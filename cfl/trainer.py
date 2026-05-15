"""End-to-end federated trainer with pluggable assignment and drift policies.

Per round
---------
1. Each client trains its current personalized model (a convex mixture
   of cluster centroids) for ``local_steps`` gradient steps on its
   local training data.
2. Server aggregates the local weights into K cluster centroids using
   the soft assignment ``pi`` (weighted average per cluster).
3. Server measures per-(client, cluster) similarity on each client's
   train data: similarity = accuracy of the cluster centroid model.
4. Server updates ``pi`` according to the assignment policy
   (``fedsoft`` or ``hflts``). HFLTS additionally falls back to a
   uniform mixture for clients whose envelopes overlap - the explicit
   "defer-update" rule from the Gap A plan.
5. Server runs a drift detector (``numeric`` or ``z_cfl``); when it
   fires for a client, that client's similarity history is reset so
   the assignment converges to the post-drift cluster.

Evaluation
----------
For every client, the personalized model is the mixture given by the
current ``pi`` row applied to the current cluster centroids. We report
per-client test accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from cfl.baseline import FedSoftServer
from cfl.data import Cohort
from cfl.hflts_cfl import HFLTSCFLServer
from cfl.model import accuracy, gradient_step, logistic_loss, mix
from hflts.term_set import LinguisticTermSet, uniform_term_set
from znumbers import ZDriftDetector
from znumbers.drift import DriftSignal, numeric_threshold_baseline


@dataclass
class TrainerConfig:
    assignment: str = "fedsoft"  # "fedsoft" | "hflts"
    drift: str = "none"  # "none" | "numeric" | "z_cfl"
    # FedSoft softmax temperature: smaller = sharper assignment.
    # 0.05 is well below the FedSoft paper default (~0.25) but is
    # necessary on this synthetic regime where the accuracy gaps
    # between cluster models stay in a narrow band.
    softmax_tau: float = 0.05
    confidence_z: float = 0.5
    n_terms: int = 11
    hflts_defer_on_overlap: bool = True
    # Symmetric defer rule for the FedSoft baseline: when no cluster's
    # posterior reaches this threshold, fall back to a uniform mixture.
    # ``None`` disables the rule and reproduces the plain FedSoft path.
    fedsoft_defer_max_pi: float | None = None
    hflts_defuzz_method: str = "midpoint_softmax"
    hflts_defuzz_softmax_tau: float = 0.3
    hflts_overlap_min: int = 2  # require overlap of >=2 terms to defer
    drift_threshold: float = 0.5
    drift_window: int = 3
    sim_window: int = 5
    local_lr: float = 0.3
    local_steps: int = 5
    l2: float = 1e-3
    neighbor_radius: int = 5


@dataclass
class TrainerState:
    cluster_W: np.ndarray
    cluster_b: np.ndarray
    pi: np.ndarray
    sim_history: list[list[np.ndarray]] = field(default_factory=list)
    loss_history: list[list[float]] = field(default_factory=list)
    drift_triggered_per_round: list[np.ndarray] = field(default_factory=list)


@dataclass
class RoundReport:
    round_idx: int
    mean_test_accuracy: float
    boundary_test_accuracy: float
    drift_actions: np.ndarray  # 1 if "re_cluster", else 0
    pi: np.ndarray


class FederatedTrainer:
    def __init__(self, cohort: Cohort, config: TrainerConfig, seed: int = 0):
        self.cohort = cohort
        self.config = config
        rng = np.random.default_rng(seed)
        n_features = cohort.train[0][0].shape[1]
        K = cohort.n_clusters
        N = cohort.n_clients
        cluster_W = rng.normal(scale=0.3, size=(K, n_features))
        cluster_b = np.zeros(K)
        # Random hard one-hot assignment at init to break the soft-CFL
        # symmetry: a uniform pi would make every cluster centroid
        # converge to the same global average after the first
        # aggregation, locking the trainer into a trivial fixed point.
        init_idx = rng.integers(0, K, size=N)
        pi = np.eye(K)[init_idx].astype(float)
        self.state = TrainerState(cluster_W=cluster_W, cluster_b=cluster_b, pi=pi)
        for _ in range(N):
            self.state.sim_history.append([])
            self.state.loss_history.append([])

        self._fedsoft = FedSoftServer(
            n_clusters=K,
            softmax_tau=config.softmax_tau,
            defer_max_pi=config.fedsoft_defer_max_pi,
        )
        self._term_set: LinguisticTermSet = uniform_term_set(config.n_terms)
        self._hflts = HFLTSCFLServer(
            n_clusters=K,
            term_set=self._term_set,
            confidence_z=config.confidence_z,
        )
        self._z_detector = ZDriftDetector()

    # ---------- public API ----------

    def run(self, n_rounds: int) -> list[RoundReport]:
        return [self.run_round(t) for t in range(n_rounds)]

    def run_round(self, round_idx: int) -> RoundReport:
        train_data = self.cohort.train_data(round_idx)
        # Capture pre-train losses: this is the "drift surprise" signal
        # before the local update has a chance to absorb the shift.
        pre_train_losses = self._compute_losses(train_data)
        local_W, local_b = self._local_train(train_data)
        self._aggregate_clusters(local_W, local_b)
        sims = self._compute_similarities(train_data)
        self._update_history(sims, pre_train_losses)
        self._update_assignment()
        drift_actions = self._run_drift_detector(train_data)
        report = self._report(round_idx, drift_actions)
        self.state.drift_triggered_per_round.append(drift_actions)
        return report

    # ---------- inference ----------

    def personalized_model(self, client_idx: int) -> tuple[np.ndarray, float]:
        return mix(
            self.state.cluster_W,
            self.state.cluster_b,
            self.state.pi[client_idx],
        )

    # ---------- per-round helpers ----------

    def _local_train(
        self, train_data: list[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[np.ndarray, np.ndarray]:
        N = self.cohort.n_clients
        K = self.cohort.n_clusters
        d = self.state.cluster_W.shape[1]
        local_W = np.zeros((N, d))
        local_b = np.zeros(N)
        for i in range(N):
            X, y = train_data[i]
            W_i, b_i = mix(
                self.state.cluster_W, self.state.cluster_b, self.state.pi[i]
            )
            W_i, b_i = gradient_step(
                W_i,
                b_i,
                X,
                y,
                lr=self.config.local_lr,
                n_steps=self.config.local_steps,
                l2=self.config.l2,
            )
            local_W[i] = W_i
            local_b[i] = b_i
        return local_W, local_b

    def _aggregate_clusters(self, local_W: np.ndarray, local_b: np.ndarray) -> None:
        K = self.cohort.n_clusters
        d = local_W.shape[1]
        for k in range(K):
            w = self.state.pi[:, k]
            denom = max(w.sum(), 1e-9)
            self.state.cluster_W[k] = (w[:, None] * local_W).sum(axis=0) / denom
            self.state.cluster_b[k] = float((w * local_b).sum() / denom)

    def _compute_similarities(
        self, train_data: list[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        N = self.cohort.n_clients
        K = self.cohort.n_clusters
        sims = np.zeros((N, K))
        for i in range(N):
            X, y = train_data[i]
            for k in range(K):
                sims[i, k] = accuracy(
                    self.state.cluster_W[k], self.state.cluster_b[k], X, y
                )
        return sims

    def _compute_losses(
        self, train_data: list[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        N = self.cohort.n_clients
        losses = np.zeros(N)
        for i in range(N):
            X, y = train_data[i]
            W_i, b_i = mix(
                self.state.cluster_W, self.state.cluster_b, self.state.pi[i]
            )
            losses[i] = logistic_loss(W_i, b_i, X, y)
        return losses

    def _update_history(
        self, sims: np.ndarray, pre_train_losses: np.ndarray
    ) -> None:
        for i in range(self.cohort.n_clients):
            self.state.sim_history[i].append(sims[i])
            self.state.loss_history[i].append(float(pre_train_losses[i]))

    def _update_assignment(self) -> None:
        N = self.cohort.n_clients
        K = self.cohort.n_clusters
        for i in range(N):
            hist = np.array(self.state.sim_history[i][-self.config.sim_window:])
            if self.config.assignment == "fedsoft":
                pi_i = self._fedsoft.assign(hist)
            elif self.config.assignment == "hflts":
                hflts_list = self._hflts.assign_hflts(hist)
                if self.config.hflts_defer_on_overlap and self._hflts.has_overlap_ambiguity(
                    hflts_list, min_overlap=self.config.hflts_overlap_min
                ):
                    pi_i = np.ones(K) / K
                else:
                    pi_i = self._hflts.defuzzify(
                        hflts_list,
                        method=self.config.hflts_defuzz_method,
                        softmax_tau=self.config.hflts_defuzz_softmax_tau,
                    )
            else:
                raise ValueError(f"unknown assignment: {self.config.assignment}")
            self.state.pi[i] = pi_i

    def _run_drift_detector(
        self, train_data: list[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        N = self.cohort.n_clients
        actions = np.zeros(N, dtype=int)
        if self.config.drift == "none":
            return actions
        # Per-client drift signal: rolling-window relative loss change.
        # neighbor_agreement: fraction of nearest-K (in pi-cosine sense)
        # clients whose drift signal sign matches.
        signals = self._build_drift_signals()
        for i in range(N):
            if self.config.drift == "numeric":
                action = numeric_threshold_baseline(
                    signals[i], threshold=self.config.drift_threshold
                )
            elif self.config.drift == "z_cfl":
                action, _, _ = self._z_detector.decide(signals[i])
            else:
                raise ValueError(f"unknown drift: {self.config.drift}")
            if action == "re_cluster":
                actions[i] = 1
                # Reset that client's similarity history so the
                # assignment is recomputed from the post-drift regime.
                self.state.sim_history[i] = []
        return actions

    def _build_drift_signals(self) -> list[DriftSignal]:
        N = self.cohort.n_clients
        w = self.config.drift_window
        sigs: list[DriftSignal] = []
        # Per-client relative loss change.
        rel_change = np.zeros(N)
        for i in range(N):
            history = self.state.loss_history[i]
            if len(history) < w + 1:
                rel_change[i] = 0.0
                continue
            recent = history[-1]
            prior = float(np.mean(history[-w - 1 : -1]))
            rel_change[i] = max(0.0, (recent - prior) / max(prior, 1e-6))
        rel_change = np.clip(rel_change, 0.0, 1.0)

        # Cohort-level agreement: an earlier version of this function
        # defined "neighbours" via pi-cosine similarity, which is
        # degenerate when the assignment is soft (every pair of clients
        # has cosine ~1, so the top-K neighbour pick is essentially
        # random). The new definition: take peers as the clients whose
        # recent loss trajectory is most similar (cosine over the last
        # ``drift_window`` loss values), independent of the assignment.
        elevated = rel_change > 0.2
        agreement = np.zeros(N)
        loss_matrix = []
        for i in range(N):
            history = self.state.loss_history[i]
            tail = history[-w:] if len(history) >= w else history + [0.0] * (w - len(history))
            loss_matrix.append(tail)
        loss_matrix = np.array(loss_matrix, dtype=float)
        norms = np.linalg.norm(loss_matrix, axis=1, keepdims=True) + 1e-9
        cos = (loss_matrix @ loss_matrix.T) / (norms @ norms.T)
        np.fill_diagonal(cos, -np.inf)
        radius = min(self.config.neighbor_radius, N - 1)
        for i in range(N):
            neighbours = np.argpartition(-cos[i], radius)[:radius]
            mine = bool(elevated[i])
            theirs = elevated[neighbours]
            agreement[i] = float(np.mean(theirs == mine))

        for i in range(N):
            n_samples = len(self.cohort.train_data(0)[i][1])
            sigs.append(
                DriftSignal(
                    magnitude=float(rel_change[i]),
                    neighbor_agreement=float(agreement[i]),
                    local_sample_size=int(n_samples),
                )
            )
        return sigs

    def _report(self, round_idx: int, drift_actions: np.ndarray) -> RoundReport:
        N = self.cohort.n_clients
        accs = np.zeros(N)
        for i in range(N):
            W_i, b_i = self.personalized_model(i)
            X_test, y_test = self.cohort.test[i]
            accs[i] = accuracy(W_i, b_i, X_test, y_test)
        boundary_mask = self.cohort.is_boundary
        if boundary_mask.any():
            boundary_acc = float(accs[boundary_mask].mean())
        else:
            boundary_acc = float(accs.mean())
        return RoundReport(
            round_idx=round_idx,
            mean_test_accuracy=float(accs.mean()),
            boundary_test_accuracy=boundary_acc,
            drift_actions=drift_actions,
            pi=self.state.pi.copy(),
        )
