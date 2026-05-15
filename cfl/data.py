"""Synthetic non-IID FL task generator with optional concept drift.

K underlying binary-classification tasks share the same input space
but have different ground-truth linear boundaries. Each client is
assigned a primary task; a configurable fraction of clients are
``boundary`` clients whose data is a 50/50 mix of two tasks - these
are the ones HFLTS-CFL is supposed to help by surfacing hesitation.

A drift event swaps the primary task of a subset of clients at a
chosen round. The trainer is responsible for picking that up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskCatalog:
    W: np.ndarray  # (K, d)
    b: np.ndarray  # (K,)

    @property
    def n_tasks(self) -> int:
        return self.W.shape[0]


def make_tasks(n_tasks: int, n_features: int, seed: int) -> TaskCatalog:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_tasks, n_features))
    # Normalize so tasks have comparable margin sizes.
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    W *= 3.0
    b = rng.normal(size=n_tasks) * 0.3
    return TaskCatalog(W=W, b=b)


def sample_from_task(
    catalog: TaskCatalog, task_id: int, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n_samples, catalog.W.shape[1]))
    z = X @ catalog.W[task_id] + catalog.b[task_id]
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n_samples) < p).astype(int)
    return X, y


@dataclass
class ClientSpec:
    primary: int
    secondary: int | None  # None for non-boundary clients
    drift_to: int | None  # task id after the drift event, else None


@dataclass
class Cohort:
    catalog: TaskCatalog
    specs: list[ClientSpec]
    train: list[tuple[np.ndarray, np.ndarray]]
    test: list[tuple[np.ndarray, np.ndarray]]
    primary_label: np.ndarray  # ground-truth cluster (pre-drift)
    is_boundary: np.ndarray
    drift_round: int | None
    drift_post_label: np.ndarray  # ground-truth cluster after drift_round
    drift_post_data: list[tuple[np.ndarray, np.ndarray]]  # post-drift train data

    @property
    def n_clients(self) -> int:
        return len(self.specs)

    @property
    def n_clusters(self) -> int:
        return self.catalog.n_tasks

    def train_data(self, round_idx: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.drift_round is None or round_idx < self.drift_round:
            return self.train
        return self.drift_post_data


def make_cohort(
    n_clients: int = 30,
    n_clusters: int = 3,
    n_features: int = 5,
    samples_per_client: int = 120,
    boundary_fraction: float = 0.3,
    drift_round: int | None = 15,
    drift_fraction: float = 0.3,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Cohort:
    rng = np.random.default_rng(seed)
    catalog = make_tasks(n_clusters, n_features, seed)

    # Primary assignment.
    primary = rng.integers(0, n_clusters, size=n_clients)
    is_boundary = np.zeros(n_clients, dtype=bool)
    is_boundary[: int(round(boundary_fraction * n_clients))] = True
    rng.shuffle(is_boundary)

    # Drift: switch primary task for drift_fraction of NON-boundary clients
    # (so the drift signal is clean rather than entangled with boundary
    # ambiguity).
    drift_to = np.full(n_clients, -1, dtype=int)
    if drift_round is not None:
        candidates = np.where(~is_boundary)[0]
        n_drift = int(round(drift_fraction * n_clients))
        n_drift = min(n_drift, len(candidates))
        drift_ids = rng.choice(candidates, size=n_drift, replace=False)
        for i in drift_ids:
            new_task = int(rng.integers(0, n_clusters))
            while new_task == primary[i]:
                new_task = int(rng.integers(0, n_clusters))
            drift_to[i] = new_task

    specs: list[ClientSpec] = []
    train: list[tuple[np.ndarray, np.ndarray]] = []
    test: list[tuple[np.ndarray, np.ndarray]] = []
    drift_post_data: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_clients):
        sec = None
        if is_boundary[i]:
            sec = int(rng.integers(0, n_clusters))
            while sec == int(primary[i]):
                sec = int(rng.integers(0, n_clusters))
        spec = ClientSpec(
            primary=int(primary[i]),
            secondary=sec,
            drift_to=int(drift_to[i]) if drift_to[i] >= 0 else None,
        )
        specs.append(spec)

        # Pre-drift train data
        total = samples_per_client
        if spec.secondary is None:
            X_train, y_train = sample_from_task(catalog, spec.primary, total, rng)
        else:
            half = total // 2
            X_a, y_a = sample_from_task(catalog, spec.primary, half, rng)
            X_b, y_b = sample_from_task(catalog, spec.secondary, total - half, rng)
            X_train = np.concatenate([X_a, X_b], axis=0)
            y_train = np.concatenate([y_a, y_b], axis=0)
        # Shuffle.
        idx = rng.permutation(len(y_train))
        X_train, y_train = X_train[idx], y_train[idx]

        # Test data drawn from the same distribution (matches the
        # post-train task identity for non-drift clients).
        n_test = max(int(total * test_fraction), 20)
        if spec.secondary is None:
            X_test, y_test = sample_from_task(catalog, spec.primary, n_test, rng)
        else:
            half = n_test // 2
            X_t_a, y_t_a = sample_from_task(catalog, spec.primary, half, rng)
            X_t_b, y_t_b = sample_from_task(
                catalog, spec.secondary, n_test - half, rng
            )
            X_test = np.concatenate([X_t_a, X_t_b], axis=0)
            y_test = np.concatenate([y_t_a, y_t_b], axis=0)

        train.append((X_train, y_train))
        test.append((X_test, y_test))

        # Post-drift train: switch to drift_to if applicable.
        if spec.drift_to is not None:
            X_post, y_post = sample_from_task(catalog, spec.drift_to, total, rng)
            drift_post_data.append((X_post, y_post))
        else:
            drift_post_data.append((X_train, y_train))

    primary_label = primary.astype(int)
    drift_post_label = primary_label.copy()
    for i in range(n_clients):
        if specs[i].drift_to is not None:
            drift_post_label[i] = specs[i].drift_to

    return Cohort(
        catalog=catalog,
        specs=specs,
        train=train,
        test=test,
        primary_label=primary_label,
        is_boundary=is_boundary,
        drift_round=drift_round,
        drift_post_label=drift_post_label,
        drift_post_data=drift_post_data,
    )
