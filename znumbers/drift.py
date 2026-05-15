"""Drift detection that emits Z-numbers and routes through fuzzy rules.

Gap C from docs/research-plan.md. Numeric drift detectors in CFL
(FedDrift, FedDAA) trigger re-clustering off a single noisy signal,
which over-reacts when the signal itself is unreliable. This module
attaches an explicit reliability dimension to every drift report.

Signal model
------------
A client emits at each round a tuple

    (magnitude, neighbor_agreement, local_sample_size)

- ``magnitude`` in [0, 1] is the raw drift signal, e.g. relative loss
  jump or histogram distance compared to a baseline window.
- ``neighbor_agreement`` in [0, 1] is the fraction of peers (from the
  same cluster in the previous round) reporting a similarly-signed
  signal. Low agreement => B should be low.
- ``local_sample_size`` is the number of local data points; saturates
  via ``saturating_sample_score``. Few samples => B should be low.

The detector folds those into a Z-number Z = (A, B) where A is
triangular around ``magnitude`` and B is triangular around an
elementwise combination of the agreement and sample-size proxies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rules import MamdaniSystem, default_drift_rules
from .z_number import TriangularFuzzyNumber, ZNumber


def saturating_sample_score(n_samples: int, saturating_at: int = 200) -> float:
    """Map a sample count to a reliability proxy in [0, 1]."""
    if n_samples <= 0:
        return 0.0
    return float(min(1.0, n_samples / max(saturating_at, 1)))


@dataclass
class DriftSignal:
    magnitude: float
    neighbor_agreement: float
    local_sample_size: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.magnitude <= 1.0:
            raise ValueError("magnitude must be in [0, 1]")
        if not 0.0 <= self.neighbor_agreement <= 1.0:
            raise ValueError("neighbor_agreement must be in [0, 1]")
        if self.local_sample_size < 0:
            raise ValueError("local_sample_size must be non-negative")


@dataclass
class ZDriftDetector:
    """Builds a Z-number per drift signal and dispatches a Mamdani action.

    ``magnitude_spread`` and ``reliability_spread`` set the triangular
    support width around the observed peak: smaller = sharper claim.
    """

    magnitude_spread: float = 0.15
    reliability_spread: float = 0.15
    saturating_at: int = 200
    rules: list = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rules is None:
            self.rules = default_drift_rules()
        self.system = MamdaniSystem(rules=self.rules)

    def to_z_number(self, signal: DriftSignal) -> ZNumber:
        peak_a = signal.magnitude
        sample_score = saturating_sample_score(
            signal.local_sample_size, self.saturating_at
        )
        peak_b = 0.5 * signal.neighbor_agreement + 0.5 * sample_score
        A = TriangularFuzzyNumber(
            left=max(0.0, peak_a - self.magnitude_spread),
            peak=peak_a,
            right=min(1.0, peak_a + self.magnitude_spread),
        )
        B = TriangularFuzzyNumber(
            left=max(0.0, peak_b - self.reliability_spread),
            peak=peak_b,
            right=min(1.0, peak_b + self.reliability_spread),
        )
        return ZNumber(A=A, B=B)

    def decide(self, signal: DriftSignal) -> tuple[str, ZNumber, dict[str, float]]:
        z = self.to_z_number(signal)
        action, strengths = self.system.evaluate(z)
        return action, z, strengths


def numeric_threshold_baseline(
    signal: DriftSignal, threshold: float = 0.5
) -> str:
    """FedDrift-style baseline: trigger purely on magnitude.

    Returns ``re_cluster`` when ``magnitude > threshold``, else
    ``no_action``. Ignores reliability entirely.
    """
    return "re_cluster" if signal.magnitude > threshold else "no_action"


def simulate_clients(
    n_clients: int,
    n_rounds: int,
    drift_round: int,
    drift_magnitude_post: float,
    base_noise: float,
    seed: int,
) -> tuple[list[list[DriftSignal]], list[list[str]]]:
    """Generate a synthetic FL drift trace.

    ``ground_truth`` is per-(client, round) and is one of:

        ``no_action`` before the drift round (noise-only),
        ``re_cluster`` from the drift round onward.

    Half of the clients (the "low-confidence" half) are given small
    sample sizes and low neighbor agreement, so a reliability-aware
    detector should NOT trigger on them even though their numeric
    magnitude post-drift looks similar.
    """
    rng = np.random.default_rng(seed)
    signals: list[list[DriftSignal]] = []
    ground_truth: list[list[str]] = []
    for i in range(n_clients):
        low_conf = i % 2 == 1
        client_signals = []
        client_truth = []
        for t in range(n_rounds):
            drifted = t >= drift_round
            if drifted:
                base_mag = drift_magnitude_post
            else:
                base_mag = 0.1
            mag = float(np.clip(base_mag + rng.normal(0, base_noise), 0.0, 1.0))
            if low_conf:
                agreement = float(np.clip(rng.uniform(0.0, 0.3), 0.0, 1.0))
                samples = int(rng.integers(5, 30))
            else:
                agreement = float(np.clip(0.7 + rng.normal(0, 0.1), 0.0, 1.0))
                samples = int(rng.integers(180, 240))
            client_signals.append(
                DriftSignal(
                    magnitude=mag,
                    neighbor_agreement=agreement,
                    local_sample_size=samples,
                )
            )
            # Ground truth: only the high-confidence half should trigger;
            # the low-confidence half is too noisy to act on.
            if drifted and not low_conf:
                client_truth.append("re_cluster")
            else:
                client_truth.append("no_action")
        signals.append(client_signals)
        ground_truth.append(client_truth)
    return signals, ground_truth
