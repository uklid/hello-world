from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hflts import HFLTS, LinguisticTermSet, between
from hflts.term_set import default_term_set


def similarity_to_hflts(
    sim_history: np.ndarray,
    term_set: LinguisticTermSet,
    *,
    confidence_z: float = 1.0,
) -> HFLTS:
    """Convert a similarity-over-rounds history to an HFLTS envelope.

    Core novelty of Gap A: replace the FedSoft point estimate
        pi_k = softmax(mean_sim_k)
    with a linguistic envelope whose width encodes hesitation.

    Procedure:
      1. Compute (mean, std) of similarity history (one cluster, T rounds).
      2. Map [mean - z*std, mean + z*std] from [0, 1] to term indices.
      3. Construct the HFLTS covering that index range.

    Low variance => narrow envelope, near a single-term assignment.
    High variance => wide envelope, surfacing hesitation explicitly.
    """
    if sim_history.ndim != 1:
        raise ValueError("sim_history must be 1-D (T,)")
    mean = float(sim_history.mean())
    std = float(sim_history.std()) if sim_history.size > 1 else 0.0
    g = term_set.g
    lower_sim = max(0.0, mean - confidence_z * std)
    upper_sim = min(1.0, mean + confidence_z * std)
    lower_idx = term_set.clamp(int(round(lower_sim * g)))
    upper_idx = term_set.clamp(int(round(upper_sim * g)))
    if upper_idx < lower_idx:
        upper_idx = lower_idx
    return between(term_set, lower_idx, upper_idx)


@dataclass
class HFLTSCFLServer:
    """HFLTS-based soft cluster assignment for CFL.

    Departs from FedSoft by representing each client's per-cluster
    membership as an HFLTS (an envelope of consecutive linguistic terms)
    rather than a scalar weight. For downstream training, the envelope
    midpoint is taken and the resulting vector is renormalized into a
    simplex; the linguistic representation is retained for inspection,
    auditing, and HFLTS-aware decisions (e.g. defer-update if multiple
    clusters share overlapping high envelopes).
    """

    n_clusters: int
    term_set: LinguisticTermSet = None  # type: ignore[assignment]
    confidence_z: float = 1.0

    def __post_init__(self) -> None:
        if self.term_set is None:
            self.term_set = default_term_set()

    def assign_hflts(self, similarity_history: np.ndarray) -> list[HFLTS]:
        """Return one HFLTS per cluster for a single client.

        similarity_history: (T, K) recent rounds of per-cluster similarity.
        """
        if similarity_history.ndim != 2:
            raise ValueError("similarity_history must be 2-D (T, K)")
        if similarity_history.shape[1] != self.n_clusters:
            raise ValueError("similarity_history shape does not match n_clusters")
        return [
            similarity_to_hflts(
                similarity_history[:, k],
                self.term_set,
                confidence_z=self.confidence_z,
            )
            for k in range(self.n_clusters)
        ]

    def defuzzify(self, hflts_list: list[HFLTS]) -> np.ndarray:
        """Midpoint-based defuzzification to a mixture weight vector."""
        mids = np.array([h.midpoint() for h in hflts_list], dtype=float)
        if mids.sum() <= 0:
            return np.ones_like(mids) / len(mids)
        return mids / mids.sum()

    def describe(self, hflts_list: list[HFLTS]) -> list[str]:
        """One comparative expression per cluster, suitable for logging."""
        return [h.as_expression() for h in hflts_list]

    def has_overlap_ambiguity(
        self, hflts_list: list[HFLTS], min_overlap: int = 1
    ) -> bool:
        """Flag when two clusters have envelopes overlapping by >= min_overlap.

        Useful for downstream "defer-update" rules in line with the
        Gap A motivation: if a client looks similar to multiple clusters
        at overlapping linguistic levels, conservative updates are safer.
        """
        intervals = sorted((h.lower, h.upper) for h in hflts_list)
        for (lo_a, hi_a), (lo_b, hi_b) in zip(intervals, intervals[1:]):
            overlap = min(hi_a, hi_b) - max(lo_a, lo_b) + 1
            if overlap >= min_overlap:
                return True
        return False
