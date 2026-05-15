from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FedSoftServer:
    """FedSoft-style soft cluster assignment baseline.

    Inspired by Ruan & Joe-Wong, FedSoft, AAAI 2022. The original
    FedSoft paper additionally runs a proximal local update; this
    stripped-down server only handles the soft assignment side of the
    algorithm (mixture-weight computation). We keep it deliberately
    small so the comparison with HFLTS-CFL isolates the *assignment*
    step rather than the wider training procedure - the trainer
    applies the same local-update loop for both.

    ``defer_max_pi``: optional ambiguity fallback. When provided, the
    server falls back to a uniform mixture if no cluster's posterior
    weight reaches this threshold; the result mirrors the
    ``defer-on-overlap`` rule on the HFLTS-CFL side and is the right
    sparring partner for it in ablations.
    """

    n_clusters: int
    softmax_tau: float = 1.0
    defer_max_pi: float | None = None

    def assign(self, similarity_history: np.ndarray) -> np.ndarray:
        if similarity_history.ndim != 2:
            raise ValueError("similarity_history must be 2-D (T, K)")
        if similarity_history.shape[1] != self.n_clusters:
            raise ValueError("similarity_history shape does not match n_clusters")
        mean_sim = similarity_history.mean(axis=0)
        logits = mean_sim / max(self.softmax_tau, 1e-9)
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        if self.defer_max_pi is not None and weights.max() < self.defer_max_pi:
            weights = np.ones_like(weights) / len(weights)
        return weights
