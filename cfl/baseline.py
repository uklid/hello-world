from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FedSoftServer:
    """FedSoft-style soft cluster assignment baseline.

    Ruan & Joe-Wong, AAAI 2022. Each client receives a numeric mixture
    weight vector pi_i in the simplex over K clusters. This stripped-down
    baseline uses a similarity-history mean as the signature (one signal
    per round) and softmaxes the similarity to clusters.
    """

    n_clusters: int
    softmax_tau: float = 1.0

    def assign(self, similarity_history: np.ndarray) -> np.ndarray:
        """Return mixture weight pi over K clusters.

        similarity_history: array of shape (T, K) - T recent rounds of
        per-cluster similarity for one client. We collapse to the mean
        across rounds and softmax with temperature tau.
        """
        if similarity_history.ndim != 2:
            raise ValueError("similarity_history must be 2-D (T, K)")
        if similarity_history.shape[1] != self.n_clusters:
            raise ValueError("similarity_history shape does not match n_clusters")
        mean_sim = similarity_history.mean(axis=0)
        logits = mean_sim / max(self.softmax_tau, 1e-9)
        logits = logits - logits.max()
        weights = np.exp(logits)
        weights = weights / weights.sum()
        return weights
