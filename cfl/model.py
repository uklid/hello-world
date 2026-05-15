"""Tiny logistic regression utilities for the federated training loop.

We avoid sklearn here because the FL loop needs to step from arbitrary
weights (the per-client mixture of cluster models), and sklearn's
LogisticRegression does not support warm-starting from arbitrary
weights without an estimator object. A few NumPy operations are
clearer than wrangling that.
"""

from __future__ import annotations

import numpy as np


def predict_proba(W: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    z = X @ W + b
    # Numerically stable sigmoid.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logistic_loss(W: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> float:
    z = X @ W + b
    # Stable softplus(z) - y*z = log(1 + exp(z)) - y*z.
    softplus = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
    return float(np.mean(softplus - y * z))


def accuracy(W: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> float:
    pred = (X @ W + b > 0).astype(int)
    return float(np.mean(pred == y))


def gradient_step(
    W: np.ndarray,
    b: float,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    n_steps: int,
    l2: float = 0.0,
) -> tuple[np.ndarray, float]:
    W = W.copy()
    for _ in range(n_steps):
        p = predict_proba(W, b, X)
        err = p - y
        grad_W = X.T @ err / len(y) + l2 * W
        grad_b = float(err.mean())
        W = W - lr * grad_W
        b = b - lr * grad_b
    return W, b


def mix(
    cluster_W: np.ndarray, cluster_b: np.ndarray, pi: np.ndarray
) -> tuple[np.ndarray, float]:
    """Convex combination of K cluster models into a single (W, b)."""
    if not np.isclose(pi.sum(), 1.0):
        pi = pi / max(pi.sum(), 1e-12)
    W = (pi[:, None] * cluster_W).sum(axis=0)
    b = float((pi * cluster_b).sum())
    return W, b
