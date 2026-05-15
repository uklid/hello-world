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
) -> tuple[np.ndarray, float | np.ndarray]:
    """Convex combination of K cluster models into a single (W, b).

    Supports both the binary case (``cluster_W`` of shape (K, d) and
    ``cluster_b`` of shape (K,)) and the multinomial case (``cluster_W``
    of shape (K, d, C) and ``cluster_b`` of shape (K, C)).
    """
    if not np.isclose(pi.sum(), 1.0):
        pi = pi / max(pi.sum(), 1e-12)
    if cluster_W.ndim == 2:
        W = (pi[:, None] * cluster_W).sum(axis=0)
        b = float((pi * cluster_b).sum())
        return W, b
    if cluster_W.ndim == 3:
        W = (pi[:, None, None] * cluster_W).sum(axis=0)
        b = (pi[:, None] * cluster_b).sum(axis=0)
        return W, b
    raise ValueError(f"unsupported cluster_W ndim={cluster_W.ndim}")


# ---------- multinomial logistic regression ----------


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(shifted)
    return ez / ez.sum(axis=1, keepdims=True)


def softmax_predict_proba(W: np.ndarray, b: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Multinomial probabilities. ``W`` is (d, C), ``b`` is (C,)."""
    return _softmax(X @ W + b)


def softmax_loss(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Cross-entropy loss, numerically stable."""
    logits = X @ W + b
    max_logit = logits.max(axis=1, keepdims=True)
    log_sum_exp = max_logit + np.log(
        np.exp(logits - max_logit).sum(axis=1, keepdims=True)
    )
    chosen = logits[np.arange(len(y)), y][:, None]
    return float(np.mean(log_sum_exp - chosen))


def softmax_accuracy(W: np.ndarray, b: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    pred = (X @ W + b).argmax(axis=1)
    return float(np.mean(pred == y))


def softmax_gradient_step(
    W: np.ndarray,
    b: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
    n_steps: int,
    l2: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    W = W.copy()
    b = b.copy()
    n_classes = W.shape[1]
    Y = np.eye(n_classes)[y]
    for _ in range(n_steps):
        P = softmax_predict_proba(W, b, X)
        diff = P - Y
        grad_W = X.T @ diff / len(y) + l2 * W
        grad_b = diff.mean(axis=0)
        W = W - lr * grad_W
        b = b - lr * grad_b
    return W, b
