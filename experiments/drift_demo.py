"""Synthetic drift demo: Z-number reliability-aware detector vs numeric threshold.

Gap C prototype from docs/research-plan.md. Compares two re-clustering
trigger policies on a synthetic FL trace where half of the clients have
low reliability (small sample size, low neighbor agreement) and should
not trigger re-clustering even when their raw drift magnitude crosses
a numeric threshold.

Metrics
-------
- precision / recall / F1 of ``re_cluster`` predictions vs ground truth
- false-positive rate on low-reliability clients (Gap C headline)

Run:
    python -m experiments.drift_demo
    python -m experiments.drift_demo --threshold 0.4
"""

from __future__ import annotations

import argparse

import numpy as np

from znumbers import ZDriftDetector
from znumbers.drift import numeric_threshold_baseline, simulate_clients


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    fpr = fp / max(fp + tn, 1)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
    }


def run(
    n_clients: int = 20,
    n_rounds: int = 30,
    drift_round: int = 15,
    drift_magnitude_post: float = 0.65,
    base_noise: float = 0.08,
    seed: int = 0,
    threshold: float = 0.5,
) -> dict:
    signals, ground_truth = simulate_clients(
        n_clients=n_clients,
        n_rounds=n_rounds,
        drift_round=drift_round,
        drift_magnitude_post=drift_magnitude_post,
        base_noise=base_noise,
        seed=seed,
    )

    detector = ZDriftDetector()

    flat_truth = []
    flat_z = []
    flat_numeric = []
    low_conf_truth = []
    low_conf_z = []
    low_conf_numeric = []
    for i in range(n_clients):
        is_low_conf = i % 2 == 1
        for t in range(n_rounds):
            sig = signals[i][t]
            truth = ground_truth[i][t]
            action_z, _, _ = detector.decide(sig)
            action_numeric = numeric_threshold_baseline(sig, threshold=threshold)
            yt = 1 if truth == "re_cluster" else 0
            yz = 1 if action_z == "re_cluster" else 0
            yn = 1 if action_numeric == "re_cluster" else 0
            flat_truth.append(yt)
            flat_z.append(yz)
            flat_numeric.append(yn)
            if is_low_conf:
                low_conf_truth.append(yt)
                low_conf_z.append(yz)
                low_conf_numeric.append(yn)

    y_true = np.array(flat_truth)
    y_z = np.array(flat_z)
    y_num = np.array(flat_numeric)
    z_metrics = _binary_metrics(y_true, y_z)
    num_metrics = _binary_metrics(y_true, y_num)

    y_true_lc = np.array(low_conf_truth)
    y_z_lc = np.array(low_conf_z)
    y_num_lc = np.array(low_conf_numeric)
    z_metrics_lc = _binary_metrics(y_true_lc, y_z_lc)
    num_metrics_lc = _binary_metrics(y_true_lc, y_num_lc)

    print("=" * 64)
    print("Z-CFL drift detector vs numeric-threshold baseline (synthetic)")
    print("=" * 64)
    print(
        f"clients={n_clients}, rounds={n_rounds}, drift@={drift_round}, "
        f"post-drift mag={drift_magnitude_post}, base_noise={base_noise}, "
        f"threshold={threshold}"
    )
    print("-" * 64)
    print("All clients:")
    print(
        f"  numeric  : P={num_metrics['precision']:.3f}  "
        f"R={num_metrics['recall']:.3f}  F1={num_metrics['f1']:.3f}  "
        f"FPR={num_metrics['fpr']:.3f}"
    )
    print(
        f"  Z-CFL    : P={z_metrics['precision']:.3f}  "
        f"R={z_metrics['recall']:.3f}  F1={z_metrics['f1']:.3f}  "
        f"FPR={z_metrics['fpr']:.3f}"
    )
    print("-" * 64)
    print("Low-reliability subgroup (Gap C headline):")
    print(
        f"  numeric  : FPR={num_metrics_lc['fpr']:.3f}  "
        f"FP={num_metrics_lc['fp']}"
    )
    print(
        f"  Z-CFL    : FPR={z_metrics_lc['fpr']:.3f}  "
        f"FP={z_metrics_lc['fp']}"
    )

    fpr_reduction = (
        (num_metrics_lc["fpr"] - z_metrics_lc["fpr"])
        / max(num_metrics_lc["fpr"], 1e-12)
    )
    print(f"  -> low-conf false-positive reduction: {fpr_reduction:.0%}")
    return {
        "z_metrics": z_metrics,
        "num_metrics": num_metrics,
        "z_metrics_low_conf": z_metrics_lc,
        "num_metrics_low_conf": num_metrics_lc,
        "fpr_reduction_low_conf": float(fpr_reduction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-clients", type=int, default=20)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--drift-round", type=int, default=15)
    parser.add_argument("--drift-magnitude-post", type=float, default=0.65)
    parser.add_argument("--base-noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
