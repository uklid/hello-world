"""Multi-seed sweep of the four end-to-end CFL configurations.

Averages the per-config metrics over N seeds and reports mean ± std.

Run:
    python -m experiments.end_to_end_sweep
    python -m experiments.end_to_end_sweep --n-seeds 10 --out-file results/end_to_end_sweep.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfl.data import make_cohort
from experiments.end_to_end import CONFIGS, evaluate


def run_sweep(
    n_seeds: int,
    n_clients: int,
    n_clusters: int,
    n_features: int,
    samples_per_client: int,
    boundary_fraction: float,
    drift_round: int,
    drift_fraction: float,
    n_rounds: int,
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {
        name: {
            "pre_mean_acc": [],
            "pre_boundary_acc": [],
            "post_mean_acc": [],
            "post_boundary_acc": [],
            "drift_precision": [],
            "drift_recall": [],
            "drift_f1": [],
            "drift_false_post": [],
        }
        for name in CONFIGS
    }
    for seed in range(n_seeds):
        cohort = make_cohort(
            n_clients=n_clients,
            n_clusters=n_clusters,
            n_features=n_features,
            samples_per_client=samples_per_client,
            boundary_fraction=boundary_fraction,
            drift_round=drift_round,
            drift_fraction=drift_fraction,
            seed=seed,
        )
        for name, cfg in CONFIGS.items():
            r = evaluate(cohort=cohort, config=cfg, n_rounds=n_rounds, seed=seed)
            for k in out[name]:
                out[name][k].append(r[k])
    return out


def format_markdown(results: dict[str, dict[str, list[float]]], n_seeds: int) -> str:
    rows = []
    headers = [
        "config",
        "pre_acc",
        "pre_bnd",
        "post_acc",
        "post_bnd",
        "drift_P",
        "drift_R",
        "drift_F1",
        "post_FP",
    ]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")

    def cell(values: list[float]) -> str:
        arr = np.array(values, dtype=float)
        return f"{arr.mean():.3f} ± {arr.std():.3f}"

    for name, metrics in results.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    cell(metrics["pre_mean_acc"]),
                    cell(metrics["pre_boundary_acc"]),
                    cell(metrics["post_mean_acc"]),
                    cell(metrics["post_boundary_acc"]),
                    cell(metrics["drift_precision"]),
                    cell(metrics["drift_recall"]),
                    cell(metrics["drift_f1"]),
                    cell(metrics["drift_false_post"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows) + f"\n\n_aggregated over {n_seeds} seeds_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-clients", type=int, default=30)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=120)
    parser.add_argument("--boundary-fraction", type=float, default=0.3)
    parser.add_argument("--drift-round", type=int, default=15)
    parser.add_argument("--drift-fraction", type=float, default=0.3)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()
    print(
        f"sweep over {args.n_seeds} seeds | "
        f"{args.n_clients} clients, {args.n_clusters} clusters, "
        f"boundary={args.boundary_fraction:.0%}, drift@{args.drift_round} "
        f"on {args.drift_fraction:.0%}",
        flush=True,
    )
    results = run_sweep(
        n_seeds=args.n_seeds,
        n_clients=args.n_clients,
        n_clusters=args.n_clusters,
        n_features=args.n_features,
        samples_per_client=args.samples_per_client,
        boundary_fraction=args.boundary_fraction,
        drift_round=args.drift_round,
        drift_fraction=args.drift_fraction,
        n_rounds=args.n_rounds,
    )
    md = format_markdown(results, args.n_seeds)
    print()
    print(md)
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(md + "\n")
        print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
