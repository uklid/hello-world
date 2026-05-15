"""End-to-end CFL experiment on UCI HAR (Human Activity Recognition).

30 subjects from the UCI HAR dataset become 30 federated clients.
KMeans on the per-subject feature mean partitions them into K task
profiles; boundary clients additionally pull samples from a second
profile, and an optional drift event reroutes a fraction of
non-boundary clients to another profile at the chosen round.

This is a multi-class problem (6 activities) and uses the trainer's
multinomial logistic regression path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfl.data import make_har_cohort
from experiments.end_to_end import CONFIGS, evaluate


def run_sweep(
    n_seeds: int,
    n_profiles: int,
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
        cohort = make_har_cohort(
            n_profiles=n_profiles,
            boundary_fraction=boundary_fraction,
            drift_round=drift_round,
            drift_fraction=drift_fraction,
            seed=seed,
        )
        for name, cfg in CONFIGS.items():
            r = evaluate(cohort=cohort, config=cfg, n_rounds=n_rounds, seed=seed)
            for k in out[name]:
                out[name][k].append(r[k])
        print(f"  seed {seed} done", flush=True)
    return out


def format_markdown(results: dict, n_seeds: int) -> str:
    rows = []
    headers = [
        "config", "pre_acc", "pre_bnd", "post_acc", "post_bnd",
        "drift_P", "drift_R", "drift_F1", "post_FP",
    ]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")

    def cell(values):
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
    return "\n".join(rows) + f"\n\n_aggregated over {n_seeds} seeds, UCI HAR (30 subjects, 6 activities)_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-profiles", type=int, default=3)
    parser.add_argument("--boundary-fraction", type=float, default=0.25)
    parser.add_argument("--drift-round", type=int, default=15)
    parser.add_argument("--drift-fraction", type=float, default=0.3)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()
    print(
        f"sweep over {args.n_seeds} seeds | HAR | "
        f"{args.n_profiles} profiles, boundary={args.boundary_fraction:.0%}, "
        f"drift@{args.drift_round} on {args.drift_fraction:.0%}",
        flush=True,
    )
    results = run_sweep(
        n_seeds=args.n_seeds,
        n_profiles=args.n_profiles,
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
