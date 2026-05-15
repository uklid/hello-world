"""End-to-end CFL experiment on UCI Adult tabular data.

Mirrors experiments/end_to_end_sweep.py but uses ``make_adult_cohort``
instead of the synthetic generator. Same four configurations:

  - fedsoft_numeric
  - hflts_numeric
  - fedsoft_z
  - hflts_z

Per-seed run is heavier than the synthetic version (105 features after
one-hot, 200 samples per client), so the default sweep is 5 seeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfl.data import make_adult_cohort
from experiments.end_to_end import CONFIGS, evaluate


def run_sweep(
    n_seeds: int,
    n_clients: int,
    samples_per_client: int,
    n_profiles: int,
    boundary_fraction: float,
    drift_round: int,
    drift_fraction: float,
    partition_by: str,
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
        cohort = make_adult_cohort(
            n_clients=n_clients,
            samples_per_client=samples_per_client,
            n_profiles=n_profiles,
            boundary_fraction=boundary_fraction,
            drift_round=drift_round,
            drift_fraction=drift_fraction,
            partition_by=partition_by,
            seed=seed,
        )
        for name, cfg in CONFIGS.items():
            r = evaluate(cohort=cohort, config=cfg, n_rounds=n_rounds, seed=seed)
            for k in out[name]:
                out[name][k].append(r[k])
        print(f"  seed {seed} done", flush=True)
    return out


def format_markdown(results: dict[str, dict[str, list[float]]], n_seeds: int) -> str:
    rows = []
    headers = [
        "config", "pre_acc", "pre_bnd", "post_acc", "post_bnd",
        "drift_P", "drift_R", "drift_F1", "post_FP",
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
    return "\n".join(rows) + f"\n\n_aggregated over {n_seeds} seeds, UCI Adult tabular cohort_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-clients", type=int, default=30)
    parser.add_argument("--samples-per-client", type=int, default=200)
    parser.add_argument("--n-profiles", type=int, default=3)
    parser.add_argument("--boundary-fraction", type=float, default=0.3)
    parser.add_argument("--drift-round", type=int, default=15)
    parser.add_argument("--drift-fraction", type=float, default=0.3)
    parser.add_argument("--partition-by", type=str, default="education-num")
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()
    print(
        f"sweep over {args.n_seeds} seeds | adult | partition_by={args.partition_by} | "
        f"{args.n_clients} clients, {args.n_profiles} profiles, "
        f"boundary={args.boundary_fraction:.0%}, drift@{args.drift_round} "
        f"on {args.drift_fraction:.0%}",
        flush=True,
    )
    results = run_sweep(
        n_seeds=args.n_seeds,
        n_clients=args.n_clients,
        samples_per_client=args.samples_per_client,
        n_profiles=args.n_profiles,
        boundary_fraction=args.boundary_fraction,
        drift_round=args.drift_round,
        drift_fraction=args.drift_fraction,
        partition_by=args.partition_by,
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
