"""Drift matrix: 4 clustering methods x 2 drift policies, on 3 datasets.

Methods (clustering only - FedAvg / Local skipped because they have
no notion of cluster reassignment to act on):
  ifca, sattler, fedsoft, hflts

Drift policies:
  numeric  : trigger when relative-loss change > 0.5
  z_cfl    : Z-number reliability-aware Mamdani rules

Datasets:
  synthetic (linear binary tasks), adult (binary income), har (6-class activities)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfl.data import make_adult_cohort, make_cohort, make_har_cohort
from cfl.trainer import FederatedTrainer, TrainerConfig
from experiments.end_to_end import evaluate


CONFIGS = {
    "fedsoft_numeric": TrainerConfig(assignment="fedsoft", drift="numeric"),
    "hflts_numeric": TrainerConfig(assignment="hflts", drift="numeric"),
    "ifca_numeric": TrainerConfig(assignment="ifca", drift="numeric"),
    "sattler_numeric": TrainerConfig(assignment="sattler", drift="numeric"),
    "fedsoft_z": TrainerConfig(assignment="fedsoft", drift="z_cfl"),
    "hflts_z": TrainerConfig(assignment="hflts", drift="z_cfl"),
    "ifca_z": TrainerConfig(assignment="ifca", drift="z_cfl"),
    "sattler_z": TrainerConfig(assignment="sattler", drift="z_cfl"),
}


def build_cohort(dataset: str, seed: int, drift_round: int):
    if dataset == "synthetic":
        return make_cohort(drift_round=drift_round, seed=seed)
    if dataset == "adult":
        return make_adult_cohort(drift_round=drift_round, seed=seed)
    if dataset == "har":
        return make_har_cohort(drift_round=drift_round, seed=seed)
    raise ValueError(f"unknown dataset {dataset!r}")


def run_dataset(dataset: str, n_seeds: int, n_rounds: int, drift_round: int) -> dict:
    out = {
        name: {
            "post_mean_acc": [],
            "drift_precision": [],
            "drift_recall": [],
            "drift_f1": [],
            "drift_false_post": [],
        }
        for name in CONFIGS
    }
    for seed in range(n_seeds):
        cohort = build_cohort(dataset, seed, drift_round)
        for name, cfg in CONFIGS.items():
            r = evaluate(cohort, cfg, n_rounds, seed)
            for k in out[name]:
                out[name][k].append(r[k])
        print(f"  [{dataset}] seed {seed} done", flush=True)
    return out


def format_dataset(name: str, results: dict, n_seeds: int) -> str:
    rows = [
        f"### {name}",
        "",
        "| config | post_acc | drift_P | drift_R | drift_F1 | post_FP |",
        "|---|---|---|---|---|---|",
    ]

    def cell(values):
        arr = np.array(values, dtype=float)
        return f"{arr.mean():.3f} ± {arr.std():.3f}"

    for cname, m in results.items():
        rows.append(
            f"| {cname} | {cell(m['post_mean_acc'])} | "
            f"{cell(m['drift_precision'])} | {cell(m['drift_recall'])} | "
            f"{cell(m['drift_f1'])} | {cell(m['drift_false_post'])} |"
        )
    return "\n".join(rows) + f"\n\n_aggregated over {n_seeds} seeds_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["synthetic", "adult", "har"],
    )
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-rounds", type=int, default=25)
    parser.add_argument("--drift-round", type=int, default=12)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()

    pieces = []
    for d in args.datasets:
        results = run_dataset(d, args.n_seeds, args.n_rounds, args.drift_round)
        piece = format_dataset(d, results, args.n_seeds)
        pieces.append(piece)
        print()
        print(piece)
        print()
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text("\n\n".join(pieces) + "\n")
        print(f"wrote {args.out_file}")


if __name__ == "__main__":
    main()
