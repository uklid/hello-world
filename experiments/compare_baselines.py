"""5-way baseline comparison without drift detection (isolates the assignment step).

Methods
-------
  - fedavg   : single global model, McMahan AISTATS 2017
  - local    : each client trains alone, no FL
  - ifca     : hard one-hot to argmax-accuracy cluster, Ghosh NeurIPS 2020
  - fedsoft  : softmax(mean similarity), Ruan & Joe-Wong AAAI 2022 (simplified)
  - hflts    : HFLTS envelope -> defuzzify, this prototype's Gap A

Run on the synthetic, UCI Adult, or UCI HAR cohort selected via --dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfl.data import make_adult_cohort, make_cohort, make_har_cohort
from cfl.trainer import FederatedTrainer, TrainerConfig


METHODS = ["fedavg", "local", "ifca", "fedsoft", "hflts"]


def build_cohort(dataset: str, seed: int):
    if dataset == "synthetic":
        return make_cohort(drift_round=None, seed=seed)
    if dataset == "adult":
        return make_adult_cohort(drift_round=None, seed=seed)
    if dataset == "har":
        return make_har_cohort(drift_round=None, seed=seed)
    raise ValueError(f"unknown dataset {dataset!r}")


def evaluate(cohort, method: str, n_rounds: int, seed: int) -> dict:
    cfg = TrainerConfig(assignment=method, drift="none")
    trainer = FederatedTrainer(cohort=cohort, config=cfg, seed=seed)
    history = trainer.run(n_rounds)
    return {
        "mean_acc": history[-1].mean_test_accuracy,
        "boundary_acc": history[-1].boundary_test_accuracy,
    }


def format_markdown(results: dict, n_seeds: int, dataset: str) -> str:
    rows = [f"### {dataset}", "", "| method | mean_acc | boundary_acc |", "|---|---|---|"]

    def cell(values):
        arr = np.array(values, dtype=float)
        return f"{arr.mean():.3f} ± {arr.std():.3f}"

    for name, m in results.items():
        rows.append(f"| {name} | {cell(m['mean_acc'])} | {cell(m['boundary_acc'])} |")
    return "\n".join(rows) + f"\n\n_aggregated over {n_seeds} seeds_"


def run_dataset(dataset: str, n_seeds: int, n_rounds: int) -> dict:
    out = {m: {"mean_acc": [], "boundary_acc": []} for m in METHODS}
    for seed in range(n_seeds):
        cohort = build_cohort(dataset, seed)
        for m in METHODS:
            r = evaluate(cohort, m, n_rounds, seed)
            out[m]["mean_acc"].append(r["mean_acc"])
            out[m]["boundary_acc"].append(r["boundary_acc"])
        print(f"  [{dataset}] seed {seed} done", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["synthetic", "adult", "har"],
        choices=["synthetic", "adult", "har"],
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()

    pieces = []
    for d in args.datasets:
        results = run_dataset(d, args.n_seeds, args.n_rounds)
        pieces.append(format_markdown(results, args.n_seeds, d))
        print()
        print(pieces[-1])
        print()
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text("\n\n".join(pieces) + "\n")
        print(f"wrote {args.out_file}")


if __name__ == "__main__":
    main()
