"""Sweep FedSoft's softmax_tau to test whether tuning closes the Gap A gap.

Lower tau is sharper. If the HFLTS-CFL win is just a side-effect of an
implicit hyperparameter choice on the FedSoft side, a well-tuned
``fedsoft_plain`` should catch up to ``hflts_no_defer``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfl.data import make_cohort
from cfl.trainer import FederatedTrainer, TrainerConfig


def evaluate(cohort, cfg: TrainerConfig, n_rounds: int, seed: int) -> float:
    trainer = FederatedTrainer(cohort=cohort, config=cfg, seed=seed)
    history = trainer.run(n_rounds)
    return history[-1].mean_test_accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.025, 0.05, 0.1, 0.15, 0.25, 0.5],
    )
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()
    print(f"sweep over {args.n_seeds} seeds, no drift")
    print()
    print("| tau | fedsoft_plain | fedsoft_defer | hflts_no_defer |")
    print("|---|---|---|---|")
    rows = []
    for tau in args.taus:
        accs_plain, accs_defer, accs_hflts = [], [], []
        for seed in range(args.n_seeds):
            cohort = make_cohort(drift_round=None, seed=seed)
            accs_plain.append(
                evaluate(
                    cohort,
                    TrainerConfig(
                        assignment="fedsoft",
                        drift="none",
                        softmax_tau=tau,
                    ),
                    args.n_rounds,
                    seed,
                )
            )
            accs_defer.append(
                evaluate(
                    cohort,
                    TrainerConfig(
                        assignment="fedsoft",
                        drift="none",
                        softmax_tau=tau,
                        fedsoft_defer_max_pi=0.55,
                    ),
                    args.n_rounds,
                    seed,
                )
            )
            accs_hflts.append(
                evaluate(
                    cohort,
                    TrainerConfig(
                        assignment="hflts",
                        drift="none",
                        softmax_tau=tau,
                        hflts_defer_on_overlap=False,
                    ),
                    args.n_rounds,
                    seed,
                )
            )
        row = (
            f"| {tau:.3f} | {np.mean(accs_plain):.3f} ± {np.std(accs_plain):.3f} "
            f"| {np.mean(accs_defer):.3f} ± {np.std(accs_defer):.3f} "
            f"| {np.mean(accs_hflts):.3f} ± {np.std(accs_hflts):.3f} |"
        )
        rows.append(row)
        print(row)
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
