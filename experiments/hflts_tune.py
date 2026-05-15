"""Tune HFLTS-CFL's own hyperparameters and pit it against tuned FedSoft.

HFLTS knobs:
  - ``confidence_z``: envelope width scaling on similarity std.
  - ``hflts_defuzz_softmax_tau``: temperature of the midpoint softmax
    used during defuzzification.
  - ``n_terms``: granularity of the linguistic term set.

Run with no drift to isolate the Gap A assignment effect.
"""

from __future__ import annotations

import argparse

import numpy as np

from cfl.data import make_cohort
from cfl.trainer import FederatedTrainer, TrainerConfig


def evaluate(cohort, cfg: TrainerConfig, n_rounds: int, seed: int) -> float:
    return FederatedTrainer(cohort=cohort, config=cfg, seed=seed).run(n_rounds)[-1].mean_test_accuracy


def sweep(taus, zs, n_terms_list, n_seeds, n_rounds) -> list[tuple[tuple, float, float]]:
    results = []
    for tau in taus:
        for z in zs:
            for nt in n_terms_list:
                accs = []
                for seed in range(n_seeds):
                    cohort = make_cohort(drift_round=None, seed=seed)
                    accs.append(
                        evaluate(
                            cohort,
                            TrainerConfig(
                                assignment="hflts",
                                drift="none",
                                hflts_defer_on_overlap=False,
                                hflts_defuzz_softmax_tau=tau,
                                confidence_z=z,
                                n_terms=nt,
                            ),
                            n_rounds,
                            seed,
                        )
                    )
                arr = np.array(accs)
                results.append(((tau, z, nt), float(arr.mean()), float(arr.std())))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-rounds", type=int, default=20)
    args = parser.parse_args()
    taus = [0.05, 0.15, 0.3]
    zs = [0.3, 0.5, 0.8]
    n_terms_list = [7, 11, 15]
    results = sweep(taus, zs, n_terms_list, args.n_seeds, args.n_rounds)
    results.sort(key=lambda r: -r[1])
    print("HFLTS top configurations (mean accuracy descending):")
    print("| defuzz_tau | confidence_z | n_terms | mean_acc | std |")
    print("|---|---|---|---|---|")
    for (tau, z, nt), m, s in results[:10]:
        print(f"| {tau:.2f} | {z:.2f} | {nt} | {m:.3f} | {s:.3f} |")
    best_params, best_m, best_s = results[0]
    print()
    print(f"best HFLTS no-defer: {best_m:.3f} ± {best_s:.3f} "
          f"(tau={best_params[0]}, z={best_params[1]}, n_terms={best_params[2]})")


if __name__ == "__main__":
    main()
