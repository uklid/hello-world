"""End-to-end CFL experiment: 4-way comparison.

Runs the federated trainer under four configurations on the same
synthetic non-IID cohort with an injected concept drift:

  - fedsoft_numeric: FedSoft assignment + numeric-threshold drift
  - hflts_numeric:   HFLTS-CFL assignment + numeric-threshold drift
  - fedsoft_z:       FedSoft assignment + Z-CFL drift
  - hflts_z:         HFLTS-CFL assignment + Z-CFL drift (combined)

Reports per-config metrics over time, plus a final summary table.

Run:
    python -m experiments.end_to_end
    python -m experiments.end_to_end --out-file results/end_to_end.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfl.data import make_cohort
from cfl.trainer import FederatedTrainer, TrainerConfig


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


def evaluate(
    cohort,
    config: TrainerConfig,
    n_rounds: int,
    seed: int,
) -> dict:
    trainer = FederatedTrainer(cohort=cohort, config=config, seed=seed)
    history = trainer.run(n_rounds)
    pre = history[cohort.drift_round - 1] if cohort.drift_round else history[-1]
    post = history[-1]
    drift_round = cohort.drift_round or n_rounds
    drift_actions_per_round = np.array(
        [r.drift_actions for r in history]
    )  # (T, N)
    # Drift truth: clients with drift_to set, fired at any round >= drift_round.
    drift_truth = np.array(
        [spec.drift_to is not None for spec in cohort.specs], dtype=int
    )
    triggered_post = (
        drift_actions_per_round[drift_round:].sum(axis=0) > 0
    ).astype(int)
    triggered_pre = (
        drift_actions_per_round[:drift_round].sum(axis=0) > 0
    ).astype(int)
    tp = int(np.sum((triggered_post == 1) & (drift_truth == 1)))
    fp_post = int(np.sum((triggered_post == 1) & (drift_truth == 0)))
    fn_post = int(np.sum((triggered_post == 0) & (drift_truth == 1)))
    fp_pre = int(np.sum(triggered_pre == 1))
    precision = tp / max(tp + fp_post, 1)
    recall = tp / max(tp + fn_post, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "pre_mean_acc": pre.mean_test_accuracy,
        "pre_boundary_acc": pre.boundary_test_accuracy,
        "post_mean_acc": post.mean_test_accuracy,
        "post_boundary_acc": post.boundary_test_accuracy,
        "drift_precision": precision,
        "drift_recall": recall,
        "drift_f1": f1,
        "drift_false_pre": fp_pre,
        "drift_false_post": fp_post,
        "history": history,
    }


def format_markdown(results: dict) -> str:
    headers = [
        "config",
        "pre_acc",
        "pre_bnd",
        "post_acc",
        "post_bnd",
        "drift_P",
        "drift_R",
        "drift_F1",
        "pre_FP",
        "post_FP",
    ]
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for name, r in results.items():
        out.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{r['pre_mean_acc']:.3f}",
                    f"{r['pre_boundary_acc']:.3f}",
                    f"{r['post_mean_acc']:.3f}",
                    f"{r['post_boundary_acc']:.3f}",
                    f"{r['drift_precision']:.3f}",
                    f"{r['drift_recall']:.3f}",
                    f"{r['drift_f1']:.3f}",
                    str(r["drift_false_pre"]),
                    str(r["drift_false_post"]),
                ]
            )
            + " |"
        )
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-clients", type=int, default=30)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=120)
    parser.add_argument("--boundary-fraction", type=float, default=0.3)
    parser.add_argument("--drift-round", type=int, default=15)
    parser.add_argument("--drift-fraction", type=float, default=0.3)
    parser.add_argument("--n-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()

    cohort = make_cohort(
        n_clients=args.n_clients,
        n_clusters=args.n_clusters,
        n_features=args.n_features,
        samples_per_client=args.samples_per_client,
        boundary_fraction=args.boundary_fraction,
        drift_round=args.drift_round,
        drift_fraction=args.drift_fraction,
        seed=args.seed,
    )

    results = {}
    for name, cfg in CONFIGS.items():
        print(f"running {name}...", flush=True)
        results[name] = evaluate(
            cohort=cohort, config=cfg, n_rounds=args.n_rounds, seed=args.seed
        )

    md = format_markdown(results)
    print()
    print("Cohort:")
    print(
        f"  {args.n_clients} clients, {args.n_clusters} clusters, "
        f"{args.boundary_fraction:.0%} boundary, "
        f"drift @ round {args.drift_round} on {args.drift_fraction:.0%}"
    )
    print()
    print(md)
    print()
    print("Legend:")
    print("  pre_acc/pre_bnd: mean / boundary-only test accuracy just before drift")
    print("  post_acc/post_bnd: same metrics at the final round")
    print("  drift_P/R/F1: precision/recall/F1 for re-clustering decisions")
    print("  pre_FP: spurious re-cluster events before the drift round")
    print("  post_FP: re-clusters on clients whose task did not actually drift")
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(md + "\n")
        print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
