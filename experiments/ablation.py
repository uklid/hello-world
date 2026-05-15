"""Ablation study: separate the HFLTS representation from the defer-on-overlap rule.

Configurations (all with drift detection disabled to isolate the
assignment-step effect):

  - fedsoft_plain: softmax(mean_sim / tau), no fallback
  - fedsoft_defer: softmax + uniform-mixture fallback when no cluster's
    posterior reaches ``fedsoft_defer_max_pi`` (mirrors the HFLTS
    defer rule on the FedSoft side)
  - hflts_no_defer: HFLTS envelope -> midpoint_softmax defuzzification,
    NO defer-on-overlap fallback (isolates the representation)
  - hflts_defer: full HFLTS-CFL with defer-on-overlap (matches the
    end-to-end Gap A configuration)

Reports per-config pre-drift / final-round accuracy and the
boundary-client accuracy that the Gap A motivation targets, averaged
across N seeds.

Run:
    python -m experiments.ablation
    python -m experiments.ablation --n-seeds 10 --out-file results/ablation.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfl.data import make_cohort
from cfl.trainer import FederatedTrainer, TrainerConfig


CONFIGS: dict[str, TrainerConfig] = {
    "fedsoft_plain": TrainerConfig(
        assignment="fedsoft",
        drift="none",
        fedsoft_defer_max_pi=None,
    ),
    "fedsoft_defer": TrainerConfig(
        assignment="fedsoft",
        drift="none",
        fedsoft_defer_max_pi=0.55,
    ),
    "hflts_no_defer": TrainerConfig(
        assignment="hflts",
        drift="none",
        hflts_defer_on_overlap=False,
    ),
    "hflts_defer": TrainerConfig(
        assignment="hflts",
        drift="none",
        hflts_defer_on_overlap=True,
    ),
}


def evaluate(cohort, config: TrainerConfig, n_rounds: int, seed: int) -> dict:
    trainer = FederatedTrainer(cohort=cohort, config=config, seed=seed)
    history = trainer.run(n_rounds)
    # All configurations run with drift disabled and the same cohort.
    return {
        "mean_acc": history[-1].mean_test_accuracy,
        "boundary_acc": history[-1].boundary_test_accuracy,
        "trajectory": np.array([r.mean_test_accuracy for r in history]),
    }


def run_sweep(
    n_seeds: int,
    n_clients: int,
    n_clusters: int,
    n_features: int,
    samples_per_client: int,
    boundary_fraction: float,
    n_rounds: int,
) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = {
        name: {"mean_acc": [], "boundary_acc": []} for name in CONFIGS
    }
    for seed in range(n_seeds):
        cohort = make_cohort(
            n_clients=n_clients,
            n_clusters=n_clusters,
            n_features=n_features,
            samples_per_client=samples_per_client,
            boundary_fraction=boundary_fraction,
            drift_round=None,
            seed=seed,
        )
        for name, cfg in CONFIGS.items():
            r = evaluate(cohort=cohort, config=cfg, n_rounds=n_rounds, seed=seed)
            out[name]["mean_acc"].append(r["mean_acc"])
            out[name]["boundary_acc"].append(r["boundary_acc"])
    return out


def format_markdown(results: dict[str, dict[str, list[float]]], n_seeds: int) -> str:
    rows = ["| config | mean_acc | boundary_acc |", "|---|---|---|"]

    def cell(values: list[float]) -> str:
        arr = np.array(values, dtype=float)
        return f"{arr.mean():.3f} ± {arr.std():.3f}"

    for name, m in results.items():
        rows.append(f"| {name} | {cell(m['mean_acc'])} | {cell(m['boundary_acc'])} |")
    return "\n".join(rows) + f"\n\n_aggregated over {n_seeds} seeds_"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--n-clients", type=int, default=30)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--n-features", type=int, default=5)
    parser.add_argument("--samples-per-client", type=int, default=120)
    parser.add_argument("--boundary-fraction", type=float, default=0.3)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()
    results = run_sweep(
        n_seeds=args.n_seeds,
        n_clients=args.n_clients,
        n_clusters=args.n_clusters,
        n_features=args.n_features,
        samples_per_client=args.samples_per_client,
        boundary_fraction=args.boundary_fraction,
        n_rounds=args.n_rounds,
    )
    md = format_markdown(results, args.n_seeds)
    print(
        f"ablation over {args.n_seeds} seeds | "
        f"{args.n_clients} clients, {args.n_clusters} clusters, "
        f"boundary={args.boundary_fraction:.0%}, no drift"
    )
    print()
    print(md)
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(md + "\n")
        print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
