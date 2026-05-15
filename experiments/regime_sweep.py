"""Regime stress test: sweep boundary_fraction across all 6 methods.

For each boundary_fraction in the sweep, run all 6 baseline methods
on the synthetic cohort (no drift) and report mean_acc / boundary_acc
/ ARI averaged over N seeds. Shows which regime each method
dominates, in particular whether HFLTS-CFL's edge on boundary
clients widens as the boundary fraction grows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from cfl.data import make_cohort
from cfl.trainer import FederatedTrainer, TrainerConfig


METHODS = ["fedavg", "local", "ifca", "sattler", "fedsoft", "hflts"]


def evaluate(cohort, method: str, n_rounds: int, seed: int) -> dict:
    cfg = TrainerConfig(assignment=method, drift="none")
    trainer = FederatedTrainer(cohort=cohort, config=cfg, seed=seed)
    history = trainer.run(n_rounds)
    pred = trainer.state.pi.argmax(axis=1)
    return {
        "mean_acc": history[-1].mean_test_accuracy,
        "boundary_acc": history[-1].boundary_test_accuracy,
        "ari": float(adjusted_rand_score(cohort.primary_label, pred)),
    }


def sweep(boundaries: list[float], n_seeds: int, n_rounds: int) -> dict:
    out: dict[float, dict[str, dict[str, list[float]]]] = {}
    for bf in boundaries:
        per_method = {m: {"mean_acc": [], "boundary_acc": [], "ari": []} for m in METHODS}
        for seed in range(n_seeds):
            cohort = make_cohort(
                boundary_fraction=bf,
                drift_round=None,
                seed=seed,
            )
            for m in METHODS:
                r = evaluate(cohort, m, n_rounds, seed)
                for k in per_method[m]:
                    per_method[m][k].append(r[k])
            print(f"  boundary={bf:.2f} seed={seed} done", flush=True)
        out[bf] = per_method
    return out


def format_table(metric: str, results: dict, boundaries: list[float]) -> str:
    rows = []
    headers = ["method"] + [f"b={b:.1f}" for b in boundaries]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("|" + "|".join(["---"] * len(headers)) + "|")
    for m in METHODS:
        cells = [m]
        for b in boundaries:
            arr = np.array(results[b][m][metric], dtype=float)
            cells.append(f"{arr.mean():.3f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boundaries",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4, 0.6, 0.8],
    )
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=20)
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()

    results = sweep(args.boundaries, args.n_seeds, args.n_rounds)
    pieces = [
        f"sweep over {args.n_seeds} seeds, synthetic cohort, no drift",
        "",
        "### mean_acc",
        format_table("mean_acc", results, args.boundaries),
        "",
        "### boundary_acc",
        format_table("boundary_acc", results, args.boundaries),
        "",
        "### ARI (cluster recovery vs primary_label)",
        format_table("ari", results, args.boundaries),
    ]
    md = "\n".join(pieces)
    print()
    print(md)
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(md + "\n")
        print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
