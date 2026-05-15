"""Sweep term-set granularity x noise level for HFLTS-CFL.

Tests the hypothesis that the ARI gap vs FedSoft at medium noise is
quantization-limited: a finer term set should close the gap, at the
cost of less human-readable labels.

Run:
    python -m experiments.sweep_granularity
    python -m experiments.sweep_granularity --out-file results/granularity.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from cfl import FedSoftServer, HFLTSCFLServer
from experiments.synthetic_demo import make_clients, similarity_history
from hflts.term_set import uniform_term_set


def evaluate_grid(
    n_clients: int,
    n_clusters: int,
    boundary_fraction: float,
    n_rounds: int,
    noise_stds: list[float],
    granularities: list[int],
    seed: int,
    softmax_tau: float,
    confidence_z: float,
) -> list[dict]:
    rows = []
    for ns in noise_stds:
        rng = np.random.default_rng(seed)
        centroids, signatures, gt = make_clients(
            n_clients, n_clusters, boundary_fraction, seed
        )
        histories = [
            similarity_history(signatures[i], centroids, n_rounds, ns, rng)
            for i in range(n_clients)
        ]
        fedsoft = FedSoftServer(n_clusters=n_clusters, softmax_tau=softmax_tau)
        fedsoft_argmax = np.array(
            [int(np.argmax(fedsoft.assign(h))) for h in histories]
        )
        row = {
            "noise_std": ns,
            "ari_fedsoft": float(adjusted_rand_score(gt, fedsoft_argmax)),
        }
        for g_terms in granularities:
            ts = uniform_term_set(g_terms)
            server = HFLTSCFLServer(
                n_clusters=n_clusters,
                term_set=ts,
                confidence_z=confidence_z,
            )
            argmax = np.array(
                [
                    int(np.argmax(server.defuzzify(server.assign_hflts(h))))
                    for h in histories
                ]
            )
            row[f"ari_g{g_terms}"] = float(adjusted_rand_score(gt, argmax))
        rows.append(row)
    return rows


def format_markdown(rows: list[dict], granularities: list[int]) -> str:
    headers = ["noise", "FedSoft"] + [f"HFLTS g={g}" for g in granularities]
    keys = ["noise_std", "ari_fedsoft"] + [f"ari_g{g}" for g in granularities]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [f"{row['noise_std']:.2f}"]
        cells += [f"{row[k]:+.3f}" for k in keys[1:]]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-clients", type=int, default=30)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--boundary-fraction", type=float, default=0.4)
    parser.add_argument("--n-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--softmax-tau", type=float, default=0.25)
    parser.add_argument("--confidence-z", type=float, default=0.5)
    parser.add_argument(
        "--noise-stds",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15, 0.20, 0.30, 0.50],
    )
    parser.add_argument(
        "--granularities",
        type=int,
        nargs="+",
        default=[7, 11, 15, 21],
    )
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()
    rows = evaluate_grid(
        n_clients=args.n_clients,
        n_clusters=args.n_clusters,
        boundary_fraction=args.boundary_fraction,
        n_rounds=args.n_rounds,
        noise_stds=args.noise_stds,
        granularities=args.granularities,
        seed=args.seed,
        softmax_tau=args.softmax_tau,
        confidence_z=args.confidence_z,
    )
    md = format_markdown(rows, args.granularities)
    print(md)
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(md + "\n")
        print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
