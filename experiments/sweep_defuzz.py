"""Sweep defuzzification methods x noise levels for HFLTS-CFL.

Reports a Markdown table comparing:
- FedSoft baseline (softmax mixture weights)
- HFLTS-CFL with four defuzzification methods: midpoint, midpoint_softmax,
  upper, score_penalty

Run:
    python -m experiments.sweep_defuzz
    python -m experiments.sweep_defuzz --out-file results/sweep.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from cfl import FedSoftServer, HFLTSCFLServer
from experiments.synthetic_demo import make_clients, similarity_history
from hflts.term_set import default_term_set


METHODS = [
    ("midpoint", {}),
    ("midpoint_softmax", {"softmax_tau": 0.4}),
    ("upper", {}),
    ("score_penalty", {"hesitation_penalty": 0.5}),
]


def evaluate(
    n_clients: int,
    n_clusters: int,
    boundary_fraction: float,
    n_rounds: int,
    noise_std: float,
    seed: int,
    softmax_tau: float,
    confidence_z: float,
) -> dict:
    rng = np.random.default_rng(seed)
    centroids, signatures, gt = make_clients(
        n_clients, n_clusters, boundary_fraction, seed
    )
    fedsoft = FedSoftServer(n_clusters=n_clusters, softmax_tau=softmax_tau)
    hflts_server = HFLTSCFLServer(
        n_clusters=n_clusters,
        term_set=default_term_set(),
        confidence_z=confidence_z,
    )

    histories = [
        similarity_history(signatures[i], centroids, n_rounds, noise_std, rng)
        for i in range(n_clients)
    ]

    fedsoft_argmax = np.array(
        [int(np.argmax(fedsoft.assign(h))) for h in histories]
    )
    hflts_lists = [hflts_server.assign_hflts(h) for h in histories]

    per_method = {}
    for name, kwargs in METHODS:
        argmax = np.array(
            [
                int(np.argmax(hflts_server.defuzzify(hl, method=name, **kwargs)))
                for hl in hflts_lists
            ]
        )
        per_method[name] = float(adjusted_rand_score(gt, argmax))

    overlap_share = float(
        np.mean([hflts_server.has_overlap_ambiguity(hl) for hl in hflts_lists])
    )
    hesitation_share = float(
        np.mean(
            [
                1.0 - hl[int(np.argmax(hflts_server.defuzzify(hl)))].is_single
                for hl in hflts_lists
            ]
        )
    )
    return {
        "noise_std": noise_std,
        "ari_fedsoft": float(adjusted_rand_score(gt, fedsoft_argmax)),
        **{f"ari_hflts_{k}": v for k, v in per_method.items()},
        "overlap_share": overlap_share,
        "hesitation_share": hesitation_share,
    }


def format_markdown(rows: list[dict]) -> str:
    headers = [
        "noise",
        "FedSoft",
        "HFLTS midpoint",
        "HFLTS softmax",
        "HFLTS upper",
        "HFLTS score-pen",
        "hesitation",
        "overlap",
    ]
    keys = [
        "noise_std",
        "ari_fedsoft",
        "ari_hflts_midpoint",
        "ari_hflts_midpoint_softmax",
        "ari_hflts_upper",
        "ari_hflts_score_penalty",
        "hesitation_share",
        "overlap_share",
    ]
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = []
        for k in keys:
            v = row[k]
            if k in ("hesitation_share", "overlap_share"):
                cells.append(f"{v:.0%}")
            elif k == "noise_std":
                cells.append(f"{v:.2f}")
            else:
                cells.append(f"{v:+.3f}")
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
    parser.add_argument("--out-file", type=Path, default=None)
    args = parser.parse_args()

    rows = [
        evaluate(
            n_clients=args.n_clients,
            n_clusters=args.n_clusters,
            boundary_fraction=args.boundary_fraction,
            n_rounds=args.n_rounds,
            noise_std=ns,
            seed=args.seed,
            softmax_tau=args.softmax_tau,
            confidence_z=args.confidence_z,
        )
        for ns in args.noise_stds
    ]
    md = format_markdown(rows)
    print(md)
    if args.out_file:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(md + "\n")
        print(f"\nwrote {args.out_file}")


if __name__ == "__main__":
    main()
