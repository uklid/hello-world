"""Synthetic demo: HFLTS-CFL vs FedSoft-style baseline.

Setup
-----
- K=3 ground-truth client clusters, each with its own data-generating
  distribution (2-D Gaussian mixture for visualization).
- N=30 clients. Each client is sampled from one of the K distributions,
  but a controlled fraction `boundary_fraction` is placed near a cluster
  boundary to elicit hesitation.
- The "similarity signal" per round per cluster is the negated L2 of the
  client signature to the cluster centroid, scaled to [0, 1] via a
  monotone squashing. We inject Gaussian noise per round to mimic the
  noisy-similarity regime that motivates Gap A.

Comparison
----------
- FedSoftServer: mean-then-softmax mixture weight pi.
- HFLTSCFLServer: per-cluster HFLTS envelope, defuzzified into a mixture
  weight by midpoint normalization. Reports comparative expressions and
  flags overlap ambiguity.

Metrics
-------
- ARI of hard-argmax cluster assignment vs ground truth.
- Hesitation share: fraction of clients whose HFLTS for the chosen
  cluster is non-singleton (interpretability proxy).
- Defer-rate: fraction flagged by has_overlap_ambiguity.

Run:
    python -m experiments.synthetic_demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from cfl import FedSoftServer, HFLTSCFLServer
from hflts.term_set import default_term_set


RNG = np.random.default_rng


def make_clients(
    n_clients: int,
    n_clusters: int,
    boundary_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroids, client_signatures, ground_truth_cluster).

    Centroids placed on a circle. Non-boundary clients sit near one
    centroid; boundary clients sit on the chord between two centroids
    plus small noise.
    """
    rng = RNG(seed)
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centroids = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 2.0

    signatures = np.zeros((n_clients, 2))
    labels = np.zeros(n_clients, dtype=int)
    n_boundary = int(round(boundary_fraction * n_clients))
    is_boundary = np.zeros(n_clients, dtype=bool)
    is_boundary[:n_boundary] = True
    rng.shuffle(is_boundary)

    for i in range(n_clients):
        if is_boundary[i]:
            a, b = rng.choice(n_clusters, size=2, replace=False)
            t = rng.uniform(0.35, 0.65)
            signatures[i] = (1 - t) * centroids[a] + t * centroids[b]
            signatures[i] += rng.normal(scale=0.15, size=2)
            labels[i] = a if t < 0.5 else b
        else:
            c = int(rng.integers(0, n_clusters))
            signatures[i] = centroids[c] + rng.normal(scale=0.25, size=2)
            labels[i] = c
    return centroids, signatures, labels


def similarity_history(
    signature: np.ndarray,
    centroids: np.ndarray,
    n_rounds: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """(T, K) similarity history, normalized to [0, 1]."""
    base_dist = np.linalg.norm(signature[None, :] - centroids, axis=1)
    sigma = 2.0
    base_sim = np.exp(-base_dist / sigma)  # in (0, 1]
    noise = rng.normal(scale=noise_std, size=(n_rounds, base_sim.size))
    history = base_sim[None, :] + noise
    return np.clip(history, 0.0, 1.0)


def run(
    n_clients: int = 30,
    n_clusters: int = 3,
    boundary_fraction: float = 0.4,
    n_rounds: int = 8,
    noise_std: float = 0.15,
    seed: int = 0,
    softmax_tau: float = 0.25,
    confidence_z: float = 1.0,
    out_dir: Path | None = None,
) -> dict:
    rng = RNG(seed)
    centroids, signatures, gt = make_clients(
        n_clients, n_clusters, boundary_fraction, seed
    )

    fedsoft = FedSoftServer(n_clusters=n_clusters, softmax_tau=softmax_tau)
    hflts_server = HFLTSCFLServer(
        n_clusters=n_clusters,
        term_set=default_term_set(),
        confidence_z=confidence_z,
    )

    fedsoft_argmax = np.zeros(n_clients, dtype=int)
    hflts_argmax = np.zeros(n_clients, dtype=int)
    hesitation_flags = np.zeros(n_clients, dtype=bool)
    overlap_flags = np.zeros(n_clients, dtype=bool)
    sample_rows = []

    for i in range(n_clients):
        history = similarity_history(
            signatures[i], centroids, n_rounds, noise_std, rng
        )
        pi = fedsoft.assign(history)
        fedsoft_argmax[i] = int(np.argmax(pi))

        hflts_list = hflts_server.assign_hflts(history)
        weights = hflts_server.defuzzify(hflts_list)
        chosen = int(np.argmax(weights))
        hflts_argmax[i] = chosen
        hesitation_flags[i] = not hflts_list[chosen].is_single
        overlap_flags[i] = hflts_server.has_overlap_ambiguity(hflts_list)

        sample_rows.append(
            {
                "client": i,
                "gt": int(gt[i]),
                "fedsoft_pi": [round(float(x), 3) for x in pi],
                "hflts_envelope": hflts_server.describe(hflts_list),
                "hflts_weights": [round(float(x), 3) for x in weights],
                "overlap_flag": bool(overlap_flags[i]),
            }
        )

    ari_fedsoft = adjusted_rand_score(gt, fedsoft_argmax)
    ari_hflts = adjusted_rand_score(gt, hflts_argmax)
    metrics = {
        "ari_fedsoft": float(ari_fedsoft),
        "ari_hflts": float(ari_hflts),
        "hesitation_share": float(hesitation_flags.mean()),
        "overlap_share": float(overlap_flags.mean()),
        "n_clients": n_clients,
        "boundary_fraction": boundary_fraction,
        "n_rounds": n_rounds,
        "noise_std": noise_std,
    }

    print("=" * 64)
    print("HFLTS-CFL vs FedSoft baseline (synthetic)")
    print("=" * 64)
    print(f"clients={n_clients}, clusters={n_clusters}, "
          f"boundary={boundary_fraction:.2f}, rounds={n_rounds}, "
          f"noise_std={noise_std:.2f}")
    print(f"ARI FedSoft  : {ari_fedsoft:+.4f}")
    print(f"ARI HFLTS-CFL: {ari_hflts:+.4f}")
    print(f"Hesitation share (HFLTS): {metrics['hesitation_share']:.2%}")
    print(f"Overlap-ambiguity share : {metrics['overlap_share']:.2%}")
    print("-" * 64)
    print("Sample assignments (first 8 clients):")
    for row in sample_rows[:8]:
        print(
            f"  client {row['client']:>2}  gt={row['gt']}  "
            f"FedSoft pi={row['fedsoft_pi']}  "
            f"HFLTS={row['hflts_envelope']}  "
            f"overlap={row['overlap_flag']}"
        )

    if out_dir is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_dir.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(6, 6))
            for k in range(n_clusters):
                ax.scatter(*centroids[k], marker="X", s=200, label=f"centroid {k}")
            for i in range(n_clients):
                color = ["C0", "C1", "C2", "C3", "C4"][gt[i] % 5]
                marker = "o" if not overlap_flags[i] else "D"
                ax.scatter(
                    signatures[i, 0],
                    signatures[i, 1],
                    c=color,
                    marker=marker,
                    edgecolors="black" if overlap_flags[i] else "none",
                    s=70,
                    alpha=0.85,
                )
            ax.set_title(
                f"Synthetic CFL signatures\n"
                f"diamonds = HFLTS flagged overlap "
                f"({metrics['overlap_share']:.0%})"
            )
            ax.legend()
            ax.set_aspect("equal")
            fig.tight_layout()
            fig.savefig(out_dir / "signatures.png", dpi=120)
            plt.close(fig)
            print(f"saved plot -> {out_dir / 'signatures.png'}")
        except Exception as exc:  # pragma: no cover - plotting optional
            print(f"plotting skipped: {exc}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-clients", type=int, default=30)
    parser.add_argument("--n-clusters", type=int, default=3)
    parser.add_argument("--boundary-fraction", type=float, default=0.4)
    parser.add_argument("--n-rounds", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--softmax-tau", type=float, default=0.25)
    parser.add_argument("--confidence-z", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    run(
        n_clients=args.n_clients,
        n_clusters=args.n_clusters,
        boundary_fraction=args.boundary_fraction,
        n_rounds=args.n_rounds,
        noise_std=args.noise_std,
        seed=args.seed,
        softmax_tau=args.softmax_tau,
        confidence_z=args.confidence_z,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
