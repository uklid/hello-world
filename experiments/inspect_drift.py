"""Diagnostic: per-round drift signals for drifters vs non-drifters."""

from __future__ import annotations

import numpy as np

from cfl.data import make_cohort
from cfl.trainer import FederatedTrainer, TrainerConfig
from znumbers import ZDriftDetector


def main() -> None:
    cohort = make_cohort(
        n_clients=30, n_clusters=3, boundary_fraction=0.3,
        drift_round=15, drift_fraction=0.3, seed=0,
    )
    drifter_ids = [i for i, s in enumerate(cohort.specs) if s.drift_to is not None]
    print(f"drifters: {drifter_ids}")
    z_det = ZDriftDetector()

    for assignment in ("fedsoft", "hflts"):
        cfg = TrainerConfig(assignment=assignment, drift="none")  # no drift triggers, just collect signals
        trainer = FederatedTrainer(cohort=cohort, config=cfg, seed=0)
        per_round_mags: list[list[float]] = [[] for _ in range(30)]
        per_round_agreement: list[list[float]] = [[] for _ in range(30)]
        for t in range(20):
            trainer.run_round(t)
            sigs = trainer._build_drift_signals()
            for i in range(30):
                per_round_mags[i].append(sigs[i].magnitude)
                per_round_agreement[i].append(sigs[i].neighbor_agreement)
        print(f"\n=== assignment={assignment} === (magnitudes per round, drifters only)")
        header = "rd:".rjust(7) + "".join(f"{t:>6}" for t in range(20))
        print(header)
        for i in drifter_ids:
            row = f"c{i:>3}D:".rjust(7) + "".join(
                f"{m:>6.2f}" for m in per_round_mags[i]
            )
            print(row)
        print("--- 5 non-drifters ---")
        for i in [j for j in range(30) if j not in drifter_ids][:5]:
            row = f"c{i:>3} :".rjust(7) + "".join(
                f"{m:>6.2f}" for m in per_round_mags[i]
            )
            print(row)

        # What action would Z-CFL pick at drift round (t=15)?
        print(f"\nat round 15 (drift round):")
        print(f"{'client':>6}  {'drift?':>6}  {'magn':>5}  {'agree':>5}  {'z_action':>12}  {'num>0.5':>7}")
        for i in [*drifter_ids, *[j for j in range(30) if j not in drifter_ids][:3]]:
            mag = per_round_mags[i][15]
            agr = per_round_agreement[i][15]
            from znumbers.drift import DriftSignal
            sig = DriftSignal(magnitude=mag, neighbor_agreement=agr, local_sample_size=120)
            action, _, _ = z_det.decide(sig)
            print(
                f"{i:>6}  {str(i in drifter_ids):>6}  {mag:>5.2f}  {agr:>5.2f}  "
                f"{action:>12}  {str(mag > 0.5):>7}"
            )


if __name__ == "__main__":
    main()
