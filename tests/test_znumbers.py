"""Unit tests for the Z-number library and the drift detector."""

from __future__ import annotations

import unittest

import numpy as np  # noqa: F401  - imported for parity with other tests

from znumbers import (
    MamdaniSystem,
    ZDriftDetector,
    ZNumber,
    default_drift_rules,
    reliability_to_alpha,
)
from znumbers.drift import (
    DriftSignal,
    numeric_threshold_baseline,
    saturating_sample_score,
    simulate_clients,
)
from znumbers.z_number import TriangularFuzzyNumber, weighted_aggregate


def make_z(peak_a: float, peak_b: float) -> ZNumber:
    return ZNumber(
        A=TriangularFuzzyNumber(peak_a - 0.1, peak_a, min(1.0, peak_a + 0.1)),
        B=TriangularFuzzyNumber(peak_b - 0.1, peak_b, min(1.0, peak_b + 0.1)),
    )


class TestTriangular(unittest.TestCase):
    def test_centroid(self) -> None:
        t = TriangularFuzzyNumber(0.2, 0.5, 0.8)
        self.assertAlmostEqual(t.centroid(), 0.5)

    def test_scale_support(self) -> None:
        t = TriangularFuzzyNumber(0.0, 0.5, 1.0)
        s = t.scale_support(0.5)
        self.assertAlmostEqual(s.left, 0.25)
        self.assertAlmostEqual(s.right, 0.75)
        self.assertAlmostEqual(s.peak, 0.5)

    def test_rejects_misordered(self) -> None:
        with self.assertRaises(ValueError):
            TriangularFuzzyNumber(0.6, 0.5, 0.8)


class TestZNumber(unittest.TestCase):
    def test_score_increases_with_reliability(self) -> None:
        z_low = make_z(0.5, 0.2)
        z_high = make_z(0.5, 0.9)
        self.assertLess(z_low.score(), z_high.score())

    def test_reliability_weighted_shrinks_support(self) -> None:
        z = make_z(0.7, 0.3)
        z_prime = z.reliability_weighted()
        self.assertLessEqual(z_prime.support_width(), z.A.support_width())

    def test_weighted_aggregate(self) -> None:
        z1 = make_z(0.2, 0.8)
        z2 = make_z(0.8, 0.4)
        agg = weighted_aggregate([z1, z2], [0.5, 0.5])
        self.assertAlmostEqual(agg.A.peak, 0.5)
        self.assertAlmostEqual(agg.B.peak, 0.6)

    def test_reliability_to_alpha_clamps(self) -> None:
        self.assertEqual(
            reliability_to_alpha(TriangularFuzzyNumber(0.0, 0.0, 0.0)), 0.0
        )
        self.assertEqual(
            reliability_to_alpha(TriangularFuzzyNumber(1.0, 1.0, 1.0)), 1.0
        )


class TestRules(unittest.TestCase):
    def test_high_drift_high_reliability_triggers(self) -> None:
        sys = MamdaniSystem(rules=default_drift_rules())
        action, _ = sys.evaluate(make_z(0.75, 0.9))
        self.assertEqual(action, "re_cluster")

    def test_high_drift_low_reliability_defers(self) -> None:
        sys = MamdaniSystem(rules=default_drift_rules())
        action, _ = sys.evaluate(make_z(0.75, 0.1))
        self.assertIn(action, {"observe", "defer"})

    def test_no_drift_high_reliability_no_action(self) -> None:
        sys = MamdaniSystem(rules=default_drift_rules())
        action, _ = sys.evaluate(make_z(0.05, 0.9))
        self.assertEqual(action, "no_action")


class TestDriftDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.det = ZDriftDetector()

    def test_high_mag_high_reliability_triggers(self) -> None:
        sig = DriftSignal(
            magnitude=0.8,
            neighbor_agreement=0.9,
            local_sample_size=300,
        )
        action, z, strengths = self.det.decide(sig)
        self.assertEqual(action, "re_cluster")
        self.assertGreater(z.alpha(), 0.5)

    def test_high_mag_low_reliability_defers(self) -> None:
        sig = DriftSignal(
            magnitude=0.8,
            neighbor_agreement=0.1,
            local_sample_size=10,
        )
        action, _, _ = self.det.decide(sig)
        self.assertNotEqual(action, "re_cluster")

    def test_baseline_ignores_reliability(self) -> None:
        sig = DriftSignal(
            magnitude=0.8,
            neighbor_agreement=0.1,
            local_sample_size=10,
        )
        self.assertEqual(numeric_threshold_baseline(sig), "re_cluster")


class TestSampleScore(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(saturating_sample_score(0), 0.0)

    def test_saturates(self) -> None:
        self.assertEqual(saturating_sample_score(1000), 1.0)

    def test_monotone(self) -> None:
        a = saturating_sample_score(50)
        b = saturating_sample_score(150)
        self.assertLess(a, b)


class TestSimulator(unittest.TestCase):
    def test_shape(self) -> None:
        signals, truth = simulate_clients(
            n_clients=4,
            n_rounds=6,
            drift_round=3,
            drift_magnitude_post=0.7,
            base_noise=0.05,
            seed=0,
        )
        self.assertEqual(len(signals), 4)
        self.assertEqual(len(truth), 4)
        for s, t in zip(signals, truth):
            self.assertEqual(len(s), 6)
            self.assertEqual(len(t), 6)

    def test_low_conf_clients_never_have_recluster_truth(self) -> None:
        _, truth = simulate_clients(
            n_clients=4,
            n_rounds=10,
            drift_round=5,
            drift_magnitude_post=0.7,
            base_noise=0.05,
            seed=0,
        )
        # Odd-indexed clients are low-confidence; ground truth must
        # never recommend re_cluster for them.
        for i in (1, 3):
            self.assertTrue(all(t == "no_action" for t in truth[i]))


if __name__ == "__main__":
    unittest.main()
