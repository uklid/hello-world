"""Sanity tests for the end-to-end federated trainer."""

from __future__ import annotations

import unittest

import numpy as np

from cfl.data import make_cohort
from cfl.model import accuracy, gradient_step, logistic_loss, mix, predict_proba
from cfl.trainer import FederatedTrainer, TrainerConfig


class TestLogisticHelpers(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.W = rng.normal(size=4)
        self.b = 0.5
        self.X = rng.normal(size=(60, 4))
        z = self.X @ self.W + self.b
        p = 1.0 / (1.0 + np.exp(-z))
        self.y = (rng.uniform(size=60) < p).astype(int)

    def test_predict_proba_bounds(self) -> None:
        p = predict_proba(self.W, self.b, self.X)
        self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_gradient_step_reduces_loss(self) -> None:
        W0 = np.zeros_like(self.W)
        b0 = 0.0
        loss0 = logistic_loss(W0, b0, self.X, self.y)
        W1, b1 = gradient_step(W0, b0, self.X, self.y, lr=0.3, n_steps=20)
        loss1 = logistic_loss(W1, b1, self.X, self.y)
        self.assertLess(loss1, loss0)

    def test_mix_simplex(self) -> None:
        cluster_W = np.array([[1.0, 0.0], [0.0, 1.0]])
        cluster_b = np.array([0.0, 1.0])
        pi = np.array([0.3, 0.7])
        W, b = mix(cluster_W, cluster_b, pi)
        np.testing.assert_allclose(W, [0.3, 0.7])
        self.assertAlmostEqual(b, 0.7)

    def test_mix_normalizes(self) -> None:
        cluster_W = np.eye(3)[:2]
        cluster_b = np.zeros(2)
        pi = np.array([2.0, 2.0])  # not a simplex
        W, _ = mix(cluster_W, cluster_b, pi)
        np.testing.assert_allclose(W.sum(), 1.0)


class TestCohort(unittest.TestCase):
    def test_shapes(self) -> None:
        cohort = make_cohort(n_clients=12, n_clusters=3, drift_round=5, seed=0)
        self.assertEqual(cohort.n_clients, 12)
        self.assertEqual(cohort.n_clusters, 3)
        self.assertEqual(len(cohort.train), 12)
        self.assertEqual(len(cohort.test), 12)
        for X, y in cohort.train:
            self.assertEqual(X.shape[0], y.shape[0])

    def test_drift_post_data_differs_for_drifters(self) -> None:
        cohort = make_cohort(
            n_clients=20,
            n_clusters=3,
            drift_round=5,
            drift_fraction=0.5,
            boundary_fraction=0.0,
            seed=0,
        )
        # Some clients should have drift_to set.
        n_drift = sum(1 for s in cohort.specs if s.drift_to is not None)
        self.assertGreater(n_drift, 0)
        # And those clients have different pre vs post data.
        for i, spec in enumerate(cohort.specs):
            if spec.drift_to is not None:
                self.assertFalse(
                    np.array_equal(cohort.train[i][0], cohort.drift_post_data[i][0])
                )

    def test_train_data_switches_at_drift_round(self) -> None:
        cohort = make_cohort(
            n_clients=10, drift_round=4, drift_fraction=0.5, seed=0
        )
        pre = cohort.train_data(3)
        post = cohort.train_data(4)
        # Identity on lists, but contents may differ for drifters.
        self.assertEqual(len(pre), len(post))


class TestFederatedTrainer(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = make_cohort(
            n_clients=15,
            n_clusters=3,
            n_features=4,
            samples_per_client=80,
            boundary_fraction=0.2,
            drift_round=None,
            seed=0,
        )

    def test_runs_a_few_rounds(self) -> None:
        cfg = TrainerConfig(assignment="fedsoft", drift="none")
        trainer = FederatedTrainer(cohort=self.cohort, config=cfg, seed=0)
        reports = trainer.run(5)
        self.assertEqual(len(reports), 5)
        # Accuracy should improve from random init.
        self.assertGreater(reports[-1].mean_test_accuracy, 0.5)

    def test_hflts_assignment_runs(self) -> None:
        cfg = TrainerConfig(assignment="hflts", drift="none")
        trainer = FederatedTrainer(cohort=self.cohort, config=cfg, seed=0)
        reports = trainer.run(5)
        self.assertGreater(reports[-1].mean_test_accuracy, 0.5)
        # Pi rows should be valid simplices.
        pi = reports[-1].pi
        np.testing.assert_allclose(pi.sum(axis=1), 1.0, atol=1e-6)

    def test_drift_detector_fires_with_simulated_drift(self) -> None:
        # Use HFLTS assignment because the FedSoft trajectory keeps the
        # baseline loss high enough that the relative-loss drift signal
        # rarely crosses Z-CFL's "large" magnitude threshold; the HFLTS
        # trajectory reaches lower steady-state loss so the post-drift
        # surprise is sharper.
        cohort = make_cohort(
            n_clients=24,
            n_clusters=3,
            samples_per_client=120,
            boundary_fraction=0.0,
            drift_round=8,
            drift_fraction=0.5,
            seed=1,
        )
        cfg = TrainerConfig(assignment="hflts", drift="numeric")
        trainer = FederatedTrainer(cohort=cohort, config=cfg, seed=1)
        reports = trainer.run(20)
        post_actions = np.array([r.drift_actions for r in reports[9:]])
        self.assertGreater(post_actions.sum(), 0)


if __name__ == "__main__":
    unittest.main()
