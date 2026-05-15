"""Unit tests for the HFLTS core library.

Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from cfl import HFLTSCFLServer, similarity_to_hflts
from hflts import (
    HFLTS,
    LinguisticTermSet,
    at_least,
    at_most,
    between,
    exactly,
    greater_than,
    hflowa,
    hflts_distance,
    hflts_similarity,
    hflwa,
    lower_than,
)
from hflts.term_set import default_term_set


class TestLinguisticTermSet(unittest.TestCase):
    def test_default_set_size(self) -> None:
        ts = default_term_set()
        self.assertEqual(ts.g, 6)
        self.assertEqual(len(ts.terms), 7)

    def test_rejects_tiny_set(self) -> None:
        with self.assertRaises(ValueError):
            LinguisticTermSet(terms=("low", "high"))

    def test_clamp(self) -> None:
        ts = default_term_set()
        self.assertEqual(ts.clamp(-2), 0)
        self.assertEqual(ts.clamp(99), ts.g)
        self.assertEqual(ts.clamp(3), 3)


class TestHFLTSConstruction(unittest.TestCase):
    def setUp(self) -> None:
        self.ts = default_term_set()

    def test_requires_consecutive(self) -> None:
        with self.assertRaises(ValueError):
            HFLTS(term_set=self.ts, indices=(1, 3))

    def test_envelope_and_width(self) -> None:
        h = between(self.ts, 2, 4)
        self.assertEqual(h.envelope, (2, 4))
        self.assertEqual(h.cardinality, 3)
        self.assertEqual(h.hesitation_width(), 2)
        self.assertFalse(h.is_single)

    def test_single_term(self) -> None:
        h = exactly(self.ts, 3)
        self.assertTrue(h.is_single)
        self.assertEqual(h.hesitation_width(), 0)
        self.assertEqual(h.midpoint(), 3.0)


class TestGrammar(unittest.TestCase):
    def setUp(self) -> None:
        self.ts = default_term_set()

    def test_at_least(self) -> None:
        h = at_least(self.ts, 4)
        self.assertEqual(h.envelope, (4, 6))
        self.assertEqual(h.as_expression(), "at_least high")

    def test_at_most(self) -> None:
        h = at_most(self.ts, 2)
        self.assertEqual(h.envelope, (0, 2))
        self.assertEqual(h.as_expression(), "at_most low")

    def test_between_unordered(self) -> None:
        h = between(self.ts, 4, 2)
        self.assertEqual(h.envelope, (2, 4))

    def test_greater_lower_than(self) -> None:
        h = greater_than(self.ts, 2)
        self.assertEqual(h.lower, 3)
        h2 = lower_than(self.ts, 4)
        self.assertEqual(h2.upper, 3)
        with self.assertRaises(ValueError):
            greater_than(self.ts, self.ts.g)
        with self.assertRaises(ValueError):
            lower_than(self.ts, 0)


class TestDistance(unittest.TestCase):
    def setUp(self) -> None:
        self.ts = default_term_set()

    def test_distance_to_self_zero(self) -> None:
        h = between(self.ts, 2, 4)
        self.assertAlmostEqual(hflts_distance(h, h), 0.0)
        self.assertAlmostEqual(hflts_similarity(h, h), 1.0)

    def test_distance_symmetric(self) -> None:
        a = at_least(self.ts, 4)
        b = at_most(self.ts, 2)
        self.assertAlmostEqual(
            hflts_distance(a, b),
            hflts_distance(b, a),
        )

    def test_distance_bounds(self) -> None:
        a = exactly(self.ts, 0)
        b = exactly(self.ts, self.ts.g)
        self.assertAlmostEqual(hflts_distance(a, b), 1.0)
        self.assertAlmostEqual(hflts_similarity(a, b), 0.0)

    def test_optimism_changes_extension(self) -> None:
        # a has lower < upper so the optimism parameter affects how the
        # shorter set is extended; the target b is shifted upward.
        a = between(self.ts, 1, 3)
        b = between(self.ts, 2, 5)
        d_pess = hflts_distance(a, b, optimism=0.0)
        d_opt = hflts_distance(a, b, optimism=1.0)
        self.assertLess(d_opt, d_pess)


class TestAggregation(unittest.TestCase):
    def setUp(self) -> None:
        self.ts = default_term_set()

    def test_hflwa_preserves_term_set(self) -> None:
        items = [exactly(self.ts, 2), exactly(self.ts, 4)]
        h = hflwa(items, [0.5, 0.5])
        self.assertIs(h.term_set, self.ts)
        self.assertEqual(h.midpoint(), 3.0)

    def test_hflwa_envelope_widening(self) -> None:
        items = [at_least(self.ts, 4), at_most(self.ts, 2)]
        h = hflwa(items, [0.5, 0.5])
        self.assertGreaterEqual(h.hesitation_width(), 2)

    def test_hflowa_reorders_descending(self) -> None:
        items = [exactly(self.ts, 1), exactly(self.ts, 5)]
        h = hflowa(items, [0.8, 0.2])
        # Reordering: descending midpoint => [5, 1]; weighted = 0.8*5+0.2*1 = 4.2 -> 4
        self.assertEqual(h.midpoint(), 4.0)

    def test_weights_must_sum_to_one(self) -> None:
        with self.assertRaises(ValueError):
            hflwa([exactly(self.ts, 1), exactly(self.ts, 2)], [0.3, 0.3])


class TestSimilarityToHFLTS(unittest.TestCase):
    def setUp(self) -> None:
        self.ts = default_term_set()

    def test_low_variance_yields_narrow_envelope(self) -> None:
        history = np.full(8, 0.8)
        h = similarity_to_hflts(history, self.ts)
        self.assertEqual(h.hesitation_width(), 0)

    def test_high_variance_widens_envelope(self) -> None:
        history = np.array([0.2, 0.95, 0.3, 0.85, 0.4, 0.9, 0.25, 0.8])
        h = similarity_to_hflts(history, self.ts)
        self.assertGreater(h.hesitation_width(), 0)


class TestHFLTSCFLServer(unittest.TestCase):
    def setUp(self) -> None:
        self.ts = default_term_set()
        self.server = HFLTSCFLServer(n_clusters=3, term_set=self.ts)

    def test_assign_returns_one_hflts_per_cluster(self) -> None:
        history = np.array(
            [
                [0.9, 0.2, 0.1],
                [0.85, 0.25, 0.05],
                [0.92, 0.18, 0.08],
            ]
        )
        out = self.server.assign_hflts(history)
        self.assertEqual(len(out), 3)
        weights = self.server.defuzzify(out)
        self.assertAlmostEqual(weights.sum(), 1.0)
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[0], weights[2])

    def test_overlap_detection(self) -> None:
        history = np.array(
            [
                [0.55, 0.50, 0.10],
                [0.60, 0.45, 0.05],
                [0.50, 0.55, 0.08],
                [0.58, 0.52, 0.10],
            ]
        )
        out = self.server.assign_hflts(history)
        self.assertTrue(self.server.has_overlap_ambiguity(out))


if __name__ == "__main__":
    unittest.main()
