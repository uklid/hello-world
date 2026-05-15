"""Minimal Mamdani-style fuzzy rule system for re-clustering decisions.

Implements just enough machinery to express rules like:

    IF drift IS moderate AND reliability IS high THEN action IS re_cluster

The antecedent uses linguistic labels on the universe [0, 1] for both
the drift magnitude (centroid of A in the Z-number) and the reliability
(alpha = centroid of B). The consequent is a categorical action with a
fuzzy weight; rules fire in parallel and the action with the largest
aggregated firing strength wins.
"""

from __future__ import annotations

from dataclasses import dataclass

from .z_number import TriangularFuzzyNumber, ZNumber


@dataclass(frozen=True)
class LinguisticLabel:
    """Named triangular fuzzy set on [0, 1]."""

    name: str
    membership: TriangularFuzzyNumber

    def degree(self, x: float) -> float:
        m = self.membership
        if x <= m.left or x >= m.right:
            return 0.0
        if x == m.peak:
            return 1.0
        if x < m.peak:
            return (x - m.left) / max(m.peak - m.left, 1e-12)
        return (m.right - x) / max(m.right - m.peak, 1e-12)


def _make_labels(spec: list[tuple[str, float, float, float]]) -> dict[str, LinguisticLabel]:
    return {
        name: LinguisticLabel(
            name=name,
            membership=TriangularFuzzyNumber(left=l, peak=p, right=r),
        )
        for name, l, p, r in spec
    }


DRIFT_LABELS = _make_labels(
    [
        ("none", 0.0, 0.0, 0.2),
        ("small", 0.05, 0.2, 0.4),
        ("moderate", 0.25, 0.45, 0.65),
        ("large", 0.5, 0.7, 0.9),
        ("severe", 0.75, 1.0, 1.0),
    ]
)

RELIABILITY_LABELS = _make_labels(
    [
        ("low", 0.0, 0.0, 0.4),
        ("medium", 0.2, 0.5, 0.8),
        ("high", 0.6, 1.0, 1.0),
    ]
)


@dataclass(frozen=True)
class FuzzyRule:
    drift_label: str
    reliability_label: str
    action: str
    weight: float = 1.0

    def firing_strength(self, drift_x: float, reliability_x: float) -> float:
        d = DRIFT_LABELS[self.drift_label].degree(drift_x)
        r = RELIABILITY_LABELS[self.reliability_label].degree(reliability_x)
        # min t-norm for AND
        return self.weight * min(d, r)


def default_drift_rules() -> list[FuzzyRule]:
    """Default rules covering every (drift, reliability) cell.

    Gap C motivation: re-cluster only when both the magnitude is high
    and the reliability is high. Downgrade to ``observe`` when one of
    the two is medium, and ``defer`` when reliability is low. The
    severe+medium and severe+low cells are explicit (a previous
    version of these rules left them undefined, which let mag=1.0
    + low-agreement signals fall through to the implicit no_action
    default).
    """
    return [
        FuzzyRule("severe", "high", "re_cluster"),
        FuzzyRule("severe", "medium", "observe"),
        FuzzyRule("severe", "low", "defer"),
        FuzzyRule("large", "high", "re_cluster"),
        FuzzyRule("large", "medium", "observe"),
        FuzzyRule("large", "low", "defer"),
        FuzzyRule("moderate", "high", "observe"),
        FuzzyRule("moderate", "medium", "observe"),
        FuzzyRule("moderate", "low", "defer"),
        FuzzyRule("small", "high", "no_action"),
        FuzzyRule("small", "medium", "no_action"),
        FuzzyRule("small", "low", "no_action"),
        FuzzyRule("none", "high", "no_action"),
        FuzzyRule("none", "medium", "no_action"),
        FuzzyRule("none", "low", "no_action"),
    ]


@dataclass
class MamdaniSystem:
    rules: list[FuzzyRule]

    def evaluate(self, z: ZNumber) -> tuple[str, dict[str, float]]:
        """Return the winning action and the firing-strength map."""
        drift_x = z.A.centroid()
        reliability_x = z.alpha()
        strengths: dict[str, float] = {}
        for rule in self.rules:
            s = rule.firing_strength(drift_x, reliability_x)
            if s <= 0:
                continue
            strengths[rule.action] = max(strengths.get(rule.action, 0.0), s)
        if not strengths:
            return "no_action", {}
        winner = max(strengths.items(), key=lambda kv: kv[1])[0]
        return winner, strengths
