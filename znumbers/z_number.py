"""Minimal Z-number primitives (Zadeh, Information Sciences 181(14):2923-2932, 2011).

A Z-number Z = (A, B) carries:
  - A: a fuzzy restriction on a variable X (the "what")
  - B: a fuzzy reliability of A (the "how sure")

Both A and B are represented here as triangular fuzzy numbers
(left, peak, right) on a common universe of discourse [0, 1]. This
is sufficient for the reliability-weighting transformation of
Kang, Wei, Li & Deng (Information Sciences 246:1-8, 2013) and the
score function used by the drift detector in this prototype.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TriangularFuzzyNumber:
    """Triangular fuzzy number with peak in [left, right]."""

    left: float
    peak: float
    right: float

    def __post_init__(self) -> None:
        if not (self.left <= self.peak <= self.right):
            raise ValueError(
                f"require left <= peak <= right, got "
                f"{self.left}, {self.peak}, {self.right}"
            )

    def centroid(self) -> float:
        """Defuzzified value (centroid of a triangle)."""
        return (self.left + self.peak + self.right) / 3.0

    def support_width(self) -> float:
        return self.right - self.left

    def scale_support(self, factor: float) -> "TriangularFuzzyNumber":
        """Scale the support around the peak by `factor`.

        Used in the Kang et al. 2013 conversion of a Z-number to an
        ordinary fuzzy number: support is scaled by sqrt(alpha) where
        alpha is the defuzzified reliability.
        """
        if factor < 0:
            raise ValueError("factor must be non-negative")
        new_left = self.peak - (self.peak - self.left) * factor
        new_right = self.peak + (self.right - self.peak) * factor
        return TriangularFuzzyNumber(left=new_left, peak=self.peak, right=new_right)


def reliability_to_alpha(B: TriangularFuzzyNumber) -> float:
    """Defuzzify a reliability triangular fuzzy number to a scalar in [0, 1].

    Uses the centroid; ``B`` is assumed to live on the universe [0, 1].
    """
    return max(0.0, min(1.0, B.centroid()))


@dataclass(frozen=True)
class ZNumber:
    """Z-number Z = (A, B) with triangular A and B on [0, 1].

    Provides a Kang et al. 2013 style conversion to a single
    reliability-weighted fuzzy number and a scalar score used for
    ranking (e.g. picking which client's drift signal to act on first).
    """

    A: TriangularFuzzyNumber
    B: TriangularFuzzyNumber

    def alpha(self) -> float:
        return reliability_to_alpha(self.B)

    def reliability_weighted(self) -> TriangularFuzzyNumber:
        """Z' = A scaled by sqrt(alpha) around its peak (Kang 2013)."""
        alpha = self.alpha()
        return self.A.scale_support(alpha**0.5)

    def score(self) -> float:
        """Scalar score for ranking: centroid(A) * alpha(B).

        High when both the restriction A points to high values AND the
        reliability is high. This is the canonical "reliability-weighted
        magnitude" used in drift triggers throughout this prototype.
        """
        return self.A.centroid() * self.alpha()


def weighted_aggregate(items: Iterable[ZNumber], weights: Iterable[float]) -> ZNumber:
    """Weighted aggregation of a sequence of Z-numbers.

    Aggregates A and B independently as weighted averages of the
    triangular parameters. The weight vector must sum to one.
    """
    items = list(items)
    weights = list(weights)
    if not items:
        raise ValueError("cannot aggregate empty sequence")
    if len(items) != len(weights):
        raise ValueError("items and weights length mismatch")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")

    def combine(attr: str, kind: str) -> float:
        return sum(w * getattr(getattr(z, kind), attr) for z, w in zip(items, weights))

    A = TriangularFuzzyNumber(
        left=combine("left", "A"),
        peak=combine("peak", "A"),
        right=combine("right", "A"),
    )
    B = TriangularFuzzyNumber(
        left=combine("left", "B"),
        peak=combine("peak", "B"),
        right=combine("right", "B"),
    )
    return ZNumber(A=A, B=B)
