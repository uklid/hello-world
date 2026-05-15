from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class LinguisticTermSet:
    """Ordered linguistic term set S = {s_0, s_1, ..., s_g}.

    Follows Rodriguez, Martinez & Herrera, IEEE TFS 20(1):109-119, 2012:
    terms are totally ordered, with index 0 representing the lowest and
    g the highest semantic value. The set must be symmetric for HFLTS
    operations defined in Liao, Xu & Zeng, Information Sciences 271:125-142,
    2014 (g should be even so that a strict middle term exists).
    """

    terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.terms) < 3:
            raise ValueError("term set needs at least 3 ordered terms")

    @property
    def g(self) -> int:
        return len(self.terms) - 1

    def index(self, term: str) -> int:
        return self.terms.index(term)

    def name(self, i: int) -> str:
        return self.terms[i]

    def clamp(self, i: int) -> int:
        return max(0, min(self.g, i))

    def all_indices(self) -> Sequence[int]:
        return range(0, self.g + 1)


DEFAULT_TERMS_7 = (
    "none",
    "very_low",
    "low",
    "medium",
    "high",
    "very_high",
    "perfect",
)


def default_term_set() -> LinguisticTermSet:
    return LinguisticTermSet(terms=DEFAULT_TERMS_7)
