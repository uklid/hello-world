from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .term_set import LinguisticTermSet


@dataclass(frozen=True)
class HFLTS:
    """Hesitant Fuzzy Linguistic Term Set.

    Represents an ordered, consecutive subset of a linguistic term set.
    The "consecutive" constraint follows the original HFLTS construction
    via the context-free grammar G_H (Rodriguez, Martinez & Herrera 2012);
    comparative expressions like "between s_2 and s_4" produce intervals,
    not arbitrary subsets.

    The envelope env(H) = [min(H), max(H)] is the basic interval-valued
    summary used by most HFLTS aggregation operators and distance measures.
    """

    term_set: LinguisticTermSet
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.indices:
            raise ValueError("HFLTS cannot be empty")
        sorted_indices = tuple(sorted(set(self.indices)))
        for i in sorted_indices:
            if i < 0 or i > self.term_set.g:
                raise ValueError(f"index {i} out of range [0, {self.term_set.g}]")
        # consecutive
        for a, b in zip(sorted_indices, sorted_indices[1:]):
            if b - a != 1:
                raise ValueError(
                    f"HFLTS indices must be consecutive, got gap between {a} and {b}"
                )
        object.__setattr__(self, "indices", sorted_indices)

    @classmethod
    def from_iterable(cls, term_set: LinguisticTermSet, items: Iterable[int]) -> "HFLTS":
        return cls(term_set=term_set, indices=tuple(items))

    @property
    def lower(self) -> int:
        return self.indices[0]

    @property
    def upper(self) -> int:
        return self.indices[-1]

    @property
    def envelope(self) -> tuple[int, int]:
        return (self.lower, self.upper)

    @property
    def cardinality(self) -> int:
        return len(self.indices)

    @property
    def is_single(self) -> bool:
        return self.cardinality == 1

    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    def hesitation_width(self) -> int:
        """Number of terms covered minus 1.

        Width 0 = no hesitation (single term).
        Width g = maximal hesitation (covers entire scale).
        """
        return self.upper - self.lower

    def terms(self) -> tuple[str, ...]:
        return tuple(self.term_set.name(i) for i in self.indices)

    def as_expression(self) -> str:
        """Render as a comparative expression for human inspection.

        Maps back to the grammar where possible: single-term, at_least,
        at_most, between. Renderers downstream (logs, plots) can use this
        directly.
        """
        if self.is_single:
            return self.term_set.name(self.lower)
        if self.lower == 0:
            return f"at_most {self.term_set.name(self.upper)}"
        if self.upper == self.term_set.g:
            return f"at_least {self.term_set.name(self.lower)}"
        return f"between {self.term_set.name(self.lower)} and {self.term_set.name(self.upper)}"

    def __repr__(self) -> str:
        return f"HFLTS({self.as_expression()!r})"
