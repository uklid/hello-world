from __future__ import annotations

from .hflts import HFLTS
from .term_set import LinguisticTermSet


def exactly(term_set: LinguisticTermSet, i: int) -> HFLTS:
    return HFLTS(term_set=term_set, indices=(i,))


def at_least(term_set: LinguisticTermSet, i: int) -> HFLTS:
    return HFLTS(term_set=term_set, indices=tuple(range(i, term_set.g + 1)))


def at_most(term_set: LinguisticTermSet, i: int) -> HFLTS:
    return HFLTS(term_set=term_set, indices=tuple(range(0, i + 1)))


def between(term_set: LinguisticTermSet, i: int, j: int) -> HFLTS:
    if i > j:
        i, j = j, i
    return HFLTS(term_set=term_set, indices=tuple(range(i, j + 1)))


def greater_than(term_set: LinguisticTermSet, i: int) -> HFLTS:
    if i >= term_set.g:
        raise ValueError("greater_than is undefined for the maximum term")
    return HFLTS(term_set=term_set, indices=tuple(range(i + 1, term_set.g + 1)))


def lower_than(term_set: LinguisticTermSet, i: int) -> HFLTS:
    if i <= 0:
        raise ValueError("lower_than is undefined for the minimum term")
    return HFLTS(term_set=term_set, indices=tuple(range(0, i)))
