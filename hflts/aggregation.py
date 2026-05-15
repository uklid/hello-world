from __future__ import annotations

from collections.abc import Sequence

from .hflts import HFLTS
from .term_set import LinguisticTermSet


def envelope_midpoint(h: HFLTS) -> float:
    return h.midpoint()


def _check_shared_term_set(items: Sequence[HFLTS]) -> LinguisticTermSet:
    if not items:
        raise ValueError("cannot aggregate empty sequence")
    ts = items[0].term_set
    for h in items[1:]:
        if h.term_set is not ts:
            raise ValueError("all HFLTS must share the same linguistic term set")
    return ts


def hflwa(items: Sequence[HFLTS], weights: Sequence[float]) -> HFLTS:
    """Hesitant Fuzzy Linguistic Weighted Average via envelope midpoints.

    Operates on envelopes (lower, upper) rather than full sets:
        out_lower = round(sum w_i * lower_i)
        out_upper = round(sum w_i * upper_i)
    This is the simplest HFLWA variant; it is lossy compared to
    full-set operators (Wei, Zhao & Tang 2014) but suffices for the
    soft cluster assignment use-case where downstream training only
    needs a defuzzified weight anyway.
    """
    ts = _check_shared_term_set(items)
    if len(items) != len(weights):
        raise ValueError("items and weights length mismatch")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")
    lower = round(sum(w * h.lower for h, w in zip(items, weights)))
    upper = round(sum(w * h.upper for h, w in zip(items, weights)))
    lower = ts.clamp(lower)
    upper = ts.clamp(upper)
    if upper < lower:
        upper = lower
    return HFLTS(term_set=ts, indices=tuple(range(lower, upper + 1)))


def hflowa(items: Sequence[HFLTS], weights: Sequence[float]) -> HFLTS:
    """Hesitant Fuzzy Linguistic Ordered Weighted Average.

    Reorders items by descending envelope midpoint, then applies the
    weight vector. Mirrors Yager-style OWA semantics in the linguistic
    domain.
    """
    ts = _check_shared_term_set(items)
    if len(items) != len(weights):
        raise ValueError("items and weights length mismatch")
    ordered = sorted(items, key=envelope_midpoint, reverse=True)
    return hflwa(ordered, list(weights))
