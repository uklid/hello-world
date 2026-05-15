from __future__ import annotations

from .hflts import HFLTS


def _extend_to_length(
    hflts: HFLTS, length: int, optimism: float
) -> list[int]:
    """Extend an HFLTS to a target length.

    Liao, Xu & Zeng (Information Sciences 271:125-142, 2014) define the
    optimism parameter beta in [0, 1]: the added element is
        x_add = beta * max(H) + (1 - beta) * min(H)
    then sorted into the sequence. beta=1 is optimist (use the upper),
    beta=0 is pessimist (use the lower), beta=0.5 is neutral.
    """
    if not 0.0 <= optimism <= 1.0:
        raise ValueError("optimism must be in [0, 1]")
    seq = list(hflts.indices)
    if len(seq) > length:
        raise ValueError("HFLTS already longer than requested length")
    if len(seq) == length:
        return seq
    add_value = round(optimism * hflts.upper + (1.0 - optimism) * hflts.lower)
    add_value = hflts.term_set.clamp(add_value)
    while len(seq) < length:
        seq.append(add_value)
    seq.sort()
    return seq


def hflts_distance(a: HFLTS, b: HFLTS, *, optimism: float = 0.5) -> float:
    """Normalized Hamming distance between two HFLTS.

    Implements eq. (5) of Liao, Xu & Zeng 2014: align the two sets to a
    common length via the optimism parameter, then average normalized
    index differences. Result is in [0, 1].
    """
    if a.term_set is not b.term_set:
        raise ValueError("HFLTS must share the same linguistic term set")
    length = max(a.cardinality, b.cardinality)
    extended_a = _extend_to_length(a, length, optimism)
    extended_b = _extend_to_length(b, length, optimism)
    g = a.term_set.g
    total = sum(abs(x - y) for x, y in zip(extended_a, extended_b))
    return total / (length * g)


def hflts_similarity(a: HFLTS, b: HFLTS, *, optimism: float = 0.5) -> float:
    """1 - normalized distance."""
    return 1.0 - hflts_distance(a, b, optimism=optimism)
