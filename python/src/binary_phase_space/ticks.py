"""Helpers for picking nice round-number tick positions."""

from __future__ import annotations

import numpy as np


_NICE_MANTISSAS = (1, 2, 5)


def _nice_step(approx_step: float) -> float:
    """Round a step size to the nearest 1-2-5 value (in log space)."""
    if approx_step <= 0:
        return 1.0
    exp = int(np.floor(np.log10(approx_step)))
    base = 10.0 ** exp
    # Candidates at this decade and the next one up
    options = [m * base for m in (1, 2, 5)] + [10.0 * base]
    return min(options, key=lambda v: abs(np.log(v / approx_step)))


def nice_symmetric_ticks(limit: float, target_count: int = 3) -> tuple[float, ...]:
    """Choose round-number ticks symmetric about zero, inside ±limit.

    Returns a tuple of approximately ``target_count`` ticks at 1-2-5 values
    within ±0.85 * limit.
    """
    if limit <= 0:
        return (0.0,)
    # Try (1, 2, 5) × 10^k for k in a reasonable range; pick the largest
    # that fits inside the window with 15% margin.
    candidates = []
    for exp in range(-3, 4):
        base = 10 ** exp
        for mantissa in _NICE_MANTISSAS:
            candidates.append(mantissa * base)
    a = max((c for c in candidates if c <= 0.85 * limit), default=round(limit, 2))
    if target_count >= 3:
        return (-a, 0.0, a)
    return (-a, a)


def nice_ticks_around(
    center: float,
    half_width: float,
    target_count: int = 3,
) -> tuple[float, ...]:
    """Choose round-number ticks within ``center ± half_width``.

    Picks a step from the 1-2-5 sequence such that approximately
    ``target_count`` ticks land inside the window. Allows ticks anywhere
    inside [center - half_width, center + half_width] so the leftmost and
    rightmost ticks can sit near the panel edges.
    """
    if half_width <= 0:
        return (float(round(center, 2)),)

    lo = center - half_width
    hi = center + half_width

    # Try each 1-2-5 step at scales straddling 2*half_width/target_count.
    approx_step = 2 * half_width / max(target_count - 1, 1)
    candidates = []
    for exp in range(-4, 5):
        base = 10.0 ** exp
        for mantissa in (1, 2, 5):
            candidates.append(mantissa * base)
    # Keep candidates whose step would yield at least one tick.
    candidates = [c for c in candidates if c > 0]

    def count_ticks(step: float) -> tuple[int, list[float]]:
        first = np.ceil(lo / step) * step
        ticks: list[float] = []
        t = first
        while t <= hi + 1e-9:
            ticks.append(float(round(t, 6)))
            t += step
        return len(ticks), ticks

    # Pick the step that gives a count closest to target_count, breaking
    # ties toward the larger step (fewer ticks = less clutter).
    best_step = min(
        candidates,
        key=lambda s: (abs(count_ticks(s)[0] - target_count), -s),
    )
    _, ticks = count_ticks(best_step)
    return tuple(ticks) if ticks else (float(round(center, 2)),)
