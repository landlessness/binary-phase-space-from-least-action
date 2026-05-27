"""Signed Stern-Brocot partition of a normalized coordinate.

This is a faithful port of stern_brocot_erase_single_native from
stern_brocot_erase.cpp. It is a *genuinely signed* Stern-Brocot tree:
the endpoints are seeded at -1/0 (-inf) and +1/0 (+inf) and the walk
starts at the central mediant 0/1, so the tree spans the whole real
line symmetrically. The central rational 0 is a first-class node
reached by the mediant recursion itself -- NOT a positive tree with a
sign reattached afterward. (An earlier version of this file made that
mistake; it produced the wrong rationals near zero and even -- rather
than odd -- microstate counts.)

Algorithm (matching the C++ exactly):

    left  = -1/0   (-inf)
    right =  1/0   (+inf)
    sel   =  0/1   (start at the center)
    while |sel - blob_center| > tol and steps < max:
        if sel < blob_center:  left  = sel-as-fraction   (emit '1')
        else:                  right = sel-as-fraction   (emit '0')
        sel = mediant(left, right)
    return sel (selected microstate), eps = sel - blob_center

Because the action capacity A is finite, the tolerance is bounded
below, so the walk halts at finite depth and many distinct blob
centers collapse onto the same microstate -- the many-to-one
granulation that is the physical content of the model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PartitionResult:
    """Outcome of one signed Stern-Brocot partition.

    Q:     selected microstate (the rational the walk halted on), signed.
    eps:   erasure displacement Q - blob_center (signed).
    bits:  partition word length (mediant steps) = Landauer bit count N.
    word:  the partition word, '0' (right) / '1' (left) per the C++.
    found: whether the walk converged within tolerance before max_bits.
    """

    Q: float
    eps: float
    bits: int
    word: str
    found: bool


def partition(blob_center: float, tol: float, *, max_bits: int = 2000) -> PartitionResult:
    """Signed Stern-Brocot partition of ``blob_center`` to tolerance ``tol``.

    Faithful scalar port of stern_brocot_erase_single_native. ``blob_center``
    is a normalized coordinate (typically in [-1, 1]); ``tol`` is the
    normalized halt tolerance ((2/pi)/A for the harmonic oscillator).
    """
    if tol <= 0:
        raise ValueError("tol must be positive (finite action capacity).")

    # Signed Stern-Brocot endpoints: left = -inf, right = +inf, start 0/1.
    left_num, left_den = -1, 0
    right_num, right_den = 1, 0
    num, den = 0, 1

    sel = num / den
    eps = sel - blob_center
    word_chars: list[str] = []
    bits = 0

    while abs(eps) > tol and bits < max_bits:
        if sel < blob_center:
            left_num, left_den = num, den
            word_chars.append("1")
        else:
            right_num, right_den = num, den
            word_chars.append("0")
        num = left_num + right_num
        den = left_den + right_den
        if den <= 0:
            break
        sel = num / den
        eps = sel - blob_center
        bits += 1

    found = abs(eps) <= tol
    return PartitionResult(Q=sel, eps=eps, bits=bits, word="".join(word_chars), found=found)
