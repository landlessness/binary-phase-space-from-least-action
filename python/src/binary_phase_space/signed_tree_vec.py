"""Vectorized signed Stern-Brocot partition.

Workhorse for the cross-section / portrait sweeps: computes the
selected microstate, erasure displacement, and found-flag for a whole
array of blob centers at one tolerance. Faithful vectorization of the
signed tree in signed_tree.py (symmetric endpoints, start at 0/1, no
sign folding). Agrees with the scalar reference to floating point.
"""

from __future__ import annotations

import numpy as np


def erase_vec(blob_centers: np.ndarray, tol: float, *, max_bits: int = 2000):
    """Signed Stern-Brocot partition over an array of blob centers.

    Returns (selected_microstate, erasure_displacement, found), each an
    array the same shape as ``blob_centers``.
    """
    if tol <= 0:
        raise ValueError("tol must be positive.")

    centers = np.asarray(blob_centers, dtype=float)
    n = centers.size

    left_num = np.full(n, -1, dtype=np.int64)
    left_den = np.zeros(n, dtype=np.int64)
    right_num = np.ones(n, dtype=np.int64)
    right_den = np.zeros(n, dtype=np.int64)
    num = np.zeros(n, dtype=np.int64)
    den = np.ones(n, dtype=np.int64)

    sel = num / den
    eps = sel - centers
    active = np.abs(eps) > tol
    it = 0
    while active.any() and it < max_bits:
        go_left = (sel < centers) & active
        go_right = (~(sel < centers)) & active
        left_num = np.where(go_left, num, left_num)
        left_den = np.where(go_left, den, left_den)
        right_num = np.where(go_right, num, right_num)
        right_den = np.where(go_right, den, right_den)
        num = left_num + right_num
        den = left_den + right_den
        den_safe = np.where(den == 0, 1, den)
        sel = num / den_safe
        eps = sel - centers
        active = np.abs(eps) > tol
        it += 1

    found = np.abs(eps) <= tol
    return sel.reshape(centers.shape), eps.reshape(centers.shape), found.reshape(centers.shape)


def partition_eps_vec(blob_centers: np.ndarray, tol: float, *, max_bits: int = 2000) -> np.ndarray:
    """Convenience: just the erasure displacement array."""
    _, eps, _ = erase_vec(blob_centers, tol, max_bits=max_bits)
    return eps
