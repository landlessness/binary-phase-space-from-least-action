"""Partition panel: the (blob center -> selected microstate) staircase
with the erasure displacement shown as faint downward scars at each
swept center.

Mirrors the middle column of plot_encoding_grid.R:
  - x = blob_center * max_ax   (no pi/2 here; that scaling is only on
                                the position-density column)
  - y = selected_microstate * max_ax
  - scar = vertical segment from (q_b, q_mu) down to (q_b, q_mu - eps*max_ax),
           gray90 at alpha 0.05, linewidth 0.2 (the dot/scar cloud reads
           as a staircase because many centers collapse onto the same
           microstate, creating horizontal bands)
  - point = small black dot at (q_b, q_mu)
  - filter: only found == 1 rows
  - sweep: dense over centers in [-1, +1]

max_ax = sqrt(A) (the classical turning point in q_0 units), matching
the R convention max_ax = p_val with p_val = sqrt(action_A).
"""

from __future__ import annotations

import numpy as np

from ..signed_tree_vec import erase_vec


def compute_partition(A, n_centers: int = 100001):
    """Run the signed tree across the unit-disk diameter and return the
    raw per-center (q_b, q_mu, eps) triples in physical q_0 units."""
    centers = np.linspace(-1.0, 1.0, n_centers)
    # tolerance matches compute_cross_section: (2/pi) / A
    tol = (2.0 / np.pi) / A
    sel, eps, found = erase_vec(centers, tol)
    keep = found == 1
    max_ax = float(np.sqrt(A))
    q_b = centers[keep] * max_ax
    q_mu = sel[keep] * max_ax
    eps_scaled = eps[keep] * max_ax
    return q_b, q_mu, eps_scaled, max_ax
