"""Partition panel: the (blob center -> selected microstate) staircase
with the erasure displacement shown as faint downward scars at each
swept center.

The signed Stern-Brocot tree runs on the normalized diameter [-1, +1].
The diameter is the x-axis (theta=0 to the right, theta=pi to the left).
For non-circular orbits the two halves stretch by different factors:
positive-side centers map to physical x = s * r_orbit(0); negative-side
centers map to x = s * r_orbit(pi) (which is a negative number times a
positive number when s < 0). For circular orbits both halves share the
same stretch sqrt(A), recovering the original symmetric behavior.

Mirrors the middle column of plot_encoding_grid.R after the stretch:
  - x = blob_center * stretch
  - y = selected_microstate * stretch
  - scar = vertical segment from (q_b, q_mu) down to (q_b, q_mu - eps*stretch)
  - point = small black dot at (q_b, q_mu)
  - filter: only found == 1 rows
  - sweep: dense over centers in [-1, +1]
"""

from __future__ import annotations

import numpy as np

from ..signed_tree_vec import erase_vec


def compute_partition(A, r_orbit=None, n_centers: int = 100001):
    """Run the signed tree across the unit-disk diameter and return the
    raw per-center (q_b, q_mu, eps) triples in physical x units.

    The diameter sweeps the +x ray for s > 0 and the -x ray for s < 0.
    Each ray gets its own stretch factor:
      r_pos = r_orbit(0)   for s > 0  (the +x reach of the orbit)
      r_neg = r_orbit(pi)  for s < 0  (the -x reach of the orbit)

    For the QHO (r_orbit=None) and symmetric orbits both equal sqrt(A),
    recovering the original symmetric stretch. For Morse / double-well /
    Kerr the partition picture is asymmetric -- matching the asymmetry of
    the cross-section and portrait columns.

    Returns (q_b, q_mu, eps_scaled, max_ax) in physical x_0 units, plus
    a (x_min, x_max) data extent for the row window.
    """
    centers = np.linspace(-1.0, 1.0, n_centers)
    tol = (2.0 / np.pi) / A
    sel, eps, found = erase_vec(centers, tol)
    keep = found == 1
    c_k = centers[keep]
    s_k = sel[keep]
    e_k = eps[keep]

    if r_orbit is None:
        r_pos = r_neg = float(np.sqrt(A))
    else:
        r_pos = float(r_orbit(0.0))
        r_neg = float(r_orbit(np.pi))

    # Per-center stretch: positive s uses r_pos, negative s uses r_neg.
    # eps is signed in the same sense (eps is the displacement of the
    # microstate from the input on the SAME ray), so it uses the SAME
    # per-center stretch as that ray.
    pos_mask = c_k >= 0
    stretch = np.where(pos_mask, r_pos, r_neg)
    q_b  = c_k * stretch
    q_mu = s_k * stretch
    eps_scaled = e_k * stretch

    max_ax = max(r_pos, r_neg)
    return q_b, q_mu, eps_scaled, max_ax
