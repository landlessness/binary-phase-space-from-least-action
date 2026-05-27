"""General phase-space portrait construction.

The radial profile is the SIGNED Stern-Brocot density along the radius,
identical to the position cross-section (signed tree on the unit-disk
diameter r in [-1, 1] at tolerance (2/pi)/A, count-based histogram of
the amplified erasure displacement). Resolution from action capacity:
1/dr ~ A ~ N sets how finely the radius resolves into rational rings.

For each angle theta the normalized radius r in [-1, 1] is stretched so
r = +-1 lands on +-r_orbit(theta). For the harmonic oscillator
r_orbit = sqrt(A) (a circle) and the profile is revolved unchanged. For
an anharmonic state r_orbit(theta) is the semiclassical contour and the
same profile stretches differently per angle.

No folding: the tree is native on [-1, 1], the center is r = 0, and the
rings are the rationals the tree resolves. The portrait reproduces the
cross-section exactly along any diameter, including its finite support.
"""

from __future__ import annotations

import numpy as np

from ..cross_section import compute_cross_section


def portrait_field(A, *, r_orbit=None, n_img: int = 1400, pad: float = 1.15):
    """Build the 2-D portrait field by stretching the radial density to the orbit.

    Returns (field, lim, support) where support = (inner, outer) physical
    radial extent of nonzero density along a diameter (for the harmonic
    case this is the +-coordinate span, matching the cross-section ink).
    """
    cs = compute_cross_section(A)
    coord = cs.coordinate                      # signed physical bin centers
    density = cs.density
    w = cs.bin_width
    # r_norm = 1 maps to the OUTERMOST BIN CENTER, which sits at the
    # classical turning point (sqrt(A) for the harmonic case). The
    # outermost ring straddles the orbit, its outer edge at center + w/2,
    # so the ink extends half a bin past r_norm = 1 -- the finite-support
    # signature that the cross-section also shows.
    extent = float(np.max(np.abs(coord)))      # = last bin center = orbit radius
    s_centers = coord / extent
    s_w = w / extent
    s_edges = np.concatenate([s_centers - s_w / 2.0, [s_centers[-1] + s_w / 2.0]])

    if r_orbit is None:
        r0 = np.sqrt(A)
        r_orbit = lambda th: np.full(np.shape(th), r0)

    th_probe = np.linspace(-np.pi, np.pi, 1441)
    r_max = float(np.max(r_orbit(th_probe)))
    # include the outermost ring's outer edge: orbit radius + half a bin,
    # stretched. The outer ring extends to r_norm = 1 + s_w/2.
    lim = r_max * (1.0 + s_w / 2.0) * pad

    xs = np.linspace(-lim, lim, n_img)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    R = np.sqrt(XX**2 + YY**2)
    TH = np.arctan2(YY, XX)
    r_orb = r_orbit(TH.ravel())

    # normalized SIGNED radial coordinate of each pixel along its diameter.
    # A pixel at (R, theta) sits at signed s = R / r_orbit(theta) on the
    # +theta ray. Look it up in the SIGNED bin edges directly -- no fold.
    # Because the profile is symmetric, +s and -s carry equal density, so
    # using +s = R/r_orbit (>= 0) and the signed edges on the >=0 side is
    # exact and preserves the full-width center bin (the bin straddling 0).
    s = R.ravel() / r_orb

    # digitize |s| against the positive half of the signed edges. The
    # center bin straddles 0, so its positive extent is [0, s_w/2]; the
    # full center disk (radius s_w/2 * extent) is the revolve of that bin.
    # Positive-side bins, built symmetric about each positive center:
    # center bin (at 0) is the disk [0, s_w/2]; bin k spans
    # [(k-1/2) s_w, (k+1/2) s_w]; the outermost bin (at r_norm=1) spans
    # [1 - s_w/2, 1 + s_w/2]. n centers -> n bins -> n+1 edges, matching
    # n densities exactly (no off-by-one).
    pos_mask = s_centers >= -1e-12
    pos_centers = s_centers[pos_mask]
    pos_density = density[pos_mask]
    # Interior bin edges are midway between centers (each ring is its value
    # +/- half the local spacing). The OUTERMOST edge is placed physically
    # at Delta_x + delta_x/2: the turning-point ring read from the center of
    # its resolution blob outward by half the resolution delta_x. In
    # normalized units delta_x/Delta_x = (1/sqrt(A))/sqrt(A) = 1/A, so the
    # outer edge is r_norm = 1 + (delta_x/2)/Delta_x.
    delta_over_Delta = 1.0 / A                    # (delta_x / Delta_x), harmonic
    interior_edges = 0.5 * (pos_centers[:-1] + pos_centers[1:])
    pos_edges = np.concatenate([[0.0], interior_edges, [1.0 + delta_over_Delta / 2.0]])

    outer = pos_edges[-1]
    idx = np.digitize(s, pos_edges) - 1
    valid = (idx >= 0) & (idx < len(pos_density)) & (s <= outer)
    field = np.zeros(R.size)
    field[valid] = pos_density[idx[valid]]
    field = field.reshape(R.shape)

    support = (coord[0] - w / 2.0, coord[-1] + w / 2.0)   # ink span (signed, physical)
    return field, lim, support, extent
