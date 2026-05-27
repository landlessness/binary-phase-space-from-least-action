"""Panel drawers for the binary phase-space figures.

  portrait_panel        2-D portrait: the signed Stern-Brocot radial
                        density stretched to the semiclassical orbit
                        (circle for the harmonic oscillator). Overlaid
                        with the Heisenberg cell A and quorum cell ã.
  cross_section_panel   the 1-D position-density skyline.

The DC floor (the lowest ring density) is mapped to a light shade of
the colormap rather than to white, so the lowest band still reads as
faint red against the panel background.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..cells import HeisenbergCell, quorum_cell
from ..cross_section import compute_cross_section
from .overlays import heisenberg_cell_patch, quorum_cell_patch, LINEWIDTH
from ._portrait import portrait_field

_RDBU = plt.get_cmap("RdBu_r")
FILL = _RDBU(0.82)


def portrait_panel(ax, A, *, r_orbit=None, hbar: float = 1.0, n_img: int = 1400):
    """Draw the phase-space portrait for action level A with cell overlays."""
    field, lim, support, extent = portrait_field(A, r_orbit=r_orbit, n_img=n_img)

    # Color scale: map the DC floor (lowest nonzero ring) to a light red,
    # not white. Render only the warm half of the diverging palette since
    # the density is non-negative. vmin is set below the floor so the
    # lowest ring is a faint-but-visible red, and zero (background) stays
    # at the panel color.
    nz = field[field > 0]
    if nz.size:
        from matplotlib.colors import ListedColormap
        floor = nz.min()
        top = field.max()
        # Build a colormap from the WARM HALF of RdBu_r only: light red at
        # the DC floor, dark red at the peak. Nothing maps to blue. Zero
        # (background) is masked so it shows the panel color.
        warm = ListedColormap(_RDBU(np.linspace(0.64, 0.95, 256)))  # floor reads as definite red
        masked = np.ma.masked_where(field <= 0, field)
        ax.imshow(
            masked.T,
            extent=[-lim, lim, -lim, lim],
            origin="lower",
            cmap=warm,
            norm=Normalize(vmin=floor - 1e-9, vmax=top),
            interpolation="nearest",
        )
    ax.set_aspect("equal", adjustable="box")

    # Heisenberg cell A and quorum cell ã. The orbit boundary IS the cell
    # ellipse, so the cell's principal radii come from probing r_orbit at
    # theta=0 (Dx) and theta=pi/2 (Dp). For the QHO (r_orbit=None,
    # circular orbit) Dx = Dp = sqrt(A), recovering the harmonic circles.
    if r_orbit is None:
        Dx = Dp = float(np.sqrt(A))
    else:
        Dx = float(r_orbit(0.0))
        Dp = float(r_orbit(np.pi / 2))
    heis = HeisenbergCell(Delta_x=Dx, Delta_p=Dp, center=(0.0, 0.0))
    qc = quorum_cell(heis, hbar=hbar)
    ax.add_patch(heisenberg_cell_patch(heis))
    ax.add_patch(quorum_cell_patch(qc))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _axis_ticks(ax, x_lim=lim, y_lim=lim)


def cross_section_panel(ax, A, *, r_orbit=None, n_img: int = 1400):
    """Draw the 1-D cross-section as the portrait sliced at p = 0.

    This is the uniform construction used across every row of the figure:
    the cross-section column is always the portrait of the same row,
    sampled along the horizontal axis (p = 0). Not a marginal, not an
    independent histogram -- one consistent operation for all states,
    matching the Wigner paper's W(x, 0) convention.

    For the harmonic case (r_orbit = circle of radius sqrt(A)) this
    slice equals compute_cross_section(A) to floating point, since the
    radial profile and the p=0 slice agree on a circle. For elliptical
    or anharmonic orbits the slice captures the orbit-stretch in the
    x-direction.
    """
    field, lim, support, extent = portrait_field(A, r_orbit=r_orbit, n_img=n_img)
    n = field.shape[0]
    mid = n // 2
    xs = np.linspace(-lim, lim, n)
    # p = 0 corresponds to the middle row of the field (YY = 0 there).
    # field is indexed [ix, iy] so the p=0 slice is field[:, mid].
    slice_y = field[:, mid]

    # Render as a filled skyline at the raster resolution. The discrete
    # bins show as flat plateaus by construction (portrait_field uses
    # nearest-bin assignment and interpolation='nearest').
    ax.fill_between(xs, slice_y, 0, where=(slice_y > 0),
                    color=FILL, alpha=0.85, linewidth=0)
    ax.plot(xs, slice_y, color="black", lw=LINEWIDTH)
    ax.set_xlim(-lim, lim)
    ymax = float(slice_y.max()) if slice_y.max() > 0 else 1.0
    ax.set_ylim(0, ymax * 1.15)
    _axis_ticks(ax, x_lim=lim)  # density y-axis left to matplotlib default


def _axis_ticks(ax, *, x_lim: float, y_lim: float | None = None):
    """Place 3 nice symmetric ticks on the x (and optionally y) axis,
    spanning the actual axis extent rather than a fixed ±sqrt(A) anchor.

    Uses nice_symmetric_ticks from the Wigner code's ticks module --
    same 1-2-5 mantissa step picking as the Wigner figures, so harmonic
    rows (lim ≈ 1.05*sqrt(A)) get the same ticks as before and
    non-harmonic rows get ticks that span their actual window.
    """
    from ..ticks import nice_symmetric_ticks
    xt = nice_symmetric_ticks(x_lim, target_count=3)
    ax.set_xticks(list(xt))
    ax.set_xticklabels([f"{t:.1f}" for t in xt])
    if y_lim is not None:
        yt = nice_symmetric_ticks(y_lim, target_count=3)
        ax.set_yticks(list(yt))
        ax.set_yticklabels([f"{t:.1f}" for t in yt])


# ---------------------------------------------------------------------------
# Partition panel (new leftmost column)
# ---------------------------------------------------------------------------

from ._partition import compute_partition  # noqa: E402


def partition_panel(ax, A, *, lim=None):
    """Draw the (blob center -> selected microstate) panel with downward
    erasure scars and a black dot at each landed point.

    Matches the middle column of plot_encoding_grid.R: faint gray scars
    from (q_b, q_mu) down to (q_b, q_mu - eps), and black dots at
    (q_b, q_mu). Many centers collapse onto the same microstate, so the
    dots form horizontal bands -- the staircase treads.
    """
    q_b, q_mu, eps_scaled, max_ax = compute_partition(A)

    if lim is None:
        lim = max_ax * 1.15

    # Faint downward scars (vertical line per center, from landing to
    # landing - eps). The R uses linewidth=0.2, alpha=0.05, color gray90.
    ax.vlines(q_b, q_mu - eps_scaled, q_mu, colors="0.85", alpha=0.05, linewidth=0.2)
    # Black landing dots.
    ax.plot(q_b, q_mu, marker="o", linestyle="none", color="black",
            markersize=0.6, markeredgewidth=0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _axis_ticks(ax, x_lim=lim, y_lim=lim)
