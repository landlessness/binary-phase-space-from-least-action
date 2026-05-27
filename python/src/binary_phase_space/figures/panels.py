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

    # Overlays: Heisenberg cell A and quorum cell ã (harmonic: circles).
    p = np.sqrt(A)
    heis = HeisenbergCell(Delta_x=p, Delta_p=p, center=(0.0, 0.0))
    qc = quorum_cell(heis, hbar=hbar)
    ax.add_patch(heisenberg_cell_patch(heis))
    ax.add_patch(quorum_cell_patch(qc))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _root_ticks(ax, A, both=True)


def cross_section_panel(ax, A):
    """Draw the 1-D position-density cross-section as a skyline step."""
    cs = compute_cross_section(A)
    w = cs.bin_width
    cq, pct = cs.coordinate, cs.density
    step_x = np.concatenate(
        [[cq[0] - w / 2], np.repeat(cq, 2) + np.tile([-w / 2, w / 2], len(cq)), [cq[-1] + w / 2]]
    )
    step_y = np.concatenate([[0.0], np.repeat(pct, 2), [0.0]])
    ax.fill(step_x, step_y, color=FILL, alpha=0.85, lw=0)
    ax.plot(step_x, step_y, color="black", lw=LINEWIDTH)
    lim = np.sqrt(A) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, pct.max() * 1.15)
    _root_ticks(ax, A, both=False)


def _root_ticks(ax, A, *, both: bool):
    r = np.sqrt(A)
    ax.set_xticks([-r, 0, r])
    ax.set_xticklabels([f"{-r:.1f}", "0.0", f"{r:.1f}"])
    if both:
        ax.set_yticks([-r, 0, r])
        ax.set_yticklabels([f"{-r:.1f}", "0.0", f"{r:.1f}"])


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
    _root_ticks(ax, A, both=True)
