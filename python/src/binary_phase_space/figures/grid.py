"""Assemble the binary phase-space figure as a 3-column grid.

  Col 1: Partition. The (blob center -> selected microstate) staircase
         with faint downward erasure scars.
  Col 2: Cross-Section. The 1-D position-density skyline (portrait at p=0).
  Col 3: Portrait. The 2-D phase-space portrait with cell overlays.

All three columns share the same per-row x-window so they align. Row
labels sit in the left margin rotated 90 degrees, matching the Wigner
manuscript's leftmost-column metadata.

ONE assembly function. Every multi-row figure (harmonic, eigen, cat,
...) goes through here. Each row is described by a Row record carrying
the action level A (in units where h/2 = 1) and optionally an orbit
function r_orbit(theta). For the harmonic oscillator r_orbit=None gives
the circle of radius sqrt(A); for the squeezed vacuum, Morse, double-well,
and Kerr crescent r_orbit is the per-state ellipse/teardrop/contour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .panels import portrait_panel, cross_section_panel, partition_panel
from ..cross_section import compute_cross_section


@dataclass
class Row:
    """One row of the figure.

    A         action capacity in units where h/2 = 1 (so vacuum -> A = 1).
              Sets the SSB tolerance and therefore the number of resolved
              microstates: A=1 gives 3, A=4 gives 13, etc.

    r_orbit   theta -> physical orbit radius. None means the QHO circle
              r_orbit(theta) = sqrt(A) (all angles equal). For elliptical
              states pass the ellipse parametrization; for anharmonic
              states pass the semiclassical contour.

    state     optional State (from systems/). When present its
              ``state.window`` (computed during state construction by the
              Wigner code's build_state_from_qobj/_from_psi from the actual
              Wigner-function extent) sets the per-row x,p window. This is
              the Wigner code path and the one that doesn't clip the
              portrait. For harmonic rows (no state) the window is
              computed from A and the orbit instead.

    label     optional row label for the left margin.
    """
    A: float
    r_orbit: Optional[Callable] = None
    state: Optional[object] = None
    label: Optional[str] = None


def _row_lim(row: Row) -> float:
    """Shared per-row x,p half-window so all three columns align.

    The window must contain BOTH the partition's hard-edged data (its
    outermost ring extending to Delta * (1 + w/2)) AND the state's own
    fade-out window from the Wigner code (state.window.x_lim, when
    present). The Wigner code sizes its window from the Wigner function,
    which fades smoothly to zero; our partition has finite hard support
    that can extend further. We take the larger of the two so neither
    construction clips.

    For harmonic rows (no state) only the partition extent matters.
    """
    cs = compute_cross_section(row.A)
    outer = float(cs.coordinate[-1] + cs.bin_width / 2.0)
    if row.r_orbit is None:
        max_orbit = float(np.sqrt(row.A))
    else:
        Dx = float(row.r_orbit(0.0))
        Dp = float(row.r_orbit(np.pi / 2))
        max_orbit = max(Dx, Dp)
    # Partition data extends to orbit * (1 + w/2 in normalized units).
    # outer / max_orbit_for_orbit_normalization captures that ratio for
    # the equivalent harmonic case at this A; for non-circular orbits
    # the largest data extent is at the longest principal radius.
    w_norm_factor = outer / float(np.sqrt(row.A))
    partition_lim = max_orbit * w_norm_factor * 1.05

    if row.state is not None:
        return max(partition_lim, float(row.state.window.x_lim))
    return partition_lim


def assemble_grid(
    rows: Iterable[Row],
    *,
    panel_width: float = 1.5,
    panel_height: float = 1.5,
    h_pad: float = 0.35,
    w_pad: float = 0.55,
    margin_top: float = 0.40,
    margin_bottom: float = 0.40,
    margin_left: float = 1.35,
    margin_right: float = 0.15,
    column_titles=("Partition", "Cross-Section", "Portrait"),
) -> Figure:
    """Build the 3-column figure across the given rows.

    rows: iterable of Row records. Each row's A and r_orbit determine
    its partition, cross-section, and portrait; the row's label appears
    in the left margin.
    """
    rows = list(rows)
    n_rows = len(rows)
    ncols = 3

    fig_width = margin_left + ncols * panel_width + (ncols - 1) * w_pad + margin_right
    fig_height = margin_top + n_rows * panel_height + (n_rows - 1) * h_pad + margin_bottom

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=n_rows, ncols=ncols, figure=fig,
        left=margin_left / fig_width, right=1 - margin_right / fig_width,
        top=1 - margin_top / fig_height, bottom=margin_bottom / fig_height,
        hspace=h_pad / panel_height, wspace=w_pad / panel_width,
    )

    for i, row in enumerate(rows):
        ax_part  = fig.add_subplot(gs[i, 0])
        ax_cross = fig.add_subplot(gs[i, 1])
        ax_port  = fig.add_subplot(gs[i, 2])

        lim = _row_lim(row)
        partition_panel(ax_part, row.A, lim=lim)
        cross_section_panel(ax_cross, row.A, r_orbit=row.r_orbit)
        portrait_panel(ax_port, row.A, r_orbit=row.r_orbit)

        # Shared x-window per row.
        ax_port.set_xlim(-lim, lim); ax_port.set_ylim(-lim, lim)
        ax_cross.set_xlim(-lim, lim)

        if i == 0:
            ax_part.set_title(column_titles[0])
            ax_cross.set_title(column_titles[1])
            ax_port.set_title(column_titles[2])

        ax_part.set_ylabel(r"$\tilde{x} / x_0$")
        ax_cross.set_ylabel(r"$\rho(x, 0)$")
        ax_port.set_ylabel(r"$p / p_0$")

        if i == n_rows - 1:
            ax_part.set_xlabel(r"$x_\mu / x_0$")
            ax_cross.set_xlabel(r"$x / x_0$")
            ax_port.set_xlabel(r"$x / x_0$")

        if row.label is not None:
            pos = ax_part.get_position()
            y_center = 0.5 * (pos.y0 + pos.y1)
            x_label = pos.x0 - 1.0 * w_pad / fig_width
            fig.text(
                x_label, y_center, row.label,
                rotation=90, ha="center", va="center",
                fontsize=plt.rcParams["axes.titlesize"],
            )

    return fig


def save_grid(fig: Figure, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved {out_path}")
