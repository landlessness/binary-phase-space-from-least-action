"""Matplotlib ellipse patches for phase-space cells.

`Ellipse` takes width and height as full extents (not semi-axes) and an
angle in degrees, so we convert from the semi-axes/radians convention used
throughout the rest of the package.

All drawn lines in the data figures — cell overlays, cross-section
traces, zero baselines — share a single line-width constant so a
tuning change flows through the whole figure consistently. Tune
``LINEWIDTH`` here.
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Ellipse

from ..cells import HeisenbergCell, BitangentBlob, QuorumCell


# Single line-width constant governing every drawn line in the data
# figures: cell-overlay ellipse strokes, cross-section data traces, and
# zero baselines. At typical PRA panel sizes 0.3-0.5 pt reads as a
# present-but-non-competing line weight. Lower if the strokes dominate
# the data; raise if lines vanish in print.
LINEWIDTH: float = 0.4


def heisenberg_cell_patch(
    cell: HeisenbergCell,
    *,
    facecolor: str = "none",
    edgecolor: str = "black",
    linewidth: float = LINEWIDTH,
    **kwargs,
) -> Ellipse:
    """Matplotlib ellipse patch for a Heisenberg cell A (axis-aligned)."""
    return Ellipse(
        xy=cell.center,
        width=2 * cell.Delta_x,
        height=2 * cell.Delta_p,
        angle=0.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )


def bitangent_blob_patch(
    blob: BitangentBlob,
    *,
    facecolor: str = "none",
    edgecolor: str = "black",
    linewidth: float = LINEWIDTH,
    **kwargs,
) -> Ellipse:
    """Matplotlib ellipse patch for a bitangent blob a_θ rotated by θ."""
    return Ellipse(
        xy=blob.center,
        width=2 * blob.r_parallel,
        height=2 * blob.r_perp,
        angle=np.degrees(blob.theta),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )


def quorum_cell_patch(
    cell: QuorumCell,
    *,
    facecolor: str = "none",
    edgecolor: str = "black",
    linewidth: float = LINEWIDTH,
    **kwargs,
) -> Ellipse:
    """Matplotlib ellipse patch for the quorum cell ã.

    Axis-aligned ellipse with semi-axes (δx, δp). The common interior
    of the bitangent family inscribed in the Heisenberg cell.
    """
    return Ellipse(
        xy=cell.center,
        width=2 * cell.delta_x,
        height=2 * cell.delta_p,
        angle=0.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        **kwargs,
    )
