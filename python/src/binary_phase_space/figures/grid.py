"""Assemble the binary phase-space figure as a 2-column grid.

Mirrors the Wigner package's figures/grid.py: fixed per-panel boxes,
figure size derived from them, aspect-equal heatmaps. Row labels are
placed in the left margin rotated 90 degrees, matching the Wigner
manuscript's leftmost-column row metadata.

Column 1: phase-space portrait with cell overlays.
Column 2: position-density cross-section.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .panels import portrait_panel, cross_section_panel


def assemble_grid(
    A_values,
    *,
    row_labels=None,
    panel_width: float = 1.7,
    panel_height: float = 1.5,
    h_pad: float = 0.35,
    w_pad: float = 0.55,
    margin_top: float = 0.40,
    margin_bottom: float = 0.40,
    margin_left: float = 1.35,
    margin_right: float = 0.15,
    column_titles=("Phase-Space Portrait", "Position Density"),
) -> Figure:
    """Build the 2-column figure across the given action levels."""
    n_rows = len(A_values)
    ncols = 2

    if row_labels is not None and len(row_labels) != n_rows:
        raise ValueError(f"row_labels has {len(row_labels)} entries, expected {n_rows}")

    fig_width = margin_left + ncols * panel_width + (ncols - 1) * w_pad + margin_right
    fig_height = margin_top + n_rows * panel_height + (n_rows - 1) * h_pad + margin_bottom

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        nrows=n_rows, ncols=ncols, figure=fig,
        left=margin_left / fig_width, right=1 - margin_right / fig_width,
        top=1 - margin_top / fig_height, bottom=margin_bottom / fig_height,
        hspace=h_pad / panel_height, wspace=w_pad / panel_width,
    )

    for i, A in enumerate(A_values):
        ax_port = fig.add_subplot(gs[i, 0])
        ax_cross = fig.add_subplot(gs[i, 1])

        portrait_panel(ax_port, A)
        cross_section_panel(ax_cross, A)

        if i == 0:
            ax_port.set_title(column_titles[0])
            ax_cross.set_title(column_titles[1])

        ax_port.set_ylabel(r"$p / p_0$")

        if i == n_rows - 1:
            ax_port.set_xlabel(r"$x / x_0$")
            ax_cross.set_xlabel(r"$x / x_0$")

        if row_labels is not None:
            pos = ax_port.get_position()
            y_center = 0.5 * (pos.y0 + pos.y1)
            x_label = pos.x0 - 1.0 * w_pad / fig_width
            fig.text(
                x_label, y_center, row_labels[i],
                rotation=90, ha="center", va="center",
                fontsize=plt.rcParams["axes.titlesize"],
            )

    return fig


def save_grid(fig: Figure, out_path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved {out_path}")