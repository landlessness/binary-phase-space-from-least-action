"""PRL-style matplotlib configuration for figures in the Wigner Resolution paper.

Usage:
    from wigner_resolution.plotstyle import use_prl_style, COLUMN_WIDTH
    use_prl_style()
"""

from __future__ import annotations

import matplotlib as mpl

# PRL column widths in inches.
COLUMN_WIDTH = 3.375  # single column
DOUBLE_WIDTH = 7.0    # full page width
GOLDEN = (1 + 5 ** 0.5) / 2

# Panel background. Matches the center color of the RdBu_r diverging
# colormap used in the heatmap (so the area outside the data extent in
# column 1, and the entire backdrop of the cross-section panels in
# columns 2 and 3, share one consistent gray). Avoids the visual jump
# where the heatmap's natural "off-white" abuts a pure-white plot area.
PANEL_BG = (0.969, 0.966, 0.965)


def use_prl_style(use_tex: bool = True) -> None:
    """Apply PRL-style matplotlib rcParams.

    Set ``use_tex=False`` if you don't have a LaTeX installation; figures
    will use matplotlib's mathtext instead.
    """
    mpl.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.facecolor": PANEL_BG,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "figure.figsize": (COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN),
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "legend.frameon": False,
        "legend.handlelength": 1.5,
    })