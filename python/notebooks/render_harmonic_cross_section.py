"""Render the harmonic-oscillator position-density cross-sections.

Recreates the "Position Density" column of the R encoding_grid figure
for the action levels A = 1, 3, 4, 5.5, 50, using the faithful signed
Stern-Brocot port in binary_phase_space.cross_section.

Output: tex/figures/harmonic_cross_section.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from binary_phase_space.cross_section import compute_cross_section
from binary_phase_space.plotstyle import use_prl_style

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "harmonic_cross_section.pdf"

use_prl_style(use_tex=True)
_RDBU = plt.get_cmap("RdBu_r")
FILL = _RDBU(0.82)

A_values = [1.0, 3.0, 4.0, 5.5, 50.0]
n = len(A_values)

fig, axes = plt.subplots(n, 1, figsize=(2.6, n * 1.2), squeeze=False)
for i, A in enumerate(A_values):
    cs = compute_cross_section(A)
    ax = axes[i, 0]
    w = cs.bin_width
    cq, pct = cs.coordinate, cs.density

    # Skyline path (matches plot_encoding_grid.R step construction).
    step_x = np.concatenate([
        [cq[0] - w / 2],
        np.repeat(cq, 2) + np.tile([-w / 2, w / 2], len(cq)),
        [cq[-1] + w / 2],
    ])
    step_y = np.concatenate([[0.0], np.repeat(pct, 2), [0.0]])

    ax.fill(step_x, step_y, color="0.85", lw=0)
    ax.plot(step_x, step_y, color="black", lw=0.4)

    lim = np.sqrt(A) * 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, pct.max() * 1.15)
    ax.set_ylabel(rf"${A}\,A_0$", rotation=90, va="center")
    ax.set_xticks([-np.sqrt(A), 0, np.sqrt(A)])
    ax.set_xticklabels([f"{-np.sqrt(A):.1f}", "0.0", f"{np.sqrt(A):.1f}"])
    if i == 0:
        ax.set_title("Position Density")
    if i == n - 1:
        ax.set_xlabel(r"Position, $x$")
    print(f"A={A}: {cs.n_states} states, extent +/-{np.max(np.abs(cq)):.2f}")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT)
print(f"Saved {OUT}")