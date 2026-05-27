"""Render the harmonic-oscillator figure.

Two columns x five rows, mirroring the Wigner Resolution paper's
render_harmonic.py: phase-space portrait, position density (= portrait
at p=0), and partition staircase. Rows are action levels A in units
where h/2 = 1.

Output: tex/figures/harmonic.pdf.
"""

from __future__ import annotations

from pathlib import Path

from binary_phase_space.figures.grid import Row, assemble_grid, save_grid
from binary_phase_space.plotstyle import use_prl_style

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "harmonic.pdf"

use_prl_style(use_tex=True)

rows = [
    Row(A=1.0,  label=r"QHO, $A = 1.0\,A_0$"),
    Row(A=3.0,  label=r"QHO, $A = 3.0\,A_0$"),
    Row(A=4.0,  label=r"QHO, $A = 4.0\,A_0$"),
    Row(A=5.5,  label=r"QHO, $A = 5.5\,A_0$"),
    Row(A=50.0, label=r"QHO, $A = 50.0\,A_0$"),
]

fig = assemble_grid(rows)
save_grid(fig, OUT)
