"""Render the harmonic-oscillator figure: phase-space portrait + position density.

Two columns x five rows, mirroring the layout and labelling of the
Wigner Resolution paper's render_harmonic.py:
  Col 1: phase-space portrait (signed Stern-Brocot rings) with the
         Heisenberg cell A and quorum cell a-tilde overlays.
  Col 2: position-density cross-section.

Rows are indexed by action level A (where the Wigner paper indexes by
Fock number n). Row labels follow the same "QHO, $...$" form.

Output: tex/figures/harmonic.pdf.
"""

from __future__ import annotations

from pathlib import Path

from binary_phase_space.figures.grid import assemble_grid, save_grid
from binary_phase_space.plotstyle import use_prl_style

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "harmonic.pdf"

use_prl_style(use_tex=True)

A_values = [1.0, 3.0, 4.0, 5.5, 50.0]

row_labels = [
    r"QHO, $A = 1.0\,A_0$",
    r"QHO, $A = 3.0\,A_0$",
    r"QHO, $A = 4.0\,A_0$",
    r"QHO, $A = 5.5\,A_0$",
    r"QHO, $A = 50.0\,A_0$",
]

fig = assemble_grid(A_values, row_labels=row_labels)
save_grid(fig, OUT)
