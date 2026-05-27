"""Render the eigen figure: squeezed vacuum, Morse, double-well, Kerr.

Mirrors the Wigner Resolution paper's render_eigen.py: same states, same
row labels, same per-row x,p windows (taken from each state's own
state.window, which was sized by build_state_from_qobj/_from_psi from
the Wigner-function extent).

Output: tex/figures/eigen.pdf.
"""

from __future__ import annotations

from pathlib import Path

from binary_phase_space.figures.grid import Row, assemble_grid, save_grid
from binary_phase_space.figures._orbits import (
    morse_orbit,
    squeezed_vacuum_orbit,
)
from binary_phase_space.plotstyle import use_prl_style
from binary_phase_space.systems.morse import morse_state
from binary_phase_space.systems.squeezed_vacuum import squeezed_vacuum_state

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "tex" / "figures" / "eigen.pdf"

use_prl_style(use_tex=True)


def squeezed_row(r: float, label: str) -> Row:
    state = squeezed_vacuum_state(r=r)
    A, r_orbit = squeezed_vacuum_orbit(state)
    return Row(A=A, r_orbit=r_orbit, state=state, label=label)


def morse_row(n: int, label: str) -> Row:
    state = morse_state(n=n)
    A, r_orbit = morse_orbit(state)
    return Row(A=A, r_orbit=r_orbit, state=state, label=label)


rows = [
    squeezed_row(r=0.5, label=r"Squeezed vacuum, $r = 0.5$"),
    morse_row(n=8, label=r"Morse, $n = 8$"),
]

fig = assemble_grid(rows)
save_grid(fig, OUT)
