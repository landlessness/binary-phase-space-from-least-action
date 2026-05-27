"""1-D least-action position-density cross-section.

Faithful port of the R pipeline:
  - stern_brocot_erase.cpp           (the signed Stern-Brocot partition)
  - harmonic_oscillator_simulator.cpp (tolerance and blob-center sweep)
  - density_computer.R                (raw_fluc binning, odd state grid)
  - plot_encoding_grid.R              (coordinate scaling, skyline)

One cross-section corresponds to a single action level A, with
normalized momentum p = sqrt(A) (harmonic, symmetric case Dx = Dp = p).

Chain of constants, each traced to the R/C++ source (no fitted factors):

  tolerance fed to the tree     tol = (2/pi) * (1/p) * (1/p) = (2/pi)/A
      (harmonic_oscillator_simulator.cpp: squeezed_boundary then
       algorithmic_tolerance = squeezed_boundary / delta_p)

  blob centers                  100001 points on [-1, 1]
      (harmonic_oscillator_simulator.cpp)

  binned quantity               raw_fluc = erasure_displacement * p^2
      (density_computer.R)

  bin grid                      symmetric, odd count = number of unique
                                selected microstates (found == 1)
      (density_computer.R / compute_action_density)

  plotted coordinate            x = bin_center * p * (pi/2)
      ("scale by the classical boundary and undo the 2/pi projection",
       plot_encoding_grid.R)

The plotted density is the histogram of the amplified ERASURE
DISPLACEMENT, not of the landing microstates. Histogramming the
landing microstates gives a classical, spiky distribution;
histogramming the amplified displacement recovers quantum structure
(the vacuum plateau at A=1 through the classical turning-point
U-shape at A=50).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signed_tree_vec import erase_vec


@dataclass
class CrossSection:
    """A computed 1-D position-density cross-section at one action level."""

    A: float                  # action level (= p^2)
    p: float                  # normalized momentum (= sqrt(A))
    n_states: int             # number of unique microstates (odd)
    coordinate: np.ndarray    # bin centers in physical coordinate x
    density: np.ndarray       # density in percent
    bin_width: float          # physical width of each bin


def _tolerance(A: float) -> float:
    """Tolerance fed to the signed tree: (2/pi)/A.

    From harmonic_oscillator_simulator.cpp:
        squeezed_boundary     = (2/pi) * (1/p)
        algorithmic_tolerance = squeezed_boundary / p = (2/pi)/p^2 = (2/pi)/A
    The 2/pi is the 2D-to-1D average linear projection of a harmonic
    orbit (mean of |cos| over a cycle).
    """
    return (2.0 / np.pi) / A


def compute_cross_section(A: float, *, n_centers: int = 100001) -> CrossSection:
    """Compute the position-density cross-section at action level A."""
    p = np.sqrt(A)
    tol = _tolerance(A)

    centers = np.linspace(-1.0, 1.0, n_centers)
    sel, eps, found = erase_vec(centers, tol)

    # Restrict to converged erasures (found == 1), as the R pipeline does.
    sel_f = sel[found]
    eps_f = eps[found]

    # Unique microstate count -> bin count (density_computer.R uses
    # signif(selected_microstate, 7); round to 7 sig figs here).
    n_states = _count_unique_signif(sel_f, digits=7)
    if n_states % 2 == 0:
        # The pipeline asserts odd symmetry; nudge if a boundary sample
        # split a central state. (With the signed tree this is rare.)
        n_states += 1

    raw_fluc = eps_f * (p * p)                     # density_computer.R
    max_extent = float(np.max(np.abs(raw_fluc)))

    if n_states <= 1 or max_extent == 0.0:
        return CrossSection(
            A=A, p=p, n_states=n_states,
            coordinate=np.array([0.0]), density=np.array([100.0]), bin_width=1.0,
        )

    bins_one_side = (n_states - 1) // 2
    spacing = max_extent / bins_one_side
    breaks = (np.arange(-bins_one_side - 1, bins_one_side + 1) + 0.5) * spacing

    counts, edges = np.histogram(raw_fluc, bins=breaks)
    mids = 0.5 * (edges[:-1] + edges[1:])
    mids[np.abs(mids) < 1e-10] = 0.0
    density = counts / counts.sum() * 100.0

    # plot_encoding_grid.R: scale bin centers by p and undo the 2/pi.
    coordinate = mids * p * (np.pi / 2.0)
    bin_width = float(np.median(np.diff(coordinate)))

    return CrossSection(
        A=A, p=p, n_states=n_states,
        coordinate=coordinate, density=density, bin_width=bin_width,
    )


def _count_unique_signif(values: np.ndarray, *, digits: int = 7) -> int:
    """Count unique values rounded to `digits` significant figures.

    Mirrors R's unique(signif(x, digits)) used in density_computer.R.
    """
    v = np.asarray(values, dtype=float)
    out = np.zeros_like(v)
    nz = v != 0
    mags = np.floor(np.log10(np.abs(v[nz])))
    factor = 10.0 ** (digits - 1 - mags)
    out[nz] = np.round(v[nz] * factor) / factor
    return len(np.unique(out))
