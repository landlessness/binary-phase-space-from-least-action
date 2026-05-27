"""Semiclassical orbits r_orbit(theta) for the eigen figure rows.

Each builder takes a State (from the systems/ subpackage) and returns a
(A, r_orbit) pair: the action capacity at which the SSB partition runs,
and the orbit function (theta -> physical radius) the partition stretches
to.

The squeezed vacuum's orbit is the Heisenberg-cell ellipse with semi-axes
Delta_x, Delta_p. The Morse, double-well, and Kerr orbits are added in
later steps.
"""

from __future__ import annotations

import numpy as np


def ellipse_orbit(Dx: float, Dp: float):
    """Axis-aligned ellipse with semi-axes Dx along x, Dp along p.

        r_orbit(theta) = Dx * Dp / sqrt((Dp cos theta)^2 + (Dx sin theta)^2)

    Hits Dx at theta = 0 (the +x axis) and Dp at theta = pi/2 (+p axis).
    """
    def r_orbit(theta):
        theta = np.asarray(theta)
        c = np.cos(theta); s = np.sin(theta)
        return (Dx * Dp) / np.sqrt((Dp * c) ** 2 + (Dx * s) ** 2)
    return r_orbit


def squeezed_vacuum_orbit(state):
    """Orbit for the squeezed vacuum: Heisenberg cell ellipse.

    The squeezed vacuum is a minimum-uncertainty (vacuum) state at every
    squeezing parameter r: Delta_x Delta_p = hbar, so

        A / (h/2) = pi Delta_x Delta_p / (pi hbar) = 1

    in the unit convention compute_cross_section uses (A passed as A/(h/2)).
    The state's *shape* changes with r (the cell ellipse axes Delta_x and
    Delta_p), but its *action capacity* does not. We pass A=1 so the
    partition runs at the same 3-microstate level as the QHO ground state.
    """
    Dx = float(state.rs.Delta_x)
    Dp = float(state.rs.Delta_p)
    A = 1.0   # vacuum action level in units where h/2 = 1
    return A, ellipse_orbit(Dx, Dp)
