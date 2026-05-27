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


def morse_orbit(state):
    """Orbit for a Morse eigenstate: the semiclassical teardrop contour.

    The SSB tree is rooted at the well bottom x=0. For each ray at angle
    theta from the root, the orbit radius r_orbit(theta) is the distance
    from the root to the level set V(x) + p^2/2 = E_n along that ray
    (with V(x) = D_e (1 - exp(-alpha x))^2 the Morse potential, and E_n
    the eigenstate's analytic energy).

    The teardrop is asymmetric in x: short reach left (steep wall), long
    reach right (soft dissociation). symmetric in p. The level set is
    star-shaped from the well bottom (single-well potential), so the
    radial scan via brentq converges unambiguously on every ray.

    A is the action capacity in h/2 units, read directly from the
    Wigner-code-built state.
    """
    from scipy.optimize import brentq
    from ..systems.morse import D_e as _D_e, alpha as _alpha, morse_V as _morse_V, morse_energy_exact

    # State energy (extract n from the state name, e.g. 'morse_n8' -> 8).
    # The state name is set by morse_state(n=..., name=None) to 'morse_n{n}'.
    name = state.name
    if name.startswith("morse_n"):
        n = int(name[len("morse_n"):])
    else:
        raise ValueError(f"Cannot infer Morse n from state name {name!r}")
    E_n = morse_energy_exact(n)

    # Action level in h/2 units (the Wigner code computes this on the state).
    A = float(state.rs.A_over_h_half)

    # Outer radius bracket for brentq: the largest distance from the
    # origin to any point on the teardrop. The x-extent is bounded by
    # x_out (soft-wall turning point); the p-extent is sqrt(2 E_n) at
    # x=0. The outer-radius bracket is the distance to the furthest
    # corner of the bounding box, plus a 5% margin.
    arg = np.sqrt(E_n / _D_e)
    x_out_tp = -np.log(1.0 - arg) / _alpha
    x_in_tp  = -np.log(1.0 + arg) / _alpha
    p_max    = np.sqrt(2.0 * E_n)
    r_max_bound = 1.05 * np.sqrt(max(x_out_tp, abs(x_in_tp))**2 + p_max**2)

    def _r_orbit_scalar(theta: float) -> float:
        """Solve V(r cos theta) + (r sin theta)^2 / 2 = E_n for r > 0."""
        ct, st = np.cos(theta), np.sin(theta)

        def f(r):
            return _morse_V(r * ct) + 0.5 * (r * st)**2 - E_n

        # At r=0 the LHS is V(0) - E_n = -E_n < 0; at r=r_max_bound it
        # exceeds E_n. Brent finds the unique zero.
        return brentq(f, 1e-9, r_max_bound, xtol=1e-8)

    def r_orbit(theta):
        scalar = np.isscalar(theta) or (hasattr(theta, "ndim") and theta.ndim == 0)
        th_arr = np.atleast_1d(np.asarray(theta, dtype=float))
        out = np.empty_like(th_arr)
        flat = out.ravel()
        for i, th in enumerate(th_arr.ravel()):
            flat[i] = _r_orbit_scalar(float(th))
        return float(out.ravel()[0]) if scalar else out

    return A, r_orbit
