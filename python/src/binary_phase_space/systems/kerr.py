"""Kerr-evolved coherent state: the crescent (banana) state.

A coherent state |alpha> evolved under the Kerr Hamiltonian

    H = chi * n_hat^2          (n_hat = a-dagger a)

for a time t shears in phase space. For small chi*t the coherent disk
bends into a crescent; the same Hamiltonian, run to chi*t = pi, produces
a two-component cat (Yurke-Stoler), so the crescent is the dynamical
bridge from a single coherent state to a cat.

We parameterize by the dimensionless shear phi = chi * t. The evolution
exp(-i phi n^2) is diagonal in the Fock basis, so it is applied exactly
as a phase on each number-basis amplitude (no ODE integration).

Frame alignment. A sheared crescent has a covariance ellipse whose
principal axes are tilted off the (x, p) coordinate axes, i.e. a
nonzero <xp> cross-term. The manuscript's cell overlays (Heisenberg
cell A, quorum cell a-tilde) are axis-aligned ellipses, valid for
states with diagonal covariance. The convolved portrait (the theta
integral) is rotation-invariant and so is unaffected by orientation,
but the cross-sections and the A / a-tilde overlays read cleanest when
a principal axis is aligned with a coordinate axis.

We therefore rotate the state rigidly in phase space, exp(i t n_hat),
choosing t so the covariance cross-term sigma_xp vanishes and the major
principal axis lands on a coordinate axis. A phase-space rotation
commutes with the Kerr shape (it only spins the crescent, preserving it
exactly), and afterward the covariance is diagonal, so the axis-aligned
overlay machinery is exact rather than approximate. This is the "rotated
ellipsoid" case the manuscript notes is reachable through the symplectic
covariance of the family; here we simply choose the frame in which the
ellipse is non-rotated.

The angle t is solved for numerically (minimize |sigma_xp(t)|), not taken
from the covariance eigenvector: a phase-space rotation by t rotates the
Wigner function by t, but the eigenvector angle of the covariance is a
different quantity and does not equal t (nor a fixed multiple of it), so
feeding the eigenvector angle into exp(i t n_hat) leaves a residual
cross-term. Solving for the nulling t directly is convention-proof and
diagonalizes the covariance to numerical precision.

Reference for Kerr cat dynamics: Yurke and Stoler, Phys. Rev. Lett. 57,
13 (1986); Kirchmair et al., Nature 495, 205 (2013) for observed
phase-space crescents.
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from ..state import DisplayWindow, State, build_state_from_qobj

# Coherent amplitude (real, placed on +q). |alpha|^2 photons. alpha=3
# (9 photons) gives a crescent whose interior spiral fringes are bold in
# the portrait while the state stays recognizably crescent-shaped; at the
# fixed shear phi below the covariance retains a defined (if modest) major
# axis, so the principal-frame alignment is well posed. Smaller alpha (e.g.
# 2) is more anisotropic but its fine fringes read faintly; larger alpha
# isotropizes the covariance and rounds the crescent toward a ring.
KERR_ALPHA_DEFAULT = 3.0

# Shear phi = chi*t. phi=0 is the bare coherent state; phi=pi is a 2-cat.
# phi ~ 0.3 gives a pronounced crescent with a CLEAR major axis (the
# rotated covariance is anisotropic, sigma_par/sigma_perp ~ 1.5); by
# phi ~ 0.5 the second moments go isotropic and there is no longer a
# well-defined "furthest extent" axis to align. 0.3 is the crescent
# sweet spot: visibly bent, modest Wigner negativity, alignable axis.
KERR_PHI_DEFAULT = 0.3

KERR_N_DEFAULT = 80


def _full_covariance(psi: qt.Qobj, N: int) -> tuple[np.ndarray, float, float]:
    """Full 2x2 (x, p) covariance of psi, plus the means <x>, <p>.

    Includes the symmetrized cross-term sigma_xp = <xp + px>/2 - <x><p>,
    which is nonzero for a sheared state.
    """
    x = qt.position(N)
    p = qt.momentum(N)
    xm = float(qt.expect(x, psi))
    pm = float(qt.expect(p, psi))
    sxx = float(qt.expect(x * x, psi)) - xm ** 2
    spp = float(qt.expect(p * p, psi)) - pm ** 2
    sxp = 0.5 * float(qt.expect(x * p + p * x, psi)) - xm * pm
    return np.array([[sxx, sxp], [sxp, spp]]), xm, pm


def _alignment_rotation_angle(psi: qt.Qobj, N: int) -> float:
    """Phase-space rotation angle that diagonalizes psi's covariance.

    A phase-space rotation is implemented as exp(i t n_hat); acting on the
    Wigner function it rotates phase space by t. To bring a state into its
    covariance principal frame we need the t that nulls the symmetrized
    cross-term sigma_xp.

    Naively one might use the covariance eigenvector angle directly, but the
    eigenvector angle and the required rotation angle t are not equal (they
    do not even differ by a fixed factor across states), because the
    eigenvector lives in the (x, p) plane while t parameterizes the
    metaplectic rotation generator n_hat. We therefore solve for t directly:
    minimize |sigma_xp(t)| over t in [0, pi). The covariance of a rotated
    state is a smooth, single-well function of t on this interval (one
    minimum where the major axis meets a coordinate axis), so a bounded
    scalar minimization converges robustly.

    Returns the rotation angle t (radians) to be applied as exp(i t n_hat).
    """
    from scipy.optimize import minimize_scalar

    num = qt.num(N)

    def abs_cross_term(t: float) -> float:
        R = (1j * t * num).expm()
        cov, _, _ = _full_covariance((R * psi).unit(), N)
        return abs(float(cov[0, 1]))

    res = minimize_scalar(abs_cross_term, bounds=(0.0, np.pi), method="bounded")
    return float(res.x)


def kerr_state(
    *,
    alpha: float = KERR_ALPHA_DEFAULT,
    phi: float = KERR_PHI_DEFAULT,
    align_major_axis: bool = True,
    name: str | None = None,
    hbar: float = 1.0,
    N: int = KERR_N_DEFAULT,
) -> State:
    """Build a Kerr-sheared coherent state (crescent / banana).

    ``alpha``  coherent amplitude (real); |alpha|^2 is the mean photon
               number. Placed on the +q axis before evolution.
    ``phi``    dimensionless shear chi*t. phi=0 is the bare coherent
               state; phi=pi is a 2-cat. The default gives a crescent.
    ``align_major_axis``  if True (default), rigidly rotate the state in
               phase space so the major principal axis of its covariance
               lands on a coordinate axis, making the covariance diagonal
               and the axis-aligned cell overlays exact.
    ``N``      Fock-basis truncation.
    """
    if N <= 0:
        raise ValueError("N must be positive.")

    # Coherent state on +q axis, then exact diagonal Kerr phase.
    psi0 = qt.coherent(N, alpha)
    n_idx = np.arange(N)
    amps = psi0.full().ravel() * np.exp(-1j * phi * n_idx.astype(float) ** 2)
    psi = qt.Qobj(amps.reshape(N, 1)).unit()

    if align_major_axis:
        # Rotate into the covariance principal frame by the rotation that
        # nulls the cross-term sigma_xp (solved numerically; see
        # _alignment_rotation_angle for why the eigenvector angle cannot be
        # used directly). Afterward sigma_xp ~ 0 to numerical precision and
        # the axis-aligned cell overlays are exact.
        t = _alignment_rotation_angle(psi, N)
        R = (1j * t * qt.num(N)).expm()
        psi = (R * psi).unit()

    return build_state_from_qobj(
        name=name or f"kerr_alpha{alpha:g}_phi{phi:g}",
        qobj=psi,
        window=DisplayWindow(x_lim=0.0, p_lim=0.0),
        hbar=hbar,
    )
