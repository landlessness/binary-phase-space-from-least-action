"""Panel drawers for the binary phase-space figures.

  portrait_panel        2-D portrait: the signed Stern-Brocot radial
                        density stretched to the semiclassical orbit
                        (circle for the harmonic oscillator). Overlaid
                        with the Heisenberg cell A and quorum cell ã.
  cross_section_panel   the 1-D position-density skyline.
  partition_panel       the (x_mu -> x-tilde) staircase with erasure scars.

The DC floor (the lowest ring density) is mapped to a light shade of
the colormap rather than to white, so the lowest band still reads as
faint red against the panel background.

OPERATING PRINCIPLE (do not drift from this):
The binary paper's cell overlays MUST be identical to the Wigner paper's.
The Heisenberg cell uses (state.rs.Delta_x, state.rs.Delta_p) for the
semi-axes and (state.cell_center_x, 0.0) for the center -- exactly as
the Wigner code's _draw_cells in wigner_resolution/figures/panels.py.
The quorum cell is derived via cells.quorum_cell(...) -- same call.
Cell sizing is NEVER derived from the orbit (the orbit is the partition
stretch only; it has no role in cell sizing). For harmonic rows that
carry no state, an equivalent minimum-uncertainty cell (sqrt(A), sqrt(A),
center origin) is synthesized to preserve a single drawing path.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..cells import HeisenbergCell, quorum_cell
from ..cross_section import compute_cross_section
from .overlays import heisenberg_cell_patch, quorum_cell_patch, LINEWIDTH
from ._portrait import portrait_field

_RDBU = plt.get_cmap("RdBu_r")
FILL = _RDBU(0.82)


def _heisenberg_for_row(state, A):
    """Build the HeisenbergCell to overlay, matching the Wigner code.

    State-based rows (squeezed, Morse, double-well, Kerr): semi-axes are
    (state.rs.Delta_x, state.rs.Delta_p), center at (state.cell_center_x, 0)
    -- identical to the Wigner code's _draw_cells.

    Harmonic rows (no state, action level only): synthesize the equivalent
    minimum-uncertainty cell -- Delta_x = Delta_p = sqrt(A), centered at
    origin -- so the same drawing helper works for both paths.
    """
    if state is not None:
        return HeisenbergCell(
            Delta_x=state.rs.Delta_x,
            Delta_p=state.rs.Delta_p,
            center=(state.cell_center_x, 0.0),
        )
    s = float(np.sqrt(A))
    return HeisenbergCell(Delta_x=s, Delta_p=s, center=(0.0, 0.0))


def _draw_cells(ax, heisenberg, *, hbar: float = 1.0,
                edgecolor: str = "black", linewidth: float = LINEWIDTH):
    """Overlay the Heisenberg cell A and the quorum cell a-tilde.

    Mirrors the Wigner code's _draw_cells (column 5 mode: A + a-tilde,
    no bitangent blob). Takes the HeisenbergCell directly so both the
    state-based and harmonic paths use the same drawing function.
    """
    ax.add_patch(heisenberg_cell_patch(
        heisenberg, edgecolor=edgecolor, linewidth=linewidth,
    ))
    qc = quorum_cell(heisenberg, hbar=hbar)
    ax.add_patch(quorum_cell_patch(
        qc, edgecolor=edgecolor, linewidth=linewidth,
    ))


def portrait_panel(ax, A, *, r_orbit=None, state=None, hbar: float = 1.0, n_img: int = 1400):
    """Draw the phase-space portrait for action level A with cell overlays.

    Cell overlays follow the Wigner-paper convention exactly: A has
    semi-axes (Delta_x, Delta_p) from the state's covariance and is
    centered at state.cell_center_x; a-tilde is its symplectic polar
    dual. For harmonic rows (no state) the equivalent minimum-uncertainty
    cell at the origin is used.
    """
    field, lim, support, extent = portrait_field(A, r_orbit=r_orbit, n_img=n_img)

    # Color scale: map the DC floor (lowest nonzero ring) to a light red,
    # not white. Render only the warm half of the diverging palette since
    # the density is non-negative. vmin is set below the floor so the
    # lowest ring is a faint-but-visible red, and zero (background) stays
    # at the panel color.
    nz = field[field > 0]
    if nz.size:
        from matplotlib.colors import ListedColormap
        floor = nz.min()
        top = field.max()
        # Build a colormap from the WARM HALF of RdBu_r only: light red at
        # the DC floor, dark red at the peak. Nothing maps to blue. Zero
        # (background) is masked so it shows the panel color.
        warm = ListedColormap(_RDBU(np.linspace(0.64, 0.95, 256)))  # floor reads as definite red
        masked = np.ma.masked_where(field <= 0, field)
        ax.imshow(
            masked.T,
            extent=[-lim, lim, -lim, lim],
            origin="lower",
            cmap=warm,
            norm=Normalize(vmin=floor - 1e-9, vmax=top),
            interpolation="nearest",
        )
    ax.set_aspect("equal", adjustable="box")

    heis = _heisenberg_for_row(state, A)
    # Thread hbar from the state (Wigner code convention) when available;
    # the harmonic path has no state and uses the panel's hbar default.
    hbar_for_cells = state.hbar if state is not None else hbar
    _draw_cells(ax, heis, hbar=hbar_for_cells)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _axis_ticks(ax, x_lim=lim, y_lim=lim)


def cross_section_panel(ax, A, *, r_orbit=None, n_img: int = 1400):
    """Draw the 1-D cross-section as the portrait sliced at p = 0.

    This is the uniform construction used across every row of the figure:
    the cross-section column is always the portrait of the same row,
    sampled along the horizontal axis (p = 0). Not a marginal, not an
    independent histogram -- one consistent operation for all states,
    matching the Wigner paper's W(x, 0) convention.

    For the harmonic case (r_orbit = circle of radius sqrt(A)) this
    slice equals compute_cross_section(A) to floating point, since the
    radial profile and the p=0 slice agree on a circle. For elliptical
    or anharmonic orbits the slice captures the orbit-stretch in the
    x-direction.
    """
    field, lim, support, extent = portrait_field(A, r_orbit=r_orbit, n_img=n_img)
    n = field.shape[0]
    mid = n // 2
    xs = np.linspace(-lim, lim, n)
    # p = 0 corresponds to the middle row of the field (YY = 0 there).
    # field is indexed [ix, iy] so the p=0 slice is field[:, mid].
    slice_y = field[:, mid]

    # Render as a filled skyline at the raster resolution. The discrete
    # bins show as flat plateaus by construction (portrait_field uses
    # nearest-bin assignment and interpolation='nearest').
    ax.fill_between(xs, slice_y, 0, where=(slice_y > 0),
                    color=FILL, alpha=0.85, linewidth=0)
    ax.plot(xs, slice_y, color="black", lw=LINEWIDTH)
    ax.set_xlim(-lim, lim)
    ymax = float(slice_y.max()) if slice_y.max() > 0 else 1.0
    ax.set_ylim(0, ymax * 1.15)
    _axis_ticks(ax, x_lim=lim)  # density y-axis left to matplotlib default


def _axis_ticks(ax, *, x_lim: float, y_lim: float | None = None):
    """Place 3 nice symmetric ticks on the x (and optionally y) axis,
    spanning the actual axis extent rather than a fixed ±sqrt(A) anchor.

    Used for harmonic rows that carry no state. State-based rows go
    through _apply_window_from_state instead, which mirrors the Wigner
    code's _apply_window exactly.

    Uses nice_symmetric_ticks from the Wigner code's ticks module --
    same 1-2-5 mantissa step picking as the Wigner figures.
    """
    from ..ticks import nice_symmetric_ticks
    xt = nice_symmetric_ticks(x_lim, target_count=3)
    ax.set_xticks(list(xt))
    ax.set_xticklabels([f"{t:.1f}" for t in xt])
    if y_lim is not None:
        yt = nice_symmetric_ticks(y_lim, target_count=3)
        ax.set_yticks(list(yt))
        ax.set_yticklabels([f"{t:.1f}" for t in yt])


def _apply_window_from_state(ax, state, *, expand_x=None, expand_p=None):
    """Apply state.window's limits and ticks to ax, exactly as the
    Wigner code's _apply_window does, with an optional expansion to
    contain our hard-edged partition data when it exceeds the Wigner
    window. Ticks are taken verbatim from state.window so they match
    the Wigner paper for the same state.

    expand_x: (xmin, xmax) data extent that must be contained. If wider
              than the Wigner window, the axis limits expand to fit.
              Ticks are unchanged (still inside the new wider range).
    expand_p: same for the p axis. Pass None for cross-section panel
              (no p axis).
    """
    w = state.window
    x_min, x_max = w.x_min, w.x_max
    if expand_x is not None:
        x_min = min(x_min, expand_x[0])
        x_max = max(x_max, expand_x[1])
    ax.set_xlim(x_min, x_max)
    if w.x_ticks:
        ax.set_xticks(list(w.x_ticks))
        ax.set_xticklabels([f"{t:.1f}" for t in w.x_ticks])

    if expand_p is not None:
        p_min, p_max = w.p_min, w.p_max
        p_min = min(p_min, expand_p[0])
        p_max = max(p_max, expand_p[1])
        ax.set_ylim(p_min, p_max)
        if w.p_ticks:
            ax.set_yticks(list(w.p_ticks))
            ax.set_yticklabels([f"{t:.1f}" for t in w.p_ticks])


# ---------------------------------------------------------------------------
# Partition panel (new leftmost column)
# ---------------------------------------------------------------------------

from ._partition import compute_partition  # noqa: E402


def partition_panel(ax, A, *, r_orbit=None, lim=None):
    """Draw the (blob center -> selected microstate) panel with downward
    erasure scars and a black dot at each landed point.

    Matches the middle column of plot_encoding_grid.R: faint gray scars
    from (q_b, q_mu) down to (q_b, q_mu - eps), and black dots at
    (q_b, q_mu). Many centers collapse onto the same microstate, so the
    dots form horizontal bands -- the staircase treads.

    For non-circular orbits, the staircase is stretched two-sided so the
    partition column lives in the same physical x as the cross-section
    and portrait columns. See compute_partition for the stretch.
    """
    q_b, q_mu, eps_scaled, max_ax = compute_partition(A, r_orbit=r_orbit)

    if lim is None:
        lim = max_ax * 1.15

    # Faint downward scars (vertical line per center, from landing to
    # landing - eps). The R uses linewidth=0.2, alpha=0.05, color gray90.
    ax.vlines(q_b, q_mu - eps_scaled, q_mu, colors="0.85", alpha=0.05, linewidth=0.2)
    # Black landing dots.
    ax.plot(q_b, q_mu, marker="o", linestyle="none", color="black",
            markersize=0.6, markeredgewidth=0)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    _axis_ticks(ax, x_lim=lim, y_lim=lim)
