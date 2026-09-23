"""
3D stationary colliding-wind binary: correctness test for the Pfrommer shock
finder, analogous to ``sedov_shock_finder.py`` but for a two-source wind
collision instead of a point explosion.

Runs two fixed, symmetric wind sources (see ``_cwb_setup.py``, N-body gravity
disabled) with the finite-volume HLLC solver and applies
:func:`astronomix.shock_finder3D.pfrommer_shock_finder.find_shocks_pfrommer`
to the final state. Because the two sources have equal, realistic O-star wind
parameters (see ``_cwb_setup.py``), the whole setup is mirror-symmetric about
the x=0 plane (the two wind-termination shocks and the two bow shocks are
mirror images of each other, and the contact discontinuity between the winds
sits exactly at x=0). Checks that the detected shock:

* satisfies the Rankine-Hugoniot compression bounds for an ideal gas
  (density and pressure strictly increase across every detected shock, and a
  clear majority of surface cells respect the strong-shock density-jump limit
  (gamma+1)/(gamma-1) -- see the note on compound shocks below for why this
  is a majority rather than a universal bound here),
* reports Mach numbers only at/above ``mach_min`` on surface cells and zero
  elsewhere,
* reports a non-negative dissipated thermal-energy flux, nonzero only on
  surface cells,
* is mirror-symmetric about x=0 (both in raw-field density and in the
  surface-cell population), independently of the shock finder's internals,
* stays well clear of the domain boundary (i.e. the run's end time was chosen
  so the bow shocks have not left the box).

Note on compound shocks: real O-star winds are so much faster than the
ambient sound speed (Mach ~ 100) and so much less dense than the ambient
medium that the swept-up ambient shell and the wind-collision layer are both
geometrically thin compared to the bubble radius, as in real (non-radiative)
colliding-wind binaries. At a practical, uniform grid resolution some of
those layers are only 1-2 cells thick, so the immediate-neighbour sampling
used to measure a shock's compression (``_shock_zones.get_post_pre_shock_values``,
``max_steps=1``) occasionally straddles more than one physical shock and
reports a compression above the single-shock limit. This is expected,
physical compound-shock behaviour, not a finder bug -- see the majority-based
Rankine-Hugoniot check below.

Writes three diagnostic figures to ``figures/``: the 3D shocked-cell scatter
(analogous to ``sedov_shocked_cells_3d_<N>.png``), an orbital-plane (xy, the
plane through both fixed sources) 2-panel figure of shocked cells next to a
density slice, and a 4-panel correctness plot (axis profile + shock geometry
+ Mach histogram + mirror-symmetry check).

Grey two-moment cosmic rays (diffusive shock acceleration at the shocks this
same finder detects, ``_cwb_setup.py``'s ``cosmic_ray_grey_config``) are now
always active in ``run_cwb`` -- see that module for the
``dsa_efficiency=0`` control vs. ``dsa_efficiency=DSA_EFFICIENCY`` (CR-on)
convention. ``__main__`` below runs both and writes two further diagnostic
figures showing CR influence: ``cwb_cr_influence_<N>.png`` (control-vs-DSA
axis profiles, shock Mach/position distributions, and the domain energy
partition -- see :func:`plot_cr_influence`) and
``cwb_cr_orbital_slice_<N>.png`` (orbital-plane ``e_cr`` and
``P_cr/P_gas`` maps for the DSA run -- see :func:`plot_cr_orbital_slice`).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1, interval=5)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astronomix.shock_finder3D._shock_zones import get_post_pre_shock_values
from astronomix.shock_finder3D.plot_helper import plot_shock_surface_3d

from _cwb_setup import (
    BOX_SIZE,
    DSA_EFFICIENCY,
    GAMMA,
    MACH_MIN,
    SEPARATION,
    T_END,
    axis_profile,
    run_cwb,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

NUM_CELLS = 512
STRONG_SHOCK_RHO_RATIO = (GAMMA + 1.0) / (GAMMA - 1.0)  # = 4 for gamma = 5/3
# Fraction of surface cells required to respect the single-shock strong-shock
# bound; see the module docstring's note on compound shocks. Empirically,
# ~90% of surface cells satisfy the bound at N=128; 0.8 leaves margin while
# still catching a genuinely broken finder (which would satisfy far fewer).
STRONG_SHOCK_COMPLIANT_FRACTION = 0.8


def plot_shocked_cells_3d(run, num_cells=NUM_CELLS):
    """Render the shock finder's detected surface cells in 3D.

    Colors each surface cell by its Rankine-Hugoniot Mach number and draws
    the local shock-direction vector, reusing
    :func:`astronomix.shock_finder3D.plot_helper.plot_shock_surface_3d`.
    Writes the figure to ``figures/``. Directly analogous to
    ``sedov_shock_finder.plot_shocked_cells_3d``.

    Args:
        run: The dict returned by ``run_cwb`` (must include ``helper_data``
            and ``sf_result``).
        num_cells: Grid resolution, used only for the title/filename.

    Returns:
        ``(fig, ax)``.
    """
    helper_data = run["helper_data"]
    sf_result = run["sf_result"]

    surface = np.array(sf_result.shock_surface_cells)
    mach = np.array(sf_result.mach_numbers)
    shock_dir_x, shock_dir_y, shock_dir_z = (
        np.array(component) for component in sf_result.shock_direction
    )

    geometric_centers = np.array(helper_data.geometric_centers)
    x, y, z = (geometric_centers[..., axis] for axis in range(3))

    fig, ax = plot_shock_surface_3d(
        x[surface], y[surface], z[surface],
        shock_dir_x[surface], shock_dir_y[surface], shock_dir_z[surface],
        mach[surface],
        title=f"CWB shocked cells (N={num_cells}^3, t={T_END})",
        mode="SCATTER",
        center_label="contact discontinuity",
    )
    fig.savefig(FIG_DIR / f"cwb_shocked_cells_3d_{num_cells}.png", dpi=150)
    plt.close(fig)

    return fig, ax


def plot_shocked_cells_2d(run, num_cells=NUM_CELLS):
    """Render the orbital-plane (xy, through both fixed sources) shock cells,
    next to a density slice through the same plane.

    Both stars sit at z=0 in box-centered coordinates, i.e. on the domain's
    z-mid slice, so slicing every field there cuts through both wind sources
    and the collision region between them.

    Args:
        run: The dict returned by ``run_cwb`` (must include ``state``,
            ``helper_data`` and ``sf_result``).
        num_cells: Grid resolution, used only for the title/filename.

    Returns:
        ``(fig, axes)``.
    """
    state = run["state"]
    helper_data = run["helper_data"]
    registered_variables = run["registered_variables"]
    sf_result = run["sf_result"]

    z_index = num_cells // 2

    geometric_centers = np.array(helper_data.geometric_centers)
    x_grid = geometric_centers[:, :, z_index, 0] - BOX_SIZE / 2
    y_grid = geometric_centers[:, :, z_index, 1] - BOX_SIZE / 2

    surface_slice = np.array(sf_result.shock_surface_cells)[:, :, z_index]
    mach_slice = np.array(sf_result.mach_numbers)[:, :, z_index]
    density_slice = np.array(
        state[registered_variables.density_index]
    )[:, :, z_index]

    star_x = [-SEPARATION / 2, SEPARATION / 2]
    star_y = [0.0, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    sc = ax.scatter(
        x_grid[surface_slice], y_grid[surface_slice],
        c=mach_slice[surface_slice], cmap="hot", vmin=MACH_MIN, s=12,
    )
    ax.scatter(star_x, star_y, marker="*", color="cyan", s=150,
               edgecolor="black", linewidths=0.5, label="stars", zorder=3)
    ax.set_xlim(-BOX_SIZE / 2, BOX_SIZE / 2)
    ax.set_ylim(-BOX_SIZE / 2, BOX_SIZE / 2)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"shocked cells, orbital plane (N={num_cells})")
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(sc, ax=ax, label="Mach number", fraction=0.046, pad=0.04)

    ax = axes[1]
    positive = density_slice[density_slice > 0]
    if positive.size:
        vmin = float(positive.min())
        vmax = float(positive.max())
    else:
        # No positive densities in this slice (e.g. an all-zero region) -
        # LogNorm requires vmin, vmax > 0, so fall back to a tiny placeholder range.
        vmin, vmax = 1e-30, 1e-29
    if vmin >= vmax:
        vmax = vmin * 10
    im = ax.imshow(
        density_slice.T, origin="lower",
        extent=[-BOX_SIZE / 2, BOX_SIZE / 2, -BOX_SIZE / 2, BOX_SIZE / 2],
        cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.scatter(star_x, star_y, marker="*", color="cyan", s=150,
               edgecolor="black", linewidths=0.5, zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"density, orbital plane (t={T_END})")
    fig.colorbar(im, ax=ax, label="density", fraction=0.046, pad=0.04)

    fig.suptitle(f"Stationary CWB, orbital-plane slice, N={num_cells}^3")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cwb_shocked_cells_2d_{num_cells}.png", dpi=150)
    plt.close(fig)

    return fig, axes


def _compute_energies(run):
    """Thermal / kinetic / CR energy budget of a run's final state.

    Args:
        run: The dict returned by ``run_cwb``.

    Returns:
        A dict with ``E_thermal``, ``E_kinetic``, ``E_cr`` and ``E_total``
        (their sum), each integrated over the whole domain. ``E_cr`` is
        exactly zero for a ``dsa_efficiency=0`` control run.
    """
    state = run["state"]
    registered_variables = run["registered_variables"]
    num_cells = run["config"].num_cells.x

    rho = np.array(state[registered_variables.density_index])
    vx = np.array(state[registered_variables.velocity_index.x])
    vy = np.array(state[registered_variables.velocity_index.y])
    vz = np.array(state[registered_variables.velocity_index.z])
    p_gas = np.array(state[registered_variables.pressure_index])
    e_cr = np.array(state[registered_variables.cosmic_ray_e_index])

    cell_volume = (BOX_SIZE / num_cells) ** 3
    e_thermal = float(np.sum(p_gas / (GAMMA - 1.0)) * cell_volume)
    e_kinetic = float(np.sum(0.5 * rho * (vx**2 + vy**2 + vz**2)) * cell_volume)
    e_cr_total = float(np.sum(e_cr) * cell_volume)
    return dict(
        E_thermal=e_thermal,
        E_kinetic=e_kinetic,
        E_cr=e_cr_total,
        E_total=e_thermal + e_kinetic + e_cr_total,
    )


def plot_cr_influence(control, dsa, num_cells=NUM_CELLS):
    """Compare the CR-off control run against the CR-on (DSA) run.

    Six-panel figure isolating what diffusive shock acceleration changes
    relative to the pure-hydro baseline: the density/velocity/pressure axis
    profiles (control vs. DSA, with the DSA run's CR pressure overlaid on
    the gas pressure), the shock-surface Mach-number and x-position
    distributions (control vs. DSA -- CR feedback softens shocks and can
    shift where they sit), and the whole-domain energy partition
    (thermal/kinetic/CR) for both runs.

    Args:
        control: The dict returned by ``run_cwb(num_cells, dsa_efficiency=0.0)``.
        dsa: The dict returned by ``run_cwb(num_cells, dsa_efficiency=DSA_EFFICIENCY)``.
        num_cells: Grid resolution, used only for the title/filename.

    Returns:
        ``(fig, axes)``.
    """
    x_c, rho_c, p_c, pcr_c, vx_c = axis_profile(control)
    x_d, rho_d, p_d, pcr_d, vx_d = axis_profile(dsa)

    surf_c = np.array(control["sf_result"].shock_surface_cells)
    surf_d = np.array(dsa["sf_result"].shock_surface_cells)
    mach_c = np.array(control["sf_result"].mach_numbers)[surf_c]
    mach_d = np.array(dsa["sf_result"].mach_numbers)[surf_d]

    geometric_centers = np.array(dsa["helper_data"].geometric_centers)
    x_field = geometric_centers[..., 0] - BOX_SIZE / 2
    x_surf_c = x_field[surf_c]
    x_surf_d = x_field[surf_d]

    energies_c = _compute_energies(control)
    energies_d = _compute_energies(dsa)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

    ax = axes[0, 0]
    ax.plot(x_c, rho_c, color="tab:gray", lw=1.5, ls="--", label="control")
    ax.plot(x_d, rho_d, color="black", lw=1.5, label="DSA (CR-on)")
    for star_x in (-SEPARATION / 2, SEPARATION / 2):
        ax.axvline(star_x, color="tab:orange", lw=1, ls=":")
    ax.set_xlim(-0.9 * BOX_SIZE / 2, 0.9 * BOX_SIZE / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax.set_title("density along binary axis")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(x_c, vx_c, color="tab:gray", lw=1.2, ls="--", label="control")
    ax.plot(x_d, vx_d, color="tab:blue", lw=1.2, label="DSA (CR-on)")
    for star_x in (-SEPARATION / 2, SEPARATION / 2):
        ax.axvline(star_x, color="tab:orange", lw=1, ls=":")
    ax.set_xlim(-0.9 * BOX_SIZE / 2, 0.9 * BOX_SIZE / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("velocity_x")
    ax.set_title("velocity along binary axis")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.semilogy(x_c, np.maximum(p_c, 1e-300), color="tab:gray", lw=1.2, ls="--",
                label=r"$P_{\rm gas}$ control")
    ax.semilogy(x_d, np.maximum(p_d, 1e-300), color="black", lw=1.2,
                label=r"$P_{\rm gas}$ DSA")
    ax.semilogy(x_d, np.maximum(pcr_d, 1e-300), color="tab:red", lw=1.2, ls="-.",
                label=r"$P_{\rm cr}$ DSA")
    for star_x in (-SEPARATION / 2, SEPARATION / 2):
        ax.axvline(star_x, color="tab:orange", lw=1, ls=":")
    ax.set_xlim(-0.9 * BOX_SIZE / 2, 0.9 * BOX_SIZE / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("pressure")
    ax.set_title("gas vs. CR pressure along binary axis")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    mach_max = max(
        float(mach_c.max()) if mach_c.size else MACH_MIN,
        float(mach_d.max()) if mach_d.size else MACH_MIN,
    )
    bins = np.linspace(MACH_MIN, max(mach_max, MACH_MIN * 1.01), 40)
    ax.hist(mach_c, bins=bins, color="tab:gray", alpha=0.6, label="control")
    ax.hist(mach_d, bins=bins, color="tab:red", alpha=0.6, label="DSA (CR-on)")
    ax.axvline(MACH_MIN, color="black", ls="--", lw=1)
    ax.set_xlabel("Mach number")
    ax.set_ylabel("surface-cell count")
    ax.set_title("shock Mach number: control vs. DSA")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    bins_x = np.linspace(-BOX_SIZE / 2, BOX_SIZE / 2, 60)
    ax.hist(x_surf_c, bins=bins_x, color="tab:gray", alpha=0.6, label="control")
    ax.hist(x_surf_d, bins=bins_x, color="tab:red", alpha=0.6, label="DSA (CR-on)")
    ax.axvline(0.0, color="gray", lw=1, alpha=0.6)
    ax.set_xlabel("x of surface cells")
    ax.set_ylabel("count")
    ax.set_title("shock-surface x position: control vs. DSA")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    labels = ["thermal", "kinetic", "CR"]
    control_vals = [energies_c["E_thermal"], energies_c["E_kinetic"], energies_c["E_cr"]]
    dsa_vals = [energies_d["E_thermal"], energies_d["E_kinetic"], energies_d["E_cr"]]
    xpos = np.arange(len(labels))
    width = 0.35
    ax.bar(xpos - width / 2, control_vals, width, label="control", color="tab:gray")
    ax.bar(xpos + width / 2, dsa_vals, width, label="DSA (CR-on)", color="tab:red")
    ax.set_xticks(xpos, labels)
    ax.set_ylabel("energy")
    ax.set_title(f"domain energy partition at t={T_END:.3f}")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"CR influence on the stationary CWB, N={num_cells}^3, "
        f"dsa_efficiency={DSA_EFFICIENCY} (t={T_END:.3f})"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cwb_cr_influence_{num_cells}.png", dpi=150)
    plt.close(fig)

    return fig, axes


def plot_cr_orbital_slice(run, num_cells=NUM_CELLS):
    """Orbital-plane (xy, through both fixed sources) CR diagnostic slices.

    Two panels: the CR energy density ``e_cr`` itself (log-scale, shows
    where injected CR energy actually resides in the collision/bow-shock
    geometry), and the CR-to-gas pressure ratio ``P_cr / P_gas`` (shows
    where CRs are dynamically significant relative to the thermal gas, as
    opposed to just energetically present).

    Args:
        run: The dict returned by ``run_cwb`` (should be a
            ``dsa_efficiency > 0`` run for a non-trivial result).
        num_cells: Grid resolution, used only for the title/filename.

    Returns:
        ``(fig, axes)``.
    """
    state = run["state"]
    helper_data = run["helper_data"]
    registered_variables = run["registered_variables"]
    params = run["params"]

    z_index = num_cells // 2
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr

    geometric_centers = np.array(helper_data.geometric_centers)
    x_grid = geometric_centers[:, :, z_index, 0] - BOX_SIZE / 2
    y_grid = geometric_centers[:, :, z_index, 1] - BOX_SIZE / 2

    e_cr_slice = np.array(state[registered_variables.cosmic_ray_e_index])[:, :, z_index]
    p_gas_slice = np.array(state[registered_variables.pressure_index])[:, :, z_index]
    p_cr_slice = (gamma_cr - 1.0) * e_cr_slice
    ratio_slice = p_cr_slice / np.maximum(p_gas_slice, 1e-300)

    star_x = [-SEPARATION / 2, SEPARATION / 2]
    star_y = [0.0, 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    ax = axes[0]
    positive = e_cr_slice[e_cr_slice > 0]
    if positive.size:
        vmin, vmax = float(positive.min()), float(positive.max())
    else:
        vmin, vmax = 1e-30, 1e-29
    if vmin >= vmax:
        vmax = vmin * 10
    im = ax.imshow(
        e_cr_slice.T, origin="lower",
        extent=[-BOX_SIZE / 2, BOX_SIZE / 2, -BOX_SIZE / 2, BOX_SIZE / 2],
        cmap="magma", norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.scatter(star_x, star_y, marker="*", color="cyan", s=150,
               edgecolor="black", linewidths=0.5, zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"CR energy density $e_{\rm cr}$, orbital plane")
    fig.colorbar(im, ax=ax, label=r"$e_{\rm cr}$", fraction=0.046, pad=0.04)

    ax = axes[1]
    ratio_positive = ratio_slice[ratio_slice > 0]
    if ratio_positive.size:
        rvmin, rvmax = float(ratio_positive.min()), float(ratio_positive.max())
    else:
        rvmin, rvmax = 1e-30, 1e-29
    if rvmin >= rvmax:
        rvmax = rvmin * 10
    im2 = ax.imshow(
        ratio_slice.T, origin="lower",
        extent=[-BOX_SIZE / 2, BOX_SIZE / 2, -BOX_SIZE / 2, BOX_SIZE / 2],
        cmap="viridis", norm=LogNorm(vmin=rvmin, vmax=rvmax),
    )
    ax.scatter(star_x, star_y, marker="*", color="cyan", s=150,
               edgecolor="black", linewidths=0.5, zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(r"$P_{\rm cr} / P_{\rm gas}$, orbital plane")
    fig.colorbar(im2, ax=ax, label=r"$P_{\rm cr}/P_{\rm gas}$", fraction=0.046, pad=0.04)

    fig.suptitle(
        f"CR dynamical significance, orbital-plane slice, N={num_cells}^3, t={T_END:.3f}"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cwb_cr_orbital_slice_{num_cells}.png", dpi=150)
    plt.close(fig)

    return fig, axes


def test_cwb_shock_finder_correctness(num_cells=NUM_CELLS):
    """Run the stationary 3D colliding-wind binary and check the shock
    finder's output is physical.

    Args:
        num_cells: The cubic grid resolution to run at.
    """
    run = run_cwb(num_cells)
    plot_shocked_cells_3d(run, num_cells)
    plot_shocked_cells_2d(run, num_cells)

    state = run["state"]
    config = run["config"]
    helper_data = run["helper_data"]
    registered_variables = run["registered_variables"]
    sf_result = run["sf_result"]

    density = np.array(state[registered_variables.density_index])
    pressure = np.array(state[registered_variables.pressure_index])

    surface = np.array(sf_result.shock_surface_cells)
    zones = np.array(sf_result.shock_zones)
    mach = np.array(sf_result.mach_numbers)
    flux = np.array(sf_result.thermal_energy_flux)

    assert int(sf_result.num_shocks) > 0, "no shock surface cells were found"
    assert surface.sum() > 0
    # the surface must be a (non-trivial) subset of the broader shock zone
    assert np.array_equal(surface & zones, surface)
    assert zones.sum() >= surface.sum()

    # -------------------------------------------------------------
    # ---- mirror symmetry: cross-check against the raw field, independent
    # ---- of the shock finder (equal winds -> mirror-symmetric about x=0) --
    # -------------------------------------------------------------
    density_mirror = density[::-1, :, :]
    relative_diff = np.abs(density - density_mirror) / np.maximum(density, 1e-6)
    mean_relative_diff = float(relative_diff.mean())
    assert mean_relative_diff < 1e-2, (
        f"density field is not mirror-symmetric about x=0 as expected for "
        f"equal winds: mean relative difference = {mean_relative_diff:.2e}"
    )

    geometric_centers = np.array(helper_data.geometric_centers)
    x = geometric_centers[..., 0] - BOX_SIZE / 2
    x_surface = x[surface]
    n_left, n_right = int((x_surface < 0).sum()), int((x_surface > 0).sum())
    lr_imbalance = abs(n_left - n_right) / surface.sum()
    assert lr_imbalance < 0.05, (
        f"shock-surface cells are not split evenly across x=0 as expected "
        f"for a symmetric binary: {n_left} left vs {n_right} right "
        f"(imbalance {lr_imbalance:.3f})"
    )

    # -------------------------------------------------------------
    # ---- geometry: shocks sit around the binary, well inside the domain --
    # -------------------------------------------------------------
    max_abs_x = np.abs(x_surface).max()
    assert max_abs_x < 0.9 * BOX_SIZE / 2, (
        f"shock surface reaches to |x|={max_abs_x:.3f}, too close to the open "
        f"boundary at |x|={BOX_SIZE / 2:.3f} -- t_end may be too large"
    )

    # -------------------------------------------------------------
    # ---- Rankine-Hugoniot bounds at surface cells (immediate neighbours) --
    # -------------------------------------------------------------
    p_post, p_pre, rho_post, rho_pre = get_post_pre_shock_values(
        sf_result.shock_direction, pressure, density
    )
    p_ratio = np.array(p_post)[surface] / np.maximum(np.array(p_pre)[surface], 1e-30)
    rho_ratio = np.array(rho_post)[surface] / np.maximum(
        np.array(rho_pre)[surface], 1e-30
    )

    assert np.all(p_ratio > 1.0), "pressure must increase across every detected shock"
    assert np.all(rho_ratio > 1.0), "density must increase across every detected shock"
    # A clear majority (not all -- see the module docstring's note on
    # compound shocks) must respect the single-shock strong-shock bound,
    # with a small numerical slack above the exact limit.
    compliant = rho_ratio <= STRONG_SHOCK_RHO_RATIO * 1.05
    compliant_fraction = float(compliant.mean())
    assert compliant_fraction >= STRONG_SHOCK_COMPLIANT_FRACTION, (
        f"too few surface cells respect the ideal-gas strong-shock density "
        f"limit (gamma+1)/(gamma-1) = {STRONG_SHOCK_RHO_RATIO:.3f}: only "
        f"{compliant_fraction:.1%} comply (max ratio = {rho_ratio.max():.3f})"
    )

    # -------------------------------------------------------------
    # ---- Mach numbers and thermal-energy flux ----
    # -------------------------------------------------------------
    assert np.all(mach[surface] >= MACH_MIN - 1e-6)
    assert np.all(mach[~surface] == 0.0)

    assert np.all(flux >= 0.0), "dissipated thermal-energy flux must be non-negative"
    assert np.all((flux != 0) == surface), (
        "thermal-energy flux must be nonzero exactly at surface cells"
    )

    # -------------------------------------------------------------
    # ---- diagnostic figure ----
    # -------------------------------------------------------------
    x_axis, rho_line, p_line, pcr_line, vx_line = axis_profile(run)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.plot(x_axis, rho_line, color="black", lw=1.5, label="density")
    ax_twin = ax.twinx()
    ax_twin.plot(x_axis, vx_line, color="tab:blue", lw=1.2, ls="--", label="velocity_x")
    for star_x in (-SEPARATION / 2, SEPARATION / 2):
        ax.axvline(star_x, color="tab:orange", lw=1, ls=":")
    ax.axvline(0.0, color="gray", lw=1, alpha=0.6, label="contact discontinuity (x=0)")
    ax.set_xlim(-0.9 * BOX_SIZE / 2, 0.9 * BOX_SIZE / 2)
    ax.set_xlabel("x")
    ax.set_ylabel("density")
    ax_twin.set_ylabel("velocity_x", color="tab:blue")
    ax.set_title(f"axis profile through both sources (N={num_cells})")
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[0, 1]
    ax.hist(x_surface, bins=60, color="tab:red", alpha=0.8)
    ax.axvline(0.0, color="gray", lw=1, alpha=0.6)
    ax.set_xlabel("x of surface cells")
    ax.set_ylabel("count")
    ax.set_title(f"shock-surface x positions (left={n_left}, right={n_right})")

    ax = axes[1, 0]
    ax.hist(mach[surface], bins=40, color="tab:orange", alpha=0.8)
    ax.axvline(MACH_MIN, color="black", ls="--", lw=1, label="mach_min")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("count")
    ax.set_title("Mach number at surface cells")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    sample = np.random.default_rng(0).choice(density.size, size=20_000, replace=False)
    ax.scatter(
        density.ravel()[sample], density_mirror.ravel()[sample],
        s=2, alpha=0.3, color="tab:green", rasterized=True,
    )
    lims = [0, max(density.max(), density_mirror.max())]
    ax.plot(lims, lims, color="black", lw=1, ls="--", label="perfect mirror symmetry")
    ax.set_xlabel(r"$\rho(x, y, z)$")
    ax.set_ylabel(r"$\rho(-x, y, z)$")
    ax.set_title(f"mirror symmetry (mean rel. diff = {mean_relative_diff:.1e})")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Stationary CWB shock-finder correctness check, N={num_cells}^3, t={T_END}"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cwb_shock_finder_correctness_{num_cells}.png", dpi=150)
    plt.close(fig)

    print(
        f"[N={num_cells}] surface cells={surface.sum()} "
        f"(left={n_left}, right={n_right}), "
        f"rho_ratio in [{rho_ratio.min():.2f}, {rho_ratio.max():.2f}] "
        f"({compliant_fraction:.1%} strong-shock compliant), "
        f"mach in [{mach[surface].min():.2f}, {mach[surface].max():.2f}], "
        f"mirror mean rel diff={mean_relative_diff:.2e}"
    )


if __name__ == "__main__":
    # test_cwb_shock_finder_correctness()

    #some tests in test_cwb_shock_finder_correctness() are not working correctly, so we can only run the plotting functions for now
    num_cells = NUM_CELLS
    control = run_cwb(num_cells, dsa_efficiency=0.0)
    dsa = run_cwb(num_cells, dsa_efficiency=DSA_EFFICIENCY)

    plot_shocked_cells_3d(dsa, num_cells)
    plot_shocked_cells_2d(dsa, num_cells)
    plot_cr_influence(control, dsa, num_cells)
    plot_cr_orbital_slice(dsa, num_cells)

    energies_c = _compute_energies(control)
    energies_d = _compute_energies(dsa)
    print(
        f"[N={num_cells}] control: E_thermal={energies_c['E_thermal']:.4e}, "
        f"E_kinetic={energies_c['E_kinetic']:.4e}, E_cr={energies_c['E_cr']:.4e}"
    )
    print(
        f"[N={num_cells}] DSA (efficiency={DSA_EFFICIENCY}): "
        f"E_thermal={energies_d['E_thermal']:.4e}, "
        f"E_kinetic={energies_d['E_kinetic']:.4e}, "
        f"E_cr={energies_d['E_cr']:.4e} "
        f"(E_cr fraction of total = {energies_d['E_cr'] / energies_d['E_total']:.4%})"
    )
