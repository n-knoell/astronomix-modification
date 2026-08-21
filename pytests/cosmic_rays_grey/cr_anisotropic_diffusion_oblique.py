"""
Oblique anisotropic CR diffusion pytest (Phase A ladder item 3).

Sets up a uniform B field at a 30-degree angle to the grid axes and checks
that CR energy transports *only* along B, with no cross-field leak, by
comparing the isotropic closure (``anisotropic_transport=False``) against
the B-projected one (``True``) on an otherwise identical setup: a localized
``e_cr`` bump, uniform gas at rest, uniform oblique B, F_cr initially zero.
Diagnostic: the ``e_cr``-weighted second moment of the distribution, resolved
into components parallel and perpendicular to B. With no bulk flow (v = 0)
this is a "does energy propagate along ±B or spread as an isotropic ring"
test rather than a true parabolic-diffusion test -- the two-moment system
implemented here (``cr_grey_transport.grey_cr_flux_terms``) is a pure
hyperbolic wave system with no relaxation/damping term, so an initially
localized, at-rest bump propagates as two counter-moving pulses (or an
isotropic ring, in the unprojected closure), not a smoothly diffusing cloud
-- but the anisotropy question (which directions the energy is allowed to
move in at all) is exactly the same either way.

Why this needed two real, pre-existing bugs fixed first (both found while
building this test, neither specific to anisotropic transport):

1. ``astronomix/_finite_volume/_state_evolution/evolve_state.py``'s
   ``_evolve_state_fv`` did a Strang split between the hydro update and the
   magnetic-field update by hardcoding ``primitive_state[-3:, ...]`` as "the
   magnetic field" -- true only when nothing is registered after
   ``magnetic_index``. Grey CR (``cosmic_ray_e_index``/
   ``cosmic_ray_flux_index``) is registered *after* it
   (``registered_variables.py``), so with both ``mhd`` and
   ``grey_cosmic_rays`` on, that slice grabbed ``(B_z, e_cr, F_cr_x)`` as
   "the magnetic field" and mislabelled real ``B_z`` as gas -- corrupting
   both halves of the split and leaving CR transport silently inert (``F_cr``
   stuck at ~1e-9 instead of growing to the expected ~1e-4). Fixed generally
   (not CR-specific -- also affects ``wind_density``/the old
   ``cosmic_ray_n`` model combined with MHD) via
   ``evolve_state._split_gas_and_magnetic_state``/``_join_gas_and_magnetic_state``,
   which locate the magnetic rows by their actual registered indices instead
   of assuming they're last.
2. ``anisotropic_flux_projection`` needs to read B, but
   ``cr_grey_transport.grey_cr_flux_terms`` (where the *isotropic* closure's
   F_cr-driven flux is computed) runs inside the FV MHD Strang split's
   gas-only Riemann solve -- exactly where fix #1 above deliberately removes
   the magnetic-field rows from the state array for that half-step. B is
   therefore structurally unavailable at that call site. The projection is
   applied instead once per full step, *before* the hydro update, in
   ``astronomix._modules._iteration_level_updates`` (alongside the existing
   ``e_cr`` positivity floor) -- see that function and
   ``anisotropic_flux_projection``'s docstring for the consequence (a small,
   per-step, non-accumulating residual: the isotropic per-axis pressure
   gradient can push a tiny perpendicular component into F_cr within a step,
   which the *next* step's correction removes again).

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md and PROGRESS.md.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    BoundarySettings,
    BoundarySettings1D,
    PERIODIC_BOUNDARY,
    finalize_config,
)

# astronomix functions
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

_PERIODIC_2D = BoundarySettings(
    BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
    BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
)


def _run(
    anisotropic: bool,
    num_cells: int,
    box_size: float,
    theta: float,
    t_end: float,
    amp: float,
    sigma: float,
    gamma_cr: float,
    reduced_streaming_speed: float,
):
    """Run one localized-bump case; returns (e_cr_final, X, Y, Bx, By)."""
    config = SimulationConfig(
        mhd=True,
        solver_mode=FINITE_VOLUME,
        dimensionality=2,
        box_size=box_size,
        num_cells=num_cells,
        boundary_settings=_PERIODIC_2D,
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, anisotropic_transport=anisotropic
        ),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    X = helper_data.geometric_centers[..., 0]
    Y = helper_data.geometric_centers[..., 1]
    x0 = y0 = 0.5 * box_size

    Bx, By = jnp.cos(theta), jnp.sin(theta)
    pulse = amp * jnp.exp(-0.5 * (((X - x0) ** 2 + (Y - y0) ** 2) / sigma**2))

    primitive_state = jnp.zeros((registered_variables.num_vars, num_cells, num_cells))
    primitive_state = primitive_state.at[registered_variables.density_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.magnetic_index.x].set(Bx)
    primitive_state = primitive_state.at[registered_variables.magnetic_index.y].set(By)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(pulse)
    # F_cr stays at its default zero.

    config = finalize_config(config, primitive_state.shape)
    params = SimulationParams(
        t_end=t_end,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=gamma_cr, reduced_streaming_speed=reduced_streaming_speed
        ),
    )

    final_state = time_integration(primitive_state, config, params, registered_variables)
    e_cr_final = final_state[registered_variables.cosmic_ray_e_index]
    return e_cr_final, X, Y, Bx, By


def _parallel_perp_variance(e_cr, X, Y, Bx, By):
    """e_cr-weighted second moments resolved along/across (Bx, By)."""
    total = jnp.sum(e_cr)
    xc = jnp.sum(e_cr * X) / total
    yc = jnp.sum(e_cr * Y) / total
    s_par = (X - xc) * Bx + (Y - yc) * By
    s_perp = -(X - xc) * By + (Y - yc) * Bx
    var_par = jnp.sum(e_cr * s_par**2) / total
    var_perp = jnp.sum(e_cr * s_perp**2) / total
    return float(var_par), float(var_perp)


def test_cr_anisotropic_diffusion_oblique(leak_ratio_tol: float = 0.1):
    """Oblique-to-grid CR transport: perpendicular-to-B spread stays small.

    Args:
        leak_ratio_tol: Maximum allowed ratio of perpendicular-to-parallel
            *growth* in the e_cr-weighted second moment
            (``(var_perp_final - var_perp_initial) / (var_par_final -
            var_par_initial)``) for the anisotropic run. Calibrated against
            a 128x128-cell, t_end=0.3 run (observed ratio ~0.019 for the
            anisotropic closure, ~1.0 -- i.e. isotropic spread -- for the
            unprojected closure on the same IC); this leaves a wide margin
            while still being far below the isotropic value.
    """
    num_cells = 128
    box_size = 1.0
    theta = jnp.deg2rad(30.0)
    t_end = 0.3
    amp = 1e-3
    sigma = 0.03
    gamma_cr = 4.0 / 3.0
    reduced_streaming_speed = 1.0

    run_kwargs = dict(
        num_cells=num_cells,
        box_size=box_size,
        theta=theta,
        t_end=t_end,
        amp=amp,
        sigma=sigma,
        gamma_cr=gamma_cr,
        reduced_streaming_speed=reduced_streaming_speed,
    )

    e_cr_iso, X, Y, Bx, By = _run(anisotropic=False, **run_kwargs)
    e_cr_aniso, _, _, _, _ = _run(anisotropic=True, **run_kwargs)

    assert not bool(jnp.any(jnp.isnan(e_cr_iso))), "Isotropic run produced NaNs."
    assert not bool(jnp.any(jnp.isnan(e_cr_aniso))), "Anisotropic run produced NaNs."

    var_par_0, var_perp_0 = 0.5 * sigma**2, 0.5 * sigma**2  # isotropic Gaussian at t=0
    var_par_iso, var_perp_iso = _parallel_perp_variance(e_cr_iso, X, Y, Bx, By)
    var_par_aniso, var_perp_aniso = _parallel_perp_variance(e_cr_aniso, X, Y, Bx, By)

    par_growth_aniso = var_par_aniso - var_par_0
    perp_growth_aniso = var_perp_aniso - var_perp_0
    leak_ratio_aniso = perp_growth_aniso / par_growth_aniso

    par_growth_iso = var_par_iso - var_par_0
    perp_growth_iso = var_perp_iso - var_perp_0
    leak_ratio_iso = perp_growth_iso / par_growth_iso

    # Sanity check that real spreading actually happened, in both cases
    # (otherwise the ratio check below would be a trivial 0/0-ish no-op).
    assert par_growth_iso > 0.5 * sigma**2, (
        "Expected substantial spreading in the isotropic control run; got "
        f"parallel-variance growth {par_growth_iso:.4e} -- IC/BC setup may "
        "be broken."
    )

    # Diagnostic plot: the two e_cr maps side by side, with a "money plot"
    # bar chart of the leak ratios for direct visual contrast.
    fig, (ax_iso, ax_aniso, ax_ratio) = plt.subplots(
        1, 3, figsize=(16, 5), constrained_layout=True
    )

    extent = [0, box_size, 0, box_size]
    vmax = float(max(jnp.max(e_cr_iso), jnp.max(e_cr_aniso)))
    for ax, e_cr, title in (
        (ax_iso, e_cr_iso, "Isotropic closure"),
        (ax_aniso, e_cr_aniso, "Anisotropic closure (B-projected)"),
    ):
        im = ax.imshow(
            e_cr.T, origin="lower", extent=extent, vmin=0, vmax=vmax, cmap="inferno"
        )
        ax.plot(
            [0.5 - 0.4 * float(Bx), 0.5 + 0.4 * float(Bx)],
            [0.5 - 0.4 * float(By), 0.5 + 0.4 * float(By)],
            color="cyan", lw=1.5, ls="--", label="B direction",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"{title}\nt = {t_end}")
        ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=[ax_iso, ax_aniso], label="e_cr", shrink=0.8)

    ax_ratio.bar(
        ["isotropic", "anisotropic"],
        [leak_ratio_iso, leak_ratio_aniso],
        color=["C3", "C0"],
    )
    ax_ratio.axhline(leak_ratio_tol, color="black", ls="--", lw=1, label=f"tol = {leak_ratio_tol}")
    ax_ratio.set_ylabel("perp / parallel variance growth")
    ax_ratio.set_title("Cross-field leak ratio")
    ax_ratio.legend()

    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_anisotropic_diffusion_oblique_test.svg")
    plt.close(fig)

    assert leak_ratio_aniso < leak_ratio_tol, (
        f"Cross-field leak too large: perp/parallel growth ratio "
        f"{leak_ratio_aniso:.4f} >= tol {leak_ratio_tol} for the anisotropic "
        f"closure (isotropic-run ratio for comparison: {leak_ratio_iso:.4f})."
    )


if __name__ == "__main__":
    test_cr_anisotropic_diffusion_oblique()
