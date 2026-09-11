"""
Mixed periodic-xy / open-z boundary condition smoke test (SILCC-ISM project, milestone M0a).

Validates the periodic-horizontal / open-vertical boundary combination this
project's stratified-box setups will all need, in complete isolation from any
new physics (no gravity, no cooling, no CR) -- the plan's own precedent for
isolating genuinely new combinations before layering anything else on top
(see this module's PROGRESS.md, e.g. items 3, 9, 10).

Setup: a uniform 3D Cartesian FV box (``HLLC``/``MINMOD``) with
``BoundarySettings(x=PERIODIC, y=PERIODIC, z=OPEN)``, carrying a smooth
tanh-tapered overdense blob (same taper pattern as
``pytests/cosmic_rays_grey/cr_sedov_taylor.py``/``cr_snr_clumpy_medium.py``)
on top of a uniform bulk velocity in x. With no y/z velocity and the blob's
full radial extent kept far from the open z-boundary, this reduces to a pure
rigid-body advection problem: the blob should cross the periodic x-face and
reappear on the other side, arriving at the expected wrapped position with
its mass and profile essentially intact (up to ordinary finite-volume
numerical diffusion from repeated reconstruction/limiting), while the open
z-boundary -- never touched by the blob -- stays inert.

Calibrated at N=128 (cube), ``v_x=0.5``, ``t_end=3.0`` (1.5 box lengths of
travel, so the blob wraps around x once and settles mid-domain at
``x=0.75`` box units, comfortably clear of the periodic seam at both the
initial and final measurement times -- avoids a wraparound-straddling
centroid ambiguity). Observed: mass conserved to a relative error of
``~2.15e-14`` (periodic x/y and the untouched open z-face together --
essentially machine precision), final x-centroid lands at ``0.749951`` vs.
the analytically expected ``0.75`` (absolute error ``~4.9e-5``, ordinary FV
numerical diffusion over one full periodic transit, not a BC bug), y/z
centroid drift is ``~1e-15`` (pure floating-point/stencil noise, as expected
for pure-x advection). See ``test_bc_smoke`` docstring for the tolerances.

**Resolution history (2026-09-11):** originally calibrated at N=32, giving
mass rel. err ``~1.19e-6``, x-centroid err ``~0.021``, y/z err ``~1e-6`` --
still passing, just less tight. Rerun at N=128 (this session's specific
request), which surfaced a real, now-fixed precision issue, not a resolution
artifact of the BC/advection physics itself: at N=128 (default float32), the
mass-conservation check *failed* (rel. err ``~5.6e-5`` against the old
``1e-5`` tolerance) -- diagnosed as float32 round-off accumulation (more
cells feeding `jnp.sum`, more CFL-limited time steps at finer `dx`), not a BC
bug; confirmed by rerunning with ``jax_enable_x64`` enabled (now on by
default in this file, same reasoning as
`stratified_hydrostatic_column.py`/`external_potential.py`), which dropped
the residual straight to the near-machine-precision numbers above.
"""

# general
from pathlib import Path

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# This is a near-machine-precision conservation check (periodic BCs should
# conserve mass up to round-off); at N=128 the increased cell/step count
# made float32 round-off accumulation large enough to matter (see module
# docstring) -- run in double precision, same reasoning as
# stratified_hydrostatic_column.py / external_potential.py.
jax.config.update("jax_enable_x64", True)

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import OPEN_BOUNDARY, PERIODIC_BOUNDARY
from astronomix import BoundarySettings, BoundarySettings1D
from astronomix import SimulationConfig, SimulationParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

# ---- physical setup ----
GAMMA = 5.0 / 3.0
N = 128
BOX = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1.0
BLOB_CONTRAST = 2.0
BLOB_RADIUS = 0.08
SMOOTH_CELLS = 2.0
V_X = 0.5
T_END = 3.0  # v_x * t_end = 1.5 box lengths -> one full periodic wrap

BLOB_X0 = 0.25 * BOX
BLOB_Y0 = 0.5 * BOX
BLOB_Z0 = 0.5 * BOX
# Expected final blob-center x, after wrapping (V_X * T_END) mod BOX into
# [0, BOX): (0.25 + 1.5) mod 1.0 = 0.75.
EXPECTED_X_FINAL = ((BLOB_X0 + V_X * T_END) % BOX)


def _run_bc_smoke():
    """Run the mixed-BC advection smoke test; return the final state and diagnostics."""
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=N,
        exact_end_time=True,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    centers = helper_data.geometric_centers
    x, y, z = centers[..., 0], centers[..., 1], centers[..., 2]
    dx = BOX / N
    cell_volume = dx**3

    dist = jnp.sqrt((x - BLOB_X0) ** 2 + (y - BLOB_Y0) ** 2 + (z - BLOB_Z0) ** 2)
    smooth_width = SMOOTH_CELLS * dx
    weight = 0.5 * (1.0 - jnp.tanh((dist - BLOB_RADIUS) / smooth_width))
    density = RHO_AMBIENT + RHO_AMBIENT * (BLOB_CONTRAST - 1.0) * weight

    velocity_x = jnp.ones_like(density) * V_X
    zero = jnp.zeros_like(density)
    gas_pressure = jnp.ones_like(density) * P_AMBIENT

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        velocity_y=zero,
        velocity_z=zero,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, gamma=GAMMA)

    final_state = time_integration(initial_state, config, params, registered_variables)

    rho_final = final_state[registered_variables.density_index]
    total_mass_initial = float(jnp.sum(density) * cell_volume)
    total_mass_final = float(jnp.sum(rho_final) * cell_volume)

    excess_final = jnp.clip(rho_final - RHO_AMBIENT, min=0.0)
    excess_mass = float(jnp.sum(excess_final))
    x_centroid_final = float(jnp.sum(excess_final * x) / jnp.sum(excess_final))
    y_centroid_final = float(jnp.sum(excess_final * y) / jnp.sum(excess_final))
    z_centroid_final = float(jnp.sum(excess_final * z) / jnp.sum(excess_final))

    return dict(
        final_state=final_state,
        rho_initial=density,
        rho_final=rho_final,
        total_mass_initial=total_mass_initial,
        total_mass_final=total_mass_final,
        excess_mass=excess_mass,
        x_centroid_final=x_centroid_final,
        y_centroid_final=y_centroid_final,
        z_centroid_final=z_centroid_final,
    )


def test_bc_smoke(
    mass_conservation_tol: float = 1e-9,
    centroid_x_tol: float = 1e-3,
    centroid_yz_tol: float = 1e-5,
):
    """Mixed periodic-xy/open-z BC smoke test: blob wraps around x, mass/shape intact.

    Args:
        mass_conservation_tol: Max relative error on total mass (initial vs.
            final) -- the periodic x/y faces should conserve mass exactly up
            to floating-point round-off, and the open z-face is never
            touched by the blob. Calibrated at N=128, float64: observed error
            ``~2.15e-14`` (>1e4x margin).
        centroid_x_tol: Max absolute error (box units) between the blob's
            final x-centroid and the analytically expected wrapped position
            (``EXPECTED_X_FINAL``). Calibrated at N=128, float64: observed
            error ``~4.9e-5`` (>20x margin).
        centroid_yz_tol: Max absolute error (box units) on the y/z centroid
            drift from its initial value (pure x-advection should leave y/z
            untouched up to numerical diffusion). Calibrated at N=128,
            float64: observed error ``~1e-15``.
    """
    run = _run_bc_smoke()

    assert not bool(jnp.any(jnp.isnan(run["final_state"]))), (
        "BC smoke test produced NaNs."
    )

    mass_rel_err = abs(run["total_mass_final"] - run["total_mass_initial"]) / run["total_mass_initial"]
    assert mass_rel_err < mass_conservation_tol, (
        f"Mass not conserved under periodic-xy/open-z BCs: initial "
        f"{run['total_mass_initial']:.8f}, final {run['total_mass_final']:.8f} "
        f"-- rel. err {mass_rel_err:.4e} >= tol {mass_conservation_tol}."
    )

    x_err = abs(run["x_centroid_final"] - EXPECTED_X_FINAL)
    assert x_err < centroid_x_tol, (
        f"Blob x-centroid ({run['x_centroid_final']:.4f}) does not match the "
        f"expected wrapped position ({EXPECTED_X_FINAL:.4f}) -- abs. err "
        f"{x_err:.4e} >= tol {centroid_x_tol}. Periodic wrap-around may be "
        f"broken."
    )

    y_err = abs(run["y_centroid_final"] - BLOB_Y0)
    z_err = abs(run["z_centroid_final"] - BLOB_Z0)
    assert max(y_err, z_err) < centroid_yz_tol, (
        f"Blob drifted in y/z during pure-x advection: y_err={y_err:.4e}, "
        f"z_err={z_err:.4e} >= tol {centroid_yz_tol}."
    )

    # Diagnostic plot: z-midplane density slices, initial vs. final.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))
    kz = N // 2
    extent = [0.0, BOX, 0.0, BOX]
    im0 = ax0.imshow(
        jnp.asarray(run["rho_initial"][:, :, kz]).T, origin="lower", extent=extent,
        vmin=RHO_AMBIENT, vmax=RHO_AMBIENT * BLOB_CONTRAST,
    )
    ax0.set_title("initial density (z-mid slice)")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    im1 = ax1.imshow(
        jnp.asarray(run["rho_final"][:, :, kz]).T, origin="lower", extent=extent,
        vmin=RHO_AMBIENT, vmax=RHO_AMBIENT * BLOB_CONTRAST,
    )
    ax1.axvline(EXPECTED_X_FINAL, color="cyan", ls="--", lw=1, label="expected x")
    ax1.set_title(f"final density (t={T_END}), wrapped once")
    ax1.legend()
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    for ax in (ax0, ax1):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "bc_smoke_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_bc_smoke()
