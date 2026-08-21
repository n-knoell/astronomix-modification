"""
Isotropic CR diffusion convergence pytest (Phase A ladder item 4).

Compares the two-moment isotropic-diffusion limit against the analytic
Gaussian Green's-function solution of the diffusion equation, at increasing
resolution, and checks the expected convergence order -- validates that the
reduced free-streaming speed (cr_grey_transport.grey_cr_fast_speed) is high
enough for the two-moment system to recover ordinary diffusion in this
regime (plan Sec. 2: "recovers anisotropic diffusion + streaming in the
appropriate limit").

**Design gap found and fixed to make this test possible at all:** as
implemented for ladder items 1-3, ``grey_cr_flux_terms`` is a *purely
hyperbolic, undamped* two-moment system -- a localized ``e_cr`` bump
propagates rigidly as a wave (verified explicitly in ladder item 3's
PROGRESS.md note), it never spreads/diffuses, no matter the resolution.
There is no diffusion limit to converge to without a relaxation/scattering
term. Added ``cr_grey_sources.cr_flux_relaxation_source`` (Jiang & Oh 2018):
``d(F_cr)/dt |_relax = -nu * F_cr``, ``nu = reduced_streaming_speed^2 /
diffusion_coefficient`` (``diffusion_coefficient`` a new
``CosmicRayGreyParams`` field, ``kappa``). At a quasi-steady balance against
the pressure-driving flux term, this relaxes ``F_cr ~= -kappa * grad(P_cr)``,
i.e. Fick's law with diffusion coefficient ``D = kappa * (gamma_cr - 1)`` for
``e_cr``. Gated behind a new ``CosmicRayGreyConfig.diffusive_relaxation``
flag (default ``False``) so ladder items 1-3's already-verified undamped-wave
behavior is unchanged unless a config explicitly opts in. See that
function's docstring and ``DESIGN.md``/``PROGRESS.md`` for the full
derivation, and the new ``_cfl_time_step`` branch
(``_finite_volume/_timestep_estimation/_timestep_estimator.py``) for the
matching parabolic-like CFL constraint this reintroduces (explicit forward
Euler on the relaxation rate ``nu``, same category as the existing
``config.diffusion`` viscous ``dt_visc`` constraint).

Setup: a narrow ``e_cr`` Gaussian bump on a uniform, static gas (``F_cr = 0``
initial), periodic BCs, run to a fixed ``t_end`` at increasing resolution;
compare each run's final profile to the analytic 1D diffusion Green's
function ``e_cr(x, t) = amp * sigma0 / sigma(t) * exp(-(x - x0)^2 /
(2 sigma(t)^2))``, ``sigma(t)^2 = sigma0^2 + 2 D t``. Parameters
(``reduced_streaming_speed = 8``, ``diffusion_coefficient = 0.06``) were
chosen, via an ad hoc calibration script (not committed), to put the
relaxation rate ``nu ~= 1067`` comfortably above the diffusion rate of the
smallest resolved scale (``1 / (D / sigma0^2) ~= 50``) -- deep enough in the
quasi-steady/diffusive regime that the telegrapher-equation-vs-diffusion
model bias is negligible next to the resolutions tested here. Calibration
run (N = 128/256/512/1024): L2 relative error 0.0089 -> 0.0042 -> 0.0020 ->
0.0010, monotonically decreasing, total ``e_cr`` conserved to 6 significant
figures at every resolution, pairwise convergence order ~1.0-1.1 (capped
near first order by the source terms' explicit-Euler operator splitting,
not by spatial truncation -- consistent, not a red flag).

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import math
from pathlib import Path

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
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


def _run_diffusion(
    num_cells: int,
    gamma_cr: float,
    reduced_streaming_speed: float,
    diffusion_coefficient: float,
    amp: float,
    sigma0: float,
    box_size: float,
    x0: float,
    t_end: float,
):
    """Run one resolution of the isotropic-diffusion setup and return the
    L2 relative error against the analytic Gaussian Green's function."""
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=box_size,
        boundary_settings=BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_relaxation=True
        ),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers

    pulse = amp * jnp.exp(-0.5 * ((x - x0) / sigma0) ** 2)

    primitive_state = jnp.zeros((registered_variables.num_vars, num_cells))
    primitive_state = primitive_state.at[registered_variables.density_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(
        pulse
    )
    # F_cr = 0 initial -- the relaxation term (not the advective closure
    # flux) is what has to build up the diffusive flux from rest.

    config = finalize_config(config, primitive_state.shape)
    params = SimulationParams(
        t_end=t_end,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=gamma_cr,
            reduced_streaming_speed=reduced_streaming_speed,
            diffusion_coefficient=diffusion_coefficient,
        ),
    )

    final_state = time_integration(primitive_state, config, params, registered_variables)
    e_cr_final = final_state[registered_variables.cosmic_ray_e_index]

    assert not bool(jnp.any(jnp.isnan(final_state))), (
        f"CR isotropic diffusion produced NaNs at N={num_cells}."
    )

    diffusion_coeff_e_cr = diffusion_coefficient * (gamma_cr - 1.0)
    sigma_final = jnp.sqrt(sigma0**2 + 2.0 * diffusion_coeff_e_cr * t_end)
    norm = amp * sigma0 / sigma_final
    analytic = norm * jnp.exp(-0.5 * ((x - x0) / sigma_final) ** 2)

    l2_err = float(jnp.sqrt(jnp.mean((e_cr_final - analytic) ** 2)) / amp)

    return l2_err, x, e_cr_final, analytic


def test_cr_isotropic_diffusion_convergence(min_order: float = 0.7):
    """Isotropic diffusion vs. the analytic Gaussian Green's function.

    Args:
        min_order: The minimum acceptable convergence order. Calibrated
            against pairwise orders of ~1.0-1.1 (see module docstring); 0.7
            leaves comfortable margin while still catching a broken/
            non-converging relaxation term.
    """
    gamma_cr = 4.0 / 3.0
    reduced_streaming_speed = 8.0
    diffusion_coefficient = 0.06
    amp = 1e-3
    sigma0 = 0.02
    box_size = 1.0
    x0 = 0.5 * box_size
    t_end = 0.24

    resolutions = (128, 256, 512, 1024)
    errors = []
    profiles = {}
    for num_cells in resolutions:
        l2_err, x, e_cr_final, analytic = _run_diffusion(
            num_cells,
            gamma_cr,
            reduced_streaming_speed,
            diffusion_coefficient,
            amp,
            sigma0,
            box_size,
            x0,
            t_end,
        )
        errors.append(l2_err)
        profiles[num_cells] = (x, e_cr_final, analytic)

    # Least-squares fit of log(error) vs log(N) across all resolutions --
    # more robust than a single pairwise ratio.
    log_n = [math.log(n) for n in resolutions]
    log_err = [math.log(e) for e in errors]
    n_pts = len(resolutions)
    mean_log_n = sum(log_n) / n_pts
    mean_log_err = sum(log_err) / n_pts
    cov = sum(
        (ln - mean_log_n) * (le - mean_log_err) for ln, le in zip(log_n, log_err)
    )
    var = sum((ln - mean_log_n) ** 2 for ln in log_n)
    convergence_order = -cov / var

    # Diagnostic plot: finest-resolution profile vs. the analytic Green's
    # function, plus the log-log convergence trend.
    fig, (ax_profile, ax_convergence) = plt.subplots(1, 2, figsize=(12, 5))

    x_fine, e_cr_fine, analytic_fine = profiles[resolutions[-1]]
    ax_profile.plot(x_fine, analytic_fine, label="analytic Green's function", color="black", lw=1.5)
    ax_profile.plot(
        x_fine, e_cr_fine, label=f"numerical (N={resolutions[-1]})", color="C0", ls="--"
    )
    ax_profile.set_xlabel("x")
    ax_profile.set_ylabel("e_cr")
    ax_profile.set_title(f"Diffused profile at t = {t_end}")
    ax_profile.legend()

    ax_convergence.loglog(resolutions, errors, "o-", color="C0", label="L2 rel. error")
    ax_convergence.set_xlabel("N (resolution)")
    ax_convergence.set_ylabel("L2 relative error")
    ax_convergence.set_title(f"Convergence order = {convergence_order:.2f}")
    ax_convergence.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_isotropic_diffusion_convergence_test.svg")
    plt.close(fig)

    assert convergence_order >= min_order, (
        f"CR isotropic diffusion did not converge as expected: order "
        f"{convergence_order:.3f} < min {min_order}. Errors: {errors}."
    )


if __name__ == "__main__":
    test_cr_isotropic_diffusion_convergence()
