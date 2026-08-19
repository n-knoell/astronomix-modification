"""
Pure CR advection pytest (Phase A ladder item 1).

Sets up a smooth ``e_cr`` pulse as a *right-moving eigenmode* of the
homogeneous (source-free) two-moment transport system that
``cr_grey_transport.grey_cr_flux_terms``/``grey_cr_fast_speed`` implement
(Jiang & Oh 2018 isotropic closure, ``P_cr = (gamma_cr - 1) * e_cr``):

    d(e_cr)/dt + d(F_cr)/dx        = 0
    d(F_cr)/dt + d(v_red^2 P_cr)/dx = 0

A pulse ``e_cr(x, 0) = f(x)``, ``F_cr(x, 0) = lambda * f(x)`` with
``lambda = v_red * sqrt(gamma_cr - 1)`` translates rigidly at speed
``lambda`` without dispersing (this is the CR analogue of the classic
radiative-transfer "beam test" in the free-streaming limit of an M1-like
closure). Running for exactly one box-crossing period
(``t_end = box_size / lambda``) under periodic BCs means the *exact* answer
is simply the initial profile again -- no shift arithmetic needed in the
reference, and the run exercises periodic-BC wraparound along the way.

The background is a pure CR-free gas (``e_cr`` background = 0, not a small
bump on a large uniform CR population) specifically so the real feedback
coupling this module also implements (``-grad(P_cr)`` momentum,
``-P_cr * div(v)`` adiabatic work -- see ``cr_grey_sources.py``) stays
second-order small in the pulse amplitude and does not contaminate this
transport-only check; ladder item 6 (``cr_shock_tube.py``) is where that
coupling itself gets tested. Verified during implementation (not asserted
here, since it needs a second, decoupled setup) via a center-of-mass
diagnostic: with a genuinely CR-free background, the numerical propagation
speed matches ``lambda`` to 5 decimal places and total ``e_cr`` is conserved
to 6, across resolutions and reduced-streaming-speed choices spanning 1-20 --
see this module's PROGRESS.md.

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md for the state-variable
and equation definitions this test exercises.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax.numpy as jnp

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


def test_cr_advection(l2_tol: float = 0.05, peak_retention_tol: float = 0.15):
    """Advect a right-moving ``e_cr`` eigenmode one full period and check
    shape/amplitude preservation against the (exact, periodic) initial
    profile.

    Args:
        l2_tol: Maximum allowed L2-norm relative error (normalized by the
            pulse amplitude) between the final and initial ``e_cr``
            profiles. Calibrated against a 512-cell run (L2 error ~0.011,
            HLL + MINMOD numerical diffusion of a resolved-but-not-huge
            Gaussian); this leaves comfortable margin.
        peak_retention_tol: Maximum allowed fractional loss of the pulse's
            peak amplitude (1 - final_peak / initial_peak). Calibrated
            against the same run (~0.06 loss observed); this leaves
            comfortable margin.
    """
    num_cells = 512
    box_size = 1.0
    gamma_cr = 4.0 / 3.0
    reduced_streaming_speed = 1.0
    amp = 1e-3
    sigma = 0.05
    x0 = 0.5 * box_size

    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=box_size,
        boundary_settings=BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers

    # The true characteristic speed of the homogeneous system (see module
    # docstring); grey_cr_fast_speed itself returns the larger, un-scaled
    # reduced_streaming_speed as a conservative CFL/Riemann bound, not this
    # exact eigenvalue -- see that function's docstring.
    wave_speed = reduced_streaming_speed * jnp.sqrt(gamma_cr - 1.0)

    pulse = amp * jnp.exp(-0.5 * ((x - x0) / sigma) ** 2)

    primitive_state = jnp.zeros((registered_variables.num_vars, num_cells))
    primitive_state = primitive_state.at[registered_variables.density_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(
        pulse
    )
    primitive_state = primitive_state.at[
        registered_variables.cosmic_ray_flux_index
    ].set(wave_speed * pulse)

    config = finalize_config(config, primitive_state.shape)

    # One full box-crossing period: periodic BCs mean the exact solution at
    # t_end is the initial profile again.
    t_end = box_size / wave_speed
    params = SimulationParams(
        t_end=t_end,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=gamma_cr, reduced_streaming_speed=reduced_streaming_speed
        ),
    )

    final_state = time_integration(primitive_state, config, params, registered_variables)

    e_cr_initial = primitive_state[registered_variables.cosmic_ray_e_index]
    e_cr_final = final_state[registered_variables.cosmic_ray_e_index]

    assert not bool(jnp.any(jnp.isnan(final_state))), "CR advection produced NaNs."

    l2_err = float(jnp.sqrt(jnp.mean((e_cr_final - e_cr_initial) ** 2)) / amp)
    peak_retention_loss = 1.0 - float(jnp.max(e_cr_final) / jnp.max(e_cr_initial))

    assert l2_err < l2_tol, (
        f"CR pulse shape not preserved: L2 relative error {l2_err:.4f} "
        f">= tol {l2_tol}."
    )
    assert peak_retention_loss < peak_retention_tol, (
        f"CR pulse amplitude not preserved: peak loss {peak_retention_loss:.4f} "
        f">= tol {peak_retention_tol}."
    )


if __name__ == "__main__":
    test_cr_advection()
