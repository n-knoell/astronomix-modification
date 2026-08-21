"""
CR transport differentiability pytest (plan Sec. 4, item 16 -- "continuous,
from Phase A").

Finite-difference vs. autodiff gradient check on a small CR transport
problem: perturb ``reduced_streaming_speed``, run a short simulation to a
scalar cost, and compare the AD gradient (``jax.grad``) against a centered
finite-difference gradient. Catches a floor/limiter/``jnp.where`` silently
killing the adjoint -- exactly the failure mode the plan's differentiability
plan (``tanh``-regularized streaming sign, no ``min``/``max``/``sign``) is
designed to avoid. See ``pytests/differentiability/sensitivity.py`` for the
general style this repo uses for gradient-correctness pytests (there: AD vs.
a closed-form analytic gradient for the non-CR Euler equations; here: AD vs.
finite differences, since a closed-form CR gradient isn't available) --
that test also establishes that ``time_integration``'s adaptive-dt loop to a
fixed ``t_end`` is end-to-end differentiable, which this test relies on too.

Setup mirrors ``cr_advection.py``'s right-moving-eigenmode pulse (a smooth
``e_cr``/``F_cr`` profile, CR-free background so the transport path
dominates, not run a full box-crossing period -- just far enough that
``reduced_streaming_speed`` visibly shapes the final state). Only
``grey_cr_flux_terms``, ``grey_cr_fast_speed``, ``cr_pressure_gradient_source``
and ``cr_adiabatic_work_source`` are exercised (the default config keeps
``anisotropic_transport``/``streaming``/``diffusive_relaxation`` off, so the
still-unimplemented ``cr_streaming_heating_source`` stays out of the path).

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md's differentiability plan.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
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

# Matches sensitivity.py's rationale: a tight AD-vs-finite-difference
# comparison needs 64-bit precision, or round-off swamps the comparison.
jax.config.update("jax_enable_x64", True)


def _cost(reduced_streaming_speed, config, gamma_cr, primitive_state, t_end, registered_variables):
    """Run the CR-grey pulse to ``t_end`` and return a scalar cost of the
    final ``e_cr`` field, as a function of ``reduced_streaming_speed``."""
    params = SimulationParams(
        t_end=t_end,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=gamma_cr, reduced_streaming_speed=reduced_streaming_speed
        ),
    )
    final_state = time_integration(primitive_state, config, params, registered_variables)
    e_cr_final = final_state[registered_variables.cosmic_ray_e_index]
    return jnp.sum(e_cr_final**2)


def test_cr_gradient_check(tol: float = 1e-2):
    """Finite-difference vs. autodiff gradient check on a small CR problem.

    Args:
        tol: The maximum allowed relative error between the AD and
            finite-difference gradients.
    """
    num_cells = 64
    box_size = 1.0
    gamma_cr = 4.0 / 3.0
    reduced_streaming_speed_0 = 1.0
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
        # Reverse-mode AD through the adaptive-dt while loop needs the
        # checkpointed backend -- see time_integration.py's
        # differentiation_mode dispatch.
        differentiation_mode=BACKWARDS,
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers

    wave_speed_0 = reduced_streaming_speed_0 * jnp.sqrt(gamma_cr - 1.0)
    pulse = amp * jnp.exp(-0.5 * ((x - x0) / sigma) ** 2)

    primitive_state = jnp.zeros((registered_variables.num_vars, num_cells))
    primitive_state = primitive_state.at[registered_variables.density_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(pulse)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_flux_index].set(
        wave_speed_0 * pulse
    )
    config = finalize_config(config, primitive_state.shape)

    # A fraction of one box-crossing period at the fiducial speed -- short
    # enough to be cheap, long enough that reduced_streaming_speed visibly
    # shapes the final state via the pulse's propagation distance.
    t_end = 0.2 * box_size / wave_speed_0

    cost_fn = lambda v: _cost(v, config, gamma_cr, primitive_state, t_end, registered_variables)

    ad_grad = float(jax.grad(cost_fn)(reduced_streaming_speed_0))

    h = 1e-4 * reduced_streaming_speed_0
    fd_grad = float(
        (cost_fn(reduced_streaming_speed_0 + h) - cost_fn(reduced_streaming_speed_0 - h))
        / (2 * h)
    )

    rel_err = abs(ad_grad - fd_grad) / abs(fd_grad)
    print(f"AD grad = {ad_grad:.6e}, FD grad = {fd_grad:.6e}, rel. err = {rel_err:.3e}")

    assert not (ad_grad != ad_grad), "AD gradient is NaN."
    assert rel_err < tol, (
        f"AD vs FD gradient mismatch: rel. err {rel_err:.3e} >= tol {tol} "
        f"(AD={ad_grad:.6e}, FD={fd_grad:.6e})."
    )


if __name__ == "__main__":
    test_cr_gradient_check()
