"""
1D CR streaming pytest (Phase A ladder item 5).

Sets up a 1D CR pressure gradient with streaming enabled and checks that the
CR population streams down the gradient at the configured (reduced)
streaming speed, and that the streaming-heating rate deposited into the gas
thermal energy matches the analytic rate -- validates
``cr_grey_transport.regularized_streaming_sign``/``streaming_flux_target``
and ``cr_grey_sources.cr_streaming_heating_source`` together.

**Design implemented (these were stubs, not filled in before this session):**
``streaming_flux_target`` (``cr_grey_transport.py``) computes, per axis,
``F_cr,axis = -sign(dP_cr/dx_axis) * reduced_streaming_speed * e_cr``, using
:func:`regularized_streaming_sign` (``tanh``) for the direction. Applied
once per full step in ``_iteration_level_updates`` as a discrete
correction that **overwrites** ``F_cr`` -- the same "instantaneous
relaxation" pattern ``anisotropic_flux_projection`` already uses for ladder
item 3 (the physical picture: in the streaming-dominated limit, self-confined
CRs are always at their streaming equilibrium, not independently evolving).
``cr_streaming_heating_source`` (``cr_grey_sources.py``) computes the
complementary, genuinely non-conservative loss this streaming motion causes
by doing work against the pressure gradient: ``Gamma = reduced_streaming_speed
* |dP_cr/dx|`` (regularized-sign version), subtracted from ``e_cr`` and (times
``streaming_heating_efficiency``) added to the gas thermal energy. See
``DESIGN.md``/``PROGRESS.md`` for the full derivation and the (documented,
not yet separately tested) interaction with ``anisotropic_transport``.

**Setup deliberately avoids periodic BCs and a sign-changing gradient.** An
early calibration attempt used a periodic Gaussian-like (cosine) ``e_cr``
profile; ``regularized_streaming_sign``'s ``tanh`` transition width in
*gradient* space, mapped back to *position* space near a gradient
zero-crossing, was narrower than one grid cell for that profile's curvature
-- an unresolved, effectively-discontinuous sign flip right at the pressure
extrema, producing large, unphysical ``F_cr`` errors (not a code bug, a bad
IC choice). This test instead uses a **linear, open-boundary ramp**
(``e_cr(x) = e_cr0 + slope * (x - box_size / 2)``, constant-sign gradient
everywhere) so ``regularized_streaming_sign`` stays saturated (``~+-1``)
across the whole domain -- matches the plan's own "linear e_cr gradient"
suggestion. Measurements are taken over a central 60% window (``margin =
0.2 * num_cells`` trimmed from each edge) to stay clear of open-boundary
ghost-cell effects.

Calibration (ad hoc script, not committed): at ``t_end = 2e-3`` (well short
of any boundary/nonlinear effects reaching the measurement window), ``F_cr``
matches ``streaming_flux_target`` evaluated on the final state to relative
error ``~2e-4`` in the bulk, and the measured gas-pressure growth rate
matches the analytic ``dP/dt = (gamma - 1) * efficiency * Gamma`` (the
``(gamma - 1)`` factor converts the conserved-energy-row heating rate
``cr_streaming_heating_source`` actually adds into a primitive-pressure
rate) to relative error ``~7e-5``.

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md.
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
    BoundarySettings1D,
    OPEN_BOUNDARY,
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
from astronomix._modules._cosmic_rays_grey.cr_grey_transport import (
    streaming_flux_target,
)


def test_cr_streaming_1d(tol: float = 1e-2):
    """Recover the streaming speed and streaming-heating rate.

    Args:
        tol: The maximum allowed relative error on the streaming speed and
            heating rate. Calibrated against observed errors of ~2e-4
            (F_cr vs. target) and ~7e-5 (heating rate) -- comfortable margin.
    """
    gamma = 5.0 / 3.0
    gamma_cr = 4.0 / 3.0
    reduced_streaming_speed = 1.0
    e_cr0 = 1.0
    slope = 1.0
    box_size = 1.0
    num_cells = 256
    t_end = 2e-3

    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=box_size,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, streaming=True
        ),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers

    e_cr_profile = e_cr0 + slope * (x - 0.5 * box_size)

    primitive_state = jnp.zeros((registered_variables.num_vars, num_cells))
    primitive_state = primitive_state.at[registered_variables.density_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(
        e_cr_profile
    )

    config = finalize_config(config, primitive_state.shape)
    params = SimulationParams(
        t_end=t_end,
        gamma=gamma,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=gamma_cr, reduced_streaming_speed=reduced_streaming_speed
        ),
    )

    final_state = time_integration(primitive_state, config, params, registered_variables)

    assert not bool(jnp.any(jnp.isnan(final_state))), "CR streaming produced NaNs."

    f_cr_index = registered_variables.cosmic_ray_flux_index
    f_cr_final = final_state[f_cr_index]
    p_final = final_state[registered_variables.pressure_index]

    target_final = streaming_flux_target(
        final_state, config, params, registered_variables
    )[f_cr_index]

    margin = int(0.2 * num_cells)
    bulk = slice(margin, num_cells - margin)

    f_cr_rel_err = float(
        jnp.max(jnp.abs(f_cr_final[bulk] - target_final[bulk]))
        / jnp.max(jnp.abs(target_final[bulk]))
    )

    p_cr_grad = (gamma_cr - 1.0) * slope
    analytic_heating_rate = reduced_streaming_speed * p_cr_grad
    # cr_streaming_heating_source adds this rate onto the conserved energy
    # row; the (gamma - 1) factor converts that into the primitive-pressure
    # growth rate actually measured here (efficiency = 1, the default).
    dp_dt_analytic = (gamma - 1.0) * analytic_heating_rate

    dp_dt_measured = (p_final[bulk] - 1.0) / t_end
    heating_rel_err = float(
        jnp.mean(jnp.abs(dp_dt_measured - dp_dt_analytic)) / dp_dt_analytic
    )

    # Diagnostic plot: F_cr vs. the streaming target, and the pressure
    # growth vs. its analytic rate.
    fig, (ax_flux, ax_heating) = plt.subplots(1, 2, figsize=(12, 5))

    ax_flux.plot(x, target_final, label="streaming target", color="black", lw=1.5)
    ax_flux.plot(x, f_cr_final, label="F_cr (final)", color="C0", ls="--")
    ax_flux.axvspan(x[margin], x[num_cells - margin - 1], color="grey", alpha=0.15, label="measurement window")
    ax_flux.set_xlabel("x")
    ax_flux.set_ylabel("F_cr")
    ax_flux.set_title(f"F_cr vs. streaming target (bulk rel. err {f_cr_rel_err:.2e})")
    ax_flux.legend()

    ax_heating.plot(x[bulk], dp_dt_measured, label="measured dP/dt", color="C0")
    ax_heating.axhline(dp_dt_analytic, color="black", lw=1.5, ls="--", label="analytic dP/dt")
    ax_heating.set_xlabel("x")
    ax_heating.set_ylabel("dP/dt")
    ax_heating.set_title(f"Streaming-heating rate (rel. err {heating_rel_err:.2e})")
    ax_heating.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_streaming_1d_test.svg")
    plt.close(fig)

    assert f_cr_rel_err < tol, (
        f"CR streaming flux does not match the streaming target: rel. err "
        f"{f_cr_rel_err:.4f} >= tol {tol}."
    )
    assert heating_rel_err < tol, (
        f"CR streaming-heating rate does not match the analytic rate: rel. "
        f"err {heating_rel_err:.4f} >= tol {tol}."
    )


if __name__ == "__main__":
    test_cr_streaming_1d()
