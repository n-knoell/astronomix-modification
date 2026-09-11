"""
Finite-volume time-step estimation.

Provides the maximum wave-speed estimate, the CFL time step (split and unsplit,
including the viscous constraint) and the experimental source-term-aware time
step that accounts for the stellar-wind injection.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    DYNAMIC_VISCOSITY,
    KINEMATIC_VISCOSITY,
    STATE_TYPE,
    UNSPLIT,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._stellar_wind.stellar_wind import _wind_injection
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._fluid_equations._equations import speed_of_sound
from astronomix._modules._cosmic_rays_grey.cr_grey_transport import grey_cr_fast_speed
from astronomix._modules._cooling._cooling import dtemperature_dt, get_temperature_from_pressure


# NOTE: these wave speeds are computed without the reconstruction. For the
# purely spatial reconstruction we do not need to know the time step a priori,
# so a coarser estimate than the one used in the Riemann solver is acceptable
# here.
# TODO: merge the duplicate wave-speed code shared with hll.py.
# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["registered_variables", "config", "flux_direction_index"]
)
def get_wave_speeds(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
    config: SimulationConfig,
    params: SimulationParams,
    flux_direction_index: int,
) -> Union[float, Float[Array, ""]]:
    """
    Return the maximum signal (wave) speed across all interfaces along an axis.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index.
        registered_variables: The registered variables.
        config: The simulation configuration.
        params: The simulation parameters, forwarded to
            :func:`grey_cr_fast_speed` when CR-grey is active.
        flux_direction_index: The state index of the velocity normal to the
            interface (the flux direction).

    Returns:
        The maximum wave speed over all interfaces along the given axis.
    """

    rho_L = primitives_left[registered_variables.density_index]
    u_L = primitives_left[flux_direction_index]

    rho_R = primitives_right[registered_variables.density_index]
    u_R = primitives_right[flux_direction_index]

    p_L = primitives_left[registered_variables.pressure_index]
    p_R = primitives_right[registered_variables.pressure_index]

    # calculate the sound speeds
    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)

    # Grey two-moment cosmic rays: combine with the CR-grey fast speed as
    # max(., .), not folded in -- see hll.py's identical pattern.
    if registered_variables.cosmic_ray_e_active:
        c_L = jnp.maximum(c_L, grey_cr_fast_speed(primitives_left, params, registered_variables))
        c_R = jnp.maximum(c_R, grey_cr_fast_speed(primitives_right, params, registered_variables))

    # A simple symmetric estimate of the maximum signal speed on either side of
    # the interface; the |u| + c form is sufficient for the time-step bound.
    wave_speeds_right_plus = jnp.abs(u_L) + c_L
    wave_speeds_left_minus = jnp.abs(u_R) + c_R

    max_wave_speed = jnp.maximum(
        jnp.max(jnp.abs(wave_speeds_right_plus)),
        jnp.max(jnp.abs(wave_speeds_left_minus)),
    )

    return max_wave_speed


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> Float[Array, ""]:
    """Calculate the time step based on the CFL condition.

    Args:
        primitive_state: The primitive state array.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The time step.

    """

    C_CFL = params.C_cfl
    grid_spacing = config.grid_spacing
    dt_max = params.dt_max
    gamma = params.gamma

    if config.split == UNSPLIT:
        rho = primitive_state[registered_variables.density_index]
        p = primitive_state[registered_variables.pressure_index]
        c = speed_of_sound(rho, p, gamma)

        # Grey two-moment cosmic rays: this branch (the default -- UNSPLIT
        # is config.split's default) previously had zero CR-grey awareness,
        # unlike the SPLIT branch's get_wave_speeds calls below. Found while
        # debugging Phase A: without this, dt is set from the gas wave speed
        # alone, so any reduced_streaming_speed exceeding the gas sound speed
        # silently violates the CR subsystem's CFL condition and blows up
        # (NaN) rather than erroring -- see
        # astronomix._modules._cosmic_rays_grey.PROGRESS.md.
        if registered_variables.cosmic_ray_e_active:
            c = jnp.maximum(c, grey_cr_fast_speed(primitive_state, params, registered_variables))

        alpha_lax = jnp.zeros((config.dimensionality,))
        for axis in range(1, config.dimensionality + 1):
            u = primitive_state[axis]
            alpha_lax_i = jnp.max(jnp.abs(u) + c)
            alpha_lax = alpha_lax.at[axis - 1].set(alpha_lax_i)

        dt = C_CFL * 1 / jnp.sum(alpha_lax / grid_spacing)

    else:
        if config.dimensionality == 3:
            # wave speeds in x direction
            primitive_state_left = primitive_state[:, :-1, :, :]
            primitive_state_right = primitive_state[:, 1:, :, :]
            max_wave_speed_x = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                params,
                registered_variables.velocity_index.x,
            )

            # wave speeds in y direction
            primitive_state_left = primitive_state[:, :, :-1, :]
            primitive_state_right = primitive_state[:, :, 1:, :]
            max_wave_speed_y = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                params,
                registered_variables.velocity_index.y,
            )

            # wave speeds in z direction
            primitive_state_left = primitive_state[:, :, :, :-1]
            primitive_state_right = primitive_state[:, :, :, 1:]
            max_wave_speed_z = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                params,
                registered_variables.velocity_index.z,
            )

            # get the maximum wave speed
            max_wave_speed = jnp.maximum(
                jnp.maximum(max_wave_speed_x, max_wave_speed_y), max_wave_speed_z
            )
        elif config.dimensionality == 2:
            # wave speeds in x direction
            primitive_state_left = primitive_state[:, :-1, :]
            primitive_state_right = primitive_state[:, 1:, :]
            max_wave_speed_x = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                params,
                registered_variables.velocity_index.x,
            )

            # wave speeds in y direction
            primitive_state_left = primitive_state[:, :, :-1]
            primitive_state_right = primitive_state[:, :, 1:]
            max_wave_speed_y = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                params,
                registered_variables.velocity_index.y,
            )

            # get the maximum wave speed
            max_wave_speed = jnp.maximum(max_wave_speed_x, max_wave_speed_y)
        else:
            # wave speeds in x direction
            primitive_state_left = primitive_state[:, :-1]
            primitive_state_right = primitive_state[:, 1:]
            max_wave_speed = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                params,
                registered_variables.velocity_index,
            )

        # calculate the time step
        dt = C_CFL * grid_spacing / max_wave_speed

        if config.use_max_adaptive_timestep:
            dt = jnp.minimum(dt, dt_max)

    # viscous time step constraint
    if config.diffusion:
        
        if config.positivity_config.clamp_in_estimates:
            rho_min = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min = jnp.min(primitive_state[registered_variables.density_index])
        
        if config.viscosity_type == DYNAMIC_VISCOSITY:
            nu_max = params.viscosity / rho_min
        elif config.viscosity_type == KINEMATIC_VISCOSITY:
            nu_max = params.viscosity

        dt_visc = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * nu_max)
        dt = jnp.minimum(dt, dt_visc)

    # CR-grey F_cr relaxation constraint: cr_flux_relaxation_source damps
    # F_cr explicitly at rate reduced_streaming_speed^2 / diffusion_coefficient
    # (see that function's docstring) -- a genuine parabolic-like stiffness,
    # same category as the viscous dt_visc constraint above. Forward-Euler
    # stability of dF/dt = -nu*F requires dt < 2/nu; mirror dt_visc's
    # conservative C_CFL-scaled convention.
    if (
        registered_variables.cosmic_ray_e_active
        and config.cosmic_ray_grey_config.diffusive_relaxation
    ):
        relaxation_rate = (
            params.cosmic_ray_grey_params.reduced_streaming_speed**2
            / params.cosmic_ray_grey_params.diffusion_coefficient
        )
        dt_relax = C_CFL / relaxation_rate
        dt = jnp.minimum(dt, dt_relax)

    # Cooling-time constraint: update_pressure_by_cooling is applied at
    # whatever dt this function returns, but nothing above is aware of
    # cooling at all. Near a strongly heating/cooling state, the local
    # relaxation time |T / dT_dt| can be far shorter than the hydro dt above
    # (confirmed directly for the Koyama & Inutsuka net-cooling curve --
    # see astronomix/_modules/_cosmic_rays_grey/PROGRESS.md's M1 entry: at
    # n_H=10 cm^-3, T=1e4 K, dt~14.5x the local cooling time made
    # update_temperature_implicit's fixed-point iteration fail to converge
    # and silently return a wrong temperature, since jax.lax.while_loop just
    # stops at max_iter regardless of whether tol was reached). Bound dt by
    # the fastest (smallest) local relaxation time anywhere on the grid,
    # mirroring dt_visc/dt_relax's C_CFL-scaled convention.
    if config.cooling_config.cooling:
        cooling_params = params.cooling_params
        density = primitive_state[registered_variables.density_index]
        pressure = primitive_state[registered_variables.pressure_index]
        temperature = get_temperature_from_pressure(
            density, pressure, cooling_params.hydrogen_mass_fraction, cooling_params.metal_mass_fraction
        )
        dT_dt = dtemperature_dt(
            density,
            temperature,
            cooling_params.hydrogen_mass_fraction,
            cooling_params.metal_mass_fraction,
            gamma,
            config.cooling_config.cooling_curve_config,
            cooling_params.cooling_curve_params,
        )
        # |dT/dt| = 0 (already at equilibrium, or cooling inactive there)
        # imposes no constraint -- floor the rate rather than the time to
        # keep this well-defined without an arbitrary large-time cap.
        # cooling_params.floor_temperature is already in the same code-unit
        # \tilde{T} convention as `temperature` here.
        relaxation_rate_cool = jnp.max(
            jnp.abs(dT_dt) / jnp.maximum(temperature, cooling_params.floor_temperature)
        )
        dt_cool = C_CFL / jnp.maximum(relaxation_rate_cool, 1e-30)
        dt = jnp.minimum(dt, dt_cool)

    return dt


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _source_term_aware_time_step(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
) -> Float[Array, ""]:
    """
    Calculate the time step based on the CFL condition and the source terms. What timestep
    would be chosen if the source terms were added under the current CFL time step?

    Args:
        state: The state array.
        config: The configuration.
        params: The parameters.
        helper_data: The helper data.

    Returns:
        The time step.
    """

    # == experimental: correct the CFL time step based on the physical sources ==

    # calculate the time step based on the CFL condition
    dt = _cfl_time_step(
        primitive_state,
        config,
        params,
        registered_variables,
    )

    # independent of config.use_max_adaptive_timestep
    dt = jnp.minimum(dt, params.dt_max)

    # ONLY ADD THE STELLAR WIND SOURCE TERM
    # DO NOT RUN ALL PHYSICS MODULES HERE

    # TODO: HOW TO HANDLE FURTHER SOURCE TERMS?

    hypothetical_new_state = _wind_injection(
        primitive_state, dt, config, params, helper_data, registered_variables
    )

    dt = _cfl_time_step(
        hypothetical_new_state,
        config,
        params,
        registered_variables,
    )

    # ===========================================================================

    return dt
