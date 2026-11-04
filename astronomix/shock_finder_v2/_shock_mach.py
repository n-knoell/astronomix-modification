from functools import partial

import jax
import jax.numpy as jnp

from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    BOOL_FIELD_TYPE,
    FIELD_TYPE,
    STATE_TYPE,
    SimulationConfig,
)
from astronomix._physics_modules._shock_finder._shock_zones import (
    get_adaptive_post_pre_shock_values,
    get_extrapolated_post_pre_shock_values,
    get_plateau_aware_post_pre_shock_values,
    get_profile_aware_post_pre_shock_values,
    get_ramp_exit_aware_post_pre_shock_values,
)


"""
Calculate Mach number for all cells, 
but only keep it at the shock surface (where shock_surface is True) via filter
* To do this to shoot follow shock direction through all cells -> get post/pre values at all cells
* Then apply some equation to get Mach number for all cells from p_post/p_pre
* then filter to keep only Mach for surface cells
"""
@partial(
    jax.jit,
    static_argnames=["registered_variables", "config", "profile_aware"],
)
def _calculate_mach_at_surface(
    primitive_state: STATE_TYPE,
    shock_surface: BOOL_FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
    surface_offsets: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    profile_aware: bool = False,
) -> FIELD_TYPE:
    gamma_gas = 5 / 3

    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    temperature = pressure / density

    sampler = (
        get_profile_aware_post_pre_shock_values
        if profile_aware
        else get_adaptive_post_pre_shock_values
    )
    p_post, p_pre, _, _, valid_samples, _, _ = sampler(
        shock_direction,
        shock_zones,
        pressure,
        temperature,
        max_steps=8,
        center_offsets=(
            shock_direction * surface_offsets[jnp.newaxis, ...]
        ),
    )

    # calculate Mach number for all cells
    # p₂/p₁ = p_post/p_pre, but clamp to 1 to avoid numerical issues with very weak shocks
    p_ratio = jnp.maximum(p_post / jnp.maximum(p_pre, 1e-30), 1.0)
    # as p₂/p₁ = (2γM² − (γ−1)) / (γ+1) so M = √[ (p₂/p₁ · (γ+1) + (γ−1)) / (2γ) ]
    M = jnp.sqrt(
        (p_ratio * (gamma_gas + 1) + (gamma_gas - 1))
        / (2 * gamma_gas)
    )

    # write Mach only at surface cells, zero elsewhere
    mach_array = jnp.where(shock_surface & valid_samples, M, 0.0)

    return mach_array


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
)
def _calculate_plateau_mach_at_surface(
    primitive_state: STATE_TYPE,
    shock_surface: BOOL_FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
    surface_offsets: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    """Calculate surface Mach numbers only from locally flat states.

    This diagnostic uses the same detected shock surface and pressure-jump
    relation as the established estimator.  Only the upstream/downstream
    selection changes: both pressure and density must reach a local plateau
    before their pressure ratio is accepted.
    """
    gamma_gas = 5.0 / 3.0
    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    (
        pressure_post,
        pressure_pre,
        _,
        _,
        valid_samples,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = get_plateau_aware_post_pre_shock_values(
        shock_direction,
        shock_zones,
        pressure,
        density,
        max_steps=8,
        center_offsets=(
            shock_direction * surface_offsets[jnp.newaxis, ...]
        ),
    )
    pressure_ratio = jnp.maximum(
        pressure_post / jnp.maximum(pressure_pre, 1.0e-30),
        1.0,
    )
    mach = jnp.sqrt(
        (
            pressure_ratio * (gamma_gas + 1.0)
            + (gamma_gas - 1.0)
        )
        / (2.0 * gamma_gas)
    )
    return jnp.where(shock_surface & valid_samples, mach, 0.0)


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
)
def _calculate_ramp_exit_mach_at_surface(
    primitive_state: STATE_TYPE,
    shock_surface: BOOL_FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
    surface_offsets: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    """Calculate Mach numbers from smoothly varying exterior states."""
    gamma_gas = 5.0 / 3.0
    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    (
        pressure_post,
        pressure_pre,
        _,
        _,
        valid_samples,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = get_ramp_exit_aware_post_pre_shock_values(
        shock_direction,
        shock_zones,
        pressure,
        density,
        max_steps=8,
        center_offsets=(
            shock_direction * surface_offsets[jnp.newaxis, ...]
        ),
    )
    pressure_ratio = jnp.maximum(
        pressure_post / jnp.maximum(pressure_pre, 1.0e-30),
        1.0,
    )
    mach = jnp.sqrt(
        (
            pressure_ratio * (gamma_gas + 1.0)
            + (gamma_gas - 1.0)
        )
        / (2.0 * gamma_gas)
    )
    return jnp.where(shock_surface & valid_samples, mach, 0.0)


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
)
def _calculate_extrapolated_mach_at_surface(
    primitive_state: STATE_TYPE,
    shock_surface: BOOL_FIELD_TYPE,
    shock_zones: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,
    surface_offsets: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    """Calculate Mach numbers from shock-front-extrapolated states."""
    gamma_gas = 5.0 / 3.0
    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    (
        pressure_post,
        pressure_pre,
        _,
        _,
        valid_samples,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = get_extrapolated_post_pre_shock_values(
        shock_direction,
        shock_zones,
        pressure,
        density,
        max_steps=8,
        center_offsets=(
            shock_direction * surface_offsets[jnp.newaxis, ...]
        ),
    )
    pressure_ratio = jnp.maximum(
        pressure_post / jnp.maximum(pressure_pre, 1.0e-30),
        1.0,
    )
    mach = jnp.sqrt(
        (
            pressure_ratio * (gamma_gas + 1.0)
            + (gamma_gas - 1.0)
        )
        / (2.0 * gamma_gas)
    )
    return jnp.where(shock_surface & valid_samples, mach, 0.0)
