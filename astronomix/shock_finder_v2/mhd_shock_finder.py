"""Find and classify fast MHD shocks without hydrodynamic jump formulas."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables

from astronomix._physics_modules._shock_finder._gradients import (
    _calculate_shock_direction,
)
from astronomix._physics_modules._shock_finder._mhd_shock import (
    calculate_fast_shock_diagnostics,
)
from astronomix._physics_modules._shock_finder._shock_surface import (
    calculate_shock_surface_offsets,
    identify_shock_surface,
)
from astronomix._physics_modules._shock_finder._shock_zones import (
    _shock_zone_criterion_aligned_gradients,
    _shock_zone_criterion_converging_flow,
    get_adaptive_post_pre_shock_values,
    get_post_pre_shock_values,
)


class MHDShockFinderResult(NamedTuple):
    """Surface geometry and orientation-aware fast-shock diagnostics."""

    shock_surface_cells: jnp.ndarray
    shock_surface_offsets: jnp.ndarray
    shock_direction: jnp.ndarray
    shock_zones: jnp.ndarray
    field_obliquity_degrees: jnp.ndarray
    upstream_fast_speed: jnp.ndarray
    downstream_fast_speed: jnp.ndarray
    normal_shock_speed: jnp.ndarray
    upstream_fast_mach: jnp.ndarray
    downstream_fast_mach: jnp.ndarray
    sampling_valid: jnp.ndarray
    shock_speed_valid: jnp.ndarray
    density_compression: jnp.ndarray
    entropy_increase: jnp.ndarray
    upstream_superfast: jnp.ndarray
    downstream_subfast: jnp.ndarray
    fast_shock_cells: jnp.ndarray


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def identify_mhd_shock_candidate_zones(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    shock_direction: FIELD_TYPE,
    minimum_relative_jump: float = 1.0e-6,
):
    """Identify compressive MHD candidates without a hydro Mach threshold.

    Fast shocks compress and heat the gas, but their jump strength cannot be
    mapped to a sonic Mach number using the hydrodynamic pressure relation.
    Candidate zones therefore retain the converging-flow and aligned-gradient
    criteria and require only resolved positive density and gas-pressure
    jumps.  The fast-mode classification is applied after adaptive sampling.
    """
    if not config.mhd:
        raise ValueError("MHD shock candidates require config.mhd=True.")

    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    radius = (
        helper_data.geometric_centers
        if config.geometry == SPHERICAL
        else None
    )
    converging = _shock_zone_criterion_converging_flow(
        primitive_state,
        config,
        registered_variables,
        radius,
    )
    aligned = _shock_zone_criterion_aligned_gradients(
        pressure,
        density,
        config,
        radius,
    )
    pressure_post, pressure_pre, density_post, density_pre = (
        get_post_pre_shock_values(
            shock_direction,
            pressure,
            density,
        )
    )
    density_scale = jnp.maximum(
        jnp.maximum(jnp.abs(density_post), jnp.abs(density_pre)),
        1.0e-30,
    )
    pressure_scale = jnp.maximum(
        jnp.maximum(jnp.abs(pressure_post), jnp.abs(pressure_pre)),
        1.0e-30,
    )
    resolved_compression = (
        density_post - density_pre > minimum_relative_jump * density_scale
    ) & (
        pressure_post - pressure_pre > minimum_relative_jump * pressure_scale
    )
    return converging & aligned & resolved_compression


def _sample_mhd_states(
    primitive_state,
    shock_zones,
    shock_direction,
    surface_offsets,
    registered_variables,
    max_steps=8,
):
    """Sample primitive MHD states immediately outside both zone boundaries."""
    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    velocity_index = registered_variables.velocity_index
    magnetic_index = registered_variables.magnetic_index
    center_offsets = shock_direction * surface_offsets[jnp.newaxis, ...]

    pressure_post, pressure_pre, density_post, density_pre, valid_pd, _, _ = (
        get_adaptive_post_pre_shock_values(
            shock_direction,
            shock_zones,
            pressure,
            density,
            max_steps=max_steps,
            center_offsets=center_offsets,
        )
    )
    vx_post, vx_pre, vy_post, vy_pre, valid_vxy, _, _ = (
        get_adaptive_post_pre_shock_values(
            shock_direction,
            shock_zones,
            primitive_state[velocity_index.x],
            primitive_state[velocity_index.y],
            max_steps=max_steps,
            center_offsets=center_offsets,
        )
    )
    vz_post, vz_pre, bx_post, bx_pre, valid_vzbx, _, _ = (
        get_adaptive_post_pre_shock_values(
            shock_direction,
            shock_zones,
            primitive_state[velocity_index.z],
            primitive_state[magnetic_index.x],
            max_steps=max_steps,
            center_offsets=center_offsets,
        )
    )
    by_post, by_pre, bz_post, bz_pre, valid_bybz, _, _ = (
        get_adaptive_post_pre_shock_values(
            shock_direction,
            shock_zones,
            primitive_state[magnetic_index.y],
            primitive_state[magnetic_index.z],
            max_steps=max_steps,
            center_offsets=center_offsets,
        )
    )

    velocity_post = jnp.stack((vx_post, vy_post, vz_post), axis=0)
    velocity_pre = jnp.stack((vx_pre, vy_pre, vz_pre), axis=0)
    magnetic_post = jnp.stack((bx_post, by_post, bz_post), axis=0)
    magnetic_pre = jnp.stack((bx_pre, by_pre, bz_pre), axis=0)
    sampling_valid = valid_pd & valid_vxy & valid_vzbx & valid_bybz
    return (
        density_pre,
        pressure_pre,
        velocity_pre,
        magnetic_pre,
        density_post,
        pressure_post,
        velocity_post,
        magnetic_post,
        sampling_valid,
    )


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def find_fast_mhd_shocks(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    gamma_gas: float = 5.0 / 3.0,
    fast_mach_minimum: float = 1.0,
) -> MHDShockFinderResult:
    """Find candidate surfaces and classify locally consistent fast shocks."""
    if not config.mhd:
        raise ValueError("Fast MHD shock finding requires config.mhd=True.")
    if config.dimensionality != 3:
        raise NotImplementedError(
            "Fast MHD shock finding currently supports three dimensions only."
        )

    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    radius = (
        helper_data.geometric_centers
        if config.geometry == SPHERICAL
        else None
    )
    shock_direction = _calculate_shock_direction(
        pressure,
        density,
        config,
        radius,
    )
    shock_zones = identify_mhd_shock_candidate_zones(
        primitive_state,
        config,
        registered_variables,
        helper_data,
        shock_direction,
    )
    shock_surface = identify_shock_surface(
        primitive_state,
        shock_zones,
        shock_direction,
        config,
        registered_variables,
    )
    surface_offsets = calculate_shock_surface_offsets(
        primitive_state,
        shock_surface,
        shock_direction,
        config,
        registered_variables,
    )
    sampled_states = _sample_mhd_states(
        primitive_state,
        shock_zones,
        shock_direction,
        surface_offsets,
        registered_variables,
    )
    diagnostics = calculate_fast_shock_diagnostics(
        *sampled_states[:8],
        shock_direction=shock_direction,
        sampling_valid=shock_surface & sampled_states[8],
        gamma_gas=gamma_gas,
        fast_mach_minimum=fast_mach_minimum,
    )

    valid_surface = shock_surface & diagnostics.sampling_valid

    def surface_float(values):
        return jnp.where(valid_surface, values, jnp.nan)

    def surface_flag(values):
        return shock_surface & values

    return MHDShockFinderResult(
        shock_surface_cells=shock_surface,
        shock_surface_offsets=surface_offsets,
        shock_direction=shock_direction,
        shock_zones=shock_zones,
        field_obliquity_degrees=surface_float(
            diagnostics.field_obliquity_degrees
        ),
        upstream_fast_speed=surface_float(diagnostics.upstream_fast_speed),
        downstream_fast_speed=surface_float(diagnostics.downstream_fast_speed),
        normal_shock_speed=surface_float(diagnostics.normal_shock_speed),
        upstream_fast_mach=surface_float(diagnostics.upstream_fast_mach),
        downstream_fast_mach=surface_float(diagnostics.downstream_fast_mach),
        sampling_valid=surface_flag(diagnostics.sampling_valid),
        shock_speed_valid=surface_flag(diagnostics.shock_speed_valid),
        density_compression=surface_flag(diagnostics.density_compression),
        entropy_increase=surface_flag(diagnostics.entropy_increase),
        upstream_superfast=surface_flag(diagnostics.upstream_superfast),
        downstream_subfast=surface_flag(diagnostics.downstream_subfast),
        fast_shock_cells=surface_flag(diagnostics.fast_shock),
    )
