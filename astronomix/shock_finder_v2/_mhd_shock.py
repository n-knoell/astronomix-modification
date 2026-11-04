"""Orientation-aware diagnostics for fast magnetohydrodynamic shocks.

The hydrodynamic shock finder derives a sonic Mach number from the gas-pressure
jump.  That relation is not valid for MHD states.  This module instead works
with already sampled upstream and downstream primitive states and computes the
local fast-magnetosonic speeds relative to an arbitrary shock normal.

Magnetic fields use the astronomix convention ``B / sqrt(mu_0)``.  Therefore
the squared Alfvén speed is simply ``B**2 / rho`` in code units.
"""

from typing import NamedTuple

import jax.numpy as jnp


class FastShockDiagnostics(NamedTuple):
    """Per-cell quantities and quality flags for a fast-shock candidate."""

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
    fast_shock: jnp.ndarray


def _unit_direction(direction, numerical_floor=1.0e-30):
    """Return unit vectors and a mask identifying finite non-zero vectors."""
    direction = jnp.asarray(direction)
    magnitude = jnp.sqrt(jnp.sum(direction**2, axis=0))
    valid = jnp.all(jnp.isfinite(direction), axis=0) & (
        magnitude > numerical_floor
    )
    safe_magnitude = jnp.where(valid, magnitude, 1.0)
    return direction / safe_magnitude[jnp.newaxis, ...], valid


def calculate_field_obliquity_degrees(
    magnetic_field,
    shock_direction,
    numerical_floor=1.0e-30,
):
    """Return the acute angle between the magnetic field and shock normal.

    The result lies between 0 degrees (parallel or anti-parallel) and 90
    degrees (perpendicular).  A vanishing or non-finite magnetic field has no
    defined orientation and is reported as NaN.
    """
    magnetic_field = jnp.asarray(magnetic_field)
    unit_normal, normal_valid = _unit_direction(
        shock_direction,
        numerical_floor=numerical_floor,
    )
    magnetic_magnitude = jnp.sqrt(jnp.sum(magnetic_field**2, axis=0))
    magnetic_valid = jnp.all(jnp.isfinite(magnetic_field), axis=0) & (
        magnetic_magnitude > numerical_floor
    )
    safe_magnitude = jnp.where(magnetic_valid, magnetic_magnitude, 1.0)
    cosine = jnp.abs(jnp.sum(magnetic_field * unit_normal, axis=0)) / (
        safe_magnitude
    )
    cosine = jnp.clip(cosine, 0.0, 1.0)
    angle = jnp.degrees(jnp.arccos(cosine))
    return jnp.where(normal_valid & magnetic_valid, angle, jnp.nan)


def calculate_fast_magnetosonic_speed(
    density,
    pressure,
    magnetic_field,
    shock_direction,
    gamma_gas=5.0 / 3.0,
    numerical_floor=1.0e-30,
):
    """Calculate the local fast-magnetosonic speed along a shock normal."""
    density = jnp.asarray(density)
    pressure = jnp.asarray(pressure)
    magnetic_field = jnp.asarray(magnetic_field)
    unit_normal, normal_valid = _unit_direction(
        shock_direction,
        numerical_floor=numerical_floor,
    )

    state_valid = (
        normal_valid
        & jnp.isfinite(density)
        & jnp.isfinite(pressure)
        & jnp.all(jnp.isfinite(magnetic_field), axis=0)
        & (density > numerical_floor)
        & (pressure > 0.0)
    )
    safe_density = jnp.where(state_valid, density, 1.0)
    safe_pressure = jnp.where(state_valid, pressure, 0.0)

    sound_speed_squared = gamma_gas * safe_pressure / safe_density
    magnetic_squared = jnp.sum(magnetic_field**2, axis=0)
    magnetic_normal = jnp.sum(magnetic_field * unit_normal, axis=0)
    alfven_speed_squared = magnetic_squared / safe_density
    normal_alfven_speed_squared = magnetic_normal**2 / safe_density

    sum_speeds_squared = sound_speed_squared + alfven_speed_squared
    discriminant = jnp.maximum(
        sum_speeds_squared**2
        - 4.0 * sound_speed_squared * normal_alfven_speed_squared,
        0.0,
    )
    fast_speed_squared = 0.5 * (
        sum_speeds_squared + jnp.sqrt(discriminant)
    )
    fast_speed = jnp.sqrt(jnp.maximum(fast_speed_squared, 0.0))
    return jnp.where(state_valid, fast_speed, jnp.nan)


def estimate_normal_shock_speed(
    density_upstream,
    density_downstream,
    velocity_upstream,
    velocity_downstream,
    shock_direction,
    minimum_relative_density_jump=1.0e-6,
    numerical_floor=1.0e-30,
):
    """Estimate the lab-frame normal shock speed from mass conservation.

    For normal velocities ``v_1`` and ``v_2`` and densities ``rho_1`` and
    ``rho_2``, conservation of mass gives

        s = (rho_2 v_2 - rho_1 v_1) / (rho_2 - rho_1).

    The estimate is invalid when the density jump is too small because the
    denominator then amplifies numerical noise.
    """
    density_upstream = jnp.asarray(density_upstream)
    density_downstream = jnp.asarray(density_downstream)
    velocity_upstream = jnp.asarray(velocity_upstream)
    velocity_downstream = jnp.asarray(velocity_downstream)
    unit_normal, normal_valid = _unit_direction(
        shock_direction,
        numerical_floor=numerical_floor,
    )

    velocity_upstream_normal = jnp.sum(
        velocity_upstream * unit_normal,
        axis=0,
    )
    velocity_downstream_normal = jnp.sum(
        velocity_downstream * unit_normal,
        axis=0,
    )
    density_difference = density_downstream - density_upstream
    density_scale = jnp.maximum(
        jnp.maximum(jnp.abs(density_upstream), jnp.abs(density_downstream)),
        numerical_floor,
    )
    valid = (
        normal_valid
        & jnp.isfinite(density_upstream)
        & jnp.isfinite(density_downstream)
        & jnp.all(jnp.isfinite(velocity_upstream), axis=0)
        & jnp.all(jnp.isfinite(velocity_downstream), axis=0)
        & (density_upstream > numerical_floor)
        & (density_downstream > numerical_floor)
        & (
            jnp.abs(density_difference)
            > minimum_relative_density_jump * density_scale
        )
    )
    safe_difference = jnp.where(valid, density_difference, 1.0)
    shock_speed = (
        density_downstream * velocity_downstream_normal
        - density_upstream * velocity_upstream_normal
    ) / safe_difference
    shock_speed = jnp.where(valid, shock_speed, jnp.nan)
    return shock_speed, valid


def calculate_fast_shock_diagnostics(
    density_upstream,
    pressure_upstream,
    velocity_upstream,
    magnetic_upstream,
    density_downstream,
    pressure_downstream,
    velocity_downstream,
    magnetic_downstream,
    shock_direction,
    sampling_valid=True,
    gamma_gas=5.0 / 3.0,
    fast_mach_minimum=1.0,
    minimum_relative_density_jump=1.0e-6,
    numerical_floor=1.0e-30,
):
    """Calculate orientation-aware fast-shock diagnostics and flags.

    This function classifies already sampled states.  It does not identify a
    shock zone or sample the simulation grid.  A conservative fast-shock flag
    requires valid samples, a resolvable compressive density jump, increasing
    gas entropy, super-fast upstream flow, and sub-fast downstream flow.
    """
    density_upstream = jnp.asarray(density_upstream)
    pressure_upstream = jnp.asarray(pressure_upstream)
    velocity_upstream = jnp.asarray(velocity_upstream)
    magnetic_upstream = jnp.asarray(magnetic_upstream)
    density_downstream = jnp.asarray(density_downstream)
    pressure_downstream = jnp.asarray(pressure_downstream)
    velocity_downstream = jnp.asarray(velocity_downstream)
    magnetic_downstream = jnp.asarray(magnetic_downstream)
    sampling_valid = jnp.asarray(sampling_valid, dtype=jnp.bool_)

    unit_normal, normal_valid = _unit_direction(
        shock_direction,
        numerical_floor=numerical_floor,
    )
    obliquity = calculate_field_obliquity_degrees(
        magnetic_upstream,
        unit_normal,
        numerical_floor=numerical_floor,
    )
    upstream_fast_speed = calculate_fast_magnetosonic_speed(
        density_upstream,
        pressure_upstream,
        magnetic_upstream,
        unit_normal,
        gamma_gas=gamma_gas,
        numerical_floor=numerical_floor,
    )
    downstream_fast_speed = calculate_fast_magnetosonic_speed(
        density_downstream,
        pressure_downstream,
        magnetic_downstream,
        unit_normal,
        gamma_gas=gamma_gas,
        numerical_floor=numerical_floor,
    )
    shock_speed, shock_speed_valid = estimate_normal_shock_speed(
        density_upstream,
        density_downstream,
        velocity_upstream,
        velocity_downstream,
        unit_normal,
        minimum_relative_density_jump=minimum_relative_density_jump,
        numerical_floor=numerical_floor,
    )

    velocity_upstream_normal = jnp.sum(
        velocity_upstream * unit_normal,
        axis=0,
    )
    velocity_downstream_normal = jnp.sum(
        velocity_downstream * unit_normal,
        axis=0,
    )
    upstream_fast_mach = jnp.abs(
        velocity_upstream_normal - shock_speed
    ) / upstream_fast_speed
    downstream_fast_mach = jnp.abs(
        velocity_downstream_normal - shock_speed
    ) / downstream_fast_speed

    finite_states = (
        normal_valid
        & jnp.isfinite(density_upstream)
        & jnp.isfinite(pressure_upstream)
        & jnp.isfinite(density_downstream)
        & jnp.isfinite(pressure_downstream)
        & jnp.all(jnp.isfinite(velocity_upstream), axis=0)
        & jnp.all(jnp.isfinite(velocity_downstream), axis=0)
        & jnp.all(jnp.isfinite(magnetic_upstream), axis=0)
        & jnp.all(jnp.isfinite(magnetic_downstream), axis=0)
        & (density_upstream > numerical_floor)
        & (density_downstream > numerical_floor)
        & (pressure_upstream > 0.0)
        & (pressure_downstream > 0.0)
    )
    density_scale = jnp.maximum(
        jnp.maximum(jnp.abs(density_upstream), jnp.abs(density_downstream)),
        numerical_floor,
    )
    density_compression = (
        density_downstream - density_upstream
        > minimum_relative_density_jump * density_scale
    )

    safe_density_upstream = jnp.maximum(density_upstream, numerical_floor)
    safe_density_downstream = jnp.maximum(density_downstream, numerical_floor)
    safe_pressure_upstream = jnp.maximum(pressure_upstream, numerical_floor)
    safe_pressure_downstream = jnp.maximum(pressure_downstream, numerical_floor)
    entropy_proxy_upstream = (
        jnp.log(safe_pressure_upstream)
        - gamma_gas * jnp.log(safe_density_upstream)
    )
    entropy_proxy_downstream = (
        jnp.log(safe_pressure_downstream)
        - gamma_gas * jnp.log(safe_density_downstream)
    )
    entropy_increase = entropy_proxy_downstream > entropy_proxy_upstream

    common_valid = (
        sampling_valid
        & finite_states
        & shock_speed_valid
        & jnp.isfinite(upstream_fast_mach)
        & jnp.isfinite(downstream_fast_mach)
    )
    upstream_superfast = common_valid & (
        upstream_fast_mach > fast_mach_minimum
    )
    downstream_subfast = common_valid & (downstream_fast_mach < 1.0)
    fast_shock = (
        common_valid
        & density_compression
        & entropy_increase
        & upstream_superfast
        & downstream_subfast
    )

    return FastShockDiagnostics(
        field_obliquity_degrees=obliquity,
        upstream_fast_speed=upstream_fast_speed,
        downstream_fast_speed=downstream_fast_speed,
        normal_shock_speed=shock_speed,
        upstream_fast_mach=upstream_fast_mach,
        downstream_fast_mach=downstream_fast_mach,
        sampling_valid=sampling_valid,
        shock_speed_valid=shock_speed_valid,
        density_compression=density_compression,
        entropy_increase=entropy_increase,
        upstream_superfast=upstream_superfast,
        downstream_subfast=downstream_subfast,
        fast_shock=fast_shock,
    )
