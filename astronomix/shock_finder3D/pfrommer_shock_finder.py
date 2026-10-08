from functools import partial

import jax
import jax.numpy as jnp

from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)

from astronomix.shock_finder3D._data_structures import ShockFinderResult
from astronomix.shock_finder3D._gradients import (
    _calculate_shock_direction,
    _calculate_velocity_divergence,
)
from astronomix.shock_finder3D._shock_zones import (
    get_post_pre_shock_values,
    get_post_pre_shock_values_adaptive,
    identify_shock_zones,
)
from astronomix.shock_finder3D._shock_surface import identify_shock_surface
from astronomix.shock_finder3D._shock_mach import (
    _calculate_mach_at_surface,
    mach_from_pressure_samples,
    mach_from_velocity_jump,
)
from astronomix.shock_finder3D._energy_dissipation import (
    calculate_thermal_energy_flux,
    thermal_energy_flux_from_pre_state,
)

def _velocity_consistent(primitive_state, config, registered_variables, shock_direction, fraction):
    """Local Rankine-Hugoniot check, see find_shocks_pfrommer's
    ``mach_velocity_consistency``: True where the immediate-neighbour
    normal-velocity jump carries at least ``fraction`` of the Mach excess the
    immediate-neighbour pressure jump implies."""
    gamma_gas = 5 / 3
    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    vel_idx = registered_variables.velocity_index
    velocity = (
        [primitive_state[vel_idx]] if isinstance(vel_idx, int)
        else [primitive_state[i] for i in (vel_idx.x, vel_idx.y, vel_idx.z)[:config.dimensionality]]
    )
    normal_velocity = sum(v * d for v, d in zip(velocity, shock_direction))
    p_post, p_pre, u_post, u_pre = get_post_pre_shock_values(shock_direction, pressure, normal_velocity)
    _, _, _, rho_pre = get_post_pre_shock_values(shock_direction, pressure, density)
    mach_pressure = mach_from_pressure_samples(p_post, p_pre, gamma_gas)
    # +d_s points to the pre-shock side: a compression has u_n(post) > u_n(pre)
    mach_velocity = mach_from_velocity_jump(u_post - u_pre, p_pre, rho_pre, gamma_gas)
    return mach_velocity - 1.0 >= fraction * (mach_pressure - 1.0)


@partial(
    jax.jit,
    static_argnames=[
        "registered_variables",
        "config",
        "mach_sampling_steps",
        "mach_sampling_adaptive",
        "mach_sampling_extend",
        "mach_velocity_consistency",
    ],
)
def find_shocks_pfrommer(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    mach_min: float = 1.3,
    mach_sampling_steps: int = 1,
    mach_sampling_adaptive: bool = False,
    mach_sampling_extend: bool = False,
    mach_velocity_consistency: float = 0.0,
) -> ShockFinderResult:
    """
    Main entry point: Identify shocks using Pfrommer et al. 2017 methodology.

    Phases:
        1. Shock direction:  d_s = -∇T / |∇T|,  shape (ndim, *spatial_shape)
        2. Shock zones:      cells satisfying all three criteria (~3-4 cells thick)
        3. Shock surface:    single cell of max compression per zone
        4. Mach numbers:     Rankine-Hugoniot M at surface cells

    Works for 1D, 2D, and 3D without modification.

    Args:
        primitive_state:      (num_vars, *spatial_shape)
        config:               simulation configuration
        registered_variables: registry of variable indices
        helper_data:          geometric centers etc.
        mach_min:             minimum Mach threshold (default 1.3)
        mach_sampling_steps:  number of cells the Rankine-Hugoniot pre-/
            post-shock sampling (Mach number + thermal-energy flux) walks
            out from each shock-surface cell, along the local shock
            direction. Default 1 (immediate neighbor only) matches
            pre-existing behavior and is a strict backward-compatible
            default; it underestimates the Mach number for any shock
            smeared over more than ~1 cell (see FIXES_TODO.md item 1b in
            pytests/shock_finder3D/). Shock-*zone* detection (criterion 3,
            `_shock_zones.py`) is intentionally unaffected by this
            parameter and always uses immediate-neighbor sampling -- only
            the reported Mach number and thermal-energy flux use this
            wider sample. When `mach_sampling_adaptive=False` (the
            default), cells within `mach_sampling_steps` of a domain
            boundary are excluded from both outputs (jnp.roll wraps at the
            edge, see `_make_interior_mask`); with `mach_sampling_adaptive=
            True`, this instead acts as the maximum number of steps the
            adaptive walk below is allowed to take.
        mach_sampling_adaptive: opt-in (default False, i.e. fully
            backward-compatible), walks each ray until it exits the shock
            zone (`get_post_pre_shock_values_adaptive`, following Schaal &
            Springel 2015's shock-surface construction method) instead of a
            fixed `mach_sampling_steps` cells, and along the continuous
            local shock-normal direction rather than a single dominant grid
            axis. See FIXES_TODO.md item 1b (round 16) for the literature
            basis and validation. Mach number and thermal-energy flux then
            both come from this one walk (the flux from the pre-shock
            pressure/density at the same point the Mach number uses).
        mach_sampling_extend: opt-in (default False), only with
            ``mach_sampling_adaptive=True``: after a ray leaves the shock
            zone, keep walking while the pressure keeps falling (pre-shock
            side) / rising (post-shock side, only while the cell being left is
            still compressive, div v < 0) by more than 2% per step, and take
            the post-shock sample at the pressure peak along the walk, so the
            samples land on the upstream/downstream plateau even when the
            zone ends inside a numerically smeared strong shock (see
            ``get_post_pre_shock_values_adaptive`` and FIXES_TODO.md round 23).
            Applies to both the Mach number and the thermal-energy flux.
        mach_velocity_consistency: opt-in (default 0, off), only with
            ``mach_sampling_adaptive=True``: Rankine-Hugoniot check of each
            shock-surface cell's *local* velocity jump. With the immediate
            neighbours along the shock direction (criterion 3's sampling),
            the pressure ratio gives M_p and the normal-velocity jump gives
            M_u (``mach_from_velocity_jump``); the cell is rejected unless
            ``M_u - 1 >= mach_velocity_consistency * (M_p - 1)``. A pressure
            jump without the matching velocity jump is not a shock: e.g. the
            far pressure tail of a smoothed Sedov IC in cold gas at rest
            passes all three zone criteria and, through the extended walk,
            re-counts the real blast's dissipated flux as its own
            (PROGRESS.md 2026-10-07). The check has to be local: sampled at
            the extended walk's end points, such a cell sees the real
            blast's own, RH-consistent jump. Measured on 48^3 Sedov blasts:
            (M_u - 1) / (M_p - 1) ~ 0.01-0.03 for those cells and for the
            not-yet-formed blast of the first steps, >= 1 for resolved
            shocks (M_u / M_p >= 0.97 at the 1st percentile, Mach 2-9 locally).
            The reported Mach number is unchanged.

    Returns:
        ShockFinderResult
    """
    if mach_sampling_extend and not mach_sampling_adaptive:
        raise ValueError("mach_sampling_extend requires mach_sampling_adaptive=True.")
    if mach_velocity_consistency > 0 and not mach_sampling_adaptive:
        raise ValueError("mach_velocity_consistency requires mach_sampling_adaptive=True.")

    pressure = primitive_state[registered_variables.pressure_index]
    density  = primitive_state[registered_variables.density_index]
    r = helper_data.geometric_centers if config.geometry == SPHERICAL else None

    # Phase 1: shock direction (ndim, *spatial_shape)
    shock_direction = _calculate_shock_direction(pressure, density, config, r)

    # Phase 2: shock zones (*spatial_shape)
    shock_zones = identify_shock_zones(
        primitive_state, config, registered_variables,
        helper_data, shock_direction, mach_min,
    )

    # Phase 3: shock surface (*spatial_shape)
    shock_surface = identify_shock_surface(
        primitive_state, shock_zones, shock_direction,
        config, registered_variables,
    )

    if mach_sampling_adaptive:
        # Phases 4-5 from one adaptive walk (Schaal & Springel 2015 Sec.
        # 2.3.3, optionally extended -- see get_post_pre_shock_values_adaptive):
        # pressure on both sides gives the Mach number, pressure and density
        # at the *same* pre-shock point give the thermal-energy flux, so the
        # two are consistent (FIXES_TODO.md round 24; previously the flux
        # used the fixed 1-cell sampler here). Rays that never leave the
        # zone within mach_sampling_steps are excluded from both.
        gamma_gas = 5 / 3
        # the extension follows the shock's compression (div v < 0) and stops
        # one cell past it -- see get_post_pre_shock_values_adaptive
        compressive = (
            _calculate_velocity_divergence(primitive_state, config, registered_variables, r) < 0
            if mach_sampling_extend else None
        )
        p_post, p_pre, _, rho_pre, exited_post, exited_pre, post_shock_steps, _ = (
            get_post_pre_shock_values_adaptive(
                shock_direction, pressure, density, shock_zones,
                max_steps=mach_sampling_steps, extend_monotone=mach_sampling_extend,
                compressive=compressive, return_sample_steps=True,
            )
        )
        pre_shock_pressure = p_pre
        valid = shock_surface & exited_post & exited_pre
        mach_pressure = mach_from_pressure_samples(p_post, p_pre, gamma_gas)
        if mach_velocity_consistency > 0:
            valid = valid & _velocity_consistent(
                primitive_state, config, registered_variables, shock_direction,
                mach_velocity_consistency,
            )
        mach_numbers = jnp.where(valid, mach_pressure, 0.0)
        thermal_energy_flux = jnp.where(
            valid,
            thermal_energy_flux_from_pre_state(mach_numbers, p_pre, rho_pre, gamma_gas),
            0.0,
        )
    else:
        post_shock_steps = None
        pre_shock_pressure = None
        # Phase 4: Mach numbers (*spatial_shape), fixed-step sampling
        mach_numbers = _calculate_mach_at_surface(
            primitive_state, shock_surface, shock_direction,
            config, registered_variables, sampling_steps=mach_sampling_steps,
        )

        # Phase 5: thermal-energy flux at shock-surface cells, same fixed
        # sampling distance as the Mach number.
        thermal_energy_flux = calculate_thermal_energy_flux(
            primitive_state=primitive_state,
            shock_surface=shock_surface,
            shock_direction=shock_direction,
            mach_numbers=mach_numbers,
            config=config,
            registered_variables=registered_variables,
            sampling_steps=mach_sampling_steps,
        )

    num_shocks     = jnp.sum(shock_surface, dtype=jnp.int32)
    shock_ids      = jnp.where(shock_surface, 1, 0)
    shock_zone_ids = jnp.where(shock_zones,   1, 0)

    return ShockFinderResult(
        shock_surface_cells=shock_surface,
        shock_direction=shock_direction,
        mach_numbers=mach_numbers,
        thermal_energy_flux=thermal_energy_flux,
        shock_zones=shock_zones,
        num_shocks=num_shocks,
        shock_ids=shock_ids,
        shock_zone_ids=shock_zone_ids,
        post_shock_steps=post_shock_steps,
        pre_shock_pressure=pre_shock_pressure,
    )