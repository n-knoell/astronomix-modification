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
from astronomix.shock_finder3D._gradients import _calculate_shock_direction
from astronomix.shock_finder3D._shock_zones import identify_shock_zones
from astronomix.shock_finder3D._shock_surface import identify_shock_surface
from astronomix.shock_finder3D._shock_mach import _calculate_mach_at_surface
from astronomix.shock_finder3D._energy_dissipation import calculate_thermal_energy_flux

@partial(jax.jit, static_argnames=["registered_variables", "config", "mach_sampling_steps"])
def find_shocks_pfrommer(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    mach_min: float = 1.3,
    mach_sampling_steps: int = 1,
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
            wider sample. Cells within `mach_sampling_steps` of a domain
            boundary are excluded from both outputs (jnp.roll wraps at the
            edge, see `_make_interior_mask`).

    Returns:
        ShockFinderResult
    """
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

    # Phase 4: Mach numbers (*spatial_shape)
    mach_numbers = _calculate_mach_at_surface(
        primitive_state, shock_surface, shock_direction,
        config, registered_variables, sampling_steps=mach_sampling_steps,
    )

    # Phase 5: thermal-energy flux at shock-surface cells. Uses the same
    # sampling_steps as Mach above so the pre-shock state (density_pre,
    # pressure_pre -> sound_speed_pre) stays consistent with the Mach number
    # it's combined with (mach_numbers * sound_speed_pre) -- sampling at
    # different distances for the two would silently mix pre-shock states
    # from different points along the shock normal.
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
    )