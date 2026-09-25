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

@partial(
    jax.jit,
    static_argnames=[
        "registered_variables",
        "config",
        "mach_sampling_steps",
        "mach_sampling_adaptive",
        "mach_sampling_extend",
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
            basis and validation. Only affects the Mach-number output
            (phase 4); thermal-energy flux (phase 5) still uses the
            fixed-step sampler regardless of this flag.
        mach_sampling_extend: opt-in (default False), only with
            ``mach_sampling_adaptive=True``: after a ray leaves the shock
            zone, keep walking while the pressure keeps falling (pre-shock
            side) / rising (post-shock side) by more than 2% per step, and take
            the post-shock sample at the pressure peak along the walk, so the
            samples land on the upstream/downstream plateau even when the
            zone ends inside a numerically smeared strong shock (see
            ``get_post_pre_shock_values_adaptive`` and FIXES_TODO.md round 23).
            Mach number only; the thermal-energy flux is unaffected.

    Returns:
        ShockFinderResult
    """
    if mach_sampling_extend and not mach_sampling_adaptive:
        raise ValueError("mach_sampling_extend requires mach_sampling_adaptive=True.")

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
        adaptive=mach_sampling_adaptive, shock_zones=shock_zones,
        extend_monotone=mach_sampling_extend,
    )

    # Phase 5: thermal-energy flux at shock-surface cells. calculate_thermal_
    # energy_flux has not been ported to the adaptive walk (see
    # get_post_pre_shock_values_adaptive's docstring) -- it always uses the
    # fixed-step sampler for its own pre-shock density/pressure. When
    # mach_sampling_adaptive=False, `mach_sampling_steps` is passed through
    # unchanged (original behavior: same sampling distance for both, so
    # pre-shock state stays consistent with the Mach number it's combined
    # with). When mach_sampling_adaptive=True, `mach_sampling_steps` instead
    # sizes the adaptive walk's step cap (can be large, e.g. 15) and must
    # NOT also be used as the flux's own fixed-step distance/boundary-margin
    # -- doing so was confirmed to desync the two (flux force-zeroed over a
    # much wider boundary margin than the now margin-free adaptive Mach,
    # breaking the "flux nonzero exactly at surface" invariant). Falls back
    # to the pre-existing default (1) instead; this means the flux's own
    # pre-shock sample point may differ from the Mach's when adaptive is on
    # -- a known, documented accuracy caveat, not yet resolved by porting
    # flux to the adaptive walk too.
    flux_sampling_steps = 1 if mach_sampling_adaptive else mach_sampling_steps
    thermal_energy_flux = calculate_thermal_energy_flux(
        primitive_state=primitive_state,
        shock_surface=shock_surface,
        shock_direction=shock_direction,
        mach_numbers=mach_numbers,
        config=config,
        registered_variables=registered_variables,
        sampling_steps=flux_sampling_steps,
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