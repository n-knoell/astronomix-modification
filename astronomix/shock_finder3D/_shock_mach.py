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
from astronomix.shock_finder3D._shock_zones import (
    get_post_pre_shock_values,
    get_post_pre_shock_values_adaptive,
    _make_interior_mask,
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
    static_argnames=["registered_variables", "config", "sampling_steps", "adaptive"],
)
def _calculate_mach_at_surface(
    primitive_state: STATE_TYPE,
    shock_surface: BOOL_FIELD_TYPE,
    shock_direction: FIELD_TYPE,        # ← needed for direction-aware p_post/p_pre
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    sampling_steps: int = 1,
    adaptive: bool = False,
    shock_zones: BOOL_FIELD_TYPE = None,
) -> FIELD_TYPE:
    gamma_gas = 5 / 3

    pressure = primitive_state[registered_variables.pressure_index]
    density  = primitive_state[registered_variables.density_index]
    temperature = pressure / density

    if adaptive:
        # Walk until leaving the shock zone (Schaal & Springel 2015, Sec.
        # 2.3.3 -- see get_post_pre_shock_values_adaptive's docstring),
        # instead of a fixed number of cells. Opt-in (default False): the
        # `sampling_steps=1` fixed-step path above is left completely
        # unchanged for every existing caller (see FIXES_TODO.md item 1a --
        # a similar zone-aware walk was previously applied as the *default*
        # and silently broke an unrelated caller; this one only ever runs
        # when explicitly requested).
        if shock_zones is None:
            raise ValueError("adaptive=True requires shock_zones to be passed.")
        p_post, p_pre, _, _, exited_post, exited_pre = get_post_pre_shock_values_adaptive(
            shock_direction, pressure, temperature, shock_zones, max_steps=sampling_steps,
        )
        converged = exited_post & exited_pre
    else:
        # direction-aware post/pre selection — same helper as criterion 3.
        # sampling_steps=1 (the default, matching pre-fix behavior) samples only
        # the immediate neighbor, which underestimates the Mach number whenever
        # the shock is smeared over more than ~1 cell (see FIXES_TODO.md item
        # 1b): the sample lands inside the transition, not past it into the
        # asymptotic pre-/post-shock plateau. A caller expecting realistic Mach
        # numbers for a numerically-smeared shock (typically ~2-4 cells wide for
        # a 2nd-order scheme, independent of grid resolution -- see FIXES_TODO.md
        # round 9/10) should pass a larger sampling_steps explicitly.
        p_post, p_pre, _, _ = get_post_pre_shock_values(
            shock_direction, pressure, temperature, max_steps=sampling_steps,
        )
        converged = jnp.ones_like(shock_surface)

    # calculate Mach number for all cells
    # p₂/p₁ = p_post/p_pre, but clamp to 1 to avoid numerical issues with very weak shocks
    p_ratio = jnp.maximum(p_post / jnp.maximum(p_pre, 1e-30), 1.0)
    # as p₂/p₁ = (2γM² − (γ−1)) / (γ+1) so M = √[ (p₂/p₁ · (γ+1) + (γ−1)) / (2γ) ]
    M = jnp.sqrt((p_ratio * (gamma_gas + 1) + (gamma_gas - 1)) / (2 * gamma_gas))

    if adaptive:
        # get_post_pre_shock_values_adaptive clamps at the boundary
        # (mode="nearest") instead of wrapping, so no boundary-margin
        # exclusion is needed here -- only rays that never found a genuine
        # zone exit within sampling_steps are excluded.
        valid_interior = converged
    else:
        # get_post_pre_shock_values samples via jnp.roll, which wraps around at
        # the domain edge -- cells within sampling_steps of a boundary would
        # otherwise report a physically meaningless wrapped-around Mach number
        # (same guard as calculate_thermal_energy_flux in _energy_dissipation.py).
        valid_interior = _make_interior_mask(pressure.shape, margin=sampling_steps)

    # write Mach only at surface cells clear of the boundary, zero elsewhere
    mach_array = jnp.where(shock_surface & valid_interior, M, 0.0)

    return mach_array