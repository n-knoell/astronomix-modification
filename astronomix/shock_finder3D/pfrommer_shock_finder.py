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
    extrapolate_pre_state_to_shock,
    get_post_pre_shock_values,
    get_post_pre_shock_values_adaptive,
    identify_shock_zones,
    post_velocity_at_end_of_jump,
    sample_along_direction,
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

def _velocity_components(primitive_state, config, registered_variables):
    vel_idx = registered_variables.velocity_index
    return (
        [primitive_state[vel_idx]] if isinstance(vel_idx, int)
        else [primitive_state[i] for i in (vel_idx.x, vel_idx.y, vel_idx.z)[:config.dimensionality]]
    )


#: pre-shock walk of ``mach_sampling_at_shock``: the extension continues only
#: while a step's pressure drop is below this fraction of the previous step's
#: (shock foot vs. background gradient; 0.5 stops inside wide feet and
#: overshoots the flux by up to 12%). The same number bounds the gradient the
#: upstream extrapolation follows: -ln(0.7) per cell, a scale height of 2.8
#: cells; anything steeper is not told apart from the shock itself.
_PRE_EXTENSION_DROP_RATIO = 0.7

#: post-shock velocity search of ``mach_sampling_at_shock``: covers pressure
#: peaks up to this many steps behind the shock cell (2-4 on resolved shocks);
#: beyond, the velocity at the peak is used as without the option.
_POST_VELOCITY_SEARCH_STEPS = 6


def _velocity_consistent(primitive_state, config, registered_variables, shock_direction, fraction,
                         gamma_gas=5 / 3):
    """Local Rankine-Hugoniot check, see find_shocks_pfrommer's
    ``mach_velocity_consistency``: True where the immediate-neighbour
    normal-velocity jump carries at least ``fraction`` of the Mach excess the
    immediate-neighbour pressure jump implies."""
    pressure = primitive_state[registered_variables.pressure_index]
    density = primitive_state[registered_variables.density_index]
    velocity = _velocity_components(primitive_state, config, registered_variables)
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
        "mach_estimator",
        "continuous_normal",
        "mach_sampling_at_shock",
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
    mach_estimator: str = "pressure",
    continuous_normal: bool = False,
    mach_sampling_at_shock: bool = False,
    gamma_gas: float = 5 / 3,
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
        mach_estimator: ``"pressure"`` (default): the Mach number from the
            gas-pressure jump between the walk's post- and pre-shock samples.
            ``"velocity"`` (only with ``mach_sampling_adaptive=True``): from
            the normal-velocity jump at the same two points,
            ``u_2 - u_1 = 2 c_1 (M^2 - 1) / ((gamma + 1) M)``
            (``mach_from_velocity_jump``); cells where it is not above
            ``mach_min`` are dropped, and so are cells whose pressure jump at
            the same points does not confirm the shock (``M_p > mach_min`` and
            ``M_p - 1 >= f (M_u - 1)`` with ``f = mach_velocity_consistency``,
            0.5 if that is off): the converse of the velocity check. Phase B decision DB4 (2026-10-19,
            ``PROGRESS_PHASEB.md``): behind a thin, curved shock (the Sedov
            shell) the pressure falls steeply, so the smeared post-shock
            pressure peak is low and the pressure Mach number reads
            0.76/0.84/0.87 of the true one at 48^3/96^3/160^3; the velocity
            falls about half as fast (0.88/0.94/0.94). On planar shocks both
            are within 1% (B8.2). On shocks that inject CRs the velocity jump
            reads high (exact 1D shocks: +1.2% at zeta = 0.1, +2.6% at KR13's
            strong-shock 0.21; the gas-pressure jump reads -2% / -4.5%), since
            the CR-loaded shock compresses more than the gas-only RH relation
            assumes. The thermal-energy flux uses this Mach number with the
            same pre-shock state.
        continuous_normal: default False here; DSA turns it on by default
            (``CosmicRayGreyConfig.dsa_continuous_normal_surface``), 2D/3D: phase 3
            along the continuous shock normal instead of the dominant grid axis
            (the surface of ``astronomix/shock_finder_v2``; the zones stay ours).
            The surface is every zone cell whose div v is a
            local minimum one cell ahead / behind along the normal
            (``_shock_surface._find_shock_surface_along_normal``), instead of
            the argmin over the whole zone segment along the dominant axis.
            The surface no longer changes where the dominant normal component
            switches axis, and it costs O(N^3) instead of the segment scan's
            O(N^4) (0.6 vs 174 ms at 160^3, 3.5 vs 1135 ms at 256^3, RTX 2080
            Ti). It can hold more than one cell per grid line through a zone,
            so its area needs ``_surface_area.column_area_weights`` instead of
            ``surface_area_weights``. Mach sampling (phases 4-5) is unchanged.
            Sedov / planar / CWB comparison:
            ``pytests/shock_finder3D/finder_v2_comparison/README.md``.
        mach_sampling_at_shock: default False here; DSA turns it on by default
            (``CosmicRayGreyConfig.dsa_sampling_at_shock``). Only with
            ``mach_sampling_adaptive`` and ``mach_sampling_extend``. Reads the
            shock's upstream and downstream state at the shock instead of a few
            cells away from it (review 2026-11-15, ``PROGRESS_PHASEB.md``
            2026-11-16):
            (1) the pre-shock extension stops where the pressure profile is no
            longer the convex foot of the shock (``pre_extension_drop_ratio``
            of the walk), instead of following any background gradient steeper
            than 2% per cell to the step cap;
            (2) pre-shock pressure and density (Mach number, dissipated flux,
            ``pre_shock_pressure`` / ``pre_shock_density`` of the result) are
            extrapolated from the walk's sample back to the shock cell
            (``extrapolate_pre_state_to_shock``);
            (3) with ``mach_estimator="velocity"`` the post-shock velocity is
            read at the end of the shock's own velocity jump, at or before
            the pressure peak (``post_velocity_at_end_of_jump``).
            A shock with flat upstream gas and a post-shock plateau (planar
            tubes, the Sedov shell) is unchanged (Mach number to 1e-4).
            ``post_shock_steps`` / ``pre_shock_steps`` still point to the
            walk's own samples.
        gamma_gas: gas adiabatic index of the Rankine-Hugoniot relations
            (zone threshold, Mach numbers, dissipated flux). Default 5/3; DSA
            passes ``params.gamma``.

    Returns:
        ShockFinderResult
    """
    if mach_sampling_extend and not mach_sampling_adaptive:
        raise ValueError("mach_sampling_extend requires mach_sampling_adaptive=True.")
    if mach_velocity_consistency > 0 and not mach_sampling_adaptive:
        raise ValueError("mach_velocity_consistency requires mach_sampling_adaptive=True.")
    if mach_estimator not in ("pressure", "velocity"):
        raise ValueError(f"Unknown mach_estimator {mach_estimator!r}.")
    if mach_estimator == "velocity" and not mach_sampling_adaptive:
        raise ValueError("mach_estimator='velocity' requires mach_sampling_adaptive=True.")
    if mach_sampling_at_shock and not mach_sampling_extend:
        raise ValueError("mach_sampling_at_shock requires mach_sampling_extend=True.")

    pressure = primitive_state[registered_variables.pressure_index]
    density  = primitive_state[registered_variables.density_index]
    r = helper_data.geometric_centers if config.geometry == SPHERICAL else None

    # Phase 1: shock direction (ndim, *spatial_shape)
    shock_direction = _calculate_shock_direction(pressure, density, config, r)

    # Phase 2: shock zones (*spatial_shape). Always along the dominant axis:
    # with zones along the continuous normal too, a compression a few cells
    # upstream left only a one-cell zone gap that the adaptive walk crossed
    # (finder_v2_comparison/README.md, Part 4)
    shock_zones = identify_shock_zones(
        primitive_state, config, registered_variables,
        helper_data, shock_direction, mach_min, gamma_gas,
    )

    # Phase 3: shock surface (*spatial_shape)
    shock_surface = identify_shock_surface(
        primitive_state, shock_zones, shock_direction,
        config, registered_variables, continuous_normal,
    )

    if mach_sampling_adaptive:
        # Phases 4-5 from one adaptive walk (Schaal & Springel 2015 Sec.
        # 2.3.3, optionally extended -- see get_post_pre_shock_values_adaptive):
        # pressure on both sides gives the Mach number, pressure and density
        # at the *same* pre-shock point give the thermal-energy flux, so the
        # two are consistent (FIXES_TODO.md round 24; previously the flux
        # used the fixed 1-cell sampler here). Rays that never leave the
        # zone within mach_sampling_steps are excluded from both.
        # the extension follows the shock's compression (div v < 0) and stops
        # one cell past it -- see get_post_pre_shock_values_adaptive
        compressive = (
            _calculate_velocity_divergence(primitive_state, config, registered_variables, r) < 0
            if mach_sampling_extend else None
        )
        p_post, p_pre, _, rho_pre, exited_post, exited_pre, post_shock_steps, pre_shock_steps = (
            get_post_pre_shock_values_adaptive(
                shock_direction, pressure, density, shock_zones,
                max_steps=mach_sampling_steps, extend_monotone=mach_sampling_extend,
                compressive=compressive, return_sample_steps=True,
                pre_extension_drop_ratio=_PRE_EXTENSION_DROP_RATIO if mach_sampling_at_shock else 0.0,
            )
        )
        if mach_sampling_at_shock:
            # upstream state at the shock cell, not pre_shock_steps ahead of it
            normal = jax.lax.stop_gradient(shock_direction)
            max_slope = -jnp.log(_PRE_EXTENSION_DROP_RATIO)
            p_pre = extrapolate_pre_state_to_shock(pressure, normal, pre_shock_steps, p_pre, max_slope)
            rho_pre = extrapolate_pre_state_to_shock(density, normal, pre_shock_steps, rho_pre, max_slope)
        pre_shock_pressure = p_pre
        pre_shock_density = rho_pre
        valid = shock_surface & exited_post & exited_pre
        mach_pressure = mach_from_pressure_samples(p_post, p_pre, gamma_gas)
        if mach_velocity_consistency > 0:
            valid = valid & _velocity_consistent(
                primitive_state, config, registered_variables, shock_direction,
                mach_velocity_consistency, gamma_gas,
            )
        if mach_estimator == "velocity":
            # normal-velocity jump between the walk's own sample points
            # (+shock_direction points to the pre-shock side)
            # The projection carries no gradient through the normal: -grad T /
            # |grad T| has a singular derivative where |grad T| -> 0 (uniform
            # ambient gas), which a masked-out cell turns into 0 * inf = NaN
            # in a rollout adjoint (as for the walk, FIXES_TODO.md round 24).
            normal = jax.lax.stop_gradient(shock_direction)
            velocity = _velocity_components(primitive_state, config, registered_variables)
            u_pre = sum(sample_along_direction(v, normal, pre_shock_steps) * d
                        for v, d in zip(velocity, normal))
            u_post = sum(sample_along_direction(v, normal, -post_shock_steps) * d
                         for v, d in zip(velocity, normal))
            if mach_sampling_at_shock:
                # end of the shock's own velocity jump, at or before the pressure peak
                u_post = post_velocity_at_end_of_jump(
                    velocity, pressure, normal, post_shock_steps, p_post, u_post,
                    search_steps=min(mach_sampling_steps, _POST_VELOCITY_SEARCH_STEPS),
                    increment_ratio=_PRE_EXTENSION_DROP_RATIO,
                )
            mach_estimate = mach_from_velocity_jump(u_post - u_pre, p_pre, rho_pre, gamma_gas)
            # The converse of mach_velocity_consistency, at the same sample
            # points: a velocity jump without the matching pressure jump is
            # not a shock either. Without this, zone cells whose walk ends in
            # unrelated gas (e.g. a rarefaction behind a shock, or a walk
            # clamped at the domain edge) with the pressure *falling* across
            # them (M_p = 1) but some velocity difference got a velocity Mach
            # number and the dissipated flux of the far side's sound speed --
            # on the B8.2 oblique tubes that drained cells to the pressure
            # floor (NaN). The Sedov shell has M_p ~ 0.87 M_u at 48^3, the
            # colliding winds M_p > M_u.
            pressure_fraction = mach_velocity_consistency if mach_velocity_consistency > 0 else 0.5
            valid = (
                valid
                & (mach_estimate > mach_min)
                & (mach_pressure > mach_min)
                & (mach_pressure - 1.0 >= pressure_fraction * (mach_estimate - 1.0))
            )
        else:
            mach_estimate = mach_pressure
        mach_numbers = jnp.where(valid, mach_estimate, 0.0)
        thermal_energy_flux = jnp.where(
            valid,
            thermal_energy_flux_from_pre_state(mach_numbers, p_pre, rho_pre, gamma_gas),
            0.0,
        )
    else:
        post_shock_steps = None
        pre_shock_steps = None
        pre_shock_pressure = None
        pre_shock_density = None
        # Phase 4: Mach numbers (*spatial_shape), fixed-step sampling
        mach_numbers = _calculate_mach_at_surface(
            primitive_state, shock_surface, shock_direction,
            config, registered_variables, sampling_steps=mach_sampling_steps,
            gamma_gas=gamma_gas,
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
            gamma_gas=gamma_gas,
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
        pre_shock_density=pre_shock_density,
        pre_shock_steps=pre_shock_steps,
    )