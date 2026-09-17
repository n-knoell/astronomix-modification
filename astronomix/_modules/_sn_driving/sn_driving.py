"""
Episodic supernova driving (SILCC-ISM project milestone M3; gap #4 in
``astronomix/_modules/_cosmic_rays_grey/DESIGN.md``'s SILCC-ISM section).

Each step, draws a single Bernoulli trial with probability ``sn_rate * dt``
(the standard thinned-Poisson-process approximation: valid whenever
``sn_rate * dt << 1``, so at most one supernova per step is overwhelmingly the
common case and a second simultaneous trigger is not modeled) and, if it
fires, a uniformly random site in the box. Deposits a smoothly-tapered
spherical energy bump at that site: a thermal part (added to gas pressure,
following ``pytests/shock_finder3D/_sedov_setup.py``'s Sedov-Taylor
normalization) and, when the CR-grey model is active, a direct cosmic-ray
part added to ``e_cr`` (Girichidis et al. 2016's convention -- a fixed
fraction of the total, not routed through the shock-triggered
``inject_crs_at_shocks``/DSA mechanism, per DESIGN.md's design decision #2 for
this project). No mass is injected -- only energy, so a run with no gravity,
cooling or open boundaries conserves total (thermal + kinetic + CR) energy
exactly except for these discrete deposits, which is what
``pytests/stratified_ism/sn_driving_energy_conservation.py`` checks.

Site placement is drawn uniformly at random over the whole box by default
("random" placement, matching Girichidis et al. 2016's own comparison case
and Simpson et al. 2016's "RAND" mode -- see DESIGN.md's SILCC-ISM section).
``SNDrivingConfig.density_weighted_placement`` switches to Simpson et al.
(2016)'s other mode instead: sites drawn from a density-weighted categorical
distribution over eligible cells rather than uniformly (SILCC-ISM project
milestone M6, ladder item 12's optional stretch cross-check) -- see
``_draw_density_weighted_site``'s docstring for the derivation and mechanics.
Either way, the injection footprint wraps periodically along any axis whose
boundary is periodic (the box-scale minimum-image convention), so a site
drawn near a periodic edge still gets its full, undistorted energy deposit.
``[SNDrivingParams.sn_z_min, sn_z_max]`` (default +/-inf, i.e. unrestricted)
restricts site eligibility to that z range under either placement mode --
added for milestone M4's tall, vertically-stratified box, where a real SN
population should track the star-forming layer near the midplane rather than
the whole column's tenuous envelope; see that field's docstring. Under
random placement x/y stay uniform over the full box; under density-weighted
placement all three axes are implicitly restricted to wherever the weighted
distribution actually puts mass.

**Momentum injection (``SNDrivingConfig.momentum_injection``, added for M4):**
at M4's box/resolution, K&I cooling time at a fresh deposit's own post-shock
state (~60 yr) is far shorter than a single hydro dynamical time, so the
thermal deposit above radiates away before it can do any work ("overcooling",
Katz 1992) -- confirmed directly (isolated unit test + an independent
cooling-time calculation, see PROGRESS.md's 2026-09-14 M4 entry), not just a
guess. When this flag is on, the thermal pressure deposit is scaled down to
``sn_momentum_thermal_floor_fraction`` of its normal value (a small nonzero
numerical floor, not exactly zero -- see that field's docstring: a fully
cold cell carrying a large velocity kick makes recovering internal energy
from ``E_total - 0.5*rho*v^2`` a catastrophic-cancellation computation that
reliably NaNs, confirmed directly) and a radial velocity kick over the same
tapered footprint is added alongside it: each affected cell receives
``Δv = v_kick * weight * r_hat``
(``r_hat`` the periodic-wrap-aware unit vector from the site to that cell),
with ``v_kick`` normalized so the mass-weighted radial momentum
``sum(rho_i * weight_i * cell_volume * v_kick)`` equals the Kim & Ostriker
(2015) fitted terminal momentum ``p_terminal = sn_momentum_coefficient *
n_0**(-0.17)`` -- ``n_0`` the *local* (weight-averaged, pre-injection)
ambient density, not some fixed reference, since M4's column spans a wide
density range. Momentum isn't radiated away by cooling, only redistributed,
so it survives where the thermal channel would not. CR-grey's own
``sn_cr_fraction * sn_energy`` deposit is unaffected either way -- e_cr isn't
subject to K&I cooling, so overcooling doesn't apply to that channel. This
is a simplified, always-on momentum injection (no adaptive resolved/
unresolved switch based on comparing the cooling radius to the grid spacing,
unlike the full Kim & Ostriker 2015 hybrid scheme) -- defensible here since
M4's setup is deeply in the unresolved regime throughout, but a future
milestone needing the resolved regime too (a finer box, or a lower-density
target) would need that comparison added.

**Delayed cooling (``SNDrivingConfig.delayed_cooling``, added for M4):** an
alternative to ``momentum_injection`` for the same overcooling finding,
tried instead of it once momentum injection turned out to trigger a third,
unresolved NaN mechanism (see PROGRESS.md's 2026-09-14 M4 entries). Rather
than changing what's injected, this leaves the plain thermal deposit as-is
and instead temporarily switches K&I cooling off within the injection
footprint, via a persistent per-cell ``cooling_shield`` field (a "time
remaining shielded" value, carried through the loop the same way as the OU
forcing field -- see ``LoopState.cooling_shield`` in
``astronomix.time_stepping.time_integration``, and
``_iteration_level_continuous_updates``'s cooling block for how it gates
the update). At trigger, every cell inside the footprint (``weight`` above a
small numerical cutoff) has its shield extended to
``max(current_shield, sn_cooling_delay_time)`` -- the *same*, untapered
duration everywhere in the footprint, deliberately not scaled by the tapered
energy-deposit ``weight``: an earlier tapered-duration version unshielded
the coolest, tapered-edge cells first and left the single hottest cell
shielded longest, so its already-cooled-back-down neighbors made the
eventual unshielding *more* discontinuous, not less (root-caused against a
real M4 run, see PROGRESS.md's 2026-09-15 entry). Every step the shield
decays by ``dt``, floored at 0. This is an Eulerian (grid-fixed), not
Lagrangian (fluid-comoving)
shield -- it does not advect with the flow, unlike a real shocked gas
parcel -- a simplification analogous to ``momentum_injection``'s
not-adaptively-resolved kick: defensible since the shield timescale is
short compared to both the hydro flow crossing a cell and the time between
triggers in M4's regime, but a future milestone with much longer delay
times or a fast-advecting flow would need the shield itself advected.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import PERIODIC_BOUNDARY, STATE_TYPE

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables


def _periodic_delta(delta, box_length: float, is_periodic: bool):
    """Minimum-image displacement along one axis.

    Args:
        delta: Raw (cell - site) displacement along this axis.
        box_length: The box size along this axis.
        is_periodic: Whether this axis is periodic (a static Python bool --
            resolved at trace time, not a traced value).

    Returns:
        ``delta`` unchanged if the axis isn't periodic, else wrapped into
        ``[-box_length / 2, box_length / 2]``.
    """
    if not is_periodic:
        return delta
    return delta - box_length * jnp.round(delta / box_length)


def _draw_density_weighted_site(
    pos_key,
    primitive_state: STATE_TYPE,
    helper_data: HelperData,
    interior_mask,
    z_lo: Union[float, Float[Array, ""]],
    z_hi: Union[float, Float[Array, ""]],
    weighting_power: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
):
    """Draw a trigger's site from a density-weighted categorical
    distribution over eligible cells -- Simpson et al. (2016)'s "density
    peak" SN-placement mode (SILCC-ISM project milestone M6), as an
    alternative to ``_inject_supernovae``'s default uniformly-random site.

    Simpson et al.'s own placement probability is proportional to a local
    star-formation-rate proxy, ``sfr_i ~ m_i / t_ff,i`` (their Sec. 2), with
    the free-fall time ``t_ff ~ rho^-0.5``. On this codebase's fixed-volume
    Cartesian grid, cell mass is simply ``m_i = rho_i * cell_volume``, so
    ``sfr_i ~ rho_i * sqrt(rho_i) = rho_i^1.5`` -- ``cell_volume`` is a
    constant common factor across all cells and drops out of the normalized
    probability, so it never needs to be computed here. ``weighting_power``
    exposes that exponent (``SNDrivingParams.sn_density_weighting_power``,
    default ``1.5``) as a tunable rather than hardcoding it.

    Unlike the uniform mode's continuous draw, this returns an eligible
    cell's own center exactly (categorical sampling over a discrete grid) --
    Simpson et al.'s own SNe are likewise placed at a star-forming *cell*,
    not an arbitrary continuum point.

    Args:
        pos_key: The PRNG key for this draw (already split off the caller's
            key -- not advanced further here).
        primitive_state: The (ghost-padded) primitive state, for density.
        helper_data: The (ghost-padded) helper data, for cell centers.
        interior_mask: Boolean array, ``True`` for real (non-ghost) cells.
        z_lo: Lower z bound of eligibility (already clipped to the domain).
        z_hi: Upper z bound of eligibility (already clipped to the domain).
        weighting_power: Exponent applied to density in the per-cell weight.
        registered_variables: The registered variables.

    Returns:
        The ``(x, y, z)`` center of the drawn cell.
    """
    rho = primitive_state[registered_variables.density_index]
    z = helper_data.geometric_centers[..., 2]
    eligible = interior_mask & (z >= z_lo) & (z <= z_hi)

    # log(rho**power) = power*log(rho), rather than computing rho**power and
    # then logging it -- avoids a spurious overflow for a large power at the
    # (physically irrelevant) high end of this box's density range.
    log_rho = jnp.log(jnp.maximum(rho, 1e-300))
    logits = jnp.where(eligible, weighting_power * log_rho, -jnp.inf)

    flat_index = jax.random.categorical(pos_key, logits.reshape(-1))
    idx = jnp.unravel_index(flat_index, rho.shape)
    return helper_data.geometric_centers[idx[0], idx[1], idx[2], :]


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _inject_supernovae(
    key,
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    cooling_shield=None,
) -> tuple:
    """Stochastically trigger and deposit at most one supernova this step.

    Args:
        key: The PRNG key (advanced and returned; carried in ``LoopState``).
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration (provides ``box_size``,
            ``grid_spacing``, ``boundary_settings``).
        params: The simulation parameters (``params.sn_driving_params`` and
            ``params.gamma``).
        registered_variables: The registered variables.
        helper_data: The helper data (provides ``geometric_centers``, the
            absolute cell-center coordinates -- 3D Cartesian only, matching
            every other point-injection module in this codebase).
        cooling_shield: The persistent per-cell "time remaining shielded"
            field (see ``LoopState.cooling_shield``), or ``None`` when
            ``config.sn_driving_config.delayed_cooling`` is off. Extended at
            a trigger's footprint when that flag is on; passed through
            unchanged otherwise.

    Returns:
        ``(key, primitive_state, cooling_shield)`` with the advanced PRNG
        key, the supernova (if triggered) deposited, and the (possibly
        extended) cooling shield field.
    """
    sn_params = params.sn_driving_params

    key, trigger_key, pos_key = jax.random.split(key, 3)

    trigger_probability = jnp.clip(sn_params.sn_rate * dt, 0.0, 1.0)
    triggered = jax.random.bernoulli(trigger_key, trigger_probability)

    box_size = config.box_size

    # helper_data/primitive_state here are ghost-padded (this function is
    # called from _iteration_level_injections with helper_data_pad) --
    # needed up front now, both to keep the density-weighted site draw below
    # from ever landing on a ghost cell and, further down, to restrict the
    # deposit weight itself to the interior (see that comment for why: a
    # ghost cell's contribution would otherwise double-count real domain
    # volume across a periodic wrap).
    ngc = config.num_ghost_cells
    num_cells = config.num_cells
    density_shape = primitive_state[registered_variables.density_index].shape
    interior_x = (jnp.arange(density_shape[0]) >= ngc) & (
        jnp.arange(density_shape[0]) < num_cells.x + ngc
    )
    interior_y = (jnp.arange(density_shape[1]) >= ngc) & (
        jnp.arange(density_shape[1]) < num_cells.y + ngc
    )
    interior_z = (jnp.arange(density_shape[2]) >= ngc) & (
        jnp.arange(density_shape[2]) < num_cells.z + ngc
    )
    interior_mask = (
        interior_x[:, None, None] & interior_y[None, :, None] & interior_z[None, None, :]
    )

    # z is restricted to [sn_z_min, sn_z_max] (clipped to the domain). At the
    # default +/-inf bounds this reduces to the unrestricted formula
    # bit-for-bit -- see SNDrivingParams.sn_z_min/sn_z_max's docstring.
    z_lo = jnp.maximum(sn_params.sn_z_min, 0.0)
    z_hi = jnp.minimum(sn_params.sn_z_max, box_size.z)

    if config.sn_driving_config.density_weighted_placement:
        site = _draw_density_weighted_site(
            pos_key, primitive_state, helper_data, interior_mask, z_lo, z_hi,
            sn_params.sn_density_weighting_power, registered_variables,
        )
    else:
        # x/y stay uniform over the whole box.
        unit_site = jax.random.uniform(pos_key, shape=(3,))
        site = jnp.array([
            unit_site[0] * box_size.x,
            unit_site[1] * box_size.y,
            unit_site[2] * (z_hi - z_lo) + z_lo,
        ])

    delta = helper_data.geometric_centers - site
    boundary_settings = config.boundary_settings
    dx = _periodic_delta(
        delta[..., 0], box_size.x,
        boundary_settings.x.left_boundary == PERIODIC_BOUNDARY,
    )
    dy = _periodic_delta(
        delta[..., 1], box_size.y,
        boundary_settings.y.left_boundary == PERIODIC_BOUNDARY,
    )
    dz = _periodic_delta(
        delta[..., 2], box_size.z,
        boundary_settings.z.left_boundary == PERIODIC_BOUNDARY,
    )
    distance = jnp.sqrt(dx**2 + dy**2 + dz**2)

    smooth_width = sn_params.sn_smooth_cells * config.grid_spacing
    weight = 0.5 * (
        1.0 - jnp.tanh((distance - sn_params.sn_injection_radius) / smooth_width)
    )

    # Restrict the weight -- both the normalization sum and the actual
    # deposit -- to the interior cells only (interior_mask computed above): a
    # ghost cell's contribution would otherwise double-count real domain
    # volume across a periodic wrap (a site near an edge is close, in raw
    # coordinates, to both the true interior cells on the far side *and* the
    # near-side ghost cells that mirror them) and then vanish when the next
    # boundary-condition application overwrites the ghost cells from the
    # interior, silently losing that fraction of sn_energy. The interior
    # deposit alone is exactly one cell per true domain volume element, so
    # the boundary handler then correctly refreshes the ghost cells (whatever
    # each axis's BC is) from the updated interior before any flux
    # computation reads them.
    weight = weight * interior_mask

    cell_volume = config.grid_spacing**3
    # Renormalize so the deposited energy is exactly sn_energy regardless of
    # resolution or how the footprint straddles the periodic wrap (mirrors
    # _sedov_setup.py's delta_p normalization).
    weight_integral_safe = jnp.maximum(jnp.sum(weight) * cell_volume, 1e-300)

    cr_fraction = (
        sn_params.sn_cr_fraction if registered_variables.cosmic_ray_e_active else 0.0
    )
    cr_energy = cr_fraction * sn_params.sn_energy

    # Thermal deposit: the terminal-momentum kick below stands in for most
    # of it when momentum_injection is on (see SNDrivingConfig.
    # momentum_injection's docstring for why: pure-thermal injection
    # overcools before it can do work, at this codebase's typical
    # resolution) -- but a small sn_momentum_thermal_floor_fraction of it is
    # still deposited, a numerical stabilization against the catastrophic
    # cancellation a zero-thermal, large-kinetic-energy cold cell produces
    # when recovering internal energy from E_total - 0.5*rho*v^2 (see that
    # field's docstring). CR energy is unaffected either way -- it isn't
    # subject to K&I thermal cooling, so overcooling doesn't apply to that
    # channel.
    # thermal_fraction is a traced params value (sn_momentum_thermal_floor_
    # fraction can be tuned/differentiated), so this can't be a Python `if`
    # guard -- it's applied unconditionally; a fraction of exactly 0.0
    # naturally makes delta_pressure zero everywhere, a harmless no-op add.
    thermal_fraction = (
        sn_params.sn_momentum_thermal_floor_fraction
        if config.sn_driving_config.momentum_injection else 1.0
    )
    thermal_energy = thermal_fraction * (1.0 - cr_fraction) * sn_params.sn_energy
    gamma = params.gamma
    delta_pressure = jnp.where(
        triggered, thermal_energy * (gamma - 1.0) / weight_integral_safe, 0.0
    ) * weight
    primitive_state = primitive_state.at[registered_variables.pressure_index].add(
        delta_pressure
    )

    if registered_variables.cosmic_ray_e_active:
        delta_e_cr = jnp.where(
            triggered, cr_energy / weight_integral_safe, 0.0
        ) * weight
        primitive_state = primitive_state.at[
            registered_variables.cosmic_ray_e_index
        ].add(delta_e_cr)

    # Terminal-momentum injection (Kim & Ostriker 2015): a radial velocity
    # kick over the same tapered footprint, normalized so the mass-weighted
    # radial momentum sums to the fitted terminal momentum p_terminal =
    # sn_momentum_coefficient * n_0**(-0.17), n_0 the *local* (weight-
    # averaged, pre-injection) ambient density in units of
    # sn_momentum_density_reference. Unlike thermal energy, momentum isn't
    # radiated away by cooling, so it survives even when the thermal channel
    # would overcool -- see SNDrivingConfig.momentum_injection's docstring.
    if config.sn_driving_config.momentum_injection:
        rho = primitive_state[registered_variables.density_index]
        weight_sum_safe = jnp.maximum(jnp.sum(weight), 1e-300)
        rho_ambient_local = jnp.sum(rho * weight) / weight_sum_safe
        n_0 = rho_ambient_local / sn_params.sn_momentum_density_reference
        p_terminal = jnp.where(
            triggered, sn_params.sn_momentum_coefficient * n_0 ** (-0.17), 0.0,
        )

        # Radial unit vector from the site to each cell, reusing dx/dy/dz's
        # already periodic-wrapped displacements; floored to avoid 0/0 at
        # the site cell itself (same smooth-floor philosophy as
        # cr_pressure_speed_floor/b_field_floor elsewhere in this codebase).
        distance_safe = jnp.maximum(distance, 1e-6 * config.grid_spacing)
        r_hat_x = dx / distance_safe
        r_hat_y = dy / distance_safe
        r_hat_z = dz / distance_safe

        mass_weight_sum_safe = jnp.maximum(jnp.sum(rho * weight) * cell_volume, 1e-300)
        v_kick = p_terminal / mass_weight_sum_safe
        primitive_state = primitive_state.at[registered_variables.velocity_index.x].add(
            v_kick * weight * r_hat_x
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.y].add(
            v_kick * weight * r_hat_y
        )
        primitive_state = primitive_state.at[registered_variables.velocity_index.z].add(
            v_kick * weight * r_hat_z
        )

    # Delayed cooling: extend every cell inside this footprint's shield to
    # the *same*, untapered sn_cooling_delay_time wherever that's longer than
    # whatever shield already remains there (an overlapping earlier
    # trigger's shield is never shortened). See SNDrivingConfig.
    # delayed_cooling's docstring -- duration is deliberately NOT tapered by
    # `weight` (unlike the energy deposit above): an earlier tapered-duration
    # version unshielded the coolest, tapered-edge cells first and left the
    # single hottest cell shielded longest, so by the time that cell's shield
    # finally lifted its already-unshielded neighbors had cooled back to
    # their local K&I equilibrium, producing a sharper, less-resolved
    # discontinuity than the original injection (root-caused against a real
    # M4 run, see PROGRESS.md's 2026-09-15 entry). A uniform duration means
    # the whole footprint unshields together, at comparable temperatures.
    if config.sn_driving_config.delayed_cooling:
        triggered_delay = jnp.where(triggered, sn_params.sn_cooling_delay_time, 0.0)
        # A raw tanh weight never reaches exact 0 far from the site (just an
        # exponentially small tail) -- but _iteration_level_continuous_
        # updates gates cooling with a hard `cooling_shield > 0` boolean, so
        # an un-floored, arbitrarily tiny tail would spuriously suppress
        # cooling for one step across the *entire* domain, not just near the
        # footprint (found via this module's own unit test, see PROGRESS.md).
        # Use the same cutoff to define the footprint's extent, but only as a
        # boolean mask now -- the shield duration itself is uniform within it.
        footprint_mask = weight > 1e-3
        cooling_shield = jnp.maximum(
            cooling_shield, jnp.where(footprint_mask, triggered_delay, 0.0)
        )

    return key, primitive_state, cooling_shield
