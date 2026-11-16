# ============================================================================
# PHASE 2: SHOCK ZONE IDENTIFICATION
# ============================================================================

from functools import partial
import jax.numpy as jnp
import jax
import jax.scipy.ndimage

from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE, BOOL_FIELD_TYPE,
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)
from astronomix.shock_finder3D._gradients import (
    _calculate_velocity_divergence,
    _calculate_temperature_gradient,
    _calculate_density_gradient,
)


"""
Criterion 1: Converging flow (∇·v < 0).
"""
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _shock_zone_criterion_converging_flow(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    r: FIELD_TYPE = None,
) -> BOOL_FIELD_TYPE:
    div_v = _calculate_velocity_divergence(primitive_state, config, registered_variables, r)
    return div_v < 0


"""
Criterion 2: Aligned gradients (∇T · ∇ρ > 0).
"""
@partial(jax.jit, static_argnames=["config"])
def _shock_zone_criterion_aligned_gradients(
    pressure: FIELD_TYPE,
    density: FIELD_TYPE,
    config: SimulationConfig,
    r: FIELD_TYPE = None,
) -> BOOL_FIELD_TYPE:
    grad_T   = _calculate_temperature_gradient(pressure, density, config, r)
    grad_rho = _calculate_density_gradient(density, config, r)

    # dot product over the ndim axis (axis=0 of the vector fields)
    dot_product = jnp.sum(grad_T * grad_rho, axis=0)
    return dot_product > 0


"""
Criterion 3: Minimum Mach number
* pick minimum Mach number
* For each cell, 
    look at the two neighbors along the shock direction (one on each side), 
    compute the pressure and temperature jumps across them, 
    -> get_post_pre_shock_values

    and check if those jumps are large enough to correspond to a shock of at least Mach mach_min
"""
def get_post_pre_shock_values(
    shock_direction,
    field_a,
    field_b,
    max_steps=1,
):
    """
    Sample two scalar fields on both sides of a candidate shock.

    The shock direction points from the hot/post-shock side toward the
    cold/pre-shock side. For every cell, the dominant component of the
    shock-direction vector determines which grid axis is used for sampling.

    Args:
        shock_direction:
            Unit-vector field with shape (ndim, *spatial_shape).

        field_a:
            First scalar field to sample, for example pressure.

        field_b:
            Second scalar field to sample, for example temperature or density.

        max_steps:
            Number of grid cells to move away from the candidate shock cell.

    Returns:
        field_a_post:
            field_a sampled on the post-shock side.

        field_a_pre:
            field_a sampled on the pre-shock side.

        field_b_post:
            field_b sampled on the post-shock side.

        field_b_pre:
            field_b sampled on the pre-shock side.
    """

    dominant_axis = jnp.argmax(
        jnp.abs(shock_direction),
        axis=0,
    )

    dominant_direction = jnp.take_along_axis(
        shock_direction,
        dominant_axis[jnp.newaxis],
        axis=0,
    )[0]

    step_sign = jnp.sign(
        dominant_direction
    ).astype(jnp.int32)

    ndim = field_a.ndim

    def shift_field(field, shift, axis):
        """
        Move a scalar field by a fixed number of cells.

        Note:
            jnp.roll wraps around at domain boundaries. Boundary cells must
            therefore be masked elsewhere before the sampled values are used.
        """

        shifted_field = field

        for _ in range(max_steps):
            shifted_field = jnp.roll(
                shifted_field,
                shift=shift,
                axis=axis,
            )

        return shifted_field

    # Default values are the original cell values. They are replaced only
    # along the locally selected dominant shock axis.
    field_a_post = field_a
    field_a_pre = field_a
    field_b_post = field_b
    field_b_pre = field_b

    for axis in range(ndim):
        uses_this_axis = dominant_axis == axis

        points_in_positive_direction = (
            uses_this_axis
            & (step_sign > 0)
        )

        points_in_negative_direction = (
            uses_this_axis
            & (step_sign < 0)
        )

        # If the shock direction points in the positive axis direction,
        # the pre-shock gas is ahead (+axis), while the post-shock gas is
        # behind (-axis).
        field_a_post_positive = shift_field(field_a, +1, axis)
        field_a_pre_positive  = shift_field(field_a, -1, axis)

        field_b_post_positive = shift_field(field_b, +1, axis)
        field_b_pre_positive  = shift_field(field_b, -1, axis)

        # Reverse the sampling sides if the shock direction points
        # in the negative axis direction.
        field_a_post_negative = shift_field(field_a, -1, axis)
        field_a_pre_negative  = shift_field(field_a, +1, axis)

        field_b_post_negative = shift_field(field_b, -1, axis)
        field_b_pre_negative  = shift_field(field_b, +1, axis)

        field_a_post = jnp.where(
            points_in_positive_direction,
            field_a_post_positive,
            jnp.where(
                points_in_negative_direction,
                field_a_post_negative,
                field_a_post,
            ),
        )

        field_a_pre = jnp.where(
            points_in_positive_direction,
            field_a_pre_positive,
            jnp.where(
                points_in_negative_direction,
                field_a_pre_negative,
                field_a_pre,
            ),
        )

        field_b_post = jnp.where(
            points_in_positive_direction,
            field_b_post_positive,
            jnp.where(
                points_in_negative_direction,
                field_b_post_negative,
                field_b_post,
            ),
        )

        field_b_pre = jnp.where(
            points_in_positive_direction,
            field_b_pre_positive,
            jnp.where(
                points_in_negative_direction,
                field_b_pre_negative,
                field_b_pre,
            ),
        )

    return (
        field_a_post,
        field_a_pre,
        field_b_post,
        field_b_pre,
    )

@partial(jax.jit, static_argnames=["max_steps", "extend_monotone", "return_sample_steps",
                                   "pre_extension_drop_ratio"])
def get_post_pre_shock_values_adaptive(
    shock_direction,
    field_a,
    field_b,
    shock_zones,
    max_steps=15,
    extend_monotone=False,
    extension_tolerance=0.02,
    compressive=None,
    return_sample_steps=False,
    pre_extension_drop_ratio=0.0,
):
    """
    Sample two scalar fields on both sides of a candidate shock, walking each
    ray until it leaves the shock zone rather than a fixed number of cells.

    This is the shock-surface construction method of Schaal & Springel (2015,
    MNRAS 446, 3992, Sec. 2.3.3, https://arxiv.org/pdf/1407.4117), itself
    building on Ryu et al. (2003) and Skillman et al. (2008): "rays are sent
    from each cell of the shock zone in the direction of the post-shock
    region... When the first cell outside of the shock zone is reached, the
    post-shock temperature is recorded". The zone (``shock_zones``, already
    computed by ``identify_shock_zones``) has a physically motivated,
    per-cell-varying width (Ryu et al. 2003's two-step "zone then surface"
    idea, ~3-4 cells typically) -- walking until leaving it, rather than a
    fixed ``max_steps``, ties the sampling distance to that actual local
    width instead of a single global hyperparameter (see FIXES_TODO.md item
    1b: a fixed 1-cell offset samples inside the transition whenever the zone
    is wider than that, and any *other* fixed offset either does the same for
    a wider zone, or overshoots into an unrelated feature for a narrower one).

    Two further departures from ``get_post_pre_shock_values``, both aimed at
    the specific failure mode item 1a's (reverted) attempt at this same idea
    hit in a real simulation:

    1. Walks along the **continuous local shock-normal direction**
       (trilinear interpolation via ``jax.scipy.ndimage.map_coordinates``),
       not the single dominant *grid axis* -- addressing the same
       grid-orientation dependence Skillman et al. (2008) already flagged in
       the coordinate-splitting approach of Ryu et al. (2003), and the
       resulting drift onto an unrelated part of a curved shock surface
       documented in FIXES_TODO.md round 10.
    2. Uses ``mode="nearest"`` (clamps at the domain boundary) instead of
       ``get_post_pre_shock_values``'s ``jnp.roll`` (wraps around) --
       structurally eliminating the wraparound-across-a-boundary mechanism
       item 1a's regression was suspected (never confirmed) to hinge on,
       rather than patching around it with a margin mask afterward.

    Args:
        shock_direction: Unit-vector field, shape (ndim, *spatial_shape).
        field_a: First scalar field to sample (e.g. pressure).
        field_b: Second scalar field to sample (e.g. temperature).
        shock_zones: Boolean shock-zone mask, shape (*spatial_shape), from
            ``identify_shock_zones`` -- the walk for a given origin cell
            stops the first time it steps outside this mask.
        max_steps: Safety cap on how far a ray can walk before giving up (a
            ray that never leaves the zone within this many steps falls back
            to its value at ``max_steps`, rather than walking indefinitely).
        extend_monotone: Opt-in (default False). The zone exit can still lie
            inside the numerically smeared shock (the zone criteria stop
            holding before the jump is complete -- e.g. a Mach ~190 CWB
            shock spread over 3 cells, FIXES_TODO.md round 22). When on, a
            ray keeps walking past the exit while ``field_a`` keeps moving
            away from the shock monotonically -- falling by more than
            ``extension_tolerance`` (relative) per step on the pre-shock
            side, rising by more than that on the post-shock side -- and
            samples the last cell of that monotone run, i.e. the start of
            the upstream/downstream plateau. On the post-shock side the
            sample is the peak of ``field_a`` along the whole walk instead
            (the same point for an isolated shock; the right one when a thin
            shocked layer shares its zone with a second shock behind it, where
            the zone exit lies beyond that second shock). Still capped by
            ``max_steps``.
        extension_tolerance: Minimum relative change per step for the
            monotone extension to continue (only with ``extend_monotone``).
        compressive: Optional boolean field (``div v < 0``). When given, the
            post-side monotone extension only steps on from a cell that is
            itself compressive, i.e. it follows the shock's compression and at
            most one cell beyond it. Without this, a post-side ray in a young
            blast (pressure rising all the way to the centre of the expanding
            interior) walks to the centre and overestimates the Mach number
            (FIXES_TODO.md round 24). Not applied on the pre side: there the
            foot of a smeared shock in a diverging upstream flow (e.g. a
            stellar wind) is already div v > 0, so the gate would stop one
            cell short of the plateau; the pre-side walk ends by itself where
            the upstream pressure flattens.
        return_sample_steps: Also return ``(steps_post, steps_pre)``: how many
            steps along the ray each side's sample lies from the origin cell
            (float fields; ``max_steps`` for rays that never left the zone).
            Used to spread DSA injection over the post-shock cells.
        pre_extension_drop_ratio: Opt-in (default 0, off), only with
            ``extend_monotone``: the pre-shock extension additionally stops
            unless this step's (linear) drop of ``field_a`` is below this
            fraction of the previous step's drop. The foot of a numerically
            smeared shock is convex (the drops shrink step by step, by
            exp(-2/w) = 0.14 / 0.37 / 0.51 for a tanh of width w = 1 / 2 / 3
            cells), a resolved background gradient is not (exp(-1/H) > 0.8
            for scale heights H >= 5 cells). Without it the 2% rule keeps
            walking down any background gradient steeper than 2% per cell, to
            the ``max_steps`` cap (dissipated flux 0.69 / 0.48 / 0.23 of
            exact at H = 40 / 20 / 10 cells). The sample then still lies a
            few cells upstream of the shock: combine with
            ``extrapolate_pre_state_to_shock``. 0.7 in the finder
            (``find_shocks_pfrommer(mach_sampling_at_shock=True)``).

    Returns:
        ``(field_a_post, field_a_pre, field_b_post, field_b_pre, exited_post,
        exited_pre)``: the sampled fields on each side (as in
        ``get_post_pre_shock_values``), plus boolean fields marking which
        origin cells' rays actually found a zone exit within ``max_steps``
        (as opposed to falling back to the ``max_steps`` cap) -- callers can
        use these to exclude non-converged rays instead of trusting an
        arbitrary cutoff value.
    """
    ndim = field_a.ndim
    shape = field_a.shape
    dtype = shock_direction.dtype

    # Where to sample is a geometric choice, not something to differentiate:
    # shock_direction = -grad T / |grad T| has a singular derivative wherever
    # |grad T| -> 0 (e.g. uniform ambient gas), and map_coordinates would
    # otherwise propagate the sampled fields' gradient into the sample
    # positions -- NaN through a live-rollout adjoint (FIXES_TODO.md round 24,
    # Phase D). Gradients still flow through the sampled field values.
    shock_direction = jax.lax.stop_gradient(shock_direction)

    base_coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(n, dtype=dtype) for n in shape], indexing="ij"),
        axis=0,
    )
    zones_f = shock_zones.astype(dtype)

    def _sample(field, coords):
        return jax.scipy.ndimage.map_coordinates(field, coords, order=1, mode="nearest")

    def _sample_mask(mask_f, coords):
        return jax.scipy.ndimage.map_coordinates(mask_f, coords, order=0, mode="nearest") > 0.5

    def _sample_zone(coords):
        return _sample_mask(zones_f, coords)

    compressive_f = compressive.astype(dtype) if compressive is not None else None

    def _walk_block(sign, base_blk, direction_blk, a0_blk, b0_blk, zones0_blk, compressive0_blk):
        # One block of origin cells (all *_blk arrays have the block's shape,
        # vector fields with a leading component axis); the sampled fields
        # themselves are always the full grids.
        #
        # jax.lax.fori_loop, not a Python-unrolled loop: the loop body is
        # traced/compiled once and executed max_steps times by XLA's own
        # While op, instead of max_steps duplicated copies of the body being
        # chained into one giant graph. The unrolled version compiled and
        # ran fine at N=64 but OOM'd at N=256 (each of the ~15 unrolled
        # steps materializes several full-grid intermediates at once) --
        # this is a real resource-usage fix, not just a style preference.
        def body_fun(step, carry):
            (a_val, b_val, exited_ever, still_inside, extending, a_peak, b_peak,
             previous_compressive, val_step, peak_step, a_previous, previous_drop) = carry
            step = step.astype(dtype)
            coords = base_blk + sign * step * direction_blk
            coords_list = [coords[d] for d in range(ndim)]
            inside_here = _sample_zone(coords_list)
            newly_exited = still_inside & (~inside_here) & (~exited_ever)

            a_here = _sample(field_a, coords_list)
            b_here = _sample(field_b, coords_list)

            if extend_monotone and compressive is not None and sign < 0:
                # post side: continue past the exit only from a compressive cell
                extending = extending & previous_compressive

            if extend_monotone and sign < 0:
                # Post side: running peak of field_a over the whole walk (zone,
                # exit cell, monotone extension). A thin shocked layer can
                # share one zone with the next shock behind it, so the zone
                # exit may lie beyond that second shock; the peak is the
                # post-shock state either way.
                walking = ~exited_ever | extending
                higher = walking & (a_here > a_peak)
                a_peak = jnp.where(higher, a_here, a_peak)
                b_peak = jnp.where(higher, b_here, b_peak)
                peak_step = jnp.where(higher, step, peak_step)

            if extend_monotone:
                # Past the exit: keep going while field_a still moves away
                # from the shock (down on the pre side, up on the post side).
                if sign > 0:
                    monotone = a_here < a_val * (1.0 - extension_tolerance)
                    if pre_extension_drop_ratio > 0:
                        # still in the (convex) shock foot, not on a background gradient
                        monotone = monotone & (
                            (a_previous - a_here) < pre_extension_drop_ratio * previous_drop
                        )
                else:
                    monotone = a_here > a_val * (1.0 + extension_tolerance)
                advance = extending & monotone
                a_val = jnp.where(advance, a_here, a_val)
                b_val = jnp.where(advance, b_here, b_val)
                val_step = jnp.where(advance, step, val_step)
                extending = advance | newly_exited

            a_val = jnp.where(newly_exited, a_here, a_val)
            b_val = jnp.where(newly_exited, b_here, b_val)
            val_step = jnp.where(newly_exited, step, val_step)

            exited_ever = exited_ever | newly_exited
            if compressive is not None:
                previous_compressive = _sample_mask(compressive_f, coords_list)
            return (a_val, b_val, exited_ever, inside_here, extending, a_peak, b_peak,
                    previous_compressive, val_step, peak_step, a_here, a_previous - a_here)

        block_shape = a0_blk.shape
        no_cells = jnp.zeros(block_shape, dtype=jnp.bool_)
        no_steps = jnp.zeros(block_shape, dtype=dtype)
        init_carry = (a0_blk, b0_blk, no_cells, zones0_blk, no_cells, a0_blk, b0_blk,
                      compressive0_blk, no_steps, no_steps, a0_blk,
                      jnp.full(block_shape, jnp.inf, dtype=a0_blk.dtype))
        a_val, b_val, exited_ever, _, _, a_peak, b_peak, _, val_step, peak_step, _, _ = (
            jax.lax.fori_loop(1, max_steps + 1, body_fun, init_carry)
        )
        if extend_monotone and sign < 0:
            a_val, b_val, val_step = a_peak, b_peak, peak_step

        # Rays still inside the zone after max_steps: fall back to the value
        # at the cap rather than extrapolating further (matches the safety
        # bound _energy_dissipation.py/_shock_mach.py already rely on for
        # the fixed-step walk's own max_steps).
        coords_final = base_blk + sign * max_steps * direction_blk
        coords_final_list = [coords_final[d] for d in range(ndim)]
        a_val = jnp.where(exited_ever, a_val, _sample(field_a, coords_final_list))
        b_val = jnp.where(exited_ever, b_val, _sample(field_b, coords_final_list))
        val_step = jnp.where(exited_ever, val_step, jnp.asarray(max_steps, dtype))

        return a_val, b_val, exited_ever, val_step

    # Origin cells are processed in slabs along the first axis when the grid
    # is large: every ray only reads the full fields, so this is exact, and
    # it bounds the walk's temporary memory (the per-step trilinear gathers
    # over all origin cells at once did not fit a 256^3 float64 run next to
    # the simulation on an 11 GB GPU -- FIXES_TODO.md round 24).
    compressive0 = compressive if compressive is not None else jnp.zeros(shape, jnp.bool_)
    num_blocks = _num_origin_blocks(shape)

    def _walk(sign):
        if num_blocks == 1:
            return _walk_block(sign, base_coords, shock_direction, field_a, field_b,
                               shock_zones, compressive0)
        rows = shape[0] // num_blocks

        def scalar_blocks(x):
            return x.reshape((num_blocks, rows) + shape[1:])

        def vector_blocks(x):
            return jnp.moveaxis(x.reshape((ndim, num_blocks, rows) + shape[1:]), 1, 0)

        blocks = (vector_blocks(base_coords), vector_blocks(shock_direction),
                  scalar_blocks(field_a), scalar_blocks(field_b),
                  scalar_blocks(shock_zones), scalar_blocks(compressive0))
        outs = jax.lax.map(lambda blk: _walk_block(sign, *blk), blocks)
        return tuple(o.reshape(shape) for o in outs)

    # +shock_direction points toward the pre-shock (cold) side, -shock_direction
    # toward the post-shock (hot) side (see get_post_pre_shock_values above).
    field_a_pre, field_b_pre, exited_pre, steps_pre = _walk(+1)
    field_a_post, field_b_post, exited_post, steps_post = _walk(-1)

    outputs = (
        field_a_post,
        field_a_pre,
        field_b_post,
        field_b_pre,
        exited_post,
        exited_pre,
    )
    if return_sample_steps:
        outputs = outputs + (steps_post, steps_pre)
    return outputs


def sample_along_direction(field, shock_direction, steps, order=1):
    """Sample ``field`` at ``cell + steps * shock_direction`` for every cell
    (trilinear, ``mode="nearest"``), i.e. at the points the adaptive walk of
    ``get_post_pre_shock_values_adaptive`` sampled when ``steps`` is one of
    its ``return_sample_steps`` outputs (negative for the post-shock side).
    Lets callers read further fields at the walk's sample points without
    repeating the walk. The sample positions carry no gradient (as in the
    walk). ``order=0`` reads the nearest cell instead of interpolating."""
    shape = field.shape
    base = jnp.stack(
        jnp.meshgrid(*[jnp.arange(n, dtype=shock_direction.dtype) for n in shape], indexing="ij"),
        axis=0,
    )
    coords = jax.lax.stop_gradient(base + steps[None] * shock_direction)
    return jax.scipy.ndimage.map_coordinates(
        field, [coords[d] for d in range(field.ndim)], order=order, mode="nearest"
    )


def extrapolate_pre_state_to_shock(field, shock_direction, pre_steps, pre_value, max_slope):
    """Pre-shock value of ``field`` extrapolated from the walk's pre-shock
    sample back to the origin (shock) cell.

    The pre-shock sample of ``get_post_pre_shock_values_adaptive`` lies
    ``pre_steps`` (typically 2-4) cells upstream of the shock cell. In a
    background gradient that is not the state the shock runs into: the density
    of an r^-2 wind is 22-33% higher there than at its termination shock
    (colliding winds, 128^3-192^3), and in an exponential atmosphere of scale
    height 10-20 cells the dissipated flux came out 0.55-0.83 of exact even
    with the walk stopped at the end of the foot. Log-linear extrapolation
    ``pre_value * exp(pre_steps * s)`` with the slope per step ``s`` taken
    beyond the sample, where the shock has not touched the gas: the minmod of
    the two one-step log-slopes between ``pre_steps``, ``+1`` and ``+2`` (zero
    where they disagree in sign: a single steep step, e.g. a contact or the
    last bit of the foot, does not extrapolate), limited to ``max_slope`` in
    magnitude. A rising result is capped at the origin cell's own value (the
    smeared shock only adds to the upstream profile). Flat upstream gas is
    returned unchanged. The sample positions carry no gradient.
    """
    floor = 1e-30
    log_0 = jnp.log(jnp.maximum(pre_value, floor))
    log_1 = jnp.log(jnp.maximum(sample_along_direction(field, shock_direction, pre_steps + 1.0), floor))
    log_2 = jnp.log(jnp.maximum(sample_along_direction(field, shock_direction, pre_steps + 2.0), floor))
    slope_near = log_0 - log_1
    slope_far = log_1 - log_2
    slope = jnp.where(
        slope_near * slope_far > 0,
        jnp.sign(slope_near) * jnp.minimum(jnp.abs(slope_near), jnp.abs(slope_far)),
        0.0,
    )
    slope = jnp.clip(slope, -max_slope, max_slope)
    extrapolated = pre_value * jnp.exp(pre_steps * slope)
    return jnp.minimum(extrapolated, jnp.maximum(pre_value, field))


@partial(jax.jit, static_argnames=["search_steps"])
def post_velocity_at_end_of_jump(
    velocity, pressure, shock_direction, post_steps, post_pressure, post_velocity,
    search_steps=6, plateau_fraction=0.9, increment_ratio=0.7,
):
    """Post-shock normal velocity at the end of the shock's own velocity jump.

    The pressure peak along the post-shock walk (``post_steps``, the walk's
    post-shock sample) is where the downstream *pressure* is best read, but
    the velocity there can contain more than the shock's jump: behind a
    wind's termination shock the flow keeps decelerating, and the peak can
    lie one or more cells into that flow.

    The velocity is read at the first step k, from the front edge of the
    pressure plateau on (first step with ``pressure >= plateau_fraction *
    post_pressure``) and at most the peak step, at which the jump stops
    being the convex approach to a plateau: the next step reverses, or it
    adds at least ``increment_ratio`` of what the step before added (the
    increments no longer shrink: a background gradient; the counterpart of
    the walk's ``pre_extension_drop_ratio``). If neither happens, or the peak
    lies more than ``search_steps`` steps away, ``post_velocity`` (the value
    at the peak) is kept.

    Measured (2026-11-16): no change on planar tubes and on the Sedov shell
    (Mach numbers to 1e-4); colliding winds at 192^3: dissipated flux -2% /
    -5% (wind 1 / 2), apex Mach number / Chevalier & Clegg 1.14 / 1.17 ->
    1.10 / 1.13. Rejected variants: "first step where the velocity reverses,
    else the front edge" reads planar shocks 2.5-3% low in Mach number
    (injection -8.5%); ending the jump where a step adds less than 2% / 0.5%
    of it reads 1% / 0.1% low (the latter: KR13 efficiency -1% at M 2.5-3);
    letting the jump end one step past the peak reads planar KR13 shocks
    0.6% high (injection +2%). The stagnating layer between two colliding
    winds (2-3 cells wide, velocity changing through the pressure peak) is
    not recognised.

    Args:
        velocity: List of velocity component fields.
        pressure: Gas pressure field.
        shock_direction: Unit normal, pointing to the pre-shock side.
        post_steps: Steps to the post-shock (pressure-peak) sample.
        post_pressure: Pressure at that sample.
        post_velocity: Normal velocity at that sample.
        search_steps: Largest peak step the search covers (static).
        plateau_fraction: Pressure fraction that defines the plateau's front edge.
        increment_ratio: Ratio of successive velocity increments from which
            on the profile is no longer the shock's.

    Returns:
        Normal velocity (along ``shock_direction``) at the chosen step.
    """
    shape = pressure.shape
    ndim = pressure.ndim
    dtype = shock_direction.dtype
    shock_direction = jax.lax.stop_gradient(shock_direction)
    base_coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(n, dtype=dtype) for n in shape], indexing="ij"),
        axis=0,
    )

    def _sample(field, coords):
        return jax.scipy.ndimage.map_coordinates(field, coords, order=1, mode="nearest")

    def _block(base_blk, direction_blk, post_steps_blk, post_pressure_blk, post_velocity_blk):
        peak_step = jnp.maximum(jnp.ceil(post_steps_blk), 1.0)

        def normal_velocity(coords):
            return sum(_sample(v, coords) * direction_blk[d] for d, v in enumerate(velocity))

        def body_fun(step, carry):
            u_previous, u_before, at_edge, ended, u_end = carry
            step = step.astype(dtype)
            coords = base_blk - step * direction_blk
            coords = [coords[d] for d in range(ndim)]
            p_here = _sample(pressure, coords)
            u_here = normal_velocity(coords)
            # +shock_direction points upstream: across the shock the normal
            # velocity rises towards the post-shock side. Did the jump end at
            # the previous step?
            increment = u_here - u_previous
            previous_increment = u_previous - u_before
            end_here = (
                at_edge & ~ended & (step - 1.0 <= peak_step)
                & ((increment < 0.0) | (increment >= increment_ratio * previous_increment))
            )
            u_end = jnp.where(end_here, u_previous, u_end)
            ended = ended | end_here
            at_edge = at_edge | (p_here >= plateau_fraction * post_pressure_blk) | (step >= peak_step)
            return u_here, u_previous, at_edge, ended, u_end

        u_origin = normal_velocity([base_blk[d] for d in range(ndim)])
        no_cells = jnp.zeros(u_origin.shape, dtype=jnp.bool_)
        init = (u_origin, u_origin, no_cells, no_cells, u_origin)
        _, _, _, ended, u_end = jax.lax.fori_loop(1, search_steps + 2, body_fun, init)
        return jnp.where(ended & (peak_step <= search_steps), u_end, post_velocity_blk)

    # slabs of origin cells along the first axis, as the walk itself
    num_blocks = _num_origin_blocks(shape)
    scalars = (post_steps, post_pressure, post_velocity)
    if num_blocks == 1:
        return _block(base_coords, shock_direction, *scalars)
    rows = shape[0] // num_blocks

    def scalar_blocks(x):
        return x.reshape((num_blocks, rows) + shape[1:])

    def vector_blocks(x):
        return jnp.moveaxis(x.reshape((ndim, num_blocks, rows) + shape[1:]), 1, 0)

    out = jax.lax.map(
        lambda blk: _block(*blk),
        (vector_blocks(base_coords), vector_blocks(shock_direction)) + tuple(scalar_blocks(x) for x in scalars),
    )
    return out.reshape(shape)


def _num_origin_blocks(shape, max_cells_per_block=2**21):
    """Smallest number of equal slabs along the first axis such that each
    slab has at most ``max_cells_per_block`` cells (1 for grids up to 128^3)."""
    cells_per_row = 1
    for n in shape[1:]:
        cells_per_row *= n
    for blocks in range(1, shape[0] + 1):
        if shape[0] % blocks == 0 and (shape[0] // blocks) * cells_per_row <= max_cells_per_block:
            return blocks
    return shape[0]


def _make_interior_mask(spatial_shape, margin=1):
    """
    Build a boolean mask that is True for cells at least `margin` cells away
    from every boundary. Shape: spatial_shape.

    `margin` must be >= the largest `max_steps`/`sampling_steps` used with
    `get_post_pre_shock_values` on a field of this shape: that function
    samples via `jnp.roll`, which wraps around at the domain edge, so cells
    closer than `margin` to a boundary can read physically meaningless
    wrapped-around values.
    """
    mask = jnp.ones(spatial_shape, dtype=jnp.bool_)
    for ax in range(len(spatial_shape)):
        sl_first = [slice(None)] * len(spatial_shape)
        sl_last  = [slice(None)] * len(spatial_shape)
        sl_first[ax] = slice(0, margin)
        sl_last[ax]  = slice(-margin, None)
        mask = mask.at[tuple(sl_first)].set(False)
        mask = mask.at[tuple(sl_last)].set(False)
    return mask


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _shock_zone_criterion_minimum_mach(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    shock_direction: FIELD_TYPE,
    mach_min: float = 1.3,
    gamma_gas: float = 5 / 3,
) -> BOOL_FIELD_TYPE:
    pressure    = primitive_state[registered_variables.pressure_index]
    density     = primitive_state[registered_variables.density_index]
    temperature = pressure / density

    # Rankine-Hugoniot thresholds at mach_min
    M2          = mach_min ** 2
    p_ratio_min = (2 * gamma_gas * M2 - (gamma_gas - 1)) / (gamma_gas + 1)
    T_ratio_min = p_ratio_min * ((gamma_gas - 1) * M2 + 2) / ((gamma_gas + 1) * M2)
    log_p_min   = jnp.log(p_ratio_min)
    log_T_min   = jnp.log(T_ratio_min)

    p_post, p_pre, T_post, T_pre = get_post_pre_shock_values(
        shock_direction, pressure, temperature
    )

    log_p_jump = jnp.log(jnp.maximum(p_post, 1e-30)) - jnp.log(jnp.maximum(p_pre, 1e-30))
    log_T_jump = jnp.log(jnp.maximum(T_post, 1e-30)) - jnp.log(jnp.maximum(T_pre, 1e-30))

    # zero out boundary cells (jnp.roll wraps around, those values are meaningless)
    interior = _make_interior_mask(pressure.shape)
    log_p_jump = jnp.where(interior, log_p_jump, 0.0)
    log_T_jump = jnp.where(interior, log_T_jump, 0.0)

    return (log_p_jump >= log_p_min) & (log_T_jump >= log_T_min)


# ============================================================================
# PUBLIC INTERFACE
# ============================================================================

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def identify_shock_zones(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    shock_direction: FIELD_TYPE,
    mach_min: float = 1.3,
    gamma_gas: float = 5 / 3,
) -> BOOL_FIELD_TYPE:
    """
    Identify all cells in shock zones (criteria 1 AND 2 AND 3).
    Results in ~3-4 cell thick zones per shock (Pfrommer et al. 2017).

    Args:
        primitive_state:      (num_vars, *spatial_shape)
        config:               simulation configuration
        registered_variables: registry of variable indices
        helper_data:          geometric centers etc.
        shock_direction:      unit vector field (ndim, *spatial_shape)
        mach_min:             minimum Mach threshold
        gamma_gas:            gas adiabatic index of the Rankine-Hugoniot thresholds

    Returns:
        Boolean field, shape (*spatial_shape)
    """
    pressure = primitive_state[registered_variables.pressure_index]
    density  = primitive_state[registered_variables.density_index]
    r = helper_data.geometric_centers if config.geometry == SPHERICAL else None

    criterion_1 = _shock_zone_criterion_converging_flow(
        primitive_state, config, registered_variables, r
    )
    criterion_2 = _shock_zone_criterion_aligned_gradients(pressure, density, config, r)
    criterion_3 = _shock_zone_criterion_minimum_mach(
        primitive_state, config, registered_variables, helper_data,
        shock_direction, mach_min, gamma_gas,
    )

    return criterion_1 & criterion_2 & criterion_3