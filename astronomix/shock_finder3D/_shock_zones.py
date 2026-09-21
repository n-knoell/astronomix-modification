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

@partial(jax.jit, static_argnames=["max_steps"])
def get_post_pre_shock_values_adaptive(
    shock_direction,
    field_a,
    field_b,
    shock_zones,
    max_steps=15,
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

    base_coords = jnp.stack(
        jnp.meshgrid(*[jnp.arange(n, dtype=dtype) for n in shape], indexing="ij"),
        axis=0,
    )
    zones_f = shock_zones.astype(dtype)

    def _sample(field, coords):
        return jax.scipy.ndimage.map_coordinates(field, coords, order=1, mode="nearest")

    def _sample_zone(coords):
        return jax.scipy.ndimage.map_coordinates(zones_f, coords, order=0, mode="nearest") > 0.5

    def _walk(sign):
        # jax.lax.fori_loop, not a Python-unrolled loop: the loop body is
        # traced/compiled once and executed max_steps times by XLA's own
        # While op, instead of max_steps duplicated copies of the body being
        # chained into one giant graph. The unrolled version compiled and
        # ran fine at N=64 but OOM'd at N=256 (each of the ~15 unrolled
        # steps materializes several full-grid intermediates at once) --
        # this is a real resource-usage fix, not just a style preference.
        def body_fun(step, carry):
            a_val, b_val, exited_ever, still_inside = carry
            step = step.astype(dtype)
            coords = base_coords + sign * step * shock_direction
            coords_list = [coords[d] for d in range(ndim)]
            inside_here = _sample_zone(coords_list)
            newly_exited = still_inside & (~inside_here) & (~exited_ever)

            a_here = _sample(field_a, coords_list)
            b_here = _sample(field_b, coords_list)
            a_val = jnp.where(newly_exited, a_here, a_val)
            b_val = jnp.where(newly_exited, b_here, b_val)

            exited_ever = exited_ever | newly_exited
            return a_val, b_val, exited_ever, inside_here

        init_carry = (field_a, field_b, jnp.zeros(shape, dtype=jnp.bool_), shock_zones)
        a_val, b_val, exited_ever, _ = jax.lax.fori_loop(1, max_steps + 1, body_fun, init_carry)

        # Rays still inside the zone after max_steps: fall back to the value
        # at the cap rather than extrapolating further (matches the safety
        # bound _energy_dissipation.py/_shock_mach.py already rely on for
        # the fixed-step walk's own max_steps).
        coords_final = base_coords + sign * max_steps * shock_direction
        coords_final_list = [coords_final[d] for d in range(ndim)]
        a_val = jnp.where(exited_ever, a_val, _sample(field_a, coords_final_list))
        b_val = jnp.where(exited_ever, b_val, _sample(field_b, coords_final_list))

        return a_val, b_val, exited_ever

    # +shock_direction points toward the pre-shock (cold) side, -shock_direction
    # toward the post-shock (hot) side (see get_post_pre_shock_values above).
    field_a_pre, field_b_pre, exited_pre = _walk(+1)
    field_a_post, field_b_post, exited_post = _walk(-1)

    return (
        field_a_post,
        field_a_pre,
        field_b_post,
        field_b_pre,
        exited_post,
        exited_pre,
    )


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
) -> BOOL_FIELD_TYPE:
    gamma_gas = 5 / 3
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
        shock_direction, mach_min,
    )

    return criterion_1 & criterion_2 & criterion_3