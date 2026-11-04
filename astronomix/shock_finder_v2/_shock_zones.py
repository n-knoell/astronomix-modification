# ============================================================================
# PHASE 2: SHOCK ZONE IDENTIFICATION
# ============================================================================

from functools import partial
import jax.numpy as jnp
import jax
from jax.scipy.ndimage import map_coordinates

from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE, BOOL_FIELD_TYPE,
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)
from astronomix._physics_modules._shock_finder._gradients import (
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
    center_offsets=None,
):
    """
    Sample two scalar fields on both sides of a candidate shock.

    The shock direction points from the hot/post-shock side toward the
    cold/pre-shock side.  Sampling follows the complete continuous direction
    vector and uses multilinear interpolation at off-grid positions.  This
    avoids discontinuities where the dominant Cartesian component changes.

    Args:
        shock_direction:
            Unit-vector field with shape (ndim, *spatial_shape).

        field_a:
            First scalar field to sample, for example pressure.

        field_b:
            Second scalar field to sample, for example temperature or density.

        max_steps:
            Distance, in grid-cell units, from the candidate shock cell.

        center_offsets:
            Optional vector displacement from each cell centre, in grid-cell
            units, with the same shape as ``shock_direction``.

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

    ndim = field_a.ndim
    if shock_direction.shape[0] != ndim:
        raise ValueError(
            "shock_direction must have one component per spatial dimension."
        )

    coordinate_dtype = jnp.result_type(field_a.dtype, jnp.float32)
    grid_coordinates = jnp.stack(
        jnp.meshgrid(
            *[
                jnp.arange(size, dtype=coordinate_dtype)
                for size in field_a.shape
            ],
            indexing="ij",
        ),
        axis=0,
    )
    if center_offsets is not None:
        if center_offsets.shape != shock_direction.shape:
            raise ValueError(
                "center_offsets must have the same shape as shock_direction."
            )
        grid_coordinates = grid_coordinates + center_offsets.astype(
            coordinate_dtype
        )
    displacement = (
        jnp.asarray(max_steps, dtype=coordinate_dtype)
        * shock_direction.astype(coordinate_dtype)
    )

    # d_shock points from hot/post-shock gas toward cold/pre-shock gas.
    post_coordinates = grid_coordinates - displacement
    pre_coordinates = grid_coordinates + displacement

    def interpolate(field, coordinates):
        return map_coordinates(
            field,
            coordinates,
            order=1,
            mode="nearest",
        )

    field_a_post = interpolate(field_a, post_coordinates)
    field_a_pre = interpolate(field_a, pre_coordinates)
    field_b_post = interpolate(field_b, post_coordinates)
    field_b_pre = interpolate(field_b, pre_coordinates)

    return (
        field_a_post,
        field_a_pre,
        field_b_post,
        field_b_pre,
    )


def get_adaptive_post_pre_shock_values(
    shock_direction,
    shock_zones,
    field_a,
    field_b,
    max_steps=8,
    center_offsets=None,
    step_size=0.25,
    outside_offset=0.25,
):
    """Sample fluid states immediately outside both shock-zone boundaries.

    Starting at each (optionally refined) shock-surface position, this sampler
    follows the complete continuous shock normal in both directions.  The
    post- and pre-shock distances are chosen independently using a sub-cell
    search for the first point outside the detected shock zone.  The final
    sample is shifted slightly farther into the exterior state to avoid
    interpolating exactly across the boolean zone boundary.  This avoids both
    whole-cell angular banding and crossing compact shocks or nearby source
    regions with one fixed distance.

    Returns the four sampled fields, a validity mask requiring a zone exit on
    both sides, and the independently selected post/pre distances in grid-cell
    units.  Rays that reach a domain boundary before leaving the zone are
    invalid.
    """
    ndim = field_a.ndim
    if shock_direction.shape[0] != ndim:
        raise ValueError(
            "shock_direction must have one component per spatial dimension."
        )
    if shock_zones.shape != field_a.shape:
        raise ValueError("shock_zones and sampled fields must have equal shape.")
    if step_size <= 0.0:
        raise ValueError("step_size must be positive.")
    if outside_offset < 0.0 or outside_offset >= max_steps:
        raise ValueError(
            "outside_offset must be non-negative and smaller than max_steps."
        )

    coordinate_dtype = jnp.result_type(field_a.dtype, jnp.float32)
    coordinates = jnp.stack(
        jnp.meshgrid(
            *[
                jnp.arange(size, dtype=coordinate_dtype)
                for size in field_a.shape
            ],
            indexing="ij",
        ),
        axis=0,
    )
    direction = shock_direction.astype(coordinate_dtype)
    if center_offsets is not None:
        if center_offsets.shape != shock_direction.shape:
            raise ValueError(
                "center_offsets must have the same shape as shock_direction."
            )
        coordinates = coordinates + center_offsets.astype(coordinate_dtype)

    upper_bounds = jnp.asarray(field_a.shape, dtype=coordinate_dtype).reshape(
        (ndim,) + (1,) * ndim
    )

    def find_exit_distance(sign):
        distance = jnp.full(field_a.shape, float(max_steps), coordinate_dtype)
        found = jnp.zeros(field_a.shape, dtype=jnp.bool_)

        num_search_steps = int((max_steps - outside_offset) / step_size)
        for step in range(1, num_search_steps + 1):
            search_distance = jnp.asarray(
                step * step_size,
                dtype=coordinate_dtype,
            )
            sample_coordinates = (
                coordinates + sign * search_distance * direction
            )
            in_bounds = jnp.all(
                (sample_coordinates >= 0.0)
                & (sample_coordinates <= upper_bounds - 1.0),
                axis=0,
            )
            sampled_zone = map_coordinates(
                shock_zones.astype(coordinate_dtype),
                sample_coordinates,
                order=0,
                mode="nearest",
            ) > 0.5
            first_exit = ~found & in_bounds & ~sampled_zone
            sampling_distance = search_distance + jnp.asarray(
                outside_offset,
                dtype=coordinate_dtype,
            )
            distance = jnp.where(first_exit, sampling_distance, distance)
            found = found | first_exit

        final_coordinates = coordinates + sign * distance * direction
        final_in_bounds = jnp.all(
            (final_coordinates >= 0.0)
            & (final_coordinates <= upper_bounds - 1.0),
            axis=0,
        )
        return distance, found & final_in_bounds

    # d_shock points from hot/post-shock gas toward cold/pre-shock gas.
    post_distance, post_found = find_exit_distance(-1.0)
    pre_distance, pre_found = find_exit_distance(1.0)
    post_coordinates = coordinates - post_distance[jnp.newaxis, ...] * direction
    pre_coordinates = coordinates + pre_distance[jnp.newaxis, ...] * direction

    def interpolate(field, sample_coordinates):
        return map_coordinates(
            field,
            sample_coordinates,
            order=1,
            mode="nearest",
        )

    return (
        interpolate(field_a, post_coordinates),
        interpolate(field_a, pre_coordinates),
        interpolate(field_b, post_coordinates),
        interpolate(field_b, pre_coordinates),
        post_found & pre_found,
        post_distance,
        pre_distance,
    )


def get_profile_aware_post_pre_shock_values(
    shock_direction,
    shock_zones,
    pressure,
    field_b,
    max_steps=8,
    center_offsets=None,
    step_size=0.25,
    outside_offsets=(0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
):
    """Select upstream and downstream states from short pressure profiles.

    The adaptive sampler identifies the first point outside each local shock-
    zone boundary.  A boolean zone can end while multilinear interpolation is
    still inside the numerically broadened pressure ramp, especially when the
    shock is oblique to the grid.  This sampler therefore evaluates several
    bounded offsets beyond each exit:

    * the downstream state is the largest valid positive pressure;
    * the upstream state is the smallest valid positive pressure;
    * ``field_b`` is taken at the same locations as the selected pressures.

    The bounded search captures the shock-adjacent downstream pressure peak
    without committing to one ramp width.  Ties are resolved in favour of the
    nearest candidate because ``jax.numpy.argmax``/``argmin`` return the first
    occurrence.  Candidates outside the domain or with non-finite/non-positive
    thermodynamic values are ignored.  At least one valid candidate is
    required on each side.

    Distances and offsets are measured in grid-cell units.  The return layout
    matches :func:`get_adaptive_post_pre_shock_values` so callers can compare
    the two estimators directly.
    """
    ndim = pressure.ndim
    if shock_direction.shape[0] != ndim:
        raise ValueError(
            "shock_direction must have one component per spatial dimension."
        )
    if shock_zones.shape != pressure.shape or field_b.shape != pressure.shape:
        raise ValueError(
            "shock_zones and sampled fields must have equal shape."
        )
    if not outside_offsets:
        raise ValueError("outside_offsets must contain at least one value.")
    if any(offset < 0.0 for offset in outside_offsets):
        raise ValueError("outside_offsets must be non-negative.")
    if tuple(outside_offsets) != tuple(sorted(outside_offsets)):
        raise ValueError("outside_offsets must be sorted in ascending order.")

    # First locate the two zone exits without adding an exterior margin.
    (
        _,
        _,
        _,
        _,
        exit_valid,
        post_exit_distance,
        pre_exit_distance,
    ) = get_adaptive_post_pre_shock_values(
        shock_direction,
        shock_zones,
        pressure,
        field_b,
        max_steps=max_steps,
        center_offsets=center_offsets,
        step_size=step_size,
        outside_offset=0.0,
    )

    coordinate_dtype = jnp.result_type(pressure.dtype, jnp.float32)
    coordinates = jnp.stack(
        jnp.meshgrid(
            *[
                jnp.arange(size, dtype=coordinate_dtype)
                for size in pressure.shape
            ],
            indexing="ij",
        ),
        axis=0,
    )
    direction = shock_direction.astype(coordinate_dtype)
    if center_offsets is not None:
        if center_offsets.shape != shock_direction.shape:
            raise ValueError(
                "center_offsets must have the same shape as shock_direction."
            )
        coordinates = coordinates + center_offsets.astype(coordinate_dtype)

    upper_bounds = jnp.asarray(
        pressure.shape, dtype=coordinate_dtype
    ).reshape((ndim,) + (1,) * ndim)

    def sample_candidates(sign, exit_distance):
        pressure_samples = []
        field_b_samples = []
        valid_samples = []
        distance_samples = []
        for offset in outside_offsets:
            distance = exit_distance + jnp.asarray(
                offset, dtype=coordinate_dtype
            )
            sample_coordinates = (
                coordinates + sign * distance[jnp.newaxis, ...] * direction
            )
            in_bounds = jnp.all(
                (sample_coordinates >= 0.0)
                & (sample_coordinates <= upper_bounds - 1.0),
                axis=0,
            )
            pressure_sample = map_coordinates(
                pressure,
                sample_coordinates,
                order=1,
                mode="nearest",
            )
            field_b_sample = map_coordinates(
                field_b,
                sample_coordinates,
                order=1,
                mode="nearest",
            )
            valid = (
                exit_valid
                & in_bounds
                & jnp.isfinite(pressure_sample)
                & jnp.isfinite(field_b_sample)
                & (pressure_sample > 0.0)
                & (field_b_sample > 0.0)
            )
            pressure_samples.append(pressure_sample)
            field_b_samples.append(field_b_sample)
            valid_samples.append(valid)
            distance_samples.append(distance)
        return (
            jnp.stack(pressure_samples, axis=0),
            jnp.stack(field_b_samples, axis=0),
            jnp.stack(valid_samples, axis=0),
            jnp.stack(distance_samples, axis=0),
        )

    post_profiles = sample_candidates(-1.0, post_exit_distance)
    pre_profiles = sample_candidates(1.0, pre_exit_distance)

    def select_profile(profiles, select_largest):
        pressure_samples, field_b_samples, valid_samples, distances = profiles
        fill_value = -jnp.inf if select_largest else jnp.inf
        selectable_pressure = jnp.where(
            valid_samples, pressure_samples, fill_value
        )
        if select_largest:
            selected_index = jnp.argmax(selectable_pressure, axis=0)
        else:
            selected_index = jnp.argmin(selectable_pressure, axis=0)
        gather_index = selected_index[jnp.newaxis, ...]

        def gather(values):
            return jnp.take_along_axis(values, gather_index, axis=0)[0]

        return (
            gather(pressure_samples),
            gather(field_b_samples),
            gather(distances),
            jnp.any(valid_samples, axis=0),
        )

    pressure_post, field_b_post, post_distance, post_valid = select_profile(
        post_profiles, select_largest=True
    )
    pressure_pre, field_b_pre, pre_distance, pre_valid = select_profile(
        pre_profiles, select_largest=False
    )
    valid = (
        post_valid
        & pre_valid
        & (pressure_post >= pressure_pre)
        & (field_b_post > 0.0)
        & (field_b_pre > 0.0)
    )

    return (
        pressure_post,
        pressure_pre,
        field_b_post,
        field_b_pre,
        valid,
        post_distance,
        pre_distance,
    )


def get_plateau_aware_post_pre_shock_values(
    shock_direction,
    shock_zones,
    pressure,
    density,
    max_steps=8,
    center_offsets=None,
    step_size=0.25,
    outside_offsets=(
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
    ),
    relative_tolerance=0.01,
    require_flat_plateau=True,
    extrapolate_smooth_profile=False,
    slope_stability_absolute_tolerance=0.002,
    slope_stability_relative_tolerance=0.25,
):
    """Select the first locally flat state beyond each shock-zone edge.

    Unlike :func:`get_profile_aware_post_pre_shock_values`, which selects
    pressure extrema from a bounded profile, this sampler explicitly checks
    that both pressure and density have stopped changing.  Candidate points
    are ordered by distance beyond the detected shock-zone boundary.  The
    selected state is the farthest point of the first three-point run whose
    two adjacent intervals both satisfy

    ``abs(q[i + 1] - q[i]) / max(abs(q[i + 1]), abs(q[i])) <= tolerance``

    for both pressure and density.  Requiring two consecutive flat intervals
    is more robust than accepting one accidentally shallow part of a
    broadened shock ramp.

    A ray is invalid when it leaves the domain, cannot leave the shock zone,
    or does not reach a pressure-and-density plateau within the configured
    exterior offset window.
    The four returned relative-change fields quantify the flatness at the
    selected downstream and upstream locations and make the sampling choice
    directly diagnosable.

    Returns
    -------
    pressure_post, pressure_pre, density_post, density_pre, valid,
    post_distance, pre_distance, post_pressure_change, pre_pressure_change,
    post_density_change, pre_density_change
    """
    ndim = pressure.ndim
    if shock_direction.shape[0] != ndim:
        raise ValueError(
            "shock_direction must have one component per spatial dimension."
        )
    if shock_zones.shape != pressure.shape or density.shape != pressure.shape:
        raise ValueError(
            "shock_zones and sampled fields must have equal shape."
        )
    minimum_offsets = 3 if require_flat_plateau else 4
    if len(outside_offsets) < minimum_offsets:
        raise ValueError(
            f"outside_offsets must contain at least {minimum_offsets} values."
        )
    if any(offset < 0.0 for offset in outside_offsets):
        raise ValueError("outside_offsets must be non-negative.")
    if tuple(outside_offsets) != tuple(sorted(outside_offsets)):
        raise ValueError("outside_offsets must be sorted in ascending order.")
    if any(offset >= max_steps for offset in outside_offsets):
        raise ValueError("outside_offsets must be smaller than max_steps.")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive.")
    if slope_stability_absolute_tolerance < 0.0:
        raise ValueError(
            "slope_stability_absolute_tolerance must be non-negative."
        )
    if slope_stability_relative_tolerance < 0.0:
        raise ValueError(
            "slope_stability_relative_tolerance must be non-negative."
        )
    if extrapolate_smooth_profile and require_flat_plateau:
        raise ValueError(
            "extrapolate_smooth_profile requires smooth-profile selection."
        )

    (
        _,
        _,
        _,
        _,
        exit_valid,
        post_exit_distance,
        pre_exit_distance,
    ) = get_adaptive_post_pre_shock_values(
        shock_direction,
        shock_zones,
        pressure,
        density,
        max_steps=max_steps,
        center_offsets=center_offsets,
        step_size=step_size,
        outside_offset=0.0,
    )

    coordinate_dtype = jnp.result_type(pressure.dtype, jnp.float32)
    coordinates = jnp.stack(
        jnp.meshgrid(
            *[
                jnp.arange(size, dtype=coordinate_dtype)
                for size in pressure.shape
            ],
            indexing="ij",
        ),
        axis=0,
    )
    direction = shock_direction.astype(coordinate_dtype)
    if center_offsets is not None:
        if center_offsets.shape != shock_direction.shape:
            raise ValueError(
                "center_offsets must have the same shape as shock_direction."
            )
        coordinates = coordinates + center_offsets.astype(coordinate_dtype)

    upper_bounds = jnp.asarray(
        pressure.shape, dtype=coordinate_dtype
    ).reshape((ndim,) + (1,) * ndim)

    def sample_candidates(sign, exit_distance):
        pressure_samples = []
        density_samples = []
        valid_samples = []
        distance_samples = []
        for offset in outside_offsets:
            distance = exit_distance + jnp.asarray(
                offset, dtype=coordinate_dtype
            )
            sample_coordinates = (
                coordinates + sign * distance[jnp.newaxis, ...] * direction
            )
            in_bounds = jnp.all(
                (sample_coordinates >= 0.0)
                & (sample_coordinates <= upper_bounds - 1.0),
                axis=0,
            )
            pressure_sample = map_coordinates(
                pressure,
                sample_coordinates,
                order=1,
                mode="nearest",
            )
            density_sample = map_coordinates(
                density,
                sample_coordinates,
                order=1,
                mode="nearest",
            )
            valid = (
                exit_valid
                & in_bounds
                & jnp.isfinite(pressure_sample)
                & jnp.isfinite(density_sample)
                & (pressure_sample > 0.0)
                & (density_sample > 0.0)
            )
            pressure_samples.append(pressure_sample)
            density_samples.append(density_sample)
            valid_samples.append(valid)
            distance_samples.append(distance)
        return (
            jnp.stack(pressure_samples, axis=0),
            jnp.stack(density_samples, axis=0),
            jnp.stack(valid_samples, axis=0),
            jnp.stack(distance_samples, axis=0),
        )

    post_profiles = sample_candidates(-1.0, post_exit_distance)
    pre_profiles = sample_candidates(1.0, pre_exit_distance)

    outside_offset_array = jnp.asarray(
        outside_offsets, dtype=coordinate_dtype
    ).reshape((len(outside_offsets),) + (1,) * ndim)

    def select_first_plateau(profiles, exit_distance):
        pressure_samples, density_samples, valid_samples, distances = profiles
        pair_valid = valid_samples[:-1] & valid_samples[1:]
        numerical_floor = jnp.asarray(1.0e-30, dtype=coordinate_dtype)

        def relative_change(values):
            denominator = jnp.maximum(
                jnp.maximum(jnp.abs(values[:-1]), jnp.abs(values[1:])),
                numerical_floor,
            )
            return jnp.abs(values[1:] - values[:-1]) / denominator

        pressure_change = relative_change(pressure_samples)
        density_change = relative_change(density_samples)
        if require_flat_plateau:
            flat_intervals = (
                pair_valid
                & (pressure_change <= relative_tolerance)
                & (density_change <= relative_tolerance)
            )
            accepted_runs = flat_intervals[:-1] & flat_intervals[1:]
            selected_index_offset = 2
            final_interval_offset = 1
        else:
            def stable_slope(changes):
                first = changes[:-1]
                second = changes[1:]
                allowed_change = (
                    slope_stability_absolute_tolerance
                    + slope_stability_relative_tolerance
                    * jnp.maximum(first, second)
                )
                return jnp.abs(second - first) <= allowed_change

            adjacent_intervals_valid = pair_valid[:-1] & pair_valid[1:]
            smooth_transitions = (
                adjacent_intervals_valid
                & stable_slope(pressure_change)
                & stable_slope(density_change)
            )
            accepted_runs = (
                smooth_transitions[:-1] & smooth_transitions[1:]
            )
            selected_index_offset = 3
            final_interval_offset = 2

        plateau_found = jnp.any(accepted_runs, axis=0)
        run_index = jnp.argmax(accepted_runs, axis=0)
        selected_index = run_index + selected_index_offset
        gather_index = selected_index[jnp.newaxis, ...]
        final_interval_index = (
            run_index + final_interval_offset
        )[jnp.newaxis, ...]

        def gather(values):
            return jnp.take_along_axis(values, gather_index, axis=0)[0]

        def gather_pair(values):
            return jnp.take_along_axis(
                values, final_interval_index, axis=0
            )[0]

        def extrapolate_four_point_run(values):
            fit_indices = (
                run_index[jnp.newaxis, ...]
                + jnp.arange(4).reshape((4,) + (1,) * ndim)
            )
            fit_values = jnp.take_along_axis(
                values, fit_indices, axis=0
            )
            fit_offsets = jnp.take_along_axis(
                jnp.broadcast_to(outside_offset_array, values.shape),
                fit_indices,
                axis=0,
            )
            mean_offset = jnp.mean(fit_offsets, axis=0)
            mean_value = jnp.mean(fit_values, axis=0)
            centered_offset = fit_offsets - mean_offset[jnp.newaxis, ...]
            denominator = jnp.sum(centered_offset**2, axis=0)
            slope = jnp.sum(
                centered_offset
                * (fit_values - mean_value[jnp.newaxis, ...]),
                axis=0,
            ) / jnp.maximum(denominator, numerical_floor)
            return mean_value - slope * mean_offset

        if extrapolate_smooth_profile:
            selected_pressure = extrapolate_four_point_run(
                pressure_samples
            )
            selected_density = extrapolate_four_point_run(density_samples)
            selected_distance = exit_distance
        else:
            selected_pressure = gather(pressure_samples)
            selected_density = gather(density_samples)
            selected_distance = gather(distances)

        return (
            selected_pressure,
            selected_density,
            selected_distance,
            plateau_found,
            gather_pair(pressure_change),
            gather_pair(density_change),
        )

    (
        pressure_post,
        density_post,
        post_distance,
        post_plateau,
        post_pressure_change,
        post_density_change,
    ) = select_first_plateau(post_profiles, post_exit_distance)
    (
        pressure_pre,
        density_pre,
        pre_distance,
        pre_plateau,
        pre_pressure_change,
        pre_density_change,
    ) = select_first_plateau(pre_profiles, pre_exit_distance)

    valid = (
        post_plateau
        & pre_plateau
        & (pressure_post > 0.0)
        & (pressure_pre > 0.0)
        & (pressure_post >= pressure_pre)
        & (density_post > 0.0)
        & (density_pre > 0.0)
    )
    return (
        pressure_post,
        pressure_pre,
        density_post,
        density_pre,
        valid,
        post_distance,
        pre_distance,
        post_pressure_change,
        pre_pressure_change,
        post_density_change,
        pre_density_change,
    )


def get_ramp_exit_aware_post_pre_shock_values(
    shock_direction,
    shock_zones,
    pressure,
    density,
    max_steps=8,
    center_offsets=None,
    step_size=0.25,
    outside_offsets=(
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
    ),
    slope_stability_absolute_tolerance=0.002,
    slope_stability_relative_tolerance=0.25,
):
    """Select states after the numerical ramp becomes smoothly varying.

    A physical downstream solution need not be flat.  This variant accepts
    the first four-point run for which the fractional pressure and density
    slopes are stable across three consecutive intervals.  It therefore
    distinguishes a rapidly changing numerical shock ramp from a resolved,
    smoothly varying physical profile without requiring a zero gradient.
    """
    return get_plateau_aware_post_pre_shock_values(
        shock_direction,
        shock_zones,
        pressure,
        density,
        max_steps=max_steps,
        center_offsets=center_offsets,
        step_size=step_size,
        outside_offsets=outside_offsets,
        require_flat_plateau=False,
        slope_stability_absolute_tolerance=(
            slope_stability_absolute_tolerance
        ),
        slope_stability_relative_tolerance=(
            slope_stability_relative_tolerance
        ),
    )


def get_extrapolated_post_pre_shock_values(
    shock_direction,
    shock_zones,
    pressure,
    density,
    max_steps=8,
    center_offsets=None,
    step_size=0.25,
    outside_offsets=(
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
    ),
    slope_stability_absolute_tolerance=0.002,
    slope_stability_relative_tolerance=0.25,
):
    """Extrapolate smooth exterior profiles back to each shock-zone exit.

    The first four-point run with stable fractional pressure and density
    slopes is fitted linearly as a function of distance beyond the detected
    shock-zone boundary.  The fit is evaluated at zero exterior offset.  This
    estimates a shock-adjacent state without mistaking a smoothly varying
    downstream solution for a constant plateau.

    This is an experimental diagnostic.  Its validity mask and reported local
    changes retain the same meaning as the ramp-exit-aware sampler, while the
    returned distances identify the zone exits where the fits are evaluated.
    """
    return get_plateau_aware_post_pre_shock_values(
        shock_direction,
        shock_zones,
        pressure,
        density,
        max_steps=max_steps,
        center_offsets=center_offsets,
        step_size=step_size,
        outside_offsets=outside_offsets,
        require_flat_plateau=False,
        extrapolate_smooth_profile=True,
        slope_stability_absolute_tolerance=(
            slope_stability_absolute_tolerance
        ),
        slope_stability_relative_tolerance=(
            slope_stability_relative_tolerance
        ),
    )


def get_post_pre_sample_relative_changes(
    shock_direction,
    pressure,
    density,
    post_distance,
    pre_distance,
    center_offsets=None,
    farther_step=0.25,
):
    """Measure how much selected states change one step farther from a shock.

    This diagnostic can be applied to any sampler that returns downstream and
    upstream distances.  A selected state is locally flat when its pressure
    and density change little between the selected position and a point
    ``farther_step`` cells farther away from the shock.
    """
    ndim = pressure.ndim
    if shock_direction.shape[0] != ndim:
        raise ValueError(
            "shock_direction must have one component per spatial dimension."
        )
    if density.shape != pressure.shape:
        raise ValueError("pressure and density must have equal shape.")
    if post_distance.shape != pressure.shape or pre_distance.shape != pressure.shape:
        raise ValueError("sampling-distance fields must match the fluid fields.")
    if farther_step <= 0.0:
        raise ValueError("farther_step must be positive.")

    coordinate_dtype = jnp.result_type(pressure.dtype, jnp.float32)
    coordinates = jnp.stack(
        jnp.meshgrid(
            *[
                jnp.arange(size, dtype=coordinate_dtype)
                for size in pressure.shape
            ],
            indexing="ij",
        ),
        axis=0,
    )
    direction = shock_direction.astype(coordinate_dtype)
    if center_offsets is not None:
        if center_offsets.shape != shock_direction.shape:
            raise ValueError(
                "center_offsets must have the same shape as shock_direction."
            )
        coordinates = coordinates + center_offsets.astype(coordinate_dtype)
    upper_bounds = jnp.asarray(
        pressure.shape, dtype=coordinate_dtype
    ).reshape((ndim,) + (1,) * ndim)

    def side_changes(sign, distance):
        selected_coordinates = (
            coordinates + sign * distance[jnp.newaxis, ...] * direction
        )
        farther_coordinates = (
            coordinates
            + sign
            * (distance + farther_step)[jnp.newaxis, ...]
            * direction
        )
        in_bounds = jnp.all(
            (selected_coordinates >= 0.0)
            & (selected_coordinates <= upper_bounds - 1.0)
            & (farther_coordinates >= 0.0)
            & (farther_coordinates <= upper_bounds - 1.0),
            axis=0,
        )

        def relative_change(field):
            selected = map_coordinates(
                field, selected_coordinates, order=1, mode="nearest"
            )
            farther = map_coordinates(
                field, farther_coordinates, order=1, mode="nearest"
            )
            denominator = jnp.maximum(
                jnp.maximum(jnp.abs(selected), jnp.abs(farther)),
                jnp.asarray(1.0e-30, dtype=coordinate_dtype),
            )
            return jnp.abs(farther - selected) / denominator

        pressure_change = relative_change(pressure)
        density_change = relative_change(density)
        valid = (
            in_bounds
            & jnp.isfinite(pressure_change)
            & jnp.isfinite(density_change)
        )
        return pressure_change, density_change, valid

    post_pressure_change, post_density_change, post_valid = side_changes(
        -1.0, post_distance
    )
    pre_pressure_change, pre_density_change, pre_valid = side_changes(
        1.0, pre_distance
    )
    return (
        post_pressure_change,
        pre_pressure_change,
        post_density_change,
        pre_density_change,
        post_valid & pre_valid,
    )

def _make_interior_mask(spatial_shape):
    """
    Build a boolean mask that is True for interior cells (not on any boundary).
    Shape: spatial_shape.
    """
    mask = jnp.ones(spatial_shape, dtype=jnp.bool_)
    for ax in range(len(spatial_shape)):
        sl_first = [slice(None)] * len(spatial_shape)
        sl_last  = [slice(None)] * len(spatial_shape)
        sl_first[ax] = 0
        sl_last[ax]  = -1
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
