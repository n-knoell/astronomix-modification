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

Site placement is drawn uniformly at random over the whole box ("random"
placement, matching Girichidis et al. 2016's own comparison case and Simpson
et al. 2016's "random" mode -- see DESIGN.md's SILCC-ISM section; "at density
peaks" is a stretch M6 cross-check, not implemented here) and the injection
footprint wraps periodically along any axis whose boundary is periodic (the
box-scale minimum-image convention), so a site drawn near a periodic edge
still gets its full, undistorted energy deposit.
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


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _inject_supernovae(
    key,
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
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

    Returns:
        ``(key, primitive_state)`` with the advanced PRNG key and the
        supernova (if triggered) deposited.
    """
    sn_params = params.sn_driving_params

    key, trigger_key, pos_key = jax.random.split(key, 3)

    trigger_probability = jnp.clip(sn_params.sn_rate * dt, 0.0, 1.0)
    triggered = jax.random.bernoulli(trigger_key, trigger_probability)

    box_size = config.box_size
    site = jax.random.uniform(pos_key, shape=(3,)) * jnp.array(
        [box_size.x, box_size.y, box_size.z]
    )

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

    # helper_data/primitive_state here are ghost-padded (this function is
    # called from _iteration_level_updates with helper_data_pad). Restrict
    # the weight -- both the normalization sum and the actual deposit -- to
    # the interior cells only: a ghost cell's contribution would otherwise
    # double-count real domain volume across a periodic wrap (a site near an
    # edge is close, in raw coordinates, to both the true interior cells on
    # the far side *and* the near-side ghost cells that mirror them) and
    # then vanish when the next boundary-condition application overwrites
    # the ghost cells from the interior, silently losing that fraction of
    # sn_energy. The interior deposit alone is exactly one cell per true
    # domain volume element, so the boundary handler then correctly refreshes
    # the ghost cells (whatever each axis's BC is) from the updated interior
    # before any flux computation reads them.
    ngc = config.num_ghost_cells
    num_cells = config.num_cells
    interior_x = (jnp.arange(weight.shape[0]) >= ngc) & (
        jnp.arange(weight.shape[0]) < num_cells.x + ngc
    )
    interior_y = (jnp.arange(weight.shape[1]) >= ngc) & (
        jnp.arange(weight.shape[1]) < num_cells.y + ngc
    )
    interior_z = (jnp.arange(weight.shape[2]) >= ngc) & (
        jnp.arange(weight.shape[2]) < num_cells.z + ngc
    )
    interior_mask = (
        interior_x[:, None, None] & interior_y[None, :, None] & interior_z[None, None, :]
    )
    weight = weight * interior_mask

    cell_volume = config.grid_spacing**3
    # Renormalize so the deposited energy is exactly sn_energy regardless of
    # resolution or how the footprint straddles the periodic wrap (mirrors
    # _sedov_setup.py's delta_p normalization).
    weight_integral_safe = jnp.maximum(jnp.sum(weight) * cell_volume, 1e-300)

    cr_fraction = (
        sn_params.sn_cr_fraction if registered_variables.cosmic_ray_e_active else 0.0
    )
    thermal_energy = (1.0 - cr_fraction) * sn_params.sn_energy
    cr_energy = cr_fraction * sn_params.sn_energy

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

    return key, primitive_state
