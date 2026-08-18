"""Shared 3D Sedov-Taylor blast-wave setup for the shock-finder tests.

Builds a spherically-symmetric point explosion (smoothly-tapered pressure
injection, following ``examples/scripts/forward/hydro/sedov_blast.py``),
integrates it with the finite-volume HLLC solver and runs
:func:`astronomix.shock_finder3D.pfrommer_shock_finder.find_shocks_pfrommer`
on the final state. Kept resolution-independent (only ``num_cells`` varies)
so the same physical setup can be reused for both the single-resolution
correctness checks and the resolution-convergence study.
"""

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD

# astronomix containers
from astronomix import SimulationConfig, SimulationParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

# ---- physical setup (matches examples/scripts/forward/hydro/sedov_blast.py) ----
GAMMA = 5.0 / 3.0
T_END = 0.1
E_EXPLOSION = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1e-4
R_EXPLOSION = 0.05
SMOOTH_CELLS = 2.0
MACH_MIN = 1.3


def run_sedov(num_cells):
    """Run one 3D Sedov blast and find its shocks.

    Args:
        num_cells: The number of cells per dimension (cubic domain).

    Returns:
        A dict with the final primitive ``state``, the ``config``,
        ``helper_data``, ``registered_variables`` and the shock finder's
        ``ShockFinderResult``.
    """
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=num_cells,
        exact_end_time=True,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (num_cells, num_cells, num_cells)
    density = jnp.ones(shape) * RHO_AMBIENT
    zeros = jnp.zeros(shape)

    # Smoothly-tapered spherical injection weight in [0, 1], then renormalise
    # the over-pressure so the deposited thermal energy above ambient is
    # exactly E_EXPLOSION regardless of resolution (see sedov_blast.py).
    dx = 1.0 / num_cells  # box_size = 1.0
    smooth_width = SMOOTH_CELLS * dx
    radius = helper_data.r
    weight = 0.5 * (1.0 - jnp.tanh((radius - R_EXPLOSION) / smooth_width))
    cell_volume = dx**3
    delta_p = E_EXPLOSION * (GAMMA - 1.0) / (jnp.sum(weight) * cell_volume)
    gas_pressure = P_AMBIENT + delta_p * weight

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, gamma=GAMMA)

    state = time_integration(initial_state, config, params, registered_variables)

    sf_result = find_shocks_pfrommer(
        state, config, registered_variables, helper_data, mach_min=MACH_MIN
    )

    return dict(
        state=state,
        config=config,
        helper_data=helper_data,
        registered_variables=registered_variables,
        sf_result=sf_result,
    )


def binned_radial_profile(r, field, num_bins=150, r_max=0.75):
    """Spherically-bin ``field`` by radius, independent of the shock finder.

    Args:
        r: Flattened cell-centre radii.
        field: Flattened field values to average per bin.
        num_bins: Number of radial bins.
        r_max: Maximum radius to bin out to.

    Returns:
        ``(bin_centers, bin_means)``, both length ``num_bins``; bins with no
        cells hold ``nan``.
    """
    bins = np.linspace(0.0, r_max, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_index = np.clip(np.digitize(r, bins) - 1, 0, num_bins - 1)
    counts = np.zeros(num_bins)
    sums = np.zeros(num_bins)
    np.add.at(counts, bin_index, 1.0)
    np.add.at(sums, bin_index, field)
    with np.errstate(invalid="ignore"):
        means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return bin_centers, means
