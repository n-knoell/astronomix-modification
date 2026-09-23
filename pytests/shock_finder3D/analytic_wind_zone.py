"""Unit tests for the optional analytic wind zone
(``WindConfig.analytic_wind_zone``, ``stellar_wind._analytic_wind_zone``).

Checks that (1) with the flag off, ``_wind_injection`` is exactly the plain 3D
EI scheme, (2) with it on, the cells inside each source's zone carry the
analytic adiabatically-cooled free-wind state, the zone radius is the
configured fraction of the stagnation distance, and everything outside the
zones is untouched, and (3) unsupported configurations are rejected.

Run: ``pytest pytests/shock_finder3D/analytic_wind_zone.py``.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
)
from astronomix.option_classes import EI, MEO, WindConfig, WindParams
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
)
from astronomix._modules._stellar_wind.stellar_wind import (
    _wind_ei3D,
    _wind_injection,
    _wind_source_distances,
)

GAMMA = 5.0 / 3.0
NUM_CELLS = 32
BOX_SIZE = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1.0

# two sources along x, the second with a 4x weaker momentum rate, so the
# stagnation point sits at 2/3 of the separation from source 1
STAR_POSITIONS = jnp.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
MASS_RATES = jnp.array([4.0, 1.0])
VEL_SCALES = jnp.array([10.0, 10.0])
BASE_RADII = jnp.array([0.01, 0.02])
BASE_TEMPERATURES = jnp.array([50.0, 80.0])
ZONE_FRACTION = 0.5
DT = jnp.asarray(1e-4)


def _setup(analytic_wind_zone, trace_wind_density=False):
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=NUM_CELLS,
        wind_config=WindConfig(
            stellar_wind=True,
            num_injection_cells=2,
            wind_injection_scheme=EI,
            trace_wind_density=trace_wind_density,
            analytic_wind_zone=analytic_wind_zone,
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    shape = (NUM_CELLS,) * 3
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.full(shape, RHO_AMBIENT),
        velocity_x=jnp.zeros(shape),
        velocity_y=jnp.zeros(shape),
        velocity_z=jnp.zeros(shape),
        gas_pressure=jnp.full(shape, P_AMBIENT),
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        gamma=GAMMA,
        wind_params=WindParams(
            wind_mass_loss_rates=MASS_RATES,
            wind_final_velocities=VEL_SCALES,
            wind_injection_positions=STAR_POSITIONS,
            wind_base_radii=BASE_RADII,
            wind_base_temperatures=BASE_TEMPERATURES,
            wind_zone_stagnation_fraction=ZONE_FRACTION,
        ),
    )
    return state, config, params, helper_data, registered_variables


def _plain_ei(state, config, params, helper_data, registered_variables):
    return _wind_ei3D(
        params, state, DT, config, helper_data, config.num_ghost_cells,
        config.wind_config.num_injection_cells, params.gamma, registered_variables,
    )


def test_flag_off_is_plain_ei():
    state, config, params, helper_data, rv = _setup(analytic_wind_zone=False)
    injected = _wind_injection(state, DT, config, params, helper_data, rv)
    np.testing.assert_array_equal(
        injected, _plain_ei(state, config, params, helper_data, rv)
    )


def test_zone_matches_analytic_profile():
    state, config, params, helper_data, rv = _setup(
        analytic_wind_zone=True, trace_wind_density=True
    )
    injected = _wind_injection(state, DT, config, params, helper_data, rv)
    ei_only = _plain_ei(state, config, params, helper_data, rv)

    # stagnation distances: r_i = d sqrt(P_i) / (sqrt(P_1) + sqrt(P_2))
    separation = 0.4
    sqrt_p = np.sqrt(np.asarray(MASS_RATES * VEL_SCALES))
    zone_radii = ZONE_FRACTION * separation * sqrt_p / sqrt_p.sum()
    np.testing.assert_allclose(zone_radii, [0.5 * 0.4 * 2 / 3, 0.5 * 0.4 / 3])

    dist = np.asarray(_wind_source_distances(config, helper_data, STAR_POSITIONS))
    box_center = BOX_SIZE / 2
    centered = np.asarray(helper_data.geometric_centers) - box_center
    radius = np.maximum(dist, 0.5 * config.grid_spacing)

    in_any_zone = np.zeros(dist.shape[1:], dtype=bool)
    for i in range(2):
        in_zone = dist[i] <= zone_radii[i]
        assert in_zone.sum() > 1
        in_any_zone |= in_zone
        r = radius[i][in_zone]
        rho = float(MASS_RATES[i]) / (4 * np.pi * r**2 * float(VEL_SCALES[i]))
        p = rho * float(BASE_TEMPERATURES[i]) * (float(BASE_RADII[i]) / r) ** (
            2 * (GAMMA - 1)
        )
        offset = centered[in_zone] - np.asarray(STAR_POSITIONS[i])
        v = float(VEL_SCALES[i]) * offset / r[:, None]

        rtol = 1e-5
        np.testing.assert_allclose(injected[rv.density_index][in_zone], rho, rtol=rtol)
        np.testing.assert_allclose(injected[rv.pressure_index][in_zone], p, rtol=rtol)
        np.testing.assert_allclose(
            injected[rv.wind_density_index][in_zone], rho, rtol=rtol
        )
        for axis, index in enumerate(
            (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)
        ):
            np.testing.assert_allclose(
                injected[index][in_zone], v[:, axis], rtol=rtol, atol=1e-6
            )

    # the zones are the only thing that differs from plain EI
    np.testing.assert_array_equal(
        np.asarray(injected)[:, ~in_any_zone], np.asarray(ei_only)[:, ~in_any_zone]
    )


def test_zone_never_smaller_than_injection_sphere():
    state, config, params, helper_data, rv = _setup(analytic_wind_zone=True)
    # a tiny fraction would give a zone smaller than the EI injection sphere
    params = params._replace(
        wind_params=params.wind_params._replace(wind_zone_stagnation_fraction=1e-3)
    )
    injected = _wind_injection(state, DT, config, params, helper_data, rv)
    injection_radius = config.wind_config.num_injection_cells * config.grid_spacing
    dist = np.asarray(_wind_source_distances(config, helper_data, STAR_POSITIONS))
    in_zone = dist[0] <= injection_radius
    radius = np.maximum(dist[0][in_zone], 0.5 * config.grid_spacing)
    rho = float(MASS_RATES[0]) / (4 * np.pi * radius**2 * float(VEL_SCALES[0]))
    np.testing.assert_allclose(injected[rv.density_index][in_zone], rho, rtol=1e-5)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(wind_injection_scheme=MEO),
        dict(stellar_wind=False),
    ],
)
def test_unsupported_config_rejected(overrides):
    _assert_rejected(FINITE_VOLUME, overrides)


def test_finite_difference_rejected():
    _assert_rejected(FINITE_DIFFERENCE, {})


def _assert_rejected(solver_mode, overrides):
    wind_config = WindConfig(
        stellar_wind=True, wind_injection_scheme=EI, analytic_wind_zone=True
    )._replace(**overrides)
    config = SimulationConfig(
        solver_mode=solver_mode, dimensionality=3, box_size=BOX_SIZE,
        num_cells=8, wind_config=wind_config,
    )
    with pytest.raises(ValueError, match="analytic_wind_zone"):
        finalize_config(config, (5, 8, 8, 8))


def test_one_dimensional_rejected():
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        box_size=BOX_SIZE,
        num_cells=8,
        wind_config=WindConfig(
            stellar_wind=True, wind_injection_scheme=EI, analytic_wind_zone=True
        ),
    )
    with pytest.raises(ValueError, match="analytic_wind_zone"):
        finalize_config(config, (3, 8))
