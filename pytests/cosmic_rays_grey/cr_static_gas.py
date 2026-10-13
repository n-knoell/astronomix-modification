"""Static heavy gas for pure CR-transport tests (Phase A test plan X6).

CR transport is linear in ``e_cr``, so a test can run the published CR values times
``E_SCALE`` in dense, cold gas: the CR pressure and the magnetic forces then move the gas
by far less than a cell, and the run is a pure transport problem in an MHD / hydro code.
First used in ``cr_anisotropic_ring.py`` (same constants; that file still has its own
copy), then in the steady-state tests (A4.4, A3.1).
"""

import jax.numpy as jnp
import numpy as np

GAS_DENSITY = 1e4
GAS_PRESSURE = 1e-4
E_SCALE = 1e-7
B0 = 1e-5


def static_gas_state(registered_variables, grid_shape, e_cr, f_cr=None, magnetic_field=None):
    """Primitive state: heavy gas at rest, ``e_cr = E_SCALE * e_cr`` and ``F_cr =
    E_SCALE * f_cr`` (zero if None, one array per flux row), ``B = B0 * magnetic_field``
    (``(B_x, B_y[, B_z])`` in units of ``B0``, MHD only)."""
    state = jnp.zeros((registered_variables.num_vars,) + tuple(grid_shape))
    state = state.at[registered_variables.density_index].set(GAS_DENSITY)
    state = state.at[registered_variables.pressure_index].set(GAS_PRESSURE)
    state = state.at[registered_variables.cosmic_ray_e_index].set(E_SCALE * np.asarray(e_cr))
    if f_cr is not None:
        flux = registered_variables.cosmic_ray_flux_index
        rows = (flux,) if isinstance(flux, int) else tuple(r for r in flux if r >= 0)
        for row, component in zip(rows, f_cr):
            state = state.at[row].set(E_SCALE * np.asarray(component))
    if magnetic_field is not None:
        b_index = registered_variables.magnetic_index
        for row, component in zip((b_index.x, b_index.y, b_index.z), magnetic_field):
            state = state.at[row].set(B0 * np.asarray(component))
    return state


def max_gas_speed(states, registered_variables, dimensionality):
    """Largest ``|u|`` over the given state(s) (leading axes: snapshots, variables)."""
    velocity = registered_variables.velocity_index
    rows = (velocity,) if isinstance(velocity, int) else tuple(velocity)[:dimensionality]
    states = np.asarray(states)
    variable_axis = states.ndim - 1 - dimensionality
    return float(np.sqrt(sum(np.take(states, row, axis=variable_axis) ** 2 for row in rows)).max())


def check_gas_at_rest(max_speed, t_end, dx, max_cells=0.1):
    """The gas moved by less than ``max_cells`` cells over the run."""
    displacement = max_speed * t_end / dx
    assert displacement < max_cells, (
        f"the gas moved {displacement:.2e} cells (>= {max_cells}); the static-gas setup does not hold."
    )
