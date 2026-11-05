"""Initial states and setups of the resolution study (PLAN.md Section 3), unrun.

Every builder returns ``(initial_state, config, params, helper_data,
registered_variables)`` with ``exact_end_time=True``, no snapshot stack and no
progress bar, for ``common.run_segments``.

* ``build_sedov_pe``: S-PE, the item-7 point explosion of
  ``cr_sedov_taylor._run_sedov`` (same constants; ``p_ambient`` and the
  efficiency model selectable for S-CP).
* ``build_sedov_ss``: S-SS / S-CP, started from the self-similar state.
* ``build_cwb``: the CWB of ``_cwb_setup.run_cwb`` with its defaults.
"""

import common  # noqa: F401  (paths, autocvd patch)

import numpy as np
import jax.numpy as jnp

from astronomix import (CARTESIAN, FINITE_VOLUME, HLLC, MINMOD, SimulationConfig, SimulationParams,
                        construct_primitive_state, finalize_config, get_helper_data,
                        get_registered_variables)
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    DSA_EFFICIENCY_KANG_RYU_2013, CosmicRayGreyConfig, CosmicRayGreyParams)

import cr_sedov_taylor as item7
import _cwb_setup


def build_sedov_pe(num_cells, zeta, p_ambient=item7.P_AMBIENT, kr13=False, t_end=item7.T_END):
    """S-PE: the item-7 point explosion (E = rho_0 = 1, tanh-tapered thermal
    deposit of radius ``R_EXPLOSION`` over ``SMOOTH_CELLS`` cells), constant
    ``zeta`` or KR13 (``kr13``, ``zeta`` then unused)."""
    cr_config = CosmicRayGreyConfig(grey_cosmic_rays=True, diffusive_shock_acceleration=True)
    if kr13:
        cr_config = cr_config._replace(dsa_efficiency_model=DSA_EFFICIENCY_KANG_RYU_2013)
    config = SimulationConfig(
        geometry=CARTESIAN, solver_mode=FINITE_VOLUME, riemann_solver=HLLC, limiter=MINMOD,
        dimensionality=3, num_cells=num_cells, exact_end_time=True, progress_bar=False,
        cosmic_ray_grey_config=cr_config,
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    shape = (num_cells,) * 3
    zeros = jnp.zeros(shape)
    dx = 1.0 / num_cells
    weight = 0.5 * (1.0 - jnp.tanh((helper_data.r - item7.R_EXPLOSION) / (item7.SMOOTH_CELLS * dx)))
    delta_p = item7.E_EXPLOSION * (item7.GAMMA - 1.0) / (jnp.sum(weight) * dx**3)
    state = construct_primitive_state(
        config=config, registered_variables=rv, density=jnp.ones(shape) * item7.RHO_AMBIENT,
        velocity_x=zeros, velocity_y=zeros, velocity_z=zeros, gas_pressure=p_ambient + delta_p * weight,
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=t_end, gamma=item7.GAMMA,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            dsa_efficiency=zeta, dsa_mach_min=item7.DSA_MACH_MIN),
    )
    return state, config, params, helper_data, rv


def build_sedov_ss(num_cells, zeta, p_ambient=item7.P_AMBIENT, r_start=0.15, kr13=False,
                   t_end=item7.T_END, density_floor=0.02, n_sub=4):
    """S-SS / S-CP: the blast started from the self-similar two-fluid state at
    radius ``r_start`` (``cr_sedov_self_similar._run``: cell averages of the
    conserved densities, core density floored, ``v_red = 0``, gamma_cr 4/3), in
    ambient pressure ``p_ambient`` (S-CP: raised). The initial CR content is the
    reference's for ``zeta`` (constant), for 0.211 with KR13 (its strong-shock
    plateau, valid while M >~ 15 at ``r_start``), none for the control."""
    from astronomix.test_setups.reference_solutions.sedov_two_fluid import solve_sedov_two_fluid
    zeta_ref = 0.211 if kr13 else zeta
    reference = (solve_sedov_two_fluid(zeta_ref, convention="gas_rh") if zeta_ref > 0
                 else solve_sedov_two_fluid(0.0))
    t_start = (r_start / reference.xi_0) ** 2.5 * (item7.RHO_AMBIENT / item7.E_EXPLOSION) ** 0.5
    cr_config = CosmicRayGreyConfig(grey_cosmic_rays=True, diffusive_shock_acceleration=True)
    if kr13:
        cr_config = cr_config._replace(dsa_efficiency_model=DSA_EFFICIENCY_KANG_RYU_2013)
    config = SimulationConfig(
        geometry=CARTESIAN, solver_mode=FINITE_VOLUME, riemann_solver=HLLC, limiter=MINMOD,
        dimensionality=3, num_cells=num_cells, exact_end_time=True, progress_bar=False,
        cosmic_ray_grey_config=cr_config,
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    dx = 1.0 / num_cells
    c = (np.arange(num_cells) + 0.5) / num_cells - 0.5
    offsets = np.stack(np.meshgrid(c, c, c, indexing="ij"))
    rho, u, p_th, e_cr = reference.cell_averages(
        offsets, dx, t_start, item7.E_EXPLOSION, item7.RHO_AMBIENT, p_ambient, n_sub=n_sub,
        density_floor=density_floor)
    state = construct_primitive_state(
        config=config, registered_variables=rv, density=jnp.asarray(rho), velocity_x=jnp.asarray(u[0]),
        velocity_y=jnp.asarray(u[1]), velocity_z=jnp.asarray(u[2]), gas_pressure=jnp.asarray(p_th),
    )
    state = state.at[rv.cosmic_ray_e_index].set(
        jnp.maximum(jnp.asarray(e_cr), CosmicRayGreyParams().minimum_e_cr))
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_start=float(t_start), t_end=t_end, gamma=item7.GAMMA,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=4.0 / 3.0, reduced_streaming_speed=0.0, dsa_efficiency=zeta,
            dsa_mach_min=item7.DSA_MACH_MIN),
    )
    return state, config, params, helper_data, rv


def build_cwb(num_cells, zeta):
    """The CWB of ``_cwb_setup.run_cwb`` (defaults) with efficiency ``zeta``."""
    state, config, params, helper_data, rv = _cwb_setup.build_cwb(num_cells, dsa_efficiency=zeta)
    return state, config._replace(progress_bar=False), params, helper_data, rv
