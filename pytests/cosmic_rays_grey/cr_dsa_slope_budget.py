"""
Gamma-ray proxy, step 3 (GAMMA_RAY_PLAN.md Sec. 3): unit check of the
alpha-budget diagnostic ``cr_grey_injection.dsa_injection_rate_by_slope``.

On a stored item-11 state (SNR in a clumpy medium, 128^3, DSA eta = 0.1):
  3a. the binned total equals what ``inject_crs_at_shocks`` actually deposits
      in a tiny dt (float64 copy of the state; the positivity cap cannot bind),
      divided by dt;
  3b. the bin sums equal a numpy re-binning of the same per-cell rates with
      alpha = (r + 2) / (r - 1) and np.histogram-style edges.
The time-integrated budget itself comes from the runs in
/export/scratch/nknoell/gamma_step3 (run_alpha_budget.py).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
jax.config.update("jax_enable_x64", True)

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from astronomix import (
    SimulationConfig, SimulationParams, get_helper_data, get_registered_variables, finalize_config,
)
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import CosmicRayGreyConfig, CosmicRayGreyParams
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_injection_rate_by_slope, dsa_mach_and_flux, dsa_shock_finder_kwargs,
    inject_crs_at_shocks,
)
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

STATE_FILE = Path("/export/scratch/nknoell/phaseB/item11_db8/data/N128_clumpy_eff0.1.npz")
SLOPE_EDGES = (2.05, 2.1, 2.15, 2.2, 2.3, 2.5, 3.0)
GAMMA, DSA_EFF, DSA_MACH_MIN = 5.0 / 3.0, 0.1, 1.3  # item 11

TOTAL_REL_TOL = 1e-6
BIN_REL_TOL = 1e-10


def _setup():
    z = np.load(STATE_FILE)
    state = jnp.asarray(z["state"], dtype=jnp.float64)
    config = SimulationConfig(
        geometry=CARTESIAN, solver_mode=FINITE_VOLUME, riemann_solver=HLLC, limiter=MINMOD,
        dimensionality=3, box_size=float(z["box_size"]), num_cells=int(z["num_cells"]),
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True, diffusive_shock_acceleration=True),
    )
    config = finalize_config(config, state.shape)
    rv = get_registered_variables(config)
    hd = get_helper_data(config)
    params = SimulationParams(gamma=GAMMA, cosmic_ray_grey_params=CosmicRayGreyParams(
        dsa_efficiency=DSA_EFF, dsa_mach_min=DSA_MACH_MIN))
    return state, config, params, rv, hd


def test_budget_total_matches_injection():
    state, config, params, rv, hd = _setup()
    rates, total = dsa_injection_rate_by_slope(state, config, params, rv, hd, SLOPE_EDGES)
    dt = 1e-9
    new = inject_crs_at_shocks(state, config, params, rv, hd, 1.0, dt)
    dv = config.grid_spacing ** 3
    injected_rate = float(jnp.sum(new[rv.cosmic_ray_e_index] - state[rv.cosmic_ray_e_index]) * dv / dt)
    rel = abs(injected_rate / float(total) - 1.0)
    rel_sum = abs(float(jnp.sum(rates)) / float(total) - 1.0)
    print(f"  3a  total {float(total):.6e}, injected/dt {injected_rate:.6e}, rel {rel:.1e}; "
          f"sum(bins)/total - 1 = {rel_sum:.1e}")
    assert rel < TOTAL_REL_TOL and rel_sum < 1e-12, (rel, rel_sum)
    return rel, np.asarray(rates), float(total)


def test_binning_vs_numpy():
    state, config, params, rv, hd = _setup()
    rates, _ = dsa_injection_rate_by_slope(state, config, params, rv, hd, SLOPE_EDGES)
    cr_config = config.cosmic_ray_grey_config
    sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=DSA_MACH_MIN, **dsa_shock_finder_kwargs(cr_config))
    mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
    w = dsa_area_weights(state, sf, cr_config, rv)
    mach, flux, w = map(np.asarray, (mach, flux, w))
    sel = mach > 0
    m2 = mach[sel] ** 2
    r = (GAMMA + 1) * m2 / ((GAMMA - 1) * m2 + 2)
    alpha = (r + 2) / (r - 1)
    per_cell = DSA_EFF * flux[sel] * w[sel] * config.grid_spacing ** 2
    edges = np.concatenate([[-np.inf], SLOPE_EDGES, [np.inf]])
    ref = np.array([per_cell[(alpha >= lo) & (alpha < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
    rel = np.max(np.abs(np.asarray(rates) - ref) / max(ref.sum(), 1e-300))
    print(f"  3b  bins vs numpy: max |diff| / total = {rel:.1e}; {sel.sum()} injecting cells")
    assert rel < BIN_REL_TOL, rel
    return rel


if __name__ == "__main__":
    r1, rates, total = test_budget_total_matches_injection()
    r2 = test_binning_vs_numpy()
    labels = ["<2.05"] + [f"[{a},{b})" for a, b in zip(SLOPE_EDGES[:-1], SLOPE_EDGES[1:])] + [">=3.0"]
    print("  instantaneous rate share by alpha (item 11 clumpy, 1 kyr, 128^3):")
    for lab, x in zip(labels, rates / total):
        print(f"    {lab:12s} {x:.3f}")
