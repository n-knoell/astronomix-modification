"""Which setting makes the coordinate-plane Mach bands of ``sedov_mach3d_test256.png``?

Swaps one difference at a time between the two figure sets: the Mach estimator ("pressure", the
test's ``_sedov_setup`` finder call, vs "velocity", the DSA call ``dsa_shock_finder_kwargs``) and
the gas state (hydro test run N = 256, t = 0.1, from ``optin_tests.py``; B7.3 CR control run
N = 160, t = 0.07). Both finder modes each. Prints, per case, the median Mach on the
coordinate-plane cells (one coordinate within 1 cell of the centre) over the rest, and the median
in the axis / face-diagonal / body-diagonal directions over the overall median.

Usage: CR_NUM_CELLS=160 CR_B73_REUSE=1 CR_PHASEB_RAW=<dir> python sedov_mach_crosscheck.py
"""

import sys
from pathlib import Path

import autocvd
autocvd.autocvd = lambda *a, **k: [0]  # pinned by CUDA_VISIBLE_DEVICES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "cosmic_rays_grey"))

import numpy as np
import jax.numpy as jnp

from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD, SimulationConfig
from astronomix import construct_primitive_state, finalize_config, get_helper_data, get_registered_variables
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import CosmicRayGreyConfig
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_shock_finder_kwargs

from cr_sedov_injection_rate import _run_with_snapshots

OPTIN = Path("/export/scratch/nknoell/finder_v2_comparison/optin")
TEST_KWARGS = dict(mach_sampling_adaptive=True, mach_sampling_steps=15, mach_sampling_extend=True)


def _test_state():
    d = np.load(OPTIN / "sedov_default.npz")
    n = d["density"].shape[0]
    config = SimulationConfig(geometry=CARTESIAN, solver_mode=FINITE_VOLUME, riemann_solver=HLLC,
                              limiter=MINMOD, dimensionality=3, num_cells=n, exact_end_time=True)
    hd = get_helper_data(config)
    rv = get_registered_variables(config)
    state = construct_primitive_state(config=config, registered_variables=rv, density=jnp.asarray(d["density"]),
                                      velocity_x=jnp.asarray(d["vx"]), velocity_y=jnp.asarray(d["vy"]),
                                      velocity_z=jnp.asarray(d["vz"]), gas_pressure=jnp.asarray(d["pressure"]))
    return state, finalize_config(config, state.shape), rv, hd, 1.3


def _b73_state():
    run = _run_with_snapshots(0.0)
    params = run["params"]
    return (jnp.asarray(run["states"][-1]), run["config"], run["registered_variables"], run["helper_data"],
            params.cosmic_ray_grey_params.dsa_mach_min)


def _pattern(surface, mach):
    n = surface.shape[0]
    idx = np.argwhere(surface)
    m = mach[surface]
    ok = m > 0
    pos = (idx[ok] + 0.5) / n - 0.5
    m = m[ok]
    u = np.sort(np.abs(pos) / np.linalg.norm(pos, axis=1)[:, None], axis=1)  # ascending components
    plane = np.min(np.abs(pos), axis=1) < 1.0 / n
    med = np.median(m)
    axis = u[:, 2] > 0.97
    face = (u[:, 0] < 0.15) & (np.abs(u[:, 1] - u[:, 2]) < 0.1)
    body = np.all(np.abs(u - 1 / np.sqrt(3)) < 0.06, axis=1)
    return (np.median(m[plane]) / np.median(m[~plane]), np.median(m[axis]) / med,
            np.median(m[face]) / med, np.median(m[body]) / med)


def main():
    cr_kwargs = dsa_shock_finder_kwargs(CosmicRayGreyConfig())
    for name, loader in (("test run N256 t=0.1", _test_state), ("B7.3 CR control N160 t=0.07", _b73_state)):
        state, config, rv, hd, mach_min = loader()
        for est_name, kwargs in (("pressure (test call)", TEST_KWARGS), ("velocity (DSA call)", cr_kwargs)):
            for mode in (False, True):
                sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=mach_min, continuous_normal=mode, **kwargs)
                plane, axis, face, body = _pattern(np.asarray(sf.shock_surface_cells), np.asarray(sf.mach_numbers))
                print(f"{name:28s} {est_name:22s} {'flag' if mode else 'default':8s} plane/rest {plane:.4f}  "
                      f"axis {axis:.4f} face-diag {face:.4f} body-diag {body:.4f}", flush=True)


if __name__ == "__main__":
    main()
