"""The shock_finder3D Sedov / CWB tests with the opt-in continuous-normal surface.

Runs ``sedov_shock_finder.test_sedov_shock_finder_correctness`` (hydro Sedov, N = 256, finder on
the final state) and the CWB test's ``__main__`` (``_cwb_setup.run_cwb`` at N = 128, CR control
and DSA zeta = 0.1 with live injection, plus ``test_cwb_shock_finder_correctness``'s checks on
the DSA run), either as they are (``default``) or with the finder's ``continuous_normal=True``
and, for the CWB, ``CosmicRayGreyConfig(dsa_continuous_normal_surface=True)`` (``hybrid``).
The tests' own figures go to ``/export/scratch/nknoell/finder_v2_comparison/optin/figures/<case>_<mode>/``,
the final states, finder output and wall times to ``.../optin/<case>_<mode>.npz``
(``plot_optin.py`` compares them).

Usage: python optin_tests.py sedov|cwb default|hybrid
"""

import functools
import json
import sys
import time
import traceback
from pathlib import Path

import autocvd
autocvd.autocvd = lambda *a, **k: [0]

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import jax

from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import CosmicRayGreyConfig
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_area_weights

OUT = Path("/export/scratch/nknoell/finder_v2_comparison/optin")
CWB_CELLS = 128
SEDOV_CELLS = 256


def _patch(setup, hybrid):
    if hybrid:
        setup.find_shocks_pfrommer = functools.partial(find_shocks_pfrommer, continuous_normal=True)
        if hasattr(setup, "CosmicRayGreyConfig"):
            setup.CosmicRayGreyConfig = functools.partial(CosmicRayGreyConfig, dsa_continuous_normal_surface=True)


def _finder_arrays(run):
    sf, rv, s = run["sf_result"], run["registered_variables"], run["state"]
    out = {name: np.asarray(getattr(sf, name)) for name in (
        "shock_surface_cells", "shock_zones", "mach_numbers", "thermal_energy_flux")}
    for name in ("density", "pressure"):
        out[name] = np.asarray(s[getattr(rv, f"{name}_index")])
    for axis in ("x", "y", "z"):
        out[f"v{axis}"] = np.asarray(s[getattr(rv.velocity_index, axis)])
    if (getattr(rv, "cosmic_ray_e_index", None) or -1) >= 0:
        out["e_cr"] = np.asarray(s[rv.cosmic_ray_e_index])
    return out


def _run_test(fn, *args):
    try:
        fn(*args)
        return "PASS"
    except Exception as error:  # report, keep going
        traceback.print_exc()
        return f"FAIL {type(error).__name__}: {str(error)[:300]}"


def sedov(hybrid, tag):
    import _sedov_setup
    _patch(_sedov_setup, hybrid)
    import sedov_shock_finder as test
    test.FIG_DIR = OUT / "figures" / tag
    test.FIG_DIR.mkdir(parents=True, exist_ok=True)
    cache = {}
    original = test.run_sedov

    def run_sedov(n):
        t0 = time.time()
        cache["run"] = original(n)
        jax.block_until_ready(cache["run"]["state"])
        cache["wall"] = time.time() - t0
        return cache["run"]

    test.run_sedov = run_sedov
    result = _run_test(test.test_sedov_shock_finder_correctness, SEDOV_CELLS)
    arrays = _finder_arrays(cache["run"])
    cr_config = CosmicRayGreyConfig(dsa_continuous_normal_surface=hybrid)
    arrays["area_weight"] = np.asarray(dsa_area_weights(
        cache["run"]["state"], cache["run"]["sf_result"], cr_config, cache["run"]["registered_variables"]))
    return {"test": result, "wall_s": cache["wall"]}, arrays


def cwb(hybrid, tag):
    import _cwb_setup
    _patch(_cwb_setup, hybrid)
    import cwb_shock_finder as test
    test.DEFAULT_FIG_DIR = test.CR_FIG_DIR = OUT / "figures" / tag
    test.DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    walls = {}
    runs = {}
    for name, zeta in (("control", 0.0), ("dsa", _cwb_setup.DSA_EFFICIENCY)):
        t0 = time.time()
        runs[name] = test.run_cwb(CWB_CELLS, dsa_efficiency=zeta)
        jax.block_until_ready(runs[name]["state"])
        walls[name] = time.time() - t0
        print(f"[{tag}] {name} run {walls[name]:.0f} s", flush=True)
    control, dsa = runs["control"], runs["dsa"]
    test.plot_shocked_cells_3d(dsa, CWB_CELLS)
    test.plot_shocked_cells_2d(dsa, CWB_CELLS)
    test.plot_cr_influence(control, dsa, CWB_CELLS)
    test.plot_cr_orbital_slice(dsa, CWB_CELLS)
    energies = {name: {k: float(v) for k, v in test._compute_energies(run).items()} for name, run in runs.items()}
    print(f"[{tag}] energies {json.dumps(energies)}", flush=True)
    # the correctness checks, on the DSA run (no third simulation)
    test.run_cwb = lambda n, **k: dsa
    result = _run_test(test.test_cwb_shock_finder_correctness, CWB_CELLS)
    arrays = _finder_arrays(dsa)
    cr_config = dsa["config"].cosmic_ray_grey_config
    arrays["area_weight"] = np.asarray(dsa_area_weights(dsa["state"], dsa["sf_result"], cr_config,
                                                        dsa["registered_variables"]))
    arrays["control_e_cr"] = np.asarray(control["state"][control["registered_variables"].cosmic_ray_e_index])
    return {"test": result, "wall_s": walls, "energies": energies,
            "hybrid_config": bool(cr_config.dsa_continuous_normal_surface)}, arrays


def main():
    case, mode = sys.argv[1], sys.argv[2]
    hybrid = {"default": False, "hybrid": True}[mode]
    tag = f"{case}_{mode}"
    summary, arrays = {"sedov": sedov, "cwb": cwb}[case](hybrid, tag)
    summary["case"], summary["mode"] = case, mode
    print(f"SUMMARY {json.dumps(summary)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{tag}.npz", **arrays)
    (OUT / f"{tag}.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
