"""DSA Mach numbers of the Sedov shell, default vs ``dsa_continuous_normal_surface``.

On the saved B7.3 control snapshots (``cr_sedov_injection_rate.py``, zeta = 0, so the gas is the
same for both modes) the production DSA finder runs in both modes; per injecting cell this saves
the Mach number the injection uses (``dsa_mach_and_flux``: velocity Mach, CR-aware), the flux, the
area weight (``dsa_area_weights``) and the finder's own pressure Mach, at t = 0.007, 0.035 and
0.07, with B7.3's self-similar R_true and M_true. ``plot_sedov_mach_optin.py`` plots them.

Usage: CR_NUM_CELLS=160 CR_B73_REUSE=1 CR_PHASEB_RAW=<dir with states_eff0_N160.npz>
       python sedov_mach_optin.py
"""

import os
import sys
from pathlib import Path

import autocvd
autocvd.autocvd = lambda *a, **k: [0]  # pinned by CUDA_VISIBLE_DEVICES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "cosmic_rays_grey"))

import numpy as np
import jax.numpy as jnp

from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights,
    dsa_mach_and_flux,
    dsa_shock_finder_kwargs,
)

from cr_sedov_injection_rate import NUM_CELLS, RAW_DIR, _run_with_snapshots

OUT_DIR = Path("/export/scratch/nknoell/finder_v2_comparison/sedov_mach")
SNAPSHOTS = (0, 4, 9)  # t = 0.007, 0.035, 0.07


def main():
    assert os.environ.get("CR_B73_REUSE", "0") != "0", "reuse the saved control states"
    run = _run_with_snapshots(0.0)
    config, params = run["config"], run["params"]
    rv, hd = run["registered_variables"], run["helper_data"]
    cr_params = params.cosmic_ray_grey_params
    ref = np.load(RAW_DIR / f"b7_3_N{NUM_CELLS}.npz")
    out = {"num_cells": NUM_CELLS}
    for i_t in SNAPSHOTS:
        t, state = float(run["times"][i_t]), jnp.asarray(run["states"][i_t])
        out[f"t{i_t}_t"] = t
        out[f"t{i_t}_R_true"] = float(ref["control_R_true"][i_t])
        out[f"t{i_t}_M_true"] = float(ref["control_M_true"][i_t])
        for mode, flag in (("default", False), ("hybrid", True)):
            cr_config = config.cosmic_ray_grey_config._replace(dsa_continuous_normal_surface=flag)
            sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=cr_params.dsa_mach_min,
                                      **dsa_shock_finder_kwargs(cr_config))
            mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
            weight = dsa_area_weights(state, sf, cr_config, rv)
            surface = np.asarray(sf.shock_surface_cells)
            key = f"t{i_t}_{mode}"
            out[f"{key}_idx"] = np.argwhere(surface).astype(np.int16)
            out[f"{key}_mach"] = np.asarray(mach)[surface]
            out[f"{key}_mach_finder"] = np.asarray(sf.mach_numbers)[surface]
            out[f"{key}_flux"] = np.asarray(flux)[surface]
            out[f"{key}_weight"] = np.asarray(weight)[surface]
            inj = out[f"{key}_flux"] > 0
            print(f"N={NUM_CELLS} t={t:.3f} {mode}: {surface.sum()} surface, {inj.sum()} injecting, "
                  f"median M/M_true {np.median(out[f'{key}_mach'][inj]) / out[f't{i_t}_M_true']:.3f}",
                  flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"sedov_mach_N{NUM_CELLS}.npz", **out)


if __name__ == "__main__":
    main()
