"""Planar oblique shocks (B8.2): per-snapshot injection of the old and the continuous-normal surface
where B8.2's probe disagrees, (2,1,0) M = 10 and (2,1,1) M = 10 at zeta = 0.1.

zeta = 0 runs once (no injection, same gas for both modes); zeta = 0.1 runs once per mode, with
``dsa_continuous_normal_surface`` set as the ``CosmicRayGreyConfig`` default, as the forced
regression does. Per snapshot and mode: D = sum(flux * area weight) dx^2 over the box interior,
the realized probe injection over zeta D, the median Mach number, the walk length and the
surface cells per line along the dominant axis. inj / exact as in the test needs B8.2's front
tracking; D ratios between the modes suffice to locate a difference.

Usage: CR_NUM_CELLS=64 python planar_inj_check2.py
"""

import sys
from pathlib import Path

import autocvd
autocvd.autocvd = lambda *a, **k: [0]  # pinned by CUDA_VISIBLE_DEVICES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "cosmic_rays_grey"))

import numpy as np
import jax.numpy as jnp

from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import CosmicRayGreyConfig
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_mach_and_flux, dsa_shock_finder_kwargs, inject_crs_at_shocks)

import cr_oblique_planar_shocks as b82


def _set_default(flag):
    defaults = list(CosmicRayGreyConfig.__new__.__defaults__)
    fields = CosmicRayGreyConfig._fields
    i = fields.index("dsa_continuous_normal_surface") - (len(fields) - len(defaults))
    defaults[i] = flag
    CosmicRayGreyConfig.__new__.__defaults__ = tuple(defaults)
    CosmicRayGreyConfig._field_defaults["dsa_continuous_normal_surface"] = flag
    assert CosmicRayGreyConfig().dsa_continuous_normal_surface == flag


def _row(run, s, t, flag, interior):
    config, params, rv, hd = run["config"], run["params"], run["rv"], run["helper_data"]
    cr_params = params.cosmic_ray_grey_params
    probe = params._replace(cosmic_ray_grey_params=cr_params._replace(dsa_efficiency=b82.PROBE_ZETA))
    dx = config.grid_spacing
    sj = jnp.asarray(s)
    cr_config = config.cosmic_ray_grey_config._replace(dsa_continuous_normal_surface=flag)
    cfg = config._replace(cosmic_ray_grey_config=cr_config)
    sf = find_shocks_pfrommer(sj, cfg, rv, hd, mach_min=cr_params.dsa_mach_min, **dsa_shock_finder_kwargs(cr_config))
    mach, flux = dsa_mach_and_flux(sj, sf, cfg, probe, rv)
    w = np.asarray(dsa_area_weights(sj, sf, cr_config, rv))
    flux, mach = np.asarray(flux), np.asarray(mach)
    surface = np.asarray(sf.shock_surface_cells)
    inj_cells = surface & (flux > 0) & interior
    d_amount = float(np.sum((flux * w)[interior])) * dx**2
    p_cr = (b82.GAMMA_CR - 1.0) * s[rv.cosmic_ray_e_index]
    c_eff = np.sqrt((b82.GAMMA * s[rv.pressure_index] + b82.GAMMA_CR * p_cr) / s[rv.density_index])
    u = np.stack([s[rv.velocity_index.x], s[rv.velocity_index.y], s[rv.velocity_index.z]])
    dt = float(params.C_cfl * dx / np.max(np.max(np.abs(u), axis=0) + c_eff))
    out = np.asarray(inject_crs_at_shocks(sj, cfg, probe, rv, hd, current_time=t, dt=dt))
    inj = float(((out[rv.cosmic_ray_e_index] - s[rv.cosmic_ray_e_index])[interior]).sum()) * dx**3 / dt
    steps = np.asarray(sf.pre_shock_steps)[inj_cells]
    return d_amount, (f"{'flag' if flag else 'old '}: D {d_amount:.4e} inj/(zD) {inj / (b82.PROBE_ZETA * d_amount):.3f} "
                      f"med M {np.median(mach[inj_cells]):.2f} walk {np.median(steps):.0f}/{steps.max():.0f} "
                      f"cells {int(inj_cells.sum())}")


def main():
    n = b82.NUM_CELLS
    idx = np.arange(n)
    inner = (idx >= b82.MARGIN) & (idx < n - b82.MARGIN)
    interior = inner[:, None, None] & inner[None, :, None] & inner[None, None, :]
    for k, zeta in (((2, 1, 0), 0.0), ((2, 1, 0), 0.1), ((2, 1, 1), 0.1)):
        runs = {}
        for flag in ((False,) if zeta == 0.0 else (False, True)):
            _set_default(flag)
            runs[flag] = b82._run(k, 10.0, zeta)
        _set_default(False)
        base = runs[False]
        for i, t in enumerate(base["times"]):
            if t < b82.MEASURE_FROM * base["t_end"] - 1e-12:
                continue
            d_old, old = _row(base, base["states"][i], t, False, interior)
            flag_run = runs.get(True, base)
            d_new, new = _row(flag_run, flag_run["states"][i], t, True, interior)
            print(f"k={k} zeta={zeta} t={t:.4f} | D flag/old {d_new / d_old:.3f} | {old} | {new}", flush=True)


if __name__ == "__main__":
    main()
