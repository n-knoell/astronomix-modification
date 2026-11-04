"""Planar oblique shocks (B8.2, ``cr_oblique_planar_shocks._run``): where does the continuous-normal
surface's extra injection at the (1,1,1) M = 10 shock come from? (walk: median / max pre-shock steps)

Per late snapshot and mode: D = sum(flux * area weight) dx^2 (the amount the injection starts
from, ``dsa_mach_and_flux`` x ``dsa_area_weights``, valid cells), the realized one-step injection
``inj`` of ``inject_crs_at_shocks`` (zeta = 0.1 probe, box interior) and inj / (zeta D), plus the
number of surface cells per grid line along the dominant axis.

Usage: CR_NUM_CELLS=64 python planar_inj_check.py
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
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_mach_and_flux, dsa_shock_finder_kwargs, inject_crs_at_shocks)

import cr_oblique_planar_shocks as b82

PROBE_ZETA = b82.PROBE_ZETA


def main():
    n = b82.NUM_CELLS
    idx = np.arange(n)
    inner = (idx >= b82.MARGIN) & (idx < n - b82.MARGIN)
    interior = inner[:, None, None] & inner[None, :, None] & inner[None, None, :]
    for k in ((1, 1, 1), (1, 0, 0), (2, 1, 1)):
        run = b82._run(k, 10.0, 0.0)
        config, params, rv, hd = run["config"], run["params"], run["rv"], run["helper_data"]
        cr_params = params.cosmic_ray_grey_params
        probe = params._replace(cosmic_ray_grey_params=cr_params._replace(dsa_efficiency=PROBE_ZETA))
        dx = config.grid_spacing
        for t, s in zip(run["times"], run["states"]):
            if t < b82.MEASURE_FROM * run["t_end"] - 1e-12:
                continue
            sj = jnp.asarray(s)
            row = []
            for flag in (False, True):
                cr_config = config.cosmic_ray_grey_config._replace(dsa_continuous_normal_surface=flag)
                cfg = config._replace(cosmic_ray_grey_config=cr_config)
                sf = find_shocks_pfrommer(sj, cfg, rv, hd, mach_min=cr_params.dsa_mach_min,
                                          **dsa_shock_finder_kwargs(cr_config))
                mach, flux = dsa_mach_and_flux(sj, sf, cfg, probe, rv)
                w = np.asarray(dsa_area_weights(sj, sf, cr_config, rv))
                flux = np.asarray(flux)
                surface = np.asarray(sf.shock_surface_cells)
                d_amount = float(np.sum((flux * w)[interior])) * dx**2
                c_ = np.sqrt((b82.GAMMA * s[rv.pressure_index]) / s[rv.density_index])
                u = np.stack([s[rv.velocity_index.x], s[rv.velocity_index.y], s[rv.velocity_index.z]])
                dt = float(params.C_cfl * dx / np.max(np.max(np.abs(u), axis=0) + c_))
                out = np.asarray(inject_crs_at_shocks(sj, cfg, probe, rv, hd, current_time=t, dt=dt))
                inj = float(((out[rv.cosmic_ray_e_index] - s[rv.cosmic_ray_e_index])[interior]).sum()) * dx**3 / dt
                dom = int(np.argmax(np.abs(k)))
                per_line = surface.sum(axis=dom)
                lines = per_line[per_line > 0]
                row.append(f"{'flag' if flag else 'old '}: cells {int(surface[interior].sum()):5d} "
                           f"per line {lines.mean():.2f} D {d_amount:.4e} inj/(zD) {inj / (PROBE_ZETA * d_amount):.3f} "
                           f"med M {np.median(np.asarray(mach)[surface & (flux > 0)]):.2f} "
                           f"walk {np.median(np.asarray(sf.pre_shock_steps)[surface & (flux > 0)]):.0f}/"
                           f"{np.max(np.asarray(sf.pre_shock_steps)[surface & (flux > 0)]):.0f}")
            print(f"k={k} t={t:.4f} | " + " | ".join(row), flush=True)


if __name__ == "__main__":
    main()
