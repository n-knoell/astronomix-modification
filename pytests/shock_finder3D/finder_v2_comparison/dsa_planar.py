"""Area weights at planar oblique shocks (exact area known): ours vs v2-surface pipelines.

Reuses B8.2's runs (``cr_oblique_planar_shocks._run``: periodic box, planar
front with normal k / |k|, gas-only shock of Mach 3 or 10, zeta = 0, N = 64)
and its measuring region (within 0.08 in theta of the front, MARGIN cells from
the faces), on the snapshots 0.4 t_end <= t <= 0.8 t_end (at t_end the upstream plateau is
gone for the short-period orientations). Per pipeline:

* ``area``: sum(w) dx^2 over the fronts' true area in the region;
* ``D``: sum(flux w) dx^2 over delta(M) rho (M c)^3 / 2 x true area;
* ``inj`` (ours and hybrid + column weight only): one complete jitted
  ``inject_crs_at_shocks`` probe (zeta = 0.1), summed over the region, over
  zeta x the exact flux x true area, i.e. B8.2's end-to-end number.

Usage: python dsa_planar.py   (results to /export/scratch/nknoell/finder_v2_comparison/dsa/planar.npz)
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "cosmic_rays_grey"))

import numpy as np
import jax
import jax.numpy as jnp

import cr_oblique_planar_shocks as b82
from astronomix._modules._cosmic_rays_grey import cr_grey_injection
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_mach_and_flux, dsa_shock_finder_kwargs
from astronomix.shock_finder3D._energy_dissipation import thermal_energy_flux_from_pre_state
from astronomix.shock_finder3D._surface_area import surface_area_weights
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

from compare_finders import find_v2
from dsa_physics import column_area_weights, find_shocks_hybrid
from dsa_timing import _patched

OUT = Path("/export/scratch/nknoell/finder_v2_comparison/dsa/planar.npz")
PROBE_ZETA = 0.1
LAST_FRACTION = 0.85


def analyse(k, mach):
    run = b82._run(k, mach, 0.0)
    config, params, rv, hd = run["config"], run["params"], run["rv"], run["helper_data"]
    theta, kabs = run["theta"], run["kabs"]
    n, dx = b82.NUM_CELLS, config.grid_spacing
    cr_params = params.cosmic_ray_grey_params
    probe = params._replace(cosmic_ray_grey_params=cr_params._replace(dsa_efficiency=PROBE_ZETA))
    f_exact = float(thermal_energy_flux_from_pre_state(mach, 1.0 / b82.GAMMA, 1.0, b82.GAMMA))
    idx = np.arange(n)
    inner = (idx >= b82.MARGIN) & (idx < n - b82.MARGIN)
    interior = inner[:, None, None] & inner[None, :, None] & inner[None, None, :]

    raw_inject = cr_grey_injection.inject_crs_at_shocks.__wrapped__
    inject_ours = jax.jit(lambda s, dt: raw_inject(s, config, probe, rv, hd, 0.0, dt))
    inject_hyb = jax.jit(lambda s, dt: raw_inject(s, config, probe, rv, hd, 0.0, dt))

    out = {}
    for t, s in zip(run["times"], run["states"]):
        # at t_end the next periodic slab's hot gas sits 1-2 cells ahead of the
        # front for |k| >= sqrt(5) (no clean upstream plateau): use 0.4-0.8 t_end
        if t < b82.MEASURE_FROM * run["t_end"] - 1e-12 or t > LAST_FRACTION * run["t_end"]:
            continue
        sj = jnp.asarray(s)
        p, rho = sj[rv.pressure_index], sj[rv.density_index]
        th_f = b82._front_theta(theta, s[rv.density_index], kabs)
        region = interior & (np.abs(np.mod(theta - th_f + 0.5, 1.0) - 0.5) < b82.WINDOW)
        true_area = b82._plane_area_in_cube(run["k"], th_f, b82.MARGIN * dx, (n - b82.MARGIN) * dx)

        sf = find_shocks_pfrommer(sj, config, rv, hd, mach_min=b82.DSA_MACH_MIN,
                                  **dsa_shock_finder_kwargs(config.cosmic_ray_grey_config))
        _, flux_o = dsa_mach_and_flux(sj, sf, config, params, rv)
        sh = find_shocks_hybrid(sj, config, rv, hd, mach_min=b82.DSA_MACH_MIN)
        _, flux_h = dsa_mach_and_flux(sj, sh, config, params, rv)
        v2d = find_v2(sj, config, rv, hd, mach_min=b82.DSA_MACH_MIN)
        v2p = find_v2(sj, config, rv, hd, mach_min=b82.DSA_MACH_MIN, profile_aware_mach=True)
        w_ninf = surface_area_weights(p, rho)
        pipelines = {
            "ours_ninf": (sf.shock_surface_cells, flux_o, w_ninf),
            "hyb_unit": (sh.shock_surface_cells, flux_h, jnp.ones_like(w_ninf)),
            "hyb_ninf": (sh.shock_surface_cells, flux_h, w_ninf),
            "hyb_col": (sh.shock_surface_cells, flux_h,
                        column_area_weights(p, rho, sh.shock_surface_cells, sh.shock_zones)),
            "v2_col": (v2d.shock_surface_cells, v2d.thermal_energy_flux,
                       column_area_weights(p, rho, v2d.shock_surface_cells, v2d.shock_zones)),
            "v2prof_col": (v2p.shock_surface_cells, v2p.thermal_energy_flux,
                           column_area_weights(p, rho, v2p.shock_surface_cells, v2p.shock_zones)),
        }
        for name, (surface, flux, w) in pipelines.items():
            surface, flux, w = (np.asarray(a) for a in (surface, flux, w))
            sel = surface & region
            out.setdefault(f"{name}__area", []).append(float(w[sel].sum()) * dx**2 / true_area)
            out.setdefault(f"{name}__D", []).append(
                float((flux * w)[region].sum()) * dx**2 / (f_exact * true_area))

        u = np.stack([s[rv.velocity_index.x], s[rv.velocity_index.y], s[rv.velocity_index.z]])
        c = np.sqrt(b82.GAMMA * s[rv.pressure_index] / s[rv.density_index])
        dt = float(params.C_cfl * dx / np.max(np.max(np.abs(u), axis=0) + c))
        for name, fn, patch in (("ours_ninf", inject_ours, False), ("hyb_col", inject_hyb, True)):
            go = lambda f=fn: np.asarray(f(sj, dt))
            injected = _patched(go) if patch else go()
            delta = (injected[rv.cosmic_ray_e_index] - s[rv.cosmic_ray_e_index])[interior]
            out.setdefault(f"{name}__inj", []).append(
                float(delta.sum()) * dx**3 / dt / (PROBE_ZETA * f_exact * true_area))
    return {key: float(np.mean(v)) for key, v in out.items()}


def main():
    results = {}
    for k in b82.ORIENTATIONS:
        for mach in b82.MACHS:
            res = analyse(k, mach)
            label = f"{b82._label(k)}_M{mach:g}"
            for key, value in res.items():
                results[f"{label}__{key}"] = value
            print(f"[{label}] " + ", ".join(f"{key} {value:.3f}" for key, value in res.items()), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, **results)


if __name__ == "__main__":
    main()
