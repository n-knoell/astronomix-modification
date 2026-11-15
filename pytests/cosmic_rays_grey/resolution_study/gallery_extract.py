"""Gallery extraction (GPU): compact per-resolution fields for ``gallery_plot.py``.

For the final state of every ζ = 0.1 run (Sedov S-PE N 32-224, CWB N 64-192) the
finder runs once more with the DSA settings, and the following is saved to
``RAW/gallery/<setup>_N<n>.npz``:

* mid-plane slices (z = 0, mean of the two central cell layers) of rho, p, e_cr, |u|;
* shell / axis profiles (Sedov: shell means in dx / 2 bins; CWB: the axis profiles
  are in the study file already);
* every injecting surface cell: position, DSA Mach number, area weight w and
  nominal injection zeta(M) flux w dx^2 (rate).

Usage: CUDA_VISIBLE_DEVICES=GPU-<uuid> python gallery_extract.py [--only sedov|cwb]
"""

import argparse

import common
from common import RAW, require_pinned_gpu, save_npz

import numpy as np
import jax.numpy as jnp

import _cwb_setup as cwb
import cr_sedov_taylor as item7
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_efficiency_of_mach, dsa_mach_and_flux, dsa_shock_finder_kwargs)
from sedov_diag import shell_profiles
from setups import build_cwb, build_sedov_pe

SEDOV_NS = (32, 48, 64, 96, 128, 192, 224, 256, 384, 448, 512)
CWB_NS = (64, 96, 128, 160, 192, 256)  # z0.1 320/384 failed (DB12)


def midplane(field):
    n = field.shape[2]
    return 0.5 * (field[:, :, n // 2 - 1] + field[:, :, n // 2])


def extract(state, config, params, rv, hd, origin):
    state = jnp.asarray(state)
    n = state.shape[1]
    dx = float(config.grid_spacing)
    cr_config, cr_params = config.cosmic_ray_grey_config, params.cosmic_ray_grey_params
    sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=cr_params.dsa_mach_min,
                              **dsa_shock_finder_kwargs(cr_config))
    mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
    w = dsa_area_weights(state, sf, cr_config, rv)
    eff = dsa_efficiency_of_mach(cr_config, cr_params)(mach)
    surface = np.asarray(sf.shock_surface_cells)
    mach, flux, w, eff = (np.asarray(a, dtype=np.float64) for a in (mach, flux, w, eff))
    inj = surface & (flux > 0)
    idx = np.argwhere(inj)
    pos = (idx + 0.5) * dx + origin

    rho = np.asarray(state[rv.density_index], dtype=np.float64)
    p = np.asarray(state[rv.pressure_index], dtype=np.float64)
    e_cr = np.asarray(state[rv.cosmic_ray_e_index], dtype=np.float64)
    u = np.stack([np.asarray(state[i], dtype=np.float64)
                  for i in (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)])
    speed = np.sqrt(np.sum(u**2, axis=0))
    out = dict(
        N=n, dx=dx, origin=origin,
        sl_rho=midplane(rho), sl_p=midplane(p), sl_ecr=midplane(e_cr), sl_speed=midplane(speed),
        surf_pos=pos.astype(np.float32), surf_mach=mach[inj].astype(np.float32),
        surf_w=w[inj].astype(np.float32), surf_inj=(eff * flux * w)[inj].astype(np.float32) * dx**2,
        E_cr=float(e_cr.sum()) * dx**3,
    )
    return out, rho, p, e_cr


def run_sedov(ns):
    for n in ns:
        d = np.load(RAW / "sedov" / "pe" / f"N{n}" / "z0.1.npz")
        key = max((k for k in d.files if k.startswith("state_")), key=lambda k: int(k.split("_")[1]))
        _, config, params, hd, rv = build_sedov_pe(n, 0.1)
        out, rho, p, e_cr = extract(d[key], config, params, rv, hd, origin=-0.5)
        r = np.asarray(hd.r).reshape(-1)
        centres, (rho_m, p_m, ecr_m), counts = shell_profiles(
            r, [rho.reshape(-1), p.reshape(-1), e_cr.reshape(-1)], 1.0 / n)
        out.update(prof_r=centres, prof_rho=rho_m, prof_p=p_m, prof_ecr=ecr_m, prof_counts=counts,
                   t=float(d["t"][-1]), R_front=float(d["R_front"][-1]))
        path = save_npz(RAW / "gallery" / f"sedov_N{n}.npz", **out)
        print(f"sedov N {n}: {out['surf_mach'].size} injecting cells, M50 {np.median(out['surf_mach']):.1f} -> {path}",
              flush=True)


def run_cwb(ns):
    for n in ns:
        d = np.load(RAW / "cwb" / f"N{n}" / "z0.1.npz")
        _, config, params, hd, rv = build_cwb(n, 0.1)
        out, *_ = extract(d["state_last"], config, params, rv, hd, origin=-0.5 * cwb.BOX_SIZE)
        out.update(t=float(d["t"][-1]), stars=np.asarray(d["stars"]), r_inj=float(d["r_inj"]))
        path = save_npz(RAW / "gallery" / f"cwb_N{n}.npz", **out)
        print(f"cwb N {n}: {out['surf_mach'].size} injecting cells -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["sedov", "cwb"], default=None)
    args = ap.parse_args()
    print("gpu", require_pinned_gpu(), flush=True)
    if args.only in (None, "sedov"):
        run_sedov(SEDOV_NS)
    if args.only in (None, "cwb"):
        run_cwb(CWB_NS)
