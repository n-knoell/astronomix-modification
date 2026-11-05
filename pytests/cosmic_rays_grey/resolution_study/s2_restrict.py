"""S2 (+ S1 orientation): finder on restricted fine states vs native coarse runs. GPU only.

For one setup (``--setup pe|ss|cp``), on the snapshots whose full states the
S1 control runs kept (indices 9, 14, 19, 24, 29; the same times at every N):

* restriction test (PLAN.md principle 5): block-average the ``--fine`` (192)
  state by 2 / 3 / 4 / 6 onto 96 / 64 / 48 / 32 and run ``sedov_diag`` there.
  Native coarse vs restricted fine at the same N = the hydro's share of the
  finder error; restricted fine vs the reference = the finder's own share.
* orientation of the dissipated flux for every kept state, native and
  restricted: D per unit exact area in each |n|_inf bin over its shell mean
  (the control runs inject nothing, so ``sedov_diag``'s injection-based
  version is empty there).

Output ``RAW/sedov/<subdir>/S2_control.npz``: one row per (kind, N, snapshot),
``kind`` = native / restricted, with every ``sedov_diag`` scalar, the Mach and
flux per |n|_inf bin, and the code stamp.

Usage: python s2_restrict.py --setup cp [--fine 192]
"""

import argparse

import common
from common import RAW, require_pinned_gpu, restrict, save_npz

import numpy as np
import jax.numpy as jnp

import cr_sedov_taylor as item7
from astronomix import get_helper_data
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_mach_and_flux, dsa_shock_finder_kwargs)
from sedov_diag import NINF_EDGES, SPHERE_BIN_FRACTIONS, analyse_snapshot
from setups import build_sedov_pe, build_sedov_ss

SUBDIR = {"pe": "pe", "ss": "ss", "cp": "cp_p0.6"}
KEPT = (9, 14, 19, 24, 29)


def setup_for(setup, n):
    """Config / params / helper data / registered variables of a control run at N (state unused)."""
    if setup == "pe":
        _, config, params, hd, rv = build_sedov_pe(n, 0.0)
    else:
        p0, r0 = (0.6, 0.08) if setup == "cp" else (item7.P_AMBIENT, 0.15)
        _, config, params, hd, rv = build_sedov_ss(n, 0.0, p_ambient=p0, r_start=r0, n_sub=1)
    return config, params, hd, rv


def flux_orientation(state, config, params, rv, hd, R):
    """Dissipated flux per unit exact area per |n|_inf bin, over its shell mean."""
    dx = float(config.grid_spacing)
    cr_config = config.cosmic_ray_grey_config
    sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=params.cosmic_ray_grey_params.dsa_mach_min,
                              **dsa_shock_finder_kwargs(cr_config))
    _, flux = dsa_mach_and_flux(state, sf, config, params, rv)
    w = dsa_area_weights(state, sf, cr_config, rv)
    e = np.asarray(flux * w, dtype=np.float64) * dx**2
    n = state.shape[1]
    c = (np.arange(n) + 0.5) * dx - 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    r = np.maximum(np.sqrt(X**2 + Y**2 + Z**2), 1e-12)
    b = np.digitize(np.max(np.abs(np.stack([X, Y, Z])), axis=0) / r, NINF_EDGES[1:-1])
    e_bin = np.array([e[b == k].sum() for k in range(5)])
    area = SPHERE_BIN_FRACTIONS * 4 * np.pi * R**2
    return (e_bin / area) / (e_bin.sum() / area.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=list(SUBDIR))
    ap.add_argument("--fine", type=int, default=192)
    args = ap.parse_args()
    print("GPU", require_pinned_gpu(), flush=True)
    base = RAW / "sedov" / SUBDIR[args.setup]
    runs = {int(p.parent.name[1:]): np.load(p) for p in sorted(base.glob("N*/control.npz"))}
    rows = []

    def record(kind, n, i, t, state, config, params, hd, rv):
        d = analyse_snapshot(state, t, config, params, rv, hd, rho0=item7.RHO_AMBIENT, gamma=item7.GAMMA)
        d["flux_per_area_ninf_bin"] = flux_orientation(state, config, params, rv, hd, d["R_front"])
        d.update(kind=kind, N=n, snap=i)
        rows.append(d)
        print(f"{kind:10s} N {n:3d} snap {i:2d} t {t:.4g} R/dx {d['R_front'] * n:6.1f} M50 {d['M50']:.3f} "
              f"A_inj {d['A_inj']:.4f} D {d['D']:.4g}", flush=True)

    # native kept states, every N
    for n, run in runs.items():
        config, params, hd, rv = setup_for(args.setup, n)
        for i in KEPT:
            if f"state_{i}" in run.files:
                record("native", n, i, float(run["t"][i]), jnp.asarray(run[f"state_{i}"]), config, params, hd, rv)
    # restricted fine states
    fine = runs[args.fine]
    gamma = float(fine["gamma"])
    for factor in (2, 3, 4, 6):
        n = args.fine // factor
        config, params, hd, rv = setup_for(args.setup, n)
        for i in KEPT:
            coarse = restrict(jnp.asarray(fine[f"state_{i}"]), rv, gamma, factor)
            record("restricted", n, i, float(fine["t"][i]), coarse, config, params, hd, rv)

    keys = [k for k in rows[0] if k != "kind"]
    out = {k: np.asarray([r[k] for r in rows]) for k in keys}
    out["kind"] = np.asarray([r["kind"] for r in rows])
    path = save_npz(base / "S2_control.npz", fine=args.fine, **out)
    print(f"saved {path}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
