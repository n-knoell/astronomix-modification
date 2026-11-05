"""Extension-D gallery extraction (GPU): compact fields of the diffusion runs for ``gallery_d_plot.py``.

For the final state of each run (W ``wind/N<n>/{adv,lam*}``, Sedov ``{ss,cp_p0.6}/N<n>/{z0.1,kr13}[_lam*]``;
the advective Sedov columns are the existing study runs) the finder runs once more with the DSA settings
(``gallery_extract.extract``: mid-plane slices of rho, p, e_cr, |u|; every injecting surface cell with
position, Mach, area weight and nominal injection) and the shell profiles (dx / 2 bins) of rho, p, e_cr
are added. Output ``RAW/gallery_d/<setup>_<name>_N<n>.npz``.

Usage: CUDA_VISIBLE_DEVICES=GPU-<uuid> python gallery_d_extract.py [--n 192] [--all-n]
"""

import argparse

import common
from common import RAW, require_pinned_gpu, save_npz

import numpy as np

from gallery_extract import extract
from sedov_diag import shell_profiles
from setups import build_sedov_ss
from wind_study import build_wind

LAMS = ("lam0.03", "lam0.1", "lam0.3")
SEDOV = (("ss", "z0.1"), ("cp_p0.6", "z0.1"), ("cp_p0.6", "kr13"))


def last_state(d):
    if "state_last" in d.files:
        return d["state_last"]
    key = max((k for k in d.files if k.startswith("state_")), key=lambda k: int(k.split("_")[1]))
    return d[key]


def profiles(hd, n, dx, rho, p, e_cr):
    r = np.asarray(hd.r).reshape(-1)
    centres, (rho_m, p_m, ecr_m), counts = shell_profiles(r, [rho.reshape(-1), p.reshape(-1), e_cr.reshape(-1)], dx)
    return dict(prof_r=centres, prof_rho=rho_m, prof_p=p_m, prof_ecr=ecr_m, prof_counts=counts)


def do_wind(n):
    for name in ("adv",) + LAMS:
        f = RAW / "wind" / f"N{n}" / f"{name}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        kappa = None if not np.isfinite(float(d["kappa"])) else float(d["kappa"])
        _, config, params, hd, rv, info = build_wind(n, kappa=kappa, vred=float(d["vred"]))
        box = float(d["box"])
        out, rho, p, e_cr = extract(last_state(d), config, params, rv, hd, origin=-0.5 * box)
        out.update(profiles(hd, n, box / n, rho, p, e_cr))
        out.update(t=float(d["t"][-1]), R_front=float(d["R_front"][-1]), box=box, rho0=info["rho0"], p0=info["p0"],
                   kappa=float(d["kappa"]), lam=0.0 if name == "adv" else float(name[3:]))
        path = save_npz(RAW / "gallery_d" / f"wind_{name}_N{n}.npz", **out)
        print(f"wind {name} N {n}: {out['surf_mach'].size} injecting cells -> {path}", flush=True)


def do_sedov(n):
    for sdir, model in SEDOV:
        for name in (model,) + tuple(f"{model}_{lam}" for lam in LAMS):
            f = RAW / "sedov" / sdir / f"N{n}" / f"{name}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            kappa = float(d["kappa"]) if "kappa" in d.files and np.isfinite(float(d["kappa"])) else None
            vred = float(d["vred"]) if "vred" in d.files else None
            p0 = float(d["p0"])
            r_start = 0.08 if sdir.startswith("cp") else 0.15
            _, config, params, hd, rv = build_sedov_ss(n, 0.1 if model == "z0.1" else 0.0, p_ambient=p0,
                                                       r_start=r_start, kr13=model == "kr13", kappa=kappa, vred=vred)
            out, rho, p, e_cr = extract(last_state(d), config, params, rv, hd, origin=-0.5)
            out.update(profiles(hd, n, 1.0 / n, rho, p, e_cr))
            lam = 0.0 if name == model else float(name.split("_lam")[1])
            out.update(t=float(d["t"][-1]), R_front=float(d["R_front"][-1]), rho0=1.0, p0=p0, lam=lam,
                       kappa=np.nan if kappa is None else kappa)
            path = save_npz(RAW / "gallery_d" / f"{sdir}_{name}_N{n}.npz", **out)
            print(f"{sdir} {name} N {n}: {out['surf_mach'].size} injecting cells -> {path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="*", default=[192])
    args = ap.parse_args()
    print("gpu", require_pinned_gpu(), flush=True)
    for n in args.n:
        if n % 32 == 0:
            do_wind(n)
        do_sedov(n)
