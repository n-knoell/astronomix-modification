"""CWB runs of the resolution study (C0, C1, C3) with per-snapshot diagnostics. GPU only.

The CWB of ``_cwb_setup.run_cwb`` (defaults; ``setups.build_cwb``), integrated in
segments; after each segment ``analyse_cwb_snapshot``:

* winds: mass flux, kinetic and total (kinetic + enthalpy) energy flux through a
  sphere of radius 1.5 injection radii around each star (fixed physical radius,
  0.047 box), against the nominal Mdot and L_w = Mdot v_inf^2 / 2 (the EI wind
  injects only part of Mdot, memory 2026-10; C0 measures whether that depends on N);
* binary axis (mean of the four central cell lines): contact x_c (u_x sign change
  between the stars), the finder's surface cells on it between each star and the
  contact (wind-1 / wind-2 termination shocks: position, DSA Mach number);
* shocks by upstream gas: each injecting surface cell is assigned to wind 1, wind 2
  or ambient by the velocity two cells upstream along the shock direction (free
  wind k moves radially away from star k at ~v_k; ambient gas is at rest):
  nominal injection zeta(M) * flux * w, weighted area, area-weighted Mach quantiles
  per class;
* energies E_th, E_kin, E_cr (total and per side of x_c).

The ``InjectionAccumulator`` (32^3 block map) adds the realized injection per step.

Output ``RAW/cwb/N<n>/<name>.npz`` (arrays over snapshots, axis profiles, the
accumulator, the last state, the code stamp). ``<name>``: ``control`` / ``z<zeta>``.

Extension D (PLAN.md Section 9): ``--kappa`` / ``--vred`` override the CR transport
(default kappa = 1, v_red = 1000); then the centred compression work per step (whole box,
``work_t``, ``work``) is stored too.

Usage: python cwb_study.py --n 64 --zeta 0.1 [--snapshots 20] [--t-first-frac 0.05]
"""

import argparse

import common
from common import RAW, InjectionAccumulator, peak_gpu_bytes, require_pinned_gpu, run_segments, save_npz

import numpy as np
import jax.numpy as jnp

import _cwb_setup as cwb
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_efficiency_of_mach, dsa_mach_and_flux, dsa_shock_finder_kwargs)
from sedov_diag import weighted_quantiles
from setups import build_cwb
from gpu_diag import CwbGeometry, analyse_cwb_snapshot_gpu
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import DSA_EFFICIENCY_KANG_RYU_2013

CLASSES = ("wind1", "wind2", "ambient")
SPHERE_RADIUS = 1.5 / 32  # 1.5 injection radii (box units), the same physical sphere at every N


def wind_fluxes(rho, u, p, gamma, centres, star, radius, dx):
    """Mass, kinetic and total energy flux through the sphere |x - star| = radius:
    the mean flux density over the shell of cells within dx / 2, times 4 pi radius^2
    (the shell's cell count deviates from 4 pi r^2 / dx^2 by up to 5% at 3 cells)."""
    X, Y, Z = centres
    d = np.stack([X - star[0], Y - star[1], Z - star[2]])
    r = np.sqrt(np.sum(d**2, axis=0))
    shell = np.abs(r - radius) < 0.5 * dx
    ur = np.sum(u * d, axis=0) / np.maximum(r, 1e-12)
    v2 = np.sum(u**2, axis=0)
    dv = 4 * np.pi * radius**2 / max(int(shell.sum()), 1)
    mdot = float(np.sum((rho * ur)[shell])) * dv
    lkin = float(np.sum((0.5 * rho * v2 * ur)[shell])) * dv
    ltot = lkin + float(np.sum((gamma / (gamma - 1.0) * p * ur)[shell])) * dv
    return mdot, lkin, ltot


def analyse_cwb_snapshot(state, t, config, params, rv, hd):
    state = jnp.asarray(state)
    n = state.shape[1]
    dx = float(config.grid_spacing)
    gamma = float(params.gamma)
    cr_config, cr_params = config.cosmic_ray_grey_config, params.cosmic_ray_grey_params
    rho = np.asarray(state[rv.density_index], dtype=np.float64)
    p = np.asarray(state[rv.pressure_index], dtype=np.float64)
    e_cr = np.asarray(state[rv.cosmic_ray_e_index], dtype=np.float64)
    u = np.stack([np.asarray(state[i], dtype=np.float64)
                  for i in (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)])
    c = (np.arange(n) + 0.5) * dx - 0.5 * cwb.BOX_SIZE
    centres = np.meshgrid(c, c, c, indexing="ij")
    stars = np.asarray(cwb.STAR_POSITIONS)
    v_inf = (cwb.v_inf1, cwb.v_inf2)
    out = {"t": float(t)}
    for k, star in enumerate(stars, start=1):
        mdot, lkin, ltot = wind_fluxes(rho, u, p, gamma, centres, star, SPHERE_RADIUS, dx)
        out.update({f"mdot{k}": mdot, f"lkin{k}": lkin, f"ltot{k}": ltot})

    # finder with the DSA settings
    sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=cr_params.dsa_mach_min,
                              **dsa_shock_finder_kwargs(cr_config))
    mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
    w = dsa_area_weights(state, sf, cr_config, rv)
    eff = dsa_efficiency_of_mach(cr_config, cr_params)(mach)
    mach, flux, w, eff = (np.asarray(a, dtype=np.float64) for a in (mach, flux, w, eff))
    surface = np.asarray(sf.shock_surface_cells)
    inj = surface & (flux > 0)
    out.update(n_surf=int(surface.sum()), n_inj=int(inj.sum()),
               D=float((flux * w).sum()) * dx**2, inj_nominal=float((eff * flux * w).sum()) * dx**2)

    # binary axis: the four central lines
    j0, j1 = n // 2 - 1, n // 2
    ax_rho = rho[:, j0:j1 + 1, j0:j1 + 1].mean(axis=(1, 2))
    ax_p = p[:, j0:j1 + 1, j0:j1 + 1].mean(axis=(1, 2))
    ax_ux = u[0][:, j0:j1 + 1, j0:j1 + 1].mean(axis=(1, 2))
    ax_ecr = e_cr[:, j0:j1 + 1, j0:j1 + 1].mean(axis=(1, 2))
    ax_surf = surface[:, j0:j1 + 1, j0:j1 + 1].any(axis=(1, 2))
    ax_mach = np.where(surface, mach, 0.0)[:, j0:j1 + 1, j0:j1 + 1].max(axis=(1, 2))
    x = c
    between = (x > stars[0][0]) & (x < stars[1][0])
    sign = np.flatnonzero(between[:-1] & between[1:] & (ax_ux[:-1] > 0) & (ax_ux[1:] <= 0))
    x_c = float(x[sign[0]] + dx * ax_ux[sign[0]] / (ax_ux[sign[0]] - ax_ux[sign[0] + 1])) if sign.size else np.nan
    out["x_contact"] = x_c
    for k, (lo, hi, pick) in enumerate(((stars[0][0], x_c, -1), (x_c, stars[1][0], 0)), start=1):
        cand = np.flatnonzero(ax_surf & (x > lo) & (x < hi)) if np.isfinite(x_c) else np.zeros(0, int)
        if cand.size:
            i = cand[pick]  # the termination shock: the surface cell closest to the contact
            out[f"x_ts{k}"] = float(x[i])
            out[f"mach_ts{k}"] = float(ax_mach[i])
        else:
            out[f"x_ts{k}"], out[f"mach_ts{k}"] = np.nan, np.nan
    out.update(ax_rho=ax_rho, ax_p=ax_p, ax_ux=ax_ux, ax_ecr=ax_ecr, ax_mach=ax_mach)

    # classify injecting cells by the gas two cells upstream along the shock direction
    idx = np.argwhere(inj)
    d_s = np.asarray(sf.shock_direction, dtype=np.float64)[:, inj].T  # (cells, 3)
    up = np.clip(np.rint(idx + 2.0 * d_s).astype(int), 0, n - 1)
    u_up = u[:, up[:, 0], up[:, 1], up[:, 2]].T
    pos = np.stack([centres[a][inj] for a in range(3)], axis=1)
    speed = np.linalg.norm(u_up, axis=1)
    cos = []
    for star in stars:
        rel = pos - star
        cos.append(np.sum(u_up * rel, axis=1) / np.maximum(speed * np.linalg.norm(rel, axis=1), 1e-30))
    cls = np.where(np.asarray(cos[0]) >= np.asarray(cos[1]), 0, 1)
    cls = np.where(speed < 0.3 * min(v_inf), 2, cls)
    e_inj = (eff * flux * w)[inj] * dx**2
    for k, name in enumerate(CLASSES):
        sel = cls == k
        out[f"inj_{name}"] = float(e_inj[sel].sum())
        out[f"A_{name}"] = float(w[inj][sel].sum()) * dx**2
        q = weighted_quantiles(mach[inj][sel], w[inj][sel], (0.16, 0.5, 0.84))
        out[f"M16_{name}"], out[f"M50_{name}"], out[f"M84_{name}"] = q

    # energies
    dv = dx**3
    side1 = (centres[0] <= x_c) if np.isfinite(x_c) else np.ones_like(rho, bool)
    out.update(E_th=float(p.sum()) / (gamma - 1.0) * dv, E_kin=float((0.5 * rho * np.sum(u**2, axis=0)).sum()) * dv,
               E_cr=float(e_cr.sum()) * dv, E_cr_side1=float(e_cr[side1].sum()) * dv,
               E_cr_side2=float(e_cr[~side1].sum()) * dv)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--zeta", type=float, default=cwb.DSA_EFFICIENCY)
    ap.add_argument("--snapshots", type=int, default=20)
    ap.add_argument("--t-first-frac", type=float, default=0.05)
    ap.add_argument("--kr13", action="store_true", help="Kang & Ryu (2013) zeta(M) (C4; --zeta then unused)")
    ap.add_argument("--cfl", type=float, default=None, help="C_cfl override")
    ap.add_argument("--name", default=None)
    ap.add_argument("--kappa", type=float, default=None, help="CR diffusion coefficient (extension D)")
    ap.add_argument("--vred", type=float, default=None, help="reduced_streaming_speed (extension D)")
    ap.add_argument("--advective", action="store_true",
                    help="CRs advected with the gas: no F_cr relaxation, v_red 0 (G7 gallery)")
    ap.add_argument("--gpu-diag", action="store_true",
                    help="per-snapshot diagnostics on the GPU (gpu_diag.py) instead of host numpy")
    args = ap.parse_args()
    gpu = require_pinned_gpu()
    transport = args.kappa is not None or args.vred is not None or args.advective
    InjectionAccumulator.install(
        coarse_cells=32, work_bins=(args.n, np.zeros(args.n**3, int), 1) if transport else None)
    state, config, params, hd, rv = build_cwb(args.n, args.zeta if not args.kr13 else cwb.DSA_EFFICIENCY,
                                              kappa=args.kappa, vred=args.vred,
                                              advective=args.advective)
    if args.kr13:
        cr = config.cosmic_ray_grey_config._replace(dsa_efficiency_model=DSA_EFFICIENCY_KANG_RYU_2013)
        config = config._replace(cosmic_ray_grey_config=cr)
    if args.cfl is not None:
        params = params._replace(C_cfl=args.cfl)
    t_end = float(params.t_end)
    times = np.linspace(args.t_first_frac * t_end, t_end, args.snapshots)
    name = args.name or ("kr13" if args.kr13 else (f"z{args.zeta:g}" if args.zeta > 0 else "control"))
    print(f"CWB N {args.n} {name}: t_end {t_end:.4g}, {len(times)} snapshots, gpu {gpu}", flush=True)
    acc = InjectionAccumulator().activate()
    rows, walls = [], []
    last = {}
    if args.gpu_diag:
        geom = CwbGeometry(args.n, float(config.grid_spacing), cwb.BOX_SIZE, np.asarray(cwb.STAR_POSITIONS),
                           SPHERE_RADIUS)

    def on_snapshot(i, t, s, wall):
        if args.gpu_diag:
            if not bool(jnp.all(jnp.isfinite(s))):
                raise FloatingPointError(f"non-finite state at t = {t}")
            d = analyse_cwb_snapshot_gpu(s, t, config, params, rv, hd, geom, np.asarray(cwb.STAR_POSITIONS),
                                         (cwb.v_inf1, cwb.v_inf2), SPHERE_RADIUS, CLASSES)
        else:
            if not np.all(np.isfinite(np.asarray(s))):
                raise FloatingPointError(f"non-finite state at t = {t}")
            d = analyse_cwb_snapshot(s, t, config, params, rv, hd)
        rows.append(d)
        walls.append(wall)
        if i == len(times) - 1:
            last["state"] = np.asarray(s)
        print(f"[{i + 1}/{len(times)}] t/t_end {t / t_end:.3f} mdot {d['mdot1']:.4g} {d['mdot2']:.4g} "
              f"x_ts1 {d['x_ts1']:.4f} x_c {d['x_contact']:.4f} x_ts2 {d['x_ts2']:.4f} "
              f"M_ts {d['mach_ts1']:.2f} {d['mach_ts2']:.2f} inj w1/w2/amb {d['inj_wind1']:.3g} "
              f"{d['inj_wind2']:.3g} {d['inj_ambient']:.3g} E_cr {d['E_cr']:.4g} acc {acc.total():.4g} "
              f"wall {wall:.1f} s peak {peak_gpu_bytes() / 1e9:.2f} GB", flush=True)

    run_segments(state, config, params, rv, times, on_snapshot)
    out = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    acc_t, acc_dt, acc_inj = acc.series()
    out.update(acc_t=acc_t, acc_dt=acc_dt, acc_inj=acc_inj, acc_coarse=acc.coarse if acc.coarse is not None else 0,
               wall=np.asarray(walls), N=args.n, zeta=args.zeta, t_end=t_end, gamma=float(params.gamma),
               mdot_nominal=np.array([cwb.mlr1, cwb.mlr2]), v_inf=np.array([cwb.v_inf1, cwb.v_inf2]),
               stars=np.asarray(cwb.STAR_POSITIONS), r_inj=(args.n // 32) / args.n, sphere_radius=SPHERE_RADIUS,
               x_axis=(np.arange(args.n) + 0.5) / args.n - 0.5, peak_gpu_GB=peak_gpu_bytes() / 1e9, gpu=gpu,
               state_last=last["state"],
               kappa=float(params.cosmic_ray_grey_params.diffusion_coefficient),
               vred=float(params.cosmic_ray_grey_params.reduced_streaming_speed))
    if transport:
        work_t, work = acc.work_series()
        if work is not None:
            out.update(work_t=work_t, work=work[:, 0])
    path = save_npz(RAW / "cwb" / f"N{args.n}" / f"{name}.npz", **out)
    print(f"saved {path}: injected {acc.total():.5g}, E_cr {rows[-1]['E_cr']:.5g}, wall {sum(walls):.0f} s",
          flush=True)


if __name__ == "__main__":
    main()
