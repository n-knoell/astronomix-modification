"""Wind-bubble runs of the resolution study, extension D (setup W, PLAN.md Section 9). GPU only.

The DSA run of ``cr_wind_bubble`` (single star, Weaver bubble, zeta = 0.1, box 1.2 = 3.6 pc,
injection radius 4 cells at 128^3, scaled with N so it is fixed physically: N must be a
multiple of 32), integrated in segments (``common.run_segments``). CR transport:
``--vred 0`` (advective anchor) or ``--kappa K --vred V`` (diffusion, F_cr relaxation).

Per snapshot:

* radial shell means (bins of dx) of rho, p, e_cr, v_r (``prof_*``, ``r_bins``);
* forward shock: front radius (half maximum of the shell's density excess, outer flank,
  ``sedov_diag.shock_front``);
* finder with the DSA settings: injecting cells split into the forward shock
  (|r - R_front| <= 4 dx), the termination shock (r < 0.6 R_front) and the rest;
  per class the area-weighted Mach quantiles, weighted area and nominal injection
  zeta(M) flux w dx^2, plus false positives (surface cells in neither class);
* energies E_th, E_kin, E_cr, E_tot.

Per step (``common.InjectionAccumulator``): the realized injection, and the centred
compression work -P_cr div(v) dV dt in radial bins of dx.

Output ``RAW/wind/N<n>/<name>.npz``. Usage:
    python wind_study.py --n 128 --name lam0.1 --kappa 0.03 --vred 100 [--snapshots 24] [--off-after 0.5]
"""

import argparse

import common
from common import RAW, InjectionAccumulator, peak_gpu_bytes, require_pinned_gpu, run_segments, save_npz

import numpy as np
import jax.numpy as jnp

import cr_wind_bubble as m
from astronomix import (CARTESIAN, FINITE_VOLUME, HLLC, MINMOD, NATIVE_JAX, BackendConfig, SimulationConfig,
                        SimulationParams, WindParams, construct_primitive_state, finalize_config,
                        get_helper_data, get_registered_variables)
from astronomix.option_classes import WindConfig
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_efficiency_of_mach, dsa_mach_and_flux, dsa_shock_finder_kwargs)
from sedov_diag import shock_front, weighted_quantiles
from gpu_diag import WindGeometry, analyse_wind_snapshot_gpu

CLASSES = ("fs", "ts", "other")


def build_wind(n, kappa=None, vred=0.0, zeta=m.DSA_EFFICIENCY):
    """The W setup at N^3 (``cr_wind_bubble`` DSA run, injection radius scaled with N, NATIVE_JAX):
    ``(state, config, params, helper_data, registered_variables, info)``; ``info`` holds rho0, p0, L_w."""
    cu = m.CODE_UNITS
    box = m.BOX_SIZE
    config = SimulationConfig(
        progress_bar=False, geometry=CARTESIAN, solver_mode=FINITE_VOLUME, riemann_solver=HLLC,
        limiter=MINMOD, dimensionality=3, box_size=box, num_cells=n, exact_end_time=True,
        wind_config=WindConfig(stellar_wind=True, num_injection_cells=m.NUM_INJECTION_CELLS * n // 128),
        cosmic_ray_grey_config=m.CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_shock_acceleration=True,
            diffusive_relaxation=kappa is not None),
        backend_config=BackendConfig(backend=NATIVE_JAX),
    )
    hd = get_helper_data(config)
    rv = get_registered_variables(config)
    rho0 = m.RHO_AMBIENT.to(cu.code_density).value
    p0 = m.P_AMBIENT.to(cu.code_pressure).value
    shape = (n, n, n)
    state = construct_primitive_state(
        config=config, registered_variables=rv, density=jnp.ones(shape) * rho0,
        velocity_x=jnp.zeros(shape), velocity_y=jnp.zeros(shape), velocity_z=jnp.zeros(shape),
        gas_pressure=jnp.ones(shape) * p0,
    )
    config = finalize_config(config, state.shape)
    mdot = m.M_DOT.to(cu.code_mass / cu.code_time).value
    vinf = m.V_INF.to(cu.code_velocity).value
    cr_params = m.CosmicRayGreyParams(dsa_efficiency=zeta, dsa_mach_min=m.DSA_MACH_MIN,
                                      reduced_streaming_speed=vred)
    if kappa is not None:
        cr_params = cr_params._replace(diffusion_coefficient=kappa)
    params = SimulationParams(
        t_end=m.T_END, gamma=m.GAMMA,
        wind_params=WindParams(wind_mass_loss_rates=jnp.array([mdot]), wind_final_velocities=jnp.array([vinf])),
        cosmic_ray_grey_params=cr_params,
    )
    return state, config, params, hd, rv, dict(rho0=rho0, p0=p0, L_w=0.5 * mdot * vinf**2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--vred", type=float, default=0.0)
    ap.add_argument("--zeta", type=float, default=m.DSA_EFFICIENCY)
    ap.add_argument("--snapshots", type=int, default=24)
    ap.add_argument("--off-after", type=float, default=None,
                    help="dsa_efficiency = 0 from this fraction of t_end on (self re-acceleration probe)")
    ap.add_argument("--gpu-diag", action="store_true",
                    help="per-snapshot diagnostics on the GPU (gpu_diag.py) instead of host numpy")
    args = ap.parse_args()
    if args.n % 32:
        raise ValueError("N must be a multiple of 32 (injection radius 4 cells at 128)")
    gpu = require_pinned_gpu()
    n = args.n
    box = m.BOX_SIZE
    dx = box / n
    cc = (np.arange(n) + 0.5) * dx - box / 2
    X, Y, Z = np.meshgrid(cc, cc, cc, indexing="ij")
    r_np = np.sqrt(X**2 + Y**2 + Z**2)
    nb = int(np.ceil(box / 2 * np.sqrt(3) / dx)) + 1
    bin_np = np.clip((r_np / dx).astype(int), 0, nb - 1)
    counts = np.bincount(bin_np.reshape(-1), minlength=nb).astype(float)
    InjectionAccumulator.install(coarse_cells=None, work_bins=(n, bin_np, nb))

    state, config, params, hd, rv, info = build_wind(n, kappa=args.kappa, vred=args.vred, zeta=args.zeta)
    assert np.allclose(np.asarray(hd.r), r_np, atol=1e-3 * dx)  # hd.r is float32
    cr_params = params.cosmic_ray_grey_params
    rho0, p0, L_w = info["rho0"], info["p0"], info["L_w"]
    t_end = m.T_END
    times = np.geomspace(0.02 * t_end, t_end, args.snapshots)
    off = None if args.off_after is None else args.off_after * t_end
    if off is not None:
        times = np.unique(np.append(times, off))
    print(f"wind {args.name} N {n} gpu {gpu} kappa {args.kappa} vred {args.vred} off_after {args.off_after}",
          flush=True)

    acc = InjectionAccumulator().activate()
    rows, profs, walls = [], [], []
    vel = (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)

    def on_snapshot(t, s, wall):
        a = np.asarray(s, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError(f"non-finite state at t = {t}")
        rho, p, ecr = a[rv.density_index], a[rv.pressure_index], a[rv.cosmic_ray_e_index]
        u = np.stack([a[c] for c in vel])
        vr = (u[0] * X + u[1] * Y + u[2] * Z) / np.maximum(r_np, 1e-30)
        prof = [np.bincount(bin_np.reshape(-1), weights=f.reshape(-1), minlength=nb) / np.maximum(counts, 1)
                for f in (rho, p, ecr, vr)]
        profs.append(prof)
        r_bins = (np.arange(nb) + 0.5) * dx
        R, R_peak, compression, width = shock_front(r_bins, prof[0], counts, rho0, dx, r_max=box / 2)
        d = {"t": float(t), "R_front": R, "R_peak_rho": R_peak, "compression": compression,
             "shell_fwhm_cells": width}

        sf = find_shocks_pfrommer(s, config, rv, hd, mach_min=cr_params.dsa_mach_min,
                                  **dsa_shock_finder_kwargs(config.cosmic_ray_grey_config))
        mach, flux = dsa_mach_and_flux(s, sf, config, params, rv)
        w = dsa_area_weights(s, sf, config.cosmic_ray_grey_config, rv)
        eff = dsa_efficiency_of_mach(config.cosmic_ray_grey_config, cr_params)(mach)
        mach, flux, w, eff = (np.asarray(q, dtype=np.float64) for q in (mach, flux, w, eff))
        surface = np.asarray(sf.shock_surface_cells)
        inj = surface & (flux > 0)
        fs = np.abs(r_np - R) <= 4.0 * dx
        ts = r_np < 0.6 * R
        cls = np.where(fs, 0, np.where(ts, 1, 2))
        d.update(n_surf=int(surface.sum()), n_inj=int(inj.sum()),
                 false_pos_frac=float((surface & (cls == 2)).sum() / max(surface.sum(), 1)))
        for k, c in enumerate(CLASSES):
            sel = inj & (cls == k)
            d[f"inj_{c}"] = float((eff * flux * w)[sel].sum()) * dx**2
            d[f"D_{c}"] = float((flux * w)[sel].sum()) * dx**2
            d[f"A_{c}"] = float(w[sel].sum()) * dx**2
            q = weighted_quantiles(mach[sel], w[sel], (0.16, 0.5, 0.84))
            d[f"M16_{c}"], d[f"M50_{c}"], d[f"M84_{c}"] = q
        dv = dx**3
        d.update(E_th=float(p.sum()) / (m.GAMMA - 1.0) * dv, E_kin=float((0.5 * rho * np.sum(u**2, axis=0)).sum()) * dv,
                 E_cr=float(ecr.sum()) * dv)
        d["E_tot"] = d["E_th"] + d["E_kin"] + d["E_cr"]
        d["inj_acc"] = acc.total()
        rows.append(d)
        walls.append(wall)
        print(f"t/t_end {t / t_end:.3f} R_fs {R:.4f} ({R / dx:.1f} cells) M50 fs {d['M50_fs']:.2f} ts {d['M50_ts']:.2f} "
              f"E_cr {d['E_cr'] / (L_w * t):.4f} L t inj {d['inj_acc'] / (L_w * t):.4f} L t "
              f"ratio {d['E_cr'] / max(d['inj_acc'], 1e-30):.3f} wall {wall:.0f} s peak {peak_gpu_bytes() / 1e9:.2f} GB",
              flush=True)

    if args.gpu_diag:
        geom = WindGeometry(r_np, bin_np, nb, X, Y, Z)

        def on_snapshot(t, s, wall):  # noqa: F811 (the GPU version of the block above; params as there)
            if not bool(jnp.all(jnp.isfinite(s))):
                raise FloatingPointError(f"non-finite state at t = {t}")
            d, prof = analyse_wind_snapshot_gpu(s, t, config, params, rv, hd, geom, dx, box, rho0, m.GAMMA, CLASSES)
            profs.append(prof)
            R = d["R_front"]
            d["inj_acc"] = acc.total()
            rows.append(d)
            walls.append(wall)
            print(f"t/t_end {t / t_end:.3f} R_fs {R:.4f} ({R / dx:.1f} cells) M50 fs {d['M50_fs']:.2f} ts {d['M50_ts']:.2f} "
                  f"E_cr {d['E_cr'] / (L_w * t):.4f} L t inj {d['inj_acc'] / (L_w * t):.4f} L t "
                  f"ratio {d['E_cr'] / max(d['inj_acc'], 1e-30):.3f} wall {wall:.0f} s peak {peak_gpu_bytes() / 1e9:.2f} GB",
                  flush=True)

    t_prev = 0.0
    p_cur = params
    for t in times:
        state = run_segments(state, config, p_cur._replace(t_start=t_prev), rv, [float(t)],
                             lambda i, tt, s, w: on_snapshot(tt, s, w))
        if off is not None and t >= off - 1e-12 * t_end and p_cur is params:
            p_cur = params._replace(cosmic_ray_grey_params=cr_params._replace(dsa_efficiency=0.0))
        t_prev = float(t)

    out = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    acc_t, acc_dt, acc_inj = acc.series()
    work_t, work = acc.work_series()
    out.update(acc_t=acc_t, acc_dt=acc_dt, acc_inj=acc_inj, work_t=work_t, work=work.astype(np.float32),
               wall=np.asarray(walls), N=n, dx=dx, box=box, zeta=args.zeta,
               kappa=np.nan if args.kappa is None else args.kappa, vred=args.vred,
               off_after=np.nan if args.off_after is None else args.off_after, L_w=L_w, t_end=t_end,
               rho0=rho0, p0=p0, gamma=m.GAMMA, r_bins=(np.arange(nb) + 0.5) * dx,
               **{f"prof_{q}": np.stack([pr[k] for pr in profs]).astype(np.float32)
                  for k, q in enumerate(("rho", "p", "ecr", "vr"))},
               peak_gpu_GB=peak_gpu_bytes() / 1e9, gpu=gpu, state_last=np.asarray(state))
    path = save_npz(RAW / "wind" / f"N{n}" / f"{args.name}.npz", **out)
    print(f"saved {path}: injected {acc.total() / (L_w * t_end):.4f} L t_end, E_cr "
          f"{rows[-1]['E_cr'] / (L_w * t_end):.4f}, wall {sum(walls):.0f} s", flush=True)


if __name__ == "__main__":
    main()
