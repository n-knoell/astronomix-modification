"""Sedov runs of the resolution study (S0-S4) with ``sedov_diag`` per snapshot. GPU only.

Setups (PLAN.md 3.1): ``pe`` point explosion, ``ss`` self-similar start
(R_0 = 0.15, p_0 = 1e-4), ``cp`` counter-pressure blast (self-similar start at
``--r-start``, raised ``--p0``, default 0.6 from S0). Integrates in segments (``common.run_segments``),
reduces each snapshot with ``sedov_diag.analyse_snapshot`` and sums the realized
injection of every step (``common.InjectionAccumulator``, 16^3 block map).
``--stop-radius`` ends the run after the first snapshot whose front radius
exceeds it (S0 sizing; the blast must stay inside the box).

Output: ``RAW/sedov/<setup dir>/N<n>/<name>.npz``: every diagnostic as an
array over snapshots, ``acc_t / acc_dt / acc_inj`` per step (step-end time),
``acc_coarse``, ``wall`` per segment, the run parameters, kept full states
(``state_<i>``) and the code stamp. ``<setup dir>`` is ``pe``, ``ss`` or
``cp_p<p0>``; ``<name>`` defaults to ``control`` / ``z<zeta>`` / ``kr13``.

Extension D (PLAN.md Section 9): ``--kappa`` (CR diffusion, F_cr relaxation) and ``--vred``
set the CR transport; then the radial shell profiles (dx / 2 bins) of rho, p, e_cr and v_r are
stored per snapshot (``prof_*``, ``prof_r``) and the centred compression work per step in
radial bins of dx (``work_t``, ``work``, ``work_r``).

Usage: python sedov_study.py --setup cp --p0 0.4 --r-start 0.08 --n 64 --zeta 0
       [--kr13] [--t-end 0.2] [--snapshots 40] [--stop-radius 0.44] [--keep -1]
"""

import argparse

import common
from common import RAW, InjectionAccumulator, peak_gpu_bytes, require_pinned_gpu, run_segments, save_npz

import numpy as np

import cr_sedov_taylor as item7
from sedov_diag import analyse_snapshot, shell_profiles
from setups import build_sedov_pe, build_sedov_ss


class _Stop(Exception):
    pass


def apply_overrides(config, params, args):
    """X1 sensitivity overrides of the cell-based numerical parameters."""
    cr = config.cosmic_ray_grey_config
    for item in args.cr:
        key, value = item.split("=")
        cr = cr._replace(**{key: type(getattr(cr, key))(float(value) if "." in value else int(value))})
    config = config._replace(cosmic_ray_grey_config=cr)
    if args.limiter:
        import astronomix.option_classes.simulation_config as sc
        config = config._replace(limiter=getattr(sc, args.limiter))
    if args.mach_min is not None:
        params = params._replace(cosmic_ray_grey_params=params.cosmic_ray_grey_params._replace(
            dsa_mach_min=args.mach_min))
    if args.cfl is not None:
        params = params._replace(C_cfl=args.cfl)
    return config, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=["pe", "ss", "cp"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--zeta", type=float, default=0.0)
    ap.add_argument("--kr13", action="store_true")
    ap.add_argument("--p0", type=float, default=None)
    ap.add_argument("--r-start", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--t-first", type=float, default=None, help="first snapshot (pe; default 0.001)")
    ap.add_argument("--snapshots", type=int, default=30)
    ap.add_argument("--stop-radius", type=float, default=None)
    ap.add_argument("--keep", type=int, nargs="*", default=[-1])
    ap.add_argument("--name", default=None)
    ap.add_argument("--cr", nargs="*", default=[], metavar="KEY=VALUE",
                    help="CosmicRayGreyConfig overrides (X1), e.g. dsa_shock_sampling_max_steps=8")
    ap.add_argument("--mach-min", type=float, default=None, help="dsa_mach_min override (X1)")
    ap.add_argument("--cfl", type=float, default=None, help="C_cfl override (X1)")
    ap.add_argument("--limiter", default=None, help="limiter override by name, e.g. VAN_LEER (X1)")
    ap.add_argument("--kappa", type=float, default=None, help="CR diffusion coefficient (extension D)")
    ap.add_argument("--vred", type=float, default=None, help="reduced_streaming_speed (extension D)")
    args = ap.parse_args()
    gpu = require_pinned_gpu()
    transport = args.kappa is not None or args.vred is not None
    dx = 1.0 / args.n
    cc = (np.arange(args.n) + 0.5) * dx - 0.5
    r_cells = np.sqrt(sum(a**2 for a in np.meshgrid(cc, cc, cc, indexing="ij")))
    n_work = int(0.5 * np.sqrt(3.0) / dx) + 2
    InjectionAccumulator.install(
        coarse_cells=16,
        work_bins=(args.n, np.clip((r_cells / dx).astype(int), 0, n_work - 1), n_work) if transport else None)

    p0 = args.p0 if args.p0 is not None else (0.6 if args.setup == "cp" else item7.P_AMBIENT)
    t_end = args.t_end if args.t_end is not None else item7.T_END
    if args.setup == "pe":
        state, config, params, hd, rv = build_sedov_pe(args.n, args.zeta, p_ambient=p0, kr13=args.kr13,
                                                       t_end=t_end, kappa=args.kappa, vred=args.vred)
        t_first = args.t_first or 0.001
    else:
        r_start = args.r_start if args.r_start is not None else (0.08 if args.setup == "cp" else 0.15)
        state, config, params, hd, rv = build_sedov_ss(args.n, args.zeta, p_ambient=p0, r_start=r_start,
                                                       kr13=args.kr13, t_end=t_end, kappa=args.kappa,
                                                       vred=args.vred)
        t_first = None
    config, params = apply_overrides(config, params, args)
    t0 = float(params.t_start)
    # log-spaced snapshots: from t_first (pe), or after the start time (ss / cp)
    times = (np.geomspace(t_first, t_end, args.snapshots) if t_first
             else np.geomspace(t0, t_end, args.snapshots + 1)[1:])
    keep = {k % len(times) for k in args.keep}
    name = args.name or ("kr13" if args.kr13 else (f"z{args.zeta:g}" if args.zeta > 0 else "control"))
    setup_dir = f"cp_p{p0:g}" if args.setup == "cp" else args.setup
    print(f"{setup_dir} N {args.n} {name}: t {t0:.4g} -> {t_end:.4g}, {len(times)} snapshots, gpu {gpu}",
          flush=True)

    acc = InjectionAccumulator().activate()
    rows, walls, kept = [], [], {}
    profs = []
    r_flat = r_cells.reshape(-1)

    def on_snapshot(i, t, s, wall):
        if not np.all(np.isfinite(np.asarray(s))):
            raise FloatingPointError(f"non-finite state at t = {t}")
        d = analyse_snapshot(s, t, config, params, rv, hd, rho0=item7.RHO_AMBIENT, gamma=item7.GAMMA)
        rows.append(d)
        walls.append(wall)
        if transport:
            a = np.asarray(s)
            ur = sum(a[c] * x for c, x in zip((rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z),
                                              np.meshgrid(cc, cc, cc, indexing="ij"))) / np.maximum(r_cells, 1e-12)
            centres, means, _ = shell_profiles(
                r_flat, [a[j].reshape(-1) for j in (rv.density_index, rv.pressure_index, rv.cosmic_ray_e_index)]
                + [ur.reshape(-1)], dx)
            profs.append((centres, means))
        if i in keep:
            kept[f"state_{i}"] = np.asarray(s)
        print(f"[{i + 1}/{len(times)}] t {t:.4g} R {d['R_front']:.4f} ({d['R_front'] * args.n:.1f} cells) "
              f"M50 {d['M50']:.2f} [{d['M16']:.2f}, {d['M84']:.2f}] compr {d['compression']:.2f} "
              f"compl {d['completeness']:.2f} E_cr {d['E_cr']:.4g} inj_acc {acc.total():.4g} "
              f"wall {wall:.1f} s peak {peak_gpu_bytes() / 1e9:.2f} GB", flush=True)
        if args.stop_radius is not None and d["R_front"] > args.stop_radius:
            kept[f"state_{i}"] = np.asarray(s)
            raise _Stop

    try:
        run_segments(state, config, params, rv, times, on_snapshot)
    except _Stop:
        print(f"stopped: front beyond {args.stop_radius}", flush=True)
    out = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    acc_t, acc_dt, acc_inj = acc.series()
    out.update(acc_t=acc_t, acc_dt=acc_dt, acc_inj=acc_inj, wall=np.asarray(walls),
               N=args.n, zeta=args.zeta, kr13=args.kr13, p0=p0, rho0=item7.RHO_AMBIENT, gamma=item7.GAMMA,
               t_start=t0, setup=args.setup, peak_gpu_GB=peak_gpu_bytes() / 1e9, gpu=gpu)
    if acc.coarse is not None:
        out["acc_coarse"] = acc.coarse
    if transport:
        out.update(E_cr0=float(np.asarray(state[rv.cosmic_ray_e_index]).sum()) * dx**3,
                   kappa=np.nan if args.kappa is None else args.kappa,
                   vred=float(params.cosmic_ray_grey_params.reduced_streaming_speed), prof_r=profs[0][0],
                   **{f"prof_{q}": np.stack([m[k] for _, m in profs]) for k, q in enumerate(("rho", "p", "ecr", "vr"))})
        work_t, work = acc.work_series()
        if work is not None:
            out.update(work_t=work_t, work=work.astype(np.float32), work_r=(np.arange(n_work) + 0.5) * dx)
    out.update(kept)
    path = save_npz(RAW / "sedov" / setup_dir / f"N{args.n}" / f"{name}.npz", **out)
    print(f"saved {path}: {len(rows)} snapshots, injected {acc.total():.5g}, E_cr {rows[-1]['E_cr']:.5g}, "
          f"wall {sum(walls):.0f} s", flush=True)


if __name__ == "__main__":
    main()
