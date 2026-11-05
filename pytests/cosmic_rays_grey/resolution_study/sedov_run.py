"""Sedov run of the resolution study with per-snapshot reduced diagnostics. GPU only.

Integrates in segments (``common.run_segments``) and, after each segment, runs
B7.3's per-snapshot analysis (``cr_sedov_injection_rate._analyse``: shock
radius, finder dissipation D and its area split, median DSA Mach number,
probe injection rate, E_cr) on that state; then B7.3's ``_derived`` (self-
similar fit, M_true, ratios). The injection accumulator adds the realized
injection of every step. Full states are kept only at ``--keep`` snapshot
indices.

Output: ``RAW/sedov/<setup>/N<n>/<name>.npz`` with the same keys as B7.3's
raw files (``<name>_t``, ``<name>_R_true``, ``<name>_mach_ratio``, ...), plus
``<name>_acc_t / _acc_dt / _acc_inj`` (per step), ``<name>_acc_coarse`` (16^3
block map of the injected energy), ``<name>_wall`` per segment, the kept
states and the code stamp.

Usage: python sedov_run.py --setup pe --n 48 --zeta 0.1 [--keep -1] [--snapshots 30]
"""

import argparse

import common
from common import RAW, InjectionAccumulator, peak_gpu_bytes, require_pinned_gpu, run_segments, save_npz

import numpy as np

import cr_sedov_injection_rate as b73
import cr_sedov_taylor as item7
from astronomix.test_setups.reference_solutions.sedov_two_fluid import solve_sedov_two_fluid
from setups import build_sedov_pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", default="pe", choices=["pe"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--zeta", type=float, default=item7.DSA_EFFICIENCY)
    ap.add_argument("--snapshots", type=int, default=30)
    ap.add_argument("--t-first", type=float, default=0.001)
    ap.add_argument("--keep", type=int, nargs="*", default=[-1], help="snapshot indices whose full state is saved")
    args = ap.parse_args()
    gpu = require_pinned_gpu()
    InjectionAccumulator.install(coarse_cells=16)

    name = "dsa" if args.zeta > 0 else "control"
    state, config, params, hd, rv = build_sedov_pe(args.n, args.zeta)
    reference = solve_sedov_two_fluid(args.zeta, convention="gas_rh") if args.zeta > 0 else solve_sedov_two_fluid(0.0)
    times = np.geomspace(args.t_first, float(params.t_end), args.snapshots)
    keep = {k % args.snapshots for k in args.keep}
    acc = InjectionAccumulator().activate()
    per_snap, walls, kept = [], [], {}

    def on_snapshot(i, t, s, wall):
        if not np.all(np.isfinite(np.asarray(s))):
            raise FloatingPointError(f"non-finite state at t = {t}")
        run = dict(states=[s], times=[t], config=config, params=params, registered_variables=rv,
                   helper_data=hd)
        per_snap.append(b73._analyse(run, reference, args.zeta))
        walls.append(wall)
        if i in keep:
            kept[f"{name}_state_{i}"] = np.asarray(s)
        r = per_snap[-1]
        print(f"[{i + 1}/{len(times)}] t {t:.4g} R {r['R_num'][0]:.4f} R/dx {r['R_num'][0] * args.n:.1f} "
              f"M_fin {r['M_finder'][0]:.1f} E_cr {r['E_cr'][0]:.4g} inj_acc {acc.total():.4g} "
              f"steps {len(acc.steps)} wall {wall:.1f} s peak {peak_gpu_bytes() / 1e9:.2f} GB", flush=True)

    run_segments(state, config, params, rv, times, on_snapshot)
    res = {k: np.concatenate([p[k] for p in per_snap]) for k in per_snap[0]}
    derived = b73._derived(res, reference)
    acc_t, acc_dt, acc_inj = acc.series()
    out = {f"{name}_{k}": v for k, v in derived.items()}
    out.update({f"{name}_acc_t": acc_t, f"{name}_acc_dt": acc_dt, f"{name}_acc_inj": acc_inj,
                f"{name}_wall": np.asarray(walls), f"{name}_N": args.n, f"{name}_zeta": args.zeta,
                f"{name}_peak_gpu_GB": peak_gpu_bytes() / 1e9, f"{name}_gpu": gpu})
    if acc.coarse is not None:
        out[f"{name}_acc_coarse"] = acc.coarse
    out.update(kept)
    path = save_npz(RAW / "sedov" / args.setup / f"N{args.n}" / f"{name}.npz", **out)
    print(f"saved {path}; injected (accumulated) {acc.total():.5g}, E_cr end {res['E_cr'][-1]:.5g}, "
          f"total wall {sum(walls):.0f} s", flush=True)


if __name__ == "__main__":
    main()
