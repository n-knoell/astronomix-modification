"""C2: CWB restriction test, finder on the restricted finest control state vs native coarse runs. GPU only.

The last state of ``RAW/cwb/N<fine>/control.npz`` is block-averaged by 2 / 3 / 4 onto
fine/2, fine/3, fine/4 and analysed with ``cwb_study.analyse_cwb_snapshot``; so are the
native control last states at every N. All with the zeta = 0.1 DSA settings (the control
state carries no CRs; zeta only scales the nominal injection), so ``inj_<class>`` is the
nominal injection per shock class the finder would give on that flow.

Native coarse vs restricted fine at the same N = the hydro share of the error, restricted
fine vs the fine native = the finder share (PLAN.md principle 5). Note the injection
region is N/32 cells at every N, so a restricted state carries the fine run's wind
structure (resolved r/R_inj profile) on the coarse grid.

Output ``RAW/cwb/C2_control.npz`` (one row per (kind, N)).

Usage: python c2_restrict.py [--fine 192]
"""

import argparse

import common
from common import RAW, require_pinned_gpu, restrict, save_npz

import numpy as np
import jax.numpy as jnp

from cwb_study import CLASSES, analyse_cwb_snapshot
from setups import build_cwb

SCALARS = ["x_ts1", "x_contact", "x_ts2", "mach_ts1", "mach_ts2", "D", "inj_nominal", "n_inj"] + [
    f"{q}_{c}" for c in CLASSES for q in ("inj", "A", "M16", "M50", "M84")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", type=int, default=192)
    args = ap.parse_args()
    print("GPU", require_pinned_gpu(), flush=True)
    rows = []

    def record(kind, n, state, t):
        _, config, params, hd, rv = build_cwb(n, 0.1)
        d = analyse_cwb_snapshot(state, t, config, params, rv, hd)
        row = {k: d[k] for k in SCALARS}
        row.update(kind=kind, N=n)
        rows.append(row)
        print(f"{kind:10s} N {n:3d} x_ts1 {d['x_ts1']:+.4f} x_c {d['x_contact']:+.4f} x_ts2 {d['x_ts2']:+.4f} "
              f"M_ts {d['mach_ts1']:.2f} {d['mach_ts2']:.2f} | M50 w1/w2 {d['M50_wind1']:.2f} {d['M50_wind2']:.2f} "
              f"A w1/w2 {d['A_wind1']:.4e} {d['A_wind2']:.4e} inj w1/w2/amb {d['inj_wind1']:.4e} "
              f"{d['inj_wind2']:.4e} {d['inj_ambient']:.4e}", flush=True)
        return config, params, rv

    runs = {int(p.parent.name[1:]): np.load(p) for p in sorted(RAW.glob("cwb/N*/control.npz"))}
    for n, run in sorted(runs.items()):
        record("native", n, jnp.asarray(run["state_last"]), float(run["t"][-1]))
    fine = runs[args.fine]
    gamma = float(fine["gamma"]) if "gamma" in fine.files else 5.0 / 3.0
    _, _, _, _, rv_fine = build_cwb(args.fine, 0.1)
    for factor in (2, 3, 4):
        coarse = restrict(jnp.asarray(fine["state_last"]), rv_fine, gamma, factor)
        record("restricted", args.fine // factor, coarse, float(fine["t"][-1]))

    out = {k: np.asarray([r[k] for r in rows]) for k in SCALARS + ["N"]}
    out["kind"] = np.asarray([r["kind"] for r in rows])
    print("saved", save_npz(RAW / "cwb" / "C2_control.npz", fine=args.fine, **out), flush=True)


if __name__ == "__main__":
    main()
