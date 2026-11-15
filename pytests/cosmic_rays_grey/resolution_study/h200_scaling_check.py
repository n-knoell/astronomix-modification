"""Quick H200 check: does |dM/M| ~ 0.55 (R/dx)^-0.6 hold beyond R/dx ~ 90? CPU only.

Uses whatever phase-1 runs are on disk (RAW/sedov/*/N*/{control,z0.1,kr13}.npz) and the
S1/S3/S4 loaders and S5 pooling (settled snapshots, R/dx >= 15). Per case:

* power law err = a (R/dx)^-p fitted to the old grids only (N <= 224) and to all grids;
* the new grids' points (N > 224) against the old-grid fit and against the study fit
  0.55 (R/dx)^-0.6 (ratio measured / predicted, median over settled snapshots);
* per-N values at t = 0.035 / 0.07 and pairwise orders vs exact (incl. 128-256, 192-384, 224-448).

S-CP reference trajectory = the finest control grid present (now 448 instead of 224 for the
control runs; 384 for z0.1 / KR13): converged to < 1% in radius (S1), so the change is small.

Figure ``pics/H200_scaling_check.png``, table ``pics/H200_scaling_check.txt``. Does not
touch the S1-S5 outputs (those are redone in the full phase-1 pass).

Usage: python h200_scaling_check.py
"""

import common  # noqa: F401  (sys.path, autocvd patch)

import numpy as np
import matplotlib

# The loaders do dict(np.load(f)), which would decompress the stored states (11-18 GB per
# 384/448 control file). Skip the state_* arrays: this check needs the diagnostics only.
_np_load = np.load


class _NoStates(dict):
    pass


def _load_without_states(f, *a, **k):
    z = _np_load(f, *a, **k)
    if not str(f).endswith(".npz"):
        return z
    with z:
        return _NoStates({key: z[key] for key in z.files if not key.startswith("state_")})


np.load = _load_without_states

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import INK, MUTED, PICS, _style, at_time  # noqa: E402
from s1_analyse import add_truth, load_setup  # noqa: E402
from s5_summary import HIGH_M, LOW_M, RDX_FIT_MIN, _settled  # noqa: E402
import s3_analyse  # noqa: E402
import s4_analyse  # noqa: E402

N_OLD_MAX = 224
STUDY = (0.55, 0.6)


def pool_by_n(runs, key, mask=None):
    out = {}
    for n, d in runs.items():
        sel = _settled(d) & (mask(d) if mask else True) & np.isfinite(d[key])
        if sel.any():
            out[n] = (d["Rdx"][sel], d[key][sel])
    return out


def fit(pts, ns):
    x = np.concatenate([pts[n][0] for n in ns])
    e = np.abs(1 - np.concatenate([pts[n][1] for n in ns]))
    ok = (x >= RDX_FIT_MIN) & (e > 0)
    if ok.sum() < 5:
        return np.nan, np.nan
    slope, inter = np.polyfit(np.log(x[ok]), np.log(e[ok]), 1)
    return np.exp(inter), -slope


def main():
    s1 = {s: add_truth(load_setup(sub), s) for s, sub in (("pe", "pe"), ("ss", "ss"), ("cp", "cp_p0.6"))}
    ref = s3_analyse.solve_sedov_two_fluid(s3_analyse.ZETA, convention="gas_rh")
    kernel = s3_analyse.b73._adiabatic_kernel(ref)
    s3 = {s: s3_analyse.load(s, ref, kernel) for s in ("pe", "ss")}
    s4 = {m: s4_analyse.load(m) for m in ("z0.1", "kr13")}
    hi = lambda d: d["M_ref"] >= HIGH_M
    lo = lambda d: d["M_ref"] <= LOW_M

    cases = [  # (metric, label, runs, key, mask)
        ("Mach", "control S-SS (M 170-760)", s1["ss"], "mach_ref", None),
        ("Mach", "control S-PE (M 170-760)", s1["pe"], "mach_ref", None),
        ("Mach", f"control S-CP M <= {LOW_M:g}", s1["cp"], "mach_ref", lo),
        ("Mach", f"control S-CP M >= {HIGH_M:g}", s1["cp"], "mach_ref", hi),
        ("Mach", "DSA zeta 0.1 S-SS", s3["ss"], "mach", None),
        ("Mach", "DSA zeta 0.1 S-PE", s3["pe"], "mach", None),
        ("Mach", f"DSA KR13 S-CP M <= {LOW_M:g}", s4["kr13"], "mach_ref", lo),
        ("D", "zeta 0.1 S-SS (M 170-760)", s3["ss"], "D_ratio", None),
        ("D", "zeta 0.1 S-PE (M 170-760)", s3["pe"], "D_ratio", None),
        ("D", f"zeta 0.1 S-CP M <= {LOW_M:g}", s4["z0.1"], "inj_ratio", lo),
        ("D", f"KR13 S-CP M <= {LOW_M:g}", s4["kr13"], "inj_ratio", lo),
    ]

    lines = ["# H200 quick scaling check: |1 - ratio| = a (R/dx)^-p, settled snapshots, R/dx >= 15",
             f"# old = N <= {N_OLD_MAX}; new-grid points: median measured / predicted (old fit; study fit "
             f"{STUDY[0]} (R/dx)^-{STUDY[1]})",
             f"{'metric':5s} {'case':32s} {'N':>16s} {'a_old':>6s} {'p_old':>5s} {'a_all':>6s} {'p_all':>5s}  new grids"]
    results = []
    for metric, label, runs, key, mask in cases:
        pts = pool_by_n(runs, key, mask)
        old = [n for n in pts if n <= N_OLD_MAX]
        new = [n for n in pts if n > N_OLD_MAX]
        a_o, p_o = fit(pts, old)
        a_a, p_a = fit(pts, list(pts))
        parts = []
        for n in new:
            x, y = pts[n]
            e = np.abs(1 - y)
            r_old = np.median(e / (a_o * x**-p_o))
            r_st = np.median(e / (STUDY[0] * x**-STUDY[1]))
            # the study fit is the Mach law; the injection error is A x that (A = 3-12), so no /study there
            parts.append(f"N{n} R/dx {x.min():.0f}-{x.max():.0f}: err {np.median(e):.4f}, /old {r_old:.2f}"
                         + (f", /study {r_st:.2f}" if metric == "Mach" else ""))
        lines.append(f"{metric:5s} {label:32s} {','.join(map(str, pts)):>16s} {a_o:6.3f} {p_o:5.2f} "
                     f"{a_a:6.3f} {p_a:5.2f}  " + ("; ".join(parts) if parts else "(no new grid)"))
        results.append((metric, label, pts, (a_o, p_o), (a_a, p_a)))

    # per-N table and pairwise orders vs exact
    lines.append("\n# per-N values (control M50 / M_true,ref; S-SS zeta 0.1 injection D / ref) and pairwise orders")
    for name, runs, key in (("S-PE control", s1["pe"], "mach_ref"), ("S-SS control", s1["ss"], "mach_ref"),
                            ("S-CP control", s1["cp"], "mach_ref"), ("S-SS z0.1 Mach", s3["ss"], "mach"),
                            ("S-SS z0.1 D", s3["ss"], "D_ratio"), ("S-CP KR13 Mach", s4["kr13"], "mach_ref")):
        lines.append(f"\n## {name}: N = {list(runs)}")
        for t in (0.035, 0.07):
            vals = {n: at_time(d, key, t) for n, d in runs.items() if d["t"].min() <= t <= d["t"].max() + 1e-12}
            rdx = {n: at_time(runs[n], "Rdx", t) for n in vals}
            lines.append(f"t {t:<6g} " + "  ".join(f"N{n} ({rdx[n]:.0f}) {v:.4f}" for n, v in vals.items()))
            pairs = [(a, b) for a, b in ((96, 192), (128, 256), (192, 384), (224, 448), (192, 224), (224, 256),
                                         (256, 384), (384, 448)) if a in vals and b in vals]
            lines.append("         orders: " + "  ".join(
                f"{a}-{b} {np.log(abs(1 - vals[a]) / abs(1 - vals[b])) / np.log(b / a):.2f}" for a, b in pairs))
    text = "\n".join(lines)
    PICS.mkdir(exist_ok=True)
    (PICS / "H200_scaling_check.txt").write_text(text + "\n")
    print(text)

    # figure: one panel per metric
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for ax, metric in zip(axes, ("Mach", "D")):
        sel = [r for r in results if r[0] == metric]
        for k, (_, label, pts, (a_o, p_o), _) in enumerate(sel):
            c = cmap(k)
            for n, (x, y) in pts.items():
                new = n > N_OLD_MAX
                ax.loglog(x, np.abs(1 - y), "o" if new else ".", ms=4.5 if new else 2.5, color=c,
                          alpha=1.0 if new else 0.35, mec="k" if new else "none", mew=0.4,
                          label=label if n == max(pts) else None)
            xs = np.geomspace(RDX_FIT_MIN, 200, 50)
            if np.isfinite(a_o):
                ax.plot(xs, a_o * xs**-p_o, color=c, lw=0.9, alpha=0.8)
        xs = np.geomspace(5, 200, 50)
        amp = 1 if metric == "Mach" else 3
        ax.plot(xs, amp * STUDY[0] * xs**-STUDY[1], color=INK, ls="--", lw=1.4,
                label=f"{'' if amp == 1 else '3 x '}{STUDY[0]} (R/dx)^-{STUDY[1]} (study fit)")
        ax.axvline(90, color=MUTED, ls=":", lw=1)
        ax.set_xlabel("R / dx (cells per shock radius)")
        ax.set_title(f"|1 - {'M_finder / M_true' if metric == 'Mach' else 'injection / exact'}|"
                     f"  (faint: N <= {N_OLD_MAX}, ringed: H200 N > {N_OLD_MAX}; lines: fit to old grids)",
                     fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=6.5)
        _style(ax)
    fig.suptitle("H200 quick check: Sedov error scaling beyond R/dx ~ 90 (phase 1, partial)", color=INK)
    fig.savefig(PICS / "H200_scaling_check.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
