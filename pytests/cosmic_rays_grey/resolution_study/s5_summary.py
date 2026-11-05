"""S5: Sedov rows of the Q5 table (cells per shock radius for a given accuracy). CPU only.

Pools every snapshot of every N per (setup, model, Mach range) against R/dx (S1/S3/S4
showed the errors collapse in R/dx once the start-up transient is over) and reports,
per metric and tolerance,

* ``measured``: the smallest R/dx above which every pooled point is within the
  tolerance (``> max`` if the finest point is not);
* ``fit``: R/dx where the power law err = a (R/dx)^-p, fitted to the pooled points
  with R/dx >= 15, reaches the tolerance (an extrapolation if beyond the data).

Metrics: Mach number M50 / M_true (finder), injection D = nominal injection / exact
(finder Mach + area through zeta(M) delta(M) M^3), and the asymptotic CR energy
E_cr / E_cr,ref = realized injection x deposit surplus (S-SS, where the exact reference
exists; S-CP: realized injection only, the surplus adds a few %). Snapshots before
2 t_start (S-SS / S-CP: the start-up transient) and the S-PE start (R/dx < 15) are
dropped.

Figure ``pics/S5_q5.png``, table ``pics/S5_summary.txt``.

Usage: python s5_summary.py
"""

import common  # noqa: F401  (sys.path, autocvd patch)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import INK, MUTED, PICS, _style  # noqa: E402
from s1_analyse import add_truth, load_setup  # noqa: E402
import s3_analyse  # noqa: E402
import s4_analyse  # noqa: E402

TOLS = {"Mach": (0.10, 0.05, 0.02), "D": (0.20, 0.10), "E_cr": (0.20, 0.10)}
RDX_FIT_MIN = 15.0
HIGH_M, LOW_M = 8.0, 4.0


def _settled(d):
    t_start = float(d.get("t_start", 0.0))
    return (d["t"] >= 2 * t_start) & (d["Rdx"] >= (RDX_FIT_MIN if t_start == 0 else 0))


def pool(runs, key, mask=None, value=None):
    x, y = [], []
    for d in runs.values():
        sel = _settled(d) & (mask(d) if mask else True)
        v = value(d) if value else d[key]
        sel &= np.isfinite(v)
        x.append(d["Rdx"][sel])
        y.append(v[sel])
    return np.concatenate(x), np.concatenate(y)


def required(x, y, tol):
    err = np.abs(1 - y)
    order = np.argsort(x)
    x, err = x[order], err[order]
    bad = np.flatnonzero(err > tol)
    measured = np.nan if not len(bad) else (x[bad[-1] + 1] if bad[-1] + 1 < len(x) else np.inf)
    if not len(bad):
        measured = x[0]
    fit = x >= RDX_FIT_MIN
    p, a = np.nan, np.nan
    if fit.sum() > 4 and np.all(err[fit] > 0):
        slope, inter = np.polyfit(np.log(x[fit]), np.log(err[fit]), 1)
        p, a = -slope, np.exp(inter)
    need = (a / tol) ** (1 / p) if p > 0 else np.nan
    return measured, need, p, x.max()


def main():
    s1 = {s: add_truth(load_setup(sub), s) for s, sub in (("pe", "pe"), ("ss", "ss"), ("cp", "cp_p0.6"))}
    ref = s3_analyse.solve_sedov_two_fluid(s3_analyse.ZETA, convention="gas_rh")
    kernel = s3_analyse.b73._adiabatic_kernel(ref)
    s3 = {s: s3_analyse.load(s, ref, kernel) for s in ("pe", "ss")}
    s4 = {m: s4_analyse.load(m) for m in ("z0.1", "kr13")}
    hi = lambda d: d["M_ref"] >= HIGH_M
    lo = lambda d: d["M_ref"] <= LOW_M

    rows = [  # (metric, case, (x, y))
        ("Mach", "control S-SS (M 170-760)", pool(s1["ss"], "mach_ref")),
        ("Mach", "control S-PE (M 170-760)", pool(s1["pe"], "mach_ref")),
        ("Mach", f"control S-CP M >= {HIGH_M:g}", pool(s1["cp"], "mach_ref", hi)),
        ("Mach", f"control S-CP M <= {LOW_M:g}", pool(s1["cp"], "mach_ref", lo)),
        ("Mach", "DSA zeta 0.1 S-SS", pool(s3["ss"], "mach")),
        ("Mach", f"DSA KR13 S-CP M >= {HIGH_M:g}", pool(s4["kr13"], "mach_ref", hi)),
        ("Mach", f"DSA KR13 S-CP M <= {LOW_M:g}", pool(s4["kr13"], "mach_ref", lo)),
        ("D", "zeta 0.1 S-SS (M 170-760)", pool(s3["ss"], "D_ratio")),
        ("D", "zeta 0.1 S-PE (M 170-760)", pool(s3["pe"], "D_ratio")),
        ("D", f"zeta 0.1 S-CP M >= {HIGH_M:g}", pool(s4["z0.1"], "inj_ratio", hi)),
        ("D", f"zeta 0.1 S-CP M <= {LOW_M:g}", pool(s4["z0.1"], "inj_ratio", lo)),
        ("D", f"KR13 S-CP M >= {HIGH_M:g}", pool(s4["kr13"], "inj_ratio", hi)),
        ("D", f"KR13 S-CP M <= {LOW_M:g}", pool(s4["kr13"], "inj_ratio", lo)),
        ("E_cr", "zeta 0.1 S-SS: realized inj x surplus",
         pool(s3["ss"], None, value=lambda d: d["inj_real_ratio"] * d["surplus"])),
        ("E_cr", "zeta 0.1 S-CP: realized inj (late)", pool(s4["z0.1"], "real_ratio", lambda d: d["t"] >= 0.03)),
        ("E_cr", "KR13 S-CP: realized inj (late)", pool(s4["kr13"], "real_ratio", lambda d: d["t"] >= 0.03)),
    ]

    lines = ["# S5: cells per shock radius R/dx for |1 - ratio| < tol (measured: all pooled points beyond;",
             "#     fit: power law over R/dx >= 15, '*' = extrapolated beyond the finest point)",
             f"{'metric':5s} {'case':40s} {'p':>5s} {'R/dx max':>8s}  tol: measured / fit"]
    for metric, case, (x, y) in rows:
        parts = []
        for tol in TOLS[metric]:
            measured, need, p, xmax = required(x, y, tol)
            m = "  >max" if np.isinf(measured) else f"{measured:6.0f}"
            f = f"{need:5.0f}{'*' if need > xmax else ' '}" if np.isfinite(need) else "    - "
            parts.append(f"{tol:4.0%}: {m} / {f}")
        lines.append(f"{metric:5s} {case:40s} {p:5.2f} {xmax:8.0f}  " + " | ".join(parts))
    text = "\n".join(lines)
    (PICS / "S5_summary.txt").write_text(text + "\n")
    print(text)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for ax, metric in zip(axes, TOLS):
        sel = [r for r in rows if r[0] == metric]
        for k, (_, case, (x, y)) in enumerate(sel):
            ax.loglog(x, np.abs(1 - y), ".", ms=3, color=cmap(k), alpha=0.6, label=case)
        for tol in TOLS[metric]:
            ax.axhline(tol, color=MUTED, ls="--", lw=0.8)
        ax.set_xlabel("R / dx")
        ax.set_title(f"|1 - {metric} ratio|", fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=6.5)
        _style(ax)
    fig.suptitle("S5: Sedov error vs cells per shock radius (all N pooled; dashed: tolerances)", color=INK)
    fig.savefig(PICS / "S5_q5.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
