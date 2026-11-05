"""DB11 analysis: CR transport in the cr_wind_bubble DSA setup (db11_probe.py runs). CPU, numpy only.

Per run: E_cr / injected over time; the end-of-run budget E_cr = injected + work (centred
-P_cr div v estimate) + rest (boundary outflow, stencil difference); the forward-shock radius
R_fs (outermost bin of the shell, rho > 2 rho_0) and the CR
precursor length l = int_{R_fs}^inf e_cr dr / e_cr(R_fs), compared with kappa / v_sh.
Output pics/DB11_transport.png, pics/DB11_summary.txt.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

RAW = Path("/export/scratch/nknoell/resolution_study/db11")
PICS = Path(__file__).resolve().parent / "pics"

# (file, label, kappa or None, colour)
KAPPAS = (("1e-3", 1e-3), ("1e-2", 1e-2), ("1e-1", 1e-1), ("1", 1.0))
BLUES = ("#9ecae1", "#4292c6", "#2171b5", "#08306b")
RUNS = ([("vred0", "advective (v_red 0)", None, "#2a2a2a"),
         ("def", "undamped wave (v_red 1)", None, "#d95f02")]
        + [(f"kappa{n}", f"kappa {k:g} (v_red 100)", k, c) for (n, k), c in zip(KAPPAS, BLUES)])


def shock_radius(rb, rho, rho0):
    """Outermost bin (inside the box half-width, no corner bins) of the swept-up shell, rho > 2 rho_0
    (nan while the early, thin shell is smeared below that in the radial mean)."""
    shell = np.where((rb < 0.6) & (rho > 2.0 * rho0))[0]
    return rb[shell.max()] if shell.size else np.nan


def analyse(name):
    d = np.load(RAW / f"{name}.npz")
    te, L, rb, dx = float(d["t_end"]), float(d["L_w"]), d["r_bins"], float(d["dx"])
    t, p, rho = d["snap_t"], d["prof_p"], d["prof_rho"]
    p0 = float(p[0][rb < 0.55][-1])
    rho0 = float(rho[0][rb < 0.55][-1])
    R = np.array([shock_radius(rb, rho[k], rho0) for k in range(len(t))])
    # v_sh at t_end from the last third of the snapshots (R ~ t^a)
    late = (t > 0.4 * te) & np.isfinite(R)
    a = np.polyfit(np.log(t[late]), np.log(R[late]), 1)[0]
    v_sh = a * R[-1] / t[-1]
    e = d["prof_ecr"][-1]
    i = np.searchsorted(rb, R[-1])
    l_pre = np.sum(e[i:][rb[i:] < 0.6]) * dx / max(e[i], 1e-30)
    st = d["step_t"]
    return dict(t=t / te, ratio=d["E_cr"] / np.maximum(d["inj"], 1e-30), E_cr=d["E_cr"][-1] / (L * te),
                inj=d["step_inj"].sum() / (L * te), work=d["step_work"].sum() / (L * te),
                R=R[-1], v_sh=v_sh, l_pre=l_pre, dx=dx, rb=rb, ecr=e / (3 * p0), rho=d["prof_rho"][-1],
                rho0=float(d["prof_rho"][0][rb < 0.55][-1]), steps=len(st))


res = {}
for name, label, kappa, colour in RUNS:
    if (RAW / f"{name}.npz").exists():
        res[name] = analyse(name)

lines = ["DB11: cr_wind_bubble DSA run (N 128) under different CR transport; energies in L_w t_end",
         f"{'run':28s} {'E_cr':>7s} {'inj':>7s} {'E_cr/inj':>8s} {'work':>7s} {'rest':>7s} "
         f"{'R_fs':>6s} {'v_sh':>5s} {'l_pre/dx':>8s} {'kap/v_sh/dx':>11s} {'steps':>6s}"]
for name, label, kappa, colour in RUNS:
    if name not in res:
        continue
    r = res[name]
    rest = r["E_cr"] - r["inj"] - r["work"]
    pred = "-" if kappa is None else f"{kappa / r['v_sh'] / r['dx']:.2f}"
    lines.append(f"{label:28s} {r['E_cr']:7.4f} {r['inj']:7.4f} {r['E_cr'] / r['inj']:8.3f} {r['work']:+7.4f} "
                 f"{rest:+7.4f} {r['R']:6.3f} {r['v_sh']:5.2f} {r['l_pre'] / r['dx']:8.2f} {pred:>11s} {r['steps']:6d}")
text = "\n".join(lines)
print(text)
(PICS / "DB11_summary.txt").write_text(text + "\n")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.6))
for name, label, kappa, colour in RUNS:
    if name not in res:
        continue
    r = res[name]
    ax1.plot(r["t"], r["ratio"], color=colour, lw=2, label=label)
    sel = (r["rb"] > 0.6 * r["R"]) & (r["rb"] < 0.6)
    ax2.plot(r["rb"][sel] / r["R"], r["ecr"][sel], color=colour, lw=2, label=label)
    ax3.plot(r["rb"][sel] / r["R"], r["rho"][sel] / r["rho0"], color=colour, lw=2, label=label)
ax1.axhline(1.0, color="#999999", lw=1, ls=":")
ax1.set_xscale("log")
ax1.set_xlabel(r"$t / t_{\rm end}$")
ax1.set_ylabel(r"$E_{\rm cr}$ / injected")
ax1.set_title("(a) CR energy held per energy injected")
ax1.legend(frameon=False, fontsize=8)
ax2.axvline(1.0, color="#999999", lw=1, ls=":")
ax2.set_yscale("log")
ax2.set_ylim(1e-3, None)
ax2.set_xlabel(r"$r / R_{\rm fs}$ (forward shock)")
ax2.set_ylabel(r"$P_{\rm cr} / p_0$ (radial mean, $t_{\rm end}$)")
ax2.set_title("(b) CR pressure around the forward shock")
ax3.axvline(1.0, color="#999999", lw=1, ls=":")
ax3.set_xlabel(r"$r / R_{\rm fs}$ (forward shock)")
ax3.set_ylabel(r"$\rho / \rho_0$ (radial mean, $t_{\rm end}$)")
ax3.set_title("(c) gas density: shell and precursor")
fig.tight_layout()
fig.savefig(PICS / "DB11_transport.png", dpi=150)
print("saved", PICS / "DB11_transport.png")
