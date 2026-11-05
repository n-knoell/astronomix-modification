"""Extension D analysis (PLAN.md Section 9, QD1-QD4). CPU, numpy / scipy only.

Usage: python d_analyse.py [wind] [sedov] [cwb] [d0]   (default: all present data)

Common: self-convergence fit q(N) = q_inf + C N^-p over every N of a series (no analytic
reference with a precursor, 9.3 item 4); the error of each N is |q(N) - q_inf| / |q_inf|, and
the cells per shock radius for 10% / 5% follow from the fit. The advective runs (W: ``adv``;
Sedov: the existing ``z0.1`` / ``kr13``; CWB: the existing ``z0.1`` at kappa 1) are the
lambda -> 0 anchor.

Precursor length l = int_{R_s}^{R_box} e_cr dr / e_cr(R_s) (radial / axis means) against
kappa / v_sh. Pre-compression: rho / rho_0 two cells ahead of the front (what the finder's
upstream sees). Energies: E_cr = E_cr0 + injected + work + rest (rest: boundary losses and the
difference between the centred work estimate and the scheme's own).

Output ``pics/D_<setup>_*.png`` and ``pics/D_<setup>_summary.txt``.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RAW = Path("/export/scratch/nknoell/resolution_study")
PICS = Path(__file__).resolve().parent / "pics"
LAMS = (0.03, 0.1, 0.3)
COLOURS = {0.0: "#2a2a2a", 0.03: "#9ecae1", 0.1: "#4292c6", 0.3: "#08306b"}


def fit_convergence(ns, qs):
    """q(N) = q_inf + C N^-p; returns (q_inf, C, p) or None (fewer than 4 points / no fit)."""
    ns, qs = np.asarray(ns, float), np.asarray(qs, float)
    ok = np.isfinite(qs)
    ns, qs = ns[ok], qs[ok]
    if ns.size < 4:
        return None
    try:
        popt, _ = curve_fit(lambda n, qi, c, p: qi + c * n ** (-p), ns, qs,
                            p0=(qs[-1], (qs[0] - qs[-1]) * ns[0], 1.0),
                            bounds=([-np.inf, -np.inf, 0.2], [np.inf, np.inf, 3.0]), maxfev=20000)
        return tuple(float(v) for v in popt)
    except (RuntimeError, ValueError):
        return None


def fit_ok(fit):
    """A fit whose exponent sits at a bound (0.2 / 3) is not a convergence statement."""
    return fit is not None and 0.21 < fit[2] < 2.99


def cells_for(fit, frac, r_cells_per_n):
    """Cells per shock radius at which |q - q_inf| = frac |q_inf| (from the fit)."""
    if not fit_ok(fit):
        return np.nan
    qi, c, p = fit
    if qi == 0:
        return np.nan
    n = (abs(c) / (frac * abs(qi))) ** (1.0 / p)
    return n * r_cells_per_n


def fmt_fit(fit):
    return "-" if fit is None else f"q_inf {fit[0]:.4g} p {fit[2]:.2f}" + ("" if fit_ok(fit) else " (bound)")


def work_split(d, t_snap, R_snap, dx, r_bins):
    """Centred compression work at and ahead of the front (r >= R(t) - 2 dx) and behind it;
    R(t) per step interpolated from the snapshots."""
    if "work" not in d.files:
        return np.nan, np.nan
    w = np.asarray(d["work"], dtype=np.float64)
    R = np.interp(d["work_t"], t_snap, R_snap)
    front = r_bins[None, :] >= (R[:, None] - 2 * dx)
    return float(w[front].sum()), float(w[~front].sum())


def precursor(r, e, R, r_max):
    """l = int_R^{r_max} e dr / e(R) on a uniform radial grid."""
    h = r[1] - r[0]
    i = int(np.searchsorted(r, R))
    sel = slice(i, int(np.searchsorted(r, r_max)))
    return float(np.sum(e[sel]) * h / max(e[i], 1e-300))


# --------------------------------------------------------------------------- W (D1)
def analyse_wind():
    ns = sorted(int(p.name[1:]) for p in (RAW / "wind").glob("N*"))
    runs = {}
    for n in ns:
        for name in ["adv"] + [f"lam{lam:g}" for lam in LAMS]:
            p = RAW / "wind" / f"N{n}" / f"{name}.npz"
            if p.exists():
                runs[(name, n)] = np.load(p)
    if not runs:
        return
    rows = {}
    for (name, n), d in runs.items():
        L = float(d["L_w"] * d["t_end"])
        t, R = d["t"], d["R_front"]
        late = (t > 0.4 * float(d["t_end"])) & np.isfinite(R)
        a = np.polyfit(np.log(t[late]), np.log(R[late]), 1)[0]
        v_sh = a * R[-1] / t[-1]
        r = d["r_bins"]
        dx = float(d["dx"])
        e = d["prof_ecr"][-1].astype(float)
        rho = d["prof_rho"][-1].astype(float)
        l = precursor(r, e, R[-1], 0.5 * float(d["box"]))
        kappa = float(d["kappa"])
        inj, work = float(d["acc_inj"].sum()), float(d["work"].sum())
        w_front, w_back = work_split(d, t, R, dx, r)
        e_amb = float(d["p0"]) / (float(d["gamma"]) - 1.0) * float(d["box"]) ** 3
        rows[(name, n)] = dict(
            E_cr=float(d["E_cr"][-1]) / L, inj=inj / L, work=work / L, work_front=w_front / L, rest=(float(d["E_cr"][-1]) - inj - work) / L,
            ratio=float(d["E_cr"][-1]) / inj, budget=(float(d["E_tot"][-1]) - e_amb) / L, R=float(R[-1]),
            R_cells=float(R[-1]) / dx, v_sh=v_sh, M50_fs=float(d["M50_fs"][-1]), M50_ts=float(d["M50_ts"][-1]),
            inj_fs=float(d["inj_fs"][-1]), inj_ts=float(d["inj_ts"][-1]), l_cells=l / dx,
            l_model_cells=(kappa / v_sh / dx) if np.isfinite(kappa) else np.nan,
            pre=float(np.interp(R[-1] + 2 * dx, r, rho)) / float(d["rho0"]), fp=float(d["false_pos_frac"][-1]),
            steps=len(d["acc_t"]), wall=float(d["wall"].sum()), dx=dx)
    lines = ["W (wind bubble, zeta 0.1): final state t_end; energies in L_w t_end; l = precursor length",
             f"{'run':8s} {'N':>4s} {'R/dx':>6s} {'E_cr':>7s} {'inj':>7s} {'work':>7s} {'w_front':>7s} {'rest':>7s} {'E_cr/inj':>8s} "
             f"{'budget':>7s} {'M50_fs':>6s} {'M50_ts':>6s} {'l/dx':>6s} {'l/model':>7s} {'pre':>5s} {'fp':>5s} {'h':>5s}"]
    for (name, n), q in sorted(rows.items(), key=lambda kv: (kv[0][0] != "adv", kv[0][0], kv[0][1])):
        lm = q["l_cells"] / q["l_model_cells"] if np.isfinite(q["l_model_cells"]) else np.nan
        lines.append(f"{name:8s} {n:4d} {q['R_cells']:6.1f} {q['E_cr']:7.4f} {q['inj']:7.4f} {q['work']:+7.4f} {q['work_front']:+7.4f} "
                     f"{q['rest']:+7.4f} {q['ratio']:8.3f} {q['budget']:7.4f} {q['M50_fs']:6.2f} {q['M50_ts']:6.2f} "
                     f"{q['l_cells']:6.2f} {lm:7.3f} {q['pre']:5.2f} {q['fp']:5.3f} {q['wall'] / 3600:5.2f}")
    lines.append("")
    lines.append("Self-convergence q(N) = q_inf + C N^-p (all N); cells per forward-shock radius for 10% / 5%")
    for name in ["adv"] + [f"lam{lam:g}" for lam in LAMS]:
        nn = [n for n in ns if (name, n) in rows]
        if not nn:
            continue
        rc = rows[(name, nn[-1])]["R"] / float(runs[(name, nn[-1])]["box"])  # R/dx = rc * N
        for qn in ("ratio", "E_cr", "inj", "M50_fs", "inj_fs"):
            f = fit_convergence(nn, [rows[(name, n)][qn] for n in nn])
            lines.append(f"  {name:8s} {qn:7s} {fmt_fit(f):28s} 10%: {cells_for(f, 0.1, rc):6.1f}  "
                         f"5%: {cells_for(f, 0.05, rc):6.1f}  values {' '.join(f'{rows[(name, n)][qn]:.4g}' for n in nn)}")
    text = "\n".join(lines)
    print(text)
    (PICS / "D_wind_summary.txt").write_text(text + "\n")

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for name in ["adv"] + [f"lam{lam:g}" for lam in LAMS]:
        lam = 0.0 if name == "adv" else float(name[3:])
        nn = [n for n in ns if (name, n) in rows]
        if not nn:
            continue
        col = COLOURS[lam]
        lab = "advective" if lam == 0 else f"λ {lam:g}"
        x = [rows[(name, n)]["R_cells"] for n in nn]
        for k, (qn, yl) in enumerate((("ratio", "E_cr / injected"), ("inj", "injected / (L_w t_end)"),
                                      ("M50_fs", "forward shock M50 (finder)"))):
            ax[0, k].plot(x, [rows[(name, n)][qn] for n in nn], "o-", color=col, label=lab)
            ax[0, k].set_xlabel("R_fs / dx")
            ax[0, k].set_ylabel(yl)
        if lam > 0:
            ax[1, 0].plot([rows[(name, n)]["l_model_cells"] for n in nn],
                          [rows[(name, n)]["l_cells"] / rows[(name, n)]["l_model_cells"] for n in nn], "o-", color=col,
                          label=lab)
        ax[1, 1].plot(x, [rows[(name, n)]["pre"] for n in nn], "o-", color=col, label=lab)
        d = runs[(name, nn[-1])]
        R = rows[(name, nn[-1])]["R"]
        r = d["r_bins"]
        sel = (r > 0.6 * R) & (r < 0.6)
        ax[1, 2].semilogy(r[sel] / R, d["prof_ecr"][-1][sel] / 3 / float(d["p0"]), color=col, label=f"{lab}, N {nn[-1]}")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("κ / v_sh in cells")
    ax[1, 0].set_ylabel("measured precursor / (κ / v_sh)")
    ax[1, 0].axhline(1, color="#999", lw=1, ls=":")
    ax[1, 1].set_xlabel("R_fs / dx")
    ax[1, 1].set_ylabel("ρ / ρ_0 at R_fs + 2 dx")
    ax[1, 2].set_xlabel("r / R_fs")
    ax[1, 2].set_ylabel("P_cr / p_0 (radial mean, t_end)")
    ax[1, 2].set_ylim(1e-3, None)
    for a in ax.flat:
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("W: wind bubble with CR diffusion (extension D1), final state")
    fig.tight_layout()
    fig.savefig(PICS / "D_wind_convergence.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- Sedov (D2)
SEDOV_RADII = (0.2, 0.3)


def sedov_rows(sdir, model):
    out = {}
    for p in sorted((RAW / "sedov" / sdir).glob(f"N*/{model}*.npz")):
        name = p.stem
        if name != model and not name.startswith(model + "_lam"):
            continue
        if "_vx2" in name:
            continue
        n = int(p.parent.name[1:])
        d = np.load(p)
        dx = 1.0 / n
        R, t = d["R_front"], d["t"]
        lam = 0.0 if name == model else float(name.split("_lam")[1])
        q = {"lam": lam, "N": n, "R_end": float(R[-1]), "R_cells_end": float(R[-1]) / dx}
        # quantities at fixed radii (linear interpolation in R)
        for rr in SEDOV_RADII + (float(R[-1]),):
            key = "end" if rr == float(R[-1]) else f"{rr:g}"
            if rr > R[-1] + 1e-9 or rr < R[0]:
                for f in ("M50", "inj_rate", "E_cr", "pre", "l_cells"):
                    q[f"{f}_{key}"] = np.nan
                continue
            q[f"M50_{key}"] = float(np.interp(rr, R, d["M50"]))
            q[f"inj_rate_{key}"] = float(np.interp(rr, R, d["inj_nominal"]))
            q[f"E_cr_{key}"] = float(np.interp(rr, R, d["E_cr"]))
            if "prof_ecr" in d.files:
                k = int(np.argmin(np.abs(R - rr)))
                r = d["prof_r"]
                q[f"pre_{key}"] = float(np.interp(R[k] + 2 * dx, r, d["prof_rho"][k])) / float(d["rho0"])
                q[f"l_cells_{key}"] = precursor(r, d["prof_ecr"][k], R[k], 0.5) / dx
            else:
                q[f"pre_{key}"], q[f"l_cells_{key}"] = np.nan, np.nan
        inj = float(d["acc_inj"].sum())
        e0 = float(d["E_cr0"]) if "E_cr0" in d.files else np.nan
        q["inj"] = inj
        q["E_cr_over_in"] = float(d["E_cr"][-1]) / (e0 + inj) if np.isfinite(e0) else np.nan
        q["work"] = float(d["work"].sum()) if "work" in d.files else np.nan
        q["work_front"] = work_split(d, t, R, dx, d["work_r"])[0] if "work_r" in d.files else np.nan
        q["kappa"] = float(d["kappa"]) if "kappa" in d.files else np.nan
        q["wall"] = float(d["wall"].sum())
        q["d"] = d
        out[(lam, n)] = q
    return out


def analyse_sedov():
    lines = []
    for sdir, model in (("ss", "z0.1"), ("cp_p0.6", "z0.1"), ("cp_p0.6", "kr13")):
        rows = sedov_rows(sdir, model)
        if not rows:
            continue
        lines.append(f"== {sdir} {model}: values at R = {SEDOV_RADII} and at the end (R_end); "
                     "M50 = finder median Mach, inj_rate = nominal injection rate, pre = rho/rho0 at R + 2 dx")
        lines.append(f"{'lam':>5s} {'N':>4s} {'R_end':>6s} {'R/dx':>6s} " + " ".join(
            f"{f + '_' + k:>11s}" for k in ("0.2", "0.3", "end") for f in ("M50", "inj_rate", "l_cells", "pre"))
                     + f" {'E_cr/(E0+inj)':>13s} {'work/inj':>8s} {'wfront/inj':>10s} {'h':>5s}")
        for (lam, n), q in sorted(rows.items()):
            lines.append(f"{lam:5g} {n:4d} {q['R_end']:6.3f} {q['R_cells_end']:6.1f} " + " ".join(
                f"{q[f + '_' + k]:11.4g}" for k in ("0.2", "0.3", "end") for f in ("M50", "inj_rate", "l_cells", "pre"))
                         + f" {q['E_cr_over_in']:13.4f} {q['work'] / q['inj'] if q['inj'] else np.nan:8.3f}"
                         f" {q['work_front'] / q['inj'] if q['inj'] else np.nan:10.3f}"
                         f" {q['wall'] / 3600:5.2f}")
        lines.append("  self-convergence (all N) at R = 0.3 and at R_end; cells per radius for 10% / 5%:")
        for lam in sorted({k[0] for k in rows}):
            nn = sorted(n for (l_, n) in rows if l_ == lam)
            for f in ("M50_0.3", "inj_rate_0.3", "M50_end", "inj_rate_end", "E_cr_over_in"):
                vals = [rows[(lam, n)][f] for n in nn]
                rr = 0.3 if f.endswith("0.3") else rows[(lam, nn[-1])]["R_end"]
                fit = fit_convergence(nn, vals)
                lines.append(f"  lam {lam:5g} {f:14s} {fmt_fit(fit):28s} 10%: {cells_for(fit, 0.1, rr):6.1f}  "
                             f"5%: {cells_for(fit, 0.05, rr):6.1f}  values {' '.join(f'{v:.4g}' for v in vals)}")
        # QD2: Mach with the precursor relative to the advective anchor at the same N
        lines.append("  M50 (lambda) / M50 (advective) at the same N and R = 0.3:")
        for lam in sorted({k[0] for k in rows if k[0] > 0}):
            nn = sorted(n for (l_, n) in rows if l_ == lam and (0.0, n) in rows)
            lines.append(f"  lam {lam:5g} " + " ".join(
                f"N{n} {rows[(lam, n)]['M50_0.3'] / rows[(0.0, n)]['M50_0.3']:.3f}" for n in nn))
        lines.append("")
        _plot_sedov(sdir, model, rows)
    if lines:
        text = "\n".join(lines)
        print(text)
        (PICS / "D_sedov_summary.txt").write_text(text + "\n")


def _plot_sedov(sdir, model, rows):
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    for lam in sorted({k[0] for k in rows}):
        col = COLOURS[lam]
        nn = sorted(n for (l_, n) in rows if l_ == lam)
        lab = "advective" if lam == 0 else f"λ {lam:g}"
        ax[0].plot([rows[(lam, n)]["R_cells_end"] * 0.3 / rows[(lam, n)]["R_end"] for n in nn],
                   [rows[(lam, n)]["M50_0.3"] for n in nn], "o-", color=col, label=lab)
        ax[1].plot([0.3 * n for n in nn], [rows[(lam, n)]["inj_rate_0.3"] for n in nn], "o-", color=col, label=lab)
        ax[2].plot([0.3 * n for n in nn], [rows[(lam, n)]["l_cells_0.3"] for n in nn], "o-", color=col, label=lab)
        d = rows[(lam, nn[-1])]["d"]
        if "prof_ecr" in d.files:
            k = int(np.argmin(np.abs(d["R_front"] - 0.3)))
            r = d["prof_r"]
            sel = (r > 0.2) & (r < 0.5)
            ax[3].semilogy(r[sel] / d["R_front"][k], d["prof_ecr"][k][sel], color=col, label=f"{lab}, N {nn[-1]}")
    for a, (xl, yl) in zip(ax, (("R / dx", "M50 at R = 0.3"), ("R / dx", "nominal injection rate at R = 0.3"),
                                ("R / dx", "precursor length / dx at R = 0.3"), ("r / R_front", "e_cr at R ≈ 0.3"))):
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Sedov {sdir} {model} with CR diffusion (extension D2)")
    fig.tight_layout()
    fig.savefig(PICS / f"D_sedov_{sdir}_{model}.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- CWB (D3)
def analyse_cwb():
    rows = {}
    for p in sorted((RAW / "cwb").glob("N*/z0.1*.npz")):
        name = p.stem
        if "_vx2" in name:
            continue
        n = int(p.parent.name[1:])
        lam = 0.0 if name == "z0.1" else float(name.split("_lam")[1])
        d = np.load(p)
        dx = 1.0 / n
        x, star2 = d["x_axis"], float(d["stars"][1][0])
        half = slice(len(d["t"]) // 2, None)
        kappa = float(d["kappa"]) if "kappa" in d.files else 1.0
        # precursor ahead of the wind-2 termination shock, along the axis towards star 2
        ls = []
        for k in range(len(d["t"]))[half]:
            xs = d["x_ts2"][k]
            if not np.isfinite(xs):
                continue
            sel = (x >= xs) & (x < star2 - 0.05)
            e = d["ax_ecr"][k][sel]
            ls.append(np.sum(e) * dx / max(e[0], 1e-300))
        inj = float(d["acc_inj"].sum())
        rows[(lam, n)] = dict(
            kappa=kappa, l_cells=float(np.mean(ls)) / dx if ls else np.nan, l_model_cells=kappa / 629.0 / dx,
            M_ts1=float(np.nanmean(d["mach_ts1"][half])), M_ts2=float(np.nanmean(d["mach_ts2"][half])),
            M50_w1=float(np.nanmean(d["M50_wind1"][half])), M50_w2=float(np.nanmean(d["M50_wind2"][half])),
            inj_w1=float(np.mean(d["inj_wind1"][half])), inj_w2=float(np.mean(d["inj_wind2"][half])),
            inj_amb=float(np.mean(d["inj_ambient"][half])), inj=inj, E_cr=float(d["E_cr"][-1]),
            ratio=float(d["E_cr"][-1]) / inj, work=float(d["work"].sum()) if "work" in d.files else np.nan,
            wall=float(d["wall"].sum()))
    if not rows:
        return
    lines = ["CWB zeta 0.1 (lam 0 = existing kappa 1 runs, lambda_wind2 0.019); second-half means; "
             "l = precursor ahead of the wind-2 TS on the axis",
             f"{'lam':>5s} {'N':>4s} {'kappa':>6s} {'M_ts1':>6s} {'M_ts2':>6s} {'M50_w1':>6s} {'M50_w2':>6s} "
             f"{'inj_w1':>9s} {'inj_w2':>9s} {'inj_amb':>9s} {'E_cr/inj':>8s} {'work/inj':>8s} {'l/dx':>6s} "
             f"{'model':>6s} {'h':>5s}"]
    for (lam, n), q in sorted(rows.items()):
        lines.append(f"{lam:5g} {n:4d} {q['kappa']:6.3g} {q['M_ts1']:6.2f} {q['M_ts2']:6.2f} {q['M50_w1']:6.2f} "
                     f"{q['M50_w2']:6.2f} {q['inj_w1']:9.3g} {q['inj_w2']:9.3g} {q['inj_amb']:9.3g} {q['ratio']:8.3f} "
                     f"{q['work'] / q['inj']:8.3f} {q['l_cells']:6.2f} {q['l_model_cells']:6.2f} {q['wall'] / 3600:5.2f}")
    lines.append("self-convergence (all N), cells per wind-2 stand-off (R_ts2 0.084) for 10% / 5%:")
    for lam in sorted({k[0] for k in rows}):
        nn = sorted(n for (l_, n) in rows if l_ == lam)
        for f in ("M50_w1", "M50_w2", "inj_w1", "inj_w2", "ratio"):
            vals = [rows[(lam, n)][f] for n in nn]
            fit = fit_convergence(nn, vals)
            lines.append(f"  lam {lam:5g} {f:7s} {fmt_fit(fit):28s} 10%: {cells_for(fit, 0.1, 0.084):6.1f}  "
                         f"5%: {cells_for(fit, 0.05, 0.084):6.1f}  values {' '.join(f'{v:.4g}' for v in vals)}")
    text = "\n".join(lines)
    print(text)
    (PICS / "D_cwb_summary.txt").write_text(text + "\n")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for lam in sorted({k[0] for k in rows}):
        nn = sorted(n for (l_, n) in rows if l_ == lam)
        col = COLOURS.get(lam, "#2a2a2a")
        lab = "κ 1 (λ 0.02)" if lam == 0 else f"λ {lam:g}"
        x = [0.084 * n for n in nn]
        ax[0].plot(x, [rows[(lam, n)]["M50_w2"] for n in nn], "o-", color=col, label=lab)
        ax[0].plot(x, [rows[(lam, n)]["M50_w1"] for n in nn], "s--", color=col)
        ax[1].plot(x, [rows[(lam, n)]["inj_w1"] + rows[(lam, n)]["inj_w2"] + rows[(lam, n)]["inj_amb"] for n in nn],
                   "o-", color=col, label=lab)
        ax[2].plot(x, [rows[(lam, n)]["l_cells"] for n in nn], "o-", color=col, label=lab)
        ax[2].plot(x, [rows[(lam, n)]["l_model_cells"] for n in nn], ":", color=col)
    for a, yl in zip(ax, ("M50 wind 2 (o) / wind 1 (s)", "nominal injection rate (all shocks)",
                          "precursor ahead of wind-2 TS / dx (dotted: κ / u_1)")):
        a.set_xlabel("wind-2 stand-off / dx")
        a.set_ylabel(yl)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle("CWB with CR diffusion (extension D3), second-half means")
    fig.tight_layout()
    fig.savefig(PICS / "D_cwb_convergence.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- D0 checks
def analyse_d0():
    lines = ["D0 checks"]
    pairs = [("wind", RAW / "wind/N96/lam0.1.npz", RAW / "wind/N96/lam0.1_vx2.npz"),
             ("S-SS", RAW / "sedov/ss/N96/z0.1_lam0.1.npz", RAW / "sedov/ss/N96/z0.1_lam0.1_vx2.npz"),
             ("S-CP", RAW / "sedov/cp_p0.6/N96/z0.1_lam0.1.npz", RAW / "sedov/cp_p0.6/N96/z0.1_lam0.1_vx2.npz"),
             ("CWB", RAW / "cwb/N96/z0.1_lam0.1.npz", RAW / "cwb/N96/z0.1_lam0.1_vx2.npz")]
    lines.append("v_red x 2 at N 96, lambda 0.1: relative change of E_cr, injected, last M50 (forward / blast / wind 2)")
    for lab, a, b in pairs:
        if not (a.exists() and b.exists()):
            lines.append(f"  {lab}: missing")
            continue
        da, db = np.load(a), np.load(b)
        mk = "M50_fs" if lab == "wind" else ("M50_wind2" if lab == "CWB" else "M50")
        lines.append(f"  {lab:5s} E_cr {db['E_cr'][-1] / da['E_cr'][-1] - 1:+.4f}  injected "
                     f"{db['acc_inj'].sum() / da['acc_inj'].sum() - 1:+.4f}  M50 {db[mk][-1] / da[mk][-1] - 1:+.4f}  "
                     f"steps {len(db['acc_t']) / len(da['acc_t']):.2f}x")
    a, b = RAW / "sedov/pe/N128/z0.1.npz", RAW / "sedov/pe/N128/z0.1_vred0.npz"
    if a.exists() and b.exists():
        da, db = np.load(a), np.load(b)
        lines.append("S-PE 128 z0.1, existing (v_red 1 placeholder) vs v_red 0: E_cr / injected "
                     f"{da['E_cr'][-1] / da['acc_inj'].sum():.4f} vs {db['E_cr'][-1] / db['acc_inj'].sum():.4f}; "
                     f"M50 end {da['M50'][-1]:.2f} vs {db['M50'][-1]:.2f}; injected {da['acc_inj'].sum():.5g} vs "
                     f"{db['acc_inj'].sum():.5g}")
        k = np.searchsorted(da["t"], 0.5 * da["t"][-1])
        lines.append(f"  E_cr ratio (v_red 1 / v_red 0) at t/t_end 0.25 / 0.5 / 0.75 / 1: " + " ".join(
            f"{np.interp(f * da['t'][-1], da['t'], da['E_cr']) / np.interp(f * db['t'][-1], db['t'], db['E_cr']):.4f}"
            for f in (0.25, 0.5, 0.75, 1.0)))
    for lab in ("lam0.1", "adv"):
        p = RAW / f"wind/N128/{lab}_off05.npz"
        if p.exists():
            d = np.load(p)
            t, E = d["t"], d["E_cr"]
            te = float(d["t_end"])
            k = int(np.argmin(np.abs(t - 0.5 * te)))
            sel = d["work_t"] >= t[k]
            lines.append(f"W 128 {lab}, injection off after t_end/2: E_cr(t_end) / E_cr(t_end/2) = {E[-1] / E[k]:.3f}; "
                         f"work over the 2nd half {d['work'][sel].sum() / E[k]:+.3f} E_cr(t_end/2)")
    text = "\n".join(lines)
    print(text)
    (PICS / "D0_checks.txt").write_text(text + "\n")


if __name__ == "__main__":
    which = sys.argv[1:] or ["d0", "wind", "sedov", "cwb"]
    for w in which:
        {"wind": analyse_wind, "sedov": analyse_sedov, "cwb": analyse_cwb, "d0": analyse_d0}[w]()
