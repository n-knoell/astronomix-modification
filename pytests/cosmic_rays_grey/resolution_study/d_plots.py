"""Extension D, time-resolved figures (CPU). Complements ``d_analyse.py`` (final-state tables).

* ``pics/D1_wind_time.png`` (W): (a) E_cr / injected over time, (b) forward-shock M50 against R_fs / dx
  over the run, (c) precursor length against kappa / v_sh (both in cells, every snapshot), (d) nominal
  injection rate at the forward and the termination shock (192^3), (e) cumulative compression work at /
  ahead of the front and behind it (192^3), (f) P_cr around the forward shock at three times (lambda 0.1,
  192^3) with the planar exp(-x v_sh / kappa).
* ``pics/D2_<setup>_<model>.png`` (S-SS, S-CP zeta 0.1 / KR13): (a) finder M50 over the advective run's
  M50 at the same N and R, (b) finder M50 over the Rankine-Hugoniot Mach number of the gas-pressure jump
  across the sub-shock (p just behind / p 3 dx ahead) and across the whole structure (p behind / p_0),
  (c) nominal injection rate over the advective run's, (d) precursor length against kappa / v_sh, (e)
  E_cr / (E_cr0 + injected) over time (192^3), (f) false-positive surface fraction.

Usage: python d_plots.py [wind] [sedov]
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import INK, MUTED, _style  # noqa: E402

RAW = Path("/export/scratch/nknoell/resolution_study")
PICS = Path(__file__).resolve().parent / "pics"
LAM_COL = {0.0: "#2a2a2a", 0.03: "#9ecae1", 0.1: "#4292c6", 0.3: "#08306b"}
N_ALPHA = {48: 0.35, 64: 0.45, 96: 0.6, 128: 0.75, 160: 0.85, 192: 1.0}
GAMMA = 5.0 / 3.0


def lam_label(lam):
    return "advective" if lam == 0 else f"λ {lam:g}"


def v_sh_of_t(t, R):
    """Shock speed from a quadratic fit of ln R against ln t (smooth derivative)."""
    ok = np.isfinite(R) & (R > 0)
    c = np.polyfit(np.log(t[ok]), np.log(R[ok]), 2)
    return (2 * c[0] * np.log(t) + c[1]) * R / t


def precursor_len(r, e, R, r_max):
    h = r[1] - r[0]
    i = int(np.searchsorted(r, R))
    j = int(np.searchsorted(r, r_max))
    return float(np.sum(e[i:j]) * h / max(e[i], 1e-300))


def m_rh(y):
    """Rankine-Hugoniot Mach number from the pressure ratio y = p2 / p1 (gamma 5/3)."""
    y = np.maximum(y, 1.0)
    return np.sqrt(((GAMMA + 1) * y + (GAMMA - 1)) / (2 * GAMMA))


def legend_lams(ax, lams, extra=()):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=LAM_COL[l], lw=2, label=lam_label(l)) for l in lams] + list(extra)
    ax.legend(handles=h, frameon=False, fontsize=8)


# ---------------------------------------------------------------------------------------- W
def wind_figure():
    runs = {}
    for p in sorted((RAW / "wind").glob("N*/*.npz")):
        if p.stem in ("adv",) or (p.stem.startswith("lam") and "_" not in p.stem):
            lam = 0.0 if p.stem == "adv" else float(p.stem[3:])
            runs[(lam, int(p.parent.name[1:]))] = np.load(p)
    if not runs:
        return
    lams = sorted({k[0] for k in runs})
    ns = sorted({k[1] for k in runs})
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.6))
    ax = ax.ravel()
    for (lam, n), d in sorted(runs.items()):
        col, al = LAM_COL[lam], N_ALPHA.get(n, 1.0)
        te = float(d["t_end"])
        t = d["t"]
        ax[0].plot(t / te, d["E_cr"] / np.maximum(d["inj_acc"], 1e-30), color=col, alpha=al, lw=1.6 if n == 192 else 1.0)
        dx = float(d["dx"])
        ax[1].plot(d["R_front"] / dx, d["M50_fs"], color=col, alpha=al, lw=1.6 if n == 192 else 1.0)
        if lam > 0:
            v = v_sh_of_t(t, d["R_front"])
            r = d["r_bins"]
            l = np.array([precursor_len(r, d["prof_ecr"][k], d["R_front"][k], 0.5 * float(d["box"]))
                          for k in range(len(t))])
            model = float(d["kappa"]) / v / dx
            late = t > 0.1 * te
            ax[2].plot(model[late], (l / dx)[late], "o-", ms=2.5, color=col, alpha=al, lw=0.8)
    ax[0].axhline(1, color=MUTED, lw=0.8, ls=":")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("t / t_end")
    ax[0].set_ylabel("E_cr / injected")
    ax[0].set_title("(a) CR energy per injected energy")
    ax[1].set_xlabel("R_fs / dx (over the run)")
    ax[1].set_ylabel("forward-shock M50 (finder)")
    ax[1].set_title("(b) forward-shock Mach over the run")
    xs = np.geomspace(0.3, 30, 50)
    ax[2].plot(xs, xs, color=INK, lw=0.8, ls="--", label="planar κ / v_sh")
    ax[2].plot(xs, 0.62 * xs, color=MUTED, lw=0.8, ls=":", label="0.62 κ / v_sh")
    ax[2].axhline(1.2, color=MUTED, lw=0.8, ls="-.", label="advective shock width")
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("κ / v_sh in cells")
    ax[2].set_ylabel("measured precursor length in cells")
    ax[2].set_title("(c) precursor, every snapshot t > 0.1 t_end")
    ax[2].legend(frameon=False, fontsize=8, loc="upper left")
    n_top = ns[-1]
    for lam in lams:
        d = runs.get((lam, n_top))
        if d is None:
            continue
        te, t, col = float(d["t_end"]), d["t"], LAM_COL[lam]
        L = float(d["L_w"])
        ax[3].plot(t / te, d["inj_fs"] / L, color=col, lw=1.6)
        ax[3].plot(t / te, d["inj_ts"] / L, color=col, lw=1.2, ls="--")
        dx = float(d["dx"])
        R = np.interp(d["work_t"], t, d["R_front"])
        w = np.asarray(d["work"], dtype=np.float64)
        front = d["r_bins"][None, :] >= (R[:, None] - 2 * dx)
        cf = np.cumsum((w * front).sum(axis=1))
        cb = np.cumsum((w * ~front).sum(axis=1))
        ax[4].plot(d["work_t"] / te, cf / (L * te), color=col, lw=1.6)
        ax[4].plot(d["work_t"] / te, cb / (L * te), color=col, lw=1.2, ls="--")
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("t / t_end")
    ax[3].set_ylabel("nominal injection rate / L_w")
    ax[3].set_title(f"(d) injection: forward shock (solid), termination shock (dashed), {n_top}³")
    ax[4].set_xlabel("t / t_end")
    ax[4].set_ylabel("cumulative −P_cr ∇·v work / (L_w t_end)")
    ax[4].set_title(f"(e) work at / ahead of the front (solid), behind (dashed), {n_top}³")
    ax[4].axhline(0, color=MUTED, lw=0.6)
    d = runs.get((0.1, n_top))
    if d is not None:
        te, t, r = float(d["t_end"]), d["t"], d["r_bins"]
        v = v_sh_of_t(t, d["R_front"])
        cm = plt.get_cmap("Blues")
        for j, frac in enumerate((0.25, 0.5, 1.0)):
            k = int(np.argmin(np.abs(t - frac * te)))
            R = d["R_front"][k]
            sel = (r > 0.6 * R) & (r < 0.6)
            pcr = d["prof_ecr"][k][sel] / 3.0 / float(d["p0"])
            c = cm(0.45 + 0.25 * j)
            ax[5].semilogy(r[sel] / R, pcr, color=c, lw=1.6, label=f"t / t_end {t[k] / te:.2f}")
            i = int(np.searchsorted(r, R))
            x = r[i:][r[i:] < 0.6]
            ax[5].semilogy(x / R, d["prof_ecr"][k][i] / 3 / float(d["p0"]) * np.exp(-(x - R) * v[k] / float(d["kappa"])),
                           color=c, lw=0.9, ls="--")
        ax[5].set_ylim(1e-3, None)
        ax[5].set_xlabel("r / R_fs")
        ax[5].set_ylabel("P_cr / p_0 (radial mean)")
        ax[5].set_title(f"(f) λ 0.1, {n_top}³: P_cr vs planar exp(−x v_sh/κ) (dashed)")
        ax[5].legend(frameon=False, fontsize=8)
    from matplotlib.lines import Line2D
    legend_lams(ax[0], lams, [Line2D([], [], color=MUTED, lw=1.6, label=f"thicker / darker: higher N ({ns[0]}-{ns[-1]})")])
    legend_lams(ax[1], lams)
    legend_lams(ax[4], lams)
    for a in ax:
        _style(a)
    fig.suptitle("W: wind bubble with CR diffusion over time (extension D1)", color=INK)
    fig.tight_layout()
    fig.savefig(PICS / "D1_wind_time.png", dpi=130)
    plt.close(fig)
    print("saved", PICS / "D1_wind_time.png")


# ---------------------------------------------------------------------------------------- Sedov
def sedov_figure(sdir, model):
    runs = {}
    for p in sorted((RAW / "sedov" / sdir).glob(f"N*/{model}*.npz")):
        name = p.stem
        if "_vx2" in name or not (name == model or name.startswith(model + "_lam")):
            continue
        lam = 0.0 if name == model else float(name.split("_lam")[1])
        runs[(lam, int(p.parent.name[1:]))] = np.load(p)
    ns_d = sorted({n for (l_, n) in runs if l_ > 0})
    runs = {k: v for k, v in runs.items() if k[1] in ns_d}
    if not ns_d:
        return
    lams = sorted({k[0] for k in runs})
    e0 = {n: float(runs[(l_, n)]["E_cr0"]) for (l_, n) in runs if l_ > 0 and "E_cr0" in runs[(l_, n)].files}
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.6))
    ax = ax.ravel()
    for (lam, n), d in sorted(runs.items()):
        col, al = LAM_COL[lam], N_ALPHA.get(n, 1.0)
        lw = 1.6 if n == ns_d[-1] else 0.9
        dx = 1.0 / n
        R, t = d["R_front"], d["t"]
        adv = runs.get((0.0, n))
        if lam > 0 and adv is not None:
            ma = np.interp(R, adv["R_front"], adv["M50"])
            ax[0].plot(R / dx, d["M50"] / ma, color=col, alpha=al, lw=lw)
            ia = np.interp(R, adv["R_front"], adv["inj_nominal"])
            ax[2].plot(R / dx, d["inj_nominal"] / ia, color=col, alpha=al, lw=lw)
        if "prof_p" in d.files:
            r = d["prof_r"]
            msub, mtot, l_cells, kmod = [], [], [], []
            v = v_sh_of_t(t, R)
            for k in range(len(t)):
                behind = (r >= R[k] - 4 * dx) & (r <= R[k])
                p2 = d["prof_p"][k][behind].max() if behind.any() else np.nan
                p1 = np.interp(R[k] + 3 * dx, r, d["prof_p"][k])
                msub.append(m_rh(p2 / p1))
                mtot.append(m_rh(p2 / float(d["p0"])))
                l_cells.append(precursor_len(r, d["prof_ecr"][k], R[k], 0.5) / dx)
                kmod.append(float(d["kappa"]) / v[k] / dx if lam > 0 else np.nan)
            ax[1].plot(R / dx, d["M50"] / np.array(msub), color=col, alpha=al, lw=lw)
            ax[1].plot(R / dx, d["M50"] / np.array(mtot), color=col, alpha=al, lw=lw, ls=":")
            if lam > 0:
                ax[3].plot(kmod, l_cells, "o-", ms=2.5, color=col, alpha=al, lw=0.8)
        if n == ns_d[-1]:
            inj = np.interp(t, d["acc_t"] + d["acc_dt"], np.cumsum(d["acc_inj"]))
            if n in e0:
                ax[4].plot(t, d["E_cr"] / (e0[n] + inj), color=col, lw=1.6)
        ax[5].plot(R / dx, d["false_pos_frac"], color=col, alpha=al, lw=lw)
    # advective baseline for (b): the stored final state (gallery_d_extract.py profiles), one point per run
    g = RAW / "gallery_d" / f"{sdir}_{model}_N{ns_d[-1]}.npz"
    adv = runs.get((0.0, ns_d[-1]))
    if g.exists() and adv is not None:
        gd = np.load(g)
        r, R, dx = gd["prof_r"], float(gd["R_front"]), 1.0 / ns_d[-1]
        behind = (r >= R - 4 * dx) & (r <= R)
        p2 = gd["prof_p"][behind].max()
        p1 = np.interp(R + 3 * dx, r, gd["prof_p"])
        ax[1].plot(R / dx, adv["M50"][-1] / m_rh(p2 / p1), "o", color=LAM_COL[0.0], ms=7,
                   label=f"advective, final state {ns_d[-1]}³, sub-shock")
        ax[1].plot(R / dx, adv["M50"][-1] / m_rh(p2 / float(gd["p0"])), "o", mfc="none", color=LAM_COL[0.0], ms=7,
                   label="advective, final state, total")
    ax[0].axhline(1, color=MUTED, lw=0.8)
    ax[0].set_ylabel("M50 / M50(advective, same N, same R)")
    ax[0].set_title("(a) finder Mach with diffusion vs without")
    ax[1].axhline(1, color=MUTED, lw=0.8)
    ax[1].set_ylabel("finder M50 / M_RH(gas-pressure jump)")
    ax[1].set_title("(b) vs sub-shock jump p(behind) / p(R + 3 dx) (solid) and total p / p₀ (dotted)")
    ax[2].axhline(1, color=MUTED, lw=0.8)
    ax[2].set_ylabel("injection rate / advective (same N, same R)")
    ax[2].set_title("(c) nominal injection rate with diffusion vs without")
    xs = np.geomspace(0.3, 60, 50)
    ax[3].plot(xs, xs, color=INK, lw=0.8, ls="--", label="planar κ / v_sh")
    ax[3].axhline(1.2, color=MUTED, lw=0.8, ls="-.", label="advective shock width")
    ax[3].set_xscale("log")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("κ / v_sh in cells")
    ax[3].set_ylabel("measured precursor length in cells")
    ax[3].set_title("(d) precursor, every snapshot")
    ax[3].legend(frameon=False, fontsize=8, loc="upper left")
    ax[4].set_xlabel("t")
    ax[4].set_ylabel("E_cr / (E_cr0 + injected)")
    ax[4].set_title(f"(e) CR energy budget, {ns_d[-1]}³")
    ax[5].set_ylabel("false-positive surface fraction (> 3 dx off the front)")
    ax[5].set_title("(f) finder false positives")
    for a in (ax[0], ax[1], ax[2], ax[5]):
        a.set_xlabel("R / dx (over the run)")
    from matplotlib.lines import Line2D
    legend_lams(ax[0], [l for l in lams if l > 0],
                [Line2D([], [], color=MUTED, lw=1.6, label=f"darker / thicker: higher N ({ns_d[0]}-{ns_d[-1]})")])
    legend_lams(ax[4], lams)
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        _style(a)
    title = {"ss": "S-SS (self-similar start, M ≈ 600 → 165)", "cp_p0.6": "S-CP (counter-pressure, M ≈ 20 → 2.4)"}[sdir]
    fig.suptitle(f"{title}, {'ζ = 0.1' if model == 'z0.1' else 'KR13'}, with CR diffusion (extension D2)", color=INK)
    fig.tight_layout()
    out = PICS / f"D2_{sdir.split('_')[0]}_{model}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    which = sys.argv[1:] or ["wind", "sedov"]
    if "wind" in which:
        wind_figure()
    if "sedov" in which:
        for sdir, model in (("ss", "z0.1"), ("cp_p0.6", "z0.1"), ("cp_p0.6", "kr13")):
            sedov_figure(sdir, model)
