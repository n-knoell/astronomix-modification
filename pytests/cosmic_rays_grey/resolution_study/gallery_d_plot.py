"""Extension-D galleries (CPU), in the style of G1: what CR diffusion does to each setup at 192^3.

Reads ``RAW/gallery_d/*.npz`` (``gallery_d_extract.py``, GPU) and the study files. One figure per
setup, one column per CR transport (advective, lambda = 0.03 / 0.1 / 0.3), final state of each run:

* row 1: density in the mid-plane; row 2: CR energy density in the mid-plane (log, 4 decades, so the
  precursor shows); row 3: the injecting shock surface in 3D (upper hemisphere; W: forward shock only),
  coloured by the DSA Mach number over the advective run's median at the same radius (Sedov) or at t_end
  (W);
* bottom: shell-mean rho and P_cr around the front (all transports, 192^3; dashed: planar precursor
  exp(-(r - R) v_sh / kappa)), and the area-weighted Mach distribution of the final state against
  cells per shock radius for every N (study files: 16 / 50 / 84%), over the advective median at the
  same N.

Figures: ``pics/G3_wind_diffusion_gallery.png``, ``G4_ss_diffusion_gallery.png``,
``G5_cp_z0.1_diffusion_gallery.png``, ``G6_cp_kr13_diffusion_gallery.png``.

Usage: python gallery_d_plot.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402, F401

from analyse import INK, MUTED, _style  # noqa: E402
from gallery_plot import show_slice, surface_3d  # noqa: E402
from sedov_diag import weighted_quantiles  # noqa: E402

RAW = Path("/export/scratch/nknoell/resolution_study")
PICS = Path(__file__).resolve().parent / "pics"
N = 192  # reset per figure in __main__ (W 320, Sedov 384)
LAM_COL = {0.0: "#2a2a2a", 0.03: "#9ecae1", 0.1: "#4292c6", 0.3: "#08306b"}
CMAP_RHO, CMAP_ECR = "viridis", "magma"


def v_sh_of_t(t, R):
    ok = np.isfinite(R) & (R > 0)
    c = np.polyfit(np.log(t[ok]), np.log(R[ok]), 2)
    return (2 * c[0] * np.log(t) + c[1]) * R / t


def load_columns(setup, names):
    cols = []
    for lam, name in names:
        f = RAW / "gallery_d" / f"{setup}_{name}_N{N}.npz"
        if f.exists():
            cols.append((lam, name, dict(np.load(f))))
    return cols


def figure(setup, model, title, out, study_dir, study_names, half_box, mach_key, wind=False):
    names = [(0.0, study_names[0])] + [(lam, f"{study_names[1]}{lam:g}") for lam in (0.03, 0.1, 0.3)]
    cols = load_columns(setup, names)
    if len(cols) < 2:
        print("missing gallery data for", setup, model)
        return
    study = {lam: np.load(study_dir / f"N{N}" / f"{name}.npz") for lam, name in names
             if (study_dir / f"N{N}" / f"{name}.npz").exists()}
    adv = study[0.0]

    def m_ref(R, n_study=None):
        a = n_study if n_study is not None else adv
        if wind:
            return float(a[mach_key][-1])
        return float(np.interp(R, a["R_front"], a[mach_key]))

    rho_max = max(np.percentile(c[2]["sl_rho"], 99.9) for c in cols)
    ecr_max = max(c[2]["sl_ecr"].max() for c in cols)
    norm_rho = Normalize(vmin=0.0, vmax=rho_max)
    norm_ecr = LogNorm(vmin=ecr_max * 1e-4, vmax=ecr_max)
    norm_m = TwoSlopeNorm(vcenter=1.0, vmin=0.8, vmax=1.2)

    fig = plt.figure(figsize=(13.0, 14.2))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.05, 1.15], hspace=0.14,
                          left=0.05, right=0.92, top=0.94, bottom=0.05)
    top = [gs[k].subgridspec(1, 4, wspace=0.04) for k in range(3)]
    for j, (lam, name, d) in enumerate(cols):
        R = float(d["R_front"])
        dx = float(d["dx"])
        ext = (-half_box, half_box, -half_box, half_box)
        crop = min(half_box, 1.3 * R)
        ax = fig.add_subplot(top[0][j])
        im_rho = show_slice(ax, d["sl_rho"] / float(d["rho0"]), ext, Normalize(0, rho_max / float(d["rho0"])),
                            CMAP_RHO, crop)
        lab = "advective" if lam == 0 else f"λ = {lam:g}  (κ = {float(d['kappa']):.2g})"
        ax.set_title(f"{lab}\nR/dx = {R / dx:.0f},  t = {float(d['t']):.3g}", color=INK, fontsize=10.5)
        if j == 0:
            ax.set_ylabel("density ρ / ρ₀\n(mid-plane)", color=INK)
        ax = fig.add_subplot(top[1][j])
        im_ecr = show_slice(ax, d["sl_ecr"], ext, norm_ecr, CMAP_ECR, crop)
        circ = plt.Circle((0, 0), R, fill=False, color="white", lw=0.6, ls=":")
        ax.add_patch(circ)
        if j == 0:
            ax.set_ylabel("CR energy density e_cr\n(mid-plane; dotted: front)", color=INK)
        ax = fig.add_subplot(top[2][j], projection="3d")
        mrat = d["surf_mach"].astype(float) / m_ref(R)
        lim = 1.05 * R
        r_s = np.sqrt(np.sum(d["surf_pos"].astype(float) ** 2, axis=1))
        keep = (lambda p, rs=r_s, R=R: (p[:, 2] >= 0.0) & (rs > 0.6 * R)) if wind else (lambda p: p[:, 2] >= 0.0)
        surface_3d(ax, d["surf_pos"], mrat, norm_m, "RdBu", int(round(2 * half_box / dx)),
                   ((-lim, lim), (-lim, lim), (0.0, lim)), keep=keep, elev=32, zoom=1.5,
                   scale=(2 * half_box / (2 * lim)) ** 2 * (0.42 / 0.5) ** 2)
        sel = keep(d["surf_pos"])
        med = weighted_quantiles(mrat[sel], d["surf_w"].astype(float)[sel], (0.5,))[0]
        ax.text2D(0.5, 1.0, f"median M / M_adv = {med:.3f}", transform=ax.transAxes, ha="center", va="top",
                  color=INK, fontsize=9.5)
        if j == 0:
            ax.text2D(-0.06, 0.5, "injecting shock surface\n(upper hemisphere)", transform=ax.transAxes,
                      rotation=90, va="center", ha="center", color=INK, fontsize=10)
    for k, (im, label) in enumerate(((im_rho, "ρ / ρ₀"), (im_ecr, "e_cr"))):
        bb = top[k][3].get_position(fig)
        cax = fig.add_axes([0.93, bb.y0 + 0.1 * bb.height, 0.012, 0.8 * bb.height])
        fig.colorbar(im, cax=cax).set_label(label, color=INK)
    bb = top[2][3].get_position(fig)
    cax = fig.add_axes([0.93, bb.y0 + 0.1 * bb.height, 0.012, 0.8 * bb.height])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_m, cmap="RdBu"), cax=cax).set_label(
        "DSA Mach / advective median" + (" (t_end)" if wind else " (same R)"), color=INK)

    bot = gs[3].subgridspec(1, 3, wspace=0.28)
    ax_r, ax_e, ax_m = (fig.add_subplot(bot[k]) for k in range(3))
    for lam, name, d in cols:
        R = float(d["R_front"])
        ok = d["prof_counts"] > 0
        x = d["prof_r"][ok] / R
        c = LAM_COL[lam]
        lab = "advective" if lam == 0 else f"λ {lam:g}"
        ax_r.plot(x, d["prof_rho"][ok] / float(d["rho0"]), color=c, lw=1.5, label=lab)
        pcr = d["prof_ecr"][ok] / 3.0 / float(d["p0"])
        ax_e.semilogy(x, pcr, color=c, lw=1.5, label=lab)
        if lam > 0:
            s = study[lam]
            v = v_sh_of_t(s["t"], s["R_front"])[-1]
            i = int(np.searchsorted(x, 1.0))
            xx = x[i:]
            ax_e.semilogy(xx, pcr[i] * np.exp(-(xx - 1.0) * R * v / float(d["kappa"])), color=c, lw=0.9, ls="--")
    for ax, ylab in ((ax_r, "shell-mean ρ / ρ₀"), (ax_e, "shell-mean P_cr / p₀")):
        ax.set_xlim(0.6, 1.6 if wind else 1.45)
        ax.set_xlabel("r / R_front", color=INK)
        ax.set_ylabel(ylab, color=INK)
        _style(ax)
    ax_e.set_ylim(max(1e-4, ax_e.get_ylim()[0]), None)
    ax_r.legend(frameon=False, fontsize=8, loc="upper left")
    ax_e.text(0.98, 0.95, "dashed: planar exp(−(r−R) v_sh/κ)", transform=ax_e.transAxes, ha="right", va="top",
              fontsize=8, color=MUTED)

    # Mach distribution over N (study files), over the advective median at the same N
    ns = sorted(int(p.name[1:]) for p in study_dir.glob("N*") if p.is_dir())
    offs = {0.0: 0.94, 0.03: 0.98, 0.1: 1.02, 0.3: 1.06}
    q = ("M16_fs", "M50_fs", "M84_fs") if wind else ("M16", "M50", "M84")
    for lam, name in names:
        for n in ns:
            f = study_dir / f"N{n}" / f"{name}.npz"
            fa = study_dir / f"N{n}" / f"{names[0][1]}.npz"
            if not (f.exists() and fa.exists()):
                continue
            s, a = np.load(f), np.load(fa)
            R = float(s["R_front"][-1])
            ref = float(a[q[1]][-1]) if wind else float(np.interp(R, a["R_front"], a[q[1]]))
            dx = float(s["dx"]) if "dx" in s.files else 1.0 / n
            xr = R / dx * offs[lam]
            ax_m.vlines(xr, s[q[0]][-1] / ref, s[q[2]][-1] / ref, color=LAM_COL[lam], lw=4)
            ax_m.plot(xr, s[q[1]][-1] / ref, "o", ms=6, color=LAM_COL[lam], mec="white", mew=1.0)
    ax_m.axhline(1.0, color=INK, lw=0.8)
    ax_m.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator, ScalarFormatter
    ax_m.xaxis.set_major_locator(FixedLocator([10, 20, 30, 50, 70, 100, 200]))
    ax_m.xaxis.set_major_formatter(ScalarFormatter())
    ax_m.xaxis.set_minor_locator(NullLocator())
    ax_m.xaxis.set_minor_formatter(NullFormatter())
    ax_m.set_xlabel("cells per shock radius R / dx (final state)", color=INK)
    ax_m.set_ylabel("DSA Mach / advective median (same N)", color=INK)
    ax_m.text(0.98, 0.04, "dot: median, bar: 16-84%\n(area-weighted, final state;\nλ offset in x for legibility)",
              transform=ax_m.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED)
    from matplotlib.lines import Line2D
    ax_m.legend(handles=[Line2D([], [], color=LAM_COL[l], lw=4, label="advective" if l == 0 else f"λ {l:g}")
                         for l in (0.0, 0.03, 0.1, 0.3)], frameon=False, fontsize=8, loc="upper left")
    _style(ax_m)
    fig.suptitle(title, color=INK, fontsize=14)
    fig.savefig(PICS / out, dpi=130)
    plt.close(fig)
    print("saved", PICS / out)


if __name__ == "__main__":
    N = 320  # W runs exist at 256 / 320
    figure("wind", None, f"Wind bubble (W, ζ = 0.1) at t_end, {N}³: advective vs CR diffusion",
           "G3_wind_diffusion_gallery.png", RAW / "wind", ("adv", "lam"), 0.6, "M50_fs", wind=True)
    N = 384  # Sedov D5 runs exist at 256 / 384
    figure("ss", "z0.1", f"Sedov S-SS (ζ = 0.1), final state, {N}³: advective vs CR diffusion",
           "G4_ss_diffusion_gallery.png", RAW / "sedov" / "ss", ("z0.1", "z0.1_lam"), 0.5, "M50")
    figure("cp_p0.6", "z0.1", f"Sedov S-CP (ζ = 0.1), final state, {N}³: advective vs CR diffusion",
           "G5_cp_z0.1_diffusion_gallery.png", RAW / "sedov" / "cp_p0.6", ("z0.1", "z0.1_lam"), 0.5, "M50")
    figure("cp_p0.6", "kr13", f"Sedov S-CP (KR13), final state, {N}³: advective vs CR diffusion",
           "G6_cp_kr13_diffusion_gallery.png", RAW / "sedov" / "cp_p0.6", ("kr13", "kr13_lam"), 0.5, "M50")
