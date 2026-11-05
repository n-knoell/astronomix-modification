"""Resolution gallery (CPU): what the Sedov blast and the CWB look like at each N.

Reads ``RAW/gallery/*.npz`` (``gallery_extract.py``, GPU) plus the study files and
makes two multi-panel figures, one column per resolution (zeta = 0.1, t_end):

* ``pics/G1_sedov_gallery.png``: S-PE point explosion, N 32 / 64 / 128 / 224.
  Rows: density and CR energy density in the mid-plane (one colour scale per row),
  the injecting shock surface in 3D (upper hemisphere) coloured by the DSA Mach
  number over the two-fluid Sedov reference; bottom: shell-mean rho and e_cr
  profiles for all seven N against the exact two-fluid solution, and the
  area-weighted Mach distribution at t_end against R / dx with the study fit
  1 - 0.55 (R/dx)^-0.6.
* ``pics/G2_cwb_gallery.png``: the CWB, N 64 / 96 / 128 / 192. Rows: density and
  e_cr in the orbital plane, injecting surface cells in 3D in the view of
  ``cwb_shocked_cells_3d_<N>.png`` (box 0-1, hot colour map); bottom: binary-axis rho, e_cr, and finder Mach for all five N.

Usage: python gallery_plot.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize, TwoSlopeNorm  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402, F401

from analyse import INK, MUTED, PICS, RAW, _style  # noqa: E402
from sedov_diag import weighted_quantiles  # noqa: E402

E_EXPLOSION, RHO0, GAMMA = 1.0, 1.0, 5.0 / 3.0
SEDOV_SHOW = (32, 64, 128, 224)
CWB_SHOW = (64, 96, 128, 192)
CMAP_RHO, CMAP_ECR = "viridis", "magma"


def _two_fluid():
    from astronomix.test_setups.reference_solutions.sedov_two_fluid import solve_sedov_two_fluid
    return solve_sedov_two_fluid(0.1, convention="gas_rh")


def load_gallery(prefix, ns):
    return {n: dict(np.load(RAW / "gallery" / f"{prefix}_N{n}.npz")) for n in ns}


def seq_colors(ns, cmap="Blues"):
    cm = plt.get_cmap(cmap)
    return {n: cm(0.35 + 0.6 * i / max(len(ns) - 1, 1)) for i, n in enumerate(ns)}


def show_slice(ax, img, extent, norm, cmap, crop):
    im = ax.imshow(img.T, origin="lower", extent=extent, norm=norm, cmap=cmap, interpolation="nearest")
    ax.set_xlim(-crop, crop)
    ax.set_ylim(-crop, crop)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.5)
    return im


def surface_3d(ax, pos, c, norm, cmap, n, lims, keep, elev=28, azim=-58, scale=1.0, zoom=1.35):
    """Injecting surface cells as square markers of about the projected cell size."""
    sel = keep(pos)
    p, cc = pos[sel], c[sel]
    span = max(hi - lo for lo, hi in lims)
    size = scale * (2.6 * 72 * zoom * (1.0 / n) / span) ** 2
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=cc, norm=norm, cmap=cmap, s=size, marker="s",
               linewidths=0, depthshade=False, rasterized=True)
    for setter, (lo, hi) in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), lims):
        setter(lo, hi)
    ax.set_box_aspect(tuple(hi - lo for lo, hi in lims), zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def cwb_surface_3d(ax, d, s, norm, cmap):
    """The 3D view of ``shock_finder3D/cwb_shock_finder.plot_shocked_cells_3d``
    (``plot_helper.plot_shock_surface_3d``, SCATTER): box coordinates 0-1 with axes and
    panes, default perspective, every cell drawn, ``hot`` colour map, cyan "+" at the
    contact; the shock-direction arrows are left out (not stored by the extraction)."""
    n = int(d["N"])
    p = d["surf_pos"] + 0.5
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=d["surf_mach"], norm=norm, cmap=cmap,
               s=15 * 0.35 * (128 / n) ** 2, alpha=0.8, linewidths=0, rasterized=True)
    st = d["stars"] + 0.5
    ax.scatter(st[:, 0], st[:, 1], st[:, 2], marker="*", s=70, color="#2a78d6", edgecolors=INK,
               linewidths=0.5, depthshade=False, label="stars")
    xc = float(s["x_contact"][-1]) + 0.5
    ax.scatter([xc], [0.5], [0.5], color="cyan", marker="+", s=110, linewidths=2, depthshade=False,
               label="contact discontinuity")
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(0, 1)
    ax.set_box_aspect([1, 1, 1], zoom=0.95)
    ax.view_init(elev=30, azim=-60)
    ticks = [0, 0.5, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.tick_params(labelsize=7, pad=0)
    for lab, setl in (("x", ax.set_xlabel), ("y", ax.set_ylabel), ("z", ax.set_zlabel)):
        setl(lab, fontsize=8, labelpad=-6)


def sedov_figure():
    ref = _two_fluid()
    ns_all = sorted(int(p.stem.split("_N")[1]) for p in (RAW / "gallery").glob("sedov_N*.npz"))
    g = load_gallery("sedov", ns_all)
    t = g[ns_all[-1]]["t"]
    R_ref = float(ref.shock_radius(t, E_EXPLOSION, RHO0))
    c0 = np.sqrt(GAMMA * 1e-4 / RHO0)
    M_ref = float(ref.shock_velocity(t, E_EXPLOSION, RHO0)) / c0

    rho_max = max(g[n]["sl_rho"].max() for n in SEDOV_SHOW)
    ecr_max = max(g[n]["sl_ecr"].max() for n in SEDOV_SHOW)
    norm_rho = Normalize(vmin=0.0, vmax=rho_max)
    norm_ecr = LogNorm(vmin=ecr_max * 1e-3, vmax=ecr_max)
    norm_m = TwoSlopeNorm(vcenter=1.0, vmin=0.8, vmax=1.2)
    cmap_m = "RdBu"

    fig = plt.figure(figsize=(13.0, 14.2))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1.05, 1.15], hspace=0.12,
                          left=0.05, right=0.92, top=0.95, bottom=0.05)
    top = [gs[k].subgridspec(1, 4, wspace=0.04) for k in range(3)]
    crop = 0.47
    for j, n in enumerate(SEDOV_SHOW):
        d = g[n]
        h = 0.5
        ext = (-h, h, -h, h)
        ax = fig.add_subplot(top[0][j])
        im_rho = show_slice(ax, d["sl_rho"], ext, norm_rho, CMAP_RHO, crop)
        ax.set_title(f"N = {n}³    R/dx = {R_ref * n:.0f}", color=INK, fontsize=11)
        if j == 0:
            ax.set_ylabel("density ρ / ρ₀\n(mid-plane)", color=INK)
        ax = fig.add_subplot(top[1][j])
        im_ecr = show_slice(ax, d["sl_ecr"], ext, norm_ecr, CMAP_ECR, crop)
        if j == 0:
            ax.set_ylabel("CR energy density e_cr\n(mid-plane)", color=INK)
        ax = fig.add_subplot(top[2][j], projection="3d")
        mrat = d["surf_mach"] / M_ref
        surface_3d(ax, d["surf_pos"], mrat, norm_m, cmap_m, n, ((-0.42, 0.42), (-0.42, 0.42), (0.0, 0.42)),
                   keep=lambda p: p[:, 2] >= 0.0, elev=32, zoom=1.5)
        med = weighted_quantiles(mrat.astype(float), d["surf_w"].astype(float), (0.5,))[0]
        ax.text2D(0.5, 1.0, f"median M / M_ref = {med:.3f}", transform=ax.transAxes,
                  ha="center", va="top", color=INK, fontsize=9.5)
        if j == 0:
            ax.text2D(-0.06, 0.5, "injecting shock surface\n(upper hemisphere)", transform=ax.transAxes,
                      rotation=90, va="center", ha="center", color=INK, fontsize=10)
    # colour bars, one per row
    for k, (im, label) in enumerate(((im_rho, "ρ / ρ₀"), (im_ecr, "e_cr"))):
        bb = top[k][3].get_position(fig)
        cax = fig.add_axes([0.93, bb.y0 + 0.1 * bb.height, 0.012, 0.8 * bb.height])
        fig.colorbar(im, cax=cax).set_label(label, color=INK)
    bb = top[2][3].get_position(fig)
    cax = fig.add_axes([0.93, bb.y0 + 0.1 * bb.height, 0.012, 0.8 * bb.height])
    sm = plt.cm.ScalarMappable(norm=norm_m, cmap=cmap_m)
    fig.colorbar(sm, cax=cax).set_label("DSA Mach / M_ref (two-fluid Sedov)", color=INK)

    # bottom row: profiles and Mach distributions
    bot = gs[3].subgridspec(1, 3, wspace=0.28)
    cols = seq_colors(ns_all)
    r_ex = np.linspace(0.02, 1.15, 2000) * R_ref
    rho_ex, _, _, pcr_ex = ref.profiles(r_ex, t, E_EXPLOSION, RHO0, p_ambient=1e-4)
    ecr_ex = 3.0 * pcr_ex  # gamma_cr 4/3
    ax_r = fig.add_subplot(bot[0])
    ax_e = fig.add_subplot(bot[1])
    for n in ns_all:
        d = g[n]
        ok = d["prof_counts"] > 0
        x = d["prof_r"][ok] / R_ref
        lw = 1.6 if n in SEDOV_SHOW else 0.9
        ax_r.plot(x, d["prof_rho"][ok], color=cols[n], lw=lw, label=f"{n}³")
        ax_e.plot(x, d["prof_ecr"][ok], color=cols[n], lw=lw, label=f"{n}³")
    ax_r.plot(r_ex / R_ref, rho_ex, color=INK, lw=1.2, ls="--", label="two-fluid Sedov")
    ax_e.plot(r_ex / R_ref, ecr_ex, color=INK, lw=1.2, ls="--", label="two-fluid Sedov")
    for ax, lab in ((ax_r, "shell-mean ρ / ρ₀"), (ax_e, "shell-mean e_cr")):
        ax.set_xlim(0.55, 1.12)
        ax.set_xlabel("r / R_ref", color=INK)
        ax.set_ylabel(lab, color=INK)
        _style(ax)
    ax_r.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

    ax_m = fig.add_subplot(bot[2])
    xs = np.geomspace(8, 120, 100)
    ax_m.plot(xs, 1 - 0.55 * xs**-0.6, color=MUTED, lw=1.2, ls="--", label="study fit 1 − 0.55 (R/dx)^−0.6")
    for n in ns_all:
        d = g[n]
        m = d["surf_mach"].astype(float) / M_ref
        w = d["surf_w"].astype(float)
        q = weighted_quantiles(m, w, (0.025, 0.16, 0.5, 0.84, 0.975))
        x = R_ref * n
        ax_m.vlines(x, q[0], q[4], color=cols[n], lw=1.0)
        ax_m.vlines(x, q[1], q[3], color=cols[n], lw=5.0)
        ax_m.plot(x, q[2], "o", ms=7, color=cols[n], mec="white", mew=1.2)
        if n in SEDOV_SHOW:
            ax_m.annotate(f"{n}³", (x, q[0]), textcoords="offset points", xytext=(0, -12), ha="center",
                          fontsize=8, color=MUTED)
    ax_m.axhline(1.0, color=INK, lw=0.8)
    ax_m.set_xscale("log")
    ax_m.set_xlabel("cells per shock radius R / dx", color=INK)
    ax_m.set_ylabel("DSA Mach / M_ref", color=INK)
    ax_m.legend(frameon=False, fontsize=8, loc="lower right")
    ax_m.set_ylim(0.55, 1.03)
    _style(ax_m)
    ax_m.text(0.98, 0.16, "dot: median, bar: 16-84%, line: 2.5-97.5%\n(area-weighted, t_end)",
              transform=ax_m.transAxes, va="bottom", ha="right", fontsize=8, color=MUTED)

    fig.suptitle(f"Sedov blast (S-PE, ζ = 0.1) at t = {t:g}: resolution gallery", color=INK, fontsize=14)
    out = PICS / "G1_sedov_gallery.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


def cwb_figure():
    ns_all = sorted(int(p.stem.split("_N")[1]) for p in (RAW / "gallery").glob("cwb_N*.npz"))
    g = load_gallery("cwb", ns_all)
    study = {n: dict(np.load(RAW / "cwb" / f"N{n}" / "z0.1.npz")) for n in ns_all}
    stars = g[ns_all[-1]]["stars"]

    rho_lo = min(np.percentile(g[n]["sl_rho"], 0.5) for n in CWB_SHOW)
    rho_hi = max(g[n]["sl_rho"].max() for n in CWB_SHOW)
    ecr_hi = max(g[n]["sl_ecr"].max() for n in CWB_SHOW)
    norm_rho = LogNorm(vmin=rho_lo, vmax=rho_hi)
    norm_ecr = LogNorm(vmin=ecr_hi * 1e-4, vmax=ecr_hi)
    norm_m = Normalize(vmin=2.5, vmax=10.0)
    cmap_m = "hot"

    fig = plt.figure(figsize=(13.0, 13.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 0.95, 1.1], hspace=0.12,
                          left=0.05, right=0.92, top=0.95, bottom=0.05)
    top = [gs[k].subgridspec(1, 4, wspace=0.04) for k in range(3)]
    crop = 0.5
    for j, n in enumerate(CWB_SHOW):
        d = g[n]
        ext = (-0.5, 0.5, -0.5, 0.5)
        r_inj = float(d["r_inj"])
        ax = fig.add_subplot(top[0][j])
        im_rho = show_slice(ax, d["sl_rho"], ext, norm_rho, CMAP_RHO, crop)
        ax.set_title(f"N = {n}³    R_inj = {r_inj * n:.0f} cells", color=INK, fontsize=11)
        if j == 0:
            ax.set_ylabel("density ρ\n(orbital plane)", color=INK)
        for s in stars:
            ax.plot(s[0], s[1], "*", ms=9, color="white", mec=INK, mew=0.6)
        ax = fig.add_subplot(top[1][j])
        im_ecr = show_slice(ax, d["sl_ecr"], ext, norm_ecr, CMAP_ECR, crop)
        if j == 0:
            ax.set_ylabel("CR energy density e_cr\n(orbital plane)", color=INK)
        ax = fig.add_subplot(top[2][j], projection="3d")
        cwb_surface_3d(ax, d, study[n], norm_m, cmap_m)
        if j == 0:
            ax.text2D(-0.08, 0.5, "injecting shock cells", transform=ax.transAxes,
                      rotation=90, va="center", ha="center", color=INK, fontsize=10)
            ax.legend(loc="upper left", fontsize=7.5, frameon=False, bbox_to_anchor=(0.0, 1.02))
    for k, (im, label) in enumerate(((im_rho, "ρ (code units)"), (im_ecr, "e_cr"))):
        bb = top[k][3].get_position(fig)
        cax = fig.add_axes([0.93, bb.y0 + 0.1 * bb.height, 0.012, 0.8 * bb.height])
        fig.colorbar(im, cax=cax).set_label(label, color=INK)
    bb = top[2][3].get_position(fig)
    cax = fig.add_axes([0.93, bb.y0 + 0.1 * bb.height, 0.012, 0.8 * bb.height])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_m, cmap=cmap_m), cax=cax).set_label(
        "DSA Mach number", color=INK)

    bot = gs[3].subgridspec(1, 3, wspace=0.28)
    cols = seq_colors(ns_all)
    axs = [fig.add_subplot(bot[k]) for k in range(3)]
    for n in ns_all:
        s = study[n]
        x = s["x_axis"]
        lw = 1.6 if n in CWB_SHOW else 0.9
        axs[0].plot(x, s["ax_rho"][-1], color=cols[n], lw=lw, label=f"{n}³")
        axs[1].plot(x, s["ax_ecr"][-1], color=cols[n], lw=lw, label=f"{n}³")
    for ax, lab in ((axs[0], "ρ on the binary axis"), (axs[1], "e_cr on the binary axis")):
        ax.set_xlabel("x (box units)", color=INK)
        ax.set_ylabel(lab, color=INK)
        ax.set_yscale("log")
        for st in stars:
            ax.axvline(st[0], color=MUTED, lw=0.6, ls=":")
        _style(ax)
    axs[1].set_xlim(-0.05, 0.3)
    axs[1].set_ylim(ecr_hi * 1e-4, ecr_hi * 3)
    axs[0].legend(frameon=False, fontsize=8, ncol=2)
    # per-shock DSA Mach at t_end against the injection radius in cells
    ax = axs[2]
    for cls, colour, label, dx_off in (("wind1", "#2a78d6", "wind-1 shock (fast, dense wind)", -0.06),
                                       ("wind2", "#eb6834", "wind-2 shock", 0.06)):
        x = np.array([float(study[n]["r_inj"]) * n for n in ns_all])
        m16, m50, m84 = (np.array([float(study[n][f"{q}_{cls}"][-1]) for n in ns_all])
                         for q in ("M16", "M50", "M84"))
        xo = x * (1 + dx_off)
        ax.vlines(xo, m16, m84, color=colour, lw=4.0, alpha=0.55)
        ax.plot(xo, m50, "o-", color=colour, ms=7, mec="white", mew=1.2, lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.set_xticklabels(["2", "3", "4", "5", "6"])
    ax.set_xlabel("injection radius R_inj (cells)", color=INK)
    ax.set_ylabel("area-weighted DSA Mach (median, 16-84%)", color=INK)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    t_end = float(study[ns_all[-1]]["t_end"])
    fig.suptitle(f"Colliding-wind binary (ζ = 0.1) at t_end = {t_end:.3g}: resolution gallery",
                 color=INK, fontsize=14)
    out = PICS / "G2_cwb_gallery.png"
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["sedov", "cwb"]
    if "cwb" in which:
        cwb_figure()
    if "sedov" in which:
        sedov_figure()
