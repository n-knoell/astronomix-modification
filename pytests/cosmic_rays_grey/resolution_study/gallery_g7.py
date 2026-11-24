"""G7: CWB (zeta = 0.1, 192^3) advective vs CR diffusion, analogous to G4 (S-SS).

Runs (``RAW/jobs/G7_cwb_*_N192.sh``, ``cwb_study.py``): ``g7_adv`` (``--advective``: no F_cr
relaxation, v_red 0) and ``g7_lam0.03 / 0.1 / 0.3`` (kappa 1.6 / 5.3 / 16, v_red 4100, the
extension-D values; RESULTS.md D3). Output ``RAW/cwb/N192/g7_*.npz``.

* ``extract`` (GPU): the finder with the DSA settings on each final state
  (``gallery_extract.extract``: orbital-plane slices, every injecting surface cell) ->
  ``RAW/gallery_d/cwb_g7_<name>_N192.npz``.
* ``plot`` (CPU): ``pics/G7_cwb_diffusion_gallery.png``. Rows: rho and e_cr (log, 4 decades)
  in the orbital plane, the injecting surface in 3D (absolute DSA Mach, as G2); bottom: rho and
  e_cr on the binary axis, per-shock DSA Mach (area-weighted median, 16-84%) for each column.

Usage: CUDA_VISIBLE_DEVICES=GPU-<uuid> python gallery_g7.py extract [--n 384];  python gallery_g7.py plot [--n 384]
(N 192: runs from ``G7_cwb_*_N192.sh``, figure ``G7_cwb_diffusion_gallery.png``; any other N: ``G7_cwb_*_N<n>.sh``
(``--gpu-diag``), figure ``G7_cwb_diffusion_gallery_N<n>.png``.)
"""

import sys

import numpy as np

from common import RAW

N = 192  # set from --n in __main__
RUNS = (("g7_adv", "advective", dict(advective=True)),
        ("g7_lam0.03", "λ = 0.03  (κ = 1.6)", dict(kappa=1.6, vred=4100.0)),
        ("g7_lam0.1", "λ = 0.1  (κ = 5.3)", dict(kappa=5.3, vred=4100.0)),
        ("g7_lam0.3", "λ = 0.3  (κ = 16)", dict(kappa=16.0, vred=4100.0)))
OUT = RAW / "gallery_d"


def run_path(name):
    return RAW / "cwb" / f"N{N}" / f"{name}.npz"


def extract_all():
    import _cwb_setup as cwb
    from common import require_pinned_gpu, save_npz
    from gallery_extract import extract
    from setups import build_cwb

    print("gpu", require_pinned_gpu(), flush=True)
    for name, _, kw in RUNS:
        if not run_path(name).exists():
            print(f"{name}: missing, skipped", flush=True)
            continue
        d = np.load(run_path(name))
        _, config, params, hd, rv = build_cwb(N, 0.1, **kw)
        out, *_ = extract(d["state_last"], config, params, rv, hd, origin=-0.5 * cwb.BOX_SIZE)
        out.update(t=float(d["t"][-1]), stars=np.asarray(d["stars"]), r_inj=float(d["r_inj"]))
        path = save_npz(OUT / f"cwb_{name}_N{N}.npz", **out)
        print(f"{name}: {out['surf_mach'].size} injecting cells -> {path}", flush=True)


def plot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize

    from analyse import INK, MUTED, PICS, _style
    from gallery_plot import CMAP_ECR, CMAP_RHO, cwb_surface_3d, show_slice

    runs = [(name, label) for name, label, _ in RUNS if (OUT / f"cwb_{name}_N{N}.npz").exists()]
    g = {name: dict(np.load(OUT / f"cwb_{name}_N{N}.npz")) for name, _ in runs}
    study = {name: dict(np.load(run_path(name))) for name, _ in runs}
    stars = g[runs[0][0]]["stars"]

    rho_lo = min(np.percentile(g[k]["sl_rho"], 0.5) for k in g)
    rho_hi = max(g[k]["sl_rho"].max() for k in g)
    ecr_hi = max(g[k]["sl_ecr"].max() for k in g)
    norm_rho = LogNorm(vmin=rho_lo, vmax=rho_hi)
    norm_ecr = LogNorm(vmin=ecr_hi * 1e-4, vmax=ecr_hi)
    norm_m = Normalize(vmin=2.5, vmax=10.0)
    cmap_m = "hot"

    fig = plt.figure(figsize=(13.0, 13.6))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 0.95, 1.1], hspace=0.14,
                          left=0.05, right=0.92, top=0.925, bottom=0.05)
    top = [gs[k].subgridspec(1, 4, wspace=0.04) for k in range(3)]
    ext = (-0.5, 0.5, -0.5, 0.5)
    for j, (name, label) in enumerate(runs):
        d, s = g[name], study[name]
        ax = fig.add_subplot(top[0][j])
        im_rho = show_slice(ax, d["sl_rho"], ext, norm_rho, CMAP_RHO, 0.5)
        ax.set_title(f"{label}\nt = {float(d['t']):.3g}", color=INK, fontsize=11)
        if j == 0:
            ax.set_ylabel("density ρ\n(orbital plane)", color=INK)
        for st in stars:
            ax.plot(st[0], st[1], "*", ms=9, color="white", mec=INK, mew=0.6)
        ax = fig.add_subplot(top[1][j])
        im_ecr = show_slice(ax, d["sl_ecr"], ext, norm_ecr, CMAP_ECR, 0.5)
        if j == 0:
            ax.set_ylabel("CR energy density e_cr\n(orbital plane)", color=INK)
        ax = fig.add_subplot(top[2][j], projection="3d")
        cwb_surface_3d(ax, d, s, norm_m, cmap_m)
        m1, m2 = (float(s[f"M50_{c}"][-1]) for c in ("wind1", "wind2"))
        ax.set_title(f"median M: wind 1 {m1:.2f}, wind 2 {m2:.2f}", color=INK, fontsize=9.5, pad=0)
        if j == 0:
            ax.text2D(-0.08, 0.5, "injecting shock cells", transform=ax.transAxes,
                      rotation=90, va="center", ha="center", color=INK, fontsize=10)
            ax.legend(loc="upper left", fontsize=7.5, frameon=False, bbox_to_anchor=(0.0, 1.02))
    for k, (im, lab) in enumerate(((im_rho, "ρ (code units)"), (im_ecr, "e_cr"))):
        bb = top[k][3].get_position(fig)
        cax = fig.add_axes([0.93, bb.y0 + 0.1 * bb.height, 0.012, 0.8 * bb.height])
        fig.colorbar(im, cax=cax).set_label(lab, color=INK)
    bb = top[2][3].get_position(fig)
    cax = fig.add_axes([0.93, bb.y0 + 0.1 * bb.height, 0.012, 0.8 * bb.height])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_m, cmap=cmap_m), cax=cax).set_label(
        "DSA Mach number", color=INK)

    bot = gs[3].subgridspec(1, 3, wspace=0.28)
    axs = [fig.add_subplot(bot[k]) for k in range(3)]
    blues = plt.get_cmap("Blues")
    cols = {"g7_adv": INK, "g7_lam0.03": blues(0.4), "g7_lam0.1": blues(0.65), "g7_lam0.3": blues(0.95)}
    short = {"g7_adv": "advective", "g7_lam0.03": "λ 0.03", "g7_lam0.1": "λ 0.1", "g7_lam0.3": "λ 0.3"}
    for name, _ in runs:
        s = study[name]
        axs[0].plot(s["x_axis"], s["ax_rho"][-1], color=cols[name], lw=1.4, label=short[name])
        axs[1].plot(s["x_axis"], s["ax_ecr"][-1], color=cols[name], lw=1.4, label=short[name])
    for ax, lab in ((axs[0], "ρ on the binary axis"), (axs[1], "e_cr on the binary axis")):
        ax.set_xlabel("x (box units)", color=INK)
        ax.set_ylabel(lab, color=INK)
        ax.set_yscale("log")
        for st in stars:
            ax.axvline(st[0], color=MUTED, lw=0.6, ls=":")
        _style(ax)
    axs[1].set_ylim(ecr_hi * 1e-4, ecr_hi * 3)
    axs[0].legend(frameon=False, fontsize=8)
    ax = axs[2]
    xs = np.arange(len(runs))
    for cls, colour, label, off in (("wind1", "#2a78d6", "wind-1 shock (fast, dense wind)", -0.08),
                                    ("wind2", "#eb6834", "wind-2 shock", 0.08)):
        m16, m50, m84 = (np.array([float(study[name][f"{q}_{cls}"][-1]) for name, _ in runs])
                         for q in ("M16", "M50", "M84"))
        ax.vlines(xs + off, m16, m84, color=colour, lw=4.0, alpha=0.55)
        ax.plot(xs + off, m50, "o", color=colour, ms=7, mec="white", mew=1.2, label=label)
    ax.set_xticks(xs)
    ax.set_xticklabels([short[name] for name, _ in runs])
    ax.set_ylabel("area-weighted DSA Mach (median, 16-84%)", color=INK)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    _style(ax)
    fig.suptitle(f"Colliding-wind binary (ζ = 0.1), final state, {N}³: advective vs CR diffusion",
                 color=INK, fontsize=14, y=0.99)
    out = PICS / ("G7_cwb_diffusion_gallery.png" if N == 192 else f"G7_cwb_diffusion_gallery_N{N}.png")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    if "--n" in sys.argv:
        N = int(sys.argv[sys.argv.index("--n") + 1])
    {"extract": extract_all, "plot": plot}[sys.argv[1]]()
