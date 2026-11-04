"""Figures + summary for ``compare_finders.py`` (reads its results_<state>.npz).

Scatter of the surface-cell Mach numbers is split two ways:

* Sedov (spherical, so any angular structure is an artefact): ln M is binned
  on the sky (cos theta x phi); ``sigma_sys`` is the spread of the bin
  medians (grid artefacts, orientation-dependent bias), ``sigma_rand`` the
  spread inside the bins (cell-to-cell noise at fixed orientation).
* any state: ``sigma_local`` is the robust spread of ln M minus the mean ln M
  of the valid surface cells in the surrounding 5^3 box, i.e. noise between
  neighbouring cells, which needs no reference solution (used for the CWB).

All spreads are robust (half the 16-84 percentile range) in ln M, i.e.
roughly fractional.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from scipy.ndimage import uniform_filter

DATA_DIR = Path("/export/scratch/nknoell/finder_v2_comparison")
PIC_DIR = Path(__file__).resolve().parent / "pics"
GAMMA = 5.0 / 3.0
SEDOV_R = 1.15167 * (0.1**2) ** 0.2
SEDOV_MACH = 0.4 * SEDOV_R / 0.1 / np.sqrt(GAMMA * 1e-4)
P_AMBIENT = 1e-4
# surface cells closer than this to a box face are dropped: the Sedov shell
# (R = 0.46 at t = 0.1) is only 2-3 cells from the faces on the axes at N = 64,
# so the pre-shock walk there runs into the open boundary in every variant
BOUNDARY_MARGIN = 6

# fixed categorical order (dataviz reference palette)
C_OURS, C_V2, C_HYB, C_OURS_U, C_V2P, C_HYB2 = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6",
                                                 "#eda100", "#e87ba4")
HEADLINE = {
    "ours (walk+extend, p)": ("zours_sours_ours_p_off0", C_OURS),
    "ours DSA (velocity)": ("zours_sours_ours_u_off0", C_OURS_U),
    "v2 default": ("zv2_sv2_v2_offv2", C_V2),
    "v2 profile-aware": ("zv2_sv2_v2_profile_offv2", C_V2P),
    "v2 geometry + our walk (p)": ("zv2_sv2_ours_p_off0", C_HYB),
    "v2 geometry + our walk (u)": ("zv2_sv2_ours_u_off0", C_HYB2),
}
# swap one stage at a time, ours -> v2 (pressure Mach throughout)
CHAIN = [
    ("ours", "zours_sours_ours_p_off0"),
    ("+v2 zones", "zv2_sours_ours_p_off0"),
    ("+v2 surface", "zv2_sv2_ours_p_off0"),
    ("+v2 sampler", "zv2_sv2_v2_off0"),
    ("+v2 offsets\n(= v2)", "zv2_sv2_v2_offv2"),
]
CROSS = [
    ("our geometry\n+ v2 sampler", "zours_sours_v2_off0"),
    ("v2 geometry\n+ our walk", "zv2_sv2_ours_p_off0"),
    ("v2 geometry\n+ our walk (u)", "zv2_sv2_ours_u_off0"),
    ("v2 profile-\naware", "zv2_sv2_v2_profile_offv2"),
]
SEQ_CMAP = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#c45a1e", "#eb6834", "#f0efec", "#2a78d6", "#184f95"])

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2,
})


def robust_sigma(x):
    if len(x) < 3:
        return np.nan
    p16, p84 = np.percentile(x, [16, 84])
    return 0.5 * (p84 - p16)


def load(name):
    return dict(np.load(DATA_DIR / f"results_{name}.npz"))


def variant(res, key):
    """Valid surface cells of one variant: positions (box-centred), Mach and samples."""
    n = int(res["num_cells"])
    mach = res[f"{key}__mach"]
    idx = res[f"{key}__idx"].astype(int)
    interior = np.all((idx >= BOUNDARY_MARGIN) & (idx < n - BOUNDARY_MARGIN), axis=1)
    valid = (mach > 0) & interior
    idx = idx[valid]
    out = dict(idx=idx, pos=(idx + 0.5) / n - 0.5, mach=mach[valid], n=n,
               num_surface=int(interior.sum()), num_valid=int(valid.sum()))
    for field in ("p_post", "p_pre", "d_post", "d_pre"):
        out[field] = res[f"{key}__{field}"][valid]
    return out


def local_residuals(v):
    """ln M minus the mean ln M of the other valid cells in the surrounding 5^3 box."""
    n = v["n"]
    grid = np.zeros((n, n, n))
    mask = np.zeros((n, n, n))
    i, j, k = v["idx"].T
    grid[i, j, k] = np.log(v["mach"])
    mask[i, j, k] = 1.0
    s = uniform_filter(grid, 5, mode="constant") * 125 - grid
    c = uniform_filter(mask, 5, mode="constant") * 125 - mask
    has = c[i, j, k] > 0.5
    return grid[i, j, k][has] - s[i, j, k][has] / c[i, j, k][has]


def local_sigma(v):
    return robust_sigma(local_residuals(v))


def sky_bins(pos, n_mu=16, n_phi=32):
    r = np.linalg.norm(pos, axis=1)
    mu = pos[:, 2] / r
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    b_mu = np.clip(((mu + 1) / 2 * n_mu).astype(int), 0, n_mu - 1)
    b_phi = np.clip(((phi + np.pi) / (2 * np.pi) * n_phi).astype(int), 0, n_phi - 1)
    return b_mu * n_phi + b_phi, n_mu, n_phi


def sedov_metrics(v):
    lnm = np.log(v["mach"])
    bins, n_mu, n_phi = sky_bins(v["pos"])
    med = np.full(n_mu * n_phi, np.nan)
    for b in np.unique(bins):
        med[b] = np.median(lnm[bins == b])
    resid = lnm - med[bins]
    counts = np.bincount(bins, minlength=n_mu * n_phi)
    return dict(
        median=np.exp(np.median(lnm)) / SEDOV_MACH,
        sigma_total=robust_sigma(lnm),
        sigma_sys=robust_sigma(med[bins]),
        sigma_rand=robust_sigma(resid),
        sigma_local=local_sigma(v),
        # equal-area bins: cells per bin should be uniform for a 1-cell shell
        count_cv=np.std(counts) / np.mean(counts),
        empty_bins=int(np.sum(counts == 0)),
        valid=v["num_valid"], surface=v["num_surface"],
        r_spread=robust_sigma(np.linalg.norm(v["pos"], axis=1)) * v["n"],
        sky=med.reshape(n_mu, n_phi),
    )


def grid_alignment(pos):
    """|n|_inf of the radial direction: 1 on the axes, 1/sqrt(3) on the cube diagonals."""
    return np.max(np.abs(pos), axis=1) / np.linalg.norm(pos, axis=1)


def surface_geometry(res, key):
    """Area and column statistics of all surface cells (valid or not) of a
    Sedov variant, with the exact radial normal: the summed DSA area weight
    sum(dx^2 / |n|_inf) over 4 pi R^2 (1 for one cell per dominant-axis
    column, as DB7 assumes), and the mean number of surface cells per
    dominant-axis column."""
    n = int(res["num_cells"])
    idx = res[f"{key}__idx"].astype(int)
    pos = (idx + 0.5) / n - 0.5
    r = np.linalg.norm(pos, axis=1)
    radius = np.median(r)
    area = np.sum((1.0 / n) ** 2 / grid_alignment(pos)) / (4 * np.pi * radius**2)
    axis = np.argmax(np.abs(pos), axis=1)
    sign = np.sign(pos[np.arange(len(pos)), axis]).astype(int)
    other = np.where(np.arange(3)[None, :] == axis[:, None], -1, idx)
    _, counts = np.unique(np.column_stack([axis, sign, other]), axis=0, return_counts=True)
    return dict(area_ratio=area, cells_per_column=len(idx) / len(counts),
                multi_columns=np.mean(counts > 1), radius=radius)


def plot_coverage(names):
    sedov = [n for n in names if n.startswith("sedov")]
    surfaces = [("ours surface", "zours_sours_ours_p_off0"), ("v2 surface", "zv2_sv2_ours_p_off0")]
    fig, axes = plt.subplots(len(surfaces), len(sedov), figsize=(3.6 * len(sedov), 5.6),
                             constrained_layout=True, squeeze=False)
    for col, name in enumerate(sedov):
        res = load(name)
        n = int(res["num_cells"])
        for row, (label, key) in enumerate(surfaces):
            idx = res[f"{key}__idx"].astype(int)
            pos = (idx + 0.5) / n - 0.5
            bins, n_mu, n_phi = sky_bins(pos)
            # cells per equal-area sky bin, normalised; excludes nothing, so the
            # axis regions at N = 64 include the boundary-affected cells
            counts = np.bincount(bins, minlength=n_mu * n_phi).reshape(n_mu, n_phi)
            g = surface_geometry(res, key)
            ax = axes[row, col]
            im = ax.imshow(counts / counts.mean(), origin="lower", extent=[-180, 180, -1, 1],
                           aspect="auto", cmap=DIV_CMAP, vmin=0.6, vmax=1.4)
            ax.grid(False)
            ax.set_title(f"{label}, {name}\narea weight {g['area_ratio']:.2f}, "
                         f"{g['cells_per_column']:.2f} cells/column", fontsize=8.5)
            ax.set_xlabel("phi [deg]")
            if col == 0:
                ax.set_ylabel("cos theta")
    fig.colorbar(im, ax=axes, shrink=0.8, label="surface cells per sky bin / mean")
    fig.suptitle("Sedov: surface-cell coverage (the visible grid pattern) and DSA area weight "
                 "sum(dx^2/|n|_inf) / 4 pi R^2")
    fig.savefig(PIC_DIR / "sedov_surface_coverage.png", dpi=150)
    plt.close(fig)


def plot_sedov_sky(name, res):
    keys = list(HEADLINE.items())
    fig, axes = plt.subplots(2, len(keys), figsize=(3.1 * len(keys), 5.6), constrained_layout=True)
    for col, (label, (key, color)) in enumerate(keys):
        v = variant(res, key)
        m = sedov_metrics(v)
        ax = axes[0, col]
        sky = np.exp(m["sky"] - np.nanmedian(m["sky"]))
        im = ax.imshow(sky, origin="lower", extent=[-180, 180, -1, 1], aspect="auto",
                       cmap=DIV_CMAP, vmin=0.85, vmax=1.15)
        ax.set_title(f"{label}\nsys {m['sigma_sys']:.3f}, rand {m['sigma_rand']:.3f}", fontsize=8.5)
        ax.set_xlabel("phi [deg]")
        ax.grid(False)
        if col == 0:
            ax.set_ylabel("cos theta")
        ax = axes[1, col]
        a = grid_alignment(v["pos"])
        ratio = v["mach"] / SEDOV_MACH
        ax.hist2d(a, ratio, bins=[40, 60], range=[[0.57, 1.0], [0.3, 1.2]], cmap=SEQ_CMAP,
                  norm=LogNorm(), cmin=1)
        edges = np.linspace(0.577, 1.0, 13)
        cen = 0.5 * (edges[1:] + edges[:-1])
        meds = [np.median(ratio[(a >= lo) & (a < hi)]) if np.any((a >= lo) & (a < hi)) else np.nan
                for lo, hi in zip(edges[:-1], edges[1:])]
        ax.plot(cen, meds, color=color, marker="o", ms=4, label="median")
        ax.axhline(1.0, color="#52514e", lw=1, ls="--")
        ax.set_xlabel("|n|_inf  (1/sqrt3 = cube diagonal, 1 = axis)")
        if col == 0:
            ax.set_ylabel("M / M_Sedov")
        ax.set_title(f"median {m['median']:.3f}, valid {m['valid']}", fontsize=8.5)
    fig.colorbar(im, ax=axes[0, :], shrink=0.8, label="sky-bin median / global median")
    fig.suptitle(f"Sedov {name}: orientation dependence of the surface Mach number "
                 f"(M_Sedov = {SEDOV_MACH:.1f})")
    fig.savefig(PIC_DIR / f"{name}_sky_and_orientation.png", dpi=150)
    plt.close(fig)


def plot_slice(name, res, reference=None):
    keys = list(HEADLINE.items())
    n = int(res["num_cells"])
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.6), constrained_layout=True)
    all_m = np.concatenate([variant(res, k)["mach"] for _, (k, _) in keys])
    lo, hi = np.percentile(all_m, [1, 99.5])
    for ax, (label, (key, color)) in zip(axes, keys):
        v = variant(res, key)
        sel = v["idx"][:, 2] == n // 2
        if name.startswith("sedov"):
            sel &= (v["pos"][:, 0] > 0) & (v["pos"][:, 1] > 0)
        sc = ax.scatter(v["pos"][sel, 0], v["pos"][sel, 1], c=v["mach"][sel], s=4 if n > 128 else 9,
                        marker="s", cmap=SEQ_CMAP, norm=LogNorm(lo, hi), linewidths=0)
        ax.set_aspect("equal")
        if not name.startswith("sedov"):
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
        ax.set_title(f"{label}\n{sel.sum()} cells in slice", fontsize=8.5)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(sc, ax=axes, shrink=0.85, label="Mach")
    fig.suptitle(f"{name}: valid shock-surface cells in the z = mid-plane slice")
    fig.savefig(PIC_DIR / f"{name}_slice.png", dpi=150)
    plt.close(fig)


def plot_attribution(names):
    sedov = [n for n in names if n.startswith("sedov")]
    rows = CHAIN + CROSS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    width = 0.8 / len(sedov)
    shades = ["#9ec5f4", "#3987e5", "#104281"]
    x = np.arange(len(rows))
    for s_i, name in enumerate(sedov):
        res = load(name)
        ms = [sedov_metrics(variant(res, key)) for _, key in rows]
        for ax, field, title in zip(axes, ("sigma_sys", "sigma_rand", "median"),
                                    ("systematic (sky-bin medians) spread of ln M",
                                     "random (within sky bin) spread of ln M",
                                     "median M / M_Sedov")):
            ax.bar(x + (s_i - (len(sedov) - 1) / 2) * width, [m[field] for m in ms], width * 0.9,
                   color=shades[s_i % 3], label=name)
            ax.set_title(title, fontsize=9)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([r for r, _ in rows], rotation=60, ha="right", fontsize=7.5)
        ax.axvline(len(CHAIN) - 0.5, color="#52514e", lw=1)
    axes[2].axhline(1.0, color="#52514e", lw=1, ls="--")
    axes[2].set_ylim(0.5, 1.05)
    axes[0].legend(frameon=False)
    fig.suptitle("Sedov: which stage causes the scatter? (left of the line: swap one stage at a time "
                 "ours -> v2; right: cross combinations)")
    fig.savefig(PIC_DIR / "sedov_stage_attribution.png", dpi=150)
    plt.close(fig)


def plot_sampling(name, res):
    keys = [("ours (walk+extend)", "zours_sours_ours_p_off0", C_OURS),
            ("v2 default (exit+0.25)", "zv2_sv2_v2_offv2", C_V2),
            ("v2 profile-aware", "zv2_sv2_v2_profile_offv2", C_V2P),
            ("v2 geometry + our walk", "zv2_sv2_ours_p_off0", C_HYB)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    for label, key, color in keys:
        v = variant(res, key)
        p_pre = v["p_pre"] / P_AMBIENT
        p_post = v["p_post"] / np.median(v["p_post"])
        kw = dict(histtype="step", lw=1.6, color=color, density=True)
        axes[0, 0].hist(np.log10(p_pre), bins=np.linspace(-0.05, 2.0, 80),
                        label=f"{label}: sigma(ln p_pre) {robust_sigma(np.log(v['p_pre'])):.3f}", **kw)
        axes[0, 1].hist(p_post, bins=np.linspace(0.4, 1.4, 80),
                        label=f"sigma(ln p_post) {robust_sigma(np.log(v['p_post'])):.3f}", **kw)
        axes[0, 2].hist(v["mach"] / SEDOV_MACH, bins=np.linspace(0.2, 1.2, 80),
                        label=f"sigma(ln M) {robust_sigma(np.log(v['mach'])):.3f}", **kw)
        axes[1, 0].hist(v["d_pre"], bins=np.arange(0, 12.01, 0.25), label=label, **kw)
        axes[1, 1].hist(v["d_post"], bins=np.arange(0, 12.01, 0.25), label=label, **kw)
        r = np.linalg.norm(v["pos"], axis=1) * v["n"]
        axes[1, 2].hist(r - np.median(r), bins=np.arange(-4, 4.01, 0.25), label=label, **kw)
    titles = ["log10 p_pre / p_ambient (ideal 0)", "p_post / median p_post", "M / M_Sedov",
              "pre-shock sample distance [cells]", "post-shock sample distance [cells]",
              "surface-cell radius - median [cells]"]
    for ax, t in zip(axes.flat, titles):
        ax.set_title(t, fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Sedov {name}: where the Mach samples land  (M^2 ~ p_post / p_pre for a strong shock)")
    fig.savefig(PIC_DIR / f"{name}_sampling.png", dpi=150)
    plt.close(fig)


def plot_resolution(names):
    sedov = [n for n in names if n.startswith("sedov")]
    if len(sedov) < 2:
        return
    ns = []
    data = {label: [] for label in HEADLINE}
    for name in sedov:
        res = load(name)
        ns.append(int(res["num_cells"]))
        for label, (key, _) in HEADLINE.items():
            data[label].append(np.percentile(variant(res, key)["mach"] / SEDOV_MACH, [16, 50, 84]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for i, (label, (_, color)) in enumerate(HEADLINE.items()):
        d = np.array(data[label])
        xs = np.array(ns) * (1 + 0.03 * (i - 2.5))
        ax.errorbar(xs, d[:, 1], yerr=[d[:, 1] - d[:, 0], d[:, 2] - d[:, 1]], color=color,
                    marker="o", ms=5, capsize=3, lw=1.6, label=label)
    ax.axhline(1.0, color="#52514e", lw=1, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("N (cells per dimension)")
    ax.set_ylabel("M / M_Sedov  (median, 16-84 %)")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Sedov: surface Mach number vs resolution")
    fig.savefig(PIC_DIR / "sedov_resolution.png", dpi=150)
    plt.close(fig)


def plot_cwb_hist(name, res):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    bins = np.logspace(np.log10(1.3), np.log10(300), 70)
    for label, (key, color) in HEADLINE.items():
        v = variant(res, key)
        sig = local_sigma(v)
        axes[0].hist(v["mach"], bins=bins, histtype="step", lw=1.6, color=color,
                     label=f"{label}: {v['num_valid']} cells, sigma_local {sig:.3f}")
        axes[1].hist(local_residuals(v), bins=np.linspace(-1.5, 1.5, 90), histtype="step", lw=1.6, color=color,
                     density=True, label=label)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Mach")
    axes[0].set_title("surface-cell Mach numbers")
    axes[1].set_xlabel("ln M - mean ln M of neighbours (5^3 box)")
    axes[1].set_title("cell-to-cell scatter")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"{name}: Mach distribution and local scatter")
    fig.savefig(PIC_DIR / f"{name}_mach_hist.png", dpi=150)
    plt.close(fig)


def write_summary(names):
    lines = []
    for name in names:
        res = load(name)
        lines.append(f"## {name} (N={int(res['num_cells'])})")
        keys = sorted({k.split("__")[0] for k in res if k.endswith("__mach") and not k.startswith(("ours_full", "v2_full"))})
        if name.startswith("sedov"):
            for label, key in (("ours surface", "zours_sours_ours_p_off0"),
                               ("v2 zones + ours surface", "zv2_sours_ours_p_off0"),
                               ("ours zones + v2 surface", "zours_sv2_ours_p_off0"),
                               ("v2 surface", "zv2_sv2_ours_p_off0")):
                g = surface_geometry(res, key)
                lines.append(f"{label:26s}: area weight {g['area_ratio']:.3f}, cells/column "
                             f"{g['cells_per_column']:.3f}, columns with >1 cell {g['multi_columns']:.3f}")
            lines.append(f"{'variant':38s} {'surf':>7s} {'valid':>7s} {'M/Msed':>7s} {'s_tot':>6s} "
                         f"{'s_sys':>6s} {'s_rand':>6s} {'s_loc':>6s} {'cnt_cv':>6s} {'empty':>5s} {'r_spr':>5s}")
            for key in keys:
                m = sedov_metrics(variant(res, key))
                lines.append(f"{key:38s} {m['surface']:7d} {m['valid']:7d} {m['median']:7.3f} "
                             f"{m['sigma_total']:6.3f} {m['sigma_sys']:6.3f} {m['sigma_rand']:6.3f} "
                             f"{m['sigma_local']:6.3f} {m['count_cv']:6.3f} {m['empty_bins']:5d} {m['r_spread']:5.2f}")
        else:
            lines.append(f"{'variant':38s} {'surf':>7s} {'valid':>7s} {'M_med':>7s} {'M_p90':>7s} {'s_tot':>6s} {'s_loc':>6s}")
            for key in keys:
                v = variant(res, key)
                lines.append(f"{key:38s} {v['num_surface']:7d} {v['num_valid']:7d} {np.median(v['mach']):7.2f} "
                             f"{np.percentile(v['mach'], 90):7.2f} {robust_sigma(np.log(v['mach'])):6.3f} "
                             f"{local_sigma(v):6.3f}")
        for label, t in zip(res["timing_labels"], res["timing_values"]):
            full = res[f"{label}__mach_grid_nonzero"]
            if not np.isfinite(t):
                lines.append(f"full finder {label:16s}: failed (out of memory)")
                continue
            lines.append(f"full finder {label:16s}: {t:7.2f} s, valid {len(full)}, median M {np.median(full):.2f}")
        lines.append("")
    text = "\n".join(lines)
    (PIC_DIR / "summary.txt").write_text(text)
    print(text)


def make_all_plots(names):
    PIC_DIR.mkdir(exist_ok=True)
    names = [n for n in names if (DATA_DIR / f"results_{n}.npz").exists()]
    for name in names:
        res = load(name)
        plot_slice(name, res)
        if name.startswith("sedov"):
            plot_sedov_sky(name, res)
            plot_sampling(name, res)
        else:
            plot_cwb_hist(name, res)
    plot_attribution(names)
    plot_coverage(names)
    plot_resolution(names)
    write_summary(names)
