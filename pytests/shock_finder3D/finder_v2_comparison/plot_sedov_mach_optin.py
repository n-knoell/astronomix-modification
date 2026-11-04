"""Sedov shell Mach numbers, default vs ``dsa_continuous_normal_surface`` (``sedov_mach_optin.py``).

* ``pics/sedov_mach3d_N{n}.png``: the shell's surface cells in 3D and as a sky map (mu = z / r
  vs azimuth, area-weighted mean per bin), coloured by the DSA Mach number over the exact
  self-similar one, both modes on the same snapshot (t_end) and the same colour scale.
* ``pics/sedov_mach3d_test256.png``: the same 3D view for ``sedov_shock_finder``'s run (hydro,
  N = 256, t = 0.1, the test's finder settings: pressure Mach), from ``optin_tests.py``.
* ``pics/sedov_mach_stats.png``: per-cell scatter and orientation dependence (grid artefacts) of
  the Mach number and of the injected energy per unit shock area, both modes.
* prints the numbers (``sedov_mach_stats.txt``).

Usage: python plot_sedov_mach_optin.py
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from plot_comparison import C_HYB, C_OURS, SEDOV_MACH

DATA = Path("/export/scratch/nknoell/finder_v2_comparison/sedov_mach")
OPTIN = Path("/export/scratch/nknoell/finder_v2_comparison/optin")
PIC_DIR = Path(__file__).resolve().parent / "pics"
MODES = (("default", "segment argmin, 1/|n|_inf (old default)", C_OURS),
         ("hybrid", "dsa_continuous_normal_surface (column w)", C_HYB))
CMAP = "viridis"
NINF_EDGES = np.linspace(1 / np.sqrt(3), 1.0, 9)


def _geometry(idx, n):
    pos = (idx + 0.5) / n - 0.5
    r = np.linalg.norm(pos, axis=1)
    return pos, r, np.max(np.abs(pos), axis=1) / r


def _wquantile(x, w, q):
    order = np.argsort(x)
    c = np.cumsum(w[order])
    return np.interp(np.asarray(q) * c[-1], c, x[order])


def _cell_data(d, key, n, m_true):
    """Injecting cells: positions, M / M_true, area weight, flux."""
    flux = d[f"{key}_flux"]
    inj = flux > 0
    pos, r, ninf = _geometry(d[f"{key}_idx"][inj].astype(float), n)
    return dict(pos=pos, r=r, ninf=ninf, m=d[f"{key}_mach"][inj] / m_true, w=d[f"{key}_weight"][inj],
                flux=flux[inj])


def _sky(pos, value, weight, n_mu=36, n_phi=72):
    mu = pos[:, 2] / np.linalg.norm(pos, axis=1)
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    b = (np.clip(((mu + 1) / 2 * n_mu).astype(int), 0, n_mu - 1) * n_phi
         + np.clip(((phi + np.pi) / (2 * np.pi) * n_phi).astype(int), 0, n_phi - 1))
    num = np.bincount(b, weights=value * weight, minlength=n_mu * n_phi)
    den = np.bincount(b, weights=weight, minlength=n_mu * n_phi)
    with np.errstate(invalid="ignore"):
        return (num / den).reshape(n_mu, n_phi)


def _scatter3d(ax, pos, value, norm, title, n):
    # front hemisphere only (towards the viewer), so the colour pattern is not overplotted
    view = np.array([np.cos(np.radians(35)) * np.cos(np.radians(40)),
                     np.cos(np.radians(35)) * np.sin(np.radians(40)), np.sin(np.radians(35))])
    front = pos @ view > 0
    ax.scatter(pos[front, 0], pos[front, 1], pos[front, 2], c=value[front], cmap=CMAP, norm=norm,
               s=0.6 * (256 / n) ** 2, linewidths=0, depthshade=False)
    ax.view_init(elev=35, azim=40)
    lim = np.abs(pos).max() * 1.02
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title, fontsize=9)


def plot_3d(n):
    d = dict(np.load(DATA / f"sedov_mach_N{n}.npz"))
    m_true, t = float(d["t9_M_true"]), float(d["t9_t"])
    cells = {mode: _cell_data(d, f"t9_{mode}", n, m_true) for mode, _, _ in MODES}
    allm = np.concatenate([c["m"] for c in cells.values()])
    norm = Normalize(*np.percentile(allm, [2, 98]))
    fig = plt.figure(figsize=(13, 9.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
    for row, (mode, label, _) in enumerate(MODES):
        c = cells[mode]
        _scatter3d(fig.add_subplot(grid[row, 0], projection="3d"), c["pos"], c["m"], norm,
                   f"{label}\n{len(c['m'])} injecting cells (front hemisphere shown)", n)
        ax = fig.add_subplot(grid[row, 1])
        sky = _sky(c["pos"], c["m"], c["w"])
        im = ax.imshow(sky, origin="lower", extent=[-180, 180, -1, 1], aspect="auto", cmap=CMAP, norm=norm)
        ax.set_xlabel("azimuth phi (deg); axes at 0, +-90, 180")
        ax.set_ylabel("mu = z / r")
        ax.grid(False)
        ax.set_title(f"sky map, area-weighted mean per bin: {label}", fontsize=9)
    fig.colorbar(im, ax=fig.axes, shrink=0.6, label="DSA Mach number / exact self-similar Mach")
    fig.suptitle(f"Sedov + CR control run (zeta = 0), N = {n}^3, t = {t:.3f}, M_true = {m_true:.1f}: "
                 "Mach numbers the injection uses (velocity Mach, CR-aware)")
    fig.savefig(PIC_DIR / f"sedov_mach3d_N{n}.png", dpi=140)
    plt.close(fig)


def plot_test256():
    fig = plt.figure(figsize=(13, 6.2), constrained_layout=True)
    data = {}
    for mode, _, _ in MODES:
        d = np.load(OPTIN / f"sedov_{mode}.npz")
        s = d["shock_surface_cells"]
        n = s.shape[0]
        pos, _, _ = _geometry(np.argwhere(s).astype(float), n)
        data[mode] = (pos, d["mach_numbers"][s] / SEDOV_MACH)
    norm = Normalize(*np.percentile(np.concatenate([m for _, m in data.values()]), [2, 98]))
    for i, (mode, label, _) in enumerate(MODES):
        pos, m = data[mode]
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _scatter3d(ax, pos, m, norm, f"{label}\n{len(m)} surface cells, median M/M_Sedov {np.median(m):.3f}", n)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    fig.colorbar(sm, ax=fig.axes, shrink=0.7, label="finder Mach number / exact Sedov Mach")
    fig.suptitle("sedov_shock_finder test run (hydro, N = 256^3, t = 0.1, M_Sedov = "
                 f"{SEDOV_MACH:.1f}): finder Mach (pressure, adaptive + extend)")
    fig.savefig(PIC_DIR / "sedov_mach3d_test256.png", dpi=140)
    plt.close(fig)


def stats(c, r_true):
    """Area-weighted scatter of M, orientation dependence of M and of the energy per area."""
    m, w = c["m"], c["w"]
    q16, q50, q84 = _wquantile(m, w, [0.16, 0.5, 0.84])
    k = np.clip(np.digitize(c["ninf"], NINF_EDGES) - 1, 0, len(NINF_EDGES) - 2)
    med_bin = np.array([_wquantile(m[k == i], w[k == i], 0.5) if np.any(k == i) else np.nan
                        for i in range(len(NINF_EDGES) - 1)])
    # exact sphere area per |n|_inf bin (Monte Carlo of uniform directions)
    u = np.random.default_rng(0).normal(size=(400000, 3))
    frac, _ = np.histogram(np.max(np.abs(u), axis=1) / np.linalg.norm(u, axis=1), NINF_EDGES)
    exact = 4 * np.pi * r_true**2 * frac / frac.sum()
    energy = np.bincount(k, weights=c["flux"] * w, minlength=len(exact)) / exact
    energy = energy / (np.sum(c["flux"] * w) / (4 * np.pi * r_true**2))
    return dict(median=q50, spread=0.5 * (q84 - q16) / q50, med_bin=med_bin / q50,
                mach_axis_diag=med_bin[-1] / med_bin[0], energy_bin=energy,
                energy_axis_diag=energy[-1] / energy[0])


def plot_stats():
    table = {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    centres = 0.5 * (NINF_EDGES[1:] + NINF_EDGES[:-1])
    for n in (96, 160):
        path = DATA / f"sedov_mach_N{n}.npz"
        if not path.exists():
            continue
        d = dict(np.load(path))
        for i_t in (0, 4, 9):
            m_true, r_true, t = float(d[f"t{i_t}_M_true"]), float(d[f"t{i_t}_R_true"]), float(d[f"t{i_t}_t"])
            for mode, label, color in MODES:
                c = _cell_data(d, f"t{i_t}_{mode}", n, m_true)
                s = stats(c, r_true)
                table[f"N{n} t={t:.3f} {mode}"] = {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray)
                                                      else round(float(v), 4)) for k, v in s.items()}
                if n == 160 and i_t == 9:
                    bins = np.linspace(0.6, 1.1, 101)
                    axes[0, 0].hist(c["m"], bins=bins, weights=c["w"], histtype="step", color=color, lw=1.5,
                                    label=f"{label}: median {s['median']:.3f}, scatter {s['spread']:.3f}")
                    axes[0, 1].plot(centres, s["med_bin"], color=color, marker="o", ms=4, label=label)
                    axes[1, 0].plot(centres, s["energy_bin"], color=color, marker="o", ms=4, label=label)
    rows = [(n, t) for n in (96, 160) for t in ("0.007", "0.035", "0.070")]
    ax = axes[1, 1]
    x = np.arange(len(rows))
    for j, (mode, label, color) in enumerate(MODES):
        vals = [table.get(f"N{n} t={t} {mode}", {}).get("spread", np.nan) for n, t in rows]
        ax.bar(x + (j - 0.5) * 0.38, vals, width=0.36, color=color, label=label)
    ax.set_xticks(x, [f"N={n}\nt={t}" for n, t in rows], fontsize=8)
    ax.set_ylabel("half 16-84% width / median (area-weighted)")
    ax.set_title("per-cell scatter of the DSA Mach number", fontsize=9)
    axes[0, 0].set_xlabel("DSA Mach / exact (injecting cells, area-weighted)")
    axes[0, 0].set_title("N = 160, t = 0.07: Mach distribution", fontsize=9)
    axes[0, 1].set_xlabel("|n|_inf of the cell's position (1: axis, 0.58: body diagonal)")
    axes[0, 1].set_ylabel("median Mach in bin / overall median")
    axes[0, 1].set_title("N = 160, t = 0.07: Mach vs orientation (grid artefact)", fontsize=9)
    axes[1, 0].set_xlabel("|n|_inf")
    axes[1, 0].set_ylabel("injected energy per exact area / shell mean")
    axes[1, 0].set_title("N = 160, t = 0.07: injected energy per unit shock area vs orientation", fontsize=9)
    for a in (axes[0, 1], axes[1, 0]):
        a.axhline(1.0, color="#52514e", lw=1, ls="--")
    for a in axes.ravel():
        a.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Sedov + CR control run: DSA Mach scatter and grid artefacts, segment surface vs continuous normal (our zones)")
    fig.savefig(PIC_DIR / "sedov_mach_stats.png", dpi=140)
    plt.close(fig)
    text = json.dumps(table, indent=1)
    (PIC_DIR / "sedov_mach_stats.txt").write_text(text)
    return table


def main():
    plot_test256()
    for n in (96, 160):
        if (DATA / f"sedov_mach_N{n}.npz").exists():
            plot_3d(n)
    for key, s in plot_stats().items():
        print(f"{key:28s} median {s['median']:.3f} scatter {s['spread']:.3f} "
              f"M axis/diag {s['mach_axis_diag']:.3f} E axis/diag {s['energy_axis_diag']:.3f}")


if __name__ == "__main__":
    main()
