"""Figures for the opt-in continuous-normal surface: optin_tests.py runs and the CR regression.

* ``pics/optin_cwb128.png``: CWB 128 with live DSA (zeta = 0.1), default vs
  ``dsa_continuous_normal_surface``: orbital-plane e_cr, their ratio, the axis
  profile, CR energy per side of the contact, Mach numbers of the injecting cells.
* ``pics/optin_sedov256.png``: hydro Sedov 256, finder on the final state, default
  vs ``continuous_normal``: Mach numbers and DSA area weights against the exact
  shell, by orientation.
* ``pics/optin_regression.txt``: every RESULT line of the regression logs and each
  registry metric that moved by more than 1% (``regress_summary``).

Usage: python plot_optin.py
"""

import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from plot_comparison import C_HYB, C_OURS, DIV_CMAP, SEQ_CMAP

OPTIN = Path("/export/scratch/nknoell/finder_v2_comparison/optin")
REGRESS = Path("/export/scratch/nknoell/finder_v2_comparison/regress")
PIC_DIR = Path(__file__).resolve().parent / "pics"
# DSA jobs of the regression with our zones + the continuous-normal surface forced on (Part 5),
# against the HEAD registry
REGRESS_JOBS = (41, 43, 45, 46, 47, 48, 49, 50, 51)
MODES = (("default", "segment argmin, 1/|n|_inf (old default)", C_OURS),
         ("hybrid", "continuous normal + column w", C_HYB))
GAMMA = 5.0 / 3.0
# _sedov_setup: E = 1, rho = 1, p = 1e-4, t = 0.1 (box-centred)
SEDOV_R = 1.15167 * 0.1**0.4
SEDOV_MACH = 0.4 * SEDOV_R / 0.1 / np.sqrt(GAMMA * 1e-4)


def _load(case):
    out = {}
    for mode, _, _ in MODES:
        path = OPTIN / f"{case}_{mode}.npz"
        if path.exists():
            out[mode] = (dict(np.load(path)), json.loads((OPTIN / f"{case}_{mode}.json").read_text()))
    return out


def plot_cwb():
    runs = _load("cwb")
    if len(runs) < 2:
        return None
    d0, d1 = runs["default"][0], runs["hybrid"][0]
    n = d0["density"].shape[0]
    mid = n // 2
    vx_axis = d0["vx"][:, mid, mid]
    i_c = int(np.argmax((vx_axis[:-1] > 0) & (vx_axis[1:] <= 0) & (np.arange(n - 1) > n // 4)))
    x = (np.arange(n) + 0.5) / n - 0.5
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    vmax = max(d0["e_cr"][:, :, mid].max(), d1["e_cr"][:, :, mid].max())
    for ax, (mode, label, _) in zip(axes[0, :2], MODES):
        im = ax.imshow(runs[mode][0]["e_cr"][:, :, mid].T, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5],
                       cmap=SEQ_CMAP, norm=LogNorm(vmax * 1e-4, vmax))
        ax.set_title(f"e_cr, orbital plane: {label}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(False)
    fig.colorbar(im, ax=axes[0, :2], shrink=0.85, label="e_cr (code units)")
    ax = axes[0, 2]
    floor = vmax * 1e-4
    ratio = (d1["e_cr"][:, :, mid] + floor) / (d0["e_cr"][:, :, mid] + floor)
    im = ax.imshow(ratio.T, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5], cmap=DIV_CMAP, vmin=0.5, vmax=1.5)
    ax.set_title("e_cr ratio continuous / segment (orbital plane)", fontsize=9)
    ax.set_xlabel("x")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1, 0]
    for mode, label, color in MODES:
        d = runs[mode][0]
        ax.semilogy(x, d["e_cr"][:, mid, mid], color=color, label=f"e_cr, {label}")
        ax.semilogy(x, d["pressure"][:, mid, mid], color=color, ls=":", lw=1, label=f"P_gas, {label}")
    ax.axvline(x[i_c], color="#52514e", lw=1, ls="--")
    ax.set_xlabel("x (binary axis)")
    ax.set_title("profile along the binary axis (dashed: contact)", fontsize=9)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 1]
    x_index = np.arange(n)[:, None, None]
    labels = ["total", "wind-1 side", "star-2 side"]
    vals = {}
    for mode, label, color in MODES:
        e = runs[mode][0]["e_cr"]
        vals[mode] = np.array([e.sum(), (e * (x_index <= i_c)).sum(), (e * (x_index > i_c)).sum()])
    r = vals["hybrid"] / vals["default"]
    bars = ax.bar(labels, r, color=C_HYB, width=0.5)
    for b, v in zip(bars, r):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.axhline(1.0, color="#52514e", lw=1, ls="--")
    lo = min(0.9, r.min() - 0.03)
    ax.set_ylim(lo, max(1.1, r.max() + 0.03))
    walls = {mode: runs[mode][1]["wall_s"]["dsa"] for mode, _, _ in MODES}
    ax.set_title(f"CR energy at t_end, continuous / segment\n(DSA run wall time {walls['default']:.0f} s vs "
                 f"{walls['hybrid']:.0f} s incl. compile)", fontsize=9)

    ax = axes[1, 2]
    bins = np.logspace(np.log10(1.3), np.log10(300), 50)
    for mode, label, color in MODES:
        d = runs[mode][0]
        inj = d["thermal_energy_flux"] > 0
        ax.hist(d["mach_numbers"][inj], bins=bins, weights=d["area_weight"][inj], histtype="step", color=color,
                lw=1.5, label=f"{label} ({inj.sum()} cells)")
    ax.set_xscale("log")
    ax.set_xlabel("Mach number of injecting surface cells (final state)")
    ax.set_ylabel("area-weighted cells")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("CWB 128, live DSA (zeta = 0.1): segment surface vs dsa_continuous_normal_surface (our zones)")
    fig.savefig(PIC_DIR / "optin_cwb128.png", dpi=150)
    plt.close(fig)
    return {"E_cr ratio total / wind-1 / star-2": [float(v) for v in r], "wall": walls,
            "tests": {m: runs[m][1]["test"] for m in runs}, "energies": {m: runs[m][1]["energies"] for m in runs}}


def plot_sedov():
    runs = _load("sedov")
    if len(runs) < 2:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    out = {}
    for mode, label, color in MODES:
        d, meta = runs[mode]
        n = d["density"].shape[0]
        dx = 1.0 / n
        idx = np.argwhere(d["shock_surface_cells"])
        pos = (idx + 0.5) * dx - 0.5
        mach = d["mach_numbers"][d["shock_surface_cells"]]
        w = d["area_weight"][d["shock_surface_cells"]]
        r = np.linalg.norm(pos, axis=1)
        ninf = np.max(np.abs(pos), axis=1) / r
        ok = mach > 0
        axes[0].hist(mach[ok] / SEDOV_MACH, bins=np.linspace(0.7, 1.1, 81), histtype="step", color=color, lw=1.5,
                     label=f"{label}: median {np.median(mach[ok]) / SEDOV_MACH:.3f}")
        # area-weighted shell area per |n|_inf bin over the exact sphere area in that bin
        edges = np.linspace(1 / np.sqrt(3), 1.0, 13)
        u = np.random.default_rng(0).normal(size=(400000, 3))
        u_ninf = np.max(np.abs(u), axis=1) / np.linalg.norm(u, axis=1)
        frac, _ = np.histogram(u_ninf, edges)
        area, _ = np.histogram(ninf, edges, weights=w * dx**2)
        exact = 4 * np.pi * SEDOV_R**2 * frac / frac.sum()
        centres = 0.5 * (edges[1:] + edges[:-1])
        axes[1].plot(centres, area / exact, color=color, marker="o", ms=4, label=label)
        cells, _ = np.histogram(ninf, edges)
        axes[2].plot(centres, cells / exact * dx**2, color=color, marker="o", ms=4, label=label)
        out[mode] = {"test": meta["test"], "wall_s": meta["wall_s"], "surface_cells": int(len(idx)),
                     "median_M_over_sedov": float(np.median(mach[ok]) / SEDOV_MACH),
                     "area_over_4piR2": float((w * dx**2).sum() / (4 * np.pi * SEDOV_R**2))}
    axes[0].set_xlabel("M / M_Sedov at the surface cells")
    axes[0].set_title("Mach numbers (finder default pressure Mach, adaptive + extend)", fontsize=9)
    axes[1].set_xlabel("|n|_inf of the cell's position (1: axis, 0.58: body diagonal)")
    axes[1].set_title("DSA area weight summed per orientation / exact shell area", fontsize=9)
    axes[1].axhline(1.0, color="#52514e", lw=1, ls="--")
    axes[2].set_xlabel("|n|_inf")
    axes[2].set_title("surface cells x dx^2 per orientation / exact shell area (unweighted)", fontsize=9)
    for ax in axes:
        ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Sedov 256 (hydro), shock finder on the final state: segment surface vs continuous_normal")
    fig.savefig(PIC_DIR / "optin_sedov256.png", dpi=150)
    plt.close(fig)
    return out


def regress_summary(jobs=REGRESS_JOBS, name="optin_regression.txt"):
    """RESULT lines per job and registry metrics that moved by more than 1%."""
    lines, moved = [], []
    for log in (REGRESS / "logs" / f"job{j}.log" for j in jobs):
        current = None
        for line in log.read_text(errors="replace").splitlines():
            m = re.match(r"=== (.*) \d\d:\d\d:\d\d$", line)
            if m:
                current = m.group(1).replace("python -u run_tests.py ", "")
            if line.startswith("RESULT"):
                lines.append(f"{log.stem:6s} {current} :: {line[7:]}")
            m = re.match(r"\s+baseline (\S+): (\S+) \(registry (\S+), gate (.*)\)", line)
            if m:
                value, base = float(m.group(2)), float(m.group(3))
                rel = (value - base) / max(abs(base), 1e-30)
                if abs(rel) > 0.01:
                    moved.append(f"{current} :: {m.group(1)}: {base:.4g} -> {value:.4g} ({rel:+.1%} rel, gate {m.group(4)})")
    text = "RESULTS\n" + "\n".join(lines) + "\n\nREGISTRY METRICS MOVED > 1%\n" + "\n".join(moved) + "\n"
    (PIC_DIR / name).write_text(text)
    return text


def main():
    print(json.dumps({"cwb": plot_cwb(), "sedov": plot_sedov()}, indent=1))
    print(regress_summary())


if __name__ == "__main__":
    main()
