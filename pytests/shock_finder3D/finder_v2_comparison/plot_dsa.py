"""Figure for the DSA question: injection physics and cost, ours vs v2-surface pipelines.

Reads dsa_physics.py (Sedov B7.3 snapshots), dsa_planar.py and the timing
text written by dsa_timing.py; writes pics/dsa_physics_and_cost.png.
"""

import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_comparison import C_HYB, C_OURS, C_OURS_U, C_V2, C_V2P, C_HYB2

DSA_DIR = Path("/export/scratch/nknoell/finder_v2_comparison/dsa")
PIC_DIR = Path(__file__).resolve().parent / "pics"
PIPES = [
    ("ours_ninf", "ours", C_OURS, "-"),
    ("hyb_col", "v2 surface + our walk, column w", C_HYB, "-"),
    ("hyb_unit", "v2 surface + our walk, unit w", C_HYB2, "--"),
    ("hyb_ninf", "v2 surface + our walk, 1/|n|_inf", C_OURS_U, ":"),
    ("v2_col", "v2 default, column w", C_V2, "-"),
    ("v2prof_col", "v2 profile-aware, column w", C_V2P, "-"),
]


def late_mean(d, key):
    return float(np.mean(d[key]))


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    ns = [48, 96, 160]
    runs = {n: np.load(DSA_DIR / f"dsa_dsa_N{n}.npz") for n in ns}
    for key, label, color, ls in PIPES:
        for ax, field in zip(axes[0], ("D_over_meas", "aniso", "axis_diag")):
            ax.plot(ns, [late_mean(runs[n], f"{key}__{field}") for n in ns], color=color, ls=ls,
                    marker="o", ms=5, label=label)
    titles = ["Sedov (zeta = 0.1): injected / true dissipation D / D_meas",
              "angular spread of injection per area (16-84 %)",
              "injection per area: axis bins / diagonal bins"]
    for ax, title in zip(axes[0], titles):
        ax.set_title(title, fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel("N")
    axes[0, 0].axhline(1.0, color="#52514e", lw=1, ls="--")
    axes[0, 2].axhline(1.0, color="#52514e", lw=1, ls="--")
    axes[0, 0].legend(frameon=False, fontsize=7.5)

    # planar oblique shocks, Mach 10 (clean snapshots)
    planar = np.load(DSA_DIR / "planar.npz")
    cases = ["100_M10", "110_M10", "210_M10", "211_M10", "111_M10"]
    ax = axes[1, 0]
    x = np.arange(len(cases))
    width = 0.8 / len(PIPES)
    for i, (key, label, color, _) in enumerate(PIPES):
        ax.bar(x + (i - (len(PIPES) - 1) / 2) * width, [planar[f"{c}__{key}__D"] for c in cases],
               width * 0.9, color=color, label=label)
    ax.axhline(1.0, color="#52514e", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("_")[0] for c in cases])
    ax.set_xlabel("front normal k")
    ax.set_title("planar Mach-10 shocks, N = 64: D / exact (area x flux)", fontsize=9)
    ax.set_ylim(0.6, 2.1)

    # area alone on the planar shocks
    ax = axes[1, 1]
    for i, (key, label, color, _) in enumerate(PIPES[:4]):
        ax.bar(x + (i - 1.5) * 0.2, [planar[f"{c}__{key}__area"] for c in cases], 0.18,
               color=color, label=label)
    ax.axhline(1.0, color="#52514e", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("_")[0] for c in cases])
    ax.set_xlabel("front normal k")
    ax.set_title("planar shocks: weighted area / exact area", fontsize=9)
    ax.set_ylim(0.8, 2.1)

    # cost
    ax = axes[1, 2]
    timing = {}
    for line in (PIC_DIR / "dsa_timing.txt").read_text().splitlines():
        m = re.match(r"N=\s*(\d+)\s+(.*?)\s+([\d.]+) ms", line)
        if m:
            timing.setdefault(m.group(2).strip(), {})[int(m.group(1))] = float(m.group(3))
    for label, color, ls in (("inject ours", C_OURS, "-"), ("inject hybrid+col", C_HYB, "-"),
                             ("stage: our surface", C_OURS, ":"), ("stage: v2 surface", C_HYB, ":"),
                             ("stage: our walk", "#52514e", "--")):
        series = timing[label]
        ax.plot(list(series), list(series.values()), color=color, ls=ls, marker="o", ms=5, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([96, 160, 256])
    ax.set_xticklabels(["96", "160", "256"])
    ax.set_xlabel("N")
    ax.set_ylabel("ms per call (RTX 2080 Ti, float32)")
    ax.set_title("cost of one DSA injection and its stages", fontsize=9)
    ax.legend(frameon=False, fontsize=7.5)
    fig.suptitle("Is v2 + area weighting better / faster for DSA?")
    fig.savefig(PIC_DIR / "dsa_physics_and_cost.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
