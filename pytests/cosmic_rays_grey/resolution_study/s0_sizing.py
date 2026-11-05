"""S0: size the counter-pressure Sedov blast (S-CP) from the 64^3 runs. CPU only.

For each ``RAW/sedov/cp_p<p0>/N64/*.npz``: M_true(t) = Rdot / c_0 from the front
trajectory (``analyse.mach_true_from_trajectory``), the time at which the front
reaches R = 0.42 (the largest radius used, 5 cells inside the box at 64^3), the
Mach range covered, the counter-pressure at the start (p_0 / p_post at R_0, strong
Sedov shock) and the finder's median Mach / M_true. Figure ``pics/S0_sizing.png``,
table ``pics/S0_sizing.txt``.

Usage: python s0_sizing.py
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import GRID, INK, MUTED, RAW, PICS, _style, mach_true_from_trajectory  # noqa: E402

R_MAX = 0.42
GAMMA = 5.0 / 3.0


def load(path):
    d = dict(np.load(path))
    c0 = np.sqrt(GAMMA * float(d["p0"]) / float(d["rho0"]))
    d["c0"] = c0
    d["M_true"] = mach_true_from_trajectory(d["t"], d["R_front"], c0)
    return d


def main():
    files = sorted(RAW.glob("sedov/S0/cp_p*/N64/*.npz"))
    runs = {f"p0={float(np.load(f)['p0']):g} {f.stem}": load(f) for f in files}
    lines = ["# S0 sizing, N = 64 (M_true from the front trajectory)",
             "run                      t(R=0.42)  M_true(R0..)  M_true(0.42)  p0/p_post(R0)  M50/M_true (median)  "
             "E_cr/E_cr0"]
    cmap = plt.get_cmap("Blues")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for k, (name, d) in enumerate(runs.items()):
        color = cmap(0.4 + 0.55 * k / max(len(runs) - 1, 1)) if "kr13" not in name else "#c2410c"
        R, t, M = d["R_front"], d["t"], d["M_true"]
        t42 = float(np.interp(R_MAX, R, t)) if R.max() >= R_MAX else np.nan
        m42 = float(np.interp(R_MAX, R, M)) if R.max() >= R_MAX else np.nan
        # strong Sedov shock at the start: V = 0.4 R / t, p_post = 2 rho V^2 / (gamma + 1)
        t0, R0 = float(d["t_start"]), R[0]
        V0 = 0.4 * R0 / max(t[0], t0)
        ratio = float(d["p0"]) / (2.0 * float(d["rho0"]) * V0**2 / (GAMMA + 1.0))
        inside = R <= R_MAX
        mr = np.nanmedian((d["M50"] / M)[inside & np.isfinite(d["M50"])])
        lines.append(f"{name:24s} {t42:9.4f}  {M[0]:12.1f}  {m42:12.2f}  {ratio:13.2e}  {mr:19.3f}  "
                     f"{d['E_cr'][inside][-1] / max(d['E_cr'][0], 1e-30):10.2f}")
        axes[0].plot(t, R, color=color, label=name)
        axes[1].plot(R, M, color=color, label=name)
        axes[1].plot(R, d["M50"], ":", color=color)
        axes[2].plot(R, d["M50"] / M, color=color, label=name)
    sedov_t = np.geomspace(1e-3, 0.3, 100)
    axes[0].plot(sedov_t, 1.15167 * sedov_t**0.4, "--", color=MUTED, lw=1, label="Sedov (p0 = 0)")
    for ax in axes:
        _style(ax)
    for ax in axes[1:]:
        ax.axvline(R_MAX, color=MUTED, ls="--", lw=1)
        ax.set_xlabel("front radius R")
    axes[0].axhline(R_MAX, color=MUTED, ls="--", lw=1)
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("R")
    axes[0].set_title("front trajectory", color=INK, fontsize=10)
    axes[1].set_yscale("log")
    axes[1].set_title("M_true = Rdot / c0 (solid), finder median (dotted)", color=INK, fontsize=10)
    for m in (2, 5, 15):
        axes[1].axhline(m, color=GRID, lw=1)
    axes[2].set_title("finder median Mach / M_true", color=INK, fontsize=10)
    axes[2].axhline(1.0, color=MUTED, ls="--", lw=1)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("S0: counter-pressure Sedov sizing, N = 64", color=INK)
    PICS.mkdir(exist_ok=True)
    fig.savefig(PICS / "S0_sizing.png", dpi=150)
    text = "\n".join(lines)
    (PICS / "S0_sizing.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
