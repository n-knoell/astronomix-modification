"""Pre-shock Mach number vs. shock distance in injection radii for the stationary CWB
runs with a fixed 2-cell wind injection (``_regen_fixed_injection.py``), against the
Chevalier & Clegg (1985) thermally driven wind.

The EI source deposits mass and thermal energy uniformly in a sphere of radius
``R_inj``; outside it the wind follows the CC85 solution (gamma = 5/3)

    M^(2/(g-1)) * [((g-1) + 2/M^2) / (g+1)]^((g+1)/(2(g-1))) = (r/R)^2,

i.e. M ~ 2.5 (r/R)^(2/3) far out. For each run, along the binary axis (mean of the
four central cell lines), this reads off for both stars' termination shocks:

* ``r_s/R_inj``: distance from the star to the first shock-surface cell on the axis;
* the finder's Mach number at that cell;
* the local pre-shock Mach number ``|v_x|/c_s``, maximum over the free wind between
  ``R_inj + dx`` and one cell before the shock.

No JAX needed; reads ``/export/scratch/nknoell/cwb_separation/data/ninj2_N<N>.npz``.
Writes ``figures/cwb_fixed_injection/cwb_mach_vs_distance_ninj2.png``.
"""

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

RAW_DIR = Path("/export/scratch/nknoell/cwb_separation/data")
FIG_DIR = Path(__file__).resolve().parent / "figures" / "cwb_fixed_injection"
GAMMA = 5.0 / 3.0
NUM_INJECTION_CELLS = 2
STAR_X = (-0.2, 0.2)  # box-centred star positions (_cwb_setup.SEPARATION = 0.4)
RESOLUTIONS = (64, 96, 128, 196, 256)
# reference blue ramp, ordinal steps 250-650 (light -> dark = low -> high N)
# Star 2's shock sits only ~4-6 injection radii from its source at N <= 128 (at N=64 the
# finder picks the wrong cell there); its points are plotted from this N on.
STAR2_MIN_N = 196
COLORS = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")


def cc85_mach(x, gamma=GAMMA):
    """Chevalier & Clegg (1985) Mach number at r/R = x > 1 (supersonic branch)."""
    g = gamma

    def residual(mach):
        return (2 / (g - 1) * np.log(mach)
                + (g + 1) / (2 * (g - 1)) * np.log(((g - 1) + 2 / mach**2) / (g + 1))
                - 2 * np.log(x))

    return brentq(residual, 1.0 + 1e-9, 1e6)


def apex_values(num_cells):
    data = np.load(RAW_DIR / f"ninj2_N{num_cells}.npz")
    state, surface, mach = data["state"], data["surface"], data["mach"]
    dx = 1.0 / num_cells
    c = num_cells // 2
    axis = (slice(None), slice(c - 1, c + 1), slice(c - 1, c + 1))
    rho = state[int(data["density_index"])][axis].mean(axis=(1, 2))
    p = state[int(data["pressure_index"])][axis].mean(axis=(1, 2))
    vx = state[int(data["vx_index"])][axis].mean(axis=(1, 2))
    x = (np.arange(num_cells) + 0.5) * dx - 0.5
    local_mach = np.abs(vx) / np.sqrt(GAMMA * p / rho)
    on_axis = surface[axis].any(axis=(1, 2))
    finder_mach = np.where(surface[axis], mach[axis], 0.0).max(axis=(1, 2))
    r_inj = NUM_INJECTION_CELLS * dx
    between = np.where(on_axis & (x > STAR_X[0]) & (x < STAR_X[1]))[0]
    out = {}
    for star, x_star, i in (("1", STAR_X[0], between.min()), ("2", STAR_X[1], between.max())):
        r_shock = abs(x[i] - x_star)
        side = np.sign(x[i] - x_star)
        upstream = ((np.abs(x - x_star) > r_inj + dx) & (np.abs(x - x_star) < r_shock - dx)
                    & (np.sign(x - x_star) == side))
        out[star] = dict(
            r_over_R=r_shock / r_inj,
            finder=float(finder_mach[i]),
            local=float(local_mach[upstream].max()) if upstream.any() else np.nan,
            cc85=cc85_mach(r_shock / r_inj),
        )
    return out


def main():
    runs = {n: apex_values(n) for n in RESOLUTIONS if (RAW_DIR / f"ninj2_N{n}.npz").exists()}
    print("N    star  r_s/R   finder  local   CC85")
    for n, res in runs.items():
        for star in ("1", "2"):
            v = res[star]
            print(f"{n:<4} {star:<5} {v['r_over_R']:<7.1f} {v['finder']:<7.1f} {v['local']:<7.1f} {v['cc85']:.1f}")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    xs = np.geomspace(1.5, 60, 300)
    ax.plot(xs, [cc85_mach(v) for v in xs], color="black", lw=2,
            label="Chevalier & Clegg (1985)", zorder=1)
    for (n, res), color in zip(runs.items(), COLORS):
        for star, marker in (("1", "o"), ("2", "s")):
            if star == "2" and n < STAR2_MIN_N:
                continue
            v = res[star]
            ax.plot(v["r_over_R"], v["local"], marker, ms=9, color=color,
                    mec="white", mew=1.5, zorder=3)
            ax.plot(v["r_over_R"], v["finder"], marker, ms=9, mfc="none",
                    mec=color, mew=2, zorder=3)
        v = res["1"]
        ax.annotate(f"N={n}", (v["r_over_R"], v["local"]), xytext=(-8, 9),
                    textcoords="offset points", ha="right", fontsize=9, color="#333333")

    # legend: encoding keys in neutral ink, resolution ramp separately
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color="black", lw=2, label="Chevalier & Clegg (1985)"),
        Line2D([], [], ls="", marker="o", ms=8, color="#555555", label="star 1 (strong wind)"),
        Line2D([], [], ls="", marker="s", ms=8, color="#555555", label="star 2 (weak wind)"),
        Line2D([], [], ls="", marker="o", ms=8, color="#555555", label=r"filled: wind just upstream, $|v_x|/c_s$ from the flow"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="none", mec="#555555", mew=2,
               label="open: shock finder, from the pressure jump"),
    ] + [Line2D([], [], ls="", marker="o", ms=8, color=c, label=f"N = {n}")
         for n, c in zip(runs, COLORS)]
    ax.legend(handles=handles, fontsize=8, loc="upper left", frameon=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1.5, 60)
    ax.set_ylim(1.5, 60)
    ticks = [2, 3, 5, 10, 20, 30, 50]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter([str(t) for t in ticks]))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"shock distance / injection radius  $r_s / R_{\rm inj}$")
    ax.set_ylabel("Mach number at the termination shock")
    ax.set_title("Stationary CWB, fixed 2-cell wind injection, pure hydro (binary axis)", fontsize=10)
    ax.grid(alpha=0.25, which="both", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "cwb_mach_vs_distance_ninj2.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
