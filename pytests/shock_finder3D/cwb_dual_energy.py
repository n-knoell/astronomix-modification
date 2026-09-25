"""Stationary CWB: dual-energy (entropy) formalism (``SimulationConfig.dual_energy``)
and a 1e4 K wind temperature floor (``WindConfig.wind_temperature_floor``),
combined with plain EI injection and the analytic wind zone. See
FIXES_TODO.md rounds 19-22.

Prints Mach statistics for all shock-surface cells and for the apex (surface
cells within 2 cells of the binary axis, split at the contact into star 1's
and star 2's shock), and the coldest pre-shock
temperature of star 1's wind on the axis. Writes to ``figures/cwb_new_inj/``:

* ``cwb_dual_energy_shocked_cells_{2d,3d}_<case>_<N>.png`` per case, laid out
  like ``cwb_shock_finder.py``'s ``cwb_shocked_cells_{2d,3d}_<N>.png``;
* ``cwb_dual_energy_axis_profile_<N>.png``: temperature and local Mach number
  along the binary axis for all cases;
* ``cwb_dual_energy_mach_hist_<N>.png``: surface-cell Mach distributions.

Usage: ``python cwb_dual_energy.py [N] [zone_fraction]`` (defaults: 64, 0.5).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1, interval=5)
# ruff: noqa: E402
# =======================

# general
import sys
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

from _cwb_setup import (
    GAMMA,
    MACH_MIN,
    P_AMBIENT,
    RHO_AMBIENT,
    SEPARATION,
    axis_profile,
    kelvin_to_code_temperature,
    run_cwb,
    v_inf1,
)
from cwb_shock_finder import plot_shocked_cells_2d, plot_shocked_cells_3d

FIG_DIR = Path(__file__).resolve().parent / "figures" / "cwb_new_inj"
FIG_DIR.mkdir(parents=True, exist_ok=True)

NUM_CELLS = 64
ZONE_FRACTION = 0.5
FLOOR_KELVIN = 1e4
KELVIN_PER_CODE = 1.0 / kelvin_to_code_temperature(1.0)
STAR_X = np.array([-SEPARATION / 2, SEPARATION / 2])


def summarize(label, run):
    sf = run["sf_result"]
    surface = np.asarray(sf.shock_surface_cells)
    mach = np.asarray(sf.mach_numbers)[surface]
    finite = bool(np.all(np.isfinite(np.asarray(run["state"]))))
    x, density, pressure, _, velocity_x = axis_profile(run)
    temperature = pressure / density
    # pre-shock wind of star 1: between star 1 and the first cell past the
    # temperature minimum on the star-2 side of star 1
    upstream = (x > STAR_X[0] + 0.02) & (x < 0.0)
    # apex: surface cells within 2 cells of the binary axis, between the stars
    centers = np.asarray(run["helper_data"].geometric_centers)
    dx = centers[1, 0, 0, 0] - centers[0, 0, 0, 0]
    box_center = 0.5 * (centers[0, 0, 0] + centers[-1, -1, -1])
    rel = centers - box_center
    near_axis = (np.hypot(rel[..., 1], rel[..., 2]) <= 2 * dx) & (np.abs(rel[..., 0]) < SEPARATION / 2)
    # split at the contact: hottest cell on the axis between the stars,
    # excluding the (hot) plain-EI injection spheres around them
    between = np.abs(x) < SEPARATION / 2 - 4 * dx
    x_contact = x[between][np.argmax(temperature[between])]
    mach_all = np.asarray(sf.mach_numbers)
    apex1 = mach_all[surface & near_axis & (rel[..., 0] < x_contact)]
    apex2 = mach_all[surface & near_axis & (rel[..., 0] > x_contact)]
    fmt = lambda m: f"{np.median(m):.1f}/{m.max():.1f} (n={m.size})" if m.size else "none"
    print(
        f"{label:>22s}: finite={finite} n_surface={surface.sum()} "
        f"Mach max {mach.max():.2f} median {np.median(mach):.2f} "
        f"p90 {np.percentile(mach, 90):.2f} | apex Mach median/max: star-1 shock "
        f"{fmt(apex1)}, star-2 shock {fmt(apex2)} | star-1 wind on axis: "
        f"T_min {temperature[upstream].min():.3g} ({temperature[upstream].min() * KELVIN_PER_CODE:.3g} K) "
        f"local Mach max {(np.abs(velocity_x) / np.sqrt(GAMMA * temperature))[upstream].max():.1f}",
        flush=True,
    )
    return mach


if __name__ == "__main__":
    num_cells = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_CELLS
    fraction = float(sys.argv[2]) if len(sys.argv) > 2 else ZONE_FRACTION

    zone = dict(analytic_wind_zone=True, wind_zone_stagnation_fraction=fraction)
    floor = dict(wind_floor_kelvin=FLOOR_KELVIN)
    cases = [
        ("plain EI", "plain", dict()),
        (f"zone f={fraction:g}", "zone", zone),
        (f"zone f={fraction:g} + DE", "zone_de", dict(zone, dual_energy=True)),
        (f"zone f={fraction:g} + 1e4 K floor", "zone_floor", dict(zone, **floor)),
        (f"zone f={fraction:g} + DE + 1e4 K floor", "zone_de_floor",
         dict(zone, dual_energy=True, **floor)),
    ]
    mach_floor = v_inf1 / np.sqrt(GAMMA * kelvin_to_code_temperature(FLOOR_KELVIN))
    print(f"star-1 wind Mach at the {FLOOR_KELVIN:g} K floor: {mach_floor:.0f}", flush=True)
    runs, machs = [], []
    for label, tag, kwargs in cases:
        run = run_cwb(num_cells, **kwargs)
        machs.append((label, summarize(label, run)))
        plot_shocked_cells_3d(
            run, num_cells, fig_dir=FIG_DIR, filename=f"cwb_dual_energy_shocked_cells_3d_{tag}_{num_cells}.png",
            title=f"CWB shocked cells, {label} (N={num_cells}^3)",
        )
        plot_shocked_cells_2d(
            run, num_cells, fig_dir=FIG_DIR, filename=f"cwb_dual_energy_shocked_cells_2d_{tag}_{num_cells}.png",
            title=f"Stationary CWB, orbital-plane slice, {label}, N={num_cells}^3",
        )
        runs.append((label, run))

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for label, run in runs:
        x, density, pressure, _, velocity_x = axis_profile(run)
        temperature = pressure / density
        ls = "-" if "DE" in label else "--"
        axes[0].semilogy(x, temperature, ls, marker="o", ms=2, label=label)
        axes[1].semilogy(x, np.abs(velocity_x) / np.sqrt(GAMMA * temperature), ls, marker="o", ms=2)
    axes[0].axhline(P_AMBIENT / RHO_AMBIENT, color="gray", lw=0.8, ls=":", label="ambient")
    axes[0].axhline(kelvin_to_code_temperature(FLOOR_KELVIN), color="gray", lw=0.8, ls="-.",
                    label=f"{FLOOR_KELVIN:g} K floor")
    axes[0].set_ylabel("T = p / rho")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("local Mach |v_x| / c_s")
    axes[1].set_xlabel("x (binary axis)")
    for ax in axes:
        for star_x in STAR_X:
            ax.axvline(star_x, color="k", lw=0.5)
    fig.suptitle(f"CWB binary axis, N={num_cells}^3: dual energy on (solid) / off (dashed)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cwb_dual_energy_axis_profile_{num_cells}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.logspace(np.log10(MACH_MIN), np.log10(max(m.max() for _, m in machs)), 40)
    for label, mach in machs:
        ax.hist(mach, bins=bins, histtype="step", lw=1.5,
                label=f"{label} (median {np.median(mach):.1f}, max {mach.max():.1f})")
    ax.set_xscale("log")
    ax.set_xlabel("Mach number (shock surface cells)")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title(f"CWB shock-finder Mach distribution, N={num_cells}^3")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cwb_dual_energy_mach_hist_{num_cells}.png", dpi=150)
    plt.close(fig)
    print("figures written to", FIG_DIR)
