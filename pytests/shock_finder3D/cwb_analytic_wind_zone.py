"""Stationary CWB with and without the analytic wind zone
(``WindConfig.analytic_wind_zone``, see FIXES_TODO.md rounds 18-20).

Runs ``_cwb_setup.run_cwb`` once with plain EI injection and once per
``ZONE_FRACTIONS`` entry with the analytic wind zone on. Writes three figures
to ``figures/``:

* ``cwb_wind_zone_axis_profile_<N>.png``: temperature ``p/rho``, density and
  local Mach number ``|v_x| / sqrt(gamma p / rho)`` along the binary axis,
  with the zone edges marked.
* ``cwb_wind_zone_mach_hist_<N>.png``: the shock finder's surface-cell Mach
  distributions.
* ``cwb_wind_zone_slices_<N>.png``: orbital-plane temperature slices and
  shocked cells (colored by Mach), with the zones drawn in.

Usage: ``python cwb_analytic_wind_zone.py [N] [fraction ...]``
(defaults: N=64, fractions 0.5 0.9).
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
from matplotlib.colors import LogNorm

from _cwb_setup import (
    BOX_SIZE,
    GAMMA,
    MACH_MIN,
    P_AMBIENT,
    RHO_AMBIENT,
    SEPARATION,
    axis_profile,
    mlr1,
    mlr2,
    run_cwb,
    v_inf1,
    v_inf2,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

NUM_CELLS = 64
ZONE_FRACTIONS = (0.5, 0.9)
STAR_X = np.array([-SEPARATION / 2, SEPARATION / 2])


def zone_radii(fraction, num_cells):
    """Zone radii of both stars, mirroring ``stellar_wind._wind_zone_radii``
    (injection radius ``num_cells // 32`` cells as the lower bound)."""
    sqrt_momentum_rate = np.sqrt(np.array([mlr1 * v_inf1, mlr2 * v_inf2]))
    stagnation = SEPARATION * sqrt_momentum_rate / sqrt_momentum_rate.sum()
    injection_radius = (num_cells // 32) * BOX_SIZE / num_cells
    return np.maximum(fraction * stagnation, injection_radius)


def summarize(label, run):
    sf = run["sf_result"]
    surface = np.asarray(sf.shock_surface_cells)
    mach = np.asarray(sf.mach_numbers)[surface]
    finite = bool(np.all(np.isfinite(np.asarray(run["state"]))))
    print(
        f"{label:>14s}: finite={finite} n_surface={surface.sum()} "
        f"Mach max {mach.max():.2f} median {np.median(mach):.2f} "
        f"p90 {np.percentile(mach, 90):.2f}",
        flush=True,
    )
    return mach


def plot_axis_profiles(runs, num_cells):
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    t_ambient = P_AMBIENT / RHO_AMBIENT
    for (label, fraction, run), color in zip(runs, plt.cm.viridis(np.linspace(0, 0.85, len(runs)))):
        x, density, pressure, _, velocity_x = axis_profile(run)
        temperature = pressure / density
        axes[0].semilogy(x, temperature, "o-", ms=3, color=color, label=label)
        axes[1].semilogy(x, density, "o-", ms=3, color=color)
        axes[2].semilogy(
            x, np.abs(velocity_x) / np.sqrt(GAMMA * temperature), "o-", ms=3, color=color
        )
        if fraction is not None:
            for star_x, radius in zip(STAR_X, zone_radii(fraction, num_cells)):
                for ax in axes:
                    ax.axvspan(star_x - radius, star_x + radius, color=color, alpha=0.08)
    axes[0].axhline(t_ambient, color="gray", ls="--", label="ambient")
    axes[0].set_ylabel("T = p / rho")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("density")
    axes[2].set_ylabel("local Mach |v_x| / c_s")
    axes[2].set_xlabel("x (binary axis)")
    for ax in axes:
        for star_x in STAR_X:
            ax.axvline(star_x, color="k", lw=0.5)
    fig.suptitle(f"CWB binary-axis profiles, N={num_cells}^3 (shaded: wind zones)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cwb_wind_zone_axis_profile_{num_cells}.png", dpi=150)
    plt.close(fig)


def plot_mach_histograms(machs, num_cells):
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
    fig.savefig(FIG_DIR / f"cwb_wind_zone_mach_hist_{num_cells}.png", dpi=150)
    plt.close(fig)


def plot_slices(runs, num_cells):
    z_index = num_cells // 2
    extent = [-BOX_SIZE / 2, BOX_SIZE / 2, -BOX_SIZE / 2, BOX_SIZE / 2]
    temperatures = []
    for _, _, run in runs:
        rv = run["registered_variables"]
        state = np.asarray(run["state"])
        temperatures.append(
            state[rv.pressure_index][:, :, z_index] / state[rv.density_index][:, :, z_index]
        )
    vmin = min(t[t > 0].min() for t in temperatures)
    vmax = max(t.max() for t in temperatures)

    fig, axes = plt.subplots(2, len(runs), figsize=(4.8 * len(runs), 9.5), squeeze=False)
    for column, ((label, fraction, run), temperature) in enumerate(zip(runs, temperatures)):
        ax = axes[0, column]
        im = ax.imshow(temperature.T, origin="lower", extent=extent, cmap="magma",
                       norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.set_title(f"T = p/rho, {label}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, column]
        sf = run["sf_result"]
        surface = np.asarray(sf.shock_surface_cells)[:, :, z_index]
        mach = np.asarray(sf.mach_numbers)[:, :, z_index]
        centers = np.asarray(run["helper_data"].geometric_centers)[:, :, z_index]
        sc = ax.scatter(centers[..., 0][surface] - BOX_SIZE / 2,
                        centers[..., 1][surface] - BOX_SIZE / 2,
                        c=mach[surface], cmap="hot", norm=LogNorm(vmin=MACH_MIN, vmax=40), s=10)
        ax.set_xlim(extent[:2])
        ax.set_ylim(extent[2:])
        ax.set_title(f"shocked cells (Mach), {label}")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

        for ax in axes[:, column]:
            ax.set_aspect("equal")
            ax.scatter(STAR_X, [0, 0], marker="*", color="cyan", s=80,
                       edgecolor="black", linewidths=0.5, zorder=3)
            if fraction is not None:
                for star_x, radius in zip(STAR_X, zone_radii(fraction, num_cells)):
                    ax.add_patch(plt.Circle((star_x, 0), radius, fill=False,
                                            color="cyan", ls="--", lw=1))
    fig.suptitle(f"CWB orbital plane, N={num_cells}^3 (dashed: wind zones)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cwb_wind_zone_slices_{num_cells}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    num_cells = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_CELLS
    fractions = tuple(float(f) for f in sys.argv[2:]) or ZONE_FRACTIONS

    runs = [("plain EI", None, run_cwb(num_cells))]
    for fraction in fractions:
        runs.append((
            f"zone f={fraction:g}",
            fraction,
            run_cwb(num_cells, analytic_wind_zone=True,
                    wind_zone_stagnation_fraction=fraction),
        ))

    machs = [(label, summarize(label, run)) for label, _, run in runs]
    for label, fraction, _ in runs:
        if fraction is not None:
            print(f"{label}: zone radii {zone_radii(fraction, num_cells)}")

    plot_axis_profiles(runs, num_cells)
    plot_mach_histograms(machs, num_cells)
    plot_slices(runs, num_cells)
    print("figures written to", FIG_DIR)
