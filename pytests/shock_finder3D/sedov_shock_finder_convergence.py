"""
3D Sedov-Taylor blast wave: resolution-convergence study for the shock finder.

Runs the same spherical point explosion (``_sedov_setup.py``) at a sweep of
grid resolutions and checks that the Pfrommer shock finder's diagnostics
behave the way better-resolved shock capturing should:

* the detected shock surface becomes a geometrically thinner spherical shell
  (the radial spread of surface-cell radii, relative to the mean shock
  radius, shrinks as the grid is refined), and
* the measured density compression ratio across the shock climbs towards the
  ideal-gas strong-shock limit (gamma+1)/(gamma-1) = 4 as numerical diffusion
  decreases with resolution.

Both diagnostics are cross-checked two ways: once directly from the shock
finder's own surface cells (immediate pre/post neighbours along the shock
direction), and once from a shock-finder-independent spherically-binned
radial density profile (the peak of the binned profile over the ambient
density). Writes a convergence figure to ``figures/``.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

from astronomix.shock_finder3D._shock_zones import get_post_pre_shock_values

from _sedov_setup import GAMMA, RHO_AMBIENT, binned_radial_profile, run_sedov

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

RESOLUTIONS = [24, 32, 48, 64, 96, 128, 256]
STRONG_SHOCK_RHO_RATIO = (GAMMA + 1.0) / (GAMMA - 1.0)  # = 4 for gamma = 5/3


def _diagnostics_at_resolution(num_cells):
    """Run one Sedov blast and reduce it to the convergence diagnostics.

    Args:
        num_cells: The cubic grid resolution to run at.

    Returns:
        A dict of scalar diagnostics for this resolution.
    """
    run = run_sedov(num_cells)
    state = run["state"]
    helper_data = run["helper_data"]
    registered_variables = run["registered_variables"]
    sf_result = run["sf_result"]

    r = np.array(helper_data.r)
    density = np.array(state[registered_variables.density_index])
    pressure = np.array(state[registered_variables.pressure_index])
    surface = np.array(sf_result.shock_surface_cells)

    assert surface.sum() > 0, f"no shock found at N={num_cells}"

    r_surface = r[surface]
    mean_r = float(r_surface.mean())
    relative_spread = float(r_surface.std() / mean_r)

    p_post, p_pre, rho_post, rho_pre = get_post_pre_shock_values(
        sf_result.shock_direction, pressure, density
    )
    rho_ratio = np.array(rho_post)[surface] / np.maximum(
        np.array(rho_pre)[surface], 1e-30
    )

    bin_centers, density_profile = binned_radial_profile(r.flatten(), density.flatten())
    valid = ~np.isnan(density_profile)
    peak_compression_independent = float(density_profile[valid].max() / RHO_AMBIENT)

    mach = np.array(sf_result.mach_numbers)

    return dict(
        num_cells=num_cells,
        num_surface_cells=int(surface.sum()),
        mean_r=mean_r,
        relative_spread=relative_spread,
        mean_rho_ratio=float(rho_ratio.mean()),
        max_rho_ratio=float(rho_ratio.max()),
        peak_compression_independent=peak_compression_independent,
        mean_mach=float(mach[surface].mean()),
    )


def test_sedov_shock_finder_convergence(resolutions=RESOLUTIONS):
    """Run the resolution sweep and check the shock finder's diagnostics converge.

    Args:
        resolutions: The cubic grid resolutions to sweep over, low to high.
    """
    rows = [_diagnostics_at_resolution(n) for n in resolutions]

    for row in rows:
        print(
            f"N={row['num_cells']:>4d}  surface_cells={row['num_surface_cells']:>7d}  "
            f"mean_r={row['mean_r']:.4f}  relative_spread={row['relative_spread']:.4f}  "
            f"mean_rho_ratio={row['mean_rho_ratio']:.3f}  "
            f"independent_peak_compression={row['peak_compression_independent']:.3f}  "
            f"mean_mach={row['mean_mach']:.2f}"
        )

    relative_spreads = [row["relative_spread"] for row in rows]
    mean_rho_ratios = [row["mean_rho_ratio"] for row in rows]
    independent_peaks = [row["peak_compression_independent"] for row in rows]

    # The shock shell must become geometrically thinner (relative to its
    # radius) as the grid is refined: the coarsest run must be the worst.
    assert relative_spreads[0] == max(relative_spreads), (
        f"lowest resolution ({resolutions[0]}) is not the worst-localised shell: "
        f"relative spreads = {relative_spreads}"
    )
    assert relative_spreads[-1] == min(relative_spreads), (
        f"highest resolution ({resolutions[-1]}) is not the best-localised shell: "
        f"relative spreads = {relative_spreads}"
    )

    # The measured compression ratio must climb towards the strong-shock
    # limit as numerical diffusion decreases with resolution, both as seen
    # by the shock finder itself and by the independent radial profile.
    assert mean_rho_ratios[-1] > mean_rho_ratios[0], (
        f"shock-finder compression ratio did not increase with resolution: "
        f"{mean_rho_ratios}"
    )
    assert independent_peaks[-1] > independent_peaks[0], (
        f"independently-measured peak compression did not increase with "
        f"resolution: {independent_peaks}"
    )
    assert independent_peaks[-1] <= STRONG_SHOCK_RHO_RATIO * 1.05, (
        f"independently-measured peak compression {independent_peaks[-1]:.3f} "
        f"exceeds the ideal-gas strong-shock limit "
        f"{STRONG_SHOCK_RHO_RATIO:.3f}"
    )

    # -------------------------------------------------------------
    # ---- convergence figure ----
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.loglog(resolutions, relative_spreads, "o-", color="tab:blue")
    ax.set_xlabel("N (cells per dimension)")
    ax.set_ylabel("std(r_surface) / mean(r_surface)")
    ax.set_title("shock-shell relative thickness")
    ax.grid(True, which="both", ls=":", alpha=0.6)

    ax = axes[1]
    ax.plot(resolutions, mean_rho_ratios, "o-", color="tab:orange", label="shock finder (mean)")
    ax.plot(resolutions, independent_peaks, "s--", color="tab:green", label="independent radial profile")
    ax.axhline(STRONG_SHOCK_RHO_RATIO, color="black", ls=":", lw=1, label="strong-shock limit (4)")
    ax.set_xlabel("N (cells per dimension)")
    ax.set_ylabel(r"$\rho_2/\rho_1$")
    ax.set_title("shock compression ratio")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.6)

    ax = axes[2]
    mean_machs = [row["mean_mach"] for row in rows]
    ax.plot(resolutions, mean_machs, "o-", color="tab:red")
    ax.set_xlabel("N (cells per dimension)")
    ax.set_ylabel("mean Mach number at surface")
    ax.set_title("shock Mach number")
    ax.grid(True, ls=":", alpha=0.6)

    fig.suptitle("Sedov blast shock-finder resolution convergence")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sedov_shock_finder_convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    test_sedov_shock_finder_convergence()
