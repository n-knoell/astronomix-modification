"""
3D Sedov-Taylor blast wave: correctness test for the Pfrommer shock finder.

Runs a spherical point explosion (see ``_sedov_setup.py``) with the
finite-volume HLLC solver and applies
:func:`astronomix.shock_finder3D.pfrommer_shock_finder.find_shocks_pfrommer`
to the final state. Checks that the detected shock:

* forms a thin, roughly spherical shell (not scattered noise),
* sits within the density/pressure transition region measured independently
  from a spherically-binned radial profile (i.e. the finder is not off in
  some unrelated part of the domain),
* satisfies the Rankine-Hugoniot compression bounds for an ideal gas
  (density jump strictly > 1 and <= (gamma+1)/(gamma-1), pressure jump > 1),
* reports Mach numbers only at/above ``mach_min`` on surface cells and zero
  elsewhere,
* reports a non-negative dissipated thermal-energy flux, nonzero only on
  surface cells.

Writes a diagnostic figure (radial profile + shock geometry + Mach
histogram) to ``figures/``.
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
from astronomix.shock_finder3D.plot_helper import plot_shock_surface_3d

from _sedov_setup import (
    GAMMA,
    MACH_MIN,
    RHO_AMBIENT,
    T_END,
    binned_radial_profile,
    run_sedov,
)

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

NUM_CELLS = 256
STRONG_SHOCK_RHO_RATIO = (GAMMA + 1.0) / (GAMMA - 1.0)  # = 4 for gamma = 5/3


def _transition_region(bin_centers, density_profile, ambient=RHO_AMBIENT):
    """Return the radial extent of the density transition, measured "by hand".

    The Sedov density profile rises from the rarefied interior to a peak
    immediately behind the shock, then falls sharply back to the ambient
    value across the (numerically smeared) shock front. This locates that
    falling edge independently of the shock finder: the peak radius, and the
    first radius beyond the peak where the binned density has returned to
    within 5% of the ambient value.

    Args:
        bin_centers: Radii of the profile bins.
        density_profile: Binned mean density (may contain nan for empty bins).
        ambient: The ambient (pre-shock) density.

    Returns:
        ``(r_peak, r_ambient_return)``.
    """
    valid = ~np.isnan(density_profile)
    centers, prof = bin_centers[valid], density_profile[valid]
    r_peak = centers[np.argmax(prof)]
    beyond_peak = centers > r_peak
    near_ambient = np.abs(prof - ambient) < 0.05 * ambient
    candidates = centers[beyond_peak & near_ambient]
    r_ambient_return = candidates.min() if candidates.size else centers[-1]
    return r_peak, r_ambient_return


def plot_shocked_cells_3d(run, num_cells=NUM_CELLS):
    """Render the shock finder's detected surface cells in 3D.

    Colors each surface cell by its Rankine-Hugoniot Mach number and draws
    the local shock-direction vector, reusing
    :func:`astronomix.shock_finder3D.plot_helper.plot_shock_surface_3d`.
    Writes the figure to ``figures/``.

    Args:
        run: The dict returned by ``run_sedov`` (must include ``helper_data``
            and ``sf_result``).
        num_cells: Grid resolution, used only for the title/filename.

    Returns:
        ``(fig, ax)``.
    """
    helper_data = run["helper_data"]
    sf_result = run["sf_result"]

    surface = np.array(sf_result.shock_surface_cells)
    mach = np.array(sf_result.mach_numbers)
    shock_dir_x, shock_dir_y, shock_dir_z = (
        np.array(component) for component in sf_result.shock_direction
    )

    geometric_centers = np.array(helper_data.geometric_centers)
    x, y, z = (geometric_centers[..., axis] for axis in range(3))

    fig, ax = plot_shock_surface_3d(
        x[surface], y[surface], z[surface],
        shock_dir_x[surface], shock_dir_y[surface], shock_dir_z[surface],
        mach[surface],
        title=f"Sedov blast shocked cells (N={num_cells}^3, t={T_END})",
        mode="SCATTER",
    )
    fig.savefig(FIG_DIR / f"sedov_shocked_cells_3d_{num_cells}.png", dpi=150)
    plt.close(fig)

    return fig, ax


def test_sedov_shock_finder_correctness(num_cells=NUM_CELLS):
    """Run the 3D Sedov blast and check the shock finder's output is physical.

    Args:
        num_cells: The cubic grid resolution to run at.
    """
    run = run_sedov(num_cells)
    plot_shocked_cells_3d(run, num_cells)

    state = run["state"]
    config = run["config"]
    helper_data = run["helper_data"]
    registered_variables = run["registered_variables"]
    sf_result = run["sf_result"]

    r = np.array(helper_data.r)
    density = np.array(state[registered_variables.density_index])
    pressure = np.array(state[registered_variables.pressure_index])

    surface = np.array(sf_result.shock_surface_cells)
    zones = np.array(sf_result.shock_zones)
    mach = np.array(sf_result.mach_numbers)
    flux = np.array(sf_result.thermal_energy_flux)

    assert int(sf_result.num_shocks) > 0, "no shock surface cells were found"
    assert surface.sum() > 0
    # the surface must be a (non-trivial) subset of the broader shock zone
    assert np.array_equal(surface & zones, surface)
    assert zones.sum() >= surface.sum()

    # -------------------------------------------------------------
    # ---- geometry: the surface should be a thin, roughly spherical shell --
    # -------------------------------------------------------------
    r_surface = r[surface]
    mean_r, std_r = r_surface.mean(), r_surface.std()
    relative_spread = std_r / mean_r
    assert relative_spread < 0.15, (
        f"shock surface radii are too scattered to be a shell: "
        f"std/mean = {relative_spread:.3f}"
    )

    # -------------------------------------------------------------
    # ---- cross-check against an independent radial density profile ----
    # -------------------------------------------------------------
    bin_centers, density_profile = binned_radial_profile(
        r.flatten(), density.flatten()
    )
    r_peak, r_ambient_return = _transition_region(bin_centers, density_profile)
    dx = 1.0 / num_cells
    margin = 3 * dx
    assert r_peak - margin <= mean_r <= r_ambient_return + margin, (
        f"shock finder's mean surface radius {mean_r:.4f} falls outside the "
        f"density transition region [{r_peak:.4f}, {r_ambient_return:.4f}] "
        f"(+/- {margin:.4f} margin) measured independently from the "
        f"radial profile"
    )

    # -------------------------------------------------------------
    # ---- Rankine-Hugoniot bounds at surface cells (immediate neighbours) --
    # -------------------------------------------------------------
    p_post, p_pre, rho_post, rho_pre = get_post_pre_shock_values(
        sf_result.shock_direction, pressure, density
    )
    p_ratio = np.array(p_post)[surface] / np.maximum(np.array(p_pre)[surface], 1e-30)
    rho_ratio = np.array(rho_post)[surface] / np.maximum(
        np.array(rho_pre)[surface], 1e-30
    )

    assert np.all(p_ratio > 1.0), "pressure must increase across every detected shock"
    assert np.all(rho_ratio > 1.0), "density must increase across every detected shock"
    # small numerical slack above the exact strong-shock bound
    assert np.all(rho_ratio <= STRONG_SHOCK_RHO_RATIO * 1.05), (
        f"density jump exceeds the ideal-gas strong-shock limit "
        f"(gamma+1)/(gamma-1) = {STRONG_SHOCK_RHO_RATIO:.3f}: "
        f"max ratio = {rho_ratio.max():.3f}"
    )

    # -------------------------------------------------------------
    # ---- Mach numbers and thermal-energy flux ----
    # -------------------------------------------------------------
    assert np.all(mach[surface] >= MACH_MIN - 1e-6)
    assert np.all(mach[~surface] == 0.0)

    assert np.all(flux >= 0.0), "dissipated thermal-energy flux must be non-negative"
    assert np.all((flux != 0) == surface), (
        "thermal-energy flux must be nonzero exactly at surface cells"
    )

    # -------------------------------------------------------------
    # ---- diagnostic figure ----
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.plot(bin_centers, density_profile, color="black", lw=1.5, label="density (binned)")
    ax.axvspan(r_peak, r_ambient_return, color="gray", alpha=0.2, label="transition region")
    ax.axvline(mean_r, color="red", lw=1.5, label="shock finder (mean surface r)")
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("radius")
    ax.set_ylabel("density")
    ax.set_title(f"radial density profile (N={num_cells})")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(r_surface, bins=40, color="tab:red", alpha=0.8)
    ax.set_xlabel("radius of surface cells")
    ax.set_ylabel("count")
    ax.set_title(f"shock-surface radii (std/mean = {relative_spread:.3f})")

    ax = axes[2]
    ax.hist(mach[surface], bins=40, color="tab:orange", alpha=0.8)
    ax.axvline(MACH_MIN, color="black", ls="--", lw=1, label="mach_min")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("count")
    ax.set_title("Mach number at surface cells")
    ax.legend(fontsize=8)

    fig.suptitle(f"Sedov blast shock-finder correctness check, N={num_cells}^3, t={T_END}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"sedov_shock_finder_correctness_{num_cells}.png", dpi=150)
    plt.close(fig)

    print(
        f"[N={num_cells}] surface cells={surface.sum()}, "
        f"mean r={mean_r:.4f} +/- {std_r:.4f}, "
        f"rho_ratio in [{rho_ratio.min():.2f}, {rho_ratio.max():.2f}], "
        f"mach in [{mach[surface].min():.2f}, {mach[surface].max():.2f}]"
    )


if __name__ == "__main__":
    test_sedov_shock_finder_correctness()
