"""
Shock-finder Mach recovery on synthetic, numerically smeared planar shocks.

A stationary planar shock along x with the exact Rankine-Hugoniot jump for a
given Mach number, every primitive blended across the shock with
``0.5 (1 + tanh((x - x_s) / w))`` to mimic a shock smeared over a few cells.
The pre-shock gas flows toward -x into the shock, so the flow converges and
the finder's zone criteria hold across the ramp.

Compares the finder's median surface-cell Mach against the exact value for
the adaptive walk with and without ``mach_sampling_extend`` (FIXES_TODO.md
round 23): the plain walk stops at the shock-zone exit, which for strong
shocks lies inside the ramp; the extension walks on while the pressure keeps
falling/rising. Also checks the dissipated thermal-energy flux against the
exact ``delta(M) * 0.5 rho1 (M c1)^3`` (round 24: in adaptive mode the flux
uses the pre-shock state from the same walk as the Mach number). Writes
``figures/sedov/synthetic_smeared_shock_mach.png``.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# numerics
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# plotting
import matplotlib.pyplot as plt

from astronomix import (
    FINITE_VOLUME,
    OPEN_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    get_helper_data,
    get_registered_variables,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix.shock_finder3D._energy_dissipation import _thermalization_efficiency

FIG_DIR = Path(__file__).resolve().parent / "figures" / "sedov"
FIG_DIR.mkdir(parents=True, exist_ok=True)

GAMMA = 5 / 3
NUM_CELLS = 64
MACH_NUMBERS = (3.0, 30.0, 150.0)
# tanh width in cells. The pressure tail of a linear-space tanh across a
# p2/p1 ~ 3e4 jump decays as 3e4 exp(-2 d / w): for w = 2 it reaches the ~2%
# per-step plateau criterion ~14 cells from the shock, inside max_steps (15)
# and the box; w = 3 would not (the exact pre-shock state is then not in the
# sampled range at all).
WIDTHS = (0.5, 1.0, 2.0)


def exact_thermal_energy_flux(mach, rho1=1.0, p1=1.0):
    c1 = np.sqrt(GAMMA * p1 / rho1)
    return float(_thermalization_efficiency(jnp.asarray(mach), GAMMA)) * 0.5 * rho1 * (mach * c1) ** 3


def planar_shock_mach(mach, width_cells, extend):
    """Median finder Mach and thermal-energy flux on the surface cells of
    one synthetic shock."""
    open_1d = BoundarySettings1D(left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY)
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME, dimensionality=3, num_cells=NUM_CELLS, box_size=1.0,
        boundary_settings=BoundarySettings(open_1d, open_1d, open_1d),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    rho1, p1 = 1.0, 1.0
    c1 = np.sqrt(GAMMA * p1 / rho1)
    rho2 = rho1 * (GAMMA + 1) * mach**2 / ((GAMMA - 1) * mach**2 + 2)
    p2 = p1 * (2 * GAMMA * mach**2 - (GAMMA - 1)) / (GAMMA + 1)
    u1 = -mach * c1           # pre-shock gas (x > x_s) flows into the shock
    u2 = u1 * rho1 / rho2     # post-shock (x < x_s), shock at rest

    x = helper_data.geometric_centers[..., 0]
    dx = 1.0 / NUM_CELLS
    blend = 0.5 * (1 + jnp.tanh((x - 0.5) / (width_cells * dx)))  # 0 post, 1 pre
    mix = lambda post, pre: post + (pre - post) * blend
    zeros = jnp.zeros_like(x)
    state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=mix(rho2, rho1), velocity_x=mix(u2, u1), velocity_y=zeros, velocity_z=zeros,
        gas_pressure=mix(p2, p1),
    )
    config = finalize_config(config, state.shape)
    sf = find_shocks_pfrommer(
        state, config, registered_variables, helper_data, mach_min=1.3,
        mach_sampling_adaptive=True, mach_sampling_steps=15, mach_sampling_extend=extend,
    )
    surface = np.asarray(sf.shock_surface_cells)
    found = surface & (np.asarray(sf.mach_numbers) > 0)
    if not found.any():
        return float("nan"), float("nan")
    return (float(np.median(np.asarray(sf.mach_numbers)[found])),
            float(np.median(np.asarray(sf.thermal_energy_flux)[found])))


def test_synthetic_smeared_shock(tol=0.1):
    results, fluxes = {}, {}
    print(f"{'M':>6s} {'w':>5s} {'adaptive':>10s} {'+extend':>10s}   flux/exact: {'adaptive':>9s} {'+extend':>9s}")
    for mach in MACH_NUMBERS:
        exact_flux = exact_thermal_energy_flux(mach)
        for width in WIDTHS:
            plain, flux_plain = planar_shock_mach(mach, width, False)
            extended, flux_extended = planar_shock_mach(mach, width, True)
            results[(mach, width)] = (plain, extended)
            fluxes[(mach, width)] = (flux_plain / exact_flux, flux_extended / exact_flux)
            print(f"{mach:6.0f} {width:5.1f} {plain:10.2f} {extended:10.2f}   "
                  f"{'':11s} {flux_plain / exact_flux:9.3f} {flux_extended / exact_flux:9.3f}")

    fig, axes = plt.subplots(1, len(MACH_NUMBERS), figsize=(13, 4))
    for ax, mach in zip(axes, MACH_NUMBERS):
        ax.plot(WIDTHS, [results[(mach, w)][0] / mach for w in WIDTHS], "o--", label="adaptive walk")
        ax.plot(WIDTHS, [results[(mach, w)][1] / mach for w in WIDTHS], "s-", label="+ monotone extension")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set(title=f"exact M = {mach:g}", xlabel="tanh width [cells]", ylabel="M_found / M_exact",
               ylim=(0, 1.2))
        ax.legend(fontsize=8)
    fig.suptitle("Synthetic smeared planar shock: finder Mach recovery")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "synthetic_smeared_shock_mach.png", dpi=150)
    plt.close(fig)

    for (mach, width), (plain, extended) in results.items():
        assert abs(extended / mach - 1) < tol, f"M={mach}, w={width}: extended walk gives {extended}"
    # f_th ~ delta(M) M^3 c1^3: a few % in M shows up ~3x in the flux
    for (mach, width), (_, flux_ratio) in fluxes.items():
        assert abs(flux_ratio - 1) < 3 * tol, f"M={mach}, w={width}: flux/exact = {flux_ratio}"


if __name__ == "__main__":
    test_synthetic_smeared_shock()
