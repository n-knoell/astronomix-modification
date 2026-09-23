"""
Full (hydro + CR) energy-budget pytest (plan Sec. 4, item 18: "Full energy
budget (thermal + kinetic + magnetic + CR) closes to round-off with
injection and streaming/collisional-loss accounting").

**Scope decision (user-confirmed): hydro + CR only, for now.** Two things
the plan's item 18 wording names are deliberately deferred, both flagged
rather than silently assumed:

1. **Magnetic energy** -- not included here; done since (2026-09-23) in
   `cr_mhd_energy_budget.py`, which closes the thermal + kinetic + magnetic
   + CR budget to ~1e-13. (The ~1e-6 plain-MHD residual that originally
   motivated deferring it turned out to be the magnetic update's fixed-point
   tolerance under the default `numerical_precision=SINGLE_PRECISION`, not
   Strang-split truncation error -- see that file's docstring.)
2. **Collisional losses** (hadronic/Coulomb on protons; synchrotron/IC/
   Coulomb/bremsstrahlung on electrons, plan Sec. 2 "Losses") -- not
   implemented anywhere in this module as a dynamical sink on `e_cr`
   (checked directly: `cr_grey_sources.py` has exactly 4 source terms --
   pressure-gradient, adiabatic-work, flux-relaxation, streaming-heating --
   none of them a collisional loss; items 13/14/15's emission code computes
   photon *diagnostics* from an assumed spectrum, it never feeds back into
   the simulation). So "collisional-loss accounting" is honestly N/A here,
   not silently assumed complete.

**What this test does check: injection (DSA, items 7/8) and streaming
(item 5) together, in the pieces that do exist.**
`cr_pressure_gradient_source`/`cr_adiabatic_work_source` are locally
energy-conserving by construction (their own docstrings: the product-rule
identity `div(P_cr v) = v.grad(P_cr) + P_cr div(v)` makes the gas energy
gain and `e_cr` loss cancel exactly up to a flux term); `inject_crs_at_shocks`
moves an exact, equal amount of energy from gas thermal to `e_cr`;
`cr_streaming_heating_source` is exactly conservative at its default
`streaming_heating_efficiency=1.0`. None of these add or remove energy from
outside the system -- unlike the SILCC-ISM project's own SN-driving energy
check (`pytests/stratified_ism/sn_driving_energy_conservation.py`), which
injects new energy at random times and has to separately account for how
much, nothing here injects energy from outside the box at all: it only
*moves* energy between thermal/kinetic/CR at a real, self-consistently
forming shock. So the correctness statement this test makes is simpler and
stronger than a differential identity: **total (thermal+kinetic+CR) energy
should be exactly constant for the whole run.**

**Technique, following that same SN-driving test's precedent (which reached
~7e-10 relative error this way): periodic box (no boundary flux loss),
`config.fixed_timestep=True` (removes any adaptive-CFL subtlety from the
comparison), no gravity/cooling.** Reuses ladder item 7's own Sedov-blast
physical setup (`cr_sedov_taylor.py`: `E_EXPLOSION=1`, `RHO_AMBIENT=1`,
`P_AMBIENT=1e-4`, tanh-tapered point deposit) but switches its default open
boundaries to periodic -- unlike item 7, this test has no reason to care
whether the shock front stays inside the domain (periodic wrap loses no
energy either way, whatever it does to the shock-finder's own detected
surface), only whether the *domain-total* energy stays constant.

**Result: genuine round-off, not just "very small".** At `NUM_CELLS=48`,
`num_timesteps=400`, `t_end=0.07`, float64, with both DSA injection
(`dsa_efficiency=0.1`) and streaming heating simultaneously active:
relative energy error `~9.5e-15` -- consistent with pure floating-point
accumulation over ~400 steps x ~1.1e5 cells, not a resolution-limited
truncation residual (contrast the MHD baseline's ~1e-6). Mass conserved
exactly (`0.0` relative error), as always for this flux-conservative FV
scheme under periodic BCs. Both mechanisms are genuinely active, not
no-ops: `E_cr` is a clearly nonzero fraction of the budget when
`dsa_efficiency=0.1` (confirms DSA fires), and disabling streaming heating
measurably changes the final state (confirms it does real work, not just
exercising a dead code path).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import SimulationConfig, SimulationParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    PERIODIC_BOUNDARY,
)

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# This is an exact energy-conservation check -- round-off accumulation in
# float32 over hundreds of fixed-timestep steps would itself be large enough
# to obscure what's being measured (same rationale as
# sn_driving_energy_conservation.py / cr_gradient_check.py).
jax.config.update("jax_enable_x64", True)

# ---- physical setup (matches cr_sedov_taylor.py / _sedov_setup.py, except
# for the boundary conditions and fixed timestep below) ----
GAMMA = 5.0 / 3.0
NUM_CELLS = 48
NUM_TIMESTEPS = 400
T_END = 0.07
E_EXPLOSION = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1e-4
R_EXPLOSION = 0.05
SMOOTH_CELLS = 2.0
DSA_MACH_MIN = 1.3
DSA_EFFICIENCY = 0.1

_PERIODIC_BOX = BoundarySettings(
    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def _run(dsa_efficiency: float, streaming: bool, grey_cosmic_rays: bool = True):
    """Run one periodic-box CR-DSA Sedov blast; return energy diagnostics."""
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        boundary_settings=_PERIODIC_BOX,
        fixed_timestep=True,
        num_timesteps=NUM_TIMESTEPS,
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=grey_cosmic_rays,
            diffusive_shock_acceleration=grey_cosmic_rays,
            streaming=streaming,
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (NUM_CELLS, NUM_CELLS, NUM_CELLS)
    density = jnp.ones(shape) * RHO_AMBIENT
    zeros = jnp.zeros(shape)

    dx = 1.0 / NUM_CELLS  # box_size = 1.0 (default)
    smooth_width = SMOOTH_CELLS * dx
    radius = helper_data.r
    weight = 0.5 * (1.0 - jnp.tanh((radius - R_EXPLOSION) / smooth_width))
    cell_volume = dx**3
    delta_p = E_EXPLOSION * (GAMMA - 1.0) / (jnp.sum(weight) * cell_volume)
    gas_pressure = P_AMBIENT + delta_p * weight

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            dsa_efficiency=dsa_efficiency, dsa_mach_min=DSA_MACH_MIN
        ),
    )

    def _energies(state):
        rho = state[registered_variables.density_index]
        vx = state[registered_variables.velocity_index.x]
        vy = state[registered_variables.velocity_index.y]
        vz = state[registered_variables.velocity_index.z]
        p_gas = state[registered_variables.pressure_index]
        e_thermal = float(jnp.sum(p_gas / (GAMMA - 1.0)) * cell_volume)
        e_kinetic = float(jnp.sum(0.5 * rho * (vx**2 + vy**2 + vz**2)) * cell_volume)
        if grey_cosmic_rays:
            e_cr = float(jnp.sum(state[registered_variables.cosmic_ray_e_index]) * cell_volume)
        else:
            e_cr = 0.0
        return e_thermal, e_kinetic, e_cr

    final_state = time_integration(initial_state, config, params, registered_variables)

    e_thermal_0, e_kinetic_0, e_cr_0 = _energies(initial_state)
    e_thermal_f, e_kinetic_f, e_cr_f = _energies(final_state)
    m0 = float(jnp.sum(density) * cell_volume)
    mf = float(jnp.sum(final_state[registered_variables.density_index]) * cell_volume)

    return dict(
        final_state=final_state,
        E_total_0=e_thermal_0 + e_kinetic_0 + e_cr_0,
        E_total_f=e_thermal_f + e_kinetic_f + e_cr_f,
        E_thermal_0=e_thermal_0,
        E_kinetic_0=e_kinetic_0,
        E_cr_0=e_cr_0,
        E_thermal_f=e_thermal_f,
        E_kinetic_f=e_kinetic_f,
        E_cr_f=e_cr_f,
        mass_0=m0,
        mass_f=mf,
    )


def test_cr_energy_budget(energy_tol: float = 1e-9, mass_tol: float = 1e-10):
    """Full (thermal + kinetic + CR) energy budget, DSA injection and
    streaming heating both active, periodic box, fixed timestep.

    Args:
        energy_tol: Maximum allowed relative error on
            ``E_total(t_end) == E_total(0)``. Calibrated observed error
            ~9.5e-15 -- a >1e5x margin.
        mass_tol: Maximum allowed relative error on total mass conservation.
    """
    run = _run(dsa_efficiency=DSA_EFFICIENCY, streaming=True)

    assert not bool(jnp.any(jnp.isnan(run["final_state"]))), "Run produced NaNs."

    rel_mass_err = abs(run["mass_f"] - run["mass_0"]) / abs(run["mass_0"])
    assert rel_mass_err < mass_tol, f"Mass not conserved: rel. err {rel_mass_err:.3e}."

    rel_energy_err = abs(run["E_total_f"] - run["E_total_0"]) / abs(run["E_total_0"])
    print(
        f"rel. mass err = {rel_mass_err:.3e}, rel. energy err = {rel_energy_err:.3e}, "
        f"E_cr(t_end) = {run['E_cr_f']:.6e}"
    )
    assert rel_energy_err < energy_tol, (
        f"Total (thermal+kinetic+CR) energy not conserved: rel. err "
        f"{rel_energy_err:.3e} >= tol {energy_tol}."
    )

    # DSA must have genuinely fired -- E_cr(t_end) a clearly nonzero,
    # bounded fraction of the budget (calibrated ~0.025-0.04 depending on
    # streaming; matches ladder item 7's own [0.01, 0.3] sanity band).
    cr_fraction = run["E_cr_f"] / run["E_total_0"]
    assert 0.005 < cr_fraction < 0.3, (
        f"DSA run's CR energy fraction ({cr_fraction:.4f}) is outside the "
        f"expected band -- injection may not be firing as intended."
    )

    # Streaming heating must be doing real work, not exercising a dead code
    # path: disabling it should measurably change the outcome.
    run_no_streaming = _run(dsa_efficiency=DSA_EFFICIENCY, streaming=False)
    state_diff = float(jnp.max(jnp.abs(run["final_state"] - run_no_streaming["final_state"])))
    assert state_diff > 1e-6, (
        f"Streaming heating had no detectable effect (max state diff "
        f"{state_diff:.3e}) -- is it actually being applied?"
    )

    # ---- plot: energy partition (with vs. without streaming heating) ----
    control = _run(dsa_efficiency=0.0, streaming=False, grey_cosmic_rays=False)

    fig, (ax_bars, ax_err) = plt.subplots(1, 2, figsize=(11, 4.5))

    labels = ["thermal", "kinetic", "CR"]
    x = jnp.arange(len(labels))
    width = 0.35
    ax_bars.bar(
        x - width / 2,
        [run_no_streaming["E_thermal_f"], run_no_streaming["E_kinetic_f"], run_no_streaming["E_cr_f"]],
        width, label="DSA only", color="C0",
    )
    ax_bars.bar(
        x + width / 2,
        [run["E_thermal_f"], run["E_kinetic_f"], run["E_cr_f"]],
        width, label="DSA + streaming", color="C1",
    )
    ax_bars.set_xticks(x, labels)
    ax_bars.set_ylabel("energy")
    ax_bars.set_title(f"Energy partition at t={T_END}\n(periodic box, fixed dt)")
    ax_bars.legend()

    ax_err.bar(
        ["hydro-only\n(control)", "DSA only", "DSA +\nstreaming"],
        [
            abs(control["E_total_f"] - control["E_total_0"]) / abs(control["E_total_0"]),
            abs(run_no_streaming["E_total_f"] - run_no_streaming["E_total_0"]) / abs(run_no_streaming["E_total_0"]),
            rel_energy_err,
        ],
        color=["grey", "C0", "C1"],
    )
    ax_err.set_yscale("log")
    ax_err.axhline(energy_tol, color="k", linestyle=":", label=f"tol = {energy_tol:.0e}")
    ax_err.set_ylabel("relative total-energy error")
    ax_err.set_title("Round-off regardless of which\nCR mechanisms are active")
    ax_err.legend()

    fig.suptitle("Ladder item 18 (hydro + CR only): full energy budget closes to round-off")
    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_energy_budget_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_cr_energy_budget()
