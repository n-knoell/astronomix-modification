"""
Baseline MHD energy-conservation check (no cosmic rays) -- an intermediate
step ahead of the grey-CR ladder's item 18 ("full energy budget: thermal +
kinetic + magnetic + CR closes to round-off with injection and streaming/
collisional-loss accounting", `pytests/shock_finder3D/
astronomix_CR_implementation_plan.md`).

**Why this exists as its own, CR-free test.** Every CR-grey ladder test to
date (items 1-17) has been hydro-only -- none has ever combined CR-grey with
real MHD dynamics (`config.mhd`, an evolved, non-static magnetic field).
Item 18 wants "thermal + kinetic + magnetic + CR" to close to round-off; if
the *plain* MHD solver (CR off entirely) doesn't already close cleanly, item
18 has no chance of doing so with CR feedback layered on top, and any
failure there would be impossible to attribute to CR-specific coupling vs.
the pre-existing MHD solver. This test isolates that question first, using
the codebase's own established, already-validated dynamic MHD IC (the 3D
circularly polarized Alfven wave, `astronomix.test_setups.mhd.alfven_wave3D`
-- periodic, exact analytic solution, genuinely nonzero velocity and a
non-uniform, evolving transverse B, unlike ladder item 3's static-B,
`v=0` anisotropic-transport setup, which never lets the induction equation
do real work).

**Result: total energy does NOT close to literal round-off, but the
residual is real truncation error, not a conservation bug.** Measured (5
full wave periods, `t_end=5.0`, `C_cfl=0.4`, float64):

    N=8  (grid 16x8x8):  relative energy error 6.33e-6
    N=16 (grid 32x16x16): relative energy error 2.29e-6

Mass is conserved *exactly* (relative error `0.0`) at both resolutions --
expected for a flux-conservative FV scheme under periodic BCs. Energy is
not, but the error shrinks with resolution (consistent with the Strang
split between the gas Riemann solve and the magnetic-field update in
`_evolve_state_fv`, `astronomix/_finite_volume/_state_evolution/
evolve_state.py` -- the same split ladder item 3's own docstring already
flagged as leaving "a small, per-step, non-accumulating residual" for a
different, CR-specific reason). This is the honest baseline item 18 needs
to design against: "closes to round-off" should be read as "closes to a
small, resolution-limited residual consistent with the scheme's own order
of accuracy," not literal machine precision, whenever MHD is in play --
unlike the SN-driving module's own exact energy check
(`pytests/stratified_ism/sn_driving_energy_conservation.py`), which reaches
genuine round-off only because it is pure hydro with `fixed_timestep=True`
and no MHD.

**Real bug found and fixed along the way (general infrastructure, not
CR-specific): the shared diagnostic
`astronomix._fluid_equations.total_quantities.calculate_total_energy` (and
the opt-in snapshot field it powers, `SnapshotSettings.return_total_energy`)
never included the magnetic-energy term.** It called
`total_energy_from_primitives` (thermal + kinetic only) unconditionally,
even though a proper MHD version already existed and is used internally by
the solver's own conserved<->primitive conversion
(`total_energy_from_primitives_mhd`, `astronomix/_fluid_equations/
_equations_mhd.py`) -- so the *solver's* internal energy accounting was
always correct, only this external diagnostic helper was not. Before the
fix, on this test's `N=8` run: correct total `4.449`, the helper's own total
`1.072` -- under-reporting by ~76%, not a rounding-level gap (this IC's
dominant, uniform `B_parallel=1` background field alone carries
`0.5 * B^2 = 0.5` of energy density, more than triple the `0.15` from
thermal pressure and far above the `~0.005` from the wave's own small
kinetic perturbation -- magnetic is the *largest* single energy channel
here). Fixed by adding the `0.5 * |B|^2` term to `calculate_total_energy`
under `config.mhd` (`total_quantities.py`); this test now asserts the
helper's own output matches this test's independent
thermal+kinetic+magnetic calculation (`_total_energy` below) to guard
against a regression, and the plot's second panel shows the (now
matching) pair rather than a live discrepancy.
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
from astronomix import FINITE_VOLUME, SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import StaticIntVector

# astronomix functions
from astronomix.test_setups.mhd.alfven_wave3D import (
    CPAlfvenWave3DSettings,
    setup_cp_alfven_wave,
)
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix._fluid_equations.total_quantities import calculate_total_energy

# Tight comparisons below need 64-bit precision (same rationale as
# cr_gradient_check.py) -- the measured residuals are ~1e-6, well below
# float32's own ~1e-7 per-op noise floor accumulated over ~1e3-1e4 cells and
# many steps.
jax.config.update("jax_enable_x64", True)


def _total_energy(state, config, params, registered_variables):
    """Correct thermal + kinetic + magnetic total energy, domain-summed
    (cell-volume-weighted). Deliberately does not use
    ``calculate_total_energy`` -- see module docstring's "Real, separately-
    found bug" note."""
    rho = state[registered_variables.density_index]
    ux = state[registered_variables.velocity_index.x]
    uy = state[registered_variables.velocity_index.y]
    uz = state[registered_variables.velocity_index.z]
    p = state[registered_variables.pressure_index]
    bx = state[registered_variables.magnetic_index.x]
    by = state[registered_variables.magnetic_index.y]
    bz = state[registered_variables.magnetic_index.z]

    energy_density = (
        p / (params.gamma - 1.0)
        + 0.5 * rho * (ux**2 + uy**2 + uz**2)
        + 0.5 * (bx**2 + by**2 + bz**2)
    )
    return jnp.sum(energy_density) * config.grid_spacing**config.dimensionality


def _total_mass(state, config, registered_variables):
    rho = state[registered_variables.density_index]
    return jnp.sum(rho) * config.grid_spacing**config.dimensionality


def _run(N: int, t_end: float, c_cfl: float = 0.4):
    """Run the CP Alfven wave at grid (2N, N, N); return the initial state,
    final state, config, params, registered_variables."""
    config = SimulationConfig(solver_mode=FINITE_VOLUME, num_cells=StaticIntVector(2 * N, N, N))
    params = SimulationParams(C_cfl=c_cfl)
    settings = CPAlfvenWave3DSettings(t_end=t_end)

    initial_state, config, params = setup_cp_alfven_wave(config, params, settings)
    registered_variables = get_registered_variables(config)

    final_state = time_integration(initial_state, config, params, registered_variables)
    return initial_state, final_state, config, params, registered_variables


def test_mhd_energy_conservation(tol: float = 1e-5):
    """Baseline (CR-off) MHD energy conservation, at two resolutions, over
    five full CP-Alfven-wave periods.

    Args:
        tol: The maximum allowed relative error in total (thermal +
            kinetic + magnetic) energy, at each resolution tested.
    """
    t_end = 5.0  # settings.CPAlfvenWave3DSettings default: 5 full periods.

    rel_errs = {}
    for N in (8, 16):
        initial_state, final_state, config, params, registered_variables = _run(N, t_end)

        assert not bool(jnp.any(jnp.isnan(final_state))), f"NaN at N={N}."

        m0 = float(_total_mass(initial_state, config, registered_variables))
        mf = float(_total_mass(final_state, config, registered_variables))
        rel_mass_err = abs(mf - m0) / abs(m0)
        assert rel_mass_err < 1e-10, f"Mass not conserved at N={N}: rel. err {rel_mass_err:.3e}."

        e0 = float(_total_energy(initial_state, config, params, registered_variables))
        ef = float(_total_energy(final_state, config, params, registered_variables))
        rel_energy_err = abs(ef - e0) / abs(e0)
        rel_errs[N] = rel_energy_err
        print(f"N={N}: rel. mass err = {rel_mass_err:.3e}, rel. energy err = {rel_energy_err:.3e}")

        assert rel_energy_err < tol, (
            f"Energy conservation worse than expected at N={N}: "
            f"rel. err {rel_energy_err:.3e} >= tol {tol}."
        )

    # The residual should shrink with resolution (truncation error), not
    # sit at a fixed floor (which would instead point to a genuine
    # conservation bug) -- see module docstring.
    assert rel_errs[16] < rel_errs[8], (
        f"Energy-conservation error did not improve with resolution "
        f"(N=8: {rel_errs[8]:.3e}, N=16: {rel_errs[16]:.3e}) -- "
        "expected truncation-error behavior, possibly a real bug."
    )

    # Regression guard for the calculate_total_energy fix (see module
    # docstring): its own output must match this test's independent
    # thermal+kinetic+magnetic calculation, not just "be close".
    _, final_state_8, config_8, params_8, registered_variables_8 = _run(8, t_end)
    correct_e_8 = float(_total_energy(final_state_8, config_8, params_8, registered_variables_8))
    helper_data_8 = get_helper_data(config_8)
    helper_e_8 = float(
        calculate_total_energy(
            final_state_8, helper_data_8, params_8.gamma, 0.0, params_8, config_8, registered_variables_8
        )
    )
    rel_helper_err = abs(helper_e_8 - correct_e_8) / abs(correct_e_8)
    assert rel_helper_err < 1e-10, (
        f"calculate_total_energy no longer matches the independent thermal+kinetic+magnetic "
        f"total (rel. err {rel_helper_err:.3e}) -- has its MHD energy term regressed?"
    )

    # ---- plot ----
    fig, (ax_conv, ax_bug) = plt.subplots(1, 2, figsize=(11, 4.5))

    ns = sorted(rel_errs)
    ax_conv.loglog(ns, [rel_errs[n] for n in ns], "o-", color="tab:blue")
    ax_conv.axhline(tol, color="k", linestyle=":", label=f"tol = {tol:.0e}")
    ax_conv.set_xlabel("N (grid: 2N x N x N)")
    ax_conv.set_ylabel("relative energy error")
    ax_conv.set_title("Shrinks with resolution -> truncation error")
    ax_conv.legend()

    ax_bug.bar(
        ["independent calc.\n(thermal+kinetic+mag.)", "calculate_total_energy\n(post-fix)"],
        [correct_e_8, helper_e_8],
        color=["tab:green", "tab:blue"],
    )
    ax_bug.set_ylabel("total energy (final state, N=8)")
    ax_bug.set_title("Shared diagnostic now includes B-energy\n(matches to round-off)")

    fig.suptitle("MHD-only baseline energy check (CR off) -- ahead of CR ladder item 18")
    fig.tight_layout()
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / "mhd_energy_conservation_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_mhd_energy_conservation()
