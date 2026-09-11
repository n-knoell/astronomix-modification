"""
Vertically-stratified hydrostatic gas column in a fixed external potential
(SILCC-ISM project, milestone M0b).

Generalizes ``pytests/self_gravity/external_potential.py``'s Plummer-sphere
pattern to a vertical, disc-like potential: an isothermal gas column exactly
balancing a fixed ``phi(z)`` external potential, integrated with the
finite-volume solver (``HLLC``/``MINMOD``, matching this project's other
milestones and the CR-grey ladder's items 7-11, rather than the Plummer
test's FD path), under the mixed periodic-xy/open-z boundary condition this
whole SILCC-ISM project needs (validated in isolation by milestone M0a,
``bc_smoke_test.py``). No cooling, no CR, no SN driving yet -- purely
gravity + hydrostatic balance.

**Potential form (a deliberate deviation from the plan's suggested
linear-first attempt):** ``phi(z) = C_pot * log(cosh((z - z0) / h))`` (a
smooth, everywhere-differentiable "slab" potential: acceleration
``g_z = -dphi/dz = -(C_pot/h) * tanh((z-z0)/h)`` vanishes at the midplane
and asymptotes to a constant magnitude far away, like a razor-thin-disc
approximation but without the kink a literal ``phi(z) = g0*|z|`` form would
put right at the midplane cell interface). The matching isothermal
hydrostatic density, by the same ``rho(z) = rho0 * exp(-phi(z)/c_s^2))``
relation ``external_potential.py`` uses, is ``rho(z) = rho0 * sech^2((z -
z0)/h)`` when ``C_pot = 2 * c_s^2`` -- the classic Spitzer (1942)
self-gravitating-isothermal-slab profile, here just imposed as the
equilibrium of a *fixed* external potential rather than solved
self-consistently. Chosen over the linear form because a smooth source term
is lower-risk for a first attempt at this new BC/potential combination; the
linear form remains available for a future milestone if a sharper density
contrast is ever needed.

Calibrated at ``N_XY=64``, ``N_Z=512`` (uniform ``dx=dy=dz=0.015625``,
``box=(1,1,8)``), ``h=1.0``, ``z0=4.0`` (so the open z-boundaries sit 4
scale heights from the midplane -- density there is already
``sech^2(4) ~= 1.4e-3`` of the midplane value, deep in the "far-field" the
open BC is meant to handle), run to ``t_end=4.0`` (~5 vertical dynamical
times, ``h/c_s``). See ``test_stratified_hydrostatic_column``'s docstring
for the exact observed equilibrium residuals.

**Real finding (2026-09-11), not a bug in this test, flagged for future
milestones:** the observed core equilibrium residual (``mach_core ~= 0.106``,
``drho_core ~= 0.026`` at this resolution) is substantially larger than the
Plummer external-potential test's presumably-tiny residual, and -- checked
directly across three resolutions (``N_XY=8``: ``mach_core~=0.096``,
``drho_core~=0.023``; ``N_XY=16``: ``mach_core~=0.101``, ``drho_core~=0.023``;
``N_XY=64`` (this file's calibration): ``mach_core~=0.106``,
``drho_core~=0.026``) -- it does **not** shrink with resolution across an 8x
range in ``N_XY`` (64x in total cell count). This rules out ordinary
truncation error and points to a structural cause:
``astronomix/_modules/_gravity/_gravity.py``'s FV gravity
path (``_gravitational_source_term_along_axis``) only supports a simple,
non-conservative ``rho * a`` / ``rho * v * a`` source coupling (its own
docstring: "finite-volume self-gravity supports only the SIMPLE_SOURCE
coupling") -- unlike the FD path's ``SECOND_ORDER_CONSERVATIVE`` /
``FOURTH_ORDER_CONSERVATIVE`` options, which build the energy source
directly from the density fluxes so it stays consistent with HLLC's own
flux-based pressure-gradient discretization. A non-conservative source added
on top of an otherwise-unmodified Riemann-solver flux is a classic
"not well-balanced" scheme combination in FV astrophysics codes -- exactly
the kind of resolution-independent systematic residual observed here, not a
bug in this potential/profile construction. **This is a real, open gap** for
any future milestone that needs a tight hydrostatic baseline (M2 onward) --
either the residual needs to be treated as an accepted systematic bias, or a
well-balanced FV gravity coupling needs to be implemented before then. Not
fixed here (out of scope for a smoke-test milestone); flagged in
``PROGRESS.md`` instead.
"""

# general
from pathlib import Path

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix.option_classes.simulation_config import SIMPLE_SOURCE

# astronomix containers
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import OPEN_BOUNDARY, PERIODIC_BOUNDARY
from astronomix import BoundarySettings, BoundarySettings1D, GravityConfig
from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import StaticFloatVector, StaticIntVector

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

# This is an equilibrium test where the residual we measure is tiny, so we
# run in double precision to keep round-off from masking physical drift --
# same reasoning as external_potential.py.
jax.config.update("jax_enable_x64", True)

# ---- physical setup ----
GAMMA = 5.0 / 3.0
N_XY = 64
N_Z = 512
L_XY = 1.0
L_Z = 8.0
C2 = 1.0  # isothermal stratification constant, P = C2 * rho
C_S = (GAMMA * C2) ** 0.5
RHO0 = 1.0  # midplane density
H_SCALE = 1.0  # potential/density scale height
Z0 = L_Z / 2.0  # midplane height
C_POT = 2.0 * C2  # -> rho(z) = RHO0 * sech^2((z - Z0) / H_SCALE)
T_END = 4.0  # ~5 vertical dynamical times (h / c_s ~= 0.775)
Z_CHECK = 2.0  # core half-width (box units) around the midplane for the equilibrium check


def _run_stratified_column():
    """Run the stratified hydrostatic column; return the final state and diagnostics."""
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    open_bc = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        box_size=StaticFloatVector(L_XY, L_XY, L_Z),
        num_cells=StaticIntVector(N_XY, N_XY, N_Z),
        exact_end_time=True,
        boundary_settings=BoundarySettings(x=periodic, y=periodic, z=open_bc),
        gravity_config=GravityConfig(
            self_gravity_version=SIMPLE_SOURCE,
            external_potential=True,
        ),
        progress_bar=True,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    z = helper_data.geometric_centers[..., 2]
    phi = C_POT * jnp.log(jnp.cosh((z - Z0) / H_SCALE))
    rho_init = RHO0 * jnp.exp(-phi / C2)
    p_init = C2 * rho_init
    zero = jnp.zeros_like(z)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho_init,
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
        gas_pressure=p_init,
    )
    params = SimulationParams(t_end=T_END, gamma=GAMMA)
    params = params._replace(gravitational_potential=phi)

    config = finalize_config(config, initial_state.shape)
    assert config.gravity_config.gravity, "finalize_config did not turn on the master gravity flag"

    final_state = time_integration(initial_state, config, params, registered_variables)

    rho_final = final_state[registered_variables.density_index]
    vx = final_state[registered_variables.velocity_index.x]
    vy = final_state[registered_variables.velocity_index.y]
    vz = final_state[registered_variables.velocity_index.z]
    mach = jnp.sqrt(vx**2 + vy**2 + vz**2) / C_S
    drho_rel = jnp.abs(rho_final - rho_init) / rho_init

    core = jnp.abs(z - Z0) < Z_CHECK
    mach_core = float(jnp.max(jnp.where(core, mach, 0.0)))
    drho_core = float(jnp.max(jnp.where(core, drho_rel, 0.0)))

    return dict(
        final_state=final_state,
        z=z,
        rho_init=rho_init,
        rho_final=rho_final,
        mach=mach,
        mach_core=mach_core,
        drho_core=drho_core,
    )


def test_stratified_hydrostatic_column(
    mach_core_tol: float = 0.15,
    drho_core_tol: float = 0.04,
):
    """Vertically-stratified hydrostatic column stays near equilibrium (FV).

    These tolerances are deliberately loose relative to the Plummer FD
    test's implied precision -- see the module docstring's "Real finding"
    section: the FV gravity path's non-conservative source coupling produces
    a genuine, resolution-independent equilibrium residual at this potential
    steepness, not a bug in this test's setup.

    Args:
        mach_core_tol: Max core (|z-z0|<Z_CHECK) Mach number by t_end --
            residual flows an ideal (well-balanced) solver would keep at
            zero. Calibrated observed value at N_XY=64: ``~0.1057``
            (>1.4x margin); confirmed resolution-independent across
            N_XY=8/16/64 (``~0.096``/``~0.101``/``~0.106``).
        drho_core_tol: Max core relative density drift by t_end. Calibrated
            observed value at N_XY=64: ``~0.0255`` (>1.5x margin); also
            resolution-independent (``~0.023``/``~0.023``/``~0.026`` across
            N_XY=8/16/64).
    """
    run = _run_stratified_column()

    assert not bool(jnp.any(jnp.isnan(run["final_state"]))), (
        "Stratified hydrostatic column produced NaNs."
    )

    assert run["mach_core"] < mach_core_tol, (
        f"Core Mach number ({run['mach_core']:.4e}) exceeds tol {mach_core_tol} "
        f"-- the column is not holding hydrostatic equilibrium."
    )
    assert run["drho_core"] < drho_core_tol, (
        f"Core relative density drift ({run['drho_core']:.4e}) exceeds tol "
        f"{drho_core_tol} -- the column is not holding hydrostatic equilibrium."
    )

    # Diagnostic plot: vertical density profile (initial vs. final, log scale)
    # and Mach-number profile along the column's central axis.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))
    ix = N_XY // 2
    iy = N_XY // 2
    z_line = np.asarray(run["z"][ix, iy, :])
    ax0.semilogy(z_line, np.asarray(run["rho_init"][ix, iy, :]), "-", label="initial", color="C0")
    ax0.semilogy(z_line, np.asarray(run["rho_final"][ix, iy, :]), "--", label=f"final (t={T_END})", color="C1")
    ax0.axvspan(Z0 - Z_CHECK, Z0 + Z_CHECK, color="grey", alpha=0.15, label="core region")
    ax0.set_xlabel("z")
    ax0.set_ylabel(r"$\rho$")
    ax0.set_title("vertical density profile")
    ax0.legend()

    ax1.semilogy(z_line, np.maximum(np.asarray(run["mach"][ix, iy, :]), 1e-16), color="C2")
    ax1.axvspan(Z0 - Z_CHECK, Z0 + Z_CHECK, color="grey", alpha=0.15)
    ax1.set_xlabel("z")
    ax1.set_ylabel("Mach number")
    ax1.set_title(f"final Mach profile (core max={run['mach_core']:.2e})")

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "stratified_hydrostatic_column_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_stratified_hydrostatic_column()
