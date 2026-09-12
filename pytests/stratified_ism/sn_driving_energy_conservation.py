"""
Episodic supernova-driving energy-conservation pytest (SILCC-ISM project
milestone M3; see ``astronomix/_modules/_cosmic_rays_grey/DESIGN.md``'s
"SILCC-ISM project" section for the multi-milestone roadmap).

Validates ``astronomix._modules._sn_driving.sn_driving._inject_supernovae`` in
isolation: a uniform 3D Cartesian **periodic** box, no gravity, cooling or
turbulent forcing, with episodic supernova driving as the only active
"physics module" besides plain hydro (+ the CR-grey advection/pressure-
coupling terms, needed so the direct CR-energy dump has somewhere real to go).
Each supernova deposits a smoothly-tapered spherical energy bump (reusing the
same ``tanh`` taper and Sedov-style renormalization as ladder items 7/11) at
a uniformly random site and time: a thermal part (added to gas pressure) plus,
when the CR-grey model is active, a direct ``sn_cr_fraction`` of the total
straight into ``e_cr`` (Girichidis et al. 2016's convention -- see gap #4 /
design decision #2 in DESIGN.md's SILCC-ISM section).

**Test design: an *exact* (not statistical) energy-conservation check**, via
two facts specific to this setup:

1. With no gravity, cooling, or open boundaries, a conservative finite-volume
   scheme's domain-integrated (thermal + kinetic + CR) energy can only change
   at a supernova deposit -- so ``E_total(t_end) - E_total(0)`` must equal
   exactly ``N_triggers * sn_energy``, independent of the (possibly quite
   complicated) subsequent shock/turbulence evolution.
2. ``N_triggers`` itself can be known exactly, not just statistically, because
   the per-step stochastic trigger is the *only* consumer of the PRNG key in
   this configuration (turbulent forcing, the other per-step key consumer, is
   off) and ``config.fixed_timestep=True`` makes every step's ``dt`` -- and
   hence the trigger probability ``sn_rate * dt`` -- identical and known in
   advance. This lets the test replay the exact same
   ``jax.random.split`` / ``jax.random.bernoulli`` sequence *outside* the
   simulation (see ``_replica_trigger_count``) and know precisely how many
   times the real run triggered, without touching the simulation's internals
   or adding any test-only hook to the module itself. (Mirrors the
   ``fixed_timestep`` + one-internal-step-per-call technique ladder item 9
   used to instrument the per-step ``e_cr`` budget -- see PROGRESS.md's
   2026-09-09 entry.)

**Real bug found and fixed while building this test:** the first version
computed the tanh-taper weight (and its normalizing sum) over the full
*ghost-padded* array ``_iteration_level_updates`` operates on. For a site
drawn near a periodic edge, that double-counts real domain volume (a nearby
ghost cell mirrors a real interior cell on the far side, so both receive
weight) in the normalization, and then the ghost cells' share of the deposit
is silently discarded the next time the boundary handler refreshes them from
the interior -- losing a fraction of ``sn_energy`` every time a site landed
near an edge. Calibrated mismatch before the fix: ``~0.125`` out of ``0.45``
total injected (over 9 triggers) -- a large, resolution-independent bias, not
noise. Fixed by restricting the weight (both the normalization sum and the
actual ``.add()`` deposit) to the interior cells only, relying on the
standard boundary handler to correctly refresh the ghost cells afterward
(exact to ``~3e-10`` relative after the fix, see calibrated numbers below).
This is a new failure mode relative to every earlier point-injection ladder
item (7/10/11), which all use *open* boundaries with the explosion safely
contained away from the domain edge -- periodic-box point injection at
arbitrary random sites is genuinely new territory for this codebase.
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

# This is an exact (replica-vs-simulation) energy-conservation check -- run
# in double precision, same reasoning as this directory's other tests
# (bc_smoke_test.py / stratified_hydrostatic_column.py): float32 round-off
# accumulation over hundreds of fixed-timestep steps is large enough to blow
# the tolerance, and can even perturb which step the exact_end_time clamp
# lands on.
jax.config.update("jax_enable_x64", True)

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD, PERIODIC_BOUNDARY
from astronomix import BoundarySettings, BoundarySettings1D
from astronomix import SimulationConfig, SimulationParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import CosmicRayGreyConfig
from astronomix._modules._sn_driving.sn_driving_options import SNDrivingConfig, SNDrivingParams

# ---- physical setup ----
GAMMA = 5.0 / 3.0
NUM_CELLS = 128
RHO_AMBIENT = 1.0
P_AMBIENT = 1.0
SN_ENERGY = 0.05
SN_CR_FRACTION = 0.1
SN_INJECTION_RADIUS = 0.05
SN_SMOOTH_CELLS = 2.0

# ---- driving / timestepping (fixed dt is required for the exact-replica
# check -- see module docstring point 2) ----
NUM_TIMESTEPS = 400
T_END = 0.05
SN_RATE = 100.0  # expected count = SN_RATE * T_END = 5 (observed 9 at the default seed)

_PERIODIC_BOX = BoundarySettings(
    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def _build_config(grey_cosmic_rays: bool) -> SimulationConfig:
    return SimulationConfig(
        progress_bar=True,
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        boundary_settings=_PERIODIC_BOX,
        fixed_timestep=True,
        num_timesteps=NUM_TIMESTEPS,
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=grey_cosmic_rays),
        sn_driving_config=SNDrivingConfig(sn_driving=True),
    )


def _sn_params() -> SNDrivingParams:
    return SNDrivingParams(
        sn_rate=SN_RATE,
        sn_energy=SN_ENERGY,
        sn_cr_fraction=SN_CR_FRACTION,
        sn_injection_radius=SN_INJECTION_RADIUS,
        sn_smooth_cells=SN_SMOOTH_CELLS,
    )


def _run_sn_driving(grey_cosmic_rays: bool):
    """Run one uniform periodic box with episodic SN driving to T_END."""
    config = _build_config(grey_cosmic_rays)
    registered_variables = get_registered_variables(config)

    shape = (NUM_CELLS, NUM_CELLS, NUM_CELLS)
    density = jnp.ones(shape) * RHO_AMBIENT
    zeros = jnp.zeros(shape)
    gas_pressure = jnp.ones(shape) * P_AMBIENT

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
    params = SimulationParams(t_end=T_END, gamma=GAMMA, sn_driving_params=_sn_params())

    final_state = time_integration(initial_state, config, params, registered_variables)

    cell_volume = (1.0 / NUM_CELLS) ** 3
    rho = final_state[registered_variables.density_index]
    vx = final_state[registered_variables.velocity_index.x]
    vy = final_state[registered_variables.velocity_index.y]
    vz = final_state[registered_variables.velocity_index.z]
    p_gas = final_state[registered_variables.pressure_index]

    E_thermal = float(jnp.sum(p_gas / (GAMMA - 1.0)) * cell_volume)
    E_kinetic = float(jnp.sum(0.5 * rho * (vx**2 + vy**2 + vz**2)) * cell_volume)
    if registered_variables.cosmic_ray_e_active:
        e_cr = final_state[registered_variables.cosmic_ray_e_index]
        E_cr = float(jnp.sum(e_cr) * cell_volume)
    else:
        e_cr = None
        E_cr = 0.0

    return dict(
        final_state=final_state,
        config=config,
        registered_variables=registered_variables,
        p_gas=p_gas,
        e_cr=e_cr,
        E_thermal=E_thermal,
        E_kinetic=E_kinetic,
        E_cr=E_cr,
        E_total=E_thermal + E_kinetic + E_cr,
    )


def _replica_trigger_count(config: SimulationConfig) -> int:
    """Independently replay the exact PRNG sequence _inject_supernovae uses.

    Valid because, in this configuration, the SN-driving trigger is the only
    per-step consumer of ``LoopState.key`` (turbulent forcing, the only other
    one in this codebase, is off) and ``fixed_timestep=True`` makes every
    step's ``dt`` -- and hence the trigger probability -- identical and known
    without running the simulation. See the module docstring's point 2.
    """
    key = jax.random.key(config.random_seed)
    dt = T_END / NUM_TIMESTEPS
    p_trigger = min(max(SN_RATE * dt, 0.0), 1.0)
    n_triggers = 0
    for _ in range(NUM_TIMESTEPS):
        key, trigger_key, _pos_key = jax.random.split(key, 3)
        n_triggers += int(jax.random.bernoulli(trigger_key, p_trigger))
    return n_triggers


def test_sn_driving_energy_conservation(
    energy_conservation_tol: float = 1e-6,
    cr_fraction_band: tuple = (0.01, 0.3),
):
    """Episodic SN driving: exact total-energy-conservation check.

    Args:
        energy_conservation_tol: Max relative error (normalized by the total
            injected energy ``N_triggers * SN_ENERGY``) on
            ``E_total(t_end) - E_total(0) == N_triggers * SN_ENERGY``, checked
            independently for the CR-active and CR-inactive runs. Calibrated
            observed error ~7e-10 (CR-active), ~8e-15 (CR-inactive) -- a
            >1000x margin either way.
        cr_fraction_band: Sanity band for the CR-active run's final
            ``E_cr / (N_triggers * SN_ENERGY)``. Not expected to equal
            ``SN_CR_FRACTION`` exactly -- once deposited, e_cr exchanges
            energy with the gas via the (always-active) adiabatic-work
            coupling verified in ladder item 1/2, so some drift away from the
            raw injected fraction is expected physics, not a bug. Calibrated
            observed ~0.096 (vs. SN_CR_FRACTION=0.1).
    """
    n_triggers = _replica_trigger_count(_build_config(grey_cosmic_rays=True))
    assert n_triggers > 0, (
        "Replica predicts zero supernovae triggered over T_END -- increase "
        "SN_RATE or T_END so this test actually exercises the injection path."
    )
    total_injected = n_triggers * SN_ENERGY

    cr_on = _run_sn_driving(grey_cosmic_rays=True)
    cr_off = _run_sn_driving(grey_cosmic_rays=False)

    for name, run in (("CR-active", cr_on), ("CR-inactive", cr_off)):
        assert not bool(jnp.any(jnp.isnan(run["final_state"]))), (
            f"SN-driving run ({name}) produced NaNs."
        )

    E_total_initial = (P_AMBIENT / (GAMMA - 1.0)) * 1.0**3  # box_size = 1.0 (default)

    for name, run in (("CR-active", cr_on), ("CR-inactive", cr_off)):
        delta_E = run["E_total"] - E_total_initial
        rel_err = abs(delta_E - total_injected) / total_injected
        assert rel_err < energy_conservation_tol, (
            f"{name} run's total energy change ({delta_E:.6e}) does not match "
            f"the replica-predicted N_triggers ({n_triggers}) * SN_ENERGY "
            f"({total_injected:.6e}) -- rel. err {rel_err:.4e} >= tol "
            f"{energy_conservation_tol}."
        )

    # CR-inactive run: the cr_fraction-forced-to-0 fallback should mean every
    # supernova is 100% thermal -- E_cr is identically 0.0, not just small.
    assert cr_off["E_cr"] == 0.0, (
        f"CR-inactive run injected nonzero CR energy: E_cr={cr_off['E_cr']:.6e} "
        f"-- the sn_cr_fraction-forced-to-0 fallback is not working."
    )

    # CR-active run: E_cr should be a clearly nonzero, physically bounded
    # fraction of the total injected energy (see docstring for why this isn't
    # an exact SN_CR_FRACTION identity).
    cr_fraction = cr_on["E_cr"] / total_injected
    assert cr_fraction_band[0] < cr_fraction < cr_fraction_band[1], (
        f"CR-active run's CR energy fraction ({cr_fraction:.4f}) is outside "
        f"the expected {cr_fraction_band} band -- either injection is not "
        f"happening, or is wildly over/under-injecting relative to "
        f"SN_CR_FRACTION={SN_CR_FRACTION}."
    )

    # Diagnostic plot: energy-budget bar chart (left) and a mid-plane e_cr
    # slice showing the randomly-placed injection sites (right).
    fig, (ax_bars, ax_slice) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["thermal", "kinetic", "CR"]
    initial_vals = [E_total_initial, 0.0, 0.0]
    final_vals = [cr_on["E_thermal"], cr_on["E_kinetic"], cr_on["E_cr"]]
    x = jnp.arange(len(labels))
    width = 0.35
    ax_bars.bar(x - width / 2, initial_vals, width, label="t=0", color="C0")
    ax_bars.bar(x + width / 2, final_vals, width, label=f"t={T_END}", color="C1")
    ax_bars.set_xticks(x, labels)
    ax_bars.set_ylabel("energy")
    ax_bars.set_title(f"Energy budget, {n_triggers} supernovae triggered")
    ax_bars.legend()

    mid = NUM_CELLS // 2
    im = ax_slice.imshow(
        cr_on["e_cr"][:, :, mid].T, origin="lower",
        extent=(0, 1, 0, 1), cmap="inferno",
    )
    ax_slice.set_xlabel("x")
    ax_slice.set_ylabel("y")
    ax_slice.set_title(f"e_cr, z-midplane slice, t={T_END}")
    fig.colorbar(im, ax=ax_slice, label="e_cr")

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "sn_driving_energy_conservation_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_sn_driving_energy_conservation()
