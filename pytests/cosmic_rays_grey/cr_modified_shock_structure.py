"""
CR-modified shock structure pytest (Phase B ladder item 9, "1D steady CR-driven
flow / CR-modified shock structure").

**Design history, in order** (see
``astronomix/_modules/_cosmic_rays_grey/PROGRESS.md``'s 2026-08-27 and
2026-09-09 entries for the full account; this docstring summarizes the
resolution):

1. The item as originally attempted used a shock held exactly stationary in
   the box (lab) frame, with ``diffusive_shock_acceleration`` injecting into
   the same grid cell every step for the whole run. This produced two real
   findings (not a test-design bug): **Finding 1**, ``reduced_streaming_speed``
   shares the Riemann-solver wave-speed bound with the gas
   (``hll.py``'s ``max(c_gas, grey_cr_fast_speed(...))``), which degrades
   shock-capturing whenever it exceeds local gas signal speeds, independent of
   whether ``diffusive_relaxation`` is on; and **Finding 2**, sustained,
   repeated DSA injection at a cell that never advects away grows the local
   CR pressure without bound. Both are documented in DESIGN.md's "Open
   questions" and are **not** fixed here -- Finding 1 in particular is a real
   change to shared FV code, out of scope to attempt unprompted.
2. Root mechanism for Finding 2 (found 2026-09-09): a *stationary* shock cell
   never releases the compressive ``-P_cr * div(v)`` term
   (``cr_grey_sources.cr_adiabatic_work_source``) the way a fluid parcel
   crossing a moving shock once would (that finite-transit-time release is
   exactly what ladder item 2 verified gives the correct
   ``P_cr ~ rho^gamma_cr`` invariant) -- confirmed by DSA theory itself, which
   describes a shock *steady in its own comoving frame*, not one pinned
   motionless in the lab frame while CR energy accumulates in the same cell
   forever. **Resolved by redesigning the test around a genuinely moving
   shock** (confirmed with the user, 2026-09-09): a Rankine-Hugoniot "piston"
   IC launched at a nonzero, constant velocity, so each grid cell only hosts
   the actively-injecting shock surface for the brief time the shock front is
   passing through it -- the same "always advancing into fresh gas" pattern
   ``cr_sedov_taylor.py``/``cr_dsa_mach_dependence.py`` already rely on.
   Verified directly (ad hoc script, not committed): the fitted shock
   velocity from several snapshots matches the nominal value to <0.1%, and
   peak ``e_cr`` at the shock saturates to a bounded value instead of growing
   without bound -- see :func:`test_cr_dsa_shock_jump`'s boundedness check
   below, which turns this verification into a permanent regression guard.
3. Calibrating the redesign surfaced a *further*, previously-unappreciated
   interaction with Finding 1 (2026-09-09): the reference precursor ODE
   (:func:`astronomix.test_setups.reference_solutions.
   cr_modified_shock_structure.cr_precursor_ode_rhs`) only holds once
   ``F_cr``'s relaxation rate ``nu = reduced_streaming_speed^2 /
   diffusion_coefficient`` is fast compared to the precursor's own advective
   formation rate ``~ v_shock / precursor_width`` -- and working through the
   two rate expressions shows this requires ``reduced_streaming_speed``
   several times the shock velocity *regardless of* ``diffusion_coefficient``
   (the ratio ``nu / omega = (gamma_cr - 1) * (reduced_streaming_speed /
   v_shock)^2`` doesn't depend on it). That is almost exactly the regime
   Finding 1 already showed causes severe HLL wave-speed-inflation dissipation
   at a resolved shock. Measured directly: raising
   ``reduced_streaming_speed`` from 4x to 8x the shock velocity improves the
   precursor-ODE match (as expected) but *worsens* the subshock jump match
   (up from ~25-40% to up to 150% on some fields) and further degrades the
   measured shock Mach; at 16x the shock velocity, the shock is degraded
   below ``dsa_mach_min`` and disappears entirely by the run's end. **A
   single run cannot quantitatively validate both reference formulas at
   once** without first fixing Finding 1. Resolved (confirmed with the user):
   split into the two independent tests below, each run at the
   ``reduced_streaming_speed`` its own reference formula actually needs,
   rather than forcing one compromise value onto both -- the same
   "independent layers" pattern ladder item 8 already used.

:func:`test_cr_dsa_shock_jump` (**Test A**): ``diffusive_relaxation`` off (DSA
injection only, no diffusive precursor), ``reduced_streaming_speed`` close to
the local gas sound speed (Finding-1-safe). Validates
:func:`modified_rankine_hugoniot_with_cr_injection` against the simulation's
own converged post-shock state, and is also this module's regression guard for
Finding 2 (peak ``e_cr`` must plateau, not grow unboundedly).

:func:`test_cr_precursor_ode` (**Test B**): ``diffusive_relaxation`` on,
``reduced_streaming_speed`` well above the shock velocity (deep quasi-steady,
per point 3 above -- accepting a weaker, Finding-1-degraded shock as the cost
of a resolvable precursor). Validates :func:`cr_precursor_ode_rhs` against the
simulation's own local gradients in the precursor, after confirming the two
first integrals (``mass_flux``, ``momentum_flux``) the ODE relies on are
genuinely close to constant across the sampled precursor window.

**Shared construction:** both tests launch the *same* Rankine-Hugoniot
"piston" shock -- an exact steady-shock solution (post-shock state given by
the standard ideal-gas RH jump at the chosen shock velocity/Mach number)
placed near the left (open) boundary and left to propagate right into a
quiescent ambient medium. Because the IC already satisfies the RH jump for
the chosen velocity, the shock genuinely propagates at that constant velocity
(verified per-test) rather than needing an inflow/piston boundary condition.
The shock's own velocity is not known a priori by the solver -- it's fit
after the fact from ``find_shocks_pfrommer``'s reported shock position across
several snapshots, then subtracted from the lab-frame velocity to boost the
sampled profile into the shock's own frame before comparing to the
(shock-frame) reference formulas.

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md.
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

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    BoundarySettings1D,
    OPEN_BOUNDARY,
    SnapshotSettings,
    finalize_config,
)

# astronomix functions
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix.test_setups.reference_solutions.cr_modified_shock_structure import (
    cr_precursor_ode_rhs,
    modified_rankine_hugoniot_with_cr_injection,
)

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# ---- shared physical setup ----
GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
RHO1 = 1.0
P1 = 1.0 / GAMMA  # pre-shock sound speed c1 = 1.0 -- see module docstring
V_SHOCK = 2.0  # nominal shock velocity (M0 = 2.0)
DSA_EFFICIENCY = 0.1
DSA_MACH_MIN = 1.3
N_CELLS = 1000
BOX_SIZE = 2.5
X0 = 0.3
T_END = 0.8
NUM_SNAPSHOTS = 8
N_FIT = 4  # number of (late) snapshots used to fit the shock velocity

# Rankine-Hugoniot post-shock state for the piston IC (exact for the plain
# ideal-gas jump; the CR-free ambient starts with zero e_cr/F_cr on both
# sides, so this is exact for t=0 regardless of the CR-grey config used).
_C1 = np.sqrt(GAMMA * P1 / RHO1)
_M0 = V_SHOCK / _C1
_R = (GAMMA + 1.0) * _M0**2 / ((GAMMA - 1.0) * _M0**2 + 2.0)
RHO2 = _R * RHO1
U2 = V_SHOCK * (1.0 - 1.0 / _R)
P2 = P1 * (2.0 * GAMMA * _M0**2 - (GAMMA - 1.0)) / (GAMMA + 1.0)


def _run_moving_shock(reduced_streaming_speed, diffusive_relaxation, diffusion_coefficient=1.0):
    """Build and run the shared moving-shock IC; return per-snapshot data."""
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=N_CELLS,
        box_size=BOX_SIZE,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True,
            diffusive_shock_acceleration=True,
            diffusive_relaxation=diffusive_relaxation,
        ),
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(return_states=True),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = np.asarray(helper_data.geometric_centers)

    left = x < X0
    rho = jnp.where(left, RHO2, RHO1)
    u = jnp.where(left, U2, 0.0)
    p = jnp.where(left, P2, P1)

    primitive_state = jnp.zeros((registered_variables.num_vars, N_CELLS))
    primitive_state = primitive_state.at[registered_variables.density_index].set(rho)
    primitive_state = primitive_state.at[registered_variables.velocity_index].set(u)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(p)

    config = finalize_config(config, primitive_state.shape)
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR,
            reduced_streaming_speed=reduced_streaming_speed,
            diffusion_coefficient=diffusion_coefficient,
            dsa_efficiency=DSA_EFFICIENCY,
            dsa_mach_min=DSA_MACH_MIN,
        ),
    )

    snapshot_data = time_integration(primitive_state, config, params, registered_variables)
    return snapshot_data, x, config, registered_variables, helper_data, params


def _shock_diagnostics(states, x, config, registered_variables, helper_data):
    """Per-snapshot shock position/Mach/peak-e_cr, via find_shocks_pfrommer."""
    shock_positions, mach_list, max_e_cr = [], [], []
    for i in range(states.shape[0]):
        sf = find_shocks_pfrommer(
            jnp.asarray(states[i]), config, registered_variables, helper_data,
            mach_min=DSA_MACH_MIN,
        )
        mach = np.asarray(sf.mach_numbers)
        idx = int(np.argmax(mach)) if float(mach.max()) > 0.0 else -1
        shock_positions.append(x[idx] if idx >= 0 else np.nan)
        mach_list.append(float(mach[idx]) if idx >= 0 else np.nan)
        max_e_cr.append(float(states[i, registered_variables.cosmic_ray_e_index].max()))
    return np.array(shock_positions), np.array(mach_list), np.array(max_e_cr)


def _fit_shock_velocity(times, shock_positions, n_fit=N_FIT):
    """Least-squares shock velocity from the last ``n_fit`` snapshots (skips
    the initial transient near t=0, before the moving RH jump has settled)."""
    A = np.vstack([times[-n_fit:], np.ones(n_fit)]).T
    v_fit, x_fit0 = np.linalg.lstsq(A, shock_positions[-n_fit:], rcond=None)[0]
    return float(v_fit), float(x_fit0)


def test_cr_dsa_shock_jump(
    density_tol: float = 0.03,
    velocity_tol: float = 0.03,
    pressure_tol: float = 0.03,
    f_cr_tol: float = 0.35,
    velocity_fit_tol: float = 0.01,
    margin_cells: int = 3,
):
    """Test A: subshock Rankine-Hugoniot-with-injection cross-check, plus the
    Finding-2 boundedness regression guard (see module docstring).

    Args:
        density_tol/velocity_tol/pressure_tol: Maximum relative error on the
            predicted vs. actual post-shock (density, velocity, pressure).
            Calibrated observed errors ~0.1-1.1% (ad hoc script, not
            committed) once the pre/post-shock sampling cells are moved
            ``margin_cells`` past the shock-finder's own ``shock_zones``
            boundary (closer cells still carry residual HLL numerical
            smearing from the discontinuity itself) -- 3% leaves a
            comfortable margin.
        f_cr_tol: Maximum relative error on the predicted vs. actual
            post-shock ``F_cr``. Calibrated observed error ~21% -- looser
            than the other three fields because, without
            ``diffusive_relaxation``, ``F_cr`` has no mechanism pinning it to
            a fixed post-shock plateau the way density/velocity/pressure
            (governed by strong RH conservation) are; the reference formula
            is exact only as an instantaneous jump condition right at the
            subshock, and some residual mismatch from finite cell width is
            expected. 35% leaves margin over the calibrated value.
        velocity_fit_tol: Maximum relative error on the fitted vs. nominal
            shock velocity -- confirms the RH "piston" IC is genuinely
            propagating at the intended constant velocity, not drifting.
        margin_cells: Cells to step past ``shock_zones`` before sampling the
            pre-/post-shock reference state (see density_tol docstring).
    """
    snapshot_data, x, config, registered_variables, helper_data, params = _run_moving_shock(
        reduced_streaming_speed=1.0, diffusive_relaxation=False,
    )
    times = np.asarray(snapshot_data.time_points)
    states = np.asarray(snapshot_data.states)
    assert not bool(np.any(np.isnan(states))), "CR-DSA moving shock (Test A) produced NaNs."

    shock_positions, mach_list, max_e_cr = _shock_diagnostics(
        states, x, config, registered_variables, helper_data
    )
    v_fit, x_fit0 = _fit_shock_velocity(times, shock_positions)
    v_rel_err = abs(v_fit - V_SHOCK) / V_SHOCK
    assert v_rel_err < velocity_fit_tol, (
        f"Fitted shock velocity ({v_fit:.5f}) deviates from the nominal "
        f"piston velocity ({V_SHOCK}) by {v_rel_err:.4e} -- the RH \"piston\" "
        f"IC is not propagating at constant velocity as designed."
    )

    # Finding-2 regression guard: peak e_cr must plateau, not grow without
    # bound, now that the shock genuinely advects through fresh cells instead
    # of sitting still. Calibrated: saturates within the first snapshot and
    # stays flat to <1% of its own value for the remainder of the run.
    late = max_e_cr[NUM_SNAPSHOTS // 2:]
    plateau_spread = (late.max() - late.min()) / late.mean()
    assert plateau_spread < 0.05, (
        f"Peak e_cr is still drifting in the second half of the run "
        f"(spread {plateau_spread:.4e} of its mean) instead of having "
        f"plateaued -- possible reintroduction of Finding 2's unbounded "
        f"growth at a shock that isn't genuinely advecting."
    )

    # Subshock RH-jump-with-injection cross-check, at the final snapshot.
    st_b = jnp.asarray(states[-1])
    sf_b = find_shocks_pfrommer(st_b, config, registered_variables, helper_data, mach_min=DSA_MACH_MIN)
    mach_b = np.asarray(sf_b.mach_numbers)
    shock_idx = int(np.argmax(mach_b))
    zone = np.asarray(sf_b.shock_zones)
    thermal_flux = np.asarray(sf_b.thermal_energy_flux)

    idx0 = shock_idx + 1
    while zone[idx0]:
        idx0 += 1
    idx0 += margin_cells
    idx2 = shock_idx - 1
    while idx2 >= 0 and zone[idx2]:
        idx2 -= 1
    idx2 -= margin_cells

    rho_b = states[-1, registered_variables.density_index]
    u_b = states[-1, registered_variables.velocity_index]
    p_gas_b = states[-1, registered_variables.pressure_index]
    e_cr_b = states[-1, registered_variables.cosmic_ray_e_index]
    p_cr_b = (GAMMA_CR - 1.0) * e_cr_b
    f_cr_b = states[-1, registered_variables.cosmic_ray_flux_index]

    rho0, u0, p_gas0 = float(rho_b[idx0]), float(u_b[idx0] - v_fit), float(p_gas_b[idx0])
    p_cr0, f_cr0 = float(p_cr_b[idx0]), float(f_cr_b[idx0])
    injected_energy_flux = float(DSA_EFFICIENCY * thermal_flux[shock_idx])
    assert injected_energy_flux > 0.0, (
        "No DSA injection detected at the final snapshot's shock cell -- the "
        "cross-check below isn't exercising anything."
    )

    rho2_pred, u2_pred, p_gas2_pred, f_cr2_pred = modified_rankine_hugoniot_with_cr_injection(
        rho0, u0, p_gas0, p_cr0, f_cr0, GAMMA, GAMMA_CR, injected_energy_flux,
    )
    rho2_act = float(rho_b[idx2])
    u2_act = float(u_b[idx2] - v_fit)
    p_gas2_act = float(p_gas_b[idx2])
    f_cr2_act = float(f_cr_b[idx2])

    checks = [
        ("density", rho2_pred, rho2_act, density_tol),
        ("velocity", u2_pred, u2_act, velocity_tol),
        ("pressure", p_gas2_pred, p_gas2_act, pressure_tol),
        ("F_cr", f_cr2_pred, f_cr2_act, f_cr_tol),
    ]
    for name, pred, act, tol in checks:
        rel_err = abs(pred - act) / max(abs(act), 1e-12)
        assert rel_err < tol, (
            f"Post-shock {name}: predicted {pred:.5f} vs. actual {act:.5f} "
            f"(rel. err {rel_err:.4e} >= tol {tol}) -- "
            f"modified_rankine_hugoniot_with_cr_injection does not match the "
            f"simulation's own converged post-shock state."
        )

    # Diagnostic plot: gas/CR profile in the shock-comoving coordinate at the
    # final snapshot, with the predicted post-shock state marked.
    xi = x - (x_fit0 + v_fit * times[-1])
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, field, name, pred_val in [
        (axes[0], rho_b, "Density", rho2_pred),
        (axes[1], u_b - v_fit, "Shock-frame velocity", u2_pred),
        (axes[2], p_gas_b, "Gas pressure", p_gas2_pred),
        (axes[3], f_cr_b, r"$F_{\rm cr}$", f_cr2_pred),
    ]:
        ax.plot(xi, field, color="C0")
        ax.axhline(pred_val, color="black", ls="--", lw=1, label="RH-jump-with-injection prediction")
        ax.axvline(0.0, color="grey", ls=":", lw=1)
        ax.set_xlim(-0.05, 0.05)
        ax.set_xlabel(r"$\xi = x - x_{\rm shock}(t)$")
        ax.set_title(name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_modified_shock_structure_jump_test.svg")
    plt.close(fig)


def test_cr_precursor_ode(
    velocity_fit_tol: float = 0.01,
    mass_flux_tol: float = 0.02,
    momentum_flux_tol: float = 0.02,
    velocity_ode_tol: float = 0.8,
    p_cr_ode_tol: float = 0.4,
    f_cr_ode_tol: float = 0.4,
    reduced_streaming_speed: float = 8.0,
    diffusion_coefficient: float = 0.2,
    precursor_window_cells: int = 150,
):
    """Test B: precursor ODE local cross-check (diffusive_relaxation on).

    ``reduced_streaming_speed`` is set well above the shock velocity here
    (unlike Test A) -- required for the quasi-steady precursor ODE to hold at
    all (see module docstring point 3), at the cost of a Finding-1-degraded
    subshock this test does not otherwise rely on.

    Args:
        velocity_fit_tol: Same purpose as in Test A.
        mass_flux_tol/momentum_flux_tol: Maximum relative spread (std over
            mean) of the two conserved first integrals
            (``mass_flux = rho * u_shockframe``, ``momentum_flux = rho *
            u_shockframe^2 + p_gas + p_cr``) across the sampled precursor
            window -- these must be genuinely close to constant for
            ``cr_precursor_ode_rhs`` (which assumes them as position-
            independent first integrals) to apply at all. Calibrated
            observed spread <0.5%; 2% leaves a comfortable margin.
        velocity_ode_tol/p_cr_ode_tol/f_cr_ode_tol: Maximum median relative
            error between the simulation's own finite-difference derivatives
            (``du/dx``, ``dP_cr/dx``, ``dF_cr/dx``) in the precursor and
            ``cr_precursor_ode_rhs``'s prediction at the same local state.
            Calibrated observed medians ~41% / 22% / 20% respectively (ad hoc
            script, not committed) at ``reduced_streaming_speed=8.0`` --
            looser than a typical formula cross-check in this module because
            Finding 1 (see module docstring) directly limits how deep into
            the quasi-steady regime this run can go before the shock itself
            degrades; ``velocity_ode_tol`` is loosest since ``du/dx`` is the
            most indirectly-determined of the three (solved from a ratio
            with cancellation, see ``cr_precursor_ode_rhs``'s derivation).
        precursor_window_cells: Number of cells upstream of the shock zone to
            include in the sampled precursor window.
    """
    snapshot_data, x, config, registered_variables, helper_data, params = _run_moving_shock(
        reduced_streaming_speed=reduced_streaming_speed,
        diffusive_relaxation=True,
        diffusion_coefficient=diffusion_coefficient,
    )
    times = np.asarray(snapshot_data.time_points)
    states = np.asarray(snapshot_data.states)
    assert not bool(np.any(np.isnan(states))), "CR-DSA moving shock (Test B) produced NaNs."

    shock_positions, mach_list, max_e_cr = _shock_diagnostics(
        states, x, config, registered_variables, helper_data
    )
    assert bool(np.all(mach_list[1:] > DSA_MACH_MIN)), (
        "The shock weakened below dsa_mach_min at this reduced_streaming_speed "
        "-- Finding 1's wave-speed-inflation dissipation has degraded the "
        "shock entirely (see module docstring point 3); reduce "
        "reduced_streaming_speed/diffusion_coefficient."
    )
    v_fit, x_fit0 = _fit_shock_velocity(times, shock_positions)
    v_rel_err = abs(v_fit - V_SHOCK) / V_SHOCK
    assert v_rel_err < velocity_fit_tol, (
        f"Fitted shock velocity ({v_fit:.5f}) deviates from the nominal "
        f"piston velocity ({V_SHOCK}) by {v_rel_err:.4e}."
    )

    st_b = jnp.asarray(states[-1])
    sf_b = find_shocks_pfrommer(st_b, config, registered_variables, helper_data, mach_min=DSA_MACH_MIN)
    mach_b = np.asarray(sf_b.mach_numbers)
    shock_idx = int(np.argmax(mach_b))
    zone = np.asarray(sf_b.shock_zones)

    rho_b = states[-1, registered_variables.density_index]
    u_b = states[-1, registered_variables.velocity_index]
    p_gas_b = states[-1, registered_variables.pressure_index]
    e_cr_b = states[-1, registered_variables.cosmic_ray_e_index]
    p_cr_b = (GAMMA_CR - 1.0) * e_cr_b
    f_cr_b = states[-1, registered_variables.cosmic_ray_flux_index]

    precursor_mask = (np.arange(N_CELLS) > shock_idx) & (~zone)
    precursor_idx = np.where(precursor_mask)[0]
    precursor_idx = precursor_idx[precursor_idx < shock_idx + precursor_window_cells]
    assert len(precursor_idx) > 20, "Precursor window too small to check -- widen precursor_window_cells."

    xi = x[precursor_idx] - x[shock_idx]
    u_sf = u_b[precursor_idx] - v_fit
    rho_p = rho_b[precursor_idx]
    p_gas_p = p_gas_b[precursor_idx]
    p_cr_p = p_cr_b[precursor_idx]
    f_cr_p = f_cr_b[precursor_idx]

    mass_flux_arr = rho_p * u_sf
    momentum_flux_arr = rho_p * u_sf**2 + p_gas_p + p_cr_p
    mass_flux = float(np.mean(mass_flux_arr))
    momentum_flux = float(np.mean(momentum_flux_arr))
    mass_flux_spread = float(np.std(mass_flux_arr)) / abs(mass_flux)
    momentum_flux_spread = float(np.std(momentum_flux_arr)) / abs(momentum_flux)
    assert mass_flux_spread < mass_flux_tol, (
        f"mass_flux is not close to constant across the precursor window "
        f"(spread {mass_flux_spread:.4e}) -- the precursor is not in the "
        f"quasi-steady regime cr_precursor_ode_rhs assumes."
    )
    assert momentum_flux_spread < momentum_flux_tol, (
        f"momentum_flux is not close to constant across the precursor window "
        f"(spread {momentum_flux_spread:.4e}) -- the precursor is not in the "
        f"quasi-steady regime cr_precursor_ode_rhs assumes."
    )

    du_dx_num = np.gradient(u_sf, xi)
    dpcr_dx_num = np.gradient(p_cr_p, xi)
    dfcr_dx_num = np.gradient(f_cr_p, xi)

    du_dx_pred = np.zeros_like(u_sf)
    dpcr_dx_pred = np.zeros_like(u_sf)
    dfcr_dx_pred = np.zeros_like(u_sf)
    for j in range(len(u_sf)):
        a, b, c = cr_precursor_ode_rhs(
            float(u_sf[j]), float(p_cr_p[j]), float(f_cr_p[j]),
            mass_flux, momentum_flux, GAMMA, GAMMA_CR, diffusion_coefficient,
        )
        du_dx_pred[j], dpcr_dx_pred[j], dfcr_dx_pred[j] = a, b, c

    # Only compare where P_cr is well above the noise floor -- far upstream
    # of the precursor, P_cr/F_cr are numerically ~0 and relative errors
    # there are dominated by float noise, not physics.
    mask = p_cr_p > 1e-4 * p_cr_p.max()
    assert int(mask.sum()) > 10, "Too few precursor cells above the noise floor to check."

    for name, num, pred, tol in [
        ("du/dx", du_dx_num, du_dx_pred, velocity_ode_tol),
        ("dP_cr/dx", dpcr_dx_num, dpcr_dx_pred, p_cr_ode_tol),
        ("dF_cr/dx", dfcr_dx_num, dfcr_dx_pred, f_cr_ode_tol),
    ]:
        n, p_ = num[mask], pred[mask]
        rel_err = np.abs(n - p_) / np.maximum(np.abs(p_), 1e-8)
        median_rel_err = float(np.median(rel_err))
        assert median_rel_err < tol, (
            f"{name}: median rel. err {median_rel_err:.4e} >= tol {tol} -- the "
            f"simulation's own precursor gradient does not match "
            f"cr_precursor_ode_rhs's prediction at the same local state."
        )

    # Diagnostic plot: P_cr/F_cr profile in the shock-comoving coordinate.
    fig, (ax_pcr, ax_fcr) = plt.subplots(1, 2, figsize=(12, 5))
    ax_pcr.semilogy(xi, np.maximum(p_cr_p, 1e-12), "o-", ms=2, color="C0")
    ax_pcr.set_xlabel(r"$\xi = x - x_{\rm shock}(t)$")
    ax_pcr.set_ylabel(r"$P_{\rm cr}$")
    ax_pcr.set_title("CR precursor (shock-comoving frame)")
    ax_fcr.plot(xi, f_cr_p, "o-", ms=2, color="C1")
    ax_fcr.set_xlabel(r"$\xi = x - x_{\rm shock}(t)$")
    ax_fcr.set_ylabel(r"$F_{\rm cr}$")
    ax_fcr.set_title("CR flux (shock-comoving frame)")
    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_modified_shock_structure_precursor_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_cr_dsa_shock_jump()
    test_cr_precursor_ode()
