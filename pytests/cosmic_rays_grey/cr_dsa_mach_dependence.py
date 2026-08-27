"""
CR-DSA Mach-dependent efficiency pytest (Phase B ladder item 8).

Ladder item 8 replaces the fixed, Mach-independent DSA efficiency (ladder
item 7, ``pytests/cosmic_rays_grey/cr_sedov_taylor.py``) with an optional
Mach-dependent model: ``CosmicRayGreyConfig.dsa_efficiency_model ==
DSA_EFFICIENCY_KANG_RYU_2013`` makes
:func:`astronomix._modules._cosmic_rays_grey.cr_grey_injection.inject_crs_at_shocks`
use
:func:`astronomix._modules._cosmic_rays_grey.cr_grey_injection.dsa_efficiency_kang_ryu_2013`
(Kang & Ryu 2013's piecewise fit to their kinetic DSA simulation results)
scaled by ``CosmicRayGreyParams.dsa_efficiency_mach_scale`` -- 1.0 reproduces
KR13 itself, 0.5 approximates Caprioli & Spitkovsky (2014) for quasi-parallel
shocks (the literature's standard implementation of CS14 as half of KR13;
see that function's docstring for sources). The scalar ``dsa_efficiency``
path (``DSA_EFFICIENCY_CONSTANT``, ladder item 7) is unchanged and remains
the default, so this is purely additive.

Test design (two independent layers, rather than one large end-to-end
check):

1. Pure-function checks on ``dsa_efficiency_kang_ryu_2013`` itself: zero
   below the Ms=2 critical Mach number, continuous within ~2% across the
   weak/intermediate piece boundary (Ms=5) and ~1% across the
   intermediate/strong-plateau boundary (Ms=15), monotonically increasing,
   and asymptoting to the paper's stated ~0.211 plateau. Fast (no
   simulation), and exercises the *entire* Mach range the ladder item cares
   about -- a real blast wave only ever samples whatever Mach numbers it
   happens to produce, so this is the layer that actually checks the fit
   shape everywhere.

2. Live-simulation checks, reusing ``cr_sedov_taylor.py``'s exact physical
   setup (matching physics/parameters lets the two ladder items' results be
   compared directly) but with ``dsa_efficiency_model`` varied:

   - No NaNs, and total-energy conservation (same identity as item 7) holds
     for both a KR13 run (``dsa_efficiency_mach_scale=1.0``) and a CS14-like
     run (``dsa_efficiency_mach_scale=0.5``).
   - ``E_cr`` is a nonzero, bounded fraction of the total energy budget for
     both -- confirms Mach-dependent injection actually fires in a real run
     (not just in the pure-function checks above).
   - The CS14-like run's ``E_cr`` is close to half the KR13 run's (within a
     few percent, calibrated ~1.9% at NUM_CELLS=48 -- not exact, since these
     are two independent, full nonlinear time integrations: diverting more
     or less thermal energy into CRs at each step changes the gas pressure,
     which feeds back into the shock's subsequent speed/Mach trajectory, so
     the *instantaneous* injection rate scales exactly with
     ``dsa_efficiency_mach_scale`` but the *time-integrated* ``E_cr`` only
     approximately does. The exact, no-feedback version of this comparison
     is the cross-check below.
   - **The main check**: an *exact* formula cross-check. A third run
     (control, ``DSA_EFFICIENCY_CONSTANT`` with ``dsa_efficiency=0`` --
     identical to ``cr_sedov_taylor.py``'s control) produces a final state
     with a real, fully-formed shock but zero injected CR energy yet. Feed
     that state into ``find_shocks_pfrommer`` directly to get its
     ``mach_numbers``/``thermal_energy_flux``, independently compute the
     expected ``delta_e_cr`` from ``dsa_efficiency_kang_ryu_2013`` applied to
     those exact outputs, then call ``inject_crs_at_shocks`` on the same
     state (with the KR13 model selected) and compare -- since
     ``inject_crs_at_shocks`` calls ``find_shocks_pfrommer`` internally with
     the same arguments (deterministic, pure), this should match to full
     floating-point precision, not just approximately. This is what actually
     proves the injection code is *using* the Mach-dependent formula and not
     something else, independent of the aggregate-energy checks above.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
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
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
    DSA_EFFICIENCY_CONSTANT,
    DSA_EFFICIENCY_KANG_RYU_2013,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_efficiency_kang_ryu_2013,
    inject_crs_at_shocks,
)

# ---- physical setup (matches cr_sedov_taylor.py / _sedov_setup.py) ----
GAMMA = 5.0 / 3.0
NUM_CELLS = 48
T_END = 0.07
E_EXPLOSION = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1e-4
R_EXPLOSION = 0.05
SMOOTH_CELLS = 2.0
DSA_MACH_MIN = 1.3

# arbitrary but fixed dt for the standalone one-step formula cross-check
# (layer 2's main check) -- any concrete value works equally well, since
# both sides of that comparison use the identical value.
CROSS_CHECK_DT = 1e-4


def test_dsa_efficiency_kang_ryu_2013_shape():
    """Pure-function checks on the KR13 piecewise fit (layer 1, see module
    docstring) -- no simulation, covers the whole Mach range at once."""
    ms = jnp.array([0.0, 1.0, 1.999, 2.0, 3.0, 5.0, 5.0001, 10.0, 15.0, 15.0001, 20.0, 1e3])
    eta = dsa_efficiency_kang_ryu_2013(ms)

    # zero below the Ms=2 critical Mach number
    assert bool(jnp.all(eta[ms < 2.0] == 0.0)), "eta should be exactly 0 below Ms=2."

    # nonzero and monotonically increasing from Ms=2 up to the plateau
    assert bool(jnp.all(jnp.diff(eta) >= 0.0)), "eta should be monotonically nondecreasing."
    assert float(eta[ms == 2.0][0]) > 0.0, "eta should be > 0 at Ms=2."

    # continuous (to a few percent) across both piece boundaries
    idx5, idx5p = int(jnp.where(ms == 5.0)[0][0]), int(jnp.where(ms == 5.0001)[0][0])
    boundary5_rel_err = abs(float(eta[idx5p] - eta[idx5])) / float(eta[idx5])
    assert boundary5_rel_err < 0.05, f"Discontinuity at Ms=5 too large: {boundary5_rel_err:.4f}"

    idx15, idx15p = int(jnp.where(ms == 15.0)[0][0]), int(jnp.where(ms == 15.0001)[0][0])
    boundary15_rel_err = abs(float(eta[idx15p] - eta[idx15])) / float(eta[idx15])
    assert boundary15_rel_err < 0.01, f"Discontinuity at Ms=15 too large: {boundary15_rel_err:.4f}"

    # asymptotes to the paper's stated ~0.211 plateau
    plateau = float(eta[ms == 1e3][0])
    assert abs(plateau - 0.211) < 1e-6, f"Strong-shock plateau should be 0.211, got {plateau}."


def _run_sedov(dsa_efficiency_model: int, dsa_efficiency: float = 0.1,
                dsa_efficiency_mach_scale: float = 1.0):
    """Run one 3D CR-DSA Sedov blast; return the final state and diagnostics.

    Identical physical setup to ``cr_sedov_taylor.py``'s ``_run_sedov``, with
    ``dsa_efficiency_model``/``dsa_efficiency_mach_scale`` exposed so this
    module can select the Mach-dependent path.
    """
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        exact_end_time=True,
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True,
            diffusive_shock_acceleration=True,
            dsa_efficiency_model=dsa_efficiency_model,
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
            dsa_efficiency=dsa_efficiency,
            dsa_mach_min=DSA_MACH_MIN,
            dsa_efficiency_mach_scale=dsa_efficiency_mach_scale,
        ),
    )

    final_state = time_integration(initial_state, config, params, registered_variables)

    rho = final_state[registered_variables.density_index]
    vx = final_state[registered_variables.velocity_index.x]
    vy = final_state[registered_variables.velocity_index.y]
    vz = final_state[registered_variables.velocity_index.z]
    p_gas = final_state[registered_variables.pressure_index]
    e_cr = final_state[registered_variables.cosmic_ray_e_index]

    e_thermal_density = p_gas / (GAMMA - 1.0)
    e_kinetic_density = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    E_thermal = float(jnp.sum(e_thermal_density) * cell_volume)
    E_kinetic = float(jnp.sum(e_kinetic_density) * cell_volume)
    E_cr = float(jnp.sum(e_cr) * cell_volume)

    return dict(
        final_state=final_state,
        config=config,
        params=params,
        registered_variables=registered_variables,
        helper_data=helper_data,
        radius=radius,
        rho=rho,
        p_gas=p_gas,
        e_cr=e_cr,
        E_thermal=E_thermal,
        E_kinetic=E_kinetic,
        E_cr=E_cr,
        E_total=E_thermal + E_kinetic + E_cr,
    )


def test_cr_dsa_mach_dependence(
    conservation_tol: float = 1e-3,
    scale_ratio_tol: float = 0.05,
    cross_check_tol: float = 1e-5,
    domain_half_width: float = 0.5,
    containment_margin: float = 0.03,
):
    """CR-DSA Mach-dependent efficiency: live-simulation checks (layer 2).

    Args:
        conservation_tol: Maximum allowed relative error on each DSA run's
            own total-energy conservation, same identity/tolerance family as
            ``cr_sedov_taylor.py``.
        scale_ratio_tol: Maximum allowed relative error on
            ``E_cr(scale=0.5) / E_cr(scale=1.0) == 0.5``. Calibrated observed
            error ~1.9% at NUM_CELLS=48 -- not near-machine-precision, since
            these are two independent full time integrations with a genuine
            (if small) dynamical feedback between them (see module
            docstring); 0.05 leaves a >2.5x margin.
        cross_check_tol: Maximum allowed relative error on the exact
            formula cross-check (layer 2's main check) -- both sides use the
            same deterministic computation, so any mismatch beyond float32
            precision (this module runs in float32; calibrated observed
            error ~8e-8, consistent with that and not with a real formula
            mismatch, which would show up at the percent level or worse)
            means the injection code isn't using the documented formula.
            1e-5 leaves a >100x margin over the calibrated value.
        domain_half_width: Half the (unit, default) box size.
        containment_margin: Minimum gap the shock front must stay within the
            domain, same purpose as in ``cr_sedov_taylor.py``.
    """
    control = _run_sedov(dsa_efficiency_model=DSA_EFFICIENCY_CONSTANT, dsa_efficiency=0.0)
    kr13 = _run_sedov(dsa_efficiency_model=DSA_EFFICIENCY_KANG_RYU_2013,
                       dsa_efficiency_mach_scale=1.0)
    cs14 = _run_sedov(dsa_efficiency_model=DSA_EFFICIENCY_KANG_RYU_2013,
                       dsa_efficiency_mach_scale=0.5)

    for name, run in (("control", control), ("kr13", kr13), ("cs14", cs14)):
        assert not bool(jnp.any(jnp.isnan(run["final_state"]))), (
            f"CR-DSA Mach-dependence ({name} run) produced NaNs."
        )

    # Containment (see cr_sedov_taylor.py for the same check/rationale).
    rho_flat = control["rho"].reshape(-1)
    r_flat = control["radius"].reshape(-1)
    shocked = jnp.abs(rho_flat / RHO_AMBIENT - 1.0) > 0.01
    r_shock = float(jnp.max(jnp.where(shocked, r_flat, 0.0)))
    assert r_shock < domain_half_width - containment_margin, (
        f"Shock front (r={r_shock:.4f}) is too close to the open boundary "
        f"for the energy-budget checks below to be meaningful."
    )

    assert control["E_cr"] == 0.0, (
        f"Control run (dsa_efficiency=0) injected nonzero CR energy: "
        f"E_cr={control['E_cr']:.6e}."
    )

    total_volume = 1.0**3
    E_ambient_initial = P_AMBIENT / (GAMMA - 1.0) * total_volume
    E_total_initial = E_ambient_initial + E_EXPLOSION
    for name, run in (("kr13", kr13), ("cs14", cs14)):
        rel_err = abs(run["E_total"] - E_total_initial) / E_total_initial
        assert rel_err < conservation_tol, (
            f"{name} run's total energy is not conserved: final "
            f"{run['E_total']:.6f} vs. initial {E_total_initial:.6f} "
            f"(rel. err {rel_err:.4e} >= tol {conservation_tol})."
        )

    # E_cr should be a clearly nonzero but bounded fraction of the total
    # energy budget. Calibrated at NUM_CELLS=48: KR13 ~4.8%, CS14-like
    # ~2.4% -- comparable to (modestly below) item 7's flat-10%-efficiency
    # model's ~5.3%. The blast's shock is genuinely strong at t=0.07 (the
    # control run's shock-surface cells span Ms~3-29, mean~11 -- see the
    # diagnostic plot), so most cells sit well above where KR13's efficiency
    # reaches 10% (~Ms=5); the total is a time-integrated quantity over the
    # whole run, including the shock's earlier (Mach-decreasing-over-time,
    # so even stronger) history, not just this final-time snapshot, so no
    # precise a priori match to item 7's flat model is expected either way.
    for name, run in (("kr13", kr13), ("cs14", cs14)):
        cr_fraction = run["E_cr"] / E_total_initial
        assert 0.001 < cr_fraction < 0.3, (
            f"{name} run's CR energy fraction ({cr_fraction:.4f}) is outside "
            f"the expected [0.001, 0.3] band -- either injection is not "
            f"happening, or is wildly over-injecting."
        )

    # dsa_efficiency_mach_scale=0.5 should give almost exactly half the CR
    # energy of scale=1.0, in a real run -- confirms the scale multiplies
    # the per-cell efficiency field itself (see module docstring).
    ratio = cs14["E_cr"] / kr13["E_cr"]
    ratio_rel_err = abs(ratio - 0.5) / 0.5
    assert ratio_rel_err < scale_ratio_tol, (
        f"CS14-like/KR13 E_cr ratio ({ratio:.6f}) should be almost exactly "
        f"0.5 -- rel. err {ratio_rel_err:.4e} >= tol {scale_ratio_tol}."
    )

    # Main check: exact formula cross-check (layer 2, see module docstring).
    # Reuse the control run's final state (a real, fully-formed shock, zero
    # CR energy yet) and grid/config, only swapping in the KR13 model.
    kr13_config = control["config"]._replace(
        cosmic_ray_grey_config=control["config"].cosmic_ray_grey_config._replace(
            dsa_efficiency_model=DSA_EFFICIENCY_KANG_RYU_2013
        )
    )
    kr13_params = control["params"]._replace(
        cosmic_ray_grey_params=control["params"].cosmic_ray_grey_params._replace(
            dsa_efficiency_mach_scale=1.0
        )
    )
    registered_variables = control["registered_variables"]
    helper_data = control["helper_data"]
    control_state = control["final_state"]

    sf_result = find_shocks_pfrommer(
        control_state, kr13_config, registered_variables, helper_data,
        mach_min=DSA_MACH_MIN,
    )
    expected_efficiency = dsa_efficiency_kang_ryu_2013(sf_result.mach_numbers)
    expected_delta_e_cr = (
        expected_efficiency * sf_result.thermal_energy_flux
        / kr13_config.grid_spacing * CROSS_CHECK_DT
    )
    assert float(jnp.max(expected_delta_e_cr)) > 0.0, (
        "Cross-check setup produced zero expected CR injection everywhere -- "
        "the control run's shock apparently never exceeds Ms=2, so this "
        "check isn't exercising anything. Increase T_END or DSA_MACH_MIN."
    )

    injected_state = inject_crs_at_shocks(
        control_state, kr13_config, kr13_params, registered_variables,
        helper_data, current_time=T_END, dt=CROSS_CHECK_DT,
    )
    actual_delta_e_cr = (
        injected_state[registered_variables.cosmic_ray_e_index]
        - control_state[registered_variables.cosmic_ray_e_index]
    )

    max_expected = float(jnp.max(jnp.abs(expected_delta_e_cr)))
    max_abs_err = float(jnp.max(jnp.abs(actual_delta_e_cr - expected_delta_e_cr)))
    cross_check_rel_err = max_abs_err / max_expected
    assert cross_check_rel_err < cross_check_tol, (
        f"inject_crs_at_shocks's actual per-cell CR injection does not match "
        f"an independent computation of dsa_efficiency_kang_ryu_2013 applied "
        f"to the same find_shocks_pfrommer output -- rel. err "
        f"{cross_check_rel_err:.4e} >= tol {cross_check_tol}. This means the "
        f"injection code is not actually using the documented formula."
    )

    # Diagnostic plot: E_cr fraction by config (left), and the KR13
    # efficiency curve with the control run's actual shock-cell Mach numbers
    # marked on it (right) -- ties the pure-function shape (layer 1) to what
    # a real blast actually samples.
    fig, (ax_bars, ax_curve) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["KR13 (scale=1.0)", "CS14-like (scale=0.5)"]
    fractions = [kr13["E_cr"] / E_total_initial, cs14["E_cr"] / E_total_initial]
    ax_bars.bar(labels, fractions, color=["C0", "C1"])
    ax_bars.set_ylabel(r"$E_{\rm cr} / E_{\rm total}$")
    ax_bars.set_title(f"CR energy fraction at t={T_END}")

    ms_curve = jnp.linspace(0.0, 20.0, 400)
    ax_curve.plot(ms_curve, dsa_efficiency_kang_ryu_2013(ms_curve), "-", color="C2",
                  label="KR13 fit")
    shock_mach = sf_result.mach_numbers.reshape(-1)
    shock_mach = shock_mach[shock_mach > 0.0]
    if shock_mach.size > 0:
        ax_curve.scatter(
            shock_mach, dsa_efficiency_kang_ryu_2013(shock_mach),
            s=8, color="C3", label="control run's shock cells",
        )
    ax_curve.axvline(2.0, color="grey", ls="--", lw=1, label="Ms=2 (critical)")
    ax_curve.set_xlabel(r"$M_s$")
    ax_curve.set_ylabel(r"$\eta_{\rm DSA}$")
    ax_curve.set_title("Kang & Ryu (2013) efficiency")
    ax_curve.legend(fontsize=8)

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_dsa_mach_dependence_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_dsa_efficiency_kang_ryu_2013_shape()
    test_cr_dsa_mach_dependence()
