"""
Koyama & Inutsuka (2002) two-phase net-cooling curve: isothermal-relaxation
pytest (SILCC-ISM project, milestone M1).

New cooling-curve type ``KOYAMA_INUTSUKA_NET_COOLING``
(``astronomix._modules._cooling.cooling_options``/``_cooling_tables``/
``_cooling.py``) implements the literature K&I heating+cooling combination

    dU/dt = n_H * Gamma - n_H^2 * Lambda(T)
    Gamma = 2e-26 erg/s
    Lambda(T) = Gamma * [1e7*exp(-1.184e5/(T+1000)) + 1.4e-2*sqrt(T)*exp(-92/T)]

(Koyama & Inutsuka 2002, ApJ 564, L97 eq. 4, corrected coefficients per the
erratum discussion in Nagashima, Inutsuka & Koyama 2006). This does not fit
the existing ``PIECEWISE_POWER_LAW`` table format (which stores
``log10(Lambda)`` and so cannot represent a curve that changes sign -- and a
sign change is exactly what a two-phase equilibrium needs: net heating below
the unstable branch, net cooling above it) -- see
``_cooling_tables.koyama_inutsuka_cooling``'s docstring for the closed-form
dispatch branch this uses instead, and for the mu_e/mu_H bookkeeping that
lets it reuse the existing ``dtemperature_dt``/``update_pressure_by_cooling``
pipeline exactly (not an approximation -- every mu_e cancels algebraically).

**A real finding that reshaped this test's design (2026-09-11):** the plan
that scoped this milestone assumed "gas relaxes toward the two stable phases
from a range of initial temperatures" *at fixed density* -- checked directly
against the actual curve (``ki_bracket`` in the reference solution below,
evaluated across T=10-1e5 K) and found it is **monotonically increasing
throughout**, with no local hump. This means the equilibrium condition
``bracket(T) = Gamma/(n_H*Gamma_scale) = 1/n_H`` (dimensionally, at fixed
``n_H``) has **exactly one root, always thermally stable** -- there is no
fixed-density bistability to demonstrate; ``update_pressure_by_cooling``
only ever touches pressure, so a fixed-density box cannot explore the other
branch at all. The classic K&I "S-curve" bistability instead lives in the
**pressure-density plane at fixed pressure**: since the (unique, per-density)
equilibrium temperature ``T_eq(n_H)`` is a *decreasing* function of ``n_H``,
the equilibrium pressure ``P_eq(n_H) = n_H * k_B * T_eq(n_H)`` is
**non-monotonic** in ``n_H`` -- it has a local maximum around ``n_H ~ 1
cm^-3`` (``P_eq/k_B ~ 4950 K/cm^3``) and a local minimum around ``n_H ~ 8.6
cm^-3`` (``P_eq/k_B ~ 1597 K/cm^3``); any pressure in between admits three
equilibrium densities (cold/dense, unstable, warm/diffuse) -- this is where
genuine bistability would show up, but only once density is free to respond
to pressure imbalance (self-gravity/stratification/turbulence -- M2 onward),
not in this milestone's isolated, no-gravity, fixed-density scope. This test
therefore checks the two things that *are* correctly scoped to an isolated
cooling-curve milestone: (1) at each of several representative densities,
gas relaxes to the unique, correct equilibrium temperature regardless of
which side of it the initial temperature started on; (2) the equilibrium
curve traced across those densities is genuinely non-monotonic in pressure
-- the defining physical signature that this is actually a two-phase-capable
curve, not just an arbitrary monotonic cooling law.

**Cooling-CFL stability question (gap #3) -- a second real finding.** Before
this milestone, cooling was applied at whatever ``dt`` the (cooling-unaware)
hydro CFL condition picked; near a strongly-heating/cooling state that
``dt`` can exceed the local instantaneous cooling time by an order of
magnitude or more (checked directly below: at ``n_H=10 cm^-3``, ``T=1e4``
K, the instantaneous cooling time is ~3438 yr). The plan's expectation was
that the module's default ``IMPLICIT_COOLING`` integrator would handle
this gracefully -- **originally checked directly and found false**:
``update_temperature_implicit``'s original naive fixed-point iteration
(``T_new = T + dT/dt(T_new)*dt``) did not converge for ``dt`` this large
relative to the local relaxation time, and ``jax.lax.while_loop`` simply
stopped at ``max_iter=50`` regardless of whether ``tol`` was reached --
silently returning a wrong (and, at more extreme stiffness, wildly
unphysical -- e.g. ``6.8e6`` K at 100x this ``dt``) temperature with no
error. **Fixed 2026-09-16: ``update_temperature_implicit`` now solves the
same backward-Euler equation via Newton's method instead** (see its own
docstring in ``_cooling.py``) -- this eliminates the silent-divergence
failure mode entirely (confirmed convergent, residual ~1e-12, in ~12
iterations even at ratios where the old solver diverged to nonsense), but
does **not** eliminate all error at large ``dt``: even an exactly-solved
single backward-Euler step is only first-order accurate in time, so
real (bounded, physically-sensible) truncation error remains at
intermediate stiffness (see ``test_ki_cooling_cfl_guard``'s updated
docstring for calibrated numbers). ``test_ki_cooling_cfl_guard`` confirms
both the historical failure mode (now fixed) and the still-real accuracy
motivation for a cooling-aware ``dt_cool`` term in
``_finite_volume/_timestep_estimation/_timestep_estimator.py``'s
``_cfl_time_step`` (mirroring the existing ``dt_visc``/``dt_relax``
pattern, gated on ``config.cooling_config.cooling`` so it is a no-op for
every existing non-cooling test), which bounds ``dt`` by the fastest local
relaxation time on the grid -- **still worth keeping even with Newton**,
since it's now an accuracy safeguard rather than a divergence safeguard.
SILCC-ISM project milestone M4 (2026-09-15) additionally needed to
decouple cooling stiffness from the *global* hydro ``dt`` entirely (one
persistently fast-cooling cell was stalling a whole simulation) via a
separate, opt-in ``CoolingConfig.subcycle_stiff_cooling`` mechanism, kept
as-is and unaffected by the Newton fix (it uses its own adaptive explicit
sub-stepping, not ``update_temperature_implicit`` internally).

**A third real finding, this one a test-setup pitfall worth flagging for any
future cooling test:** this codebase's cooling module defines ``n_H =
density / mu_H`` (``get_particle_number_density``'s convention), so a
uniform box at a target hydrogen number density must be built with
``density = n_H * mu_H * m_p`` -- *not* ``density = n_H * m_p`` (which is
the convention every existing CR-grey pytest uses, since none of them ever
call the cooling module's ``n_H``-dependent machinery). Missing the
``mu_H`` factor here silently builds a box at ``n_H / mu_H`` instead, which
-- because ``T_eq(n_H)`` is strongly density-dependent -- relaxes cleanly
and confidently to the *wrong* equilibrium temperature (caught by
comparing against the independent reference below, which is exactly why
that cross-check is there rather than only checking "did it converge to
some value").

See ``astronomix/test_setups/reference_solutions/koyama_inutsuka_equilibrium.py``
for the independent (plain numpy/scipy, no astronomix/JAX) reference used to
cross-check every claim above.
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

# Stiff, near-machine-precision checks (density conservation under cooling,
# which should touch only pressure) -- same reasoning as bc_smoke_test.py /
# stratified_hydrostatic_column.py.
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# units
from astropy import units as u
import astropy.constants as c

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import FINITE_VOLUME, HLLC, MINMOD
from astronomix import CodeUnits
from astronomix import SimulationConfig, SimulationParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

# astronomix modules
from astronomix._finite_volume._timestep_estimation._timestep_estimator import _cfl_time_step
from astronomix._modules._cooling._cooling import (
    get_effective_molecular_weights,
    get_pressure_from_temperature,
    get_temperature_from_pressure,
    update_temperature_implicit,
)
from astronomix._modules._cooling._cooling_tables import koyama_inutsuka_cooling
from astronomix._modules._cooling.cooling_options import (
    IMPLICIT_COOLING,
    KOYAMA_INUTSUKA_NET_COOLING,
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
)

# independent reference (plain numpy/scipy, no astronomix/JAX)
from astronomix.test_setups.reference_solutions.koyama_inutsuka_equilibrium import (
    find_equilibrium_temperatures,
    integrate_isochoric_relaxation,
    ki_net_rate,
)

# ---- physical setup ----
GAMMA = 5.0 / 3.0
X_H = 0.76  # hydrogen mass fraction (CoolingParams default)
Z_METAL = 0.02  # metal mass fraction (CoolingParams default)
MU, MU_E, MU_H = get_effective_molecular_weights(X_H, Z_METAL)
C_CFL_DEFAULT = SimulationParams().C_cfl

CODE_UNITS = CodeUnits(1 * u.pc, 1 * u.M_sun, 1 * u.km / u.s)
KI_PARAMS = koyama_inutsuka_cooling(CODE_UNITS, X_H, Z_METAL)
FLOOR_TEMPERATURE_KELVIN = 10.0
FLOOR_TEMPERATURE_CODE = FLOOR_TEMPERATURE_KELVIN * KI_PARAMS.code_temperature_per_kelvin

NUM_CELLS = 8
BOX_SIZE_PHYS = 1.0 * u.pc
BOX_SIZE = BOX_SIZE_PHYS.to(CODE_UNITS.code_length).value

# Representative ISM densities spanning the K&I equilibrium curve's warm
# (n_H~0.3), canonical (n_H~1), and cold (n_H~10) regimes -- see module
# docstring for why these are probed at fixed density rather than seeking
# fixed-density bistability.
N_H_VALUES_CGS = [0.3, 1.0, 10.0]
T_EQ_KELVIN = {n_h: find_equilibrium_temperatures(n_h)[0] for n_h in N_H_VALUES_CGS}

# Perturbation factors (below/above the true equilibrium) and a
# density-specific t_end, generous relative to the ODE-estimated time to
# reach 1% of equilibrium (checked directly in an ad hoc script before
# committing these numbers -- see PROGRESS.md's M1 entry). Cheap: this is a
# uniform, velocity-free 8-cell box, so even the longest of these needs only
# a few thousand CFL-limited steps.
T_FACTORS = [0.8, 1.2]
T_END_YEARS = {0.3: 6.0e6, 1.0: 3.0e7, 10.0: 5.0e5}


def _run_relaxation(n_h_cgs: float, t_init_factor: float):
    """Run one uniform, fixed-density relaxation box; return diagnostics."""
    t_init_kelvin = t_init_factor * T_EQ_KELVIN[n_h_cgs]
    t_end_years = T_END_YEARS[n_h_cgs]

    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=1,
        num_cells=NUM_CELLS,
        box_size=BOX_SIZE,
        exact_end_time=True,
        cooling_config=CoolingConfig(
            cooling=True,
            cooling_method=IMPLICIT_COOLING,
            cooling_curve_config=CoolingCurveConfig(cooling_curve_type=KOYAMA_INUTSUKA_NET_COOLING),
        ),
    )
    registered_variables = get_registered_variables(config)

    # n_H = density / mu_H is this codebase's own convention
    # (get_particle_number_density), so density = n_H * mu_H * m_p -- not
    # n_H * m_p (mu_H != 1 in general; missing it silently shifts which
    # equilibrium the run actually relaxes to, see PROGRESS.md's M1 entry
    # for how this was caught).
    rho_code = float((n_h_cgs / u.cm ** 3 * MU_H * c.m_p).to(CODE_UNITS.code_density).value)
    t_init_code = t_init_kelvin * KI_PARAMS.code_temperature_per_kelvin
    p_init_code = float(get_pressure_from_temperature(rho_code, t_init_code, X_H, Z_METAL))

    density = jnp.ones(NUM_CELLS) * rho_code
    zero = jnp.zeros(NUM_CELLS)
    gas_pressure = jnp.ones(NUM_CELLS) * p_init_code

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zero,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, initial_state.shape)

    t_end_code = (t_end_years * u.yr).to(CODE_UNITS.code_time).value
    params = SimulationParams(
        t_end=t_end_code,
        gamma=GAMMA,
        cooling_params=CoolingParams(
            hydrogen_mass_fraction=X_H,
            metal_mass_fraction=Z_METAL,
            floor_temperature=FLOOR_TEMPERATURE_CODE,
            cooling_curve_params=KI_PARAMS,
        ),
    )

    final_state = time_integration(initial_state, config, params, registered_variables)

    rho_final = final_state[registered_variables.density_index]
    p_final = final_state[registered_variables.pressure_index]
    t_final_code = get_temperature_from_pressure(rho_final, p_final, X_H, Z_METAL)
    t_final_kelvin = t_final_code / KI_PARAMS.code_temperature_per_kelvin

    # Independent (scipy, no astronomix/JAX) trajectory at the same t_end,
    # using the same mu the simulation's own EOS uses for the heat capacity.
    ode_solution = integrate_isochoric_relaxation(
        n_h_cgs, t_init_kelvin, t_end_years, MU, MU_H, GAMMA
    )
    t_ode_kelvin_final = float(ode_solution.y[0, -1])

    return dict(
        final_state=final_state,
        rho_init=rho_code,
        rho_final=rho_final,
        t_init_kelvin=t_init_kelvin,
        t_final_kelvin=t_final_kelvin,
        t_eq_kelvin=T_EQ_KELVIN[n_h_cgs],
        t_ode_kelvin_final=t_ode_kelvin_final,
        t_end_years=t_end_years,
    )


def test_ki_cooling_relaxation(
    density_conservation_tol: float = 1e-10,
    eq_rel_tol: float = 0.01,
    ode_match_rel_tol: float = 0.005,
):
    """Isochoric relaxation to the K&I equilibrium temperature, at 3 densities.

    For each of ``N_H_VALUES_CGS``, runs both an initially-cold
    (``T_FACTORS[0] * T_eq``) and an initially-hot (``T_FACTORS[1] * T_eq``)
    uniform box to a density-specific ``t_end`` and checks:

    1. No NaNs; density unchanged (cooling only touches pressure).
    2. The simulated final temperature matches the independent
       scipy-integrated ODE trajectory at the same ``t_end`` -- validates
       ``update_pressure_by_cooling``'s actual per-step integration, not
       just the final fixed point.
    3. The simulated final temperature has converged close to the true
       equilibrium ``T_eq`` (from ``t_end_years``'s generous margin over the
       ODE-estimated relaxation time -- see module docstring).
    4. The equilibrium pressure ``P_eq(n_H) = n_H * T_eq(n_H)`` (from the
       simulation's own converged temperatures) is non-monotonic in
       ``n_H`` -- the defining two-phase signature (see module docstring).

    Calibrated 2026-09-11 (all six (n_H, factor) combinations): density is
    conserved exactly (rel. err ``0.0`` at float64); ``eq_rel_err`` ranges
    ``1.5e-5`` to ``3.4e-3`` (worst case at ``n_H=10``, tol has >2.9x
    margin); ``ode_rel_err`` ranges ``6.0e-8`` to ``1.2e-3`` (same worst
    case, tol has >4x margin). Simulated ``P_eq/k_B`` at
    ``n_H={0.3, 1.0, 10.0}`` is ``{2017, 5006, 1605}`` -- matches the
    independent reference's scan to <1%, confirming the non-monotonic
    (two-phase) structure.

    Args:
        density_conservation_tol: Max relative density drift under cooling.
        eq_rel_tol: Max relative error between the simulated final T and
            the true equilibrium T_eq, at t_end.
        ode_match_rel_tol: Max relative error between the simulated final T
            and the independently-integrated ODE's T at the same t_end.
    """
    p_eq_sim = {}

    for n_h in N_H_VALUES_CGS:
        for factor in T_FACTORS:
            run = _run_relaxation(n_h, factor)

            assert not bool(jnp.any(jnp.isnan(run["final_state"]))), (
                f"n_H={n_h}, factor={factor}: relaxation run produced NaNs."
            )

            rho_rel_err = float(jnp.max(jnp.abs(run["rho_final"] - run["rho_init"]) / run["rho_init"]))
            assert rho_rel_err < density_conservation_tol, (
                f"n_H={n_h}, factor={factor}: density drifted under cooling "
                f"(rel. err {rho_rel_err:.4e} >= tol {density_conservation_tol})."
            )

            t_sim = float(jnp.mean(run["t_final_kelvin"]))
            eq_rel_err = abs(t_sim - run["t_eq_kelvin"]) / run["t_eq_kelvin"]
            assert eq_rel_err < eq_rel_tol, (
                f"n_H={n_h}, factor={factor}: simulated final T ({t_sim:.2f} K) "
                f"did not converge to T_eq ({run['t_eq_kelvin']:.2f} K) -- "
                f"rel. err {eq_rel_err:.4e} >= tol {eq_rel_tol}."
            )

            ode_rel_err = abs(t_sim - run["t_ode_kelvin_final"]) / run["t_ode_kelvin_final"]
            assert ode_rel_err < ode_match_rel_tol, (
                f"n_H={n_h}, factor={factor}: simulated final T ({t_sim:.2f} K) "
                f"does not match the independent ODE trajectory "
                f"({run['t_ode_kelvin_final']:.2f} K) at the same t_end -- "
                f"rel. err {ode_rel_err:.4e} >= tol {ode_match_rel_tol}."
            )

            p_eq_sim.setdefault(n_h, []).append(n_h * t_sim)

    p_eq_by_density = {n_h: float(np.mean(vals)) for n_h, vals in p_eq_sim.items()}
    assert p_eq_by_density[1.0] > p_eq_by_density[0.3], (
        "Equilibrium pressure did not rise from n_H=0.3 to n_H=1.0 -- "
        "expected the K&I curve's rising (warm) branch here."
    )
    assert p_eq_by_density[1.0] > p_eq_by_density[10.0], (
        "Equilibrium pressure did not fall from n_H=1.0 to n_H=10.0 -- "
        "expected the K&I curve's falling (unstable) branch here."
    )

    # Diagnostic plot: the K&I equilibrium curve (reference) with the
    # simulation's own converged (n_H, T_eq) points overlaid.
    n_scan = np.logspace(np.log10(0.05), np.log10(50), 200)
    t_scan = np.array([find_equilibrium_temperatures(n)[0] for n in n_scan])
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))
    ax0.loglog(n_scan, t_scan, "-", color="C0", label="reference T_eq(n_H)")
    ax0.loglog(N_H_VALUES_CGS, [T_EQ_KELVIN[n] for n in N_H_VALUES_CGS], "o", color="C1", label="calibrated points")
    ax0.set_xlabel(r"$n_H$ [cm$^{-3}$]")
    ax0.set_ylabel(r"$T_{eq}$ [K]")
    ax0.set_title("K&I equilibrium curve")
    ax0.legend()

    ax1.loglog(n_scan, n_scan * t_scan, "-", color="C0", label="reference P_eq/k_B(n_H)")
    ax1.loglog(
        list(p_eq_by_density.keys()), list(p_eq_by_density.values()), "o", color="C1",
        label="simulated P_eq/k_B",
    )
    ax1.set_xlabel(r"$n_H$ [cm$^{-3}$]")
    ax1.set_ylabel(r"$P_{eq}/k_B$ [K cm$^{-3}$]")
    ax1.set_title("non-monotonic two-phase pressure curve")
    ax1.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "ki_cooling_thermal_relaxation_test.svg")
    plt.close(fig)


def test_ki_cooling_cfl_guard(
    newton_rel_err_max: float = 0.22,
    newton_vs_old_naive_rel_err_max: float = 0.3,
    guarded_dt_ratio_max: float = 0.5,
    guarded_dt_match_rel_tol: float = 1e-4,
):
    """Confirms the stiff-dt behavior of the (now Newton-based) implicit
    solver (gap #3, and its 2026-09-16 fix), plus the CFL guard's continued
    accuracy role.

    At ``n_H=10 cm^-3``, ``T=1e4`` K (a hot cell suddenly at high density --
    e.g. freshly-shocked or freshly-injected ejecta), the instantaneous
    cooling time is ~3438 yr (checked directly below). **Original finding
    (2026-09-11):** calling ``update_temperature_implicit`` with a single
    ``dt=5e4`` yr step (~14.5x that cooling time) did not converge -- its
    then-naive fixed-point iteration (``T_new = T + dT/dt(T_new)*dt``)
    diverged for ``dt`` this large, and ``jax.lax.while_loop`` just stopped
    at ``max_iter=50`` regardless of whether ``tol`` was reached, silently
    returning a wrong temperature (``10068`` K vs. the reference's ``5855``
    K here; far worse -- literally millions of percent off, e.g. ``6.8e6``
    K -- at higher stiffness ratios, checked separately). **Fixed
    2026-09-16: ``update_temperature_implicit`` now solves the same
    backward-Euler equation via Newton's method** (see its docstring in
    ``_cooling.py``). Part 1 below now confirms the *opposite* premise from
    before: Newton converges cleanly here (residual ~1e-12) and gets
    materially closer to the reference than the old naive result did, but
    **not exactly right** -- a single, however-exactly-solved backward-Euler
    step is still only first-order accurate in time, so real truncation
    error remains at this stiffness (``rel. err ~0.187`` here, vs. the old
    naive result's ``0.72``). Part 2 confirms ``_cfl_time_step``'s
    ``dt_cool`` term (mirroring the existing ``dt_visc``/``dt_relax``
    pattern) still meaningfully shrinks ``dt`` at this same stiff state,
    correctly sized to the local cooling time -- worth keeping even with
    Newton, since its role is now accuracy (bounding truncation error) more
    than the divergence-safety it was originally added for.

    Calibrated 2026-09-16 (Part 1 renumbered from the original naive-failure
    calibration): Newton's single-step result is ``6951`` K vs. the
    reference's ``5855`` K (rel. err ``0.187``, tol has ~15% margin) --
    dramatically closer than the old naive solver's ``10068`` K (rel. err
    ``0.72``), demonstrating the fix, though not exact. ``_cfl_time_step``'s
    guarded dt is ``0.4296x`` the pure-hydro dt (tol has ~14% margin -- this
    is a fully deterministic ratio, not a noisy measurement, so this margin
    is real); it matches ``C_cfl *`` the independent instantaneous-cooling-
    time estimate to ``~1.4e-15`` (exact at float64, as expected -- both are
    the same single-uniform-cell calculation by construction) -- tol left
    much looser (``1e-4``) to tolerate backend/precision differences on
    other machines.

    Args:
        newton_rel_err_max: Max relative error the (now Newton-based)
            single-step call may show against the reference at this stiff
            dt -- confirms Newton converges to something reasonably close,
            not that it is exact (it isn't, by construction of a single
            large backward-Euler step).
        newton_vs_old_naive_rel_err_max: Max relative error allowed,
            expressed as a fraction of the old naive solver's own
            historical error (``0.72``, hardcoded from the 2026-09-11
            calibration since that solver no longer exists to call
            directly) -- confirms Newton is a real, substantial
            improvement, not just "still wrong but slightly different."
        guarded_dt_ratio_max: Max allowed ratio of the cooling-guarded dt to
            the pure-hydro dt at this stiff state -- confirms the guard is
            actually active and materially reducing dt.
        guarded_dt_match_rel_tol: Max relative error between the
            cooling-guarded dt and the independently-computed instantaneous
            cooling time -- confirms the guard is correctly sized, not just
            "some smaller number".
    """
    n_h_cgs = 10.0
    t_init_kelvin = 1.0e4
    dt_years = 5.0e4

    n_tot_cgs = n_h_cgs * MU_H / MU  # true total particle number density, see reference module docstring
    instantaneous_t_cool_years = (
        (n_tot_cgs * (1.380649e-16) * t_init_kelvin / (GAMMA - 1.0))
        / abs(ki_net_rate(n_h_cgs, t_init_kelvin))
    ) / 3.15576e7
    assert dt_years > 10.0 * instantaneous_t_cool_years, (
        f"Test setup is not actually stiff: dt={dt_years} yr is not >>10x "
        f"the instantaneous cooling time ({instantaneous_t_cool_years:.2f} yr)."
    )

    rho_code = float((n_h_cgs / u.cm ** 3 * MU_H * c.m_p).to(CODE_UNITS.code_density).value)
    t_init_code = t_init_kelvin * KI_PARAMS.code_temperature_per_kelvin
    dt_code = (dt_years * u.yr).to(CODE_UNITS.code_time).value

    cooling_curve_config = CoolingCurveConfig(cooling_curve_type=KOYAMA_INUTSUKA_NET_COOLING)

    # --- Part 1: confirm Newton converges to something reasonably close
    # (not exact -- see docstring) at this stiff dt, and is a real,
    # substantial improvement over the old naive solver's historical result.
    t_implicit_code = update_temperature_implicit(
        jnp.array(rho_code), jnp.array(t_init_code), dt_code, X_H, Z_METAL, GAMMA,
        cooling_curve_config, KI_PARAMS,
    )
    t_implicit_kelvin = float(t_implicit_code) / KI_PARAMS.code_temperature_per_kelvin

    ode_solution = integrate_isochoric_relaxation(n_h_cgs, t_init_kelvin, dt_years, MU, MU_H, GAMMA)
    t_reference_kelvin = float(ode_solution.y[0, -1])

    newton_rel_err = abs(t_implicit_kelvin - t_reference_kelvin) / t_reference_kelvin
    assert newton_rel_err < newton_rel_err_max, (
        f"Newton-based update_temperature_implicit should converge to "
        f"within {newton_rel_err_max} rel. err of the reference at this "
        f"stiff dt, but got rel. err {newton_rel_err:.4e} (implicit="
        f"{t_implicit_kelvin:.2f} K, reference={t_reference_kelvin:.2f} K) "
        f"-- either a real regression in the solver, or this stiff test "
        f"point's own truncation error grew for an unrelated reason and "
        f"the calibration needs revisiting."
    )

    # Hardcoded from the 2026-09-11 calibration of the old naive fixed-point
    # solver (no longer callable directly -- see this test's docstring) --
    # confirms Newton is a substantial, not marginal, improvement.
    old_naive_rel_err = 0.71966
    assert newton_rel_err < newton_vs_old_naive_rel_err_max * old_naive_rel_err, (
        f"Expected Newton's rel. err ({newton_rel_err:.4e}) to be well "
        f"below {newton_vs_old_naive_rel_err_max}x the old naive solver's "
        f"historical rel. err ({old_naive_rel_err}) at this same stiff "
        f"state -- if this fails, the Newton fix is not actually improving "
        f"on the old behavior here."
    )

    # --- Part 2: confirm _cfl_time_step's new dt_cool term catches this. ---
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=1,
        num_cells=NUM_CELLS,
        box_size=BOX_SIZE,
        cooling_config=CoolingConfig(
            cooling=True, cooling_method=IMPLICIT_COOLING, cooling_curve_config=cooling_curve_config
        ),
    )
    registered_variables = get_registered_variables(config)
    density = jnp.ones(NUM_CELLS) * rho_code
    zero = jnp.zeros(NUM_CELLS)
    gas_pressure = jnp.ones(NUM_CELLS) * float(get_pressure_from_temperature(rho_code, t_init_code, X_H, Z_METAL))
    state = construct_primitive_state(
        config=config, registered_variables=registered_variables, density=density, velocity_x=zero,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        gamma=GAMMA,
        cooling_params=CoolingParams(
            hydrogen_mass_fraction=X_H, metal_mass_fraction=Z_METAL,
            floor_temperature=FLOOR_TEMPERATURE_CODE, cooling_curve_params=KI_PARAMS,
        ),
    )

    dt_guarded = float(_cfl_time_step(state, config, params, registered_variables))

    config_no_cooling = config._replace(cooling_config=CoolingConfig(cooling=False))
    dt_hydro_only = float(_cfl_time_step(state, config_no_cooling, params, registered_variables))

    assert dt_guarded / dt_hydro_only < guarded_dt_ratio_max, (
        f"Cooling-guarded dt ({dt_guarded:.4e} code time) is not "
        f"meaningfully smaller than the pure-hydro dt ({dt_hydro_only:.4e} "
        f"code time) at this stiff state -- the new dt_cool term does not "
        f"appear to be active."
    )

    instantaneous_t_cool_code = (instantaneous_t_cool_years * u.yr).to(CODE_UNITS.code_time).value
    dt_cool_match_rel_err = abs(dt_guarded - C_CFL_DEFAULT * instantaneous_t_cool_code) / (
        C_CFL_DEFAULT * instantaneous_t_cool_code
    )
    assert dt_cool_match_rel_err < guarded_dt_match_rel_tol, (
        f"Cooling-guarded dt ({dt_guarded:.4e} code time) is not close to "
        f"C_cfl * the independently-computed instantaneous cooling time "
        f"({C_CFL_DEFAULT * instantaneous_t_cool_code:.4e} code time) -- "
        f"rel. err {dt_cool_match_rel_err:.4e} >= tol {guarded_dt_match_rel_tol}."
    )


if __name__ == "__main__":
    test_ki_cooling_relaxation()
    test_ki_cooling_cfl_guard()
