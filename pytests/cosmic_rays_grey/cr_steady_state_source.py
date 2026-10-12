"""
Steady-state CR diffusion with a source (Phase A test plan A4.4).

A time-constant source ``Q = Q0 sin(k x)`` on ``e_cr`` (the test-only hook
``CosmicRayGreyParams.test_e_cr_source``, added inside every RK stage) drives the
two-moment system to the steady state ``e = e_off + Q0 / (kappa k^2) sin(k x)``,
``F = -kappa grad(e)``. It is the isotropic analogue of the Sovinec test (A3.1) and
checks the discrete steady operator on its own, without the transient.

Spec (PROGRESS_PHASEA.md, 2026-10-12, corrected from the plan): the steady state is
independent of the time step (the in-stage source and the ETD2RK relaxation keep
``L(e) + Q = 0`` an exact fixed point of the step), but *not* of ``v_red``: the CR
rows' HLL dissipation ``R(tau) v_red / sqrt(3) (q_R - q_L)`` is part of the discrete
steady operator, so the steady state carries the A4.2 scheme error (MINMOD:
``~ v_red dx^2`` in optically thin cells, ``~ kappa dx`` in thick ones, ``tau =
(gamma_cr - 1) v_red dx / kappa``).

Setup: periodic box 1, ``gamma_cr = 4/3``, ``kappa = 0.1`` (5x item 4's, so the
slowest mode relaxes at ``kappa k^2 = 3.95`` and a steady state is affordable),
``k = 2 pi``, amplitude 0.5 on an offset 1 (units of ``E_SCALE``), ``C_cfl = 0.4``,
float64, static heavy gas (``rho = 1e4``, ``P = 1e-4``, ``E_SCALE = 1e-7``). Start from
the cell-averaged analytic state, run to ``t = 22 / (kappa k^2)`` (the initial
deviation, ~ the scheme error, decays by e^-22). Errors are L2 over cells of the
deviation from the cell-averaged analytic profile, per amplitude.

1. ``test_steady_state_dt_independence``: ``C_cfl`` 0.4 / 0.2 / 0.1 give the same
   steady state, in a thin (``v_red = 8``, N 128, ``tau`` 0.21) and a thick (``v_red =
   32``, N 32, ``tau`` 3.3) case, for ``e_cr`` and ``F_cr``; also the stage-wise
   backward-Euler relaxation (``CR_RELAXATION_STAGEWISE_IMPLICIT``), whose diffusive
   fixed point is exact as well (DESIGN.md Problem 3). And the run is steady: the
   last two snapshots agree.
2. ``test_steady_state_convergence``: ``v_red`` 4 / 8 / 16 / 32 at N 32 ... 1024 with
   MINMOD; MC (``DOUBLE_MINMOD``) at ``v_red`` 32 for comparison.
3. ``test_steady_state_vred_dependence``: the ``v_red`` dependence of the steady
   state, ``L2(e(v_red = 32) - e(v_red = 4))``, vanishes with N.
4. ``test_steady_state_2d``: the diagonal mode ``sin(k (x + y))`` in 2D (hydro), N
   32 / 64 / 128 at ``v_red = 8``: dt-independence and convergence.
5. ``test_steady_state_figure`` (``pics/04_isotropic_diffusion/
   cr_steady_state_source_test.svg``), no assertions.

Measured 2026-10-12 (GPU, float64; logs ``/export/scratch/nknoell/phaseA/a44/logs/``;
numbers in ``baselines.json``, ``cr_steady_state_source.py::*``, and PROGRESS_PHASEA.md):

- The steady state is a fixed point of the step: ``C_cfl`` 0.4 / 0.2 / 0.1 agree to
  <= 4e-13 of the amplitude, thin and thick, ETD2RK and stage-wise alike.
- MINMOD: second order once the cells are thin (2.00 at N 512 -> 1024 for every
  ``v_red``), first order where they are thick (``v_red = 32``, ``tau`` 3.3 -> 1.7:
  1.14); the error at fixed fine N grows ~ ``v_red^0.92``. MC: orders 2.1-2.4 at
  ``v_red = 32``, 4-20x smaller (N 32-256). This is A4.2's scheme error, now in the
  steady operator.
- The ``v_red`` dependence of the steady state converges at second order (1.99).
- 2D diagonal mode: orders 1.88 / 1.99, dt-independent to 3e-14.

Gates: see each test's docstring.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import sys
from functools import lru_cache
from pathlib import Path

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# baseline registry
from cr_baselines import Baselines

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    DOUBLE_MINMOD,
    FINITE_VOLUME,
    MINMOD,
    BoundarySettings,
    BoundarySettings1D,
    DOUBLE_PRECISION,
    PERIODIC_BOUNDARY,
    SnapshotSettings,
    finalize_config,
)

# astronomix functions
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CR_RELAXATION_ETD2RK,
    CR_RELAXATION_STAGEWISE_IMPLICIT,
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_transport import cr_flux_rows

jax.config.update("jax_enable_x64", True)

GAMMA_CR = 4.0 / 3.0
KAPPA = 0.1
BOX_SIZE = 1.0
WAVENUMBER = 2.0 * np.pi
AMPLITUDE = 0.5
E_OFFSET = 1.0
C_CFL = 0.4
RELAX_EFOLDS = 22.0
# static heavy gas and a small, offset e_cr (as in cr_telegrapher_reference.py)
GAS_DENSITY = 1e4
GAS_PRESSURE = 1e-4
E_SCALE = 1e-7

V_REDS = (4.0, 8.0, 16.0, 32.0)
RESOLUTIONS = (32, 64, 128, 256, 512, 1024)
MC_RESOLUTIONS = (32, 64, 128, 256)
CFLS = (0.4, 0.2, 0.1)
RESOLUTIONS_2D = (32, 64, 128)

PICS_DIR = Path(__file__).resolve().parent / "pics" / "04_isotropic_diffusion"
TEST_FILE = "cr_steady_state_source.py"


def _optical_depth(v_red: float, num_cells: int) -> float:
    """``tau = (gamma_cr - 1) v_red dx / kappa`` (``cr_wave_speed_reduction``)."""
    return (GAMMA_CR - 1.0) * v_red * BOX_SIZE / num_cells / KAPPA


def _sinc(z: float) -> float:
    return float(np.sinc(z / np.pi))


@lru_cache(maxsize=None)
def _steady_run(
    num_cells: int,
    v_red: float,
    C_cfl: float = C_CFL,
    limiter: int = MINMOD,
    integrator: int = CR_RELAXATION_ETD2RK,
    dimensionality: int = 1,
):
    """Run from the cell-averaged analytic steady state to ``t_end = RELAX_EFOLDS /
    rate`` with ``Q = Q0 sin(k . x)``, ``k`` along x (1D) or the diagonal (2D).

    Returns:
        dict with ``error_e`` / ``error_f`` (L2 / amplitude vs the analytic cell
        averages; ``F`` per ``kappa k`` amplitude), ``drift`` (max change of ``e``
        between the snapshots at ``0.9 t_end`` and ``t_end``, per amplitude),
        ``e`` / ``f`` (final fields in units of ``E_SCALE``, ``f`` stacked over
        axes), ``max_u`` and ``t_end``.
    """
    dx = BOX_SIZE / num_cells
    if dimensionality == 1:
        boundary_settings = BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        )
    else:
        periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
        boundary_settings = BoundarySettings(periodic, periodic)
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        limiter=limiter,
        dimensionality=dimensionality,
        num_cells=num_cells,
        box_size=BOX_SIZE,
        boundary_settings=boundary_settings,
        numerical_precision=DOUBLE_PRECISION,
        return_snapshots=True,
        num_snapshots=2,
        use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True,
            diffusive_relaxation=True,
            flux_relaxation_integrator=integrator,
        ),
    )
    registered_variables = get_registered_variables(config)
    flux_rows = cr_flux_rows(registered_variables)
    if dimensionality == 1:
        x = (np.arange(num_cells) + 0.5) * dx
        phase = WAVENUMBER * x
    else:
        centers = np.asarray(get_helper_data(config).geometric_centers)
        phase = WAVENUMBER * (centers[..., 0] + centers[..., 1])
    # cell averages: one sinc factor per axis along which the phase varies
    sinc = _sinc(0.5 * WAVENUMBER * dx) ** dimensionality
    k_squared = dimensionality * WAVENUMBER**2
    rate = KAPPA * k_squared
    q0 = AMPLITUDE * rate
    e_exact = E_OFFSET + AMPLITUDE * sinc * np.sin(phase)
    f_exact = -KAPPA * AMPLITUDE * WAVENUMBER * sinc * np.cos(phase)
    source = q0 * sinc * np.sin(phase)

    shape = (registered_variables.num_vars,) + (num_cells,) * dimensionality
    state = jnp.zeros(shape)
    state = state.at[registered_variables.density_index].set(GAS_DENSITY)
    state = state.at[registered_variables.pressure_index].set(GAS_PRESSURE)
    state = state.at[registered_variables.cosmic_ray_e_index].set(E_SCALE * e_exact)
    for row in flux_rows:
        state = state.at[row].set(E_SCALE * f_exact)
    config = finalize_config(config, state.shape)
    t_end = RELAX_EFOLDS / rate
    params = SimulationParams(
        t_end=t_end,
        C_cfl=C_cfl,
        snapshot_timepoints=jnp.array([0.9 * t_end, t_end]),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR,
            reduced_streaming_speed=v_red,
            diffusion_coefficient=KAPPA,
            test_e_cr_source=jnp.asarray(E_SCALE * source),
        ),
    )
    result = time_integration(state, config, params, registered_variables)
    states = np.asarray(result.states)
    assert not np.any(np.isnan(states)), f"NaNs at N = {num_cells}, v_red = {v_red}."
    e_cr = states[:, registered_variables.cosmic_ray_e_index] / E_SCALE
    f_cr = np.stack([states[-1, row] for row in flux_rows]) / E_SCALE
    velocity = registered_variables.velocity_index
    velocity_rows = (velocity,) if isinstance(velocity, int) else tuple(velocity)[:dimensionality]
    max_u = max(float(np.abs(states[:, row]).max()) for row in velocity_rows)
    f_scale = KAPPA * AMPLITUDE * WAVENUMBER
    return dict(
        error_e=float(np.sqrt(np.mean((e_cr[-1] - e_exact) ** 2))) / AMPLITUDE,
        error_f=float(np.sqrt(np.mean((f_cr - f_exact[None]) ** 2))) / f_scale,
        drift=float(np.abs(e_cr[-1] - e_cr[0]).max()) / AMPLITUDE,
        e=e_cr[-1],
        f=f_cr,
        max_u=max_u,
        t_end=t_end,
    )


def _max_diff(a: dict, b: dict) -> float:
    """Max difference of two steady states, ``e`` per amplitude and ``F`` per its
    amplitude, whichever is larger."""
    f_scale = KAPPA * AMPLITUDE * WAVENUMBER
    return max(
        float(np.abs(a["e"] - b["e"]).max()) / AMPLITUDE,
        float(np.abs(a["f"] - b["f"]).max()) / f_scale,
    )


def _order(n_coarse: int, n_fine: int, e_coarse: float, e_fine: float) -> float:
    return float(np.log(e_coarse / e_fine) / np.log(n_fine / n_coarse))


def _check_gas_at_rest(run: dict, num_cells: int, max_displacement: float = 1e-3):
    """The CR push on the heavy gas over the run stays far below a cell."""
    displacement = run["max_u"] * run["t_end"] / (BOX_SIZE / num_cells)
    assert displacement < max_displacement, f"gas moved {displacement:.1e} cells."


DT_CASES = ((8.0, 128), (32.0, 32))


def test_steady_state_dt_independence(max_dt_diff: float = 1e-10, max_drift: float = 1e-9):
    """The steady state is an exact fixed point of the step: independent of
    ``C_cfl`` (0.4 / 0.2 / 0.1) to round-off, for ``e_cr`` and ``F_cr``, in a thin
    and a thick case, with ETD2RK (default) and with the stage-wise relaxation.

    Gates: max difference over ``C_cfl`` <= ``max_dt_diff`` (per amplitude) for
    both integrators; the runs are steady (snapshot drift <= ``max_drift``); gas
    at rest.

    Args:
        max_dt_diff: Allowed difference between the steady states at different
            ``C_cfl``.
        max_drift: Allowed change of ``e`` over the last 10% of the run. Not
            round-off: the heavy gas is not in equilibrium (nothing balances
            ``grad P_cr``), it accelerates at ~1e-11 and its ``-P_cr div u`` work
            changes ``e`` by an estimated ~1e-10 over the run (measured drift
            6e-11 to 2e-10), identically at every ``C_cfl``.
    """
    failures = []
    for v_red, n in DT_CASES:
        for integrator, name in (
            (CR_RELAXATION_ETD2RK, "ETD2RK"),
            (CR_RELAXATION_STAGEWISE_IMPLICIT, "stage-wise"),
        ):
            runs = [_steady_run(n, v_red, C_cfl=c, integrator=integrator) for c in CFLS]
            diffs = [_max_diff(runs[0], r) for r in runs[1:]]
            drift = max(r["drift"] for r in runs)
            print(
                f"v_red {v_red:g}, N {n} (tau {_optical_depth(v_red, n):.2f}), {name}: "
                f"error {runs[0]['error_e']:.4e}; C_cfl 0.4 vs 0.2 / 0.1: "
                + " / ".join(f"{d:.2e}" for d in diffs) + f"; drift {drift:.1e}"
            )
            for r in runs:
                _check_gas_at_rest(r, n)
            if max(diffs) > max_dt_diff:
                failures.append(f"{name}, v_red {v_red:g}, N {n}: dt dependence {max(diffs):.2e}")
            if drift > max_drift:
                failures.append(f"{name}, v_red {v_red:g}, N {n}: not steady, drift {drift:.2e}")
    assert not failures, "; ".join(failures)


def test_steady_state_convergence(
    min_thin_order: float = 1.8,
    thick_order_range: tuple = (0.8, 1.3),
    min_mc_order: float = 1.8,
):
    """The steady-state error with MINMOD: second order in thin cells, first
    order in thick ones (the A4.2 scheme error); MC second order in both.

    Gates: pairwise order at N 512 -> 1024 >= ``min_thin_order`` for every
    ``v_red`` (``tau`` <= 0.1); ``v_red = 32`` at N 32 -> 64 (``tau`` 3.3 -> 1.7)
    within ``thick_order_range``; MC at ``v_red = 32``, N 32 -> 128, order >=
    ``min_mc_order``; every error within 1.5x of ``baselines.json``.
    """
    baselines = Baselines(f"{TEST_FILE}::test_steady_state_convergence")
    failures = []
    for v_red in V_REDS:
        errors = [_steady_run(n, v_red)["error_e"] for n in RESOLUTIONS]
        f_errors = [_steady_run(n, v_red)["error_f"] for n in RESOLUTIONS]
        orders = [_order(RESOLUTIONS[i], RESOLUTIONS[i + 1], errors[i], errors[i + 1])
                  for i in range(len(RESOLUTIONS) - 1)]
        print(
            f"MINMOD v_red {v_red:g}: e error " + " / ".join(f"{e:.3e}" for e in errors)
            + " (N " + " / ".join(str(n) for n in RESOLUTIONS) + "), orders "
            + " / ".join(f"{o:.2f}" for o in orders)
            + "; F error " + " / ".join(f"{e:.3e}" for e in f_errors)
            + "; tau " + " / ".join(f"{_optical_depth(v_red, n):.2g}" for n in RESOLUTIONS)
        )
        for n, e in zip(RESOLUTIONS, errors):
            baselines.upper(f"minmod_v{v_red:g}_n{n}", e)
        if orders[-1] < min_thin_order:
            failures.append(f"v_red {v_red:g}: thin order {orders[-1]:.2f} < {min_thin_order}")
        if v_red == 32.0 and not thick_order_range[0] <= orders[0] <= thick_order_range[1]:
            failures.append(f"v_red 32: thick order {orders[0]:.2f} outside {thick_order_range}")
        for n in RESOLUTIONS:
            _check_gas_at_rest(_steady_run(n, v_red), n)
    mc = [_steady_run(n, 32.0, limiter=DOUBLE_MINMOD)["error_e"] for n in MC_RESOLUTIONS]
    mc_orders = [_order(MC_RESOLUTIONS[i], MC_RESOLUTIONS[i + 1], mc[i], mc[i + 1])
                 for i in range(len(MC_RESOLUTIONS) - 1)]
    print(
        "MC v_red 32: e error " + " / ".join(f"{e:.3e}" for e in mc) + " (N "
        + " / ".join(str(n) for n in MC_RESOLUTIONS) + "), orders "
        + " / ".join(f"{o:.2f}" for o in mc_orders)
        + "; MINMOD / MC " + " / ".join(
            f"{_steady_run(n, 32.0)['error_e'] / e:.1f}" for n, e in zip(MC_RESOLUTIONS, mc))
    )
    for n, e in zip(MC_RESOLUTIONS, mc):
        baselines.upper(f"mc_v32_n{n}", e)
    if min(mc_orders[:2]) < min_mc_order:
        failures.append(f"MC orders {mc_orders[:2]} < {min_mc_order}")
    assert not failures, "; ".join(failures)
    baselines.check()


def test_steady_state_vred_dependence(min_order: float = 1.8, thin_slope_range: tuple = (0.8, 1.2)):
    """The ``v_red`` dependence of the steady state is a discretization error:
    ``L2(e(v_red = 32) - e(v_red = 4)) / amplitude`` vanishes with N, and in thin
    cells the error grows ~ ``v_red^1`` at fixed N (A4.2's ``v_red dx^2``).

    Gates: pairwise order of the ``v_red`` difference at N 512 -> 1024 >=
    ``min_order``; slope of the error vs ``v_red`` at N = 1024 within
    ``thin_slope_range``; the differences within 1.5x of ``baselines.json``.
    """
    baselines = Baselines(f"{TEST_FILE}::test_steady_state_vred_dependence")
    diffs = [
        float(np.sqrt(np.mean((_steady_run(n, 32.0)["e"] - _steady_run(n, 4.0)["e"]) ** 2))) / AMPLITUDE
        for n in RESOLUTIONS
    ]
    orders = [_order(RESOLUTIONS[i], RESOLUTIONS[i + 1], diffs[i], diffs[i + 1])
              for i in range(len(RESOLUTIONS) - 1)]
    thin = [_steady_run(RESOLUTIONS[-1], v)["error_e"] for v in V_REDS]
    slope = float(np.polyfit(np.log(V_REDS), np.log(thin), 1)[0])
    print(
        "L2(e(v_red 32) - e(v_red 4)) / amplitude: " + " / ".join(f"{d:.3e}" for d in diffs)
        + " (N " + " / ".join(str(n) for n in RESOLUTIONS) + "), orders "
        + " / ".join(f"{o:.2f}" for o in orders)
        + f"; N {RESOLUTIONS[-1]} error vs v_red: " + " / ".join(f"{e:.3e}" for e in thin)
        + f", slope {slope:.3f}"
    )
    for n, d in zip(RESOLUTIONS, diffs):
        baselines.upper(f"vred_diff_n{n}", d)
    assert orders[-1] >= min_order, f"v_red dependence converges at order {orders[-1]:.2f}."
    assert thin_slope_range[0] <= slope <= thin_slope_range[1], f"thin-cell slope vs v_red {slope:.3f}."
    baselines.check()


def test_steady_state_2d(
    max_dt_diff: float = 1e-10, min_order: float = 1.8, max_drift: float = 1e-9, v_red: float = 8.0
):
    """The diagonal mode ``sin(k (x + y))`` in 2D: the multi-D steady operator
    (both axes' fluxes and the ETD2RK relaxation of both ``F_cr`` components).

    Gates: ``C_cfl`` 0.4 vs 0.2 at N = 64 <= ``max_dt_diff``; pairwise order at N
    64 -> 128 >= ``min_order``; snapshot drift <= ``max_drift`` (see
    :func:`test_steady_state_dt_independence`); errors within 1.5x of
    ``baselines.json``; gas at rest.
    """
    baselines = Baselines(f"{TEST_FILE}::test_steady_state_2d")
    runs = [_steady_run(n, v_red, dimensionality=2) for n in RESOLUTIONS_2D]
    errors = [r["error_e"] for r in runs]
    orders = [_order(RESOLUTIONS_2D[i], RESOLUTIONS_2D[i + 1], errors[i], errors[i + 1])
              for i in range(len(RESOLUTIONS_2D) - 1)]
    dt_diff = _max_diff(_steady_run(64, v_red, dimensionality=2),
                        _steady_run(64, v_red, C_cfl=0.2, dimensionality=2))
    drift = max(r["drift"] for r in runs)
    print(
        f"2D diagonal mode, v_red {v_red:g}: e error " + " / ".join(f"{e:.3e}" for e in errors)
        + " (N " + " / ".join(str(n) for n in RESOLUTIONS_2D) + "), orders "
        + " / ".join(f"{o:.2f}" for o in orders)
        + f"; F error " + " / ".join(f"{r['error_f']:.3e}" for r in runs)
        + f"; C_cfl 0.4 vs 0.2 at N 64: {dt_diff:.2e}; drift {drift:.1e}"
    )
    for n, r in zip(RESOLUTIONS_2D, runs):
        _check_gas_at_rest(r, n)
        baselines.upper(f"v{v_red:g}_n{n}", r["error_e"])
    assert dt_diff <= max_dt_diff, f"2D steady state depends on dt: {dt_diff:.2e}."
    assert drift <= max_drift, f"2D run not steady: drift {drift:.2e}."
    assert orders[-1] >= min_order, f"2D order {orders[-1]:.2f} < {min_order}."
    baselines.check()


def test_steady_state_figure():
    """Figure (no assertions): steady-state error vs N per ``v_red`` (MINMOD,
    MC), and the ``v_red`` dependence."""
    fig, (ax_n, ax_v) = plt.subplots(1, 2, figsize=(11, 4.2))
    n = np.array(RESOLUTIONS)
    for i, v in enumerate(V_REDS):
        ax_n.loglog(n, [_steady_run(m, v)["error_e"] for m in RESOLUTIONS], "o-", color=f"C{i}",
                    label=rf"MINMOD, $v_{{red}}$ = {v:g}")
    ax_n.loglog(MC_RESOLUTIONS, [_steady_run(m, 32.0, limiter=DOUBLE_MINMOD)["error_e"] for m in MC_RESOLUTIONS],
                "s--", color="C3", alpha=0.7, label=r"MC, $v_{red}$ = 32")
    ref = _steady_run(RESOLUTIONS[0], 32.0)["error_e"]
    ax_n.loglog(n, ref * (n / n[0]) ** -1, "k:", lw=0.8, label=r"$N^{-1}$")
    ax_n.loglog(n, ref * (n / n[0]) ** -2, "k-.", lw=0.8, label=r"$N^{-2}$")
    ax_n.set_xlabel("N")
    ax_n.set_ylabel("L2 error / amplitude (steady state)")
    ax_n.set_title(r"Steady state with source, $\kappa$ = 0.1")
    ax_n.legend(fontsize=7)
    v = np.array(V_REDS)
    for j, m in enumerate((64, 256, 1024)):
        ax_v.loglog(v, [_steady_run(m, x)["error_e"] for x in V_REDS], "o-", color=f"C{j}",
                    label=f"N = {m}")
    ax_v.loglog(v, _steady_run(1024, 4.0)["error_e"] * v / v[0], "k:", lw=0.8, label=r"$\propto v_{red}$")
    ax_v.set_xlabel(r"$v_{red}$")
    ax_v.set_ylabel("L2 error / amplitude")
    ax_v.set_title(r"$v_{red}$ dependence of the steady state (MINMOD)")
    ax_v.legend(fontsize=7)
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_steady_state_source_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    tests = (
        test_steady_state_dt_independence,
        test_steady_state_convergence,
        test_steady_state_vred_dependence,
        test_steady_state_2d,
        test_steady_state_figure,
    )
    selected = sys.argv[1:]
    if selected:
        tests = tuple(t for t in tests if t.__name__ in selected)
    failed = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as error:
            failed.append(test.__name__)
            print(f"FAIL {test.__name__}: {error}")
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
