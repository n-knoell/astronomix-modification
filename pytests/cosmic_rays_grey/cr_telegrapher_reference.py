"""
Isotropic CR diffusion against the exact two-moment ("telegrapher") solution
(Phase A test plan A4.1), and the ``v_red`` scan of the model floor (A4.2).

Ladder item 4 (``cr_isotropic_diffusion_convergence.py``) compares the
two-moment scheme with the diffusion Green's function. That mixes two errors:
the discretization error and the model difference between the two-moment
system at finite ``v_red`` and pure diffusion, which item 4 sees as a floor of
~1.2e-4. Here the reference is the exact solution of the two-moment system
itself (``test_setups/reference_solutions/linear_theory.py``): the FFT of the
discrete initial ``e_cr`` (``F_cr = 0``), every mode evolved with its exact 2x2
propagator, inverse FFT. The error against it is purely discretization.

Model, with ``c^2 = (gamma_cr - 1) v_red^2`` and ``nu = c^2 / kappa``:
``d(e)/dt + dF/dx = 0``, ``d(F)/dt + c^2 de/dx = -nu F``. A mode ``k`` has the
roots ``s = (-nu +- sqrt(nu^2 - 4 c^2 k^2)) / 2``: overdamped below
``k_c = nu / (2 c)``, a damped wave (rate ``nu / 2``) above.

Setup: item 4's parameters (``gamma_cr = 4/3``, ``v_red = 8``, ``kappa = 0.02``,
box 1, periodic, ``C_cfl = 0.4``; ``nu = 1067``, ``k_c = 115.5``, i.e. mode
number 18.4), float64, static heavy gas (``rho = 1e4``, ``P = 1e-4``,
``e_cr = 1e-7 (0.1 + profile)``; the offset keeps ``e_cr`` far above
``minimum_e_cr``, the scale keeps the CR push on the gas below 1e-10; max abs(u)
2.7e-12 measured).

1. ``test_telegrapher_model_floor`` (no simulation): the telegrapher-vs-diffusion
   difference for item 4's Gaussian at ``v_red`` = 4 ... 64.
2. ``test_cr_gaussian_vs_telegrapher``: item 4's Gaussian (``sigma0 = 0.02``,
   ``t_end = 0.24``) at N = 128 ... 4096 against both references.
3. ``test_cr_single_mode_vs_telegrapher``: ``e = 1 + cos(2 pi m x) / 2`` with
   ``m`` = 2, 8, 18, 24, 48 (``k / k_c`` = 0.11, 0.44, 0.98, 1.31, 2.61) until
   three e-folds of the slow root, error = max over 24 snapshots of
   L2(e - e_ref) / 0.5, at the default ``C_cfl = 0.4`` and at 0.1.
4. ``test_cr_single_mode_second_order``: the plan's expectation (second order)
   for the single modes at the default ``C_cfl``.
5. ``test_cr_relaxation_integrators``: ETD2RK against the stage-wise backward-Euler
   relaxation used until 2026-10-10 (``CR_RELAXATION_STAGEWISE_IMPLICIT``), on m = 24.
6. A4.2 (``test_vred_scan_*``): item 4's Gaussian at ``v_red`` = 4 ... 64 and N = 128 ...
   8192, against the diffusion Green's function and the telegrapher reference; the
   default MINMOD limiter, plus MC (``DOUBLE_MINMOD``) for comparison.

All runs use the default ``flux_relaxation_integrator`` (ETD2RK, since 2026-10-10)
unless stated.

Measured 2026-10-10 (GPU, float64; logs ``/export/scratch/nknoell/phaseA/a41/logs/`` and
``/export/scratch/nknoell/phaseA/regress_etd/logs/``):

1. Model floor (L2 / amplitude, telegrapher - diffusion at ``t = 0.24``): 4.41e-4 /
   1.08e-4 / 2.69e-5 / 6.71e-6 / 1.68e-6 at ``v_red`` = 4 / 8 / 16 / 32 / 64, slope
   -2.008. At ``v_red = 8`` it is item 4's ~1.2e-4 floor: the floor is the model.
2. Gaussian, L2 / amplitude at N = 128 / 256 / 512 / 1024 / 2048 / 4096:
   - vs telegrapher: 1.64e-3 / 4.87e-4 / 1.26e-4 / 3.18e-5 / 7.94e-6 / 1.98e-6,
     least-squares order 1.93 (N 128-2048), pairwise 1.75 / 1.95 / 1.99 / 2.00 / 2.00;
     ``F_cr`` 5.6e-4 ... 7.0e-7;
   - vs diffusion: 1.67e-3 ... 1.08e-4, i.e. 1.004x the model floor at N = 4096.
   - No floor against the telegrapher reference: item 4's flattening is the model, not
     the scheme.
   - Stage-wise relaxation (before 2026-10-10): 1.66e-3 ... 3.42e-6, order 1.86,
     pairwise down to 1.55 at 4096 (its time error).
3. Single modes, max-in-time L2 / amplitude at N = 128 ... 2048, ETD2RK:

   ======  =======  =============================================  ==============
   m       k/k_c    C_cfl 0.4 (default)                             C_cfl 0.1
   ======  =======  =============================================  ==============
   2       0.11     7.1e-3 ... 3.27e-5, order 1.81 ... 2.00         3.26e-5
   8       0.44     3.4e-2 ... 1.56e-4, order 1.79 ... 2.00         1.52e-4
   18      0.98     8.3e-2 ... 5.06e-4, order 1.47 ... 1.95         5.06e-4
   24      1.31     1.2e-1 ... 8.64e-4, order 1.34 ... 1.89         8.55e-4
   48      2.61     2.8e-1 ... 3.11e-3, order 1.10 ... 1.95         3.02e-3
   ======  =======  =============================================  ==============

   Second order at fine N and independent of ``C_cfl`` to 1-3%: the error is spatial.

**History: the stage-wise F_cr relaxation was first order in time.** Until 2026-10-10
``evolve_state`` applied the implicit relaxation ``F <- F / (1 + nu dt)`` at the end of
each SSP-RK2 stage and averaged the stages. For pure relaxation that is the step factor
``(1 + 1 / (1 + a)^2) / 2 = 1 - a + 3 a^2 / 2`` (``a = nu dt``) against ``exp(-a) = 1 - a
+ a^2 / 2``: a local error ``a^2``, a global error ``O(nu dt)``. A steady, gradient-slaved
``F_cr`` is still an exact fixed point (DESIGN.md Problem 3), so the diffusion limit was
unaffected; the error showed in transients and for every mode that is not ``<< k_c``.
With it, the single modes at ``C_cfl = 0.4`` converged at first order at fine N (pairwise
0.92 / 0.71 / 0.69 / 0.44 for m = 8 / 18 / 24 / 48 at 1024 -> 2048; N = 2048 errors
1.09e-3 / 3.59e-3 / 4.75e-3 / 9.91e-3, 3-7x the ETD2RK ones), with errors ~linear in
``C_cfl`` above ~0.1. ETD2RK (``cr_grey_sources.cr_flux_relaxation_etd2rk``; DESIGN.md
"F_cr relaxation time accuracy: ETD2RK") fixed it; test 4 was a strict expected failure
until then. Test 5 keeps the old scheme's first-order behavior visible (m = 24: order
0.69, ETD2RK 5.5x more accurate at N = 2048).

**A4.2, measured 2026-10-11** (GPU, float64, ETD2RK, ``C_cfl = 0.4``; logs
``/export/scratch/nknoell/phaseA/a42/logs/``). L2 / amplitude; ``tau = (gamma_cr - 1) v_red
dx / kappa`` is the cell size in CR mean free paths (``cr_wave_speed_reduction``'s optical
depth):

- **Error against diffusion = model floor + scheme error.**

  ===========  ==========================================  ===========  ===========
  ``v_red``    floor (telegrapher - diffusion)              N = 2048     N = 8192
  ===========  ==========================================  ===========  ===========
  4            4.41e-4                                      4.42e-4      4.41e-4
  8            1.08e-4                                      1.10e-4      1.08e-4
  16           2.69e-5                                      3.34e-5      2.71e-5
  32           6.71e-6                                      3.21e-5      7.29e-6
  ===========  ==========================================  ===========  ===========

  - Fitted slope ``-1.975`` at N = 8192 (plan gate ``-2 +- 0.3``).
  - At the plan's N = 2048 the slope is ``-1.31``. There the scheme error against the
    telegrapher reference (4.2e-6 / 7.9e-6 / 1.5e-5 / 3.0e-5) grows with ``v_red`` and
    overtakes the falling floor near ``v_red ~ 16``. This combination is also what made item
    4's floor look like ``v_red^-1`` (plan Sec. 4.3 step 3, now closed).
- **Scheme error with MINMOD (the default and the production limiter):**
  - It is spatial: ``C_cfl`` 0.1 and 0.4 agree to < 1% (v_red 8 and 32, N 128 ... 8192).
  - Optically thin cells (``tau << 1``): second order, ``~ v_red^1`` at fixed N (slope 0.956 at
    N = 8192: 2.6e-7 / 5.0e-7 / 9.7e-7 / 1.9e-6).
  - Optically thick cells (``tau >~ 2``): independent of ``v_red`` (1.977e-3 / 1.983e-3 /
    1.980e-3 at N = 128, ``tau`` = 2.1 / 4.2 / 8.3) and **first order** (``v_red = 64``: orders
    1.02 / 1.04 over N = 128 ... 512).
- **Cause: the HLL dissipation of the CR rows times minmod's face jumps.**
  - Minmod takes one-sided slopes, which are first-order accurate. The face jumps ``q_R -
    q_L`` are therefore ``O(dx^2 q'')``, and the dissipative flux ``R(tau) c q_jump / 2``
    contributes ``O(R c dx^2)``.
  - With ``R ~ 1`` (thin cells) that is ``~ v_red dx^2``.
  - With ``R ~ 1 / tau ~ kappa / (v_red dx)`` (thick cells) it is ``~ kappa dx``, first
    order and ``v_red``-independent.
  - Central slopes (MC) have ``O(dx^3)`` jumps, i.e. ``O(R c dx^3)``, which stays second order
    when the cells are thick.
  - Measured with MC (``DOUBLE_MINMOD``):
    - ``v_red = 64`` (``tau`` 4.2 / 2.1 / 1.0): 6.5e-5 / 1.6e-5 / 3.7e-6, orders 2.01 / 2.14,
      vs MINMOD 9.8e-4 / 4.8e-4 / 1.9e-4.
    - At N = 2048: 15x smaller than MINMOD at ``v_red = 8`` (5.2e-7 vs 7.9e-6), and nearly
      ``v_red``-independent (``v_red`` 64 / 8: 1.42, MINMOD 7.20). This is the plan's A4.2
      expectation, which MINMOD does not meet.
  - Van Albada (prototype, same logs) is within 10% of MC (1% at N = 2048), but only because
    its unscaled ``3 dx`` epsilon makes it the unlimited central slope for e_cr in code units.
  - HLLC gives the same CR-row fluxes as HLL (``hll._grey_cr_hll_rows``): identical errors.
- **Relevance:** M5-M7 run MINMOD. M7 at ``v_red = 1e4`` km/s has ``tau ~ 3`` (``kappa =
  1e28``, dx 31 pc), so its CR diffusion is in the first-order regime. Relative to the bump
  RMS that costs 0.6-2.4% at 13-50 cells per sigma. Decision 2026-10-12: MINMOD kept
  (DESIGN.md "Decided: CR diffusion accuracy and the slope limiter").

Gates: 1 slope -2 +- 0.05; 2 order >= 1.8 (plan A4.1), error at every N within 1.5x of
``baselines.json``, finest vs-diffusion error = model floor within 5%, gas at rest; 3
tracked bounds (error decreasing from N = 256, N = 2048 within 1.5x of ``baselines.json``);
4 pairwise order >= 1.8 at N 1024 -> 2048 for m >= 8; 5 stage-wise order <= 1.2
and ETD2RK >= 3x more accurate at N = 2048. A4.2:
- floor: slope vs diffusion at N = 8192 over ``v_red`` 4-32 is ``-2 +- 0.3`` (plan), and
  at N = 2048 the scheme error exceeds the floor at ``v_red = 32`` (the documented
  explanation of the shallower slope there);
- scheme error (MINMOD): thin-cell slope vs ``v_red`` ``1 +- 0.2``; thick cells
  ``v_red``-independent within 5% and orders in [0.8, 1.2];
- limiter: MC thick-cell orders >= 1.8, MINMOD / MC >= 5 at ``v_red = 8``, MC ratio ``v_red``
  64 / 8 <= 2;
- ``test_vred_scan_figure`` (``pics/04_isotropic_diffusion/cr_telegrapher_vred_scan_test.svg``)
  has no assertions.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
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
    BoundarySettings1D,
    DOUBLE_PRECISION,
    PERIODIC_BOUNDARY,
    SnapshotSettings,
    finalize_config,
)

# astronomix functions
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.test_setups.reference_solutions.linear_theory import (
    telegrapher_critical_wavenumber,
    telegrapher_rates,
    telegrapher_spectral_solution,
)

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CR_RELAXATION_ETD2RK,
    CR_RELAXATION_STAGEWISE_IMPLICIT,
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

jax.config.update("jax_enable_x64", True)

# item 4's parameters (cr_isotropic_diffusion_convergence.py)
GAMMA_CR = 4.0 / 3.0
V_RED = 8.0
KAPPA = 0.02
BOX_SIZE = 1.0
SIGMA0 = 0.02
T_END = 0.24
C_CFL = 0.4
# static heavy gas and a small, offset e_cr (see the module docstring)
GAS_DENSITY = 1e4
GAS_PRESSURE = 1e-4
E_SCALE = 1e-7
E_OFFSET = 0.1

GAUSS_RESOLUTIONS = (128, 256, 512, 1024, 2048, 4096)
MODE_NUMBERS = (2, 8, 18, 24, 48)
MODE_RESOLUTIONS = (128, 256, 512, 1024, 2048)
MODE_CFLS = (C_CFL, 0.1)
MODE_AMPLITUDE = 0.5
MODE_SNAPSHOTS = 24
FLOOR_V_REDS = (4.0, 8.0, 16.0, 32.0, 64.0)

PICS_DIR = Path(__file__).resolve().parent / "pics" / "04_isotropic_diffusion"


def _cell_centers(num_cells: int) -> np.ndarray:
    return (np.arange(num_cells) + 0.5) * BOX_SIZE / num_cells


def _gaussian(x: np.ndarray, sigma: float) -> np.ndarray:
    return SIGMA0 / sigma * np.exp(-0.5 * ((x - 0.5 * BOX_SIZE) / sigma) ** 2)


def _mode_t_end(mode: int) -> float:
    """Three e-folds of the slow root, at most ``T_END``."""
    s_slow, _ = telegrapher_rates(2.0 * np.pi * mode / BOX_SIZE, GAMMA_CR, V_RED, KAPPA)
    return float(min(T_END, 3.0 / abs(s_slow.real)))


def _run(
    num_cells: int,
    profile: np.ndarray,
    t_end: float,
    C_cfl: float,
    timepoints=None,
    integrator: int = CR_RELAXATION_ETD2RK,
    v_red: float = V_RED,
    limiter: int = MINMOD,
):
    """Evolve ``e_cr = E_SCALE * profile`` (``F_cr = 0``) in static heavy gas,
    with the given ``flux_relaxation_integrator``, ``reduced_streaming_speed``
    and slope limiter.

    Returns:
        ``(times, e_cr, F_cr, max abs(u))`` with ``e_cr`` / ``F_cr`` of shape
        ``(len(times), num_cells)`` in units of ``E_SCALE``.
    """
    snapshots = timepoints is not None
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        limiter=limiter,
        dimensionality=1,
        num_cells=num_cells,
        box_size=BOX_SIZE,
        boundary_settings=BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        numerical_precision=DOUBLE_PRECISION,
        return_snapshots=snapshots,
        num_snapshots=len(timepoints) if snapshots else 10,
        use_specific_snapshot_timepoints=snapshots,
        snapshot_settings=SnapshotSettings(return_states=True),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True,
            diffusive_relaxation=True,
            flux_relaxation_integrator=integrator,
        ),
    )
    registered_variables = get_registered_variables(config)
    state = jnp.zeros((registered_variables.num_vars, num_cells))
    state = state.at[registered_variables.density_index].set(GAS_DENSITY)
    state = state.at[registered_variables.pressure_index].set(GAS_PRESSURE)
    state = state.at[registered_variables.cosmic_ray_e_index].set(E_SCALE * profile)
    # F_cr = 0 initial.
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=t_end,
        C_cfl=C_cfl,
        snapshot_timepoints=jnp.array(timepoints if snapshots else [t_end]),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR, reduced_streaming_speed=v_red, diffusion_coefficient=KAPPA
        ),
    )
    result = time_integration(state, config, params, registered_variables)
    if snapshots:
        states = np.asarray(result.states)
        times = np.asarray(result.time_points)
    else:
        states = np.asarray(result)[None]
        times = np.array([t_end])
    assert not np.any(np.isnan(states)), f"NaNs at N = {num_cells}."
    e_cr = states[:, registered_variables.cosmic_ray_e_index] / E_SCALE
    f_cr = states[:, registered_variables.cosmic_ray_flux_index] / E_SCALE
    max_u = float(np.abs(states[:, registered_variables.velocity_index]).max())
    return times, e_cr, f_cr, max_u


def _l2(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a**2)))


def _ls_order(resolutions, errors) -> float:
    """Least-squares convergence order of ``errors`` vs ``resolutions``."""
    return float(-np.polyfit(np.log(resolutions), np.log(errors), 1)[0])


def _model_floor(v_red: float, num_cells: int = 4096) -> float:
    """L2(telegrapher - diffusion) / amplitude for item 4's Gaussian at ``T_END``."""
    x = _cell_centers(num_cells)
    e_tel, _ = telegrapher_spectral_solution(
        _gaussian(x, SIGMA0), BOX_SIZE, T_END, GAMMA_CR, v_red, KAPPA
    )
    sigma_end = np.sqrt(SIGMA0**2 + 2.0 * KAPPA * T_END)
    return _l2(e_tel - _gaussian(x, sigma_end))


@lru_cache(maxsize=None)
def _gaussian_run(num_cells: int):
    x = _cell_centers(num_cells)
    profile = E_OFFSET + _gaussian(x, SIGMA0)
    _, e_cr, f_cr, max_u = _run(num_cells, profile, T_END, C_CFL)
    e_tel, f_tel = telegrapher_spectral_solution(profile, BOX_SIZE, T_END, GAMMA_CR, V_RED, KAPPA)
    sigma_end = np.sqrt(SIGMA0**2 + 2.0 * KAPPA * T_END)
    e_diffusion = E_OFFSET + _gaussian(x, sigma_end)
    return dict(
        x=x,
        e_cr=e_cr[-1],
        e_tel=e_tel,
        e_diffusion=e_diffusion,
        error_tel=_l2(e_cr[-1] - e_tel),
        error_flux_tel=_l2(f_cr[-1] - f_tel[0]),
        error_diffusion=_l2(e_cr[-1] - e_diffusion),
        max_u=max_u,
    )


@lru_cache(maxsize=None)
def _mode_run(mode: int, num_cells: int, C_cfl: float, integrator: int = CR_RELAXATION_ETD2RK):
    x = _cell_centers(num_cells)
    profile = 1.0 + MODE_AMPLITUDE * np.cos(2.0 * np.pi * mode * x / BOX_SIZE)
    t_end = _mode_t_end(mode)
    timepoints = np.linspace(0.0, t_end, MODE_SNAPSHOTS + 1)[1:]
    times, e_cr, _, max_u = _run(num_cells, profile, t_end, C_cfl, timepoints, integrator)
    cos_mode = np.cos(2.0 * np.pi * mode * x / BOX_SIZE)
    errors, amplitudes, amplitudes_ref = [], [], []
    for time, e in zip(times, e_cr):
        e_ref, _ = telegrapher_spectral_solution(profile, BOX_SIZE, time, GAMMA_CR, V_RED, KAPPA)
        errors.append(_l2(e - e_ref) / MODE_AMPLITUDE)
        amplitudes.append(2.0 * np.mean(e * cos_mode) / MODE_AMPLITUDE)
        amplitudes_ref.append(2.0 * np.mean(e_ref * cos_mode) / MODE_AMPLITUDE)
    return dict(
        times=times,
        error=float(max(errors)),
        amplitude=np.array(amplitudes),
        amplitude_ref=np.array(amplitudes_ref),
        max_u=max_u,
    )


def test_telegrapher_model_floor(slope_tol: float = 0.05):
    """The telegrapher-vs-diffusion difference scales as ``v_red^-2`` (plan A4.1,
    closing Sec. 4.3 step 3), and at ``v_red = 8`` it is item 4's floor.

    Args:
        slope_tol: Allowed deviation of the fitted log-log slope from -2.
    """
    floors = [_model_floor(v_red) for v_red in FLOOR_V_REDS]
    slope = float(np.polyfit(np.log(FLOOR_V_REDS), np.log(floors), 1)[0])
    print(
        "telegrapher - diffusion, L2 / amplitude: "
        + ", ".join(f"v_red {v:g}: {f:.3e}" for v, f in zip(FLOOR_V_REDS, floors))
        + f"; slope {slope:.3f}"
    )
    assert abs(slope + 2.0) < slope_tol, f"model floor slope {slope:.3f}, expected -2."


def test_cr_gaussian_vs_telegrapher(
    min_order: float = 1.8,
    floor_rel_tol: float = 0.05,
    max_gas_speed: float = 1e-10,
):
    """Item 4's Gaussian against the exact two-moment solution.

    The L2 error against the telegrapher reference at every N is gated at 1.5x of
    ``baselines.json``.

    Args:
        min_order: Minimum least-squares order over N = 128-2048 against the
            telegrapher reference (plan A4.1).
        floor_rel_tol: The error against the diffusion solution at the finest N
            must equal the model floor (:func:`_model_floor`) to this relative
            tolerance: the item-4 floor is the model, not the scheme.
        max_gas_speed: The gas must stay at rest.
    """
    runs = {n: _gaussian_run(n) for n in GAUSS_RESOLUTIONS}
    error_tel = np.array([runs[n]["error_tel"] for n in GAUSS_RESOLUTIONS])
    error_diffusion = np.array([runs[n]["error_diffusion"] for n in GAUSS_RESOLUTIONS])
    error_flux = np.array([runs[n]["error_flux_tel"] for n in GAUSS_RESOLUTIONS])
    fit = [n <= 2048 for n in GAUSS_RESOLUTIONS]
    order = _ls_order(np.array(GAUSS_RESOLUTIONS)[fit], error_tel[fit])
    floor = _model_floor(V_RED)
    floor_rel = abs(error_diffusion[-1] - floor) / floor
    max_u = max(runs[n]["max_u"] for n in GAUSS_RESOLUTIONS)
    pairwise = np.log2(error_tel[:-1] / error_tel[1:])

    for n, et, ef, ed in zip(GAUSS_RESOLUTIONS, error_tel, error_flux, error_diffusion):
        print(f"N = {n:5d}: L2 vs telegrapher e {et:.3e} F {ef:.3e}, vs diffusion {ed:.3e}")
    print(
        f"order vs telegrapher (N 128-2048) {order:.3f}, pairwise "
        + " / ".join(f"{p:.2f}" for p in pairwise)
        + f"; model floor {floor:.3e}, finest vs-diffusion error / floor - 1 = {floor_rel:+.3f}; "
        f"max abs(u) {max_u:.1e}"
    )

    fig, (ax_profile, ax_conv, ax_floor) = plt.subplots(1, 3, figsize=(17, 5))
    coarse = runs[GAUSS_RESOLUTIONS[0]]
    ax_profile.plot(coarse["x"], coarse["e_tel"], "k-", lw=1.5, label="telegrapher (exact)")
    ax_profile.plot(coarse["x"], coarse["e_diffusion"], "C1:", lw=1.5, label="diffusion")
    ax_profile.plot(coarse["x"], coarse["e_cr"], "C0--", label=f"scheme, N = {GAUSS_RESOLUTIONS[0]}")
    ax_profile.set_xlim(0.1, 0.9)
    ax_profile.set_xlabel("x")
    ax_profile.set_ylabel(r"$e_{cr}$ / scale")
    ax_profile.set_title(f"Gaussian at t = {T_END}")
    ax_profile.legend()
    ax_conv.loglog(GAUSS_RESOLUTIONS, error_tel, "o-", label="vs telegrapher")
    ax_conv.loglog(GAUSS_RESOLUTIONS, error_diffusion, "s-", label="vs diffusion")
    ax_conv.axhline(floor, color="C1", ls=":", label="model floor (telegrapher - diffusion)")
    ref_n = np.array(GAUSS_RESOLUTIONS, dtype=float)
    ax_conv.loglog(ref_n, error_tel[0] * (ref_n / ref_n[0]) ** -2, "k:", lw=0.8, label=r"$N^{-2}$")
    ax_conv.set_xlabel("N")
    ax_conv.set_ylabel("L2 error / amplitude")
    ax_conv.set_title(f"Order vs telegrapher (N 128-2048) = {order:.2f}")
    ax_conv.legend()
    floors = [_model_floor(v) for v in FLOOR_V_REDS]
    ax_floor.loglog(FLOOR_V_REDS, floors, "o-", label="telegrapher - diffusion")
    v = np.array(FLOOR_V_REDS)
    ax_floor.loglog(v, floors[1] * (v / v[1]) ** -2, "k:", lw=0.8, label=r"$v_{red}^{-2}$")
    ax_floor.set_xlabel(r"$v_{red}$")
    ax_floor.set_ylabel("L2 / amplitude")
    ax_floor.set_title("Model floor of item 4")
    ax_floor.legend()
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_telegrapher_reference_gaussian_test.svg")
    plt.close(fig)

    assert max_u < max_gas_speed, f"gas not at rest: max abs(u) {max_u:.1e}."
    assert order >= min_order, f"order vs telegrapher {order:.3f} < {min_order}: {error_tel}."
    assert floor_rel <= floor_rel_tol, (
        f"finest error vs diffusion {error_diffusion[-1]:.3e} is not the model floor {floor:.3e}."
    )
    baselines = Baselines("cr_telegrapher_reference.py::test_cr_gaussian_vs_telegrapher")
    for n, error in zip(GAUSS_RESOLUTIONS, error_tel):
        baselines.upper(f"l2_tel_n{n}", float(error))
    baselines.check()


def test_cr_single_mode_vs_telegrapher(baseline_factor: float = 1.5, max_gas_speed: float = 1e-10):
    """Single Fourier modes across ``k_c`` against their exact evolution.

    Tracked bounds: the max-in-time error decreases with N (from N = 256) and
    stays below ``baseline_factor`` x the measured value at N = 2048, for both
    ``C_cfl`` (the measured values are in ``baselines.json``).

    Args:
        baseline_factor: Allowed growth over the baseline.
        max_gas_speed: The gas must stay at rest.
    """
    k_c = telegrapher_critical_wavenumber(GAMMA_CR, V_RED, KAPPA)
    errors = {
        (m, c): np.array([_mode_run(m, n, c)["error"] for n in MODE_RESOLUTIONS])
        for m in MODE_NUMBERS
        for c in MODE_CFLS
    }
    failures = []
    baselines = Baselines("cr_telegrapher_reference.py::test_cr_single_mode_vs_telegrapher")
    for (m, c), err in errors.items():
        pairwise = np.log2(err[:-1] / err[1:])
        print(
            f"m = {m:2d} (k/k_c {2 * np.pi * m / BOX_SIZE / k_c:.2f}, t_end {_mode_t_end(m):.4g}), "
            f"C_cfl {c}: max_t L2 / amplitude "
            + ", ".join(f"{e:.3e}" for e in err)
            + "; pairwise orders "
            + " / ".join(f"{p:.2f}" for p in pairwise)
        )
        if np.any(np.diff(err[1:]) >= 0):
            failures.append(f"m = {m}, C_cfl {c}: error not decreasing from N = 256: {err}")
        baselines.upper(f"m{m}_cfl{c:g}_max_l2_n{MODE_RESOLUTIONS[-1]}", float(err[-1]), factor=baseline_factor)
    max_u = max(_mode_run(m, n, c)["max_u"] for m in MODE_NUMBERS for n in MODE_RESOLUTIONS for c in MODE_CFLS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_conv, ax_amp = axes
    for i, m in enumerate(MODE_NUMBERS):
        label = f"m = {m} (k/k_c = {2 * np.pi * m / BOX_SIZE / k_c:.2f})"
        ax_conv.loglog(MODE_RESOLUTIONS, errors[(m, C_CFL)], "o-", color=f"C{i}", label=label)
        ax_conv.loglog(MODE_RESOLUTIONS, errors[(m, 0.1)], "s--", color=f"C{i}", alpha=0.6)
        run = _mode_run(m, 256, C_CFL)
        ax_amp.plot(run["times"] / run["times"][-1], run["amplitude_ref"], "-", color=f"C{i}", label=label)
        ax_amp.plot(run["times"] / run["times"][-1], run["amplitude"], "o", ms=3, color=f"C{i}")
    n = np.array(MODE_RESOLUTIONS, dtype=float)
    ax_conv.loglog(n, 3e-2 * (n / n[0]) ** -1, "k:", lw=0.8, label=r"$N^{-1}$")
    ax_conv.loglog(n, 3e-2 * (n / n[0]) ** -2, "k-.", lw=0.8, label=r"$N^{-2}$")
    ax_conv.set_xlabel("N")
    ax_conv.set_ylabel("max over t of L2 error / amplitude")
    ax_conv.set_title(f"Single modes: solid C_cfl = {C_CFL}, dashed 0.1")
    ax_conv.legend(fontsize=8)
    ax_amp.axhline(0.0, color="grey", lw=0.5)
    ax_amp.set_xlabel(r"$t / t_{end}$ (three e-folds of the slow root)")
    ax_amp.set_ylabel("mode amplitude / initial")
    ax_amp.set_title(f"Lines exact, dots scheme (N = 256, C_cfl = {C_CFL})")
    ax_amp.legend(fontsize=8)
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_telegrapher_reference_modes_test.svg")
    plt.close(fig)

    assert max_u < max_gas_speed, f"gas not at rest: max abs(u) {max_u:.1e}."
    assert not failures, "; ".join(failures)
    baselines.check()


def test_cr_single_mode_second_order(min_order: float = 1.8, min_mode: int = 8):
    """The plan's A4.1 expectation for the single modes: second order down to
    the finest N at the default ``C_cfl``, for every mode with ``m >= min_mode``
    (pairwise order N = 1024 -> 2048). An expected failure until 2026-10-10,
    with the stage-wise backward-Euler relaxation; passes with ETD2RK.

    Args:
        min_order: Minimum pairwise order at the two finest N.
        min_mode: Smallest mode number checked.
    """
    orders = {}
    for m in MODE_NUMBERS:
        if m < min_mode:
            continue
        err = [_mode_run(m, n, C_CFL)["error"] for n in MODE_RESOLUTIONS[-2:]]
        orders[m] = float(np.log2(err[0] / err[1]))
    print("pairwise orders N 1024 -> 2048 at C_cfl 0.4: " + ", ".join(f"m = {m}: {o:.2f}" for m, o in orders.items()))
    assert min(orders.values()) >= min_order, f"single-mode orders {orders} below {min_order}."


def test_cr_relaxation_integrators(mode: int = 24, min_gain: float = 3.0, max_stagewise_order: float = 1.2):
    """ETD2RK against the stage-wise backward-Euler relaxation
    (``CR_RELAXATION_STAGEWISE_IMPLICIT``, the scheme until 2026-10-10) on one
    damped-wave mode at the default ``C_cfl``: the old scheme stays first order
    (so this test still sees the defect ETD2RK removes) and ETD2RK is at least
    ``min_gain`` times more accurate at N = 2048.

    Args:
        mode: The mode number (``k / k_c = 1.31`` for 24).
        min_gain: Minimum error ratio stage-wise / ETD2RK at N = 2048.
        max_stagewise_order: Maximum pairwise order N 1024 -> 2048 of the
            stage-wise scheme.
    """
    resolutions = MODE_RESOLUTIONS[-2:]
    etd = [_mode_run(mode, n, C_CFL)["error"] for n in resolutions]
    stagewise = [_mode_run(mode, n, C_CFL, CR_RELAXATION_STAGEWISE_IMPLICIT)["error"] for n in resolutions]
    order_stagewise = float(np.log2(stagewise[0] / stagewise[1]))
    gain = stagewise[1] / etd[1]
    print(
        f"m = {mode}, C_cfl {C_CFL}, N {resolutions[0]} / {resolutions[1]}: "
        f"ETD2RK {etd[0]:.3e} / {etd[1]:.3e}, stage-wise {stagewise[0]:.3e} / {stagewise[1]:.3e} "
        f"(order {order_stagewise:.2f}); gain at N {resolutions[1]} {gain:.2f}"
    )
    assert order_stagewise <= max_stagewise_order, (
        f"stage-wise relaxation order {order_stagewise:.2f} > {max_stagewise_order}: did it change?"
    )
    assert gain >= min_gain, f"ETD2RK gain {gain:.2f} < {min_gain}."


# ---- A4.2: v_red scan of the model floor ----

SCAN_V_REDS = (4.0, 8.0, 16.0, 32.0, 64.0)
SCAN_RESOLUTIONS = (128, 512, 2048, 8192)
# The plan's resolution (A4.2) and the one where the model floor dominates up to v_red 32.
PLAN_N, FINE_N = 2048, 8192
FLOOR_FIT_V_REDS = (4.0, 8.0, 16.0, 32.0)


def _optical_depth(v_red: float, num_cells: int) -> float:
    """``tau = (gamma_cr - 1) v_red dx / kappa``, the cell size in CR mean free
    paths (``cr_grey_transport.cr_wave_speed_reduction``)."""
    return (GAMMA_CR - 1.0) * v_red * BOX_SIZE / num_cells / KAPPA


@lru_cache(maxsize=None)
def _scan_run(v_red: float, num_cells: int, limiter: int = MINMOD):
    """Item 4's Gaussian at ``v_red``: L2 / amplitude against the telegrapher
    reference, against diffusion, and the model floor (telegrapher - diffusion)."""
    x = _cell_centers(num_cells)
    profile = E_OFFSET + _gaussian(x, SIGMA0)
    _, e_cr, _, max_u = _run(num_cells, profile, T_END, C_CFL, v_red=v_red, limiter=limiter)
    e_tel, _ = telegrapher_spectral_solution(profile, BOX_SIZE, T_END, GAMMA_CR, v_red, KAPPA)
    sigma_end = np.sqrt(SIGMA0**2 + 2.0 * KAPPA * T_END)
    e_diffusion = E_OFFSET + _gaussian(x, sigma_end)
    return dict(
        error_tel=_l2(e_cr[-1] - e_tel),
        error_diffusion=_l2(e_cr[-1] - e_diffusion),
        floor=_l2(e_tel - e_diffusion),
        max_u=max_u,
    )


def _slope(xs, ys) -> float:
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def test_vred_scan_model_floor(slope_tol: float = 0.3, max_gas_speed: float = 1e-10):
    """Plan A4.2: the error against the diffusion Green's function scales as
    ``v_red^-2`` once the model floor dominates it.

    Gate (plan): fitted slope ``-2 +- slope_tol`` over ``v_red`` = 4 ... 32, at
    ``FINE_N`` = 8192. At the plan's N = 2048 the slope is shallower; the
    reason is documented (module docstring) and checked: there the scheme
    error against the telegrapher reference exceeds the floor at ``v_red =
    32``.

    Args:
        slope_tol: Allowed deviation of the fitted slope from -2.
        max_gas_speed: The gas must stay at rest.
    """
    rows = {}
    for n in (PLAN_N, FINE_N):
        runs = [_scan_run(v, n) for v in FLOOR_FIT_V_REDS]
        rows[n] = dict(
            error_diffusion=[r["error_diffusion"] for r in runs],
            error_tel=[r["error_tel"] for r in runs],
            floor=[r["floor"] for r in runs],
            slope=_slope(FLOOR_FIT_V_REDS, [r["error_diffusion"] for r in runs]),
        )
        print(
            f"N = {n}: v_red " + " / ".join(f"{v:g}" for v in FLOOR_V_REDS[:4])
            + ": vs diffusion " + " / ".join(f"{e:.3e}" for e in rows[n]["error_diffusion"])
            + ", vs telegrapher " + " / ".join(f"{e:.3e}" for e in rows[n]["error_tel"])
            + ", floor " + " / ".join(f"{e:.3e}" for e in rows[n]["floor"])
            + f"; slope vs diffusion {rows[n]['slope']:.3f}"
        )
    max_u = max(_scan_run(v, n)["max_u"] for v in FLOOR_FIT_V_REDS for n in (PLAN_N, FINE_N))
    assert max_u < max_gas_speed, f"gas not at rest: max abs(u) {max_u:.1e}."
    assert abs(rows[FINE_N]["slope"] + 2.0) <= slope_tol, (
        f"error vs diffusion at N = {FINE_N} scales as v_red^{rows[FINE_N]['slope']:.3f}, not -2."
    )
    # The documented explanation of the shallower slope at the plan's N.
    assert rows[PLAN_N]["error_tel"][-1] > rows[PLAN_N]["floor"][-1], (
        "at N = 2048, v_red = 32 the scheme error no longer exceeds the model floor: "
        "the module docstring's explanation of the N = 2048 slope is out of date."
    )


def test_vred_scan_scheme_error(
    thin_slope: float = 1.0,
    thin_slope_tol: float = 0.2,
    thick_spread_tol: float = 0.05,
    thick_order_range: tuple = (0.8, 1.2),
):
    """The scheme error against the telegrapher reference with the default
    (MINMOD) limiter: HLL dissipation ``R(tau) v_red`` times minmod's O(dx^2)
    face jumps (module docstring).

    - Optically thin cells (``tau <= 0.07`` at ``FINE_N`` for ``v_red <= 32``):
      the error grows as ``v_red^1`` at fixed N.
    - Optically thick cells (``tau >= 2``, ``R ~ 1 / tau``): the error does
      not depend on ``v_red`` and converges at first order.

    Args:
        thin_slope: Expected slope of the error vs ``v_red`` at ``FINE_N``.
        thin_slope_tol: Allowed deviation from it.
        thick_spread_tol: Maximum relative spread of the error over ``v_red``
            = 16 / 32 / 64 at N = 128 (``tau`` = 2.1 / 4.2 / 8.3).
        thick_order_range: Allowed pairwise orders at ``v_red = 64``, N = 128 ->
            512 (``tau`` 8.3 -> 2.1).
    """
    thin = [_scan_run(v, FINE_N)["error_tel"] for v in FLOOR_FIT_V_REDS]
    slope = _slope(FLOOR_FIT_V_REDS, thin)
    thick = [_scan_run(v, 128)["error_tel"] for v in (16.0, 32.0, 64.0)]
    spread = (max(thick) - min(thick)) / min(thick)
    thick_n = (128, 256, 512)
    thick_errors = [_scan_run(64.0, n)["error_tel"] for n in thick_n]
    orders = [float(np.log2(thick_errors[i] / thick_errors[i + 1])) for i in range(2)]
    print(
        f"thin (N = {FINE_N}, tau <= {_optical_depth(32.0, FINE_N):.2f}): error vs telegrapher "
        + " / ".join(f"{e:.3e}" for e in thin) + f" at v_red 4 / 8 / 16 / 32, slope {slope:.3f}; "
        f"thick (N = 128, tau 2.1 / 4.2 / 8.3): " + " / ".join(f"{e:.4e}" for e in thick)
        + f", spread {spread:.4f}; v_red 64, N 128 / 256 / 512: "
        + " / ".join(f"{e:.3e}" for e in thick_errors) + ", orders " + " / ".join(f"{o:.2f}" for o in orders)
    )
    assert abs(slope - thin_slope) <= thin_slope_tol, f"thin-cell error ~ v_red^{slope:.3f}, expected ~1."
    assert spread <= thick_spread_tol, f"thick-cell error depends on v_red: spread {spread:.3f}."
    assert all(thick_order_range[0] <= o <= thick_order_range[1] for o in orders), (
        f"thick-cell orders {orders} outside {thick_order_range}."
    )
    baselines = Baselines("cr_telegrapher_reference.py::test_vred_scan_scheme_error")
    for v_red, error in zip(FLOOR_FIT_V_REDS, thin):
        baselines.upper(f"thin_v{v_red:g}_n{FINE_N}", error)
    for n, error in zip(thick_n, thick_errors):
        baselines.upper(f"thick_v64_n{n}", error)
    baselines.check()


def test_vred_scan_limiter(min_order: float = 1.8, min_gain: float = 5.0, max_v_red_ratio: float = 2.0):
    """The MINMOD dependence pinned down: with the MC limiter (``DOUBLE_MINMOD``,
    central slopes, O(dx^3) face jumps) the scheme error is second order also
    in optically thick cells and nearly independent of ``v_red`` (the plan's
    A4.2 expectation). Documents the alternative; the default stays MINMOD.

    Args:
        min_order: Minimum pairwise order at ``v_red = 64``, N = 256 -> 1024.
        min_gain: Minimum error ratio MINMOD / MC at ``v_red = 8``, N = 2048.
        max_v_red_ratio: Maximum MC error ratio ``v_red`` 64 / 8 at N = 2048
            (MINMOD: ~7).
    """
    mc_thick_n = (256, 512, 1024)
    mc_thick = [_scan_run(64.0, n, DOUBLE_MINMOD)["error_tel"] for n in mc_thick_n]
    orders = [float(np.log2(mc_thick[i] / mc_thick[i + 1])) for i in range(2)]
    gain = _scan_run(8.0, PLAN_N)["error_tel"] / _scan_run(8.0, PLAN_N, DOUBLE_MINMOD)["error_tel"]
    ratio = _scan_run(64.0, PLAN_N, DOUBLE_MINMOD)["error_tel"] / _scan_run(8.0, PLAN_N, DOUBLE_MINMOD)["error_tel"]
    ratio_minmod = _scan_run(64.0, PLAN_N)["error_tel"] / _scan_run(8.0, PLAN_N)["error_tel"]
    print(
        f"MC, v_red 64, N 256 / 512 / 1024 (tau 4.2 / 2.1 / 1.0): " + " / ".join(f"{e:.3e}" for e in mc_thick)
        + ", orders " + " / ".join(f"{o:.2f}" for o in orders)
        + f"; MINMOD / MC at v_red 8, N {PLAN_N}: {gain:.1f}; error v_red 64 / 8 at N {PLAN_N}: "
        f"MC {ratio:.2f}, MINMOD {ratio_minmod:.2f}"
    )
    assert min(orders) >= min_order, f"MC thick-cell orders {orders} < {min_order}."
    assert gain >= min_gain, f"MINMOD / MC error ratio {gain:.1f} < {min_gain}."
    assert ratio <= max_v_red_ratio, f"MC error still grows with v_red: ratio {ratio:.2f}."


def test_vred_scan_figure():
    """Figure of the A4.2 scan (no assertions): error against the telegrapher
    reference vs N for every ``v_red`` (MINMOD; MC for 8 and 64), and the
    errors against diffusion and telegrapher vs ``v_red`` at N = 2048 / 8192
    with the model floor."""
    fig, (ax_n, ax_v) = plt.subplots(1, 2, figsize=(14, 5.5))
    n = np.array(SCAN_RESOLUTIONS, dtype=float)
    for i, v in enumerate(SCAN_V_REDS):
        err = [_scan_run(v, m)["error_tel"] for m in SCAN_RESOLUTIONS]
        ax_n.loglog(n, err, "o-", color=f"C{i}", label=f"v_red = {v:g} (MINMOD)")
        for m, e in zip(SCAN_RESOLUTIONS, err):
            ax_n.annotate(f"{_optical_depth(v, m):.2g}", (m, e), fontsize=6, color=f"C{i}",
                          xytext=(3, 3), textcoords="offset points")
        if v in (8.0, 64.0):
            err_mc = [_scan_run(v, m, DOUBLE_MINMOD)["error_tel"] for m in SCAN_RESOLUTIONS[:3]]
            ax_n.loglog(n[:3], err_mc, "s--", color=f"C{i}", alpha=0.7, label=f"v_red = {v:g} (MC)")
    ax_n.loglog(n, 2e-3 * (n / n[0]) ** -1, "k:", lw=0.8, label=r"$N^{-1}$")
    ax_n.loglog(n, 2e-3 * (n / n[0]) ** -2, "k-.", lw=0.8, label=r"$N^{-2}$")
    ax_n.set_xlabel("N")
    ax_n.set_ylabel("L2 error / amplitude vs telegrapher")
    ax_n.set_title(r"Scheme error (labels: optical depth $\tau$ of a cell)")
    ax_n.legend(fontsize=7)
    v = np.array(SCAN_V_REDS)
    for m, style in ((PLAN_N, "--"), (FINE_N, "-")):
        ax_v.loglog(v, [_scan_run(x, m)["error_diffusion"] for x in SCAN_V_REDS], "o" + style, color="C0",
                    label=f"vs diffusion, N = {m}")
        ax_v.loglog(v, [_scan_run(x, m)["error_tel"] for x in SCAN_V_REDS], "s" + style, color="C3",
                    label=f"vs telegrapher (scheme), N = {m}")
    floors = [_scan_run(x, FINE_N)["floor"] for x in SCAN_V_REDS]
    ax_v.loglog(v, floors, "k-", lw=2, alpha=0.4, label="model floor (telegrapher - diffusion)")
    ax_v.loglog(v, floors[0] * (v / v[0]) ** -2, "k:", lw=0.8, label=r"$v_{red}^{-2}$")
    ax_v.set_xlabel(r"$v_{red}$")
    ax_v.set_ylabel("L2 error / amplitude")
    ax_v.set_title("Error vs diffusion = model floor + scheme error (MINMOD)")
    ax_v.legend(fontsize=7)
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_telegrapher_vred_scan_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    tests = (
        test_telegrapher_model_floor,
        test_cr_gaussian_vs_telegrapher,
        test_cr_single_mode_vs_telegrapher,
        test_cr_single_mode_second_order,
        test_cr_relaxation_integrators,
        test_vred_scan_model_floor,
        test_vred_scan_scheme_error,
        test_vred_scan_limiter,
        test_vred_scan_figure,
    )
    failed = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as error:
            failed.append(test.__name__)
            print(f"FAIL {test.__name__}: {error}")
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
