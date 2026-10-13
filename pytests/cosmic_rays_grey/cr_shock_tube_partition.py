"""
Post-shock CR/thermal partition in two-fluid shock tubes: Phase A test plan A6.1, A6.2 and A6.3
(``astronomix/_modules/_cosmic_rays_grey/astronomix_CR_phaseA_test_plan.md``, Sec. 6.4).

**What it measures.** The item-6 shock tube (``cr_shock_tube.py``) gates only the box-averaged
error, which hides the one quantity the two-fluid model leaves open at a shock: how the
dissipated energy is split between gas and CRs. The reference (Pfrommer et al. 2006, App. B;
``pfrommer_riemann_solver.py``) compresses the CRs adiabatically through the shock, so the CR
entropy ``K_cr = P_cr / rho^gamma_cr`` of the shocked gas equals the upstream ``K_cr,R``. Our
scheme evolves ``e_cr`` with a non-conservative ``-P_cr div v`` source (Gupta et al.'s
"Eg+Ecr pdv" class); for that class the post-shock partition is not fixed by the equations but
by the dissipation inside the numerically smeared shock (Gupta, Sharma & Mignone 2021;
Semenov, Kravtsov & Diemer 2021). All runs: float64, ``v_red = 0`` (``F_cr = 0`` exactly), box
[0, 1], diaphragm at 0.5, open boundaries.

Metrics (windows fixed in x, so they do not depend on N):

- shocked plateau, shock side (55-85% of the way from contact to shock): ``K_cr / K_cr,R - 1``
  (``dK_2``), ``P_th``, ``P_cr``, ``rho`` against the reference, and the spurious CR energy per
  post-shock thermal energy, ``dP_cr / (gamma_cr - 1) / (P_th,2 / (gamma - 1))``, which is
  directly comparable to a DSA efficiency. The contact half is left out: the smeared contact
  carries the left side's much higher ``K_cr`` into it (contact diffusion, A1.3);
- left star plateau (middle 50%) as the control: ``K_cr / K_cr,L - 1``. That gas only went
  through a smooth rarefaction, so the error must converge to 0;
- shock and contact positions (density mid-level crossing) against the reference, in cells.

**Results (re-measured 2026-10-09 with the stage-wise CR coupling; HLL + minmod, ``C_cfl = 0.4``
unless stated).** The CR feedback is applied inside every RK stage since 2026-10-09
(``cr_adiabatic_coupling.py``, DESIGN.md "Resolved: stage-wise CR-gas coupling"). The values
with the earlier operator-split coupling (2026-10-08) are given in brackets.

A6.1, item-6 tube (Sod-like, 40% CR pressure both sides; Mach 1.64, compression 1.94):

  ======  =========  =========  =========  =========  ==========  ==========
  N       dK_cr,2    dP_th,2    dP_cr,2    drho_2     spurious    dK_cr,L
  ======  =========  =========  =========  =========  ==========  ==========
  100     +1.28e-2   -9.7e-3    +1.35e-2   +5.5e-4    +1.29e-2    +1.7e-3
  400     +1.27e-2   -8.9e-3    +1.62e-2   +2.6e-3    +1.54e-2    -2.1e-5
  1600    +1.27e-2   -9.1e-3    +1.62e-2   +2.6e-3    +1.54e-2    -7.5e-7
  3200    +1.27e-2   -9.1e-3    +1.62e-2   +2.6e-3    +1.54e-2    -3.4e-9
  ======  =========  =========  =========  =========  ==========  ==========

  [operator-split: dK_cr,2 1.06% at every N, dK_cr,L 5.4e-4 ... 4.8e-5.] ``C_cfl`` 0.1 / 0.4 / 0.8:
  dK_cr,2 = 1.26 / 1.27 / 1.31% [1.21 / 1.06 / 0.82%]; MC (``DOUBLE_MINMOD``) 1.30%,
  ``VAN_ALBADA`` 1.28%, each the same at N = 400 and 1600. HLLC 0.42% (N = 400) with the
  total-pressure HLLC flux of 2026-10-10 (DESIGN.md "Resolved: total-pressure HLLC flux"), 1.24%
  with the stage-wise source coupling: P2 shrinks the error 3x but leaves it non-zero, so the
  partition stays method-dependent. The smooth control now
  converges at order ~1.1 [0.85]. The shock lags by a fixed ~5e-4 in x (more CR energy makes the
  post-shock gas softer, hence denser and slower).

A6.2, Gupta et al. (2021) Table 2 (their N = 1000; t = 0.1 and 1e-4):

  - Tube A (``L {1, 0, 2, 1}``, ``R {0.2, 0, 0.02, 0.1}``; CR-dominated upstream, Mach 2.59,
    compression 2.94): dK_cr,2 = +6.83% [6.35%], ``P_cr`` +10.7%, ``P_th`` **-11.3%**, spurious
    fraction **21%** [19%], identical for N = 250-4000. ``C_cfl`` 0.1 / 0.8: 6.7 / 7.3% [6.6 /
    6.0%].
  - Tube B (``L {1, 0, 6.7e4, 1.3e5}``, ``R {0.2, 0, 240, 240}``; Mach 9.89, compression 3.90,
    reference post-shock ``P_cr/P`` = 0.028, cf. Gupta et al.'s ~0.03): the shocked plateau is
    only 15 cells wide at N = 1000, so N <= 1000 is unresolved. N = 2000 / 4000: dK_cr,2 =
    12.9% [10.6%], spurious fraction 0.76% [0.62%]. Essentially the Pfrommer M = 10 tube below.

A6.3, Pfrommer et al. (2006) Sec. 5.2 Mach scan (``rho 1 | 0.2``, ``X_cr = 2 | 1``,
``P_th,L = 1e5 (gamma - 1)``, right pressure solved for the composite-sound-speed Mach number;
t such that the shock travels 0.35), N = 3200 (N = 1600 identical to 3 digits):

  =====  ======  ================  =========  =============
  M      comp.   dK_cr,2 [op.-sp.] dP_th,2    spurious
  =====  ======  ================  =========  =============
  1.4    1.62    +0.48% [0.41%]    -0.63%     +1.07%
  2      2.37    +2.98% [2.56%]    -2.48%     +4.38%
  3      3.09    +7.09% [5.99%]    -2.79%     +5.02%
  6      3.73    +11.6% [9.61%]    -1.07%     +1.93%
  10     3.90    +12.9% [10.6%]    -0.41%     +0.75%
  30     3.99    +13.6% [11.1%]    -0.05%     +0.085%
  100    4.00    +13.6% [11.2%]    -0.004%    +0.0077%
  =====  ======  ================  =========  =============

  - The CR entropy error grows with Mach number and saturates at ~14% [11%] (Semenov et al.
    2021 report ~20% for their strong-shock test). ``C_cfl`` 0.1 / 0.8 moves it by -2 / +5%
    relative [+15 / -18%]: the stage-wise coupling makes the error larger but much less
    method-dependent, as Gupta et al. (2021) report for their "Unsplit-pdv".
  - The spurious energy is ~14% of the *adiabatically compressed upstream* CR energy, so it
    scales with the upstream CR pressure, not with the dissipated energy. Per post-shock thermal
    energy it peaks at ~5% for M = 2-3 and falls as ~M^-2 above (tube A, with 5x more CR than
    thermal pressure upstream: 21%).

So the partition error is a property of the scheme, not a truncation error, and at weak,
CR-loaded shocks it is as large as typical DSA efficiencies. This is decision D2 of the plan.
The energy-scheme numbers above stay as the control curves (dashed in the figures); the gates
only bound them from above and check the smooth control. Since 2026-10-12 (Phase A plan
X1/X2) the upper bounds are 1.5x the measured values in ``baselines.json``
(``cr_baselines.py``), on top of the looser fixed bounds in the test signatures.

**D2 decided (2026-10-13): HLLC + hybrid CR entropy is the grey-CR default** (``DEFAULT_SCHEME``;
DESIGN.md "Decided: CR shock partition"). Solid curves and black markers in the figures; gated
at ``max_default_error = 1e-3`` and against ``baselines.json``. Measured (float64, ``C_cfl =
0.4``):

  - A6.1 item-6 tube: dK_cr,2 = -2.8e-5 / +8.4e-5 / +8.6e-5 / +8.6e-5 / +8.6e-5 at N = 100 /
    200 / 400 / 800 / 1600 (energy scheme 1.27%). N = 400: ``C_cfl`` 0.1 / 0.8 give 6.9e-5 /
    3.9e-4, MC -1.1e-5, ``VAN_ALBADA`` 4.7e-6.
  - A6.2 (N = 4000): tube A dK_cr,2 = +1.3e-4, ``P_th`` -2.0e-4, spurious fraction 3.7e-4
    (energy scheme 6.83% / -11.3% / 21%); tube B +7.2e-6 (12.9%).
  - A6.3 (N = 3200): dK_cr,2 = 1.03e-4 / 1.00e-4 / 1.31e-4 / 9.3e-6 / 9.1e-6 / 9.0e-6 / 9.0e-6
    at M = 1.4 / 2 / 3 / 6 / 10 / 30 / 100; ``abs(dP_th,2)`` <= 1.3e-4, spurious fraction <=
    2.3e-4. The left-plateau control ``dK_cr,L`` grows to -2e-4 at M >= 30 (energy scheme
    <= 7e-7; not investigated, two orders below the partition error it replaces).
  - 3D Sedov (``test_cr_entropy_hybrid_sedov_3d``, N = 48): shell mean ``abs(dK_cr)`` 1.9e-4
    (energy scheme 1.4e-2).

Runtime on CPU (``JAX_PLATFORMS=cpu``): A6.1 ~1 min, A6.2 ~1 min, A6.3 ~4 min.

See ``PROGRESS_PHASEA.md`` in the module directory.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import warnings
from pathlib import Path
from typing import NamedTuple

# numerics
import numpy as np
from scipy.optimize import brentq

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
    FINITE_VOLUME,
    BoundarySettings1D,
    DOUBLE_PRECISION,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    DOUBLE_MINMOD,
    HLL,
    HLLC,
    MINMOD,
    VAN_ALBADA,
    finalize_config,
)

# astronomix functions
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.test_setups.reference_solutions.pfrommer_riemann_solver import (
    _hugoniot_rho_star,
    _rho_from_total_pressure,
    _solve_star_region,
    _sound_speed,
    pfrommer_riemann_solution,
)

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CR_ENTROPY_TRANSFER_GLOBAL,
    CR_ENTROPY_TRANSFER_SHOCKS,
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# Plateau averages at the 1e-5 level need float64.
jax.config.update("jax_enable_x64", True)

GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
X0 = 0.5  # diaphragm; box [0, 1]


class ShockTube(NamedTuple):
    """Left/right states ``(rho, u, P_th, P_cr)`` and end time of one shock tube."""

    left: tuple
    right: tuple
    t_end: float


# A6.1: the item-6 tube of cr_shock_tube.py.
ITEM6 = ShockTube(left=(1.0, 0.0, 0.6, 0.4), right=(0.125, 0.0, 0.06, 0.04), t_end=0.2)

PICS_DIR = Path(__file__).resolve().parent / "pics" / "06_shock_tube"

# The CR-run default since 2026-10-13 (simulation_config.resolve_auto_defaults): HLLC with the
# total-pressure flux and the hybrid CR entropy (DESIGN.md "Decided: CR shock partition").
# run_shock_tube's own defaults are the energy scheme with HLL, the control.
DEFAULT_SCHEME = dict(riemann_solver=HLLC, cr_entropy=True, cr_entropy_transfer=CR_ENTROPY_TRANSFER_SHOCKS)


def _reference(tube, x):
    """Reference (rho, u, P_th, P_cr) at positions ``x``."""
    with warnings.catch_warnings():
        # quad's round-off warning inside the rarefaction integral; harmless (~1e-13).
        warnings.simplefilter("ignore")
        return pfrommer_riemann_solution(
            *tube.left, *tube.right, GAMMA, GAMMA_CR, np.asarray(x), tube.t_end, X0
        )


def reference_structure(tube):
    """Wave positions and star states of the reference, for a left rarefaction and a right
    shock (all tubes here). Also the shock Mach number with the composite upstream sound speed
    ``c_1^2 = (gamma P_th,1 + gamma_cr P_cr,1) / rho_1`` (Pfrommer et al. 2006, App. B)."""
    (rho_l, u_l, p_th_l, p_cr_l), (rho_r, u_r, p_th_r, p_cr_r) = tube.left, tube.right
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_star, u_star, _, p_r = _solve_star_region(
            *tube.left, *tube.right, GAMMA, GAMMA_CR
        )
        rho_star_left = _rho_from_total_pressure(p_star, rho_l, p_th_l, p_cr_l, GAMMA, GAMMA_CR)
        rho_2 = _hugoniot_rho_star(p_star, rho_r, p_th_r, p_cr_r, GAMMA, GAMMA_CR)
    assert p_star > p_r and p_star < p_th_l + p_cr_l, "Expected a left rarefaction + right shock."
    a_tail = _sound_speed(rho_star_left, rho_l, p_th_l, p_cr_l, GAMMA, GAMMA_CR)
    v_shock = u_r + np.sqrt((p_star - p_r) / (1.0 / rho_r - 1.0 / rho_2)) / rho_r
    p_cr_2 = p_cr_r * (rho_2 / rho_r) ** GAMMA_CR
    c_1 = np.sqrt((GAMMA * p_th_r + GAMMA_CR * p_cr_r) / rho_r)
    return dict(
        x_head=X0 + (u_l - _sound_speed(rho_l, rho_l, p_th_l, p_cr_l, GAMMA, GAMMA_CR)) * tube.t_end,
        x_tail=X0 + (u_star - a_tail) * tube.t_end,
        x_contact=X0 + u_star * tube.t_end,
        x_shock=X0 + v_shock * tube.t_end,
        rho_star_left=rho_star_left,
        rho_2=rho_2,
        p_th_2=p_star - p_cr_2,
        p_cr_2=p_cr_2,
        mach=(v_shock - u_r) / c_1,
        compression=rho_2 / rho_r,
    )


def run_shock_tube(
    tube, num_cells, C_cfl=0.4, limiter=MINMOD, riemann_solver=HLL, cr_entropy=False,
    cr_entropy_transfer=CR_ENTROPY_TRANSFER_GLOBAL,
):
    """Run a shock tube in float64. Returns ``x, rho, u, P_th, P_cr`` (numpy).

    ``cr_entropy`` selects the conservative CR entropy scheme
    (``CosmicRayGreyConfig.cr_entropy``, off by default like the library).
    """
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=1.0,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
        ),
        numerical_precision=DOUBLE_PRECISION,
        limiter=limiter,
        riemann_solver=riemann_solver,
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, cr_entropy=cr_entropy, cr_entropy_transfer=cr_entropy_transfer
        ),
    )
    registered_variables = get_registered_variables(config)
    x = get_helper_data(config).geometric_centers

    left = x < X0
    state = jnp.zeros((registered_variables.num_vars, num_cells))
    rows = (
        registered_variables.density_index,
        registered_variables.velocity_index,
        registered_variables.pressure_index,
    )
    for row, value_l, value_r in zip(rows, tube.left[:3], tube.right[:3]):
        state = state.at[row].set(jnp.where(left, value_l, value_r))
    state = state.at[registered_variables.cosmic_ray_e_index].set(
        jnp.where(left, tube.left[3], tube.right[3]) / (GAMMA_CR - 1.0)
    )

    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=tube.t_end,
        gamma=GAMMA,
        C_cfl=C_cfl,
        # v_red = 0: tightly coupled, F_cr stays exactly 0 (see cr_shock_tube.py).
        cosmic_ray_grey_params=CosmicRayGreyParams(gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0),
    )
    final = np.asarray(time_integration(state, config, params, registered_variables))
    assert not np.any(np.isnan(final)), f"NaNs in the shock tube at N = {num_cells}."
    return (
        np.asarray(x),
        final[registered_variables.density_index],
        final[registered_variables.velocity_index],
        final[registered_variables.pressure_index],
        final[registered_variables.cosmic_ray_e_index] * (GAMMA_CR - 1.0),
    )


def _crossing(x, y, level, lo, hi):
    """Last linear-interpolated crossing of ``y = level`` in ``lo < x < hi``."""
    m = (x > lo) & (x < hi)
    xx, yy = x[m], y[m]
    i = np.nonzero(np.diff(np.sign(yy - level)))[0][-1]
    return xx[i] + (level - yy[i]) * (xx[i + 1] - xx[i]) / (yy[i + 1] - yy[i])


def partition_metrics(tube, x, rho, p_th, p_cr, ref):
    """Plateau and position metrics of one run (see the module docstring)."""
    dx = x[1] - x[0]
    k_cr = p_cr / rho**GAMMA_CR
    k_left = tube.left[3] / tube.left[0] ** GAMMA_CR
    k_right = tube.right[3] / tube.right[0] ** GAMMA_CR

    def window(a, b, lo, hi):
        return (x > a + lo * (b - a)) & (x < a + hi * (b - a))

    # Shock side of the shocked plateau: the smeared contact carries the left side's much
    # higher K_cr into the contact half (63x / 6600x for Gupta B / M = 100), which is contact
    # diffusion (A1.3), not the shock partition.
    shocked = window(ref["x_contact"], ref["x_shock"], 0.55, 0.85)
    left_star = window(ref["x_tail"], ref["x_contact"], 0.25, 0.75)
    gap = 0.1 * (ref["x_shock"] - ref["x_contact"])
    x_shock = _crossing(x, rho, 0.5 * (ref["rho_2"] + tube.right[0]), ref["x_contact"] + gap, 1.0)
    x_contact = _crossing(
        x, rho, 0.5 * (ref["rho_star_left"] + ref["rho_2"]), ref["x_tail"], ref["x_shock"] - gap
    )
    return dict(
        dK_2=k_cr[shocked].mean() / k_right - 1.0,
        dP_th_2=p_th[shocked].mean() / ref["p_th_2"] - 1.0,
        dP_cr_2=p_cr[shocked].mean() / ref["p_cr_2"] - 1.0,
        drho_2=rho[shocked].mean() / ref["rho_2"] - 1.0,
        # Spurious CR energy per post-shock thermal energy, comparable to a DSA efficiency.
        spurious_fraction=(p_cr[shocked].mean() - ref["p_cr_2"])
        / (GAMMA_CR - 1.0)
        / (ref["p_th_2"] / (GAMMA - 1.0)),
        dK_L=k_cr[left_star].mean() / k_left - 1.0,
        shock_offset_cells=(x_shock - ref["x_shock"]) / dx,
        contact_offset_cells=(x_contact - ref["x_contact"]) / dx,
        cells_in_window=int(shocked.sum()),
    )


def test_cr_shock_tube_partition(
    resolutions=(100, 200, 400, 800, 1600),
    max_partition_error=0.02,
    min_control_order=0.7,
    max_control_error=1e-3,
    max_default_error=1e-3,
):
    """A6.1: post-shock ``K_cr`` vs N and vs method, for the energy scheme with HLL (the
    control; the scheme the module docstring's tables describe) and the CR-run default since
    2026-10-13 (``DEFAULT_SCHEME``: HLLC + hybrid CR entropy).

    Args:
        resolutions: Cell counts of the convergence study.
        max_partition_error: Upper bound on ``|K_cr,2 / K_cr,R - 1|`` (tracked metric, see the
            module docstring; measured 1.27%, 1.31% at ``C_cfl = 0.8``).
        min_control_order: Minimum least-squares order of the left-plateau ``K_cr`` error
            (smooth flow, fitted over N >= 200; measured 1.12).
        max_control_error: Bound on the left-plateau ``|K_cr / K_cr,L - 1|`` at the finest N
            (measured 7.5e-7 at N = 1600).
        max_default_error: Bound on the default scheme's ``|dK_cr,2|`` at every N and for every
            method variant.
    """
    ref = reference_structure(ITEM6)
    default_runs = {n: run_shock_tube(ITEM6, n, **DEFAULT_SCHEME) for n in resolutions}
    default_dK = np.array([
        partition_metrics(ITEM6, x, rho, p_th, p_cr, ref)["dK_2"]
        for (x, rho, _, p_th, p_cr) in default_runs.values()
    ])

    runs = {n: run_shock_tube(ITEM6, n) for n in resolutions}
    metrics = {
        n: partition_metrics(ITEM6, x, rho, p_th, p_cr, ref)
        for n, (x, rho, _, p_th, p_cr) in runs.items()
    }

    n_method = 400
    variants = {
        "C_cfl 0.1": dict(C_cfl=0.1),
        "C_cfl 0.4": dict(),
        "C_cfl 0.8": dict(C_cfl=0.8),
        "MC limiter": dict(limiter=DOUBLE_MINMOD),
        "van Albada": dict(limiter=VAN_ALBADA),
        "HLLC": dict(riemann_solver=HLLC),
    }
    method_dK = {}
    for name, kwargs in variants.items():
        x, rho, _, p_th, p_cr = runs[n_method] if not kwargs else run_shock_tube(ITEM6, n_method, **kwargs)
        method_dK[name] = partition_metrics(ITEM6, x, rho, p_th, p_cr, ref)["dK_2"]
    default_method_dK = {}
    for name, kwargs in variants.items():
        if name == "HLLC":
            continue
        x, rho, _, p_th, p_cr = run_shock_tube(ITEM6, n_method, **{**DEFAULT_SCHEME, **kwargs})
        default_method_dK[name] = partition_metrics(ITEM6, x, rho, p_th, p_cr, ref)["dK_2"]

    ns = np.array(resolutions, dtype=float)
    dK_2 = np.array([metrics[n]["dK_2"] for n in resolutions])
    dK_L = np.abs([metrics[n]["dK_L"] for n in resolutions])
    # N = 100 is pre-asymptotic (dK_L changes sign between N = 100 and 200), so fit from N = 200.
    control_order = -np.polyfit(np.log(ns[1:]), np.log(dK_L[1:]), 1)[0]

    for n in resolutions:
        m = metrics[n]
        print(
            f"N={n:5d}  dK_2={m['dK_2']:+.3e}  dP_th_2={m['dP_th_2']:+.3e}  "
            f"dP_cr_2={m['dP_cr_2']:+.3e}  drho_2={m['drho_2']:+.3e}  dK_L={m['dK_L']:+.3e}  "
            f"shock={m['shock_offset_cells']:+.2f} dx  contact={m['contact_offset_cells']:+.2f} dx"
        )
    print(f"left-plateau K_cr order {control_order:.2f}")
    print("method dependence (N = 400): " + ", ".join(f"{k} {v:+.3e}" for k, v in method_dK.items()))
    print("default (HLLC + hybrid) dK_2 vs N: " + ", ".join(f"{n} {v:+.2e}" for n, v in zip(resolutions, default_dK)))
    print("default method dependence (N = 400): " + ", ".join(f"{k} {v:+.2e}" for k, v in default_method_dK.items()))

    # ---- figure ----
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ax_profile, ax_conv, ax_control, ax_method = axes

    k_right = ITEM6.right[3] / ITEM6.right[0] ** GAMMA_CR
    for i, n in enumerate(resolutions[::2]):
        for scheme_runs, style, name in ((runs, "--", "energy, HLL"), (default_runs, "-", "default")):
            x, rho, _, _, p_cr = scheme_runs[n]
            m = (x > ref["x_contact"] - 0.05) & (x < ref["x_shock"] + 0.03)
            ax_profile.plot(x[m], p_cr[m] / rho[m] ** GAMMA_CR / k_right - 1.0, style, color=f"C{i}",
                            label=f"N = {n}, {name}")
    ax_profile.axhline(0.0, color="black", label="reference (adiabatic CRs)")
    for xv in (ref["x_contact"], ref["x_shock"]):
        ax_profile.axvline(xv, color="grey", ls=":")
    ax_profile.set_ylim(-0.03, 0.05)
    ax_profile.set_xlabel("x")
    ax_profile.set_ylabel(r"$K_{cr}/K_{cr,R} - 1$")
    ax_profile.set_title("CR entropy of the shocked gas (left of contact: off scale)")
    ax_profile.legend()

    for key, label in [
        ("dK_2", r"$K_{cr,2}$"),
        ("dP_cr_2", r"$P_{cr,2}$"),
        ("dP_th_2", r"$P_{th,2}$"),
        ("drho_2", r"$\rho_2$"),
    ]:
        ax_conv.plot(ns, [metrics[n][key] for n in resolutions], "o-", label=label)
    ax_conv.plot(ns, default_dK, "s-", color="black", label=r"$K_{cr,2}$, default (HLLC + hybrid)")
    ax_conv.axhline(0.0, color="black", lw=0.8)
    ax_conv.set_xscale("log")
    ax_conv.set_xlabel("N")
    ax_conv.set_ylabel("plateau mean / reference - 1")
    ax_conv.set_title("Shocked plateau (energy scheme: does not converge)")
    ax_conv.legend()

    ax_control.loglog(ns, dK_L, "o-", label=r"$|K_{cr}/K_{cr,L} - 1|$, left star plateau")
    ax_control.loglog(ns, dK_L[0] * ns[0] / ns, "k--", label=r"$\propto N^{-1}$")
    ax_control.set_xlabel("N")
    ax_control.set_title(f"Smooth control: order {control_order:.2f}")
    ax_control.legend()

    names = list(method_dK)
    xs = np.arange(len(names))
    ax_method.bar(xs - 0.2, [max(abs(100 * method_dK[k]), 1e-6) for k in names], 0.4, color="C0",
                  label="energy scheme (HLL; last bar HLLC)")
    ax_method.bar(xs[:-1] + 0.2, [max(abs(100 * default_method_dK[k]), 1e-6) for k in names[:-1]], 0.4,
                  color="black", label="default (HLLC + hybrid)")
    ax_method.set_xticks(xs, names, rotation=30)
    ax_method.set_yscale("log")
    ax_method.set_ylabel(r"$|K_{cr,2}/K_{cr,R} - 1|$ [%]")
    ax_method.set_title(f"Method dependence, N = {n_method}")
    ax_method.legend(fontsize=8)

    fig.suptitle(
        f"A6.1 post-shock CR/thermal partition. Default (HLLC + hybrid CR entropy): dK_cr,2 = "
        f"{100 * default_dK[-1]:+.3f}% at N = {resolutions[-1]}, <= {100 * np.abs(list(default_method_dK.values())).max():.3f}% "
        f"over methods. Energy scheme (HLL): {100 * dK_2[-1]:+.2f}%, "
        f"{100 * min(method_dK.values()):.2f}-{100 * max(method_dK.values()):.2f}% over methods"
    )
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_shock_tube_partition_test.svg")
    plt.close(fig)

    # ---- gates ----
    worst = max(np.abs(dK_2).max(), max(abs(v) for v in method_dK.values()))
    assert worst < max_partition_error, (
        f"Post-shock CR entropy error {worst:.3e} exceeds the tracked bound {max_partition_error}."
    )
    assert control_order > min_control_order, (
        f"Left-plateau K_cr error converges at order {control_order:.2f} < {min_control_order}: "
        "the smooth (rarefaction) part of the solution is no longer adiabatic."
    )
    assert dK_L[-1] < max_control_error, (
        f"Left-plateau K_cr error {dK_L[-1]:.3e} at N = {resolutions[-1]} exceeds {max_control_error}."
    )
    default_worst = max(np.abs(default_dK).max(), max(abs(v) for v in default_method_dK.values()))
    assert default_worst < max_default_error, (
        f"Default scheme post-shock CR entropy error {default_worst:.3e} exceeds {max_default_error}."
    )
    baselines = Baselines("cr_shock_tube_partition.py::test_cr_shock_tube_partition")
    baselines.upper("default_abs_dK_2_worst", float(default_worst), floor=2e-4)
    baselines.upper(f"abs_dK_2_n{resolutions[-1]}", abs(float(dK_2[-1])), floor=1e-3)
    baselines.upper("abs_dK_2_worst_method", max(abs(v) for v in method_dK.values()), floor=1e-3)
    baselines.upper(f"dK_L_n{resolutions[-1]}", float(dK_L[-1]), floor=1e-6)
    baselines.lower("control_order", control_order, margin=0.15)
    baselines.check()



# A6.2: Gupta, Sharma & Mignone (2021), Table 2, problems 2 and 3.
GUPTA_A = ShockTube(left=(1.0, 0.0, 2.0, 1.0), right=(0.2, 0.0, 0.02, 0.1), t_end=0.1)
GUPTA_B = ShockTube(left=(1.0, 0.0, 6.7e4, 1.3e5), right=(0.2, 0.0, 2.4e2, 2.4e2), t_end=1e-4)


def pfrommer_mach_tube(mach):
    """A6.3: Pfrommer et al. (2006) Sec. 5.2 composite tube with shock Mach number ``mach``.

    Left: ``rho = 1``, ``P_th = (gamma - 1) 1e5``, ``X_cr = P_cr / P_th = 2``. Right:
    ``rho = 0.2``, ``X_cr = 1``, pressure solved for the requested Mach number (composite
    upstream sound speed; the paper gives no table). ``t_end`` lets the shock travel 0.35 (and
    keeps the rarefaction head inside the box).
    """
    p_th_left = (GAMMA - 1.0) * 1e5
    left = (1.0, 0.0, p_th_left, 2.0 * p_th_left)

    def mach_residual(log_p):
        right = (0.2, 0.0, np.exp(log_p), np.exp(log_p))
        return reference_structure(ShockTube(left, right, 1.0))["mach"] - mach

    log_p = brentq(mach_residual, np.log(1e-12 * p_th_left), np.log(0.9 * p_th_left), xtol=1e-13)
    right = (0.2, 0.0, np.exp(log_p), np.exp(log_p))
    ref = reference_structure(ShockTube(left, right, 1.0))
    t_end = min(0.35 / (ref["x_shock"] - X0), 0.4 / (X0 - ref["x_head"]))
    return ShockTube(left, right, t_end)


def test_cr_shock_tube_gupta(
    resolutions_a=(250, 1000, 4000),
    resolutions_b=(1000, 2000, 4000),
    max_partition_error_a=0.1,
    max_partition_error_b=0.2,
    min_resolved_cells=8,
    max_default_error=1e-3,
):
    """A6.2: Gupta et al. (2021) shock tubes A and B: the energy scheme with HLL (control,
    dashed) at all N, and the CR-run default (``DEFAULT_SCHEME``, HLLC + hybrid CR entropy,
    solid) at the finest N.

    Args:
        resolutions_a: Cell counts for tube A (Gupta et al. use 1000).
        resolutions_b: Cell counts for tube B.
        max_partition_error_a: Bound on ``|dK_cr,2|`` for tube A (measured 6.83%).
        max_partition_error_b: Bound on ``|dK_cr,2|`` for tube B, resolved runs only (measured
            12.9%).
        min_resolved_cells: Minimum cells in the shock-side window for a run to count as
            resolved (tube B: 4 at N = 1000, 9 at N = 2000).
        max_default_error: Bound on the default scheme's ``|dK_cr,2|`` (finest N).
    """
    results = {}
    default = {}
    for label, tube, n in [("A", GUPTA_A, resolutions_a[-1]), ("B", GUPTA_B, resolutions_b[-1])]:
        ref = reference_structure(tube)
        x, rho, _, p_th, p_cr = run_shock_tube(tube, n, **DEFAULT_SCHEME)
        default[label] = (x, rho, p_th, p_cr, ref, partition_metrics(tube, x, rho, p_th, p_cr, ref))
        m = default[label][-1]
        print(f"Gupta {label} default (HLLC + hybrid) N={n:5d}  dK_2={m['dK_2']:+.3e}  dP_th_2={m['dP_th_2']:+.3e}  "
              f"dP_cr_2={m['dP_cr_2']:+.3e}  spurious={m['spurious_fraction']:+.3e}")
    for label, tube, resolutions in [("A", GUPTA_A, resolutions_a), ("B", GUPTA_B, resolutions_b)]:
        ref = reference_structure(tube)
        for n in resolutions:
            x, rho, _, p_th, p_cr = run_shock_tube(tube, n)
            results[label, n] = (x, rho, p_th, p_cr, ref, partition_metrics(tube, x, rho, p_th, p_cr, ref))
            m = results[label, n][-1]
            print(
                f"Gupta {label} (M = {ref['mach']:.2f}) N={n:5d}  dK_2={m['dK_2']:+.3e}  "
                f"dP_th_2={m['dP_th_2']:+.3e}  dP_cr_2={m['dP_cr_2']:+.3e}  "
                f"spurious={m['spurious_fraction']:+.3e}  cells={m['cells_in_window']}"
            )

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for row, (label, tube, resolutions) in enumerate(
        [("A", GUPTA_A, resolutions_a), ("B", GUPTA_B, resolutions_b)]
    ):
        ref = results[label, resolutions[0]][4]
        lo = ref["x_tail"] - 0.3 * (ref["x_shock"] - ref["x_tail"])
        hi = ref["x_shock"] + 0.15 * (ref["x_shock"] - ref["x_tail"])
        x_fine = np.linspace(lo, hi, 4001)
        rho_ref, _, p_th_ref, p_cr_ref = _reference(tube, x_fine)
        k_right = tube.right[3] / tube.right[0] ** GAMMA_CR
        panels = [
            (p_th_ref, lambda r: r[2], r"$P_{th}$"),
            (p_cr_ref, lambda r: r[3], r"$P_{cr}$"),
            (p_cr_ref / rho_ref**GAMMA_CR / k_right, lambda r: r[3] / r[1] ** GAMMA_CR / k_right, r"$K_{cr}/K_{cr,R}$"),
        ]
        for ax, (ref_curve, getter, name) in zip(axes[row], panels):
            ax.plot(x_fine, ref_curve, color="black", label="reference (adiabatic CRs)")
            for n in resolutions:
                r = results[label, n]
                m = (r[0] > lo) & (r[0] < hi)
                ax.plot(r[0][m], getter(r)[m], ls="--", label=f"N = {n}, energy scheme (HLL)")
            r = default[label]
            m = (r[0] > lo) & (r[0] < hi)
            ax.plot(r[0][m], getter(r)[m], ls="-", color="C3", lw=1.2,
                    label=f"N = {resolutions[-1]}, default (HLLC + hybrid)")
            ax.set_xlim(lo, hi)
            ax.set_xlabel("x")
            ax.set_title(f"Gupta {label} (M = {ref['mach']:.2f}): {name}")
        axes[row, 2].set_ylim(0.5, 1.5)
        axes[row, 2].axhline(1.0, color="grey", lw=0.8)
        axes[row, 0].legend()
    m_a = results["A", resolutions_a[-1]][-1]
    m_b = results["B", resolutions_b[-1]][-1]
    d_a, d_b = default["A"][-1], default["B"][-1]
    fig.suptitle(
        f"A6.2 Gupta et al. (2021) tubes, N = {resolutions_a[-1]}. Default (HLLC + hybrid): A dK_cr,2 = "
        f"{100 * d_a['dK_2']:+.3f}%, P_th,2 {100 * d_a['dP_th_2']:+.2f}% | B dK_cr,2 = {100 * d_b['dK_2']:+.3f}%.\n"
        f"Energy scheme (HLL): A dK_cr,2 = {100 * m_a['dK_2']:+.2f}%, "
        f"P_th,2 {100 * m_a['dP_th_2']:+.1f}%, spurious {100 * m_a['spurious_fraction']:.1f}% | "
        f"B dK_cr,2 = {100 * m_b['dK_2']:+.2f}%, spurious {100 * m_b['spurious_fraction']:.2f}%"
    )
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_shock_tube_gupta_test.svg")
    plt.close(fig)

    for n in resolutions_a:
        dK = results["A", n][-1]["dK_2"]
        assert abs(dK) < max_partition_error_a, f"Gupta A N={n}: dK_cr,2 = {dK:.3e}."
    for n in resolutions_b:
        m = results["B", n][-1]
        if m["cells_in_window"] >= min_resolved_cells:
            assert abs(m["dK_2"]) < max_partition_error_b, f"Gupta B N={n}: dK_cr,2 = {m['dK_2']:.3e}."
    assert results["B", resolutions_b[-1]][-1]["cells_in_window"] >= min_resolved_cells
    for label in ("A", "B"):
        dK = default[label][-1]["dK_2"]
        assert abs(dK) < max_default_error, f"Gupta {label} default scheme: dK_cr,2 = {dK:.3e}."
    baselines = Baselines("cr_shock_tube_partition.py::test_cr_shock_tube_gupta")
    for label in ("A", "B"):
        baselines.upper(f"default_{label}_abs_dK_2", abs(default[label][-1]["dK_2"]), floor=2e-4)
    for (label, n), result in results.items():
        baselines.upper(f"{label}_abs_dK_2_n{n}", abs(result[-1]["dK_2"]), floor=1e-3)
    baselines.check()


def test_cr_shock_tube_mach_scan(
    machs=(1.4, 2.0, 3.0, 6.0, 10.0, 30.0, 60.0, 100.0),
    resolutions=(1600, 3200),
    max_partition_error=0.2,
    max_control_error=1e-3,
    max_default_error=1e-3,
):
    """A6.3: CR entropy error vs Mach number (Pfrommer et al. 2006 Sec. 5.2 tubes), energy
    scheme with HLL (control) at both N and the CR-run default (``DEFAULT_SCHEME``, HLLC +
    hybrid CR entropy) at the finer N.

    Args:
        machs: Shock Mach numbers (the paper's eight).
        resolutions: Cell counts; the shock-side window has >= 42 cells at N = 1600 for all M.
        max_partition_error: Bound on ``|dK_cr,2|`` (measured <= 13.6%, 14.3% at ``C_cfl = 0.8``).
        max_control_error: Bound on the left-plateau ``|dK_cr,L|`` (measured <= 6.5e-7).
        max_default_error: Bound on the default scheme's ``|dK_cr,2|``.
    """
    rows = []
    default_rows = []
    for mach in machs:
        tube = pfrommer_mach_tube(mach)
        ref = reference_structure(tube)
        assert abs(ref["mach"] - mach) < 1e-8 * mach
        for n in resolutions:
            x, rho, _, p_th, p_cr = run_shock_tube(tube, n)
            m = partition_metrics(tube, x, rho, p_th, p_cr, ref)
            rows.append(dict(mach=mach, n=n, compression=ref["compression"], **m))
            print(
                f"M={mach:6.1f} N={n:5d} compression={ref['compression']:.3f}  dK_2={m['dK_2']:+.3e}  "
                f"dP_th_2={m['dP_th_2']:+.3e}  dP_cr_2={m['dP_cr_2']:+.3e}  "
                f"spurious={m['spurious_fraction']:+.3e}  dK_L={m['dK_L']:+.2e}"
            )
        x, rho, _, p_th, p_cr = run_shock_tube(tube, resolutions[-1], **DEFAULT_SCHEME)
        m = partition_metrics(tube, x, rho, p_th, p_cr, ref)
        default_rows.append(dict(mach=mach, **m))
        print(f"M={mach:6.1f} N={resolutions[-1]:5d} default (HLLC + hybrid)  dK_2={m['dK_2']:+.3e}  "
              f"dP_th_2={m['dP_th_2']:+.3e}  spurious={m['spurious_fraction']:+.3e}  dK_L={m['dK_L']:+.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for n, marker in zip(resolutions, ("o", "s")):
        sel = [r for r in rows if r["n"] == n]
        ms = [r["mach"] for r in sel]
        axes[0].semilogx(ms, [100 * r["dK_2"] for r in sel], marker + "-", label=rf"$K_{{cr,2}}$, N = {n}")
        axes[0].semilogx(ms, [100 * r["dP_th_2"] for r in sel], marker + ":", label=rf"$P_{{th,2}}$, N = {n}")
        axes[1].loglog(ms, [abs(r["spurious_fraction"]) for r in sel], marker + "--", label=f"N = {n}, energy scheme (HLL)")
    ms = [r["mach"] for r in default_rows]
    axes[0].semilogx(ms, [100 * r["dK_2"] for r in default_rows], "D-", color="black",
                     label=rf"$K_{{cr,2}}$, default (HLLC + hybrid), N = {resolutions[-1]}")
    axes[1].loglog(ms, [max(abs(r["spurious_fraction"]), 1e-8) for r in default_rows], "D-", color="black",
                   label=f"N = {resolutions[-1]}, default (HLLC + hybrid)")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[0].set_xlabel("Mach number")
    axes[0].set_ylabel("post-shock plateau / reference - 1 [%]")
    axes[0].set_title("Post-shock partition error vs Mach")
    axes[0].legend()
    m_ref = np.array([6.0, 100.0])
    axes[1].loglog(m_ref, 0.016 * (m_ref / 6.0) ** -2, "k--", label=r"$\propto M^{-2}$")
    axes[1].set_xlabel("Mach number")
    axes[1].set_ylabel(r"spurious $\Delta e_{cr}$ / post-shock $e_{th}$")
    axes[1].set_title("Spurious CR energy (compare: DSA efficiency)")
    axes[1].legend()
    fig.suptitle(
        "A6.3 Pfrommer et al. (2006) Mach scan (X_cr = 2 | 1). Default (HLLC + hybrid): |dK_cr,2| <= "
        f"{100 * max(abs(r['dK_2']) for r in default_rows):.3f}%. Energy scheme (HLL): dK_cr,2 saturates at "
        f"{100 * max(r['dK_2'] for r in rows):.1f}%, spurious fraction peaks at "
        f"{100 * max(r['spurious_fraction'] for r in rows):.1f}%"
    )
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_shock_tube_mach_scan_test.svg")
    plt.close(fig)

    for r in rows:
        assert abs(r["dK_2"]) < max_partition_error, f"M={r['mach']} N={r['n']}: dK_cr,2 = {r['dK_2']:.3e}."
        assert abs(r["dK_L"]) < max_control_error, f"M={r['mach']} N={r['n']}: dK_cr,L = {r['dK_L']:.3e}."
    for r in default_rows:
        assert abs(r["dK_2"]) < max_default_error, f"M={r['mach']} default scheme: dK_cr,2 = {r['dK_2']:.3e}."
    baselines = Baselines("cr_shock_tube_partition.py::test_cr_shock_tube_mach_scan")
    baselines.upper("default_abs_dK_2_worst", max(abs(r["dK_2"]) for r in default_rows), floor=2e-4)
    for r in rows:
        if r["n"] == resolutions[-1]:
            baselines.upper(f"m{r['mach']:g}_abs_dK_2_n{r['n']}", abs(r["dK_2"]), floor=1e-3)
    baselines.check()


def test_cr_entropy_shock_partition(
    max_entropy_error=1e-6,
    max_entropy_error_gupta_b=1e-4,
    max_plateau_error=2e-3,
):
    """A6.1-A6.3 with ``CosmicRayGreyConfig.cr_entropy``: the CRs pass the shock adiabatically.

    Acceptance test of the conservative CR entropy scheme (DESIGN.md "Open: conservative CR
    entropy at shocks"). Same tubes and metrics as the energy-scheme tests above, with the flag
    on; the energy scheme runs alongside for the figure.

    Args:
        max_entropy_error: Bound on ``|dK_cr,2|`` (measured <= 2e-9 at N = 1600 for all tubes
            and all ``C_cfl``; energy scheme 1.3-13.6%).
        max_entropy_error_gupta_b: Bound for Gupta B at N = 4000, whose plateau is 18 cells
            wide in the window (measured 7e-6).
        max_plateau_error: Bound on the shocked-plateau ``P_th``, ``P_cr`` and ``rho`` errors
            (measured <= 7e-4; truncation of the smeared shock).
    """
    cases = [
        ("item 6", ITEM6, 1600, 0.1),
        ("item 6", ITEM6, 1600, 0.4),
        ("item 6", ITEM6, 1600, 0.8),
        ("Gupta A", GUPTA_A, 2000, 0.4),
        ("Gupta B", GUPTA_B, 4000, 0.4),
        ("M = 2", pfrommer_mach_tube(2.0), 1600, 0.4),
        ("M = 10", pfrommer_mach_tube(10.0), 1600, 0.4),
        ("M = 100", pfrommer_mach_tube(100.0), 1600, 0.4),
    ]
    rows = []
    profiles = {}
    for label, tube, n, cfl in cases:
        ref = reference_structure(tube)
        per_flag = {}
        for flag in (False, True):
            x, rho, _, p_th, p_cr = run_shock_tube(tube, n, C_cfl=cfl, cr_entropy=flag)
            per_flag[flag] = partition_metrics(tube, x, rho, p_th, p_cr, ref)
            if cfl == 0.4 and label in ("item 6", "Gupta A", "M = 10"):
                profiles[label, flag] = (x, p_cr / rho**GAMMA_CR / (tube.right[3] / tube.right[0] ** GAMMA_CR), ref)
        rows.append((label, n, cfl, per_flag[False], per_flag[True]))
        m = per_flag[True]
        print(
            f"{label:8s} N={n:5d} C_cfl={cfl}  energy scheme dK_2={per_flag[False]['dK_2']:+.3e} | "
            f"cr_entropy dK_2={m['dK_2']:+.3e} dP_th_2={m['dP_th_2']:+.2e} dP_cr_2={m['dP_cr_2']:+.2e} "
            f"drho_2={m['drho_2']:+.2e} spurious={m['spurious_fraction']:+.2e}"
        )

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, label in zip(axes[:3], ("item 6", "Gupta A", "M = 10")):
        for flag, style, name in ((False, "--", "energy scheme"), (True, "-", "cr_entropy")):
            x, k_ratio, ref = profiles[label, flag]
            width = ref["x_shock"] - ref["x_contact"]
            m = (x > ref["x_contact"] - 0.3 * width) & (x < ref["x_shock"] + 0.2 * width)
            ax.plot(x[m], k_ratio[m] - 1.0, style, label=name)
        ax.axhline(0.0, color="black", lw=0.8, label="reference (adiabatic CRs)")
        ax.axvline(ref["x_contact"], color="grey", ls=":")
        ax.axvline(ref["x_shock"], color="grey", ls=":")
        ax.set_ylim(-0.03, 0.15)
        ax.set_xlabel("x")
        ax.set_ylabel(r"$K_{cr}/K_{cr,R} - 1$")
        ax.set_title(f"{label} (M = {ref['mach']:.2f})")
        ax.legend()
    labels = [f"{r[0]}\nN={r[1]}, C={r[2]}" for r in rows]
    xs = np.arange(len(rows))
    axes[3].bar(xs - 0.2, [max(abs(r[3]["dK_2"]), 1e-16) for r in rows], 0.4, label="energy scheme")
    axes[3].bar(xs + 0.2, [max(abs(r[4]["dK_2"]), 1e-16) for r in rows], 0.4, label="cr_entropy")
    axes[3].set_yscale("log")
    axes[3].set_ylim(1e-15, 1e3)  # room for the legend above the bars
    axes[3].set_xticks(xs, labels, rotation=60, fontsize=7)
    axes[3].set_ylabel(r"$|K_{cr,2}/K_{cr,R} - 1|$")
    axes[3].set_title("Post-shock CR entropy error")
    axes[3].legend()
    worst = max(abs(r[4]["dK_2"]) for r in rows)
    fig.suptitle(
        f"Conservative CR entropy (cr_entropy=True): post-shock |dK_cr,2| <= {worst:.1e} "
        f"(energy scheme up to {100 * max(abs(r[3]['dK_2']) for r in rows):.1f}%)"
    )
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_shock_tube_cr_entropy_test.svg")
    plt.close(fig)

    for label, n, cfl, _, m in rows:
        bound = max_entropy_error_gupta_b if label == "Gupta B" else max_entropy_error
        assert abs(m["dK_2"]) < bound, f"{label} N={n} C_cfl={cfl}: dK_cr,2 = {m['dK_2']:.3e}."
        for key in ("dP_th_2", "dP_cr_2", "drho_2"):
            assert abs(m[key]) < max_plateau_error, f"{label} N={n} C_cfl={cfl}: {key} = {m[key]:.3e}."


def test_cr_shock_tube_energy_conservation(num_cells=800, tol=1e-12):
    """A6.5: mass and total energy ``E_gas + e_cr`` in a closed (reflecting) shock tube.

    The ``-v.grad P_cr`` / ``-P_cr div u`` coupling telescopes, so the energy scheme conserves
    the total to round-off; the ``cr_entropy`` transfer moves energy between ``e_cr`` and the
    gas only, so it must too. Item-6 and Gupta A tubes, run to 3 t_end (several wall
    reflections), both schemes. Measured: mass <= 2e-16, energy <= 5e-14.

    Args:
        num_cells: Cell count.
        tol: Bound on the relative drift of mass and total energy.
    """
    for label, tube in (("item 6", ITEM6), ("Gupta A", GUPTA_A)):
        for flag in (False, True):
            config = SimulationConfig(
                solver_mode=FINITE_VOLUME,
                dimensionality=1,
                num_cells=num_cells,
                box_size=1.0,
                boundary_settings=BoundarySettings1D(
                    left_boundary=REFLECTIVE_BOUNDARY, right_boundary=REFLECTIVE_BOUNDARY
                ),
                numerical_precision=DOUBLE_PRECISION,
                # A6.5 as calibrated: the energy scheme and the global transfer, HLL. The hybrid's
                # closed-box energy is checked in test_cr_entropy_shock_only (HLL and HLLC).
                riemann_solver=HLL,
                cosmic_ray_grey_config=CosmicRayGreyConfig(
                    grey_cosmic_rays=True, cr_entropy=flag, cr_entropy_transfer=CR_ENTROPY_TRANSFER_GLOBAL
                ),
            )
            registered_variables = get_registered_variables(config)
            x = get_helper_data(config).geometric_centers
            left = x < X0
            state = jnp.zeros((registered_variables.num_vars, num_cells))
            rows = (
                registered_variables.density_index,
                registered_variables.velocity_index,
                registered_variables.pressure_index,
            )
            for row, value_l, value_r in zip(rows, tube.left[:3], tube.right[:3]):
                state = state.at[row].set(jnp.where(left, value_l, value_r))
            state = state.at[registered_variables.cosmic_ray_e_index].set(
                jnp.where(left, tube.left[3], tube.right[3]) / (GAMMA_CR - 1.0)
            )
            config = finalize_config(config, state.shape)
            params = SimulationParams(
                t_end=3 * tube.t_end,
                gamma=GAMMA,
                cosmic_ray_grey_params=CosmicRayGreyParams(gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0),
            )
            final = np.asarray(time_integration(state, config, params, registered_variables))

            def totals(s):
                rho = s[registered_variables.density_index]
                u = s[registered_variables.velocity_index]
                p_th = s[registered_variables.pressure_index]
                e_cr = s[registered_variables.cosmic_ray_e_index]
                return rho.sum(), (p_th / (GAMMA - 1.0) + 0.5 * rho * u**2 + e_cr).sum()

            (m0, e0), (m1, e1) = totals(np.asarray(state)), totals(final)
            d_mass, d_energy = abs(m1 / m0 - 1.0), abs(e1 / e0 - 1.0)
            print(f"{label:8s} cr_entropy={flag!s:5}  mass drift {d_mass:.1e}  energy drift {d_energy:.1e}")
            assert not np.any(np.isnan(final))
            assert d_mass < tol and d_energy < tol, (
                f"{label} cr_entropy={flag}: mass drift {d_mass:.2e}, energy drift {d_energy:.2e}."
            )


def test_cr_entropy_cold_rarefaction(
    num_cells=400, p_th=1e-4, max_entropy_error=1e-3, max_entropy_error_dual=3e-2, tol=1e-12,
    max_energy_drift=1e-4,
):
    """``cr_entropy`` gas-positivity safeguard in cold, CR-dominated gas.

    Periodic box, ``rho = 1``, ``P_cr = 1``, ``P_th = 1e-4``, ``u = -1.5 | +1.5``: a double
    rarefaction (Einfeldt's 1-2-3 problem) in the middle, a colliding shock at the wrap,
    ``t = 0.15``. The energy scheme turns ~9% of the rarefied CRs' energy into spurious gas
    heat at start-up; ``cr_entropy`` takes it back, more than the cold gas has, so without the
    safeguard (``cr_entropy_max_thermal_drain``) the gas pressure goes negative (NaN).

    Checks with the flag on, with and without the gas ``dual_energy``: no NaN, the CR entropy
    in the rarefaction (``|x - 0.5| < 0.15``) within ``max_entropy_error`` /
    ``max_entropy_error_dual`` of its initial value and below the energy scheme's, mass
    conserved to ``tol``, total-energy drift below ``max_energy_drift``.

    Measured with the stage-wise CR coupling (2026-10-09): K_cr error 6.3e-5 / 2.2e-2 without /
    with dual energy (energy scheme 5.3% / 2.6%); energy drift 1.0e-5 (energy scheme 3.8e-8; it no
    longer hits the gas pressure floor at start-up). With the earlier operator-split coupling:
    0.89% / 0.75% (energy scheme 9.1% / 7.4%), drift 2.1e-4 (energy scheme 4.6e-4). With dual
    energy the safeguard now limits more, because the energy scheme leaves less spurious
    start-up heat in the gas.

    Known trade-off (not gated): energy conservation moves the energy scheme's truncation-level
    CR excess into the cold gas, ~1e-4 of the CR energy. With P_th/P_cr = 1e-6 and dual energy
    that heats the start-up region 8x more than the energy scheme does; without dual energy
    cr_entropy halves the spurious gas heating.

    Args:
        num_cells: Cell count.
        p_th: Initial gas pressure.
        max_entropy_error: Bound on ``|K_cr / K_cr,0 - 1|`` in the rarefaction.
        max_entropy_error_dual: The same with gas dual energy.
        tol: Bound on the relative mass drift.
        max_energy_drift: Bound on the relative total-energy drift.
    """
    results = {}
    for dual in (False, True):
        for flag in (False, True):
            config = SimulationConfig(
                solver_mode=FINITE_VOLUME,
                dimensionality=1,
                num_cells=num_cells,
                box_size=1.0,
                dual_energy=dual,
                boundary_settings=BoundarySettings1D(
                    left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
                ),
                numerical_precision=DOUBLE_PRECISION,
                # This test is about the global transfer's safeguard (the hybrid does not act in a
                # rarefaction), with the HLL solver it was calibrated with.
                riemann_solver=HLL,
                cosmic_ray_grey_config=CosmicRayGreyConfig(
                    grey_cosmic_rays=True, cr_entropy=flag, cr_entropy_transfer=CR_ENTROPY_TRANSFER_GLOBAL
                ),
            )
            registered_variables = get_registered_variables(config)
            x = np.asarray(get_helper_data(config).geometric_centers)
            state = jnp.zeros((registered_variables.num_vars, num_cells))
            state = state.at[registered_variables.density_index].set(1.0)
            state = state.at[registered_variables.velocity_index].set(jnp.where(x < 0.5, -1.5, 1.5))
            state = state.at[registered_variables.pressure_index].set(p_th)
            state = state.at[registered_variables.cosmic_ray_e_index].set(1.0 / (GAMMA_CR - 1.0))
            config = finalize_config(config, state.shape)
            params = SimulationParams(
                t_end=0.15,
                gamma=GAMMA,
                cosmic_ray_grey_params=CosmicRayGreyParams(gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0),
            )
            final = np.asarray(time_integration(state, config, params, registered_variables))

            def totals(st):
                rho = st[registered_variables.density_index]
                u = st[registered_variables.velocity_index]
                e = st[registered_variables.pressure_index] / (GAMMA - 1.0) + 0.5 * rho * u**2
                return rho.sum(), (e + st[registered_variables.cosmic_ray_e_index]).sum()

            rho = final[registered_variables.density_index]
            k_cr = final[registered_variables.cosmic_ray_e_index] * (GAMMA_CR - 1.0) / rho**GAMMA_CR
            middle = np.abs(x - 0.5) < 0.15
            (m0, e0), (m1, e1) = totals(np.asarray(state)), totals(final)
            results[dual, flag] = dict(
                nan=bool(np.any(np.isnan(final))),
                k_err=float(np.max(np.abs(k_cr[middle] - 1.0))),
                d_mass=abs(m1 / m0 - 1.0),
                d_energy=abs(e1 / e0 - 1.0),
            )
            r = results[dual, flag]
            print(
                f"dual_energy={dual!s:5} cr_entropy={flag!s:5}  NaN={r['nan']}  "
                f"max|K_cr/K_cr,0 - 1| (rarefaction) {r['k_err']:.2e}  "
                f"mass drift {r['d_mass']:.1e}  energy drift {r['d_energy']:.1e}"
            )

    for dual in (False, True):
        r = results[dual, True]
        assert not r["nan"], f"cr_entropy, dual_energy={dual}: NaN."
        bound = max_entropy_error_dual if dual else max_entropy_error
        assert r["k_err"] < bound, f"dual_energy={dual}: K_cr error {r['k_err']:.3e}."
        assert r["k_err"] < results[dual, False]["k_err"], "cr_entropy is not better than the energy scheme."
    r = results[False, True]
    assert r["d_mass"] < tol, f"cr_entropy: mass drift {r['d_mass']:.2e}."
    assert r["d_energy"] < max_energy_drift, f"cr_entropy: energy drift {r['d_energy']:.2e}."


def _run_1d(variant, num_cells, state_fn, t_end, boundary, reduced_streaming_speed=0.0, riemann_solver=HLL):
    """1D float64 run for the hybrid tests. ``variant``: "off", "global" or "shocks";
    ``state_fn(x, registered_variables)`` returns the initial primitive state."""
    cr_kwargs = dict(
        off=dict(cr_entropy=False),
        **{"global": dict(cr_entropy=True, cr_entropy_transfer=CR_ENTROPY_TRANSFER_GLOBAL)},
        shocks=dict(cr_entropy=True, cr_entropy_transfer=CR_ENTROPY_TRANSFER_SHOCKS),
    )[variant]
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=1.0,
        boundary_settings=BoundarySettings1D(left_boundary=boundary, right_boundary=boundary),
        numerical_precision=DOUBLE_PRECISION,
        riemann_solver=riemann_solver,
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True, **cr_kwargs),
    )
    registered_variables = get_registered_variables(config)
    x = np.asarray(get_helper_data(config).geometric_centers)
    state = state_fn(x, registered_variables)
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=t_end,
        gamma=GAMMA,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR, reduced_streaming_speed=reduced_streaming_speed
        ),
    )
    final = np.asarray(time_integration(state, config, params, registered_variables))
    assert not np.any(np.isnan(final)), f"{variant}: NaN."
    return x, np.asarray(state), final, registered_variables


def _total_energy(state, registered_variables):
    rho = state[registered_variables.density_index]
    u = state[registered_variables.velocity_index]
    return (
        state[registered_variables.pressure_index] / (GAMMA - 1.0)
        + 0.5 * rho * u**2
        + state[registered_variables.cosmic_ray_e_index]
    ).sum()


def test_cr_entropy_shock_only(
    max_entropy_error_hllc=1e-3,
    max_entropy_error_hll=5e-3,
    min_gain_hllc=10.0,
    min_gain_hll=4.0,
    tol=1e-12,
    max_gas_entropy_error=1e-2,
    max_rarefaction_difference=1e-5,
):
    """``CR_ENTROPY_TRANSFER_SHOCKS``, the hybrid (2026-10-13; DESIGN.md "Decided: CR shock
    partition", refined plan): the energy scheme everywhere, except in the shock zones
    of ``cr_grey_sources.cr_entropy_shock_mask`` (Gupta et al. 2021 detector), where ``e_cr :=
    e(s_cr)`` and the difference goes to the gas. ``s_cr`` is re-synced from ``e_cr`` every
    step. Checks:

    1. Post-shock CR entropy: item-6, Gupta A, M = 2 (below the DSA threshold) and M = 10
       tubes, HLL and HLLC (total-pressure flux). Gates: ``abs(dK_cr,2)`` below
       ``max_entropy_error_hllc`` / ``_hll`` and at least ``min_gain`` below the energy scheme.
    2. Total energy in the closed (reflecting) item-6 box: exact (the transfer moves energy
       between ``e_cr`` and the gas only).
    3. CR-dominated smooth squeeze (item-2 setup, P_cr/P_th = 33): the gas entropy is not
       damaged and the ringing is no larger than the energy scheme's.
    4. Cold, CR-dominated double rarefaction: the rarefaction (``abs(x - 1/2) < 0.15``) is no
       shock zone, so it is identical to the energy scheme there (the global transfer is what
       fixes the start-up error, at the price of the mixing error of 5). The converging flow
       at the periodic edge does shock, and the hybrid acts there; what that changes reaches
       the window at the 5e-7 level (measured 4.8e-7, ``max_rarefaction_difference``).
    5. Mixing at a pressure-balanced, advected density contact (contrast 10, one crossing, HLLC):
       ``E_cr`` and the total pressure as exact as the energy scheme. The global transfer gains
       CR energy there (Jensen: ``e(s_cr)`` of a mixed cell exceeds the mixed ``e_cr``); that
       error, first order in dx, is why the mode before 2026-10-13 (``e(s_cr)`` everywhere,
       the difference discarded outside shocks) created 11-15% of the CR injection in M7.

    Measured (2026-10-13, PROGRESS_PHASEA.md "D2 restarted"):
    1. ``dK_cr,2`` HLLC 9e-6 ... 1.3e-4, HLL 1.5-2.4e-3 (energy scheme HLLC 0.42-6.2%, HLL
       1.3-12.9%); shock within 0.9 cells of the reference.
    2. Drift 3.0e-14 (HLL and HLLC).
    3. Squeeze identical to the energy scheme (ringing 0.0007, gas entropy 2.7e-6; global 1.5e-3).
    4. ``K_cr`` error 5.29e-2 in both; ``e_cr`` differs by 4.8e-7.
    5. ``dE_cr`` -1e-14, ``max abs(P_tot - 1)`` 6e-14, as the energy scheme; global +2.3e-2 / 2.4e-2.

    Args:
        max_entropy_error_hllc: Bound on ``abs(dK_cr,2)`` with HLLC.
        max_entropy_error_hll: Bound on ``abs(dK_cr,2)`` with HLL.
        min_gain_hllc: Minimum ratio of the energy scheme's ``abs(dK_cr,2)`` to the hybrid's,
            HLLC (measured 49-6900).
        min_gain_hll: The same for HLL (measured 5.7 for item 6, the weakest shock).
        tol: Bound on the closed-box energy drift and on the mixing-test changes.
        max_gas_entropy_error: Bound on ``max abs(K_th / K_th,0 - 1)`` in the squeeze.
        max_rarefaction_difference: Bound on the relative ``e_cr`` difference hybrid vs energy
            scheme inside the cold rarefaction.
    """
    # 1. shocks
    tubes = (("item 6", ITEM6, 800), ("Gupta A", GUPTA_A, 2000), ("M = 2", pfrommer_mach_tube(2.0), 1600),
             ("M = 10", pfrommer_mach_tube(10.0), 1600))
    for solver, bound, min_gain in ((HLLC, max_entropy_error_hllc, min_gain_hllc), (HLL, max_entropy_error_hll, min_gain_hll)):
        name = "HLLC" if solver == HLLC else "HLL"
        for label, tube, n in tubes:
            ref = reference_structure(tube)
            m = {}
            for flag in (False, True):
                x, rho, _, p_th, p_cr = run_shock_tube(
                    tube, n, riemann_solver=solver, cr_entropy=flag,
                    cr_entropy_transfer=CR_ENTROPY_TRANSFER_SHOCKS,
                )
                m[flag] = partition_metrics(tube, x, rho, p_th, p_cr, ref)
            print(f"{name:4s} {label:8s} N={n}: dK_cr,2 hybrid {m[True]['dK_2']:+.2e} (energy scheme "
                  f"{m[False]['dK_2']:+.2e}), dP_th,2 {m[True]['dP_th_2']:+.2e}, shock offset "
                  f"{m[True]['shock_offset_cells']:+.2f} cells")
            assert abs(m[True]["dK_2"]) < bound, f"{name} {label}: dK_cr,2 = {m[True]['dK_2']:.3e}."
            assert abs(m[False]["dK_2"]) > min_gain * abs(m[True]["dK_2"]), (
                f"{name} {label}: hybrid {m[True]['dK_2']:.3e} vs energy scheme {m[False]['dK_2']:.3e}."
            )

    # 2. closed box
    def tube_state(tube):
        def fn(x, rv):
            left = x < X0
            state = jnp.zeros((rv.num_vars, x.size))
            for row, a, b in zip((rv.density_index, rv.velocity_index, rv.pressure_index), tube.left[:3], tube.right[:3]):
                state = state.at[row].set(jnp.where(left, a, b))
            return state.at[rv.cosmic_ray_e_index].set(jnp.where(left, tube.left[3], tube.right[3]) / (GAMMA_CR - 1.0))
        return fn

    for solver in (HLL, HLLC):
        _, s0, f, rv = _run_1d("shocks", 800, tube_state(ITEM6), 3 * ITEM6.t_end, REFLECTIVE_BOUNDARY,
                               riemann_solver=solver)
        drift = abs(_total_energy(f, rv) / _total_energy(s0, rv) - 1.0)
        print(f"closed item-6 box ({'HLLC' if solver == HLLC else 'HLL'}): total-energy drift {drift:.2e}")
        assert drift < tol, f"hybrid energy drift {drift:.3e}."

    # 3. CR-dominated smooth squeeze (item 2)
    def squeeze(x, rv):
        state = jnp.zeros((rv.num_vars, x.size)).at[rv.density_index].set(1.0)
        state = state.at[rv.velocity_index].set(-0.2 * (x - 0.5)).at[rv.pressure_index].set(0.01)
        return state.at[rv.cosmic_ray_e_index].set(1.0)

    ringing, gas_error = {}, {}
    for variant in ("off", "global", "shocks"):
        x, _, f, rv = _run_1d(variant, 512, squeeze, 1.0, OPEN_BOUNDARY, reduced_streaming_speed=0.1)
        rho = f[rv.density_index]
        window = (x > 0.28) & (x < 0.42)
        interior = (x > 0.1) & (x < 0.9)
        ringing[variant] = float(rho[window].max() - rho[window].min())
        gas_error[variant] = float(np.max(np.abs(f[rv.pressure_index][interior] / rho[interior] ** GAMMA / 0.01 - 1.0)))
        print(f"squeeze {variant:6s}: rho ringing {ringing[variant]:.4f}, max abs(K_th - 1) {gas_error[variant]:.2e}")
    assert gas_error["shocks"] < max_gas_entropy_error, f"hybrid gas entropy error {gas_error['shocks']:.3e}."
    assert ringing["shocks"] <= ringing["off"] * (1.0 + 1e-6), "hybrid ringing exceeds the energy scheme's."

    # 4. cold rarefaction: no shock zone
    def rarefaction(x, rv):
        state = jnp.zeros((rv.num_vars, x.size)).at[rv.density_index].set(1.0)
        state = state.at[rv.velocity_index].set(jnp.where(x < 0.5, -1.5, 1.5)).at[rv.pressure_index].set(1e-4)
        return state.at[rv.cosmic_ray_e_index].set(1.0 / (GAMMA_CR - 1.0))

    e_cr = {}
    for variant in ("off", "shocks"):
        x, _, f, rv = _run_1d(variant, 400, rarefaction, 0.15, PERIODIC_BOUNDARY)
        e_cr[variant] = f[rv.cosmic_ray_e_index][np.abs(x - 0.5) < 0.15]
        k_cr = f[rv.cosmic_ray_e_index] * (GAMMA_CR - 1.0) / f[rv.density_index] ** GAMMA_CR
        print(f"cold rarefaction {variant:6s}: max abs(K_cr - 1) {np.max(np.abs(k_cr[np.abs(x - 0.5) < 0.15] - 1.0)):.2e}")
    diff = float(np.max(np.abs(e_cr["shocks"] / e_cr["off"] - 1.0)))
    print(f"cold rarefaction: hybrid vs energy scheme, max relative e_cr difference {diff:.2e}")
    assert diff < max_rarefaction_difference, f"cold rarefaction: hybrid differs from the energy scheme by {diff:.3e}."

    # 5. mixing at an advected, pressure-balanced density contact
    def contact(x, rv):
        state = jnp.zeros((rv.num_vars, x.size))
        state = state.at[rv.density_index].set(jnp.where(np.abs(x - 0.5) < 0.25, 1.0, 0.1))
        state = state.at[rv.velocity_index].set(1.0).at[rv.pressure_index].set(0.5)
        return state.at[rv.cosmic_ray_e_index].set(0.5 / (GAMMA_CR - 1.0))

    for variant in ("off", "global", "shocks"):
        _, s0, f, rv = _run_1d(variant, 512, contact, 1.0, PERIODIC_BOUNDARY, riemann_solver=HLLC)
        d_ecr = float(f[rv.cosmic_ray_e_index].sum() / s0[rv.cosmic_ray_e_index].sum() - 1.0)
        dp = float(np.max(np.abs(f[rv.pressure_index] + (GAMMA_CR - 1.0) * f[rv.cosmic_ray_e_index] - 1.0)))
        print(f"mixing {variant:6s}: dE_cr/E_cr {d_ecr:+.2e}, max abs(P_tot - 1) {dp:.2e}")
        if variant != "global":
            assert abs(d_ecr) < tol and dp < tol, f"mixing {variant}: dE_cr {d_ecr:.2e}, dP_tot {dp:.2e}."


def test_cr_entropy_hybrid_sedov_3d(num_cells=48, t_end=0.07, max_entropy_error=1e-2, min_gain=5.0):
    """Multi-D check of the hybrid: a 3D Sedov blast (the item-7 setup, thermal explosion,
    ``E = 1`` in ``rho = 1``, ``P_th = 1e-4``) into a uniform CR background ``P_cr = 1e-4``.
    The CRs only see adiabatic compression, so the exact ``K_cr = P_cr / rho^gamma_cr`` stays
    at its ambient value everywhere, at every shock orientation. HLLC (total-pressure flux) and
    HLL, energy scheme vs hybrid: the mass-weighted mean of ``abs(K_cr / K_0 - 1)`` over the
    shocked shell (``rho > 1.5``), and its maximum.

    Gates: hybrid mean below ``max_entropy_error`` and at least ``min_gain`` below the energy
    scheme. The shell is ~2-4 cells thick at N = 48, so this checks the mask in 3D, not the
    plateau value.

    Measured (2026-10-13, float32 default precision): shell mean ``abs(dK_cr)`` HLLC 1.4e-2
    (energy scheme) -> 1.9e-4 (hybrid), maximum 3.4e-2 -> 3.6e-3; HLL 3.5e-2 -> 2.1e-4, maximum
    0.10 -> 4.8e-3.

    Args:
        num_cells: Cells per axis.
        t_end: End time (the shock stays inside the box, as in item 7).
        max_entropy_error: Bound on the hybrid's shell mean ``abs(dK_cr)``.
        min_gain: Minimum ratio energy scheme / hybrid of that mean.
    """
    from astronomix import CARTESIAN, construct_primitive_state

    results = {}
    for solver in (HLLC, HLL):
        for flag in (False, True):
            config = SimulationConfig(
                geometry=CARTESIAN,
                solver_mode=FINITE_VOLUME,
                riemann_solver=solver,
                limiter=MINMOD,
                dimensionality=3,
                num_cells=num_cells,
                exact_end_time=True,
                cosmic_ray_grey_config=CosmicRayGreyConfig(
                    grey_cosmic_rays=True, cr_entropy=flag, cr_entropy_transfer=CR_ENTROPY_TRANSFER_SHOCKS
                ),
            )
            helper_data = get_helper_data(config)
            rv = get_registered_variables(config)
            dx = 1.0 / num_cells
            weight = 0.5 * (1.0 - jnp.tanh((helper_data.r - 0.05) / (2.0 * dx)))
            gas_pressure = 1e-4 + (GAMMA - 1.0) / (jnp.sum(weight) * dx**3) * weight
            zeros = jnp.zeros((num_cells,) * 3)
            state = construct_primitive_state(
                config=config, registered_variables=rv, density=zeros + 1.0, velocity_x=zeros,
                velocity_y=zeros, velocity_z=zeros, gas_pressure=gas_pressure,
            )
            state = state.at[rv.cosmic_ray_e_index].set(1e-4 / (GAMMA_CR - 1.0))
            config = finalize_config(config, state.shape)
            params = SimulationParams(
                t_end=t_end, gamma=GAMMA,
                cosmic_ray_grey_params=CosmicRayGreyParams(gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0),
            )
            final = np.asarray(time_integration(state, config, params, rv))
            assert not np.any(np.isnan(final))
            rho = final[rv.density_index]
            dk = np.abs(final[rv.cosmic_ray_e_index] * (GAMMA_CR - 1.0) / rho**GAMMA_CR / 1e-4 - 1.0)
            shell = rho > 1.5
            results[solver, flag] = (float((dk * rho)[shell].sum() / rho[shell].sum()), float(dk[shell].max()))
            print(f"Sedov 3D N={num_cells} {'HLLC' if solver == HLLC else 'HLL':4s} "
                  f"{'hybrid' if flag else 'energy':6s}: shell mean abs(dK_cr) {results[solver, flag][0]:.2e}, "
                  f"max {results[solver, flag][1]:.2e}, shell cells {int(shell.sum())}")
    for solver in (HLLC, HLL):
        hybrid, energy = results[solver, True][0], results[solver, False][0]
        assert hybrid < max_entropy_error, f"Sedov 3D hybrid shell dK_cr {hybrid:.3e}."
        assert energy > min_gain * hybrid, f"Sedov 3D: hybrid {hybrid:.3e} vs energy scheme {energy:.3e}."


if __name__ == "__main__":
    test_cr_shock_tube_partition()
    test_cr_shock_tube_gupta()
    test_cr_shock_tube_mach_scan()
    test_cr_entropy_shock_partition()
    test_cr_shock_tube_energy_conservation()
    test_cr_entropy_cold_rarefaction()
    test_cr_entropy_shock_only()
    test_cr_entropy_hybrid_sedov_3d()
