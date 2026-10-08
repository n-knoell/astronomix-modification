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

**Results (2026-10-08, HLL + minmod, ``C_cfl = 0.4`` unless stated).**

A6.1, item-6 tube (Sod-like, 40% CR pressure both sides; Mach 1.64, compression 1.94):

  ======  =========  =========  =========  =========  ==========  ==========
  N       dK_cr,2    dP_th,2    dP_cr,2    drho_2     spurious    dK_cr,L
  ======  =========  =========  =========  =========  ==========  ==========
  100     +1.06e-2   -5.7e-3    +1.20e-2   +1.1e-3    +1.15e-2    +5.4e-4
  400     +1.06e-2   -7.5e-3    +1.35e-2   +2.1e-3    +1.28e-2    -3.4e-4
  1600    +1.06e-2   -7.6e-3    +1.35e-2   +2.2e-3    +1.28e-2    -9.3e-5
  3200    +1.06e-2   -7.5e-3    +1.35e-2   +2.2e-3    +1.28e-2    -4.8e-5
  ======  =========  =========  =========  =========  ==========  ==========

  ``C_cfl`` 0.1 / 0.4 / 0.8: dK_cr,2 = 1.21 / 1.06 / 0.82%; MC (``DOUBLE_MINMOD``) 0.99%,
  ``VAN_ALBADA`` 1.04%, HLLC 1.02%, each identical at N = 400 and 1600. The shock lags by a
  fixed ~5e-4 in x (more CR energy makes the post-shock gas softer, hence denser and slower).

A6.2, Gupta et al. (2021) Table 2 (their N = 1000; t = 0.1 and 1e-4):

  - Tube A (``L {1, 0, 2, 1}``, ``R {0.2, 0, 0.02, 0.1}``; CR-dominated upstream, Mach 2.59,
    compression 2.94): dK_cr,2 = +6.35%, ``P_cr`` +9.9%, ``P_th`` **-10.5%**, spurious
    fraction **19%**, identical for N = 250-4000. ``C_cfl`` 0.1 / 0.8: 6.6 / 6.0%.
  - Tube B (``L {1, 0, 6.7e4, 1.3e5}``, ``R {0.2, 0, 240, 240}``; Mach 9.89, compression 3.90,
    reference post-shock ``P_cr/P`` = 0.028, cf. Gupta et al.'s ~0.03): the shocked plateau is
    only 15 cells wide at N = 1000, so N <= 1000 is unresolved. N = 2000 / 4000: dK_cr,2 =
    10.6%, spurious fraction 0.62%. Essentially the Pfrommer M = 10 tube below.

A6.3, Pfrommer et al. (2006) Sec. 5.2 Mach scan (``rho 1 | 0.2``, ``X_cr = 2 | 1``,
``P_th,L = 1e5 (gamma - 1)``, right pressure solved for the composite-sound-speed Mach number;
t such that the shock travels 0.35), N = 3200 (N = 1600 identical to 3 digits):

  =====  ======  =========  =========  ==========
  M      comp.   dK_cr,2    dP_th,2    spurious
  =====  ======  =========  =========  ==========
  1.4    1.62    +0.41%     -0.53%     +0.91%
  2      2.37    +2.56%     -2.12%     +3.76%
  3      3.09    +5.99%     -2.34%     +4.23%
  6      3.73    +9.61%     -0.87%     +1.60%
  10     3.90    +10.6%     -0.33%     +0.61%
  30     3.99    +11.1%     -0.02%     +0.07%
  100    4.00    +11.2%     +0.01%     +0.006%
  =====  ======  =========  =========  ==========

  - The CR entropy error grows with Mach number and saturates at ~11% (Semenov et al. 2021
    report ~20% for their strong-shock test). ``C_cfl`` 0.1 / 0.8 moves it by +15 / -18%.
  - The spurious energy is ~11% of the *adiabatically compressed upstream* CR energy, so it
    scales with the upstream CR pressure, not with the dissipated energy. Per post-shock thermal
    energy it peaks at ~4% for M = 2-3 and falls as ~M^-2 above (tube A, with 5x more CR than
    thermal pressure upstream: 19%).

So the partition error is a property of the scheme, not a truncation error, and at weak,
CR-loaded shocks it is as large as typical DSA efficiencies. This is decision D2 of the plan.
Until D2 is made the partition is a tracked metric: the gates only bound it from above (so a
fix that drives it to 0 still passes) and check the smooth control.

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

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    BoundarySettings1D,
    DOUBLE_PRECISION,
    OPEN_BOUNDARY,
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


def run_shock_tube(tube, num_cells, C_cfl=0.4, limiter=MINMOD, riemann_solver=HLL):
    """Run a shock tube in float64. Returns ``x, rho, u, P_th, P_cr`` (numpy)."""
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
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True),
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
):
    """A6.1: post-shock ``K_cr`` vs N and vs method.

    Args:
        resolutions: Cell counts of the convergence study.
        max_partition_error: Upper bound on ``|K_cr,2 / K_cr,R - 1|`` (tracked metric, see the
            module docstring; measured 1.06%, 1.21% at ``C_cfl = 0.1``).
        min_control_order: Minimum least-squares order of the left-plateau ``K_cr`` error
            (smooth flow, fitted over N >= 200; measured 0.85).
        max_control_error: Bound on the left-plateau ``|K_cr / K_cr,L - 1|`` at the finest N
            (measured 9e-5 at N = 1600).
    """
    ref = reference_structure(ITEM6)

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

    # ---- figure ----
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    ax_profile, ax_conv, ax_control, ax_method = axes

    k_right = ITEM6.right[3] / ITEM6.right[0] ** GAMMA_CR
    for n in resolutions[::2]:
        x, rho, _, _, p_cr = runs[n]
        m = (x > ref["x_contact"] - 0.05) & (x < ref["x_shock"] + 0.03)
        ax_profile.plot(x[m], p_cr[m] / rho[m] ** GAMMA_CR / k_right - 1.0, label=f"N = {n}")
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
    ax_conv.axhline(0.0, color="black", lw=0.8)
    ax_conv.set_xscale("log")
    ax_conv.set_xlabel("N")
    ax_conv.set_ylabel("plateau mean / reference - 1")
    ax_conv.set_title("Shocked plateau: does not converge")
    ax_conv.legend()

    ax_control.loglog(ns, dK_L, "o-", label=r"$|K_{cr}/K_{cr,L} - 1|$, left star plateau")
    ax_control.loglog(ns, dK_L[0] * ns[0] / ns, "k--", label=r"$\propto N^{-1}$")
    ax_control.set_xlabel("N")
    ax_control.set_title(f"Smooth control: order {control_order:.2f}")
    ax_control.legend()

    names = list(method_dK)
    ax_method.bar(names, [100 * method_dK[k] for k in names], color="C0")
    ax_method.set_ylabel(r"$K_{cr,2}/K_{cr,R} - 1$ [%]")
    ax_method.set_title(f"Method dependence, N = {n_method}")
    ax_method.tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"A6.1 post-shock CR/thermal partition: dK_cr,2 = {100 * dK_2[-1]:+.2f}% at N = "
        f"{resolutions[-1]} (spread over N {100 * np.ptp(dK_2):.2f}%), "
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
):
    """A6.2: Gupta et al. (2021) shock tubes A and B.

    Args:
        resolutions_a: Cell counts for tube A (Gupta et al. use 1000).
        resolutions_b: Cell counts for tube B.
        max_partition_error_a: Bound on ``|dK_cr,2|`` for tube A (measured 6.35%).
        max_partition_error_b: Bound on ``|dK_cr,2|`` for tube B, resolved runs only (measured
            10.6%).
        min_resolved_cells: Minimum cells in the shock-side window for a run to count as
            resolved (tube B: 4 at N = 1000, 9 at N = 2000).
    """
    results = {}
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
                ax.plot(r[0][m], getter(r)[m], ls="--", label=f"N = {n}")
            ax.set_xlim(lo, hi)
            ax.set_xlabel("x")
            ax.set_title(f"Gupta {label} (M = {ref['mach']:.2f}): {name}")
        axes[row, 2].set_ylim(0.5, 1.5)
        axes[row, 2].axhline(1.0, color="grey", lw=0.8)
        axes[row, 0].legend()
    m_a = results["A", resolutions_a[-1]][-1]
    m_b = results["B", resolutions_b[-1]][-1]
    fig.suptitle(
        f"A6.2 Gupta et al. (2021) tubes, N = {resolutions_a[-1]}: A dK_cr,2 = {100 * m_a['dK_2']:+.2f}%, "
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


def test_cr_shock_tube_mach_scan(
    machs=(1.4, 2.0, 3.0, 6.0, 10.0, 30.0, 60.0, 100.0),
    resolutions=(1600, 3200),
    max_partition_error=0.2,
    max_control_error=1e-3,
):
    """A6.3: CR entropy error vs Mach number (Pfrommer et al. 2006 Sec. 5.2 tubes).

    Args:
        machs: Shock Mach numbers (the paper's eight).
        resolutions: Cell counts; the shock-side window has >= 42 cells at N = 1600 for all M.
        max_partition_error: Bound on ``|dK_cr,2|`` (measured <= 11.2%, 12.9% at ``C_cfl = 0.1``).
        max_control_error: Bound on the left-plateau ``|dK_cr,L|`` (measured <= 1.6e-4).
    """
    rows = []
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

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for n, marker in zip(resolutions, ("o", "s")):
        sel = [r for r in rows if r["n"] == n]
        ms = [r["mach"] for r in sel]
        axes[0].semilogx(ms, [100 * r["dK_2"] for r in sel], marker + "-", label=rf"$K_{{cr,2}}$, N = {n}")
        axes[0].semilogx(ms, [100 * r["dP_th_2"] for r in sel], marker + ":", label=rf"$P_{{th,2}}$, N = {n}")
        axes[1].loglog(ms, [r["spurious_fraction"] for r in sel], marker + "-", label=f"N = {n}")
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
        "A6.3 Pfrommer et al. (2006) Mach scan (X_cr = 2 | 1): dK_cr,2 saturates at "
        f"{100 * max(r['dK_2'] for r in rows):.1f}%; spurious fraction peaks at "
        f"{100 * max(r['spurious_fraction'] for r in rows):.1f}%"
    )
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_shock_tube_mach_scan_test.svg")
    plt.close(fig)

    for r in rows:
        assert abs(r["dK_2"]) < max_partition_error, f"M={r['mach']} N={r['n']}: dK_cr,2 = {r['dK_2']:.3e}."
        assert abs(r["dK_L"]) < max_control_error, f"M={r['mach']} N={r['n']}: dK_cr,L = {r['dK_L']:.3e}."


if __name__ == "__main__":
    test_cr_shock_tube_partition()
    test_cr_shock_tube_gupta()
    test_cr_shock_tube_mach_scan()
