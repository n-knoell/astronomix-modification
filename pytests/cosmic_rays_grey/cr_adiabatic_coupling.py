"""
CR-gas coupling tests: Phase A test plan A2.2 and A2.3
(``astronomix/_modules/_cosmic_rays_grey/astronomix_CR_phaseA_test_plan.md``, Sec. 2.4).

Ladder item 2 (``cr_adiabatic_compression.py``) checks the local adiabatic invariant
``e_cr ~ rho^gamma_cr``. These tests check the *dynamics* of the coupling (``-grad P_cr`` on the
gas with its work, ``-P_cr div u`` on the CRs), all in float64 with ``v_red = 0`` (``F_cr = 0``,
tightly coupled):

- **A2.2, CR-modified linear sound wave.** Periodic box, the exact eigenvector of the
  composite fluid (``delta rho / rho = eps``, ``delta u = c_eff eps``, ``delta P_th = gamma P_th
  eps``, ``delta P_cr = gamma_cr P_cr eps``, ``c_eff^2 = (gamma P_th + gamma_cr P_cr) / rho``),
  ``eps = 1e-6``, one period, ``P_cr / P_th`` in {0, 0.5, 2, 10}, N = 64-512. Gates: L1 order of
  ``rho``, phase speed, the eigenvector ratio, and stability at ``C_cfl = 0.8``.
- **A2.3, equivalence with gamma = 4/3 hydro.** With ``P_th = 1e-6 P_cr`` the two-fluid system
  is the single-fluid gamma = 4/3 Euler system. (a) A simple wave (``u = 0.1 sin 2 pi x``,
  constant Riemann invariant) up to half its shock-formation time against the exact
  characteristic solution and against the code's own gamma = 4/3 hydro; (b) the item-2 squeeze
  (``u = -0.2 (x - 0.5)``, open boundaries, t = 1) against gamma = 4/3 hydro.

**History (2026-10-09).** Until then the coupling was operator-split: evaluated once from the
pre-step state and added after the whole RK2 step, i.e. a forward-Euler step of an
oscillatory subsystem. Measured with that scheme:

- A2.2: order 1.09 / 0.68 / -0.09 at ``P_cr / P_th`` = 0.5 / 2 / 10, against 1.86 without CRs.
  The error was a wave train of fixed 8-12 cells, growing exponentially, first order in dt for
  ``C_cfl <= 0.2`` and unstable above (``C_cfl = 0.8``: 1e5 eps at ``P_cr / P_th = 10``).
- A2.3: simple-wave error growing with N (order -0.31); squeeze ringing 0.08 / 0.36 at
  N = 512 / 2048, while gamma = 4/3 hydro gives 0.0012. This was the ringing of ladder item 2.

The coupling is now applied inside every RK stage (Gupta, Sharma & Mignone 2021, Et+Ecr
"Unsplit-pdv"; ``evolve_state._evolve_gas_state_unsplit_inner``). Measured (2026-10-09):

- A2.2: L1/eps at N = 512: 4.3e-4 / 4.1e-4 / 3.9e-4 / 3.7e-4, order 1.85-1.87 for every
  ``P_cr / P_th``; phase speed within 2e-5; ``C_cfl = 0.8``: <= 1.1e-3 eps.
- A2.3a: two-fluid vs exact 7.1e-6 at N = 1024 (order 1.86; hydro 8.3e-6, order 1.87),
  two-fluid vs hydro order 1.85.
- A2.3b: ringing 0.0026 / 0.0027 (N = 512 / 2048); hydro 0.0012 / 0.0013.

See ``PROGRESS_PHASEA.md`` (2026-10-09).
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
    PERIODIC_BOUNDARY,
    finalize_config,
)

# astronomix functions
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# Amplitudes of 1e-6 need float64.
jax.config.update("jax_enable_x64", True)

GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
PICS_DIR = Path(__file__).resolve().parent / "pics" / "02_adiabatic_compression"


def _run(num_cells, rho, u, p_th, p_cr, t_end, boundary, cosmic_rays=True, gamma=GAMMA, C_cfl=0.4):
    """1D float64 run from profile functions of x; ``cosmic_rays=False`` is plain hydro."""
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=1.0,
        numerical_precision=DOUBLE_PRECISION,
        boundary_settings=BoundarySettings1D(left_boundary=boundary, right_boundary=boundary),
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=cosmic_rays),
    )
    rv = get_registered_variables(config)
    x = np.asarray(get_helper_data(config).geometric_centers)
    state = jnp.zeros((rv.num_vars, num_cells))
    state = state.at[rv.density_index].set(rho(x)).at[rv.velocity_index].set(u(x))
    state = state.at[rv.pressure_index].set(p_th(x))
    if cosmic_rays:
        state = state.at[rv.cosmic_ray_e_index].set(p_cr(x) / (GAMMA_CR - 1.0))
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=t_end,
        gamma=gamma,
        C_cfl=C_cfl,
        cosmic_ray_grey_params=CosmicRayGreyParams(gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0),
    )
    final = np.asarray(time_integration(state, config, params, rv))
    assert not np.any(np.isnan(final)), "NaN in the run."
    return x, final[rv.density_index], final[rv.velocity_index], final[rv.pressure_index], (
        final[rv.cosmic_ray_e_index] * (GAMMA_CR - 1.0) if cosmic_rays else np.zeros(num_cells)
    )


def _order(resolutions, errors):
    return -np.polyfit(np.log(resolutions), np.log(errors), 1)[0]


def test_cr_linear_sound_wave(
    ratios=(0.0, 0.5, 2.0, 10.0),
    resolutions=(64, 128, 256, 512),
    min_order=1.8,
    max_phase_error=1e-4,
    max_eigenvector_error=1e-3,
    max_error_cfl08=1e-2,
):
    """A2.2: the CR-modified sound wave converges at second order and is stable.

    Args:
        ratios: ``P_cr / P_th`` values (``rho = P_th = 1``).
        resolutions: Cell counts.
        min_order: Minimum L1 order of ``rho`` (measured 1.85-1.87).
        max_phase_error: Bound on the relative phase-speed error at the finest N (measured
            <= 2e-5).
        max_eigenvector_error: Bound on ``abs(ratio - 1)`` of
            ``(delta P_cr / P_cr) / (delta rho / rho) / gamma_cr`` at the finest N.
        max_error_cfl08: Bound on L1/eps at the finest N with ``C_cfl = 0.8`` (measured <= 1.1e-3;
            the operator-split coupling gave up to 1e5).
    """
    eps = 1e-6
    wave = lambda x: np.sin(2.0 * np.pi * x)  # noqa: E731
    results = {}
    for ratio in ratios:
        c_eff = np.sqrt(GAMMA + GAMMA_CR * ratio)
        profiles = (
            lambda x: 1.0 + eps * wave(x),
            lambda x: c_eff * eps * wave(x),
            lambda x: 1.0 + GAMMA * eps * wave(x),
            lambda x, r=ratio: r * (1.0 + GAMMA_CR * eps * wave(x)),
        )
        errors = []
        for n in resolutions:
            x, rho, _, _, p_cr = _run(n, *profiles, 1.0 / c_eff, PERIODIC_BOUNDARY)
            errors.append(float(np.mean(np.abs(rho - (1.0 + eps * wave(x))))) / eps)
        mode_rho = np.sum((rho - 1.0) * np.exp(-2j * np.pi * x))
        phase_error = float(-np.angle(mode_rho / (-0.5j * resolutions[-1] * eps)) / (2.0 * np.pi))
        eigen_error = (
            float(abs(abs(np.sum((p_cr - ratio) * np.exp(-2j * np.pi * x))) / ratio / abs(mode_rho) / GAMMA_CR - 1.0))
            if ratio > 0
            else 0.0
        )
        x8, rho8, _, _, _ = _run(resolutions[-1], *profiles, 1.0 / c_eff, PERIODIC_BOUNDARY, C_cfl=0.8)
        error_cfl08 = float(np.mean(np.abs(rho8 - (1.0 + eps * wave(x8))))) / eps
        results[ratio] = dict(errors=errors, order=_order(resolutions, errors), phase=phase_error,
                              eigen=eigen_error, cfl08=error_cfl08, final=(x, rho))
        print(
            f"P_cr/P_th {ratio:4.1f}: L1/eps {' '.join(f'{e:.2e}' for e in errors)}, order "
            f"{results[ratio]['order']:.2f}, phase {phase_error:+.1e}, eigenvector {eigen_error:.1e}, "
            f"C_cfl 0.8: {error_cfl08:.2e}"
        )

    fig, (ax_conv, ax_err) = plt.subplots(1, 2, figsize=(13, 5))
    for ratio, r in results.items():
        ax_conv.loglog(resolutions, r["errors"], "o-", label=f"P_cr/P_th = {ratio:g} (order {r['order']:.2f})")
        x, rho = r["final"]
        ax_err.plot(x, (rho - 1.0 - eps * wave(x)) / eps, label=f"P_cr/P_th = {ratio:g}")
    n = np.array(resolutions, dtype=float)
    ax_conv.loglog(n, results[ratios[0]]["errors"][0] * (n[0] / n) ** 2, "k--", label=r"$\propto N^{-2}$")
    ax_conv.set_xlabel("N")
    ax_conv.set_ylabel(r"L1($\rho$ error) / $\epsilon$ after one period")
    ax_conv.set_title("A2.2: CR-modified sound wave")
    ax_conv.legend(fontsize=8)
    ax_err.set_xlabel("x")
    ax_err.set_ylabel(r"$\rho$ error / $\epsilon$ at N = %d" % resolutions[-1])
    ax_err.set_title("Error after one period")
    ax_err.legend(fontsize=8)
    fig.suptitle("A2.2 (stage-wise CR coupling): second order at every CR pressure fraction")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_linear_sound_wave_test.svg")
    plt.close(fig)

    for ratio, r in results.items():
        assert r["order"] >= min_order, f"P_cr/P_th {ratio}: order {r['order']:.2f} < {min_order}."
        assert abs(r["phase"]) < max_phase_error, f"P_cr/P_th {ratio}: phase error {r['phase']:.2e}."
        assert r["eigen"] < max_eigenvector_error, f"P_cr/P_th {ratio}: eigenvector error {r['eigen']:.2e}."
        assert r["cfl08"] < max_error_cfl08, f"P_cr/P_th {ratio}: C_cfl 0.8 error {r['cfl08']:.2e}."


def test_cr_hydro_equivalence(
    resolutions=(64, 128, 256, 512, 1024),
    min_order=1.8,
    max_ringing_ratio=3.0,
    squeeze_resolutions=(512, 2048),
):
    """A2.3: the CR-only limit reproduces gamma = 4/3 hydro.

    Args:
        resolutions: Cell counts for the simple wave.
        min_order: Minimum L1 order of the two-fluid run vs the exact solution and vs hydro
            (measured 1.86 / 1.85).
        max_ringing_ratio: Bound on the squeeze ringing (``rho`` range in 0.28 < x < 0.42)
            relative to gamma = 4/3 hydro (measured ~2.1).
        squeeze_resolutions: Cell counts for the squeeze.
    """
    amplitude = 0.1
    c0 = np.sqrt(GAMMA_CR)
    t_shock = 1.0 / ((GAMMA_CR + 1.0) / 2.0 * 2.0 * np.pi * amplitude)
    t_end = 0.5 * t_shock

    def simple_wave(x):
        u = amplitude * np.sin(2.0 * np.pi * x)
        rho = ((c0 + (GAMMA_CR - 1.0) / 2.0 * u) / c0) ** (2.0 / (GAMMA_CR - 1.0))
        return u, rho, rho**GAMMA_CR

    def exact(x, t):
        x0 = x.copy()
        for _ in range(300):  # foot of the C+ characteristic: x = x0 + (u + c)(x0) t
            x0 = x - (c0 + (GAMMA_CR + 1.0) / 2.0 * amplitude * np.sin(2.0 * np.pi * x0)) * t
        return simple_wave(x0)

    errors_two_fluid, errors_hydro, differences = [], [], []
    for n in resolutions:
        x, rho_cr, _, _, _ = _run(
            n, lambda x: simple_wave(x)[1], lambda x: simple_wave(x)[0],
            lambda x: 1e-6 * simple_wave(x)[2], lambda x: simple_wave(x)[2], t_end, PERIODIC_BOUNDARY,
        )
        _, rho_hy, _, _, _ = _run(
            n, lambda x: simple_wave(x)[1], lambda x: simple_wave(x)[0], lambda x: simple_wave(x)[2],
            None, t_end, PERIODIC_BOUNDARY, cosmic_rays=False, gamma=GAMMA_CR,
        )
        rho_exact = exact(x, t_end)[1]
        errors_two_fluid.append(float(np.mean(np.abs(rho_cr - rho_exact))))
        errors_hydro.append(float(np.mean(np.abs(rho_hy - rho_exact))))
        differences.append(float(np.mean(np.abs(rho_cr - rho_hy))))
    order_two_fluid = _order(resolutions, errors_two_fluid)
    order_difference = _order(resolutions, differences)
    print(f"simple wave: two-fluid vs exact {' '.join(f'{e:.2e}' for e in errors_two_fluid)} (order {order_two_fluid:.2f}); "
          f"hydro vs exact order {_order(resolutions, errors_hydro):.2f}; two-fluid vs hydro order {order_difference:.2f}")

    ringing = {}
    squeeze_profiles = {}
    for n in squeeze_resolutions:
        x, rho_cr, _, _, _ = _run(
            n, lambda x: 1.0 + 0.0 * x, lambda x: -0.2 * (x - 0.5), lambda x: 1e-6 / 3.0 + 0.0 * x,
            lambda x: 1.0 / 3.0 + 0.0 * x, 1.0, OPEN_BOUNDARY,
        )
        _, rho_hy, _, _, _ = _run(
            n, lambda x: 1.0 + 0.0 * x, lambda x: -0.2 * (x - 0.5), lambda x: 1.0 / 3.0 + 0.0 * x,
            None, 1.0, OPEN_BOUNDARY, cosmic_rays=False, gamma=GAMMA_CR,
        )
        window = (x > 0.28) & (x < 0.42)
        ringing[n] = (float(np.ptp(rho_cr[window])), float(np.ptp(rho_hy[window])))
        squeeze_profiles[n] = (x, rho_cr, rho_hy)
        print(f"squeeze N = {n}: rho ringing two-fluid {ringing[n][0]:.4f}, hydro {ringing[n][1]:.4f}")

    fig, (ax_conv, ax_squeeze) = plt.subplots(1, 2, figsize=(13, 5))
    ax_conv.loglog(resolutions, errors_two_fluid, "o-", label=f"two-fluid vs exact (order {order_two_fluid:.2f})")
    ax_conv.loglog(resolutions, errors_hydro, "s--", label="gamma = 4/3 hydro vs exact")
    ax_conv.loglog(resolutions, differences, "^:", label=f"two-fluid vs hydro (order {order_difference:.2f})")
    ax_conv.set_xlabel("N")
    ax_conv.set_ylabel(r"L1($\rho$)")
    ax_conv.set_title("A2.3a: simple wave, t = 0.5 t_shock")
    ax_conv.legend(fontsize=8)
    x, rho_cr, rho_hy = squeeze_profiles[squeeze_resolutions[-1]]
    ax_squeeze.plot(x, rho_hy, "k-", lw=1, label="gamma = 4/3 hydro")
    ax_squeeze.plot(x, rho_cr, "C1--", lw=1, label=r"two-fluid, $P_{th} = 10^{-6} P_{cr}$")
    ax_squeeze.set_xlabel("x")
    ax_squeeze.set_ylabel(r"$\rho / \rho_0$")
    ax_squeeze.set_title(f"A2.3b: item-2 squeeze, N = {squeeze_resolutions[-1]}")
    ax_squeeze.legend(fontsize=8)
    fig.suptitle("A2.3 (stage-wise CR coupling): the CR-only limit reproduces gamma = 4/3 hydro")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_hydro_equivalence_test.svg")
    plt.close(fig)

    assert order_two_fluid >= min_order, f"simple wave order {order_two_fluid:.2f} < {min_order}."
    assert order_difference >= min_order, f"two-fluid vs hydro order {order_difference:.2f} < {min_order}."
    for n, (two_fluid, hydro) in ringing.items():
        assert two_fluid <= max_ringing_ratio * hydro, (
            f"squeeze N = {n}: ringing {two_fluid:.4f} > {max_ringing_ratio} x hydro {hydro:.4f}."
        )


if __name__ == "__main__":
    test_cr_linear_sound_wave()
    test_cr_hydro_equivalence()
