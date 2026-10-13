"""
Sovinec test: direct measurement of the cross-field numerical diffusivity
``kappa_perp,num`` (Phase A test plan A3.1, with A3.5 merged in).

Sovinec et al. (2004, JCP 195, 355, Sec. 4.2): ``B`` tangent to the contours of ``psi =
cos(pi x) cos(pi y)``, so ``b . grad psi = 0`` and ``div(K grad psi) = -2 pi^2 kappa_perp
psi``: the lowest mode of the anisotropic operator sees only the perpendicular
diffusivity. Whatever decays it beyond ``kappa_perp`` is numerical. Used for CR transport
by Pakmor et al. (2016, Sec. 3.4) and Dubois & Commerçon (2016, Sec. 3.6); no two-moment
CR paper runs it (PROGRESS_PHASEA.md, 2026-10-12 literature checks). The spec followed
here is the PROGRESS_PHASEA.md log entry "A3.1 / A3.5 / A4.4 and the source hook".

Setup (deviations from the papers, none change the transport):

- Periodic box ``[-1, 1]^2`` (four Sovinec cells; ``FIXED_BOUNDARY`` cannot carry the
  field-dependent Dirichlet problem in MHD). With N divisible by 4 the separatrices x, y
  = +-1/2 are cell faces, and for an odd-symmetric scheme this is the Dirichlet problem
  with odd-reflection ghost cells; checked in every run (antisymmetry of ``e - e_mean``
  across x, y = 1/2).
- ``B = B0 (cos(pi x) sin(pi y), -sin(pi x) cos(pi y))`` at cell centres, heavy static gas
  (``cr_static_gas``: ``rho = 1e4``, ``E_SCALE = 1e-7``, ``B0 = 1e-5``), 2D MHD,
  ``anisotropic_transport`` + ``diffusive_relaxation``, float64, ``C_cfl = 0.4``.
- ``kappa_par = 1``, ``kappa_perp`` = 0 or 1e-2 (M7 / Girichidis et al. 2016 ratio).
- ``v_red = 96`` (optical depth ``tau = (gamma_cr - 1) v_red dx / kappa_par`` = 2 / 1 /
  0.5 / 0.25 at 16 / 32 / 64 / 128 cells per unit length, through M7's thick regime ``tau
  ~ 3`` into thin cells); the A3.5 scan varies ``v_red`` = 24 ... 384 at 32 and 64 cells
  per unit length. The plan's ``v_red`` 4 ... 32 would leave the parallel transport
  non-diffusive at ``kappa_par = 1`` (``nu = c^2 / kappa_par`` <= ``kappa_par k^2``).

Measurement (our variant, not published): the **decay rate** of the ``psi`` mode.
Start from ``e = e_off + A psi`` (cell averages, ``A = 0.5``, ``e_off = 1``), ``F =
-K grad e``, no source, and fit ``lambda`` to the projected amplitude ``a(t) = <e -
e_mean, psi> / <psi, psi>`` over ``t`` in [0.5, 2] (after the parallel and two-moment
transients). ``kappa_tot = lambda / (2 pi^2)``.

- ``kappa_perp = 0``: ``kappa_perp,num / kappa_par = kappa_tot / kappa_par``.
- ``kappa_perp > 0``: ``kappa_perp,num / kappa_perp = lambda / lambda_iso - 1`` with
  ``lambda_iso`` from an isotropic run with ``kappa = kappa_perp`` on the same grid (the
  Dubois & Commerçon normalization, which cancels the Laplacian's own error).
- At ``kappa_perp = 0`` every ``f(psi)`` is steady in the continuum, so the fitted rate
  only measures the damping of the ``psi`` projection: the residual of the projection is
  recorded, and the steady-state anchor (source hook ``test_e_cr_source``, ``Q = Q0
  psi``) checks the rate.

Tests (gates in each docstring; numbers in ``baselines.json`` and PROGRESS_PHASEA.md):

1. ``test_sovinec_resolution``: ``kappa_perp,num`` at 16 ... 128 cells per unit length,
   for the variants limiter (MINMOD, MC, van Leer) x wave speed (``CR_WAVE_SPEED_FACE_NORMAL``,
   ``CR_WAVE_SPEED_FIELD_ALIGNED``) and ``CR_WAVE_SPEED_PROJECTED`` with the monotonicity
   guard off, plus MINMOD / MC / van Leer face-normal with the guard on, at ``kappa_perp``
   = 0 and 1e-2. ``DEFAULT_VARIANT`` (van Leer, face-normal, guard on) is the library
   default since 2026-10-13. ``SOVINEC_KAPPA_PERPS=0`` (or ``0.01``) and
   ``SOVINEC_VARIANTS=0,1`` (indices into ``VARIANTS``) restrict a run, to split the test
   over GPUs.
2. ``test_sovinec_vred_scan`` (A3.5): ``d ln kappa_perp,num / d ln v_red`` at 32 and 64
   cells per unit length, face-normal and field-aligned wave speed (MINMOD, guard off).
3. ``test_sovinec_steady_anchor``: steady state with the source agrees with the decay
   rate within 10% (32 and 64 cells per unit length, ``kappa_perp`` 0 and 1e-2), on
   ``ANCHOR_VARIANT`` (MINMOD, face-normal, guard off): it checks the measure, not the
   scheme, and with the less diffusive limiters the steady state takes far longer.
4. ``test_sovinec_figure``: ``pics/03_anisotropic_diffusion/cr_sovinec_test.svg``, from the
   values registered in ``baselines.json`` by tests 1 and 2 (no runs).

Guard misfire (2026-10-12, fixed 2026-10-13): with the three-face shape sensor the guard
fired on smooth extrema in optically thick cells. MINMOD clips the extrema of the ``psi``
mode slightly; the sensor then read the flattened top as a step (0.47 > onset 0.45 next to
a near-zero jump) and raised ``R`` towards 1, i.e. the dissipation by up to ``tau``-fold
(isotropic, ``kappa = 0.01``, ``tau = 200``: decay rate 4.62x the exact one; anisotropic
``kappa_perp = 0``, 16 cells per unit length: 0.051 instead of 0.021; odd symmetry broken).
The five-face sensor (``cr_guard_wide_sensor``, default) does not fire on the mode: with the
guard on, every limiter gives the guard-off values to within 0.4%.

Results (2026-10-13; full tables in PROGRESS_PHASEA.md): ``kappa_perp,num / kappa_par`` of
the default (van Leer, face-normal) is 3.4e-3 / 1.1e-3 / 2.2e-4 / 3.5e-5 at 16 ... 128 cells
per unit length (orders 1.6 / 2.3 / 2.7), 6-29x below MINMOD (2.0e-2 / 8.9e-3 / 3.4e-3 /
1.0e-3, the default until 2026-10-12) and ~3x above MC. Field-aligned and projected speeds
leak 2-4x more in thick cells. The leak adds to a physical ``kappa_perp`` (1e-2, van Leer:
0.41 / 0.11 / 0.022 / 0.0033 of it; MINMOD 2.0 / 0.91 / 0.32 / 0.095). A3.5: ~``v_red`` in
thin cells, ``v_red``-independent for ``tau`` >~ 2. The steady-state anchor agrees with the
decay rate to <= 3.2%.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import os
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

# baseline registry and the static-gas setup
from cr_baselines import Baselines, load as load_baselines
from cr_static_gas import E_SCALE, check_gas_at_rest, max_gas_speed, static_gas_state

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    DOUBLE_MINMOD,
    FINITE_VOLUME,
    MINMOD,
    VAN_LEER,
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
    CR_WAVE_SPEED_FACE_NORMAL,
    CR_WAVE_SPEED_FIELD_ALIGNED,
    CR_WAVE_SPEED_PROJECTED,
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

jax.config.update("jax_enable_x64", True)

GAMMA_CR = 4.0 / 3.0
BOX_SIZE = 2.0  # [-1, 1]^2
C_CFL = 0.4
KAPPA_PAR = 1.0
KAPPA_PERPS = (0.0, 1e-2)
V_RED = 96.0
AMPLITUDE = 0.5
E_OFFSET = 1.0
K_SQUARED = 2.0 * np.pi**2
T_END = 2.0
FIT_START = 0.5
NUM_SNAPSHOTS = 21

# cells per unit length; N = 2 x this (box 2), divisible by 4
CELLS_PER_UNIT = (16, 32, 64, 128)
SCAN_CELLS_PER_UNIT = (32, 64)
SCAN_V_REDS = (24.0, 48.0, 96.0, 192.0, 384.0)
ANCHOR_CELLS_PER_UNIT = (32, 64)

GUARD_OFF = dict(cr_guard_sensor_onset=2.0, cr_guard_sensor_full=3.0)
MODEL_NAMES = {
    CR_WAVE_SPEED_FACE_NORMAL: "face-normal",
    CR_WAVE_SPEED_FIELD_ALIGNED: "field-aligned",
    CR_WAVE_SPEED_PROJECTED: "projected",
}
LIMITER_NAMES = {MINMOD: "MINMOD", DOUBLE_MINMOD: "MC", VAN_LEER: "van Leer"}
# (limiter, wave-speed model, guard). The limiter is CosmicRayGreyConfig.cr_limiter
# (CR rows only). Guard on = the library default guard (five-face sensor, MINMOD
# blend since 2026-10-13); guard off = the M7 production setting. The three-face
# sensor used until 2026-10-12 fired on MINMOD's clipped extrema in thick cells
# (module docstring).
VARIANTS = (
    (MINMOD, CR_WAVE_SPEED_FACE_NORMAL, False),
    (MINMOD, CR_WAVE_SPEED_FIELD_ALIGNED, False),
    (DOUBLE_MINMOD, CR_WAVE_SPEED_FACE_NORMAL, False),
    (DOUBLE_MINMOD, CR_WAVE_SPEED_FIELD_ALIGNED, False),
    (MINMOD, CR_WAVE_SPEED_PROJECTED, False),
    (MINMOD, CR_WAVE_SPEED_FACE_NORMAL, True),
    (DOUBLE_MINMOD, CR_WAVE_SPEED_FACE_NORMAL, True),
    (VAN_LEER, CR_WAVE_SPEED_FACE_NORMAL, False),
    (VAN_LEER, CR_WAVE_SPEED_FACE_NORMAL, True),
)
DEFAULT_VARIANT = VARIANTS[8]  # the library default since 2026-10-13
# The steady-state anchor checks the decay-rate measure, not a scheme. It runs to
# 3 / rate, so it uses the leakiest variant: with van Leer or MC at kappa_perp = 0
# that is 10^2-10^3 time units (one 64/unit run did not finish in 7.5 h).
ANCHOR_VARIANT = VARIANTS[0]

PICS_DIR = Path(__file__).resolve().parent / "pics" / "03_anisotropic_diffusion"
TEST_FILE = "cr_sovinec.py"


def _variant_name(variant) -> str:
    limiter, model, guard = variant
    return f"{LIMITER_NAMES[limiter]}, {MODEL_NAMES[model]}" + (", guard on" if guard else "")


def _variant_key(variant) -> str:
    limiter, model, guard = variant
    return f"{LIMITER_NAMES[limiter].lower().replace(' ', '_')}_{MODEL_NAMES[model].replace('-', '_')}" + ("_guard" if guard else "")


def _optical_depth(v_red: float, cells_per_unit: int) -> float:
    return (GAMMA_CR - 1.0) * v_red / cells_per_unit / KAPPA_PAR


def _cell_average_factor(dx: float) -> float:
    """Cell average of ``cos(pi x)`` / ``sin(pi x)`` over a cell, per point value."""
    return float(np.sinc(0.5 * dx))  # np.sinc(z) = sin(pi z) / (pi z)


@lru_cache(maxsize=None)
def _sovinec_run(
    cells_per_unit: int,
    kappa_perp: float,
    variant=DEFAULT_VARIANT,
    v_red: float = V_RED,
    isotropic: bool = False,
    source_rate: float = 0.0,
    initial_amplitude: float = AMPLITUDE,
    t_end: float = T_END,
):
    """Run the ``psi`` mode. ``isotropic=True``: hydro, isotropic ``kappa =
    kappa_perp`` (the reference rate). ``source_rate`` > 0: ``Q = source_rate psi``.

    Returns:
        dict with ``times``, ``amplitude`` (projection per snapshot), ``residual``
        (final L2 of ``e - e_mean - a psi`` per ``a``), ``antisymmetry`` (max over x, y
        of the odd-reflection violation across 1/2, per ``A``), ``max_speed``, ``dx``.
    """
    limiter, model, guard = variant
    num_cells = 2 * cells_per_unit
    dx = BOX_SIZE / num_cells
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        mhd=not isotropic,
        solver_mode=FINITE_VOLUME,
        limiter=limiter,
        dimensionality=2,
        num_cells=num_cells,
        box_size=BOX_SIZE,
        boundary_settings=BoundarySettings(periodic, periodic),
        numerical_precision=DOUBLE_PRECISION,
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True,
            diffusive_relaxation=True,
            anisotropic_transport=not isotropic,
            cr_wave_speed_model=model,
            cr_limiter=limiter,
            # The test-only e_cr source does not support the CR entropy row (on by default
            # since 2026-10-13); static gas has no shocks, so the hybrid would not act anyway.
            cr_entropy=False,
        ),
    )
    registered_variables = get_registered_variables(config)
    centers = np.asarray(get_helper_data(config).geometric_centers)
    x = centers[..., 0] - 0.5 * BOX_SIZE
    y = centers[..., 1] - 0.5 * BOX_SIZE
    s = _cell_average_factor(dx)
    psi = np.cos(np.pi * x) * np.cos(np.pi * y) * s**2
    # F = -K grad e = -kappa_perp grad e (b . grad psi = 0; isotropic: kappa =
    # kappa_perp), cell averages
    grad_scale = initial_amplitude * np.pi * s**2
    f_cr = (
        kappa_perp * grad_scale * np.sin(np.pi * x) * np.cos(np.pi * y),
        kappa_perp * grad_scale * np.cos(np.pi * x) * np.sin(np.pi * y),
    )
    magnetic_field = None
    if not isotropic:
        magnetic_field = (
            np.cos(np.pi * x) * np.sin(np.pi * y),
            -np.sin(np.pi * x) * np.cos(np.pi * y),
            np.zeros_like(x),
        )
    state = static_gas_state(
        registered_variables, (num_cells, num_cells),
        e_cr=E_OFFSET + initial_amplitude * psi, f_cr=f_cr, magnetic_field=magnetic_field,
    )
    config = finalize_config(config, state.shape)
    cr_params = dict(
        gamma_cr=GAMMA_CR,
        reduced_streaming_speed=v_red,
        diffusion_coefficient=kappa_perp if isotropic else KAPPA_PAR,
        perpendicular_diffusion_coefficient=0.0 if isotropic else kappa_perp,
        **({} if guard else GUARD_OFF),
    )
    if source_rate > 0.0:
        cr_params["test_e_cr_source"] = jnp.asarray(E_SCALE * source_rate * psi)
    params = SimulationParams(
        t_end=t_end,
        C_cfl=C_CFL,
        snapshot_timepoints=jnp.linspace(0.0, t_end, NUM_SNAPSHOTS),
        cosmic_ray_grey_params=CosmicRayGreyParams(**cr_params),
    )
    result = time_integration(state, config, params, registered_variables)
    states = np.asarray(result.states)
    e = states[:, registered_variables.cosmic_ray_e_index] / E_SCALE
    assert np.all(np.isfinite(e)), f"NaNs at {cells_per_unit} cells per unit length."
    deviation = e - e.mean(axis=(1, 2), keepdims=True)
    amplitude = np.einsum("tij,ij->t", deviation, psi) / np.sum(psi**2)
    residual = float(np.sqrt(np.mean((deviation[-1] - amplitude[-1] * psi) ** 2)
                             / np.mean((amplitude[-1] * psi) ** 2)))
    # odd reflection across x = 1/2 (cell i <-> 3N/2 - 1 - i) and y = 1/2
    mirror = (3 * num_cells // 2 - 1 - np.arange(num_cells)) % num_cells
    antisymmetry = max(
        float(np.abs(deviation[-1] + deviation[-1][mirror, :]).max()),
        float(np.abs(deviation[-1] + deviation[-1][:, mirror]).max()),
    ) / initial_amplitude
    return dict(
        e=e,
        times=np.asarray(result.time_points),
        amplitude=amplitude,
        residual=residual,
        antisymmetry=antisymmetry,
        max_speed=max_gas_speed(states, registered_variables, 2),
        dx=dx,
        t_end=t_end,
    )


def _decay_rate(run: dict, t_start: float = FIT_START):
    """Least-squares rate of ``ln a(t)`` over ``t >= t_start``, and the relative
    difference of the rates over the two halves of that window (constancy)."""
    times, amplitude = run["times"], run["amplitude"]
    window = times >= t_start - 1e-12

    def rate(mask):
        return -float(np.polyfit(times[mask], np.log(amplitude[mask]), 1)[0])

    total = rate(window)
    middle = 0.5 * (t_start + times[-1])
    first, second = rate(window & (times <= middle + 1e-12)), rate(times >= middle - 1e-12)
    return total, abs(first - second) / total


@lru_cache(maxsize=None)
def _kappa_num(cells_per_unit: int, kappa_perp: float, variant=DEFAULT_VARIANT, v_red: float = V_RED):
    """``kappa_perp,num / kappa_par`` (``kappa_perp = 0``) or ``kappa_perp,num /
    kappa_perp`` (``kappa_perp > 0``), with the run diagnostics."""
    run = _sovinec_run(cells_per_unit, kappa_perp, variant, v_red)
    rate, rate_drift = _decay_rate(run)
    check_gas_at_rest(run["max_speed"], run["t_end"], run["dx"])
    if kappa_perp == 0.0:
        value = rate / K_SQUARED / KAPPA_PAR
        rate_iso = None
    else:
        # isotropic reference with the variant's limiter, guard off
        iso_variant = (variant[0], CR_WAVE_SPEED_FACE_NORMAL, False)
        rate_iso, _ = _decay_rate(_sovinec_run(cells_per_unit, kappa_perp, iso_variant, v_red, isotropic=True))
        value = rate / rate_iso - 1.0
    return dict(value=value, rate=rate, rate_iso=rate_iso, rate_drift=rate_drift,
                residual=run["residual"], antisymmetry=run["antisymmetry"])


def _orders(cells, values):
    return [float(np.log(values[i] / values[i + 1]) / np.log(cells[i + 1] / cells[i]))
            for i in range(len(cells) - 1)]


def _print_row(label, cells, rows):
    values = [r["value"] for r in rows]
    print(
        f"{label}: " + " / ".join(f"{v:.3e}" for v in values)
        + " (cells/unit " + " / ".join(str(c) for c in cells) + "), orders "
        + " / ".join(f"{o:.2f}" for o in _orders(cells, values))
        + "; rate drift <= " + f"{max(r['rate_drift'] for r in rows):.1e}"
        + ", residual <= " + f"{max(r['residual'] for r in rows):.1e}"
        + ", antisymmetry <= " + f"{max(r['antisymmetry'] for r in rows):.1e}"
    )


def test_sovinec_resolution(max_antisymmetry: float = 1e-9, max_rate_drift: float = 0.05, kappa_perps=None):
    """``kappa_perp,num`` against resolution for every variant, ``kappa_perp`` = 0
    (per ``kappa_par``) and 1e-2 (per ``kappa_perp``).

    Gates (guard-off variants): the periodic box is the Dirichlet problem
    (antisymmetry across the separatrices <= ``max_antisymmetry``, round-off: up to
    1.8e-10 measured); at ``kappa_perp`` > 0 the decay is exponential (rates over the
    two halves of the fit window within ``max_rate_drift``; <= 1.3e-2 measured). At
    ``kappa_perp = 0`` the numerical leak is not a single-mode decay (drift up to 0.34,
    MC at 16 cells per unit length), so the drift is only printed there; the steady-state
    anchor checks that rate. Guard-on variants: the same gates since 2026-10-13 (the
    three-face sensor broke the symmetry, 0.1-0.3 of ``A``; the five-face one does not
    fire on this mode). All: gas at rest;
    ``kappa_perp,num`` of the default variant decreases with resolution; every
    value within 1.5x of ``baselines.json``. The order (>= 1.5 in the plan) and
    the absolute level (D1) are tracked, not gated.
    """
    baselines = Baselines(f"{TEST_FILE}::test_sovinec_resolution")
    failures = []
    if kappa_perps is None:
        kappa_perps = tuple(float(k) for k in os.environ.get("SOVINEC_KAPPA_PERPS", "").split(",") if k) or KAPPA_PERPS
    for kappa_perp in kappa_perps:
        unit = "kappa_par" if kappa_perp == 0.0 else "kappa_perp"
        selected = os.environ.get("SOVINEC_VARIANTS", "")
        variants = tuple(VARIANTS[int(i)] for i in selected.split(",")) if selected else VARIANTS
        for variant in variants:
            if kappa_perp > 0.0 and variant[1] == CR_WAVE_SPEED_PROJECTED:
                continue
            rows = [_kappa_num(c, kappa_perp, variant) for c in CELLS_PER_UNIT]
            _print_row(f"kappa_perp {kappa_perp:g}, {_variant_name(variant)}: kappa_perp,num / {unit}",
                       CELLS_PER_UNIT, rows)
            for c, r in zip(CELLS_PER_UNIT, rows):
                baselines.upper(f"kperp{kappa_perp:g}_{_variant_key(variant)}_c{c}", r["value"])
                if r["antisymmetry"] > max_antisymmetry:
                    failures.append(f"{_variant_name(variant)}, {c}/unit: antisymmetry {r['antisymmetry']:.1e}")
                if kappa_perp > 0.0 and r["rate_drift"] > max_rate_drift:
                    failures.append(f"{_variant_name(variant)}, {c}/unit: rate drift {r['rate_drift']:.2f}")
            if variant == DEFAULT_VARIANT:
                values = [r["value"] for r in rows]
                if not all(a > b for a, b in zip(values, values[1:])):
                    failures.append(f"kappa_perp {kappa_perp:g}: default kappa_perp,num not decreasing {values}")
                baselines.lower(f"kperp{kappa_perp:g}_default_order_fine",
                                _orders(CELLS_PER_UNIT, values)[-1], margin=0.15)
        if kappa_perp > 0.0:
            iso = [_kappa_num(c, kappa_perp)["rate_iso"] / (K_SQUARED * kappa_perp) - 1.0 for c in CELLS_PER_UNIT]
            print("  isotropic reference: lambda_iso / (2 pi^2 kappa_perp) - 1 = "
                  + " / ".join(f"{v:.2e}" for v in iso))
    assert not failures, "; ".join(failures)
    baselines.check()


def test_sovinec_vred_scan():
    """A3.5: the ``v_red`` dependence of ``kappa_perp,num`` (``kappa_perp = 0``) at
    fixed resolution. A slope near 1 in thin cells means the HLL dissipation
    (``~ v_red``) dominates the leak; a flat curve, the gradient projection.

    Gates: values within 1.5x of ``baselines.json``; slopes tracked.
    """
    baselines = Baselines(f"{TEST_FILE}::test_sovinec_vred_scan")
    for cells in SCAN_CELLS_PER_UNIT:
        for model in (CR_WAVE_SPEED_FACE_NORMAL, CR_WAVE_SPEED_FIELD_ALIGNED):
            variant = (MINMOD, model, False)
            values = [_kappa_num(cells, 0.0, variant, v)["value"] for v in SCAN_V_REDS]
            slopes = [float(np.log(values[i + 1] / values[i]) / np.log(SCAN_V_REDS[i + 1] / SCAN_V_REDS[i]))
                      for i in range(len(SCAN_V_REDS) - 1)]
            print(
                f"{cells}/unit, {_variant_name(variant)}: kappa_perp,num / kappa_par at v_red "
                + " / ".join(f"{v:g}" for v in SCAN_V_REDS) + " (tau "
                + " / ".join(f"{_optical_depth(v, cells):.2g}" for v in SCAN_V_REDS) + "): "
                + " / ".join(f"{v:.3e}" for v in values) + "; local slopes "
                + " / ".join(f"{s:.2f}" for s in slopes)
            )
            for v, value in zip(SCAN_V_REDS, values):
                baselines.upper(f"c{cells}_{MODEL_NAMES[model].replace('-', '_')}_v{v:g}", value)
    baselines.check()


def test_sovinec_steady_anchor(max_mismatch: float = 0.1):
    """The decay rate against the published measure, the steady state with the
    source ``Q = Q0 psi`` (``ANCHOR_VARIANT``: MINMOD, face-normal, guard off).

    Start at 1.2x the amplitude the decay rate predicts (``Q0 / lambda``), run to
    ``3 / lambda`` and extrapolate the steady amplitude from three late snapshots
    (Aitken); ``lambda_ss = Q0 / a_ss``.

    Gate: ``|lambda_ss / lambda - 1| <= max_mismatch`` at ``kappa_perp`` 0 and 1e-2,
    32 and 64 cells per unit length.
    """
    failures = []
    for kappa_perp in KAPPA_PERPS:
        for cells in ANCHOR_CELLS_PER_UNIT:
            rate = _kappa_num(cells, kappa_perp, ANCHOR_VARIANT)["rate"]
            q0 = AMPLITUDE * rate
            t_end = 3.0 / rate
            run = _sovinec_run(cells, kappa_perp, ANCHOR_VARIANT, V_RED, source_rate=q0,
                               initial_amplitude=1.2 * AMPLITUDE, t_end=t_end)
            check_gas_at_rest(run["max_speed"], run["t_end"], run["dx"])
            a = run["amplitude"]
            a1, a2, a3 = a[-11], a[-6], a[-1]  # equally spaced at t_end / 2, 3/4, 1
            a_ss = (a1 * a3 - a2**2) / (a1 + a3 - 2.0 * a2)
            rate_ss = q0 / a_ss
            mismatch = rate_ss / rate - 1.0
            print(
                f"kappa_perp {kappa_perp:g}, {cells}/unit: decay rate {rate:.5e}, steady-state rate "
                f"{rate_ss:.5e} (a_ss {a_ss:.6f}, a(t_end) {a3:.6f}), mismatch {mismatch:+.2e}; "
                f"residual {run['residual']:.1e}, antisymmetry {run['antisymmetry']:.1e}"
            )
            if abs(mismatch) > max_mismatch:
                failures.append(f"kappa_perp {kappa_perp:g}, {cells}/unit: mismatch {mismatch:+.2e}")
    assert not failures, "; ".join(failures)


def test_sovinec_figure():
    """Figure (no assertions) from the values tests 1 and 2 registered in
    ``baselines.json``: ``kappa_perp,num`` against resolution per variant
    (``kappa_perp = 0``) and against ``v_red`` (A3.5)."""
    registry = load_baselines()
    resolution = registry[f"{TEST_FILE}::test_sovinec_resolution"]
    scan = registry[f"{TEST_FILE}::test_sovinec_vred_scan"]
    fig, (ax_n, ax_v) = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    c = np.array(CELLS_PER_UNIT)
    for i, variant in enumerate(VARIANTS):
        values = [resolution[f"kperp0_{_variant_key(variant)}_c{n}"]["value"] for n in CELLS_PER_UNIT]
        ax_n.loglog(c, values, "x--" if variant[2] else "o-", color=f"C{i}", label=_variant_name(variant))
    ref = resolution[f"kperp0_{_variant_key(DEFAULT_VARIANT)}_c{CELLS_PER_UNIT[0]}"]["value"]
    ax_n.loglog(c, ref * (c / c[0]) ** -1, "k:", lw=0.8, label=r"$\propto dx$")
    ax_n.loglog(c, ref * (c / c[0]) ** -2, "k-.", lw=0.8, label=r"$\propto dx^2$")
    ax_n.set_xlabel("cells per unit length")
    ax_n.set_ylabel(r"$\kappa_{\perp,num} / \kappa_\parallel$")
    ax_n.set_title(rf"Sovinec decay rate, $\kappa_\perp = 0$, $v_{{red}}$ = {V_RED:g}")
    ax_n.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    v = np.array(SCAN_V_REDS)
    for j, cells in enumerate(SCAN_CELLS_PER_UNIT):
        for model, style in ((CR_WAVE_SPEED_FACE_NORMAL, "o-"), (CR_WAVE_SPEED_FIELD_ALIGNED, "s--")):
            values = [scan[f"c{cells}_{MODEL_NAMES[model].replace('-', '_')}_v{x:g}"]["value"] for x in SCAN_V_REDS]
            ax_v.loglog(v, values, style, color=f"C{j}", label=f"{cells}/unit, {MODEL_NAMES[model]}")
    ax_v.set_xlabel(r"$v_{red}$")
    ax_v.set_ylabel(r"$\kappa_{\perp,num} / \kappa_\parallel$")
    ax_v.set_title(r"A3.5: $v_{red}$ dependence (MINMOD, guard off)")
    ax_v.legend(fontsize=7)
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_sovinec_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    tests = (
        test_sovinec_resolution,
        test_sovinec_vred_scan,
        test_sovinec_steady_anchor,
        test_sovinec_figure,
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
