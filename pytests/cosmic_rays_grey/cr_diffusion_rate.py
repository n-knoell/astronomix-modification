"""
CR diffusion-rate pytests (DESIGN.md "Open: CR diffusion correctness", fix-plan
step 0): measure the ``e_cr`` diffusivity the two-moment scheme actually
produces, isotropic (1D, T1) and B-projected anisotropic (2D MHD, T2).

Written *before* the fix, per the plan's TDD rule -- most tests here fail on
the current code. Each test's docstring names the fix step that should make it
pass, so the file doubles as the gate for steps 1-4:

- step 1, convention: ``diffusion_coefficient`` is the ``e_cr`` diffusivity,
  ``d(e_cr)/dt = div(kappa grad e_cr)`` (Girichidis et al. 2016). Every
  target below is ``D = diffusion_coefficient``; the current code gives
  ``kappa/3``.
- step 2, relaxation inside every RK stage: removes the ``+nu dt_gas/2`` bias
  of the operator-split relaxation (D must not depend on the time step).
- step 3, tensor relaxation: removes the per-step perpendicular leak of the
  once-per-step B-projection, and adds ``perpendicular_diffusion_coefficient``.
- step 4, CR-specific HLL wave speed with the optical-depth reduction:
  removes most of the numerical diffusion in the stiff (optically thick per
  cell) regime and across a grid-aligned field.

Measurement: a tiny-amplitude Gaussian ``e_cr`` bump on a uniform static gas
in a periodic box (``F_cr = 0`` initially). D is half the slope of the
``e_cr``-weighted second moment over ``t >= t_end/3`` (skipping the
relaxation transient); in 2D the moments are resolved along/across B. With a
static gas the advective terms vanish, so D is the pure diffusion rate.

The time step is varied through ``C_cfl``: ``params.dt_max`` is ignored by the
UNSPLIT branch of ``_cfl_time_step`` (only the SPLIT branch applies it). Both
the hydro CFL and the current relaxation limit scale with ``C_cfl``, so
``nu dt`` scales with it either way.

Numbers measured on the current code (2026-10-04, before any fix step) are in
each test's docstring; units are ``kappa``.

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md.
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

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
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
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# D is a small difference of second moments -- needs float64.
jax.config.update("jax_enable_x64", True)

GAMMA_CR = 4.0 / 3.0
REDUCED_STREAMING_SPEED = 8.0
AMP = 1e-3
SIGMA0 = 0.05
NUM_SNAPSHOTS = 11

# T1 (1D isotropic): kappa / (v_red dx) = 1.25 at N = 256, i.e. diffusive but
# only a few mean free paths per cell.
KAPPA_1D = 0.02
T_END_1D = 0.05
# T1 stiff case: optical depth per cell tau = (gamma_cr - 1) v_red dx / kappa
# = 7.4 at N = 512, nu dt ~ 3 at C_cfl = 0.4 once the relaxation no longer
# limits dt.
KAPPA_STIFF = 0.0007
T_END_STIFF = 0.5

# T2 (2D MHD, |B| = 1 at angle theta to x).
KAPPA_2D = 0.02
T_END_2D = 0.05
ANGLES_DEG = (0.0, 30.0, 45.0)
RESOLUTIONS_2D = (64, 128, 256)

PERP_PARAM = "perpendicular_diffusion_coefficient"


@lru_cache(maxsize=None)
def _measure_diffusivity(
    dimensionality: int,
    num_cells: int,
    kappa: float,
    c_cfl: float,
    t_end: float,
    theta_deg: float = 0.0,
    kappa_perp: float = 0.0,
):
    """Run one bump and return ``(D_par, D_perp, num_iterations)``.

    1D: ``D_perp`` is NaN. 2D: MHD with uniform B at ``theta_deg``,
    ``anisotropic_transport=True``; ``D_par``/``D_perp`` are measured along
    and across B. ``kappa_perp > 0`` sets ``perpendicular_diffusion_coefficient``
    (fix step 3).
    """
    periodic = BoundarySettings1D(
        left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
    )
    config = SimulationConfig(
        mhd=dimensionality == 2,
        solver_mode=FINITE_VOLUME,
        dimensionality=dimensionality,
        num_cells=num_cells,
        box_size=1.0,
        boundary_settings=(
            periodic if dimensionality == 1 else BoundarySettings(periodic, periodic)
        ),
        numerical_precision=DOUBLE_PRECISION,
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(return_states=True),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True,
            diffusive_relaxation=True,
            anisotropic_transport=dimensionality == 2,
        ),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    theta = np.deg2rad(theta_deg)
    shape = (num_cells,) * dimensionality
    primitive_state = jnp.zeros((registered_variables.num_vars,) + shape)
    primitive_state = primitive_state.at[registered_variables.density_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(1.0)
    if dimensionality == 1:
        x = np.asarray(helper_data.geometric_centers)
        r2 = (x - 0.5) ** 2
    else:
        x = np.asarray(helper_data.geometric_centers[..., 0])
        y = np.asarray(helper_data.geometric_centers[..., 1])
        r2 = (x - 0.5) ** 2 + (y - 0.5) ** 2
        b_index = registered_variables.magnetic_index
        primitive_state = primitive_state.at[b_index.x].set(np.cos(theta))
        primitive_state = primitive_state.at[b_index.y].set(np.sin(theta))
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(
        AMP * jnp.exp(-0.5 * r2 / SIGMA0**2)
    )
    # F_cr = 0 initial.

    config = finalize_config(config, primitive_state.shape)

    cr_params = dict(
        gamma_cr=GAMMA_CR,
        reduced_streaming_speed=REDUCED_STREAMING_SPEED,
        diffusion_coefficient=kappa,
    )
    if kappa_perp > 0.0:
        assert PERP_PARAM in CosmicRayGreyParams._fields, (
            f"CosmicRayGreyParams has no {PERP_PARAM} yet -- physical "
            f"perpendicular diffusion is fix-plan step 3 (DESIGN.md)."
        )
        cr_params[PERP_PARAM] = kappa_perp
    params = SimulationParams(
        t_end=t_end,
        C_cfl=c_cfl,
        cosmic_ray_grey_params=CosmicRayGreyParams(**cr_params),
    )

    snapshots = time_integration(primitive_state, config, params, registered_variables)
    times = np.asarray(snapshots.time_points)
    e_cr = np.asarray(snapshots.states[:, registered_variables.cosmic_ray_e_index])
    assert np.all(np.isfinite(e_cr)), (
        f"CR diffusion run produced NaNs (dim={dimensionality}, N={num_cells}, "
        f"kappa={kappa}, C_cfl={c_cfl}, theta={theta_deg})."
    )

    var_par, var_perp = [], []
    for e in e_cr:
        w = e / e.sum()
        if dimensionality == 1:
            xc = (w * x).sum()
            var_par.append((w * (x - xc) ** 2).sum())
        else:
            xc, yc = (w * x).sum(), (w * y).sum()
            s_par = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
            s_perp = -(x - xc) * np.sin(theta) + (y - yc) * np.cos(theta)
            var_par.append((w * s_par**2).sum())
            var_perp.append((w * s_perp**2).sum())

    window = times >= t_end / 3.0
    d_par = 0.5 * np.polyfit(times[window], np.array(var_par)[window], 1)[0]
    d_perp = (
        0.5 * np.polyfit(times[window], np.array(var_perp)[window], 1)[0]
        if dimensionality == 2
        else float("nan")
    )
    return float(d_par), float(d_perp), int(snapshots.num_iterations)


def _save(fig, name):
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(pics_dir / name)
    plt.close(fig)


# -------------------------------------------------------------
# ==================== T1: 1D isotropic =======================
# -------------------------------------------------------------


def test_cr_diffusion_time_step_independence_1d(
    num_cells: int = 512,
    c_cfl_values: tuple = (0.4, 0.2, 0.1),
    tol: float = 0.02,
):
    """T1a: the diffusion rate must not depend on the time step.

    Passes after fix step 2. The operator-split relaxation integrates
    ``F_cr`` undamped through the RK2 gas step, so today
    ``D = D_0 (1 + nu dt / 2)``. Before the fix, with ``nu dt`` = 0.31 / 0.16 /
    0.08, D / kappa was 0.393 / 0.368 / 0.355. The range relative to the
    smallest-dt value is 11%.

    Args:
        num_cells: The resolution.
        c_cfl_values: The CFL numbers scanned.
        tol: Maximum relative spread of D over the scan.
    """
    diffusivities = [
        _measure_diffusivity(1, num_cells, KAPPA_1D, c, T_END_1D)[0]
        for c in c_cfl_values
    ]
    spread = (max(diffusivities) - min(diffusivities)) / diffusivities[-1]
    print(f"T1a: C_cfl {c_cfl_values} -> D/kappa "
          f"{[round(d / KAPPA_1D, 4) for d in diffusivities]}, spread {spread:.4f}")
    assert spread < tol, (
        f"D depends on the time step: relative spread {spread:.4f} >= {tol} over "
        f"C_cfl = {c_cfl_values} (D/kappa = {[d / KAPPA_1D for d in diffusivities]}). "
        f"Operator-split relaxation bias, DESIGN.md Problem 3."
    )


def test_cr_diffusion_rate_1d(
    resolutions: tuple = (256, 512, 1024),
    c_cfl: float = 0.2,
    tol: float = 0.02,
):
    """T1b: ``D -> diffusion_coefficient`` under refinement.

    Passes after fix step 1. Before it, D / kappa was 0.400 / 0.368 / 0.349 at
    N = 256 / 512 / 1024: ``1/3`` (the convention) times the time-step bias
    ``1 + nu dt / 2``, moving *away* from 1 under refinement. After step 1:
    1.080 / 1.033 / 1.015. About 1.3 points of the last value are the step-2
    bias (``nu dt = 0.026``), so step 2 should leave mostly the spatial
    error.

    Args:
        resolutions: Increasing resolutions.
        c_cfl: The CFL number.
        tol: Maximum ``|D/kappa - 1|`` at the finest resolution.
    """
    errors = []
    for num_cells in resolutions:
        d = _measure_diffusivity(1, num_cells, KAPPA_1D, c_cfl, T_END_1D)[0]
        errors.append(d / KAPPA_1D - 1.0)
    print(f"T1b: N {resolutions} -> D/kappa - 1 = {[round(e, 4) for e in errors]}")

    fig, (ax_dt, ax_n) = plt.subplots(1, 2, figsize=(11, 4.5))
    c_scan = (0.4, 0.2, 0.1)
    d_scan = [_measure_diffusivity(1, 512, KAPPA_1D, c, T_END_1D) for c in c_scan]
    dt_scan = [T_END_1D / it for _, _, it in d_scan]
    ax_dt.plot(dt_scan, [d / KAPPA_1D for d, _, _ in d_scan], "o-", color="C0")
    ax_dt.axhline(1.0, color="black", lw=1, label="target D = kappa")
    ax_dt.set_xlabel("dt (N = 512)")
    ax_dt.set_ylabel("D / kappa")
    ax_dt.set_title("T1a: time-step dependence")
    ax_dt.legend()
    ax_n.plot(resolutions, [1.0 + e for e in errors], "o-", color="C1")
    ax_n.axhline(1.0, color="black", lw=1, label="target D = kappa")
    ax_n.set_xscale("log", base=2)
    ax_n.set_xlabel("N")
    ax_n.set_ylabel("D / kappa")
    ax_n.set_title(f"T1b: refinement (C_cfl = {c_cfl})")
    ax_n.legend()
    _save(fig, "cr_diffusion_rate_1d_test.svg")

    assert abs(errors[-1]) < tol, (
        f"D/kappa = {1 + errors[-1]:.4f} at N = {resolutions[-1]}, expected 1 "
        f"within {tol} (diffusion_coefficient is the e_cr diffusivity, fix step 1)."
    )
    assert all(abs(b) < abs(a) for a, b in zip(errors, errors[1:])), (
        f"|D/kappa - 1| does not decrease under refinement: {errors}."
    )


def test_cr_diffusion_stiff_1d(
    num_cells: int = 512,
    c_cfl: float = 0.4,
    tol: float = 0.1,
    step_margin: float = 1.05,
):
    """T1c: stiff regime, ``nu dt ~ 3`` and optical depth 7.4 per cell.

    Passes after fix steps 1, 2 and 4.
    - The relaxation must no longer limit dt (step 2). This is checked as at
      most the hydro-CFL step count at ``|u| + c = v_red``.
    - HLL dissipation at ``v_red`` must not swamp ``D`` (step 4).
    - Before the fix: relaxation-limited, 114286 steps against 5120 from the
      hydro CFL (22x), and D / kappa = 0.68. The plan check measured the
      Riemann part of this regime at +27% with a per-stage implicit
      relaxation, and +4% with a reduced CR wave speed.

    Args:
        num_cells: The resolution.
        c_cfl: The CFL number.
        tol: Maximum ``|D/kappa - 1|``.
        step_margin: Allowed factor over the hydro-CFL step count.
    """
    d, _, num_iterations = _measure_diffusivity(
        1, num_cells, KAPPA_STIFF, c_cfl, T_END_STIFF
    )
    hydro_steps = T_END_STIFF * REDUCED_STREAMING_SPEED * num_cells / c_cfl
    print(f"T1c: D/kappa = {d / KAPPA_STIFF:.4f}, steps {num_iterations} "
          f"(hydro CFL {hydro_steps:.0f})")
    assert num_iterations <= step_margin * hydro_steps, (
        f"{num_iterations} steps vs. {hydro_steps:.0f} from the hydro CFL: the "
        f"F_cr relaxation still limits dt (fix step 2)."
    )
    assert abs(d / KAPPA_STIFF - 1.0) < tol, (
        f"D/kappa = {d / KAPPA_STIFF:.4f} in the stiff regime, expected 1 within "
        f"{tol} (numerical diffusion of the CR rows, fix step 4)."
    )


# -------------------------------------------------------------
# ================= T2: 2D MHD anisotropic ====================
# -------------------------------------------------------------


def test_cr_anisotropic_no_leak(
    num_cells: int = 96,
    c_cfl_values: tuple = (0.4, 0.1),
    tol: float = 0.01,
):
    """T2a: no time-step-dependent leak across B (grid-aligned B).

    Passes after fix step 3. The once-per-step projection lets the first gas
    half-step build ``F_perp`` undamped. Before the fix, ``D_perp / kappa``
    was 0.349 at ``nu dt = 0.4`` and 0.306 at 0.1, a leak of
    ``0.14 nu dt kappa``.

    Args:
        num_cells: The resolution.
        c_cfl_values: The CFL numbers compared.
        tol: Maximum ``|Delta D_perp| / kappa`` between them.
    """
    d_perp = [
        _measure_diffusivity(2, num_cells, KAPPA_2D, c, T_END_2D, 0.0)[1]
        for c in c_cfl_values
    ]
    change = abs(d_perp[0] - d_perp[1]) / KAPPA_2D
    print(f"T2a: C_cfl {c_cfl_values} -> D_perp/kappa "
          f"{[round(d / KAPPA_2D, 4) for d in d_perp]}")
    assert change < tol, (
        f"D_perp changes by {change:.4f} kappa between C_cfl = {c_cfl_values}: "
        f"per-step leak of the B-projection (fix step 3)."
    )


def _anisotropic_resolution_study():
    """``{theta: [(N, D_par/kappa, D_perp/kappa), ...]}`` at C_cfl = 0.2."""
    return {
        theta: [
            (n, *(d / KAPPA_2D for d in
                  _measure_diffusivity(2, n, KAPPA_2D, 0.2, T_END_2D, theta)[:2]))
            for n in RESOLUTIONS_2D
        ]
        for theta in ANGLES_DEG
    }


def test_cr_anisotropic_parallel_rate(tol: float = 0.1):
    """T2b: ``D_par -> diffusion_coefficient`` along B at 0, 30 and 45 deg.

    Passes after fix step 1; steps 2-4 lower the error further. Before step
    1, ``D_par / kappa`` at N = 64 / 128 / 256 was 0.89 / 0.49 / 0.38 at 0
    deg: ``1/3`` times the time-step bias, plus numerical diffusion that
    shrinks under refinement. After step 1: 1.51 / 1.14 / 1.04 at 0 deg, and
    1.046 (30 deg) and 1.046 (45 deg) at N = 256.

    Args:
        tol: Maximum ``|D_par/kappa - 1|`` at the finest resolution.
    """
    study = _anisotropic_resolution_study()
    _plot_anisotropic(study)
    for theta, rows in study.items():
        errors = [d_par - 1.0 for _, d_par, _ in rows]
        print(f"T2b: theta {theta:4.1f} -> D_par/kappa - 1 = {[round(e, 4) for e in errors]}")
    for theta, rows in study.items():
        errors = [d_par - 1.0 for _, d_par, _ in rows]
        assert abs(errors[-1]) < tol, (
            f"D_par/kappa = {1 + errors[-1]:.4f} at theta = {theta}, "
            f"N = {RESOLUTIONS_2D[-1]}, expected 1 within {tol}."
        )
        assert all(abs(b) < abs(a) for a, b in zip(errors, errors[1:])), (
            f"|D_par/kappa - 1| does not decrease under refinement at "
            f"theta = {theta}: {errors}."
        )


def test_cr_anisotropic_perpendicular_numerical(aligned_tol: float = 0.01):
    """T2c: numerical cross-field diffusion with ``kappa_perp = 0``.

    - Every angle: ``D_perp`` must decrease under refinement. This already
      holds today and guards against a regression.
    - Grid-aligned B: ``D_perp < aligned_tol * kappa`` at the finest
      resolution. Passes after fix steps 3 and 4: the face-normal optical
      depth ``tau_n`` is infinite across an aligned B, so the reduced CR wave
      speed removes the HLL dissipation there.
    - Oblique B keeps ``R ~ 1`` on both axes (DESIGN.md fix step 4). Its
      ``D_perp`` is reported and plotted but not gated on an absolute value.
    - Before the fix, ``D_perp / kappa`` at N = 64 / 128 / 256 was 0.63 /
      0.20 / 0.064 at 0 deg, and within 5% of that at 30 and 45 deg.

    Args:
        aligned_tol: Maximum ``D_perp/kappa`` for grid-aligned B.
    """
    study = _anisotropic_resolution_study()
    for theta, rows in study.items():
        d_perp = [d for _, _, d in rows]
        print(f"T2c: theta {theta:4.1f} -> D_perp/kappa = {[round(d, 4) for d in d_perp]}")
        assert all(b < a for a, b in zip(d_perp, d_perp[1:])), (
            f"D_perp does not decrease under refinement at theta = {theta}: {d_perp}."
        )
    d_aligned = study[0.0][-1][2]
    assert d_aligned < aligned_tol, (
        f"D_perp/kappa = {d_aligned:.4f} across grid-aligned B at "
        f"N = {RESOLUTIONS_2D[-1]} (kappa_perp = 0), expected < {aligned_tol}: "
        f"numerical diffusion of the CR rows across B (fix steps 3-4)."
    )


def test_cr_anisotropic_kappa_perp(
    num_cells: int = 128,
    theta_deg: float = 30.0,
    kappa_perp_ratio: float = 0.01,
    tol: float = 0.25,
):
    """T2d: ``perpendicular_diffusion_coefficient`` adds its own value to D_perp.

    Passes after fix step 3. The numerical part of ``D_perp`` is the same with
    and without ``kappa_perp``, so the difference isolates the physical part.
    Before the fix the parameter does not exist and the test fails on that.

    Args:
        num_cells: The resolution.
        theta_deg: The field angle.
        kappa_perp_ratio: ``kappa_perp / kappa_par``.
        tol: Maximum ``|Delta D_perp / kappa_perp - 1|``.
    """
    kappa_perp = kappa_perp_ratio * KAPPA_2D
    d_without = _measure_diffusivity(2, num_cells, KAPPA_2D, 0.2, T_END_2D, theta_deg)[1]
    d_with = _measure_diffusivity(
        2, num_cells, KAPPA_2D, 0.2, T_END_2D, theta_deg, kappa_perp
    )[1]
    ratio = (d_with - d_without) / kappa_perp
    print(f"T2d: Delta D_perp / kappa_perp = {ratio:.4f}")
    assert abs(ratio - 1.0) < tol, (
        f"kappa_perp = {kappa_perp} adds {ratio:.4f} kappa_perp to D_perp, "
        f"expected 1 within {tol}."
    )


def _plot_anisotropic(study):
    fig, (ax_par, ax_perp) = plt.subplots(1, 2, figsize=(11, 4.5))
    for i, (theta, rows) in enumerate(study.items()):
        n = [r[0] for r in rows]
        ax_par.plot(n, [r[1] for r in rows], "o-", color=f"C{i}", label=f"{theta:.0f} deg")
        ax_perp.plot(n, [r[2] for r in rows], "o-", color=f"C{i}", label=f"{theta:.0f} deg")
    ax_par.axhline(1.0, color="black", lw=1, label="target")
    ax_par.set_ylabel("D_par / kappa_par")
    ax_par.set_title("T2b: along B")
    ax_perp.set_yscale("log")
    ax_perp.set_ylabel("D_perp / kappa_par (kappa_perp = 0)")
    ax_perp.set_title("T2c: across B (numerical)")
    for ax in (ax_par, ax_perp):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("N")
        ax.legend()
    _save(fig, "cr_diffusion_rate_2d_test.svg")


if __name__ == "__main__":
    tests = (
        test_cr_diffusion_time_step_independence_1d,
        test_cr_diffusion_rate_1d,
        test_cr_diffusion_stiff_1d,
        test_cr_anisotropic_no_leak,
        test_cr_anisotropic_parallel_rate,
        test_cr_anisotropic_perpendicular_numerical,
        test_cr_anisotropic_kappa_perp,
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
