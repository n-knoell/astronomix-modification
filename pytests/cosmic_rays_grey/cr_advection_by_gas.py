"""
Passive CR blob advected by the gas: Phase A test plan A1.1
(``astronomix/_modules/_cosmic_rays_grey/astronomix_CR_phaseA_test_plan.md``, Sec. 1.4).

Ladder item 1 (``cr_advection.py``) propagates a free CR wave through gas at rest, so the
``u_n e_cr`` / ``u_n F_cr`` advective fluxes are never tested on their own. Here the gas moves
uniformly (``rho = P = 1``, ``u0`` along x in 1D and along the box diagonal in 3D, periodic) and
carries a CR blob of amplitude ``1e-6`` on a ``1e-7`` background, so the CR pressure does not
disturb the gas (passive). The exact solution is the initial profile translated by ``u0 t``;
after one crossing (``t = 1``) it is the initial profile. float64. ``v_red = 0`` (``F_cr = 0``
exactly); the 1D test also runs a relaxation variant (``v_red = 1``, ``kappa = 1e-6``).

Measured (2026-10-09, stage-wise CR coupling, HLL + minmod, ``C_cfl = 0.4``):

- 1D Gaussian (sigma = 0.05): L1/amp 4.7e-2 / 1.9e-2 / 6.0e-3 / 2.1e-3 at N = 64-512, order 1.51
  (the minmod value for a smooth extremum); centroid within 0.012 dx; minimum 3e-12 of amp
  above 0; ``F_cr`` exactly 0; max abs(u - u0) 1.3e-7.
- 1D top-hat (width 0.2): L1/amp 8.7e-2 ... 2.2e-2, order 0.66 (the expected ~2/3 for a
  discontinuity); no new extrema (min -1e-11, max 1.0000 of amp).
- Relaxation variant: identical to 4 digits; ``F_cr`` stays below 4e-5 of amp.
- 3D (diagonal ``u0 = (1, 1, 1)``, Gaussian sigma = 0.1): order 1.56 / 1.55 (N = 32-64-128),
  round to 4e-5, centroid within 0.025 cells; see ``test_cr_advection_by_gas_3d``.

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
    BoundarySettings,
    BoundarySettings1D,
    DOUBLE_PRECISION,
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
AMPLITUDE = 1e-6
BACKGROUND = 1e-7
PICS_DIR = Path(__file__).resolve().parent / "pics" / "01_advection"


def _gaussian(sigma):
    return lambda r2: np.exp(-0.5 * r2 / sigma**2)


def _run_1d(num_cells, shape, relaxation=False):
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=1.0,
        numerical_precision=DOUBLE_PRECISION,
        boundary_settings=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True, diffusive_relaxation=relaxation),
    )
    rv = get_registered_variables(config)
    x = np.asarray(get_helper_data(config).geometric_centers)
    profile = shape(x)
    state = jnp.zeros((rv.num_vars, num_cells)).at[rv.density_index].set(1.0)
    state = state.at[rv.velocity_index].set(1.0).at[rv.pressure_index].set(1.0)
    state = state.at[rv.cosmic_ray_e_index].set(BACKGROUND + AMPLITUDE * profile)
    config = finalize_config(config, state.shape)
    cr_params = (
        CosmicRayGreyParams(gamma_cr=GAMMA_CR, reduced_streaming_speed=1.0, diffusion_coefficient=1e-6)
        if relaxation
        else CosmicRayGreyParams(gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0)
    )
    final = np.asarray(
        time_integration(state, config, SimulationParams(t_end=1.0, gamma=GAMMA, cosmic_ray_grey_params=cr_params), rv)
    )
    assert not np.any(np.isnan(final)), "NaN in the 1D advection run."
    blob = (final[rv.cosmic_ray_e_index] - BACKGROUND) / AMPLITUDE
    centroid = np.angle(np.sum(blob * np.exp(2j * np.pi * x))) / (2.0 * np.pi) % 1.0 - 0.5
    return dict(
        x=x,
        blob=blob,
        exact=profile,
        l1=float(np.mean(np.abs(blob - profile))),
        centroid_cells=float(centroid * num_cells),
        min=float(blob.min()),
        max=float(blob.max()),
        max_flux=float(np.abs(final[rv.cosmic_ray_flux_index]).max() / AMPLITUDE),
        max_du=float(np.abs(final[rv.velocity_index] - 1.0).max()),
    )


def _order(resolutions, errors):
    return -np.polyfit(np.log(resolutions), np.log(errors), 1)[0]


def test_cr_advection_by_gas_1d(
    resolutions=(64, 128, 256, 512),
    min_order_smooth=1.4,
    max_error_smooth=3e-3,
    min_order_tophat=0.55,
    max_centroid_cells=0.05,
    max_overshoot=1e-9,
    max_du=1e-6,
):
    """A1.1 in 1D: Gaussian and top-hat blob, one crossing at ``u0 = 1``.

    Args:
        resolutions: Cell counts.
        min_order_smooth: Minimum L1 order for the Gaussian (measured 1.51).
        max_error_smooth: Bound on the Gaussian L1/amp at the finest N (measured 2.1e-3).
        min_order_tophat: Minimum L1 order for the top-hat (measured 0.66).
        max_centroid_cells: Bound on the centroid offset in cells (measured <= 0.012).
        max_overshoot: Bound on new extrema, as a fraction of the amplitude (measured 1e-11).
        max_du: Bound on the gas velocity perturbation (passive CRs; measured 1.3e-7).
    """
    shapes = {
        "Gaussian": lambda x: _gaussian(0.05)((x - 0.5) ** 2),
        "top-hat": lambda x: ((x > 0.4) & (x < 0.6)).astype(float),
    }
    results = {}
    for relaxation in (False, True):
        for name, shape in shapes.items():
            runs = [_run_1d(n, shape, relaxation) for n in resolutions]
            errors = [r["l1"] for r in runs]
            results[name, relaxation] = dict(runs=runs, errors=errors, order=_order(resolutions, errors))
            print(
                f"{name:8s} relaxation={relaxation!s:5}: L1/amp {' '.join(f'{e:.2e}' for e in errors)}, order "
                f"{results[name, relaxation]['order']:.2f}, centroid [cells] "
                f"{' '.join(f'{r['centroid_cells']:+.1e}' for r in runs)}, min/max {runs[-1]['min']:+.1e}/{runs[-1]['max']:.4f}, "
                f"max|F_cr|/amp {runs[-1]['max_flux']:.1e}, max|u - u0| {runs[-1]['max_du']:.1e}"
            )

    fig, (ax_profile, ax_conv) = plt.subplots(1, 2, figsize=(13, 5))
    for name in shapes:
        r = results[name, False]["runs"][-1]
        ax_profile.plot(r["x"], r["exact"], "k-", lw=1)
        ax_profile.plot(r["x"], r["blob"], "--", label=f"{name}, N = {resolutions[-1]}")
        ax_conv.loglog(resolutions, results[name, False]["errors"], "o-",
                       label=f"{name} (order {results[name, False]['order']:.2f})")
    ax_profile.set_xlabel("x")
    ax_profile.set_ylabel(r"$(e_{cr} - e_{bg}) / A$")
    ax_profile.set_title("After one crossing (black: exact)")
    ax_profile.legend()
    ax_conv.set_xlabel("N")
    ax_conv.set_ylabel("L1 / A")
    ax_conv.set_title("Convergence")
    ax_conv.legend()
    fig.suptitle("A1.1 (1D): passive CR blob advected by the gas, u0 = 1, v_red = 0")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_advection_by_gas_1d_test.svg")
    plt.close(fig)

    for (name, relaxation), r in results.items():
        last = r["runs"][-1]
        assert all(abs(run["centroid_cells"]) < max_centroid_cells for run in r["runs"]), f"{name}: centroid offset."
        assert last["min"] > -max_overshoot and last["max"] < 1.0 + max_overshoot, f"{name}: new extrema."
        assert last["max_du"] < max_du, f"{name}: the CRs are not passive (max|u - u0| {last['max_du']:.2e})."
        if not relaxation:
            assert last["max_flux"] == 0.0, f"{name}: F_cr != 0 with v_red = 0."
    assert results["Gaussian", False]["order"] >= min_order_smooth
    assert results["Gaussian", False]["errors"][-1] < max_error_smooth
    assert results["top-hat", False]["order"] >= min_order_tophat


def _run_3d(num_cells, shape):
    per = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=3,
        num_cells=num_cells,
        box_size=1.0,
        numerical_precision=DOUBLE_PRECISION,
        boundary_settings=BoundarySettings(x=per, y=per, z=per),
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True),
    )
    rv = get_registered_variables(config)
    centers = np.asarray(get_helper_data(config).geometric_centers)
    x, y, z = centers[..., 0], centers[..., 1], centers[..., 2]
    profile = shape((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2)
    v = rv.velocity_index
    state = jnp.zeros((rv.num_vars,) + x.shape).at[rv.density_index].set(1.0).at[rv.pressure_index].set(1.0)
    state = state.at[v.x].set(1.0).at[v.y].set(1.0).at[v.z].set(1.0)
    state = state.at[rv.cosmic_ray_e_index].set(BACKGROUND + AMPLITUDE * profile)
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=1.0, gamma=GAMMA, cosmic_ray_grey_params=CosmicRayGreyParams(gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0)
    )
    final = np.asarray(time_integration(state, config, params, rv))
    assert not np.any(np.isnan(final)), "NaN in the 3D advection run."
    blob = (final[rv.cosmic_ray_e_index] - BACKGROUND) / AMPLITUDE
    centroid = [
        float((np.angle(np.sum(blob * np.exp(2j * np.pi * q))) / (2.0 * np.pi) % 1.0 - 0.5) * num_cells) for q in (x, y, z)
    ]
    # second moments along and across the advection direction (1, 1, 1) / sqrt(3)
    dx, dy, dz = (((q - 0.5 + 0.5) % 1.0) - 0.5 for q in (x, y, z))
    weight = np.clip(blob, 0.0, None)
    along = np.sum(weight * ((dx + dy + dz) / np.sqrt(3.0)) ** 2) / weight.sum()
    total = np.sum(weight * (dx**2 + dy**2 + dz**2)) / weight.sum()
    du = max(float(np.abs(final[c] - 1.0).max()) for c in (v.x, v.y, v.z))
    return dict(
        blob=blob, exact=profile, l1=float(np.mean(np.abs(blob - profile))), centroid=centroid,
        roundness=float(along / ((total - along) / 2.0)), min=float(blob.min()), max=float(blob.max()), max_du=du,
    )


def test_cr_advection_by_gas_3d(
    resolutions=(32, 64),
    min_order=1.3,
    max_centroid_cells=0.1,
    max_roundness_error=1e-3,
    max_overshoot=1e-7,
    max_du=1e-6,
):
    """A1.1 in 3D: a Gaussian (sigma = 0.1) and a top-hat sphere (r = 0.2) advected along the box
    diagonal, ``u0 = (1, 1, 1)``, for one crossing (``t = 1``), periodic.

    Measured (2026-10-09): Gaussian L1/amp 6.8e-3 / 2.3e-3 / 7.9e-4 at N = 32 / 64 / 128 (order
    1.56 for 32-64, 1.55 for 64-128); second-moment ratio along / across the diagonal 1.00004 /
    1.00000 / 1.00000 (round); identical moments along x, y and z; centroid within 0.025 cells;
    minimum >= -3e-9 of the amplitude. Top-hat sphere at N = 64: L1/amp 2.0e-2, min -5.8e-9, max 0.98 (no new extrema). The
    default runs N = 32 and 64 (~2 min on a GPU); add 128 for the third point (~6 min, 128^3 float64).

    Args:
        resolutions: Cell counts for the Gaussian.
        min_order: Minimum L1 order of the Gaussian.
        max_centroid_cells: Bound on the centroid offset per axis, in cells.
        max_roundness_error: Bound on ``abs(along / across - 1)`` of the second moments.
        max_overshoot: Bound on new extrema, as a fraction of the amplitude (top-hat).
        max_du: Bound on the gas velocity perturbation.
    """
    gaussian = [_run_3d(n, _gaussian(0.1)) for n in resolutions]
    tophat = _run_3d(64, lambda r2: (r2 < 0.2**2).astype(float))
    errors = [r["l1"] for r in gaussian]
    order = _order(resolutions, errors)
    for n, r in zip(resolutions, gaussian):
        print(f"Gaussian N = {n}: L1/amp {r['l1']:.2e}, centroid [cells] {[round(c, 3) for c in r['centroid']]}, "
              f"along/across {r['roundness']:.6f}, min {r['min']:.1e}, max|u - u0| {r['max_du']:.1e}")
    print(f"Gaussian order {order:.2f}; top-hat N = 64: L1/amp {tophat['l1']:.2e}, min {tophat['min']:+.1e}, max {tophat['max']:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    n = resolutions[-1]
    mid = n // 2
    for ax, data, title in ((axes[0], gaussian[-1]["blob"], f"Gaussian, N = {n}"), (axes[1], tophat["blob"], "top-hat sphere, N = 64")):
        image = ax.imshow(data[:, :, data.shape[2] // 2].T, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
        ax.set_title(f"{title}: z = 0.5 slice after one crossing")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax)
    diag = np.arange(n)
    axes[2].plot(diag, gaussian[-1]["exact"][diag, diag, diag], "k-", label="exact")
    axes[2].plot(diag, gaussian[-1]["blob"][diag, diag, diag], "C0--", label=f"N = {n}")
    axes[2].set_xlabel("cell index along the main diagonal")
    axes[2].set_ylabel(r"$(e_{cr} - e_{bg}) / A$")
    axes[2].set_title(f"Gaussian along (1, 1, 1); order {order:.2f}")
    axes[2].legend()
    fig.suptitle("A1.1 (3D): passive CR blob advected along the box diagonal")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_advection_by_gas_3d_test.svg")
    plt.close(fig)

    assert order >= min_order, f"3D Gaussian order {order:.2f} < {min_order}."
    for r in gaussian + [tophat]:
        assert all(abs(c) < max_centroid_cells for c in r["centroid"]), f"centroid {r['centroid']}."
        assert abs(r["roundness"] - 1.0) < max_roundness_error, f"roundness {r['roundness']:.6f}."
        assert r["max_du"] < max_du, f"the CRs are not passive (max|u - u0| {r['max_du']:.2e})."
    assert tophat["min"] > -max_overshoot and tophat["max"] < 1.0 + max_overshoot, "top-hat: new extrema."


if __name__ == "__main__":
    test_cr_advection_by_gas_1d()
    test_cr_advection_by_gas_3d()
