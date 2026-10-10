"""
CR-thermal contact in pressure balance: Phase A test plan A1.3
(``astronomix/_modules/_cosmic_rays_grey/astronomix_CR_phaseA_test_plan.md``, Sec. 1.4;
Thomas, Pfrommer & Pakmor 2021, Sec. 4.3; Gupta, Sharma & Mignone 2021, Sec. 5.1).

**Setup.** ``rho = 1`` everywhere, ``[P_th, P_cr] = [1/8, 7/8]`` in one region and ``[3/4, 1/4]``
in the other, so ``P_th + P_cr = 1``: a contact discontinuity in pressure balance, which the
exact two-fluid equations keep static (or advect unchanged). ``gamma = 5/3``, ``gamma_cr =
4/3``, float64, ``v_red = 0`` (``F_cr = 0``, tightly coupled), ``t = 1``.

- 1D: static contact (open boundaries) and an advected slab (``u0 = 1``, periodic, one
  crossing), N = 64-512. A diffusive variant (``v_red = 10``, ``kappa = 1/300``) is reported
  only: there the CRs diffuse out of the contact and the gas must respond, so the imbalance is
  physical.
- 3D: periodic slabs with oblique normal ``(1, 1, 1)`` (static, and advected with ``u0 = (1,
  0, 0)``), hydro and MHD with a uniform tangential field ``B = 0.5 (1, -1, 0) / sqrt(2)``
  (1D finite-volume MHD is not supported), and a static spherical CR bubble (``r = 0.2``). In
  MHD the balance is ``P_th + P_cr + B^2/2``.

**Expected:** pressure balance and ``u = u0`` to round-off, as for the "pdv" schemes in Gupta
et al. (2021, Fig. 3).

**Measured (2026-10-10), HLLC + minmod, total-pressure flux
(``cr_grey_transport.cr_total_pressure_flux``): balanced to round-off.**

- 1D: max abs(P_tot - 1) at most 5e-15 (static) and 1e-13 (advected), the same for ``u``.
- 3D hydro: oblique slab 2e-16 / 3e-14 (static / advected), bubble 3e-16 (N = 64).
- The diffusive variant (``v_red = 10``, reported only): ~2e-3 static, ~1.5e-2 advected, with
  max abs(u) ~1e-2. This is physical: the CRs diffuse out of the contact.

**Source-term coupling (HLL, and MHD with any solver): the balance is NOT kept.** Measured
2026-10-09 with the stage-wise CR coupling, HLL + minmod. The errors converge only at order
~0.5-0.75:

- 1D static: max abs(P_tot - 1) 2.9e-2 / 1.6e-2 / 9.1e-3 / 5.9e-3 (N = 64-512), max abs(u)
  1.3e-2 ... 3.4e-3. Early on it is larger: 0.16 and 0.055 at t = 0.01, before the waves leave
  through the open boundaries.
- 1D advected: 6.8e-3 / 4.3e-3 / 2.7e-3 / 2.0e-3, max abs(u - u0) 7.4e-3 ... 1.7e-3. HLLC
  without the total-pressure flux was no better (static 4.7e-2 ... 9.0e-3).
- 3D oblique slab (hydro, HLL), static: 4.0e-2 / 2.6e-2 / 1.7e-2 (N = 32 / 64 / 128), max
  abs(u) 5.6e-3 ... 4.6e-3; advected 1.4e-2 / 9.8e-3 / 6.1e-3, 1.6e-2 ... 1.1e-2. Bubble 4.4e-2
  / 2.2e-2 / 1.6e-2, max abs(u) 2.4e-2 ... 1.2e-2.
- 3D MHD (incl. B^2/2): static 4.6e-2 / 3.3e-2 / 2.1e-2, advected 1.4e-2 / 9.2e-3 / 6.7e-3
  (N = 32 / 64 / 128), the same as hydro.
  - The field is uniform at t = 0. It stays exact with no contact, or with B along the normal.
  - The spurious flows compress the tangential field: max abs(dB) up to 0.2 next to the static
    contact.

**Cause (diagnosed 2026-10-09):** with the source coupling, the gas energy and the CR energy are
smeared by different operators.
- The HLL flux diffuses the gas energy row, ``S_L S_R (E_R - E_L) / (S_R - S_L)``, and ``E``
  jumps at the contact.
- The CR row only moves with the mass flux (mass flux x upwind ``e_cr / rho``,
  ``hll._grey_cr_hll_rows``), which is zero here.
- So ``P_th`` smears over 2 cells within ``t = 1e-3`` while ``P_cr`` stays sharp, and their sum
  is no longer constant.
- The spurious velocity grows like ``t^2`` (3.2e-5 at ``t = 1e-4``, 3.2e-3 at ``1e-3``), i.e.
  from this growing imbalance, not from a force present at t = 0.
- In the advected case, the gas-energy work ``-v . grad(P_cr)`` also differentiates the jump in
  ``P_cr``. This is the "vdp" form that fails Gupta et al.'s Fig. 3.
- A pure-gas density contact at uniform pressure stays exact to 1e-15 with HLL and HLLC.

**Fix (2026-10-10):** HLLC carries the CR pressure (Gupta et al. 2021, "Eg+Ecr Unsplit-pdv";
Thomas, Pfrommer & Pakmor 2021, Sec. 4.3).
- The total pressure goes into the momentum flux, the gas energy flux and the HLLC star state.
- The adiabatic work uses the Riemann face velocities.
- HLLC then sees a contact, not a pressure jump, and neither energy is diffused across it.

The HLL and MHD runs below are tracked bounds: the errors must not exceed the measured values
(with a margin) and must decrease with N.

See ``PROGRESS_PHASEA.md`` (2026-10-09/10).
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
    HLL,
    HLLC,
    BoundarySettings,
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

jax.config.update("jax_enable_x64", True)

GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
P_TH_IN, P_CR_IN = 1.0 / 8.0, 7.0 / 8.0
P_TH_OUT, P_CR_OUT = 3.0 / 4.0, 1.0 / 4.0
PICS_DIR = Path(__file__).resolve().parent / "pics" / "01_advection"


def _run_1d(num_cells, advected, reduced_streaming_speed=0.0, riemann_solver=HLLC):
    boundary = PERIODIC_BOUNDARY if advected else OPEN_BOUNDARY
    relaxation = reduced_streaming_speed > 0.0
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=1.0,
        numerical_precision=DOUBLE_PRECISION,
        riemann_solver=riemann_solver,
        boundary_settings=BoundarySettings1D(boundary, boundary),
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True, diffusive_relaxation=relaxation),
    )
    rv = get_registered_variables(config)
    x = np.asarray(get_helper_data(config).geometric_centers)
    inside = ((x > 0.25) & (x < 0.75)) if advected else (x < 0.5)
    u0 = 1.0 if advected else 0.0
    state = jnp.zeros((rv.num_vars, num_cells)).at[rv.density_index].set(1.0).at[rv.velocity_index].set(u0)
    state = state.at[rv.pressure_index].set(np.where(inside, P_TH_IN, P_TH_OUT))
    state = state.at[rv.cosmic_ray_e_index].set(np.where(inside, P_CR_IN, P_CR_OUT) / (GAMMA_CR - 1.0))
    config = finalize_config(config, state.shape)
    cr_params = CosmicRayGreyParams(
        gamma_cr=GAMMA_CR, reduced_streaming_speed=reduced_streaming_speed,
        **({"diffusion_coefficient": 1.0 / 300.0} if relaxation else {}),
    )
    final = np.asarray(time_integration(state, config, SimulationParams(t_end=1.0, gamma=GAMMA, cosmic_ray_grey_params=cr_params), rv))
    assert not np.any(np.isnan(final)), "NaN in the 1D contact run."
    p_th = final[rv.pressure_index]
    p_cr = final[rv.cosmic_ray_e_index] * (GAMMA_CR - 1.0)
    return dict(
        x=x, p_th=p_th, p_cr=p_cr, u=final[rv.velocity_index],
        max_dp=float(np.abs(p_th + p_cr - 1.0).max()), max_du=float(np.abs(final[rv.velocity_index] - u0).max()),
    )


def _decreasing(values):
    return all(b < a for a, b in zip(values, values[1:]))


def test_cr_contact_pressure_balance_1d(
    resolutions=(64, 128, 256, 512),
    max_error_hllc=1e-10,
    max_dp_static_hll=1e-2,
    max_du_static_hll=5e-3,
    max_dp_advected_hll=4e-3,
    max_du_advected_hll=3e-3,
):
    """A1.3 in 1D: HLLC with the total-pressure flux, and HLL with the source coupling (tracked).

    Args:
        resolutions: Cell counts.
        max_error_hllc: Bound on max abs(P_tot - 1) and max abs(u - u0) with HLLC, every N, static
            and advected (measured <= 1.2e-13).
        max_dp_static_hll, max_du_static_hll: Bounds on max abs(P_tot - 1) and max abs(u) with
            HLL at the finest N, static contact (measured 5.9e-3, 3.4e-3).
        max_dp_advected_hll, max_du_advected_hll: The same for the advected slab (measured
            2.0e-3, 1.7e-3).
    """
    results = {}
    for solver, name in ((HLLC, "HLLC"), (HLL, "HLL ")):
        for advected in (False, True):
            for v_red in ((0.0, 10.0) if solver == HLLC else (0.0,)):
                runs = [_run_1d(n, advected, v_red, solver) for n in resolutions]
                results[solver, advected, v_red] = runs
                print(
                    f"{name} {'advected' if advected else 'static':8s} v_red = {v_red:4.1f}: max|P_tot - 1| "
                    f"{' '.join(f'{r['max_dp']:.1e}' for r in runs)}, max|u - u0| {' '.join(f'{r['max_du']:.1e}' for r in runs)}"
                )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (advected, title) in zip(axes[:2], ((False, "static contact"), (True, "advected slab, one crossing"))):
        r = results[HLLC, advected, 0.0][-1]
        r_hll = results[HLL, advected, 0.0][-1]
        u0 = 1.0 if advected else 0.0
        ax.plot(r["x"], r["p_th"], label=r"$P_{th}$ (HLLC)")
        ax.plot(r["x"], r["p_cr"], label=r"$P_{cr}$ (HLLC)")
        ax.plot(r["x"], r["p_th"] + r["p_cr"], "k-", label=r"$P_{th} + P_{cr}$ (HLLC)")
        ax.plot(r_hll["x"], r_hll["p_th"] + r_hll["p_cr"], "k--", lw=0.8, label=r"$P_{th} + P_{cr}$ (HLL, source coupling)")
        ax.plot(r["x"], r["u"] - u0 + 0.5, "C3:", label=r"$u - u_0 + 0.5$ (HLLC)")
        ax.set_xlabel("x")
        ax.set_title(f"{title}, N = {resolutions[-1]}, v_red = 0")
        ax.legend(fontsize=8)
    for (solver, advected, v_red), runs in results.items():
        axes[2].loglog(resolutions, [max(r["max_dp"], 1e-17) for r in runs], "o-" if v_red == 0 else "s:",
                       label=f"{'HLLC' if solver == HLLC else 'HLL'}, {'advected' if advected else 'static'}, v_red = {v_red:g}")
    axes[2].set_xlabel("N")
    axes[2].set_ylabel(r"max $|P_{th} + P_{cr} - 1|$")
    axes[2].set_title("Pressure-balance error (exact: 0)")
    axes[2].legend(fontsize=8)
    fig.suptitle("A1.3 (1D): CR-thermal contact in pressure balance -- HLLC total-pressure flux vs HLL source coupling")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_contact_pressure_balance_1d_test.svg")
    plt.close(fig)

    for advected in (False, True):
        for r in results[HLLC, advected, 0.0]:
            assert r["max_dp"] < max_error_hllc, f"HLLC advected={advected}: max|P_tot - 1| {r['max_dp']:.2e}."
            assert r["max_du"] < max_error_hllc, f"HLLC advected={advected}: max|u - u0| {r['max_du']:.2e}."
    for advected, max_dp, max_du in (
        (False, max_dp_static_hll, max_du_static_hll), (True, max_dp_advected_hll, max_du_advected_hll)
    ):
        runs = results[HLL, advected, 0.0]
        assert _decreasing([r["max_dp"] for r in runs]), f"HLL advected={advected}: P_tot error not decreasing with N."
        assert runs[-1]["max_dp"] < max_dp, f"HLL advected={advected}: max|P_tot - 1| {runs[-1]['max_dp']:.2e}."
        assert runs[-1]["max_du"] < max_du, f"HLL advected={advected}: max|u - u0| {runs[-1]['max_du']:.2e}."


def _run_3d(num_cells, geometry, advected=False, mhd=False, riemann_solver=HLLC):
    per = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=3,
        num_cells=num_cells,
        box_size=1.0,
        numerical_precision=DOUBLE_PRECISION,
        mhd=mhd,
        riemann_solver=riemann_solver,
        boundary_settings=BoundarySettings(x=per, y=per, z=per),
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True),
    )
    rv = get_registered_variables(config)
    centers = np.asarray(get_helper_data(config).geometric_centers)
    x, y, z = centers[..., 0], centers[..., 1], centers[..., 2]
    if geometry == "bubble":
        inside = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2 < 0.2**2
    else:
        phase = (x + y + z) % 1.0
        inside = (phase >= 0.25) & (phase < 0.75)
    v = rv.velocity_index
    u0 = 1.0 if advected else 0.0
    state = jnp.zeros((rv.num_vars,) + x.shape).at[rv.density_index].set(1.0).at[v.x].set(u0)
    state = state.at[rv.pressure_index].set(np.where(inside, P_TH_IN, P_TH_OUT))
    state = state.at[rv.cosmic_ray_e_index].set(np.where(inside, P_CR_IN, P_CR_OUT) / (GAMMA_CR - 1.0))
    b = np.array([0.5, -0.5, 0.0]) / np.sqrt(2.0) if mhd else np.zeros(3)
    if mhd:
        m = rv.magnetic_index
        state = state.at[m.x].set(b[0]).at[m.y].set(b[1]).at[m.z].set(b[2])
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=1.0, gamma=GAMMA, cosmic_ray_grey_params=CosmicRayGreyParams(gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0)
    )
    final = np.asarray(time_integration(state, config, params, rv))
    assert not np.any(np.isnan(final)), f"NaN in the 3D {geometry} run."
    p_tot = final[rv.pressure_index] + final[rv.cosmic_ray_e_index] * (GAMMA_CR - 1.0)
    p_ref = 1.0
    if mhd:
        m = rv.magnetic_index
        p_tot = p_tot + 0.5 * (final[m.x] ** 2 + final[m.y] ** 2 + final[m.z] ** 2)
        p_ref += 0.5 * float(b @ b)
    du = max(float(np.abs(final[v.x] - u0).max()), float(np.abs(final[v.y]).max()), float(np.abs(final[v.z]).max()))
    return dict(max_dp=float(np.abs(p_tot - p_ref).max()), max_du=du, slice=(p_tot - p_ref)[:, :, num_cells // 2])


def test_cr_contact_pressure_balance_3d(
    resolutions=(32, 64),
    cases=(
        ("slab", False, False, None),
        ("slab", True, False, None),
        ("bubble", False, False, None),
        ("slab", False, True, 4.5e-2),
        ("slab", True, True, 1.5e-2),
    ),
    max_error_hllc=1e-10,
    max_du_mhd=3e-2,
):
    """A1.3 in 3D: oblique slabs (static and advected) and a static CR bubble with HLLC and the
    total-pressure flux; oblique MHD slabs with HLL and the source coupling (tracked; MHD has no
    total-pressure flux yet).

    Args:
        resolutions: Cell counts (N = 128 adds ~4-6 min per case in float64).
        cases: ``(geometry, advected, mhd, bound)``. Hydro (``bound`` None): max abs(P_tot -
            P_ref) and max abs(u - u0) below ``max_error_hllc`` at every N (measured <= 3e-14).
            MHD: max abs(P_tot - P_ref) below ``bound`` at the finest N and decreasing with N
            (measured at N = 64: 3.3e-2 / 9.2e-3 static / advected).
        max_error_hllc: See ``cases``.
        max_du_mhd: Bound on the spurious velocity in MHD (measured <= 1.8e-2 at N = 64).
    """
    results = {}
    for geometry, advected, mhd, _ in cases:
        runs = [_run_3d(n, geometry, advected, mhd, HLL if mhd else HLLC) for n in resolutions]
        results[geometry, advected, mhd] = runs
        print(
            f"{geometry:6s} {'advected' if advected else 'static':8s} {'MHD, HLL ' if mhd else 'hydro, HLLC'}: "
            f"max|P_tot - P_ref| {' '.join(f'{r['max_dp']:.1e}' for r in runs)}, "
            f"max|u - u0| {' '.join(f'{r['max_du']:.1e}' for r in runs)}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, title in (
        (axes[0], ("slab", False, True), "oblique slab, static, MHD (HLL)"),
        (axes[1], ("bubble", False, False), "CR bubble, static, hydro (HLLC)"),
    ):
        image = ax.imshow(results[key][-1]["slice"].T, origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r")
        ax.set_title(f"{title}: P_tot - P_ref (z = 0.5), N = {resolutions[-1]}", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax)
    for (geometry, advected, mhd), runs in results.items():
        axes[2].loglog(resolutions, [max(r["max_dp"], 1e-17) for r in runs], "o-",
                       label=f"{geometry}, {'advected' if advected else 'static'}, {'MHD (HLL)' if mhd else 'hydro (HLLC)'}")
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("max |P_tot - P_ref|")
    axes[2].set_title("Pressure-balance error (exact: 0)")
    axes[2].legend(fontsize=8)
    fig.suptitle("A1.3 (3D): CR-thermal contacts in pressure balance -- hydro HLLC to round-off, MHD source coupling not")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_contact_pressure_balance_3d_test.svg")
    plt.close(fig)

    for geometry, advected, mhd, bound in cases:
        runs = results[geometry, advected, mhd]
        label = f"{geometry} advected={advected} mhd={mhd}"
        if bound is None:
            for r in runs:
                assert r["max_dp"] < max_error_hllc, f"{label}: max|P_tot - P_ref| {r['max_dp']:.2e}."
                assert r["max_du"] < max_error_hllc, f"{label}: max|u - u0| {r['max_du']:.2e}."
        else:
            assert _decreasing([r["max_dp"] for r in runs]), f"{label}: P_tot error not decreasing with N."
            assert runs[-1]["max_dp"] < bound, f"{label}: max|P_tot - P_ref| {runs[-1]['max_dp']:.2e}."
            assert runs[-1]["max_du"] < max_du_mhd, f"{label}: max|u - u0| {runs[-1]['max_du']:.2e}."


if __name__ == "__main__":
    test_cr_contact_pressure_balance_1d()
    test_cr_contact_pressure_balance_3d()
