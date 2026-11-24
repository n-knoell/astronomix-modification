"""
Obliquity-dependent DSA efficiency at planar MHD shocks (GAMMA_RAY_PLAN.md "Optional --
obliquity"): does ``dsa_obliquity_dependence`` turn ``zeta(M)`` into ``zeta(M) g(theta_Bn)``
with the right angle?

``g(theta) = [tanh((theta_crit - theta) / delta) + 1] / 2``, theta_crit = pi/4, delta = pi/18
(Pais et al. 2018, MNRAS 478, 5278, Eq. 7). ``theta_Bn = arccos(|n_s . b|)`` is the angle between
the finder's shock normal and the *upstream* field. Downstream, the shock compresses B_t by the
density ratio r, so the field there is refracted towards the shock plane
(``tan theta_2 = r tan theta_1``: 45 deg upstream reads 76 deg downstream, g = 0.5 vs 0.001);
reading the wrong side would switch off injection at all but nearly parallel shocks.

**Setup.** 3D periodic unit box, N^3 (``CR_NUM_CELLS``, default 64), FV MHD, HLLC + MINMOD,
grey CRs with ``v_red = 0``, constant ``zeta = 0.1``, obliquity switch on. A planar slab of the
gas-only Rankine-Hugoniot post-shock state of Mach ``MACH`` (as ``cr_oblique_planar_shocks.py``:
phase ``frac(k . x)`` in (0.05, 0.45), 1-cell tanh edges) for normals along the grid (1,0,0) and
at 26.6 deg to it (2,1,0). Uniform upstream field of plasma beta ``BETA`` (weak: the shock stays
a gas shock) at angle theta_true to the normal, in the plane of the normal and the tangent
``t``; in the slab B_t is compressed by the local density ratio (frozen-in, div B = 0 since B_n
is uniform).

**Measured** on the snapshots at t >= 0.4 t_end, on shock-surface cells at least ``MARGIN``
cells from the faces and in a window that follows the front (``WINDOW_BEHIND`` cells behind it
for the post-shock deposit, ``WINDOW_AHEAD`` ahead; it keeps out the shock that the slab's back
edge sends round the periodic box, which reaches the front region by t_end at N = 32):

* the obliquity the injection uses (``shock_obliquity``: B at the walk's pre-shock sample,
  projected on the sigma = 2 smoothed-temperature normal), flux-weighted median, minus
  theta_true; for the figure also with the finder's own normal (off by 4-5 deg at (2,1,0),
  N = 32) and with B read at the surface cell itself (what a downstream read would give);
* the injected energy with the switch on over that with it off, one probe call of
  ``inject_crs_at_shocks`` each on the same state (spread and positivity cap included), against
  g(theta_true). Gated with the gas-only Hugoniot in both probes (``ratio``), where on / off is the
  obliquity factor alone; with the default CR-aware Hugoniot (``ratio_full``, reported) on / off
  is ~2% higher at 45 deg, because the off probe reads the less CR-loaded shock's velocity jump
  with the full zeta and so a slightly lower Mach number (N = 32: 0.509 vs 0.500);
* AD vs central FD of the injected energy with respect to theta_crit (theta_true = 45 deg).

Gates (N = 64): |median theta - theta_true| <= THETA_TOL; |ratio - g(theta_true)| <= RATIO_TOL_FAR
for |theta_true - 45 deg| >= 15 deg and <= RATIO_TOL_NEAR inside (there dg/dtheta = 0.05 per
degree, so the finder's angle error shows up directly); AD/FD <= 1e-2.

Raw values ``/export/scratch/nknoell/gamma_obliquity/obliquity_N<N>.npz``; figure
``pics/gamma_obliquity/dsa_obliquity_mhd.png``.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import dataclasses
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    PERIODIC_BOUNDARY,
    SnapshotSettings,
)
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_shock_finder_kwargs,
    inject_crs_at_shocks,
    obliquity_efficiency_factor,
    shock_obliquity,
)

NUM_CELLS = int(os.environ.get("CR_NUM_CELLS", 64))
GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
MACH = 10.0
BETA = 100.0
ZETA = 0.1
DSA_MACH_MIN = 1.3
THETA_CRIT, DELTA = np.pi / 4, np.pi / 18
ORIENTATIONS = {(1, 0, 0): (0.0, 1.0, 0.0), (2, 1, 0): (-1.0, 2.0, 0.0)}  # normal: tangent
THETAS_DEG = (0.0, 15.0, 30.0, 40.0, 45.0, 50.0, 60.0, 75.0, 90.0)
THETA_BACK, THETA_FRONT, TRAVEL = 0.05, 0.45, 0.35
SNAPSHOT_FRACTIONS = np.array([0.4, 0.6, 0.8, 1.0])
MARGIN = NUM_CELLS // 8  # the front ends at phase 0.8 < 1 - 1/8
WINDOW_BEHIND, WINDOW_AHEAD = 10.0, 4.0  # cells along the normal
THETA_TOL = 1.0  # degrees (N = 64: max 0.53)
RATIO_TOL_FAR, RATIO_TOL_NEAR = 0.01, 0.04  # N = 64: max 0.0032, 0.016
AD_FD_TOL = 1e-2
RAW_DIR = Path(os.environ.get("CR_GAMMA_RAW", "/export/scratch/nknoell/gamma_obliquity"))
PICS_DIR = Path(__file__).resolve().parent / "pics" / "gamma_obliquity"


def g_of_theta(theta):
    return 0.5 * (np.tanh((THETA_CRIT - theta) / DELTA) + 1.0)


def _rh_post(mach):
    p1 = 1.0 / GAMMA
    x = (GAMMA + 1.0) * mach**2 / ((GAMMA - 1.0) * mach**2 + 2.0)
    p2 = p1 * (2.0 * GAMMA * mach**2 - (GAMMA - 1.0)) / (GAMMA + 1.0)
    return x, mach * (1.0 - 1.0 / x), p2


def _setup(k, tangent, theta_true):
    kvec = np.asarray(k, float)
    kabs = float(np.linalg.norm(kvec))
    khat = kvec / kabs
    that = np.asarray(tangent, float) / np.linalg.norm(tangent)
    per = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    t_end = TRAVEL / (kabs * MACH)
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        box_size=1.0,
        mhd=True,
        boundary_settings=BoundarySettings(x=per, y=per, z=per),
        exact_end_time=True,
        return_snapshots=True,
        num_snapshots=len(SNAPSHOT_FRACTIONS),
        use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_shock_acceleration=True,
            dsa_obliquity_dependence=True,
        ),
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    centres = np.moveaxis(np.asarray(helper_data.geometric_centers), -1, 0)
    phase = np.mod(np.tensordot(kvec, centres, axes=1), 1.0)
    width = kabs / NUM_CELLS
    w = (0.25 * (1.0 + np.tanh((phase - THETA_BACK) / width))
         * (1.0 - np.tanh((phase - THETA_FRONT) / width)))
    rho2, u2, p2 = _rh_post(MACH)
    rho = 1.0 + w * (rho2 - 1.0)
    u = w * u2
    b0 = np.sqrt(2.0 * (1.0 / GAMMA) / BETA)  # p_B = B^2 / 2
    b_n = b0 * np.cos(theta_true) * np.ones_like(rho)
    b_t = b0 * np.sin(theta_true) * rho  # frozen-in, rho_1 = 1
    b = b_n[None] * khat[:, None, None, None] + b_t[None] * that[:, None, None, None]
    state = construct_primitive_state(
        config=config,
        registered_variables=rv,
        density=jnp.asarray(rho),
        velocity_x=jnp.asarray(u * khat[0]),
        velocity_y=jnp.asarray(u * khat[1]),
        velocity_z=jnp.asarray(u * khat[2]),
        magnetic_field_x=jnp.asarray(b[0]),
        magnetic_field_y=jnp.asarray(b[1]),
        magnetic_field_z=jnp.asarray(b[2]),
        gas_pressure=jnp.asarray(1.0 / GAMMA + w * (p2 - 1.0 / GAMMA)),
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=t_end,
        gamma=GAMMA,
        snapshot_timepoints=jnp.array(SNAPSHOT_FRACTIONS * t_end),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0,
            dsa_efficiency=ZETA, dsa_mach_min=DSA_MACH_MIN,
        ),
    )
    return state, config, params, rv, helper_data, phase, t_end


def _injected(state, config, params, rv, helper_data, mask, dt):
    out = inject_crs_at_shocks(state, config, params, rv, helper_data, 1.0, dt)
    return jnp.sum(jnp.where(mask, out[rv.cosmic_ray_e_index] - state[rv.cosmic_ray_e_index], 0.0))


def _measure(k, tangent, theta_deg, ad_check=False):
    theta_true = np.deg2rad(theta_deg)
    state0, config, params, rv, helper_data, phase, t_end = _setup(k, tangent, theta_true)
    snaps = time_integration(state0, config, params, rv)
    states = np.asarray(snaps.states)
    assert np.all(np.isfinite(states)), f"non-finite state, k = {k}, theta = {theta_deg}"
    def with_cr(**kw):
        return config._replace(cosmic_ray_grey_config=config.cosmic_ray_grey_config._replace(**kw))

    # gate probes: gas-only Hugoniot in both, so on / off is exactly the obliquity factor;
    # the default (CR-aware) pair is reported as ratio_full
    probes = dict(
        ratio=(with_cr(dsa_mach_cr_aware=False),
               with_cr(dsa_mach_cr_aware=False, dsa_obliquity_dependence=False)),
        ratio_full=(config, with_cr(dsa_obliquity_dependence=False)),
    )
    n = NUM_CELLS
    interior = np.zeros((n, n, n), bool)
    interior[MARGIN:n - MARGIN, MARGIN:n - MARGIN, MARGIN:n - MARGIN] = True
    kabs = float(np.linalg.norm(k))
    dt = 0.1 / (n * MACH)
    cr = params.cosmic_ray_grey_params
    out = dict(theta_up=[], theta_surf=[], theta_finder=[], ratio=[], ratio_full=[], weight_sum=[])
    masks = []
    for s, t in zip(states, np.asarray(snaps.time_points)):
        s = jnp.asarray(s)
        # window that follows the front (shock speed M c_1 = M along the normal), in cells
        # along the normal: WINDOW_BEHIND behind (the post-shock deposit), WINDOW_AHEAD ahead;
        # keeps out the shock that the slab's back edge sends round the periodic box
        cells_ahead = (np.mod(phase - (THETA_FRONT + TRAVEL * t / t_end) + 0.5, 1.0) - 0.5) * n / kabs
        mask = jnp.asarray(interior & (cells_ahead > -WINDOW_BEHIND) & (cells_ahead < WINDOW_AHEAD))
        masks.append(mask)
        sf = find_shocks_pfrommer(
            s, config, rv, helper_data, mach_min=cr.dsa_mach_min, gamma_gas=GAMMA,
            **dsa_shock_finder_kwargs(config.cosmic_ray_grey_config),
        )
        sigma = config.cosmic_ray_grey_config.dsa_area_normal_smoothing
        theta_up, _ = shock_obliquity(s, sf, rv, normal_smoothing=sigma)
        theta_finder, _ = shock_obliquity(s, sf, rv, normal_smoothing=None)
        surf = dataclasses.replace(sf, pre_shock_steps=jnp.zeros_like(sf.pre_shock_steps))
        theta_surf, _ = shock_obliquity(s, surf, rv, normal_smoothing=sigma)
        weight = np.asarray(jnp.where(mask, sf.thermal_energy_flux, 0.0))
        sel = weight > 0
        assert sel.any(), f"no shock cells in the measuring region, k = {k}, theta = {theta_deg}"
        out["theta_up"].append(_wmedian(np.asarray(theta_up)[sel], weight[sel]))
        out["theta_surf"].append(_wmedian(np.asarray(theta_surf)[sel], weight[sel]))
        out["theta_finder"].append(_wmedian(np.asarray(theta_finder)[sel], weight[sel]))
        out["weight_sum"].append(weight.sum())
        for key, (c_on, c_off) in probes.items():
            e_on = float(_injected(s, c_on, params, rv, helper_data, mask, dt))
            e_off = float(_injected(s, c_off, params, rv, helper_data, mask, dt))
            out[key].append(e_on / e_off)
    res = {key: np.asarray(v) for key, v in out.items()}
    if ad_check:
        s, mask = jnp.asarray(states[-1]), masks[-1]

        def e_of(theta_crit):
            p = params._replace(cosmic_ray_grey_params=cr._replace(dsa_obliquity_theta_crit=theta_crit))
            return _injected(s, config, p, rv, helper_data, mask, dt)

        ad = float(jax.grad(e_of)(jnp.float32(THETA_CRIT)))
        h = np.deg2rad(0.5)
        fd = float((e_of(THETA_CRIT + h) - e_of(THETA_CRIT - h)) / (2 * h))
        res["ad"], res["fd"] = ad, fd
    return res


def _wmedian(x, w):
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    return float(x[order][np.searchsorted(cw, 0.5 * cw[-1])])


def _plot(results, path):
    th = np.linspace(0, np.pi / 2, 400)
    r = _rh_post(MACH)[0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(np.rad2deg(th), g_of_theta(th), "k-", lw=1.2, label=r"$g(\theta)$ (Pais+18 Eq. 7)")
    ax[0].plot(np.rad2deg(th), g_of_theta(np.arctan(r * np.tan(th))), "k:", lw=1.2,
               label=r"$g$ of the downstream angle (B read behind the shock)")
    markers = {(1, 0, 0): "o", (2, 1, 0): "s"}
    for (k, theta_deg), res in sorted(results.items()):
        lab = f"normal {k}" if theta_deg == THETAS_DEG[0] else None
        ax[0].plot(theta_deg, np.mean(res["ratio"]), markers[k], mfc="none", color="C0" if k == (1, 0, 0) else "C3", label=lab)
        ax[1].plot(theta_deg, np.rad2deg(np.median(res["theta_up"])) - theta_deg, markers[k],
                   color="C0" if k == (1, 0, 0) else "C3", mfc="none", label=f"upstream sample, {k}" if lab else None)
        ax[1].plot(theta_deg, np.rad2deg(np.median(res["theta_finder"])) - theta_deg, markers[k],
                   color="C1", mfc="none", label=f"finder normal, {k}" if lab else None)
        ax[1].plot(theta_deg, np.rad2deg(np.median(res["theta_surf"])) - theta_deg, markers[k],
                   color="0.6", label=f"B at surface cell, {k}" if lab else None)
    ax[0].set_xlabel(r"$\theta_{Bn}$ (true, upstream) [deg]")
    ax[0].set_ylabel("injected energy, obliquity on / off")
    ax[0].legend(fontsize=8)
    ax[1].axhspan(-THETA_TOL, THETA_TOL, color="C2", alpha=0.15, lw=0)
    ax[1].set_xlabel(r"$\theta_{Bn}$ (true) [deg]")
    ax[1].set_ylabel(r"measured $-$ true obliquity [deg]")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"DSA obliquity, planar MHD shocks: M = {MACH:g}, beta = {BETA:g}, N = {NUM_CELLS}$^3$, "
                 f"t >= 0.4 t_end", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def test_dsa_obliquity_mhd():
    results = {}
    ad_fd = None
    for k, tangent in ORIENTATIONS.items():
        for theta_deg in THETAS_DEG:
            ad = k == (1, 0, 0) and theta_deg == 45.0
            res = _measure(k, tangent, theta_deg, ad_check=ad)
            results[(k, theta_deg)] = res
            if ad:
                ad_fd = (res["ad"], res["fd"])
            print(f"k={k} theta={theta_deg:5.1f}  ratio={np.mean(res['ratio']):.4f} "
                  f"(g={g_of_theta(np.deg2rad(theta_deg)):.4f}, CR-aware {np.mean(res['ratio_full']):.4f})  "
                  f"theta_up-true={np.rad2deg(np.median(res['theta_up'])) - theta_deg:+.2f} deg  "
                  f"theta_surf-true={np.rad2deg(np.median(res['theta_surf'])) - theta_deg:+.2f} deg  "
                  f"finder-normal {np.rad2deg(np.median(res['theta_finder'])) - theta_deg:+.2f} deg",
                  flush=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    flat = {}
    for (k, theta_deg), res in results.items():
        tag = f"k{''.join(map(str, k))}_t{int(theta_deg)}"
        for key, v in res.items():
            flat[f"{tag}_{key}"] = v
    np.savez(RAW_DIR / f"obliquity_N{NUM_CELLS}.npz", **flat)
    _plot(results, PICS_DIR / "dsa_obliquity_mhd.png")

    failures = []
    for (k, theta_deg), res in results.items():
        d_theta = abs(np.rad2deg(np.median(res["theta_up"])) - theta_deg)
        if d_theta > THETA_TOL:
            failures.append(f"{k} {theta_deg}: theta error {d_theta:.2f} deg")
        tol = RATIO_TOL_FAR if abs(theta_deg - 45.0) >= 15.0 else RATIO_TOL_NEAR
        d_ratio = abs(np.mean(res["ratio"]) - g_of_theta(np.deg2rad(theta_deg)))
        if d_ratio > tol:
            failures.append(f"{k} {theta_deg}: ratio off g by {d_ratio:.4f} (tol {tol})")
    ad, fd = ad_fd
    print(f"AD {ad:.6e}  FD {fd:.6e}  rel {abs(ad - fd) / abs(fd):.2e}")
    if abs(ad - fd) > AD_FD_TOL * abs(fd):
        failures.append(f"AD/FD {abs(ad - fd) / abs(fd):.2e}")
    assert not failures, "; ".join(failures)


if __name__ == "__main__":
    test_dsa_obliquity_mhd()
