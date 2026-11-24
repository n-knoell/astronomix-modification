"""
Obliquity-dependent DSA in an MHD Sedov blast in a uniform field (GAMMA_RAY_PLAN.md "Optional --
obliquity", the SNR check after the planar test ``cr_dsa_obliquity_mhd.py``).

The shock of a blast in a uniform field B = B_0 z has a radial normal, so its obliquity at polar
angle psi (from the field axis) is theta_Bn = psi folded to [0, pi/2]. Injection with
``dsa_obliquity_dependence`` is then zeta g(psi): polar caps (quasi-parallel) inject, the equatorial
belt (quasi-perpendicular) does not. Over the whole shell the injected energy is the sphere
average <g> = int_0^{pi/2} g(theta) sin(theta) d theta = 0.3017 of that without the switch (Pais
et al. 2018, Eq. 20: 0.302 zeta_0).

**Setup** (as ``cr_mhd_energy_budget.py``): 3D periodic unit box, N^3 (``CR_NUM_CELLS``, default 128),
FV MHD, HLLC + MINMOD, rho = 1, p = 1e-4, E = 1 deposited as pressure in r < 0.05 (2-cell tanh
edge), B_0 for plasma beta ``BETA`` = 100 (the field does not change the dynamics), grey CRs with
``v_red = 0``, constant zeta = 0.1 (small, so the blast stays close to spherical; Pais et al. use
0.5 to amplify the dynamical effect). The shock has M ~ 200 and radius 0.26 -> 0.38 (33 -> 49
cells at 128^3) over the measured snapshots. Two runs: switch on and off.

**Measured** on the snapshots of the switch-on run (probe calls of ``inject_crs_at_shocks`` with the
switch on and off on the same state, gas-only Hugoniot in both, as the planar test):

1. global on / off ratio against <g> = 0.3017 (gate ``GLOBAL_TOL``);
2. on / off per polar-angle bin (5 deg) against the sin-weighted bin mean of g (gate ``BIN_TOL``);
   and the median obliquity the injection reads minus the cell's polar angle per bin (diagnostic);
3. final-state maps of the gamma-ray proxy ``int rho e_cr dl`` (global alpha: every band is this
   map times a constant), line of sight perpendicular and parallel to B, both runs; the injected
   and final E_cr with the switch on over off; shell radius along and across B (diagnostic).

Results 2026-11-24 (128^3 PASSES; 64^3 smoke in brackets): global on/off 0.3033 [0.3010] vs 0.3017;
max bin deviation 0.034 [0.070]; E_cr(t_end) on/off 0.312 [0.308]. The obliquity read is
pulled towards 45 deg by up to 1.9 deg [3.1 deg] (normals on the shell tilt away from the grid
axes; first order in N, independent of the shell radius at fixed N), which smears the bin profile
of g slightly; it cancels in the global ratio. The shell radius along and across B is the same
in both runs at zeta = 0.1 (Pais et al. find eccentricity 0.25 only with zeta_0 = 0.5).

Raw values ``/export/scratch/nknoell/gamma_obliquity/sedov_N<N>.npz``; figure
``pics/gamma_obliquity/dsa_obliquity_sedov_mhd.png``.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import os
from pathlib import Path

import numpy as np
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
    shock_obliquity,
)

NUM_CELLS = int(os.environ.get("CR_NUM_CELLS", 128))
GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
E_EXPLOSION, RHO_AMBIENT, P_AMBIENT = 1.0, 1.0, 1e-4
R_EXPLOSION, SMOOTH_CELLS = 0.05, 2.0
BETA = 100.0
ZETA = 0.1
DSA_MACH_MIN = 1.3
T_END = 0.0627  # Sedov radius ~0.38
SNAPSHOT_FRACTIONS = np.array([0.4, 0.6, 0.8, 1.0])
THETA_CRIT, DELTA = np.pi / 4, np.pi / 18
G_SPHERE = 0.30166  # int_0^{pi/2} g sin, quad
BIN_DEG = 5.0
GLOBAL_TOL = 0.02
BIN_TOL = 0.06
RAW_DIR = Path(os.environ.get("CR_GAMMA_RAW", "/export/scratch/nknoell/gamma_obliquity"))
PICS_DIR = Path(__file__).resolve().parent / "pics" / "gamma_obliquity"


def g_of_theta(theta):
    return 0.5 * (np.tanh((THETA_CRIT - theta) / DELTA) + 1.0)


def _run(obliquity):
    per = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
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
            dsa_obliquity_dependence=obliquity,
        ),
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    n = NUM_CELLS
    shape = (n, n, n)
    dx = 1.0 / n
    weight = 0.5 * (1.0 - jnp.tanh((helper_data.r - R_EXPLOSION) / (SMOOTH_CELLS * dx)))
    delta_p = E_EXPLOSION * (GAMMA - 1.0) / (jnp.sum(weight) * dx**3)
    b0 = np.sqrt(2.0 * P_AMBIENT / BETA)  # p_B = B^2 / 2
    zeros = jnp.zeros(shape)
    state = construct_primitive_state(
        config=config,
        registered_variables=rv,
        density=jnp.ones(shape) * RHO_AMBIENT,
        velocity_x=zeros, velocity_y=zeros, velocity_z=zeros,
        magnetic_field_x=zeros, magnetic_field_y=zeros,
        magnetic_field_z=jnp.ones(shape) * b0,
        gas_pressure=P_AMBIENT + delta_p * weight,
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        snapshot_timepoints=jnp.array(SNAPSHOT_FRACTIONS * T_END),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0,
            dsa_efficiency=ZETA, dsa_mach_min=DSA_MACH_MIN,
        ),
    )
    snaps = time_integration(state, config, params, rv)
    states = np.asarray(snaps.states)
    assert np.all(np.isfinite(states)), f"non-finite state, obliquity = {obliquity}"
    return dict(states=states, times=np.asarray(snaps.time_points), config=config,
                params=params, rv=rv, helper_data=helper_data)


def _polar_angle(helper_data):
    c = np.moveaxis(np.asarray(helper_data.geometric_centers), -1, 0) - 0.5
    r = np.sqrt(np.sum(c**2, axis=0))
    return np.arccos(np.clip(np.abs(c[2]) / np.maximum(r, 1e-12), 0.0, 1.0)), r


def _probe(run):
    config, params, rv, hd = run["config"], run["params"], run["rv"], run["helper_data"]
    cr = params.cosmic_ray_grey_params

    def with_cr(**kw):
        return config._replace(cosmic_ray_grey_config=config.cosmic_ray_grey_config._replace(**kw))

    c_on = with_cr(dsa_mach_cr_aware=False, dsa_obliquity_dependence=True)
    c_off = with_cr(dsa_mach_cr_aware=False, dsa_obliquity_dependence=False)
    psi, _ = _polar_angle(hd)
    edges = np.deg2rad(np.arange(0.0, 90.0 + BIN_DEG, BIN_DEG))
    idx = np.clip(np.digitize(psi, edges) - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    dt = 0.1 / (NUM_CELLS * 200.0)
    on_bins, off_bins, dtheta_bins, on_tot, off_tot = [], [], [], [], []
    for s in run["states"]:
        s = jnp.asarray(s)
        d = {}
        for key, c in (("on", c_on), ("off", c_off)):
            out = inject_crs_at_shocks(s, c, params, rv, hd, 1.0, dt)
            d[key] = np.asarray(out[rv.cosmic_ray_e_index] - s[rv.cosmic_ray_e_index], dtype=np.float64)
        on_bins.append(np.bincount(idx.ravel(), d["on"].ravel(), nb))
        off_bins.append(np.bincount(idx.ravel(), d["off"].ravel(), nb))
        on_tot.append(d["on"].sum())
        off_tot.append(d["off"].sum())
        sf = find_shocks_pfrommer(
            s, config, rv, hd, mach_min=cr.dsa_mach_min, gamma_gas=GAMMA,
            **dsa_shock_finder_kwargs(config.cosmic_ray_grey_config),
        )
        theta, _ = shock_obliquity(s, sf, rv, normal_smoothing=config.cosmic_ray_grey_config.dsa_area_normal_smoothing)
        flux = np.asarray(sf.thermal_energy_flux)
        diff = np.asarray(theta) - psi
        med = np.full(nb, np.nan)
        for b in range(nb):
            sel = (idx == b) & (flux > 0)
            if sel.any():
                med[b] = np.median(diff[sel])
        dtheta_bins.append(med)
    g_bins = np.array([
        np.sum(g_of_theta(t) * np.sin(t)) / np.sum(np.sin(t))
        for t in (np.linspace(lo, hi, 201) for lo, hi in zip(edges[:-1], edges[1:]))
    ])
    return dict(edges=edges, on_bins=np.array(on_bins), off_bins=np.array(off_bins),
                dtheta_bins=np.array(dtheta_bins), on_tot=np.array(on_tot), off_tot=np.array(off_tot),
                g_bins=g_bins)


def _gamma_map(state, rv, axis):
    """Gamma-ray proxy int rho e_cr dl (global alpha: any band is this times a constant)."""
    return np.sum(state[rv.density_index] * state[rv.cosmic_ray_e_index], axis=axis) / NUM_CELLS


def _shell_radius(state, rv, along):
    """Radius of the density peak along a ray from the centre (z: along B, x: across)."""
    n = NUM_CELLS
    c = n // 2
    rho = state[rv.density_index]
    ray = rho[c, c, c:] if along == "z" else rho[c:, c, c]
    return (np.argmax(ray) + 0.5) / n


def _plot(res, maps, info, path):
    edges = np.rad2deg(res["edges"])
    centres = 0.5 * (edges[:-1] + edges[1:])
    ratio_bins = res["on_bins"].sum(0) / res["off_bins"].sum(0)
    fig = plt.figure(figsize=(15, 8.5))
    ax = fig.add_subplot(2, 3, 1)
    th = np.linspace(0, 90, 300)
    ax.plot(th, g_of_theta(np.deg2rad(th)), "k-", lw=1.0, label=r"$g(\psi)$")
    ax.step(edges, np.append(res["g_bins"], res["g_bins"][-1]), "k:", where="post", lw=1.0,
            label=r"$g$ bin mean")
    ax.plot(centres, ratio_bins, "o", mfc="none", color="C3", label="injected on / off")
    ax.set_xlabel(r"polar angle from $B$, $\psi$ [deg]")
    ax.set_ylabel("injected on / off")
    ax.set_title(f"global on/off = {info['global']:.4f} (sphere <g> = {G_SPHERE:.4f})", fontsize=10)
    ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 3, 4)
    for i, f in enumerate(SNAPSHOT_FRACTIONS):
        ax.plot(centres, np.rad2deg(res["dtheta_bins"][i]), ".-", lw=0.8, label=f"t = {f:g} t_end")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\psi$ [deg]")
    ax.set_ylabel(r"median $\theta_{Bn}$ read $-\ \psi$ [deg]")
    ax.legend(fontsize=8)
    vmax = max(m.max() for m in maps.values())
    panels = [("on_perp", "switch on, LOS $\\perp B$ (B vertical)", 2),
              ("off_perp", "switch off, LOS $\\perp B$", 3),
              ("on_par", "switch on, LOS $\\parallel B$", 5),
              ("off_par", "switch off, LOS $\\parallel B$", 6)]
    for key, title, pos in panels:
        ax = fig.add_subplot(2, 3, pos)
        img = maps[key]
        if key.endswith("perp"):
            img = img.T  # rows z (B up), columns y
        im = ax.imshow(img / vmax, origin="lower", extent=(0, 1, 0, 1), cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=fig.axes[-4:], shrink=0.8, label=r"$\int \rho\, e_{\rm cr}\, dl$ / max")
    fig.suptitle(
        f"MHD Sedov, uniform B (beta = {BETA:g}), zeta = {ZETA:g}, N = {NUM_CELLS}$^3$, t_end = {T_END}: "
        f"E_cr on/off = {info['ecr_ratio']:.3f}; shell radius along/across B = "
        f"{info['r_par']:.3f}/{info['r_perp']:.3f} (on), {info['r_par_off']:.3f}/{info['r_perp_off']:.3f} (off)",
        fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def test_dsa_obliquity_sedov_mhd():
    run_on = _run(True)
    res = _probe(run_on)
    run_off = _run(False)
    rv = run_on["rv"]
    s_on, s_off = run_on["states"][-1], run_off["states"][-1]
    dv = 1.0 / NUM_CELLS**3
    ecr_on = float(np.sum(s_on[rv.cosmic_ray_e_index], dtype=np.float64) * dv)
    ecr_off = float(np.sum(s_off[rv.cosmic_ray_e_index], dtype=np.float64) * dv)
    maps = dict(on_perp=_gamma_map(s_on, rv, 0), off_perp=_gamma_map(s_off, rv, 0),
                on_par=_gamma_map(s_on, rv, 2), off_par=_gamma_map(s_off, rv, 2))
    global_ratio = float(res["on_tot"].sum() / res["off_tot"].sum())
    info = dict(global_=global_ratio, ecr_ratio=ecr_on / ecr_off,
                r_par=_shell_radius(s_on, rv, "z"), r_perp=_shell_radius(s_on, rv, "x"),
                r_par_off=_shell_radius(s_off, rv, "z"), r_perp_off=_shell_radius(s_off, rv, "x"))
    info["global"] = info.pop("global_")
    ratio_bins = res["on_bins"].sum(0) / res["off_bins"].sum(0)
    dev = ratio_bins - res["g_bins"]
    print("per-snapshot global on/off:", np.round(res["on_tot"] / res["off_tot"], 4))
    print(f"global on/off {global_ratio:.4f} vs <g> {G_SPHERE:.4f}; E_cr(t_end) on/off {info['ecr_ratio']:.4f}; "
          f"E_cr on {ecr_on:.4e} off {ecr_off:.4e}")
    print("bin centres [deg]:", 0.5 * (np.rad2deg(res["edges"][:-1]) + np.rad2deg(res["edges"][1:])))
    print("on/off per bin:   ", np.round(ratio_bins, 3))
    print("g bin mean:       ", np.round(res["g_bins"], 3))
    print("theta - psi [deg] (t_end):", np.round(np.rad2deg(res["dtheta_bins"][-1]), 2))
    print(f"shell radius along / across B: on {info['r_par']:.4f} / {info['r_perp']:.4f}, "
          f"off {info['r_par_off']:.4f} / {info['r_perp_off']:.4f}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RAW_DIR / f"sedov_N{NUM_CELLS}.npz", **res, **{k: v for k, v in info.items()},
             **{f"map_{k}": v for k, v in maps.items()}, times=run_on["times"])
    np.savez(RAW_DIR / f"sedov_N{NUM_CELLS}_final_states.npz", on=s_on, off=s_off)
    _plot(res, maps, info, PICS_DIR / "dsa_obliquity_sedov_mhd.png")
    failures = []
    if abs(global_ratio - G_SPHERE) > GLOBAL_TOL:
        failures.append(f"global on/off {global_ratio:.4f} vs {G_SPHERE} (tol {GLOBAL_TOL})")
    if np.max(np.abs(dev)) > BIN_TOL:
        failures.append(f"max bin deviation {np.max(np.abs(dev)):.4f} (tol {BIN_TOL})")
    assert not failures, "; ".join(failures)


if __name__ == "__main__":
    test_dsa_obliquity_sedov_mhd()
