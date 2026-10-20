"""
Finder Mach number, shock area and DSA injection at planar shocks oblique to the grid: Phase B
test plan B8.2, and step 2 of the DB4/DB7 fix plan (``PROGRESS_PHASEB.md`` 2026-10-19).

**Setup.** 3D periodic unit box, N^3 (``CR_NUM_CELLS``), HLLC + MINMOD as item 7, grey CRs with
``v_red = 0`` and the constant DSA model. Planar fronts with normal ``k / |k|`` for integer ``k``
in ``ORIENTATIONS``: along the grid, 26.6 deg and 45 deg in a coordinate plane, (2,1,1) and the
body diagonal. With the phase ``theta = frac(k . x)`` the state is periodic; fronts are
``1 / |k|`` apart, so the true shock area per unit volume is ``|k|``. The gas-only RH post-shock
state of Mach number ``M`` (ambient ``rho = 1``, ``c = 1``) fills ``0.05 < theta < 0.45``
(1-cell tanh edges). The front at ``theta = 0.45`` is a single shock until rarefactions from
the slab's back edge arrive, well after ``t_end`` (the shock travels 0.35 of a period).
Runs: ``zeta = 0`` (plain gas shock) and ``zeta = 0.1``, M = 3 and 10.

**Measured** on 4 snapshots (``t >= 0.4 t_end``), on the shock-surface cells within 0.08 in
``theta`` of the front and at least ``MARGIN`` cells from the box faces (the finder's walk clamps
at the faces instead of wrapping):

* ``M_true``: the initial RH Mach number for ``zeta = 0``; with injection, that times the
  ratio of tracked shock speeds (front of the theta-binned density profile, linear fit over the
  measured snapshots) to the ``zeta = 0`` run (the tracking alone reads 1-2% high).
* True area: the fronts' intersection with the measuring cube (``_plane_area_in_cube``).
* Median finder Mach (gas pressure) and velocity-jump Mach over ``M_true`` (``_finder_estimators``).
* Area: surface cells x ``dx^2`` over the true area, unweighted and with ``1 / |n|_inf`` from
  the exact, the finder's and the smoothed-temperature (``sigma = 2``) normals.
* Injection: one probe call of ``inject_crs_at_shocks`` (``zeta = 0.1``, CFL-sized ``dt``) per
  snapshot, summed over the measuring region, over ``zeta delta(M_true) rho_1 (M_true c_1)^3 / 2``
  times the true area: the end-to-end injected flux per unit area (1 = exact).

Raw values: ``/export/scratch/nknoell/phaseB/b8_2/b8_2_N<N>_<k>.npz`` (``CR_B82_ORIENT`` picks a
subset, e.g. ``"111,211"``); figure ``pics/08_dsa_mach_dependence/cr_oblique_planar_shocks.png``
from every file found. Diagnostic, no gates yet (plan target: angle dependence of the injected
flux below 5% at N = 128).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import os
from pathlib import Path

# numerics
import numpy as np
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    PERIODIC_BOUNDARY,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix.shock_finder3D._energy_dissipation import thermal_energy_flux_from_pre_state

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_shock_finder_kwargs,
    inject_crs_at_shocks,
)

# estimators at the finder's sample points, normal variants
from _finder_estimators import finder_samples, ninf, normal_variants, unit

NUM_CELLS = int(os.environ.get("CR_NUM_CELLS", 64))
GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
DSA_MACH_MIN = 1.3
PROBE_ZETA = 0.1
ORIENTATIONS = ((1, 0, 0), (2, 1, 0), (1, 1, 0), (2, 1, 1), (1, 1, 1))
MACHS = (3.0, 10.0)
ZETAS = (0.0, 0.1)
THETA_BACK, THETA_FRONT, TRAVEL = 0.05, 0.45, 0.35
SNAPSHOT_FRACTIONS = np.array([0.25, 0.4, 0.6, 0.8, 1.0])
MEASURE_FROM = 0.4  # fraction of t_end
WINDOW = 0.08  # in theta around the front
MARGIN = 10  # cells from the box faces
RAW_DIR = Path(os.environ.get("CR_PHASEB_RAW", "/export/scratch/nknoell/phaseB")) / "b8_2"
PICS_DIR = Path(__file__).resolve().parent / "pics" / "08_dsa_mach_dependence"
NORMALS = ("exact", "finder", "gaussT2")


def _label(k):
    return "".join(str(c) for c in k)


def _rh_post(mach):
    rho1, p1 = 1.0, 1.0 / GAMMA
    x = (GAMMA + 1.0) * mach**2 / ((GAMMA - 1.0) * mach**2 + 2.0)
    p2 = p1 * (2.0 * GAMMA * mach**2 - (GAMMA - 1.0)) / (GAMMA + 1.0)
    return x * rho1, mach * (1.0 - 1.0 / x), p2


def _run(k, mach, zeta):
    kvec = np.asarray(k, float)
    kabs = float(np.linalg.norm(kvec))
    khat = kvec / kabs
    per = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    t_end = TRAVEL / (kabs * mach)
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        box_size=1.0,
        boundary_settings=BoundarySettings(x=per, y=per, z=per),
        exact_end_time=True,
        return_snapshots=True,
        num_snapshots=len(SNAPSHOT_FRACTIONS),
        use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_shock_acceleration=True
        ),
    )
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)
    centres = np.moveaxis(np.asarray(helper_data.geometric_centers), -1, 0)
    theta = np.mod(np.tensordot(kvec, centres, axes=1), 1.0)
    width = kabs / NUM_CELLS  # one cell along the normal, in theta
    w = (0.25 * (1.0 + np.tanh((theta - THETA_BACK) / width))
         * (1.0 - np.tanh((theta - THETA_FRONT) / width)))
    rho2, u2, p2 = _rh_post(mach)
    u = w * u2
    state = construct_primitive_state(
        config=config,
        registered_variables=rv,
        density=jnp.asarray(1.0 + w * (rho2 - 1.0)),
        velocity_x=jnp.asarray(u * khat[0]),
        velocity_y=jnp.asarray(u * khat[1]),
        velocity_z=jnp.asarray(u * khat[2]),
        gas_pressure=jnp.asarray(1.0 / GAMMA + w * (p2 - 1.0 / GAMMA)),
    )
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=t_end,
        gamma=GAMMA,
        snapshot_timepoints=jnp.array(SNAPSHOT_FRACTIONS * t_end),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR, reduced_streaming_speed=0.0,
            dsa_efficiency=zeta, dsa_mach_min=DSA_MACH_MIN,
        ),
    )
    snaps = time_integration(state, config, params, rv)
    states, times = np.asarray(snaps.states), np.asarray(snaps.time_points)
    assert not np.any(np.isnan(states)), f"NaN, k = {k}, M = {mach}, zeta = {zeta}"
    return dict(states=states, times=times, config=config, params=params, rv=rv,
                helper_data=helper_data, theta=theta, kabs=kabs, k=kvec, khat=khat, t_end=t_end)


def _front_theta(theta, rho, kabs):
    """Front of the theta-binned density profile: last half-level crossing ahead of the
    post-shock plateau (bins one cell wide along the normal)."""
    nb = int(round(NUM_CELLS / kabs))
    edges = np.linspace(0.0, 1.0, nb + 1)
    idx = np.clip(np.digitize(theta.ravel(), edges) - 1, 0, nb - 1)
    prof = np.bincount(idx, rho.ravel(), nb) / np.maximum(np.bincount(idx, minlength=nb), 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    behind = (centres > THETA_BACK + 0.1) & (centres < 0.97)
    j = int(np.argmax(np.where(behind, prof, -np.inf)))
    half = 0.5 * (prof[j] + 1.0)
    i = j
    while i + 1 < nb and prof[i + 1] > half:
        i += 1
    return centres[i] + (prof[i] - half) / (prof[i] - prof[i + 1]) * (centres[1] - centres[0])


def _plane_area_in_cube(k, theta_f, lo, hi, samples=2000):
    """Total area of the fronts ``k . x = theta_f + j`` (all integers ``j``) inside the cube
    ``[lo, hi)^3``: each sheet is a graph over the two minor axes, ``x_d = (c - k_o . x_o) / k_d``
    along the dominant axis ``d``, so its area is ``|k| / |k_d|`` times the area of the minor-axis
    square where ``lo <= x_d < hi`` (midpoint quadrature)."""
    k = np.asarray(k, float)
    d = int(np.argmax(np.abs(k)))
    o1, o2 = [i for i in range(3) if i != d]
    g = lo + (np.arange(samples) + 0.5) / samples * (hi - lo)
    y, z = np.meshgrid(g, g, indexing="ij")
    s = k[o1] * y + k[o2] * z
    corners = np.array([[a, b, c] for a in (lo, hi) for b in (lo, hi) for c in (lo, hi)]) @ k
    covered = 0.0
    for j in range(int(np.floor(corners.min() - theta_f)) - 1, int(np.ceil(corners.max() - theta_f)) + 2):
        x = (theta_f + j - s) / k[d]
        covered += np.mean((x >= lo) & (x < hi))
    return covered * (hi - lo) ** 2 * np.linalg.norm(k) / abs(k[d])


def _analyse(run, mach, zeta, v_s0=None):
    """Diagnostics of one run. The Mach number the ratios refer to is the initial RH Mach
    number for ``zeta = 0`` (the front tracking itself reads the speed 1-2% high, from the
    profile steepening out of the smooth initial step); with injection, ``mach`` times the
    tracked speed over that of the matching ``zeta = 0`` run (``v_s0``), so the bias cancels."""
    config, params, rv = run["config"], run["params"], run["rv"]
    helper_data, theta, kabs, khat = run["helper_data"], run["theta"], run["kabs"], run["khat"]
    cr_config, cr_params = config.cosmic_ray_grey_config, params.cosmic_ray_grey_params
    kwargs = dsa_shock_finder_kwargs(cr_config)
    dx = config.grid_spacing
    probe_params = params._replace(
        cosmic_ray_grey_params=cr_params._replace(dsa_efficiency=PROBE_ZETA))

    fronts = np.array([_front_theta(theta, s[rv.density_index], kabs) for s in run["states"]])
    late = run["times"] >= MEASURE_FROM * run["t_end"] - 1e-12  # past the start-up transient
    fit = np.polyfit(run["times"][late], fronts[late] / kabs, 1)
    v_s = fit[0]
    M_true = mach if v_s0 is None else mach * v_s / v_s0
    f_exact = float(thermal_energy_flux_from_pre_state(M_true, 1.0 / GAMMA, 1.0, GAMMA))

    idx = np.arange(NUM_CELLS)
    inner = (idx >= MARGIN) & (idx < NUM_CELLS - MARGIN)
    interior = inner[:, None, None] & inner[None, :, None] & inner[None, None, :]

    out = {k: [] for k in ("t", "mach_p", "mach_u", "mach_finder", "D_p", "D_u", "area_cells",
                           "inj", "n_cells", "closure_p", "angle_finder", "angle_gaussT2")}
    for k in NORMALS:
        out[f"area_{k}"], out[f"D_p_{k}"], out[f"D_u_{k}"] = [], [], []
    for t, s, th_f in zip(run["times"], run["states"], fronts):
        if t < MEASURE_FROM * run["t_end"] - 1e-12:
            continue
        sj = jnp.asarray(s)
        sf = find_shocks_pfrommer(sj, config, rv, helper_data, mach_min=DSA_MACH_MIN, **kwargs)
        dtheta = np.mod(theta - th_f + 0.5, 1.0) - 0.5
        region = interior & (np.abs(dtheta) < WINDOW)
        true_area = _plane_area_in_cube(run["k"], th_f, MARGIN * dx, (NUM_CELLS - MARGIN) * dx)
        surface = np.asarray(sf.shock_surface_cells) & region
        smp = finder_samples(s, sf, config, rv, kwargs, GAMMA, GAMMA_CR, select=region)
        valid = smp["valid"]

        normals = normal_variants(s, sf, rv, dx, mode="wrap")
        all_surface = np.asarray(sf.shock_surface_cells)
        in_region = region[all_surface]
        normals = {k: v[:, in_region] for k, v in normals.items()}
        normals["exact"] = np.repeat(khat[:, None], int(surface.sum()), axis=1)
        valid_in_surface = valid[surface]

        # injection probe: what one step deposits in the region, per unit time
        u = np.stack([s[rv.velocity_index.x], s[rv.velocity_index.y], s[rv.velocity_index.z]])
        p_cr = (GAMMA_CR - 1.0) * s[rv.cosmic_ray_e_index]
        c_eff = np.sqrt((GAMMA * s[rv.pressure_index] + GAMMA_CR * p_cr) / s[rv.density_index])
        dt_probe = float(params.C_cfl * dx / np.max(np.max(np.abs(u), axis=0) + c_eff))
        injected = np.asarray(inject_crs_at_shocks(sj, config, probe_params, rv, helper_data,
                                                   current_time=t, dt=dt_probe))
        delta = (injected[rv.cosmic_ray_e_index] - s[rv.cosmic_ray_e_index])[interior]
        inj = float(delta.sum()) * dx**3 / dt_probe

        norm = f_exact * true_area
        out["t"].append(float(t))
        out["mach_p"].append(float(np.median(smp["M"]["p"])) / M_true)
        out["mach_u"].append(float(np.median(smp["M"]["u"])) / M_true)
        out["mach_finder"].append(float(np.median(np.asarray(sf.mach_numbers)[valid])) / M_true)
        out["D_p"].append(float(smp["flux"]["p"].sum()) * dx**2 / norm)
        out["D_u"].append(float(smp["flux"]["u"].sum()) * dx**2 / norm)
        out["area_cells"].append(float(surface.sum()) * dx**2 / true_area)
        out["inj"].append(inj / (PROBE_ZETA * norm))
        out["n_cells"].append(int(valid.sum()))
        out["closure_p"].append(smp["closure_p"])
        for k in NORMALS:
            wgt = 1.0 / np.maximum(ninf(normals[k]), 1e-6)
            out[f"area_{k}"].append(float(wgt.sum()) * dx**2 / true_area)
            out[f"D_p_{k}"].append(float(np.sum(smp["flux"]["p"] * wgt[valid_in_surface]))
                                   * dx**2 / norm)
            out[f"D_u_{k}"].append(float(np.sum(smp["flux"]["u"] * wgt[valid_in_surface]))
                                   * dx**2 / norm)
        for k in ("finder", "gaussT2"):
            cos = np.clip(np.sum(normals[k] * normals["exact"], axis=0), -1.0, 1.0)
            out[f"angle_{k}"].append(float(np.degrees(np.median(np.arccos(cos)))))
    res = {k: np.asarray(v) for k, v in out.items()}
    res["M_true"] = np.asarray(M_true)
    res["v_fit"] = np.asarray(v_s)
    res["M_init"] = np.asarray(mach)
    res["zeta"] = np.asarray(zeta)
    res["front_fit_rms"] = np.asarray(np.std(
        fronts[late] / kabs - np.polyval(fit, run["times"][late])) / dx)
    return res


def test_cr_oblique_planar_shocks():
    chosen = os.environ.get("CR_B82_ORIENT")
    orientations = [k for k in ORIENTATIONS if chosen is None or _label(k) in chosen.split(",")]
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for k in orientations:
        res = {}
        for mach in MACHS:
            v_s0 = None
            for zeta in ZETAS:  # zeta = 0 first: its tracked speed calibrates the others
                r = _analyse(_run(k, mach, zeta), mach, zeta, v_s0)
                if zeta == 0:
                    v_s0 = float(r["v_fit"])
                tag = f"M{mach:g}_z{zeta:g}"
                res.update({f"{tag}_{key}": v for key, v in r.items()})
                print(f"N = {NUM_CELLS}, k = {k} (|n|_inf = {ninf(unit(np.array(k, float)[:, None]))[0]:.4f}), "
                      f"M = {mach:g}, zeta = {zeta:g}: M_true {float(r['M_true']):.4f} (tracked {float(r['v_fit']):.4f}), "
                      f"front fit rms {float(r['front_fit_rms']):.3f} dx, cells {r['n_cells'].min()}.., "
                      f"closure {r['closure_p'].min():.5f}")
                print("   mean: M_p {:.4f}  M_u {:.4f}  M_fin {:.4f} | area cells {:.4f}  exact {:.4f}  "
                      "finder {:.4f}  gaussT2 {:.4f} | D_p {:.4f}  D_u {:.4f}  D_u_gaussT2 {:.4f}  "
                      "D_p_exact {:.4f} | inj {:.4f} | angle finder {:.2f}  gaussT2 {:.2f}".format(
                          *(r[x].mean() for x in ("mach_p", "mach_u", "mach_finder", "area_cells",
                                                  "area_exact", "area_finder", "area_gaussT2", "D_p",
                                                  "D_u", "D_u_gaussT2", "D_p_exact", "inj",
                                                  "angle_finder", "angle_gaussT2"))))
        np.savez(RAW_DIR / f"b8_2_N{NUM_CELLS}_{_label(k)}.npz", **res)
    _plot()


def _plot():
    files = sorted(RAW_DIR.glob("b8_2_N*_*.npz"))
    if not files:
        return
    data = {}
    for f in files:
        n, lab = f.stem.split("_")[2:4]
        data[int(n[1:]), lab] = np.load(f)
    ns = sorted({n for n, _ in data})
    labels = [_label(k) for k in ORIENTATIONS if any((n, _label(k)) in data for n in ns)]
    x = np.arange(len(labels))
    ninfs = [ninf(unit(np.array([int(c) for c in lab], float)[:, None]))[0] for lab in labels]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    styles = {3.0: "o", 10.0: "s"}
    for n_i, n in enumerate(ns):
        ls = "-" if n_i == len(ns) - 1 else ":"
        for mach in MACHS:
            tag = f"M{mach:g}_z0"
            def series(key):
                return [data[n, lab][f"{tag}_{key}"].mean() if (n, lab) in data else np.nan
                        for lab in labels]
            m = styles[mach]
            axes[0].plot(x, series("mach_p"), ls, marker=m, color="C0",
                         label=f"pressure, M = {mach:g}, N = {n}")
            axes[0].plot(x, series("mach_u"), ls, marker=m, color="C1",
                         label=f"velocity, M = {mach:g}, N = {n}")
            axes[1].plot(x, series("area_cells"), ls, marker=m, color="k",
                         label=f"cell count, M = {mach:g}, N = {n}")
            axes[1].plot(x, series("area_finder"), ls, marker=m, color="C2",
                         label=rf"$1/|n|_\infty$ finder, M = {mach:g}, N = {n}")
            axes[1].plot(x, series("area_gaussT2"), ls, marker=m, color="C4",
                         label=rf"$1/|n|_\infty$ $\sigma = 2$, M = {mach:g}, N = {n}")
            axes[2].plot(x, series("inj"), ls, marker=m, color="C0",
                         label=f"code now (probe), M = {mach:g}, N = {n}")
            axes[2].plot(x, series("D_u_gaussT2"), ls, marker=m, color="C3",
                         label=rf"velocity + $\sigma = 2$ weight, M = {mach:g}, N = {n}")
    axes[1].plot(x, ninfs, "k--", lw=0.8, label=r"$|n|_\infty$ (prediction)")
    axes[0].set_ylabel(r"median $M_{\rm est} / M_{\rm true}$")
    axes[0].set_title(r"(a) Mach estimators ($\zeta = 0$)")
    axes[1].set_ylabel("surface area / true area")
    axes[1].set_title(r"(b) Area ($\zeta = 0$)")
    axes[2].set_ylabel("injected (dissipated) flux per area / exact")
    axes[2].set_title(r"(c) Injection at $\zeta = 0.1$ per area ($\zeta = 0$ runs)")
    for ax in axes:
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(x, [f"({','.join(lab)})\n{v:.3f}" for lab, v in zip(labels, ninfs)])
        ax.set_xlabel(r"shock normal $k$, $|n|_\infty$")
        ax.legend(fontsize=6)
    fig.suptitle("B8.2: planar shocks oblique to the grid (3D periodic), finder diagnostics")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_oblique_planar_shocks.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    test_cr_oblique_planar_shocks()
