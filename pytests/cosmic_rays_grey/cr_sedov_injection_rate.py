"""
Instantaneous DSA injection rate of the item-7 Sedov blast against the
self-similar two-fluid rate (Phase B test plan B7.3).

Item 7 (``cr_sedov_taylor.py``) only gates the time-integrated ``E_cr``, which
is dominated by the least-resolved early blast (plan Y4: ~39% of the reference
``E_cr`` sits in gas shocked while ``R_s < R/4``). This test splits the
deficit: it reruns the item-7 setup unchanged (``dsa_efficiency`` 0 and 0.1,
same box, explosion, sampling, CR transport) with ten snapshots, and on every
snapshot calls the pure functions the run itself uses,
``find_shocks_pfrommer`` and ``inject_crs_at_shocks``. Per snapshot:

- ``R_num``: the shock front, where the shell-averaged density excess falls to
  half its peak on the outer flank, and the parabolic peak of the total
  pressure. At 48^3 the front sits ~1 cell ahead of and the peak ~1 cell behind
  the shock. Both are fitted with ``R = R_off + xi (E t^2/rho_0)^(1/5)`` over
  ``t >= 0.03`` (the offset absorbs the estimator bias); the mean ``xi`` gives
  the simulated shock's radius and speed ``V = (2/5) R / t``, and
  ``M_true = V / c_0``. The two ``xi`` differ by ~2% at 48^3.
- ``D``: the finder's dissipation rate as the injection counts it,
  ``sum(thermal_energy_flux w) dx^2`` with the area weight ``w = 1 / |n|_inf``
  of ``dsa_area_weighting`` (default since 2026-10-19, DB7; ``w = 1``
  without), split as ``D = A_inj * <f>`` (the weighted surface area and the
  mean flux per unit area).
- ``M_finder``: median Mach number of the surface cells (from the velocity
  jump, ``dsa_mach_from_velocity``, default since 2026-10-19, DB4).
- the nominal injection rate ``zeta D`` (what ``inject_crs_at_shocks`` spreads
  before its positivity cap; the spread conserves the total, item 8's
  cross-check), and the realized rate after the cap, probed with a CFL-sized
  ``dt`` (``C_cfl dx / max(|u|_inf + c_eff, v_red)``, an estimate of the run's
  step; the cap only binds when ``dt`` is large).
- ``E_cr / E``.

Reference (``sedov_two_fluid.py``, B7.1): the control blast is the classic
Sedov solution, the DSA blast the self-similar solution with ``zeta`` = 0.1 in
the code's ``"gas_rh"`` convention (CR flux = ``zeta`` x gas-only RH dissipated
flux). Both rates are constant in units of ``E / t``:

- finder dissipation ``D t / E`` = 0.458 (control), 0.416 (DSA),
- injection ``t / E dE_inj/dt`` = 0.0416 (DSA),
- ``E_cr / E`` = 0.140, constant in time (injection balances the adiabatic
  losses).

``D`` is compared three ways: at the run's time (``D t / E``, which also
contains the time offset of the finite explosion region); at the run's own
radius, ``D / D_ref(R)`` with ``D_ref(R) = 0.416 E / t_R`` and ``t_R`` the time
at which the reference reaches ``R``; and against the simulated shock itself,
``D / D_meas`` with ``D_meas = 4 pi R^2 delta rho_0 V^3 / 2`` (strong-shock
``delta`` = 0.5625). The last one isolates the finder from the hydro:
``D / D_meas = (A_inj / 4 pi R^2) (<f> / f_meas)``.

- ``A_inj / 4 pi R^2``: a shock with normal ``n`` crosses ``|n|_inf`` cells
  per ``dx^2`` of its area (``A_cells / 4 pi R^2``, the plain cell count, is
  ~0.84 at every resolution; ``<|n|_inf>`` = 0.831 over a sphere). With the
  area weight each cell counts ``dx^2 / |n|_inf`` with the normal of the
  Gaussian-smoothed temperature (``A_oblique``; ~1.02-1.07 here, slightly
  above 1 since the normal is still tilted ~3 deg away from the grid axes).
- ``<f> / f_meas``: at these Mach numbers (~150-500, ``delta(M)`` saturated)
  ~ ``(M_finder / M_true)^3``. So the finder's Mach bias enters the injection
  with its third power. Behind the thin Sedov shell the pressure falls
  steeply, so the gas-pressure Mach number (the default until 2026-10-19)
  reads 0.76/0.84/0.87 of the true one at 48^3/96^3/160^3; the velocity jump
  reads ~0.88/0.94/0.94 (``cr_sedov_mach_estimators.py``).

Gates. Hard: the control run injects nothing; the realized rate never exceeds
the nominal one; the front and peak fits agree on ``xi`` within 3%, and the
control blast's ``xi`` is Sedov's within 3% (checks the snapshot analysis; the
control blast is validated in item 7). Everything else is tracked in ``baselines.json`` per
resolution (``CR_NUM_CELLS``, metric suffix ``_N<cells>``); the plan's target
(``D / D_ref`` >= 0.9 at N = 160) is reported, not gated.

Adiabatic memory: ``E_cr`` is also predicted from the run's own injection
rate with the reference's adiabatic kernel (``_adiabatic_kernel``: a shell
keeps ``K ~ (R_s/R)^(3/4)`` of its CR energy, Bell 2015), so that a mismatch in
``E_cr`` can be split into injection and losses. Hard gate: the kernel closes
on the reference (``injection rate x int K ds/s = f_cr``) to 1e-3.

Raw per-snapshot numbers go to ``/export/scratch/nknoell/phaseB/b7_3/``; the
figure (``pics/07_sedov_taylor/cr_sedov_injection_rate.png``) overlays every
resolution found there, each quantity as a ratio to the reference.
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

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import SnapshotSettings

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix.shock_finder3D._surface_area import surface_area_weights

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_shock_finder_kwargs,
    inject_crs_at_shocks,
)

# reference
from astronomix.test_setups.reference_solutions.sedov_two_fluid import (
    solve_sedov_two_fluid,
)

# item-7 setup, unchanged
from cr_sedov_taylor import (
    DSA_EFFICIENCY,
    DSA_MACH_MIN,
    E_EXPLOSION,
    GAMMA,
    P_AMBIENT,
    R_EXPLOSION,
    RHO_AMBIENT,
    SMOOTH_CELLS,
    T_END,
)
from cr_baselines import Baselines

NUM_CELLS = int(os.environ.get("CR_NUM_CELLS", 48))
SNAPSHOT_TIMES = np.linspace(0.007, T_END, 10)
LATE_TIME = 0.03  # plan B7.3: rates should have settled by t >= 0.03
GAMMA_CR = 4.0 / 3.0
RAW_DIR = Path(os.environ.get("CR_PHASEB_RAW", "/export/scratch/nknoell/phaseB/b7_3"))
PICS_DIR = Path(__file__).resolve().parent / "pics" / "07_sedov_taylor"
# reanalyse the snapshot states saved in RAW_DIR instead of rerunning
REUSE_STATES = os.environ.get("CR_B73_REUSE", "0") not in ("", "0")


def _run_with_snapshots(dsa_efficiency: float):
    """The item-7 run (``cr_sedov_taylor._run_sedov``) with snapshots."""
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        exact_end_time=True,
        return_snapshots=True,
        num_snapshots=len(SNAPSHOT_TIMES),
        use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_shock_acceleration=True
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (NUM_CELLS, NUM_CELLS, NUM_CELLS)
    zeros = jnp.zeros(shape)
    dx = 1.0 / NUM_CELLS
    weight = 0.5 * (1.0 - jnp.tanh((helper_data.r - R_EXPLOSION) / (SMOOTH_CELLS * dx)))
    delta_p = E_EXPLOSION * (GAMMA - 1.0) / (jnp.sum(weight) * dx**3)
    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.ones(shape) * RHO_AMBIENT,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=P_AMBIENT + delta_p * weight,
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        snapshot_timepoints=jnp.array(SNAPSHOT_TIMES),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            dsa_efficiency=dsa_efficiency, dsa_mach_min=DSA_MACH_MIN
        ),
    )
    states_file = RAW_DIR / f"states_eff{dsa_efficiency:g}_N{NUM_CELLS}.npz"
    if REUSE_STATES and states_file.exists():
        saved = np.load(states_file)
        states, times = saved["states"], saved["times"]
    else:
        snapshots = time_integration(initial_state, config, params, registered_variables)
        states, times = np.asarray(snapshots.states), np.asarray(snapshots.time_points)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(states_file, states=states, times=times)
    return dict(
        states=states,
        times=times,
        config=config,
        params=params,
        registered_variables=registered_variables,
        helper_data=helper_data,
    )


def _shell_mean(r_flat, field_flat, dx):
    """Shell average of a field in radial bins of width dx/2."""
    edges = np.arange(0.0, 0.5 * np.sqrt(3.0) + dx, 0.5 * dx)
    idx = np.digitize(r_flat, edges) - 1
    counts = np.bincount(idx, minlength=len(edges))[: len(edges) - 1]
    sums = np.bincount(idx, weights=field_flat, minlength=len(edges))[: len(edges) - 1]
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres, sums / np.maximum(counts, 1), counts


def _shock_radius(r_flat, rho_flat, p_flat, dx):
    """Shock radius from shell averages: ``R_front``, where the density excess
    falls to half its peak on the outer flank (the smeared front's midpoint),
    and ``R_peak``, the parabolic peak of the total pressure (lags the front by
    about a cell at 48^3)."""
    centres, rho_mean, counts = _shell_mean(r_flat, rho_flat, dx)
    _, p_mean, _ = _shell_mean(r_flat, p_flat, dx)
    valid = (centres > 2.0 * R_EXPLOSION) & (centres < 0.5) & (counts > 0)
    i = int(np.argmax(np.where(valid, p_mean, -np.inf)))
    y0, y1, y2 = p_mean[i - 1], p_mean[i], p_mean[i + 1]
    curv = y0 - 2.0 * y1 + y2
    r_peak = centres[i] + (0.5 * (y0 - y2) / curv if curv != 0 else 0.0) * (centres[1] - centres[0])
    excess = rho_mean - RHO_AMBIENT
    j = int(np.argmax(np.where(valid, excess, -np.inf)))
    half = 0.5 * excess[j]
    k = j
    while k + 1 < len(centres) and excess[k + 1] > half:
        k += 1
    # linear interpolation between k (above half) and k + 1 (below)
    r_front = centres[k] + (excess[k] - half) / (excess[k] - excess[k + 1]) * (centres[1] - centres[0])
    return r_front, r_peak


def _analyse(run, reference, zeta):
    """Per-snapshot shock radius, finder dissipation, Mach, injection, E_cr."""
    config, params = run["config"], run["params"]
    rv, helper_data = run["registered_variables"], run["helper_data"]
    cr_config = config.cosmic_ray_grey_config
    cr_params = params.cosmic_ray_grey_params
    dx = config.grid_spacing
    r_flat = np.asarray(helper_data.r).reshape(-1)
    c_0 = np.sqrt(GAMMA * P_AMBIENT / RHO_AMBIENT)
    xi_0 = reference.xi_0
    out = {k: [] for k in (
        "t", "R_num", "R_peak", "R_ref", "t_R", "D", "A_cells", "A_oblique", "A_inj", "n_surf",
        "M_finder",
        "inj_nominal", "inj_realized", "dt_probe", "E_cr", "E_total",
    )}
    for t, state in zip(run["times"], run["states"]):
        state = jnp.asarray(state)
        rho = state[rv.density_index]
        p_gas = state[rv.pressure_index]
        e_cr = state[rv.cosmic_ray_e_index]
        u = jnp.stack([state[rv.velocity_index.x], state[rv.velocity_index.y],
                       state[rv.velocity_index.z]])
        p_cr = (GAMMA_CR - 1.0) * e_cr
        R_num, R_peak = _shock_radius(
            r_flat, np.asarray(rho).reshape(-1), np.asarray(p_gas + p_cr).reshape(-1), dx
        )
        # time at which the self-similar blast has radius R_num
        t_R = (R_num / xi_0) ** 2.5 * np.sqrt(RHO_AMBIENT / E_EXPLOSION)

        sf = find_shocks_pfrommer(
            state, config, rv, helper_data,
            mach_min=cr_params.dsa_mach_min, **dsa_shock_finder_kwargs(cr_config),
        )
        surface = np.asarray(sf.shock_surface_cells)
        mach = np.asarray(sf.mach_numbers)[surface]
        flux = np.asarray(sf.thermal_energy_flux)
        # A shock with normal n crosses |n|_inf cells per dx^2 of its area (one
        # surface cell per grid line along the dominant axis), so the true area
        # a surface cell represents is dx^2 / |n|_inf (DB7). A_oblique: that
        # area with the smoothed-temperature normal; the injection weights by
        # it when dsa_area_weighting is on (default), else counts dx^2 per cell.
        weights = np.asarray(surface_area_weights(
            p_gas, rho, sigma=cr_config.dsa_area_normal_smoothing))
        w_inj = weights if cr_config.dsa_area_weighting else np.ones_like(weights)
        D = float(np.sum(flux * w_inj)) * dx**2
        A_oblique = float(np.sum(weights[surface])) * dx**2
        A_inj = float(np.sum(w_inj[surface])) * dx**2

        # CFL-sized probe dt for the positivity cap (estimate of the run's step)
        c_eff = jnp.sqrt((GAMMA * p_gas + GAMMA_CR * p_cr) / rho)
        signal = jnp.max(jnp.max(jnp.abs(u), axis=0) + c_eff)
        signal = max(float(signal), cr_params.reduced_streaming_speed)
        dt_probe = params.C_cfl * dx / signal
        injected = inject_crs_at_shocks(
            state, config, params, rv, helper_data, current_time=t, dt=dt_probe,
        )
        delta = injected[rv.cosmic_ray_e_index] - e_cr
        inj_realized = float(jnp.sum(delta)) * dx**3 / dt_probe

        E_kin = 0.5 * rho * jnp.sum(u**2, axis=0)
        out["t"].append(float(t))
        out["R_num"].append(R_num)
        out["R_peak"].append(R_peak)
        out["R_ref"].append(reference.shock_radius(float(t), E_EXPLOSION, RHO_AMBIENT))
        out["t_R"].append(t_R)
        out["D"].append(D)
        out["A_cells"].append(float(surface.sum()) * dx**2)
        out["A_oblique"].append(A_oblique)
        out["A_inj"].append(A_inj)
        out["n_surf"].append(int(surface.sum()))
        out["M_finder"].append(float(np.median(mach)) if mach.size else 0.0)
        out["inj_nominal"].append(zeta * D)
        out["inj_realized"].append(inj_realized)
        out["dt_probe"].append(dt_probe)
        out["E_cr"].append(float(jnp.sum(e_cr)) * dx**3)
        out["E_total"].append(float(jnp.sum(p_gas / (GAMMA - 1.0) + E_kin + e_cr)) * dx**3)
    return {k: np.asarray(v) for k, v in out.items()}


def _derived(res, reference):
    """Ratios against the self-similar reference."""
    E = E_EXPLOSION
    d = dict(res)
    c_0 = np.sqrt(GAMMA * P_AMBIENT / RHO_AMBIENT)
    # The simulated shock's radius and speed: fit R = R_off + xi (E t^2/rho_0)^(1/5)
    # for t >= LATE_TIME. R_off absorbs the constant offset of the radius
    # estimator (front midpoint ahead, pressure peak behind the shock by about
    # a cell at 48^3); xi gives the shock radius and speed. Fitted separately to
    # the front and the peak, the two xi must agree.
    late = res["t"] >= LATE_TIME
    scale = (E / RHO_AMBIENT * res["t"] ** 2) ** 0.2
    design = np.stack([np.ones(late.sum()), scale[late]], axis=1)
    (d["R_off_front"], d["xi_front"]), *_ = np.linalg.lstsq(design, res["R_num"][late], rcond=None)
    (d["R_off_peak"], d["xi_peak"]), *_ = np.linalg.lstsq(design, res["R_peak"][late], rcond=None)
    # the front sits ahead of, the pressure peak behind the shock: take the mean
    # (their difference, ~2% at 48^3, is the radius uncertainty)
    d["xi_fit"] = 0.5 * (d["xi_front"] + d["xi_peak"])
    d["R_true"] = d["xi_fit"] * scale
    d["V_true"] = 0.4 * d["R_true"] / res["t"]
    d["M_true"] = d["V_true"] / c_0
    # gas-only RH dissipated flux of the simulated shock (strong limit,
    # delta = 4/(gamma+1)^2) over the sphere of radius R_true
    delta_strong = 4.0 / (GAMMA + 1.0) ** 2
    d["D_meas"] = 4.0 * np.pi * d["R_true"] ** 2 * delta_strong * 0.5 * RHO_AMBIENT * d["V_true"] ** 3
    d["D_over_meas"] = res["D"] / d["D_meas"]
    # reference dissipation at the run's own radius: D_ref * E / t_R
    t_R = (d["R_true"] / reference.xi_0) ** 2.5 * np.sqrt(RHO_AMBIENT / E)
    d["D_over_ref_R"] = res["D"] / (reference.finder_dissipation_rate * E / t_R)
    d["D_t_over_E"] = res["D"] * res["t"] / E
    d["D_over_ref_t"] = d["D_t_over_E"] / reference.finder_dissipation_rate
    d["area_ratio"] = res["A_cells"] / (4.0 * np.pi * d["R_true"] ** 2)
    d["area_ratio_oblique"] = res["A_oblique"] / (4.0 * np.pi * d["R_true"] ** 2)
    d["area_ratio_inj"] = res["A_inj"] / (4.0 * np.pi * d["R_true"] ** 2)
    d["flux_ratio"] = d["D_over_meas"] / d["area_ratio_inj"]
    d["mach_ratio"] = res["M_finder"] / d["M_true"]
    d["inj_t_over_E"] = res["inj_nominal"] * res["t"] / E
    d["cap_loss"] = np.where(
        res["inj_nominal"] > 0, 1.0 - res["inj_realized"] / np.maximum(res["inj_nominal"], 1e-300), 0.0
    )
    d["E_cr_over_E"] = res["E_cr"] / E
    d["R_over_ref"] = res["R_num"] / res["R_ref"]
    return d


def _late_mean(d, key):
    late = d["t"] >= LATE_TIME
    return float(np.mean(d[key][late]))


def test_cr_sedov_injection_rate():
    """B7.3: instantaneous injection rate and finder dissipation vs. the reference."""
    ref_ctrl = solve_sedov_two_fluid(0.0)
    ref_dsa = solve_sedov_two_fluid(DSA_EFFICIENCY, convention="gas_rh")
    results = {}
    for name, zeta, reference in (("control", 0.0, ref_ctrl), ("dsa", DSA_EFFICIENCY, ref_dsa)):
        run = _run_with_snapshots(zeta)
        assert not np.any(np.isnan(run["states"])), f"{name} run produced NaNs"
        res = _analyse(run, reference, zeta)
        del run
        results[name] = _derived(res, reference)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RAW_DIR / f"b7_3_N{NUM_CELLS}.npz",
             **{f"{name}_{k}": v for name, d in results.items() for k, v in d.items()})

    ctrl, dsa = results["control"], results["dsa"]
    print(f"N = {NUM_CELLS}; reference: D t/E {ref_ctrl.finder_dissipation_rate:.4f} (control), "
          f"{ref_dsa.finder_dissipation_rate:.4f} (DSA), injection t/E "
          f"{ref_dsa.injection_rate:.4f}, E_cr/E {ref_dsa.f_cr:.4f}")
    header = ("   t     R_front R_peak R/Rref D t/E  D/Dref(R) D/Dmeas A/4piR2 Aobl/4piR2 "
              "f/fmeas M_fin  M_true (M ratio)^3 inj t/E  cap      E_cr/E")
    for name, d in results.items():
        print(f"{name}: xi fit (t >= {LATE_TIME}) front {float(d['xi_front']):.4f} "
              f"(offset {float(d['R_off_front']):+.4f}), peak {float(d['xi_peak']):.4f} "
              f"(offset {float(d['R_off_peak']):+.4f}); reference "
              f"{(ref_ctrl if name == 'control' else ref_dsa).xi_0:.4f}\n{header}")
        for i in range(len(d["t"])):
            print(f"  {d['t'][i]:.4f} {d['R_num'][i]:.4f}  {d['R_peak'][i]:.4f} "
                  f"{d['R_over_ref'][i]:.4f} "
                  f"{d['D_t_over_E'][i]:.4f} {d['D_over_ref_R'][i]:.4f}    "
                  f"{d['D_over_meas'][i]:.4f}  {d['area_ratio'][i]:.4f}  "
                  f"{d['area_ratio_oblique'][i]:.4f}     "
                  f"{d['flux_ratio'][i]:.4f} {d['M_finder'][i]:6.1f} {d['M_true'][i]:6.1f} "
                  f"  {d['mach_ratio'][i] ** 3:.4f}     {d['inj_t_over_E'][i]:.4f}  "
                  f"{d['cap_loss'][i]:+.1e} {d['E_cr_over_E'][i]:.4f}")

    # hard gates
    assert np.all(ctrl["inj_realized"] == 0.0) and np.all(ctrl["inj_nominal"] == 0.0), (
        "control run (dsa_efficiency = 0) injects CR energy")
    assert np.all(dsa["inj_realized"] <= dsa["inj_nominal"] * (1.0 + 1e-4) + 1e-12), (
        "realized injection exceeds the nominal zeta * D (the cap can only remove energy)")
    for name, d, reference in (("control", ctrl, ref_ctrl), ("dsa", dsa, ref_dsa)):
        xi_front, xi_peak = float(d["xi_front"]), float(d["xi_peak"])
        assert abs(xi_front - xi_peak) < 0.03 * reference.xi_0, (
            f"{name}: front and peak fits disagree (xi {xi_front:.4f} vs {xi_peak:.4f})")
        if name == "control":
            xi_fit = float(d["xi_fit"])
            assert abs(xi_fit / reference.xi_0 - 1.0) < 0.03, (
                f"control blast xi {xi_fit:.4f} vs Sedov {reference.xi_0:.4f}")

    # tracked metrics (late-time means), per resolution
    late_metrics = {
        "ctrl_D_over_ref_R": _late_mean(ctrl, "D_over_ref_R"),
        "dsa_D_over_ref_R": _late_mean(dsa, "D_over_ref_R"),
        "dsa_D_over_ref_t": _late_mean(dsa, "D_over_ref_t"),
        "ctrl_D_over_meas": _late_mean(ctrl, "D_over_meas"),
        "dsa_D_over_meas": _late_mean(dsa, "D_over_meas"),
        "ctrl_xi_over_ref": float(ctrl["xi_fit"]) / ref_ctrl.xi_0,
        "dsa_xi_over_ref": float(dsa["xi_fit"]) / ref_dsa.xi_0,
        "dsa_area_ratio": _late_mean(dsa, "area_ratio"),
        "dsa_area_ratio_oblique": _late_mean(dsa, "area_ratio_oblique"),
        "dsa_area_ratio_inj": _late_mean(dsa, "area_ratio_inj"),
        "dsa_flux_ratio": _late_mean(dsa, "flux_ratio"),
        "dsa_mach_ratio": _late_mean(dsa, "mach_ratio"),
        "dsa_inj_over_ref": _late_mean(dsa, "inj_t_over_E") / ref_dsa.injection_rate,
        "dsa_cap_loss": _late_mean(dsa, "cap_loss"),
        "dsa_E_cr_over_ref_end": float(dsa["E_cr_over_E"][-1]) / ref_dsa.f_cr,
    }
    # adiabatic memory: E_cr predicted from the run's own injection rate with
    # the reference's adiabatic kernel (are the run's losses the reference's?)
    kernel = _adiabatic_kernel(ref_dsa)
    closure = ref_dsa.injection_rate * kernel[2][-1] / ref_dsa.f_cr
    assert abs(closure - 1.0) < 1e-3, f"adiabatic kernel closure {closure:.5f}"
    pred = _kernel_prediction(dsa["t"], dsa["inj_nominal"], dsa["E_cr"], kernel)
    print(f"adiabatic kernel: closure {closure:.5f}, K(0.1) = "
          f"{np.interp(0.1, kernel[0], kernel[1]):.3f}, share of E_cr injected before "
          f"0.1 t: {np.interp(0.1, kernel[0], kernel[2]) / kernel[2][-1]:.3f}; "
          f"E_cr(t_end) run / kernel prediction {dsa['E_cr'][-1] / pred[-1]:.3f}")
    print("late-time (t >= 0.03) means:")
    for k, v in late_metrics.items():
        print(f"  {k}: {v:.4f}")
    baselines = Baselines("cr_sedov_injection_rate.py::test_cr_sedov_injection_rate")
    for k, v in late_metrics.items():
        if k == "dsa_cap_loss":
            baselines.upper(f"{k}_N{NUM_CELLS}", v, floor=1e-3)
        else:
            baselines.close(f"{k}_N{NUM_CELLS}", v, rel=0.05)
    _plot(ref_ctrl, ref_dsa)
    baselines.check()


def _adiabatic_kernel(reference):
    """Adiabatic memory of the self-similar reference.

    A shell shocked at ``t_s = s t`` received the CR energy injected at
    ``t_s``; at ``t`` it holds the fraction ``K(s)`` of it. Its specific CR
    energy scales as ``V_s^2 Pi_cr / G`` with ``V_s^2 ~ t^(-6/5)`` and the shell
    sits where ``r_0 / R = s^(2/5)``. Asymptotically ``K ~ s^0.3 =
    (R_s/R)^(3/4)`` (relativistic gas in the near-isobaric Sedov interior,
    ``P ~ R^-3``, energy per particle ``~ P^(1/4)``; Bell 2015), not the
    ``(R_s/R)^1`` of homologous expansion. Returns ``s``, ``K`` and the
    cumulative ``F(s) = int_0^s K ds'/s'`` normalised to 1 at ``s = 1`` (the
    fraction of the reference's ``E_cr(t)`` injected before ``s t``);
    ``(injection rate) x int_0^1 K ds/s = f_cr`` is the closure check.
    """
    s = reference.r0_over_R ** 2.5
    eps = reference.Pi_cr / reference.G
    K = s ** 1.2 * eps / eps[-1]
    order = np.argsort(s)
    s, K = s[order], K[order]
    keep = s > 0
    s, K = s[keep], K[keep]
    # power-law tail below the innermost tabulated shell
    p = np.log(K[1] / K[0]) / np.log(s[1] / s[0])
    cum = K[0] / p + np.concatenate(
        [[0.0], np.cumsum(0.5 * (K[1:] / s[1:] + K[:-1] / s[:-1]) * np.diff(s))]
    )
    return s, K, cum


def _kernel_prediction(t, inj, e_cr, kernel):
    """``E_cr(t_j)`` predicted from the run's own injection rate and the
    reference's adiabatic kernel: the CRs present at the first snapshot decay
    like a self-similar population, every later injection with ``K``."""
    s, K, cum = kernel
    pred = [e_cr[0]]
    for j in range(1, len(t)):
        tt = np.linspace(t[0], t[j], 2001)
        rate = np.interp(tt, t, inj)
        weight = np.interp(tt / t[j], s, K)
        old = e_cr[0] * np.interp(t[0] / t[j], s, cum) / cum[-1]
        pred.append(old + np.trapezoid(rate * weight, tt))
    return np.asarray(pred)


# N = 48 / 96 / 160: the first three slots of the dataviz reference palette
# (validated all-pairs incl. colour-blind), plus a marker each
RES_STYLE = {48: ("#2a78d6", "o"), 96: ("#eb6834", "s"), 160: ("#1baf7a", "^")}


def _plot(ref_ctrl, ref_dsa):
    """All resolutions found in RAW_DIR, every quantity as a ratio to the
    self-similar reference (1 = correct), read top-left to bottom-right:
    blast -> finder Mach -> finder dissipation -> injection -> E_cr -> summary."""
    runs = sorted(
        (int(p.stem.split("_N")[-1]), np.load(p)) for p in RAW_DIR.glob("b7_3_N*.npz")
    )
    kernel = _adiabatic_kernel(ref_dsa)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7})
    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.45)
    ax = {k: fig.add_subplot(gs[i, j]) for k, (i, j) in zip("abcdef", np.ndindex(2, 3))}
    t_lines = "abcde"
    for n, data in runs:
        color, marker = RES_STYLE.get(n, ("0.3", "x"))
        kw = dict(color=color, marker=marker, ms=3.5, lw=1.4, label=f"N = {n}$^3$")
        t = data["dsa_t"]
        # (a) blast radius: mean of front midpoint and pressure peak
        r_mid = 0.5 * (data["dsa_R_num"] + data["dsa_R_peak"]) / data["dsa_R_ref"]
        ax["a"].plot(t, r_mid, **kw)
        # (b) finder Mach number
        mach = data["dsa_mach_ratio"]
        ax["b"].plot(t, np.where(mach > 0, mach, np.nan), **kw)
        # (c) finder dissipation vs the simulated shock; dotted: area x Mach^3
        ax["c"].plot(t, data["dsa_D_over_meas"], **kw)
        # (the area split is meaningless while the surface is not yet a shell)
        prod = data["dsa_area_ratio_inj"] * mach**3
        ax["c"].plot(t, np.where(t >= LATE_TIME, prod, np.nan), color=color, ls=":", lw=1.6)
        # (d) injection rate vs reference
        ax["d"].plot(t, data["dsa_inj_t_over_E"] / ref_dsa.injection_rate, **kw)
        # (e) E_cr vs reference; dotted: kernel prediction from the run's injection
        pred = _kernel_prediction(t, data["dsa_inj_nominal"], data["dsa_E_cr"], kernel)
        ax["e"].plot(t, data["dsa_E_cr_over_E"] / ref_dsa.f_cr, **kw)
        ax["e"].plot(t, pred / E_EXPLOSION / ref_dsa.f_cr, color=color, ls=":", lw=1.6)
    # (f) late-time summary, grouped by quantity, one bar per resolution
    groups = [
        ("area", lambda d: _late_mean_npz(d, "dsa_area_ratio_inj")),
        ("(Mach ratio)$^3$", lambda d: _late_mean_npz(d, "dsa_mach_ratio", power=3)),
        ("dissipation (c)", lambda d: _late_mean_npz(d, "dsa_D_over_meas")),
        ("injection (d)", lambda d: _late_mean_npz(d, "dsa_inj_t_over_E") / ref_dsa.injection_rate),
        ("$E_{cr}$ at $t_{end}$ (e)", lambda d: float(d["dsa_E_cr_over_E"][-1]) / ref_dsa.f_cr),
    ]
    height = 0.8 / max(len(runs), 1)
    for i, (n, data) in enumerate(runs):
        color, _ = RES_STYLE.get(n, ("0.3", "x"))
        y = np.arange(len(groups)) + (i - 0.5 * (len(runs) - 1)) * height
        ax["f"].barh(y, [g(data) for _, g in groups], height=0.85 * height, color=color)
    ax["f"].set_yticks(np.arange(len(groups)), [g for g, _ in groups])
    ax["f"].invert_yaxis()
    ax["f"].axvline(1.0, color="k", lw=0.8)
    ax["f"].set(xlim=(0, 1.15), xlabel="ratio to reference",
                title="(f) summary: late-time means\n($t \\geq 0.03$), ratio to reference")

    titles = {
        "a": "(a) blast radius / reference\n(hydro: correct to 1-2%)",
        "b": "(b) finder Mach / true Mach\n(velocity jump, DB4)",
        "c": "(c) finder dissipation / exact\n"
             "dotted: area $\\times$ Mach$^3$",
        "d": "(d) CR injection rate / reference\n"
             f"(reference $t\\,\\dot E_{{inj}}/E$ = {ref_dsa.injection_rate:.4f})",
        "e": "(e) $E_{cr}$ / reference "
             f"({ref_dsa.f_cr:.3f} $E$)\ndotted: expected from (d) + ref. losses",
    }
    ylims = {"a": (0.95, 1.05), "b": (0.5, 1.1), "c": (0, 1.1), "d": (0, 1.1), "e": (0, 1.1)}
    for k in t_lines:
        a = ax[k]
        a.axhline(1.0, color="k", lw=0.8)
        a.axvspan(0.0, LATE_TIME, color="0.9", lw=0, zorder=0)
        a.set(title=titles[k], xlim=(0.0, 0.0735), ylim=ylims[k], xlabel="time $t$")
        a.set_xticks([0.0, 0.02, 0.04, 0.06])
    ax["a"].text(0.0015, 0.953, "early:\nnot yet\nself-similar", fontsize=6, va="bottom", color="0.35")
    for k, a in ax.items():
        a.grid(axis="x" if k == "f" else "y", color="0.88", lw=0.5)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    handles, labels = ax["a"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.text(0.5, 0.935, f"Sedov blast with DSA ($\\zeta$ = {DSA_EFFICIENCY:g}): every quantity divided "
             "by the self-similar two-fluid reference (black line = 1); grey: $t < 0.03$",
             ha="center", fontsize=7.5, color="0.25")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.835, bottom=0.08)
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_sedov_injection_rate.png", dpi=250)
    plt.close(fig)
    plt.rcdefaults()


def _late_mean_npz(data, key, power=1):
    late = data["dsa_t"] >= LATE_TIME
    return float(np.mean(data[key][late] ** power))



if __name__ == "__main__":
    test_cr_sedov_injection_rate()
