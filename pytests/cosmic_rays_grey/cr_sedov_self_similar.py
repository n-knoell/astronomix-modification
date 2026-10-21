"""
CR-DSA Sedov blast started from the self-similar two-fluid state, tightly
coupled CRs (Phase B test plan B7.2, with the shock slowdown B7.5).

Item 7 (``cr_sedov_taylor.py``) starts from a point explosion, so its ``E_cr``
remembers the unresolved early blast (plan Y4: 53% of the reference ``E_cr(t)``
is injected before ``0.1 t``; B7.3 found the late injection rate at 0.77 / 0.86
/ 0.92 of the reference but ``E_cr`` at 0.52 / 0.67 / 0.74, N = 48 / 96 / 160).
Here the blast starts from the self-similar solution itself
(``sedov_two_fluid.py``, B7.1, ``SedovTwoFluidSolution.cell_averages``) at
``R_0 = 0.15`` (7 cells at 48^3), so what remains is the scheme's own error:
injection, deposit location, and the CR scheme at an injecting shock.

Setup: the item-7 box, ambient gas and time ``T_END``; ``t_start = t_0`` with
``R(t_0) = R_0`` (the reference's clock, ``E = rho_0 = 1``), so the final state
is directly comparable with item 7 and B7.3 at the same time. The core density
is floored at ``DENSITY_FLOOR`` (the self-similar ``G`` is 1e-9 at
``lambda = 0.05``; the point-explosion runs have ``rho_min`` 0.015-0.14);
pressure and velocity are kept, so the initial energy is the reference's to
1e-3 (the shell's interpolation; measured, and the ratios below use the
grid's own energy). ``v_red = 0`` (``F_cr = 0``: the tightly coupled limit of
the reference, plan DB2 proposal), constant efficiency, code convention
``"gas_rh"``. Runs per resolution: the classic Sedov control (``zeta = 0``),
and ``zeta`` = 0.1 (item 7's value) and 0.211 (the reference ``E_cr/E`` of
item 8's KR13 strong blast, 0.2625), each with the grey-CR default (HLLC +
hybrid CR entropy) and the energy scheme (``cr_entropy = False``), plan Y7,
plus ``zeta`` = 0.1 with the opt-in density-consistent deposit
(``dsa_density_consistent_deposit``, DB9 = b).

Measured per snapshot (B7.3's ``_analyse``/``_derived``, plus):

- ``E_cr / E`` against the reference ``f_cr`` (constant in time), and the
  adiabatic-kernel prediction from the run's own injection rate (B7.3: a run
  holding more ``E_cr`` than its injection predicts has losses or gains the
  reference does not have).
- At the final state, relative to the simulated shock radius ``R_true`` (fit):
  the shell-window ratio ``X_cr = int P_cr dV / int P_th dV`` over
  ``lambda`` in [0.7, 0.9] (all gas there was shocked during the run:
  ``r_0/R`` >= 0.33 there, ``R_0 / R(T_END)`` = 0.38), and the mean ``P_cr``
  there in units of ``rho_0 V_true^2`` (both against the reference); the
  peak of the shell-averaged ``P_cr`` near the shock over the reference's
  post-shock ``P_cr`` (a deposit spike would exceed 1, plan 7.3 step 5; the
  smeared shell gives < 1); the L1 errors
  of ``rho``, ``P_tot`` and ``P_cr`` against the reference's cell averages at
  ``T_END``.
- B7.5: ``xi_dsa / xi_ctrl`` against ``xi_0(zeta) / xi_0(0)`` (0.981 / 0.963),
  and the exponent of ``R ~ t^alpha`` (0.4).

Gates. Hard: no NaN; the initial state carries the reference's energy (2e-3)
and ``E_cr/E`` (1e-2); total energy is conserved to 1e-3 (the blast stays
inside the open box); the control blast's ``xi`` is Sedov's within 3% and the
front and peak fits agree within 3% (B7.3's checks of the radius analysis).
Everything else is tracked in ``baselines.json`` per resolution
(``CR_NUM_CELLS``, metric suffix ``_N<cells>``). Plan targets (reported, not
gated): ``|E_cr/E_ref - 1| < 0.1`` and ``|X_cr/X_ref - 1| < 0.2`` at
N >= 96, monotone in N.

Findings (2026-10-21; numbers in ``baselines.json`` and ``PROGRESS_PHASEB.md``):
``E_cr/E`` stays within 5% / 2% / 0.5% of the reference (N = 48 / 96 / 160),
so item 7's deficit is the point-explosion start. The injection rate alone is
only 0.77 / 0.87 / 0.92 of the reference; the rest is a deposit surplus. CRs
deposited into the smeared, under-compressed shell (density peak ~0.5 / 0.65
of the RH post-shock density at N = 48 / 96) later lose less to expansion than
CRs injected at the RH density (+18 / +14 / +11% on the injected part; the
adiabatic losses of CRs already present are right to 3.5 / 1.6 / 0.9%). The
shock slowdown (B7.5) matches the reference to < 0.5% at every N.
With the density-consistent deposit (opt-in, ``z0.1_dcd``) the run holds what
its injection predicts to 1-1.5% (default +6 to +11%), so ``E_cr/E`` drops to
0.84 / 0.90 / 0.93: the injection deficit, no longer offset.

Raw data: ``/export/scratch/nknoell/phaseB/b7_2/`` (``CR_PHASEB_RAW``;
``CR_B72_REUSE=1`` reanalyses saved states); figure
``pics/07_sedov_taylor/cr_sedov_self_similar.png`` (all resolutions found
there).
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

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# reference
from astronomix.test_setups.reference_solutions.sedov_two_fluid import (
    solve_sedov_two_fluid,
)

# item-7 setup and the B7.3 snapshot analysis
from cr_sedov_taylor import DSA_MACH_MIN, E_EXPLOSION, GAMMA, P_AMBIENT, RHO_AMBIENT, T_END
from cr_sedov_injection_rate import (
    GAMMA_CR,
    LATE_TIME,
    RES_STYLE,
    _adiabatic_kernel,
    _analyse,
    _derived,
    _kernel_prediction,
    _shell_mean,
)
from cr_baselines import Baselines

NUM_CELLS = int(os.environ.get("CR_NUM_CELLS", 48))
R_0 = 0.15
DENSITY_FLOOR = 0.02
N_SUB = 4
NUM_SNAPSHOTS = 10
#: (name, zeta, cr_entropy, dsa_density_consistent_deposit): None = the grey-CR default
#: (hybrid), False = energy scheme
RUNS = (
    ("control", 0.0, None, False),
    ("z0.1", 0.1, None, False),
    ("z0.1_energy", 0.1, False, False),
    ("z0.211", 0.211, None, False),
    ("z0.211_energy", 0.211, False, False),
    ("z0.1_dcd", 0.1, None, True),
)
SHELL_WINDOW = (0.7, 0.9)
SPIKE_WINDOW = (0.85, 1.1)
RAW_DIR = Path(os.environ.get("CR_PHASEB_RAW", "/export/scratch/nknoell/phaseB/b7_2"))
B73_DIR = Path("/export/scratch/nknoell/phaseB/b7_3")
PICS_DIR = Path(__file__).resolve().parent / "pics" / "07_sedov_taylor"
REUSE_STATES = os.environ.get("CR_B72_REUSE", "0") not in ("", "0")


def _offsets(n):
    """Cell-centre offsets from the box centre, ``(3, n, n, n)`` (unit box)."""
    c = (np.arange(n) + 0.5) / n - 0.5
    return np.stack(np.meshgrid(c, c, c, indexing="ij"))


def _run(name, zeta, cr_entropy, reference, density_consistent=False):
    """One blast from the self-similar state at ``R_0`` to ``T_END``, with snapshots."""
    t_0 = (R_0 / reference.xi_0) ** 2.5 * np.sqrt(RHO_AMBIENT / E_EXPLOSION)
    times = np.linspace(t_0, T_END, NUM_SNAPSHOTS)
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        exact_end_time=True,
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_shock_acceleration=True, cr_entropy=cr_entropy,
            dsa_density_consistent_deposit=density_consistent,
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    dx = 1.0 / NUM_CELLS
    offsets = _offsets(NUM_CELLS)
    assert np.allclose(np.sqrt(np.sum(offsets**2, axis=0)), np.asarray(helper_data.r), atol=1e-4 * dx)  # helper r is float32
    rho, u, p_th, e_cr = reference.cell_averages(
        offsets, dx, t_0, E_EXPLOSION, RHO_AMBIENT, P_AMBIENT, n_sub=N_SUB,
        density_floor=DENSITY_FLOOR,
    )
    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.asarray(rho),
        velocity_x=jnp.asarray(u[0]),
        velocity_y=jnp.asarray(u[1]),
        velocity_z=jnp.asarray(u[2]),
        gas_pressure=jnp.asarray(p_th),
    )
    initial_state = initial_state.at[registered_variables.cosmic_ray_e_index].set(
        jnp.maximum(jnp.asarray(e_cr), CosmicRayGreyParams().minimum_e_cr)
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_start=float(t_0),
        t_end=T_END,
        gamma=GAMMA,
        snapshot_timepoints=jnp.array(times),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR,
            reduced_streaming_speed=0.0,
            dsa_efficiency=zeta,
            dsa_mach_min=DSA_MACH_MIN,
        ),
    )
    states_file = RAW_DIR / f"states_{name}_N{NUM_CELLS}.npz"
    if REUSE_STATES and states_file.exists():
        saved = np.load(states_file)
        states, times = saved["states"], saved["times"]
    else:
        snapshots = time_integration(initial_state, config, params, registered_variables)
        states, times = np.asarray(snapshots.states), np.asarray(snapshots.time_points)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(states_file, states=states, times=times)
    return dict(
        states=states, times=times, config=config, params=params,
        registered_variables=registered_variables, helper_data=helper_data, t_0=t_0,
    )


def _window_mean(lam, values, weights, window):
    sel = (lam >= window[0]) & (lam <= window[1])
    return float(np.sum(values[sel] * weights[sel]) / np.sum(weights[sel]))


def _final_profiles(run, d, reference):
    """Shell profiles, window ratios, spike and L1 errors of the final state."""
    rv = run["registered_variables"]
    state = run["states"][-1]
    dx = 1.0 / NUM_CELLS
    r_flat = np.asarray(run["helper_data"].r).reshape(-1)
    rho = state[rv.density_index]
    p_th = state[rv.pressure_index]
    p_cr = (GAMMA_CR - 1.0) * state[rv.cosmic_ray_e_index]
    centres, rho_m, counts = _shell_mean(r_flat, rho.reshape(-1), dx)
    _, pth_m, _ = _shell_mean(r_flat, p_th.reshape(-1), dx)
    _, pcr_m, _ = _shell_mean(r_flat, p_cr.reshape(-1), dx)
    R_true, V_true = float(d["R_true"][-1]), float(d["V_true"][-1])
    lam = centres / R_true
    shell_vol = counts.astype(float)
    out = dict(lam=lam, rho_m=rho_m, pth_m=pth_m, pcr_m=pcr_m, counts=counts)

    # reference over the same window (lambda relative to its own shock)
    ref_l = reference.lam
    w_ref = ref_l**2
    sel_ref = (ref_l >= SHELL_WINDOW[0]) & (ref_l <= SHELL_WINDOW[1])
    if reference.f_cr > 0:
        x_ref = np.trapezoid((reference.Pi_cr * w_ref)[sel_ref], ref_l[sel_ref]) / np.trapezoid(
            (reference.Pi_th * w_ref)[sel_ref], ref_l[sel_ref])
        pcr_ref = np.trapezoid((reference.Pi_cr * w_ref)[sel_ref], ref_l[sel_ref]) / np.trapezoid(
            w_ref[sel_ref], ref_l[sel_ref])
        x_sim = _window_mean(lam, pcr_m, shell_vol, SHELL_WINDOW) / _window_mean(
            lam, pth_m, shell_vol, SHELL_WINDOW)
        pcr_sim = _window_mean(lam, pcr_m, shell_vol, SHELL_WINDOW) / (RHO_AMBIENT * V_true**2)
        spike_sel = (lam >= SPIKE_WINDOW[0]) & (lam <= SPIKE_WINDOW[1])
        x_shell = np.where(pth_m > 0, pcr_m / np.maximum(pth_m, 1e-300), 0.0)
        out.update(
            xcr_shell_ratio=x_sim / x_ref,
            pcr_shell_ratio=pcr_sim / pcr_ref,
            pcr_peak_ratio=float(np.max(pcr_m[spike_sel])) / (RHO_AMBIENT * V_true**2 * reference.Pi_cr[-1]),
            x_shell=x_shell,
        )

    # L1 against the reference's cell averages at the final time
    ref_rho, _, ref_pth, ref_ecr = reference.cell_averages(
        _offsets(NUM_CELLS), dx, float(run["times"][-1]), E_EXPLOSION, RHO_AMBIENT, P_AMBIENT,
        n_sub=N_SUB,
    )
    ref_pcr = (GAMMA_CR - 1.0) * ref_ecr
    out["L1_rho"] = float(np.sum(np.abs(rho - ref_rho)) / np.sum(np.abs(ref_rho - RHO_AMBIENT)))
    out["L1_p_tot"] = float(np.sum(np.abs(p_th + p_cr - ref_pth - ref_pcr)) / np.sum(ref_pth + ref_pcr))
    if reference.f_cr > 0:
        out["L1_p_cr"] = float(np.sum(np.abs(p_cr - ref_pcr)) / np.sum(ref_pcr))
    return out


def _alpha(d):
    """Exponent of ``R ~ t^alpha`` (front/peak mean, ``t >= LATE_TIME``)."""
    late = d["t"] >= LATE_TIME
    r_mid = 0.5 * (d["R_num"] + d["R_peak"])
    return float(np.polyfit(np.log(d["t"][late]), np.log(r_mid[late]), 1)[0])


def test_cr_sedov_self_similar():
    """B7.2 + B7.5: the self-similar blast keeps its CR partition, profiles and speed."""
    e_ambient = P_AMBIENT / (GAMMA - 1.0)
    results, profiles, refs = {}, {}, {}
    for name, zeta, cr_entropy, density_consistent in RUNS:
        reference = solve_sedov_two_fluid(zeta, convention="gas_rh")
        refs[name] = reference
        run = _run(name, zeta, cr_entropy, reference, density_consistent)
        assert not np.any(np.isnan(run["states"])), f"{name}: NaNs"
        assert abs(run["times"][0] - run["t_0"]) < 1e-6 and abs(run["times"][-1] - T_END) < 1e-6, (
            f"{name}: snapshot times {run['times']}")
        res = _analyse(run, reference, zeta)
        d = _derived(res, reference)
        # the grid's own blast energy (the ambient subtracted) and the kernel prediction
        e_grid = res["E_total"][0] - e_ambient
        d["E_grid"] = e_grid
        d["E_cr_over_ref"] = (res["E_cr"] / e_grid / reference.f_cr if zeta > 0
                              else np.full_like(res["E_cr"], np.nan))
        d["conservation"] = np.abs(res["E_total"] / res["E_total"][0] - 1.0)
        d["alpha"] = _alpha(d)
        d["inj_realized_over_nominal"] = res["inj_realized"] / np.maximum(res["inj_nominal"], 1e-300)
        if zeta > 0:
            kernel = _adiabatic_kernel(reference)
            d["E_cr_pred_over_ref"] = (_kernel_prediction(d["t"], res["inj_nominal"], res["E_cr"], kernel)
                                       / e_grid / reference.f_cr)
        profiles[name] = _final_profiles(run, d, reference)
        del run
        results[name] = d
        print(f"{name}: E_grid {e_grid:.5f}, E_cr/E(t_0) / ref {d['E_cr_over_ref'][0]:.4f}, "
              f"conservation {d['conservation'].max():.2e}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RAW_DIR / f"b7_2_N{NUM_CELLS}.npz",
        **{f"{name}_{k}": np.asarray(v) for name, d in results.items() for k, v in d.items()},
        **{f"{name}_prof_{k}": np.asarray(v) for name, p in profiles.items() for k, v in p.items()},
    )

    # report
    ctrl = results["control"]
    header = "   t      R/Rref  M_fin/M  inj/ref  E_cr/E/ref  pred/ref"
    for name, d in results.items():
        print(f"{name}: xi front {float(d['xi_front']):.4f} peak {float(d['xi_peak']):.4f} "
              f"(fit {float(d['xi_fit']):.4f}, ref {refs[name].xi_0:.4f}), alpha {d['alpha']:.4f}")
        if refs[name].f_cr > 0:
            print(header)
            for i in range(len(d["t"])):
                print(f"  {d['t'][i]:.4f}  {d['R_over_ref'][i]:.4f}  {d['mach_ratio'][i]:.4f}   "
                      f"{d['inj_t_over_E'][i] / refs[name].injection_rate:.4f}   "
                      f"{d['E_cr_over_ref'][i]:.4f}      {d['E_cr_pred_over_ref'][i]:.4f}")
        p = profiles[name]
        print("  final: " + ", ".join(f"{k} {p[k]:.4f}" for k in (
            "xcr_shell_ratio", "pcr_shell_ratio", "pcr_peak_ratio", "L1_rho", "L1_p_tot", "L1_p_cr") if k in p))

    # hard gates
    for name, d in results.items():
        reference = refs[name]
        assert abs(d["E_grid"] / E_EXPLOSION - 1.0) < 2e-3, f"{name}: initial energy {d['E_grid']:.5f}"
        if reference.f_cr > 0:
            assert abs(d["E_cr_over_ref"][0] - 1.0) < 1e-2, (
                f"{name}: initial E_cr/E is {d['E_cr_over_ref'][0]:.4f} of the reference")
        assert d["conservation"].max() < 1e-3, f"{name}: total energy error {d['conservation'].max():.2e}"
        assert abs(float(d["xi_front"]) - float(d["xi_peak"])) < 0.03 * reference.xi_0, (
            f"{name}: front and peak fits disagree ({float(d['xi_front']):.4f} vs {float(d['xi_peak']):.4f})")
    assert abs(float(ctrl["xi_fit"]) / refs["control"].xi_0 - 1.0) < 0.03, (
        f"control xi {float(ctrl['xi_fit']):.4f} vs Sedov {refs['control'].xi_0:.4f}")

    # tracked metrics
    baselines = Baselines("cr_sedov_self_similar.py::test_cr_sedov_self_similar")
    sfx = f"_N{NUM_CELLS}"
    baselines.close("ctrl_xi_over_ref" + sfx, float(ctrl["xi_fit"]) / refs["control"].xi_0, rel=0.01)
    baselines.close("ctrl_alpha" + sfx, ctrl["alpha"], rel=0.01)
    baselines.close("ctrl_L1_rho" + sfx, profiles["control"]["L1_rho"], rel=0.1)
    for name, d in results.items():
        if name == "control":
            continue
        reference, p = refs[name], profiles[name]
        late = d["t"] >= LATE_TIME
        metrics = {
            "E_cr_over_ref_end": float(d["E_cr_over_ref"][-1]),
            "E_cr_run_over_pred_end": float(d["E_cr_over_ref"][-1] / d["E_cr_pred_over_ref"][-1]),
            "inj_over_ref": float(np.mean(d["inj_t_over_E"][late])) / reference.injection_rate,
            "mach_ratio": float(np.mean(d["mach_ratio"][late])),
            "xi_over_ref": float(d["xi_fit"]) / reference.xi_0,
            # B7.5: shock slowdown relative to the control, over the reference's
            "slowdown_over_ref": (float(d["xi_fit"]) / float(ctrl["xi_fit"]))
                                 / (reference.xi_0 / refs["control"].xi_0),
            "alpha": d["alpha"],
            "xcr_shell_ratio": p["xcr_shell_ratio"],
            "pcr_shell_ratio": p["pcr_shell_ratio"],
            "pcr_peak_ratio": p["pcr_peak_ratio"],
            "L1_rho": p["L1_rho"],
            "L1_p_tot": p["L1_p_tot"],
            "L1_p_cr": p["L1_p_cr"],
        }
        if name.endswith("_dcd"):
            # share of the nominal injection the density factor deposits
            metrics["inj_realized_over_nominal"] = float(np.mean(d["inj_realized_over_nominal"][late]))
        for k, v in metrics.items():
            baselines.close(f"{name}_{k}{sfx}", v, rel=0.1 if k.startswith("L1") else 0.03)
    _plot(refs)
    baselines.check()


#: line style per run in the figure
RUN_STYLE = {"z0.1": "-", "z0.1_energy": "--", "z0.211": "-", "z0.211_energy": "--", "z0.1_dcd": "-."}


def _plot(refs):
    """All resolutions in RAW_DIR. Top: zeta = 0.1 / 0.211 E_cr against the reference over
    time (with B7.3's point explosion at zeta = 0.1), the injection rate. Bottom: final
    shell profiles (X_cr, P_cr) at zeta = 0.1 and a summary of the late-time ratios."""
    runs = sorted((int(p.stem.split("_N")[-1]), np.load(p)) for p in RAW_DIR.glob("b7_2_N*.npz"))
    b73 = {int(p.stem.split("_N")[-1]): np.load(p) for p in B73_DIR.glob("b7_3_N*.npz")}
    ref = refs["z0.1"]
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 6.5})
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.5)
    ax = {k: fig.add_subplot(gs[i, j]) for k, (i, j) in zip("abcdef", np.ndindex(2, 3))}
    for n, data in runs:
        color, marker = RES_STYLE.get(n, ("0.3", "x"))
        for name, ls in RUN_STYLE.items():
            if f"{name}_t" not in data:
                continue
            key = "b" if name.startswith("z0.211") else "a"
            t = data[f"{name}_t"]
            ax[key].plot(t, data[f"{name}_E_cr_over_ref"], color=color, ls=ls, lw=1.3,
                         marker=marker if ls == "-" else None, ms=3)
            if ls == "-":
                ax[key].plot(t, data[f"{name}_E_cr_pred_over_ref"], color=color, ls=":", lw=1.2)
        if n in b73:
            ax["a"].plot(b73[n]["dsa_t"], b73[n]["dsa_E_cr_over_E"] / ref.f_cr, color=color,
                         lw=0.8, alpha=0.45)
        t = data["z0.1_t"]
        ax["c"].plot(t, data["z0.1_inj_t_over_E"] / ref.injection_rate, color=color, marker=marker,
                     ms=3, lw=1.3, label=f"N = {n}$^3$")
        ax["c"].plot(t, data["z0.1_energy_inj_t_over_E"] / ref.injection_rate, color=color, ls="--", lw=1.1)
        lam = data["z0.1_prof_lam"]
        sel = (lam > 0.3) & (lam < 1.15)
        ax["d"].plot(lam[sel], data["z0.1_prof_x_shell"][sel] / ref.x_cr_0, color=color, lw=1.2)
        V2 = RHO_AMBIENT * data["z0.1_V_true"][-1] ** 2
        ax["e"].plot(lam[sel], data["z0.1_prof_pcr_m"][sel] / V2, color=color, lw=1.2)
    # references
    rl = ref.lam
    xr = np.where(ref.Pi_th > 0, ref.Pi_cr / ref.Pi_th, 0.0) / ref.x_cr_0
    ax["d"].plot(rl[rl > 0.3], xr[rl > 0.3], color="k", lw=0.9, label="reference")
    ax["e"].plot(rl[rl > 0.3], ref.Pi_cr[rl > 0.3], color="k", lw=0.9, label="reference")
    for k in "de":
        ax[k].axvspan(*SHELL_WINDOW, color="0.92", lw=0, zorder=0)
        ax[k].legend(loc="upper left", frameon=False)
    # summary: zeta = 0.1 default scheme, per resolution
    groups = [
        ("$E_{cr}$ end", lambda d: float(d["z0.1_E_cr_over_ref"][-1])),
        ("injection", lambda d: float(np.mean(d["z0.1_inj_t_over_E"][d["z0.1_t"] >= LATE_TIME]))
         / ref.injection_rate),
        ("$X_{cr}$ shell", lambda d: float(d["z0.1_prof_xcr_shell_ratio"])),
        ("$E_{cr}$/kernel", lambda d: float(d["z0.1_E_cr_over_ref"][-1] / d["z0.1_E_cr_pred_over_ref"][-1])),
        ("slowdown", lambda d: (float(d["z0.1_xi_fit"]) / float(d["control_xi_fit"]))
         / (ref.xi_0 / refs["control"].xi_0)),
        ("$E_{cr}$ end (d.c.)", lambda d: float(d["z0.1_dcd_E_cr_over_ref"][-1])
         if "z0.1_dcd_t" in d else np.nan),
        ("$E_{cr}$/kernel (d.c.)", lambda d: float(d["z0.1_dcd_E_cr_over_ref"][-1]
                                                       / d["z0.1_dcd_E_cr_pred_over_ref"][-1])
         if "z0.1_dcd_t" in d else np.nan),
    ]
    height = 0.8 / max(len(runs), 1)
    for i, (n, data) in enumerate(runs):
        color, _ = RES_STYLE.get(n, ("0.3", "x"))
        y = np.arange(len(groups)) + (i - 0.5 * (len(runs) - 1)) * height
        ax["f"].barh(y, [g(data) for _, g in groups], height=0.85 * height, color=color)
    ax["f"].set_yticks(np.arange(len(groups)), [g for g, _ in groups])
    ax["f"].invert_yaxis()
    ax["f"].axvline(1.0, color="k", lw=0.8)
    ax["f"].set(xlim=(0.6, 1.25), xlabel="ratio to reference",
                title="(f) $\\zeta$ = 0.1, late / final ratios\n(d.c.: dens.-consistent deposit)")
    titles = {
        "a": "(a) $E_{cr}/E$ / reference\n$\\zeta$ = 0.1",
        "b": "(b) $E_{cr}/E$ / reference\n$\\zeta$ = 0.211",
        "c": "(c) injection rate / reference\n$\\zeta$ = 0.1",
        "d": "(d) $X_{cr} = P_{cr}/P_{th}$ / post-shock $X_{cr,0}$\nshell means at $t_{end}$, $\\zeta$ = 0.1",
        "e": "(e) $P_{cr} / \\rho_0 V_s^2$ at $t_{end}$, $\\zeta$ = 0.1\n(grey: shell window)",
    }
    for k in "abc":
        ax[k].axhline(1.0, color="k", lw=0.8)
        ax[k].set(xlabel="time $t$", xlim=(0.0, 0.0735), ylim=(0.4, 1.15))
        ax[k].set_xticks([0.0, 0.02, 0.04, 0.06])
    ax["d"].set(xlabel="$\\lambda = r / R_s$", xlim=(0.3, 1.15), ylim=(0, 2.5))
    ax["e"].set(xlabel="$\\lambda = r / R_s$", xlim=(0.3, 1.15))
    for k, a in ax.items():
        if k in titles:
            a.set_title(titles[k])
        a.grid(axis="x" if k == "f" else "y", color="0.88", lw=0.5)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    handles, labels = ax["c"].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(runs), frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.text(0.5, 0.905, "solid: default scheme (HLLC + hybrid CR entropy), dashed: energy scheme, "
             "dash-dot: density-consistent deposit (opt-in);\n"
             "dotted: expected from the run's own injection + reference losses; "
             "faint in (a): point explosion (item 7 / B7.3).\n"
             "Self-similar start at $R_0$ = 0.15, $v_{red}$ = 0, black line = reference",
             ha="center", fontsize=6.5, color="0.25")
    fig.subplots_adjust(left=0.08, right=0.985, top=0.78, bottom=0.08)
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_sedov_self_similar.png", dpi=250)
    plt.close(fig)
    plt.rcdefaults()


if __name__ == "__main__":
    test_cr_sedov_self_similar()
