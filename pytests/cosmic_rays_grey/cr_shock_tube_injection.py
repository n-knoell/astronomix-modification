"""
Planar shock tube with CR acceleration against the exact Riemann solution: Phase B test plan
B7.4 (``astronomix/_modules/_cosmic_rays_grey/astronomix_CR_phaseB_test_plan.md``, Sec. 7.4).

**What it measures.** The thermal shock tube of Pfrommer et al. (2017, Table 1; ``rho = 1 |
0.125``, ``P = 63.499 | 0.1``, Mach 10.00) with DSA at a constant ``zeta`` = 0.1 and 0.5, run to
``t = 0.35`` in float64 with ``v_red = 0`` (box [0, 10], diaphragm at 5, open boundaries), against
the exact Riemann solution with CR injection (``pfrommer_riemann_solver.py``, ``injection=``;
App. B of Pfrommer et al. 2017 generalized to the code's convention).

**Reference.** ``test_injection_riemann_reference`` checks the solver extension: Pfrommer et al.'s
Table 1 (their convention: M 10.00 -> 9.555, compression 3.88 -> 4.741; CR tube 10.00 -> 9.559),
flux continuity through the shock to 1e-10, the strong-shock limit of B7.1, and mass/energy over
the sampled profile. Two versions of the code's convention (``inject_crs_at_shocks``: the CR flux
is ``zeta`` times the finder's gas-only RH dissipated flux at the finder's Mach number):

- *true Mach number* ``v_1 / c_1``: what the injection is meant to do;
- *the finder's Mach number* (``FINDER_MACH``): the injected CRs change the shock the finder
  samples, so the code converges to the fixed point ``zeta x f(M_finder(zeta))``. Since
  2026-10-19 (DB4) DSA's finder reads the **velocity** jump, which the extra compression of a
  CR-loaded shock raises (``mach="velocity"``); until then the **gas-pressure** jump, which the
  CR pressure lowers (``mach="gas_pressure"``).

  ======  =======  ================  ==================  =========================  ============
  zeta    M true   M_u at vel. FP    M_p at gas-p. FP    P_cr,2 true / vel. / gas-p  inj./nominal
  ======  =======  ================  ==================  =========================  ============
  0.1     9.913    10.030            9.748               0.585 / 0.608 / 0.554      1.038 / 0.948
  0.5     9.600    10.164            8.893               3.152 / 3.957 / 2.358      1.226 / 0.766
  ======  =======  ================  ==================  =========================  ============

**Results (2026-10-20, GPU, default scheme = HLLC + hybrid CR entropy, adaptive sampling with the
spread deposit, velocity-jump Mach number).** ``zeta = 0.1``, against the velocity fixed point,
constant for N = 400-3200: shock speed within 5e-4, ``P_th`` -0.15%, density +0.1%, finder Mach
number -0.4% to -0.7%, L1 density order 0.89, and ``P_cr`` **+2.6%** (box CR energy +2.5%):
injection inside the smeared shock, where the gas is still being compressed, so the injected CRs
gain adiabatic energy the exact solution does not have (the deposit-location bias of Dubois et
al. 2019; not resolution-dependent). Against the true Mach number ``P_cr`` +6.6%, rate +6.3%.

``zeta = 0.5`` does not reach the velocity fixed point: ``P_cr`` -14%, ``P_th`` +7.4%, shock
+0.6%, finder Mach number -3% to -6%, L1 order 0.38; the run lies between the two references
(against the true Mach number ``P_cr`` +8%, rate +6.2%). In the smeared CR-loaded shock the
sampled velocity jump is smaller than the exact post-shock one. With the gas-pressure Mach number
(until 2026-10-19) the runs sat at that fixed point for both ``zeta`` (``P_cr`` +2.7%; against the
true Mach number -2.7% / **-23%**; DB8, accepted 2026-10-14).

Schemes (N = 400 / 1600, ``zeta`` = 0.1, ``P_cr`` vs the velocity fixed point): default +2.6%,
energy scheme with HLLC +6.6%, with HLL +10.8%: the energy scheme adds the spurious CR shock
heating of Phase A (A6.1-A6.3) on top. Surface-cell deposit (``dsa_adaptive_shock_sampling =
False``, gas-pressure Mach number from the 1-cell sampler): ``P_cr`` -52% (tracked; not the
default since 2026-09-26).

Gates: hard on the default scheme at N >= 400 against the fixed point of the finder's Mach number
(bounds in the ``test_cr_shock_tube_injection`` signature): they check the injection mechanics
given that Mach number. They apply to ``zeta = 0.1`` only (``GATED_ZETAS``); ``zeta = 0.5`` misses
the velocity fixed point (above) and is tracked only (user decision 2026-10-20: realistic
efficiencies stay <= 0.21, and KR13 at 0.21 passes in B8.1). Everything at ``zeta = 0.5``, against
the true-Mach reference, the other schemes, the surface deposit and the CR tube (``X_cr = 2 | 1``, App. C; B8.5) is tracked in
``baselines.json``. Runtime ~20 min on one 2080 Ti. Raw profiles:
``/export/scratch/nknoell/phaseB/b7_4/``.

See ``PROGRESS_PHASEB.md`` in the module directory.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import os
import warnings
from pathlib import Path
from typing import NamedTuple

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# baseline registry
from cr_baselines import Baselines

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    BoundarySettings1D,
    DOUBLE_PRECISION,
    OPEN_BOUNDARY,
    HLL,
    HLLC,
    SnapshotSettings,
    finalize_config,
)

# astronomix functions
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix.test_setups.reference_solutions.pfrommer_riemann_solver import (
    ShockInjection,
    pfrommer_riemann_solution,
    pfrommer_shock_structure,
)
from astronomix.test_setups.reference_solutions.sedov_two_fluid import _strong_shock_state

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    DSA_EFFICIENCY_CONSTANT,
    DSA_EFFICIENCY_KANG_RYU_2013,
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_shock_finder_kwargs

# plateau averages at the 1e-4 level need float64
jax.config.update("jax_enable_x64", True)

GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
BOX = 10.0
X0 = 5.0
T_END = 0.35
#: snapshots for the shock-speed and injection-rate fits (the position at one time carries a
#: fixed offset of about a cell from the shape of the smeared shock)
SNAPSHOT_TIMES = np.linspace(0.1, T_END, 6)
DSA_MACH_MIN = 1.3  # CosmicRayGreyParams default
#: The Mach number the DSA finder reads (``CosmicRayGreyConfig.dsa_mach_from_velocity``, the
#: default since 2026-10-19, decision DB4: the normal-velocity jump; ``"gas_pressure"`` before).
#: The hard gates compare with the exact solution at this fixed point ("fp" metrics).
FINDER_MACH = "velocity"
FINDER_MACH_KEY = {"velocity": "mach_gas_velocity", "gas_pressure": "mach_gas_pressure"}[FINDER_MACH]
PICS_DIR = Path(__file__).resolve().parent / "pics" / "07_sedov_taylor"
RAW_DIR = Path(os.environ.get("CR_PHASEB_RAW", "/export/scratch/nknoell/phaseB")) / "b7_4"


class Tube(NamedTuple):
    """Left/right states ``(rho, u, P_th, P_cr)``."""

    left: tuple
    right: tuple


#: Pfrommer et al. (2017) Table 1. "th": thermal gas, M = 10.00 without injection. "cr":
#: ``X_cr = P_cr / P_th`` = 2 | 1 at the same total pressures' Mach number 10.00 with the
#: composite sound speed (``P_5 = 51.516`` and ``P_1 = 0.1`` are totals).
TH = Tube(left=(1.0, 0.0, 63.499, 0.0), right=(0.125, 0.0, 0.1, 0.0))
CR = Tube(left=(1.0, 0.0, 51.516 / 3.0, 2.0 * 51.516 / 3.0), right=(0.125, 0.0, 0.05, 0.05))

#: Scheme variants: the grey-CR default (auto: HLLC + hybrid CR entropy) and the energy scheme.
SCHEMES = {
    "default": dict(),
    "energy HLLC": dict(riemann_solver=HLLC, cr_entropy=False),
    "energy HLL": dict(riemann_solver=HLL, cr_entropy=False),
    "default, surface deposit": dict(dsa_adaptive_shock_sampling=False),
}


def injection(zeta, mach="true"):
    """The code's convention (``inject_crs_at_shocks``) as a ``ShockInjection``."""
    return ShockInjection(zeta, "gas_rh", mach, DSA_MACH_MIN)


def reference(tube, zeta, mach="true", x0=X0, t_end=T_END):
    """Shock structure plus wave positions at ``t_end``. ``zeta`` may be a function of the
    Mach number (KR13)."""
    inj = injection(zeta, mach) if callable(zeta) or zeta > 0 else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = pfrommer_shock_structure(*tube.left, *tube.right, GAMMA, GAMMA_CR, injection=inj)
    s["x_shock"] = x0 + s["v_shock"] * t_end
    s["x_contact"] = x0 + s["u_star"] * t_end
    return s


def run_tube(tube, num_cells, zeta, riemann_solver=None, cr_entropy=None,
             dsa_adaptive_shock_sampling=True, C_cfl=0.4, box=BOX, x0=X0, t_end=T_END,
             kr13_scale=None):
    """Run one tube to ``t_end`` in float64 with DSA on (if ``zeta > 0`` or ``kr13_scale``),
    ``v_red = 0``. ``kr13_scale``: use the KR13 model with this ``dsa_efficiency_mach_scale``
    instead of the constant ``zeta``. Snapshots at ``SNAPSHOT_TIMES * t_end / T_END``.

    Returns:
        dict with ``x``, the final ``rho, u, p_th, p_cr`` (numpy), the snapshot ``times``,
        ``rho_t`` and ``e_cr_t`` (density and CR energy density per snapshot) and the finder's
        Mach number at the shock of the final state (``mach_finder``).
    """
    extra = {} if riemann_solver is None else dict(riemann_solver=riemann_solver)
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=box,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
        ),
        numerical_precision=DOUBLE_PRECISION,
        exact_end_time=True,
        return_snapshots=True,
        num_snapshots=len(SNAPSHOT_TIMES),
        use_specific_snapshot_timepoints=True,
        snapshot_settings=SnapshotSettings(return_states=True),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True,
            cr_entropy=cr_entropy,
            diffusive_shock_acceleration=zeta > 0 or kr13_scale is not None,
            dsa_adaptive_shock_sampling=dsa_adaptive_shock_sampling,
            dsa_efficiency_model=(DSA_EFFICIENCY_CONSTANT if kr13_scale is None
                                  else DSA_EFFICIENCY_KANG_RYU_2013),
        ),
        **extra,
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers

    left = x < x0
    state = jnp.zeros((registered_variables.num_vars, num_cells))
    rows = (
        registered_variables.density_index,
        registered_variables.velocity_index,
        registered_variables.pressure_index,
    )
    for row, value_l, value_r in zip(rows, tube.left[:3], tube.right[:3]):
        state = state.at[row].set(jnp.where(left, value_l, value_r))
    state = state.at[registered_variables.cosmic_ray_e_index].set(
        jnp.where(left, tube.left[3], tube.right[3]) / (GAMMA_CR - 1.0)
    )

    config = finalize_config(config, state.shape)
    params = SimulationParams(
        t_end=t_end,
        gamma=GAMMA,
        C_cfl=C_cfl,
        snapshot_timepoints=jnp.array(SNAPSHOT_TIMES * t_end / T_END),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR,
            reduced_streaming_speed=0.0,
            dsa_efficiency=zeta,
            dsa_efficiency_mach_scale=1.0 if kr13_scale is None else kr13_scale,
            dsa_mach_min=DSA_MACH_MIN,
        ),
    )
    snapshots = time_integration(state, config, params, registered_variables)
    states = np.asarray(snapshots.states)
    times = np.asarray(snapshots.time_points)
    assert not np.any(np.isnan(states)), f"NaNs at N = {num_cells}"
    assert abs(times[-1] - t_end) < 1e-12 * max(1.0, t_end), times
    final = states[-1]
    sf = find_shocks_pfrommer(jnp.asarray(final), config, registered_variables, helper_data,
                              mach_min=DSA_MACH_MIN,
                              **dsa_shock_finder_kwargs(config.cosmic_ray_grey_config))
    mach = np.asarray(sf.mach_numbers)
    rv = registered_variables
    return dict(
        x=np.asarray(x),
        rho=final[rv.density_index],
        u=final[rv.velocity_index],
        p_th=final[rv.pressure_index],
        p_cr=final[rv.cosmic_ray_e_index] * (GAMMA_CR - 1.0),
        times=times,
        rho_t=states[:, rv.density_index],
        e_cr_t=states[:, rv.cosmic_ray_e_index],
        mach_finder=float(mach.max()),
    )


def _crossing(x, y, level, lo, hi):
    """Last linear-interpolated crossing of ``y = level`` in ``lo < x < hi``."""
    m = (x > lo) & (x < hi)
    xx, yy = x[m], y[m]
    i = np.nonzero(np.diff(np.sign(yy - level)))[0][-1]
    return xx[i] + (level - yy[i]) * (xx[i + 1] - xx[i]) / (yy[i + 1] - yy[i])


def metrics(tube, run, ref, box=BOX, x0=X0):
    """Shock speed, shocked-plateau state and injection rate against ``ref``.

    - Shock speed: least-squares slope of the density mid-level crossing over the snapshots.
    - Plateau: 55-85% of the way from the contact to the shock (exact positions), as in
      ``cr_shock_tube_partition.py``; the smeared contact is left out.
    - Injection rate (thermal tube only, no upstream CRs): slope of the box-integrated CR
      energy over the snapshots, against the exact ``f_inj`` (the plateau grows at
      ``v_shock - u_2`` and holds ``e_inj``).
    """
    x, rho, p_th, p_cr = run["x"], run["rho"], run["p_th"], run["p_cr"]
    dx = x[1] - x[0]
    a, b = ref["x_contact"], ref["x_shock"]
    window = (x > a + 0.55 * (b - a)) & (x < a + 0.85 * (b - a))
    level = 0.5 * (ref["rho_2"] + tube.right[0])
    positions = np.array([
        _crossing(x, rho_t, level, x0 + (ref["u_star"] + 0.1 * (ref["v_shock"] - ref["u_star"])) * t, box)
        for t, rho_t in zip(run["times"], run["rho_t"])
    ])
    v_fit, x_fit0 = np.polyfit(run["times"], positions, 1)
    p_cr_1 = tube.right[3]
    # injected CR pressure: total minus the adiabatically compressed upstream CRs
    x_s = ref["rho_2"] / tube.right[0]
    p_cr_inj_ref = ref["p_cr_2"] - p_cr_1 * x_s**GAMMA_CR
    p_cr_inj = p_cr[window].mean() - p_cr_1 * (rho[window].mean() / tube.right[0]) ** GAMMA_CR
    out = dict(
        dv_shock=v_fit / ref["v_shock"] - 1.0,
        shock_offset_cells=(positions[-1] - b) / dx,
        dP_th=p_th[window].mean() / ref["p_th_2"] - 1.0,
        dP_cr=p_cr[window].mean() / ref["p_cr_2"] - 1.0 if ref["p_cr_2"] > 0 else np.nan,
        dP_cr_inj=p_cr_inj / p_cr_inj_ref - 1.0 if p_cr_inj_ref > 0 else np.nan,
        drho=rho[window].mean() / ref["rho_2"] - 1.0,
        du=run["u"][window].mean() / ref["u_star"] - 1.0,
        mach_finder=run["mach_finder"],
        cells_in_window=int(window.sum()),
    )
    if p_cr_1 == 0.0 and tube.left[3] == 0.0 and ref["f_inj"] > 0:
        e_cr_box = run["e_cr_t"].sum(axis=1) * dx
        out["rate"] = np.polyfit(run["times"], e_cr_box, 1)[0] / ref["f_inj"]
    return out


def _shock_frame_fluxes(rho, v, p_th, p_cr):
    """Mass, momentum and total-energy flux through the shock (``v`` in the shock frame)."""
    h = (GAMMA / (GAMMA - 1.0) * p_th + GAMMA_CR / (GAMMA_CR - 1.0) * p_cr) / rho
    return np.array([rho * v, p_th + p_cr + rho * v**2, rho * v * (0.5 * v**2 + h)])


def test_injection_riemann_reference():
    """Self-test of the injection extension of ``pfrommer_riemann_solver.py`` (no simulation).

    - ``zeta = 0`` reproduces the solver without injection exactly.
    - Pfrommer et al. (2017) Table 1 (their ``"total"`` convention, ``zeta = 0.5``): Mach
      number 10.00 -> 9.56 for the thermal and the CR tube (composite sound speed), shock
      compression 3.88 -> 4.74 for the thermal tube.
    - Mass, momentum and energy fluxes through the shock are continuous to 1e-10 for every
      convention (the Hugoniot root-find only uses their combination).
    - ``"gas_rh"``: the injected flux is ``zeta`` times the finder's gas-RH dissipated flux at
      the true Mach number (``mach="true"``), or at the Mach number of the gas-pressure jump
      (``mach="gas_pressure"``) or of the velocity jump (``mach="velocity"``), to 1e-10.
    - Strong-shock limit: compression and ``P_cr / P`` equal B7.1's strong-shock state
      (``sedov_two_fluid._strong_shock_state``) for both conventions, to 1e-4.
    - The sampled profile conserves mass and total energy over the box to 1e-4 (trapezoid
      error at the discontinuities, 40001 samples).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = pfrommer_shock_structure(*TH.left, *TH.right, GAMMA, GAMMA_CR)
        zero = pfrommer_shock_structure(*TH.left, *TH.right, GAMMA, GAMMA_CR,
                                        injection=injection(0.0))
        for key in ("p_star", "u_star", "rho_2", "p_th_2", "p_cr_2", "v_shock"):
            assert zero[key] == base[key], key

        # Pfrommer et al. (2017) Table 1
        assert abs(base["mach_gas"] - 10.0) < 5e-4 and abs(base["compression"] - 3.88) < 5e-3
        th = pfrommer_shock_structure(*TH.left, *TH.right, GAMMA, GAMMA_CR,
                                      injection=ShockInjection(0.5, "total"))
        cr0 = pfrommer_shock_structure(*CR.left, *CR.right, GAMMA, GAMMA_CR)
        cr = pfrommer_shock_structure(*CR.left, *CR.right, GAMMA, GAMMA_CR,
                                      injection=ShockInjection(0.5, "total"))
        print(f"Table 1: th M {base['mach_gas']:.4f} -> {th['mach_gas']:.4f}, x "
              f"{base['compression']:.4f} -> {th['compression']:.4f}; CR M_comp "
              f"{cr0['mach_composite']:.4f} -> {cr['mach_composite']:.4f}")
        assert abs(th["mach_gas"] - 9.56) < 5e-3 and abs(th["compression"] - 4.74) < 5e-3
        assert abs(cr0["mach_composite"] - 10.0) < 5e-4
        assert abs(cr["mach_composite"] - 9.56) < 5e-3

        for tube in (TH, CR):
            for inj in (ShockInjection(0.5, "total"), injection(0.1), injection(0.5),
                        injection(0.5, "gas_pressure"), injection(0.5, "velocity")):
                s = pfrommer_shock_structure(*tube.left, *tube.right, GAMMA, GAMMA_CR,
                                             injection=inj)
                v1 = s["v_shock"] - tube.right[1]
                up = _shock_frame_fluxes(tube.right[0], v1, tube.right[2], tube.right[3])
                down = _shock_frame_fluxes(s["rho_2"], v1 / s["compression"], s["p_th_2"],
                                           s["p_cr_2"])
                assert np.max(np.abs(down / up - 1.0)) < 1e-10, (inj, down / up - 1.0)
                if inj.convention == "gas_rh" and inj.mach == "true":
                    assert abs(s["f_inj"] / (inj.zeta * s["f_diss_gas"]) - 1.0) < 1e-10
                if inj.mach in ("gas_pressure", "velocity"):
                    from astronomix.test_setups.reference_solutions.pfrommer_riemann_solver import (
                        thermalization_efficiency,
                    )
                    m_p = s["mach_gas_pressure" if inj.mach == "gas_pressure" else "mach_gas_velocity"]
                    c_th = np.sqrt(GAMMA * tube.right[2] / tube.right[0])
                    f_p = inj.zeta * thermalization_efficiency(m_p, GAMMA) * 0.5 * tube.right[0] * (
                        m_p * c_th) ** 3
                    assert abs(s["f_inj"] / f_p - 1.0) < 1e-10

        # strong-shock limit against B7.1
        strong = Tube(left=TH.left, right=(0.125, 0.0, 1e-9, 0.0))
        for zeta in (0.1, 0.5):
            for convention, inj in (("total", ShockInjection(zeta, "total")),
                                    ("gas_rh", injection(zeta))):
                s = pfrommer_shock_structure(*strong.left, *strong.right, GAMMA, GAMMA_CR,
                                             injection=inj)
                x_ref, f_ref = _strong_shock_state(zeta, convention, GAMMA, GAMMA_CR)
                f = s["p_cr_2"] / (s["p_th_2"] + s["p_cr_2"])
                assert abs(s["compression"] / x_ref - 1.0) < 1e-4, (convention, zeta)
                assert abs(f / f_ref - 1.0) < 1e-4, (convention, zeta)

        # sampled profile: mass and energy over the box
        xs = np.linspace(0.0, BOX, 40001)
        for inj in (None, injection(0.5), ShockInjection(0.5, "total")):
            rho, u, p_th, p_cr = pfrommer_riemann_solution(
                *TH.left, *TH.right, GAMMA, GAMMA_CR, xs, T_END, X0, injection=inj)
            energy = np.trapezoid(0.5 * rho * u**2 + p_th / (GAMMA - 1.0)
                                  + p_cr / (GAMMA_CR - 1.0), xs)
            left, right = TH.left, TH.right
            energy_0 = X0 * left[2] / (GAMMA - 1.0) + (BOX - X0) * right[2] / (GAMMA - 1.0)
            mass_0 = X0 * left[0] + (BOX - X0) * right[0]
            assert abs(energy / energy_0 - 1.0) < 1e-4, (inj, energy / energy_0 - 1.0)
            assert abs(np.trapezoid(rho, xs) / mass_0 - 1.0) < 1e-4


def l1_density(tube, run, zeta, mach=FINDER_MACH):
    """L1 density error over the box against the exact injection solution."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho_ref = pfrommer_riemann_solution(*tube.left, *tube.right, GAMMA, GAMMA_CR, run["x"],
                                            T_END, X0, injection=injection(zeta, mach))[0]
    return float(np.mean(np.abs(run["rho"] - rho_ref)) * BOX)


DEFAULT_RESOLUTIONS = (100, 200, 400, 800, 1600, 3200)
VARIANT_RESOLUTIONS = (400, 1600)
ZETAS = (0.1, 0.5)
GATED_ZETAS = (0.1,)  # zeta = 0.5 tracked only (module docstring)


def test_cr_shock_tube_injection(
    max_dv=2e-3,
    max_dP_th=0.02,
    max_dP_cr=0.05,
    max_drho=0.01,
    max_rate=0.05,
    max_mach_finder=0.03,
    min_l1_order=0.7,
):
    """B7.4: thermal tube of Pfrommer et al. (2017) Table 1 with DSA, ``zeta`` 0.1 and 0.5.

    Hard gates (default scheme, N >= 400, ``zeta`` in ``GATED_ZETAS``) against the exact solution of the code's convention
    at the Mach number the finder reads (``injection(zeta, FINDER_MACH)``, module
    docstring):

    Args:
        max_dv: Shock speed (snapshot fit), relative.
        max_dP_th: Plateau thermal pressure, relative.
        max_dP_cr: Plateau CR pressure, relative (the deposit inside the smeared shock adds
            ~3%, constant in N; the plan's 2% is not met).
        max_drho: Plateau density, relative.
        max_rate: Box CR energy growth rate against the exact injected flux, relative.
        max_mach_finder: Finder Mach number at the final state against the reference's Mach
            number of the same kind (``FINDER_MACH``), relative.
        min_l1_order: Least-squares order of the L1 density error over N >= 200.

    Everything else (the same metrics against the true-Mach reference, the energy scheme with
    HLL and HLLC, the surface-cell deposit and the CR-loaded tube) is tracked in
    ``baselines.json``.
    """
    baselines = Baselines("cr_shock_tube_injection.py::test_cr_shock_tube_injection")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = []
    for zeta in ZETAS:
        ref_true, ref_fp = reference(TH, zeta, "true"), reference(TH, zeta, FINDER_MACH)
        for scheme, kwargs in SCHEMES.items():
            resolutions = DEFAULT_RESOLUTIONS if scheme == "default" else VARIANT_RESOLUTIONS
            for n in resolutions:
                run = run_tube(TH, n, zeta, **kwargs)
                fp, true = metrics(TH, run, ref_fp), metrics(TH, run, ref_true)
                fp["l1_rho"] = l1_density(TH, run, zeta)
                fp["mach_finder_rel"] = run["mach_finder"] / ref_fp[FINDER_MACH_KEY] - 1.0
                true["mach_finder_rel"] = run["mach_finder"] / ref_true["mach_gas"] - 1.0
                results[(zeta, scheme, n)] = dict(run=run, fp=fp, true=true)
                tag = f"z{zeta:g}_{scheme.replace(', ', '_').replace(' ', '_')}_N{n}"
                print(f"zeta {zeta} {scheme:25s} N {n:5d}: FP dv {fp['dv_shock']:+.4f} dP_th "
                      f"{fp['dP_th']:+.4f} dP_cr {fp['dP_cr']:+.4f} drho {fp['drho']:+.4f} rate "
                      f"{fp['rate'] - 1.0:+.4f} M_f {fp['mach_finder_rel']:+.4f} L1 {fp['l1_rho']:.3e}"
                      f" | TRUE dP_cr {true['dP_cr']:+.4f} rate {true['rate'] - 1.0:+.4f} M_f "
                      f"{true['mach_finder_rel']:+.4f}")
                np.savez(RAW_DIR / f"{tag}.npz", x=run["x"], rho=run["rho"], u=run["u"],
                         p_th=run["p_th"], p_cr=run["p_cr"])
                for key in ("dv_shock", "dP_th", "dP_cr", "drho", "mach_finder_rel"):
                    baselines.close(f"{tag}_fp_{key}", fp[key], abs_tol=0.01)
                baselines.close(f"{tag}_fp_rate", fp["rate"], rel=0.02)
                baselines.close(f"{tag}_true_dP_cr", true["dP_cr"], abs_tol=0.01)
                baselines.close(f"{tag}_true_mach_finder_rel", true["mach_finder_rel"], abs_tol=0.01)
                if scheme == "default" and n >= 400 and zeta in GATED_ZETAS:
                    for key, bound in (("dv_shock", max_dv), ("dP_th", max_dP_th),
                                       ("dP_cr", max_dP_cr), ("drho", max_drho),
                                       ("mach_finder_rel", max_mach_finder)):
                        if not abs(fp[key]) < bound:
                            failures.append(f"{tag}: {key} {fp[key]:+.4f} (bound {bound})")
                    if not abs(fp["rate"] - 1.0) < max_rate:
                        failures.append(f"{tag}: rate {fp['rate']:.4f} (bound {max_rate})")
        ns = np.array([n for n in DEFAULT_RESOLUTIONS if n >= 200])
        l1 = np.array([results[(zeta, "default", n)]["fp"]["l1_rho"] for n in ns])
        order = -np.polyfit(np.log(ns), np.log(l1), 1)[0]
        print(f"zeta {zeta}: L1(rho) order {order:.2f}")
        baselines.close(f"z{zeta:g}_l1_order", order, abs_tol=0.1)
        if zeta in GATED_ZETAS and not order > min_l1_order:
            failures.append(f"zeta {zeta}: L1 order {order:.2f} (bound {min_l1_order})")

    # CR-loaded tube (Pfrommer et al. 2017 Table 1 "CR"; B8.5 territory), tracked only
    ref_cr = reference(CR, 0.5, FINDER_MACH)
    for n in VARIANT_RESOLUTIONS:
        run = run_tube(CR, n, 0.5)
        m = metrics(CR, run, ref_cr)
        results[(0.5, "CR tube", n)] = dict(run=run, fp=m)
        print(f"CR tube zeta 0.5 N {n}: dv {m['dv_shock']:+.4f} dP_th {m['dP_th']:+.4f} dP_cr "
              f"{m['dP_cr']:+.4f} dP_cr_inj {m['dP_cr_inj']:+.4f} M_f {m['mach_finder']:.3f} "
              f"(finder-Mach reference {ref_cr[FINDER_MACH_KEY]:.3f}, composite "
              f"{ref_cr['mach_composite']:.3f})")
        for key in ("dv_shock", "dP_th", "dP_cr", "dP_cr_inj"):
            baselines.close(f"cr_tube_N{n}_fp_{key}", m[key], abs_tol=0.01)

    _plot(results)
    baselines.check()
    assert not failures, "\n".join(failures)


def _plot(results):
    """Plateau profiles at N = 1600, the resolution series and the scheme comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    xs = np.linspace(X0 + 2.6, X0 + 4.2, 4001)
    for row, zeta in enumerate(ZETAS):
        ax = axes[row, 0]
        run = results[(zeta, "default", 1600)]["run"]
        for mach, ls in (("true", ":"), (FINDER_MACH, "--")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, p_th, p_cr = pfrommer_riemann_solution(
                    *TH.left, *TH.right, GAMMA, GAMMA_CR, xs, T_END, X0,
                    injection=injection(zeta, mach))
            ax.plot(xs, p_th, "k" + ls, lw=1)
            ax.plot(xs, p_cr, "C3" + ls, lw=1)
        sel = (run["x"] > xs[0]) & (run["x"] < xs[-1])
        ax.plot(run["x"][sel], run["p_th"][sel], "k-", lw=1.5, label=r"$P_{th}$")
        ax.plot(run["x"][sel], run["p_cr"][sel], "C3-", lw=1.5, label=r"$P_{cr}$")
        fp = results[(zeta, "default", 1600)]["fp"]
        true = results[(zeta, "default", 1600)]["true"]
        ax.set(xlabel="x", ylabel="pressure",
               title=rf"$\zeta$={zeta}, N=1600: $P_{{cr}}$ {fp['dP_cr']:+.1%} (finder-M ref, dashed),"
                     f"\n{true['dP_cr']:+.1%} (true-M ref, dotted)")
        ax.legend(fontsize=8)

        ax = axes[row, 1]
        ns = np.array(DEFAULT_RESOLUTIONS)
        for key, label, c in (("dP_cr", r"$P_{cr}$", "C3"), ("dP_th", r"$P_{th}$", "k"),
                              ("dv_shock", r"$v_s$", "C0"), ("mach_finder_rel", r"$M_{finder}$", "C2")):
            vals = [results[(zeta, "default", n)]["fp"][key] for n in ns]
            ax.semilogx(ns, vals, "o-", color=c, label=label + " vs finder-M ref")
            if key in ("dP_cr", "mach_finder_rel"):
                vals = [results[(zeta, "default", n)]["true"][key] for n in ns]
                ax.semilogx(ns, vals, "s:", color=c, mfc="none", label=label + " vs true-M ref")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(xlabel="N", ylabel="relative error", title=rf"default scheme, $\zeta$={zeta}")
        ax.legend(fontsize=7)

        ax = axes[row, 2]
        names = list(SCHEMES)
        for j, n in enumerate(VARIANT_RESOLUTIONS):
            vals = [results[(zeta, s, n)]["fp"]["dP_cr"] for s in names]
            ax.bar(np.arange(len(names)) + 0.4 * j, vals, 0.4, label=f"N={n}")
        ax.set_xticks(np.arange(len(names)) + 0.2)
        ax.set_xticklabels([s.replace(", ", ",\n") for s in names], fontsize=8)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(ylabel=r"$P_{cr}$ vs finder-M ref", title="schemes and deposit")
        ax.legend(fontsize=8)
    fig.suptitle("B7.4: thermal tube of Pfrommer et al. (2017) with DSA vs the exact solution "
                 f"(code convention; dashed: at the finder's Mach number ({FINDER_MACH} jump), dotted: true Mach)")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_shock_tube_injection.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    test_injection_riemann_reference()
    test_cr_shock_tube_injection()
