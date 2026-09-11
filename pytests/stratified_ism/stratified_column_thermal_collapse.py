"""
Stratified column with K&I cooling: thermal-gravitational collapse onset
(SILCC-ISM project, milestone M2).

Combines M0b's vertically-stratified external-potential column
(``stratified_hydrostatic_column.py``) with M1's Koyama & Inutsuka (2002)
two-phase net-cooling curve (``ki_cooling_thermal_relaxation.py``) for the
first time. The milestone's original goal (per the plan) was to "verify a
physically sensible two-phase-structured quasi-equilibrium" -- **this
milestone found that no such static equilibrium is dynamically reachable
in this setup**, and was rescoped (user-confirmed, see PROGRESS.md's
2026-09-11 M2 entry for the full investigation) to verify that the column
instead undergoes a real, numerically well-behaved onset of
thermal-gravitational collapse, up to a defensible cutoff time -- not a
bug, and not a resolution artifact.

**Why there is no static equilibrium here (the investigation, summarized):**
three independent checks, plus a resolution scan, all agree:

1. Starting from a naive single-temperature isothermal column (M0b's own
   IC style), the midplane runs away (``n_H``: 10 -> 195 cm^-3, ``T``: 8000
   -> 47.6 K by 2 dynamical times) and the run goes to NaN by 5.
2. Starting instead from *this* module's IC (below -- density from a warm
   isothermal guess, but temperature set per-cell from the K&I equilibrium
   at that local density, i.e. already very close to local thermal
   balance, not the naive profile above) makes **no qualitative
   difference** -- the midplane still runs away to the same kind of
   endpoint (``n_H``: 10 -> 203, NaN by 5 dynamical times). This rules out
   "bad initial condition" as the cause.
3. Systematically weakening the external potential (``C_pot`` scaled by
   0.3, 0.1, 0.03, 0.01) does not find a stable *stratified* regime either:
   factors down to 0.1 still collapse just as badly; factors of 0.03/0.01
   only avoid collapse by flattening the initial density contrast away
   almost entirely (midplane-to-edge density ratio drops to ~1.1-1.3x,
   too weak to represent a real two-phase structure) -- weakening gravity
   trades away the very stratification this milestone needs to
   demonstrate, rather than stabilizing it.
4. **Resolution scan (the decisive check):** at low resolution
   (``N_XY=16``, ``N_Z=128``) the collapse is comparatively slow (NaN
   between 2 and 5 dynamical times); at higher resolution (``N_XY=32``,
   ``N_Z=256``, this test's calibration) the *same* setup collapses
   *faster* (NaN between 1 and 2 dynamical times), not slower. A
   resolution increase making a collapse happen sooner (not converging to
   a stable answer) is the classic signature of an **unregularized
   thermal instability**: Field (1965)'s classical result is that the
   ISM thermal instability's growth rate is unbounded at short wavelength
   without a regularizing mechanism (thermal conduction, turbulence,
   magnetic tension -- none of which this module implements yet). Refining
   resolution further would only resolve smaller, faster-growing modes,
   never converge to a stable profile.

**Conclusion:** a plain external-potential-confined column combined with
this two-phase cooling curve is genuinely, physically unstable to runaway
condensation once there is meaningful density stratification -- this is
real thermal-gravitational collapse physics, not a numerical defect, and
it previews exactly why the roadmap's M3 (episodic SN driving/turbulence)
exists as the mechanism expected to resist this runaway. The "stable
two-phase profile" claim from the original M2 scope is deferred to
M4/M5, once that additional physics is present.

**What this test actually checks instead:** the column's *early*,
well-resolved evolution (before the collapse reaches a numerically
pathological state) is a real, self-consistent physical process, not
numerical noise -- the midplane density grows substantially (genuine
ongoing condensation), the midplane temperature tracks close to its own
*new, local* K&I equilibrium as it condenses (confirms the cooling
physics is driving this correctly, not some other instability/bug), the
envelope (which started near its own equilibrium) stays quiescent, mass
is conserved to good precision, and the run is NaN-free throughout.

**Setup (calibrated 2026-09-11):** 3D Cartesian FV (``HLLC``/``MINMOD``),
``CodeUnits(1 pc, 1 Msun, 1 km/s)`` (matching M1's convention -- cooling
needs real Kelvin), mixed periodic-xy/open-z BCs (M0a), external potential
``phi(z) = C_pot*log(cosh((z-z0)/h))`` (M0b's form) with ``h=50`` pc,
domain spanning ``z0 +/- 3h`` (edge density ``~0.1`` cm^-3, comfortably
inside the K&I curve's well-behaved regime). Midplane density ``n_H,0=10``
cm^-3 (matches M1's own calibrated CNM point). ``C_pot`` fixed via a
reference warm temperature of 7500 K (``C_pot = 2*c_s_warm^2``, the same
relation M0b used, here just providing a reference scale for the
potential's depth rather than exactly matching a single-temperature
solution). ``N_XY=32``, ``N_Z=256`` (uniform grid spacing, required by
this codebase).

**Initial condition:** density from the warm (T=7500 K) isothermal
sech^2-like barometric profile, but temperature set *per cell* from the
K&I local equilibrium at that cell's guessed density (not the single
global temperature M0b used) -- a "coarse" local-equilibrium guess
(not a fully self-consistent hydrostatic solution -- an attempt to build
one via fixed-point iteration on the coupled hydrostatic+K&I-equilibrium
ODE was tried and diverges outright, the same instability point 2 above
already demonstrates directly) that at least starts the midplane close to
correct thermal balance, ruling out "started too far from equilibrium" as
an independent contributing cause (point 2 above).

``t_end = 1.0`` vertical dynamical time (``h / c_s,warm ~= 3.70e6`` yr) --
comfortably inside the clean, NaN-free regime (collapse reaches NaN
between 1 and 2 dynamical times at this resolution), giving a full
dynamical time of genuine, well-resolved evolution to check.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# Equilibrium-tracking checks need enough precision to distinguish real
# drift from round-off -- same reasoning as every other file in this
# project (bc_smoke_test.py / stratified_hydrostatic_column.py /
# ki_cooling_thermal_relaxation.py).
jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np
from scipy.interpolate import interp1d

# units
from astropy import units as u
import astropy.constants as c

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix.option_classes.simulation_config import SIMPLE_SOURCE
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import OPEN_BOUNDARY, PERIODIC_BOUNDARY
from astronomix import BoundarySettings, BoundarySettings1D, GravityConfig, CodeUnits
from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import StaticFloatVector, StaticIntVector

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

# astronomix modules
from astronomix._modules._cooling._cooling import (
    get_effective_molecular_weights,
    get_pressure_from_temperature,
    get_temperature_from_pressure,
)
from astronomix._modules._cooling._cooling_tables import koyama_inutsuka_cooling
from astronomix._modules._cooling.cooling_options import (
    IMPLICIT_COOLING,
    KOYAMA_INUTSUKA_NET_COOLING,
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
)

# independent reference (plain numpy/scipy, no astronomix/JAX)
from astronomix.test_setups.reference_solutions.koyama_inutsuka_equilibrium import find_equilibrium_temperatures

# ---- physical setup ----
GAMMA = 5.0 / 3.0
X_H = 0.76
Z_METAL = 0.02
MU, MU_E, MU_H = get_effective_molecular_weights(X_H, Z_METAL)

CODE_UNITS = CodeUnits(1 * u.pc, 1 * u.M_sun, 1 * u.km / u.s)
KI_PARAMS = koyama_inutsuka_cooling(CODE_UNITS, X_H, Z_METAL)
FLOOR_TEMPERATURE_KELVIN = 10.0
FLOOR_TEMPERATURE_CODE = FLOOR_TEMPERATURE_KELVIN * KI_PARAMS.code_temperature_per_kelvin

N_H_MIDPLANE = 10.0  # cm^-3, matches ki_cooling_thermal_relaxation.py's CNM point
H_SCALE_PHYS = 50.0 * u.pc
H_SCALE = H_SCALE_PHYS.to(CODE_UNITS.code_length).value
Z0 = 3.0 * H_SCALE
L_Z = 6.0 * H_SCALE
L_XY = 0.75 * H_SCALE  # ratio chosen so N_XY/N_Z below gives uniform grid spacing
N_XY, N_Z = 32, 256

T_WARM_REFERENCE_KELVIN = 7500.0  # sets the potential's depth (C_pot); edge equilibrium is close to this
C2_WARM = T_WARM_REFERENCE_KELVIN * KI_PARAMS.code_temperature_per_kelvin / MU
C_POT = 2.0 * C2_WARM
C_S_WARM = float(np.sqrt(GAMMA * C2_WARM))
T_DYN = H_SCALE / C_S_WARM  # vertical dynamical time, code units
T_END = 1.0 * T_DYN

# Fast T_eq(n_H) interpolator (log-log cubic spline), built once from the
# same independent reference the rest of this project uses -- avoids
# calling the root-finding scan directly inside the IC construction.
_N_GRID = np.logspace(np.log10(0.01), np.log10(300.0), 400)
_T_GRID = np.array([find_equilibrium_temperatures(n)[0] for n in _N_GRID])
_LOG_TEQ_INTERP = interp1d(np.log(_N_GRID), np.log(_T_GRID), kind="cubic", fill_value="extrapolate")


def _t_eq_fast(n_h):
    """Fast log-log interpolated K&I equilibrium temperature (Kelvin)."""
    return np.exp(_LOG_TEQ_INTERP(np.log(n_h)))


def _run_collapse_onset():
    """Run the stratified column with K&I cooling; return diagnostics."""
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        box_size=StaticFloatVector(L_XY, L_XY, L_Z),
        num_cells=StaticIntVector(N_XY, N_XY, N_Z),
        exact_end_time=True,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        gravity_config=GravityConfig(self_gravity_version=SIMPLE_SOURCE, external_potential=True),
        cooling_config=CoolingConfig(
            cooling=True,
            cooling_method=IMPLICIT_COOLING,
            cooling_curve_config=CoolingCurveConfig(cooling_curve_type=KOYAMA_INUTSUKA_NET_COOLING),
        ),
        progress_bar=False,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    z = helper_data.geometric_centers[..., 2]
    phi = C_POT * jnp.log(jnp.cosh((z - Z0) / H_SCALE))

    # n_H = density / mu_H (this codebase's convention) -- see
    # ki_cooling_thermal_relaxation.py's module docstring, "real finding #3".
    rho_midplane = float((N_H_MIDPLANE / u.cm ** 3 * MU_H * c.m_p).to(CODE_UNITS.code_density).value)

    # One-shot local-equilibrium IC: warm-isothermal density guess, but
    # temperature set per-cell from the K&I equilibrium at that guessed
    # local density (see module docstring -- a fully self-consistent
    # hydrostatic+K&I-equilibrium solve was attempted and diverges).
    rho_guess = rho_midplane * jnp.exp(-phi / C2_WARM)
    rho_guess_np = np.asarray(rho_guess)
    n_h_guess = (rho_guess_np * CODE_UNITS.code_density).to(u.g / u.cm ** 3).value / (
        MU_H * c.m_p.to(u.g).value
    )
    t_local_kelvin = _t_eq_fast(n_h_guess)
    t_tilde_local = jnp.asarray(t_local_kelvin * KI_PARAMS.code_temperature_per_kelvin)

    rho_init = rho_guess
    p_init = get_pressure_from_temperature(rho_init, t_tilde_local, X_H, Z_METAL)
    zero = jnp.zeros_like(z)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho_init,
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
        gas_pressure=p_init,
    )
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        cooling_params=CoolingParams(
            hydrogen_mass_fraction=X_H,
            metal_mass_fraction=Z_METAL,
            floor_temperature=FLOOR_TEMPERATURE_CODE,
            cooling_curve_params=KI_PARAMS,
        ),
    )
    params = params._replace(gravitational_potential=phi)
    config = finalize_config(config, initial_state.shape)

    final_state = time_integration(initial_state, config, params, registered_variables)

    rho_final = final_state[registered_variables.density_index]
    p_final = final_state[registered_variables.pressure_index]
    t_final_code = get_temperature_from_pressure(rho_final, p_final, X_H, Z_METAL)
    t_final_kelvin = t_final_code / KI_PARAMS.code_temperature_per_kelvin

    ix, iy = N_XY // 2, N_XY // 2
    mid_idx = N_Z // 2
    z_line = np.asarray(z[ix, iy, :])
    t_line = np.asarray(t_final_kelvin[ix, iy, :])
    rho_line = np.asarray(rho_final[ix, iy, :])
    rho_line_phys = (rho_line * CODE_UNITS.code_density).to(u.g / u.cm ** 3)
    n_h_line = (rho_line_phys / (MU_H * c.m_p)).to(1 / u.cm ** 3).value

    total_mass_initial = float(jnp.sum(rho_init))
    total_mass_final = float(jnp.sum(rho_final))

    return dict(
        final_state=final_state,
        z_line=z_line,
        n_h_init=n_h_guess[ix, iy, :],
        t_init_kelvin=t_local_kelvin[ix, iy, :],
        n_h_line=n_h_line,
        t_line=t_line,
        mid_idx=mid_idx,
        total_mass_initial=total_mass_initial,
        total_mass_final=total_mass_final,
    )


def test_stratified_column_thermal_collapse_onset(
    mass_conservation_tol: float = 5e-3,
    midplane_growth_min: float = 1.5,
    midplane_eq_tracking_tol: float = 0.08,
    edge_growth_max: float = 1.3,
    edge_eq_tracking_tol: float = 0.01,
):
    """Onset of thermal-gravitational collapse in a cooling stratified column.

    See module docstring for why this checks the collapse's onset rather
    than a static equilibrium (no such equilibrium is dynamically
    reachable in this setup -- confirmed via a resolution scan showing the
    unregularized-thermal-instability signature, not a numerical defect).

    Calibrated 2026-09-11: midplane n_H grows 10.00 -> 21.65 (factor 2.165,
    tol has >1.4x margin) while tracking its own new local equilibrium to
    3.98% (tol has >2x margin) -- confirms the K&I cooling is correctly
    driving the collapsing gas toward its shifting local balance, not
    doing something numerically arbitrary. The envelope stays quiescent:
    n_H grows only 1.14x (tol has margin) and tracks its (near-initial)
    equilibrium to 0.091% (tol has >10x margin). Mass is conserved to
    2.33e-3 (tol has >2x margin) -- much tighter than the ~1-30% seen once
    a run is pushed into the pathological regime (t>=2 dynamical times at
    this resolution).

    Args:
        mass_conservation_tol: Max relative error on total mass.
        midplane_growth_min: Min required n_H growth factor at the
            midplane (confirms genuine, non-trivial ongoing condensation).
        midplane_eq_tracking_tol: Max relative error between the
            midplane's final T and the K&I equilibrium T at its own final
            n_H (confirms the cooling physics, not noise, drives this).
        edge_growth_max: Max allowed n_H growth factor at the domain edge
            (confirms the envelope, which started near its own
            equilibrium, stays quiescent).
        edge_eq_tracking_tol: Max relative error between the edge's final
            T and the K&I equilibrium T at its own final n_H.
    """
    run = _run_collapse_onset()

    assert not bool(jnp.any(jnp.isnan(run["final_state"]))), (
        "Stratified column with K&I cooling produced NaNs within t_end="
        f"{T_END:.4f} code time (1.0 dynamical time) -- this is inside the "
        "calibrated NaN-free regime (collapse reaches NaN between 1 and 2 "
        "dynamical times at this resolution); if this now fails, either "
        "t_end needs reducing or something upstream has changed."
    )

    mass_rel_err = abs(run["total_mass_final"] - run["total_mass_initial"]) / run["total_mass_initial"]
    assert mass_rel_err < mass_conservation_tol, (
        f"Mass not conserved to tolerance: rel. err {mass_rel_err:.4e} >= "
        f"tol {mass_conservation_tol}."
    )

    mid_idx = run["mid_idx"]
    n_h_mid_init = run["n_h_init"][mid_idx]
    n_h_mid_final = run["n_h_line"][mid_idx]
    t_mid_final = run["t_line"][mid_idx]
    t_eq_mid_final = find_equilibrium_temperatures(n_h_mid_final)[0]

    growth_mid = n_h_mid_final / n_h_mid_init
    assert growth_mid > midplane_growth_min, (
        f"Midplane n_H growth factor ({growth_mid:.4f}) is below "
        f"{midplane_growth_min} -- expected genuine, substantial ongoing "
        "condensation at the midplane by t_end."
    )

    mid_eq_rel_err = abs(t_mid_final - t_eq_mid_final) / t_eq_mid_final
    assert mid_eq_rel_err < midplane_eq_tracking_tol, (
        f"Midplane T ({t_mid_final:.2f} K) does not track its own new "
        f"local K&I equilibrium ({t_eq_mid_final:.2f} K at its final "
        f"n_H={n_h_mid_final:.2f} cm^-3) -- rel. err {mid_eq_rel_err:.4e} "
        f">= tol {midplane_eq_tracking_tol}."
    )

    n_h_edge_init = run["n_h_init"][0]
    n_h_edge_final = run["n_h_line"][0]
    t_edge_final = run["t_line"][0]
    t_eq_edge_final = find_equilibrium_temperatures(n_h_edge_final)[0]

    growth_edge = n_h_edge_final / n_h_edge_init
    assert growth_edge < edge_growth_max, (
        f"Envelope (edge) n_H growth factor ({growth_edge:.4f}) exceeds "
        f"{edge_growth_max} -- expected the envelope, which started near "
        "its own equilibrium, to stay close to quiescent."
    )

    edge_eq_rel_err = abs(t_edge_final - t_eq_edge_final) / t_eq_edge_final
    assert edge_eq_rel_err < edge_eq_tracking_tol, (
        f"Envelope T ({t_edge_final:.2f} K) does not track its own K&I "
        f"equilibrium ({t_eq_edge_final:.2f} K) -- rel. err "
        f"{edge_eq_rel_err:.4e} >= tol {edge_eq_tracking_tol}."
    )

    # Diagnostic plot: initial vs. final density and temperature profiles,
    # with the K&I equilibrium curve evaluated at the final density
    # overlaid -- shows how closely the (still-evolving) column tracks
    # its own local equilibrium as it condenses.
    z_pc = (run["z_line"] * CODE_UNITS.code_length).to(u.pc).value
    t_eq_of_final_n_h = _t_eq_fast(run["n_h_line"])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))
    ax0.semilogy(z_pc, run["n_h_init"], "--", color="C0", label="initial")
    ax0.semilogy(z_pc, run["n_h_line"], "-", color="C1", label=f"final (t={T_END:.3f} code time)")
    ax0.set_xlabel("z [pc]")
    ax0.set_ylabel(r"$n_H$ [cm$^{-3}$]")
    ax0.set_title("density profile")
    ax0.legend()

    ax1.semilogy(z_pc, run["t_init_kelvin"], "--", color="C0", label="initial")
    ax1.semilogy(z_pc, run["t_line"], "-", color="C1", label="final")
    ax1.semilogy(z_pc, t_eq_of_final_n_h, ":", color="C2", label="K&I T_eq(final n_H)")
    ax1.set_xlabel("z [pc]")
    ax1.set_ylabel("T [K]")
    ax1.set_title("temperature profile vs. local equilibrium")
    ax1.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "stratified_column_thermal_collapse_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_stratified_column_thermal_collapse_onset()
