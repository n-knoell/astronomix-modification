"""
SILCC-ISM project milestone M4: stratified column + episodic SN driving +
K&I cooling, with the *delayed-cooling* overcooling mitigation.

**NOT a committed pytest -- an exploratory script**, per this milestone's own
established status in PROGRESS.md ("M4 itself remains an exploratory
scratchpad script, not a committed pytest yet"). It is meant to be run and
inspected directly, not asserted against fixed tolerances.

Combines M2's exact stratified+cooling column
(``stratified_column_thermal_collapse.py``) with M3's episodic SN driving
(``sn_driving_energy_conservation.py`` / ``sn_driving_options.py``), plus the
new ``SNDrivingConfig.delayed_cooling`` mitigation (2026-09-15) for the real
overcooling finding this milestone hit (see PROGRESS.md's 2026-09-14 M4
entry): the K&I cooling time at a fresh midplane injection's own post-shock
state was measured at ``~60 yr``, far shorter than a single hydro dynamical
time, so a pure-thermal deposit radiated away before it could do any work and
SN driving was effectively thermally inert -- the column collapsed just like
M2's baseline. ``momentum_injection`` (Kim & Ostriker 2015) was the first
mitigation tried; it fixed the overcooling but hit a third, still-unresolved
NaN mechanism, so this script tries ``delayed_cooling`` instead (temporarily
suppressing K&I cooling within a fresh injection's footprint) -- see
``SNDrivingConfig.delayed_cooling``'s docstring in ``sn_driving_options.py``
and PROGRESS.md's 2026-09-15 entry for the mechanism and its own isolated
unit-test verification (not yet tried against this real run).

Design decisions reused directly from PROGRESS.md's 2026-09-14 M4 entry
(grounded there, not re-derived here):

1. Box/potential/IC: identical to M2 (same constants below) -- same box
   rather than a literature-scale one (cost), same one-shot local-equilibrium
   IC.
2. SN rate: boosted ~10x the area-scaled literature rate, targeting
   ``EXPECTED_TRIGGERS_PER_T_DYN`` expected triggers per dynamical time (M2's
   own box is too small for the literature areal rate to trigger more than
   about once per collapse timescale).
3. Site placement: ``sn_z_min``/``sn_z_max`` restrict real SN sites to within
   one scale height of the midplane, so random sites land in the
   astrophysically relevant dense gas rather than the tenuous envelope.
4. Injection radius calibrated (6 pc, ~5.1 grid cells at this resolution) as
   a compromise between numerical gentleness and footprint size.

``sn_cooling_delay_time`` here is *not* hand-copied from PROGRESS.md's
``~60 yr`` number -- it is recomputed fresh via ``_probe_post_shock_cooling_time``
below (same technique: a hand-built single deposit at the midplane state,
independent of any PRNG trigger), then scaled by ``DELAY_MULTIPLIER``, so the
calibration is self-consistent with this exact config even if grid spacing,
injection radius, or the cooling curve's calibration ever drift.
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
from astronomix.option_classes.simulation_config import SIMPLE_SOURCE, SnapshotSettings
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
    cooling_time,
)
from astronomix._modules._cooling._cooling_tables import koyama_inutsuka_cooling
from astronomix._modules._cooling.cooling_options import (
    IMPLICIT_COOLING,
    KOYAMA_INUTSUKA_NET_COOLING,
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
)
from astronomix._modules._sn_driving.sn_driving_options import SNDrivingConfig, SNDrivingParams

# independent reference (plain numpy/scipy, no astronomix/JAX)
from astronomix.test_setups.reference_solutions.koyama_inutsuka_equilibrium import find_equilibrium_temperatures

# ---- physical setup (identical to stratified_column_thermal_collapse.py, M2) ----
GAMMA = 5.0 / 3.0
X_H = 0.76
Z_METAL = 0.02
MU, MU_E, MU_H = get_effective_molecular_weights(X_H, Z_METAL)

CODE_UNITS = CodeUnits(1 * u.pc, 1 * u.M_sun, 1 * u.km / u.s)
KI_PARAMS = koyama_inutsuka_cooling(CODE_UNITS, X_H, Z_METAL)
FLOOR_TEMPERATURE_KELVIN = 10.0
FLOOR_TEMPERATURE_CODE = FLOOR_TEMPERATURE_KELVIN * KI_PARAMS.code_temperature_per_kelvin

N_H_MIDPLANE = 10.0  # cm^-3
H_SCALE_PHYS = 50.0 * u.pc
H_SCALE = H_SCALE_PHYS.to(CODE_UNITS.code_length).value
Z0 = 3.0 * H_SCALE
L_Z = 6.0 * H_SCALE
L_XY = 0.75 * H_SCALE
N_XY, N_Z = 32, 256
GRID_SPACING_CODE = L_XY / N_XY

T_WARM_REFERENCE_KELVIN = 7500.0
C2_WARM = T_WARM_REFERENCE_KELVIN * KI_PARAMS.code_temperature_per_kelvin / MU
C_POT = 2.0 * C2_WARM
C_S_WARM = float(np.sqrt(GAMMA * C2_WARM))
T_DYN = H_SCALE / C_S_WARM  # vertical dynamical time, code units
T_DYN_YEARS = (T_DYN * CODE_UNITS.code_time).to(u.yr).value

_N_GRID = np.logspace(np.log10(0.01), np.log10(300.0), 400)
_T_GRID = np.array([find_equilibrium_temperatures(n)[0] for n in _N_GRID])
_LOG_TEQ_INTERP = interp1d(np.log(_N_GRID), np.log(_T_GRID), kind="cubic", fill_value="extrapolate")


def _t_eq_fast(n_h):
    return np.exp(_LOG_TEQ_INTERP(np.log(n_h)))


# ---- M4-specific setup ----
T_END = 2.0 * T_DYN  # checkpoint window PROGRESS.md's earlier momentum-injection attempt used

RHO_MIDPLANE_CODE = float((N_H_MIDPLANE / u.cm ** 3 * MU_H * c.m_p).to(CODE_UNITS.code_density).value)

EXPECTED_TRIGGERS_PER_T_DYN = 8.0
SN_RATE_CODE = EXPECTED_TRIGGERS_PER_T_DYN / T_DYN

SN_ENERGY_CODE = (1.0e51 * u.erg).to(CODE_UNITS.code_energy).value
SN_INJECTION_RADIUS_CODE = (6.0 * u.pc).to(CODE_UNITS.code_length).value
SN_SMOOTH_CELLS = 2.0
SN_Z_MIN = Z0 - H_SCALE
SN_Z_MAX = Z0 + H_SCALE

# Safety margin over the measured post-shock cooling time (see
# _probe_post_shock_cooling_time below) -- long enough for the deposit to do
# real PdV work before cooling resumes, short compared to the ~1/SN_RATE_CODE
# inter-arrival time so the shield never covers a large fraction of the box
# between triggers. 50x worked cleanly in this feature's own unit test
# (pytests/stratified_ism -- see PROGRESS.md's 2026-09-15 entry); tune here
# if the run below still overcools or, conversely, never lets the shocked gas
# cool down again.
DELAY_MULTIPLIER = 50.0

# Problem-scale positivity floor (2026-09-15): the generic SimulationParams
# defaults (minimum_density/minimum_pressure = 1e-14) are ~25 trillion times
# smaller than this run's own ambient density (RHO_MIDPLANE_CODE ~ 0.325) --
# tried directly against the real M4 run and had zero effect, since flooring
# a near-vacuum cell all the way down to 1e-14 just creates an even sharper
# density discontinuity against its still-normal neighbors than the
# unfloored negative-pressure state did (see PROGRESS.md's 2026-09-15
# "continued still further" entry). MIN_DENSITY_CODE instead sits well below
# any density this run's SN-driven blow-out actually reaches (the lowest
# ambient n_H seen before the earlier NaN was ~0.43, i.e. code density
# ~0.014, and the pathological cell itself was ~6.9e-4) while staying only
# ~1-2 orders of magnitude below that pathological cell's own value, not
# many orders of magnitude below it. MIN_PRESSURE_CODE is derived from it at
# this module's own FLOOR_TEMPERATURE_CODE (already used by K&I cooling) so
# the floor represents a physically coherent "cold gas at floor density"
# state rather than an arbitrary number.
MIN_DENSITY_CODE = RHO_MIDPLANE_CODE * 1e-4
MIN_PRESSURE_CODE = float(get_pressure_from_temperature(
    MIN_DENSITY_CODE, FLOOR_TEMPERATURE_CODE, X_H, Z_METAL,
))


def _base_config() -> SimulationConfig:
    return SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        # VAN_ALBADA_PP was tried (2026-09-15) to address a low-density NaN
        # in the hydro evolve -- silently reverted to MINMOD by
        # finalize_config's own gravity guard (external_potential=True below
        # makes config.gravity_config.gravity True). split=SPLIT+
        # time_integrator=MUSCL was also tried as an alternative -- ran but
        # was inconclusive (never revisited the low-density regime).
        # Reverted to defaults (MINMOD, unsplit, RK2_SSP): the actual fix now
        # lives in _evolve_gas_state_unsplit_inner itself (a positivity
        # floor after the combined multi-axis conserved-state update -- see
        # its docstring/comment there and PROGRESS.md's 2026-09-15 entries).
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
            # 2026-09-15: without this, a run-long stall was found -- some
            # diffuse SN-blown-out cell is persistently in a fast-cooling
            # state, and _cfl_time_step's dt_cool term (a domain-wide
            # minimum) pins the *global* dt to that one cell's local cooling
            # time forever. See CoolingConfig.subcycle_stiff_cooling's
            # docstring and PROGRESS.md's 2026-09-15 entry.
            subcycle_stiff_cooling=True,
        ),
        sn_driving_config=SNDrivingConfig(sn_driving=True, delayed_cooling=True),
        progress_bar=True,
        return_snapshots=True,
        num_snapshots=40,
        snapshot_settings=SnapshotSettings(
            return_states=True,
            return_final_state=True,
            return_total_mass=True,
            return_internal_energy=True,
            return_kinetic_energy=True,
        ),
    )


def _potential_and_ic(config, registered_variables):
    helper_data = get_helper_data(config)
    z = helper_data.geometric_centers[..., 2]
    phi = C_POT * jnp.log(jnp.cosh((z - Z0) / H_SCALE))

    rho_guess = RHO_MIDPLANE_CODE * jnp.exp(-phi / C2_WARM)
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
    return phi, initial_state


def _probe_post_shock_cooling_time(config) -> float:
    """K&I cooling time (code units) at a fresh midplane SN deposit's own
    post-shock state -- a hand-built single deposit (same tanh-taper formula
    ``_inject_supernovae`` uses, duplicated rather than driving the real PRNG
    trigger -- same technique as ``sn_driving_with_cooling.py``'s
    ``_synthetic_post_injection_pressure``), evaluated on a small local patch
    around the box-center/midplane site, independent of resolution elsewhere
    in the box.
    """
    n_local = 24
    half_width = 3.0 * SN_INJECTION_RADIUS_CODE
    coords_1d = jnp.linspace(-half_width, half_width, n_local)
    dx, dy, dz = jnp.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    distance = jnp.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    smooth_width = SN_SMOOTH_CELLS * GRID_SPACING_CODE
    weight = 0.5 * (1.0 - jnp.tanh((distance - SN_INJECTION_RADIUS_CODE) / smooth_width))
    local_cell_volume = (2.0 * half_width / n_local) ** 3
    weight_integral = jnp.sum(weight) * local_cell_volume

    thermal_energy = SN_ENERGY_CODE  # sn_cr_fraction=0 for this run
    delta_pressure = thermal_energy * (GAMMA - 1.0) / weight_integral * weight

    p_ambient = float(get_pressure_from_temperature(
        RHO_MIDPLANE_CODE,
        _t_eq_fast(N_H_MIDPLANE) * KI_PARAMS.code_temperature_per_kelvin,
        X_H, Z_METAL,
    ))
    p_hot = p_ambient + delta_pressure
    t_hot_code = get_temperature_from_pressure(RHO_MIDPLANE_CODE, p_hot, X_H, Z_METAL)
    t_hot_max_code = float(jnp.max(t_hot_code))
    t_hot_max_kelvin = t_hot_max_code / KI_PARAMS.code_temperature_per_kelvin

    t_cool_code = float(cooling_time(
        jnp.array(RHO_MIDPLANE_CODE), jnp.array(t_hot_max_code), X_H, Z_METAL, GAMMA,
        config.cooling_config.cooling_curve_config, KI_PARAMS,
    ))
    t_cool_years = (t_cool_code * CODE_UNITS.code_time).to(u.yr).value
    print(
        f"[probe] midplane post-shock: T_hot={t_hot_max_kelvin:.3e} K, "
        f"t_cool={t_cool_years:.2f} yr = {t_cool_code:.6g} code"
    )
    return t_cool_code


def run_m4():
    config = _base_config()
    registered_variables = get_registered_variables(config)
    phi, initial_state = _potential_and_ic(config, registered_variables)
    config = finalize_config(config, initial_state.shape)

    t_cool_code = _probe_post_shock_cooling_time(config)
    sn_cooling_delay_time = DELAY_MULTIPLIER * t_cool_code
    delay_years = (sn_cooling_delay_time * CODE_UNITS.code_time).to(u.yr).value
    inter_arrival_years = (1.0 / SN_RATE_CODE * CODE_UNITS.code_time).to(u.yr).value
    print(
        f"sn_cooling_delay_time = {DELAY_MULTIPLIER:.0f} x t_cool = {delay_years:.2f} yr "
        f"({sn_cooling_delay_time:.6g} code); mean SN inter-arrival = "
        f"{inter_arrival_years:.2f} yr (ratio {delay_years / inter_arrival_years:.4f})"
    )
    print(
        f"T_DYN = {T_DYN_YEARS:.4e} yr; T_end = 2*T_DYN = "
        f"{(T_END * CODE_UNITS.code_time).to(u.yr).value:.4e} yr; "
        f"expected SN triggers over T_end ~= {SN_RATE_CODE * T_END:.2f}"
    )
    print(
        f"minimum_density = {MIN_DENSITY_CODE:.6g} code "
        f"({MIN_DENSITY_CODE / RHO_MIDPLANE_CODE:.3g} x RHO_MIDPLANE_CODE); "
        f"minimum_pressure = {MIN_PRESSURE_CODE:.6g} code "
        f"(cold gas at floor density, T={FLOOR_TEMPERATURE_KELVIN:.0f} K)"
    )

    sn_driving_params = SNDrivingParams(
        sn_rate=SN_RATE_CODE,
        sn_energy=SN_ENERGY_CODE,
        sn_cr_fraction=0.0,
        sn_injection_radius=SN_INJECTION_RADIUS_CODE,
        sn_smooth_cells=SN_SMOOTH_CELLS,
        sn_z_min=SN_Z_MIN,
        sn_z_max=SN_Z_MAX,
        sn_cooling_delay_time=sn_cooling_delay_time,
    )

    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        minimum_density=MIN_DENSITY_CODE,
        minimum_pressure=MIN_PRESSURE_CODE,
        cooling_params=CoolingParams(
            hydrogen_mass_fraction=X_H,
            metal_mass_fraction=Z_METAL,
            floor_temperature=FLOOR_TEMPERATURE_CODE,
            cooling_curve_params=KI_PARAMS,
        ),
        sn_driving_params=sn_driving_params,
    )
    params = params._replace(gravitational_potential=phi)

    print("Starting M4 run (delayed cooling)...")
    snapshot_data = time_integration(initial_state, config, params, registered_variables)

    time_points_years = (np.asarray(snapshot_data.time_points) * CODE_UNITS.code_time).to(u.yr).value
    states = snapshot_data.states  # (num_snapshots, num_vars, nx, ny, nz)

    print("\n--- per-snapshot summary ---")
    first_nan_idx = None
    for i, t_yr in enumerate(time_points_years):
        state_i = states[i]
        has_nan = bool(jnp.any(jnp.isnan(state_i)))
        if has_nan and first_nan_idx is None:
            first_nan_idx = i
        if has_nan:
            print(f"  snapshot {i:3d}  t={t_yr:12.1f} yr   NaN present")
            continue
        rho_i = state_i[registered_variables.density_index]
        p_i = state_i[registered_variables.pressure_index]
        t_i_kelvin = get_temperature_from_pressure(rho_i, p_i, X_H, Z_METAL) / KI_PARAMS.code_temperature_per_kelvin
        n_h_i = (rho_i * CODE_UNITS.code_density).to(u.g / u.cm ** 3).value / (MU_H * c.m_p.to(u.g).value)
        mid = N_XY // 2
        n_h_mid = float(n_h_i[mid, mid, N_Z // 2])
        print(
            f"  snapshot {i:3d}  t={t_yr:12.1f} yr   max(T)={float(jnp.max(t_i_kelvin)):12.3e} K   "
            f"n_H(midplane)={n_h_mid:8.3f} cm^-3"
        )

    if first_nan_idx is not None:
        print(f"\nRun went to NaN at snapshot {first_nan_idx} "
              f"(t~={time_points_years[first_nan_idx]:.1f} yr).")
    else:
        print("\nRun completed with no NaNs through t_end.")

    # Diagnostic plot: midplane n_H and max(T) over time.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 5))
    n_h_mid_series, t_max_series = [], []
    for i in range(len(time_points_years)):
        state_i = states[i]
        if bool(jnp.any(jnp.isnan(state_i))):
            n_h_mid_series.append(np.nan)
            t_max_series.append(np.nan)
            continue
        rho_i = state_i[registered_variables.density_index]
        p_i = state_i[registered_variables.pressure_index]
        t_i_kelvin = get_temperature_from_pressure(rho_i, p_i, X_H, Z_METAL) / KI_PARAMS.code_temperature_per_kelvin
        n_h_i = (rho_i * CODE_UNITS.code_density).to(u.g / u.cm ** 3).value / (MU_H * c.m_p.to(u.g).value)
        mid = N_XY // 2
        n_h_mid_series.append(float(n_h_i[mid, mid, N_Z // 2]))
        t_max_series.append(float(jnp.max(t_i_kelvin)))

    ax0.semilogy(time_points_years, n_h_mid_series, "-o", ms=3)
    ax0.set_xlabel("t [yr]")
    ax0.set_ylabel(r"midplane $n_H$ [cm$^{-3}$]")
    ax0.set_title("Midplane density vs. time")

    ax1.semilogy(time_points_years, t_max_series, "-o", ms=3, color="C1")
    ax1.set_xlabel("t [yr]")
    ax1.set_ylabel("max(T) [K]")
    ax1.set_title("Peak temperature vs. time (SN heating survives if this spikes)")

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    out_path = pics_dir / "m4_stratified_column_sn_driving_delayed_cooling.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nDiagnostic plot written to {out_path}")

    return snapshot_data


if __name__ == "__main__":
    run_m4()
