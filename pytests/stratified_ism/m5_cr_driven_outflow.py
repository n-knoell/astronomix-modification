"""
SILCC-ISM project milestone M5: M4's stratified column + episodic SN driving
(delayed-cooling overcooling mitigation) + K&I cooling, with grey two-moment
cosmic-ray transport now turned ON -- the code-comparison anchor against
Girichidis et al. (2016), ApJL 816, L19, "Launching Cosmic-Ray-driven
Outflows from the Magnetized Interstellar Medium" (arXiv:1509.07247).

**NOT a committed pytest -- an exploratory script**, same status as M4's own
script (``m4_stratified_column_sn_driving_delayed_cooling.py``): meant to be
run and inspected directly, not asserted against fixed tolerances. The
box/resolution/SN-rate choices below (reused unchanged from M2/M3/M3.5/M4)
are already a cost-scoped-down approximation to Girichidis 2016's own much
larger box -- this milestone was scoped (2026-09-16, discussed with the
user) as a *qualitative* code-comparison, not a precision reproduction.

Design decisions for this milestone, discussed with the user before
implementation:

1. **Reuse M4's exact box/potential/IC/SN-driving/cooling stack unchanged,
   turning CR-DSA-style direct injection on** (rather than building a fresh
   literature-scale box) -- M4's stack just finished a long stabilization
   fight and is verified stable end-to-end; adding CR transport is additive
   physics on top of it, not a new numerical-stability project.
2. **M0a/M0b's known, unfixed FV self-gravity bias (~10% Mach / ~2% density,
   see PROGRESS.md's 2026-09-16 entry) is NOT fixed before this run** -- it
   is one more documented systematic alongside the box-size/SN-rate
   deviations from the paper, not a blocker; revisit only if it changes the
   qualitative conclusion.

Cosmic-ray physics parameters, derived from the paper (not guessed) via
``astropy`` unit conversion against this script's own ``CODE_UNITS``, same
pattern as ``KI_PARAMS``/``FLOOR_TEMPERATURE_CODE`` below:

- ``sn_cr_fraction = 0.1``: Girichidis et al. (2016)'s fiducial CR energy
  injection per SN, 1e50 erg = 10% of the fiducial 1e51 erg thermal SN
  energy -- already this module's own ``SNDrivingParams.sn_cr_fraction``
  default (``sn_driving_options.py``), M4 was the one that overrode it to
  0.0; this script just doesn't override it.
- ``diffusion_coefficient``: **attempt 1 used kappa_parallel = 1e28 cm^2/s
  (the paper's along-field-lines value) as an isotropic stand-in and this
  was a real mistake, not just an approximation** -- applied isotropically
  (including the vertical/confining direction, where the paper's own
  kappa_perp = 1e26 cm^2/s, 100x smaller, is what actually keeps a vertical
  CR pressure gradient), the diffusion length over T_end
  (``sqrt(kappa*T_end) ~= 495 pc``) came out far bigger than this box's own
  300 pc height, so CR pressure fully homogenized (measured <0.1 dex
  variation end to end) and both H_cr and the mass-loading/outflow-velocity
  comparison were meaningless (confirmed directly from attempt 1's pressure-
  profile plot -- see PROGRESS.md's 2026-09-16 M5 attempt-1 entry). **Fixed
  (attempt 2, discussed with the user): use kappa_perp = 1e26 cm^2/s
  instead**, isotropically -- not because it is isotropically correct
  either (it still is not; real anisotropic transport is M6), but because
  it gives ``sqrt(kappa_perp*T_end) ~= 49.5 pc``, close to one scale height
  (``H_SCALE = 50`` pc), the physically sensible regime for *this* box size
  -- confined enough for a real gradient to survive the run, not so confined
  that nothing moves at all. This is the paper's own quoted value, not a
  free hand-tune.
- ``reduced_streaming_speed``: attempt 1 picked 3000 km/s (see the same
  reasoning below); **attempt 2 scales it down to 300 km/s alongside the
  100x-smaller diffusion_coefficient, chosen to hold the relaxation rate
  ``nu = reduced_streaming_speed^2 / diffusion_coefficient`` (and therefore
  the explicit stability bound ``dt <~ diffusion_coefficient /
  reduced_streaming_speed^2`` this term imposes on ``_cfl_time_step``, see
  ``cr_flux_relaxation_source``'s docstring) fixed at attempt 1's own value**
  -- so this recalibration should not materially change per-step cost
  relative to the run that already completed in ~80 minutes. This still
  clears the diffusive-limit floor sqrt(diffusion_coefficient / T_DYN) ~= 9.3
  km/s by the same 32x margin as attempt 1. It does **not** clear attempt 1's
  other margin, however -- 300 km/s is well below the ~1500 km/s sound speed
  seen in M4's most violent disruption spikes (T~1e8 K) -- a real, accepted
  trade-off for this run: attempt 2's own actual peak temperature never
  exceeded ~2e6 K (sound speed ~210 km/s, so 300 km/s still exceeds the
  gas signal speeds *this particular run* produced, with a thin ~1.4x
  margin), but a run that reached M4-scale disruption temperatures could
  under-resolve CR transport at 300 km/s. Not re-checked via a fresh
  convergence study; DESIGN.md's open question remains open.

Diagnostics computed (Girichidis et al. 2016's own reported quantities,
Sec. 3-4 there): mass-loading factor, outflow velocity, and the CR-vs-gas
pressure vertical scale height, all evaluated at ``Z0 + H_SCALE`` (one scale
height above the midplane) rather than the paper's fixed "1 kpc" -- our box
is ~13x shorter than theirs, so a box-relative height is the natural
analogue, not a literal match. Mass-loading factor needs a star-formation
rate this setup does not track directly (SN driving here is a prescribed
Poisson rate, not tied to a self-consistent SF model) -- converted via the
standard Chabrier/Kroupa-IMF core-collapse assumption of ~100 Msun of stars
formed per SN (a documented assumption, not independently verified against
the paper's own conversion).

Attempt 3 (kappa_perp + the ``_apply_gravity_source`` positivity-floor fix,
see ``_finite_volume/_state_evolution/evolve_state.py``) completed
successfully: no NaNs, H_cr/H_gas ~= 3.1 (reproduces the paper's headline
qualitative claim), mass-loading/outflow-velocity within an order of
magnitude of the paper -- see PROGRESS.md's 2026-09-16 M5 entries for the
full numbers.

**Attempt 4 (not yet run -- resolution doubled per-axis, ``_RESOLUTION_FACTOR``
below, at the user's request to better match Girichidis 2016's own
resolution; box size unchanged). Also new: ``_girichidis_fig1_style_plot``
now uses physical units and midplane/box-centered axes to more directly
match the user-provided reference figure (``pics/girichidis_SN.jpeg``,
Girichidis et al. 2016 Fig. 1) rather than code units and this codebase's
native [0, box_size) convention.** See ``_RESOLUTION_FACTOR``'s own comment
for the memory/cost math (~8x the cells, ~16x attempt 3's wall-clock time)
-- deliberately not launched by the same session that made this change; the
resulting memory footprint (~8x attempt 3's ~8288 MiB) does not fit on this
cluster's 2080 Ti nodes (11 GB each), so this needs bigger/different
hardware before it can run at all.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1, interval=10)
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
from matplotlib.colors import LogNorm

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
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# independent reference (plain numpy/scipy, no astronomix/JAX)
from astronomix.test_setups.reference_solutions.koyama_inutsuka_equilibrium import find_equilibrium_temperatures

# ---- physical setup (identical to M2/M3/M3.5/M4) ----
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
# Resolution doubled in each of the 3 dimensions (2026-09-16, attempt 3's own
# run measured ~8288 MiB on an RTX 2080 Ti) -- box size held fixed (same
# physical setup as attempts 1-3, still a cost-scoped-down approximation to
# Girichidis 2016's box; only resolution changes here), so this codebase's
# own uniform-grid-spacing requirement (grid_spacing = L_XY/N_XY = L_Z/N_Z,
# enforced in simulation_helper_data.py's _normalize_config_vectors) forces
# N_XY and N_Z to scale by the *same* factor. Total cell count scales as
# that factor cubed, so a factor of 2 gives 2^3=8x the cells -- and, since
# per-cell state-array memory dominates the GPU footprint, roughly 8x the
# memory of attempt 3's run (~8288 MiB -> ~66 GB), matching a GPU "with 8x
# the capacity" (no single GPU on this cluster's 2080 Ti nodes has that much
# VRAM -- this needs bigger/different hardware, hence not run here).
# **Not just 8x more expensive: dx also halves, which roughly halves the
# hydro CFL step too, so expect total wall-clock cost to grow by something
# closer to ~8x * ~2x = ~16x attempt 3's ~43 minutes (i.e. many hours), not
# just 8x** -- budget GPU time accordingly before launching this.
_RESOLUTION_FACTOR = 2
N_XY, N_Z = 32 * _RESOLUTION_FACTOR, 256 * _RESOLUTION_FACTOR
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


# ---- M4-inherited setup ----
T_END = 2.0 * T_DYN

RHO_MIDPLANE_CODE = float((N_H_MIDPLANE / u.cm ** 3 * MU_H * c.m_p).to(CODE_UNITS.code_density).value)

EXPECTED_TRIGGERS_PER_T_DYN = 8.0
SN_RATE_CODE = EXPECTED_TRIGGERS_PER_T_DYN / T_DYN

SN_ENERGY_CODE = (1.0e51 * u.erg).to(CODE_UNITS.code_energy).value
SN_INJECTION_RADIUS_CODE = (6.0 * u.pc).to(CODE_UNITS.code_length).value
SN_SMOOTH_CELLS = 2.0
SN_Z_MIN = Z0 - H_SCALE
SN_Z_MAX = Z0 + H_SCALE

DELAY_MULTIPLIER = 50.0

MIN_DENSITY_CODE = RHO_MIDPLANE_CODE * 1e-4
MIN_PRESSURE_CODE = float(get_pressure_from_temperature(
    MIN_DENSITY_CODE, FLOOR_TEMPERATURE_CODE, X_H, Z_METAL,
))

# ---- M5-specific: CR-grey physics (Girichidis et al. 2016 parameters) ----
SN_CR_FRACTION = 0.1  # Girichidis et al. (2016)'s fiducial fCR

# Attempt 2 (2026-09-16): kappa_perp, not kappa_parallel -- attempt 1's
# kappa_parallel=1e28 cm^2/s isotropic stand-in over-diffused CR pressure to
# a flat profile in this box (diffusion length ~495 pc >> 300 pc box height).
# kappa_perp gives a diffusion length ~= one scale height over T_end instead
# -- see the module docstring's "diffusion_coefficient" entry for the numbers.
KAPPA_PERP_CGS = 1.0e26 * u.cm ** 2 / u.s  # Girichidis et al. (2016), Sec. 2
DIFFUSION_COEFFICIENT_CODE = KAPPA_PERP_CGS.to(
    CODE_UNITS.code_length ** 2 / CODE_UNITS.code_time
).value

# Scaled down from attempt 1's 3000 km/s by the same factor as
# diffusion_coefficient dropped (100x), so the relaxation rate nu =
# reduced_streaming_speed^2/diffusion_coefficient -- and hence the explicit
# stability bound this source term imposes on the timestep -- is unchanged
# from attempt 1's already-verified, tractable ~80 minute run. See the
# module docstring's "reduced_streaming_speed" entry for the accepted
# trade-off (no longer safely above M4's most extreme disruption sound
# speed, only above this run's own realized peak).
REDUCED_STREAMING_SPEED_PHYS = 300.0 * u.km / u.s
REDUCED_STREAMING_SPEED_CODE = REDUCED_STREAMING_SPEED_PHYS.to(CODE_UNITS.code_velocity).value

GAMMA_CR = 4.0 / 3.0

# Reference height for the outflow/scale-height diagnostics: one scale
# height above the midplane (box-relative analogue of the paper's fixed
# "1 kpc", since this box's total half-height is only 3*H_SCALE).
Z_REF_CODE = Z0 + H_SCALE

# Standard Chabrier/Kroupa-IMF core-collapse-SN yield used to convert the
# prescribed SN rate into an equivalent SFR for the mass-loading factor
# (this setup has no separate star-formation model to read an SFR from
# directly) -- a documented assumption, not from the paper itself.
M_STAR_PER_SN_CODE = (100.0 * u.M_sun).to(CODE_UNITS.code_mass).value
SFR_EQUIVALENT_CODE = SN_RATE_CODE * M_STAR_PER_SN_CODE  # code mass / code time


def _base_config() -> SimulationConfig:
    return SimulationConfig(
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
            subcycle_stiff_cooling=True,
        ),
        sn_driving_config=SNDrivingConfig(sn_driving=True, delayed_cooling=True),
        # M5: grey two-moment CR transport, diffusive (Fick's-law) relaxation
        # on -- streaming stays off, matching Girichidis et al. (2016)'s own
        # diffusion-only model (their streaming follow-up is a later paper).
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_relaxation=True,
        ),
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
    return phi, initial_state, helper_data


def _probe_post_shock_cooling_time(config) -> float:
    """K&I cooling time (code units) at a fresh midplane SN deposit's own
    post-shock state, accounting for M5's nonzero ``sn_cr_fraction`` (only
    ``(1 - sn_cr_fraction)`` of ``sn_energy`` goes into the thermal deposit
    this probes -- M4's version of this function assumed sn_cr_fraction=0).
    Same technique as M4/``sn_driving_with_cooling.py``'s own probes: a
    hand-built single deposit, independent of any PRNG trigger.
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

    thermal_energy = (1.0 - SN_CR_FRACTION) * SN_ENERGY_CODE
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
        f"[probe] midplane post-shock (thermal fraction only): T_hot={t_hot_max_kelvin:.3e} K, "
        f"t_cool={t_cool_years:.2f} yr = {t_cool_code:.6g} code"
    )
    return t_cool_code


def _mass_loading_and_outflow_velocity(state, registered_variables, z_index):
    """Mass-outflow rate and mass-flux-weighted outflow velocity through the
    horizontal plane at ``z_index``, plus the mass-loading factor relative
    to ``SFR_EQUIVALENT_CODE``. Returns plain Python floats.
    """
    rho = np.asarray(state[registered_variables.density_index])
    vz = np.asarray(state[registered_variables.velocity_index.z])
    cell_area = GRID_SPACING_CODE ** 2

    rho_slice = rho[:, :, z_index]
    vz_slice = vz[:, :, z_index]
    outflow_mask = vz_slice > 0.0

    flux = np.where(outflow_mask, rho_slice * vz_slice, 0.0)
    mdot_out = float(np.sum(flux) * cell_area)

    mass_out = np.where(outflow_mask, rho_slice, 0.0)
    v_out_mean = float(np.sum(flux) / np.sum(mass_out)) if np.any(outflow_mask) else 0.0

    eta = mdot_out / SFR_EQUIVALENT_CODE
    return mdot_out, v_out_mean, eta


def _pressure_scale_heights(state, registered_variables, z_axis_code, mid_index):
    """Horizontally-averaged gas- and CR-pressure e-folding scale heights,
    measured upward from ``mid_index`` (the midplane cell). Returns
    (H_gas_code, H_cr_code); NaN if the profile never drops to 1/e within
    the domain.
    """
    p_gas = np.asarray(state[registered_variables.pressure_index])
    e_cr = np.asarray(state[registered_variables.cosmic_ray_e_index])
    p_cr = (GAMMA_CR - 1.0) * e_cr

    p_gas_z = p_gas.mean(axis=(0, 1))
    p_cr_z = p_cr.mean(axis=(0, 1))

    def _scale_height(profile_1d):
        p0 = profile_1d[mid_index]
        if p0 <= 0.0:
            return float("nan")
        target = p0 / np.e
        upper = profile_1d[mid_index:]
        z_upper = z_axis_code[mid_index:]
        below = np.nonzero(upper <= target)[0]
        if below.size == 0:
            return float("nan")
        idx = below[0]
        if idx == 0:
            return float(z_upper[0] - z_axis_code[mid_index])
        # linear interpolation in log(p) vs z between idx-1 and idx
        p_a, p_b = upper[idx - 1], upper[idx]
        z_a, z_b = z_upper[idx - 1], z_upper[idx]
        if p_a <= 0.0 or p_b <= 0.0:
            return float(z_b - z_axis_code[mid_index])
        frac = (np.log(p_a) - np.log(target)) / (np.log(p_a) - np.log(p_b))
        z_target = z_a + frac * (z_b - z_a)
        return float(z_target - z_axis_code[mid_index])

    return _scale_height(p_gas_z), _scale_height(p_cr_z), p_gas_z, p_cr_z


def _midplane_clumpiness(state, registered_variables, mid_index):
    """std(log10(density)) across the midplane face-on slice -- a simple
    proxy for Simpson et al. (2016)'s qualitative "smooth" (density-weighted
    placement) vs. "clumpy" (random placement) structure comparison.
    Identical to ``m6_sn_placement_comparison.py``'s function of the same
    name, added here (2026-09-22) to close PROGRESS.md's flagged M6-vs-M5
    clumpiness gap -- M5's original attempt-3 run never saved states, so
    this needed a fresh run with the diagnostic in place, not a retrofit
    onto old data. See this module's docstring -- not a metric from the
    paper itself.
    """
    rho = np.asarray(state[registered_variables.density_index])[:, :, mid_index]
    log_rho = np.log10(np.maximum(rho, 1e-30))
    return float(np.std(log_rho))


def _clumpiness_evolution_plot(
    states, time_points_years, clumpiness_series, registered_variables, mid_index,
    out_path, title_prefix,
):
    """Small filmstrip (~4 snapshots) of the midplane face-on density field,
    from the initial condition through the clumpiness *peak* (the onset/
    collapse transient) to the late-time quasi-steady state.

    Added 2026-09-22 alongside the M5-vs-M6 clumpiness finding it exists to
    illustrate: the two placement modes' *late-time-average* clumpiness came
    out statistically tied (0.467 both), which doesn't show Simpson et al.
    (2016)'s predicted "density-weighted -> smoother" signature -- but the
    signature *does* show up as a much higher transient peak during the
    onset/collapse phase for random placement, which a single late-time
    number can't convey. This filmstrip makes that transient visible
    directly, rather than relying on the reader to infer it from the
    ``clump=`` column of the per-snapshot printout. See PROGRESS.md's
    2026-09-22 entry for the full numbers.

    The peak snapshot is found dynamically (``nanargmax`` of the actual
    ``clumpiness_series``), not a hardcoded index, so this keeps working if
    the run parameters (resolution, SN rate, ...) ever change and shift
    which snapshot the peak lands on.
    """
    clump_arr = np.asarray(clumpiness_series)
    valid_idx = np.where(~np.isnan(clump_arr))[0]
    if valid_idx.size == 0:
        print(f"  [warning] skipping clumpiness evolution plot -- no valid (non-NaN) snapshots")
        return

    idx_first = int(valid_idx[0])
    idx_last = int(valid_idx[-1])
    idx_peak = int(valid_idx[np.argmax(clump_arr[valid_idx])])
    mid_target = (idx_peak + idx_last) // 2
    idx_settle = int(valid_idx[np.searchsorted(valid_idx, mid_target)])

    indices = sorted(dict.fromkeys([idx_first, idx_peak, idx_settle, idx_last]))
    candidates = list(valid_idx)
    while len(indices) < 4 and len(indices) < len(candidates):
        gaps = [(indices[i + 1] - indices[i], indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
        gaps.sort(reverse=True)
        _, lo, hi = gaps[0]
        mid = int(candidates[np.searchsorted(candidates, (lo + hi) // 2)])
        if mid in indices:
            break
        indices.append(mid)
        indices.sort()

    fig, axes = plt.subplots(1, len(indices), figsize=(4.3 * len(indices), 4.6))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        rho = np.asarray(states[idx][registered_variables.density_index])[:, :, mid_index]
        vmin = max(float(rho.min()), 1e-30)
        vmax = max(float(rho.max()), vmin * 10)
        im = ax.imshow(
            rho.T, origin="lower", cmap="magma", aspect="equal",
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        t_myr = time_points_years[idx] / 1e6
        tag = " (peak)" if idx == idx_peak else (" (final)" if idx == idx_last else "")
        ax.set_title(f"t={t_myr:.2f} Myr{tag}\nclump={clump_arr[idx]:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\rho$ [code]")

    fig.suptitle(f"{title_prefix}: midplane density, clumpiness evolution")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _girichidis_fig1_style_plot(state, registered_variables, mid_index, t_yr, out_path):
    """Structure plot styled directly after Girichidis et al. (2016) Fig. 1
    (``pics/girichidis_SN.jpeg``, provided by the user -- a 3x3 grid: edge-on
    density / face-on density / midplane CR energy density rows, one column
    per feedback mode: thermal-only, CR-only, thermal+CR). We only have one
    run (thermal+CR, their rightmost "SNe: thermal + CR" column), so this is
    one column, not three -- M6 (MHD/anisotropic transport) is the only
    planned follow-up that would add a second physically distinct
    configuration to compare a second column against.

    Matches the reference figure's presentation, not just its layout:
    physical units (density in g/cm^3, CR energy density in erg/cm^3, via
    this script's own CODE_UNITS -- not code units) so the color-scale range
    is directly comparable to the paper's own colorbars, and axes centered on
    the midplane/box center (paper: x,y in [-1,1] kpc, z in [-1.5,1.5] kpc)
    rather than this codebase's native [0, box_size) convention.
    """
    rho_code = np.asarray(state[registered_variables.density_index])
    e_cr_code = np.asarray(state[registered_variables.cosmic_ray_e_index])

    if not (np.any(rho_code > 0) and np.any(e_cr_code > 0)):
        # Defensive fallback, not expected to trigger given run_m5's own
        # corrupted-state filtering before this is called -- but a degenerate
        # (all non-positive) density or e_cr field makes LogNorm's vmin/vmax
        # invalid (vmax < vmin), which crashes matplotlib's colorbar deep
        # inside its callback chain rather than failing cleanly here.
        print(f"  [warning] skipping structure plot at t={t_yr:.1f} yr -- "
              f"density or e_cr field is degenerate (all non-positive)")
        return

    # Physical units, matching the reference figure's own colorbars.
    rho = (rho_code * CODE_UNITS.code_density).to(u.g / u.cm ** 3).value
    e_cr_density_unit = CODE_UNITS.code_energy / CODE_UNITS.code_length ** 3
    e_cr = (e_cr_code * e_cr_density_unit).to(u.erg / u.cm ** 3).value

    # Centered axes: x, y relative to the box center; z relative to the
    # midplane (Z0) -- matches the reference figure's [-1,1]/[-1.5,1.5] kpc
    # convention (ours is pc-scale, not kpc, given this box's own size).
    x_extent = [-L_XY / 2, L_XY / 2]
    y_extent = [-L_XY / 2, L_XY / 2]
    z_extent = [-Z0, L_Z - Z0]

    y_mid = N_XY // 2
    edge_on_density = rho[:, y_mid, :].T  # (z, x), z vertical
    face_on_density = rho[:, :, mid_index].T  # (y, x)
    face_on_e_cr = e_cr[:, :, mid_index].T  # (y, x)

    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(6, 15))

    im0 = ax0.imshow(
        edge_on_density, origin="lower", extent=x_extent + z_extent,
        norm=LogNorm(vmin=max(edge_on_density.min(), 1e-30), vmax=edge_on_density.max()),
        cmap="magma", aspect="auto",
    )
    ax0.axhline(0.0, color="cyan", ls=":", lw=1)
    ax0.set_xlabel("x [pc]")
    ax0.set_ylabel("z [pc] (0 = midplane)")
    ax0.set_title("Density, edge-on (y=0)")
    fig.colorbar(im0, ax=ax0, label=r"density [g cm$^{-3}$]")

    im1 = ax1.imshow(
        face_on_density, origin="lower", extent=x_extent + y_extent,
        norm=LogNorm(vmin=max(face_on_density.min(), 1e-30), vmax=face_on_density.max()),
        cmap="magma", aspect="equal",
    )
    ax1.set_xlabel("x [pc]")
    ax1.set_ylabel("y [pc]")
    ax1.set_title("Density, face-on (midplane)")
    fig.colorbar(im1, ax=ax1, label=r"density [g cm$^{-3}$]")

    e_cr_floor = max(float(face_on_e_cr.max()) * 1e-8, 1e-30)
    im2 = ax2.imshow(
        face_on_e_cr, origin="lower", extent=x_extent + y_extent,
        norm=LogNorm(vmin=max(face_on_e_cr.min(), e_cr_floor), vmax=max(face_on_e_cr.max(), e_cr_floor * 10)),
        cmap="viridis", aspect="equal",
    )
    ax2.set_xlabel("x [pc]")
    ax2.set_ylabel("y [pc]")
    ax2.set_title("CR energy density, midplane")
    fig.colorbar(im2, ax=ax2, label=r"$E_{CR}$ [erg cm$^{-3}$]")

    t_myr = t_yr / 1e6
    fig.suptitle(
        f"SNe: thermal + CR  |  Time: {t_myr:.1f} Myr\n"
        f"(styled after Girichidis et al. 2016 Fig. 1, pics/girichidis_SN.jpeg)"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_m5():
    config = _base_config()
    registered_variables = get_registered_variables(config)
    phi, initial_state, helper_data_unpadded = _potential_and_ic(config, registered_variables)
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
    diffusion_length_pc = float(np.sqrt(DIFFUSION_COEFFICIENT_CODE * T_END))
    print(
        f"diffusion_coefficient = {DIFFUSION_COEFFICIENT_CODE:.6g} code "
        f"(kappa_perp=1e26 cm^2/s); reduced_streaming_speed = "
        f"{REDUCED_STREAMING_SPEED_CODE:.6g} code (= km/s); "
        f"diffusive-limit floor sqrt(D/T_DYN) = "
        f"{np.sqrt(DIFFUSION_COEFFICIENT_CODE / T_DYN):.4g} km/s; "
        f"diffusion length over T_end sqrt(D*T_end) = {diffusion_length_pc:.1f} pc "
        f"(box height = {L_Z:.0f} pc, H_SCALE = {H_SCALE:.0f} pc)"
    )

    sn_driving_params = SNDrivingParams(
        sn_rate=SN_RATE_CODE,
        sn_energy=SN_ENERGY_CODE,
        sn_cr_fraction=SN_CR_FRACTION,
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
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR,
            diffusion_coefficient=DIFFUSION_COEFFICIENT_CODE,
            reduced_streaming_speed=REDUCED_STREAMING_SPEED_CODE,
        ),
    )
    params = params._replace(gravitational_potential=phi)

    print("Starting M5 run (CR-grey diffusion + SN driving + delayed cooling)...")
    snapshot_data = time_integration(initial_state, config, params, registered_variables)

    time_points_years = (np.asarray(snapshot_data.time_points) * CODE_UNITS.code_time).to(u.yr).value
    states = snapshot_data.states  # (num_snapshots, num_vars, nx, ny, nz)

    z_axis_code = np.asarray(helper_data_unpadded.geometric_centers[0, 0, :, 2])
    mid_index = int(np.argmin(np.abs(z_axis_code - Z0)))
    z_ref_index = int(np.argmin(np.abs(z_axis_code - Z_REF_CODE)))

    print("\n--- per-snapshot summary ---")
    first_nan_idx = None
    n_h_mid_series, t_max_series = [], []
    mdot_series, v_out_series, eta_series = [], [], []
    h_gas_series, h_cr_series = [], []
    clumpiness_series = []
    for i, t_yr in enumerate(time_points_years):
        state_i = states[i]
        # A literal jnp.isnan hit is not the only corruption signature this
        # project has seen (see PROGRESS.md's M4 NaN investigations): once
        # dt/current_time themselves go bad, later snapshot slots can come
        # back as an all-zero fill (t=0, no NaN anywhere) rather than NaN --
        # density is never exactly 0 in a real run (MIN_DENSITY_CODE > 0 is
        # an unconditional floor), so treat that as corrupted too, not just
        # literal NaN (attempt 2 (2026-09-16) hit exactly this: has_nan alone
        # stayed False for 34 straight degenerate snapshots after the real
        # failure, silently feeding a zeroed-out state into the scale-height/
        # mass-loading diagnostics and crashing the structure plot's colorbar
        # on a degenerate all-zero array).
        has_nan = bool(jnp.any(jnp.isnan(state_i)))
        looks_corrupted = has_nan or bool(jnp.all(state_i[registered_variables.density_index] == 0))
        if looks_corrupted and first_nan_idx is None:
            first_nan_idx = i
        if looks_corrupted:
            print(f"  snapshot {i:3d}  t={t_yr:12.1f} yr   NaN/corrupted-state present")
            n_h_mid_series.append(np.nan)
            t_max_series.append(np.nan)
            mdot_series.append(np.nan)
            v_out_series.append(np.nan)
            eta_series.append(np.nan)
            h_gas_series.append(np.nan)
            h_cr_series.append(np.nan)
            clumpiness_series.append(np.nan)
            continue

        rho_i = state_i[registered_variables.density_index]
        p_i = state_i[registered_variables.pressure_index]
        t_i_kelvin = get_temperature_from_pressure(rho_i, p_i, X_H, Z_METAL) / KI_PARAMS.code_temperature_per_kelvin
        n_h_i = (np.asarray(rho_i) * CODE_UNITS.code_density).to(u.g / u.cm ** 3).value / (MU_H * c.m_p.to(u.g).value)
        mid = N_XY // 2
        n_h_mid = float(n_h_i[mid, mid, N_Z // 2])

        mdot_out, v_out_mean, eta = _mass_loading_and_outflow_velocity(
            state_i, registered_variables, z_ref_index
        )
        h_gas, h_cr, _, _ = _pressure_scale_heights(
            state_i, registered_variables, z_axis_code, mid_index
        )
        clumpiness = _midplane_clumpiness(state_i, registered_variables, mid_index)

        n_h_mid_series.append(n_h_mid)
        t_max_series.append(float(jnp.max(t_i_kelvin)))
        mdot_series.append(mdot_out)
        v_out_series.append(v_out_mean)
        eta_series.append(eta)
        h_gas_series.append(h_gas)
        h_cr_series.append(h_cr)
        clumpiness_series.append(clumpiness)

        print(
            f"  snapshot {i:3d}  t={t_yr:12.1f} yr   max(T)={float(jnp.max(t_i_kelvin)):10.3e} K   "
            f"n_H(mid)={n_h_mid:8.3f} cm^-3   Mdot_out={mdot_out: .3e}   "
            f"v_out={v_out_mean:8.2f} km/s   eta={eta: .3e}   "
            f"H_gas={h_gas:8.2f}   H_cr={h_cr:8.2f}   clump={clumpiness:6.3f}"
        )

    if first_nan_idx is not None:
        last_good_t = time_points_years[first_nan_idx - 1] if first_nan_idx > 0 else 0.0
        print(f"\nRun went to NaN/corrupted state at snapshot {first_nan_idx} -- last good "
              f"snapshot {first_nan_idx - 1} was at t={last_good_t:.1f} yr, so the failure "
              f"happened between there and this snapshot's nominal time.")
    else:
        print("\nRun completed with no NaNs through t_end.")

    # Late-time comparison to Girichidis et al. (2016)'s reported ranges
    # (mass loading ~ order unity, outflow velocity ~10-50 km/s at their
    # reference height, H_cr > H_gas): average over the last quarter of the
    # (NaN-free) snapshots, where the run has had time to settle into a
    # quasi-steady CR-driven state.
    valid = ~np.isnan(np.asarray(eta_series))
    if np.any(valid):
        tail = np.where(valid)[0][-max(1, int(0.25 * np.sum(valid))):]
        print("\n--- late-time (last quarter of valid snapshots) averages ---")
        print(f"  mass-loading factor eta = {np.nanmean(np.asarray(eta_series)[tail]):.3e} "
              f"(Girichidis 2016: order unity)")
        print(f"  outflow velocity v_out  = {np.nanmean(np.asarray(v_out_series)[tail]):.2f} km/s "
              f"(Girichidis 2016: 10-50 km/s at their 1 kpc reference height)")
        print(f"  H_gas = {np.nanmean(np.asarray(h_gas_series)[tail]):.2f} code, "
              f"H_cr = {np.nanmean(np.asarray(h_cr_series)[tail]):.2f} code "
              f"(Girichidis 2016: H_cr > H_gas)")
        print(f"  midplane clumpiness = {np.nanmean(np.asarray(clumpiness_series)[tail]):.3f} "
              f"(M6_SN_PLACEMENT_COMPARISON's own M6_CLUMPINESS constant, if set, is the "
              f"density-weighted-placement counterpart to compare against)")

    # Diagnostic plots.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax0, ax1, ax2, ax3 = axes.flat

    ax0.semilogy(time_points_years, n_h_mid_series, "-o", ms=3)
    ax0.set_xlabel("t [yr]")
    ax0.set_ylabel(r"midplane $n_H$ [cm$^{-3}$]")
    ax0.set_title("Midplane density vs. time")

    ax1.semilogy(time_points_years, t_max_series, "-o", ms=3, color="C1")
    ax1.set_xlabel("t [yr]")
    ax1.set_ylabel("max(T) [K]")
    ax1.set_title("Peak temperature vs. time")

    ax2.plot(time_points_years, v_out_series, "-o", ms=3, color="C2")
    ax2.set_xlabel("t [yr]")
    ax2.set_ylabel(r"$v_{out}$ [km/s] at $z=Z_0+H$")
    ax2.set_title("Outflow velocity vs. time")

    ax3.semilogy(time_points_years, np.abs(eta_series), "-o", ms=3, color="C3")
    ax3.set_xlabel("t [yr]")
    ax3.set_ylabel(r"mass-loading factor $\eta$")
    ax3.set_title("Mass loading vs. time")

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    out_path = pics_dir / "m5_cr_driven_outflow.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nDiagnostic plot written to {out_path}")

    # CR-vs-gas pressure profile at the final valid snapshot.
    if np.any(valid):
        last_valid = np.where(valid)[0][-1]
        _, _, p_gas_z, p_cr_z = _pressure_scale_heights(
            states[last_valid], registered_variables, z_axis_code, mid_index
        )
        fig2, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(z_axis_code, p_gas_z, label="gas pressure")
        ax.semilogy(z_axis_code, p_cr_z, label="CR pressure")
        ax.axvline(Z0, color="k", ls=":", lw=1, label="midplane")
        ax.set_xlabel("z [code = pc]")
        ax.set_ylabel("horizontally-averaged pressure [code]")
        ax.set_title(f"Pressure profiles at snapshot {last_valid} "
                      f"(t={time_points_years[last_valid]:.1f} yr)")
        ax.legend()
        fig2.tight_layout()
        out_path2 = pics_dir / "m5_pressure_profiles.svg"
        fig2.savefig(out_path2)
        plt.close(fig2)
        print(f"Pressure-profile plot written to {out_path2}")

        out_path3 = pics_dir / "m5_structure_girichidis_fig1_style.svg"
        _girichidis_fig1_style_plot(
            states[last_valid], registered_variables, mid_index,
            time_points_years[last_valid], out_path3,
        )
        print(f"Girichidis-Fig.1-style structure plot written to {out_path3}")

    out_path4 = pics_dir / "m5_clumpiness_evolution.svg"
    _clumpiness_evolution_plot(
        states, time_points_years, clumpiness_series, registered_variables, mid_index,
        out_path4, title_prefix="M5: SNe random placement",
    )
    print(f"Clumpiness-evolution filmstrip written to {out_path4}")

    return snapshot_data


if __name__ == "__main__":
    run_m5()
