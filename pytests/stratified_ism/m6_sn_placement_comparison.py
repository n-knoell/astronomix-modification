"""
SILCC-ISM project milestone M6: Simpson et al. (2016), ApJL 827, L29,
"The Role of Cosmic-Ray Pressure in Accelerating Galactic Outflows"
(arXiv:1606.02324) -- the SN-placement comparison branch of ladder item 12's
optional M6 stretch (user picked this over MHD + anisotropic CR diffusion,
2026-09-16 -- see the saved plan `~/.claude/plans/memoized-discovering-scone.md`
for the full "and/or" framing).

**Density-weighted ("density peak") SN site selection, instead of M5's
uniformly-random placement -- everything else held identical to M5 attempt 3**
(same box, K&I cooling, delayed-cooling mitigation, subcycled cooling, and CR
-grey diffusion parameters: kappa_perp=1e26 cm^2/s, reduced_streaming_speed=
300 km/s), at M5 attempt 3's own resolution (``N_XY, N_Z = 32, 256`` --
**deliberately not M5's now-doubled attempt-4 resolution**, so this is a fair,
apples-to-apples comparison against the attempt-3 numbers already on record,
not entangled with the separate, not-yet-run resolution-doubling change).

Simpson et al. (2016) compare two SN-placement modes on an otherwise-identical
stratified box (their Sec. 2/3):

- **RAND**: SNe placed randomly, uniformly in the horizontal plane, following
  the vertical density profile's shape -- this is M5's existing default
  placement (``SNDrivingConfig.density_weighted_placement=False``), already
  run as M5 attempt 3 (see PROGRESS.md's 2026-09-16 M5 entries for the full
  numbers) -- **not rerun here**, to avoid duplicate GPU cost; this script
  compares its own new run directly against those already-recorded numbers.
- **Density-weighted** ("local SFR"/density-peak placement): per-cell trigger
  probability proportional to a local star-formation-rate proxy, ``sfr_i ~
  m_i / t_ff,i ~ rho_i^1.5`` on this codebase's fixed-volume grid (their
  Sec. 2's own derivation) -- implemented as
  ``SNDrivingConfig.density_weighted_placement=True`` +
  ``_draw_density_weighted_site`` in ``_modules/_sn_driving/sn_driving.py``
  (new this session). This is what this script runs.

Simpson et al.'s own reported qualitative differences between the two modes
(their Sec. 3-4, quoting directly): density-weighted placement gives "a much
smoother flow" driven as a "pressure-driven wind" (CR pressure dominant),
while random placement gives "the clumpy nature of gas" via a "ballistic
wind" (kinetic/ram pressure dominant above ~2 scale heights) with the
mid-plane undergoing "thermal runaway" collapse into dense clumps -- despite
comparable mass-loading between the two modes. Three new diagnostics target
these specific qualitative claims (Girichidis-2016-style diagnostics already
built for M5 -- mass-loading, outflow velocity, H_cr/H_gas -- are reused
unchanged):

- ``_midplane_clumpiness``: std(log10(rho)) across the midplane face-on
  slice -- a simple, easy-to-compute proxy for "smooth" vs. "clumpy"
  structure. Not from the paper (they show this qualitatively, via
  Fig. 1-style density maps, not a single number) -- an independent,
  documented choice of metric for this comparison.
- ``_outflow_pressure_budget``: mean CR pressure, gas (thermal) pressure, and
  ram pressure (``rho * v_z^2``) at the same one-scale-height-above-midplane
  reference height M5's other diagnostics use -- a proxy for "pressure-driven"
  (CR pressure comparable to or exceeding ram pressure) vs. "ballistic" (ram
  pressure dominant) driving.
- Midplane ``n_H`` time series (already tracked, reused unchanged): watches
  for the same "thermal runaway"/unbounded-collapse signature Simpson report
  for their RAND run, vs. a regulated, bounded trajectory.

Physics parameters, box, cooling, and CR-grey transport setup below are
copied from ``m5_cr_driven_outflow.py`` (attempt 3's own values, at that
attempt's resolution) rather than imported from it, matching this project's
established convention of self-contained per-milestone scripts (M2 through
M5 each copy-and-modify the previous milestone's setup inline, rather than
cross-importing) -- and, in this specific case, also deliberately decoupling
this script from M5's own current (in-progress, not-yet-run) resolution
change.
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

# ---- physical setup (identical to M2/M3/M3.5/M4/M5) ----
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
# Resolution doubled in each of the 3 dimensions compared 
# to the base run
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


# ---- M4/M5-inherited setup ----
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

# ---- CR-grey physics: identical to M5 attempt 3 (the successful run) ----
SN_CR_FRACTION = 0.1  # Girichidis et al. (2016)'s fiducial fCR

KAPPA_PERP_CGS = 1.0e26 * u.cm ** 2 / u.s  # Girichidis et al. (2016), Sec. 2
DIFFUSION_COEFFICIENT_CODE = KAPPA_PERP_CGS.to(
    CODE_UNITS.code_length ** 2 / CODE_UNITS.code_time
).value

REDUCED_STREAMING_SPEED_PHYS = 300.0 * u.km / u.s
REDUCED_STREAMING_SPEED_CODE = REDUCED_STREAMING_SPEED_PHYS.to(CODE_UNITS.code_velocity).value

GAMMA_CR = 4.0 / 3.0

Z_REF_CODE = Z0 + H_SCALE

M_STAR_PER_SN_CODE = (100.0 * u.M_sun).to(CODE_UNITS.code_mass).value
SFR_EQUIVALENT_CODE = SN_RATE_CODE * M_STAR_PER_SN_CODE  # code mass / code time

# ---- M5 attempt 3's already-recorded numbers (RAND placement), for the
# comparison table at the end of this run -- NOT recomputed here. See
# PROGRESS.md's 2026-09-16 M5 entries for the full write-up. ----
M5_ATTEMPT3_ETA = 6.155
M5_ATTEMPT3_V_OUT_KMS = 5.83
M5_ATTEMPT3_H_GAS = 28.41
M5_ATTEMPT3_H_CR = 86.81
# Clumpiness wasn't tracked by the original attempt-3 run (its states were
# never saved to disk, so it couldn't be computed retroactively either) --
# this flagged gap (PROGRESS.md) was closed 2026-09-22 by adding
# _midplane_clumpiness to m5_cr_driven_outflow.py and rerunning it at this
# same attempt-3 resolution. Same late-time-quarter-average convention as
# the other M5_ATTEMPT3_* constants above.
M5_ATTEMPT3_CLUMP = 0.467


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
        # M6: density_weighted_placement=True is the one change from M5
        # attempt 3's config -- Simpson et al. (2016)'s "density peak" mode
        # instead of the default uniformly-random site (see
        # sn_driving.py's _draw_density_weighted_site).
        sn_driving_config=SNDrivingConfig(
            sn_driving=True, delayed_cooling=True, density_weighted_placement=True,
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_relaxation=True,
        ),
        progress_bar=True,
        return_snapshots=True,
        # 10, not 40 like m5_cr_driven_outflow.py: with return_states=True,
        # the full snapshot buffer stays live in GPU memory for the whole
        # run, and this config's extra density-weighted-placement memory
        # need (on top of a similar baseline to M5) pushes an 11GB 2080 Ti
        # over the edge at higher snapshot counts -- confirmed directly
        # (2026-09-22): 40 needed an extra 7.46GiB in one allocation, 20
        # needed 4.67GiB, both OOM'd; 10 fits (~8.3GB total, matching M5's
        # healthy baseline) and was confirmed to complete successfully
        # (twice, independently). Coarser time sampling than M5's 40 as a
        # result -- fine for this milestone's per-snapshot diagnostics and
        # the clumpiness-evolution filmstrip (only needs ~4 of them), but
        # worth knowing if a future use of this script wants finer sampling.
        num_snapshots=10,
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
    post-shock state, accounting for the nonzero ``sn_cr_fraction`` -- see
    m5_cr_driven_outflow.py's identical probe for the full docstring.
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
    """Mass-outflow rate, mass-flux-weighted outflow velocity, and the
    mass-loading factor at ``z_index`` -- identical to
    m5_cr_driven_outflow.py's diagnostic of the same name.
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
    """Horizontally-averaged gas- and CR-pressure e-folding scale heights --
    identical to m5_cr_driven_outflow.py's diagnostic of the same name.
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
    placement) vs. "clumpy" (random placement) structure comparison. See
    this module's docstring -- not a metric from the paper itself.
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
    collapse transient) to the late-time quasi-steady state. Identical to
    ``m5_cr_driven_outflow.py``'s function of the same name -- see that
    module's docstring for the full rationale (added 2026-09-22 alongside
    the M5-vs-M6 clumpiness finding it exists to illustrate).
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


def _outflow_pressure_budget(state, registered_variables, z_index):
    """Mean CR pressure, gas pressure, and ram pressure (rho*v_z^2) at
    ``z_index`` -- a proxy for Simpson et al. (2016)'s "pressure-driven"
    (CR pressure comparable to/exceeding ram pressure) vs. "ballistic" (ram
    pressure dominant) outflow-mechanism comparison. See this module's
    docstring -- not a metric from the paper itself.
    """
    rho = np.asarray(state[registered_variables.density_index])[:, :, z_index]
    vz = np.asarray(state[registered_variables.velocity_index.z])[:, :, z_index]
    p_gas = np.asarray(state[registered_variables.pressure_index])[:, :, z_index]
    e_cr = np.asarray(state[registered_variables.cosmic_ray_e_index])[:, :, z_index]
    p_cr = (GAMMA_CR - 1.0) * e_cr
    p_ram = rho * vz ** 2
    return float(np.mean(p_gas)), float(np.mean(p_cr)), float(np.mean(p_ram))


def _girichidis_fig1_style_plot(state, registered_variables, mid_index, t_yr, out_path, title_prefix):
    """Structure plot styled after Girichidis et al. (2016) Fig. 1 --
    identical to m5_cr_driven_outflow.py's diagnostic of the same name,
    parameterized by ``title_prefix`` so the two milestones' plots are
    labeled distinctly.
    """
    rho_code = np.asarray(state[registered_variables.density_index])
    e_cr_code = np.asarray(state[registered_variables.cosmic_ray_e_index])

    if not (np.any(rho_code > 0) and np.any(e_cr_code > 0)):
        print(f"  [warning] skipping structure plot at t={t_yr:.1f} yr -- "
              f"density or e_cr field is degenerate (all non-positive)")
        return

    rho = (rho_code * CODE_UNITS.code_density).to(u.g / u.cm ** 3).value
    e_cr_density_unit = CODE_UNITS.code_energy / CODE_UNITS.code_length ** 3
    e_cr = (e_cr_code * e_cr_density_unit).to(u.erg / u.cm ** 3).value

    x_extent = [-L_XY / 2, L_XY / 2]
    y_extent = [-L_XY / 2, L_XY / 2]
    z_extent = [-Z0, L_Z - Z0]

    y_mid = N_XY // 2
    edge_on_density = rho[:, y_mid, :].T
    face_on_density = rho[:, :, mid_index].T
    face_on_e_cr = e_cr[:, :, mid_index].T

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
    fig.suptitle(f"{title_prefix}  |  Time: {t_myr:.1f} Myr")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_m6():
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
    print("SN placement mode: DENSITY-WEIGHTED (Simpson et al. 2016 'density peak', "
          f"weighting power {1.5:.2f}) -- vs. M5 attempt 3's RANDOM placement.")

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

    print("Starting M6 run (density-weighted SN placement + CR-grey diffusion + delayed cooling)...")
    snapshot_data = time_integration(initial_state, config, params, registered_variables)

    time_points_years = (np.asarray(snapshot_data.time_points) * CODE_UNITS.code_time).to(u.yr).value
    states = snapshot_data.states

    z_axis_code = np.asarray(helper_data_unpadded.geometric_centers[0, 0, :, 2])
    mid_index = int(np.argmin(np.abs(z_axis_code - Z0)))
    z_ref_index = int(np.argmin(np.abs(z_axis_code - Z_REF_CODE)))

    print("\n--- per-snapshot summary ---")
    first_nan_idx = None
    n_h_mid_series, t_max_series = [], []
    mdot_series, v_out_series, eta_series = [], [], []
    h_gas_series, h_cr_series = [], []
    clumpiness_series = []
    p_gas_out_series, p_cr_out_series, p_ram_out_series = [], [], []
    for i, t_yr in enumerate(time_points_years):
        state_i = states[i]
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
            p_gas_out_series.append(np.nan)
            p_cr_out_series.append(np.nan)
            p_ram_out_series.append(np.nan)
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
        p_gas_out, p_cr_out, p_ram_out = _outflow_pressure_budget(
            state_i, registered_variables, z_ref_index
        )

        n_h_mid_series.append(n_h_mid)
        t_max_series.append(float(jnp.max(t_i_kelvin)))
        mdot_series.append(mdot_out)
        v_out_series.append(v_out_mean)
        eta_series.append(eta)
        h_gas_series.append(h_gas)
        h_cr_series.append(h_cr)
        clumpiness_series.append(clumpiness)
        p_gas_out_series.append(p_gas_out)
        p_cr_out_series.append(p_cr_out)
        p_ram_out_series.append(p_ram_out)

        print(
            f"  snapshot {i:3d}  t={t_yr:12.1f} yr   max(T)={float(jnp.max(t_i_kelvin)):10.3e} K   "
            f"n_H(mid)={n_h_mid:8.3f} cm^-3   Mdot_out={mdot_out: .3e}   "
            f"v_out={v_out_mean:8.2f} km/s   eta={eta: .3e}   "
            f"H_gas={h_gas:8.2f}   H_cr={h_cr:8.2f}   clump={clumpiness:6.3f}   "
            f"P_gas={p_gas_out: .2e}   P_cr={p_cr_out: .2e}   P_ram={p_ram_out: .2e}"
        )

    if first_nan_idx is not None:
        last_good_t = time_points_years[first_nan_idx - 1] if first_nan_idx > 0 else 0.0
        print(f"\nRun went to NaN/corrupted state at snapshot {first_nan_idx} -- last good "
              f"snapshot {first_nan_idx - 1} was at t={last_good_t:.1f} yr.")
    else:
        print("\nRun completed with no NaNs through t_end.")

    valid = ~np.isnan(np.asarray(eta_series))
    if np.any(valid):
        tail = np.where(valid)[0][-max(1, int(0.25 * np.sum(valid))):]
        eta_m6 = float(np.nanmean(np.asarray(eta_series)[tail]))
        v_out_m6 = float(np.nanmean(np.asarray(v_out_series)[tail]))
        h_gas_m6 = float(np.nanmean(np.asarray(h_gas_series)[tail]))
        h_cr_m6 = float(np.nanmean(np.asarray(h_cr_series)[tail]))
        clump_m6 = float(np.nanmean(np.asarray(clumpiness_series)[tail]))
        p_gas_m6 = float(np.nanmean(np.asarray(p_gas_out_series)[tail]))
        p_cr_m6 = float(np.nanmean(np.asarray(p_cr_out_series)[tail]))
        p_ram_m6 = float(np.nanmean(np.asarray(p_ram_out_series)[tail]))

        print("\n--- late-time (last quarter of valid snapshots) averages: "
              "M6 (density-weighted) vs. M5 attempt 3 (random, already on record) ---")
        print(f"  mass-loading factor eta : M6={eta_m6:.3e}   M5-attempt3={M5_ATTEMPT3_ETA:.3e}   "
              f"(Simpson et al. 2016: comparable between placement modes)")
        print(f"  outflow velocity v_out  : M6={v_out_m6:.2f} km/s   M5-attempt3={M5_ATTEMPT3_V_OUT_KMS:.2f} km/s")
        print(f"  H_gas                   : M6={h_gas_m6:.2f}   M5-attempt3={M5_ATTEMPT3_H_GAS:.2f}")
        print(f"  H_cr                    : M6={h_cr_m6:.2f}   M5-attempt3={M5_ATTEMPT3_H_CR:.2f}")
        print(f"  midplane clumpiness     : M6={clump_m6:.3f}   M5-attempt3={M5_ATTEMPT3_CLUMP:.3f}   "
              f"(Simpson et al. 2016: density-weighted placement -> smoother/lower clumpiness "
              f"than random placement; late-time average found these statistically tied -- see "
              f"PROGRESS.md's 2026-09-22 entry: the qualitative signature shows up in the *peak* "
              f"clumpiness during the onset/collapse phase instead, not the late-time average)")
        print(f"  outflow pressure budget : M6 P_gas={p_gas_m6: .3e}  P_cr={p_cr_m6: .3e}  "
              f"P_ram={p_ram_m6: .3e}   (Simpson et al. 2016: density-weighted placement -> "
              f"'pressure-driven' [P_cr comparable to/exceeding P_ram]; random placement -> "
              f"'ballistic' [P_ram dominant])")

    # Diagnostic plots.
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    ax0, ax1, ax2, ax3, ax4, ax5 = axes.flat

    ax0.semilogy(time_points_years, n_h_mid_series, "-o", ms=3)
    ax0.set_xlabel("t [yr]")
    ax0.set_ylabel(r"midplane $n_H$ [cm$^{-3}$]")
    ax0.set_title("Midplane density vs. time (watch for unbounded runaway)")

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

    ax4.plot(time_points_years, clumpiness_series, "-o", ms=3, color="C4")
    ax4.set_xlabel("t [yr]")
    ax4.set_ylabel(r"std(log$_{10}\rho$) at midplane")
    ax4.set_title("Midplane clumpiness vs. time")

    ax5.semilogy(time_points_years, np.abs(p_gas_out_series), "-o", ms=3, label="P_gas")
    ax5.semilogy(time_points_years, np.abs(p_cr_out_series), "-o", ms=3, label="P_cr")
    ax5.semilogy(time_points_years, np.abs(p_ram_out_series), "-o", ms=3, label="P_ram")
    ax5.set_xlabel("t [yr]")
    ax5.set_ylabel("pressure [code] at $z=Z_0+H$")
    ax5.set_title("Outflow pressure budget vs. time")
    ax5.legend(fontsize=8)

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    out_path = pics_dir / "m6_sn_placement_comparison.svg"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"\nDiagnostic plot written to {out_path}")

    if np.any(valid):
        last_valid = np.where(valid)[0][-1]
        out_path3 = pics_dir / "m6_structure_girichidis_fig1_style.svg"
        _girichidis_fig1_style_plot(
            states[last_valid], registered_variables, mid_index,
            time_points_years[last_valid], out_path3,
            title_prefix="M6: SNe density-weighted placement",
        )
        print(f"Girichidis-Fig.1-style structure plot written to {out_path3}")

    out_path4 = pics_dir / "m6_clumpiness_evolution.svg"
    _clumpiness_evolution_plot(
        states, time_points_years, clumpiness_series, registered_variables, mid_index,
        out_path4, title_prefix="M6: SNe density-weighted placement",
    )
    print(f"Clumpiness-evolution filmstrip written to {out_path4}")

    return snapshot_data


if __name__ == "__main__":
    run_m6()
