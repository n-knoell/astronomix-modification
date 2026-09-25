"""Shared 3D stationary colliding-wind-binary setup for the shock-finder tests.

Two equal-mass, equal-wind O-star sources (3D thermal-energy-injection
scheme, ``EI``) sit at fixed, symmetric positions on the x-axis, separated by
a realistic binary separation, in a homogeneous warm-ISM ambient medium.
``config.nbody_config`` is left at its default (``NBodyConfig(nbody=False)``),
so the sources do not orbit -- ``wind_params.wind_injection_positions`` is used
directly instead of being overridden by N-body positions (see
``astronomix._modules._stellar_wind.stellar_wind._wind_source_params``).

Physical parameters (stellar mass, mass-loss rate, terminal wind velocity,
binary separation, ambient density/pressure) and their conversion to code
units via :class:`astronomix.CodeUnits` follow
``examples/stellar_wind/colliding_wind_binary.py``: a ~45 solar-mass O star
with a ~7e-7 Msun/yr, ~2000 km/s wind, 20 au from its (identical) companion,
in a warm neutral/ionized ambient medium (n ~ 2 cm^-3, T ~ 1.5e4 K). Unlike
that example, the two stars are given *equal* wind parameters here (rather
than the example's unequal M1/M2 pair) so the whole setup stays exactly
mirror-symmetric about x=0 -- a physically realistic "twin" massive binary,
and a strong, resolution-independent cross-check for the tests below (see
``cwb_shock_finder.py``).

With equal mass-loss rates and terminal velocities for both stars, each wind
free-expands, is slowed by a termination (reverse) shock facing the collision
region, and the two shocked winds meet at a contact discontinuity at x=0.
Each wind bubble also drives its own forward/bow shock into the ambient
medium. Integrated with the finite-volume HLLC solver, then reduced with
:func:`astronomix.shock_finder3D.pfrommer_shock_finder.find_shocks_pfrommer`.

Realistic O-star wind velocities are highly supersonic with respect to the
ambient sound speed (Mach number ~ 100) and vastly denser than the ambient
medium at the collision radius, so the swept-up ambient shell and the
wind-collision layer are both geometrically thin compared to the bubble
radius -- as in real (non-radiative) colliding-wind binaries, some of the
shock finder's surface cells sit close enough together that the immediate
neighbour sampled on each side of a candidate shock (see
``_shock_zones.get_post_pre_shock_values``) straddles more than one physical
shock. ``cwb_shock_finder.py`` accounts for this explicitly.

Kept resolution-independent (only ``num_cells`` varies) so the same physical
setup can be reused across tests, mirroring ``_sedov_setup.py``.
"""

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# units
import astropy.constants as const
from astropy import units as u

# astronomix constants
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC
from astronomix.option_classes.simulation_config import MINMOD, MUSCL, SPLIT, VAN_ALBADA_PP
from astronomix import OPEN_BOUNDARY, FORWARDS

# astronomix containers
from astronomix import CodeUnits, SimulationConfig, SimulationParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix import (
    CodeUnits,
    WindParams,
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings,
)
from astronomix.option_classes import EI, WindConfig, WindParams
from astronomix.option_classes import NBodyConfig, NBodyParams, NGP

from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

from astronomix._modules._nbody._nbody import binary_starting_orbits_at_phase

# cooling (see FIXES_TODO.md item 1b, round 14: the pre-shock wind can only
# ever be as cold as the grid can numerically resolve via adiabatic PdV
# expansion alone, which item 1b rounds 12-13 showed is nowhere near enough
# at any tractable resolution -- radiative cooling is a local, pointwise
# process (rate ~ n^2 * Lambda(T)) that doesn't need that spatial resolution
# at all, and is real missing physics for an O-star wind, not just a
# numerical workaround)
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    IMPLICIT_COOLING,
    PIECEWISE_POWER_LAW,
)
from astronomix._modules._cooling._cooling_tables import schure_cooling

# ---- physical setup (see examples/stellar_wind/colliding_wind_binary.py) ----
GAMMA = 5.0 / 3.0
BOX_SIZE = 1.0
SEPARATION = 0.4  # binary separation, in code units (box-centered coordinates)

# STAR_MASS = 45 * u.M_sun  # equal for both stars -> mirror-symmetric collision
M1 = 50 * u.M_sun
M2 = 40 * u.M_sun
# MASS_LOSS_RATE = 7e-7 * u.M_sun / u.yr  # equal for both stars
# WIND_VELOCITY = 2000 * u.km / u.s  # equal for both stars
SEPARATION_PHYSICAL = 50 * u.au

# warm neutral/ionized ambient medium (same values as the example script)
RHO_0 = 2 * const.m_p / u.cm**3
P_0 = 3e4 * u.K / u.cm**3 * const.k_B

# code units: length set by the physical separation, mass by one star's mass,
# velocity by the resulting Keplerian scale (matches CodeUnits usage in
# colliding_wind_binary.py, though these stars do not orbit here).
# CODE_LENGTH = SEPARATION_PHYSICAL / SEPARATION
# CODE_MASS = M1 # STAR_MASS
# CODE_VELOCITY = np.sqrt(const.G * CODE_MASS / CODE_LENGTH).to(u.km / u.s)
# CODE_UNITS = CodeUnits(CODE_LENGTH, CODE_MASS, CODE_VELOCITY)

sep_in_au = 50  # upper limit: 80, lower limit: 5
mass_temp = 1
# code_length chosen so SEPARATION (the dimensionless box-separation above)
# times code_length equals sep_in_au exactly, i.e. this setup actually
# simulates a sep_in_au-au binary. Previously this used an unrelated
# `sep_in_au / 10` scaling for code_length, decoupled from SEPARATION -- the
# separation actually simulated was silently 25x smaller than sep_in_au
# claimed (0.8 au vs the documented 20 au; see FIXES_TODO.md round 12).
# Restores the originally-intended CODE_LENGTH = SEPARATION_PHYSICAL /
# SEPARATION relationship (see the commented-out block above).
code_length = (sep_in_au / SEPARATION) * u.au
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(const.G * code_mass / code_length).to(u.km / u.s)
CODE_UNITS = CodeUnits(code_length, code_mass, code_velocity)

# WIND_MASS_LOSS_RATE = MASS_LOSS_RATE.to(
#     CODE_UNITS.code_mass / CODE_UNITS.code_time
# ).value
# WIND_TERMINAL_VELOCITY = WIND_VELOCITY.to(CODE_UNITS.code_velocity).value
RHO_AMBIENT = RHO_0.to(CODE_UNITS.code_density).value
P_AMBIENT = P_0.to(CODE_UNITS.code_pressure).value

# Schure et al. (2009) + Dalgarno & McCray (1972) piecewise cooling curve
# (astronomix._modules._cooling._cooling_tables.schure_cooling); the
# tabulated rate is exactly zero below its own ~6300 K floor (see
# _cooling._evaluate_piecewise_power_law -- out-of-range returns 0.0, not an
# extrapolation), so the wind cannot be cooled colder than that via this
# curve regardless of floor_temperature below. HYDROGEN_MASS_FRACTION /
# METAL_MASS_FRACTION left at CoolingParams' own defaults (0.76 / 0.02).
HYDROGEN_MASS_FRACTION = 0.76
METAL_MASS_FRACTION = 0.02
FLOOR_TEMPERATURE_KELVIN = 1e2  # numerical-only floor, well below where the
# Schure curve's own tabulated rate already reaches zero -- see above.
FLOOR_TEMPERATURE = (FLOOR_TEMPERATURE_KELVIN * u.K * const.k_B / const.m_p).to(
    CODE_UNITS.code_energy / CODE_UNITS.code_mass
).value
COOLING_CURVE_PARAMS = schure_cooling(CODE_UNITS)

MACH_MIN = 1.3

# Analytic wind zone (WindConfig.analytic_wind_zone, opt-in via run_cwb's
# analytic_wind_zone argument; FIXES_TODO.md round 18): the wind's
# adiabatic-expansion base state, the same illustrative photospheric values
# expected_cwb_mach.py uses for its adiabatic-ceiling estimate. The base
# temperature is converted to the code pseudo-temperature p/rho with the same
# mu = 1 convention as RHO_0/P_0 above.
WIND_BASE_RADIUS = (20 * u.R_sun).to(CODE_UNITS.code_length).value
WIND_BASE_TEMPERATURE = (3.5e4 * u.K * const.k_B / const.m_p).to(
    CODE_UNITS.code_velocity**2
).value

# Wind temperature floor (WindConfig.wind_temperature_floor, opt-in via
# run_cwb's wind_floor_kelvin argument; FIXES_TODO.md round 22): a
# photoionised O-star wind sits near 1e4 K instead of cooling adiabatically.
# Same mu = 1 convention as above. The ambient medium (1.5e4 K) is above it.
def kelvin_to_code_temperature(kelvin):
    return (kelvin * u.K * const.k_B / const.m_p).to(CODE_UNITS.code_velocity**2).value
WIND_ZONE_STAGNATION_FRACTION = 0.5

# diffusive shock acceleration: inject this fraction of each detected shock's
# dissipated kinetic-energy flux into e_cr instead of gas thermal energy
# (CosmicRayGreyParams.dsa_efficiency). Same fixed-efficiency value already
# calibrated for the wind-blown-bubble ladder item (cr_wind_bubble.py) --
# reused here rather than re-tuned, since this is a first CR-DSA pass on the
# CWB setup, not a new calibration exercise. dsa_mach_min reuses MACH_MIN
# above so the injected-CR population matches exactly the shocks the finder
# itself reports.
DSA_EFFICIENCY = 0.1

# box-centered coordinates (box center at the origin), same convention as
# helper_data.geometric_centers - box_center; see
# astronomix._modules._stellar_wind.stellar_wind._wind_source_distances.
STAR_POSITIONS = jnp.array([[-SEPARATION / 2, 0.0, 0.0], [SEPARATION / 2, 0.0, 0.0]])

mass_loss_rate_1 = 6.5e-7 * u.M_sun / u.yr
mass_loss_rate_2 = 6.5e-8 * u.M_sun / u.yr
wind_velocity_1 = 2200 * u.km / u.s
wind_velocity_2 = 1850 * u.km / u.s
eccentricity = 0.0
cos_inclination = 1.0
turbulence_strength = 0.0

a = SEPARATION

m1 = M1.to(CODE_UNITS.code_mass).value
m2 = M2.to(CODE_UNITS.code_mass).value
Period = 2 * np.pi * np.sqrt(a**3 / ((m1 + m2)))

mlr1 = mass_loss_rate_1.to(CODE_UNITS.code_mass / CODE_UNITS.code_time).value
mlr2 = mass_loss_rate_2.to(CODE_UNITS.code_mass / CODE_UNITS.code_time).value
v_inf1 = wind_velocity_1.to(CODE_UNITS.code_velocity).value
v_inf2 = wind_velocity_2.to(CODE_UNITS.code_velocity).value
inclination_deg = float(jnp.rad2deg(jnp.arccos(cos_inclination)))

# CosmicRayGreyParams.reduced_streaming_speed's global default (1.0) is a
# placeholder (see DESIGN.md's open questions) that is nowhere near this
# setup's actual velocity scale: v_inf1/v_inf2 above are ~826/~694 in the
# same code units. Left at the default, the CR two-moment system's own
# characteristic signal speed is ~800x slower than the wind that advects its
# e_cr/F_cr state -- the same "reduced_streaming_speed undercut by the real
# flow speed" failure mode already documented for the SILCC-ISM project (see
# this module's CR-grey PROGRESS.md, 2026-09-16 M5 entry), just far more
# extreme here. Set comfortably above both terminal wind speeds instead.
REDUCED_STREAMING_SPEED = 1000.0

CROSSING_TIME_SAFETY_MARGIN = 1.1
furthest_edge_distance = ((BOX_SIZE / 2 + SEPARATION / 2)**2 + (BOX_SIZE / 2)**2 + (BOX_SIZE / 2)**2)**0.5
wind_crossing_time = furthest_edge_distance / min(v_inf1, v_inf2)
T_END = CROSSING_TIME_SAFETY_MARGIN * wind_crossing_time
# T_END = Period / 8
print(f"Orbital period in code units: {Period}")
print(f"Wind boundary-crossing time in code units: {wind_crossing_time}")
print(f"End time in code units: {T_END} ({T_END / Period:.4%} of the orbital period)")

masses = jnp.array([m1, m2])
# initial N-body phase-space state [t, x, y, z, vx, vy, vz] per body,
# flattened -- seeds astronomix._modules._nbody._nbody's RK4 integrator,
# advanced jointly with the hydro update every step.
nbody_state = binary_starting_orbits_at_phase(
    a, eccentricity, inclination_deg, m1, m2, phi=0.0, true_anom_deg=0.0,
)


def run_cwb(
    num_cells,
    dsa_efficiency=DSA_EFFICIENCY,
    analytic_wind_zone=False,
    wind_zone_stagnation_fraction=WIND_ZONE_STAGNATION_FRACTION,
    dual_energy=False,
    dual_energy_shock_dilation=1,
    wind_floor_kelvin=None,
):
    """Run one 3D stationary colliding-wind-binary simulation and find its shocks.

    Grey two-moment cosmic rays are always active (``grey_cosmic_rays=True``,
    ``diffusive_shock_acceleration=True``) so the registered-variable layout
    (and hence array shapes/indices) is identical across calls -- only
    ``dsa_efficiency`` varies between a CR-off control (``dsa_efficiency=0``,
    DSA code path live but injecting nothing) and a CR-on run. Everything
    else about the physical setup (wind/ambient/cooling/resolution/BCs) is
    unchanged from the original hydro-only CWB setup.

    Args:
        num_cells: The number of cells per dimension (cubic domain).
        dsa_efficiency: Fraction of each detected shock's dissipated
            kinetic-energy flux diverted into ``e_cr`` (diffusive shock
            acceleration). ``0.0`` gives a CR-free control run with the same
            state layout as a CR-on run.
        analytic_wind_zone: Overwrite an extended zone around each star with
            the analytic adiabatically-cooled free wind every step
            (``WindConfig.analytic_wind_zone``, base state
            ``WIND_BASE_RADIUS`` / ``WIND_BASE_TEMPERATURE``). ``False`` is
            the plain EI injection.
        wind_zone_stagnation_fraction: Zone radius as a fraction of each
            star's distance to the stagnation point
            (``WindParams.wind_zone_stagnation_fraction``). Only read when
            ``analytic_wind_zone`` is on.
        dual_energy: Use the dual-energy (entropy) pressure in cold,
            unshocked, supersonic cells (``SimulationConfig.dual_energy``),
            so the free wind is not re-heated by kinetic-energy truncation
            error between injection and the shock. ``False`` is the plain
            total-energy scheme.
        dual_energy_shock_dilation: Shock-zone widening (cells) for the
            dual-energy switch (``SimulationConfig.dual_energy_shock_dilation``).
        wind_floor_kelvin: Wind temperature floor in K
            (``WindConfig.wind_temperature_floor``), e.g. ``1e4``. ``None``
            (default) leaves the wind unfloored.

    Returns:
        A dict with the final primitive ``state``, the ``config``,
        ``helper_data``, ``registered_variables`` and the shock finder's
        ``ShockFinderResult``.
    """
    config = SimulationConfig(
        progress_bar=True,
        first_order_fallback=False,
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        split=SPLIT,
        # HLLC + full 2nd-order MUSCL reconstruction is not unconditionally
        # positivity-preserving the way HLL + first-order is; at this wind's
        # Mach ~100 injection contrast that showed up as a genuine blow-up
        # (checkify pinpointed it: NaN pressure straight out of
        # _evolve_state_along_axis, confirmed at N=128). PositivityConfig
        # (default_positivity_protection, vacuum_rest, nan_safe, ...) does
        # NOT help here -- it's only ever consulted by the finite-difference
        # SSPRK integrator (_apply_stage_positivity is called exclusively
        # from astronomix/_finite_difference/_time_integrators/_ssprk.py);
        # the finite-volume evolve path this config uses has no positivity
        # floor wired in at all, so those knobs are silently inert here
        # (confirmed: turning them all on made no difference).
        #
        # VAN_ALBADA_PP was tried first (2026-09-18) but turned out to be
        # silently inert for split=SPLIT: its positivity clamp was only
        # wired into the unsplit reconstruction path. Switched to MINMOD
        # instead (2026-09-18/19): survives the injection-region blow-up on
        # its own (limited gradient collapses near a sharp jump instead of
        # amplifying it), but a *different* failure remains -- a gradual,
        # multi-step density/pressure erosion at the domain's open-boundary
        # corner as the wind front sweeps past it (see FIXES_TODO.md item 4
        # for the full trace, including four ad hoc post-hoc fix attempts
        # that all fell short).
        #
        # 2026-09-19: properly ported VAN_ALBADA_PP's actual positivity-
        # preserving gradient-scaling algorithm (the real alpha/kappa/beta
        # formula, not an ad hoc clamp) from _reconstruct_at_interface_unsplit
        # to the split path (_reconstruct_at_interface_split), scaling the
        # reconstructed *output* delta rather than the pre-A_W input (the
        # latter doesn't work -- A_W mixes variables, so scaling its input
        # can't bound a blow-up that only exists in its mixed output; see
        # that function's own comments). This genuinely fixes the original
        # injection-region blow-up (verified clean). It does not yet fix
        # the full run: a new failure mode appears at N=64 around t~0.13%
        # of T_END, well before the old boundary-corner failure -- dt
        # collapses ~500x within a single step at a cell just outside the
        # injection sphere, from a localized pressure/density (temperature)
        # spike, root cause not yet found. See FIXES_TODO.md item 4 for the
        # full trace.
        #
        # Confirmed (2026-09-19): re-running the real correctness test with
        # MINMOD at N=64 reproduces the known boundary-corner failure
        # (fully NaN by T_END), consistent with everything documented
        # above.
        #
        # Back to MINMOD (2026-09-19, again): the ported positivity-
        # preserving output-delta scaling (see _reconstruct_at_interface_split)
        # is mathematically limiter-agnostic -- it operates on the actual
        # reconstructed delta (primitives_left/right - primitive_state),
        # not on the raw limited_gradients a specific limiter produced -- so
        # it was extended to also apply when config.limiter == MINMOD, not
        # just VAN_ALBADA_PP. MINMOD's own gradients are already far more
        # conservative than VAN_ALBADA's (confirmed: this scaling is a
        # near-total no-op for MINMOD in the injection region, alpha/kappa/
        # beta ~= 1 almost everywhere), so the hope is this only actually
        # engages at the genuinely extreme boundary-corner scenario it's
        # meant to fix, without the broader over-triggering that made some
        # of VAN_ALBADA_PP's own naive (now-reverted) fix attempts regress.
        # See FIXES_TODO.md item 4 for whether this actually fixes MINMOD's
        # ~15.47%-of-T_END boundary-corner failure.
        limiter=MINMOD,
        riemann_solver=HLLC,
        time_integrator=MUSCL,
        differentiation_mode=FORWARDS,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        exact_end_time=True,
        dual_energy=dual_energy,
        dual_energy_shock_dilation=dual_energy_shock_dilation,
        wind_config=WindConfig(
            stellar_wind=True,
            num_injection_cells=num_cells // 32,
            # wind_injection_scheme=EI,
            trace_wind_density=False,
            analytic_wind_zone=analytic_wind_zone,
            wind_temperature_floor=wind_floor_kelvin is not None,
        ),
        
        nbody_config=NBodyConfig(
            nbody=False,
            deposit_particles=NGP,
            central_object_only=False,
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_shock_acceleration=True
        ),
        # FIXES_TODO.md item 1b, round 14: radiative cooling, disabled by
        # default (CoolingConfig()'s own default is cooling=False), enabled
        # here to test whether it -- not just grid resolution -- is what the
        # pre-shock wind needs to reach a realistically low temperature.
        # IMPLICIT_COOLING (Newton's method, astronomix._modules._cooling.
        # _cooling.update_temperature_implicit) rather than EXPLICIT_COOLING:
        # far more robust across a wide dt/cooling-time ratio, which this
        # setup's huge injection-region density/temperature contrast (item 4:
        # ~17 orders of magnitude in pressure alone) will produce somewhere.
        # subcycle_stiff_cooling=True is not optional here: without it,
        # _cfl_time_step's dt_cool term clamps the *global* dt to the
        # single fastest-cooling cell in the whole domain (almost certainly
        # right at the injection region) -- documented (cooling_options.py)
        # to have stalled a previous run (SILCC-ISM M4) indefinitely under
        # exactly this "one persistently fast-cooling cell" pattern.
        cooling_config=CoolingConfig(
            cooling=False,   
            cooling_method=IMPLICIT_COOLING,
            subcycle_stiff_cooling=True,
            cooling_curve_config=CoolingCurveConfig(
                cooling_curve_type=PIECEWISE_POWER_LAW,
            ),
        ),
        boundary_settings=BoundarySettings(
            BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
            BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
            BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        # nbody_config left at its default (nbody=False): the two wind
        # sources stay fixed at STAR_POSITIONS instead of following an
        # N-body orbit -- a *stationary* colliding-wind binary.
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (num_cells, num_cells, num_cells)
    density = jnp.ones(shape) * RHO_AMBIENT
    zeros = jnp.zeros(shape)
    gas_pressure = jnp.ones(shape) * P_AMBIENT

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        C_cfl=0.4,
        # Used by the SPLIT-path positivity floor/scaling in
        # reconstruction.py/evolve_state.py (active for limiter in
        # {MINMOD, VAN_ALBADA_PP} -- see FIXES_TODO.md item 4).
        # SimulationParams' global default (1e-14) is *larger* than this
        # setup's ambient density/pressure (RHO_AMBIENT/P_AMBIENT ~1.26e-17),
        # which would floor -- i.e. artificially inflate -- the entire
        # ambient medium rather than only clamping genuine violations.
        # Scale them well below ambient instead.
        minimum_density=1e-3 * RHO_AMBIENT,
        minimum_pressure=1e-3 * P_AMBIENT,
        nbody_params=NBodyParams(
            masses=masses,
            nbody_state=nbody_state,
        ),
        wind_params=WindParams(
            wind_mass_loss_rates=jnp.array(
                [mlr1, mlr2]
            ),
            wind_final_velocities=jnp.array(
                [v_inf1, v_inf2]
            ),
            # Required now that nbody_config.nbody=False: _wind_source_params
            # falls back to this field for source positions, and its default
            # ([[0,0,0]]) is a single source at the box center, not two stars.
            wind_injection_positions=STAR_POSITIONS,
            # Only read when analytic_wind_zone=True.
            wind_base_radii=jnp.array([WIND_BASE_RADIUS, WIND_BASE_RADIUS]),
            wind_base_temperatures=jnp.array(
                [WIND_BASE_TEMPERATURE, WIND_BASE_TEMPERATURE]
            ),
            wind_zone_stagnation_fraction=wind_zone_stagnation_fraction,
            wind_floor_temperature=(
                0.0 if wind_floor_kelvin is None
                else kelvin_to_code_temperature(wind_floor_kelvin)
            ),
        ),
        # Inert unless config.cooling_config.cooling=True (see above).
        cooling_params=CoolingParams(
            hydrogen_mass_fraction=HYDROGEN_MASS_FRACTION,
            metal_mass_fraction=METAL_MASS_FRACTION,
            floor_temperature=FLOOR_TEMPERATURE,
            cooling_curve_params=COOLING_CURVE_PARAMS,
        ),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            dsa_efficiency=dsa_efficiency, dsa_mach_min=MACH_MIN,
            reduced_streaming_speed=REDUCED_STREAMING_SPEED,
        ),
    )

    state = time_integration(initial_state, config, params, registered_variables)

    # mach_sampling_adaptive: walks each ray until it exits the shock zone
    # (Schaal & Springel 2015 Sec. 2.3.3) instead of a fixed 1-cell offset --
    # see FIXES_TODO.md item 1b, round 16. Validated against a synthetic
    # smeared shock (recovers M within ~12% of truth up to 12-cell smearing,
    # vs. the fixed default's collapse to M~2) and the Sedov-Taylor
    # correctness test (median Mach 7.4->126, no invariant violated). On the
    # real N=64 stationary CWB run it gives essentially the same result as
    # the fixed default (median 4.29 vs 3.12, max 10.36 vs 10.00, cap=15 vs
    # cap=30 identical) -- confirms the low measured Mach here is a real
    # limitation of the pre-shock physical state, not a shock-finder-
    # formalism artifact.
    sf_result = find_shocks_pfrommer(
        state, config, registered_variables, helper_data, mach_min=MACH_MIN,
        mach_sampling_adaptive=True, mach_sampling_steps=15,
    )

    return dict(
        state=state,
        config=config,
        params=params,
        helper_data=helper_data,
        registered_variables=registered_variables,
        sf_result=sf_result,
    )


def axis_profile(run):
    """Density / pressure / CR pressure / x-velocity profile along the binary
    (x) axis.

    Samples the line of cells through the domain's y/z center -- i.e. through
    both fixed wind sources, which sit on the x-axis (y = z = 0 in
    box-centered coordinates). Used as an independent cross-check of the
    shock finder's output against the raw field, analogous to
    ``_sedov_setup.binned_radial_profile``.

    Args:
        run: The dict returned by ``run_cwb``.

    Returns:
        ``(x, density, pressure, pressure_cr, velocity_x)``, each a 1D numpy
        array in box-centered x-coordinates. ``pressure_cr`` is
        ``(gamma_cr - 1) * e_cr`` and is exactly zero everywhere for a
        ``dsa_efficiency=0`` control run.
    """
    state = run["state"]
    registered_variables = run["registered_variables"]
    helper_data = run["helper_data"]
    params = run["params"]
    num_cells = run["config"].num_cells.x

    y = num_cells // 2
    z = num_cells // 2
    density = np.array(state[registered_variables.density_index])[:, y, z]
    pressure = np.array(state[registered_variables.pressure_index])[:, y, z]
    e_cr = np.array(state[registered_variables.cosmic_ray_e_index])[:, y, z]
    pressure_cr = (params.cosmic_ray_grey_params.gamma_cr - 1.0) * e_cr
    velocity_x = np.array(
        state[registered_variables.velocity_index.x]
    )[:, y, z]
    x = np.array(helper_data.geometric_centers)[:, y, z, 0] - BOX_SIZE / 2
    return x, density, pressure, pressure_cr, velocity_x
