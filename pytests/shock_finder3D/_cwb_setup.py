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
from astronomix.option_classes.simulation_config import MUSCL, SPLIT, VAN_ALBADA_PP
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

from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

from astronomix._modules._nbody._nbody import binary_starting_orbits_at_phase

# ---- physical setup (see examples/stellar_wind/colliding_wind_binary.py) ----
GAMMA = 5.0 / 3.0
BOX_SIZE = 1.0
SEPARATION = 0.4  # binary separation, in code units (box-centered coordinates)

# STAR_MASS = 45 * u.M_sun  # equal for both stars -> mirror-symmetric collision
M1 = 50 * u.M_sun
M2 = 40 * u.M_sun
# MASS_LOSS_RATE = 7e-7 * u.M_sun / u.yr  # equal for both stars
# WIND_VELOCITY = 2000 * u.km / u.s  # equal for both stars
SEPARATION_PHYSICAL = 20 * u.au

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

sep_in_au = 20 # upper limit: 80, lower limit: 5
length_temp = sep_in_au / 10
mass_temp = 1
code_length = length_temp * u.au
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(const.G * code_mass / code_length).to(u.km / u.s)
CODE_UNITS = CodeUnits(code_length, code_mass, code_velocity)

# WIND_MASS_LOSS_RATE = MASS_LOSS_RATE.to(
#     CODE_UNITS.code_mass / CODE_UNITS.code_time
# ).value
# WIND_TERMINAL_VELOCITY = WIND_VELOCITY.to(CODE_UNITS.code_velocity).value
RHO_AMBIENT = RHO_0.to(CODE_UNITS.code_density).value
P_AMBIENT = P_0.to(CODE_UNITS.code_pressure).value

# T_END is a small fraction of the code time unit: real O-star winds are so
# much faster than the ambient sound speed and so much less dense than the
# ambient medium that the wind-blown bubble sweeps across an order-unity
# fraction of the domain within a tiny fraction of the (Keplerian) code time
# unit. Chosen empirically (at NUM_INJECTION_CELLS / num_cells below) so the
# two wind bubbles have merged into a genuine wind-collision region (not just
# two independent, not-yet-touching bubbles) while the bow shocks stay
# comfortably inside the domain (max |x| well below BOX_SIZE / 2).
# T_END = 4e-3
MACH_MIN = 1.3

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

# T_END = (3.5 * u.yr).to(CODE_UNITS.code_time).value
T_END = Period
print(f"End time in code units: {T_END}")

m1 = M1.to(CODE_UNITS.code_mass).value
m2 = M2.to(CODE_UNITS.code_mass).value
mlr1 = mass_loss_rate_1.to(CODE_UNITS.code_mass / CODE_UNITS.code_time).value
mlr2 = mass_loss_rate_2.to(CODE_UNITS.code_mass / CODE_UNITS.code_time).value
v_inf1 = wind_velocity_1.to(CODE_UNITS.code_velocity).value
v_inf2 = wind_velocity_2.to(CODE_UNITS.code_velocity).value
inclination_deg = float(jnp.rad2deg(jnp.arccos(cos_inclination)))

masses = jnp.array([m1, m2])
# initial N-body phase-space state [t, x, y, z, vx, vy, vz] per body,
# flattened -- seeds astronomix._modules._nbody._nbody's RK4 integrator,
# advanced jointly with the hydro update every step.
nbody_state = binary_starting_orbits_at_phase(
    a, eccentricity, inclination_deg, m1, m2, phi=0.0, true_anom_deg=0.0,
)


def run_cwb(num_cells):
    """Run one 3D stationary colliding-wind-binary simulation and find its shocks.

    Args:
        num_cells: The number of cells per dimension (cubic domain).

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
        # (confirmed: turning them all on made no difference). VAN_ALBADA_PP
        # is the FV-path-aware fix instead -- a genuine positivity-preserving
        # MUSCL reconstruction (astronomix/_finite_volume/_state_evolution/
        # reconstruction.py) that clamps the interface extrapolation itself
        # so it can't overshoot into negative density/pressure, rather than
        # patching a bad state after the fact.
        limiter=VAN_ALBADA_PP,
        riemann_solver=HLLC,
        time_integrator=MUSCL,
        differentiation_mode=FORWARDS,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        exact_end_time=True,
        wind_config=WindConfig(
            stellar_wind=True,
            num_injection_cells=num_cells // 32,
            # wind_injection_scheme=EI,
            trace_wind_density=False,
        ),
        
        nbody_config=NBodyConfig(
            nbody=True,
            deposit_particles=NGP,
            central_object_only=False,
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
            # wind_injection_positions=STAR_POSITIONS,
        ),
    )

    state = time_integration(initial_state, config, params, registered_variables)

    sf_result = find_shocks_pfrommer(
        state, config, registered_variables, helper_data, mach_min=MACH_MIN
    )

    return dict(
        state=state,
        config=config,
        helper_data=helper_data,
        registered_variables=registered_variables,
        sf_result=sf_result,
    )


def axis_profile(run):
    """Density / pressure / x-velocity profile along the binary (x) axis.

    Samples the line of cells through the domain's y/z center -- i.e. through
    both fixed wind sources, which sit on the x-axis (y = z = 0 in
    box-centered coordinates). Used as an independent cross-check of the
    shock finder's output against the raw field, analogous to
    ``_sedov_setup.binned_radial_profile``.

    Args:
        run: The dict returned by ``run_cwb``.

    Returns:
        ``(x, density, pressure, velocity_x)``, each a 1D numpy array in
        box-centered x-coordinates.
    """
    state = run["state"]
    registered_variables = run["registered_variables"]
    helper_data = run["helper_data"]
    num_cells = run["config"].num_cells.x

    y = num_cells // 2
    z = num_cells // 2
    density = np.array(state[registered_variables.density_index])[:, y, z]
    pressure = np.array(state[registered_variables.pressure_index])[:, y, z]
    velocity_x = np.array(
        state[registered_variables.velocity_index.x]
    )[:, y, z]
    x = np.array(helper_data.geometric_centers)[:, y, z, 0] - BOX_SIZE / 2
    return x, density, pressure, velocity_x
