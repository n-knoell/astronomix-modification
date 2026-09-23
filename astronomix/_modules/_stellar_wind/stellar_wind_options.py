"""
Configuration and parameter containers for stellar-wind injection.

``WindConfig`` holds the static choices (which injection scheme, how many cells
to inject into) and ``WindParams`` the physical wind parameters. The module-level
integer constants name the available injection schemes from
https://arxiv.org/abs/2107.14673.
"""

# typing
from typing import NamedTuple, Tuple, Union

# jax
import jax.numpy as jnp

# Wind injection schemes (see https://arxiv.org/abs/2107.14673).
MEO = 0  # momentum and energy overwrite
EI = 1  # thermal energy injection
MEI = 2  # momentum and energy injection


class WindConfig(NamedTuple):
    stellar_wind: bool = False
    num_injection_cells: int = 10
    wind_injection_scheme: int = EI
    trace_wind_density: bool = False

    #: Use tabulated, time-dependent stellar-evolution wind tracks
    #: (``WindParams.real_params``, see ``stellar_wind_functions.get_wind_parameters``)
    #: instead of the static ``wind_mass_loss_rate(s)`` / ``wind_final_velocity(ies)``.
    #: Only supported by the 3D EI scheme (``_wind_ei3D`` / ``_wind_ei3D_source``).
    real_wind_params: bool = False

    #: After the EI source term, overwrite an extended "wind zone" around each
    #: source with the analytic freely-expanding, adiabatically-cooled wind
    #: (``rho = Mdot / (4 pi r^2 v_inf)``, radial ``v_inf`` relative to the
    #: source, ``p = rho * T0 * (r0 / r)**(2 (gamma - 1))``), every step. Lets
    #: the wind arrive at the collision region with the thermal state it would
    #: have after expanding from the (unresolvable) photosphere, instead of
    #: the far too hot state a numerically large injection sphere produces
    #: (see pytests/shock_finder3D/FIXES_TODO.md, rounds 12 and 18). The zone
    #: radius is ``WindParams.wind_zone_stagnation_fraction`` times each
    #: source's distance to its nearest stagnation point. Only supported by
    #: the 3D EI scheme on the finite-volume path; ``False`` leaves the
    #: default injection untouched.
    analytic_wind_zone: bool = False


class WindParams(NamedTuple):
    # Single-source parameters, used by the 1D injection schemes
    # (MEO / MEI / EI).
    wind_mass_loss_rate: float = 0.0
    wind_final_velocity: float = 0.0

    # Only required for the MEO injection scheme.
    pressure_floor: float = 100000.0

    #: Per-source mass-loss rate / terminal velocity and injection positions
    #: for the 3D EI scheme (``_wind_ei3D`` / ``_wind_ei3D_source``), shapes
    #: (n_sources,) / (n_sources,) / (n_sources, 3). Positions are in the same
    #: box-centered coordinates as the N-body state (box center at the
    #: origin; see ``astronomix._modules._nbody._nbody``). Ignored — and
    #: overridden by the current N-body positions — when
    #: ``config.nbody_config.nbody`` is also enabled.
    wind_mass_loss_rates: jnp.ndarray = jnp.array([0.0])
    wind_final_velocities: jnp.ndarray = jnp.array([0.0])
    wind_injection_positions: jnp.ndarray = jnp.array([[0.0, 0.0, 0.0]])

    #: Tabulated (time, log-mass-loss-rate, wind-velocity) tracks, one row of
    #: tracks per source: a 3-tuple ``(time, mass_loss_rate, wind_velocity)``,
    #: each of shape (n_sources, n_track_points), already converted to code
    #: units by the caller (see ``stellar_wind_functions.get_wind_parameters``,
    #: which returns the raw Ekstrom+2012 track values and does not itself do
    #: any unit conversion). Interpolated at the current simulation time by
    #: ``stellar_wind_functions.get_current_wind_params`` in place of
    #: ``wind_mass_loss_rates`` / ``wind_final_velocities`` when
    #: ``config.wind_config.real_wind_params``.
    real_params: Union[Tuple, None] = None

    #: Analytic wind zone (``WindConfig.analytic_wind_zone``) only. Per-source
    #: wind base radius ``r0`` (code length, e.g. the photospheric radius) and
    #: base pseudo-temperature ``T0 = p / rho`` at ``r0`` (code units, the
    #: same convention the shock finder's Mach estimate uses), shapes
    #: (n_sources,). Anchor the adiabatic law ``T(r) = T0 (r0 / r)**(2 (gamma - 1))``.
    wind_base_radii: jnp.ndarray = jnp.array([1.0])
    wind_base_temperatures: jnp.ndarray = jnp.array([0.0])

    #: Analytic wind zone only. Each source's zone radius is this fraction of
    #: its distance to the nearest ram-pressure stagnation point (the point
    #: between two sources where ``Mdot_i v_i / r_i**2`` balance), clipped to
    #: at least the EI injection radius and at most ``wind_zone_max_radius``.
    #: Keep it in (0, 1) so zones never overlap and ordinary hydro cells
    #: remain in front of the shock. With a single source there is no
    #: stagnation point and the zone radius is ``wind_zone_max_radius``,
    #: which must then be set to a finite value.
    wind_zone_stagnation_fraction: float = 0.5
    wind_zone_max_radius: float = float("inf")

    #: Set internally, once per step, by the time-integration loop (see
    #: astronomix.time_stepping.time_integration) to the current simulation
    #: time. Only needed to make "now" available to the finite-difference
    #: source-term path (which runs per RK stage, without direct access to
    #: the absolute time) for the ``real_params`` time interpolation; not
    #: meant to be set directly by the user.
    current_time: float = 0.0
