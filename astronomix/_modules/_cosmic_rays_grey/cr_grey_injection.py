"""
Diffusive shock acceleration (DSA) injection for the grey two-moment CR model
(plan Sec. 2 "Injection"; test ladder items 7-8, Phase B).

Detects shocks with the general N-D Pfrommer shock finder
(``astronomix.shock_finder3D.pfrommer_shock_finder.find_shocks_pfrommer``,
PR #4) and converts a configurable fraction of each shock's dissipated
kinetic-energy flux into cosmic-ray energy density, deposited directly into
``e_cr`` at the shock-surface cells the finder reports.

This replaces the retired single-scalar ``_cosmic_rays`` model's
``inject_crs_at_strongest_shock`` (see this module's DESIGN.md,
"Consolidating with the old ``_cosmic_rays`` model"), which (a) only ever
injected at the single strongest shock in a 1D-only domain, and (b) targeted
a different, legacy 1D-only shock finder that predates PR #4. This
implementation injects at *every* shock-surface cell the N-D finder reports
-- it already supports multiple simultaneous shocks natively (one boolean
array over the whole domain), so no "strongest shock" reduction is needed or
performed -- and is dimensionality-agnostic (1D/2D/3D), matching
``find_shocks_pfrommer``'s own scope.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax.numpy as jnp
import jax

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def inject_crs_at_shocks(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    current_time: Union[float, Float[Array, ""]],
    dt: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    """Inject cosmic-ray energy at every detected shock-surface cell.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters (gas ``gamma`` and
            ``params.cosmic_ray_grey_params``' DSA knobs).
        registered_variables: The registered variables.
        helper_data: The helper data (passed through to the shock finder).
        current_time: The current simulation time.
        dt: The time step.

    Returns:
        The primitive state array with injected cosmic rays.

    Implementation note -- energy accounting: ``find_shocks_pfrommer``'s
    ``thermal_energy_flux`` is the Rankine-Hugoniot-predicted dissipated
    kinetic-energy flux at each shock-surface cell (energy / area / time),
    already zero everywhere except those cells (see
    ``calculate_thermal_energy_flux``'s own boundary/masking logic) -- so no
    additional masking by ``shock_surface_cells`` is needed here; a
    domain with no shock yet (e.g. before one has formed, or below
    ``dsa_mach_min``) naturally yields an all-zero flux and this function is
    a no-op, without needing an explicit "any shock present" branch the way
    the retired old model's ``jax.lax.cond`` gate did.

    A fixed ``dsa_efficiency`` fraction of that dissipated flux is diverted
    from the gas thermal channel into ``e_cr`` instead -- exact, local
    energy conservation (kinetic energy/velocity untouched; only the portion
    of thermal energy the shock would have deposited this step is
    repartitioned). Cartesian, uniform ``grid_spacing`` only for now
    (matches every grey-CR ladder test to date, and the plan's own FV-first
    staging): a shock cell's cross-sectional area is
    ``grid_spacing^(dimensionality - 1)`` and its volume is
    ``grid_spacing^dimensionality``, so area / volume = ``1 /
    grid_spacing`` -- mirrors the retired old model's identical Cartesian
    formula (see ``git log`` for ``cr_injection.py``'s prior
    ``config.grid_spacing ** (config.dimensionality - 1)``). Curvilinear/
    spherical geometry is not supported here; extend if a future ladder item
    needs it.

    Ladder item 7 uses a fixed, Mach-independent ``dsa_efficiency`` (mirrors
    the retired old model's constant efficiency knob). Ladder item 8's
    Mach-dependent DSA efficiency (Kang & Ryu 2013; Caprioli & Spitkovsky
    2014) is a separate follow-up: it only needs to replace the scalar
    ``dsa_efficiency`` below with a function of ``sf_result.mach_numbers``,
    not change the injection mechanics.
    """
    cr_params = params.cosmic_ray_grey_params
    gamma_gas = params.gamma

    sf_result = find_shocks_pfrommer(
        primitive_state,
        config,
        registered_variables,
        helper_data,
        mach_min=cr_params.dsa_mach_min,
    )

    delta_e_cr_density = (
        cr_params.dsa_efficiency * sf_result.thermal_energy_flux / config.grid_spacing * dt
    )

    # Injecting only after a certain amount of time is an ad-hoc guard
    # against spurious detections before a real shock has formed (mirrors
    # the retired old model's identical ``diffusive_shock_acceleration_start_time``
    # convention); defaults to 0.0 (no delay).
    started = current_time >= cr_params.dsa_start_time
    delta_e_cr_density = jnp.where(started, delta_e_cr_density, 0.0)

    e_cr_new = (
        primitive_state[registered_variables.cosmic_ray_e_index] + delta_e_cr_density
    )
    p_gas_new = (
        primitive_state[registered_variables.pressure_index]
        - delta_e_cr_density * (gamma_gas - 1.0)
    )

    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(
        e_cr_new
    )
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        p_gas_new
    )

    return primitive_state
