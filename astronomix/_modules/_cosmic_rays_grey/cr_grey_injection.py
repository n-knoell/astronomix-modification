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
from astronomix.option_classes.simulation_config import STATE_TYPE, FIELD_TYPE
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    DSA_EFFICIENCY_CONSTANT,
    DSA_EFFICIENCY_KANG_RYU_2013,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer


@jax.jit
def dsa_efficiency_kang_ryu_2013(mach_numbers: FIELD_TYPE) -> FIELD_TYPE:
    """DSA acceleration efficiency vs. sonic Mach number (Kang & Ryu 2013).

    Piecewise fit to KR13's kinetic DSA simulation results, in the form
    standard across the cluster/cosmological CR literature (e.g. Vazza et al.
    2016; the CRESCENDO code, Girichidis et al. 2022) -- KR13 itself reports
    simulation tables/figures rather than a closed-form fit. Continuous
    across both piece boundaries (checked by hand: ~2% agreement at Ms=5,
    <1% at Ms=15) and consistent with KR13's own stated asymptotic value
    (eta -> ~0.2 for Ms >~ 10).

    Caprioli & Spitkovsky (2014) has no independent closed-form fit either;
    the literature's standard approximation is half of this function's
    output (Vazza et al. 2016) -- see
    ``CosmicRayGreyParams.dsa_efficiency_mach_scale``.

    Args:
        mach_numbers: Sonic Mach number field (e.g.
            ``ShockFinderResult.mach_numbers``); zero, negative, or any value
            away from a detected shock surface is fine (the fit below is
            zero for Ms < 2, so those cells contribute nothing).

    Returns:
        Efficiency field, same shape as ``mach_numbers``, in [0, ~0.211].
    """
    # ms**4 in the denominator of the intermediate piece below is only ever
    # evaluated (for the *returned value*) where Ms > 5, but jnp.where's VJP
    # differentiates every branch everywhere -- an un-selected 1/Ms**4 at
    # Ms=0 (the typical "no shock here" value away from shock-surface cells)
    # has an infinite local gradient, which contaminates the total gradient
    # via 0 * inf = nan even though the forward value is masked out cleanly.
    # Same "nan-safe forward, nan-unsafe backward" jnp.where gotcha this
    # module has hit before (see cr_pressure_speed_floor); floor the value
    # used *inside* the discardable branches instead of the real Mach number
    # driving the branch selection.
    ms = jnp.maximum(mach_numbers, 0.0)
    ms_safe = jnp.maximum(ms, 1.0)

    # weak-shock piece, 2 <= Ms <= 5
    c, d, e = -5.95e-4, 1.88e-5, 5.334
    weak = c + d * ms**e

    # intermediate piece, 5 < Ms <= 15
    b = (-2.87, 9.67, -8.88, 1.94, 0.18)
    intermediate = sum(bn * (ms_safe - 1.0) ** n for n, bn in enumerate(b)) / ms_safe**4

    # strong-shock plateau, Ms > 15
    strong = 0.211

    return jnp.where(
        ms < 2.0,
        0.0,
        jnp.where(ms <= 5.0, weak, jnp.where(ms <= 15.0, intermediate, strong)),
    )


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

    A fraction of that dissipated flux -- ``dsa_efficiency`` (constant) or
    ``dsa_efficiency_mach_scale * dsa_efficiency_kang_ryu_2013(sf_result.
    mach_numbers)`` (Mach-dependent), per ``config.cosmic_ray_grey_config.
    dsa_efficiency_model`` -- is diverted from the gas thermal channel into
    ``e_cr`` instead -- exact, local energy conservation (kinetic
    energy/velocity untouched; only the portion of thermal energy the shock
    would have deposited this step is repartitioned). Cartesian, uniform
    ``grid_spacing`` only for now
    (matches every grey-CR ladder test to date, and the plan's own FV-first
    staging): a shock cell's cross-sectional area is
    ``grid_spacing^(dimensionality - 1)`` and its volume is
    ``grid_spacing^dimensionality``, so area / volume = ``1 /
    grid_spacing`` -- mirrors the retired old model's identical Cartesian
    formula (see ``git log`` for ``cr_injection.py``'s prior
    ``config.grid_spacing ** (config.dimensionality - 1)``). Curvilinear/
    spherical geometry is not supported here; extend if a future ladder item
    needs it.

    Ladder item 7's fixed, Mach-independent ``dsa_efficiency`` remains the
    default (``dsa_efficiency_model == DSA_EFFICIENCY_CONSTANT``); ladder
    item 8 adds ``DSA_EFFICIENCY_KANG_RYU_2013`` as an opt-in alternative
    (see ``dsa_efficiency_kang_ryu_2013`` above) without changing the
    injection mechanics or the default behavior any existing config/test
    relies on.
    """
    cr_params = params.cosmic_ray_grey_params
    cr_config = config.cosmic_ray_grey_config
    gamma_gas = params.gamma

    sf_result = find_shocks_pfrommer(
        primitive_state,
        config,
        registered_variables,
        helper_data,
        mach_min=cr_params.dsa_mach_min,
    )

    if cr_config.dsa_efficiency_model == DSA_EFFICIENCY_KANG_RYU_2013:
        efficiency = cr_params.dsa_efficiency_mach_scale * dsa_efficiency_kang_ryu_2013(
            sf_result.mach_numbers
        )
    elif cr_config.dsa_efficiency_model == DSA_EFFICIENCY_CONSTANT:
        efficiency = cr_params.dsa_efficiency
    else:
        raise ValueError("Invalid dsa_efficiency_model")

    delta_e_cr_density = (
        efficiency * sf_result.thermal_energy_flux / config.grid_spacing * dt
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
