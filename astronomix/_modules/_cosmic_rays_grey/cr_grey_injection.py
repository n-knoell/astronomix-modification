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


def dsa_shock_finder_kwargs(cr_config) -> dict:
    """``find_shocks_pfrommer`` sampling options DSA injection uses for this
    ``CosmicRayGreyConfig`` -- for callers (tests, diagnostics) that need the
    same Mach numbers / fluxes the injection sees."""
    if cr_config.dsa_adaptive_shock_sampling:
        return dict(
            mach_sampling_adaptive=True,
            mach_sampling_extend=True,
            mach_sampling_steps=cr_config.dsa_shock_sampling_max_steps,
        )
    return dict()


@partial(jax.jit, static_argnames=["max_steps"])
def _spread_over_post_shock_cells(
    amount, shock_direction, post_steps, pressure, pre_shock_pressure, max_steps
):
    """Redistribute per-surface-cell injection over the post-shock cells.

    For every cell with ``amount > 0`` (a shock-surface cell), the cells
    ``k = 0 .. post_steps`` steps along ``-shock_direction`` (nearest grid
    cell; ``k = 0`` is the surface cell itself) each receive a share
    proportional to their pressure excess over the pre-shock pressure,
    ``max(p_k - p_pre, 0)`` -- i.e. to their share of the shock-heated thermal
    energy, the weighting of Pfrommer et al. (2017, Sec. 3.1.2) restricted to
    thermal energy since that is what the injection draws from. A ray with no
    positive weight keeps its amount in the surface cell. The total is
    conserved.

    Args:
        amount: Per-cell injection (energy density), nonzero at surface cells.
        shock_direction: Unit shock-direction field, shape (ndim, *shape),
            pointing to the pre-shock side.
        post_steps: Steps to each cell's post-shock sample
            (``ShockFinderResult.post_shock_steps``).
        pressure: Gas pressure field.
        pre_shock_pressure: Sampled pre-shock pressure per cell.
        max_steps: Maximum ray length (static).

    Returns:
        The redistributed injection field (same total as ``amount``).
    """
    shape = amount.shape
    ndim = amount.ndim
    base = jnp.stack(jnp.meshgrid(*[jnp.arange(n) for n in shape], indexing="ij"))
    upper = jnp.array(shape).reshape((ndim,) + (1,) * ndim) - 1

    def ray_cell(k):
        position = base - k * shock_direction
        return jnp.clip(jnp.round(position).astype(jnp.int32), 0, upper)

    def weight(k, cell):
        p_k = pressure[tuple(cell[d] for d in range(ndim))]
        on_ray = k <= post_steps
        return jnp.where(on_ray, jnp.maximum(p_k - pre_shock_pressure, 0.0), 0.0)

    def accumulate(k, total):
        return total + weight(k, ray_cell(k))

    total = jax.lax.fori_loop(0, max_steps + 1, accumulate, jnp.zeros(shape, amount.dtype))
    has_weight = total > 0
    safe_total = jnp.where(has_weight, total, 1.0)

    def deposit(k, out):
        cell = ray_cell(k)
        share = jnp.where(
            has_weight,
            amount * weight(k, cell) / safe_total,
            jnp.where(k == 0, amount, 0.0),
        )
        flat = jnp.ravel_multi_index(tuple(cell[d] for d in range(ndim)), shape, mode="clip")
        return out.at[flat.ravel()].add(share.ravel())

    out = jax.lax.fori_loop(0, max_steps + 1, deposit, jnp.zeros(amount.size, amount.dtype))
    return out.reshape(shape)


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
    would have deposited this step is repartitioned), **except when the
    nominal amount would drive that cell's gas pressure below
    ``params.minimum_pressure``, in which case the diverted amount is capped
    so it doesn't** -- see the positivity-floor note below. Cartesian, uniform
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
        **dsa_shock_finder_kwargs(cr_config),
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

    if cr_config.dsa_adaptive_shock_sampling:
        # Spread each shock's injection over its surface cell and the
        # numerically broadened post-shock cells behind it (Pfrommer et al.
        # 2017, Sec. 3.1.2), where the dissipated heat actually sits. With the
        # adaptive sampling the flux is the full dissipated flux of the
        # shock; the surface cell alone (often the foot of the smeared shock)
        # can hold far less thermal energy than that (FIXES_TODO.md round 24).
        delta_e_cr_density = _spread_over_post_shock_cells(
            delta_e_cr_density,
            sf_result.shock_direction,
            sf_result.post_shock_steps,
            primitive_state[registered_variables.pressure_index],
            sf_result.pre_shock_pressure,
            cr_config.dsa_shock_sampling_max_steps,
        )

    # Injecting only after a certain amount of time is an ad-hoc guard
    # against spurious detections before a real shock has formed (mirrors
    # the retired old model's identical ``diffusive_shock_acceleration_start_time``
    # convention); defaults to 0.0 (no delay).
    started = current_time >= cr_params.dsa_start_time
    delta_e_cr_density = jnp.where(started, delta_e_cr_density, 0.0)

    # Positivity floor: cap the diverted amount so p_gas_new can never drop
    # below params.minimum_pressure, mirroring the identical fix applied to
    # _apply_gravity_source (SILCC-ISM project M5 root cause -- see
    # DESIGN.md/PROGRESS.md). Without this, a shock-surface cell whose gas
    # pressure is already near the floor (e.g. a still-near-vacuum cell right
    # at the edge of a freshly-seeded, extremely fast wind/SN injection, where
    # find_shocks_pfrommer can report a very large apparent Mach number) can
    # have far more energy diverted than it actually has -- driving pressure
    # sharply negative and NaN-ing the very next Riemann solve. Bounding
    # delta_e_cr_density itself (rather than clamping p_gas_new after the
    # fact) keeps the e_cr/p_gas exchange exactly self-consistent: whatever
    # amount e_cr actually gains is exactly what gas thermal energy loses,
    # even in the clamped case.
    p_gas_pre = primitive_state[registered_variables.pressure_index]
    max_available = jnp.maximum(p_gas_pre - params.minimum_pressure, 0.0) / (
        gamma_gas - 1.0
    )
    delta_e_cr_density = jnp.minimum(delta_e_cr_density, max_available)

    e_cr_new = (
        primitive_state[registered_variables.cosmic_ray_e_index] + delta_e_cr_density
    )
    p_gas_new = p_gas_pre - delta_e_cr_density * (gamma_gas - 1.0)
    # max_available above bounds p_gas_new >= minimum_pressure only exactly
    # in infinite precision -- floating-point roundoff in this subtraction
    # can still leave it a hair below (observed: ~-4.5e-13), which is enough
    # for speed_of_sound's unguarded sqrt(gamma*p/rho) to NaN. Re-clamp here
    # to make the floor exact rather than "exact up to roundoff".
    p_gas_new = jnp.maximum(p_gas_new, params.minimum_pressure)

    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(
        e_cr_new
    )
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        p_gas_new
    )

    return primitive_state
