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
from astronomix.shock_finder3D._surface_area import surface_area_weights
from astronomix.shock_finder3D._shock_zones import sample_along_direction
from astronomix.shock_finder3D._energy_dissipation import (
    _thermalization_efficiency,
    thermal_energy_flux_from_pre_state,
)


def dsa_shock_finder_kwargs(cr_config) -> dict:
    """``find_shocks_pfrommer`` sampling options DSA injection uses for this
    ``CosmicRayGreyConfig`` -- for callers (tests, diagnostics) that need the
    same Mach numbers / fluxes the injection sees."""
    if cr_config.dsa_adaptive_shock_sampling:
        return dict(
            mach_sampling_adaptive=True,
            mach_sampling_extend=True,
            mach_sampling_steps=cr_config.dsa_shock_sampling_max_steps,
            mach_velocity_consistency=cr_config.dsa_velocity_consistency,
            mach_estimator="velocity" if cr_config.dsa_mach_from_velocity else "pressure",
        )
    return dict()


@partial(jax.jit, static_argnames=["max_steps"])
def _spread_over_post_shock_cells(
    amount, shock_direction, post_steps, pressure, pre_shock_pressure, max_steps,
    density=None, post_shock_density=None, gamma_cr=4.0 / 3.0,
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
    conserved, unless ``density`` and ``post_shock_density`` are given
    (``dsa_density_consistent_deposit``): then each share is further scaled
    by ``min(1, (rho_k / rho_post)^(gamma_cr - 1))``, with ``rho_post`` the
    surface cell's Rankine-Hugoniot post-shock density, so that the CRs a
    still under-compressed cell receives reach the intended energy only
    once it is compressed to ``rho_post``; the remainder is not deposited.

    Args:
        amount: Per-cell injection (energy density), nonzero at surface cells.
        shock_direction: Unit shock-direction field, shape (ndim, *shape),
            pointing to the pre-shock side.
        post_steps: Steps to each cell's post-shock sample
            (``ShockFinderResult.post_shock_steps``).
        pressure: Gas pressure field.
        pre_shock_pressure: Sampled pre-shock pressure per cell.
        max_steps: Maximum ray length (static).
        density: Gas density field (only with ``post_shock_density``).
        post_shock_density: Rankine-Hugoniot post-shock density per
            surface cell, or None (no density factor).
        gamma_cr: CR adiabatic index of the density factor.

    Returns:
        The redistributed injection field (same total as ``amount`` without
        the density factor, at most that with it).
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

    def density_factor(cell):
        if post_shock_density is None:
            return 1.0
        rho_k = density[tuple(cell[d] for d in range(ndim))]
        return jnp.minimum(1.0, (rho_k / post_shock_density) ** (gamma_cr - 1.0))

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
        ) * density_factor(cell)
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


def cr_aware_velocity_mach(
    mach_u, cr_ratio, efficiency_of_mach, mach_min, gamma, gamma_cr, iterations=60
):
    """Upstream gas Mach number ``v_1 / c_th,1`` of a CR-loaded shock with DSA
    injection from its normal-velocity jump (``dsa_mach_cr_aware``, Phase B DB8).

    ``mach_u`` is the finder's gas-only Rankine-Hugoniot reading of the jump
    (0 away from shocks), which fixes ``w = du / c_th,1 = 2 (M_u - 1 / M_u) /
    (gamma + 1)``; ``cr_ratio`` is ``X = P_cr,1 / P_th,1`` at the pre-shock
    sample. In units ``rho_1 = c_th,1 = 1`` and with the unknown ``s = rho_1 /
    rho_2`` (``M = w / (1 - s)``), mass and momentum conservation leave the
    energy jump with the upstream CRs compressed adiabatically and the injected
    specific energy ``zeta(M) delta(M) M^2 / 2`` (the code's convention, a
    fraction of the gas-only dissipated flux). It has exactly one root on
    ``[max(0.1, 1 - w), 1)`` (checked for X 0-2, zeta 0-0.5 / KR13, M
    1.5-100; exact to 3e-15 against the steady planar shocks), found by
    bisection. A final Newton step from the stop-gradient root gives the
    implicit-function derivatives (with respect to the jump, X and the
    efficiency parameters), which the bisection alone would not; it changes
    only the derivative, not the value.

    Args:
        mach_u: Gas-only velocity-jump Mach number (0 = no shock).
        cr_ratio: Upstream ``P_cr / P_th``.
        efficiency_of_mach: The DSA efficiency ``zeta(M)``.
        mach_min: Injection threshold (``zeta = 0`` at or below it).
        gamma: Gas adiabatic index.
        gamma_cr: CR adiabatic index.
        iterations: Bisection steps.

    Returns:
        ``(M, rho_2 / rho_1)``, M = 0 where ``mach_u`` is 0.
    """
    shock = mach_u > 0.0
    a = gamma / (gamma - 1.0)
    b = gamma_cr / (gamma_cr - 1.0)
    # harmless stand-ins away from shocks (masked below; keep gradients finite)
    m_u = jnp.where(shock, jnp.maximum(mach_u, 1.0), 2.0)
    w = jnp.maximum(2.0 * (m_u - 1.0 / m_u) / (gamma + 1.0), 1e-6)
    x_cr = jnp.where(shock, jnp.maximum(cr_ratio, 0.0), 0.0)
    p_1 = (1.0 + x_cr) / gamma
    h_1 = (a + b * x_cr) / gamma

    def residual(s):
        m = w / (1.0 - s)
        zeta = jnp.where(m > mach_min, efficiency_of_mach(m), 0.0)
        zeta_delta = zeta * _thermalization_efficiency(m, gamma)
        return (
            0.5 * m**2 * (1.0 - s**2) + h_1 - a * s * (p_1 + m**2 * (1.0 - s))
            - (b - a) * (
                x_cr / gamma * s ** (1.0 - gamma_cr)
                + (gamma_cr - 1.0) * zeta_delta * 0.5 * m**2
            )
        )

    lower = jnp.maximum(0.1, 1.0 - jax.lax.stop_gradient(w))
    upper = jnp.ones_like(w) - 1e-6

    # while_loop rather than fori_loop: XLA unrolls a short fixed-length
    # fori_loop and keeps every iteration's full-grid temporaries
    def bisect(carry):
        i, lo, hi = carry
        mid = 0.5 * (lo + hi)
        positive = jax.lax.stop_gradient(residual(mid)) > 0.0
        return i + 1, jnp.where(positive, mid, lo), jnp.where(positive, hi, mid)

    _, lower, upper = jax.lax.while_loop(
        lambda carry: carry[0] < iterations, bisect, (0, lower, upper)
    )
    s_root = jax.lax.stop_gradient(0.5 * (lower + upper))
    r, dr_ds = jax.jvp(residual, (s_root,), (jnp.ones_like(s_root),))
    dr_ds = jax.lax.stop_gradient(dr_ds)
    newton = -r / jnp.where(dr_ds != 0.0, dr_ds, 1.0)
    # value: the bisection root (a Newton step could leave it where zeta(M)
    # jumps); derivative: the implicit-function one, -dr/dtheta / dr/ds
    s_root = s_root + (newton - jax.lax.stop_gradient(newton))
    mach = jnp.where(shock, w / (1.0 - s_root), 0.0)
    return mach, 1.0 / s_root


def dsa_efficiency_of_mach(cr_config, cr_params):
    """The DSA efficiency ``zeta(M)`` of ``cr_config.dsa_efficiency_model`` as a
    function of the Mach-number field."""
    if cr_config.dsa_efficiency_model == DSA_EFFICIENCY_KANG_RYU_2013:
        def efficiency_of_mach(mach):
            return cr_params.dsa_efficiency_mach_scale * dsa_efficiency_kang_ryu_2013(mach)
    elif cr_config.dsa_efficiency_model == DSA_EFFICIENCY_CONSTANT:
        def efficiency_of_mach(mach):
            return cr_params.dsa_efficiency * jnp.ones_like(mach)
    else:
        raise ValueError("Invalid dsa_efficiency_model")
    return efficiency_of_mach


def dsa_mach_and_flux(primitive_state, sf_result, config, params, registered_variables):
    """Mach number and dissipated thermal-energy flux per shock-surface cell
    that DSA injection uses, from the finder result of
    ``find_shocks_pfrommer(..., **dsa_shock_finder_kwargs(cr_config))``: the
    finder's own, or with ``dsa_mach_cr_aware`` (Phase B DB8) recomputed from
    its velocity jump with the CR-loaded Hugoniot. For callers (tests,
    diagnostics) that need the same values as ``inject_crs_at_shocks``.

    Returns:
        ``(mach_numbers, thermal_energy_flux)``, zero away from injecting cells.
    """
    cr_params = params.cosmic_ray_grey_params
    cr_config = config.cosmic_ray_grey_config
    gamma_gas = params.gamma
    mach_numbers = sf_result.mach_numbers
    thermal_energy_flux = sf_result.thermal_energy_flux
    cr_aware = cr_config.dsa_mach_cr_aware
    if cr_aware is None:  # config not finalized: resolve as resolve_auto_defaults does
        cr_aware = cr_config.dsa_adaptive_shock_sampling and cr_config.dsa_mach_from_velocity
    if not cr_aware:
        return mach_numbers, thermal_energy_flux
    if not (cr_config.dsa_adaptive_shock_sampling and cr_config.dsa_mach_from_velocity):
        raise ValueError(
            "dsa_mach_cr_aware needs dsa_adaptive_shock_sampling and dsa_mach_from_velocity"
        )
    # the gas-only reading of the velocity jump is biased by the injected
    # CRs' extra compression and by upstream CR pressure
    e_cr_pre = sample_along_direction(
        primitive_state[registered_variables.cosmic_ray_e_index],
        jax.lax.stop_gradient(sf_result.shock_direction),
        sf_result.pre_shock_steps,
    )
    cr_ratio = (cr_params.gamma_cr - 1.0) * e_cr_pre / jnp.maximum(
        sf_result.pre_shock_pressure, 1e-30
    )
    mach_numbers, _ = cr_aware_velocity_mach(
        sf_result.mach_numbers, cr_ratio, dsa_efficiency_of_mach(cr_config, cr_params),
        cr_params.dsa_mach_min, gamma_gas, cr_params.gamma_cr,
    )
    valid = mach_numbers > cr_params.dsa_mach_min
    mach_numbers = jnp.where(valid, mach_numbers, 0.0)
    thermal_energy_flux = jnp.where(
        valid,
        thermal_energy_flux_from_pre_state(
            mach_numbers, sf_result.pre_shock_pressure, sf_result.pre_shock_density,
            gamma_gas,
        ),
        0.0,
    )
    return mach_numbers, thermal_energy_flux


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
    ``grid_spacing^(dimensionality - 1) / |n|_inf`` (``dsa_area_weighting``,
    default since 2026-10-19: an oblique shock has ``|n|_inf`` surface cells
    per ``grid_spacing^(dimensionality - 1)`` of its area) and its volume is
    ``grid_spacing^dimensionality``, so area / volume = ``1 /
    (|n|_inf grid_spacing)`` (``1 / grid_spacing`` without the weighting) -- mirrors the retired old model's identical Cartesian
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

    With ``dsa_mach_cr_aware`` (the default with adaptive sampling and the
    velocity-jump Mach number, Phase B DB8) the Mach number and the
    dissipated flux are recomputed from the finder's velocity jump with the
    CR-loaded Hugoniot (``cr_aware_velocity_mach``) before the efficiency is
    applied.
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

    efficiency_of_mach = dsa_efficiency_of_mach(cr_config, cr_params)
    mach_numbers, thermal_energy_flux = dsa_mach_and_flux(
        primitive_state, sf_result, config, params, registered_variables
    )

    delta_e_cr_density = (
        efficiency_of_mach(mach_numbers) * thermal_energy_flux / config.grid_spacing * dt
    )

    if cr_config.dsa_area_weighting:
        # A surface cell represents dx^(ndim-1) / |n|_inf of shock area, not
        # dx^(ndim-1) (Phase B DB7; shock_finder3D._surface_area).
        delta_e_cr_density = delta_e_cr_density * surface_area_weights(
            primitive_state[registered_variables.pressure_index],
            primitive_state[registered_variables.density_index],
            sigma=cr_config.dsa_area_normal_smoothing,
        )

    if cr_config.dsa_adaptive_shock_sampling:
        # Spread each shock's injection over its surface cell and the
        # numerically broadened post-shock cells behind it (Pfrommer et al.
        # 2017, Sec. 3.1.2), where the dissipated heat actually sits. With the
        # adaptive sampling the flux is the full dissipated flux of the
        # shock; the surface cell alone (often the foot of the smeared shock)
        # can hold far less thermal energy than that (FIXES_TODO.md round 24).
        if cr_config.dsa_density_consistent_deposit:
            # gas-only RH post-shock density of each surface cell (Phase B
            # DB9): deposit only the share the cell's CRs will have once it
            # is compressed to it (see _spread_over_post_shock_cells). From
            # the finder's gas-only Mach number even with dsa_mach_cr_aware:
            # the CR-aware Mach number / compression over-corrects (B7.4
            # zeta 0.1: -1.7% instead of -0.8%, 2026-10-27)
            mach_sq = sf_result.mach_numbers**2
            post_shock_density = (
                sf_result.pre_shock_density * (gamma_gas + 1.0) * mach_sq
                / ((gamma_gas - 1.0) * mach_sq + 2.0)
            )
            density_kwargs = dict(
                density=primitive_state[registered_variables.density_index],
                # zero away from shock cells (amount = 0 there): keep finite
                post_shock_density=jnp.maximum(post_shock_density, params.minimum_density),
                gamma_cr=cr_params.gamma_cr,
            )
        else:
            density_kwargs = dict()
        delta_e_cr_density = _spread_over_post_shock_cells(
            delta_e_cr_density,
            sf_result.shock_direction,
            sf_result.post_shock_steps,
            primitive_state[registered_variables.pressure_index],
            sf_result.pre_shock_pressure,
            cr_config.dsa_shock_sampling_max_steps,
            **density_kwargs,
        )
    elif cr_config.dsa_density_consistent_deposit:
        raise ValueError(
            "dsa_density_consistent_deposit needs dsa_adaptive_shock_sampling"
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
