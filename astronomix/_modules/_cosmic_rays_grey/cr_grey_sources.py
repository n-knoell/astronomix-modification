"""
Grey CR feedback source terms (plan Sec. 2): the coupling from the
independently-tracked ``e_cr``/``F_cr`` state back into the gas.

Each function returns a conserved-state-shaped **rate** (not yet multiplied
by ``dt``), matching the convention of the existing FD source terms in
``astronomix._modules._time_integrator_sources`` (e.g. ``fd_viscosity_source``),
so callers there compose them as ``source_term += cr_xxx_source(...) * dt``.
Phase A scaffolding: typed signatures, ``NotImplementedError`` bodies.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._cosmic_rays_grey.cr_grey_fluid_equations import (
    pressure_from_e_cr,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_transport import (
    regularized_streaming_sign,
)
from astronomix._stencil_operations._stencil_operations import _stencil_add


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def cr_pressure_gradient_source(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
) -> STATE_TYPE:
    """``-grad(P_cr)`` added to the gas momentum equation -- the CR-driven
    wind/outflow forcing term (plan Sec. 2).

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        config: The simulation configuration.
        registered_variables: The registered variables.
        params: The simulation parameters.

    Returns:
        The momentum-row source-term rate.

    Implementation note: this also adds the matching work-rate
    ``-v . grad(P_cr)`` to the gas total-energy row (``pressure_index``
    doubles as the energy slot in the conserved layout) -- the same
    momentum-source-dot-velocity pattern
    ``_gravity._gravitational_source_term_along_axis`` uses for self-gravity
    (there: momentum row gets ``rho * a``, energy row gets ``rho * v * a``).
    Without it, the CR pressure force would still push the gas but never pay
    for the kinetic energy it hands out, breaking total-energy conservation
    (plan Sec. 4, test 18). Combined with
    :func:`cr_adiabatic_work_source`'s ``-P_cr * div(v)`` on ``e_cr``, the
    product-rule identity ``div(P_cr v) = v . grad(P_cr) + P_cr div(v)``
    makes gas-energy-gain and e_cr-loss exactly cancel up to a
    ``div(P_cr v)`` flux term -- i.e. locally energy-conserving, not just on
    integration over a periodic domain.
    """
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr
    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    p_cr = pressure_from_e_cr(e_cr, gamma_cr)

    source_term = jnp.zeros_like(primitive_state)
    energy_source = jnp.zeros_like(p_cr)

    for axis in range(1, config.dimensionality + 1):
        # 2nd-order centered gradient; the stencil axis is ``axis - 1``
        # because ``p_cr`` has no leading variable axis (see
        # _gravity._gravitational_source_term_along_axis's identical
        # pattern for the gravitational-potential gradient).
        grad_p_cr_axis = _stencil_add(
            p_cr, indices=(1, -1), factors=(1.0, -1.0), axis=axis - 1
        ) / (2 * config.grid_spacing)

        v_axis = primitive_state[axis]

        # momentum source: -grad(P_cr). ``axis`` is itself the momentum-row
        # index for this axis (velocity_index is allocated at rows 1/2/3),
        # matching every other FV hot-path use of ``flux_direction_index``/
        # ``axis`` as a direct row index (e.g. hll.py, evolve_state.py).
        source_term = source_term.at[axis].add(-grad_p_cr_axis)

        # work done on the gas by that force: -v . grad(P_cr), accumulated
        # over axes.
        energy_source = energy_source - v_axis * grad_p_cr_axis

    source_term = source_term.at[registered_variables.pressure_index].add(
        energy_source
    )

    return source_term


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def cr_adiabatic_work_source(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
) -> STATE_TYPE:
    """Adiabatic ``-P_cr * div(v)`` work term on ``e_cr`` (plan Sec. 2).

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        config: The simulation configuration.
        registered_variables: The registered variables.
        params: The simulation parameters.

    Returns:
        The ``e_cr``-row source-term rate.

    See :func:`cr_pressure_gradient_source`'s docstring for how this pairs
    with the gas-energy work-rate term to conserve total energy.
    """
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr
    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    p_cr = pressure_from_e_cr(e_cr, gamma_cr)

    # div(v), matching the pattern in
    # _finite_volume._riemann_solver.hll._am_hllc_solver's identical
    # divergence computation.
    div_v = sum(
        _stencil_add(
            primitive_state[axis], indices=(1, -1), factors=(1.0, -1.0), axis=axis - 1
        )
        / (2 * config.grid_spacing)
        for axis in range(1, config.dimensionality + 1)
    )

    source_term = jnp.zeros_like(primitive_state)
    source_term = source_term.at[registered_variables.cosmic_ray_e_index].set(
        -p_cr * div_v
    )

    return source_term


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def cr_flux_relaxation_source(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
) -> STATE_TYPE:
    """``F_cr`` scattering/relaxation term (Jiang & Oh 2018), ``-nu * F_cr``
    with ``nu = reduced_streaming_speed^2 / diffusion_coefficient``. Only
    active when ``config.cosmic_ray_grey_config.diffusive_relaxation``.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        config: The simulation configuration.
        registered_variables: The registered variables.
        params: The simulation parameters (carries
            ``params.cosmic_ray_grey_params.diffusion_coefficient``).

    Returns:
        The ``F_cr``-row source-term rate.

    Without this term, ``grey_cr_flux_terms``'s two-moment system is a pure
    undamped wave equation (verified in ladder items 1-3: a localized e_cr
    bump propagates rigidly, it does not spread). This term damps ``F_cr``
    at rate ``nu`` toward its flux-gradient forcing
    (``-v_red^2 * grad(P_cr)``, from ``grey_cr_flux_terms``'s pressure-driving
    term); at a quasi-steady balance (``d(F_cr)/dt ~ 0`` on timescales long
    compared to ``1/nu``, ignoring bulk advection) this gives
    ``F_cr ~= -diffusion_coefficient * grad(P_cr)``, i.e. Fick's law with
    diffusion coefficient ``diffusion_coefficient * (gamma_cr - 1)`` for
    ``e_cr`` itself (``P_cr = (gamma_cr - 1) * e_cr``) -- see
    ``cr_isotropic_diffusion_convergence.py`` (ladder item 4).

    Applied as a plain additive rate via the same explicit
    ``source_term * dt`` composition as every other CR-grey source in
    ``_time_integrator_sources`` (no special implicit/exact-exponential
    treatment) -- this reintroduces a genuine parabolic-like CFL constraint
    (``dt <~ diffusion_coefficient / reduced_streaming_speed^2``), handled the
    same way the existing viscosity module's ``dt_visc`` constrains
    ``_cfl_time_step`` (see that function's ``diffusive_relaxation`` branch).
    Deliberately opt-in (``diffusive_relaxation`` defaults to False) so
    ladder items 1-3's already-verified undamped-wave behavior is unchanged.
    """
    diffusion_coefficient = params.cosmic_ray_grey_params.diffusion_coefficient
    reduced_streaming_speed = params.cosmic_ray_grey_params.reduced_streaming_speed
    relaxation_rate = reduced_streaming_speed**2 / diffusion_coefficient

    f_cr_index = registered_variables.cosmic_ray_flux_index
    source_term = jnp.zeros_like(primitive_state)
    for axis_index in (
        (f_cr_index,)
        if config.dimensionality == 1
        else (f_cr_index.x, f_cr_index.y, f_cr_index.z)[: config.dimensionality]
    ):
        source_term = source_term.at[axis_index].set(
            -relaxation_rate * primitive_state[axis_index]
        )

    return source_term


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def cr_streaming_heating_source(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
) -> STATE_TYPE:
    """Streaming losses on ``e_cr`` deposited as heating into the gas thermal
    energy (plan Sec. 2). Only active when
    ``config.cosmic_ray_grey_config.streaming``; uses
    :func:`astronomix._modules._cosmic_rays_grey.cr_grey_transport.regularized_streaming_sign`
    for the direction of the streaming flux along B.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        config: The simulation configuration.
        registered_variables: The registered variables.
        params: The simulation parameters.

    Returns:
        The combined ``e_cr``/gas-thermal source-term rate.

    Implementation note: the streaming velocity that
    ``cr_grey_transport.streaming_flux_target`` pins ``F_cr`` to
    (``v_st,axis = -sign(dP_cr/dx_axis) * reduced_streaming_speed``, i.e.
    always directed down the local CR pressure gradient) does work against
    that gradient as it streams -- the standard CR-streaming heating picture
    (Wiener et al. 2017): the streaming instability that self-confines the
    CRs damps into gas heat at rate ``Gamma = -v_st . grad(P_cr) =
    reduced_streaming_speed * |grad(P_cr)|`` (per axis, regularized-sign
    version below), always ``>= 0``. This is a genuine, non-conservative
    loss from ``e_cr`` -- distinct from ``streaming_flux_target``'s
    conservative spatial redistribution of ``e_cr`` (that alone moves
    energy around without creating or destroying it). ``e_cr`` always loses
    the full rate ``Gamma``; only the ``streaming_heating_efficiency``
    fraction reappears as gas-thermal heating (the rest represents energy
    escaping into channels this grey model doesn't track, e.g. into
    higher-frequency wave turbulence) -- so this term is exactly conservative
    only when ``streaming_heating_efficiency == 1`` (the default).
    """
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr
    reduced_streaming_speed = params.cosmic_ray_grey_params.reduced_streaming_speed
    efficiency = params.cosmic_ray_grey_params.streaming_heating_efficiency
    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    p_cr = pressure_from_e_cr(e_cr, gamma_cr)

    heating_rate = jnp.zeros_like(p_cr)
    for axis in range(1, config.dimensionality + 1):
        grad_p_cr_axis = _stencil_add(
            p_cr, indices=(1, -1), factors=(1.0, -1.0), axis=axis - 1
        ) / (2 * config.grid_spacing)
        sign = regularized_streaming_sign(
            grad_p_cr_axis, params.cosmic_ray_grey_params
        )
        # sign(x) * x -> |x| away from the tanh regularization scale.
        heating_rate = heating_rate + reduced_streaming_speed * sign * grad_p_cr_axis

    source_term = jnp.zeros_like(primitive_state)
    source_term = source_term.at[registered_variables.cosmic_ray_e_index].add(
        -heating_rate
    )
    source_term = source_term.at[registered_variables.pressure_index].add(
        efficiency * heating_rate
    )

    return source_term
