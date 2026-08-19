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
    """
    raise NotImplementedError(
        "Phase A: CR streaming heating into gas thermal energy. See DESIGN.md."
    )
