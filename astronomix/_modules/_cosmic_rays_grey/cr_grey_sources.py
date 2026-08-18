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

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables


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
    """
    raise NotImplementedError(
        "Phase A: -grad(P_cr) momentum feedback source. See DESIGN.md."
    )


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
    """
    raise NotImplementedError(
        "Phase A: -P_cr div(v) adiabatic work term on e_cr. See DESIGN.md."
    )


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
