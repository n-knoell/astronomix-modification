"""
Two-moment grey CR transport (Jiang & Oh 2018; Thomas & Pfrommer 2019).

Phase A scaffolding: typed signatures for every transport hook the flux,
Riemann-solver and CFL code needs, with physics bodies left as
``NotImplementedError`` stubs to be filled in test-first (see
``pytests/cosmic_rays_grey/`` and this module's ``DESIGN.md``). The one
implemented function, :func:`regularized_streaming_sign`, is a
self-contained utility with an unambiguous definition (plan Sec. 2:
"regularize the streaming sign with a tanh to keep the adjoint clean") and
does not depend on the rest of the still-open transport design.

WARNING: ``gamma_cr`` and ``reduced_streaming_speed`` are fixed here as
module-level constants rather than read from ``CosmicRayGreyParams``, for the
same reason the older ``cr_fluid_equations.py`` hardcodes its adiabatic
indices: none of ``_euler_flux``, ``_hll_solver``/``_hllc_solver``,
``_reconstruct_at_interface_split`` or ``get_wave_speeds`` currently receive
``SimulationParams`` (only some of their siblings, e.g.
``_reconstruct_at_interface_unsplit``, do -- the plumbing is already
inconsistent across the FV hot path). Threading ``params`` through all of
them is real, separate follow-up work, not part of this scaffold; until then
these constants are the values to move into ``CosmicRayGreyParams`` lookups.
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
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyParams,
)

# See module WARNING above -- mirror these into CosmicRayGreyParams lookups
# once params is threaded through the FV hot path.
gamma_cr = 4.0 / 3.0
reduced_streaming_speed = 1.0


@partial(
    jax.jit, static_argnames=["config", "registered_variables", "flux_direction_index"]
)
def grey_cr_flux_terms(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int,
) -> STATE_TYPE:
    """Two-moment flux contribution for the ``e_cr``/``F_cr`` rows.

    Unlike a passively advected scalar (e.g. the old ``n_cr``, or
    ``wind_density``), ``e_cr`` is transported by the independent flux
    variable ``F_cr`` rather than by ``u * e_cr``, and ``F_cr`` itself has a
    reduced-speed closure flux -- so both rows need an explicit flux term
    here rather than relying on ``_euler_flux``'s generic ``u_n * q``
    pass-through for unlisted rows.

    Called from ``astronomix._fluid_equations._fluxes._euler_flux`` when
    ``registered_variables.cosmic_ray_e_active``, and added onto the
    ``cosmic_ray_e_index``/``cosmic_ray_flux_index`` rows of the flux vector.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        gamma: The gas adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.
        flux_direction_index: The index of the velocity component in the
            flux direction of interest.

    Returns:
        The flux contribution to add onto the ``e_cr``/``F_cr`` rows.
    """
    raise NotImplementedError(
        "Phase A: two-moment e_cr/F_cr flux terms (Jiang & Oh 2018 closure). "
        "See DESIGN.md."
    )


@partial(jax.jit, static_argnames=["registered_variables"])
def grey_cr_fast_speed(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
) -> Float[Array, "..."]:
    """Maximum signal speed of the two-moment CR subsystem.

    This is the reduced free-streaming speed (a tunable accuracy/cost knob,
    plan Sec. 2), not folded into the gas sound speed the way the old
    polytropic model's ``speed_of_sound_crs`` was -- the two subsystems are
    combined as ``max(u_gas + c_gas, v_cr_fast)`` at each call site (hll.py,
    reconstruction.py, the CFL estimator) rather than added inside one
    effective sound speed.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        registered_variables: The registered variables.

    Returns:
        The CR-grey fast/reduced-streaming speed.
    """
    raise NotImplementedError(
        "Phase A: CR-grey reduced free-streaming speed for the CFL/Riemann "
        "wave-speed bound. See DESIGN.md."
    )


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def anisotropic_flux_projection(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Project the CR flux/diffusion along the local magnetic-field direction.

    Monotonicity-safe operator (Sharma & Hammett 2007): heat/CR energy stays
    along B, no cross-field leak. Only meaningful when
    ``config.cosmic_ray_grey_config.anisotropic_transport`` is set; the
    isotropic closure is the default.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The B-projected CR flux.
    """
    raise NotImplementedError(
        "Phase A: Sharma & Hammett (2007) anisotropic projection. See DESIGN.md."
    )


@jax.jit
def regularized_streaming_sign(
    x: Float[Array, "..."],
    params: CosmicRayGreyParams,
) -> Float[Array, "..."]:
    """Smooth (``tanh``) stand-in for ``sign(x)``, for a differentiable
    streaming term.

    ``sign``/``min``/``max`` all have zero or discontinuous gradients at
    ``x = 0``; a ``tanh`` with a small regularization scale keeps the adjoint
    finite there while matching ``sign(x)`` away from it (plan Sec. 2).

    Args:
        x: The quantity whose sign gates the streaming direction (e.g. the
            CR pressure gradient along B).
        params: The grey CR parameters (carries the regularization scale).

    Returns:
        ``tanh(x / params.streaming_sign_regularization)``.
    """
    return jnp.tanh(x / params.streaming_sign_regularization)
