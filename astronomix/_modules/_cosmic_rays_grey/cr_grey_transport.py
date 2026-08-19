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

``params`` (``SimulationParams``, carrying ``CosmicRayGreyParams`` at
``params.cosmic_ray_grey_params``) is threaded through the full FV hot path
that needs it -- ``_euler_flux``, ``_hll_solver``/``_hllc_solver``/
``_am_hllc_solver``, ``_reconstruct_at_interface_split`` and
``get_wave_speeds`` all take ``params`` now -- so ``gamma_cr`` and
``reduced_streaming_speed`` are read from real, differentiable
``CosmicRayGreyParams`` values in every function below rather than
module-level constants (DESIGN.md's former "known scaffold gap", resolved).
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
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyParams,
)

# astronomix functions
from astronomix._modules._cosmic_rays_grey.cr_grey_fluid_equations import (
    pressure_from_e_cr,
)


@partial(
    jax.jit, static_argnames=["config", "registered_variables", "flux_direction_index"]
)
def grey_cr_flux_terms(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
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
        params: The simulation parameters (carries ``CosmicRayGreyParams`` at
            ``params.cosmic_ray_grey_params``, e.g. ``gamma_cr``/
            ``reduced_streaming_speed``).
        registered_variables: The registered variables.
        flux_direction_index: The index of the velocity component in the
            flux direction of interest.

    Returns:
        The flux contribution to add onto the ``e_cr``/``F_cr`` rows.

    Implementation note (Jiang & Oh 2018's reduced-speed-of-light two-moment
    closure, isotropic Eddington tensor ``P_cr = (gamma_cr - 1) * e_cr`` --
    the same factor used for the gas-momentum coupling in
    ``cr_grey_sources.cr_pressure_gradient_source``):

        d(e_cr)/dt + d(F_cr)/dx      = 0
        d(F_cr)/dt + d(v_red^2 P_cr)/dx = 0

    so the ``e_cr`` row's flux along ``flux_direction_index`` is simply the
    matching component of ``F_cr``, and that same component of the ``F_cr``
    row's flux is ``v_red^2 * P_cr``. The isotropic closure has no
    off-diagonal pressure-tensor terms, so the *other* ``F_cr`` components
    (e.g. ``F_cr,y`` when ``flux_direction_index`` is the x-axis) get zero
    flux here -- matching how ``_euler_flux`` only adds the pressure term to
    the flux-direction momentum component, not the others. Caller
    (``_euler_flux``) reads every ``cosmic_ray_flux_index`` component back
    out of the returned array regardless of axis, so this function must zero
    those out explicitly rather than leave them unset.
    """
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr
    reduced_streaming_speed = params.cosmic_ray_grey_params.reduced_streaming_speed

    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    p_cr = pressure_from_e_cr(e_cr, gamma_cr)

    if config.dimensionality == 1:
        f_cr_index_along_flux_direction = registered_variables.cosmic_ray_flux_index
    else:
        # cosmic_ray_flux_index is allocated the same way velocity_index is
        # (consecutive x/y/z rows), and flux_direction_index is itself the
        # velocity-row index for this axis (1/2/3) -- see e.g. _euler_flux's
        # ``primitive_state[flux_direction_index]``. So the matching F_cr row
        # is the same offset into cosmic_ray_flux_index.x.
        f_cr_index_along_flux_direction = (
            registered_variables.cosmic_ray_flux_index.x + (flux_direction_index - 1)
        )

    flux_vector = jnp.zeros_like(primitive_state)
    flux_vector = flux_vector.at[registered_variables.cosmic_ray_e_index].set(
        primitive_state[f_cr_index_along_flux_direction]
    )
    flux_vector = flux_vector.at[f_cr_index_along_flux_direction].set(
        reduced_streaming_speed**2 * p_cr
    )

    return flux_vector


@partial(jax.jit, static_argnames=["registered_variables"])
def grey_cr_fast_speed(
    primitive_state: STATE_TYPE,
    params: SimulationParams,
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
        params: The simulation parameters (carries
            ``params.cosmic_ray_grey_params.reduced_streaming_speed``).
        registered_variables: The registered variables.

    Returns:
        The CR-grey fast/reduced-streaming speed.

    Implementation note: the true characteristic speed of
    :func:`grey_cr_flux_terms`'s isotropic-closure system is
    ``reduced_streaming_speed * sqrt(gamma_cr - 1)`` (< ``reduced_streaming_speed``
    for ``gamma_cr = 4/3``), but every call site here uses this as a safety
    bound for the Riemann solver's wave-speed clamp / the CFL estimate, not
    as the exact eigenvalue -- so, matching standard reduced-speed-of-light
    two-moment practice, this returns the un-scaled ``reduced_streaming_speed``
    itself as a conservative (never-too-small) bound, independent of the
    local state.
    """
    return jnp.full(
        primitive_state[registered_variables.density_index].shape,
        params.cosmic_ray_grey_params.reduced_streaming_speed,
    )


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def anisotropic_flux_projection(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
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
        params: The simulation parameters (carries ``CosmicRayGreyParams``).
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
