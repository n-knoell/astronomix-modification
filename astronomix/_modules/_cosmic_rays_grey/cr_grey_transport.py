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
from astronomix._stencil_operations._stencil_operations import _stencil_add


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

        d(e_cr)/dt + d(u_n e_cr + F_cr)/dx        = 0
        d(F_cr)/dt + d(u_n F_cr + v_red^2 P_cr)/dx = 0

    Both rows now carry the same generic ``u_n * q`` bulk-advection piece
    that ``_euler_flux`` already forms for every *other* row (CRs are
    carried by the gas, on top of the relative/pressure-like transport) --
    see ``jnp.zeros_like`` -> ``u_n * primitive_state`` below -- plus an
    extra term added only to the flux-direction component: ``F_cr`` for the
    ``e_cr`` row, ``v_red^2 * P_cr`` for the matching ``F_cr`` row. The
    isotropic closure has no off-diagonal pressure-tensor terms, so the
    *other* ``F_cr`` components (e.g. ``F_cr,y`` when
    ``flux_direction_index`` is the x-axis) get only their advective piece,
    matching how ``_euler_flux`` only adds the pressure term to the
    flux-direction momentum component, not the others.

    Without the advective piece, a non-uniform ``e_cr`` profile sitting in a
    uniformly-flowing gas (``div(v) = 0``, so the adiabatic-work source term
    is silently zero) would never be swept downstream with the flow --
    breaking wind/SNe-driven CR transport, the Phase C science target. It
    also mismatches the plan's adiabatic-compression invariant (Sec. 4, test
    2): a homologous squeeze ``v = H(t) x`` gives
    ``d(e_cr)/dt = -(e_cr + P_cr) div(v)`` only once the ``u_n e_cr`` piece
    is present (the ``e_cr`` term supplies the "volume dilution" a
    conservative advective flux gives for free); without it the source term
    in ``cr_grey_sources.cr_adiabatic_work_source`` alone integrates to the
    wrong exponent, ``e_cr ~ rho^(gamma_cr - 1)`` instead of
    ``rho^gamma_cr``. See ``PROGRESS.md`` for the full derivation.
    """
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr
    reduced_streaming_speed = params.cosmic_ray_grey_params.reduced_streaming_speed

    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    p_cr = pressure_from_e_cr(e_cr, gamma_cr)
    u_n = primitive_state[flux_direction_index]

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

    # Generic u_n * q bulk-advection piece for every CR row (e_cr and every
    # F_cr component) -- only the rows the caller reads back
    # (cosmic_ray_e_index / cosmic_ray_flux_index.*) matter, so populating
    # the rest of the array is harmless.
    #
    # Anisotropic transport (plan Sec. 2) is NOT applied here: this function
    # runs deep inside the FV MHD Strang split's gas-only Riemann solve
    # (_evolve_state_fv -> _evolve_gas_state_*), where the magnetic-field
    # rows have already been split off into a separate array
    # (evolve_state._split_gas_and_magnetic_state) and are not part of
    # ``primitive_state`` here -- so this function structurally cannot read
    # B. Projecting F_cr onto B instead happens once per full step, before
    # the hydro update, in _iteration_level_updates (see that module and
    # anisotropic_flux_projection's docstring) -- by the time this function
    # runs, F_cr is already B-aligned if anisotropic_transport is on, so the
    # isotropic-looking flux below is correct either way.
    flux_vector = u_n * primitive_state
    flux_vector = flux_vector.at[registered_variables.cosmic_ray_e_index].add(
        primitive_state[f_cr_index_along_flux_direction]
    )
    flux_vector = flux_vector.at[f_cr_index_along_flux_direction].add(
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
    :func:`grey_cr_flux_terms`'s isotropic-closure system, *relative to the
    local advecting velocity* ``u_n`` (the CR subsystem now advects with the
    gas -- see that function's docstring), is
    ``reduced_streaming_speed * sqrt(gamma_cr - 1)`` (< ``reduced_streaming_speed``
    for ``gamma_cr = 4/3``), but every call site here uses this as a safety
    bound for the Riemann solver's wave-speed clamp / the CFL estimate, not
    as the exact eigenvalue -- so, matching standard reduced-speed-of-light
    two-moment practice, this returns the un-scaled ``reduced_streaming_speed``
    itself as a conservative (never-too-small) bound, independent of the
    local state. Every call site adds this to ``|u_n|`` itself (the same
    ``jnp.maximum(c, grey_cr_fast_speed(...))`` widening of the *sound-speed*
    term, before the surrounding ``|u| + c`` combination) rather than here,
    so this function must NOT add ``u_n`` itself -- doing so would double
    count it.

    Second contribution -- the CR-pressure momentum coupling (found while
    building the ladder-item-2 adiabatic-compression test): ``-grad(P_cr)``
    on gas momentum (``cr_grey_sources.cr_pressure_gradient_source``) and
    ``-P_cr div(v)`` back onto ``e_cr`` (``cr_adiabatic_work_source``) form a
    genuine coupled gas+CR acoustic mode, independent of ``F_cr``/
    ``reduced_streaming_speed`` entirely -- linearizing the coupled
    continuity/momentum/adiabatic equations around a uniform background
    gives ``omega^2 = k^2 (c_gas^2 + gamma_cr (gamma_cr - 1) e_cr / rho)``,
    i.e. a real, stable, *faster*-than-``c_gas`` sound speed, not an
    instability -- but nothing in the CFL/Riemann wave-speed bound accounted
    for it before now (only ``reduced_streaming_speed`` was returned here),
    so any state with ``e_cr`` large enough for this term to matter silently
    violated CFL and blew up to NaN. `ladder item 1 never exercised this
    (its CR background was zero, so the momentum coupling was a no-op the
    whole run). Added as a straightforward sum (not the tighter
    ``sqrt(a^2+b^2)``, to keep this a simple, easily-conservative widening
    of an already-"never-too-small" bound, matching this function's existing
    contract) on top of ``reduced_streaming_speed``.
    """
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr
    speed_floor = params.cosmic_ray_grey_params.cr_pressure_speed_floor
    rho = primitive_state[registered_variables.density_index]
    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    # Floor added in quadrature under the sqrt (not jnp.maximum on the
    # result) so the gradient stays finite at e_cr = 0 -- see
    # CosmicRayGreyParams.cr_pressure_speed_floor's docstring.
    cr_pressure_coupling_speed = jnp.sqrt(
        jnp.maximum(gamma_cr * (gamma_cr - 1.0) * e_cr / rho, 0.0) + speed_floor**2
    )
    return (
        params.cosmic_ray_grey_params.reduced_streaming_speed
        + cr_pressure_coupling_speed
    )


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def anisotropic_flux_projection(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Project the CR flux onto the local magnetic-field direction.

    Returns ``F_cr_parallel = (F_cr . b_hat) * b_hat``, ``b_hat = B / |B|``,
    i.e. the component of ``F_cr`` along the local field, with the
    perpendicular component discarded. Only meaningful when
    ``config.mhd`` and ``config.cosmic_ray_grey_config.anisotropic_transport``
    are both set (requires an actual field to project onto); the isotropic
    closure (raw, unprojected ``F_cr``) is the default.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        config: The simulation configuration.
        params: The simulation parameters (carries ``CosmicRayGreyParams``,
            in particular ``b_field_floor``).
        registered_variables: The registered variables.

    Returns:
        A ``STATE_TYPE``-shaped array with the ``cosmic_ray_flux_index``
        row(s) set to the B-projected flux; other rows are zero and unused
        by the caller.

    Where this is called from, and why (not from :func:`grey_cr_flux_terms`):
    the FV MHD Strang split (``evolve_state._evolve_state_fv``) splits
    ``primitive_state`` into a gas-only sub-array and a separate magnetic
    sub-array for each half-step (``evolve_state._split_gas_and_magnetic_state``)
    -- the magnetic-field rows are structurally unavailable inside the
    gas-only Riemann solve where :func:`grey_cr_flux_terms` runs, so this
    function cannot be called from there. Instead it's applied once per full
    step, as a primitive-state correction in
    ``astronomix._modules._iteration_level_updates`` (alongside the ``e_cr``
    positivity floor), overwriting ``F_cr`` with its B-projected value
    *before* the hydro update runs -- so by the time
    :func:`grey_cr_flux_terms` reads ``F_cr`` mid-step, it is already
    B-aligned if ``anisotropic_transport`` is on. Consequence: within a
    single step, the isotropic per-axis ``v_red^2 * P_cr`` driving term
    (unchanged, not projected -- see :func:`grey_cr_flux_terms`) can still
    push a small (order ``dt``) perpendicular component into ``F_cr`` before
    the *next* step's correction removes it again -- a per-step residual,
    not a steady-state leak. Scope note (plan Sec. 2: "the flux limiter is
    largely avoided [in two-moment form]... keep the S&H-style guard only
    where needed"): this is a direct projection, not the full Sharma &
    Hammett (2007) monotonicity-preserving flux-limiting scheme (which
    exists for *diffusive/parabolic* oblique discretizations; this system is
    hyperbolic/Riemann-solved, and the plan itself expects less limiting
    machinery to be needed here). See PROGRESS.md for the numerical
    verification of how small the residual leak actually is.
    """
    b_index = registered_variables.magnetic_index
    f_cr_index = registered_variables.cosmic_ray_flux_index

    b_x = primitive_state[b_index.x]
    b_y = primitive_state[b_index.y]
    f_x = primitive_state[f_cr_index.x]
    f_y = primitive_state[f_cr_index.y]
    if config.dimensionality == 3:
        b_z = primitive_state[b_index.z]
        f_z = primitive_state[f_cr_index.z]
    else:
        b_z = jnp.zeros_like(b_x)
        f_z = jnp.zeros_like(f_x)

    # Smooth (always-positive, differentiable) floor on |B| -- same "prefer
    # smooth regularization over min/max" philosophy as
    # regularized_streaming_sign (plan Sec. 2) -- instead of jnp.maximum,
    # which would have a non-smooth gradient at the floor.
    b_field_floor = params.cosmic_ray_grey_params.b_field_floor
    b_mag = jnp.sqrt(b_x**2 + b_y**2 + b_z**2 + b_field_floor**2)
    b_hat_x = b_x / b_mag
    b_hat_y = b_y / b_mag
    b_hat_z = b_z / b_mag

    f_dot_b = f_x * b_hat_x + f_y * b_hat_y + f_z * b_hat_z

    flux_vector = jnp.zeros_like(primitive_state)
    flux_vector = flux_vector.at[f_cr_index.x].set(f_dot_b * b_hat_x)
    flux_vector = flux_vector.at[f_cr_index.y].set(f_dot_b * b_hat_y)
    if config.dimensionality == 3:
        flux_vector = flux_vector.at[f_cr_index.z].set(f_dot_b * b_hat_z)

    return flux_vector


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def streaming_flux_target(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Per-axis CR streaming-flux target (plan Sec. 2, ladder item 5).

    Physical picture (Wiener et al. 2017; the "streaming-dominated,
    always-at-equilibrium" limit): self-confined CRs stream down their own
    pressure gradient at the reduced free-streaming speed, so in this limit
    ``F_cr`` isn't an independently evolving quantity any more -- it's
    pinned to ``F_cr,axis = -sign(dP_cr/dx_axis) * reduced_streaming_speed *
    e_cr`` (regularized via :func:`regularized_streaming_sign` instead of a
    hard ``sign``, for the same adjoint reason as everywhere else in this
    module). Applied once per full step in
    ``astronomix._modules._iteration_level_updates`` as a discrete
    correction that **overwrites** ``F_cr`` -- the same "instantaneous
    relaxation" pattern :func:`anisotropic_flux_projection` already uses for
    item 3 (see that function's docstring), not a new stiff relaxation-rate
    source term.

    Isotropic per-axis, not projected along a true magnetic-field direction
    -- this does not require ``config.mhd`` (matches the plan's own "1D
    streaming" staging for this ladder item). If both ``streaming`` and
    ``anisotropic_transport`` are enabled, ``_iteration_level_updates``
    applies this correction first and the B-projection second, so the
    combination is "isotropic streaming target, then projected onto B" --
    a reasonable but **not separately verified** approximation (no ladder
    item tests the combination); flagged here rather than assumed correct.

    The complementary energy loss this transport implies (streaming does
    work against the pressure gradient, converting some ``e_cr`` into gas
    heat) is computed separately by
    :func:`astronomix._modules._cosmic_rays_grey.cr_grey_sources.cr_streaming_heating_source`
    from the same gradient/sign -- this function only returns the
    conservative flux target, no energy is created or destroyed by it alone.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        config: The simulation configuration.
        params: The simulation parameters (carries
            ``params.cosmic_ray_grey_params.reduced_streaming_speed`` and
            the streaming-sign regularization scale).
        registered_variables: The registered variables.

    Returns:
        A ``STATE_TYPE``-shaped array with the ``cosmic_ray_flux_index``
        row(s) set to the streaming-flux target; other rows are zero and
        unused by the caller.
    """
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr
    reduced_streaming_speed = params.cosmic_ray_grey_params.reduced_streaming_speed
    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    p_cr = pressure_from_e_cr(e_cr, gamma_cr)

    f_cr_index = registered_variables.cosmic_ray_flux_index
    flux_vector = jnp.zeros_like(primitive_state)
    for axis in range(1, config.dimensionality + 1):
        grad_p_cr_axis = _stencil_add(
            p_cr, indices=(1, -1), factors=(1.0, -1.0), axis=axis - 1
        ) / (2 * config.grid_spacing)
        sign = regularized_streaming_sign(
            grad_p_cr_axis, params.cosmic_ray_grey_params
        )
        axis_index = (
            f_cr_index
            if config.dimensionality == 1
            else (f_cr_index.x, f_cr_index.y, f_cr_index.z)[axis - 1]
        )
        flux_vector = flux_vector.at[axis_index].set(
            -sign * reduced_streaming_speed * e_cr
        )

    return flux_vector


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
