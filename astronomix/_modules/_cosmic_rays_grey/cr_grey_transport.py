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
    # B. The anisotropy instead lives in the implicit F_cr update applied
    # after every RK stage (cr_grey_sources.cr_flux_relaxation_update, with
    # B passed in separately): each stage starts from a B-aligned F_cr (up to
    # kappa_perp) if anisotropic_transport is on, so the isotropic-looking
    # flux below is correct either way.
    flux_vector = u_n * primitive_state
    flux_vector = flux_vector.at[registered_variables.cosmic_ray_e_index].add(
        primitive_state[f_cr_index_along_flux_direction]
    )
    flux_vector = flux_vector.at[f_cr_index_along_flux_direction].add(
        reduced_streaming_speed**2 * p_cr
    )

    return flux_vector


def cr_flux_rows(registered_variables: RegisteredVariables) -> tuple:
    """The allocated ``F_cr`` state rows, in x/y/z order.

    One row in 1D, x/y in 2D hydro, x/y/z in 2D MHD and 3D (2D MHD carries
    ``F_z`` for field-aligned transport with ``B_z != 0``, see
    ``registered_variables.py``). Every loop over the ``F_cr`` rows should use
    this rather than slicing by ``config.dimensionality``.

    Args:
        registered_variables: The registered variables.

    Returns:
        The row indices.
    """
    f_cr_index = registered_variables.cosmic_ray_flux_index
    rows = (f_cr_index,) if isinstance(f_cr_index, int) else tuple(f_cr_index)
    return tuple(row for row in rows if row >= 0)


@partial(jax.jit, static_argnames=["registered_variables"])
def cr_pressure_coupling_speed(
    primitive_state: STATE_TYPE,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> Float[Array, "..."]:
    """CR-pressure contribution to the *gas* signal speed,
    ``sqrt(gamma_cr (gamma_cr - 1) e_cr / rho)``.

    Combined with the gas sound speed in quadrature,
    ``sqrt(c_gas^2 + c_cr^2)``, for the gas rows of the Riemann solver and for
    the CFL estimate.

    Args:
        primitive_state: The primitive state of the fluid on all cells.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The CR-pressure coupling speed.

    Why (found while building the ladder-item-2 adiabatic-compression test):
    ``-grad(P_cr)`` on gas momentum
    (``cr_grey_sources.cr_pressure_gradient_source``) and ``-P_cr div(v)``
    back onto ``e_cr`` (``cr_adiabatic_work_source``) form a genuine coupled
    gas+CR acoustic mode, independent of ``F_cr``/``reduced_streaming_speed``
    -- linearizing the coupled continuity/momentum/adiabatic equations around
    a uniform background gives ``omega^2 = k^2 (c_gas^2 + gamma_cr (gamma_cr -
    1) e_cr / rho)``, a real, stable, *faster*-than-``c_gas`` sound speed.
    Without it in the wave-speed bounds, any state with ``e_cr`` large enough
    for this term to matter violated CFL and blew up to NaN. Until
    2026-10-04 it was added *linearly* on top of ``reduced_streaming_speed``
    and the sum shared by all rows (``grey_cr_fast_speed``); see
    :func:`cr_closure_signal_speed` for the split.

    The floor (``CosmicRayGreyParams.cr_pressure_speed_floor``) is added in
    quadrature under the sqrt, not as ``jnp.maximum`` on the result, so the
    gradient stays finite at ``e_cr = 0`` -- the CR-free-background case
    (confirmed to NaN reverse-mode AD otherwise, see cr_gradient_check.py).
    """
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr
    speed_floor = params.cosmic_ray_grey_params.cr_pressure_speed_floor
    rho = primitive_state[registered_variables.density_index]
    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    return jnp.sqrt(
        jnp.maximum(gamma_cr * (gamma_cr - 1.0) * e_cr / rho, 0.0) + speed_floor**2
    )


def cr_closure_signal_speed(params: SimulationParams) -> Union[float, Float[Array, ""]]:
    """Signal speed of the two-moment ``(e_cr, F_cr)`` subsystem relative to
    the gas, ``reduced_streaming_speed * sqrt(gamma_cr - 1)`` (``v_red /
    sqrt(3)`` for ``gamma_cr = 4/3``).

    The eigenvalues of :func:`grey_cr_flux_terms`'s isotropic-closure system
    are ``u_n +- v_red sqrt(gamma_cr - 1)``. The Riemann solver gives the CR
    rows their own HLL flux with this speed (scaled by
    :func:`cr_wave_speed_reduction` in the diffusive regime), and the CFL
    estimate uses ``max(gas speed, this)``. Until 2026-10-04 the un-scaled
    ``v_red`` (plus the CR-pressure coupling speed) was a single bound shared
    by gas and CR rows: ``sqrt(3)`` too much dissipation on the CR rows, and
    ``v_red``-inflated dissipation on the gas rows that smeared shocks
    (DESIGN.md "Finding 1"; "Open: CR diffusion correctness", fix step 4).

    Args:
        params: The simulation parameters.

    Returns:
        The CR closure signal speed.
    """
    cr_params = params.cosmic_ray_grey_params
    return cr_params.reduced_streaming_speed * jnp.sqrt(cr_params.gamma_cr - 1.0)


def cr_wave_speed_reduction(
    kappa_normal: Float[Array, "..."],
    params: SimulationParams,
    grid_spacing: Union[float, Float[Array, ""]],
) -> Float[Array, "..."]:
    """Optical-depth reduction ``R`` of the CR rows' HLL wave speed (Jiang &
    Oh 2018, Sec. 3.2.1, after Jiang et al. 2013).

    ``R = sqrt((1 - exp(-tau^2)) / tau^2)``, ``tau = (gamma_cr - 1) v_red dx /
    kappa_normal``: the cell size in units of the CR mean free path. In the
    diffusive regime (``tau >> 1``) the HLL dissipation at the full signal
    speed would be a numerical diffusivity ``~ v_red dx`` far above
    ``kappa``; ``R ~ 1 / tau`` scales it down to ``~ kappa``. For ``tau
    << 1`` (free streaming) ``R -> 1``.

    ``kappa_normal`` is the ``e_cr`` diffusivity along the face normal ``n``:
    ``kappa`` isotropically, ``n . K . n = kappa_par b_n^2 + kappa_perp (1 -
    b_n^2)`` with anisotropic transport. JO18 define ``tau`` for a scalar
    ``sigma_c``; the face-normal projection is this module's choice
    (validated by ``cr_diffusion_rate.py`` T2c).

    Written as ``R^2 = x^2 (1 - exp(-1/x^2))`` with ``x = 1/tau`` plus a tiny
    floor, so ``kappa_normal = 0`` (B along another axis, ``kappa_perp = 0``)
    gives ``R ~ 0`` with finite values and gradients.

    Args:
        kappa_normal: The face-normal ``e_cr`` diffusivity, per cell.
        params: The simulation parameters.
        grid_spacing: The cell size.

    Returns:
        ``R``, same shape as ``kappa_normal``.
    """
    cr_params = params.cosmic_ray_grey_params
    kappa_dx = (cr_params.gamma_cr - 1.0) * cr_params.reduced_streaming_speed * grid_spacing
    x2 = (kappa_normal / kappa_dx) ** 2 + 1e-30
    return jnp.sqrt(x2 * -jnp.expm1(-1.0 / x2))


def cr_flux_realizability_cap(
    primitive_state: STATE_TYPE,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """Two-moment realizability cap ``|F_cr| <= v_red e_cr`` (Rosdahl et al.
    2025, Sec. 2.3).

    The CR energy cannot flow faster than the (reduced) speed of light, so a
    realizable state has ``|F_cr| <= v_red e_cr``. The isotropic (P1) closure
    does not enforce this by itself. Rosdahl et al. check it "each time an
    interpolation is needed ... or after the source step" and, where it is
    violated, "reduce the magnitude of the flux vector so that
    ``F_c = c_tilde e_c``", as M1 radiation schemes do. Here
    ``F_cr <- F_cr min(1, v_red max(e_cr, 0) / |F_cr|)`` on all ``F_cr`` rows.
    It is local and adds no dissipation.

    Found as the coincidence of undershoots and ``|F_cr| > v_red e_cr`` in
    the ring tests without the monotonicity guard (DESIGN.md "Open: CR
    diffusion follow-up", step 4). ``min`` makes it piecewise smooth (a kink
    at the cap only); the ``1e-30`` keeps ``|F| = 0`` finite.

    Args:
        primitive_state: A (cell or interface) primitive state with the CR rows.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The state with ``F_cr`` capped.
    """
    rows = cr_flux_rows(registered_variables)
    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    flux_magnitude = jnp.sqrt(sum(primitive_state[row] ** 2 for row in rows) + 1e-30)
    cap = params.cosmic_ray_grey_params.reduced_streaming_speed * jnp.maximum(e_cr, 0.0)
    scale = jnp.minimum(1.0, cap / flux_magnitude)
    for row in rows:
        primitive_state = primitive_state.at[row].multiply(scale)
    return primitive_state


def cr_monotonicity_guard(
    e_cr: Float[Array, "..."],
    params: SimulationParams,
) -> Float[Array, "..."]:
    """Per-cell weight ``g`` in [0, 1] of the CR rows' monotonicity guard:
    the optical-depth reduction ``R`` of the Rusanov dissipation is raised to
    ``R_eff = R + (1 - R) g`` on the faces of cells near a sharp ``e_cr``
    front (DESIGN.md "Open: CR diffusion follow-up", step 3).

    A flux-limiter blend of the reduced-dissipation flux (``R``) and the full
    Rusanov flux (``R = 1``) driven by a shape sensor, not by the size of the
    jump. Per face, ``psi = |d_c| / (|d_l| + |d_c| + |d_r|)``, with ``d_c`` the
    ``e_cr`` jump across the face and ``d_l``, ``d_r`` the jumps across its two
    neighbouring faces along the same axis. Values: <= 1/3 on any linear or
    exponential profile, ~0.2 at a smooth extremum, <= 0.38 for a Gaussian
    with 2.6 cells per sigma, 1 for a one-cell step. ``psi`` relates to the
    ratio of consecutive gradients of TVD flux limiters (Sweby 1984) and the
    self-adjusting hybrid schemes of Harten & Zwas (1972); Sharma & Hammett
    (2007) limit anisotropic fluxes with the same kind of comparison.
    The face weight is a C1 smoothstep of ``psi`` from
    ``cr_guard_sensor_onset`` to ``cr_guard_sensor_full``. Each cell takes
    the largest weight of its faces on all axes, and that is spread over the
    3^d neighbourhood, as Jameson, Schmidt & Turkel (1981) spread their shock
    sensor over neighbouring cells. Spreading matters: the undershoots form
    in cells next to the front's corners, via the transverse direction, on
    faces that look smooth along their own axis; a face-local guard reduced
    them only 10-15x.

    Why: the face-normal ``R`` makes the CR-row flux central across faces
    nearly perpendicular to B (S&H's unlimited centered differencing),
    giving undershoots of 2-3% of the jump and negative ``e_cr`` at
    cross-field fronts in ``cr_anisotropic_ring.py``. A sensor scaled by
    ``e_cr`` itself was tried first and fired on every Gaussian tail running
    into a CR-free background.

    The absolute values are smoothed, ``sqrt(d^2 + eps^2)``, so the sensor is
    differentiable and gives ``psi = 1/3`` (off) in exactly flat or CR-free
    regions instead of 0/0.

    Args:
        e_cr: The cell-centered ``e_cr`` (spatial axes only).
        params: The simulation parameters.

    Returns:
        ``g`` per cell; a face uses the larger value of its two cells.
    """
    cr_params = params.cosmic_ray_grey_params
    eps = 1e-15

    def smooth_abs(d):
        return jnp.sqrt(d**2 + eps**2)

    cell_guard = jnp.zeros_like(e_cr)
    for axis in range(e_cr.ndim):
        jump_center = e_cr - jnp.roll(e_cr, 1, axis=axis)  # face i: e_i - e_{i-1}
        jump_left = jnp.roll(jump_center, 1, axis=axis)
        jump_right = jnp.roll(jump_center, -1, axis=axis)
        psi = smooth_abs(jump_center) / (
            smooth_abs(jump_left) + smooth_abs(jump_center) + smooth_abs(jump_right)
        )
        ramp = jnp.clip(
            (psi - cr_params.cr_guard_sensor_onset)
            / (cr_params.cr_guard_sensor_full - cr_params.cr_guard_sensor_onset),
            0.0,
            1.0,
        )
        face_guard = ramp**2 * (3.0 - 2.0 * ramp)
        # Both cells adjacent to face i (cells i - 1 and i) see it.
        cell_guard = jnp.maximum(
            cell_guard,
            jnp.maximum(face_guard, jnp.roll(face_guard, -1, axis=axis)),
        )
    # Spread over the 3^d neighbourhood.
    for axis in range(e_cr.ndim):
        cell_guard = jnp.maximum(
            cell_guard,
            jnp.maximum(jnp.roll(cell_guard, 1, axis=axis), jnp.roll(cell_guard, -1, axis=axis)),
        )
    return cell_guard


def magnetic_unit_vector(
    magnetic_field: Float[Array, "3 ..."],
    params: SimulationParams,
) -> Float[Array, "3 ..."]:
    """Smoothly floored unit vector ``b_hat = B / sqrt(|B|^2 + b_floor^2)``.

    Used by the anisotropic ``F_cr`` update
    (``cr_grey_sources.cr_flux_relaxation_update``). The floor
    (``CosmicRayGreyParams.b_field_floor``) keeps ``b_hat`` well-defined and
    differentiable at ``B = 0`` (where it goes to zero, so the update treats
    the cell as purely perpendicular), instead of a hard
    ``jnp.maximum``/``where``.

    Args:
        magnetic_field: The cell-centered ``(B_x, B_y, B_z)`` (all three
            components, also in 2D).
        params: The simulation parameters.

    Returns:
        ``b_hat``, same shape as ``magnetic_field``.
    """
    b_field_floor = params.cosmic_ray_grey_params.b_field_floor
    b_mag = jnp.sqrt(jnp.sum(magnetic_field**2, axis=0) + b_field_floor**2)
    return magnetic_field / b_mag


def cr_wave_speed_factors(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    magnetic_field: Union[Float[Array, "3 ..."], None] = None,
) -> Union[Float[Array, "..."], None]:
    """Per-cell, per-axis :func:`cr_wave_speed_reduction` for the Riemann
    solver, shape ``(dimensionality, *grid)``; None (no reduction) without
    ``diffusive_relaxation``.

    The face-normal diffusivity is ``diffusion_coefficient`` isotropically,
    and ``kappa_par b_n^2 + kappa_perp (1 - b_n^2)`` with
    ``anisotropic_transport`` (``b`` from ``magnetic_field``, the split-off
    B of the FV MHD gas half-step, constant within it).

    Args:
        primitive_state: The (gas) primitive state, for the grid shape.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        magnetic_field: The cell-centered ``(B_x, B_y, B_z)``; required with
            ``anisotropic_transport``.

    Returns:
        The factors, or None.
    """
    cr_config = config.cosmic_ray_grey_config
    if not (registered_variables.cosmic_ray_e_active and cr_config.diffusive_relaxation):
        return None
    cr_params = params.cosmic_ray_grey_params
    grid_shape = primitive_state.shape[1:]
    if cr_config.anisotropic_transport:
        b_hat = magnetic_unit_vector(magnetic_field, params)
        kappa_normal = jnp.stack([
            cr_params.diffusion_coefficient * b_hat[i] ** 2
            + cr_params.perpendicular_diffusion_coefficient * (1.0 - b_hat[i] ** 2)
            for i in range(config.dimensionality)
        ])
    else:
        kappa_normal = jnp.full(
            (config.dimensionality,) + grid_shape, cr_params.diffusion_coefficient
        )
    return cr_wave_speed_reduction(kappa_normal, params, config.grid_spacing)


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
    ``astronomix._modules._iteration_level_continuous_updates`` as a discrete
    correction that **overwrites** ``F_cr`` -- an "instantaneous
    relaxation", not a stiff relaxation-rate source term.

    Isotropic per-axis, not projected along a true magnetic-field direction
    -- this does not require ``config.mhd`` (matches the plan's own "1D
    streaming" staging for this ladder item). If both ``streaming`` and
    ``anisotropic_transport`` are enabled, this correction runs first and the
    per-RK-stage F_cr update then projects it onto B
    (``cr_grey_sources.cr_flux_relaxation_update``), so the combination is
    "isotropic streaming target, then projected onto B" --
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
