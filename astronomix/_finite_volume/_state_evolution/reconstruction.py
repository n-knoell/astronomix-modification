"""
Interface reconstruction for the finite-volume scheme.

Provides the MUSCL split reconstruction (with the predictor half-step) and the
unsplit reconstructions (including the positivity-preserving van Albada-PP
variant) that extrapolate the cell-centred primitive state to the left/right
states at each interface.
"""

# general
from functools import partial

# typing
from typing import Union
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    MINMOD,
    MUSCL,
    STATE_TYPE,
    STATE_TYPE_ALTERED,
    VAN_ALBADA,
    VAN_ALBADA_PP,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._modules._cosmic_rays_grey.cr_grey_transport import grey_cr_fast_speed
from astronomix._modules._gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._finite_volume._state_evolution.limiters import _van_albada_limiter, _minmod
from astronomix._stencil_operations._stencil_operations import _stencil_add
from astronomix._fluid_equations._equations import speed_of_sound
from astronomix._finite_volume._state_evolution.limited_gradients import _calculate_limited_gradients


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables", "axis"])
def _reconstruct_at_interface_split(
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    axis: int,
) -> tuple[STATE_TYPE_ALTERED, STATE_TYPE_ALTERED]:

    # get fluid variables for convenience
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    u = primitive_state[axis]

    # get the limited gradients on the cells
    limited_gradients = _calculate_limited_gradients(
        primitive_state, config, helper_data, axis=axis
    )

    if config.time_integrator == MUSCL:
        # calculate the sound speed
        c = speed_of_sound(rho, p, gamma)

        # Grey two-moment cosmic rays: combine with the CR-grey fast speed as
        # max(., .), not folded in -- see hll.py's identical pattern.
        if registered_variables.cosmic_ray_e_active:
            c = jnp.maximum(c, grey_cr_fast_speed(primitive_state, params, registered_variables))

        # ================ construct A_W, the "primitive Jacabian" (not an actual Jacabian) ================
        # see https://diglib.uibk.ac.at/download/pdf/4422963.pdf, 2.11

        # calculate the vectors making up A_W
        A_W = jnp.zeros((registered_variables.num_vars,) + primitive_state.shape)

        # set u diagonal, this way all quantities are automatically advected
        A_W = A_W.at[
            jnp.arange(registered_variables.num_vars),
            jnp.arange(registered_variables.num_vars),
        ].set(u)

        # set rest
        #
        # Positivity/robustness: the (axis, pressure) entry is 1/rho -- a
        # cell with legitimately near-vanishing density (a near-vacuum
        # ambient medium, or a supersonic wind's leading edge reaching an
        # open boundary) makes this term huge, which the einsum below then
        # multiplies straight into a projected *velocity* gradient. Flooring
        # rho here (not clamping the output primitives, which is too late --
        # the Jacobian projection already happened) fixes that specific
        # mechanism at its root; cheap and unconditional, complementary to
        # the VAN_ALBADA_PP positivity scaling applied further below (that
        # one covers the *output* of this whole predictor+spatial
        # reconstruction; this one covers the A_W matrix itself, which the
        # output-side scaling has no visibility into).
        rho_floor_for_jacobian = jnp.maximum(rho, params.minimum_density)
        A_W = A_W.at[registered_variables.density_index, axis].set(rho)
        A_W = A_W.at[registered_variables.pressure_index, 1].set(rho * c**2)
        A_W = A_W.at[axis, registered_variables.pressure_index].set(
            1 / rho_floor_for_jacobian
        )

        # ====================================================================================================

        # project the gradients
        if config.dimensionality == 1:
            projected_gradients = jnp.einsum("bax, ax -> bx", A_W, limited_gradients)
        elif config.dimensionality == 2:
            projected_gradients = jnp.einsum("baxy, axy -> bxy", A_W, limited_gradients)
        elif config.dimensionality == 3:
            projected_gradients = jnp.einsum(
                "baxyz, axyz -> bxyz", A_W, limited_gradients
            )

        # predictor step
        predictors = primitive_state - dt / 2 * projected_gradients

        # Sanitize: a literal inf/NaN here (an overflow inside the A_W/
        # predictor-step einsum) would otherwise sail straight through
        # everything below, including the VAN_ALBADA_PP positivity scaling
        # further down -- that scaling can only *shrink* an already-finite
        # delta towards primitive_state, it cannot repair a delta that's
        # already NaN (jnp.where(cond, nan, x) still propagates nan through
        # the comparison-selected branch once nan has entered any of these
        # expressions). Falls back to the plain cell-centred value, exactly
        # matching what config.first_order_fallback=True would have used
        # for that cell.
        predictors = jnp.where(jnp.isfinite(predictors), predictors, primitive_state)
    else:
        raise ValueError(
            f"Time integrator {config.time_integrator} not supported for split reconstruction. Only MUSCL is supported."
        )

    # compute primitives at the interfaces
    if config.geometry == CARTESIAN:
        distances_to_left_interfaces = (
            config.grid_spacing / 2
        )  # distances r_i - r_{i-1/2}
        distances_to_right_interfaces = (
            config.grid_spacing / 2
        )  # distances r_{i+1/2} - r_i
    else:
        r = helper_data.geometric_centers
        rv = helper_data.volumetric_centers

        distances_to_left_interfaces = rv - (r - config.grid_spacing / 2)
        distances_to_right_interfaces = (r + config.grid_spacing / 2) - rv

    primitives_left = predictors - distances_to_left_interfaces * limited_gradients
    primitives_right = predictors + distances_to_right_interfaces * limited_gradients

    # ============ VAN_ALBADA_PP / MINMOD: positivity-preserving output scaling ============
    #
    # Ported from _reconstruct_at_interface_unsplit's own multi-dimensional
    # positivity machinery (same alpha_density/kappa_pressure/beta formula,
    # same derivation) -- see that function's comments for the original
    # context. That version has no MUSCL predictor step: it scales the pure
    # spatial difference (limited_gradients * grid_spacing / 2) directly,
    # symmetrically, before extrapolating to the interface.
    #
    # First attempt at this port (2026-09-19, reverted) applied the same
    # alpha/kappa/beta scaling to limited_gradients itself, *before* the A_W
    # einsum above -- reasoning that A_W's 1/rho coupling (the mechanism
    # found, see FIXES_TODO.md item 4, to turn an ordinary pressure gradient
    # into a wildly unphysical projected *velocity* gradient) would then
    # only ever see an already-safe input. Empirically this made things
    # *worse* than doing nothing (failed at step 0, immediately, at the same
    # injection-region cell the very first VAN_ALBADA_PP investigation
    # found). Root cause: A_W *mixes* variables -- a pressure gradient
    # scaled down by kappa_pressure (chosen using only pressure's own
    # positivity requirement) still flows through A_W's 1/rho entry into
    # the *velocity* row of projected_gradients, entirely unconstrained by
    # beta (the velocity-specific, generally much stricter bound). Scaling
    # the input can't fix a blow-up that only exists in the matrix's output.
    #
    # This version instead scales the *output* -- the actual total
    # perturbation (predictor step + spatial extrapolation combined)
    # applied to get from primitive_state to primitives_left/right -- since
    # that already has A_W's mixing baked in, whatever it is. This departs
    # from the unsplit algorithm's assumption of one *symmetric* difference
    # per axis (primitive_state -/+ the same difference): the MUSCL
    # predictor step generally makes primitives_left/right's own deltas
    # from primitive_state asymmetric. Nothing in the underlying
    # alpha_density/kappa_pressure/beta derivation actually requires
    # symmetry (each factor only bounds "how far can primitive_state move
    # in this specific direction and stay positive"), so left and right are
    # scaled independently, each using its own delta as the formula's
    # "differences" input.
    #
    # Extended to config.limiter == MINMOD too (2026-09-19): the formula
    # below is mathematically limiter-agnostic -- it operates on the actual
    # reconstructed delta (primitives_left/right - primitive_state), not on
    # the raw limited_gradients a specific limiter produced, so nothing
    # here depends on VAN_ALBADA_PP specifically. MINMOD's own gradients
    # are already far more conservative than VAN_ALBADA's (confirmed:
    # alpha_density/kappa_pressure/beta come out ~= 1, i.e. this is a
    # near-total no-op, everywhere in the CWB injection region under
    # MINMOD), so this is meant to only actually engage where MINMOD's own
    # reconstruction is genuinely at risk -- see FIXES_TODO.md item 4 for
    # whether it fixes MINMOD's own remaining (boundary-corner) failure.
    if config.limiter == VAN_ALBADA_PP or config.limiter == MINMOD:
        eps = 1e-14

        alpha_lax = jnp.zeros((config.dimensionality,))
        for axis_i in range(1, config.dimensionality + 1):
            u_i = primitive_state[axis_i]
            alpha_lax = alpha_lax.at[axis_i - 1].set(jnp.max(jnp.abs(u_i) + c))
        C_axis = alpha_lax[axis - 1] / jnp.sum(alpha_lax)

        # q = 1/C_cfl, exactly the unsplit convention (no MHD dt/2 case here
        # -- this split path isn't used for MHD). Reused unchanged across
        # every Strang sub-step of this step (x/2, y/2, z, y/2, x/2): each
        # sub-step's own sub_dt is <= the full dt that C_cfl was calibrated
        # against, so this is if anything more conservative (more
        # positivity margin) than strictly necessary for the halved
        # sub-steps, never less.
        q = 1 / params.C_cfl

        density_index = registered_variables.density_index
        pressure_index = registered_variables.pressure_index
        vx0 = registered_variables.velocity_index.x

        def _positivity_scale(primitives_side):
            delta = primitives_side - primitive_state

            density_diff_protected = jnp.where(
                jnp.abs(delta[density_index]) > eps, delta[density_index], eps
            )
            pressure_diff_protected = jnp.where(
                jnp.abs(delta[pressure_index]) > eps, delta[pressure_index], eps
            )
            alpha_density = jnp.where(
                jnp.abs(delta[density_index]) > eps,
                jnp.minimum(rho / (jnp.abs(density_diff_protected) * (1 + eps)), 1),
                1,
            )
            kappa_pressure = jnp.where(
                jnp.abs(delta[pressure_index]) > eps,
                jnp.minimum(p / (jnp.abs(pressure_diff_protected) * (1 + eps)), 1),
                1,
            )

            delta_v = delta[vx0 : vx0 + config.dimensionality]
            vsum = jnp.sum(delta_v**2, axis=0)
            A1 = (C_axis * alpha_density * delta[density_index] * jnp.sum(delta_v, axis=0)) ** 2
            A2 = C_axis * vsum
            A1 = jnp.where(vsum > eps, A1, eps)
            A2 = jnp.where(vsum > eps, A2, eps)

            beta = jnp.where(
                vsum > eps,
                jnp.minimum(
                    jnp.sqrt(
                        ((q - 2) ** 2 * rho * p)
                        / ((gamma - 1) * (2 * A1 + (q - 2) * rho**2 * A2))
                    ),
                    1,
                ),
                1,
            )

            scaled_delta = delta
            scaled_delta = scaled_delta.at[density_index].set(
                delta[density_index] * alpha_density
            )
            scaled_delta = scaled_delta.at[pressure_index].set(
                delta[pressure_index] * kappa_pressure
            )
            scaled_delta = scaled_delta.at[vx0 : vx0 + config.dimensionality].set(
                delta_v * beta
            )
            result = primitive_state + scaled_delta

            # alpha_density/kappa_pressure only guarantee *non*-negativity
            # (their limiting case, alpha = rho / |diff|, drives the scaled
            # delta to exactly cancel rho/p in the worst case -- density or
            # pressure landing at exactly 0.0, not some small positive
            # value). Confirmed directly (2026-09-19, scratch
            # debug_port_nan.py): an interface reconstructed with rho=0,
            # p=0 exactly (but a nonzero velocity survives, since beta's
            # own derivation doesn't force velocity to zero alongside a
            # vanishing density) is a valid, physical vacuum state in
            # principle, but speed_of_sound(rho=0, p=0, gamma) computes a
            # literal 0/0 -- sqrt(gamma*p/rho) -- which is NaN, not 0, and
            # that NaN then poisons the whole HLLC flux at that interface.
            # A small strictly-positive floor (not the scaling's own job;
            # this is purely about keeping downstream sqrt/division
            # well-defined) closes that gap.
            result = result.at[density_index].set(
                jnp.maximum(result[density_index], params.minimum_density)
            )
            result = result.at[pressure_index].set(
                jnp.maximum(result[pressure_index], params.minimum_pressure)
            )
            return result

        primitives_left = _positivity_scale(primitives_left)
        primitives_right = _positivity_scale(primitives_right)
    # ============ end VAN_ALBADA_PP scaling ============

    # primitives left at i is the left state at the interface
    # between i-1 and i so the right extrapolation from the cell i-1
    p_left_interface = jnp.roll(primitives_right, shift=1, axis=axis)

    # primitives right at i is the right state at the interface
    # between i-1 and i so the left extrapolation from the cell i
    p_right_interface = primitives_left

    return p_left_interface, p_right_interface


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _reconstruct_at_interface_unsplit(
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
):
    """
    Unsplit reconstruction (all axes at once).

    Computes the limited gradients along every axis simultaneously, which is
    needed for the multidimensional positivity-preserving van Albada-PP
    limiting. NOTE: this materialises a full ``(dimensionality, *state_shape)``
    gradient buffer and is therefore comparatively memory-hungry.
    """

    # Limited gradients buffer, shape (dimensionality, *state_shape).
    limited_gradients = jnp.zeros((config.dimensionality,) + primitive_state.shape)

    for axis in range(1, config.dimensionality + 1):
        limited_gradients = limited_gradients.at[axis - 1].set(
            _calculate_limited_gradients(
                primitive_state, config, helper_data, axis=axis
            )
        )

    differences = limited_gradients * config.grid_spacing / 2

    primitives_left_interface = jnp.zeros(
        (config.dimensionality,) + primitive_state.shape
    )
    primitives_right_interface = jnp.zeros(
        (config.dimensionality,) + primitive_state.shape
    )

    if config.limiter == VAN_ALBADA_PP:
        # positivity preserving reconstruction

        eps = 1e-14

        rho = primitive_state[registered_variables.density_index]
        p = primitive_state[registered_variables.pressure_index]
        c = speed_of_sound(rho, p, gamma)
        alpha_lax = jnp.zeros((config.dimensionality,))
        for axis in range(1, config.dimensionality + 1):
            u = primitive_state[axis]
            alpha_lax_i = jnp.max(jnp.abs(u) + c)
            alpha_lax = alpha_lax.at[axis - 1].set(alpha_lax_i)

        # NOTE: formula will change for different grid spacings along dimensions!!!

        C = alpha_lax / jnp.sum(alpha_lax)

        if config.mhd:
            # in the MHD case we go half-time steps as of
            # the strang splitting, so effectively, for the
            # hydro part we use C_cfl / 2, so 1 / (C_cfl / 2)
            # = 2 / C_cfl
            q = 2 / params.C_cfl
        else:
            q = 1 / params.C_cfl

        density_diff_protected = jnp.where(
            jnp.abs(differences[:, registered_variables.density_index]) > eps,
            differences[:, registered_variables.density_index],
            eps,
        )

        pressure_diff_protected = jnp.where(
            jnp.abs(differences[:, registered_variables.pressure_index]) > eps,
            differences[:, registered_variables.pressure_index],
            eps,
        )

        alpha_density = jnp.where(
            jnp.abs(differences[:, registered_variables.density_index]) > eps,
            jnp.minimum(
                primitive_state[registered_variables.density_index]
                / (jnp.abs(density_diff_protected) * (1 + eps)),
                1,
            ),
            1,
        )

        kappa_pressure = jnp.where(
            jnp.abs(differences[:, registered_variables.pressure_index]) > eps,
            jnp.minimum(
                primitive_state[registered_variables.pressure_index]
                / (jnp.abs(pressure_diff_protected) * (1 + eps)),
                1,
            ),
            1,
        )

        if config.dimensionality == 1:
            A1 = jnp.sum(
                jnp.sum(
                    C[:, None]
                    * alpha_density
                    * differences[:, registered_variables.density_index]
                    * differences[:, registered_variables.velocity_index],
                    axis=0,
                )
                ** 2,
                axis=0,
            )
            A2 = jnp.sum(
                C[:, None]
                * jnp.sum(
                    differences[:, registered_variables.velocity_index] ** 2, axis=1
                ),
                axis=0,
            )
        elif config.dimensionality == 2:
            A1 = jnp.sum(
                jnp.sum(
                    C[:, None, None]
                    * alpha_density
                    * differences[:, registered_variables.density_index]
                    * differences[
                        :,
                        registered_variables.velocity_index.x : registered_variables.velocity_index.x
                        + config.dimensionality,
                    ],
                    axis=0,
                )
                ** 2,
                axis=0,
            )
            A2 = jnp.sum(
                C[:, None, None]
                * jnp.sum(
                    differences[
                        :,
                        registered_variables.velocity_index.x : registered_variables.velocity_index.x
                        + config.dimensionality,
                    ]
                    ** 2,
                    axis=1,
                ),
                axis=0,
            )
        elif config.dimensionality == 3:
            A1 = jnp.sum(
                jnp.sum(
                    C[:, None, None, None]
                    * alpha_density
                    * differences[:, registered_variables.density_index]
                    * differences[
                        :,
                        registered_variables.velocity_index.x : registered_variables.velocity_index.x
                        + config.dimensionality,
                    ],
                    axis=0,
                )
                ** 2,
                axis=0,
            )
            A2 = jnp.sum(
                C[:, None, None, None]
                * jnp.sum(
                    differences[
                        :,
                        registered_variables.velocity_index.x : registered_variables.velocity_index.x
                        + config.dimensionality,
                    ]
                    ** 2,
                    axis=1,
                ),
                axis=0,
            )
        vsum = jnp.sum(
            jnp.sum(
                differences[
                    :,
                    registered_variables.velocity_index.x : registered_variables.velocity_index.x
                    + config.dimensionality,
                ]
                ** 2,
                axis=1,
            ),
            axis=0,
        )
        A1 = jnp.where(vsum > eps, A1, eps)
        A2 = jnp.where(vsum > eps, A2, eps)

        beta = jnp.where(
            vsum > eps,
            jnp.minimum(
                jnp.sqrt(
                    (
                        (q - 2) ** 2
                        * primitive_state[registered_variables.density_index]
                        * primitive_state[registered_variables.pressure_index]
                    )
                    / (
                        (gamma - 1)
                        * (
                            2 * A1
                            + (q - 2)
                            * primitive_state[registered_variables.density_index] ** 2
                            * A2
                        )
                    )
                ),
                1,
            ),
            1,
        )

        differences_pp = differences

        differences_pp = differences_pp.at[:, registered_variables.density_index].set(
            differences[:, registered_variables.density_index] * alpha_density
        )

        differences_pp = differences_pp.at[:, registered_variables.pressure_index].set(
            differences[:, registered_variables.pressure_index] * kappa_pressure
        )

        differences_pp = differences_pp.at[
            :,
            registered_variables.velocity_index.x : registered_variables.velocity_index.x
            + config.dimensionality,
        ].set(
            differences[
                :,
                registered_variables.velocity_index.x : registered_variables.velocity_index.x
                + config.dimensionality,
            ]
            * beta
        )

        differences = differences_pp

    for axis in range(1, config.dimensionality + 1):
        # i-1/2R, ...
        primitives_left_center = (
            primitive_state - differences[axis - 1]
        )  # left of the cell center but the right of the interface
        # i+1/2L, ...
        primitives_right_center = (
            primitive_state + differences[axis - 1]
        )  # right of the cell center but the left of the interface

        # primitives left at i is the left state at the interface
        # between i-1 and i so the right extrapolation from the cell i-1
        p_left_interface = jnp.roll(primitives_right_center, shift=1, axis=axis)

        # primitives right at i is the right state at the interface
        # between i-1 and i so the left extrapolation from the cell i
        p_right_interface = primitives_left_center

        # set the values
        primitives_left_interface = primitives_left_interface.at[axis - 1].set(
            p_left_interface
        )
        primitives_right_interface = primitives_right_interface.at[axis - 1].set(
            p_right_interface
        )

    return primitives_left_interface, primitives_right_interface


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "axis"])
def _reconstruct_at_interface_unsplit_single(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    helper_data: HelperData,
    axis: int,
):
    """
    Unsplit reconstruction.
    """

    limited_gradients = _calculate_limited_gradients(
        primitive_state, config, helper_data, axis=axis
    )
    differences = limited_gradients * config.grid_spacing / 2

    # i-1/2R, ...
    primitives_left_center = (
        primitive_state - differences
    )  # left of the cell center but the right of the interface
    # i+1/2L, ...
    primitives_right_center = (
        primitive_state + differences
    )  # right of the cell center but the left of the interface

    # primitives left at i is the left state at the interface
    # between i-1 and i so the right extrapolation from the cell i-1
    p_left_interface = jnp.roll(primitives_right_center, shift=1, axis=axis)

    # primitives right at i is the right state at the interface
    # between i-1 and i so the left extrapolation from the cell i
    p_right_interface = primitives_left_center

    return p_left_interface, p_right_interface
