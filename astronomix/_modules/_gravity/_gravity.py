"""
Self-gravity source terms coupling the gravitational potential to the fluid.

Assembles the total gravitational potential (self-gravity from the FFT Poisson
solve plus any external potential) and turns it into momentum and energy source
terms for the fluid. Several couplings are supported: a simple non-conservative
source and two conservative flux-based formulations (second- and fourth-order)
used by the finite-difference solver.
"""

# general
from functools import partial

# typing
from typing import Tuple, Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
    STATE_TYPE,
    UNSPLIT,
    VAN_ALBADA_PP,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._modules._gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._modules._gravity._utils import _pad_external_potential
from astronomix._modules._nbody._nbody import _deposit_nbody_density
from astronomix._stencil_operations._stencil_operations import _shift, _stencil_add
from astronomix._finite_volume._state_evolution.reconstruction import (
    _reconstruct_at_interface_unsplit_single,
)

@partial(jax.jit, static_argnames=["grid_spacing", "config", "registered_variables"])
def _compute_total_potential(
    gas_density: FIELD_TYPE,
    grid_spacing: float,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    G: Union[float, Float[Array, ""]] = 1.0,
) -> FIELD_TYPE:
    """
    Compute the total gravitational potential, including contributions from
    self-gravity (optionally combined with the deposited N-body masses) and
    any external potentials.

    Args:
        gas_density: The gas density field (ghost-cell padded, i.e. the
            shape of a single state field).
        grid_spacing: The grid spacing.
        config: The simulation configuration.
        params: The simulation parameters (provides the external potential
            and, when ``config.nbody_config.nbody``, the current N-body
            state/masses).
        registered_variables: The registered variables.
        G: The gravitational constant.

    Returns:
        The total gravitational potential, with the same shape as gas_density.
    """
    total_potential = jnp.zeros_like(gas_density)

    # Self-gravity contribution from the FFT Poisson solve. When the N-body
    # solver is active, the point masses are deposited onto the grid and
    # combined with the gas density before the solve, so the gas feels the
    # combined gas-plus-N-body potential. This coupling is one-directional:
    # the N-body integration (rk4_step_nbody) only ever depends on the other
    # bodies' masses, never on the gas, so the N-body masses gravitationally
    # act on the gas but the gas never acts back on them.
    if config.gravity_config.self_gravity:
        combined_density = gas_density
        if config.nbody_config.nbody:
            combined_density = combined_density + _deposit_nbody_density(
                params.nbody_params.nbody_state,
                params.nbody_params.masses,
                gas_density.shape,
                grid_spacing,
                config,
            )

        total_potential = total_potential + _compute_gravitational_potential(
            combined_density,
            grid_spacing,
            config,
            G,
        )

    # External-potential contribution. The external potential is supplied on the
    # bare grid, so it is given ghost cells matching the (here padded) density
    # field, filled according to the boundary conditions.
    if config.gravity_config.external_potential:
        external_potential = _pad_external_potential(
            params.gravitational_potential,
            gas_density,
            config,
            registered_variables,
            params,
        )
        total_potential = total_potential + external_potential

    return total_potential

def _fd_gravity_source(
    primitive_state: STATE_TYPE,
    density_fluxes,
    drho,
    dt,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    Build the finite-difference self-gravity source term for the full state.

    Computes the total gravitational potential and assembles the momentum and
    energy source contributions for every spatial axis, according to the
    configured coupling (``SIMPLE_SOURCE`` or one of the conservative,
    flux-based schemes).

    Args:
        primitive_state: The primitive state array.
        density_fluxes: The per-axis density fluxes at the cell faces, used by
            the conservative energy couplings.
        drho: The density change over the step, used by the conservative energy
            couplings to keep the ``phi * drho`` term consistent.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The full-state source term to be added over this time step.
    """

    S = jnp.zeros_like(primitive_state)

    gravitational_potential = _compute_total_potential(
        primitive_state[registered_variables.density_index],
        config.grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )

    if config.gravity_config.self_gravity_version == SIMPLE_SOURCE:

        for axis in range(1, config.dimensionality + 1):
            rho = primitive_state[registered_variables.density_index]
            v_axis = primitive_state[axis]

            # 6th-order centered finite difference for the gravitational
            # acceleration, a_i = -(phi_{i+3} - 9 phi_{i+2} + 45 phi_{i+1}
            # - 45 phi_{i-1} + 9 phi_{i-2} - phi_{i-3}) / (60 dx). The stencil
            # axis is ``axis - 1`` because the leading state axis indexes the
            # fields, not the spatial dimensions.
            acceleration = -_stencil_add(
                gravitational_potential,
                indices=(3, 2, 1, -1, -2, -3),
                factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
                axis=axis - 1,
            ) / (60.0 * config.grid_spacing)

            # Simple (non-conservative) coupling: rho * a for momentum and
            # rho * v * a for energy.
            S_axis = jnp.zeros_like(primitive_state)
            S_axis = S_axis.at[axis].set(rho * acceleration)
            S_axis = S_axis.at[registered_variables.pressure_index].set(
                rho * v_axis * acceleration
            )

            S += S_axis * dt
    elif config.gravity_config.self_gravity_version == SECOND_ORDER_CONSERVATIVE:

        for axis in range(1, config.dimensionality + 1):
            rho = primitive_state[registered_variables.density_index]
            phi_cell = gravitational_potential

            # Momentum source from the 6th-order centered potential gradient.
            acceleration = -_stencil_add(
                gravitational_potential,
                indices=(3, 2, 1, -1, -2, -3),
                factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
                axis=axis - 1,
            ) / (60.0 * config.grid_spacing)

            S_axis = jnp.zeros_like(primitive_state)
            S_axis = S_axis.at[axis].set(rho * acceleration)

            # Energy source built from the density fluxes so it is consistent
            # with the conservative update (no separate ``drho`` term needed).
            # The potential is interpolated to the right cell face i+1/2 with a
            # 6th-order symmetric stencil; the left face value is obtained by a
            # shift.
            phi_face = _stencil_add(
                gravitational_potential,
                indices=(-2, -1, 0, 1, 2, 3),
                factors=(3.0, -25.0, 150.0, 150.0, -25.0, 3.0),
                axis=axis - 1,
            ) / 256.0

            F_right = density_fluxes[axis - 1]  # density flux at i+1/2
            F_left = _shift(density_fluxes[axis - 1], 1, axis=axis - 1)  # at i-1/2
            phi_face_left = _shift(phi_face, 1, axis=axis - 1)  # phi at i-1/2

            # Energy source W_i = -[F_right (phi_right - phi_i)
            # + F_left (phi_i - phi_left)] / dx, which is the discrete form of
            # -div(F phi) + phi div(F) = -rho v grad(phi).
            energy_source = -(
                F_right * (phi_face - phi_cell)
                + F_left * (phi_cell - phi_face_left)
            ) / config.grid_spacing

            S_axis = S_axis.at[registered_variables.energy_index].set(energy_source)

            S += S_axis * dt

    elif config.gravity_config.self_gravity_version == FOURTH_ORDER_CONSERVATIVE:
        for axis in range(1, config.dimensionality + 1):
            spatial_axis = axis - 1

            rho = primitive_state[registered_variables.density_index]
            v_axis = primitive_state[axis]
            dx = config.grid_spacing

            # 6th-order interpolation of the potential to the cell faces.
            phi_face = _stencil_add(
                gravitational_potential,
                indices=(-2, -1, 0, 1, 2, 3),
                factors=(3.0, -25.0, 150.0, 150.0, -25.0, 3.0),
                axis=spatial_axis,
            ) / 256.0

            # 6th-order gravitational acceleration at the cell centers.
            acceleration = -_stencil_add(
                gravitational_potential,
                indices=(3, 2, 1, -1, -2, -3),
                factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
                axis=spatial_axis,
            ) / (60.0 * dx)

            S_axis = jnp.zeros_like(primitive_state)
            S_axis = S_axis.at[axis].set(rho * acceleration)

            # Corrected product form for the energy source. The fourth-order
            # product flux needs a correction term built from the curvature of
            # the potential and the gradient of the momentum density; second
            # order on the correction is sufficient to reach the overall order.
            f = rho * v_axis  # momentum density (rho v) at cell centers
            dPhi = -acceleration  # phi' at cell centers (6th order, reused)

            d2Phi = (  # phi'' at cell centers (2nd order)
                _shift(gravitational_potential, -1, axis=spatial_axis)
                - 2.0 * gravitational_potential
                + _shift(gravitational_potential, 1, axis=spatial_axis)
            ) / dx**2

            df = (  # f' at cell centers (2nd order)
                _shift(f, -1, axis=spatial_axis) - _shift(f, 1, axis=spatial_axis)
            ) / (2.0 * dx)

            # Correction at the cell centers, then averaged onto the faces.
            corr_cc = d2Phi * f + 2.0 * dPhi * df
            corr_face = 0.5 * (corr_cc + _shift(corr_cc, -1, axis=spatial_axis))

            # Corrected product flux and the resulting energy source -div(q_hat).
            q_hat = density_fluxes[axis - 1] * phi_face - (dx**2 / 24.0) * corr_face
            S_energy = -1.0 / dx * (q_hat - _shift(q_hat, 1, axis=spatial_axis))

            S_axis = S_axis.at[registered_variables.pressure_index].set(S_energy)
            S += S_axis * dt

        # Account for the change in potential energy due to the density change.
        S = S.at[registered_variables.energy_index].add(
            -drho * gravitational_potential
        )
    else:
        raise NotImplementedError("This scheme is not implemented.")

    return S

# -------------------------------------------------------------
# ============ Well-balanced FV coupling (opt-in) ==============
# -------------------------------------------------------------
#
# Both the plain SIMPLE_SOURCE and the momentum-conservative source above
# were checked against pytests/stratified_ism/stratified_hydrostatic_column.py
# (the SILCC-ISM M0b hydrostatic-column test): the momentum-conservative form
# left the equilibrium residual essentially unchanged (core Mach ~0.108 vs.
# ~0.106 for the plain form), confirming the dominant error is *not* the
# source term's own stencil, but the fact that the Riemann-solver's pressure
# flux (from a nonlinear solve on reconstructed states) has no reason to
# match *any* local closed-form source-term formula. The functions below
# implement the standard fix for that (Käppeli & Mishra 2014-style hydrostatic
# reconstruction, specialized for a contact-preserving solver like HLLC):
#
# 1. Build a *discrete* hydrostatic reference pressure P_eq (cell-centered,
#    arbitrary up to an additive constant -- only interface differences are
#    used) by cumulatively integrating the same interface density/potential
#    difference used everywhere else in this module.
# 2. Reconstruct the *perturbation* P - P_eq (not the raw pressure) with the
#    ordinary limiter, then add the same single-valued P_eq_face back onto
#    both the left and right interface states.
# 3. Replace the gravity momentum/energy source with P_eq_face's own flux
#    difference.
# 4. Evaluate that source at the *same* time-integration weight as step 2's
#    flux, not as a single once-per-step operator-split correction. Steps 2
#    and 3 cancel exactly only when evaluated from the same state; the
#    unsplit FV scheme's flux divergence (step 2's contribution) goes
#    through both stages of SSP-RK2, so the source has to as well, or the
#    two pick up different time-integration weights and stop canceling even
#    though they're spatially identical at every instant -- confirmed by
#    this being wrong in an earlier version of this fix: the once-per-step
#    presolve/apply pattern used for every *other* FV source (which is fine
#    there, since nothing is meant to cancel it exactly) left a genuine,
#    non-round-off residual even from an exactly discretely-hydrostatic
#    initial condition. See evolve_state.py's
#    ``well_balanced_inline_gravity``.
#
# The key property this relies on: for a *contact-preserving* Riemann solver
# (HLLC; the "C" is literally for this), an interface with matched pressure
# and zero velocity on both sides resolves as a stationary contact with
# *exactly* zero mass/energy flux and momentum flux equal to that matched
# pressure -- a numerical-flux consistency property (F(q, q) = f(q)), not an
# asymptotic one. So when the perturbation is exactly zero (discrete
# equilibrium), step 2 feeds the Riemann solver P_L = P_R = P_eq_face at
# every interface, its momentum flux is *exactly* P_eq_face (regardless of
# resolution, limiter, or solver-internal averaging details), and step 3's
# source exactly cancels that flux's divergence by construction -- round-off
# cancellation, not just reduced truncation error. Density needs no matching
# treatment: it only enters the flux via advection (rho * v), which is
# already exactly zero when v = 0 on both sides.


def _hydrostatic_pressure_reference(
    gravitational_potential: FIELD_TYPE,
    rho: FIELD_TYPE,
    axis: int,
) -> FIELD_TYPE:
    """
    Discrete cell-centered hydrostatic reference pressure P_eq along `axis`.

    Built by a running sum of the per-face increment P_eq_{i+1} - P_eq_i =
    -0.5*(rho_i + rho_{i+1})*(phi_{i+1} - phi_i), with P_eq_0 := 0 (the
    integration constant is arbitrary -- only interface *differences* are
    used downstream, both here and by
    :func:`_hydrostatic_pressure_reference_face`). Note: the increment that
    would wrap from the last cell back to the first (relevant only for a
    periodic axis) is computed but never read, so this is correct for both
    open and periodic axes without special-casing.

    Args:
        gravitational_potential: The gravitational potential.
        rho: The gas density.
        axis: The state-array axis (1-indexed; spatial axis is axis - 1).

    Returns:
        The reference pressure field, same shape as rho.
    """
    spatial_axis = axis - 1
    n = rho.shape[spatial_axis]

    phi_plus = _shift(gravitational_potential, -1, axis=spatial_axis)  # phi_{i+1}
    rho_plus = _shift(rho, -1, axis=spatial_axis)  # rho_{i+1}
    increment = -0.5 * (rho + rho_plus) * (phi_plus - gravitational_potential)

    cumulative = jnp.cumsum(increment, axis=spatial_axis)
    cumulative_interior = jax.lax.slice_in_dim(cumulative, 0, n - 1, axis=spatial_axis)
    zero_first = jnp.zeros_like(jax.lax.slice_in_dim(rho, 0, 1, axis=spatial_axis))
    return jnp.concatenate([zero_first, cumulative_interior], axis=spatial_axis)


def _hydrostatic_pressure_reference_face(
    p_eq: FIELD_TYPE,
    axis: int,
) -> FIELD_TYPE:
    """
    Face-centered P_eq, at the same i-1/2 faces (indexed at cell i) that
    :func:`_reconstruct_at_interface_unsplit_single` and the flux-divergence
    update use.
    """
    spatial_axis = axis - 1
    p_eq_minus = _shift(p_eq, 1, axis=spatial_axis)  # P_eq_{i-1}
    return 0.5 * (p_eq + p_eq_minus)  # at face i-1/2


def _reconstruct_pressure_well_balanced(
    primitive_state: STATE_TYPE,
    gravitational_potential: FIELD_TYPE,
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    axis: int,
):
    """
    Drop-in replacement for ``_reconstruct_at_interface_unsplit_single``
    (same call signature minus ``primitive_state``/``axis`` ordering) that
    reconstructs the pressure row relative to the discrete hydrostatic
    reference instead of raw pressure -- see this section's module-level
    comment for why.

    Only the pressure row is touched: density/velocity/B (and any other
    registered field) go through the same reconstruction, unperturbed, since
    reconstruction here is row-independent (no characteristic decomposition).

    Returns:
        ``(primitives_left_interface, primitives_right_interface)``, exactly
        as ``_reconstruct_at_interface_unsplit_single`` would.
    """
    p_index = registered_variables.pressure_index
    rho = primitive_state[registered_variables.density_index]

    p_eq = _hydrostatic_pressure_reference(gravitational_potential, rho, axis)
    perturbed_state = primitive_state.at[p_index].add(-p_eq)

    p_left, p_right = _reconstruct_at_interface_unsplit_single(
        perturbed_state, config, helper_data, axis
    )

    p_eq_face = _hydrostatic_pressure_reference_face(p_eq, axis)
    p_left = p_left.at[p_index].add(p_eq_face)
    p_right = p_right.at[p_index].add(p_eq_face)
    return p_left, p_right

# -------------------------------------------------------------
# ========== ↑ Well-balanced FV coupling (opt-in) ↑ ============
# -------------------------------------------------------------

# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["axis", "grid_spacing", "registered_variables", "config"]
)
def _gravitational_source_term_along_axis(
    gravitational_potential: FIELD_TYPE,
    primitive_state: STATE_TYPE,
    grid_spacing: float,
    registered_variables: RegisteredVariables,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    axis: int,
) -> STATE_TYPE:
    """
    Compute the source term for the self-gravity solver along a single axis.

    Momentum-conservative, face-based coupling: the force density is built
    from *interface* quantities -- density and potential difference averaged
    onto the same i+1/2 / i-1/2 faces the Riemann-solver flux divergence
    uses -- rather than a single cell-centered a_i = -(phi_{i+1} -
    phi_{i-1})/(2 dx) estimate. The two differ whenever rho varies across the
    stencil (i.e. exactly the stratified-ISM regime this module targets),
    and averaging the two half-face contributions makes the discrete force
    on any neighboring cell pair equal and opposite (Newton's third law
    holds exactly at the discrete level), which the old wide-stencil,
    cell-centered form did not guarantee.

    This does not make the FV coupling fully well-balanced (the pressure
    side of the force still comes from a nonlinear Riemann solve on
    reconstructed states, which this source does not mirror), but it removes
    the stencil-width mismatch (2 dx vs. dx) identified as the likely driver
    of the resolution-independent hydrostatic-equilibrium residual found at
    SILCC-ISM milestone M0b (see PROGRESS.md and
    pytests/stratified_ism/stratified_hydrostatic_column.py) -- that residual
    was observed to *not* shrink under grid refinement, which rules out
    ordinary truncation error and points at a structural stencil mismatch
    rather than a resolution effect.

    Args:
        gravitational_potential: The gravitational potential.
        primitive_state: The primitive state.
        grid_spacing: The grid spacing.
        registered_variables: The registered variables.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.
        axis: The axis along which to compute the source term.

    Returns:
        The source term.

    """

    rho = primitive_state[registered_variables.density_index]
    v_axis = primitive_state[axis]
    spatial_axis = axis - 1

    # This must mirror *exactly* the condition evolve_state.py's
    # _evolve_gas_state_unsplit_inner uses to pick the well-balanced
    # reconstruction (_reconstruct_pressure_well_balanced) over the plain
    # one. well_balanced_fv_gravity alone is not enough: the SPLIT scheme
    # never calls the well-balanced reconstruction (it only exists in the
    # unsplit inner loop) and VAN_ALBADA_PP forces the plain reconstruction
    # even when well_balanced_fv_gravity is set (see that function's elif
    # chain and simulation_config.py's well_balanced_fv_gravity docstring).
    # Using the well-balanced *source* there anyway would pair it with a
    # flux built from the raw (non-perturbation) pressure -- not a
    # mismatched-but-harmless combination, but one that cancels nothing
    # (the source would subtract a P_eq gradient the flux never added in
    # the first place), so it must fall back to the plain source too.
    use_well_balanced = (
        config.gravity_config.well_balanced_fv_gravity
        and config.split == UNSPLIT
        and config.limiter != VAN_ALBADA_PP
    )

    if use_well_balanced:
        # See this file's "Well-balanced FV coupling" section: the momentum
        # source is P_eq_face's own flux difference, built from the exact
        # same P_eq the paired reconstruction wrapper
        # (_reconstruct_pressure_well_balanced) uses, so the two cancel
        # exactly for a discretely-hydrostatic state -- *provided* the two
        # are evaluated from the same state at the same time-integration
        # weight; see evolve_state.py's well_balanced_inline_gravity for the
        # RK-stage-level half of this.
        p_eq = _hydrostatic_pressure_reference(gravitational_potential, rho, axis)
        p_eq_face = _hydrostatic_pressure_reference_face(p_eq, axis)
        p_eq_face_plus = _shift(p_eq_face, -1, axis=spatial_axis)  # face i+1/2
        momentum_source = (p_eq_face_plus - p_eq_face) / grid_spacing

        source_term = jnp.zeros_like(primitive_state)
        source_term = source_term.at[axis].set(momentum_source)
        source_term = source_term.at[registered_variables.pressure_index].set(
            momentum_source * v_axis
        )
        return source_term

    # Neighbor values via the shared _shift stencil (periodic; open/reflective
    # BCs are enforced separately by the ghost-cell handler before this runs).
    phi_plus = _shift(gravitational_potential, -1, axis=spatial_axis)  # phi_{i+1}
    phi_minus = _shift(gravitational_potential, 1, axis=spatial_axis)  # phi_{i-1}
    rho_plus = _shift(rho, -1, axis=spatial_axis)  # rho_{i+1}
    rho_minus = _shift(rho, 1, axis=spatial_axis)  # rho_{i-1}

    # Face-centered acceleration and density, on the same i+1/2 / i-1/2
    # faces (dx apart, not 2 dx) that the flux divergence differences.
    g_right = -(phi_plus - gravitational_potential) / grid_spacing  # at i+1/2
    g_left = -(gravitational_potential - phi_minus) / grid_spacing  # at i-1/2
    rho_right = 0.5 * (rho + rho_plus)  # at i+1/2
    rho_left = 0.5 * (rho + rho_minus)  # at i-1/2

    # Momentum-conservative force density: the average of the two half-face
    # contributions, each of which is shared exactly (equal and opposite)
    # with the corresponding neighbor cell.
    momentum_source = 0.5 * (rho_right * g_right + rho_left * g_left)

    source_term = jnp.zeros_like(primitive_state)

    # set momentum source
    source_term = source_term.at[axis].set(momentum_source)

    # finite-volume self-gravity supports only the SIMPLE_SOURCE coupling
    # (the FD-only conservative flux schemes live in _fd_gravity_source).
    source_term = source_term.at[registered_variables.pressure_index].set(
        momentum_source * v_axis
    )

    return source_term