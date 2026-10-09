"""
Grey CR feedback source terms (plan Sec. 2): the coupling from the
independently-tracked ``e_cr``/``F_cr`` state back into the gas.

Each function returns a conserved-state-shaped **rate** (not yet multiplied
by ``dt``), matching the convention of the existing FD source terms in
``astronomix._modules._time_integrator_sources`` (e.g. ``fd_viscosity_source``),
so callers there compose them as ``source_term += cr_xxx_source(...) * dt``.
Exception: :func:`cr_flux_relaxation_update` returns the updated state, applied
implicitly inside every RK stage (see its docstring).
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
    cr_entropy_from_e_cr,
    e_cr_from_cr_entropy,
    pressure_from_e_cr,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_transport import (
    cr_flux_rows,
    cr_passive_row_flux,
    magnetic_unit_vector,
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
def cr_flux_relaxation_update(
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    magnetic_field: Union[Float[Array, "3 ..."], None] = None,
) -> STATE_TYPE:
    """Implicit ``F_cr`` scattering/relaxation step (Jiang & Oh 2018),
    ``d(F_cr)/dt = -nu . F_cr`` over ``dt`` (backward Euler), isotropic or
    along/across B.

    - Isotropic (``diffusive_relaxation``): ``F_cr <- F_cr / (1 + nu dt)``,
      ``nu = (gamma_cr - 1) v_red^2 / kappa``.
    - Anisotropic (``anisotropic_transport``): the rate is a tensor,
      ``nu = (gamma_cr - 1) v_red^2 K^-1`` with
      ``K = kappa_par b b + kappa_perp (I - b b)``, so

          F_cr <- [ a_par b b + a_perp (I - b b) ] . F_cr,
          a = kappa / (kappa + (gamma_cr - 1) v_red^2 dt)

      per direction (``kappa_par = diffusion_coefficient``,
      ``kappa_perp = perpendicular_diffusion_coefficient``). The kappa form
      stays finite and smooth at ``kappa_perp = 0``, where ``a_perp = 0``
      removes ``F_perp`` completely (the B-projection). Without
      ``diffusive_relaxation`` the parallel part is undamped (``a_par = 1``,
      pure wave transport along B, ladder item 3).

    Args:
        primitive_state: The primitive state after an explicit hydro stage.
        dt: The stage time step.
        config: The simulation configuration.
        registered_variables: The registered variables.
        params: The simulation parameters (carries
            ``params.cosmic_ray_grey_params.diffusion_coefficient`` and
            ``perpendicular_diffusion_coefficient``).
        magnetic_field: The cell-centered ``(B_x, B_y, B_z)`` on the same
            grid as ``primitive_state``; required with
            ``anisotropic_transport``. Passed separately because in FV MHD
            the gas half-steps run on a gas-only state
            (``evolve_state._split_gas_and_magnetic_state``). B does not
            change during a gas half-step, so this is exact.

    Returns:
        The state with the updated ``F_cr`` row(s).

    Without this term, ``grey_cr_flux_terms``'s two-moment system is a pure
    undamped wave equation (verified in ladder items 1-3: a localized e_cr
    bump propagates rigidly, it does not spread). This term damps ``F_cr``
    at rate ``nu`` against its flux-gradient forcing
    (``-v_red^2 * grad(P_cr)``, from ``grey_cr_flux_terms``'s pressure-driving
    term) -- *toward zero*, the drive stays in the flux; relaxing toward
    ``-K . grad(e_cr)`` as well would count it twice. At a quasi-steady
    balance (``d(F_cr)/dt ~ 0`` on timescales long compared to ``1/nu``,
    ignoring bulk advection) this gives ``F_cr ~= -(v_red^2 / nu) grad(P_cr)
    = -K . grad(e_cr)``, i.e. Fick's law with diffusivity ``K`` for ``e_cr``
    itself -- the literature convention ``d(e_cr)/dt = div(K grad e_cr)``
    (Girichidis et al. 2016) -- see ``cr_isotropic_diffusion_convergence.py``
    (ladder item 4) and ``cr_diffusion_rate.py``. Until 2026-10-04 the rate
    had no ``(gamma_cr - 1)`` factor, so the ``e_cr`` diffusivity was
    ``diffusion_coefficient / 3`` (DESIGN.md "Open: CR diffusion
    correctness", Problem 1).

    Called by ``_evolve_gas_state_unsplit`` after *every* forward-Euler hydro
    stage of the SSP-RK2 step (Jiang & Oh 2018 likewise add the source
    implicitly in each stage of their integrator). Placement matters:
    - Applied once per step on the operator-split source path (the
      explicit ``-nu F_cr`` rate used until 2026-10-04), ``F_cr`` evolves
      undamped through both RK stages, and the stage-averaged ``e_cr`` flux,
      hence D, comes out too large by ``1 + nu dt / 2``. Applied implicitly
      once after the step, the same bias grows without bound (measured
      ``+0.49 nu dt``). Per stage, a steady ``F_cr`` is a fixed point of
      every stage, so the bias is gone (DESIGN.md Problem 3).
    - The same holds across B: the old once-per-step projection let each
      step's first stage build ``F_perp`` undamped, a leak of ~0.14
      ``nu dt kappa`` (DESIGN.md Problem 2). Here every stage starts from
      ``F_perp = a_perp F_perp``.
    - Backward Euler is unconditionally stable, so there is no ``dt <~ 1 /
      nu`` limit on the time step (formerly a ``_cfl_time_step`` branch),
      also not for small ``kappa_perp``.
    - Every factor is smooth, so the update stays differentiable.

    Deliberately opt-in (``diffusive_relaxation`` and
    ``anisotropic_transport`` default to False) so ladder items 1-3's
    already-verified undamped-wave behavior is unchanged.
    """
    cr_config = config.cosmic_ray_grey_config
    cr_params = params.cosmic_ray_grey_params
    # (gamma_cr - 1) v_red^2 dt, the "kappa" of one implicit step.
    kappa_step = (
        (cr_params.gamma_cr - 1.0) * cr_params.reduced_streaming_speed**2 * dt
    )
    if cr_config.diffusive_relaxation:
        damping_par = cr_params.diffusion_coefficient / (
            cr_params.diffusion_coefficient + kappa_step
        )
    else:
        damping_par = 1.0

    flux_rows = cr_flux_rows(registered_variables)

    if not cr_config.anisotropic_transport:
        for row in flux_rows:
            primitive_state = primitive_state.at[row].multiply(damping_par)
        return primitive_state

    kappa_perp = cr_params.perpendicular_diffusion_coefficient
    damping_perp = kappa_perp / (kappa_perp + kappa_step)

    # flux_rows is in x/y/z order. 2D MHD carries F_z as well (B_z is
    # evolved): with B_z != 0 part of the field-aligned flux is F_z, and
    # dropping it would shrink the in-plane parallel flux every stage
    # (DESIGN.md "Open: CR diffusion follow-up", F1).
    b_hat = magnetic_unit_vector(magnetic_field, params)
    f_dot_b = sum(primitive_state[row] * b_hat[i] for i, row in enumerate(flux_rows))
    for i, row in enumerate(flux_rows):
        primitive_state = primitive_state.at[row].set(
            damping_perp * primitive_state[row]
            + (damping_par - damping_perp) * f_dot_b * b_hat[i]
        )
    return primitive_state


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


@partial(jax.jit, static_argnames=["registered_variables"])
def cr_entropy_sync(
    primitive_state: STATE_TYPE,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
) -> STATE_TYPE:
    """Set the CR entropy row from ``e_cr``, ``s_cr = P_cr rho^(1 - gamma_cr)``.

    Called at the start of every hydro update, so every ``e_cr`` change made
    outside it (injections, the floor, continuous updates, initial
    conditions) is picked up without those call sites knowing about
    ``s_cr``. Mirrors the gas dual-energy ``_dual_energy_sync``.
    """
    rho = primitive_state[registered_variables.density_index]
    e_cr = primitive_state[registered_variables.cosmic_ray_e_index]
    return primitive_state.at[registered_variables.cosmic_ray_entropy_index].set(
        cr_entropy_from_e_cr(e_cr, rho, params.cosmic_ray_grey_params.gamma_cr)
    )


@partial(jax.jit, static_argnames=["registered_variables"])
def cr_entropy_to_energy(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    gas_energy_cells=None,
) -> STATE_TYPE:
    """Take ``e_cr`` from the advected CR entropy and give the difference to the
    gas thermal energy (DESIGN.md "Open: conservative CR entropy at shocks",
    design step 5).

    At the end of a hydro step the conservative ``e_cr`` row (fluxes plus the
    operator-split ``-P_cr div u``) and the gas energy conserve
    ``E_gas + e_cr`` exactly, but the partition between them is not unique at
    shocks. ``s_cr`` carries the CRs adiabatically, so ``e_cr := e(s_cr)``
    and ``P_th += (gamma - 1) (e_cr,old - e_cr,new)`` keeps the total and
    fixes the partition. Where that would take more than
    ``cr_entropy_max_thermal_drain`` of the gas thermal energy, the transfer
    is limited and the rest stays in ``e_cr``.

    Args:
        primitive_state: The primitive state after the full hydro step,
            including the operator-split sources.
        gamma: The gas adiabatic index.
        registered_variables: The registered variables.
        params: The simulation parameters.
        gas_energy_cells: Optional boolean mask of the cells whose gas pressure
            comes from the total energy (gas dual energy). Elsewhere ``e_cr``
            is taken from ``s_cr`` and the gas is left unchanged: the
            dual-energy scheme already discards the total-energy residue there.
            None: all cells.

    Returns:
        The state with ``e_cr`` and the gas pressure updated.
    """
    rho = primitive_state[registered_variables.density_index]
    e_cr_energy = primitive_state[registered_variables.cosmic_ray_e_index]
    e_cr_entropy = e_cr_from_cr_entropy(
        primitive_state[registered_variables.cosmic_ray_entropy_index],
        rho,
        params.cosmic_ray_grey_params.gamma_cr,
    )
    delta = e_cr_energy - e_cr_entropy

    # Gas-positivity safeguard: a negative delta takes energy from the gas;
    # take at most cr_entropy_max_thermal_drain of its thermal energy above
    # the floor and leave the rest in e_cr.
    p_th = primitive_state[registered_variables.pressure_index]
    available = jnp.maximum(p_th - params.minimum_pressure, 0.0) / (gamma - 1.0)
    delta = jnp.maximum(
        delta, -params.cosmic_ray_grey_params.cr_entropy_max_thermal_drain * available
    )

    e_cr_new = e_cr_energy - delta
    if gas_energy_cells is not None:
        e_cr_new = jnp.where(gas_energy_cells, e_cr_new, e_cr_entropy)
        delta = jnp.where(gas_energy_cells, delta, 0.0)

    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(e_cr_new)
    primitive_state = primitive_state.at[registered_variables.pressure_index].add(
        (gamma - 1.0) * delta
    )
    # Keep s_cr consistent with the (possibly limited) e_cr.
    return cr_entropy_sync(primitive_state, registered_variables, params)


@partial(jax.jit, static_argnames=["config", "registered_variables", "axis"])
def cr_entropy_closure_source(
    conserved_change: STATE_TYPE,
    fluxes: STATE_TYPE,
    primitives_left_interface: STATE_TYPE,
    primitives_right_interface: STATE_TYPE,
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    axis: int,
) -> STATE_TYPE:
    """Add the non-adiabatic CR transport of one axis to the CR entropy row
    (DESIGN.md "Open: conservative CR entropy at shocks", design step 4).

    The ``e_cr`` interface flux is the advective part (mass flux x upwind
    ``e_cr / rho``, which ``s_cr`` already gets for its own row) plus the
    closure part (``F_cr`` and its Rusanov term: diffusion relative to the
    gas). This recomputes the advective part with the same arithmetic as
    ``hll._grey_cr_hll_rows``, takes the closure part of this axis' ``e_cr``
    change as the difference, and adds it to ``s_cr`` cell by cell with the
    weight ``(gamma_cr - 1) rho^(1 - gamma_cr)`` of the stage-start density.
    The weight is cell-centred on purpose: a face-weighted flux would add a
    spurious ``C d(w)/dx`` term in stratified gas. For static gas the
    resulting ``e(s_cr)`` equals the ``e_cr`` row exactly.

    Args:
        conserved_change: This axis' conserved-state change, ``-dt/dx`` times
            the flux difference.
        fluxes: This axis' interface fluxes, as returned by the Riemann solver.
        primitives_left_interface, primitives_right_interface: The interface
            states the Riemann solver was called with.
        primitive_state: The stage-start primitive state.
        dt: The stage time step.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        axis: The flux direction (1, 2 or 3).

    Returns:
        ``conserved_change`` with the closure source added to the ``s_cr`` row.
    """
    e = registered_variables.cosmic_ray_e_index
    rho = registered_variables.density_index
    gamma_cr = params.cosmic_ray_grey_params.gamma_cr

    advective_flux = cr_passive_row_flux(
        fluxes[rho],
        primitives_left_interface[e],
        primitives_right_interface[e],
        primitives_left_interface[rho],
        primitives_right_interface[rho],
    )
    advective_change = (
        1
        / config.grid_spacing
        * _stencil_add(advective_flux, indices=(0, 1), factors=(1.0, -1.0), axis=axis - 1)
        * dt
    )
    closure_change = conserved_change[e] - advective_change
    weight = (gamma_cr - 1.0) * primitive_state[rho] ** (1.0 - gamma_cr)
    return conserved_change.at[registered_variables.cosmic_ray_entropy_index].add(
        weight * closure_change
    )
