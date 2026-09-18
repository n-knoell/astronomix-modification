"""
CR differentiability pytest (plan Sec. 4, item 16 -- "continuous, from Phase
A", own wording: "transport, then injection, then emission").

Finite-difference vs. autodiff gradient checks on small CR problems: perturb
a parameter, compute a scalar cost, and compare the AD gradient
(``jax.grad``) against a centered finite-difference gradient. Catches a
floor/limiter/``jnp.where`` silently killing the adjoint -- exactly the
failure mode the plan's differentiability plan (``tanh``-regularized
streaming sign, no ``min``/``max``/``sign``) is designed to avoid. See
``pytests/differentiability/sensitivity.py`` for the general style this repo
uses for gradient-correctness pytests (there: AD vs. a closed-form analytic
gradient for the non-CR Euler equations; here: AD vs. finite differences,
since a closed-form CR gradient isn't available) -- that test also
establishes that ``time_integration``'s adaptive-dt loop to a fixed
``t_end`` is end-to-end differentiable, which ``test_cr_gradient_check``
relies on too.

Three stages, added incrementally as the corresponding ladder phase landed
(transport at Phase A; injection/emission added once Phases B/C were both
complete, per DESIGN.md's "Next per the plan's own staging" note):

1. ``test_cr_gradient_check`` (Phase A): transport only. Setup mirrors
   ``cr_advection.py``'s right-moving-eigenmode pulse (a smooth
   ``e_cr``/``F_cr`` profile, CR-free background so the transport path
   dominates, not run a full box-crossing period -- just far enough that
   ``reduced_streaming_speed`` visibly shapes the final state). Only
   ``grey_cr_flux_terms``, ``grey_cr_fast_speed``,
   ``cr_pressure_gradient_source`` and ``cr_adiabatic_work_source`` are
   exercised (the default config keeps
   ``anisotropic_transport``/``streaming``/``diffusive_relaxation`` off, so
   the still-unimplemented ``cr_streaming_heating_source`` stays out of the
   path).

2. ``test_cr_gradient_check_injection`` (Phase B): DSA shock injection,
   targeting ``dsa_efficiency_kang_ryu_2013``'s Mach-dependent piecewise fit
   specifically -- the one part of the injection path with a nontrivial
   ``jnp.where``-guarded rational-function branch (5 < Ms <= 15), same
   differentiability-gotcha category as this test's own
   ``cr_pressure_speed_floor`` fix (see ``test_cr_gradient_check``'s
   docstring in DESIGN.md) and ``pion_decay_photon_spectrum``'s regime
   splits below.

3. ``test_cr_gradient_check_emission`` (Phase C): chains the same injection
   step into ``proton_spectrum_normalized_to_energy`` and
   ``pion_decay_photon_spectrum``, directly exercising the plan's own Sec. 3
   goal ("gradients flow sim -> spectrum -> map") end to end, not just the
   per-formula smoke checks ladder items 13-15 did inline (finite/nonzero,
   never cross-checked against finite differences through a real injection
   step -- see DESIGN.md's ladder-item-15 section, "Next per the plan's own
   staging").

**Design note for stages 2-3, found empirically (not a guess): differentiate
through a single ``inject_crs_at_shocks`` call on a *fixed* pre-shocked
state, not through the full adaptive ``time_integration`` loop with DSA
feedback enabled.** A first attempt did the latter (DSA on throughout a
short shock-tube run, cost as a function of
``dsa_efficiency_mach_scale``) and got AD vs. FD relative error ~20% --
not a differentiability bug, but real nonlinear feedback (diverting more
thermal energy into CRs measurably weakens the shock, which shifts the
adaptive-dt trajectory and which cell the shock finder's per-zone argmax
selects as the surface cell) compounding over many steps, the same category
of trap flagged in this project's M4 work ("the same run at finer resolution
is not actually the same run"). Isolating the injection formula against a
*fixed* (``jax.lax.stop_gradient``-wrapped) control state -- generated once
by a real DSA-off run, then reused for every perturbed parameter value --
removes both confounds and brought the relative error to ~1e-9. This
mirrors ``cr_dsa_mach_dependence.py``'s own "layer 2's main check", which
isolates the same function the same way for the same reason (an exact
formula cross-check, not an aggregate-energy comparison, needs a fixed
pre-shock state).

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md's differentiability plan.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_VOLUME,
    BoundarySettings1D,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    finalize_config,
)

# astronomix functions
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
    DSA_EFFICIENCY_KANG_RYU_2013,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import inject_crs_at_shocks
from astronomix._modules._cosmic_rays_grey.cr_grey_emission import (
    proton_spectrum_normalized_to_energy,
    pion_decay_photon_spectrum,
)

# Matches sensitivity.py's rationale: a tight AD-vs-finite-difference
# comparison needs 64-bit precision, or round-off swamps the comparison.
jax.config.update("jax_enable_x64", True)


def _cost(reduced_streaming_speed, config, gamma_cr, primitive_state, t_end, registered_variables):
    """Run the CR-grey pulse to ``t_end`` and return a scalar cost of the
    final ``e_cr`` field, as a function of ``reduced_streaming_speed``."""
    params = SimulationParams(
        t_end=t_end,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=gamma_cr, reduced_streaming_speed=reduced_streaming_speed
        ),
    )
    final_state = time_integration(primitive_state, config, params, registered_variables)
    e_cr_final = final_state[registered_variables.cosmic_ray_e_index]
    return jnp.sum(e_cr_final**2)


def test_cr_gradient_check(tol: float = 1e-2):
    """Finite-difference vs. autodiff gradient check on a small CR problem.

    Args:
        tol: The maximum allowed relative error between the AD and
            finite-difference gradients.
    """
    num_cells = 64
    box_size = 1.0
    gamma_cr = 4.0 / 3.0
    reduced_streaming_speed_0 = 1.0
    amp = 1e-3
    sigma = 0.05
    x0 = 0.5 * box_size

    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=box_size,
        boundary_settings=BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True),
        # Reverse-mode AD through the adaptive-dt while loop needs the
        # checkpointed backend -- see time_integration.py's
        # differentiation_mode dispatch.
        differentiation_mode=BACKWARDS,
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers

    wave_speed_0 = reduced_streaming_speed_0 * jnp.sqrt(gamma_cr - 1.0)
    pulse = amp * jnp.exp(-0.5 * ((x - x0) / sigma) ** 2)

    primitive_state = jnp.zeros((registered_variables.num_vars, num_cells))
    primitive_state = primitive_state.at[registered_variables.density_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(pulse)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_flux_index].set(
        wave_speed_0 * pulse
    )
    config = finalize_config(config, primitive_state.shape)

    # A fraction of one box-crossing period at the fiducial speed -- short
    # enough to be cheap, long enough that reduced_streaming_speed visibly
    # shapes the final state via the pulse's propagation distance.
    t_end = 0.2 * box_size / wave_speed_0

    cost_fn = lambda v: _cost(v, config, gamma_cr, primitive_state, t_end, registered_variables)

    ad_grad = float(jax.grad(cost_fn)(reduced_streaming_speed_0))

    h = 1e-4 * reduced_streaming_speed_0
    fd_grad = float(
        (cost_fn(reduced_streaming_speed_0 + h) - cost_fn(reduced_streaming_speed_0 - h))
        / (2 * h)
    )

    rel_err = abs(ad_grad - fd_grad) / abs(fd_grad)
    print(f"AD grad = {ad_grad:.6e}, FD grad = {fd_grad:.6e}, rel. err = {rel_err:.3e}")

    assert not (ad_grad != ad_grad), "AD gradient is NaN."
    assert rel_err < tol, (
        f"AD vs FD gradient mismatch: rel. err {rel_err:.3e} >= tol {tol} "
        f"(AD={ad_grad:.6e}, FD={fd_grad:.6e})."
    )


def _shocked_control_state():
    """Build a fixed, real shock (no CR injection yet) as a stop-gradient
    input for the injection/emission gradient checks below.

    A 1D open-boundary shock-tube-like setup (no rarefaction-side CR
    pressure -- this is a plain gas shock, unlike ``cr_shock_tube.py``'s
    two-fluid problem), tuned so the shock's detected Mach number
    (``max_Ms ~= 8``) lands inside ``dsa_efficiency_kang_ryu_2013``'s
    "intermediate" piece (5 < Ms <= 15) -- the branch with the
    differentiability-sensitive ``1 / ms_safe**4`` term. ``dsa_efficiency =
    0`` (DSA code path live, injecting nothing) keeps this a pure control
    run: a real shock, zero ``e_cr`` yet, exactly the precondition
    ``inject_crs_at_shocks`` needs to be exercised for the first time by the
    functions below.

    Returns:
        Tuple of ``(config, registered_variables, helper_data,
        state_with_shock, current_time, dt)`` -- the last two are the fixed
        ``current_time``/``dt`` arguments the gradient checks below pass to
        ``inject_crs_at_shocks`` directly (no time-integration loop).
    """
    num_cells = 128
    box_size = 1.0
    x0 = 0.3 * box_size
    gamma = 5.0 / 3.0
    t_end = 0.02
    dsa_mach_min = 1.3
    rho_L, p_L = 1.0, 50.0
    rho_R, p_R = 0.01, 0.01

    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=box_size,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True,
            diffusive_shock_acceleration=True,
            dsa_efficiency_model=DSA_EFFICIENCY_KANG_RYU_2013,
        ),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers

    left = x < x0
    rho = jnp.where(left, rho_L, rho_R)
    p_th = jnp.where(left, p_L, p_R)

    primitive_state = jnp.zeros((registered_variables.num_vars, num_cells))
    primitive_state = primitive_state.at[registered_variables.density_index].set(rho)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(p_th)
    config = finalize_config(config, primitive_state.shape)

    control_params = SimulationParams(
        t_end=t_end,
        gamma=gamma,
        cosmic_ray_grey_params=CosmicRayGreyParams(dsa_efficiency=0.0, dsa_mach_min=dsa_mach_min),
    )
    state_with_shock = time_integration(
        primitive_state, config, control_params, registered_variables
    )
    state_with_shock = jax.lax.stop_gradient(state_with_shock)

    # Arbitrary fixed dt for the standalone injection-formula check -- any
    # concrete value works equally well (matches cr_dsa_mach_dependence.py's
    # identical choice for its own single-step cross-check), since nothing
    # here re-derives dt from a CFL condition.
    dt = 1e-4
    return config, registered_variables, helper_data, state_with_shock, t_end, dt


def _injected_e_cr(mach_scale, config, registered_variables, helper_data, state_with_shock, current_time, dt):
    """One ``inject_crs_at_shocks`` call on the fixed shocked state, as a
    function of ``dsa_efficiency_mach_scale``. Returns the resulting
    ``e_cr`` field."""
    params = SimulationParams(
        cosmic_ray_grey_params=CosmicRayGreyParams(
            dsa_mach_min=1.3, dsa_efficiency_mach_scale=mach_scale
        ),
    )
    new_state = inject_crs_at_shocks(
        state_with_shock, config, params, registered_variables, helper_data, current_time, dt
    )
    return new_state[registered_variables.cosmic_ray_e_index]


def test_cr_gradient_check_injection(tol: float = 1e-2):
    """Finite-difference vs. autodiff gradient check on DSA shock injection.

    Args:
        tol: The maximum allowed relative error between the AD and
            finite-difference gradients.
    """
    config, registered_variables, helper_data, state_with_shock, current_time, dt = (
        _shocked_control_state()
    )

    def cost_fn(mach_scale):
        e_cr = _injected_e_cr(
            mach_scale, config, registered_variables, helper_data, state_with_shock, current_time, dt
        )
        return jnp.sum(e_cr**2)

    mach_scale_0 = 0.5
    ad_grad = float(jax.grad(cost_fn)(mach_scale_0))

    h = 1e-4 * mach_scale_0
    fd_grad = float((cost_fn(mach_scale_0 + h) - cost_fn(mach_scale_0 - h)) / (2 * h))

    rel_err = abs(ad_grad - fd_grad) / abs(fd_grad)
    print(f"AD grad = {ad_grad:.6e}, FD grad = {fd_grad:.6e}, rel. err = {rel_err:.3e}")

    assert not (ad_grad != ad_grad), "AD gradient is NaN."
    assert rel_err < tol, (
        f"AD vs FD gradient mismatch: rel. err {rel_err:.3e} >= tol {tol} "
        f"(AD={ad_grad:.6e}, FD={fd_grad:.6e})."
    )


def test_cr_gradient_check_emission(tol: float = 1e-2):
    """Finite-difference vs. autodiff gradient check chaining DSA injection
    into the pion-decay emission formula (sim -> spectrum -> emission).

    The proton spectrum's assumed shape (``alpha=2.0``, ``e_cutoff=1e5``
    GeV) and the GeV/gas-density scale below are illustrative, chosen only
    to exercise ``pion_decay_photon_spectrum``'s regime-split
    differentiability at a representative point on its domain -- unlike
    ladder item 15, this is not a physically-calibrated prediction (no
    ``CodeUnits`` conversion; the "small CR problem" here is a
    differentiability check, not a science result).

    Args:
        tol: The maximum allowed relative error between the AD and
            finite-difference gradients.
    """
    config, registered_variables, helper_data, state_with_shock, current_time, dt = (
        _shocked_control_state()
    )

    # Arbitrary code-energy -> GeV scale (illustrative, see docstring) and a
    # reference photon energy/gas density landing well inside
    # pion_decay_photon_spectrum's smooth interior, away from any regime
    # boundary's floating-point edge.
    gev_scale = 1e40
    gas_number_density_cm3 = 1.0
    photon_energy_gev = 3.0

    def cost_fn(mach_scale):
        e_cr = _injected_e_cr(
            mach_scale, config, registered_variables, helper_data, state_with_shock, current_time, dt
        )
        total_energy_gev = jnp.sum(e_cr) * gev_scale
        spectrum = proton_spectrum_normalized_to_energy(total_energy_gev, alpha=2.0, e_cutoff=1e5)
        return pion_decay_photon_spectrum(spectrum, photon_energy_gev, gas_number_density_cm3)

    mach_scale_0 = 0.5
    ad_grad = float(jax.grad(cost_fn)(mach_scale_0))

    h = 1e-4 * mach_scale_0
    fd_grad = float((cost_fn(mach_scale_0 + h) - cost_fn(mach_scale_0 - h)) / (2 * h))

    rel_err = abs(ad_grad - fd_grad) / abs(fd_grad)
    print(f"AD grad = {ad_grad:.6e}, FD grad = {fd_grad:.6e}, rel. err = {rel_err:.3e}")

    assert not (ad_grad != ad_grad), "AD gradient is NaN."
    assert rel_err < tol, (
        f"AD vs FD gradient mismatch: rel. err {rel_err:.3e} >= tol {tol} "
        f"(AD={ad_grad:.6e}, FD={fd_grad:.6e})."
    )


if __name__ == "__main__":
    test_cr_gradient_check()
    test_cr_gradient_check_injection()
    test_cr_gradient_check_emission()
