"""
Phase D milestone D2: gradient-based inference demo -- fitting DSA injection
efficiency (``CosmicRayGreyParams.dsa_efficiency_mach_scale``) to a synthetic
pion-decay emission map, through a **full live rollout** with real shock
propagation and DSA feedback on (plan Sec. 1: "gradient-based inference demo
(fit kappa / injection efficiency to a synthetic map)").

**Why this is riskier than D1 (``cr_phase_d_kappa_inference.py``).**
``cr_gradient_check.py``'s own injection-efficiency gradient check
(``test_cr_gradient_check_injection``) only validated
``dsa_efficiency_mach_scale`` gradients against a *fixed,
``stop_gradient``-wrapped pre-shock state* feeding a single
``inject_crs_at_shocks`` call -- not a full live rollout. A first attempt at
differentiating through a full live rollout there gave ~20% AD-vs-FD
relative error: not a differentiability bug, but a real effect -- as
``dsa_efficiency_mach_scale`` varies, the shock weakens at a different rate,
which shifts *which cell* the shock finder's per-zone selection flags as the
surface cell at each step, injecting discrete, poorly-behaved jumps into the
gradient. This script confronts that directly rather than sidestepping it,
per explicit user direction (see ``PROGRESS.md``'s Phase D entry).

**Base setup**: identical to ``cr_dsa_mach_dependence.py``'s ``_run_sedov``
(itself identical to ``cr_sedov_taylor.py``'s setup) -- a 3D Sedov-Taylor
blast, ``NUM_CELLS=48``, KR13 Mach-dependent DSA model, since a real
propagating shock is required for injection efficiency to have any effect at
all. Unlike D1, the emission map's target-gas weighting is the run's own
*simulated* ``rho`` field (not a fixed synthetic profile) -- physically
meaningful here since the shocked/swept-up shell's density genuinely varies
in space and is exactly the "dense target gas is the signal" mechanism the
plan's emission section describes. Map projection (line-of-sight sum along
z) and the exact-linearity emission trick match D1 and ladder item 15's
``_build_emission_maps`` -- see D1's module docstring for the full
derivation.

**Structure, per this module's "investigate, don't dig indefinitely"
pattern**: (1) measure the actual AD-vs-FD gradient error *in this specific
setup* first (don't assume item 16's ~20% number transfers unchanged); (2)
attempt the optax/Adam fit regardless -- Adam's momentum may average out a
noisy per-step gradient well enough to still converge; (3) report the actual
outcome (clean convergence / noisy-but-convergent / divergent) rather than
tuning a tolerance to force a pass.

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md's Phase D section.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import time
from pathlib import Path

# jax
import jax
import jax.numpy as jnp
import optax

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import SimulationConfig, SimulationParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import BACKWARDS

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
    DSA_EFFICIENCY_KANG_RYU_2013,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_emission import (
    pion_decay_photon_spectrum_per_ev,
    proton_spectrum_normalized_to_energy,
)

# Matches D1/cr_gradient_check.py's rationale: a tight AD-vs-FD comparison
# needs 64-bit precision.
jax.config.update("jax_enable_x64", True)

# ---- physical setup (matches cr_dsa_mach_dependence.py's/cr_sedov_taylor.py's
# _run_sedov exactly) ----
GAMMA = 5.0 / 3.0
NUM_CELLS = 48
T_END = 0.07
E_EXPLOSION = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1e-4
R_EXPLOSION = 0.05
SMOOTH_CELLS = 2.0
DSA_MACH_MIN = 1.3

# ---- illustrative emission-map conversion (see D1's module docstring for
# the linearity-trick derivation; same constants reused here for
# consistency -- absolute scale is irrelevant since the loss below is a
# relative squared error) ----
GEV_SCALE = 1e38
SPECTRUM_ALPHA = 2.0
SPECTRUM_E_CUTOFF_GEV = 1e5
PHOTON_ENERGY_GEV = 3.0

# ---- inference target/guess ----
MACH_SCALE_TRUE = 1.0
MACH_SCALE_GUESS_INIT = 0.4
LEARNING_RATE = 0.1
NUM_STEPS = 40


def _build_config_and_ic():
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        exact_end_time=True,
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True,
            diffusive_shock_acceleration=True,
            dsa_efficiency_model=DSA_EFFICIENCY_KANG_RYU_2013,
        ),
        # Reverse-mode AD through the adaptive-dt while loop needs the
        # checkpointed backend -- see module docstring.
        differentiation_mode=BACKWARDS,
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (NUM_CELLS, NUM_CELLS, NUM_CELLS)
    density = jnp.ones(shape) * RHO_AMBIENT
    zeros = jnp.zeros(shape)

    dx = 1.0 / NUM_CELLS  # box_size = 1.0 (default)
    smooth_width = SMOOTH_CELLS * dx
    radius = helper_data.r
    weight = 0.5 * (1.0 - jnp.tanh((radius - R_EXPLOSION) / smooth_width))
    cell_volume = dx**3
    delta_p = E_EXPLOSION * (GAMMA - 1.0) / (jnp.sum(weight) * cell_volume)
    gas_pressure = P_AMBIENT + delta_p * weight

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, initial_state.shape)
    return config, registered_variables, initial_state


_CONFIG, _REGVARS, _IC = _build_config_and_ic()

# Photons/(s eV) per unit (1 GeV total proton energy content) per unit (1
# cm^-3 gas density), at PHOTON_ENERGY_GEV -- see D1's module docstring for
# the exact-linearity derivation this exploits.
_UNIT_SPECTRUM = proton_spectrum_normalized_to_energy(
    1.0, alpha=SPECTRUM_ALPHA, e_cutoff=SPECTRUM_E_CUTOFF_GEV
)
_K_PION_1GEV = pion_decay_photon_spectrum_per_ev(_UNIT_SPECTRUM, PHOTON_ENERGY_GEV, 1.0)


def _emission_map(e_cr, rho):
    """``(e_cr, rho) -> synthetic pion-decay emission map`` (2D,
    line-of-sight-summed along z, photons/(s eV)) -- uses the run's own
    simulated density as the target-gas weighting (unlike D1's fixed
    synthetic profile), via the linearity trick (see module docstring).
    Stays entirely in ``jnp`` so gradients flow through it."""
    total_energy_gev = e_cr * GEV_SCALE
    rate = total_energy_gev * rho * _K_PION_1GEV
    return jnp.sum(rate, axis=2)


def forward_model(mach_scale):
    """``dsa_efficiency_mach_scale -> (synthetic emission map, final
    simulation state)``, running the real 3D Sedov-Taylor CR-DSA blast to
    ``T_END`` with the shock, DSA injection, and CR feedback all live."""
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            dsa_efficiency=0.0,  # unused by the KR13 path; kept explicit at 0
            dsa_mach_min=DSA_MACH_MIN,
            dsa_efficiency_mach_scale=mach_scale,
        ),
    )
    final_state = time_integration(_IC, _CONFIG, params, _REGVARS)
    e_cr_final = final_state[_REGVARS.cosmic_ray_e_index]
    rho_final = final_state[_REGVARS.density_index]
    return _emission_map(e_cr_final, rho_final), final_state


def _loss(log_mach_scale, map_target):
    """Relative squared error between the predicted and target maps."""
    mach_scale = jnp.exp(log_mach_scale)
    map_pred, _ = forward_model(mach_scale)
    return jnp.sum((map_pred - map_target) ** 2) / jnp.sum(map_target**2)


def _fd_vs_ad_gradient_check(log_mach_scale_0, map_target):
    """Step 1: measure the actual AD-vs-FD gradient error in this setup, at
    the optimizer's own starting point (not at MACH_SCALE_TRUE itself, where
    the loss/gradient is ~0 and a relative-error comparison is ill-defined).
    Returns (ad_grad, fd_grad, rel_err)."""
    ad_grad = float(jax.grad(_loss)(log_mach_scale_0, map_target))

    h = 1e-3
    fd_grad = float(
        (_loss(log_mach_scale_0 + h, map_target) - _loss(log_mach_scale_0 - h, map_target))
        / (2 * h)
    )
    rel_err = abs(ad_grad - fd_grad) / abs(fd_grad) if fd_grad != 0.0 else float("inf")
    return ad_grad, fd_grad, rel_err


def _optimize(map_target, log_mach_scale_init, num_steps, lr):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(log_mach_scale_init)

    @jax.jit
    def step(log_mach_scale, opt_state):
        loss_value, grad = jax.value_and_grad(_loss)(log_mach_scale, map_target)
        updates, opt_state = optimizer.update(grad, opt_state, log_mach_scale)
        log_mach_scale = optax.apply_updates(log_mach_scale, updates)
        return log_mach_scale, opt_state, loss_value

    log_mach_scale = log_mach_scale_init
    losses, mach_scales = [], []
    for i in range(num_steps):
        log_mach_scale, opt_state, loss_value = step(log_mach_scale, opt_state)
        losses.append(float(loss_value))
        mach_scales.append(float(jnp.exp(log_mach_scale)))
        print(f"  step {i:3d}: mach_scale = {mach_scales[-1]:.5f}, loss = {losses[-1]:.6e}")

    return log_mach_scale, losses, mach_scales


def test_cr_phase_d_injection_efficiency_inference():
    """Investigation, not a guaranteed-pass test (see module docstring):
    measures the live-rollout AD-vs-FD gradient error for
    ``dsa_efficiency_mach_scale`` in this setup, attempts a gradient-based
    fit regardless, and reports the actual outcome. Only asserts what should
    hold unconditionally (no NaNs); does not assert parameter recovery to a
    pre-chosen tolerance."""
    t0 = time.time()
    map_target, target_state = forward_model(MACH_SCALE_TRUE)
    map_target = jax.lax.stop_gradient(map_target)
    one_run_s = time.time() - t0
    print(f"One forward run: {one_run_s:.2f}s")

    assert not bool(jnp.any(jnp.isnan(target_state))), (
        "Synthetic-target forward run at MACH_SCALE_TRUE produced NaNs."
    )

    log_mach_scale_init = jnp.asarray(jnp.log(MACH_SCALE_GUESS_INIT))

    t0 = time.time()
    ad_grad, fd_grad, rel_err = _fd_vs_ad_gradient_check(log_mach_scale_init, map_target)
    one_grad_check_s = time.time() - t0
    print(
        f"Step 1 -- live-rollout AD-vs-FD check at the initial guess "
        f"({one_grad_check_s:.2f}s, {NUM_STEPS} optimizer steps planned):\n"
        f"  AD grad = {ad_grad:.6e}, FD grad = {fd_grad:.6e}, rel. err = {rel_err:.3e}"
    )

    print(
        f"Step 2 -- attempting the fit regardless: mach_scale true="
        f"{MACH_SCALE_TRUE}, initial guess={MACH_SCALE_GUESS_INIT}"
    )
    log_mach_scale_final, losses, mach_scales = _optimize(
        map_target, log_mach_scale_init, NUM_STEPS, LEARNING_RATE
    )

    assert not any(loss_value != loss_value for loss_value in losses), (
        "Loss went NaN during optimization."
    )

    mach_scale_final = float(jnp.exp(log_mach_scale_final))
    rel_err_final = abs(mach_scale_final - MACH_SCALE_TRUE) / MACH_SCALE_TRUE
    print(
        f"Step 3 -- outcome: recovered mach_scale = {mach_scale_final:.5f} "
        f"(true = {MACH_SCALE_TRUE}), rel. err = {rel_err_final:.3e}, "
        f"loss trajectory {losses[0]:.4e} -> {losses[-1]:.4e}"
    )

    map_pred_final, _ = forward_model(mach_scale_final)

    fig, (ax_loss, ax_scale, ax_target, ax_pred) = plt.subplots(1, 4, figsize=(20, 4.5))

    ax_loss.plot(losses, color="tab:blue")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("optimizer step")
    ax_loss.set_ylabel("relative squared error")
    ax_loss.set_title("Training loss")

    ax_scale.plot(mach_scales, "o-", color="tab:blue", ms=3, label="fitted")
    ax_scale.axhline(MACH_SCALE_TRUE, color="k", ls=":", label="true")
    ax_scale.set_xlabel("optimizer step")
    ax_scale.set_ylabel("dsa_efficiency_mach_scale")
    ax_scale.set_title(
        f"Parameter recovery (Step 1 AD-vs-FD rel. err = {rel_err:.2f})"
    )
    ax_scale.legend()

    im0 = ax_target.imshow(map_target.T, origin="lower", cmap="inferno")
    ax_target.set_title("Synthetic target map")
    fig.colorbar(im0, ax=ax_target, fraction=0.046)

    im1 = ax_pred.imshow(map_pred_final.T, origin="lower", cmap="inferno")
    ax_pred.set_title("Final fitted map")
    fig.colorbar(im1, ax=ax_pred, fraction=0.046)

    fig.suptitle(
        "Phase D (D2): gradient-based injection-efficiency inference through "
        "a full live rollout"
    )
    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_phase_d_injection_efficiency_inference_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_cr_phase_d_injection_efficiency_inference()
