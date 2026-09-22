"""
Phase D milestone D1: gradient-based inference demo -- fitting kappa
(``CosmicRayGreyParams.diffusion_coefficient``) to a synthetic pion-decay
emission map (plan Sec. 1: "gradient-based inference demo (fit kappa /
injection efficiency to a synthetic map)").

Reuses ladder item 4's exact physical setup
(``cr_isotropic_diffusion_convergence.py``'s ``_run_diffusion``: a narrow
``e_cr`` Gaussian bump, isotropic diffusive relaxation, uniform static gas,
periodic BCs) and ladder item 16's live-rollout gradient path
(``differentiation_mode=BACKWARDS``, already validated for this exact
parameter/setup family in ``cr_gradient_check.py``'s ``test_cr_gradient_check``,
tol 1e-2) -- so the only genuinely new risk here is the emission-map layer on
top, not the transport gradient itself.

**Map construction.** The final ``e_cr(x)`` profile is converted to a
synthetic pion-decay photon map via the same exact-linearity trick ladder
item 15's ``_build_emission_maps`` uses (``pion_decay_photon_spectrum`` is
exactly linear in both the proton spectrum's ``amplitude`` and the gas
density, so a full grid's worth of emission values reduces to one pion-decay
integral, computed once at unit amplitude/density, times a cheap elementwise
product) -- kept entirely in ``jnp`` here (unlike item 15's diagnostic-only
``np.asarray`` cast) so gradients flow map -> spectrum -> transport. A fixed,
non-uniform *target-gas* density profile (``_N_GAS``, a Gaussian "cloud" bump
offset from the CR injection site -- not part of the simulation's own
uniform ``rho``, just a post-hoc emissivity weighting, matching how the plan
treats target gas as external context) is applied so the resulting map is a
genuinely different spatial shape from the raw ``e_cr(x)`` field, not merely
a rescaled copy of it -- kappa's effect (how far the bump spreads by
``T_END``) interacts with this fixed weighting nontrivially. The GeV
conversion (``GEV_SCALE``) and assumed spectral shape
(``SPECTRUM_ALPHA``/``SPECTRUM_E_CUTOFF_GEV``) are illustrative, not a
physically calibrated ``CodeUnits`` conversion -- matches
``cr_gradient_check.py``'s own "Arbitrary code-energy -> GeV scale"
precedent (this is a differentiability/inference demo, not a science
prediction).

**Inference.** A synthetic target map is generated once at ``KAPPA_TRUE``
(no noise -- an exact target, since this is a clean gradient/optimization
sanity check, not a statistical inverse-problem study). ``log_kappa`` (not
``kappa`` directly) is the optimized variable, so Adam's unconstrained steps
can never drive the physical diffusion coefficient negative. Starting from
``kappa_guess = 2 * KAPPA_TRUE``, ``jax.grad`` + ``optax.adam`` minimize the
relative squared error between the predicted and target maps.

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
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_VOLUME,
    BoundarySettings1D,
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
)
from astronomix._modules._cosmic_rays_grey.cr_grey_emission import (
    pion_decay_photon_spectrum_per_ev,
    proton_spectrum_normalized_to_energy,
)

# Matches cr_gradient_check.py's rationale: a tight AD-vs-FD-adjacent
# optimization (log-space Adam steps of a few percent) needs 64-bit
# precision, or round-off swamps the small, physically-meaningful signal.
jax.config.update("jax_enable_x64", True)

# ---- physical setup (matches cr_isotropic_diffusion_convergence.py's own
# calibration exactly, at a single resolution rather than a convergence
# scan) ----
NUM_CELLS = 128
BOX_SIZE = 1.0
GAMMA_CR = 4.0 / 3.0
REDUCED_STREAMING_SPEED = 8.0
AMP = 1e-3
SIGMA0 = 0.02
X0 = 0.5 * BOX_SIZE
T_END = 0.24
KAPPA_TRUE = 0.06

# ---- fixed "target gas" emissivity weighting (see module docstring) ----
X_CLOUD = X0 + 0.15
SIGMA_CLOUD = 0.05
N_GAS_BASE = 1.0
N_GAS_CLOUD_AMP = 3.0

# ---- illustrative emission-map conversion (see module docstring) ----
GEV_SCALE = 1e38
SPECTRUM_ALPHA = 2.0
SPECTRUM_E_CUTOFF_GEV = 1e5
PHOTON_ENERGY_GEV = 3.0

# ---- optimization ----
KAPPA_GUESS_INIT = 2.0 * KAPPA_TRUE
LEARNING_RATE = 0.1
NUM_STEPS = 60


def _build_config_and_ic():
    """Build the (fixed, reused-every-call) config/IC -- matches
    ``cr_isotropic_diffusion_convergence.py``'s ``_run_diffusion`` setup."""
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=NUM_CELLS,
        box_size=BOX_SIZE,
        boundary_settings=BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_relaxation=True
        ),
        # Reverse-mode AD through the adaptive-dt while loop needs the
        # checkpointed backend -- see cr_gradient_check.py.
        differentiation_mode=BACKWARDS,
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers

    pulse = AMP * jnp.exp(-0.5 * ((x - X0) / SIGMA0) ** 2)
    primitive_state = jnp.zeros((registered_variables.num_vars, NUM_CELLS))
    primitive_state = primitive_state.at[registered_variables.density_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(1.0)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(pulse)

    config = finalize_config(config, primitive_state.shape)
    n_gas = N_GAS_BASE + N_GAS_CLOUD_AMP * jnp.exp(-0.5 * ((x - X_CLOUD) / SIGMA_CLOUD) ** 2)
    return config, registered_variables, primitive_state, x, n_gas


_CONFIG, _REGVARS, _IC, _X, _N_GAS = _build_config_and_ic()

# Photons/(s eV) per unit (1 GeV total proton energy content) per unit (1
# cm^-3 gas density), at PHOTON_ENERGY_GEV -- computed once, reused for every
# forward-model call, via pion_decay_photon_spectrum's exact linearity in
# both spectrum.amplitude and gas_number_density_cm3 (see module docstring).
_UNIT_SPECTRUM = proton_spectrum_normalized_to_energy(
    1.0, alpha=SPECTRUM_ALPHA, e_cutoff=SPECTRUM_E_CUTOFF_GEV
)
_K_PION_1GEV = pion_decay_photon_spectrum_per_ev(_UNIT_SPECTRUM, PHOTON_ENERGY_GEV, 1.0)


def _emission_map(e_cr):
    """``e_cr(x) -> synthetic pion-decay emission map`` (1D, photons/(s
    eV)), via the fixed target-gas weighting and the linearity trick above.
    Stays entirely in ``jnp`` (see module docstring) so gradients flow
    through it."""
    total_energy_gev = e_cr * GEV_SCALE
    return total_energy_gev * _N_GAS * _K_PION_1GEV


def forward_model(kappa):
    """``kappa -> (synthetic emission map, final simulation state)``,
    running the real CR-grey diffusive-relaxation transport to ``T_END``."""
    params = SimulationParams(
        t_end=T_END,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR,
            reduced_streaming_speed=REDUCED_STREAMING_SPEED,
            diffusion_coefficient=kappa,
        ),
    )
    final_state = time_integration(_IC, _CONFIG, params, _REGVARS)
    e_cr_final = final_state[_REGVARS.cosmic_ray_e_index]
    return _emission_map(e_cr_final), final_state


def _loss(log_kappa, map_target):
    """Relative squared error between the predicted and target maps --
    scale-invariant, so it doesn't depend on the illustrative GEV_SCALE/
    _K_PION_1GEV normalization chosen above."""
    kappa = jnp.exp(log_kappa)
    map_pred, _ = forward_model(kappa)
    return jnp.sum((map_pred - map_target) ** 2) / jnp.sum(map_target**2)


def _optimize(map_target, log_kappa_init, num_steps, lr):
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(log_kappa_init)

    @jax.jit
    def step(log_kappa, opt_state):
        loss_value, grad = jax.value_and_grad(_loss)(log_kappa, map_target)
        updates, opt_state = optimizer.update(grad, opt_state, log_kappa)
        log_kappa = optax.apply_updates(log_kappa, updates)
        return log_kappa, opt_state, loss_value

    log_kappa = log_kappa_init
    losses, kappas = [], []
    for i in range(num_steps):
        log_kappa, opt_state, loss_value = step(log_kappa, opt_state)
        losses.append(float(loss_value))
        kappas.append(float(jnp.exp(log_kappa)))
        print(f"  step {i:3d}: kappa = {kappas[-1]:.5f}, loss = {losses[-1]:.6e}")

    return log_kappa, losses, kappas


def test_cr_phase_d_kappa_inference(
    kappa_rel_tol: float = 0.05, loss_reduction_factor: float = 0.1
):
    """Recover ``KAPPA_TRUE`` by fitting a synthetic pion-decay emission map.

    Args:
        kappa_rel_tol: Maximum allowed relative error between the final
            fitted ``kappa`` and ``KAPPA_TRUE``.
        loss_reduction_factor: The final loss must drop below this fraction
            of the initial loss -- a coarse "did the optimizer do anything
            at all" guard, independent of the tighter parameter-recovery
            check above.
    """
    t0 = time.time()
    map_target, target_state = forward_model(KAPPA_TRUE)
    map_target = jax.lax.stop_gradient(map_target)
    one_run_s = time.time() - t0
    print(f"One forward run: {one_run_s:.2f}s")

    assert not bool(jnp.any(jnp.isnan(target_state))), (
        "Synthetic-target forward run at KAPPA_TRUE produced NaNs."
    )

    t0 = time.time()
    log_kappa_init = jnp.asarray(jnp.log(KAPPA_GUESS_INIT))
    _ = jax.grad(_loss)(log_kappa_init, map_target)
    one_grad_s = time.time() - t0
    print(f"One jax.grad eval: {one_grad_s:.2f}s ({NUM_STEPS} steps planned)")

    print(f"Fitting kappa: true={KAPPA_TRUE}, initial guess={KAPPA_GUESS_INIT}")
    log_kappa_final, losses, kappas = _optimize(
        map_target, log_kappa_init, NUM_STEPS, LEARNING_RATE
    )

    assert not any(loss_value != loss_value for loss_value in losses), (
        "Loss went NaN during optimization."
    )

    kappa_final = float(jnp.exp(log_kappa_final))
    rel_err = abs(kappa_final - KAPPA_TRUE) / KAPPA_TRUE
    print(
        f"Recovered kappa = {kappa_final:.5f} (true = {KAPPA_TRUE}), "
        f"rel. err = {rel_err:.3e}"
    )

    assert losses[-1] < loss_reduction_factor * losses[0], (
        f"Loss did not drop enough: final {losses[-1]:.4e} vs. initial "
        f"{losses[0]:.4e} (ratio {losses[-1] / losses[0]:.4f} >= "
        f"{loss_reduction_factor})."
    )
    assert rel_err < kappa_rel_tol, (
        f"Recovered kappa ({kappa_final:.5f}) too far from KAPPA_TRUE "
        f"({KAPPA_TRUE}): rel. err {rel_err:.4e} >= tol {kappa_rel_tol}."
    )

    map_pred_final, _ = forward_model(kappa_final)

    fig, (ax_loss, ax_kappa, ax_map) = plt.subplots(1, 3, figsize=(16, 4.5))

    ax_loss.plot(losses, color="tab:blue")
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("optimizer step")
    ax_loss.set_ylabel("relative squared error")
    ax_loss.set_title("Training loss")

    ax_kappa.plot(kappas, "o-", color="tab:blue", ms=3, label=r"fitted $\kappa$")
    ax_kappa.axhline(KAPPA_TRUE, color="k", ls=":", label=r"true $\kappa$")
    ax_kappa.set_xlabel("optimizer step")
    ax_kappa.set_ylabel(r"$\kappa$")
    ax_kappa.set_title("Parameter recovery")
    ax_kappa.legend()

    ax_map.plot(_X, map_target, color="k", lw=1.5, label="synthetic target")
    ax_map.plot(_X, map_pred_final, color="tab:red", ls="--", label="fitted forward model")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("emission map [photons / (s eV)]")
    ax_map.set_title("Final map comparison")
    ax_map.legend()

    fig.suptitle(
        "Phase D (D1): gradient-based kappa inference from a synthetic "
        "pion-decay emission map"
    )
    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_phase_d_kappa_inference_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_cr_phase_d_kappa_inference()
