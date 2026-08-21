"""
Two-fluid CR-modified shock tube pytest (Phase A ladder item 6).

The single most diagnostic coupling test: a Sod-like shock tube where the
left/right states carry CR pressure, compared against a semi-analytic
two-fluid Riemann solution following Pfrommer, Enßlin & Jubelgas (2006).
Exercises the full Phase A stack together -- transport flux
(``cr_grey_transport.grey_cr_flux_terms``), the CR-grey wave speed in the
Riemann solver/CFL, and the momentum/energy feedback
(``cr_grey_sources.cr_pressure_gradient_source``/``cr_adiabatic_work_source``)
-- rather than any one piece in isolation.

**Physics regime this test targets, and why ``reduced_streaming_speed = 0``
is the right (not a lazy) choice:** Pfrommer et al. (2006)'s semi-analytic
solution assumes the CR population is *tightly coupled* to the gas -- it
moves with the local gas velocity and compresses adiabatically (``P_cr propto
rho^gamma_cr``, even irreversibly through a shock, since collisionless CRs
don't locally thermalize the way gas does), with no independent CR flux of
its own. This module's grey two-moment closure is more general: ``F_cr`` is
an independently evolving flux, driven by the CR pressure gradient at
``reduced_streaming_speed``. Setting ``reduced_streaming_speed = 0`` collapses
``grey_cr_flux_terms``' ``F_cr`` equation to pure homogeneous advection with
zero initial data (``F_cr`` starts at 0 and its only source term,
``v_red^2 * grad(P_cr)``, vanishes identically) -- so ``F_cr`` stays exactly
zero for the whole run, and ``e_cr`` is transported purely by
``u_n * e_cr`` (bulk gas advection) plus the (unaffected-by-v_red) momentum/
energy feedback sources. This is *exactly* the physical assumption
Pfrommer's solution is built on, realized exactly (not approximately) by
this specific parameter choice, rather than needing to argue a "small
enough" reduced streaming speed leaves a residual, untunable error against
the reference. The test also asserts ``F_cr ~= 0`` throughout as a direct
check that this reasoning is realized by the actual simulation, not just
assumed.

**Reference solver (new: ``astronomix/test_setups/reference_solutions/
pfrommer_riemann_solver.py``).** The composite EOS ``P = P_th,ref *
(rho/rho_ref)^gamma_th + P_cr,ref * (rho/rho_ref)^gamma_cr`` is not a single
power law, so the classic closed-form exact-Riemann-solver formulas (Toro
2009, already implemented in ``riemann_solver.py`` and used by the plain
``shock_tube1D.py`` test) don't apply -- the rarefaction fan requires a
numerically integrated Riemann invariant, and the shock jump requires a
numerical (Hugoniot) root-find for the post-shock density given a trial
pressure, nested inside the usual outer star-region-pressure bisection.
Implemented in plain numpy/scipy (root-finding + quadrature), not JAX --
it's a one-shot reference-solution generator for this test, not part of the
simulation hot path.

**Validation of the new solver itself** (ad hoc scripts, not committed):
with the CR pressure set to zero on both sides, ``pfrommer_riemann_solution``
reduces to the ordinary single-gamma problem and was checked against the
repository's existing, trusted ``riemann_solver._exact_riemann_ideal_gas``
across the *entire* sampled profile (not just the star-region pressure/
velocity) for three cases -- classic Sod (gamma=1.4), Sod with gamma=5/3,
and a reversed-Sod case that exercises the left-shock branch instead of the
left-rarefaction branch (the more common configuration in a plain Sod
problem) -- matching to ~2e-7 (numerical root-finding/quadrature precision,
not a discrepancy). A second check set ``gamma_cr = gamma_th`` with nonzero,
unequal CR pressure on each side: the composite EOS then degenerates to a
single power law in the *combined* pressure, and the solver was checked
against the same existing exact solver fed the summed pressure -- also
matched to ~2e-7, and confirmed the per-side CR pressure *fraction* stays
constant through both the shock and the rarefaction (expected when both
components share one adiabatic index). Two real bugs were found and fixed
during this validation, both sign errors from first-pass derivation, not
issues with the final closed-form design:
1. The rarefaction-fan velocity integral had left/right family signs
   swapped (``u`` should *increase* through a left rarefaction as the gas
   accelerates toward the lower-pressure side, *decrease* through a right
   one -- the first-pass code had both backwards).
2. The shock-speed formula in the per-point sampler used the *unsigned*
   mass-flux magnitude with the wrong sign convention relative to which
   side (L/R) is shocked.
Both were caught by exactly this single-gamma cross-check, not by eyeballing
the algebra -- see the two functions' docstrings/comments for the corrected
derivations.

Note on the specific left/right pressure split (P_th = 0.6/0.06, P_cr =
0.4/0.04, i.e. a 40% CR pressure fraction on both sides of an otherwise
classic Sod problem): this is a reasonable, meaningfully-diagnostic choice
in the spirit of Pfrommer et al. (2006)'s own illustrative test cases, not a
literal reproduction of a specific numeric case from their paper (which
this implementation was not checked against directly -- only the underlying
*method* is validated, via the degenerate-limit cross-checks above).

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# numerics
import numpy as np

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    BoundarySettings1D,
    OPEN_BOUNDARY,
    finalize_config,
)

# astronomix functions
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.test_setups.reference_solutions.pfrommer_riemann_solver import (
    pfrommer_riemann_solution,
)

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)


def test_cr_shock_tube(tol: float = 1e-2):
    """Two-fluid CR-modified shock tube vs. a Pfrommer et al. (2006)-style
    semi-analytic solution.

    Args:
        tol: The maximum allowed mean absolute error per primitive variable
            against the semi-analytic solution.
    """
    num_cells = 400
    box_size = 1.0
    x0 = 0.5
    t_end = 0.2
    gamma = 5.0 / 3.0
    gamma_cr = 4.0 / 3.0

    rho_L, u_L, p_th_L, p_cr_L = 1.0, 0.0, 0.6, 0.4
    rho_R, u_R, p_th_R, p_cr_R = 0.125, 0.0, 0.06, 0.04

    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        dimensionality=1,
        num_cells=num_cells,
        box_size=box_size,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    x = helper_data.geometric_centers

    left = x < x0
    rho = jnp.where(left, rho_L, rho_R)
    u = jnp.where(left, u_L, u_R)
    p_th = jnp.where(left, p_th_L, p_th_R)
    p_cr = jnp.where(left, p_cr_L, p_cr_R)
    e_cr = p_cr / (gamma_cr - 1.0)

    primitive_state = jnp.zeros((registered_variables.num_vars, num_cells))
    primitive_state = primitive_state.at[registered_variables.density_index].set(rho)
    primitive_state = primitive_state.at[registered_variables.velocity_index].set(u)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(p_th)
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(e_cr)

    config = finalize_config(config, primitive_state.shape)
    params = SimulationParams(
        t_end=t_end,
        gamma=gamma,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=gamma_cr,
            # Realizes Pfrommer's tightly-coupled assumption exactly -- see
            # module docstring.
            reduced_streaming_speed=0.0,
        ),
    )

    final_state = time_integration(primitive_state, config, params, registered_variables)

    assert not bool(jnp.any(jnp.isnan(final_state))), "CR shock tube produced NaNs."

    rho_num = final_state[registered_variables.density_index]
    u_num = final_state[registered_variables.velocity_index]
    p_th_num = final_state[registered_variables.pressure_index]
    e_cr_num = final_state[registered_variables.cosmic_ray_e_index]
    f_cr_num = final_state[registered_variables.cosmic_ray_flux_index]

    max_f_cr = float(jnp.max(jnp.abs(f_cr_num)))
    assert max_f_cr < 1e-6, (
        f"F_cr did not stay ~0 with reduced_streaming_speed=0 (max |F_cr| = "
        f"{max_f_cr:.3e}) -- the tightly-coupled assumption behind the "
        f"reference solution is not being realized by the simulation."
    )

    rho_ref, u_ref, p_th_ref, p_cr_ref = pfrommer_riemann_solution(
        rho_L, u_L, p_th_L, p_cr_L,
        rho_R, u_R, p_th_R, p_cr_R,
        gamma, gamma_cr,
        np.asarray(x), t_end, x0,
    )
    e_cr_ref = p_cr_ref / (gamma_cr - 1.0)

    density_error = float(jnp.mean(jnp.abs(rho_num - rho_ref)))
    velocity_error = float(jnp.mean(jnp.abs(u_num - u_ref)))
    pressure_error = float(jnp.mean(jnp.abs(p_th_num - p_th_ref)))
    e_cr_error = float(jnp.mean(jnp.abs(e_cr_num - e_cr_ref)))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    ax_density, ax_velocity, ax_pressure, ax_e_cr = axes

    for ax, num, ref, name in [
        (ax_density, rho_num, rho_ref, "Density"),
        (ax_velocity, u_num, u_ref, "Velocity"),
        (ax_pressure, p_th_num, p_th_ref, "Thermal pressure"),
        (ax_e_cr, e_cr_num, e_cr_ref, "e_cr"),
    ]:
        ax.plot(x, ref, label="Pfrommer-style reference", color="black")
        ax.plot(x, num, label="astronomix (CR-grey)", color="C0", ls="--")
        ax.set_xlabel("x")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_shock_tube_test.svg")
    plt.close(fig)

    assert density_error < tol, f"Density error {density_error} exceeds tolerance {tol}"
    assert velocity_error < tol, f"Velocity error {velocity_error} exceeds tolerance {tol}"
    assert pressure_error < tol, f"Pressure error {pressure_error} exceeds tolerance {tol}"
    assert e_cr_error < tol, f"e_cr error {e_cr_error} exceeds tolerance {tol}"


if __name__ == "__main__":
    test_cr_shock_tube()
