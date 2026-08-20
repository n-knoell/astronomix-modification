"""
Adiabatic CR compression pytest (Phase A ladder item 2).

Applies a homologous squeeze -- ``v(x, 0) = -alpha0 * (x - x_center)``, a
smooth converging velocity field, with uniform density/pressure/``e_cr`` --
to a CR-loaded gas and checks that ``e_cr`` scales as ``rho^gamma_cr``,
matching the grey closure ``P_cr = (gamma_cr - 1) * e_cr`` under adiabatic
compression (plan Sec. 4, test 2).

Why this exponent (not ``gamma_cr - 1``): for a Lagrangian fluid element
under adiabatic compression, first-law bookkeeping (``dE_cr = -P_cr dV`` on
the whole compressing region) gives
``d(e_cr)/dt = -(e_cr + P_cr) div(v) = -gamma_cr * e_cr * div(v)``, which
integrates to ``e_cr ~ rho^gamma_cr``. Reproducing this in the code requires
``e_cr`` to be advected with the gas -- the ``-P_cr div(v)`` term alone
(``cr_grey_sources.cr_adiabatic_work_source``) only supplies the ``P_cr``
piece; the missing ``-e_cr div(v)`` "volume dilution" piece is supplied by
the advective ``u_n * e_cr`` flux term in
``cr_grey_transport.grey_cr_flux_terms`` (see that function's docstring for
the full derivation -- this was a real, pre-existing gap: fixed together
with this test, since ``grey_cr_flux_terms`` previously only moved ``e_cr``
via ``F_cr``, never via the bulk gas velocity).

Robust check design: rather than compare the simulated density/e_cr against
the idealized analytic homologous solution (which assumes a perfectly
uniform interior -- broken in practice by the open boundaries' zero-gradient
approximation to the nonzero analytic edge velocity, which turns out to
perturb the density profile well before t_end, not just a thin edge layer),
this test checks the *pointwise* scaling relation ``e_cr / e_cr0 ~=
(rho / rho0)^gamma_cr`` using the simulation's own ``rho`` at each cell --
a purely local thermodynamic identity that holds regardless of the density
profile's shape. Calibrated against a 512-cell, t_end=1 run: with the
correct exponent this relation holds to <0.6% everywhere (including the
boundary-adjacent cells) and to <0.1% away from a 10% edge margin, while the
wrong exponent (``gamma_cr - 1``, i.e. without the advective fix above)
would show ~15% disagreement -- see this module's PROGRESS.md for the full
numerical verification, including the CFL fix (below) required to make this
run stable at all.

CFL note: this test exercises ``cr_pressure_gradient_source`` with
non-negligible ``e_cr`` for the first time in the ladder (item 1's CR
background was zero) and this surfaced a second real gap:
``grey_cr_transport.grey_cr_fast_speed`` didn't account for the CR-pressure
momentum coupling's own acoustic mode
(``sqrt(gamma_cr * (gamma_cr - 1) * e_cr / rho)``, from linearizing the
coupled continuity/momentum/adiabatic equations -- a real, stable wave, just
one the CFL/Riemann wave-speed bound never saw), so runs with substantial
``e_cr`` silently violated CFL and blew up to NaN. Also fixed alongside this
test -- see ``grey_cr_fast_speed``'s docstring and PROGRESS.md.

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md for the state-variable
and equation definitions this test exercises.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

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

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)


def test_cr_adiabatic_compression(edge_margin: float = 0.1, tol: float = 5e-3):
    """Homologous squeeze: check e_cr ~ rho^gamma_cr pointwise.

    Args:
        edge_margin: Fraction of the box (from each edge) excluded from the
            check, to stay clear of the open boundaries' zero-gradient
            approximation to the (nonzero) analytic edge velocity. Default
            0.1 keeps 410/512 cells; the relation holds even at margin=0
            (see module docstring), this just adds comfortable safety
            margin.
        tol: Maximum allowed relative error on the pointwise e_cr/rho^gamma_cr
            scaling. Calibrated against a 512-cell, t_end=1 run (max error
            ~1e-3 at this margin); the wrong (pre-fix) exponent gives ~0.15,
            so this tolerance leaves a wide margin while still being far
            tighter than the two exponents' separation.
    """
    num_cells = 512
    box_size = 1.0
    x_center = 0.5 * box_size
    gamma_gas = 5.0 / 3.0
    gamma_cr = 4.0 / 3.0
    reduced_streaming_speed = 0.1
    rho0 = 1.0
    p0 = 0.01
    e_cr0 = 1.0
    alpha0 = 0.2
    t_end = 1.0

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

    primitive_state = jnp.zeros((registered_variables.num_vars, num_cells))
    primitive_state = primitive_state.at[registered_variables.density_index].set(rho0)
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(p0)
    primitive_state = primitive_state.at[registered_variables.velocity_index].set(
        -alpha0 * (x - x_center)
    )
    primitive_state = primitive_state.at[registered_variables.cosmic_ray_e_index].set(
        e_cr0
    )
    # F_cr stays at its default zero -- P_cr is spatially uniform initially,
    # so nothing drives it away from zero except numerical noise.

    config = finalize_config(config, primitive_state.shape)

    params = SimulationParams(
        t_end=t_end,
        gamma=gamma_gas,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=gamma_cr, reduced_streaming_speed=reduced_streaming_speed
        ),
    )

    final_state = time_integration(primitive_state, config, params, registered_variables)

    assert not bool(jnp.any(jnp.isnan(final_state))), (
        "CR adiabatic compression produced NaNs."
    )

    rho_final = final_state[registered_variables.density_index]
    e_cr_final = final_state[registered_variables.cosmic_ray_e_index]

    interior = (x > edge_margin * box_size) & (x < (1.0 - edge_margin) * box_size)

    rho_ratio = rho_final[interior] / rho0
    e_cr_ratio = e_cr_final[interior] / e_cr0
    predicted_e_cr_ratio = rho_ratio**gamma_cr

    # Sanity check that real compression actually happened (otherwise the
    # scaling check below would be trivially near-1 ~= 1 regardless of
    # whether the source term is implemented correctly at all).
    assert float(jnp.max(rho_ratio)) > 1.05, (
        "Expected the homologous squeeze to produce >5% compression in the "
        "interior; got at most "
        f"{float(jnp.max(rho_ratio)) - 1.0:.4f} -- IC/BC setup may be broken."
    )

    rel_err = jnp.abs(e_cr_ratio / predicted_e_cr_ratio - 1.0)
    max_rel_err = float(jnp.max(rel_err))

    # Diagnostic plot. Left: the compression profiles across the whole
    # domain, with the interior region actually used by the assertion
    # shaded -- shows why comparing against the idealized (spatially
    # uniform) analytic solution directly would have been too strict (see
    # module docstring: the profile is not flat, even away from the
    # boundary). Right: the "money plot" -- the pointwise scaling relation
    # this test actually checks, with the correct (gamma_cr) and the wrong
    # pre-fix (gamma_cr - 1) power laws overlaid for contrast.
    fig, (ax_profiles, ax_scaling) = plt.subplots(1, 2, figsize=(12, 5))

    ax_profiles.plot(x, rho_final / rho0, label=r"$\rho / \rho_0$", color="C0")
    ax_profiles.plot(x, e_cr_final / e_cr0, label=r"$e_{cr} / e_{cr,0}$", color="C1")
    ax_profiles.axvspan(
        edge_margin * box_size,
        (1.0 - edge_margin) * box_size,
        color="grey",
        alpha=0.15,
        label="region used in the check",
    )
    ax_profiles.set_xlabel("x")
    ax_profiles.set_ylabel("ratio to initial value")
    ax_profiles.set_title(f"Homologous compression profiles at t = {t_end:.2f}")
    ax_profiles.legend()

    rho_curve = jnp.linspace(1.0, float(jnp.max(rho_ratio)), 200)
    ax_scaling.scatter(
        rho_ratio, e_cr_ratio, s=8, color="C0", alpha=0.6, label="simulation (interior cells)"
    )
    ax_scaling.plot(
        rho_curve,
        rho_curve**gamma_cr,
        color="black",
        lw=1.5,
        label=r"$(\rho/\rho_0)^{\gamma_{cr}}$ (correct)",
    )
    ax_scaling.plot(
        rho_curve,
        rho_curve ** (gamma_cr - 1.0),
        color="C3",
        ls="--",
        lw=1.5,
        label=r"$(\rho/\rho_0)^{\gamma_{cr}-1}$ (pre-fix, wrong)",
    )
    ax_scaling.set_xlabel(r"$\rho / \rho_0$")
    ax_scaling.set_ylabel(r"$e_{cr} / e_{cr,0}$")
    ax_scaling.set_title(f"Pointwise adiabatic invariant (max rel. err = {max_rel_err:.2e})")
    ax_scaling.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_adiabatic_compression_test.svg")
    plt.close(fig)

    assert max_rel_err < tol, (
        f"e_cr does not scale as rho^gamma_cr under adiabatic compression: "
        f"max pointwise relative error {max_rel_err:.4e} >= tol {tol}."
    )


if __name__ == "__main__":
    test_cr_adiabatic_compression()
