"""
CR-DSA Sedov-Taylor blast pytest (Phase B ladder item 7).

Runs a 3D Cartesian point explosion (matching
``pytests/shock_finder3D/_sedov_setup.py``'s physical setup) twice, identical
except for the diffusive-shock-acceleration (DSA) efficiency:
``dsa_efficiency = 0`` (control -- DSA code path live, but injecting nothing)
and ``dsa_efficiency = 0.1``. Checks the thermal/CR/kinetic energy partition
(plan Sec. 4, test 7) via
:func:`astronomix._modules._cosmic_rays_grey.cr_grey_injection.inject_crs_at_shocks`,
which the previous session wired up fresh against the real PR #4 finder
(``astronomix.shock_finder3D.pfrommer_shock_finder.find_shocks_pfrommer``)
after the old single-scalar ``_cosmic_rays`` model (with its own, legacy
shock-finder-based injection) was retired -- see this module's DESIGN.md,
"Consolidating with the old ``_cosmic_rays`` model".

Test design: rather than compare against an analytic self-similar Sedov
profile (a genuine global self-similar solution that, like ladder item 2's
homologous-compression test, would need careful late-time/open-boundary
matching this module has previously found unreliable -- see
``cr_adiabatic_compression.py``'s docstring), this test uses a *differential*
energy-conservation identity between the two runs: since
``inject_crs_at_shocks`` is built to conserve energy exactly (it adds
``delta_e_cr`` to ``e_cr`` and removes the identical energy from gas thermal
pressure, touching neither density nor velocity -- see that function's
docstring), the total energy the DSA run "loses" from thermal+kinetic
relative to the control run must equal the CR energy it gained:

    (E_thermal_control + E_kinetic_control) - (E_thermal_dsa + E_kinetic_dsa)
        ~= E_cr_dsa

Calibrated at N=48, t_end=0.07 (chosen so the shock stays well inside the
open-boundary domain -- r_shock ~= 0.455 vs. domain half-width 0.5, checked
below), with the default adaptive shock sampling
(``CosmicRayGreyConfig.dsa_adaptive_shock_sampling=True``, re-run
2026-10-02): this identity holds to a relative error of ~1.8e-5, and
``E_cr_dsa / E_total ~= 6.1%`` of the total (initial) energy budget, a
clearly nonzero and physically bounded signal at ``dsa_efficiency = 0.1``,
``dsa_mach_min = 1.3``. Each run's own total-energy conservation (initial
ambient + E_EXPLOSION vs. final thermal+kinetic+CR) independently holds to
~1.6e-5 (control) / ~1.7e-5 (DSA). (Old fixed 1-cell sampling: identity
~2.1e-5, conservation ~1.7e-5.)

**Re-measured 2026-10-10 with the total-pressure HLLC flux** (P_cr in the HLLC flux, pdv work from
the face velocities; DESIGN.md "Resolved: total-pressure HLLC flux"; values of 2026-10-09 with the
stage-wise source coupling in brackets):
``E_cr / E_total`` 0.0401 [0.0434], energy-partition identity 1.0e-6 [3.0e-6], total-energy
conservation 1.3e-5 (control) / 1.3e-5 (DSA) [same].
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import os
from pathlib import Path

# jax
import jax.numpy as jnp

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

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# ---- physical setup (matches pytests/shock_finder3D/_sedov_setup.py) ----
GAMMA = 5.0 / 3.0
# grid cells per axis; override with the CR_NUM_CELLS environment variable for
# resolution studies (tolerances below are calibrated at 48)
NUM_CELLS = int(os.environ.get("CR_NUM_CELLS", 48))
T_END = 0.07
E_EXPLOSION = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1e-4
R_EXPLOSION = 0.05
SMOOTH_CELLS = 2.0
DSA_MACH_MIN = 1.3
DSA_EFFICIENCY = 0.1


def _run_sedov(dsa_efficiency: float):
    """Run one 3D CR-DSA Sedov blast; return the final state and diagnostics."""
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        exact_end_time=True,
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_shock_acceleration=True
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (NUM_CELLS, NUM_CELLS, NUM_CELLS)
    density = jnp.ones(shape) * RHO_AMBIENT
    zeros = jnp.zeros(shape)

    # Smoothly-tapered spherical injection weight in [0, 1], renormalised so
    # the deposited thermal energy above ambient is exactly E_EXPLOSION.
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
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            dsa_efficiency=dsa_efficiency, dsa_mach_min=DSA_MACH_MIN
        ),
    )

    final_state = time_integration(initial_state, config, params, registered_variables)

    rho = final_state[registered_variables.density_index]
    vx = final_state[registered_variables.velocity_index.x]
    vy = final_state[registered_variables.velocity_index.y]
    vz = final_state[registered_variables.velocity_index.z]
    p_gas = final_state[registered_variables.pressure_index]
    e_cr = final_state[registered_variables.cosmic_ray_e_index]

    e_thermal_density = p_gas / (GAMMA - 1.0)
    e_kinetic_density = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    E_thermal = float(jnp.sum(e_thermal_density) * cell_volume)
    E_kinetic = float(jnp.sum(e_kinetic_density) * cell_volume)
    E_cr = float(jnp.sum(e_cr) * cell_volume)

    return dict(
        final_state=final_state,
        radius=radius,
        rho=rho,
        p_gas=p_gas,
        e_cr=e_cr,
        E_thermal=E_thermal,
        E_kinetic=E_kinetic,
        E_cr=E_cr,
        E_total=E_thermal + E_kinetic + E_cr,
    )


def test_cr_sedov_taylor(
    energy_partition_tol: float = 1e-2,
    conservation_tol: float = 1e-3,
    domain_half_width: float = 0.5,
    containment_margin: float = 0.03,
):
    """CR-DSA Sedov blast: check the thermal/CR/kinetic energy partition.

    Args:
        energy_partition_tol: Maximum allowed relative error on the
            control-vs-DSA energy-partition identity described in the module
            docstring. Calibrated observed error ~1.8e-5 at NUM_CELLS=48; this
            leaves a >450x margin.
        conservation_tol: Maximum allowed relative error on each run's own
            total-energy conservation (initial ambient + E_EXPLOSION vs.
            final thermal+kinetic+CR), a weaker but independent sanity check.
            Calibrated observed error ~2e-5.
        domain_half_width: Half the (unit, default) box size -- the nearest
            open boundary's distance from the explosion centre.
        containment_margin: Minimum gap (in the same units) the shock front
            must stay within the domain, so the energy-conservation checks
            above aren't silently passing because energy already left
            through the open boundaries.
    """
    control = _run_sedov(dsa_efficiency=0.0)
    dsa = _run_sedov(dsa_efficiency=DSA_EFFICIENCY)

    for name, run in (("control", control), ("dsa", dsa)):
        assert not bool(jnp.any(jnp.isnan(run["final_state"]))), (
            f"CR Sedov-Taylor ({name} run) produced NaNs."
        )

    # Containment: the shock front (defined as the outermost cell where
    # density has departed >1% from ambient) must stay well inside the open
    # boundaries, or the energy-conservation checks below would be
    # vacuous/misleading (energy having already left the domain).
    rho_flat = control["rho"].reshape(-1)
    r_flat = control["radius"].reshape(-1)
    shocked = jnp.abs(rho_flat / RHO_AMBIENT - 1.0) > 0.01
    r_shock = float(jnp.max(jnp.where(shocked, r_flat, 0.0)))
    assert r_shock < domain_half_width - containment_margin, (
        f"Shock front (r={r_shock:.4f}) is too close to the open boundary "
        f"(domain half-width {domain_half_width}) for the energy-budget "
        f"checks below to be meaningful -- reduce t_end or increase the "
        f"domain."
    )

    # Control run's DSA code path is live (diffusive_shock_acceleration=True)
    # but dsa_efficiency=0 should inject exactly zero CR energy -- confirms
    # the efficiency knob is a genuine off switch, not just "never called".
    # Since c740cc3 the e_cr floor (CosmicRayGreyParams.minimum_e_cr) is applied
    # every step, so a run without injection holds floor-level e_cr (up to ~10x
    # the floor where the shock compresses it) instead of exactly 0. Real
    # injection is ~1e9x larger.
    max_e_cr = float(jnp.max(control["e_cr"]))
    assert max_e_cr <= 100.0 * CosmicRayGreyParams().minimum_e_cr, (
        f"Control run (dsa_efficiency=0) injected CR energy: max e_cr "
        f"{max_e_cr:.3e} is above floor level (E_cr={control['E_cr']:.6e})."
    )

    # Each run's own total-energy conservation (initial ambient thermal
    # energy + the injected E_EXPLOSION, vs. final thermal+kinetic+CR).
    total_volume = 1.0**3  # box_size = 1.0 (default)
    E_ambient_initial = P_AMBIENT / (GAMMA - 1.0) * total_volume
    E_total_initial = E_ambient_initial + E_EXPLOSION
    for name, run in (("control", control), ("dsa", dsa)):
        rel_err = abs(run["E_total"] - E_total_initial) / E_total_initial
        print(f"{name}: total-energy rel. err {rel_err:.3e}")
        assert rel_err < conservation_tol, (
            f"{name} run's total energy is not conserved: final "
            f"{run['E_total']:.6f} vs. initial {E_total_initial:.6f} "
            f"(rel. err {rel_err:.4e} >= tol {conservation_tol})."
        )

    # E_cr should be a clearly nonzero but bounded fraction of the total
    # energy budget at dsa_efficiency=0.1, dsa_mach_min=1.3 (calibrated
    # ~6.1% -- see module docstring; 18.7% after fix step 4 of 2026-10-04,
    # which was spurious injection; 4.3% at N = 48 with the DSA velocity
    # check of 2026-10-07 -- see cr_dsa_mach_dependence.py; 4.0% with the
    # total-pressure HLLC flux of 2026-10-10).
    cr_fraction = dsa["E_cr"] / E_total_initial
    print(f"E_cr / E_total = {cr_fraction:.4f}")
    assert 0.01 < cr_fraction < 0.3, (
        f"DSA run's CR energy fraction ({cr_fraction:.4f}) is outside the "
        f"expected [0.01, 0.3] band for dsa_efficiency={DSA_EFFICIENCY} -- "
        f"either injection is not happening, or is wildly over-injecting."
    )

    # The main check: the differential energy-partition identity (see module
    # docstring) -- energy lost from thermal+kinetic in the DSA run relative
    # to the control run must equal the CR energy gained.
    thermal_kinetic_control = control["E_thermal"] + control["E_kinetic"]
    thermal_kinetic_dsa = dsa["E_thermal"] + dsa["E_kinetic"]
    energy_diverted = thermal_kinetic_control - thermal_kinetic_dsa
    partition_rel_err = abs(energy_diverted - dsa["E_cr"]) / dsa["E_cr"]
    print(f"energy-partition identity rel. err {partition_rel_err:.3e}")
    assert partition_rel_err < energy_partition_tol, (
        f"Energy-partition identity violated: thermal+kinetic energy "
        f"diverted from the control run ({energy_diverted:.6f}) does not "
        f"match the DSA run's CR energy ({dsa['E_cr']:.6f}) -- rel. err "
        f"{partition_rel_err:.4e} >= tol {energy_partition_tol}."
    )

    # Diagnostic plot: energy-partition bar chart (left) and radial e_cr /
    # gas-pressure profile at t_end for the DSA run (right).
    fig, (ax_bars, ax_profile) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["thermal", "kinetic", "CR"]
    control_vals = [control["E_thermal"], control["E_kinetic"], control["E_cr"]]
    dsa_vals = [dsa["E_thermal"], dsa["E_kinetic"], dsa["E_cr"]]
    x = jnp.arange(len(labels))
    width = 0.35
    ax_bars.bar(x - width / 2, control_vals, width, label="control (efficiency=0)", color="C0")
    ax_bars.bar(x + width / 2, dsa_vals, width, label=f"DSA (efficiency={DSA_EFFICIENCY})", color="C1")
    ax_bars.set_xticks(x, labels)
    ax_bars.set_ylabel("energy")
    ax_bars.set_title(f"Energy partition at t={T_END}")
    ax_bars.legend()

    r_flat_dsa = dsa["radius"].reshape(-1)
    order = jnp.argsort(r_flat_dsa)
    ax_profile.plot(
        r_flat_dsa[order], dsa["p_gas"].reshape(-1)[order], ".", ms=1, color="C0",
        label=r"$P_{\rm gas}$",
    )
    ax_profile.plot(
        r_flat_dsa[order],
        ((dsa["e_cr"].reshape(-1)[order]) * (4.0 / 3.0 - 1.0)),
        ".", ms=1, color="C1", label=r"$P_{\rm cr}$",
    )
    ax_profile.axvline(r_shock, color="grey", ls="--", lw=1, label=f"r_shock={r_shock:.3f}")
    ax_profile.set_xlabel("r")
    ax_profile.set_ylabel("pressure")
    ax_profile.set_title("DSA run: radial pressure profile")
    ax_profile.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics" / "07_sedov_taylor"
    pics_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if NUM_CELLS == 48 else f"_{NUM_CELLS}"
    fig.savefig(pics_dir / f"cr_sedov_taylor_test{suffix}.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_cr_sedov_taylor()
