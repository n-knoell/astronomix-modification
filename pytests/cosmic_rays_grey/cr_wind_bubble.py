"""
CR wind-blown-bubble pytest (Phase B/C ladder item 10).

Runs a 3D Cartesian, non-MHD, finite-volume stellar-wind bubble (single
source at the box center, thermal-energy-injection ``EI`` scheme, blown into
a homogeneous ambient ISM) twice, identical except for the diffusive-shock-
acceleration (DSA) efficiency: ``dsa_efficiency = 0`` (control -- DSA code
path live, but injecting nothing) and ``dsa_efficiency = 0.1``. CR injection
reuses items 7/8's shock-DSA machinery
(:func:`astronomix._modules._cosmic_rays_grey.cr_grey_injection.inject_crs_at_shocks`
via ``find_shocks_pfrommer``) at the wind's forward shock -- no dedicated
"wind CR source term" is built here (a real design decision; see the
module's PROGRESS.md 2026-09-10 entry for the discussion and the "verify
against the analytic Weaver solution first, since it had never been tested
against this repo's wind module at all" scope agreed with the user).

**Two independent validation layers, matching the user-chosen "Weaver
baseline + differential CR check" scope:**

1. **Weaver et al. (1977) baseline (control run only).** The plain-hydro
   Weaver self-similar wind-bubble solution
   (``astronomix._modules._stellar_wind.weaver.Weaver``) already existed in
   this repo but was never exercised by any pytest -- a real, previously
   untested gap this closes. The control run's forward (outer) shock radius,
   found via the same ``find_shocks_pfrommer`` used for DSA injection
   (the outermost detected shock-surface cell), matches Weaver's ``R_2(t)``
   to <1% at the calibrated resolution/setup below. The shocked-wind
   interior pressure plateau (averaged over a radial shell strictly between
   the inner and critical radii, away from both shocks) matches Weaver's
   analytic interior-pressure formula to ~19%. The inner (wind-termination)
   shock is deliberately **not** checked quantitatively against Weaver's
   ``R_1(t)`` -- confirmed (by comparing the detected inner shock's radius
   to the injection region's own edge, ``num_injection_cells * grid_spacing``)
   that at this resolution the shock finder's innermost detection sits right
   at the wind source term's injection-region boundary, not the physically
   resolved termination shock; this is a resolution limitation of a very
   small (``num_injection_cells=4``) source region, not a bug.

2. **Differential CR energy-partition identity (both runs), same identity
   as item 7's Sedov-Taylor test:** since ``inject_crs_at_shocks`` conserves
   energy exactly (diverts thermal pressure into ``e_cr``, touching neither
   density nor velocity), the thermal+kinetic energy the DSA run "loses"
   relative to the control run must equal the CR energy it gained. Calibrated:
   holds to a relative error of ~2.4e-4 (dsa_efficiency=0.1, dsa_mach_min=1.3,
   forward-shock Mach ~2.9). ``E_cr`` is ~1.6% of the DSA run's own realized
   total (thermal+kinetic+CR) energy -- clearly nonzero, physically bounded,
   and lower than item 7's Sedov blast (~5.3%) as expected for a much weaker
   (Mach ~3 vs. Mach ~3-29) shock.

**A real, pre-existing finding surfaced while calibrating, not fixed here
(out of scope -- unrelated to CR-grey):** the wind module's 3D thermal-
energy-injection scheme (``_wind_ei3D``) normalizes its injected power by
the *nominal* spherical injection volume
(``4/3 * pi * (num_injection_cells * grid_spacing)**3``), but the actual
per-cell injection mask uses a conservative radius shrunk by half a grid
cell (``dist <= injection_radius - grid_spacing / 2``). At small
``num_injection_cells`` this is a large fractional volume mismatch --
verified directly: at ``num_injection_cells=4`` the shrunk-vs-nominal volume
ratio is ``(3.5/4)**3 ~= 0.67``, and the measured total injected energy
(final thermal+kinetic+CR minus initial ambient thermal, across ~1e4 yr) is
``~66%`` of the nominal ``L_w * t_end`` wind luminosity -- matching this
predicted ratio almost exactly. Because of this, this test does **not**
assert an absolute energy budget against the nominal ``L_w * t_end``
formula (it would fail by a wide, resolution-dependent margin that has
nothing to do with CR-grey correctness). Instead it uses two budget checks
that are insensitive to this bias: (a) the differential CR-partition
identity above, which only compares the two runs against each other, and
(b) a tight self-consistency check that the control and DSA runs' own
*realized* total energy (thermal+kinetic+CR) agree with each other (both
runs share the same, biased, injected-energy budget -- ``dsa_efficiency``
only redirects a couple of percent of it into ``e_cr``), calibrated to
agree to ~4e-6 relative error.
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

# units
from astropy import units as u
import astropy.constants as c

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import CodeUnits, WindParams
from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes import WindConfig

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
from astronomix._modules._stellar_wind.weaver import Weaver
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

# ---- physical setup (calibrated 2026-09-10) ----
GAMMA = 5.0 / 3.0
NUM_CELLS = 128
BOX_SIZE = 1.2  # code_units.code_length units (-> 3.6 pc physical box)
NUM_INJECTION_CELLS = 4
DSA_MACH_MIN = 1.3
DSA_EFFICIENCY = 0.1

CODE_UNITS = CodeUnits(3 * u.parsec, 1 * u.M_sun, 100 * u.km / u.s)

M_STAR = 40 * u.M_sun
V_INF = 2000 * u.km / u.s
M_DOT = 2.965e-3 / (1e6 * u.yr) * M_STAR

RHO_AMBIENT = 2 * c.m_p / u.cm**3
P_AMBIENT = 3e4 * u.K / u.cm**3 * c.k_B

T_END_PHYS = 1e4 * u.yr
T_END = T_END_PHYS.to(CODE_UNITS.code_time).value


def _run_wind_bubble(dsa_efficiency: float):
    """Run one 3D CR-DSA wind-bubble simulation; return final state and diagnostics."""
    config = SimulationConfig(
        progress_bar=True,
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=NUM_CELLS,
        exact_end_time=True,
        wind_config=WindConfig(
            stellar_wind=True,
            num_injection_cells=NUM_INJECTION_CELLS,
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_shock_acceleration=True
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (NUM_CELLS, NUM_CELLS, NUM_CELLS)
    density = jnp.ones(shape) * RHO_AMBIENT.to(CODE_UNITS.code_density).value
    zeros = jnp.zeros(shape)
    gas_pressure = jnp.ones(shape) * P_AMBIENT.to(CODE_UNITS.code_pressure).value

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

    wind_params = WindParams(
        wind_mass_loss_rates=jnp.array(
            [M_DOT.to(CODE_UNITS.code_mass / CODE_UNITS.code_time).value]
        ),
        wind_final_velocities=jnp.array(
            [V_INF.to(CODE_UNITS.code_velocity).value]
        ),
    )

    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        wind_params=wind_params,
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
    radius = helper_data.r

    cell_volume = (BOX_SIZE / NUM_CELLS) ** 3
    e_thermal_density = p_gas / (GAMMA - 1.0)
    e_kinetic_density = 0.5 * rho * (vx**2 + vy**2 + vz**2)

    E_thermal = float(jnp.sum(e_thermal_density) * cell_volume)
    E_kinetic = float(jnp.sum(e_kinetic_density) * cell_volume)
    E_cr = float(jnp.sum(e_cr) * cell_volume)

    shock_result = find_shocks_pfrommer(
        final_state, config, registered_variables, helper_data, mach_min=DSA_MACH_MIN
    )
    surf = np.array(shock_result.shock_surface_cells)
    r_np = np.array(radius)
    r_shock_max = float(r_np[surf].max()) if surf.any() else float("nan")

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
        r_shock_max=r_shock_max,
    )


def test_cr_wind_bubble(
    weaver_shock_radius_tol: float = 0.05,
    weaver_interior_pressure_tol: float = 0.35,
    partition_tol: float = 1e-2,
    self_consistency_tol: float = 1e-3,
    domain_half_width: float = BOX_SIZE / 2,
    containment_margin: float = 0.1,
):
    """CR wind-blown bubble: Weaver baseline + differential CR energy check.

    Args:
        weaver_shock_radius_tol: Max relative error on the control run's
            forward-shock radius (from ``find_shocks_pfrommer``) vs. Weaver's
            analytic ``R_2(t)``. Calibrated observed error ~0.4%.
        weaver_interior_pressure_tol: Max relative error on the control run's
            shocked-wind interior pressure plateau vs. Weaver's analytic
            value. Calibrated observed error ~19%; a looser tolerance than
            the shock-radius check since this compares a resolution-sensitive
            volume average, not a single well-defined front position.
        partition_tol: Max relative error on the differential CR
            energy-partition identity (see module docstring). Calibrated
            observed error ~2.4e-4.
        self_consistency_tol: Max relative error between the control and DSA
            runs' own realized total (thermal+kinetic+CR) energy -- both draw
            from the same (systematically biased, see module docstring)
            injected energy budget, so these must closely agree regardless of
            that bias. Calibrated observed error ~4e-6.
        domain_half_width: Half the box size -- the nearest open boundary's
            distance from the wind source.
        containment_margin: Minimum gap the forward shock must stay within
            the domain, so the energy-budget checks aren't contaminated by
            energy having already left through the open boundaries.
    """
    control = _run_wind_bubble(dsa_efficiency=0.0)
    dsa = _run_wind_bubble(dsa_efficiency=DSA_EFFICIENCY)

    for name, run in (("control", control), ("dsa", dsa)):
        assert not bool(jnp.any(jnp.isnan(run["final_state"]))), (
            f"CR wind bubble ({name} run) produced NaNs."
        )

    # Containment: the forward shock must stay well inside the open
    # boundaries, or the checks below would be contaminated by boundary
    # losses.
    assert control["r_shock_max"] < domain_half_width - containment_margin, (
        f"Forward shock (r={control['r_shock_max']:.4f}) is too close to the "
        f"open boundary (domain half-width {domain_half_width}) for the "
        f"checks below to be meaningful -- reduce t_end or increase the "
        f"domain."
    )

    # Control run's DSA code path is live (diffusive_shock_acceleration=True)
    # but dsa_efficiency=0 should inject exactly zero CR energy.
    assert control["E_cr"] == 0.0, (
        f"Control run (dsa_efficiency=0) injected nonzero CR energy: "
        f"E_cr={control['E_cr']:.6e}."
    )

    # --- Weaver baseline (control run) ---
    weaver = Weaver(V_INF, M_DOT, RHO_AMBIENT, P_AMBIENT, gamma=GAMMA)
    code_length_pc = CODE_UNITS.code_length.to(u.pc)

    r_shock_weaver = weaver.get_outer_shock_radius(T_END_PHYS).to(u.pc).value
    r_shock_sim = control["r_shock_max"] * code_length_pc
    shock_rel_err = abs(r_shock_sim - r_shock_weaver) / r_shock_weaver
    assert shock_rel_err < weaver_shock_radius_tol, (
        f"Control run's forward-shock radius (r={r_shock_sim:.4f} pc) does "
        f"not match Weaver's R_2(t)={r_shock_weaver:.4f} pc -- rel. err "
        f"{shock_rel_err:.4e} >= tol {weaver_shock_radius_tol}."
    )

    r1_weaver = weaver.get_inner_shock_radius(T_END_PHYS).to(u.pc).value
    r_c_weaver = weaver.get_critical_radius(T_END_PHYS).to(u.pc).value
    r_np_pc = np.array(control["radius"]) * code_length_pc
    p_np = np.array(control["p_gas"])
    interior_mask = (r_np_pc > 1.3 * r1_weaver) & (r_np_pc < 0.9 * r_c_weaver)
    assert interior_mask.any(), "No cells found in the shocked-wind interior shell."
    p_interior_sim = (
        float(p_np[interior_mask].mean())
        * CODE_UNITS.code_pressure.to(u.erg / u.cm**3)
    ) * (u.erg / u.cm**3)
    p_interior_weaver = (
        5
        / (22 * np.pi * (weaver.xi_crit * weaver.alpha) ** 3)
        * (weaver.L_w**2 * weaver.rho_0**3) ** (1 / 5)
        * T_END_PHYS ** (-4 / 5)
    ).to(u.erg / u.cm**3)
    interior_rel_err = float(
        abs(p_interior_sim - p_interior_weaver) / p_interior_weaver
    )
    assert interior_rel_err < weaver_interior_pressure_tol, (
        f"Control run's shocked-wind interior pressure ({p_interior_sim:.4e}) "
        f"does not match Weaver's analytic value ({p_interior_weaver:.4e}) -- "
        f"rel. err {interior_rel_err:.4e} >= tol {weaver_interior_pressure_tol}."
    )

    # --- Differential CR checks (both runs) ---
    self_consistency_rel_err = abs(control["E_total"] - dsa["E_total"]) / control["E_total"]
    assert self_consistency_rel_err < self_consistency_tol, (
        f"Control and DSA runs' realized total energy disagree: "
        f"control={control['E_total']:.6e}, dsa={dsa['E_total']:.6e} -- rel. "
        f"err {self_consistency_rel_err:.4e} >= tol {self_consistency_tol}."
    )

    cr_fraction = dsa["E_cr"] / dsa["E_total"]
    assert 0.001 < cr_fraction < 0.2, (
        f"DSA run's CR energy fraction ({cr_fraction:.4f}) is outside the "
        f"expected (0.001, 0.2) band for dsa_efficiency={DSA_EFFICIENCY} -- "
        f"either injection is not happening, or is wildly over-injecting."
    )

    thermal_kinetic_control = control["E_thermal"] + control["E_kinetic"]
    thermal_kinetic_dsa = dsa["E_thermal"] + dsa["E_kinetic"]
    energy_diverted = thermal_kinetic_control - thermal_kinetic_dsa
    partition_rel_err = abs(energy_diverted - dsa["E_cr"]) / dsa["E_cr"]
    assert partition_rel_err < partition_tol, (
        f"Energy-partition identity violated: thermal+kinetic energy "
        f"diverted from the control run ({energy_diverted:.6e}) does not "
        f"match the DSA run's CR energy ({dsa['E_cr']:.6e}) -- rel. err "
        f"{partition_rel_err:.4e} >= tol {partition_tol}."
    )

    # Diagnostic plot: energy-partition bar chart (left) and radial
    # density/pressure profile at t_end vs. the Weaver solution (right).
    fig, (ax_bars, ax_profile) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["thermal", "kinetic", "CR"]
    control_vals = [control["E_thermal"], control["E_kinetic"], control["E_cr"]]
    dsa_vals = [dsa["E_thermal"], dsa["E_kinetic"], dsa["E_cr"]]
    x = np.arange(len(labels))
    width = 0.35
    ax_bars.bar(x - width / 2, control_vals, width, label="control (efficiency=0)", color="C0")
    ax_bars.bar(x + width / 2, dsa_vals, width, label=f"DSA (efficiency={DSA_EFFICIENCY})", color="C1")
    ax_bars.set_xticks(x, labels)
    ax_bars.set_ylabel("energy")
    ax_bars.set_title(f"Energy partition at t={T_END_PHYS}")
    ax_bars.legend()

    r_sim_pc = np.array(dsa["radius"]).reshape(-1) * code_length_pc
    p_sim = np.array(dsa["p_gas"]).reshape(-1)
    p_cr_sim = np.array(dsa["e_cr"]).reshape(-1) * (4.0 / 3.0 - 1.0)
    order = np.argsort(r_sim_pc)
    ax_profile.plot(
        r_sim_pc[order],
        p_sim[order] * CODE_UNITS.code_pressure.to(u.erg / u.cm**3),
        ".", ms=1, color="C0", label=r"$P_{\rm gas}$ (sim)", rasterized=True,
    )
    ax_profile.plot(
        r_sim_pc[order],
        p_cr_sim[order] * CODE_UNITS.code_pressure.to(u.erg / u.cm**3),
        ".", ms=1, color="C1", label=r"$P_{\rm cr}$ (sim)", rasterized=True,
    )
    delta_R = 0.02 * u.pc
    r_w, p_w = weaver.get_pressure_profile(delta_R, domain_half_width * code_length_pc * u.pc, T_END_PHYS)
    ax_profile.plot(
        r_w.to(u.pc).value, p_w.to(u.erg / u.cm**3).value,
        "--", color="grey", label="Weaver",
    )
    ax_profile.axvline(r_shock_sim, color="grey", ls=":", lw=1, label=f"r_shock={r_shock_sim:.3f} pc")
    ax_profile.set_yscale("log")
    # Floor: e_cr's positivity floor (far from any shock) is orders of
    # magnitude below any physically meaningful pressure here -- clip the
    # view so it doesn't dominate the log-scale range.
    p_ambient_phys = P_AMBIENT.to(u.erg / u.cm**3).value
    ax_profile.set_ylim(bottom=p_ambient_phys * 1e-3)
    ax_profile.set_xlabel("r [pc]")
    ax_profile.set_ylabel(r"pressure [erg cm$^{-3}$]")
    ax_profile.set_title("DSA run: radial pressure profile")
    ax_profile.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_wind_bubble_test.svg", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    test_cr_wind_bubble()
