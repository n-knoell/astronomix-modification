"""
CR-DSA SNR expanding into a uniform vs. clumpy medium pytest (Phase B/C ladder item 11).

Runs a 3D Cartesian, non-MHD, finite-volume supernova-remnant (SNR) point
explosion -- physically scaled (``E_SN = 1e51`` erg into a warm/diffuse ISM,
``n = 1 cm^-3``, ``T = 1e4`` K, matching item 10's move from item 7's toy
code-unit Sedov setup to real astropy-unit ISM parameters) -- in four
combinations: ``{uniform, clumpy}`` ambient medium x
``{dsa_efficiency=0 (control), dsa_efficiency=0.1 (DSA)}``. CR injection
reuses items 7/8/10's shock-DSA machinery
(:func:`astronomix._modules._cosmic_rays_grey.cr_grey_injection.inject_crs_at_shocks`
via ``find_shocks_pfrommer``) at the SNR forward shock -- no new injection
code is needed, matching item 10's "new test of existing machinery" pattern.

**Clumpy medium (user-confirmed design, see PROGRESS.md's 2026-09-11 entry):**
three discrete overdense spheres (density contrast 10x ambient, tanh-tapered
edges reusing the Sedov explosion's own ``weight`` pattern generalized to
off-center positions) are placed along three non-axis-aligned directions at
a fixed distance from the explosion center -- far enough out to sit outside
the explosion's own smoothing region, close enough that the forward shock
reaches and crosses them well before ``t_end``. Only density is perturbed;
gas pressure stays uniform, so each clump starts with a *lower* local sound
speed than the ambient medium -- when the forward shock arrives, the local
sonic/DSA Mach number at a clump is higher than it is in the surrounding
uniform gas, a concrete, checkable physical signature (see Check 2 below).

**Two validation layers (both user-confirmed):**

1. **Differential CR energy-partition identity** (same identity as items
   7/9/10), applied separately to the uniform and clumpy configurations:
   since ``inject_crs_at_shocks`` conserves energy exactly (diverts thermal
   pressure into ``e_cr``, touching neither density nor velocity), the
   thermal+kinetic energy the DSA run "loses" relative to its own
   same-medium control run must equal the CR energy it gained. This check is
   medium-agnostic by construction -- it must hold whether or not the
   ambient medium has clumps in it.

2. **Clumps-matter signature**: the clumpy-DSA run's total injected
   ``E_cr`` must differ *measurably* from the uniform-DSA run's, at fixed
   ``dsa_efficiency``/``dsa_mach_min``. This is the check that actually
   exercises the clumpy geometry, not just injection bookkeeping (which
   Check 1 alone would pass even if the clumps had zero physical effect).
   **The calibrated sign of this effect is a ~18% *decrease*, not the
   naively-expected increase** -- see the calibration note below.

**Calibration note on Check 2's sign (2026-09-11):** the working hypothesis
going in was that clumps, being colder at fixed pressure (lower local sound
speed), would locally raise the Mach number the shock encounters and so
*increase* total DSA injection. The locally-elevated-Mach part of this is
confirmed by the data (the clumpy run's minimum surface Mach number is
~1.9-2.0 vs. the uniform run's ~1.35, using the same ``dsa_mach_min=1.3``
threshold in both). But the net, whole-shock-surface effect on total
injected ``E_cr`` goes the other way: the clumps' extra inertia measurably
slows the *overall* forward shock (the clumpy run's max shock radius is
~15% smaller than the uniform run's at the same ``t_end``, and it flags
~11% fewer shock-surface cells overall), so less total thermal energy is
processed through the shock surface within a fixed ``t_end`` -- and less
total energy is available to divert into CR at fixed ``dsa_efficiency``.
The net decrease in total ``E_cr`` wins over the locally-higher-Mach
increase at the clump surfaces themselves. This is a genuine, real
physical prediction of the injection model as built (not a bug) -- flagged
here because it's the opposite of the naive intuition, and worth knowing
before this scaffold's later Phase C emission work (item 15's SNR-cloud
case) interprets similar behavior.

See ``astronomix/_modules/_cosmic_rays_grey/PROGRESS.md``'s 2026-09-11
entry for exact calibrated numbers, tolerances, and the reasoning behind
this test's specific parameter choices.
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
from astronomix import CodeUnits
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
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

# ---- physical setup (calibrated 2026-09-11) ----
GAMMA = 5.0 / 3.0
NUM_CELLS = 256
DSA_MACH_MIN = 1.3
DSA_EFFICIENCY = 0.1

CODE_UNITS = CodeUnits(5 * u.parsec, 1 * u.M_sun, 100 * u.km / u.s)

E_SN = 1e51 * u.erg
N_AMBIENT = 1 / u.cm**3
T_AMBIENT = 1e4 * u.K
RHO_AMBIENT = N_AMBIENT * c.m_p
P_AMBIENT = N_AMBIENT * c.k_B * T_AMBIENT

BOX_SIZE_PHYS = 20.0 * u.pc
BOX_SIZE = BOX_SIZE_PHYS.to(CODE_UNITS.code_length).value

T_END_PHYS = 1000.0 * u.yr
T_END = T_END_PHYS.to(CODE_UNITS.code_time).value

# Explosion energy deposited as a smooth spherical pressure bump at the box
# center, tapered over SMOOTH_CELLS grid cells (same tanh-taper pattern as
# item 7's toy Sedov test, generalized to physical units).
R_EXPLOSION_PHYS = 2.0 * (BOX_SIZE_PHYS / NUM_CELLS)
SMOOTH_CELLS = 2.0

# Three discrete overdense clumps: non-axis-aligned directions (to break
# spherical symmetry), fixed distance from the explosion center, tanh-tapered
# edges, density-only perturbation (pressure stays uniform).
_CLUMP_DIRECTIONS_RAW = np.array(
    [
        [1.0, 0.3, 0.2],
        [-0.4, 1.0, -0.3],
        [0.2, -0.5, 1.0],
    ]
)
CLUMP_DIRECTIONS = _CLUMP_DIRECTIONS_RAW / np.linalg.norm(
    _CLUMP_DIRECTIONS_RAW, axis=1, keepdims=True
)
CLUMP_DISTANCE_PHYS = 3.0 * u.pc
CLUMP_RADIUS_PHYS = 1.0 * u.pc
CLUMP_SMOOTH_CELLS = 2.0
CLUMP_DENSITY_CONTRAST = 10.0


def _run_snr(dsa_efficiency: float, clumpy: bool):
    """Run one 3D CR-DSA SNR simulation; return the final state and diagnostics."""
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
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_shock_acceleration=True
        ),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (NUM_CELLS, NUM_CELLS, NUM_CELLS)
    dx = BOX_SIZE / NUM_CELLS
    cell_volume = dx**3

    rho_ambient_code = RHO_AMBIENT.to(CODE_UNITS.code_density).value
    p_ambient_code = P_AMBIENT.to(CODE_UNITS.code_pressure).value

    density = jnp.ones(shape) * rho_ambient_code

    if clumpy:
        axis = (jnp.arange(NUM_CELLS) + 0.5) * dx
        X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
        box_center = BOX_SIZE / 2.0
        clump_distance_code = CLUMP_DISTANCE_PHYS.to(CODE_UNITS.code_length).value
        clump_radius_code = CLUMP_RADIUS_PHYS.to(CODE_UNITS.code_length).value
        clump_smooth_width = CLUMP_SMOOTH_CELLS * dx
        for direction in CLUMP_DIRECTIONS:
            cx = box_center + direction[0] * clump_distance_code
            cy = box_center + direction[1] * clump_distance_code
            cz = box_center + direction[2] * clump_distance_code
            dist = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
            weight = 0.5 * (
                1.0 - jnp.tanh((dist - clump_radius_code) / clump_smooth_width)
            )
            density = density + rho_ambient_code * (CLUMP_DENSITY_CONTRAST - 1.0) * weight

    zeros = jnp.zeros(shape)

    radius = helper_data.r
    r_explosion_code = R_EXPLOSION_PHYS.to(CODE_UNITS.code_length).value
    smooth_width = SMOOTH_CELLS * dx
    explosion_weight = 0.5 * (1.0 - jnp.tanh((radius - r_explosion_code) / smooth_width))
    e_sn_code = E_SN.to(CODE_UNITS.code_energy).value
    delta_p = e_sn_code * (GAMMA - 1.0) / (jnp.sum(explosion_weight) * cell_volume)
    gas_pressure = p_ambient_code + delta_p * explosion_weight

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

    shock_result = find_shocks_pfrommer(
        final_state, config, registered_variables, helper_data, mach_min=DSA_MACH_MIN
    )
    surf = np.array(shock_result.shock_surface_cells)
    r_np = np.array(radius)
    r_shock_max = float(r_np[surf].max()) if surf.any() else float("nan")
    mach_np = np.array(shock_result.mach_numbers)
    mach_at_surface = mach_np[surf] if surf.any() else np.array([])

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
        num_shocks=int(np.array(shock_result.num_shocks)),
        mach_at_surface=mach_at_surface,
    )


def test_cr_snr_clumpy_medium(
    partition_tol: float = 1e-2,
    conservation_tol: float = 1e-3,
    clumpy_effect_min: float = 0.05,
    domain_half_width: float = BOX_SIZE / 2,
    containment_margin: float = 0.4,
):
    """CR-DSA SNR: uniform-vs-clumpy medium energy-partition and clumps-matter checks.

    Args:
        partition_tol: Max relative error on the differential CR
            energy-partition identity (Check 1, see module docstring),
            applied separately to the uniform and clumpy configurations.
            Calibrated observed error ~6.5e-4 (uniform), ~2e-4 (clumpy).
        conservation_tol: Max relative error on each run's own total-energy
            conservation (initial ambient thermal + E_SN vs. final
            thermal+kinetic+CR) -- a weaker but independent sanity check,
            identical across all four runs since ambient pressure (not
            density) sets the initial thermal energy and is uniform even in
            the clumpy configuration. Calibrated observed error ~1.5e-6.
        clumpy_effect_min: Minimum required *magnitude* of the fractional
            difference between the clumpy-DSA run's E_cr and the
            uniform-DSA run's E_cr (Check 2, the clumps-matter signature) --
            see module docstring's calibration note. Calibrated observed
            magnitude ~0.179 (a decrease), a >3.5x margin.
        domain_half_width: Half the (code-unit) box size -- the nearest open
            boundary's distance from the explosion center.
        containment_margin: Minimum gap (code units) the forward shock must
            stay within the domain in every configuration, so the energy
            checks aren't contaminated by boundary losses. Calibrated
            observed shock radii: 1.431 (uniform), 1.210 (clumpy), vs.
            domain_half_width=2.0.
    """
    runs = {
        (clumpy, dsa_eff): _run_snr(dsa_efficiency=dsa_eff, clumpy=clumpy)
        for clumpy in (False, True)
        for dsa_eff in (0.0, DSA_EFFICIENCY)
    }
    uni_ctrl = runs[(False, 0.0)]
    uni_dsa = runs[(False, DSA_EFFICIENCY)]
    clumpy_ctrl = runs[(True, 0.0)]
    clumpy_dsa = runs[(True, DSA_EFFICIENCY)]

    for name, run in (
        ("uniform control", uni_ctrl),
        ("uniform DSA", uni_dsa),
        ("clumpy control", clumpy_ctrl),
        ("clumpy DSA", clumpy_dsa),
    ):
        assert not bool(jnp.any(jnp.isnan(run["final_state"]))), (
            f"CR SNR ({name} run) produced NaNs."
        )
        assert run["r_shock_max"] < domain_half_width - containment_margin, (
            f"{name} run's forward shock (r={run['r_shock_max']:.4f}) is too "
            f"close to the open boundary (domain half-width "
            f"{domain_half_width:.4f}) for the checks below to be "
            f"meaningful -- reduce t_end or increase the domain."
        )

    # Control runs' DSA code path is live but dsa_efficiency=0 should inject
    # exactly zero CR energy in both media.
    for name, run in (("uniform control", uni_ctrl), ("clumpy control", clumpy_ctrl)):
        assert run["E_cr"] == 0.0, (
            f"{name} run (dsa_efficiency=0) injected nonzero CR energy: "
            f"E_cr={run['E_cr']:.6e}."
        )

    # Each run's own total-energy conservation. Ambient pressure (not
    # density) sets the initial thermal energy, so this is identical across
    # uniform and clumpy configurations even though the clumps carry extra
    # mass.
    total_volume = BOX_SIZE**3
    p_ambient_code = P_AMBIENT.to(CODE_UNITS.code_pressure).value
    e_sn_code = E_SN.to(CODE_UNITS.code_energy).value
    E_ambient_initial = p_ambient_code / (GAMMA - 1.0) * total_volume
    E_total_initial = E_ambient_initial + e_sn_code
    for name, run in (
        ("uniform control", uni_ctrl),
        ("uniform DSA", uni_dsa),
        ("clumpy control", clumpy_ctrl),
        ("clumpy DSA", clumpy_dsa),
    ):
        rel_err = abs(run["E_total"] - E_total_initial) / E_total_initial
        assert rel_err < conservation_tol, (
            f"{name} run's total energy is not conserved: final "
            f"{run['E_total']:.6f} vs. initial {E_total_initial:.6f} "
            f"(rel. err {rel_err:.4e} >= tol {conservation_tol})."
        )

    # --- Check 1: differential CR energy-partition identity, per medium ---
    for name, ctrl, dsa in (("uniform", uni_ctrl, uni_dsa), ("clumpy", clumpy_ctrl, clumpy_dsa)):
        thermal_kinetic_ctrl = ctrl["E_thermal"] + ctrl["E_kinetic"]
        thermal_kinetic_dsa = dsa["E_thermal"] + dsa["E_kinetic"]
        energy_diverted = thermal_kinetic_ctrl - thermal_kinetic_dsa
        partition_rel_err = abs(energy_diverted - dsa["E_cr"]) / dsa["E_cr"]
        assert partition_rel_err < partition_tol, (
            f"{name}: energy-partition identity violated -- thermal+kinetic "
            f"energy diverted from the control run ({energy_diverted:.6e}) "
            f"does not match the DSA run's CR energy ({dsa['E_cr']:.6e}) -- "
            f"rel. err {partition_rel_err:.4e} >= tol {partition_tol}."
        )

    # --- Check 2: clumps-matter signature ---
    # Calibrated sign is a *decrease* (see module docstring's calibration
    # note) -- check the magnitude of the difference, not a presumed
    # direction.
    clumpy_effect = (clumpy_dsa["E_cr"] - uni_dsa["E_cr"]) / uni_dsa["E_cr"]
    assert abs(clumpy_effect) > clumpy_effect_min, (
        f"Clumpy-DSA run's E_cr ({clumpy_dsa['E_cr']:.6e}) does not differ "
        f"measurably from the uniform-DSA run's ({uni_dsa['E_cr']:.6e}) -- "
        f"fractional difference {clumpy_effect:.4f}, magnitude <= min "
        f"{clumpy_effect_min} -- the clumpy geometry is not producing any "
        f"detectable effect on total DSA injection."
    )

    # Diagnostic plot: energy-partition bar chart (left) and a density slice
    # through the clumpy-DSA run showing the clumps and the shock (right).
    fig, (ax_bars, ax_slice) = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["thermal", "kinetic", "CR"]
    x = np.arange(len(labels))
    width = 0.2
    for i, (name, run) in enumerate(
        (("uniform, ctrl", uni_ctrl), ("uniform, DSA", uni_dsa),
         ("clumpy, ctrl", clumpy_ctrl), ("clumpy, DSA", clumpy_dsa))
    ):
        vals = [run["E_thermal"], run["E_kinetic"], run["E_cr"]]
        ax_bars.bar(x + (i - 1.5) * width, vals, width, label=name)
    ax_bars.set_xticks(x, labels)
    ax_bars.set_ylabel("energy")
    ax_bars.set_title(f"Energy partition at t={T_END_PHYS}")
    ax_bars.legend(fontsize=7)

    mid = NUM_CELLS // 2
    rho_slice = np.array(clumpy_dsa["rho"])[:, :, mid]
    im = ax_slice.imshow(rho_slice.T, origin="lower", extent=[0, BOX_SIZE, 0, BOX_SIZE])
    fig.colorbar(im, ax=ax_slice, label=r"$\rho$ (code units)")
    ax_slice.set_title("Clumpy-DSA run: density slice (z=mid)")
    ax_slice.set_xlabel("x")
    ax_slice.set_ylabel("y")

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_snr_clumpy_medium_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_cr_snr_clumpy_medium()
