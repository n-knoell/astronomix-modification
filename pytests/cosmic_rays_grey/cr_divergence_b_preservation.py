"""
Div-B preservation pytest (plan Sec. 4, item 19: "∇·B preservation
unaffected by the CR module (both schemes)").

**Scope finding, not new -- already documented, just not yet checked
against: CR-grey has no finite-difference implementation at all.**
`registered_variables.py`'s `FINITE_DIFFERENCE` branch explicitly does not
allocate `cosmic_ray_e_index`/`cosmic_ray_flux_index` (confirmed by grep: no
file under `astronomix/_finite_difference/` mentions cosmic rays), and
`DESIGN.md`'s "FD limitation" section already records why (FD's WENO
reconstruction does characteristic decomposition against a hardcoded
eigensystem with no hook for extra registered scalars -- "real, separate
follow-up work -- not attempted in this scaffold"). So "both schemes" is
only half-testable today: there is no FD+CR combination for the CR module
to possibly disturb `div(B)` in. This test checks FV (the only scheme
where CR-grey and MHD can coexist) for the actual "unaffected by the CR
module" claim, and separately checks FD's own (CR-absent) `div(B)`
baseline just to have a clean reference for whenever the FD follow-up
lands.

**Setup**: the same validated, genuinely dynamic 3D circularly polarized
Alfven wave used by the MHD-only energy baseline
(`pytests/mhd/mhd_energy_conservation.py`) -- periodic, divergence-free IC
to machine precision, real evolving oblique B. A small localized `e_cr`
pulse (amplitude `0.01`, tiny relative to the wave's own energy budget) is
superposed for the CR-on configurations, reusing ladder item 3's own
technique of giving `anisotropic_transport` a real, nonzero, genuinely
time-varying field to project against -- item 3's own setup only ever used
a *static*, uniform B (`v=0` throughout), so this is the first time
anisotropic CR transport is exercised against a real evolving field, not a
frozen one.

**Why this isn't a vacuous check: ladder item 3 already found and fixed a
real bug in exactly this area** (`evolve_state._evolve_state_fv`'s Strang
split originally sliced `primitive_state[-3:]` as "the magnetic field",
which silently grabbed CR rows registered after it and mislabelled real
`B_z` as gas whenever both `mhd` and `grey_cosmic_rays` were on -- fixed via
`_split_gas_and_magnetic_state`/`_join_gas_and_magnetic_state`, which locate
magnetic rows by their actual registered indices). That fix was verified
against CR *transport shape*, never directly against `div(B)` itself -- this
test closes that gap.

**Result: clean pass, `div(B)` unaffected by the CR module.** `N=8`,
`t_end=5.0` (5 full periods), float64, `max(|div(B)|)` via the same discrete
operator the (previously unexercised -- no existing pytest uses it)
`magnetic_divergence` snapshot diagnostic calls
(`divergence3D`/`_interface_field_divergence`,
`astronomix/_finite_volume/_magnetic_update/_vector_maths.py` /
`astronomix/_spatial_operators/_differencing.py`):

    FV, CR off:                          6.8e-15
    FV, CR on (isotropic):                8.4e-15
    FV, CR on + anisotropic_transport:    9.5e-15
    FD baseline (CR absent, for reference): 5.4e-14

All four sit at genuine floating-point round-off, not a resolution-limited
residual -- CR-grey does not measurably perturb `div(B)`, with or without
anisotropic transport actively reading a real, time-varying field, matching
the physical expectation that no CR-grey source/transport term ever writes
to a magnetic-field row.

**Minor, unrelated documentation mismatch noticed in passing, not fixed
here (low stakes, no correctness impact):** `SnapshotData.magnetic_divergence`'s
own docstring says "Mean absolute magnetic field divergence", but
`_compute_magnetic_divergence` (`_snapshotting/_snapshot_diagnostics.py`)
actually computes the *max* absolute divergence, not the mean. Arguably the
max is the more appropriate diagnostic (a mean could mask a localized
violation), so this is flagged as a stale docstring, not treated as a bug
to fix.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix containers
from astronomix import FINITE_DIFFERENCE, FINITE_VOLUME, SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import StaticIntVector

# astronomix functions
from astronomix.test_setups.mhd.alfven_wave3D import (
    CPAlfvenWave3DSettings,
    setup_cp_alfven_wave,
)
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix._finite_volume._magnetic_update._vector_maths import divergence3D
from astronomix._spatial_operators._differencing import _interface_field_divergence

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# Round-off-level comparisons need 64-bit precision (same rationale as
# mhd_energy_conservation.py / cr_gradient_check.py).
jax.config.update("jax_enable_x64", True)

N = 8
T_END = 5.0
PULSE_AMPLITUDE = 0.01
PULSE_SIGMA = 0.3  # in box-length units


def _run_fv(grey_cosmic_rays: bool, anisotropic_transport: bool):
    """Run the CP Alfven wave (FV) with the given CR-grey config; return
    (initial_state, final_state, config, registered_variables)."""
    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        num_cells=StaticIntVector(2 * N, N, N),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=grey_cosmic_rays, anisotropic_transport=anisotropic_transport
        ),
    )
    params = SimulationParams(C_cfl=0.4)
    settings = CPAlfvenWave3DSettings(t_end=T_END)

    initial_state, config, params = setup_cp_alfven_wave(config, params, settings)
    registered_variables = get_registered_variables(config)

    if grey_cosmic_rays:
        helper_data = get_helper_data(config)
        box_center = config.box_size.x / 2, config.box_size.y / 2, config.box_size.z / 2
        x, y, z = (
            helper_data.geometric_centers[..., 0],
            helper_data.geometric_centers[..., 1],
            helper_data.geometric_centers[..., 2],
        )
        r2 = (
            ((x - box_center[0]) / (PULSE_SIGMA * config.box_size.x)) ** 2
            + ((y - box_center[1]) / (PULSE_SIGMA * config.box_size.y)) ** 2
            + ((z - box_center[2]) / (PULSE_SIGMA * config.box_size.z)) ** 2
        )
        pulse = PULSE_AMPLITUDE * jnp.exp(-0.5 * r2)
        initial_state = initial_state.at[registered_variables.cosmic_ray_e_index].set(pulse)

    final_state = time_integration(initial_state, config, params, registered_variables)
    return initial_state, final_state, config, registered_variables


def _max_abs_div_b_fv(state, config, registered_variables) -> float:
    b = state[registered_variables.magnetic_index.x : registered_variables.magnetic_index.z + 1]
    return float(jnp.max(jnp.abs(divergence3D(b, config.grid_spacing))))


def _run_fd_baseline():
    """FD's own (CR-absent) div(B) baseline, for reference only -- CR-grey
    has no FD implementation to test (see module docstring)."""
    config = SimulationConfig(solver_mode=FINITE_DIFFERENCE, num_cells=StaticIntVector(2 * N, N, N))
    params = SimulationParams(C_cfl=0.4)
    settings = CPAlfvenWave3DSettings(t_end=T_END)
    initial_state, config, params = setup_cp_alfven_wave(config, params, settings)
    registered_variables = get_registered_variables(config)
    final_state = time_integration(initial_state, config, params, registered_variables)

    iface = registered_variables.interface_magnetic_field_index
    d0 = _interface_field_divergence(
        initial_state[iface.x], initial_state[iface.y], initial_state[iface.z], config.grid_spacing
    )
    df = _interface_field_divergence(
        final_state[iface.x], final_state[iface.y], final_state[iface.z], config.grid_spacing
    )
    return float(jnp.max(jnp.abs(d0))), float(jnp.max(jnp.abs(df))), bool(jnp.any(jnp.isnan(final_state)))


def test_cr_divergence_b_preservation(tol: float = 1e-10):
    """div(B) preservation, FV scheme, CR-grey on vs. off (with and
    without anisotropic transport actively reading a real, time-varying B).

    Args:
        tol: Maximum allowed max(|div(B)|) at t_end, for every
            configuration tested. Calibrated observed values ~6e-15 to
            1.2e-14 (FV) and ~5.5e-14 (FD baseline) -- a >1e4x margin.
    """
    configs = {
        "FV, CR off": (False, False),
        "FV, CR on (isotropic)": (True, False),
        "FV, CR on + anisotropic": (True, True),
    }

    divb_final = {}
    for label, (grey, aniso) in configs.items():
        initial_state, final_state, config, registered_variables = _run_fv(grey, aniso)

        assert not bool(jnp.any(jnp.isnan(final_state))), f"{label}: run produced NaNs."

        divb_0 = _max_abs_div_b_fv(initial_state, config, registered_variables)
        divb_f = _max_abs_div_b_fv(final_state, config, registered_variables)
        divb_final[label] = divb_f
        print(f"{label}: max|div(B)| t=0 -> {divb_0:.3e}, t={T_END} -> {divb_f:.3e}")

        assert divb_f < tol, (
            f"{label}: max|div(B)| at t_end ({divb_f:.3e}) exceeds tol {tol} -- "
            "the CR module may be degrading div(B) preservation."
        )

    # FD's own baseline, for reference/context (see module docstring for
    # why CR-grey itself can't be exercised on FD).
    fd_divb_0, fd_divb_f, fd_nan = _run_fd_baseline()
    assert not fd_nan, "FD baseline run produced NaNs."
    print(f"FD baseline (CR absent): max|div(B)| t=0 -> {fd_divb_0:.3e}, t={T_END} -> {fd_divb_f:.3e}")
    divb_final["FD baseline\n(CR absent)"] = fd_divb_f

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = list(divb_final.keys())
    values = [divb_final[k] for k in labels]
    colors = ["C0", "C1", "C2", "grey"]
    ax.bar(labels, values, color=colors[: len(labels)])
    ax.axhline(tol, color="k", linestyle=":", label=f"tol = {tol:.0e}")
    ax.set_yscale("log")
    ax.set_ylabel(f"max(|div(B)|) at t={T_END}")
    ax.set_title(
        "Ladder item 19 (FV, CR-grey-vs-off): div(B) unaffected by the CR module\n"
        "(FD baseline shown for reference -- CR-grey has no FD implementation)"
    )
    ax.legend()
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_divergence_b_preservation_test.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_cr_divergence_b_preservation()
