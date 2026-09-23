"""
Full (thermal + kinetic + magnetic + CR) energy-budget pytest -- ladder
item 18's MHD extension (plan Sec. 4, item 18: "Full energy budget (thermal
+ kinetic + magnetic + CR) closes to round-off with injection and
streaming/collisional-loss accounting"). The hydro+CR half of item 18 lives
in `cr_energy_budget.py`; this file adds the magnetic channel that one
deferred.

**Why MHD could be deferred there and can be done here.** The companion
CR-off baseline (`pytests/mhd/mhd_energy_conservation.py`) originally
measured a ~1e-6 residual for plain FV MHD and attributed it to Strang-split
truncation error. Instrumenting every sub-step of a real `time_integration`
run showed that attribution was wrong:

- Both gas half-steps (`_evolve_gas_state_unsplit`, which is also where every
  CR-grey source term lives) conserve thermal+kinetic(+CR) energy to
  round-off (~1e-14 absolute per step).
- The whole residual came from `magnetic_update`, the Pang & Wu implicit-
  midpoint B/v update. That update is exactly energy-conserving *at
  convergence*: its centered, `jnp.roll`-based curl satisfies discrete
  summation by parts on a periodic grid, so the midpoint kinetic and magnetic
  energy changes cancel identically (measured: -2.9e-16 summed over a
  whole run). But its fixed-point iteration stops on a tolerance chosen by
  `config.numerical_precision`, which defaults to `SINGLE_PRECISION` (1e-5)
  even in float64 runs -- every step was stopping 4-5 iterations in with
  a ~1e-6 midpoint residual. With `numerical_precision=DOUBLE_PRECISION`
  (1e-10) the same CP-Alfven-wave baseline closes to 7.5e-12 (N=8) /
  2.1e-12 (N=16) instead of 6.3e-6 / 2.3e-6.

So the correct statement for MHD is the same as for hydro: energy closes to
(fixed-point-tolerance-limited) round-off, provided the run asks for the
double-precision tolerance. This test does, and additionally re-runs the main
configuration with the default `SINGLE_PRECISION` tolerance to pin that
explanation down.

**How the CR module couples to B, and so what this test exercises.** CR-grey
only reads B in `anisotropic_flux_projection` (projects F_cr onto the local
B direction once per step, `_iteration_level_continuous_updates`), which
rewrites F_cr but moves no energy. Streaming (`streaming_flux_target`/
`cr_streaming_heating_source`) uses the isotropic `reduced_streaming_speed`,
not the Alfven speed. DSA injection (`inject_crs_at_shocks`) runs on the full
state, B rows included. The CR source terms run inside the gas half-steps,
where B is split out. So the energy-relevant new coupling is indirect: CR
pressure changes the gas velocity field, which then does work against B in
`magnetic_update`. The test makes sure that path is actually exercised (the
magnetic energy changes by a clearly nonzero amount) and that anisotropic
transport does real work (turning it off changes the result).

**Setup.** `cr_energy_budget.py`'s periodic-box, fixed-timestep CR-DSA Sedov
blast (`E_EXPLOSION=1`, `RHO_AMBIENT=1`, `P_AMBIENT=1e-4`), threaded by a
uniform `B_x = B0`. No gravity, cooling or collisional losses (the latter are
not implemented as a dynamical e_cr sink anywhere in this module, see
`cr_energy_budget.py`'s docstring), so nothing enters or leaves the box and
total (thermal+kinetic+magnetic+CR) energy must be exactly constant.
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
import jax
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
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    DOUBLE_PRECISION,
    PERIODIC_BOUNDARY,
    SINGLE_PRECISION,
)

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# Exact energy-conservation check -- needs float64 (same rationale as
# cr_energy_budget.py).
jax.config.update("jax_enable_x64", True)

# ---- physical setup (cr_energy_budget.py's, plus a uniform B_x) ----
GAMMA = 5.0 / 3.0
NUM_CELLS = 128
# Fixed dt must respect the initial blast's CFL bound, which tightens faster
# than 1/NUM_CELLS (the tanh-tapered deposit sharpens with resolution):
# measured effective Courant number 0.63 at 48^3 / 400 steps, 2.77 at
# 128^3 / 400 steps. 1800 steps puts 128^3 back at ~0.6.
NUM_TIMESTEPS = 1800
T_END = 0.07
E_EXPLOSION = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1e-4
B0 = 0.2
R_EXPLOSION = 0.05
SMOOTH_CELLS = 2.0
DSA_MACH_MIN = 1.3
DSA_EFFICIENCY = 0.1

_PERIODIC_BOX = BoundarySettings(
    x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
)


def _run(
    grey_cosmic_rays: bool = True,
    anisotropic_transport: bool = True,
    numerical_precision: int = DOUBLE_PRECISION,
):
    """Run one periodic-box magnetized CR-DSA Sedov blast; return energy
    diagnostics. DSA and streaming are on whenever CRs are."""
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=NUM_CELLS,
        mhd=True,
        numerical_precision=numerical_precision,
        boundary_settings=_PERIODIC_BOX,
        fixed_timestep=True,
        num_timesteps=NUM_TIMESTEPS,
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=grey_cosmic_rays,
            diffusive_shock_acceleration=grey_cosmic_rays,
            streaming=grey_cosmic_rays,
            anisotropic_transport=grey_cosmic_rays and anisotropic_transport,
        ),
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
        magnetic_field_x=jnp.ones(shape) * B0,
        magnetic_field_y=zeros,
        magnetic_field_z=zeros,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        cosmic_ray_grey_params=CosmicRayGreyParams(
            dsa_efficiency=DSA_EFFICIENCY, dsa_mach_min=DSA_MACH_MIN
        ),
    )

    def _energies(state):
        rho = state[registered_variables.density_index]
        v = state[registered_variables.velocity_index.x : registered_variables.velocity_index.z + 1]
        b = state[registered_variables.magnetic_index.x : registered_variables.magnetic_index.z + 1]
        p_gas = state[registered_variables.pressure_index]
        e_thermal = float(jnp.sum(p_gas / (GAMMA - 1.0)) * cell_volume)
        e_kinetic = float(jnp.sum(0.5 * rho * jnp.sum(v**2, axis=0)) * cell_volume)
        e_magnetic = float(jnp.sum(0.5 * jnp.sum(b**2, axis=0)) * cell_volume)
        if grey_cosmic_rays:
            e_cr = float(jnp.sum(state[registered_variables.cosmic_ray_e_index]) * cell_volume)
        else:
            e_cr = 0.0
        return dict(thermal=e_thermal, kinetic=e_kinetic, magnetic=e_magnetic, cr=e_cr)

    final_state = time_integration(initial_state, config, params, registered_variables)

    e_0 = _energies(initial_state)
    e_f = _energies(final_state)
    m0 = float(jnp.sum(density) * cell_volume)
    mf = float(jnp.sum(final_state[registered_variables.density_index]) * cell_volume)

    return dict(
        final_state=final_state,
        registered_variables=registered_variables,
        E_0=e_0,
        E_f=e_f,
        E_total_0=sum(e_0.values()),
        E_total_f=sum(e_f.values()),
        mass_0=m0,
        mass_f=mf,
    )


def _rel_energy_err(run):
    return abs(run["E_total_f"] - run["E_total_0"]) / abs(run["E_total_0"])


def _cr_elongation(e_cr_slice, coords):
    """e_cr-weighted second-moment ratio <(x-x_c)^2> / <(y-y_c)^2> in a
    midplane slice: > 1 means CRs are spread further along B (x) than across."""
    w = e_cr_slice / np.sum(e_cr_slice)
    xx, yy = np.meshgrid(coords, coords, indexing="ij")
    xc, yc = np.sum(w * xx), np.sum(w * yy)
    return float(np.sum(w * (xx - xc) ** 2) / np.sum(w * (yy - yc) ** 2))


def _mhd_fields_plot(run, run_iso, path):
    """Midplane (z = box centre) view of the magnetized CR blast at t_end:
    how the blast reshapes B and where the CRs sit relative to it, for
    anisotropic vs. isotropic CR transport."""
    rv = run["registered_variables"]
    k = NUM_CELLS // 2
    coords = (np.arange(NUM_CELLS) + 0.5) / NUM_CELLS
    extent = (0.0, 1.0, 0.0, 1.0)

    def _slices(state):
        state = np.asarray(state)
        return dict(
            rho=state[rv.density_index, :, :, k],
            p=state[rv.pressure_index, :, :, k],
            bx=state[rv.magnetic_index.x, :, :, k],
            by=state[rv.magnetic_index.y, :, :, k],
            bz=state[rv.magnetic_index.z, :, :, k],
            e_cr=state[rv.cosmic_ray_e_index, :, :, k],
        )

    an, iso = _slices(run["final_state"]), _slices(run_iso["final_state"])
    b2 = an["bx"] ** 2 + an["by"] ** 2 + an["bz"] ** 2
    log_beta = np.log10(2.0 * an["p"] / b2)

    def _field_lines(ax, s):
        # arrays are indexed [x, y]; streamplot/imshow want [y, x]
        ax.streamplot(
            coords, coords, s["bx"].T, s["by"].T,
            color="white", linewidth=0.6, density=1.1, arrowsize=0.6,
        )

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    im = ax.imshow(np.log10(an["rho"]).T, origin="lower", extent=extent, cmap="Greys")
    fig.colorbar(im, ax=ax, label=r"$\log_{10}\rho$")
    ax.set_title("Density")

    ax = axes[0, 1]
    im = ax.imshow((b2 / B0**2).T, origin="lower", extent=extent, cmap="Purples")
    _field_lines(ax, an)
    fig.colorbar(im, ax=ax, label=r"$|B|^2 / B_0^2$")
    ax.set_title("Magnetic pressure + in-plane field lines\n(shell compression across B, not along it)")

    ax = axes[0, 2]
    lim = float(np.max(np.abs(log_beta)))
    im = ax.imshow(log_beta.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.contour(coords, coords, log_beta.T, levels=[0.0], colors="k", linewidths=0.8)
    fig.colorbar(im, ax=ax, label=r"$\log_{10}\beta$,  $\beta = 2P_{\rm gas}/B^2$")
    ax.set_title(r"Plasma $\beta$ (black: $\beta = 1$)")

    e_max = float(np.maximum(np.max(an["e_cr"]), np.max(iso["e_cr"])))
    e_min = e_max * 1e-4
    for ax, s, name in ((axes[1, 0], an, "anisotropic"), (axes[1, 1], iso, "isotropic")):
        im = ax.imshow(
            np.log10(np.clip(s["e_cr"], e_min, None)).T,
            origin="lower", extent=extent, cmap="Oranges",
            vmin=np.log10(e_min), vmax=np.log10(e_max),
        )
        _field_lines(ax, s)
        fig.colorbar(im, ax=ax, label=r"$\log_{10} e_{\rm cr}$")
        ax.set_title(
            f"CR energy, {name} transport\n"
            f"elongation along B: {_cr_elongation(s['e_cr'], coords):.3f}"
        )

    ax = axes[1, 2]
    for s, name, color in ((iso, "isotropic", "C0"), (an, "anisotropic", "C1")):
        ax.semilogy(coords, s["e_cr"][:, NUM_CELLS // 2], color=color, lw=2, label=f"{name}, along x (∥ B)")
        ax.semilogy(coords, s["e_cr"][NUM_CELLS // 2, :], color=color, lw=2, ls="--", label=f"{name}, along y (⊥ B)")
    ax.set_ylim(e_min, 2 * e_max)
    ax.set_xlabel("position through box centre")
    ax.set_ylabel(r"$e_{\rm cr}$")
    ax.set_title("CR energy cuts through the centre")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for ax in axes.flat[:5]:
        ax.set_xlabel("x  (B$_0$ direction)")
        ax.set_ylabel("y")

    fig.suptitle(
        f"Ladder item 18 (MHD + CR): midplane z = 0.5 at t = {T_END}, "
        f"$B_0 = {B0}\\,\\hat{{x}}$, {NUM_CELLS}$^3$"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def test_cr_mhd_energy_budget(energy_tol: float = 1e-9, mass_tol: float = 1e-10):
    """Full (thermal + kinetic + magnetic + CR) energy budget, DSA injection,
    streaming and anisotropic transport all active, periodic box, fixed
    timestep, double-precision magnetic fixed-point tolerance.

    Args:
        energy_tol: Maximum allowed relative error on
            ``E_total(t_end) == E_total(0)``.
        mass_tol: Maximum allowed relative error on total mass conservation.
    """
    run = _run()

    assert not bool(jnp.any(jnp.isnan(run["final_state"]))), "Run produced NaNs."

    rel_mass_err = abs(run["mass_f"] - run["mass_0"]) / abs(run["mass_0"])
    assert rel_mass_err < mass_tol, f"Mass not conserved: rel. err {rel_mass_err:.3e}."

    rel_energy_err = _rel_energy_err(run)
    print(
        f"MHD+CR: rel. mass err = {rel_mass_err:.3e}, rel. energy err = {rel_energy_err:.3e}\n"
        f"  E(0)     = {run['E_0']}\n  E(t_end) = {run['E_f']}"
    )
    assert rel_energy_err < energy_tol, (
        f"Total (thermal+kinetic+magnetic+CR) energy not conserved: rel. err "
        f"{rel_energy_err:.3e} >= tol {energy_tol}."
    )

    # DSA must have fired: same sanity band as cr_energy_budget.py.
    cr_fraction = run["E_f"]["cr"] / run["E_total_0"]
    assert 0.005 < cr_fraction < 0.3, (
        f"CR energy fraction ({cr_fraction:.4f}) is outside the expected band -- "
        f"injection may not be firing as intended."
    )

    # The magnetic channel must genuinely take part in the exchange, or the
    # "+ magnetic" in the budget is vacuous.
    d_magnetic = abs(run["E_f"]["magnetic"] - run["E_0"]["magnetic"]) / run["E_total_0"]
    assert d_magnetic > 1e-3, (
        f"Magnetic energy barely changed (|dE_mag|/E_total = {d_magnetic:.3e}) -- "
        f"the blast is not doing work against B."
    )

    # Anisotropic transport (the only place CR-grey reads B) must do real work.
    run_iso = _run(anisotropic_transport=False)
    state_diff = float(jnp.max(jnp.abs(run["final_state"] - run_iso["final_state"])))
    assert state_diff > 1e-6, (
        f"Anisotropic transport had no detectable effect (max state diff "
        f"{state_diff:.3e}) -- is it actually being applied?"
    )
    assert _rel_energy_err(run_iso) < energy_tol, (
        f"Isotropic-transport MHD+CR run does not conserve energy: rel. err "
        f"{_rel_energy_err(run_iso):.3e}."
    )

    # CR-off MHD control, and the main configuration again with the default
    # (SINGLE_PRECISION) magnetic fixed-point tolerance -- the latter must be
    # clearly worse, confirming that the tolerance, not the CR coupling or a
    # splitting error, is what limits MHD energy conservation.
    control = _run(grey_cosmic_rays=False)
    loose = _run(numerical_precision=SINGLE_PRECISION)
    print(
        f"MHD control (CR off): rel. energy err = {_rel_energy_err(control):.3e}\n"
        f"MHD+CR isotropic:     rel. energy err = {_rel_energy_err(run_iso):.3e}\n"
        f"MHD+CR, SINGLE_PRECISION tolerance: rel. energy err = {_rel_energy_err(loose):.3e}"
    )
    assert _rel_energy_err(control) < energy_tol
    assert _rel_energy_err(loose) > 10 * rel_energy_err, (
        "Loosening the magnetic fixed-point tolerance did not degrade energy "
        "conservation -- the explanation in this module's docstring no longer holds."
    )

    # ---- plot ----
    fig, (ax_bars, ax_err) = plt.subplots(1, 2, figsize=(12, 4.5))

    channels = ["thermal", "kinetic", "magnetic", "cr"]
    x = jnp.arange(len(channels))
    width = 0.27
    ax_bars.bar(x - width, [run["E_0"][c] for c in channels], width, label="t = 0", color="grey")
    ax_bars.bar(x, [run_iso["E_f"][c] for c in channels], width, label="isotropic CR transport", color="C0")
    ax_bars.bar(x + width, [run["E_f"][c] for c in channels], width, label="anisotropic CR transport", color="C1")
    ax_bars.set_xticks(x, ["thermal", "kinetic", "magnetic", "CR"])
    ax_bars.set_ylabel("energy")
    ax_bars.set_title(f"Energy partition at t={T_END}\n(periodic box, fixed dt, B_x = {B0})")
    ax_bars.legend()

    labels = ["MHD only\n(control)", "MHD + CR\nisotropic", "MHD + CR\nanisotropic", "MHD + CR aniso.\nSINGLE_PREC. tol."]
    errs = [_rel_energy_err(r) for r in (control, run_iso, run, loose)]
    ax_err.bar(labels, errs, color=["grey", "C0", "C1", "C3"])
    ax_err.set_yscale("log")
    ax_err.axhline(energy_tol, color="k", linestyle=":", label=f"tol = {energy_tol:.0e}")
    ax_err.set_ylabel("relative total-energy error")
    ax_err.set_title("Closes to round-off with the double-precision\nmagnetic fixed-point tolerance")
    ax_err.legend()

    fig.suptitle("Ladder item 18 (MHD + CR): thermal + kinetic + magnetic + CR energy budget")
    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_mhd_energy_budget_test.svg")
    plt.close(fig)

    _mhd_fields_plot(run, run_iso, pics_dir / "cr_mhd_energy_budget_fields.png")


if __name__ == "__main__":
    test_cr_mhd_energy_budget()
