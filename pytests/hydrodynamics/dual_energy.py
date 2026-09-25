"""
Dual-energy (entropy) formalism test (``SimulationConfig.dual_energy``).

Three checks, each on the Strang-split MUSCL path used by the CWB setup and on
the unsplit SSP-RK2 path:

1. Cold Hubble expansion (1D and 2D). Uniform density, ``u = U (x - x_c)``,
   pressure ~1e-5 of the kinetic energy at the box edge (Mach ~ 10^3). The
   exact solution stays uniform and adiabatic, ``rho = rho0 / (1 + U t)^d``,
   ``p = p0 (rho / rho0)^gamma``. The total-energy scheme heats the gas
   through kinetic-energy truncation error; the dual-energy scheme must track
   the adiabat, i.e. conserve p / rho^gamma (checked for 0.08 < |x - x_c| < 0.25, i.e. away from the open
   boundaries and from the stagnation point at the centre, where the flow is
   not cold relative to its kinetic energy and the switch does not engage).
2. Colliding cold streams (1D). Two Mach ~10^3 streams collide and drive two
   strong shocks; the post-shock plateau must match Rankine-Hugoniot
   (``rho = 4 rho1``, ``p = 4/3 rho1 u1^2`` for gamma = 5/3). Run with
   ``dual_energy_eta`` huge, so every unshocked cell uses the entropy
   pressure and only the shock-finder switch keeps shock heating: with the
   switch the plateau is correct, with detection disabled
   (``dual_energy_mach_min`` huge) the streams fail to heat.
3. Sod shock tube: dual energy (default eta) changes nothing measurable.

Writes ``figures/dual_energy.png``.
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
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    HLLC,
    MINMOD,
    OPEN_BOUNDARY,
)
from astronomix.option_classes.simulation_config import (
    MUSCL,
    RK2_SSP,
    SPLIT,
    UNSPLIT,
    finalize_config,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    time_integration,
    get_registered_variables,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.test_setups.hydrodynamics.shock_tube1D import (
    setup_sod_shock_tube,
    sod_shock_tube_solution,
)

FIG_DIR = Path(__file__).parent / "figures"

GAMMA = 5 / 3

SCHEMES = {
    "split/MUSCL": dict(split=SPLIT, time_integrator=MUSCL),
    "unsplit/RK2": dict(split=UNSPLIT, time_integrator=RK2_SSP),
}


def _base_config(dimensionality, num_cells, scheme, dual_energy):
    open_1d = BoundarySettings1D(left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY)
    return SimulationConfig(
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=dimensionality,
        num_cells=num_cells,
        box_size=1.0,
        boundary_settings=open_1d if dimensionality == 1 else BoundarySettings(open_1d, open_1d),
        dual_energy=dual_energy,
        **SCHEMES[scheme],
    )


def _run(config, params, build_state):
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    state = build_state(config, registered_variables, helper_data)
    config = finalize_config(config, state.shape)
    final_state = time_integration(state, config, params, registered_variables)
    return final_state, registered_variables, helper_data


# -------------------------------------------------------------
# ==================== ↓ Cold expansion ↓ =====================
# -------------------------------------------------------------

def run_cold_expansion(dimensionality, scheme, dual_energy, num_cells=128,
                       expansion_rate=1.0, rho0=1.0, p0=1e-6, t_end=0.5):
    """Relative pressure error against the exact adiabat, interior cells."""
    config = _base_config(dimensionality, num_cells, scheme, dual_energy)
    params = SimulationParams(C_cfl=0.4, gamma=GAMMA, t_end=t_end,
                              minimum_pressure=1e-20, minimum_density=1e-20)

    def build_state(config, registered_variables, helper_data):
        x = helper_data.geometric_centers - 0.5
        if dimensionality == 1:
            return construct_primitive_state(
                config=config, registered_variables=registered_variables,
                density=jnp.full_like(x, rho0),
                velocity_x=expansion_rate * x,
                gas_pressure=jnp.full_like(x, p0),
            )
        rx, ry = x[..., 0], x[..., 1]
        return construct_primitive_state(
            config=config, registered_variables=registered_variables,
            density=jnp.full_like(rx, rho0),
            velocity_x=expansion_rate * rx,
            velocity_y=expansion_rate * ry,
            gas_pressure=jnp.full_like(rx, p0),
        )

    final_state, rv, helper_data = _run(config, params, build_state)
    rho_exact = rho0 / (1 + expansion_rate * t_end) ** dimensionality
    p_exact = p0 * (rho_exact / rho0) ** GAMMA

    x = helper_data.geometric_centers - 0.5
    # Away from the open boundaries and from the expansion centre: at the
    # centre u -> 0, the flow is not cold relative to its kinetic energy, the
    # dual-energy switch (by design) does not engage, and both schemes share
    # the same stagnation-point start-up error.
    dist = jnp.abs(x) if dimensionality == 1 else jnp.linalg.norm(x, axis=-1)
    interior = (dist > 0.08) & (dist < 0.25)
    # Error in the adiabatic invariant p / rho^gamma, which is what the
    # dual-energy scheme is meant to preserve. (Dimension-split sweeps also
    # leave a small density bump along the x_c / y_c lines in both schemes --
    # a continuity-equation artifact; p follows it adiabatically.)
    p = final_state[rv.pressure_index]
    rho = final_state[rv.density_index]
    invariant = p / rho**GAMMA / (p0 / rho0**GAMMA)
    err = jnp.max(jnp.abs(invariant[interior] - 1.0))
    return float(err), final_state, rv, helper_data, p_exact


# -------------------------------------------------------------
# ================== ↓ Colliding streams ↓ ====================
# -------------------------------------------------------------

def run_colliding_streams(scheme, dual_energy, num_cells=400, u1=1.0, rho1=1.0,
                          p1=1e-6, t_end=0.6, eta=1e6, mach_min=1.3):
    """Post-shock plateau (centre) density and pressure."""
    config = _base_config(1, num_cells, scheme, dual_energy)
    params = SimulationParams(C_cfl=0.4, gamma=GAMMA, t_end=t_end,
                              dual_energy_eta=eta, dual_energy_mach_min=mach_min,
                              minimum_pressure=1e-20, minimum_density=1e-20)

    def build_state(config, registered_variables, helper_data):
        x = helper_data.geometric_centers
        return construct_primitive_state(
            config=config, registered_variables=registered_variables,
            density=jnp.full_like(x, rho1),
            velocity_x=jnp.where(x < 0.5, u1, -u1),
            gas_pressure=jnp.full_like(x, p1),
        )

    final_state, rv, helper_data = _run(config, params, build_state)
    x = helper_data.geometric_centers
    # the shocks move out at u1/3; sample well inside the plateau
    plateau = jnp.abs(x - 0.5) < 0.5 * u1 / 3 * t_end
    rho_c = float(jnp.median(final_state[rv.density_index][plateau]))
    p_c = float(jnp.median(final_state[rv.pressure_index][plateau]))
    return rho_c, p_c, final_state, rv, helper_data


def test_dual_energy():
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # ---- 1. cold expansion ----
    print("cold expansion: max |(p/rho^gamma)/(p0/rho0^gamma) - 1| in the interior")
    for dimensionality in (1, 2):
        num_cells = 128 if dimensionality == 1 else 64
        for scheme in SCHEMES:
            err_off, state_off, rv_off, hd, p_exact = run_cold_expansion(
                dimensionality, scheme, False, num_cells=num_cells)
            err_on, state_on, rv_on, _, _ = run_cold_expansion(
                dimensionality, scheme, True, num_cells=num_cells)
            print(f"  {dimensionality}D {scheme:12s} total energy {err_off:9.3e}   dual energy {err_on:9.3e}")
            assert err_on < 1e-2, f"dual energy should track the adiabat ({err_on})"
            # (NaN counts as worse: the 2D unsplit total-energy run fails)
            assert not (err_off <= 10 * err_on), "the total-energy scheme should be visibly worse"
            if dimensionality == 1 and scheme == "split/MUSCL":
                x = hd.geometric_centers
                axes[0].semilogy(x, state_off[rv_off.pressure_index] / p_exact, label="total energy")
                axes[0].semilogy(x, state_on[rv_on.pressure_index] / p_exact, label="dual energy")
                for lo, hi in ((0.25, 0.42), (0.58, 0.75)):
                    axes[0].axvspan(lo, hi, color="0.9", zorder=-1)
    axes[0].set(title="1D cold expansion (split/MUSCL), shaded: checked", xlabel="x", ylabel="p / p_exact")
    axes[0].legend()

    # ---- 2. colliding streams ----
    rho_rh, p_rh = 4.0, 4.0 / 3.0
    print("colliding streams: plateau rho, p (Rankine-Hugoniot 4, 1.333)")
    for scheme in SCHEMES:
        rho_off, p_off, s_off, rv0, hd = run_colliding_streams(scheme, False)
        rho_on, p_on, s_on, rv1, _ = run_colliding_streams(scheme, True)
        rho_ns, p_ns, s_ns, rv2, _ = run_colliding_streams(scheme, True, mach_min=1e6)
        print(f"  {scheme:12s} total energy ({rho_off:.3f}, {p_off:.4f})   "
              f"dual+switch ({rho_on:.3f}, {p_on:.4f})   dual, no switch ({rho_ns:.3f}, {p_ns:.3e})")
        for rho_c, p_c in ((rho_off, p_off), (rho_on, p_on)):
            assert abs(rho_c / rho_rh - 1) < 0.05 and abs(p_c / p_rh - 1) < 0.05
        assert p_ns < 0.5 * p_rh, "without shock detection the entropy pressure must fail to heat"
        if scheme == "split/MUSCL":
            x = hd.geometric_centers
            axes[1].plot(x, s_off[rv0.pressure_index], label="total energy")
            axes[1].plot(x, s_on[rv1.pressure_index], "--", label="dual energy + shock switch")
            axes[1].plot(x, s_ns[rv2.pressure_index], ":", label="dual energy, no shock switch")
    axes[1].axhline(p_rh, color="k", lw=0.8)
    axes[1].set(title="colliding Mach~10^3 streams, eta=1e6", xlabel="x", ylabel="pressure")
    axes[1].legend(fontsize=8)

    # ---- 3. Sod ----
    print("Sod: max |difference| dual vs total energy")
    for scheme in SCHEMES:
        finals = []
        for dual in (False, True):
            config = _base_config(1, 200, scheme, dual)
            rv = get_registered_variables(config)
            hd = get_helper_data(config)
            # C_cfl=0.4: the 1D split/MUSCL path NaNs at 0.8 with or
            # without dual energy
            state, config, params = setup_sod_shock_tube(config, rv, SimulationParams(C_cfl=0.4), hd)
            config = finalize_config(config, state.shape)
            finals.append((time_integration(state, config, params, rv), rv))
        (a, rva), (b, rvb) = finals
        diff = max(float(jnp.max(jnp.abs(a[i] - b[i]))) for i in (rva.density_index, rva.velocity_index, rva.pressure_index))
        print(f"  {scheme:12s} {diff:.3e}")
        assert diff < 1e-3
        if scheme == "split/MUSCL":
            ref = sod_shock_tube_solution(config, rvb, params, hd)
            axes[2].plot(hd.geometric_centers, ref[rvb.pressure_index], "k", lw=0.8, label="exact")
            axes[2].plot(hd.geometric_centers, a[rva.pressure_index], label="total energy")
            axes[2].plot(hd.geometric_centers, b[rvb.pressure_index], "--", label="dual energy")
    axes[2].set(title="Sod shock tube (split/MUSCL)", xlabel="x", ylabel="pressure")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "dual_energy.png", dpi=130)


if __name__ == "__main__":
    test_dual_energy()
