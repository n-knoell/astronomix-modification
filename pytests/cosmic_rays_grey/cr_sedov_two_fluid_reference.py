"""
Self-test of the self-similar two-fluid Sedov reference (Phase B test plan B7.1).

``astronomix/test_setups/reference_solutions/sedov_two_fluid.py`` solves the
point explosion with a constant CR injection fraction ``zeta`` at the strong
shock and adiabatic, tightly coupled CRs behind it (Chevalier 1983; Pfrommer
et al. 2017, Sec. 4.2). Items 7 and 8 (and B7.2-B7.5) compare against it, so
this file checks the solver itself. Pure numpy/scipy, no simulation.

Gates (physics, hard-coded):

- ``zeta = 0`` is the classic Sedov blast: ``xi_0`` = 1.15167 (``gamma`` = 5/3)
  and 1.0328 (7/5) (Sedov 1959; Kamm & Timmes 2007), kinetic fraction 0.2828
  (5/3) and 0.2187 (7/5).
- Energy integral: the energy flux through every sphere co-moving with fixed
  ``lambda = r/R`` vanishes (Sedov's first integral; the ODEs only satisfy it
  implicitly), below 1e-4 of the post-shock flux. Mass inside the shock is the
  swept-up mass, to 1e-5.
- Boundary: ``X_cr,0 = P_cr/P_th = zeta / (2 (1 - zeta))`` (Pfrommer et al.
  2017) and the compression ``x = 2 k - 1`` of the caloric index.
- Continuity at ``zeta -> 0`` (1e-6 vs exactly 0).
- ``profiles()`` (the physical-unit evaluator the 3D tests use) integrates back
  to ``E`` and to zero excess mass on a radial grid.
- Coarse cross-check: Pais et al. (2018) eq. 16's effective single-fluid
  ``gamma(zeta)`` gives a classic Sedov ``xi_0`` within 3% of the two-fluid one
  (it is an approximation; measured 1.1-1.8%).
- ``"gas_rh"`` convention (what ``inject_crs_at_shocks`` does): the CR flux
  through the shock is exactly ``zeta`` times the gas-only RH dissipated flux.

The values the plan quotes (its Sec. 7.2 table) are kept as regression values
to 1e-3, so the plan, the tests and the solver cannot drift apart.

References:
    Chevalier (1983), ApJ 272, 765. Pfrommer et al. (2017), MNRAS 465, 4500.
    Pais et al. (2018), MNRAS 478, 5278. Kamm & Timmes (2007), LA-UR-07-2849.
"""

# general
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix reference solutions
from astronomix.test_setups.reference_solutions.sedov_two_fluid import (
    solve_sedov_two_fluid,
)

PICS_DIR = Path(__file__).resolve().parent / "pics" / "07_sedov_taylor"

#: plan Sec. 7.2 table: zeta -> (xi_0, f_kin, f_th, f_cr, dE_inj/dt * t/E)
PLAN_TABLE = {
    0.0: (1.1516, 0.2828, 0.7172, 0.0, 0.0),
    0.1: (1.1289, 0.2673, 0.5898, 0.1429, 0.0425),
    0.211: (1.1066, 0.2531, 0.4731, 0.2738, 0.0835),
    0.375: (1.0775, 0.2362, 0.3330, 0.4308, 0.1357),
    0.5: (1.0578, 0.2258, 0.2455, 0.5287, 0.1704),
}


def test_sedov_two_fluid_classic_limit():
    """``zeta = 0``: classic Sedov constants for gamma 5/3 and 7/5."""
    for gamma, xi_ref, kin_ref in ((5.0 / 3.0, 1.15167, 0.2828), (1.4, 1.0328, 0.2187)):
        sol = solve_sedov_two_fluid(0.0, gamma_th=gamma)
        print(f"gamma {gamma:.4f}: xi_0 {sol.xi_0:.6f} (ref {xi_ref}), "
              f"E_kin/E {sol.f_kin:.4f} (ref {kin_ref}), energy closure "
              f"{sol.energy_closure:.1e}, mass closure {sol.mass_closure:.1e}")
        assert abs(sol.xi_0 - xi_ref) < 2e-4, f"xi_0 {sol.xi_0} vs {xi_ref}"
        assert abs(sol.f_kin - kin_ref) < 1e-3, f"E_kin/E {sol.f_kin} vs {kin_ref}"
        assert sol.f_cr == 0.0
        assert sol.energy_closure < 1e-4 and sol.mass_closure < 1e-5
        # gas-only dissipation rate 4 pi R^2 * (rho_0 V_s^3 delta/2) in E/t;
        # 0.458 for 5/3 matches the independent ~0.46 E/t estimate (PROGRESS.md 2026-10-07)
        assert abs(sol.dissipation_rate - sol.finder_dissipation_rate) < 1e-12
    sol_5_3 = solve_sedov_two_fluid(0.0)
    assert abs(sol_5_3.dissipation_rate - 0.4583) < 1e-3


def test_sedov_two_fluid_with_crs():
    """``zeta > 0``: closure, boundary, continuity, profiles, Pais et al. check."""
    for zeta, (xi_ref, kin_ref, th_ref, cr_ref, rate_ref) in PLAN_TABLE.items():
        sol = solve_sedov_two_fluid(zeta)
        print(f"zeta {zeta:5.3f}: xi_0 {sol.xi_0:.5f}, x {sol.compression:.4f}, "
              f"X_cr,0 {sol.x_cr_0:.4f}, E_kin/E_th/E_cr {sol.f_kin:.4f}/"
              f"{sol.f_th:.4f}/{sol.f_cr:.4f}, rate*t/E {sol.injection_rate:.4f}, "
              f"closure {sol.energy_closure:.1e}/{sol.mass_closure:.1e}")
        assert sol.energy_closure < 1e-4 and sol.mass_closure < 1e-5
        assert abs(sol.f_kin + sol.f_th + sol.f_cr - 1.0) < 1e-12
        for name, value, ref in (("xi_0", sol.xi_0, xi_ref), ("E_kin/E", sol.f_kin, kin_ref),
                                 ("E_th/E", sol.f_th, th_ref), ("E_cr/E", sol.f_cr, cr_ref),
                                 ("rate", sol.injection_rate, rate_ref)):
            assert abs(value - ref) < 1e-3, f"zeta {zeta}: {name} {value} vs plan {ref}"
        if zeta == 0.0:
            continue
        assert abs(sol.x_cr_0 - zeta / (2.0 * (1.0 - zeta))) < 1e-12
        f = sol.x_cr_0 / (1.0 + sol.x_cr_0)
        assert abs(sol.compression - (2.0 * (2.5 * (1.0 - f) + 4.0 * f) - 1.0)) < 1e-12
        # each shell keeps P_th rho^-5/3 and P_cr rho^-4/3 from the same shock
        # state rho_2 = x rho_0: X_cr = X_cr,0 (G / x)^(-1/3) (Pfrommer et al.
        # 2017, eq. 64). Exact in the log-form ODEs, a consistency check.
        x_cr = sol.Pi_cr / sol.Pi_th
        x_pred = sol.x_cr_0 * (sol.G / sol.compression) ** (-1.0 / 3.0)
        assert np.max(np.abs(x_cr / x_pred - 1.0)) < 1e-8
        # Pais et al. (2018) eq. 16 effective single-fluid index
        gamma_eff = (4.0 / 3.0) * (8.0 * zeta + 5.0) / (9.0 * zeta + 4.0)
        xi_eff = solve_sedov_two_fluid(0.0, gamma_th=gamma_eff).xi_0
        print(f"  Pais et al. gamma_eff {gamma_eff:.4f}: xi_0 {xi_eff:.5f} "
              f"({sol.xi_0 / xi_eff - 1.0:+.3%})")
        assert abs(sol.xi_0 / xi_eff - 1.0) < 0.03
        # profiles() integrates back to E = 1 and zero excess mass
        t = 0.07
        r = np.linspace(0.0, 1.2 * sol.shock_radius(t), 200001)
        rho, u, p_th, p_cr = sol.profiles(r, t)
        energy = np.trapezoid((0.5 * rho * u**2 + 1.5 * p_th + 3.0 * p_cr) * 4 * np.pi * r**2, r)
        excess_mass = np.trapezoid((rho - 1.0) * 4 * np.pi * r**2, r)
        assert abs(energy - 1.0) < 1e-4, f"profiles() energy {energy}"
        assert abs(excess_mass) < 1e-4, f"profiles() excess mass {excess_mass}"

    # continuity at zeta -> 0
    assert abs(solve_sedov_two_fluid(1e-6).xi_0 - solve_sedov_two_fluid(0.0).xi_0) < 1e-5

    # "gas_rh" convention: CR flux = zeta * gas-only RH dissipated flux
    for zeta in (0.1, 0.211, 0.5):
        sol = solve_sedov_two_fluid(zeta, convention="gas_rh")
        assert abs(sol.injection_rate - zeta * sol.finder_dissipation_rate) < 1e-10
        print(f"gas_rh zeta {zeta}: E_cr/E {sol.f_cr:.4f}, rate*t/E "
              f"{sol.injection_rate:.4f}, X_cr,0 {sol.x_cr_0:.4f}")

    _plot()


def _plot():
    """Similarity profiles and Lagrangian map for the plan's zeta values."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    zetas = (0.0, 0.1, 0.211, 0.5)
    for i, zeta in enumerate(zetas):
        sol = solve_sedov_two_fluid(zeta)
        c = f"C{i}"
        lab = rf"$\zeta$={zeta} ($\xi_0$={sol.xi_0:.4f}, $E_{{cr}}/E$={sol.f_cr:.3f})"
        axes[0, 0].plot(sol.lam, sol.G, c, label=lab)
        axes[0, 1].plot(sol.lam, sol.U, c)
        axes[0, 2].plot(sol.lam, sol.Pi_th, c)
        if zeta > 0:
            axes[0, 2].plot(sol.lam, sol.Pi_cr, c + "--")
            axes[1, 0].plot(sol.lam, sol.Pi_cr / sol.Pi_th, c)
            q = np.linspace(0.02, 1.0, 50)
            axes[1, 2].plot(q, [sol.cr_energy_fraction_shocked_before(qq) for qq in q], c)
        axes[1, 1].plot(sol.lam, sol.r0_over_R, c)
    axes[0, 0].set(ylabel=r"$G=\rho/\rho_0$", title="density")
    axes[0, 1].set(ylabel=r"$U=u/V_s$", title="velocity")
    axes[0, 2].set(ylabel=r"$\Pi=P/(\rho_0 V_s^2)$", title=r"$P_{th}$ (solid), $P_{cr}$ (dashed)")
    axes[1, 0].set(ylabel=r"$X_{cr}=P_{cr}/P_{th}$", yscale="log", ylim=(1e-2, 1e3),
                   title=r"$X_{cr}\propto G^{-1/3}$")
    axes[1, 1].set(ylabel=r"$r_0/R$", title="Lagrangian origin of the shell")
    axes[1, 2].set(xlabel=r"$q$", ylabel="fraction of $E_{cr}$",
                   title=r"$E_{cr}$ in gas shocked while $R_s<qR$")
    for ax in axes.flat[:5]:
        ax.set_xlabel(r"$\lambda=r/R$")
        ax.set_xlim(0, 1.02)
    axes[1, 2].axvline(0.25, color="grey", lw=0.8, ls=":")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Self-similar two-fluid Sedov blast (B7.1), "
                 r"$\gamma_{th}=5/3$, $\gamma_{cr}=4/3$, $e_{cr,2}=\zeta\,e_{int,2}$")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_sedov_two_fluid_reference.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    test_sedov_two_fluid_classic_limit()
    test_sedov_two_fluid_with_crs()
