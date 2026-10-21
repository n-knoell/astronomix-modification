"""
Self-similar two-fluid (gas + cosmic-ray) Sedov-Taylor blast with CR
acceleration at the shock (Chevalier 1983; Pfrommer et al. 2017, Sec. 4.2).

A point explosion of energy ``E`` in a uniform, cold medium of density
``rho_0``. At the (strong) shock a fixed fraction of the dissipated energy goes
into cosmic rays; behind the shock both components are adiabatic, tightly
coupled to the gas (no diffusion, no streaming) and each keeps its own
entropy, ``P_th rho^-gamma_th`` and ``P_cr rho^-gamma_cr``. With a constant
injection fraction the problem has no new scale, so the blast stays
self-similar:

    R(t) = xi_0 (E t^2 / rho_0)^(1/5),    V_s = dR/dt = (2/5) R / t,

and with ``lambda = r / R``

    u = V_s U(lambda),  rho = rho_0 G(lambda),  P_i = rho_0 V_s^2 Pi_i(lambda).

The similarity ODEs (``a = U - lambda``) are

    continuity:  G U' + a G' = -2 U G / lambda
    momentum:    a U' + (Pi_th' + Pi_cr') / G = (3/2) U
    entropy i:   -gamma_i a G' / G + a Pi_i' / Pi_i = 3

(the ``3/2`` and ``3`` come from ``dV_s/dt = -(3/2) V_s / t``). They are
integrated inward from the strong-shock state at ``lambda = 1``. ``xi_0``
follows from the energy integral

    E = (16 pi / 25) xi_0^5 E * I,
    I = int_0^1 (G U^2 / 2 + Pi_th / (gamma_th - 1) + Pi_cr / (gamma_cr - 1)) lambda^2 dlambda.

Strong-shock boundary. With ``f = P_cr,2 / P_2`` the caloric index of the
post-shock mixture fixes the compression, ``x = 2 k - 1`` with
``k = (1 - f) gamma_th / (gamma_th - 1) + f gamma_cr / (gamma_cr - 1)``, and
``U = 1 - 1/x``, ``G = x``, ``Pi = 1 - 1/x`` (split ``1 - f : f``). Two
conventions for the injection fraction ``zeta`` set ``f``:

* ``"total"`` (Pfrommer et al. 2017's ``zeta``): ``e_cr,2 = zeta e_int,2``,
  i.e. ``X_cr,0 = P_cr/P_th = zeta (gamma_cr - 1) / ((1 - zeta)(gamma_th - 1))``
  (``zeta / (2 (1 - zeta))`` for 5/3 and 4/3).
* ``"gas_rh"`` (what ``cr_grey_injection.inject_crs_at_shocks`` does): the
  CR energy flux through the shock is ``zeta`` times the *gas-only*
  Rankine-Hugoniot dissipated flux ``delta(M) rho_0 V_s^3 / 2`` the shock
  finder reports (strong limit ``delta = 4 / (gamma + 1)^2`` = 0.5625 for
  5/3), while the more compressive two-fluid shock dissipates slightly more.
  Solved for ``f`` by root finding. For ``zeta = 0.1`` the effective
  ``"total"`` fraction is 0.098 (``E_cr/E`` 0.140 instead of 0.143).

Validated in ``pytests/cosmic_rays_grey/cr_sedov_two_fluid_reference.py``
(B7.1 of ``astronomix_CR_phaseB_test_plan.md``): ``zeta = 0`` reproduces the
classic Sedov constants (``xi_0`` = 1.15167 for 5/3, 1.03277 for 7/5) and the
kinetic fraction; energy and mass close; the boundary ratio is
``zeta/(2(1-zeta))``.

Implemented in plain numpy/scipy (like ``pfrommer_riemann_solver.py``): a
one-shot reference generator for tests, not part of the simulation hot path.

References:
    Chevalier, R. A. (1983). "Blast waves with cosmic-ray pressure". ApJ, 272, 765.
    Pfrommer, C., Pakmor, R., Schaal, K., Simpson, C. M., & Springel, V. (2017).
    "Simulating cosmic ray physics on a moving mesh". MNRAS, 465, 4500 (Sec. 4.2).
    Sedov, L. I. (1959). Similarity and Dimensional Methods in Mechanics.
"""

# general
from typing import NamedTuple

# numerics
import numpy as np
from scipy.integrate import solve_ivp, cumulative_trapezoid
from scipy.optimize import brentq


class SedovTwoFluidSolution(NamedTuple):
    """Self-similar two-fluid Sedov solution (dimensionless, see module docstring)."""

    #: injection fraction and its convention ("total" or "gas_rh")
    zeta: float
    convention: str
    gamma_th: float
    gamma_cr: float
    #: ``R = xi_0 (E t^2 / rho_0)^(1/5)``
    xi_0: float
    #: shock compression ``rho_2 / rho_0``
    compression: float
    #: post-shock ``P_cr / P_th``
    x_cr_0: float
    #: similarity profiles on ``lam`` (increasing, ``lam[-1] = 1``)
    lam: np.ndarray
    U: np.ndarray
    G: np.ndarray
    Pi_th: np.ndarray
    Pi_cr: np.ndarray
    #: Lagrangian origin of each shell, ``r_0 / R = m(lam)^(1/3)``
    r0_over_R: np.ndarray
    #: energy fractions (kinetic, thermal, CR) of ``E``
    f_kin: float
    f_th: float
    f_cr: float
    #: CR injection rate ``dE_inj/dt`` in units of ``E / t``
    injection_rate: float
    #: gas-only RH dissipated energy flux through the shock
    #: (``delta_strong rho_0 V_s^3 / 2`` over ``4 pi R^2``) in units of ``E / t``:
    #: what the shock finder's ``thermal_energy_flux`` integrates to
    finder_dissipation_rate: float
    #: dissipated (post-shock internal energy) flux of the two-fluid shock, ``E / t``
    dissipation_rate: float
    #: relative closure errors of the energy and mass integrals
    energy_closure: float
    mass_closure: float

    def shock_radius(self, t, e_explosion=1.0, rho_0=1.0):
        """``R(t)``."""
        return self.xi_0 * (e_explosion * t**2 / rho_0) ** 0.2

    def shock_velocity(self, t, e_explosion=1.0, rho_0=1.0):
        """``V_s(t) = (2/5) R / t``."""
        return 0.4 * self.shock_radius(t, e_explosion, rho_0) / t

    def profiles(self, r, t, e_explosion=1.0, rho_0=1.0, p_ambient=0.0):
        """Physical ``rho, u_r, P_th, P_cr`` at radii ``r`` and time ``t``.

        Outside the shock the ambient medium (``rho_0``, at rest, ``p_ambient``,
        no CRs). Inside the innermost tabulated shell the innermost values are
        held (``G -> 0``, the pressures tend to a constant).
        """
        r = np.asarray(r, dtype=float)
        R = self.shock_radius(t, e_explosion, rho_0)
        V = self.shock_velocity(t, e_explosion, rho_0)
        lam = r / R
        inside = lam <= 1.0
        lam_c = np.clip(lam, self.lam[0], 1.0)
        G = np.interp(lam_c, self.lam, self.G)
        U = np.interp(lam_c, self.lam, self.U)
        Pt = np.interp(lam_c, self.lam, self.Pi_th)
        Pc = np.interp(lam_c, self.lam, self.Pi_cr)
        # U ~ lambda near the centre (homologous core): scale instead of holding
        U = np.where(lam < self.lam[0], self.U[0] * lam / self.lam[0], U)
        rho = np.where(inside, rho_0 * G, rho_0)
        u = np.where(inside, V * U, 0.0)
        p_th = np.where(inside, rho_0 * V**2 * Pt + p_ambient, p_ambient)
        p_cr = np.where(inside, rho_0 * V**2 * Pc, 0.0)
        return rho, u, p_th, p_cr

    def cell_averages(self, offsets, dx, t, e_explosion=1.0, rho_0=1.0, p_ambient=0.0,
                      n_sub=4, density_floor=0.0):
        """Cell averages of the blast on a Cartesian grid (B7.2 initial state).

        ``offsets``: ``(3, ...)`` cell-centre positions relative to the blast
        centre; ``dx``: the (uniform) cell size. Each cell is sampled at
        ``n_sub^3`` points and the *conserved* densities are averaged (mass,
        momentum, total gas energy, CR energy), so the thin post-shock shell
        keeps its mass and energy at any resolution. ``density_floor`` (in
        units of ``rho_0``) raises ``G`` in the hot core, where it falls like
        ``lambda^(3/(gamma - 1))`` (1e-9 at ``lambda`` = 0.05) and the sound
        speed would set a vanishing time step; pressure and velocity are kept,
        so the energy changes only by the core's kinetic energy (negligible,
        ``U ~ lambda`` there).

        Returns ``rho``, the velocity ``(3, ...)``, ``p_th`` and ``e_cr``.
        """
        offsets = np.asarray(offsets, dtype=float)
        R = self.shock_radius(t, e_explosion, rho_0)
        V = self.shock_velocity(t, e_explosion, rho_0)
        sums = np.zeros((6,) + offsets.shape[1:])
        sub = (np.arange(n_sub) + 0.5) / n_sub - 0.5
        for sx in sub:
            for sy in sub:
                for sz in sub:
                    pos = offsets + dx * np.array([sx, sy, sz]).reshape((3,) + (1,) * (offsets.ndim - 1))
                    r = np.sqrt(np.sum(pos**2, axis=0))
                    lam_c = np.clip(r / R, self.lam[0], 1.0)
                    inside = r <= R
                    G = np.maximum(np.interp(lam_c, self.lam, self.G), density_floor)
                    U = np.interp(lam_c, self.lam, self.U)
                    U = np.where(r / R < self.lam[0], self.U[0] * r / R / self.lam[0], U)
                    rho = np.where(inside, rho_0 * G, rho_0)
                    u_r = np.where(inside, V * U, 0.0)
                    p_th = p_ambient + np.where(
                        inside, rho_0 * V**2 * np.interp(lam_c, self.lam, self.Pi_th), 0.0)
                    p_cr = np.where(inside, rho_0 * V**2 * np.interp(lam_c, self.lam, self.Pi_cr), 0.0)
                    unit = pos / np.maximum(r, 1e-300)
                    sums[0] += rho
                    sums[1:4] += rho * u_r * unit
                    sums[4] += p_th / (self.gamma_th - 1.0) + 0.5 * rho * u_r**2
                    sums[5] += p_cr / (self.gamma_cr - 1.0)
        sums /= n_sub**3
        rho = sums[0]
        velocity = sums[1:4] / rho
        p_th = (self.gamma_th - 1.0) * (sums[4] - 0.5 * np.sum(sums[1:4] ** 2, axis=0) / rho)
        return rho, velocity, p_th, sums[5]

    def cr_energy_fraction_shocked_before(self, q):
        """Fraction of the CR energy that sits in gas shocked while the shock
        radius was below ``q R(t)`` (i.e. shells with ``r_0 / R < q``)."""
        w = self.Pi_cr * self.lam**2 / (self.gamma_cr - 1.0)
        total = np.trapezoid(w, self.lam)
        sel = self.r0_over_R < q
        return float(np.trapezoid(np.where(sel, w, 0.0), self.lam) / total)


def _strong_shock_state(zeta, convention, gamma_th, gamma_cr):
    """Post-shock compression ``x`` and CR pressure fraction ``f = P_cr/P``."""

    def compression(f):
        k = (1.0 - f) * gamma_th / (gamma_th - 1.0) + f * gamma_cr / (gamma_cr - 1.0)
        return 2.0 * k - 1.0

    if convention == "total":
        x_cr = zeta * (gamma_cr - 1.0) / ((1.0 - zeta) * (gamma_th - 1.0))
        f = x_cr / (1.0 + x_cr)
        return compression(f), f
    if convention == "gas_rh":
        # gas-only strong shock: dissipated flux e_int,2 (V_s / x_g) / (rho_0 V_s^3)
        x_g = (gamma_th + 1.0) / (gamma_th - 1.0)
        d_gas = (1.0 - 1.0 / x_g) / (gamma_th - 1.0) / x_g
        if zeta == 0.0:
            return x_g, 0.0

        def residual(f):
            # e_cr,2 (V_s / x) = zeta d_gas rho_0 V_s^3, P_cr,2 = f (1 - 1/x)
            x = compression(f)
            return f * (1.0 - 1.0 / x) / (gamma_cr - 1.0) / x - zeta * d_gas

        f = brentq(residual, 0.0, 1.0 - 1e-12, xtol=1e-15)
        return compression(f), f
    raise ValueError(f"unknown convention {convention!r}")


def solve_sedov_two_fluid(
    zeta: float,
    convention: str = "total",
    gamma_th: float = 5.0 / 3.0,
    gamma_cr: float = 4.0 / 3.0,
    lam_min: float = 1e-3,
    rtol: float = 1e-10,
) -> SedovTwoFluidSolution:
    """Solve the self-similar two-fluid Sedov blast for injection fraction ``zeta``.

    ``zeta = 0`` gives the classic single-fluid Sedov solution (the CR entropy
    equation is dropped). ``lam_min`` is the inner end of the integration; the
    neglected core holds ``~Pi_c lam_min^3`` of the energy integral.
    """
    if not 0.0 <= zeta < 1.0:
        raise ValueError("zeta must be in [0, 1)")
    x, f = _strong_shock_state(zeta, convention, gamma_th, gamma_cr)
    p2 = 1.0 - 1.0 / x
    with_cr = f > 0.0

    # Unknowns U, ln G, ln Pi_th, ln Pi_cr against s = ln(lambda): G falls like
    # lambda^(3/(gamma-1)) towards the centre, which makes the linear form of
    # the ODEs stiff and ill-conditioned. Eliminating the entropy equations
    # (ln Pi_i)' = gamma_i (ln G)' + 3/a gives the explicit system below; its
    # denominator vanishes only at a sonic point, which the blast interior has not.
    def rhs(s, y):
        lam = np.exp(s)
        U, G, Pt = y[0], np.exp(y[1]), np.exp(y[2])
        Pc = np.exp(y[3]) if with_cr else 0.0
        a = U - lam
        num = 1.5 * U + 2.0 * a * U / lam - 3.0 * (Pt + Pc) / (a * G)
        den = (gamma_th * Pt + gamma_cr * Pc) / G - a**2
        g = num / den  # d ln G / d lambda
        dU = -2.0 * U / lam - a * g
        out = [lam * dU, lam * g, lam * (gamma_th * g + 3.0 / a)]
        if with_cr:
            out.append(lam * (gamma_cr * g + 3.0 / a))
        return out

    y0 = [1.0 - 1.0 / x, np.log(x), np.log(p2 * (1.0 - f))]
    if with_cr:
        y0.append(np.log(p2 * f))
    sol = solve_ivp(
        rhs, [0.0, np.log(lam_min)], y0, method="DOP853", rtol=rtol, atol=1e-12,
        dense_output=True,
    )
    if not sol.success:
        raise RuntimeError(f"similarity ODE integration failed: {sol.message}")

    # dense, increasing lambda grid (log-spaced towards the centre)
    lam = np.unique(np.concatenate([
        np.geomspace(lam_min, 1.0, 4000), np.linspace(0.5, 1.0, 4001),
    ]))
    y = sol.sol(np.log(lam))
    U, G, Pt = y[0], np.exp(y[1]), np.exp(y[2])
    Pc = np.exp(y[3]) if with_cr else np.zeros_like(lam)

    w_kin = 0.5 * G * U**2 * lam**2
    w_th = Pt / (gamma_th - 1.0) * lam**2
    w_cr = Pc / (gamma_cr - 1.0) * lam**2
    # core lam < lam_min: G, U -> 0, pressures ~ constant
    core = lam_min**3 / 3.0
    I_kin = np.trapezoid(w_kin, lam)
    I_th = np.trapezoid(w_th, lam) + Pt[0] / (gamma_th - 1.0) * core
    I_cr = np.trapezoid(w_cr, lam) + Pc[0] / (gamma_cr - 1.0) * core
    I = I_kin + I_th + I_cr
    xi_0 = (25.0 / (16.0 * np.pi * I)) ** 0.2

    # Lagrangian map: m(lam) = 3 int_0^lam G l^2 dl (= 1 at the shock)
    m = 3.0 * cumulative_trapezoid(G * lam**2, lam, initial=0.0)
    mass_closure = abs(m[-1] - 1.0)
    r0_over_R = np.cbrt(np.clip(m / m[-1], 0.0, 1.0))

    # Independent energy check (Sedov's energy integral): the energy inside a
    # sphere co-moving with fixed lambda is constant in time, so the energy flux
    # through it vanishes at every lambda,
    #   (G U^2/2 + eps) (U - lambda) + (Pi_th + Pi_cr) U = 0,
    # which the ODEs only satisfy implicitly (holds exactly at the shock).
    # Normalized by the post-shock work flux P_2 U_2: in the core the flux
    # itself goes to zero (~lambda) and a local ratio only measures the
    # interpolation of U there (G < 1e-9 for lambda < 0.01).
    eps = Pt / (gamma_th - 1.0) + Pc / (gamma_cr - 1.0)
    flux = (0.5 * G * U**2 + eps) * (U - lam) + (Pt + Pc) * U
    energy_closure = np.max(np.abs(flux)) / (p2 * (1.0 - 1.0 / x))

    # fluxes through the shock: 4 pi R^2 rho_0 V_s^3 = 4 pi 0.4^3 xi_0^5 E / t
    geometric = 4.0 * np.pi * 0.4**3 * xi_0**5
    e_cr2 = p2 * f / (gamma_cr - 1.0)
    e_int2 = p2 * (1.0 - f) / (gamma_th - 1.0) + e_cr2
    injection_rate = geometric * e_cr2 / x
    dissipation_rate = geometric * e_int2 / x
    x_g = (gamma_th + 1.0) / (gamma_th - 1.0)
    d_gas = (1.0 - 1.0 / x_g) / (gamma_th - 1.0) / x_g
    finder_dissipation_rate = geometric * d_gas

    return SedovTwoFluidSolution(
        zeta=zeta, convention=convention, gamma_th=gamma_th, gamma_cr=gamma_cr,
        xi_0=float(xi_0), compression=float(x),
        x_cr_0=float(f / (1.0 - f)),
        lam=lam, U=U, G=G, Pi_th=Pt, Pi_cr=Pc, r0_over_R=r0_over_R,
        f_kin=float(I_kin / I), f_th=float(I_th / I), f_cr=float(I_cr / I),
        injection_rate=float(injection_rate),
        finder_dissipation_rate=float(finder_dissipation_rate),
        dissipation_rate=float(dissipation_rate),
        energy_closure=float(energy_closure), mass_closure=float(mass_closure),
    )
