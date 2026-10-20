"""
Semi-analytic two-fluid (gas + cosmic-ray) Riemann solver (Pfrommer,
Enßlin & Jubelgas 2006).

Generalizes the classic single-gamma exact Riemann solver
(``riemann_solver.py``) to a composite equation of state where the total
pressure is the sum of two power laws in density,

    P(rho) = P_th,ref * (rho / rho_ref)^gamma_th
            + P_cr,ref * (rho / rho_ref)^gamma_cr,

with the thermal component (gamma_th = 5/3) obeying the usual irreversible
Rankine-Hugoniot jump across a shock, while the cosmic-ray component
(gamma_cr = 4/3) is assumed collisionless and compresses **adiabatically
through the shock too** (Pfrommer et al. 2006's key simplifying assumption
-- no local CR thermalization at the shock, DSA injection is a separate,
not-yet-implemented process; see this module's docstring for the physical
picture). This closes the jump conditions for a mixture whose two
components have different adiabatic indices.

Because the composite EOS is not a single power law, the classic
closed-form rarefaction-fan and shock relations (Toro 2009, ch. 4) no
longer apply: the rarefaction-fan velocity is a numerically-integrated
Riemann invariant, and the shock jump requires a numerical (Hugoniot)
root-find for the post-shock density given the trial pressure, nested
inside the usual outer bisection for the star-region pressure. Implemented
in plain numpy/scipy (root-finding + quadrature) rather than JAX: this is a
one-shot reference-solution generator for tests, not part of the simulation
hot path, and does not need to be jit- or grad-compatible.

Validated (see ``pytests/cosmic_rays_grey/cr_shock_tube.py``'s docstring)
by setting the CR pressure to zero on both sides, which reduces this solver
to the pure single-gamma problem, and checking it reproduces
``riemann_solver._exact_riemann_ideal_gas`` to numerical-integration
precision across the full sampled profile -- strong evidence the general
bisection/quadrature machinery is correct before trusting it with a
genuine two-component EOS the repository has no independent reference for.

**CR acceleration at the shock** (``injection=ShockInjection(...)``, Phase B
test plan B7.4/B8.1/B8.5; Pfrommer et al. 2017, App. B/C): a fraction of the
energy the shock dissipates goes into CRs. Only the split of the post-shock
internal energy changes -- mass, momentum and total energy are conserved
across the shock as before -- so the same Hugoniot root-find applies with
``P_cr,2 = P_cr,1 x^gamma_cr + (gamma_cr - 1) e_inj`` and
``P_th,2 = P_2 - P_cr,2`` (``x = rho_2 / rho_1``). Two definitions of
``e_inj``:

- ``"total"`` (Pfrommer et al. 2017): ``e_inj = zeta e_diss`` with
  ``e_diss = e_th,2 + e_inj - e_th,1 x^gamma_th`` the dissipated energy
  density before the split, i.e. ``e_inj = zeta / (1 - zeta)
  (e_th,2 - e_th,1 x^gamma_th)``;
- ``"gas_rh"`` (``inject_crs_at_shocks``): the CR energy flux leaving the
  shock is ``zeta`` times the finder's gas-only Rankine-Hugoniot dissipated
  flux ``delta(M) rho_1 (M c_th,1)^3 / 2`` (Schaal & Springel 2015, eqs.
  8-9; ``c_th,1`` from the gas pressure alone). Per unit mass crossing the
  shock this is ``e_inj = zeta delta(M) rho_1 (M c_th,1)^3 x / (2 v_1)``
  with ``v_1`` the upstream speed in the shock frame. ``mach="true"`` uses
  ``M = v_1 / c_th,1``; ``mach="gas_pressure"`` uses the Mach number the
  finder reads from the *gas* pressure jump ``P_th,2 / P_th,1`` (gas-only
  RH), which the injected CR pressure lowers -- the self-consistent state a
  shock that is resolved and sampled downstream of the deposit converges to;
  ``mach="velocity"`` the Mach number the finder reads from the normal-velocity
  jump ``v_1 (1 - 1/x)`` (gas-only RH, ``mach_estimator="velocity"``, the DSA
  default since 2026-10-19), which the extra compression of a CR-loaded shock
  raises.

``zeta`` may be a function of the Mach number used (the KR13 model, B8.1);
the root-find then solves the fixed point ``zeta = eta(M(zeta))``.
``pfrommer_shock_structure`` returns the wave speeds, star states, Mach
numbers and injected energy.

References:
    Pfrommer, C., Pakmor, R., Schaal, K., Simpson, C. M., & Springel, V.
    (2017). MNRAS, 465, 4500 (App. B: thermal gas with CR injection; App. C:
    with upstream CRs; Table 1).
    Schaal, K., & Springel, V. (2015). MNRAS, 446, 3992 (``delta(M)``).
    Pfrommer, C., Enßlin, T. A., & Jubelgas, M. (2006). "Simulating cosmic
    ray physics on a moving mesh". MNRAS, 367(1), 113-131.
    (The composite two-fluid Riemann problem used for their code
    validation; the polytropic gamma_cr = 4/3 grey CR closure and the
    "CRs compress adiabatically through shocks" assumption both match this
    repository's ``_modules/_cosmic_rays_grey`` module.)
"""

# general
from functools import partial
from typing import Callable, NamedTuple, Union

# numerics
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def _total_pressure(rho, rho_ref, p_th_ref, p_cr_ref, gamma_th, gamma_cr):
    """Composite total pressure P_th(rho) + P_cr(rho), both power laws in
    rho anchored at the reference state (rho_ref, p_th_ref, p_cr_ref)."""
    x = rho / rho_ref
    return p_th_ref * x**gamma_th + p_cr_ref * x**gamma_cr


def _cr_pressure(rho, rho_ref, p_cr_ref, gamma_cr):
    """CR pressure alone, adiabatic invariant P_cr(rho) = P_cr,ref * (rho / rho_ref)^gamma_cr."""
    return p_cr_ref * (rho / rho_ref) ** gamma_cr


def _sound_speed(rho, rho_ref, p_th_ref, p_cr_ref, gamma_th, gamma_cr):
    """Composite adiabatic sound speed a(rho) = sqrt(dP/drho), both
    components varying isentropically (valid inside a smooth rarefaction;
    NOT valid across a shock, where only the CR component stays adiabatic)."""
    x = rho / rho_ref
    dP_drho = (
        gamma_th * p_th_ref / rho_ref * x ** (gamma_th - 1.0)
        + gamma_cr * p_cr_ref / rho_ref * x ** (gamma_cr - 1.0)
    )
    return np.sqrt(np.maximum(dP_drho, 0.0))


def _specific_internal_energy(rho, p_th, p_cr, gamma_th, gamma_cr):
    """Specific (per unit mass) internal energy: thermal + CR, each with
    its own adiabatic index, eps = p_th / ((gamma_th - 1) rho) + p_cr /
    ((gamma_cr - 1) rho)."""
    return p_th / ((gamma_th - 1.0) * rho) + p_cr / ((gamma_cr - 1.0) * rho)


def _rho_from_total_pressure(p_target, rho_ref, p_th_ref, p_cr_ref, gamma_th, gamma_cr):
    """Invert the composite (isentropic) P(rho) relation for rho, given a
    target total pressure -- used for rarefaction-branch sampling, where
    both components individually stay on their own adiabat. Monotonically
    increasing in rho, so a bracketed bisection is unconditionally robust."""
    def residual(rho):
        return _total_pressure(rho, rho_ref, p_th_ref, p_cr_ref, gamma_th, gamma_cr) - p_target

    lo, hi = 1e-12 * rho_ref, rho_ref
    while residual(hi) < 0.0:
        hi *= 2.0
    while residual(lo) > 0.0:
        lo *= 0.5
    return brentq(residual, lo, hi, xtol=1e-14, rtol=1e-14)


def _rarefaction_u_of_rho(rho, rho_K, u_K, p_th_K, p_cr_K, gamma_th, gamma_cr, side):
    """Velocity at density ``rho`` along the isentropic simple wave
    connecting to state K, via the (numerically integrated) Riemann
    invariant -- generalizes Toro's closed-form
    ``u* = u_K -+ g4 * a_K * ((p*/p_K)^g1 - 1)`` to a composite EOS with no
    closed-form integral.

    Args:
        side: ``"L"`` for the left-family (C-) wave (u decreases as rho
            drops below rho_K), ``"R"`` for the right-family (C+) wave (u
            increases as rho drops below rho_K).
    """
    integrand = lambda r: _sound_speed(  # noqa: E731
        r, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr
    ) / r
    integral, _ = quad(integrand, rho, rho_K, limit=200)
    # integral = int_{rho}^{rho_K} a/rho' drho' >= 0 (rho <= rho_K inside a
    # rarefaction connecting to a lower density). Along a left rarefaction
    # (u - a family), u increases as the gas expands toward lower rho
    # (accelerating toward the lower-pressure side); along a right
    # rarefaction (u + a family) u decreases -- i.e. the opposite sign.
    return u_K + integral if side == "L" else u_K - integral


class ShockInjection(NamedTuple):
    """CR acceleration at shocks (see the module docstring).

    Attributes:
        zeta: Efficiency, a constant or a function of the Mach number
            (``"total"``: the composite-sound-speed Mach number; ``"gas_rh"``:
            the one selected by ``mach``).
        convention: ``"total"`` (Pfrommer et al. 2017) or ``"gas_rh"``
            (``inject_crs_at_shocks``).
        mach: ``"gas_rh"`` only: ``"true"`` (``v_1 / c_th,1``) or
            ``"gas_pressure"`` (from ``P_th,2 / P_th,1``, what the finder read
            until 2026-10-19) or ``"velocity"`` (from the velocity jump, what
            DSA's finder reads since then).
        mach_min: No injection at or below this Mach number (the finder's
            ``mach_min``; KR13 is zero below 2 by itself).
    """

    zeta: Union[float, Callable[[float], float]]
    convention: str = "total"
    mach: str = "true"
    mach_min: float = 1.0


def thermalization_efficiency(mach, gamma):
    """Gas-only RH dissipated fraction ``delta(M)`` of the incoming kinetic flux
    (Schaal & Springel 2015, eqs. 8-9), numpy copy of
    ``shock_finder3D._energy_dissipation._thermalization_efficiency``."""
    mach = np.asarray(mach, dtype=float)
    m2 = np.maximum(mach, 1.0) ** 2
    x = (gamma + 1.0) * m2 / ((gamma - 1.0) * m2 + 2.0)
    y = (2.0 * gamma * m2 - (gamma - 1.0)) / (gamma + 1.0)
    delta = 2.0 / (gamma * (gamma - 1.0) * m2 * x) * (y - x**gamma)
    return np.where(mach > 1.0, np.maximum(delta, 0.0), 0.0)


def _gas_rh_mach_from_pressure(pressure_ratio, gamma):
    """Gas-only RH Mach number from the pressure jump (the finder's
    ``mach_from_pressure_samples``)."""
    return np.sqrt(max((gamma + 1.0) * pressure_ratio + (gamma - 1.0), 0.0) / (2.0 * gamma))


def _gas_rh_mach_from_velocity(jump_over_c, gamma):
    """Gas-only RH Mach number from the velocity jump over the upstream sound
    speed (the finder's ``mach_from_velocity_jump``)."""
    y = 0.5 * (gamma + 1.0) * jump_over_c
    return 0.5 * (y + np.sqrt(y * y + 4.0))


def _zeta_at(injection, mach):
    if mach <= injection.mach_min:
        return 0.0
    return float(injection.zeta(mach)) if callable(injection.zeta) else float(injection.zeta)


def _post_shock_partition(p_star, rho_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr, injection):
    """Thermal/CR split of the post-shock total pressure ``p_star`` at density
    ``rho_star`` (module docstring). Returns ``(p_th*, p_cr*, e_inj, M_used)``;
    ``M_used`` is the Mach number the efficiency was evaluated at (0 without
    injection)."""
    x = rho_star / rho_K
    p_cr_ad = p_cr_K * x**gamma_cr
    p_rest = p_star - p_cr_ad
    if injection is None or p_star <= p_th_K + p_cr_K or rho_star <= rho_K:
        return p_rest, p_cr_ad, 0.0, 0.0
    p_K = p_th_K + p_cr_K
    v1 = np.sqrt((p_star - p_K) / (1.0 / rho_K - 1.0 / rho_star)) / rho_K
    if injection.convention == "total":
        c1 = np.sqrt((gamma_th * p_th_K + gamma_cr * p_cr_K) / rho_K)
        mach = v1 / c1
        r = _zeta_at(injection, mach)
        r = r / (1.0 - r)
        e_th_ad = p_th_K * x**gamma_th / (gamma_th - 1.0)
        e_inj = r * (p_rest / (gamma_th - 1.0) - e_th_ad) / (
            1.0 + r * (gamma_cr - 1.0) / (gamma_th - 1.0)
        )
        e_inj = max(e_inj, 0.0)
        return p_rest - (gamma_cr - 1.0) * e_inj, p_cr_ad + (gamma_cr - 1.0) * e_inj, e_inj, mach
    if injection.convention != "gas_rh":
        raise ValueError(f"unknown injection convention {injection.convention!r}")
    c_th = np.sqrt(gamma_th * p_th_K / rho_K)

    def e_inj_at(mach):
        flux = _zeta_at(injection, mach) * thermalization_efficiency(mach, gamma_th) * (
            0.5 * rho_K * (mach * c_th) ** 3
        )
        return float(flux) * x / v1

    if injection.mach == "true":
        mach = v1 / c_th
        e_inj = e_inj_at(mach)
    elif injection.mach == "gas_pressure":
        # p_th + (gamma_cr - 1) e_inj(M(p_th)) = p_rest, monotone in p_th; KR13's zeta(M)
        # is monotone too, so the root is unique.
        def residual(p_th):
            return p_th + (gamma_cr - 1.0) * e_inj_at(
                _gas_rh_mach_from_pressure(p_th / p_th_K, gamma_th)
            ) - p_rest

        if p_rest <= p_th_K or residual(p_rest) <= 0.0:
            return p_rest, p_cr_ad, 0.0, _gas_rh_mach_from_pressure(max(p_rest, 0.0) / p_th_K, gamma_th)
        p_th = brentq(residual, p_th_K, p_rest, xtol=1e-15, rtol=1e-14) if residual(p_th_K) < 0.0 else p_th_K
        mach = _gas_rh_mach_from_pressure(p_th / p_th_K, gamma_th)
        e_inj = (p_rest - p_th) / (gamma_cr - 1.0)
    elif injection.mach == "velocity":
        # the velocity jump is fixed by (p_star, rho_star) alone, so no inner root-find
        mach = _gas_rh_mach_from_velocity(v1 * (1.0 - 1.0 / x) / c_th, gamma_th)
        e_inj = e_inj_at(mach)
    else:
        raise ValueError(f"unknown injection mach {injection.mach!r}")
    return p_rest - (gamma_cr - 1.0) * e_inj, p_cr_ad + (gamma_cr - 1.0) * e_inj, e_inj, mach


def _hugoniot_rho_star(p_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr, injection=None):
    """Post-shock density solving the general-EOS Hugoniot (energy-jump)
    relation ``eps* - eps_K = 0.5 * (p* + p_K) * (1/rho_K - 1/rho*)``, with
    the CR pressure staying on its own adiabat through the shock
    (``p_cr* = p_cr_K * (rho*/rho_K)^gamma_cr``) and the thermal pressure
    absorbing whatever's left of the (given) total: ``p_th* = p* - p_cr*``.
    Monotonic in rho* for rho* > rho_K (the physical, compressive branch),
    so a bracketed bisection is unconditionally robust.

    With ``injection`` the post-shock split follows ``_post_shock_partition``
    (CR acceleration; the total energy jump is unchanged).
    """
    p_K = _total_pressure(rho_K, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
    eps_K = _specific_internal_energy(rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)

    def residual(rho_star):
        p_th_star, p_cr_star, _, _ = _post_shock_partition(
            p_star, rho_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr, injection
        )
        # p_th_star can transiently go negative for trial rho_star far from
        # the true root (e.g. while scanning for a bracket, before the
        # search has localized near the physical solution) -- clamp to a
        # tiny positive floor so eps_star stays finite/evaluable there
        # instead of producing a NaN that would kill the bracket search.
        p_th_star = max(p_th_star, 1e-300 * p_star)
        eps_star = _specific_internal_energy(rho_star, p_th_star, p_cr_star, gamma_th, gamma_cr)
        return (eps_star - eps_K) - 0.5 * (p_star + p_K) * (1.0 / rho_K - 1.0 / rho_star)

    # residual's sign as a function of rho_star isn't a fixed a priori
    # direction (unlike _total_pressure, a manifestly increasing sum of
    # positive powers), and for a strong shock the true root sits at a
    # *bounded* compression ratio even as p_star grows arbitrarily large
    # (the classic strong-shock compression-ratio limit) -- so a naive
    # "double hi until signs differ" search can run past the root without
    # ever detecting the crossing and diverge/overflow. Scan a wide,
    # log-spaced grid for the first sign change instead, then bisect within
    # that bracket -- robust regardless of where the root actually falls.
    # Physically, even an arbitrarily strong shock has a *bounded*
    # compression ratio (the classic strong-shock limit, ~(gamma+1)/(gamma-1)
    # per component) -- so the scan grid only needs to span a modest
    # multiple of rho_K, not an unbounded one; a wide grid risks float64
    # overflow in intermediate powers long before it would be needed.
    # rho_K itself starts the grid (residual > 0 there for p_star > p_K): an infinitesimal
    # shock (e.g. the weak left wave of a tube started from an RH state) has its root below
    # 1 + 1e-9.
    with np.errstate(over="ignore", invalid="ignore"):
        grid = rho_K * np.concatenate([[1.0], np.geomspace(1.0 + 1e-9, 50.0, 400)])
        residuals = np.array([residual(r) for r in grid])
    finite = np.isfinite(residuals)
    sign = np.where(finite, np.sign(residuals), 0.0)
    sign_changes = np.nonzero(
        finite[:-1] & finite[1:] & (sign[:-1] * sign[1:] < 0.0)
    )[0]
    if len(sign_changes) == 0:
        raise RuntimeError(
            f"_hugoniot_rho_star: no sign change found for p_star={p_star}, "
            f"rho_K={rho_K} -- widen the scan grid."
        )
    i = sign_changes[0]
    return brentq(residual, grid[i], grid[i + 1], xtol=1e-14, rtol=1e-14)


def _shock_mass_flux(p_star, p_K, rho_K, rho_star, p_th_K, p_cr_K, gamma_th, gamma_cr):
    """Mass flux ``j`` through a shock; the acoustic limit ``rho_K a_K`` when the shock is
    so weak that ``rho_star`` rounds to ``rho_K``."""
    dv = 1.0 / rho_K - 1.0 / rho_star
    if dv <= 1e-14 / rho_K:
        return rho_K * _sound_speed(rho_K, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
    return np.sqrt((p_star - p_K) / dv)


def _f_K(p_star, p_K, rho_K, u_K, p_th_K, p_cr_K, gamma_th, gamma_cr, side, injection=None):
    """Toro-style ``f_K(p*)`` (defined so that ``u* = u_L - f_L(p*)`` on the
    left, ``u* = u_R + f_R(p*)`` on the right; solving
    ``f_L(p*) + f_R(p*) + (u_R - u_L) = 0`` for p* is exactly the classic
    exact-Riemann-solver structure, generalized here to the composite EOS.
    """
    if p_star > p_K:
        rho_star = _hugoniot_rho_star(p_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr, injection)
        j = _shock_mass_flux(p_star, p_K, rho_K, rho_star, p_th_K, p_cr_K, gamma_th, gamma_cr)
        return (p_star - p_K) / j
    else:
        rho_star = _rho_from_total_pressure(p_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
        u_star = _rarefaction_u_of_rho(rho_star, rho_K, u_K, p_th_K, p_cr_K, gamma_th, gamma_cr, side)
        return (u_K - u_star) if side == "L" else (u_star - u_K)


def _solve_star_region(
    rho_L, u_L, p_th_L, p_cr_L, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr, injection=None
):
    """Bisect for the star-region total pressure p*, then recover u*."""
    p_L = _total_pressure(rho_L, rho_L, p_th_L, p_cr_L, gamma_th, gamma_cr)
    p_R = _total_pressure(rho_R, rho_R, p_th_R, p_cr_R, gamma_th, gamma_cr)

    def f(p_star):
        f_L = _f_K(p_star, p_L, rho_L, u_L, p_th_L, p_cr_L, gamma_th, gamma_cr, "L", injection)
        f_R = _f_K(p_star, p_R, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr, "R", injection)
        return f_L + f_R + (u_R - u_L)

    p_lo, p_hi = 1e-12 * min(p_L, p_R), 10.0 * max(p_L, p_R)
    f_lo, f_hi = f(p_lo), f(p_hi)
    while f_lo * f_hi > 0.0:
        p_hi *= 2.0
        f_hi = f(p_hi)
    p_star = brentq(f, p_lo, p_hi, xtol=1e-14, rtol=1e-14)

    f_L = _f_K(p_star, p_L, rho_L, u_L, p_th_L, p_cr_L, gamma_th, gamma_cr, "L", injection)
    f_R = _f_K(p_star, p_R, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr, "R", injection)
    u_star = u_L - f_L
    u_star_check = u_R + f_R
    assert abs(u_star - u_star_check) < 1e-8 * max(1.0, abs(u_star)), (
        "Star-region velocity mismatch between the two sides -- solver bug."
    )
    return p_star, u_star, p_L, p_R


def _sample_side(
    S, p_star, u_star, rho_K, u_K, p_th_K, p_cr_K, gamma_th, gamma_cr, side, injection=None
):
    """Sample density/velocity/thermal-/CR-pressure at self-similar
    coordinate(s) ``S`` for one side (K = L or R), covering the uniform
    upstream region, a shock, or a rarefaction fan (head/interior/tail) as
    appropriate. ``S`` is a numpy array; returns arrays of the same shape.
    """
    p_K = _total_pressure(rho_K, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
    a_K = _sound_speed(rho_K, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
    sign = -1.0 if side == "L" else 1.0

    rho = np.empty_like(S)
    u = np.empty_like(S)
    p_th = np.empty_like(S)
    p_cr = np.empty_like(S)

    if p_star > p_K:
        # Shock: piecewise constant either side of the shock speed D.
        rho_star = _hugoniot_rho_star(p_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr, injection)
        j = _shock_mass_flux(p_star, p_K, rho_K, rho_star, p_th_K, p_cr_K, gamma_th, gamma_cr)
        # Shock speed from mass-flux conservation. j here is the *magnitude*
        # of the mass flux (positive by construction above); the signed
        # flux is +j for a left-side shock and -j for a right-side shock
        # (derived from momentum conservation p_K - p* = j_signed*(u*-u_K)
        # together with the known sign of u* - u_K on each side), giving
        # D = u_K - j_signed/rho_K = u_K + sign*j/rho_K.
        D = u_K + sign * j / rho_K
        upstream = (S < D) if side == "L" else (S > D)
        p_th_star, p_cr_star, _, _ = _post_shock_partition(
            p_star, rho_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr, injection
        )
        rho[:] = np.where(upstream, rho_K, rho_star)
        u[:] = np.where(upstream, u_K, u_star)
        p_th[:] = np.where(upstream, p_th_K, p_th_star)
        p_cr[:] = np.where(upstream, p_cr_K, p_cr_star)
    else:
        # Rarefaction: head/tail speeds bound the fan.
        rho_star = _rho_from_total_pressure(p_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
        a_star = _sound_speed(rho_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
        S_head = u_K + sign * a_K
        S_tail = u_star + sign * a_star
        upstream = (S < S_head) if side == "L" else (S > S_head)
        beyond_tail = (S > S_tail) if side == "L" else (S < S_tail)
        p_cr_star = _cr_pressure(rho_star, rho_K, p_cr_K, gamma_cr)
        p_th_star = p_star - p_cr_star
        rho[:] = np.where(upstream, rho_K, np.where(beyond_tail, rho_star, np.nan))
        u[:] = np.where(upstream, u_K, np.where(beyond_tail, u_star, np.nan))
        p_th[:] = np.where(upstream, p_th_K, np.where(beyond_tail, p_th_star, np.nan))
        p_cr[:] = np.where(upstream, p_cr_K, np.where(beyond_tail, p_cr_star, np.nan))

        fan = ~upstream & ~beyond_tail
        for i in np.nonzero(fan)[0]:
            S_target = S[i]

            def residual(rho_trial, S_target=S_target):
                u_trial = _rarefaction_u_of_rho(
                    rho_trial, rho_K, u_K, p_th_K, p_cr_K, gamma_th, gamma_cr, side
                )
                a_trial = _sound_speed(rho_trial, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
                return (u_trial + sign * a_trial) - S_target

            lo, hi = (rho_star, rho_K) if rho_star < rho_K else (rho_K, rho_star)
            rho_i = brentq(residual, lo, hi, xtol=1e-13, rtol=1e-13)
            u_i = _rarefaction_u_of_rho(
                rho_i, rho_K, u_K, p_th_K, p_cr_K, gamma_th, gamma_cr, side
            )
            p_cr_i = _cr_pressure(rho_i, rho_K, p_cr_K, gamma_cr)
            p_th_i = _total_pressure(rho_i, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr) - p_cr_i
            rho[i], u[i], p_th[i], p_cr[i] = rho_i, u_i, p_th_i, p_cr_i

    return rho, u, p_th, p_cr


def pfrommer_riemann_solution(
    rho_L, u_L, p_th_L, p_cr_L,
    rho_R, u_R, p_th_R, p_cr_R,
    gamma_th, gamma_cr,
    x, t, x0,
    injection=None,
):
    """Semi-analytic two-fluid Riemann solution (Pfrommer et al. 2006),
    sampled at positions ``x`` and time ``t > 0``.

    Args:
        rho_L, u_L, p_th_L, p_cr_L: Left-state density, velocity, thermal
            pressure, and CR pressure.
        rho_R, u_R, p_th_R, p_cr_R: Right-state counterparts.
        gamma_th: Thermal-gas adiabatic index.
        gamma_cr: CR adiabatic index.
        x: Positions (array-like) at which to evaluate the solution.
        t: Time at which to evaluate the solution (must be > 0).
        x0: Position of the initial diaphragm.
        injection: ``ShockInjection`` for CR acceleration at shocks (module
            docstring), or None (CRs only compressed adiabatically).

    Returns:
        (rho, u, p_th, p_cr): numpy arrays, same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    p_star, u_star, p_L, p_R = _solve_star_region(
        rho_L, u_L, p_th_L, p_cr_L, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr, injection
    )

    S = (x - x0) / t
    left = S <= u_star

    rho = np.empty_like(x)
    u = np.empty_like(x)
    p_th = np.empty_like(x)
    p_cr = np.empty_like(x)

    if np.any(left):
        S_L = S[left]
        rho_l, u_l, p_th_l, p_cr_l = _sample_side(
            S_L, p_star, u_star, rho_L, u_L, p_th_L, p_cr_L, gamma_th, gamma_cr, "L", injection
        )
        rho[left], u[left], p_th[left], p_cr[left] = rho_l, u_l, p_th_l, p_cr_l

    if np.any(~left):
        S_R = S[~left]
        rho_r, u_r, p_th_r, p_cr_r = _sample_side(
            S_R, p_star, u_star, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr, "R", injection
        )
        rho[~left], u[~left], p_th[~left], p_cr[~left] = rho_r, u_r, p_th_r, p_cr_r

    return rho, u, p_th, p_cr


def pfrommer_shock_structure(
    rho_L, u_L, p_th_L, p_cr_L,
    rho_R, u_R, p_th_R, p_cr_R,
    gamma_th, gamma_cr,
    injection=None,
):
    """Star region and right-going shock of the two-fluid Riemann problem
    (the tubes of B7.4/B8.1: left rarefaction or weak wave, right shock).

    Returns:
        dict with ``p_star``, ``u_star``, the shocked state ``rho_2``,
        ``p_th_2``, ``p_cr_2``, the injected CR energy density ``e_inj``,
        ``v_shock`` (lab frame), ``compression``, the Mach numbers
        ``mach_gas`` (``v_1 / c_th,1``), ``mach_composite`` (``v_1 / c_1``,
        Pfrommer et al.'s) and ``mach_gas_pressure`` (gas-only RH Mach of
        ``P_th,2 / P_th,1``), ``mach_gas_velocity`` (gas-only RH Mach of the
        velocity jump), the injected CR energy
        flux ``f_inj = e_inj (v_shock - u_2)`` and the finder's gas-RH
        dissipated flux at the true Mach number ``f_diss_gas``; the left
        star density ``rho_star_L`` and the left wave type.
    """
    p_star, u_star, _, p_R = _solve_star_region(
        rho_L, u_L, p_th_L, p_cr_L, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr, injection
    )
    if not p_star > p_R:
        raise ValueError("no right-going shock in this Riemann problem")
    rho_2 = _hugoniot_rho_star(p_star, rho_R, p_th_R, p_cr_R, gamma_th, gamma_cr, injection)
    p_th_2, p_cr_2, e_inj, _ = _post_shock_partition(
        p_star, rho_2, rho_R, p_th_R, p_cr_R, gamma_th, gamma_cr, injection
    )
    j = np.sqrt((p_star - p_R) / (1.0 / rho_R - 1.0 / rho_2))
    v1 = j / rho_R
    c_th = np.sqrt(gamma_th * p_th_R / rho_R)
    c_comp = np.sqrt((gamma_th * p_th_R + gamma_cr * p_cr_R) / rho_R)
    mach_gas = v1 / c_th
    p_L = p_th_L + p_cr_L
    if p_star > p_L:
        rho_star_L = _hugoniot_rho_star(p_star, rho_L, p_th_L, p_cr_L, gamma_th, gamma_cr, injection)
        left_wave = "shock"
    else:
        rho_star_L = _rho_from_total_pressure(p_star, rho_L, p_th_L, p_cr_L, gamma_th, gamma_cr)
        left_wave = "rarefaction"
    return dict(
        p_star=p_star,
        u_star=u_star,
        rho_2=rho_2,
        p_th_2=p_th_2,
        p_cr_2=p_cr_2,
        e_inj=e_inj,
        v_shock=u_R + v1,
        compression=rho_2 / rho_R,
        mach_gas=mach_gas,
        mach_composite=v1 / c_comp,
        mach_gas_pressure=_gas_rh_mach_from_pressure(p_th_2 / p_th_R, gamma_th),
        mach_gas_velocity=_gas_rh_mach_from_velocity(v1 * (1.0 - rho_R / rho_2) / c_th, gamma_th),
        f_inj=e_inj * v1 * rho_R / rho_2,
        f_diss_gas=float(thermalization_efficiency(mach_gas, gamma_th)) * 0.5 * rho_R * v1**3,
        rho_star_L=rho_star_L,
        left_wave=left_wave,
    )
