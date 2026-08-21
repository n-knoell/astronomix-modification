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

References:
    Pfrommer, C., Enßlin, T. A., & Jubelgas, M. (2006). "Simulating cosmic
    ray physics on a moving mesh". MNRAS, 367(1), 113-131.
    (The composite two-fluid Riemann problem used for their code
    validation; the polytropic gamma_cr = 4/3 grey CR closure and the
    "CRs compress adiabatically through shocks" assumption both match this
    repository's ``_modules/_cosmic_rays_grey`` module.)
"""

# general
from functools import partial

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


def _hugoniot_rho_star(p_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr):
    """Post-shock density solving the general-EOS Hugoniot (energy-jump)
    relation ``eps* - eps_K = 0.5 * (p* + p_K) * (1/rho_K - 1/rho*)``, with
    the CR pressure staying on its own adiabat through the shock
    (``p_cr* = p_cr_K * (rho*/rho_K)^gamma_cr``) and the thermal pressure
    absorbing whatever's left of the (given) total: ``p_th* = p* - p_cr*``.
    Monotonic in rho* for rho* > rho_K (the physical, compressive branch),
    so a bracketed bisection is unconditionally robust.
    """
    p_K = _total_pressure(rho_K, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
    eps_K = _specific_internal_energy(rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)

    def residual(rho_star):
        p_cr_star = _cr_pressure(rho_star, rho_K, p_cr_K, gamma_cr)
        # p_th_star can transiently go negative for trial rho_star far from
        # the true root (e.g. while scanning for a bracket, before the
        # search has localized near the physical solution) -- clamp to a
        # tiny positive floor so eps_star stays finite/evaluable there
        # instead of producing a NaN that would kill the bracket search.
        p_th_star = max(p_star - p_cr_star, 1e-300 * p_star)
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
    with np.errstate(over="ignore", invalid="ignore"):
        grid = rho_K * np.geomspace(1.0 + 1e-9, 50.0, 400)
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


def _f_K(p_star, p_K, rho_K, u_K, p_th_K, p_cr_K, gamma_th, gamma_cr, side):
    """Toro-style ``f_K(p*)`` (defined so that ``u* = u_L - f_L(p*)`` on the
    left, ``u* = u_R + f_R(p*)`` on the right; solving
    ``f_L(p*) + f_R(p*) + (u_R - u_L) = 0`` for p* is exactly the classic
    exact-Riemann-solver structure, generalized here to the composite EOS.
    """
    if p_star > p_K:
        rho_star = _hugoniot_rho_star(p_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
        j = np.sqrt((p_star - p_K) / (1.0 / rho_K - 1.0 / rho_star))
        return (p_star - p_K) / j
    else:
        rho_star = _rho_from_total_pressure(p_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
        u_star = _rarefaction_u_of_rho(rho_star, rho_K, u_K, p_th_K, p_cr_K, gamma_th, gamma_cr, side)
        return (u_K - u_star) if side == "L" else (u_star - u_K)


def _solve_star_region(rho_L, u_L, p_th_L, p_cr_L, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr):
    """Bisect for the star-region total pressure p*, then recover u*."""
    p_L = _total_pressure(rho_L, rho_L, p_th_L, p_cr_L, gamma_th, gamma_cr)
    p_R = _total_pressure(rho_R, rho_R, p_th_R, p_cr_R, gamma_th, gamma_cr)

    def f(p_star):
        f_L = _f_K(p_star, p_L, rho_L, u_L, p_th_L, p_cr_L, gamma_th, gamma_cr, "L")
        f_R = _f_K(p_star, p_R, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr, "R")
        return f_L + f_R + (u_R - u_L)

    p_lo, p_hi = 1e-12 * min(p_L, p_R), 10.0 * max(p_L, p_R)
    f_lo, f_hi = f(p_lo), f(p_hi)
    while f_lo * f_hi > 0.0:
        p_hi *= 2.0
        f_hi = f(p_hi)
    p_star = brentq(f, p_lo, p_hi, xtol=1e-14, rtol=1e-14)

    f_L = _f_K(p_star, p_L, rho_L, u_L, p_th_L, p_cr_L, gamma_th, gamma_cr, "L")
    f_R = _f_K(p_star, p_R, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr, "R")
    u_star = u_L - f_L
    u_star_check = u_R + f_R
    assert abs(u_star - u_star_check) < 1e-8 * max(1.0, abs(u_star)), (
        "Star-region velocity mismatch between the two sides -- solver bug."
    )
    return p_star, u_star, p_L, p_R


def _sample_side(
    S, p_star, u_star, rho_K, u_K, p_th_K, p_cr_K, gamma_th, gamma_cr, side
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
        rho_star = _hugoniot_rho_star(p_star, rho_K, p_th_K, p_cr_K, gamma_th, gamma_cr)
        j = np.sqrt((p_star - p_K) / (1.0 / rho_K - 1.0 / rho_star))
        # Shock speed from mass-flux conservation. j here is the *magnitude*
        # of the mass flux (positive by construction above); the signed
        # flux is +j for a left-side shock and -j for a right-side shock
        # (derived from momentum conservation p_K - p* = j_signed*(u*-u_K)
        # together with the known sign of u* - u_K on each side), giving
        # D = u_K - j_signed/rho_K = u_K + sign*j/rho_K.
        D = u_K + sign * j / rho_K
        upstream = (S < D) if side == "L" else (S > D)
        p_cr_star = _cr_pressure(rho_star, rho_K, p_cr_K, gamma_cr)
        p_th_star = p_star - p_cr_star
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

    Returns:
        (rho, u, p_th, p_cr): numpy arrays, same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    p_star, u_star, p_L, p_R = _solve_star_region(
        rho_L, u_L, p_th_L, p_cr_L, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr
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
            S_L, p_star, u_star, rho_L, u_L, p_th_L, p_cr_L, gamma_th, gamma_cr, "L"
        )
        rho[left], u[left], p_th[left], p_cr[left] = rho_l, u_l, p_th_l, p_cr_l

    if np.any(~left):
        S_R = S[~left]
        rho_r, u_r, p_th_r, p_cr_r = _sample_side(
            S_R, p_star, u_star, rho_R, u_R, p_th_R, p_cr_R, gamma_th, gamma_cr, "R"
        )
        rho[~left], u[~left], p_th[~left], p_cr[~left] = rho_r, u_r, p_th_r, p_cr_r

    return rho, u, p_th, p_cr
