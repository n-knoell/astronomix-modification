"""
Semi-analytic cross-checks for the steady, diffusion-mediated CR-modified
shock structure (Phase B ladder item 9: "1D steady CR-driven flow /
CR-modified shock structure"; see
``astronomix/_modules/_cosmic_rays_grey/PROGRESS.md``'s 2026-08-27 entry for
the full derivation this module implements).

Physical picture. Run a stationary (or slowly-relaxing) 1D shock with both
``diffusive_shock_acceleration`` (ladder items 7-8) and ``diffusive_relaxation``
(ladder item 4's quasi-steady Fick's-law limit, ``F_cr ~= -diffusion_coefficient
* de_cr/dx``) turned on together for the first time. CR pressure accelerated at
the shock diffuses upstream, building a smooth *precursor* that pre-decelerates
and pre-heats the incoming gas before it reaches the (numerically thin, i.e.
diffusion-unresolved) viscous subshock -- the textbook signature of a
CR-modified shock (Drury & Voelk 1981; Axford, Leer & Skadron 1977; see also
the review in Malkov & Drury 2001).

This module does **not** independently solve the full boundary-value problem
from idealized x = -infinity/+infinity conditions (that requires shooting from
an unstable fixed point with an eigenvector-selected perturbation, matching to
a second, a priori unknown, downstream fixed point -- a much harder and more
fragile numerical problem than is needed here). Instead it provides the
*local* governing equations so the actual simulation's own converged profile
can be checked against them directly, in the same spirit as ladder item 8's
"exact formula cross-check" (feed the code's own state into the documented
formula, compare to what the code produced) -- this is what
``pytests/cosmic_rays_grey/cr_modified_shock_structure.py`` does, not an
independently-integrated reference profile.

Two pieces, both derived (not guessed) from this module's actual coupled
equations -- see ``cr_precursor_ode_rhs``'s and
``modified_rankine_hugoniot_with_cr_injection``'s docstrings for the
step-by-step derivations:

1. :func:`cr_precursor_ode_rhs` -- the local ODE the smooth precursor upstream
   of the subshock must satisfy (no injection there; injection is localized at
   the subshock).
2. :func:`modified_rankine_hugoniot_with_cr_injection` -- the (mass, momentum,
   energy) jump conditions across the subshock itself, given the CR energy
   flux DSA injects there. Assumes ``P_cr`` continuous through the subshock
   (a resolved precursor).
3. :func:`rankine_hugoniot_with_downstream_crs` and
   :func:`cr_energy_flux_jump_bounds` -- the jump when the CRs are instead
   created behind the shock with no precursor, so ``P_cr`` itself jumps
   (the pytest's Test A): total momentum/energy conservation for the gas
   state, plus a bracket on the CR energy-flux jump that checks the
   injected amount.

Plain numpy/scipy (root-finding only) -- a one-shot reference-formula module
like ``pfrommer_riemann_solver.py``, not part of the simulation hot path.
"""

# numerics
import numpy as np
from scipy.optimize import brentq


def cr_precursor_ode_rhs(
    u: float,
    p_cr: float,
    f_cr: float,
    mass_flux: float,
    momentum_flux: float,
    gamma_gas: float,
    gamma_cr: float,
    diffusion_coefficient: float,
) -> tuple:
    """Local steady-state ODE for the smooth CR precursor (no injection here).

    Derivation (see PROGRESS.md ladder item 9 for the full algebra): in the
    smooth precursor, mass flux ``mass_flux = rho * u`` and total momentum
    flux ``momentum_flux = rho * u^2 + P_gas + P_cr`` are exact, position-
    independent first integrals of the steady 1D equations (continuity; gas
    momentum with the ``-grad(P_cr)`` coupling from
    ``cr_grey_sources.cr_pressure_gradient_source``) -- no injection or
    diffusion term appears in either, so they hold everywhere, precursor and
    subshock and downstream alike.

    Two more equations come from the actual coupled per-species energy
    equations (steady, 1D, using ``rho = mass_flux / u`` and ``P_gas =
    momentum_flux - mass_flux * u - p_cr`` to eliminate ``rho``/``P_gas``):

    - Gas energy (``cr_pressure_gradient_source``'s ``-u * dP_cr/dx`` work
      term): ``d/dx[(1/2) rho u^3 + gamma_gas/(gamma_gas-1) P_gas u] = -u *
      dP_cr/dx``. Solving for ``du/dx`` gives the first return value.
    - CR energy (``grey_cr_flux_terms``'s ``u * e_cr + F_cr`` flux --
      *not* an enthalpy-like ``u * P_cr`` term, unlike the gas equation, see
      that function's flux formula -- plus ``cr_adiabatic_work_source``'s
      ``-P_cr * du/dx`` work term, plus ``F_cr = -diffusion_coefficient *
      de_cr/dx`` from ``cr_flux_relaxation_source``'s quasi-steady limit,
      ladder item 4): ``d/dx[u * P_cr/(gamma_cr-1) + F_cr] = -P_cr * du/dx``.
      Solving for ``dF_cr/dx`` gives the third return value, using the
      already-solved ``du/dx`` and ``dP_cr/dx = -(gamma_cr - 1) F_cr /
      diffusion_coefficient``.

    Verified (ad hoc script, during development, not committed): the point
    ``(u, P_cr, F_cr) = (u1, 0, 0)`` (far-upstream ambient conditions) is an
    exact fixed point of this ODE; linearizing around it gives a 2x2 system
    in ``(P_cr, F_cr)`` with eigenvalues ``0`` and ``u1 / diffusion_coefficient``
    (the growing precursor mode) and growing-eigenvector
    direction ``F_cr = -[u1 / (gamma_cr - 1)] * P_cr``. Integrating this ODE
    numerically from a small perturbation along that exact eigenvector
    direction (``scipy.integrate.solve_ivp``, tight tolerances) gives a
    physically sensible precursor (``u`` decreasing smoothly, ``P_cr``
    growing smoothly and staying positive) along which the total energy flux
    ``E = (1/2) rho u^3 + gamma_gas/(gamma_gas-1) P_gas u + gamma_cr/(gamma_cr-1)
    P_cr u + F_cr`` -- conserved by construction only in the *sum* of the gas
    and CR energy equations, not by either equation individually, so this is
    a genuine cross-check that ``du_dx`` (derived from the gas equation) and
    ``df_cr_dx`` (derived from the CR equation) are mutually consistent --
    stays constant to ~1.6e-7 relative error even after ``u`` has dropped by
    30% and ``P_cr`` has grown by four orders of magnitude from its seed
    perturbation.

    Args:
        u: Local gas velocity (shock-frame).
        p_cr: Local CR pressure ``(gamma_cr - 1) * e_cr``.
        f_cr: Local CR flux.
        mass_flux: The conserved ``rho * u``.
        momentum_flux: The conserved ``rho * u^2 + P_gas + P_cr``.
        gamma_gas: Gas adiabatic index.
        gamma_cr: CR adiabatic index.
        diffusion_coefficient: ``CosmicRayGreyParams.diffusion_coefficient``
            (the quasi-steady-limit ``e_cr`` diffusivity, ladder item 4; until
            2026-10-04 this module used the old ``P_cr``-diffusivity
            convention, ``dP_cr/dx = -F_cr / diffusion_coefficient``).

    Returns:
        ``(du_dx, dp_cr_dx, df_cr_dx)``.
    """
    A = gamma_gas / (gamma_gas - 1.0)

    dp_cr_dx = -(gamma_cr - 1.0) * f_cr / diffusion_coefficient

    denom = (1.0 - 2.0 * A) * mass_flux * u + A * (momentum_flux - p_cr)
    du_dx = u * (A - 1.0) * dp_cr_dx / denom

    df_cr_dx = (
        -p_cr * du_dx * gamma_cr / (gamma_cr - 1.0)
        - u * dp_cr_dx / (gamma_cr - 1.0)
    )

    return du_dx, dp_cr_dx, df_cr_dx


def modified_rankine_hugoniot_with_cr_injection(
    rho0: float,
    u0: float,
    p_gas0: float,
    p_cr0: float,
    f_cr0: float,
    gamma_gas: float,
    gamma_cr: float,
    injected_energy_flux: float,
) -> tuple:
    """Jump conditions across the (diffusion-unresolved) subshock.

    Derivation (PROGRESS.md ladder item 9): the subshock is thin compared to
    the CR diffusion length (``diffusion_coefficient / u0``, the precursor's
    own scale) but thick compared to nothing on the CR side -- CRs don't
    thermalize collisionally, so ``P_cr`` stays *continuous* through it (only
    gas density/velocity/pressure genuinely jump). Since ``P_cr`` is the same
    on both sides, it cancels out of the total-momentum first integral
    (``rho u^2 + P_gas + P_cr = const``), leaving the *ordinary single-fluid*
    mass+momentum Rankine-Hugoniot relations for ``(rho, u, P_gas)`` alone.

    Energy is different: DSA injection removes ``injected_energy_flux``
    (energy / area / time -- ``eta_cr(M0) * thermal_energy_flux0`` from
    ``cr_grey_injection.inject_crs_at_shocks``, evaluated at this function's
    ``rho0``/``u0``/``p_gas0``) from the gas channel at exactly this point, so
    the gas-only energy jump is a standard "radiative-shock-like" jump with a
    known energy sink:

        (1/2) rho2 u2^3 + gamma_gas/(gamma_gas-1) P_gas2 u2
      = (1/2) rho0 u0^3 + gamma_gas/(gamma_gas-1) P_gas0 u0 - s * injected_energy_flux

    with ``s = sign(u0)`` the flow direction in the shock frame: the energy
    fluxes carry the sign of ``u``, so removing energy from the gas shrinks
    the flux's *magnitude*. (Before 2026-10-03 this read ``- injected_energy_flux``
    unconditionally, which is only right for ``u0 > 0``; for the ``u0 < 0``
    piston shock of the item-9 pytest it *added* energy to the gas instead --
    see PROGRESS.md's 2026-10-03 entry.)

    Solved via a 1D root-find over the compression ratio ``r = rho2/rho0``
    (mass and gas-momentum conservation give ``u2``/``P_gas2`` as functions of
    ``r`` directly). The post-shock CR flux follows from integrating the
    CR-only energy equation (see :func:`cr_precursor_ode_rhs`'s docstring)
    across the same jump, where the ``-P_cr * du/dx`` term becomes, in the
    distributional (zero-width-shock) limit, ``-P_cr(0) * (u2 - u0)`` (a
    finite jump in ``u``, not a delta -- ``P_cr`` is finite and continuous, so
    only the ordinary product survives):

        F_cr2 = F_cr0 - P_cr0 * (u2 - u0) * gamma_cr/(gamma_cr-1) + s * injected_energy_flux

    This assumes ``P_cr`` continuous through the jump, i.e. CRs that already
    fill the shock region (a resolved precursor). It does **not** apply when
    the CRs are created behind the shock with no precursor (``P_cr`` itself
    jumps); use :func:`rankine_hugoniot_with_downstream_crs` and
    :func:`cr_energy_flux_jump_bounds` for that case.

    Validation (see this module's pytest / a standalone check run during
    development): with ``p_cr0 = f_cr0 = injected_energy_flux = 0`` this must
    reduce exactly to the ordinary ideal-gas Rankine-Hugoniot jump (density
    ratio ``(gamma+1)M^2 / ((gamma-1)M^2+2)``) -- checked to root-finder
    precision (~1e-10).

    Args:
        rho0: Pre-subshock gas density (already CR-precursor-modified, not
            necessarily the far-upstream ambient value).
        u0: Pre-subshock gas velocity (shock frame).
        p_gas0: Pre-subshock gas pressure.
        p_cr0: Pre-subshock CR pressure (continuous through the subshock).
        f_cr0: Pre-subshock CR flux.
        gamma_gas: Gas adiabatic index.
        gamma_cr: CR adiabatic index.
        injected_energy_flux: The CR energy flux DSA injects at this point
            (``>= 0``; use ``0.0`` to recover a plain gas shock with an
            unmodified but continuous ``P_cr`` riding through it).

    Returns:
        ``(rho2, u2, p_gas2, f_cr2)``, the post-subshock state.
    """
    mass_flux = rho0 * u0
    momentum_flux_gas_only = rho0 * u0**2 + p_gas0

    flow_sign = float(np.sign(u0))
    energy_flux0 = (
        0.5 * rho0 * u0**3
        + gamma_gas / (gamma_gas - 1.0) * p_gas0 * u0
        - flow_sign * injected_energy_flux
    )

    def residual(r):
        u2 = u0 / r
        rho2 = rho0 * r
        p_gas2 = momentum_flux_gas_only - rho2 * u2**2
        return (
            0.5 * rho2 * u2**3
            + gamma_gas / (gamma_gas - 1.0) * p_gas2 * u2
            - energy_flux0
        )

    r_max = (gamma_gas + 1.0) / (gamma_gas - 1.0) * 4.0

    # Bracket-finding note (found while validating this function against a
    # genuinely CR-precursor-modified upstream state, not the pristine
    # ambient state the original degenerate-limit validation used): with
    # ``injected_energy_flux > 0``, ``residual(r=1) = sign(u0) * injected_energy_flux``
    # (not exactly 0, as in the plain-gas case where ``r=1`` is the exact
    # trivial "no jump" root) -- this shifts a vestige of that trivial root to
    # a spurious crossing very close to ``r=1``, so ``residual`` can go
    # positive -> negative -> positive across ``(1, r_max)`` instead of the
    # single negative -> positive crossing the plain-gas case has. A naive
    # two-point ``brentq(residual, 1+eps, r_max)`` then fails outright (equal
    # signs at both ends) or, worse, could silently converge to the
    # unphysical near-1 pseudo-root if the two probed endpoints happened to
    # bracket it instead. Scan for sign changes and take the one closest to
    # ``r_max``: the physical, entropy-producing compression root is a single
    # well-separated crossing well above 1 (verified against the plain-gas
    # RH ratio at the local Mach number), while the near-1 artifact is
    # confined to a narrow neighborhood of ``r=1`` by construction.
    r_probe = np.concatenate([np.array([1.0 + 1e-9]), np.geomspace(1e-3, 1.0, 400) * (r_max - 1.0) + 1.0])
    r_probe = np.sort(r_probe)
    residual_probe = np.array([residual(rp) for rp in r_probe])
    sign_changes = np.where(np.diff(np.sign(residual_probe)) != 0)[0]
    if len(sign_changes) == 0:
        raise RuntimeError(
            "modified_rankine_hugoniot_with_cr_injection: no sign change found "
            "for the compression-ratio root over (1, r_max) -- check inputs."
        )
    lo_idx = sign_changes[-1]
    r = brentq(residual, r_probe[lo_idx], r_probe[lo_idx + 1])

    u2 = u0 / r
    rho2 = rho0 * r
    p_gas2 = momentum_flux_gas_only - rho2 * u2**2

    f_cr2 = (
        f_cr0
        - p_cr0 * (u2 - u0) * gamma_cr / (gamma_cr - 1.0)
        + flow_sign * injected_energy_flux
    )

    return rho2, u2, p_gas2, f_cr2


def rankine_hugoniot_with_downstream_crs(
    rho0: float,
    u0: float,
    p_gas0: float,
    p_cr0: float,
    f_cr0: float,
    p_cr2: float,
    f_cr2: float,
    gamma_gas: float,
    gamma_cr: float,
) -> tuple:
    """Gas jump across a shock whose CR pressure itself jumps (no precursor).

    The case of the item-9 pytest's Test A: without ``diffusive_relaxation``
    the injected CRs are created behind the shock and advected away, so
    ``P_cr`` rises from ``p_cr0`` to ``p_cr2`` across the jump and does not
    cancel from the momentum balance (unlike
    :func:`modified_rankine_hugoniot_with_cr_injection`'s precursor case).

    Only the *total* (gas + CR) momentum and energy fluxes are conserved
    exactly across such a jump: the gas<->CR exchange terms
    (``-u dP_cr/dx`` in the gas energy, ``-P_cr du/dx`` in the CR energy)
    become products of two quantities that jump at the same place, whose
    split depends on where inside the shock the CRs appear -- but their sum,
    ``-d(u P_cr)/dx``, is a total derivative. With the downstream CR state
    taken as given, mass, total momentum and total energy

        rho u = const
        rho u^2 + P_gas + P_cr = const
        (1/2) rho u^3 + gamma_gas/(gamma_gas-1) P_gas u
          + gamma_cr/(gamma_cr-1) P_cr u + F_cr = const

    (``F_cr`` is the CR flux relative to the gas, so it is Galilean
    invariant, while ``u`` is the shock-frame velocity) fix the downstream
    gas state. The amount of injection enters only through ``p_cr2``/
    ``f_cr2``; check it separately with :func:`cr_energy_flux_jump_bounds`.

    Args:
        rho0, u0, p_gas0: Upstream gas state (``u0`` in the shock frame).
        p_cr0, f_cr0: Upstream CR pressure and flux.
        p_cr2, f_cr2: Downstream CR pressure and flux.
        gamma_gas, gamma_cr: Adiabatic indices.

    Returns:
        ``(rho2, u2, p_gas2)``.
    """
    A = gamma_gas / (gamma_gas - 1.0)
    B = gamma_cr / (gamma_cr - 1.0)
    momentum_flux = rho0 * u0**2 + p_gas0 + p_cr0
    energy_flux = 0.5 * rho0 * u0**3 + A * p_gas0 * u0 + B * p_cr0 * u0 + f_cr0

    def gas_state(r):
        u2 = u0 / r
        rho2 = rho0 * r
        return rho2, u2, momentum_flux - rho2 * u2**2 - p_cr2

    def residual(r):
        rho2, u2, p_gas2 = gas_state(r)
        return 0.5 * rho2 * u2**3 + A * p_gas2 * u2 + B * p_cr2 * u2 + f_cr2 - energy_flux

    # Same sign-change scan as modified_rankine_hugoniot_with_cr_injection:
    # take the crossing closest to r_max (the physical compression root).
    r_max = (gamma_gas + 1.0) / (gamma_gas - 1.0) * 4.0
    r_probe = np.sort(np.concatenate([np.array([1.0 + 1e-9]), np.geomspace(1e-3, 1.0, 400) * (r_max - 1.0) + 1.0]))
    residual_probe = np.array([residual(rp) for rp in r_probe])
    sign_changes = np.where(np.diff(np.sign(residual_probe)) != 0)[0]
    if len(sign_changes) == 0:
        raise RuntimeError(
            "rankine_hugoniot_with_downstream_crs: no sign change found for the "
            "compression-ratio root over (1, r_max) -- check inputs."
        )
    lo_idx = sign_changes[-1]
    r = brentq(residual, r_probe[lo_idx], r_probe[lo_idx + 1])
    return gas_state(r)


def cr_energy_flux_jump_bounds(
    u0: float,
    u2: float,
    p_cr0: float,
    p_cr2: float,
    injected_energy_flux: float,
) -> tuple:
    """Allowed range of the jump in the CR energy flux ``u e_cr + F_cr``.

    Integrating the steady CR energy equation
    ``d(u e_cr + F_cr)/dx = -P_cr du/dx + q`` (``q`` the DSA injection rate
    per volume, integrating to ``injected_energy_flux`` across the shock)
    from upstream to downstream gives

        Phi2 - Phi0 = s * injected_energy_flux + <P_cr> * (u0 - u2)

    with ``s = sign(u0)`` and ``<P_cr>`` the CR pressure averaged over the
    velocity jump. If ``P_cr`` changes monotonically through the shock,
    ``<P_cr>`` lies between ``p_cr0`` and ``p_cr2``: ``p_cr0`` if the CRs
    appear only after the gas is compressed, ``p_cr2`` if they are already
    there before. This brackets the jump without having to resolve the
    shock's inner structure.

    Returns:
        ``(lower, upper)`` bounds on ``Phi2 - Phi0``.
    """
    flow_sign = float(np.sign(u0))
    a = flow_sign * injected_energy_flux + p_cr0 * (u0 - u2)
    b = flow_sign * injected_energy_flux + p_cr2 * (u0 - u2)
    return min(a, b), max(a, b)
