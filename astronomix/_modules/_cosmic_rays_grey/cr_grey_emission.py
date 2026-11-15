"""
Hadronic (pion-decay) gamma-ray emission from the grey CR-proton population
(plan Sec. 3 "Emission operators"; test ladder item 13, Phase C).

JAX reimplementation of the Kafexhiu et al. (2014) parametrization of the
p-p -> pi0 -> gamma differential cross section -- the same equations
naima's ``radiative.PionDecay`` class (Zabalza 2015) uses, so that class
(with its lookup-table fast path disabled, ``useLUT=False``, to compare
against the same underlying analytic formula rather than a cached fit to
it) serves directly as this module's validation reference; see
``pytests/cosmic_rays_grey/cr_pion_decay_emission.py``. No new reference
solver needed to be built for this ladder item, unlike most earlier ones.

Scope for this ladder item: given a *known* proton population (the plan's
grey "particle-spectrum object" -- an exponential-cutoff power law in total
proton energy, matching naima's ``ExponentialCutoffPowerLaw`` convention
exactly) and a target gas density, compute the pion-decay photon spectrum.
This is a standalone physics-formula check, decoupled from any running
simulation. Deriving a local spectrum's normalization from a cell's
``e_cr`` field and line-of-sight integrating to a map (the plan's "spectrum
normalization from e_cr" and "Line-of-sight integrate to a map/SED") is
ladder item 15's job (the end-to-end SNR-cloud case), not this one's.

Units: energies are plain floats in GeV throughout (matching Kafexhiu et
al. (2014)'s own convention, and naima's internal ``_Ep``/``_J`` arrays
before their final eV conversion) -- not code units, and not astropy
Quantities (this module operates on bare JAX arrays so it stays
``jax.grad``-differentiable end to end). ``pion_decay_photon_spectrum``
returns photons / (s * GeV); multiply by ``1e-9`` to compare directly
against naima's ``1/(s eV)`` convention.

Differentiability: every piecewise branch below is evaluated
unconditionally across the whole ``Tp``/``Egamma`` domain and combined with
``jnp.where`` (this module's established pattern, e.g.
``cr_grey_injection.dsa_efficiency_kang_ryu_2013``) -- so every
``sqrt``/``log``/fractional power is floored against invalid domains
(negative arguments, division by zero) even in branches that get masked out
by the final ``jnp.where``, to avoid the "``jnp.where`` differentiates every
branch everywhere" NaN-gradient gotcha (same category as
``CosmicRayGreyParams.cr_pressure_speed_floor``).
"""

# general
import math

# typing
from typing import NamedTuple

# jax
import jax
import jax.numpy as jnp

# units (host-side constants only, never traced)
from astropy import constants as _const
from astropy import units as _u

#: proton rest energy, GeV (Kafexhiu et al. (2014) / naima's radiative.PionDecay).
M_P_GEV = float((_const.m_p * _const.c ** 2).to(_u.GeV).value)
#: neutral-pion rest mass, GeV (Kafexhiu et al. (2014) Sec. II).
M_PI0_GEV = 0.1349766
#: kinetic-energy threshold for pi0 production in p-p collisions, GeV.
T_TH_GEV = 0.27966184
#: speed of light, cm/s.
_C_CGS = float(_const.c.to(_u.cm / _u.s).value)

#: floor guarding sqrt/log/division/fractional-power arguments against
#: exactly-zero or slightly-negative (floating-point rounding) inputs, in
#: every branch -- see module docstring's "Differentiability" note.
_TINY = 1e-30

#: high-energy differential-cross-section models (Kafexhiu et al. (2014)
#: Table IV) -- coefficients for _sigma_pi_hiE's ``a`` argument.
_A_TABLE = {
    "Geant4": (0.728, 0.596, 0.491, 0.2503, 0.117),
    "Pythia8": (0.652, 0.0016, 0.488, 0.1928, 0.483),
    "SIBYLL": (5.436, 0.254, 0.072, 0.075, 0.166),
    "QGSJET": (0.908, 0.0009, 6.089, 0.176, 0.448),
}

#: Table V's fixed (lambda, alpha, beta, gamma) for each high-energy model's
#: own regime (Tp > _E_TRANS[model]) -- the low/mid-energy regimes below
#: fill beta/gamma dynamically from Tp itself (_F, below).
_F_MP_TABLE = {
    "Geant4": (3.0, 0.5, 4.9, 1.0),
    "Pythia8": (3.5, 0.5, 4.0, 1.0),
    "SIBYLL": (3.55, 0.5, 3.6, 1.0),
    "QGSJET": (3.55, 0.5, 4.5, 1.0),
}

#: Table VII's (b1, b2, b3) for each high-energy model's own regime.
_B_TABLE = {
    "Geant4_0": (9.53, 0.52, 0.054),
    "Geant4": (9.13, 0.35, 9.7e-3),
    "Pythia8": (9.06, 0.3795, 0.01105),
    "SIBYLL": (10.77, 0.412, 0.01264),
    "QGSJET": (13.16, 0.4419, 0.01439),
}

#: proton kinetic energy (GeV) above which each high-energy model is valid.
_E_TRANS = {"Pythia8": 50.0, "SIBYLL": 100.0, "QGSJET": 100.0, "Geant4": 100.0}


class GreyProtonSpectrum(NamedTuple):
    """Exponential-cutoff power-law proton spectrum -- the plan's grey
    "particle-spectrum object" (normalization, slope, cutoff).

    Matches naima's ``ExponentialCutoffPowerLaw`` exactly (same functional
    form and parameter meaning), so a naima particle distribution built with
    the same numbers gives the same J(E):

        J(E) = amplitude * (E / e_0)^(-alpha) * exp(-(E / e_cutoff)^beta)

    All energies (``e_0``, ``e_cutoff``, and the ``e_total_gev`` passed to
    :func:`proton_number_density`) are in GeV. ``amplitude`` is in particles
    / GeV -- a *total* particle-count spectrum (as naima's radiative models
    expect), not a volumetric density; see module docstring.
    """

    amplitude: float
    e_0: float
    alpha: float
    e_cutoff: float
    beta: float = 1.0


def proton_number_density(spectrum: GreyProtonSpectrum, e_total_gev):
    """J(E): particles per unit total proton energy, particles/GeV."""
    x = e_total_gev / spectrum.e_0
    return (
        spectrum.amplitude
        * x ** (-spectrum.alpha)
        * jnp.exp(-((e_total_gev / spectrum.e_cutoff) ** spectrum.beta))
    )


def proton_spectrum_normalized_to_energy(
    total_energy_gev,
    e_0: float = 1.0,
    alpha: float = 2.0,
    e_cutoff: float = 1e5,
    beta: float = 1.0,
    e_p_min_gev: float = M_P_GEV + T_TH_GEV + 1e-4,
    e_p_max_gev: float = 1e7,
    n_e_p_per_decade: int = 100,
) -> GreyProtonSpectrum:
    """Build a :class:`GreyProtonSpectrum` whose total energy content equals
    ``total_energy_gev`` -- the plan's "spectrum normalization from e_cr"
    (Sec. 3), deliberately deferred by ladder items 13/14 (which validated
    the emission formulas against directly-specified spectra) to whichever
    item first needs to turn a simulation's local ``e_cr`` into an assumed
    proton population. First exercised by ladder item 15.

    "Total energy content" uses the same convention as naima's own
    ``BaseProton.Wp``/``compute_Wp`` (and ``BaseElectron.We``): the integral
    of *total* energy (not kinetic) weighted by the number spectrum,
    ``integral(E * J(E) dE)`` over ``[e_p_min_gev, e_p_max_gev]`` -- so
    ``total_energy_gev`` should be a proton population's total energy
    content in GeV (e.g. a cell's ``e_cr`` energy density converted to
    physical erg, times that cell's volume, then converted erg -> GeV).

    ``total_energy_gev`` may be an array (e.g. one value per grid cell) --
    the returned spectrum's ``amplitude`` field then broadcasts to the same
    shape, one independently-normalized spectrum per input value, while
    ``e_0``/``alpha``/``e_cutoff``/``beta`` (the assumed shared spectral
    *shape* -- e.g. ``alpha=2.0`` is the standard test-particle strong-shock
    DSA prediction) stay shared scalars.

    Args:
        total_energy_gev: Target total energy content, GeV (scalar or array).
        e_0: Reference energy for the assumed shape, GeV.
        alpha: Power-law index of the assumed shape.
        e_cutoff: Exponential cutoff energy of the assumed shape, GeV.
        beta: Cutoff exponent of the assumed shape.
        e_p_min_gev: Lower integration limit (energy content, not the
            emission formula's own integration grid -- callers passing the
            returned spectrum to :func:`pion_decay_photon_spectrum` should
            use matching values there for consistency).
        e_p_max_gev: Upper integration limit, GeV.
        n_e_p_per_decade: Number of integration-grid points per decade.

    Returns:
        A :class:`GreyProtonSpectrum` with the given shape and an
        ``amplitude`` normalized so its total energy content matches.
    """
    n_ep = max(10, int(n_e_p_per_decade * math.log10(e_p_max_gev / e_p_min_gev)))
    ep_grid = jnp.logspace(math.log10(e_p_min_gev), math.log10(e_p_max_gev), n_ep)
    unit_spectrum = GreyProtonSpectrum(amplitude=1.0, e_0=e_0, alpha=alpha, e_cutoff=e_cutoff, beta=beta)
    unit_energy_gev = _trapz_loglog(ep_grid * proton_number_density(unit_spectrum, ep_grid), ep_grid)
    amplitude = total_energy_gev / unit_energy_gev
    return GreyProtonSpectrum(amplitude=amplitude, e_0=e_0, alpha=alpha, e_cutoff=e_cutoff, beta=beta)


# ---------------------------------------------------------------------------
# Momentum-space spectrum normalised to the fluid's *kinetic* e_cr
# (GAMMA_RAY_PLAN.md step 1).
#
# ``proton_spectrum_normalized_to_energy`` above (kept unchanged for the
# existing item-15 / Phase-D / gradient-check scripts) uses a power law in
# *total* energy and matches ``integral(E J dE)`` -- rest mass included, above
# the pion threshold only -- to the target energy. The fluid variable e_cr is
# the *kinetic* energy density of the whole CR population (P_cr = e_cr / 3 in
# the relativistic limit), so that convention biases the gamma-ray level by
# tens of per cent (GAMMA_RAY_PLAN.md Sec. 3). The functions below follow
# Pfrommer et al. (2017, ApJL 847, L13, Eqs. 1-2) instead:
#
#     f(p) = C g(p),  g(p) = p^-alpha * [smooth break] * exp(-(p/p_max)^beta_c)
#                            * theta(p - q),
#     e_cr = integral_q^inf f(p) T(p) dp,
#
# with p the dimensionless momentum P / (m_p c) and T the kinetic energy.
# ---------------------------------------------------------------------------

#: erg -> GeV.
ERG_TO_GEV = float((1.0 * _u.erg).to(_u.GeV).value)


class ProtonMomentumSpectrum(NamedTuple):
    """Shape of a grey CR-proton spectrum in dimensionless momentum
    ``p = P / (m_p c)`` (unit amplitude; the amplitude comes from ``e_cr``).

    ``g(p) = p^-alpha [1 + (p/p_break)^((alpha_2 - alpha)/break_smoothness)]^-break_smoothness
    exp(-(p/p_max)^cutoff_beta) theta(p - q)``

    - Single power law: ``alpha_2=None`` (static; the break factor is dropped).
    - Ackermann et al. (2013, Science 339, 807, Eq. 1) smoothly broken power
      law: ``alpha=s1``, ``alpha_2=s2``, ``p_break=p_br / (m_p c)``. Their
      paper does not state the proton smoothness parameter; the default 0.1
      (the value they fix for their *photon* broken power law) is an
      assumption.
    - ``p_max=None`` (static) drops the exponential cutoff.

    Defaults (GAMMA_RAY_PLAN.md Sec. 6): alpha = 2.3, q = 0.5 (Pfrommer et al.
    2017), p_max = 1e5 (~94 TeV). Float fields may be traced (differentiable).
    """

    alpha: float = 2.3
    q: float = 0.5
    p_max: float | None = 1.0e5
    cutoff_beta: float = 1.0
    alpha_2: float | None = None
    p_break: float = 1.0e3
    break_smoothness: float = 0.1


#: Ackermann et al. (2013) best-fit proton spectra (s1, s2, p_br [GeV/c]).
ACKERMANN13_IC443 = dict(alpha=2.36, alpha_2=3.1, p_break=239.0 / M_P_GEV)
ACKERMANN13_W44 = dict(alpha=2.36, alpha_2=3.5, p_break=22.0 / M_P_GEV)


def proton_kinetic_energy_gev(p_hat):
    """Kinetic energy T(p) = m_p c^2 (sqrt(1 + p^2) - 1), GeV, written as
    m_p p^2 / (sqrt(1 + p^2) + 1) to avoid cancellation at small p."""
    p2 = p_hat * p_hat
    return M_P_GEV * p2 / (jnp.sqrt(1.0 + p2) + 1.0)


def proton_momentum_shape(spectrum: ProtonMomentumSpectrum, p_hat):
    """Unit-amplitude shape g(p) of :class:`ProtonMomentumSpectrum`, per unit
    dimensionless momentum. Zero below ``q``."""
    p_safe = jnp.maximum(p_hat, _TINY)
    g = p_safe ** (-spectrum.alpha)
    if spectrum.alpha_2 is not None:
        x = p_safe / spectrum.p_break
        w = spectrum.break_smoothness
        g = g * (1.0 + x ** ((spectrum.alpha_2 - spectrum.alpha) / w)) ** (-w)
    if spectrum.p_max is not None:
        g = g * jnp.exp(-((p_safe / spectrum.p_max) ** spectrum.cutoff_beta))
    return jnp.where(p_hat >= spectrum.q, g, 0.0)


def proton_unit_kinetic_energy_gev(
    spectrum: ProtonMomentumSpectrum,
    p_hat_max: float = 1.0e7,
    n_per_decade: int = 100,
):
    """``integral_q^p_hat_max g(p) T(p) dp`` for the unit-amplitude shape, GeV
    (per unit amplitude). The grid starts exactly at ``q`` (no step inside the
    integral) and has a static size, so ``q`` may be traced; it covers
    ``q >= 1e-3`` with at least ``n_per_decade`` points per decade."""
    n = max(10, int(n_per_decade * (math.log10(p_hat_max) + 3.0)))
    q = jnp.asarray(spectrum.q)
    p = q * (p_hat_max / q) ** jnp.linspace(0.0, 1.0, n)
    return _trapz_loglog(proton_momentum_shape(spectrum, p) * proton_kinetic_energy_gev(p), p)


def proton_number_per_total_energy(spectrum: ProtonMomentumSpectrum, amplitude, e_total_gev):
    """J(E) = C g(p(E)) dp/dE, particles per GeV of *total* proton energy --
    the form the Kafexhiu et al. (2014) integral (and naima) expects.
    With p = P/(m_p c): dp/dE = E / (m_p^2 p)."""
    p_hat = jnp.sqrt(jnp.maximum(e_total_gev ** 2 - M_P_GEV ** 2, _TINY)) / M_P_GEV
    return amplitude * proton_momentum_shape(spectrum, p_hat) * e_total_gev / (M_P_GEV ** 2 * p_hat)


def pion_decay_emissivity_per_ecr(
    photon_energy_gev,
    spectrum: ProtonMomentumSpectrum = ProtonMomentumSpectrum(),
    *,
    nuclear_enhancement: bool = True,
    hi_e_model: str = "Pythia8",
    e_p_min_gev: float = M_P_GEV + T_TH_GEV + 1e-4,
    e_p_max_gev: float = 1e7,
    n_e_p_per_decade: int = 100,
    p_hat_norm_max: float = 1.0e7,
    n_norm_per_decade: int = 100,
):
    """Pion-decay emissivity kernel K(E_gamma) per unit target density and
    per unit CR *kinetic* energy density (GAMMA_RAY_PLAN.md step 1):

        j_gamma(E) [photons s^-1 GeV^-1 cm^-3] = n_H [cm^-3] * e_cr [erg cm^-3] * K(E)

    ``n_H`` is the target *hydrogen* density, as in Kafexhiu et al. (2014) /
    naima's ``nh``; helium and heavier nuclei (in target and CRs) enter
    through ``nuclear_enhancement`` and must not be added to ``n_H`` again.
    ``e_cr`` is the fluid CR energy density, interpreted as protons with
    the shape ``spectrum``.

    The proton-energy integration (``e_p_*``) matches
    :func:`pion_decay_photon_spectrum` / naima's defaults; ``p_hat_norm_max``
    and ``n_norm_per_decade`` set the normalisation integral
    (:func:`proton_unit_kinetic_energy_gev`). All of these are static.
    K is a per-unit quantity (~1e-13 for E ~ 1 GeV), so it is float32-safe;
    differentiable in the float fields of ``spectrum``.

    Returns:
        K, photons s^-1 GeV^-1 per (cm^-3 * erg cm^-3), same shape as
        ``photon_energy_gev``.
    """
    amplitude = ERG_TO_GEV / proton_unit_kinetic_energy_gev(spectrum, p_hat_norm_max, n_norm_per_decade)
    n_ep = max(10, int(n_e_p_per_decade * math.log10(e_p_max_gev / e_p_min_gev)))
    Ep = jnp.logspace(math.log10(e_p_min_gev), math.log10(e_p_max_gev), n_ep)
    J = proton_number_per_total_energy(spectrum, amplitude, Ep)
    return _pp_photon_rate(Ep, J, photon_energy_gev, 1.0, nuclear_enhancement, hi_e_model)


# ---------------------------------------------------------------------------
# Maps and SEDs from (stored) simulation states -- GAMMA_RAY_PLAN.md step 2.
#
# Everything is linear in the per-cell "hadronic target" field
# w = n_H [cm^-3] * e_cr [erg cm^-3], so a map/SED is K(E) (or its band
# integral) times a sum of w over cells or along the line of sight. All
# functions are plain jnp (differentiable in the fields and the float
# spectrum parameters); unit conversion factors are host-side floats.
# ---------------------------------------------------------------------------

#: 4 pi, for intensities per steradian.
_FOUR_PI = 4.0 * math.pi


def hadronic_target_field(rho_code, e_cr_code, code_units, hydrogen_mass_fraction: float):
    """Per-cell ``w = n_H e_cr`` [cm^-3 * erg cm^-3] from code-unit density
    and CR energy density.

    ``n_H = hydrogen_mass_fraction * rho / m_p``. Pass the value the setup
    assumes: the item-11/15 setups define ``rho = n m_p`` with ``n`` meant as
    hydrogen, i.e. ``1.0``; a cosmic mixture is ~0.71. Helium is *not* added to
    n_H -- the nuclear-enhancement factor in K already accounts for it.

    Args:
        rho_code: density field, code units.
        e_cr_code: CR energy density field, code units (energy / length^3).
        code_units: an ``astronomix.CodeUnits`` instance.
        hydrogen_mass_fraction: X_H (required, see above).
    """
    rho_to_cgs = float((1.0 * code_units.code_density).to(_u.g / _u.cm ** 3).value)
    ecr_to_cgs = float((1.0 * code_units.code_energy / code_units.code_length ** 3).to(_u.erg / _u.cm ** 3).value)
    n_h_per_rho = hydrogen_mass_fraction * rho_to_cgs / float(_const.m_p.to(_u.g).value)
    return (n_h_per_rho * ecr_to_cgs) * rho_code * e_cr_code


def pion_decay_band_energy_per_ecr(
    e_min_gev: float,
    e_max_gev: float,
    spectrum: ProtonMomentumSpectrum = ProtonMomentumSpectrum(),
    n_per_decade: int = 64,
    **kernel_kwargs,
):
    """Band-integrated energy emissivity per unit target field,
    ``integral_{e_min}^{e_max} E K(E) dE``, erg s^-1 per (cm^-3 * erg cm^-3).
    The band edges are static. Second-order convergent; the default 64 points
    per decade agree with 640 to < 1e-4 in every band from 30 MeV to 10 TeV
    (worst in the pion bump; 16/decade would give ~1.3e-3;
    ``pytests/cosmic_rays_grey/cr_gamma_maps.py``)."""
    n = max(4, int(math.ceil(n_per_decade * math.log10(e_max_gev / e_min_gev))) + 1)
    E = jnp.logspace(math.log10(e_min_gev), math.log10(e_max_gev), n)
    K = pion_decay_emissivity_per_ecr(E, spectrum, **kernel_kwargs)
    return _trapz_loglog(E * K, E) / ERG_TO_GEV


def pion_decay_sed(
    target_field,
    cell_volume_cm3: float,
    photon_energy_gev,
    spectrum: ProtonMomentumSpectrum = ProtonMomentumSpectrum(),
    distance_cm: float | None = None,
    **kernel_kwargs,
):
    """Domain-total pion-decay SED from a ``hadronic_target_field``.

    Returns ``E^2 dL/dE`` [erg s^-1] (``distance_cm=None``) or
    ``E^2 dF/dE = E^2 dL/dE / (4 pi d^2)`` [erg cm^-2 s^-1], at
    ``photon_energy_gev``."""
    E = jnp.asarray(photon_energy_gev)
    e2k_erg = E * E * pion_decay_emissivity_per_ecr(E, spectrum, **kernel_kwargs) / ERG_TO_GEV
    sed = e2k_erg * (jnp.sum(target_field) * cell_volume_cm3)
    if distance_cm is not None:
        sed = sed / (_FOUR_PI * distance_cm ** 2)
    return sed


def pion_decay_band_map(
    target_field,
    cell_length_cm: float,
    e_min_gev: float,
    e_max_gev: float,
    spectrum: ProtonMomentumSpectrum = ProtonMomentumSpectrum(),
    axis: int = 2,
    **band_kwargs,
):
    """Projected band energy intensity along ``axis`` (optically thin):

        I = (1 / 4 pi) sum_LOS w dl * integral_band E K dE   [erg cm^-2 s^-1 sr^-1]

    Distance-independent. The energy flux of one pixel (side ``cell_length``)
    at distance d is ``I * (cell_length / d)^2``; summed over the map it equals
    the band integral of :func:`pion_decay_sed`'s ``E dF/dE``."""
    column = jnp.sum(target_field, axis=axis) * cell_length_cm
    return column * pion_decay_band_energy_per_ecr(e_min_gev, e_max_gev, spectrum, **band_kwargs) / _FOUR_PI


def _sigma_pp_inel(Tp):
    """Inelastic p-p cross section, Kafexhiu et al. (2014) Eq. 1, cm^2."""
    Tp_safe = jnp.maximum(Tp, _TINY)
    L = jnp.log(Tp_safe / T_TH_GEV)
    sigma = 30.7 - 0.96 * L + 0.18 * L ** 2
    ratio = T_TH_GEV / Tp_safe
    sigma = sigma * (1.0 - ratio ** 1.9) ** 3
    return sigma * 1e-27  # mbarn -> cm^2


def _sigma_pi_loE(Tp):
    """Inclusive pi0-production cross section, T_th < Tp < 2 GeV (fit to
    experimental data)."""
    Tp_safe = jnp.maximum(Tp, _TINY)
    m_p, m_pi = M_P_GEV, M_PI0_GEV
    Mres, Gres = 1.1883, 0.2264
    s = 2.0 * m_p * (Tp_safe + 2.0 * m_p)
    s_safe = jnp.maximum(s, _TINY)
    gamma_bw = jnp.sqrt(Mres ** 2 * (Mres ** 2 + Gres ** 2))
    K = jnp.sqrt(8.0) * Mres * Gres * gamma_bw / (jnp.pi * jnp.sqrt(Mres ** 2 + gamma_bw))
    fBW = m_p * K / (((jnp.sqrt(s_safe) - m_p) ** 2 - Mres ** 2) ** 2 + Mres ** 2 * Gres ** 2)
    mu_arg = (s_safe - m_pi ** 2 - 4.0 * m_p ** 2) ** 2 - 16.0 * m_pi ** 2 * m_p ** 2
    mu = jnp.sqrt(jnp.maximum(mu_arg, 0.0)) / (2.0 * m_pi * jnp.sqrt(s_safe))
    sigma0 = 7.66e-3
    sigma1pi = sigma0 * mu ** 1.95 * (1.0 + mu + mu ** 5) * jnp.maximum(fBW, _TINY) ** 1.86
    sigma2pi = 5.7 / (1.0 + jnp.exp(-9.3 * (Tp_safe - 1.4)))
    sigma2pi = jnp.where(Tp_safe < 0.56, 0.0, sigma2pi)
    return (sigma1pi + sigma2pi) * 1e-27


def _sigma_pi_midE(Tp):
    """Geant4.10.0-model multiplicity fit, 2 GeV <= Tp < 5 GeV."""
    Tp_safe = jnp.maximum(Tp, _TINY)
    Qp = (Tp_safe - T_TH_GEV) / M_P_GEV
    multip = -6e-3 + 0.237 * Qp - 0.023 * Qp ** 2
    return _sigma_pp_inel(Tp_safe) * multip


def _sigma_pi_hiE(Tp, a):
    """General high-energy multiplicity fit, Tp >= 5 GeV (Eq. 7)."""
    Tp_safe = jnp.maximum(Tp, 3.0 + _TINY)
    csip = jnp.maximum((Tp_safe - 3.0) / M_P_GEV, _TINY)
    m1 = a[0] * csip ** a[3] * (1.0 + jnp.exp(-a[1] * csip ** a[4]))
    m2 = 1.0 - jnp.exp(-a[2] * csip ** 0.25)
    return _sigma_pp_inel(Tp_safe) * m1 * m2


def _sigma_pi(Tp, hi_e_model):
    """Inclusive pi0-production cross section, all Tp (piecewise combine)."""
    e_trans = _E_TRANS[hi_e_model]
    lo = _sigma_pi_loE(Tp)
    mid = _sigma_pi_midE(Tp)
    hi_geant4 = _sigma_pi_hiE(Tp, _A_TABLE["Geant4"])
    hi_model = _sigma_pi_hiE(Tp, _A_TABLE[hi_e_model])
    result = jnp.where(Tp < 2.0, lo, mid)
    result = jnp.where(Tp >= 5.0, hi_geant4, result)
    result = jnp.where(Tp >= e_trans, hi_model, result)
    return result


def _b_params(Tp, hi_e_model):
    e_trans = _E_TRANS[hi_e_model]
    g0, g, hi = _B_TABLE["Geant4_0"], _B_TABLE["Geant4"], _B_TABLE[hi_e_model]
    b1 = jnp.where(Tp < 5.0, g0[0], g[0])
    b1 = jnp.where(Tp >= e_trans, hi[0], b1)
    b2 = jnp.where(Tp < 5.0, g0[1], g[1])
    b2 = jnp.where(Tp >= e_trans, hi[1], b2)
    b3 = jnp.where(Tp < 5.0, g0[2], g[2])
    b3 = jnp.where(Tp >= e_trans, hi[2], b3)
    return 5.9, b1, b2, b3


def _calc_EpimaxLAB(Tp):
    """Eq. 10: max LAB-frame pion energy for a given proton kinetic energy."""
    Tp_safe = jnp.maximum(Tp, _TINY)
    m_p, m_pi = M_P_GEV, M_PI0_GEV
    s = jnp.maximum(2.0 * m_p * (Tp_safe + 2.0 * m_p), _TINY)
    EpiCM = (s - 4.0 * m_p ** 2 + m_pi ** 2) / (2.0 * jnp.sqrt(s))
    PpiCM = jnp.sqrt(jnp.maximum(EpiCM ** 2 - m_pi ** 2, 0.0))
    gCM = (Tp_safe + 2.0 * m_p) / jnp.sqrt(s)
    betaCM = jnp.sqrt(jnp.maximum(1.0 - jnp.maximum(gCM, _TINY) ** -2, 0.0))
    return gCM * (EpiCM + PpiCM * betaCM)


def _calc_Egmax(Tp):
    """Max LAB-frame photon energy from pi0 decay (kinematic endpoint)."""
    m_pi = M_PI0_GEV
    EpimaxLAB = jnp.maximum(_calc_EpimaxLAB(Tp), m_pi * (1.0 + _TINY))
    gpiLAB = EpimaxLAB / m_pi
    betapiLAB = jnp.sqrt(jnp.maximum(1.0 - gpiLAB ** -2, 0.0))
    return (m_pi / 2.0) * gpiLAB * (1.0 + betapiLAB)


def _kappa(Tp):
    thetap = jnp.maximum(Tp / M_P_GEV, _TINY)
    return 3.29 - thetap ** -1.5 / 5.0


def _mu(Tp):
    q = jnp.maximum((Tp - 1.0) / M_P_GEV, _TINY)
    x = 5.0 / 4.0
    return x * q ** x * jnp.exp(-x * q)


def _F_func(Tp, Egamma, lamb, alpha, beta, gamma):
    """Eq. 11: the shape function F(Tp, Egamma) for one model-parameter set."""
    m_pi = M_PI0_GEV
    Egmax = jnp.maximum(_calc_Egmax(Tp), _TINY)
    Egamma_safe = jnp.maximum(Egamma, _TINY)
    Yg = Egamma_safe + m_pi ** 2 / (4.0 * Egamma_safe)
    Ygmax = Egmax + m_pi ** 2 / (4.0 * Egmax)
    Xg = (Yg - m_pi) / jnp.maximum(Ygmax - m_pi, _TINY)
    # analytically Xg >= 0 always (Yg, Ygmax >= m_pi for any positive energy);
    # only the upper bound needs clipping (Egamma beyond this Tp's kinematic
    # reach), per Kafexhiu et al. (2014) below Eq. 11 -- clip both defensively
    # since the floors above can perturb Xg by a tiny amount near the edges.
    Xg = jnp.clip(Xg, 0.0, 1.0)
    C = lamb * m_pi / Ygmax
    F = (1.0 - Xg ** alpha) ** beta / (1.0 + Xg / jnp.maximum(C, _TINY)) ** gamma
    return F


def _F(Tp, Egamma, hi_e_model):
    """F(Tp, Egamma), combined across all of Kafexhiu et al. (2014)'s Tp
    regimes (Table V) for the chosen high-energy model."""
    kappa = _kappa(Tp)
    F_expdata = _F_func(Tp, Egamma, 1.0, 1.0, kappa, 0.0)
    mu = _mu(Tp)
    F_g0 = _F_func(Tp, Egamma, 3.0, 1.0, mu + 2.45, mu + 1.45)
    F_g1 = _F_func(Tp, Egamma, 3.0, 1.0, 1.5 * mu + 4.95, mu + 1.50)
    F_g2 = _F_func(Tp, Egamma, 3.0, 0.5, 4.2, 1.0)
    hi_lambda, hi_alpha, hi_beta, hi_gamma = _F_MP_TABLE[hi_e_model]
    F_hi = _F_func(Tp, Egamma, hi_lambda, hi_alpha, hi_beta, hi_gamma)

    result = jnp.where(Tp < T_TH_GEV, 0.0, F_expdata)
    result = jnp.where(Tp > 1.0, F_g0, result)
    result = jnp.where(Tp > 4.0, F_g1, result)
    result = jnp.where(Tp > 20.0, F_g2, result)
    result = jnp.where(Tp > _E_TRANS[hi_e_model], F_hi, result)
    return result


def _Amax(Tp, hi_e_model):
    """Eq. 12: amplitude of the differential cross section vs. Egamma."""
    Tp_safe = jnp.maximum(Tp, _TINY)
    b0, b1, b2, b3 = _b_params(Tp_safe, hi_e_model)
    EpimaxLAB = jnp.maximum(_calc_EpimaxLAB(Tp_safe), _TINY)
    sigma_pi_val = _sigma_pi(Tp_safe, hi_e_model)
    loE_val = b0 * sigma_pi_val / EpimaxLAB
    thetap = jnp.maximum(Tp_safe / M_P_GEV, _TINY)
    hiE_val = (
        b1 * thetap ** (-b2) * jnp.exp(b3 * jnp.log(thetap) ** 2) * sigma_pi_val / M_P_GEV
    )
    return jnp.where(Tp_safe < 1.0, loE_val, hiE_val)


def _nuclear_factor(Tp):
    """Sec. IV's nuclear-enhancement factor for local-ISM abundances
    (Sihver et al. (1993) proton-nucleus inelastic cross section)."""
    Tp_safe = jnp.maximum(Tp, _TINY)
    sigmaRpp = 10.0 * jnp.pi * 1e-27
    sigmainel = _sigma_pp_inel(Tp_safe)
    sigmainel0 = _sigma_pp_inel(jnp.asarray(1e3))
    f = sigmainel / jnp.maximum(sigmainel0, _TINY)
    G = 1.0 + jnp.log(jnp.maximum(jnp.where(f > 1.0, f, 1.0), _TINY))
    epsC, eps1, eps2 = 1.37, 0.29, 0.1
    epstotal = jnp.where(
        Tp_safe > T_TH_GEV,
        epsC + (eps1 + eps2) * sigmaRpp * G / jnp.maximum(sigmainel, _TINY),
        0.0,
    )
    epstotal = jnp.where((Tp_safe > T_TH_GEV) & (Tp_safe < 1.0), 1.9141, epstotal)
    return epstotal


def _diffsigma(Tp, Egamma, nuclear_enhancement, hi_e_model):
    """dsigma/dEgamma, Kafexhiu et al. (2014) Eq. 8, cm^2/GeV."""
    diffsigma = _Amax(Tp, hi_e_model) * _F(Tp, Egamma, hi_e_model)
    if nuclear_enhancement:
        diffsigma = diffsigma * _nuclear_factor(Tp)
    return diffsigma


def _trapz_loglog(y, x):
    """Composite trapezoidal rule in log-log space, along the last axis --
    matches ``naima.utils.trapz_loglog`` exactly (see that function for the
    derivation: locally treats each bin as an exact power law and integrates
    it analytically, falling back to the ``\\int 1/x = log(x)`` case when the
    local index is -1).

    Zero-valued endpoints are masked (bin contributes 0, as in naima), *not*
    floored: an absolute floor (``max(y, _TINY)``, used until 2026-11-13)
    silently turned every bin with ``y < 1e-30`` into a constant (b = 0)
    bin. That biased unit-amplitude pion kernels (item 15, Phase D) high by
    up to ~1.2 % above a few GeV (GAMMA_RAY_PLAN.md step 1).

    "Zero" means below the dtype's smallest *normal* number: subnormal
    values (e.g. an exponential cutoff tail underflowing) would make the
    backward pass of ``log(y)`` overflow (1/y = inf, times 0 -> NaN)."""
    y1, y2 = y[..., :-1], y[..., 1:]
    x1, x2 = x[..., :-1], x[..., 1:]
    y_min = jnp.finfo(jnp.result_type(y, float)).tiny
    pos1 = y1 > y_min
    pos2 = y2 > y_min
    y1_safe = jnp.where(pos1, y1, 1.0)
    y2_safe = jnp.where(pos2, y2, 1.0)
    # b = 0 in masked bins: a masked endpoint's placeholder 1.0 next to a
    # tiny real value gives |b| ~ 1e4, (x2/x1)**b = inf, and NaN in the
    # backward pass even though the bin is masked in the forward pass.
    # log(y2) - log(y1), not log(y2 / y1): the quotient's derivative
    # y2 / y1**2 overflows for y ~ 1e-300 (cutoff tails).
    b = jnp.where(pos1 & pos2, (jnp.log(y2_safe) - jnp.log(y1_safe)) / jnp.log(x2 / x1), 0.0)
    denom = b + 1.0
    is_regular = jnp.abs(denom) > 1e-10
    denom_safe = jnp.where(is_regular, denom, 1.0)
    powerlaw_term = (y1 * (x2 * (x2 / x1) ** b - x1)) / denom_safe
    loglinear_term = x1 * y1 * jnp.log(x2 / x1)
    trapz_i = jnp.where(is_regular, powerlaw_term, loglinear_term)
    trapz_i = jnp.where(pos1 & pos2, trapz_i, 0.0)
    return jnp.sum(trapz_i, axis=-1)


def pion_decay_photon_spectrum(
    spectrum: GreyProtonSpectrum,
    photon_energy_gev,
    gas_number_density_cm3,
    *,
    nuclear_enhancement: bool = True,
    hi_e_model: str = "Pythia8",
    e_p_min_gev: float = M_P_GEV + T_TH_GEV + 1e-4,
    e_p_max_gev: float = 1e7,
    n_e_p_per_decade: int = 100,
):
    """Differential pion-decay photon production rate dN/dE/dt at the given
    photon energies, for a known proton population -- ladder item 13.

    Matches ``naima.radiative.PionDecay(particle_distribution, nh,
    nuclear_enhancement, useLUT=False)._spectrum(photon_energy)`` up to the
    eV/GeV unit convention (see module docstring); see
    ``pytests/cosmic_rays_grey/cr_pion_decay_emission.py`` for the direct
    comparison. ``e_p_min_gev``/``e_p_max_gev``/``n_e_p_per_decade`` are
    static (Python float/int) integration hyperparameters, not differentiable
    simulation state, matching naima's own ``Epmin``/``Epmax``/``nEpd``
    defaults exactly.

    Args:
        spectrum: The proton population's assumed spectral shape.
        photon_energy_gev: Photon energy (or array of energies), GeV.
        gas_number_density_cm3: Target (ambient) gas number density, cm^-3.
        nuclear_enhancement: Apply the ISM-abundance nuclear-enhancement
            factor (Sec. IV). Matches naima's default (``True``).
        hi_e_model: One of ``"Pythia8"`` (default, matches naima's default),
            ``"Geant4"``, ``"SIBYLL"``, ``"QGSJET"``.
        e_p_min_gev: Lower edge of the proton-energy integration grid, GeV.
        e_p_max_gev: Upper edge of the proton-energy integration grid, GeV.
        n_e_p_per_decade: Number of proton-energy grid points per decade.

    Returns:
        Photon production rate, photons / (s * GeV), same shape as
        ``photon_energy_gev``.
    """
    n_ep = max(10, int(n_e_p_per_decade * math.log10(e_p_max_gev / e_p_min_gev)))
    Ep = jnp.logspace(math.log10(e_p_min_gev), math.log10(e_p_max_gev), n_ep)
    J = proton_number_density(spectrum, Ep)
    return _pp_photon_rate(Ep, J, photon_energy_gev, gas_number_density_cm3, nuclear_enhancement, hi_e_model)


def _pp_photon_rate(Ep, J, photon_energy_gev, gas_number_density_cm3, nuclear_enhancement, hi_e_model):
    """Kafexhiu et al. (2014) photon production rate for J(E_p) [1/GeV] on the
    total-energy grid ``Ep`` [GeV]: ``n c integral(dsigma/dE_gamma J dE_p)``,
    photons / (s GeV). Shared by both spectrum conventions."""
    Tp = Ep - M_P_GEV
    photon_energy_gev = jnp.asarray(photon_energy_gev)
    scalar_input = photon_energy_gev.ndim == 0

    def _spec_at_one_energy(Eg):
        diffsigma = _diffsigma(Tp, Eg, nuclear_enhancement, hi_e_model)
        return _trapz_loglog(diffsigma * J, Ep)

    specpp = jax.vmap(_spec_at_one_energy)(jnp.atleast_1d(photon_energy_gev))
    rate = specpp * gas_number_density_cm3 * _C_CGS
    return rate[0] if scalar_input else rate


def pion_decay_photon_spectrum_per_ev(spectrum, photon_energy_gev, gas_number_density_cm3, **kwargs):
    """Same as :func:`pion_decay_photon_spectrum`, but in naima's own
    ``1/(s eV)`` output convention (1 GeV = 1e9 eV) for a direct comparison
    without any unit bookkeeping on the caller's side."""
    return 1e-9 * pion_decay_photon_spectrum(spectrum, photon_energy_gev, gas_number_density_cm3, **kwargs)
