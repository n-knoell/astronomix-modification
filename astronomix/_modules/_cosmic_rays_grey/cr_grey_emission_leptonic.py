"""
Leptonic (synchrotron + inverse-Compton) emission from the grey CR-electron
population (plan Sec. 3 "Emission operators"; test ladder item 14, Phase C).

JAX reimplementation of the same closed-form approximations naima's
``radiative.Synchrotron``/``radiative.InverseCompton`` classes (Zabalza
2015) use -- Aharonian, Kelner & Prosekin (2010) for synchrotron in a
random magnetic field, and Khangulyan, Aharonian & Kelner (2014) for
inverse-Compton scattering off an isotropic thermal (Planck/gray-body) seed
photon field -- so those classes serve directly as this module's validation
reference, matching ``cr_grey_emission.py``'s approach for the hadronic
channel (ladder item 13). The plan's own citation (Blumenthal & Gould 1970)
is the foundational physics both of these later, more numerically
convenient closed-form approximations build on; naima itself does not
implement B&G's formulas directly, so -- exactly as item 13 followed
Kafexhiu et al. (2014) rather than a from-scratch B&G/Kelner reimplementation
-- this module follows naima's actual algorithms so the comparison is
meaningful (naima IS the plan's chosen validation reference, Sec. 3).

Scope for this ladder item, matching its literal wording ("Synchrotron + IC
SED for a *known* electron population vs naima"): given an externally
specified electron spectrum (``GreyElectronSpectrum`` -- the same
exponential-cutoff-power-law "particle-spectrum object" convention as
``cr_grey_emission.GreyProtonSpectrum``) and a magnetic field strength /
isotropic thermal seed photon field, compute the synchrotron and IC photon
spectra. Deriving an electron population from the local CR-grey state
(fixed ``K_ep`` vs. a separately-evolved grey electron energy density -- the
plan's still-open "electron treatment" design decision, Sec. 6) and
line-of-sight integration to a map are both deferred to later ladder items,
exactly as ladder item 13 deferred the analogous proton-side questions to
item 15 -- this is a standalone physics-formula check, decoupled from any
running simulation or from *where* the spectrum's normalization comes from.

Only the isotropic-thermal-seed-photon-field IC case (naima's
``_iso_ic_on_planck``, i.e. the ``"CMB"``/``"FIR"``/``"NIR"`` presets) is
ported -- naima's anisotropic and monochromatic/tabulated seed-field cases
are not needed for this ladder item's "known population" SED check and are
left for whichever later ladder item first needs them.

Units: energies are plain floats in eV throughout (matching naima's own
default output convention, ``1/(s eV)``, and this module's electron-energy
grid), not astropy Quantities -- kept as bare JAX arrays so this stays
``jax.grad``-differentiable end to end, same convention as
``cr_grey_emission.py`` (there: GeV, matching Kafexhiu et al. (2014)'s own
units).

Differentiability: every formula below is floored against invalid domains
(division by zero, log/pow of a non-positive argument) even where a
``jnp.where`` mask would forward-value-mask it out, to avoid the
"``jnp.where`` differentiates every branch everywhere" NaN-gradient gotcha
-- see ``cr_grey_emission.py``'s module docstring for the same pattern
applied to the hadronic channel.
"""

# general
import math

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp

# units (host-side constants only, never traced)
from astropy import constants as _const
from astropy import units as _u

from astronomix._modules._cosmic_rays_grey.cr_grey_emission import _TINY, _trapz_loglog

#: electron rest energy, eV.
M_E_C2_EV = float((_const.m_e * _const.c ** 2).to(_u.eV).value)
#: erg per eV, for the synchrotron formula's photon-energy unit conversion.
_ERG_PER_EV = float((1.0 * _u.eV).to(_u.erg).value)

# naima.radiative reassigns its module-level `e` to the Gaussian/esu value
# (`e = e.gauss`) before using it in the synchrotron formula alongside
# cgs-valued m_e/c/hbar/B -- per that module's own comment on the gyroradius
# calculation, plain SI Coulombs would not convert correctly there. Matched
# here explicitly (`.gauss`, not the bare SI `.value`) so this stays
# numerically identical to the validation reference.
_E_CHARGE = float(_const.e.gauss.value)
_M_E_CGS = float(_const.m_e.cgs.value)
_C_CGS = float(_const.c.cgs.value)
_HBAR_CGS = float(_const.hbar.cgs.value)

#: Thomson cross section and the Khangulyan et al. (2014) IC prefactor,
#: copied as literal magic numbers from naima.radiative.InverseCompton (see
#: that class's ``_iso_ic_on_planck`` for the cgs derivation comment) rather
#: than re-derived via astropy, to guarantee bit-exact agreement with the
#: validation reference.
_IC_PREFACTOR = 2.6318735743809104e16
#: k_B * 1 Kelvin, in units of m_e c^2 (Khangulyan et al. (2014)'s own
#: convention for the dimensionless thermal-seed-photon "temperature").
_K_TO_MEC2 = 1.6863699549e-10
#: radiation constant a = 4*sigma_SB/c, erg/(cm^3 K^4) -- lets a seed
#: field's actual energy density differ from a pure blackbody's (naima's
#: "uf" dilution factor).
_A_RAD_CGS = float((4.0 * _const.sigma_sb / _const.c).to(_u.erg / (_u.cm ** 3 * _u.K ** 4)).value)


class GreyElectronSpectrum(NamedTuple):
    """Exponential-cutoff power-law electron spectrum -- the same
    "particle-spectrum object" convention as
    ``cr_grey_emission.GreyProtonSpectrum``, just for the electron
    population (matching naima's ``ExponentialCutoffPowerLaw``). All
    energies (``e_0``, ``e_cutoff``, and the ``e_total_ev`` passed to
    :func:`electron_number_density`) are in eV; ``amplitude`` is in
    particles / eV (a *total* particle-count spectrum, not a volumetric
    density -- see this module's and ``cr_grey_emission``'s docstrings)."""

    amplitude: float
    e_0: float
    alpha: float
    e_cutoff: float
    beta: float = 1.0


def electron_number_density(spectrum: GreyElectronSpectrum, e_total_ev):
    """J(E): particles per unit total electron energy, particles/eV."""
    x = e_total_ev / spectrum.e_0
    return (
        spectrum.amplitude
        * x ** (-spectrum.alpha)
        * jnp.exp(-((e_total_ev / spectrum.e_cutoff) ** spectrum.beta))
    )


def _lorentz_factor_grid(e_e_min_ev, e_e_max_ev, n_e_e_per_decade):
    """Electron Lorentz-factor grid + J(E) resampled onto it, matching
    naima's ``BaseElectron._gam``/``_nelec`` exactly (including the
    ``dN/dE -> dN/dgamma`` Jacobian, ``mec2``)."""
    log10_g_min = math.log10(e_e_min_ev / M_E_C2_EV)
    log10_g_max = math.log10(e_e_max_ev / M_E_C2_EV)
    n_g = max(10, int(n_e_e_per_decade * (log10_g_max - log10_g_min)))
    return jnp.logspace(log10_g_min, log10_g_max, n_g)


def _nelec_of(spectrum, gam):
    """particles per unit Lorentz factor: n(gamma) = J(E) * mec2, E = gamma*mec2."""
    return electron_number_density(spectrum, gam * M_E_C2_EV) * M_E_C2_EV


def _g_tilde(x):
    """AKP10 Eq. D7 -- the synchrotron kernel's fitting-function approximation."""
    x_safe = jnp.maximum(x, _TINY)
    cb = jnp.cbrt(x_safe)
    gt1 = 1.808 * cb / jnp.sqrt(1.0 + 3.4 * cb ** 2.0)
    gt2 = 1.0 + 2.210 * cb ** 2.0 + 0.347 * cb ** 4.0
    gt3 = 1.0 + 1.353 * cb ** 2.0 + 0.217 * cb ** 4.0
    return gt1 * (gt2 / gt3) * jnp.exp(-x_safe)


def synchrotron_photon_spectrum(
    spectrum: GreyElectronSpectrum,
    photon_energy_ev,
    b_field_gauss,
    *,
    e_e_min_ev: float = 1e9,
    e_e_max_ev: float = 1e9 * M_E_C2_EV,
    n_e_e_per_decade: int = 100,
):
    """Synchrotron differential photon spectrum dN/dE/dt, Aharonian, Kelner
    & Prosekin (2010), for a known electron population in a random magnetic
    field -- ladder item 14 (synchrotron half).

    Matches ``naima.radiative.Synchrotron(particle_distribution,
    B)._spectrum(photon_energy)`` (see
    ``pytests/cosmic_rays_grey/cr_synchrotron_ic_emission.py`` for the
    direct comparison). ``e_e_min_ev``/``e_e_max_ev``/``n_e_e_per_decade``
    are static integration hyperparameters, matching naima's own
    ``Eemin``/``Eemax``/``nEed`` defaults (1 GeV to ``1e9 * m_e c^2``).

    Args:
        spectrum: The electron population's assumed spectral shape.
        photon_energy_ev: Photon energy (or array of energies), eV.
        b_field_gauss: Isotropic random magnetic field strength, Gauss.

    Returns:
        Photon production rate, photons / (s * eV), same shape as
        ``photon_energy_ev``.
    """
    gam = _lorentz_factor_grid(e_e_min_ev, e_e_max_ev, n_e_e_per_decade)
    nelec = _nelec_of(spectrum, gam)

    # Floor B itself (the only quantity below that can be legitimately
    # exactly zero) rather than any of the CGS-erg-scale quantities derived
    # from it: those routinely sit at ~1e-17 to 1e-51 for realistic photon
    # energies/fields -- this module's shared `_TINY=1e-30` is sized for the
    # *dimensionless* quantities in cr_grey_emission.py and would silently
    # clobber these much smaller (but perfectly legitimate) values instead
    # of guarding an actual zero (confirmed directly: this was a real bug
    # caught by the naima cross-check, not a hypothetical one).
    b_field_gauss = jnp.maximum(b_field_gauss, 1e-30)

    photon_energy_ev = jnp.asarray(photon_energy_ev)
    photon_energy_erg = photon_energy_ev * _ERG_PER_EV

    cs1_0 = jnp.sqrt(3.0) * _E_CHARGE ** 3 * b_field_gauss
    cs1_1 = 2.0 * jnp.pi * _M_E_CGS * _C_CGS ** 2 * _HBAR_CGS * photon_energy_erg
    cs1 = cs1_0 / cs1_1  # shape (n_photon,); photon_energy_erg > 0 always, no floor needed

    e_c_erg = 3.0 * _E_CHARGE * _HBAR_CGS * b_field_gauss * gam ** 2 / (2.0 * _M_E_CGS * _C_CGS)
    eg_ec = photon_energy_erg[None, :] / e_c_erg[:, None]  # (n_gamma, n_photon); e_c_erg > 0 given the B floor above

    dnde = cs1[None, :] * _g_tilde(eg_ec)  # 1/(s erg), shape (n_gamma, n_photon)
    integrand = nelec[:, None] * dnde
    spec_per_erg = _trapz_loglog(integrand.T, gam)  # 1/(s erg), shape (n_photon,)
    return spec_per_erg * _ERG_PER_EV  # 1/(s eV)


def _g_planck_pair(x, params):
    """G34, Eqs. 20/24/25 of Khangulyan et al. (2014) -- the isotropic
    thermal-seed-photon IC kernel's fitting-function approximation."""
    alpha, a, beta, b, c = params
    x_safe = jnp.maximum(x, _TINY)
    pi26 = jnp.pi ** 2 / 6.0
    tmp = (1.0 + c * x_safe) / (1.0 + pi26 * c * x_safe)
    big_g = pi26 * tmp * jnp.exp(-x_safe)
    tmp2 = 1.0 + b * x_safe ** beta
    little_g = 1.0 / (a * x_safe ** alpha / tmp2 + 1.0)
    return big_g * little_g


#: Eqs. 26, 27 of Khangulyan et al. (2014).
_A3 = (0.606, 0.443, 1.481, 0.540, 0.319)
_A4 = (0.461, 0.726, 1.457, 0.382, 6.620)


def _iso_ic_on_planck(gam, seed_temperature_k, gamma_energy_mec2):
    """IC cross-section-like integrand for isotropic scattering off a
    blackbody photon spectrum, Eq. 14 of Khangulyan et al. (2014). ``gam``
    (electron Lorentz factor) and ``gamma_energy_mec2`` (photon energy in
    units of m_e c^2) broadcast to shape (n_gamma, n_photon)."""
    theta = seed_temperature_k * _K_TO_MEC2  # dimensionless k_B T / (m_e c^2)
    gam_col = gam[:, None]
    gamma_energy = gamma_energy_mec2[None, :]

    z = gamma_energy / gam_col
    # z = E_gamma / E_e must be < 1 (the photon can't leave with more energy
    # than the scattering electron carried); the `valid` mask below zeroes
    # out the unphysical (z >= 1) region's forward value regardless, but
    # computing (1 - z) directly (rather than 1 - clip(z, ..., 1 - _TINY))
    # matters even so: at this module's `_TINY = 1e-30`, `1.0 - (1.0 -
    # 1e-30)` rounds to exactly `1.0` in float64 (1e-30 is far below
    # machine epsilon at scale 1), so the clip is silently a no-op and
    # `1 - z_safe` can land on an exact `0.0` for any z close to or above 1
    # -- `1/(1-z_safe)` then genuinely diverges. The forward pass still
    # looks fine (the final `jnp.where` masks the *output*), but
    # `jnp.where` differentiates both branches, so this `inf`/NaN
    # contaminates the gradient even though it never affects the value --
    # confirmed directly (a real bug, not hypothetical): this exact
    # mechanism produced a NaN gradient during this ladder item's own
    # development. Flooring `(1 - z)` itself, at a scale (`1e-10`) that
    # actually survives float64 subtraction, avoids the cancellation
    # instead of trying to patch it after the fact.
    one_minus_z = jnp.clip(1.0 - z, 1e-10, 1.0)
    z_safe = 1.0 - one_minus_z
    x = z_safe / one_minus_z / (4.0 * gam_col * theta)

    cross_section = z_safe ** 2 / (2.0 * one_minus_z) * _g_planck_pair(x, _A3) + _g_planck_pair(x, _A4)
    cross_section = (theta / gam_col) ** 2 * _IC_PREFACTOR * cross_section

    valid = (gamma_energy < gam_col) & (gam_col > 1.0)
    return jnp.where(valid, cross_section, 0.0)


def inverse_compton_photon_spectrum_planck(
    spectrum: GreyElectronSpectrum,
    photon_energy_ev,
    seed_temperature_k: float,
    seed_energy_density_erg_cm3: float,
    *,
    e_e_min_ev: float = 1e9,
    e_e_max_ev: float = 1e9 * M_E_C2_EV,
    n_e_e_per_decade: int = 100,
):
    """Inverse-Compton differential photon spectrum dN/dE/dt off an
    isotropic thermal (Planck/gray-body) seed photon field, Khangulyan,
    Aharonian & Kelner (2014), for a known electron population -- ladder
    item 14 (IC half).

    Matches ``naima.radiative.InverseCompton(particle_distribution,
    seed_photon_fields=[["seed", T, u]])._spectrum(photon_energy)`` for a
    single isotropic thermal seed (see
    ``pytests/cosmic_rays_grey/cr_synchrotron_ic_emission.py``). Passing
    ``seed_energy_density_erg_cm3`` different from the pure-blackbody value
    at ``seed_temperature_k`` reproduces naima's diluted-blackbody
    convention (e.g. its ``"FIR"``/``"NIR"`` presets).

    Args:
        spectrum: The electron population's assumed spectral shape.
        photon_energy_ev: Photon energy (or array of energies), eV.
        seed_temperature_k: Seed photon field temperature, Kelvin.
        seed_energy_density_erg_cm3: Seed photon field energy density, erg/cm^3.

    Returns:
        Photon production rate, photons / (s * eV), same shape as
        ``photon_energy_ev``.
    """
    gam = _lorentz_factor_grid(e_e_min_ev, e_e_max_ev, n_e_e_per_decade)
    nelec = _nelec_of(spectrum, gam)

    photon_energy_ev = jnp.asarray(photon_energy_ev)
    eph_mec2 = photon_energy_ev / M_E_C2_EV

    blackbody_u = jnp.maximum(_A_RAD_CGS * seed_temperature_k ** 4, _TINY)
    dilution = seed_energy_density_erg_cm3 / blackbody_u

    gamint = _iso_ic_on_planck(gam, seed_temperature_k, eph_mec2)  # (n_gamma, n_photon)
    integral = _trapz_loglog((nelec[:, None] * gamint).T, gam)  # 1/s, shape (n_photon,)
    lum = dilution * eph_mec2 * integral  # 1/s
    return lum / jnp.maximum(photon_energy_ev, _TINY)  # 1/(s eV)
