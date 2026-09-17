"""
Synchrotron + inverse-Compton SED pytest (plan Sec. 4, item 14 -- "Emission
validation", Phase C).

Validates :mod:`astronomix._modules._cosmic_rays_grey.cr_grey_emission_leptonic`'s
JAX reimplementations of naima's ``radiative.Synchrotron`` (Aharonian, Kelner
& Prosekin 2010) and ``radiative.InverseCompton`` (isotropic thermal seed,
Khangulyan, Aharonian & Kelner 2014) against naima itself (Zabalza 2015) --
the plan's own choice of validation reference (Sec. 3), exactly as ladder
item 13 validated the hadronic channel against naima's Kafexhiu et al.
(2014) implementation. naima is a dev-only dependency (see item 13's pytest).

Scope, matching this ladder item's literal wording ("Synchrotron + IC SED
for a *known* electron population vs naima"): given an externally specified
exponential-cutoff power-law electron spectrum (``GreyElectronSpectrum``,
the same "particle-spectrum object" convention as
``cr_grey_emission.GreyProtonSpectrum``) plus a magnetic field strength
(synchrotron) or an isotropic thermal seed photon field (IC), check that
the computed photon spectra match naima's -- across photon energy, across
several spectral shapes/field strengths, and across naima's CMB/FIR/NIR
presets plus a custom diluted gray-body seed for IC. Deriving an electron
population from the local CR-grey state (the plan's still-open "electron
treatment" design decision, Sec. 6: fixed ``K_ep`` vs. a separately-evolved
grey electron energy density) and line-of-sight integration to a map are
both deferred to later ladder items, exactly as item 13 deferred the
analogous proton-side questions to item 15.

**Tolerance note:** both spectra span many decades in photon energy and
fall off exponentially past their cutoff, so far out in that tail (>~1e-30
of the peak value) floating-point-level noise between naima's numpy
implementation and this module's JAX one is no longer numerically
meaningful (values sub-`1e-100`, orders of magnitude past any physical
significance) -- comparisons below are restricted to photon energies where
naima's own value is within ``PEAK_FLOOR_FRACTION`` of that spectrum's
peak, exactly the same "don't chase floating-point noise past machine-
precision-scale values" judgment call as ladder item 8's tolerance
calibration.

Also includes a differentiability smoke check (``jax.grad`` through both
channels is finite and nonzero everywhere tested) -- catches a
floor/power/log silently killing the adjoint, same category as ladder item
13's own gradient check.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
# Same reasoning as cr_pion_decay_emission.py (item 13): these spectra span
# a huge dynamic range (a *total* particle-count normalization integrated
# over a wide electron-energy grid), which overflows JAX's float32 default.
jax.config.update("jax_enable_x64", True)

# general
from pathlib import Path

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# units (naima's own interface is astropy-Quantity-based)
import astropy.units as u

# naima (validation reference for this ladder item only -- see module docstring)
from naima.models import ExponentialCutoffPowerLaw
from naima.radiative import Synchrotron, InverseCompton

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cr_grey_emission_leptonic import (
    GreyElectronSpectrum,
    synchrotron_photon_spectrum,
    inverse_compton_photon_spectrum_planck,
)

# ---- physical setup: a naima-tutorial-style ECPL electron population ----
AMPLITUDE_PER_EV = 1e33
E_0_TEV = 1.0
ALPHA = 2.0
E_CUTOFF_TEV = 100.0
B_FIELD_GAUSS = 100e-6  # 100 microgauss, a typical ISM-ish random field

# photon energies spanning radio through gamma-ray
PHOTON_ENERGY_EV = jnp.logspace(-6, 12, 60)

REL_TOL = 1e-6
#: see module docstring's "Tolerance note".
PEAK_FLOOR_FRACTION = 1e-30


def _make_spectrum(amplitude_per_ev, e_0_ev, alpha, e_cutoff_ev):
    return GreyElectronSpectrum(amplitude=amplitude_per_ev, e_0=e_0_ev, alpha=alpha, e_cutoff=e_cutoff_ev)


def _naima_and_mine_sync(alpha, e_cutoff_tev, b_gauss):
    amplitude = AMPLITUDE_PER_EV / u.eV
    e_0 = E_0_TEV * u.TeV
    e_cutoff = e_cutoff_tev * u.TeV

    pdist = ExponentialCutoffPowerLaw(amplitude, e_0, alpha, e_cutoff)
    sy = Synchrotron(pdist, B=b_gauss * u.G)
    naima_spec = jnp.asarray(sy._spectrum(PHOTON_ENERGY_EV.copy() * u.eV).to("1/(s eV)").value)

    spectrum = _make_spectrum(
        float(amplitude.to(1 / u.eV).value), float(e_0.to(u.eV).value), alpha, float(e_cutoff.to(u.eV).value)
    )
    mine = synchrotron_photon_spectrum(spectrum, PHOTON_ENERGY_EV, b_gauss)
    return naima_spec, mine


def _naima_and_mine_ic(alpha, e_cutoff_tev, seed):
    amplitude = AMPLITUDE_PER_EV / u.eV
    e_0 = E_0_TEV * u.TeV
    e_cutoff = e_cutoff_tev * u.TeV

    pdist = ExponentialCutoffPowerLaw(amplitude, e_0, alpha, e_cutoff)
    ic = InverseCompton(pdist, seed_photon_fields=[seed])
    naima_spec = jnp.asarray(ic._spectrum(PHOTON_ENERGY_EV.copy() * u.eV).to("1/(s eV)").value)

    seed_info = ic.seed_photon_fields[next(iter(ic.seed_photon_fields))]
    seed_t = float(seed_info["T"].to(u.K).value)
    seed_u = float(seed_info["u"].to(u.erg / u.cm ** 3).value)

    spectrum = _make_spectrum(
        float(amplitude.to(1 / u.eV).value), float(e_0.to(u.eV).value), alpha, float(e_cutoff.to(u.eV).value)
    )
    mine = inverse_compton_photon_spectrum_planck(spectrum, PHOTON_ENERGY_EV, seed_t, seed_u)
    return naima_spec, mine


def _assert_matches(naima_spec, mine, label):
    mask = naima_spec > jnp.max(naima_spec) * PEAK_FLOOR_FRACTION
    rel_err = jnp.abs(mine[mask] - naima_spec[mask]) / jnp.abs(naima_spec[mask])
    max_rel_err = float(jnp.max(rel_err))
    assert max_rel_err < REL_TOL, (
        f"{label} does not match naima closely enough within {PEAK_FLOOR_FRACTION:.0e} of "
        f"the peak: max rel. err {max_rel_err:.3e} >= tol {REL_TOL}."
    )


def test_synchrotron_matches_naima():
    """Default configuration (100 uG) plus a weak-field/soft-spectrum and a
    strong-field/hard-cutoff variant."""
    naima_spec, mine = _naima_and_mine_sync(ALPHA, E_CUTOFF_TEV, B_FIELD_GAUSS)
    _assert_matches(naima_spec, mine, "Default synchrotron SED")

    naima_weak, mine_weak = _naima_and_mine_sync(1.7, 5.0, 3.24e-6)
    _assert_matches(naima_weak, mine_weak, "Weak-field/soft-spectrum synchrotron SED")

    naima_strong, mine_strong = _naima_and_mine_sync(2.5, 1e3, 1e-3)
    _assert_matches(naima_strong, mine_strong, "Strong-field/hard-cutoff synchrotron SED")

    return naima_spec, mine


def test_inverse_compton_matches_naima():
    """CMB, FIR, and NIR presets, plus a soft-spectrum variant."""
    naima_cmb, mine_cmb = _naima_and_mine_ic(ALPHA, E_CUTOFF_TEV, "CMB")
    _assert_matches(naima_cmb, mine_cmb, "IC-on-CMB SED")

    naima_fir, mine_fir = _naima_and_mine_ic(ALPHA, E_CUTOFF_TEV, "FIR")
    _assert_matches(naima_fir, mine_fir, "IC-on-FIR SED")

    naima_nir, mine_nir = _naima_and_mine_ic(ALPHA, E_CUTOFF_TEV, "NIR")
    _assert_matches(naima_nir, mine_nir, "IC-on-NIR SED")

    naima_soft, mine_soft = _naima_and_mine_ic(1.6, 10.0, "CMB")
    _assert_matches(naima_soft, mine_soft, "IC-on-CMB, soft-spectrum SED")

    return naima_cmb, mine_cmb


def test_inverse_compton_matches_naima_diluted_greybody():
    """A custom (diluted) gray body -- energy density different from the
    pure blackbody value at that temperature -- exercises naima's "custom
    seed" convention, not just its three presets."""
    amplitude = AMPLITUDE_PER_EV / u.eV
    e_0 = E_0_TEV * u.TeV
    e_cutoff = E_CUTOFF_TEV * u.TeV
    custom_t = 20.0 * u.K
    custom_u = 0.2 * u.eV / u.cm ** 3

    pdist = ExponentialCutoffPowerLaw(amplitude, e_0, ALPHA, e_cutoff)
    ic = InverseCompton(pdist, seed_photon_fields=[["custom", custom_t, custom_u]])
    naima_spec = jnp.asarray(ic._spectrum(PHOTON_ENERGY_EV.copy() * u.eV).to("1/(s eV)").value)

    spectrum = _make_spectrum(
        float(amplitude.to(1 / u.eV).value), float(e_0.to(u.eV).value), ALPHA, float(e_cutoff.to(u.eV).value)
    )
    mine = inverse_compton_photon_spectrum_planck(
        spectrum, PHOTON_ENERGY_EV, float(custom_t.to(u.K).value), float(custom_u.to(u.erg / u.cm ** 3).value)
    )
    _assert_matches(naima_spec, mine, "IC-on-diluted-20K-graybody SED")


def test_synchrotron_and_ic_gradients_are_finite():
    """Differentiability smoke check: jax.grad through spectrum params (and
    B / seed temperature+density) to a summed photon rate is finite and
    nonzero everywhere tested. Also regression-guards a real bug caught
    during this ladder item's development: an inadequately-scaled
    positivity floor (sized for dimensionless quantities, reused here where
    the natural scale is very different -- CGS erg-scale terms for
    synchrotron, and a near-unity kinematic ratio for IC) silently produced
    either wildly wrong forward values (synchrotron) or a NaN gradient
    (IC) despite a numerically clean forward pass -- see this module's
    ``cr_grey_emission_leptonic.py`` for the fixes and their reasoning."""

    def total_sync_rate(amplitude, alpha, e_cutoff, e_0, b_field):
        spectrum = GreyElectronSpectrum(amplitude=amplitude, e_0=e_0, alpha=alpha, e_cutoff=e_cutoff)
        return jnp.sum(synchrotron_photon_spectrum(spectrum, PHOTON_ENERGY_EV, b_field))

    def total_ic_rate(amplitude, alpha, e_cutoff, e_0, seed_t, seed_u):
        spectrum = GreyElectronSpectrum(amplitude=amplitude, e_0=e_0, alpha=alpha, e_cutoff=e_cutoff)
        return jnp.sum(inverse_compton_photon_spectrum_planck(spectrum, PHOTON_ENERGY_EV, seed_t, seed_u))

    e_0_ev = E_0_TEV * 1e12
    e_cutoff_ev = E_CUTOFF_TEV * 1e12
    amplitude_per_ev = AMPLITUDE_PER_EV

    sync_grads = jax.grad(total_sync_rate, argnums=(0, 1, 2, 3, 4))(
        amplitude_per_ev, ALPHA, e_cutoff_ev, e_0_ev, B_FIELD_GAUSS
    )
    assert all(jnp.isfinite(g) for g in sync_grads), f"Non-finite synchrotron gradient: {sync_grads}."
    assert all(g != 0.0 for g in sync_grads), f"A silently-zero synchrotron gradient: {sync_grads}."

    ic_grads = jax.grad(total_ic_rate, argnums=(0, 1, 2, 3, 4, 5))(
        amplitude_per_ev, ALPHA, e_cutoff_ev, e_0_ev, 2.72548, 4.1685946e-13
    )
    assert all(jnp.isfinite(g) for g in ic_grads), f"Non-finite IC gradient: {ic_grads}."
    assert all(g != 0.0 for g in ic_grads), f"A silently-zero IC gradient: {ic_grads}."


if __name__ == "__main__":
    naima_sync, mine_sync = test_synchrotron_matches_naima()
    naima_ic, mine_ic = test_inverse_compton_matches_naima()
    test_inverse_compton_matches_naima_diluted_greybody()
    test_synchrotron_and_ic_gradients_are_finite()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax_spec, ax_err, naima_spec, mine, title in (
        (axes[0, 0], axes[0, 1], naima_sync, mine_sync, "Synchrotron (100 uG)"),
        (axes[1, 0], axes[1, 1], naima_ic, mine_ic, "Inverse Compton (CMB)"),
    ):
        ax_spec.loglog(PHOTON_ENERGY_EV, jnp.maximum(naima_spec, 1e-300), "-", lw=2.5, color="C0", label="naima")
        ax_spec.loglog(PHOTON_ENERGY_EV, jnp.maximum(mine, 1e-300), "--", lw=1.5, color="C1", label="astronomix (JAX)")
        ax_spec.set_xlabel(r"$E_\gamma$ [eV]")
        ax_spec.set_ylabel(r"$dN/dE\,dt$ [1/(s eV)]")
        ax_spec.set_title(title)
        ax_spec.legend()

        mask = naima_spec > jnp.max(naima_spec) * PEAK_FLOOR_FRACTION
        rel_err = jnp.abs(mine - naima_spec) / jnp.abs(naima_spec)
        ax_err.loglog(PHOTON_ENERGY_EV[mask], jnp.maximum(rel_err[mask], 1e-20), "-", color="C2")
        ax_err.axhline(REL_TOL, color="grey", ls="--", lw=1, label=f"tol={REL_TOL:.0e}")
        ax_err.set_xlabel(r"$E_\gamma$ [eV]")
        ax_err.set_ylabel("relative error vs. naima")
        ax_err.set_title(f"{title}: agreement with naima")
        ax_err.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_synchrotron_ic_emission.svg")
    plt.close(fig)
