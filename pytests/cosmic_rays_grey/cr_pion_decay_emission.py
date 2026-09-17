"""
Pion-decay SED pytest (plan Sec. 4, item 13 -- "Emission validation", Phase C).

Validates :mod:`astronomix._modules._cosmic_rays_grey.cr_grey_emission`'s
JAX reimplementation of the Kafexhiu et al. (2014) pion-decay parametrization
against `naima`'s own ``radiative.PionDecay`` class (Zabalza 2015) -- the
plan's own choice of validation reference (Sec. 3), so unlike most earlier
ladder items no new reference solver needed to be built here; naima is
installed as a dev dependency for this check (not a runtime astronomix
dependency).

Scope, matching this ladder item's literal wording ("Pion-decay SED for a
*known* proton population vs naima"): given an externally-specified
exponential-cutoff power-law proton spectrum (this module's
``GreyProtonSpectrum`` -- the plan's grey "particle-spectrum object") and a
target gas density, check that the computed photon spectrum matches naima's,
across photon energy, across the four Monte Carlo high-energy models
Kafexhiu et al. (2014) supports, with and without the ISM nuclear-enhancement
factor, and across different proton spectral shapes. Deriving a spectrum's
normalization from a simulation's local ``e_cr`` field and line-of-sight
integrating to a map is ladder item 15's job, not this one's -- this is a
standalone physics-formula check (naima's own ``useLUT=False`` forces its
analytic formula path rather than a cached fit to it, so the comparison is
against the same underlying physics naima's docstring cites, Kafexhiu et al.
2014, not merely against naima's numerics).

Also includes a differentiability smoke check (``jax.grad`` through the
whole spectrum -> photon-rate chain is finite, no NaN) -- the plan's own
"Reimplement Kafexhiu in JAX so gradients flow sim -> spectrum -> map" goal
(Sec. 3), a lighter version of ladder items 16/17's dedicated FD-vs-AD
gradient checks (not yet extended to the emission operators as of this
ladder item).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
# A tight cross-check against naima needs 64-bit precision: this ladder
# item's spectrum amplitudes/rates span >40 orders of magnitude (particle
# count normalizations ~1e36-1e45 down to photon rates ~1e12), which
# overflows JAX's float32 default outright (confirmed: silently produces
# NaN through the log-log trapezoidal integration) long before precision
# would ever become the limiting factor.
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
from naima.radiative import PionDecay

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cr_grey_emission import (
    GreyProtonSpectrum,
    pion_decay_photon_spectrum,
    pion_decay_photon_spectrum_per_ev,
)

# ---- physical setup: a naima-tutorial-style ECPL proton population ----
AMPLITUDE_PER_EV = 1e36  # particles / eV, at E_0
E_0_TEV = 1.0
ALPHA = 2.1
E_CUTOFF_TEV = 30.0
NH_CM3 = 2.5  # target (ambient) gas number density

# photon energies to compare across: 100 MeV to 100 TeV, the standard naima
# tutorial range for this exact case
PHOTON_ENERGY_GEV = jnp.logspace(-1, 5, 40)

# naima's four Monte Carlo high-energy differential-cross-section models
# (Kafexhiu et al. (2014) Sec. III.B); Pythia8 is naima's (and this module's)
# default.
HI_E_MODELS = ("Pythia8", "Geant4", "SIBYLL", "QGSJET")

REL_TOL = 1e-6  # generous margin above the ~1e-12 float64 agreement observed


def _naima_and_mine(alpha, e_cutoff_tev, nuclear_enhancement, hi_e_model):
    """Compute the same SED both ways for one (alpha, e_cutoff, ...) choice."""
    amplitude = AMPLITUDE_PER_EV / u.eV
    e_0 = E_0_TEV * u.TeV
    e_cutoff = e_cutoff_tev * u.TeV
    nh = NH_CM3 / u.cm ** 3

    pdist = ExponentialCutoffPowerLaw(amplitude, e_0, alpha, e_cutoff)
    pp = PionDecay(pdist, nh=nh, nuclear_enhancement=nuclear_enhancement)
    pp.useLUT = False  # force the analytic formula (see module docstring)
    pp.hiEmodel = hi_e_model

    energy = PHOTON_ENERGY_GEV.copy() * u.GeV
    naima_spec = jnp.asarray(pp._spectrum(energy).to("1/(s eV)").value)

    spectrum = GreyProtonSpectrum(
        amplitude=float(amplitude.to(1 / u.GeV).value),
        e_0=float(e_0.to(u.GeV).value),
        alpha=alpha,
        e_cutoff=float(e_cutoff.to(u.GeV).value),
    )
    mine = pion_decay_photon_spectrum_per_ev(
        spectrum, PHOTON_ENERGY_GEV, NH_CM3,
        nuclear_enhancement=nuclear_enhancement, hi_e_model=hi_e_model,
    )
    return naima_spec, mine


def test_pion_decay_matches_naima_default():
    """Default configuration (Pythia8, nuclear enhancement on) across the
    full 100 MeV - 100 TeV photon-energy range."""
    naima_spec, mine = _naima_and_mine(ALPHA, E_CUTOFF_TEV, True, "Pythia8")
    rel_err = jnp.abs(mine - naima_spec) / jnp.abs(naima_spec)
    max_rel_err = float(jnp.max(rel_err))
    assert max_rel_err < REL_TOL, (
        f"Default pion-decay SED does not match naima's Kafexhiu et al. "
        f"(2014) implementation closely enough: max rel. err "
        f"{max_rel_err:.3e} >= tol {REL_TOL}."
    )
    return naima_spec, mine


def test_pion_decay_matches_naima_all_hi_e_models_and_nuclear_enhancement():
    """Cross-check across all four Monte Carlo high-energy models, with and
    without the nuclear-enhancement factor, and a couple of different
    spectral shapes -- makes sure agreement isn't a coincidence of one
    particular parameter choice."""
    cases = [
        dict(alpha=ALPHA, e_cutoff_tev=E_CUTOFF_TEV, nuclear_enhancement=False, hi_e_model="Pythia8"),
        dict(alpha=2.3, e_cutoff_tev=50.0, nuclear_enhancement=True, hi_e_model="Geant4"),
        dict(alpha=1.8, e_cutoff_tev=100.0, nuclear_enhancement=True, hi_e_model="SIBYLL"),
        dict(alpha=2.5, e_cutoff_tev=10.0, nuclear_enhancement=True, hi_e_model="QGSJET"),
    ]
    for case in cases:
        naima_spec, mine = _naima_and_mine(
            case["alpha"], case["e_cutoff_tev"], case["nuclear_enhancement"], case["hi_e_model"]
        )
        rel_err = jnp.abs(mine - naima_spec) / jnp.abs(naima_spec)
        max_rel_err = float(jnp.max(rel_err))
        assert max_rel_err < REL_TOL, f"Mismatch for {case}: max rel. err {max_rel_err:.3e}."


def test_pion_decay_gradients_are_finite():
    """Differentiability smoke check: jax.grad through spectrum params and
    gas density, all the way to a summed photon rate, is finite everywhere
    -- catches a floor/power/log silently killing the adjoint (same category
    as cr_pressure_speed_floor / dsa_efficiency_kang_ryu_2013's own
    NaN-gradient guards)."""

    def total_rate(amplitude, alpha, e_cutoff, e_0, nh):
        spectrum = GreyProtonSpectrum(amplitude=amplitude, e_0=e_0, alpha=alpha, e_cutoff=e_cutoff)
        rate = pion_decay_photon_spectrum(spectrum, PHOTON_ENERGY_GEV, nh)
        return jnp.sum(rate)

    e_0_gev = E_0_TEV * 1e3
    e_cutoff_gev = E_CUTOFF_TEV * 1e3
    amplitude_per_gev = AMPLITUDE_PER_EV * 1e9

    grad_fn = jax.grad(total_rate, argnums=(0, 1, 2, 3, 4))
    grads = grad_fn(amplitude_per_gev, ALPHA, e_cutoff_gev, e_0_gev, NH_CM3)

    assert all(jnp.isfinite(g) for g in grads), (
        f"Non-finite gradient through the pion-decay emission chain: {grads}."
    )
    assert all(g != 0.0 for g in grads), (
        f"A gradient that's exactly zero everywhere would mean this parameter "
        f"has no effect on the output, i.e. the adjoint was silently killed: {grads}."
    )


if __name__ == "__main__":
    naima_spec, mine = test_pion_decay_matches_naima_default()
    test_pion_decay_matches_naima_all_hi_e_models_and_nuclear_enhancement()
    test_pion_decay_gradients_are_finite()

    # Diagnostic plot: naima vs. this module's JAX reimplementation, overlaid.
    fig, (ax_spec, ax_err) = plt.subplots(1, 2, figsize=(12, 5))

    ax_spec.loglog(PHOTON_ENERGY_GEV, naima_spec, "-", lw=2.5, color="C0", label="naima (Kafexhiu+14)")
    ax_spec.loglog(PHOTON_ENERGY_GEV, mine, "--", lw=1.5, color="C1", label="astronomix (JAX)")
    ax_spec.set_xlabel(r"$E_\gamma$ [GeV]")
    ax_spec.set_ylabel(r"$dN/dE\,dt$ [1/(s eV)]")
    ax_spec.set_title("Pion-decay photon spectrum (ladder item 13)")
    ax_spec.legend()

    rel_err = jnp.abs(mine - naima_spec) / jnp.abs(naima_spec)
    ax_err.loglog(PHOTON_ENERGY_GEV, jnp.maximum(rel_err, 1e-20), "-", color="C2")
    ax_err.axhline(REL_TOL, color="grey", ls="--", lw=1, label=f"tol={REL_TOL:.0e}")
    ax_err.set_xlabel(r"$E_\gamma$ [GeV]")
    ax_err.set_ylabel("relative error vs. naima")
    ax_err.set_title("Agreement with naima")
    ax_err.legend()

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_pion_decay_emission.svg")
    plt.close(fig)
