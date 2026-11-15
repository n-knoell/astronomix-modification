"""
Gamma-ray proxy, step 1 (GAMMA_RAY_PLAN.md Sec. 3): pion-decay emissivity per
unit kinetic CR energy density, K(E_gamma), for a proton spectrum that is a
power law in *momentum* (Pfrommer et al. 2017, ApJL 847, L13, Eqs. 1-2).

    j_gamma(E) = n_H * e_cr * K(E)

Checks:
  1a. normalisation integral (JAX, log-log trapezoid) vs scipy.integrate.quad;
  1a'. same vs Pfrommer et al. (2017) Eq. 2 (incomplete beta function, no cutoff);
  1b. K(E) vs naima.PionDecay (useLUT=False) fed with an independently built
      numpy particle distribution (single and Ackermann et al. 2013 broken
      power laws, several hi-E models, nuclear enhancement on/off);
  1c. AD vs central FD of a band-weighted K for every float spectrum parameter.
Also records the level change against the legacy convention
(``proton_spectrum_normalized_to_energy``: total-energy power law, total energy
above threshold) and plots E^2 K(E).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
jax.config.update("jax_enable_x64", True)

# general
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import astropy.units as u
from scipy import integrate, special

from naima.radiative import PionDecay

from astronomix._modules._cosmic_rays_grey.cr_grey_emission import (
    ACKERMANN13_IC443,
    ACKERMANN13_W44,
    ERG_TO_GEV,
    M_P_GEV,
    ProtonMomentumSpectrum,
    pion_decay_emissivity_per_ecr,
    pion_decay_photon_spectrum,
    proton_spectrum_normalized_to_energy,
    proton_unit_kinetic_energy_gev,
)

PICS_DIR = Path(__file__).resolve().parent / "pics" / "gamma_step1_emissivity_per_ecr"

PHOTON_ENERGY_GEV = np.logspace(-1.5, 4, 34)  # ~30 MeV - 10 TeV

NORM_REL_TOL = 3e-5      # JAX trapezoid vs quad (measured worst 9.1e-6, 100 pts/decade)
ANALYTIC_REL_TOL = 3e-5  # JAX (+ analytic tail) vs Pfrommer Eq. 2 (measured worst 7.7e-6)
NAIMA_REL_TOL = 3e-5     # K vs naima (measured worst 9.1e-6 = normalisation quadrature)
GRAD_REL_TOL = 2e-6      # AD vs central FD (measured worst 3.0e-7, FD noise)

SPECTRA = {
    "single a=2.3 q=0.5 (default)": ProtonMomentumSpectrum(),
    "single a=2.0 q=0.1": ProtonMomentumSpectrum(alpha=2.0, q=0.1),
    "single a=2.6 q=1.0 pmax=1e3": ProtonMomentumSpectrum(alpha=2.6, q=1.0, p_max=1e3),
    "IC 443 (Ackermann+13)": ProtonMomentumSpectrum(**ACKERMANN13_IC443),
    "W44 (Ackermann+13)": ProtonMomentumSpectrum(**ACKERMANN13_W44),
}


# ---------------- independent numpy/scipy reference ----------------
def _g_np(s: ProtonMomentumSpectrum, p):
    p = np.asarray(p, dtype=float)
    g = p ** (-s.alpha)
    if s.alpha_2 is not None:
        w = s.break_smoothness
        g = g * (1.0 + (p / s.p_break) ** ((s.alpha_2 - s.alpha) / w)) ** (-w)
    if s.p_max is not None:
        g = g * np.exp(-((p / s.p_max) ** s.cutoff_beta))
    return np.where(p >= s.q, g, 0.0)


def _norm_quad(s: ProtonMomentumSpectrum, p_hat_max=1e7):
    """integral_q^p_hat_max g(p) T(p) dp, by adaptive quadrature in ln p."""
    def integrand(lnp):
        p = np.exp(lnp)
        return _g_np(s, p) * M_P_GEV * (np.sqrt(1.0 + p * p) - 1.0) * p
    pts = [np.log(x) for x in (1.0, getattr(s, "p_break", 1.0), s.p_max or 1.0) if s.q < x < p_hat_max]
    val, _ = integrate.quad(integrand, np.log(s.q), np.log(p_hat_max), points=sorted(set(pts)) or None,
                            epsabs=0.0, epsrel=1e-13, limit=2000)
    return val


def _naima_K(s: ProtonMomentumSpectrum, nuclear_enhancement=True, hi_e_model="Pythia8"):
    """K(E) [1/(s GeV) per (cm^-3 erg cm^-3)] from naima, with the particle
    distribution built in numpy (amplitude from quad, Jacobian dp/dE)."""
    C = ERG_TO_GEV / _norm_quad(s)  # protons per unit p-hat in 1 cm^3 holding 1 erg

    def pdist(energy):
        E = np.atleast_1d(energy.to_value(u.GeV))
        p = np.sqrt(np.maximum(E * E - M_P_GEV ** 2, 1e-300)) / M_P_GEV
        J_per_gev = C * _g_np(s, p) * E / (M_P_GEV ** 2 * p)
        return J_per_gev * 1e-9 / u.eV

    pp = PionDecay(pdist, nh=1.0 / u.cm ** 3, nuclear_enhancement=nuclear_enhancement)
    pp.useLUT = False
    pp.hiEmodel = hi_e_model
    return pp._spectrum(PHOTON_ENERGY_GEV * u.GeV).to_value("1/(s eV)") * 1e9


# ---------------- tests ----------------
def test_normalisation_vs_quadrature():
    worst = 0.0
    for name, s in SPECTRA.items():
        mine = float(proton_unit_kinetic_energy_gev(s))
        ref = _norm_quad(s)
        rel = abs(mine / ref - 1.0)
        worst = max(worst, rel)
        print(f"  1a  {name:32s} norm JAX/quad - 1 = {mine / ref - 1:+.2e}")
        assert rel < NORM_REL_TOL, f"{name}: normalisation rel. err {rel:.2e}"
    return worst


def _pfrommer17_eq2(alpha, q):
    """Pfrommer et al. (2017) Eq. 2 with C_p = 1, GeV (2 < alpha < 3)."""
    a, b = (alpha - 2.0) / 2.0, (3.0 - alpha) / 2.0
    x = 1.0 / (1.0 + q * q)
    inc_beta = special.betainc(a, b, x) * special.beta(a, b)
    return M_P_GEV / (alpha - 1.0) * (0.5 * inc_beta + q ** (1.0 - alpha) * (np.sqrt(1.0 + q * q) - 1.0))


def test_normalisation_vs_pfrommer17_analytic():
    p_top = 1e12
    worst = 0.0
    for alpha in (2.3, 2.5, 2.7, 2.9):
        for q in (0.1, 0.5, 1.0):
            s = ProtonMomentumSpectrum(alpha=alpha, q=q, p_max=None)
            mine = float(proton_unit_kinetic_energy_gev(s, p_hat_max=p_top))
            # analytic tail beyond p_top: T ~ m_p (p - 1 + 1/(2p)) for p >> 1
            tail = M_P_GEV * (p_top ** (2 - alpha) / (alpha - 2) - p_top ** (1 - alpha) / (alpha - 1)
                              + 0.5 * p_top ** (-alpha) / alpha)
            ref = _pfrommer17_eq2(alpha, q)
            rel = abs((mine + tail) / ref - 1.0)
            worst = max(worst, rel)
            print(f"  1a' alpha={alpha} q={q}: (JAX+tail)/Eq.2 - 1 = {(mine + tail) / ref - 1:+.2e}")
            assert rel < ANALYTIC_REL_TOL, f"alpha={alpha}, q={q}: rel. err {rel:.2e}"
    return worst


def test_emissivity_vs_naima():
    cases = [(name, s, True, "Pythia8") for name, s in SPECTRA.items()]
    cases += [
        ("default, no nuclear enh., Geant4", ProtonMomentumSpectrum(), False, "Geant4"),
        ("default, QGSJET", ProtonMomentumSpectrum(), True, "QGSJET"),
    ]
    worst = 0.0
    for name, s, nuc, model in cases:
        mine = np.asarray(pion_decay_emissivity_per_ecr(
            jnp.asarray(PHOTON_ENERGY_GEV), s, nuclear_enhancement=nuc, hi_e_model=model))
        ref = _naima_K(s, nuc, model)
        ok = ref > 0
        rel = np.max(np.abs(mine[ok] / ref[ok] - 1.0))
        assert np.all(mine[~ok] == 0.0)
        worst = max(worst, rel)
        print(f"  1b  {name:36s} max |K/naima - 1| = {rel:.2e}")
        assert rel < NAIMA_REL_TOL, f"{name}: K vs naima rel. err {rel:.2e}"
    return worst


def _band_weighted_K(params, base: ProtonMomentumSpectrum):
    s = base._replace(**params)
    E = jnp.asarray([0.3, 1.0, 3.0, 10.0, 30.0])
    return jnp.sum(E * E * pion_decay_emissivity_per_ecr(E, s))


def test_gradients_ad_vs_fd():
    cases = [
        (ProtonMomentumSpectrum(), dict(alpha=2.3, q=0.5, p_max=1e5, cutoff_beta=1.0)),
        (ProtonMomentumSpectrum(alpha=2.6, q=1.0, p_max=1e2),  # cutoff in band -> p_max sensitivity
         dict(p_max=1e2, cutoff_beta=1.0)),
        (ProtonMomentumSpectrum(**ACKERMANN13_W44),
         dict(alpha=2.36, alpha_2=3.5, p_break=ACKERMANN13_W44["p_break"], break_smoothness=0.1)),
    ]
    worst = 0.0
    for base, params in cases:
        params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
        ad = jax.grad(_band_weighted_K)(params, base)
        f0 = float(_band_weighted_K(params, base))
        for k, v in params.items():
            h = 1e-5 * abs(float(v))
            fp = float(_band_weighted_K({**params, k: v + h}, base))
            fm = float(_band_weighted_K({**params, k: v - h}, base))
            fd = (fp - fm) / (2 * h)
            rel = abs(float(ad[k]) - fd) / max(abs(fd), 1e-12 * abs(f0) / abs(float(v)))
            worst = max(worst, rel)
            print(f"  1c  d/d{k:16s} AD={float(ad[k]):+.6e} FD={fd:+.6e} rel={rel:.1e}")
            assert np.isfinite(float(ad[k])) and rel < GRAD_REL_TOL, f"{k}: AD vs FD rel. err {rel:.2e}"
    return worst


def legacy_vs_new_level():
    """K(new) / K(legacy) for the same alpha and cutoff (~94 TeV)."""
    rows = []
    E = jnp.asarray([0.1, 0.3, 1.0, 3.0, 10.0, 100.0])
    for alpha in (2.0, 2.3, 2.5):
        new = np.asarray(pion_decay_emissivity_per_ecr(E, ProtonMomentumSpectrum(alpha=alpha)))
        legacy_spec = proton_spectrum_normalized_to_energy(ERG_TO_GEV, alpha=alpha, e_cutoff=1e5 * M_P_GEV)
        old = np.asarray(pion_decay_photon_spectrum(legacy_spec, E, 1.0))
        rows.append((alpha, new / old))
    return np.asarray(E), rows


if __name__ == "__main__":
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["Gamma-ray proxy step 1: K(E) per unit kinetic e_cr (cr_pion_emissivity_per_ecr.py)", ""]
    print("1a normalisation vs quad");           w1 = test_normalisation_vs_quadrature()
    print("1a' normalisation vs Pfrommer Eq.2"); w2 = test_normalisation_vs_pfrommer17_analytic()
    print("1b K vs naima");                      w3 = test_emissivity_vs_naima()
    print("1c AD vs FD");                        w4 = test_gradients_ad_vs_fd()
    lines += [f"1a  worst rel. err (norm vs quad)          {w1:.2e}  (tol {NORM_REL_TOL:.0e})",
              f"1a' worst rel. err (norm vs Pfrommer Eq.2) {w2:.2e}  (tol {ANALYTIC_REL_TOL:.0e})",
              f"1b  worst rel. err (K vs naima)            {w3:.2e}  (tol {NAIMA_REL_TOL:.0e})",
              f"1c  worst rel. err (AD vs FD)              {w4:.2e}  (tol {GRAD_REL_TOL:.0e})", "",
              "K(new) / K(legacy), same alpha, cutoff ~94 TeV:",
              "  E [GeV]  " + "  ".join(f"{e:8.1f}" for e in [0.1, 0.3, 1.0, 3.0, 10.0, 100.0])]
    E_l, rows = legacy_vs_new_level()
    for alpha, r in rows:
        lines.append(f"  a={alpha:<4}  " + "  ".join(f"{x:8.3f}" for x in r))
    k1 = float(pion_decay_emissivity_per_ecr(jnp.asarray(1.0)))
    lines += ["", f"K(1 GeV), default spectrum (alpha 2.3, q 0.5, p_max 1e5): {k1:.4e} "
                  "photons s^-1 GeV^-1 per (cm^-3 erg cm^-3)"]
    text = "\n".join(lines)
    print(text)
    (PICS_DIR / "summary.txt").write_text(text + "\n")

    # ---- figure: E^2 K(E) ----
    E = np.logspace(-1.5, 4, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, (name, s) in enumerate(SPECTRA.items()):
        K = np.asarray(pion_decay_emissivity_per_ecr(jnp.asarray(E), s))
        ax.loglog(E, E * E * K, color=f"C{i}", lw=2.2 if i == 0 else 1.4, label=name)
    legacy = proton_spectrum_normalized_to_energy(ERG_TO_GEV, alpha=2.0, e_cutoff=1e5 * M_P_GEV)
    K_old = np.asarray(pion_decay_photon_spectrum(legacy, jnp.asarray(E), 1.0))
    ax.loglog(E, E * E * K_old, "k--", lw=1.2, label="legacy convention, a=2.0")
    ax.set_xlabel(r"$E_\gamma$ [GeV]")
    ax.set_ylabel(r"$E^2 K$ [GeV s$^{-1}$ per (cm$^{-3}$ erg cm$^{-3}$)]")
    ax.set_title(r"$j_\gamma = n_H\, e_{cr}\, K(E)$, per unit kinetic $e_{cr}$")
    ax.set_ylim(1e-15, None)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(PICS_DIR / "E2K_per_ecr.png", dpi=150)
    plt.close(fig)
    print(f"wrote {PICS_DIR}")
