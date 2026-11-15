# γ-ray modelling from the grey CR model — plan (slim version)

**Date:** 2026-11-13 (slimmed the same day on user request). **Status:** plan only, nothing implemented or
run. Branch `feature/cosmic-rays-grey`.

**Goal:** a γ-ray proxy that is physically accurate but needs no new simulation fields, no extra compute
and no extra resolution.

**Recommendation.** Post-process stored states with

    j_γ(E) = n_H · e_cr · K(E; α, q, E_cut)

- `K` is the Kafexhiu et al. (2014) π⁰-decay emissivity per unit **kinetic** CR energy density, for a proton
  spectrum that is a power law in **momentum**.
- `K` is computed once per spectral shape, then multiplied with the per-cell `n_H e_cr`.
- α is **one global, differentiable parameter**, not a per-cell field. The default is **α = 2.3** (§6).

The richer options (per-cell slope tracking, losses, instrument model) were considered and moved to §5:
for the current targets they add complexity for little accuracy (§2).

---

## 1. What already exists

| Piece | Where | State |
|---|---|---|
| Kafexhiu (2014) p–p → π⁰ → γ, JAX, differentiable | `cr_grey_emission.py` | matches `naima` (`useLUT=False`) to floating point (item 13) |
| Synchrotron + IC on isotropic Planck fields | `cr_grey_emission_leptonic.py` | matches `naima` (item 14) |
| `e_cr` → spectrum normalisation | `proton_spectrum_normalized_to_energy` | one shared shape (α = 2.0, E_cut = 100 TeV); **convention issue, §3 step 1** |
| Linearity trick: `K(E_γ)` once, map = `K ⊗ (amplitude · n)` | `cr_snr_molecular_cloud_pion_bump.py::_build_emission_maps` | works at 128³ |
| SNR–MC pion bump | item 15 | morphology + 50 MeV suppression pass; SED shape/level vs IC 443/W44 open (`PROGRESS.md` "Open (2026-10-04)") |
| Electrons | fixed `K_ep = 0.01` | decided for Phase C |

So the proxy is mostly a re-parametrisation of what item 15 already does, plus one convention fix.

---

## 2. Why a single global slope is enough

With the test-particle slope α = (r+2)/(r−1) (γ = 5/3), r = 4M²/(M²+3):

| M | r | α |
|---|---|---|
| 3 | 3.00 | 2.50 |
| 5 | 3.57 | 2.17 |
| 10 | 3.88 | 2.04 |

The energy injected per unit shock area scales as ζ(M)·M³ (same upstream sound speed):
- With Kang & Ryu (2013) (`dsa_efficiency_kang_ryu_2013`), an M = 3 shock gives ~1e-3 of the energy of an M = 10 shock.
- With the constant-ζ default it gives ~3 %.

So the CR energy, and therefore the γ-rays, comes from strong shocks with α ≈ 2.0–2.1. The weak shocks
that would vary the slope carry almost no energy. They are also where the finder's Mach error is largest
(~10 % at ~64 cells per radius, resolution study Q1). A slope read per cell from the local Mach number
would be noisy exactly where it varies, while a global slope does not depend on resolution.

The real spectral uncertainty is not Mach-driven. Observed SNRs need α ≈ 2.2–2.4, steeper than
test-particle theory gives for strong shocks, so α is better treated as a **parameter** (default 2.3, see §6; or
fitted) than predicted.

Hadronic/Coulomb losses do not shape the spectrum here either: t_pp ≈ 7e7 yr / n, about 7e5 yr at
n = 100 cm⁻³, far longer than SNR or CWB ages.

---

## 3. Steps

### Step 1 — Fix the `e_cr` → spectrum normalisation (core, ~0.5 day) — **DONE 2026-11-13**

**Result (2026-11-13, uncommitted).**
- **Code** (`cr_grey_emission.py`): `ProtonMomentumSpectrum` (single PL, or the Ackermann broken PL via
  `ACKERMANN13_IC443` / `ACKERMANN13_W44`) and `pion_decay_emissivity_per_ecr(E, spectrum)` → K.
  `j_γ = n_H e_cr K`; at the default α = 2.3, `K(1 GeV) = 1.818e-14` photons s⁻¹ GeV⁻¹ per (cm⁻³ · erg cm⁻³).
  The legacy functions are unchanged.
- **Test** `pytests/cosmic_rays_grey/cr_pion_emissivity_per_ecr.py` (GPU, passes). Measured worst errors:
  - 1a, normalisation vs `scipy.quad`: 9.1e-6;
  - 1a′, vs Pfrommer et al. 2017b Eq. 2 (incomplete beta function): 7.7e-6, which confirms that formula;
  - 1b, K vs `naima.PionDecay` fed an independent numpy distribution, for 7 cases (single and broken PL, Pythia8/Geant4/QGSJET, nuclear enhancement on/off): 9.1e-6;
  - 1c, AD vs FD for all 7 float spectrum parameters: 3e-7.
  - Figure and summary are in `pytests/cosmic_rays_grey/pics/gamma_step1_emissivity_per_ecr/`.
- **The level change is smaller than the estimate below.** K(new)/K(legacy) at the same α:

  | α | 0.1 GeV | 1 GeV | ≥ 3 GeV |
  |---|---|---|---|
  | 2.0 | 1.21 | 1.05 | 1.03 |
  | 2.3 | 1.30 | 1.06 | 1.03 |
  | 2.5 | 1.32 | 1.03 | 0.99 |

  The rest-mass over-count and the change from an energy-space to a momentum-space shape partly cancel. The
  difference is mainly in the pion bump (< 1 GeV).
- **Pre-existing bug found and fixed in `_trapz_loglog`.** The absolute floor `max(y, 1e-30)` made every bin
  with y < 1e-30 piecewise constant. This hit the unit-amplitude pion kernels in item 15 and Phase D: they
  came out up to 1.2 % **high** above ~30 GeV (+0.1 % at 1 GeV, ~0 at 50 MeV).
  - The leptonic kernels and the item-13 test (amplitude 1e36) were unaffected; items 13 and 14 still pass.
  - Two follow-up NaN-gradient guards were added: subnormal values are masked, with b = 0 in masked bins, and
    `log y2 − log y1` replaces `log(y2/y1)`.
  - Item 15 and Phase D were **not** rerun.

**The issue (as planned).** `proton_spectrum_normalized_to_energy` uses a power law in **total** energy, `J(E) ∝ E^-α`.
It sets `∫ E J dE` (rest mass included) over `E > E_th` (pion threshold) equal to the cell's CR energy. The
fluid `e_cr` is the **kinetic** energy of the whole population. A quick pre-implementation estimate put the
level bias at ~10–35 %; the measured change is in the table above.

**The fix** (Pfrommer et al. 2017b, Eqs. 1–2):
- `f(p) = C p^-α exp(-(p/p_max)^β) θ(p − q)`, with p in m_p c.
- `e_cr = ∫ f(p) T(p) dp` from q (T is the kinetic energy).
- Hand Kafexhiu `J(E) = f(p) dp/dE`.
- Expose `q` (default 0.5 as in Pfrommer et al. 2017b; state the choice) and `p_max`.
- Add the Ackermann et al. (2013) smoothly broken power law in momentum (`s1, s2, p_br`) as a second shape (PROGRESS fix #2).

**Interface:** a function `pion_emissivity_per_ecr(E_γ, alpha, q, p_max, shape=...)` returns `K(E_γ)` in
photons s⁻¹ GeV⁻¹ per (cm⁻³ · erg cm⁻³). Keep it jit/grad-able in α so α can be fitted.

**Tests:**
- 1a: normalisation integral vs by-hand quadrature (< 1e-10).
- 1b: γ spectrum vs `naima.PionDecay` with a `TableModel` of the same momentum spectrum.
- 1c: d/dα of a band flux, AD vs FD.

### Step 2 — Proxy maps and SEDs from stored states (~0.5–1 day) — **DONE 2026-11-15**

**Result (2026-11-15, uncommitted).**
- **Code** (`cr_grey_emission.py`):
  - `hadronic_target_field(rho, e_cr, code_units, hydrogen_mass_fraction)` gives `w = n_H e_cr` per cell. X_H is required on purpose: item 11/15 use ρ = n m_p, i.e. X_H = 1.
  - `pion_decay_band_energy_per_ecr` integrates over a band (64 points per decade, second order; < 1e-4 vs 640 per decade; 16 per decade would give 1.3e-3 in the pion bump).
  - `pion_decay_sed` returns E²dL/dE, or E²dF/dE given a distance.
  - `pion_decay_band_map` returns the projected intensity in erg cm⁻² s⁻¹ sr⁻¹ (pixel flux = I·(dx/d)²).
- **Test** `pytests/cosmic_rays_grey/cr_gamma_maps.py` (GPU, passes):
  - units vs astropy: 4e-16;
  - band integral convergence: 9.4e-5;
  - summed map flux = SED band integral: 4e-16 (9e-8 on the float32 stored state);
  - uniform sphere vs the analytic chord: 4 %, within the voxelisation tolerance of 6.5 %;
  - AD vs FD (α, one cell's e_cr): 5e-9.
- **Demo** on the stored item-11 240³ states (1 kyr, η = 0.1), no rerun. Output in `pics/gamma_step2_maps/`.
  - At α = 2.3 and d = 1.5 kpc, E²F(1 GeV) = 8.4e-12 (clumpy) and 7.2e-12 (uniform) erg cm⁻² s⁻¹. That is ~12× below IC 443, as expected for 1 kyr, n = 1 and 10× clumps.
  - L(0.1–100 GeV) = 1.1e34 erg/s, consistent with the hadronic loss rate (7.44e-16 Σ n e dV × ~1/3 × band fraction).
  - Clumpy/uniform = 1.17 at equal E_cr, and the three clumps are the map hotspots.
- **Property of the proxy:** with one global spectrum, the band maps are the same image times a constant.
  Energy-dependent morphology needs §5 option F (spectral CRs).

- **Map per band:** `F_band(pixel) = [∫_band K dE] · Σ_LOS n_H e_cr Δl / (4π d²)`.
- **SED:** `E² F(E) = E² K(E) · Σ_cells n_H e_cr ΔV / (4π d²)`, in erg cm⁻² s⁻¹, so it sits on Ackermann et al. (2013) axes.
- The target density is **hydrogen** only, `n_H = X_H ρ / m_p` (X_H ≈ 0.7). Kafexhiu/naima's nuclear-enhancement factor (on by default) already accounts for helium and heavier nuclei, in both the target gas and the CRs, so don't use Pfrommer's `n_N = ρ / m_p` with it (corrected 2026-11-13).
- Runs on stored states (`/export/scratch/nknoell`), with no reruns ([[feedback_no_reruns_for_plots]]). Item 15's script does not cache its final state: add caching the next time it runs anyway.
- **Fast cross-check:** the analytic Pfrommer et al. 2017b Eq. 3/7 emissivity (δ(α) = 0.14 α^-1.6 + 0.44, σ_pp(α) = 32 (0.96 + e^{4.4−2.4α}) mbarn). Measure the agreement per band, as a sanity check only.

### Step 3 — α-budget check (justifies the single slope; small) — **DONE 2026-11-15: PASS**

**Result (2026-11-15, uncommitted).**
- **How it was measured.** Instead of a per-step accumulator inside the solver, a pure diagnostic
  `cr_grey_injection.dsa_injection_rate_by_slope` is evaluated in the run's `snapshot_callable` at 400 evenly
  spaced times, and the rates are integrated in time on the host. The solver is untouched.
  - The diagnostic uses the same finder kwargs, Mach/flux, ζ(M) and area weights as `inject_crs_at_shocks`.
  - Unit test `pytests/cosmic_rays_grey/cr_dsa_slope_budget.py`: the binned total vs what the injection
    actually deposits agrees to 2e-12; the binning vs numpy to 2e-16.
- **Runs** (`/export/scratch/nknoell/gamma_step3/`, wrapper `run_alpha_budget.py`, η = 0.1 constant, 2080 Ti):
  item 15 at 128³, 192³ and 240³ (2 kyr), and item-11 uniform at 240³ (1 kyr).
- **Result: > 99.99 % of the injected CR energy is at α < 2.05 in all four runs.**
  - At 2 kyr, 3.3 % of the injecting cells have M < 9 (reflected or internal shocks), but they carry 4e-4 of
    the instantaneous rate. The median Mach number is 111.
  - E_inj = 1.66 / 2.03 / 2.15e50 erg at 128³ / 192³ / 240³, the known resolution under-count; E_cr(end) is
    about half of that because of adiabatic losses.
- **Caveat: this is specific to these setups.** The ambient gas is 1e4 K, so M ~ 100. Re-run the check
  (wrapper above) before using the proxy for hot media (1e6 K, M ~ 3–10) or CWB wings.
- **Memory note.** item 15's pytest enables x64 globally, so its simulation ran in float64. The wrapper turns
  x64 off (float32, like item 11). It also needs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`; then 240³ with the
  callback fits in 9.1 GB.

Add a run diagnostic (opt-in, global scalars, no fields): at each step, add the injected ΔE_cr into ~8
bins of α(M_CR-aware) and write the totals to the log/output.

**Pass criterion:** at least 90 % of the injected energy is in α < 2.2 for the SNR and SNR–MC runs. If it
fails for a target (for example a CWB with strong off-apex weakening), revisit §5 option A for that target only.

### Step 4 — Demo: item 15 redo (GPU, [[feedback_tests_on_gpu]]) — **DONE 2026-11-15**

**Result.** `pytests/cosmic_rays_grey/cr_gamma_snr_cloud_demo.py` reads the cached 240³ item-15 state
(2 kyr) and writes `pics/gamma_step3_4/`.
- With α = 2.3, E²F(1 GeV, 1.5 kpc) = 1.4e-11 erg cm⁻² s⁻¹. The SED peaks at **0.83 GeV**; IC 443's fit
  shape peaks at ~0.8 GeV. The legacy α = 2.0 peaked at ~150 GeV, so the PROGRESS 2026-10-04 shape issue is
  fixed.
- The level is 8.5× below IC 443's observed peak, as expected: the remnant is younger (2 kyr), the cloud is
  ~10 M_sun, and the box is not scaled to IC 443.
- Pion bump: at 50 MeV, E²F is 96 % below the extrapolation of the 3–30 GeV slope.
- The 1–10 GeV map is brightest 1.0 cell from the cloud centre (cloud radius 12 cells).
- IC on the CMB (K_ep = 0.01, no cooling) is 10× fainter than pion decay at 1 GeV, but dominates above
  ~100 GeV. Without cooling this is an upper limit.
- Band maps were not repeated: under a global α they are scaled copies (step 2).

On a stored (or newly cached) item-15 state:
- the SED for the default α = 2.3, with α = 2.0 and the Ackermann broken PL as comparison curves, against the IC 443/W44 points. This compares shape only; the setup is not scaled to either remnant (level discussion in `PROGRESS.md`).
- the projected band maps at 0.1–1, 1–10 and 10–100 GeV;
- the α-budget from step 3;
- the existing IC (`K_ep`) for comparison at 1 GeV.

### Optional — obliquity ζ(θ_Bn), only for MHD SNR runs (after step 4, per §6)

One multiplicative factor at injection, no new fields:

`g(θ) = ½ [tanh((θ_crit − θ)/δ) + 1]`, with θ_crit ≈ π/4 and δ ≈ π/18

The form comes from Pais et al. 2018 (**check the constants against the paper**), motivated by Caprioli &
Spitkovsky 2014. θ_Bn is measured between the finder normal and the upstream B. Opt-in flag, default off.
It is what sets SNR γ-ray *morphology* (polar caps vs a full shell). It is also the gap listed in
`CR_GAMMA_paper_scope.md`.

Test: oblique planar MHD shocks θ = 0…90° (`cr_oblique_planar_shocks.py`): injected/ζ(M) ≈ g(θ).

---

## 4. Effort

| Step | Touches dynamics? | Re-baseline? | Size |
|---|---|---|---|
| 1 normalisation + `K(E; α)` | no | item-15 numbers only | ~0.5 day |
| 2 maps/SEDs from stored states | no | no | ~0.5–1 day |
| 3 α-budget diagnostic | diagnostics only (opt-in) | no | ~0.5 day |
| 4 demo | no (post-processing) | — | ~1 day |
| opt. obliquity | yes (opt-in) | no | 1–2 days |

Total core: ~2.5–3 days, with no new state variables and no resolution requirement beyond what the run already has.

## 5. Later, only if needed (deferred, with the trigger that would bring each back)

| Option | What | Trigger |
|---|---|---|
| A. Slope bins | K passive fractions of `e_cr`, advected with F_cr, injected by α(r) | step 3 fails for a target |
| B. Hadronic + Coulomb losses | Λ_h = −7.44e-16 n_e e_cr, Λ_C = −2.78e-16 n_e e_cr (cgs), 1/6 of Λ_h heats gas (Pfrommer et al. 2017a; **coefficients to verify against the PDF**) | dense, long-lived gas (n ≳ 1e3 cm⁻³, age ≳ 1e5 yr), or ISM/galaxy boxes; then also consider steady-state spectra (Werhahn et al. 2021) |
| C. Instrument model | PSF/exposure/Poisson likelihood in JAX, validated with gammapy | fitting real Fermi-LAT/CTA data |
| D. Bremsstrahlung / secondary e± | port naima `Bremsstrahlung`; Kelner et al. 2006 secondaries | the band below ~200 MeV in dense clouds matters |
| E. CWB extras | anisotropic IC on stellar photons, γγ absorption | CWB γ-ray track (IC likely dominates there; Reitberger et al. 2014, Pittard et al. 2021) |
| F. Spectral CRs | Phase E with Girichidis | energy-dependent escape / diffusion into clouds (W28-type; Gabici et al. 2009) |

**Cross-section systematic:** Kafexhiu vs AAfrag/Kelner differ by about ±25 % in γ normalisation
(Kachelrieß et al. 2021). Quote it; don't port alternatives.

## 6. Decisions (resolved 2026-11-13)

1. **Default α = 2.3**, a power law in momentum with q = 0.5 and p_max = 1e5 (E_cut ≈ 100 TeV).
   - Why: it sits between Pfrommer et al. 2017b's 2.2 and Ackermann et al. 2013's s1 = 2.36 for IC 443/W44 below their breaks.
   - It is consistent with the Galactic source index (~2.2–2.4) implied by the observed ~2.7 CR spectrum minus a diffusion index of ~0.3–0.5.
   - It is softer than test-particle α ≈ 2.0, as observed SNRs require.
   - The step-4 demo also shows α = 2.0 (test particle) and the Ackermann broken PL as comparison curves.
2. **Obliquity after the demo.** It stays in "Optional" and is not started until step 4 is done.
3. **Code home: here.** Emission code goes in `astronomix/_modules/_cosmic_rays_grey/cr_grey_emission.py`; tests go in `pytests/cosmic_rays_grey/`.

---

## References

[A] = abstract/summary read; [F] = equations fetched from the arXiv HTML; **unverified** = check before coding.

- Ackermann et al. 2013, Science 339, 807 — pion bump in IC 443/W44; broken PL in momentum. arXiv:1302.3307
- Kafexhiu et al. 2014, PRD 90, 123014 — π⁰ γ cross-section (in code). [arXiv:1406.7369](https://arxiv.org/abs/1406.7369)
- Pfrommer, Pakmor, Simpson & Springel 2017b, ApJL 847, L13 — γ-rays from grey CRs; Eqs. 1–7. [F] [arXiv:1709.05343](https://arxiv.org/abs/1709.05343)
- Pfrommer, Pakmor, Schaal, Simpson & Springel 2017a, MNRAS 465, 4500 — CR loss rates. [F, §3.3 **unverified**] [arXiv:1604.07399](https://arxiv.org/abs/1604.07399)
- Pfrommer & Enßlin 2004, A&A 413, 17 — analytic hadronic emissivity. [F] [arXiv:astro-ph/0306257](https://arxiv.org/abs/astro-ph/0306257)
- Kang & Ryu 2013 — DSA efficiency ζ(M) (in code).
- Kachelrieß et al. 2021, PRD 104, 123027 — cross-section comparison. [A] [arXiv:2110.00496](https://arxiv.org/abs/2110.00496)
- Pais et al. 2018, MNRAS — obliquity-dependent acceleration, SN 1006 / Vela Jr. [A; g(θ) **unverified**] [arXiv:1805.03216](https://arxiv.org/abs/1805.03216); Pais & Pfrommer 2020 [arXiv:2009.06649](https://ar5iv.labs.arxiv.org/html/2009.06649)
- Caprioli & Spitkovsky 2014, ApJ 783, 91 — efficiency vs θ_Bn. [A] [arXiv:1310.2943](https://arxiv.org/abs/1310.2943)
- Werhahn et al. 2021, MNRAS 505 — steady-state spectra (deferred option B). [A] [arXiv:2105.10509](https://arxiv.org/abs/2105.10509), [arXiv:2105.11463](https://www.arxiv.org/abs/2105.11463)
- Winner et al. 2019 [arXiv:1903.01467](https://arxiv.org/abs/1903.01467); Wittor et al. 2017 [arXiv:1610.05305](https://arxiv.org/abs/1610.05305) — tracer-based slope tracking (deferred option A). [A]
- Girichidis et al. 2020 [arXiv:1909.12840](https://arxiv.org/abs/1909.12840); Böss et al. 2023 [arXiv:2207.05087](https://arxiv.org/abs/2207.05087) — spectral CRs (option F). [A]
- Gabici, Aharonian & Casanova 2009 [arXiv:0810.0035](https://www.arxiv.org/abs/0810.0035); Reitberger et al. 2014 [arXiv:1401.1323](https://arxiv.org/abs/1401.1323); Pittard et al. 2021 [arXiv:2104.07399](https://ar5iv.labs.arxiv.org/html/2104.07399). [A]
- Zabalza 2015 (naima) — validation reference.
