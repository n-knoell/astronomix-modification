# Cosmic Rays in `astronomix` — Implementation Plan (Grey Phase)

**Goal.** Add grey (energy-integrated) cosmic-ray magnetohydrodynamics to `astronomix`, in **both** the finite-volume (`fv_mhd`) and finite-difference (`fd_mhd`) schemes, wired so that we can forward-model **hadronic γ-ray** and **leptonic radio/synchrotron + inverse-Compton** emission from **stellar winds and supernovae**. Keep everything differentiable end-to-end. Design the emission interface so the later **spectrally-resolved** upgrade (with P. Girichidis) is a drop-in replacement, not a rewrite.

---

## 1. Staged scope

- **Phase A — grey CR transport + feedback** (both schemes), analytic tests pass.
- **Phase B — injection** (shock DSA via the Pfrommer finder from PR #4; SN/wind source terms) + integration tests.
- **Phase C — differentiable emission operators** (hadronic γ; leptonic synchrotron/IC), validated against `naima`; first-light wind bubble + SNR–cloud case.
- **Phase D — gradient-based inference** demo (fit κ / injection efficiency to a synthetic map).
- **Phase E — spectrally-resolved CRs** (Girichidis collaboration) replacing the assumed spectral shape.

The wind/SNe *emission science* runs on `fv_mhd` first, because it needs **open/outflow boundaries** — which `fv_mhd` already has and `fd_mhd` does not (currently periodic-only, and unstable at B→0). CRs go into `fd_mhd` too, but validated on **periodic** tests and turbulence boxes until FD gets open BCs. Two-moment transport (below) is hyperbolic, so it fits both the FV Riemann framework and the FD WENO framework with one shared design.

---

## 2. Design decisions

**Transport: two-moment, not classical diffusion.** Evolve the CR energy density `e_cr` **and** the CR flux `F_cr` as an independent variable with a reduced free-streaming speed (Jiang & Oh 2018; Thomas & Pfrommer 2019). This makes the system hyperbolic + explicit (no stiff parabolic solve, no global linear system), GPU-friendly, and — critically for us — smooth and differentiable. It recovers anisotropic diffusion + streaming in the appropriate limit. Cost: one extra scalar (`e_cr`) plus the flux vector, and a CFL tied to the reduced speed (a tunable accuracy/cost knob). Regularize the streaming sign with a `tanh` to keep the adjoint clean.

**Grey closure.** Single CR-proton energy density, `γ_cr = 4/3`, `P_cr = (γ_cr − 1) e_cr`. Optionally a second grey CR-**electron** energy density for leptonic emission (see §3).

**Feedback coupling.** `−∇P_cr` added to the gas momentum equation (the wind driver); CR and gas-thermal energy tracked separately; adiabatic `−P_cr ∇·v` work term; streaming heating deposits CR energy into gas thermal. Positivity floor on `e_cr`; CR fast speed folded into the adaptive CFL.

**Anisotropic transport along B.** Diffusion/streaming along **B** with a monotonicity-safe operator (Sharma & Hammett 2007). In two-moment form the flux limiter is largely avoided; keep the S&H-style guard only where needed and prefer smooth regularization over `min/max` for differentiability.

**Injection.**
- *Shock DSA* via the Pfrommer shock finder (PR #4): it already returns Mach number, shock surfaces, and thermal-energy flux, so DSA injection is "the fraction of the kinetic-energy flux that goes to CRs instead of heat," `η_cr(M)` from Kang & Ryu (2013) / Caprioli & Spitkovsky (2014).
- *SN/wind source terms*: inject a prescribed CR energy fraction (canonically ~10% of `E_SN`) at the source, time-sequenced for clustered sources.

**Losses.** Hadronic (pion) losses on protons and Coulomb losses; synchrotron + IC + Coulomb + bremsstrahlung on electrons (electrons cool fast — needed even in grey, see §3). Hadronic loss rate is also the input to the γ-ray operator.

**FV vs FD wiring.**
- `fv_mhd`: `e_cr`, `F_cr` as advected quantities with the CR pressure contribution in the flux; source terms operator-split (reuse the existing operator-split structure). Inherits open/reflective/periodic BCs and positivity machinery. **Primary science path.**
- `fd_mhd`: `e_cr`, `F_cr` evolved with the existing 5th-order WENO/CT stencils. Periodic tests + turbulence boxes only until open BCs land. Add a robust B→0 fallback so CR transport doesn't inherit the FD zero-field instability.

---

## 3. Emission operators (differentiable JAX)

Design a single `emission` interface that takes a **particle-spectrum object** + local gas/field state and returns emissivity. In the grey phase the spectrum object is `(normalization from e_cr, slope, cutoff)`; in the spectral phase it becomes the binned spectrum. Same downstream code.

**Hadronic γ-ray (protons).** Pion-decay emissivity via the Kafexhiu et al. (2014) parametrization of the p–p → π⁰ → γ cross section (modern standard; Kelner et al. 2006 as alternative). Emissivity ∝ `n_gas × n_CR,p`, so the multiphase/dense target gas *is* the signal. Line-of-sight integrate to a map/SED. Reimplement Kafexhiu in JAX so gradients flow sim → spectrum → map; validate against `naima` (Zabalza 2015).

**Leptonic radio/synchrotron + IC (electrons).** Synchrotron from the grey electron population + local `|B|`; IC off the ISRF/CMB/dust field; bremsstrahlung optional. Formulae: Blumenthal & Gould (1970); validate against `naima`.
- *Electron treatment, grey.* Start with a fixed electron/proton injection ratio `K_ep ≈ 0.01–0.02` deriving electrons from the proton field (simplest, good for first light). Move to a **separately evolved** grey electron energy density with synchrotron+IC losses next, because electrons cool fast and their spatial distribution near shocks differs from protons (matters for synchrotron shell morphology and aged SNRs).

**Grey caveat (state it plainly).** With grey CRs the emission *normalization and morphology* are predicted, but the spectral *shape* is **assumed** (power law with slope from the shock Mach number / a fixed index, plus a cutoff). That is enough for luminosities, maps, and an SED-under-assumption — and it is exactly the piece the spectral-CR phase makes first-principles.

---

## 4. Test & validation ladder (accuracy + physical correctness)

Run in order; each is a CI gate. Follow group TDD conventions — write the test before the code.

**Unit / analytic (Phase A)**
1. Pure CR advection (translation of a `e_cr` blob) — shape/amplitude preserved.
2. Adiabatic compression: uniform squeeze ⇒ `e_cr ∝ ρ^{γ_cr}`.
3. Anisotropic diffusion oblique to the grid (Sharma & Hammett 2007) — heat stays along **B**, no cross-field leak.
4. Isotropic diffusion vs the analytic Gaussian Green's function (convergence order).
5. 1D streaming: recover streaming speed and streaming-heating rate.
6. **Two-fluid (CR-modified) shock tube** vs the semi-analytic solution (Pfrommer et al. 2006) — the single most diagnostic coupling test.

**Injection (Phase B)**
7. CR Sedov–Taylor blast: thermal/CR/kinetic energy partition.
8. DSA efficiency vs Kang & Ryu (2013) / Caprioli & Spitkovsky (2014) as a function of Mach number, using the PR #4 finder output.

**Integration / physical (Phase B–C)**
9. 1D steady CR-driven flow / CR-modified shock structure.
10. Wind-blown bubble with CR pressure (first-light emission case).
11. SNR expanding into a uniform then a clumpy medium.
12. Reproduce a published grey CR-ISM result as a code-comparison anchor (e.g. a SILCC-style stratified box, Girichidis et al. 2016; Simpson et al. 2016).

**Emission validation (Phase C)**
13. Pion-decay SED for a known proton population vs `naima`.
14. Synchrotron + IC SED for a known electron population vs `naima`.
15. End-to-end SNR–molecular-cloud case reproducing a pion-bump source morphology/SED (IC 443 / W44 analog; Ackermann et al. 2013).

**Differentiability (continuous, from Phase A)**
16. Finite-difference vs autodiff gradient check on a small CR problem (transport, then injection, then emission).
17. Gradient stability across a modest rollout — catches a floor/limiter silently killing the adjoint.

**Conservation & consistency**
18. Full energy budget (thermal + kinetic + magnetic + CR) closes to round-off with injection and streaming/collisional-loss accounting.
19. `∇·B` preservation unaffected by the CR module (both schemes).
20. FV vs FD consistency on shared periodic tests (tests 1–8).

---

## 5. Working bibliography

*Verify exact volumes/DOIs against ADS before citing in a paper — these are the working set.*

**Two-moment CR transport / numerics**
- Jiang & Oh (2018), ApJ — two-moment CR transport (reduced-speed flux variable).
- Thomas & Pfrommer (2019), MNRAS — CR hydrodynamics, two-moment.
- Sharma & Hammett (2007), JCP — monotonicity-preserving anisotropic diffusion.
- Pakmor et al. (2016), ApJL — anisotropic CR diffusion on a moving mesh.

**CR-MHD in the ISM / winds / feedback**
- Girichidis et al. (2016), ApJL — CR-driven outflows (SILCC).
- Simpson et al. (2016), ApJL — CR-driven winds from the ISM.
- Pfrommer et al. (2017), MNRAS — CR feedback in galaxy formation.
- Farber et al. (2018), ApJ — temperature-dependent (two-κ) transport in the multiphase ISM.
- Ruszkowski & Pfrommer (2023), A&A Review — CR feedback review (overview + parameter landscape).

**Shocks / DSA injection**
- Pfrommer et al. (2006), MNRAS — shock finder / Mach-number method (basis of PR #4).
- Schaal & Springel (2015), MNRAS — grid shock finder.
- Kang & Ryu (2013), ApJ — DSA acceleration efficiency vs Mach number.
- Caprioli & Spitkovsky (2014), ApJ — DSA efficiency from PIC.

**Emission**
- Kafexhiu et al. (2014), PRD — pion-decay γ-ray parametrization (primary).
- Kelner, Aharonian & Bugayov (2006), PRD — p–p interaction spectra (alternative).
- Blumenthal & Gould (1970), Rev. Mod. Phys. — synchrotron / IC / bremsstrahlung.
- Zabalza (2015), `naima` — reference implementation for validation.
- Ackermann et al. (2013), Science — pion-bump detection in IC 443 / W44 (validation target).

**Spectrally-resolved CRs (Phase E, with Girichidis)**
- Girichidis et al. (2020; 2022) — spectrally-resolved CR transport.
- Ogrodnik et al. (2021) — spectral CR method.
- Hopkins et al. (2022) — full-spectrum CR transport in galaxy simulations.

**Streaming / heating**
- Wiener et al. (2017) — CR streaming and heating in the ISM.

---

## 6. Open decisions for the team

- Reduced free-streaming speed: pick the largest value that leaves wind/emission properties unchanged (short convergence study).
- Electron treatment: fixed `K_ep` post-processing vs separately-evolved grey electrons — decide before Phase C emission work.
- FD open boundaries: implement now (unblocks wind/SNe on FD) or defer and keep FD for periodic turbulence only?
- Spectral-interface contract with Girichidis: agree the `spectrum` object API early so Phase E is a swap.

---

## 7. Claude Code kickoff prompt

Paste the block below into Claude Code at the repo root. It sets up the feature branch and scopes the work to Phase A with a design-doc-first, test-first approach.

```
We are adding grey cosmic-ray MHD to astronomix (JAX, differentiable). Read README.md,
the fv_mhd and fd_mhd scheme code, the existing physics modules under
astronomix/_physics_modules/, and the open shock-finder PR #4
(astronomix/_physics_modules/_shock_finder/) before writing anything.

Create and work on a new branch: feature/cosmic-rays-grey. Do not modify the shock-finder
PR; treat its ShockFinderResult (Mach number, shock surfaces, thermal-energy flux) as the
injection API you will consume later.

Scope for THIS session — Phase A only (grey CR transport + feedback), no emission yet:
- New module astronomix/_physics_modules/_cosmic_rays/, mirroring the _shock_finder layout.
- Grey CR closure: evolve CR energy density e_cr and CR flux F_cr (two-moment transport,
  reduced free-streaming speed, hyperbolic/explicit) following Jiang & Oh (2018) and
  Thomas & Pfrommer (2019). gamma_cr = 4/3, P_cr = (gamma_cr - 1) * e_cr.
- Feedback: -grad(P_cr) in gas momentum, adiabatic -P_cr div(v) work, separate CR/thermal
  energy, streaming heating into gas thermal, positivity floor on e_cr, CR fast speed in
  the adaptive CFL.
- Anisotropic transport along B; use smooth (tanh) regularization for the streaming sign
  instead of min/max/sign, to keep the code differentiable.
- Wire into BOTH schemes: fv_mhd (operator-split source terms, primary path, inherits open
  BCs + positivity) and fd_mhd (WENO/CT stencils, periodic-only for now; add a robust B->0
  fallback so CR transport does not inherit the FD zero-field instability).

Method of work (follow the group conventions — TDD, reproducibility, CI, and the
exploratory-vs-production-library-code distinction):
1. First write a short DESIGN.md in the new module: state variables, equations, operator-
   split ordering, the two-moment update, BC handling per scheme, and the differentiability
   plan. Stop and show it to me before implementing.
2. Then write tests BEFORE the implementation, as pytest cases + example notebooks matching
   the existing notebooks/ structure. Minimum set:
     (a) pure CR advection,
     (b) adiabatic compression: e_cr proportional to rho^gamma_cr,
     (c) oblique-to-grid anisotropic diffusion (Sharma & Hammett 2007) — no cross-field leak,
     (d) isotropic diffusion vs the analytic Gaussian Green's function (check convergence order),
     (e) two-fluid CR-modified shock tube vs the semi-analytic solution (Pfrommer et al. 2006),
     (f) a finite-difference-vs-autodiff gradient check on a small CR transport problem.
   Add these as CI gates.
3. Implement until the tests pass. Keep the emission operators, DSA injection, and SN source
   terms as clearly-marked stubs with typed interfaces (an emission function should take a
   "spectrum object" argument so spectrally-resolved CRs can replace grey later without a
   rewrite) — do not implement them this session.
4. Full energy-budget conservation check (thermal + kinetic + magnetic + CR) to round-off.

When running any JAX GPU code, use autocvd to select a free GPU. Keep single- and double-
precision paths working. Do not break existing MHD tests — run the current test suite before
and after. Summarize what changed and what is still stubbed at the end.
```

---

*Prepared for the astronomix team (Buck group, IWR Heidelberg). Phase E assumes the Girichidis spectral-CR collaboration.*
