# Differentiable CR-MHD and γ-ray maps: the first cosmic-ray paper on `astronomix`, scoped

**Date:** 2026-10-29. **Requested by:** TB. The group has brainstormed cosmic rays (CRs) in
`astronomix`, and Niklas has pushed the implementation plan and most of its ladder. TB wants
a first paper on differentiable, GPU-accelerated CR physics. Its science case should be sharpened
toward gradient-based reconstruction, or a Fisher forecast, of γ-ray emission from supernova
remnants (SNRs) and/or colliding-wind binaries (CWBs). It should show what a gradient buys that a
non-differentiable production code such as AREPO does not. **No prompt preceded it.**

**Status: scope only. Nothing was run.** All code facts are read from Niklas's fork,
`n-knoell/astronomix-modification`, branch `feature/cosmic-rays-grey` at `ac4e411`
(2026-10-28):
- `astronomix/_modules/_cosmic_rays_grey/astronomix_CR_implementation_plan.md` (the plan);
- `DESIGN.md`, `PROGRESS.md` and `PROGRESS_PHASEB.md` in the same directory.

**None of it is on `leo1200/astronomix` `main`.** Numbers quoted from those logs are the logs'
numbers, not re-measured here.

**No earlier CR paper plan was found.** The places searched:
- this repository;
- Niklas's fork;
- earlier Claude Code sessions;
- TB's claude.ai docs.

The closest documents:
- the implementation plan, whose Phase D is a two-parameter "fit κ / injection efficiency to a
  synthetic map" demo;
- the *ASGARD Wind Papers I–III* doc (2026-10-21), which lists cosmic rays as **out of scope** for
  that series.

**Update, same day.** TB then supplied the group's drafted outline (Knöll et al., A&A format).
§12 reconciles this scope with it. Where they differ, the outline is the paper, and §12 says what
should change in it.

**References** are in §11, with the marks used in
[`SHOWCASE_bar_sgr_scope.md`](SHOWCASE_bar_sgr_scope.md):
- **[A]**: the claim was read in the abstract;
- **[M]**: metadata only, cited for existence and attribution;
- **unverified**: may not migrate into the paper.

---

## 0. The plan in ten lines

1. **The code is further along than a "first paper" usually is.** The ladder Phases A–D are done
   on Niklas's branch:
   - grey two-moment CR transport;
   - DSA injection through the 3D shock finder;
   - pion-decay and synchrotron + IC emission, matched to `naima`;
   - an SNR–molecular-cloud pion bump;
   - FD-vs-AD checks;
   - a κ fit to 0.4 %.

   The paper is a methods paper with a science demonstration, not a feasibility study.
2. **The science case as first posed fails the programme's own test.** A fit or Fisher forecast of
   a handful of global parameters fails it: for example κ, η, α and a cloud density for an SNR, or
   Ṁ, v∞, e and i for a CWB.
   [`RATIONALE.md` §4](../RATIONALE.md) concedes that a double-precision Fisher forecast at that
   size gains nothing from autodiff. Niklas's own CWB pipeline already does a 7-parameter
   inference with SBI. AREPO plus a grid of runs does it too.
3. **The version that passes is field-level.** It makes the unknowns a field: the
   CR-acceleration efficiency over the shock surface, η(Ω) or η(θ_Bn), and the 3D target-gas
   density of the cloud. Then asks *how many independent combinations of them a γ-ray map
   actually constrains*.
   - That is P ~ 10²–10⁵.
   - The parameters reach the observable through the dynamics (D-028).
   - The answer is publishable whichever way it comes out.
4. **The gradient's real contribution is access to `Jᵀ`, not cost alone.** A low-rank
   Gauss–Newton Fisher / Laplace posterior costs ~2k tangent-plus-adjoint solves for k informed
   directions, independent of P. A finite difference gives `J v` in two runs but never `Jᵀ w`, so
   a non-differentiable code cannot build that low-rank Fisher in fewer than ~P runs.
5. **Lead with the SNR–molecular-cloud system (IC 443 / W44 class), not the CWB.**
   - Its shock is expanding: Niklas measured 0.1 % AD-vs-FD through live DSA.
   - Its target gas *is* the signal.
   - The grey approximation is adequate for maps.
6. **The CWB is a weaker headline but a valuable control.** Its parameters are global; its
   wind-collision shocks are quasi-stationary, the regime where Niklas measured ~20 % AD-vs-FD
   error; and its collision region is unstable. It belongs in a second paper, Niklas's, as a
   gradient-versus-SBI comparison on his existing pipeline (§8).
7. **One ablation turns D-028 into a figure.** Compute the Fisher information with the dynamics
   live and with the dynamics frozen (emission post-processed on fixed snapshots, the way a forward
   grid is analysed). The difference is the information that arrives *through the dynamics*.
8. **Gate 0 is the frozen-`dt` gradient.** It is the ROADMAP's named prerequisite for any
   `astronomix` gradient paper, and the `stop_gradient` on `dt` is still at
   `time_integration.py:607–625` on Niklas's branch.
9. **"What AREPO can't do" must be worded as capability, never speed.**
   [`RATIONALE.md` §6](../RATIONALE.md) forbids an unmeasured speed comparison. AREPO's CR physics
   is richer: spectral CRs and a moving mesh. Concede that in the introduction.
10. **Before an Overleaf project is made, TB's call is needed on three points** (§10, proposed
    O-20):
    - the lead author, presumably Niklas;
    - the paper track, under O-15;
    - whether Girichidis is involved from Paper 1 or only from the spectral follow-up.

---

## 1. What exists, and what it does not cover

**On Niklas's branch, verified by its own tests** (ladder items in the plan's numbering):

| Piece | State | Source |
|---|---|---|
| Grey two-moment transport (`e_cr`, `F_cr`, reduced speed), FV unsplit RK2, HLLC + hybrid CR entropy, ETD2RK relaxation, realizability cap | done; items 1–6 pass | `DESIGN.md` "Resolved" sections, `PROGRESS_PHASEA.md` |
| Anisotropic transport along **B** (κ∥, κ⊥) | done, with a recorded cross-field-leak limitation at sharp CR structure | `DESIGN.md` "Known limitation…" (2026-10-05) |
| Streaming | implemented, but `DESIGN.md` records the physically correct version as **open** | `DESIGN.md` "Open: physically correct CR streaming" |
| DSA injection: 3D Pfrommer-type shock finder, Kang & Ryu (2013) η(M), CR-aware velocity-jump Mach | done; Mach bias ≈ 10 % at 64 cells per radius, shrinking with resolution | `PROGRESS.md` 2026-10-14, `PROGRESS_PHASEB.md` |
| Pion decay (Kafexhiu et al. 2014, a JAX port of `naima`'s analytic path) | matches `naima` to floating point | `DESIGN.md` item 13 |
| Synchrotron + IC, fixed `K_ep = 0.01` | matches `naima` | `DESIGN.md` item 14 |
| SNR–molecular-cloud pion bump, 128³ | peak 3.2 cells from the cloud centre; 95 % suppression at 50 MeV | `DESIGN.md` item 15 |
| FD vs AD: transport, injection on a frozen state, emission | 1e-2 tol; injection 3.4e-10, emission 1.8e-8 | `DESIGN.md` items 16–17 |
| Phase D: κ fit (D1); DSA-efficiency fit through a live rollout (D2) | D1 0.40 %; D2 6.35 % after 40 Adam steps, AD-vs-FD 0.10 % | `DESIGN.md` "Resolved: Phase D" |
| Energy budget, ∇·B | closes to ~1e-13 (MHD); ∇·B unaffected | items 18–19 |
| CWB with N-body orbit, DSA and CR diffusion, 64³/128³ | runs; 256³ does not fit 11 GB | `PROGRESS_PHASEB.md` 2026-10-20 |

**What the paper cannot lean on yet:**

- **Spectral shape is assumed, not predicted.** It is the plan's own "grey caveat". Maps and
  band-integrated fluxes are predictions. SED shape is not, so spectral index and cutoff enter
  only the observable. They are the control parameters that *do not* pass through the dynamics.
  - **Energy-dependent escape cannot be represented.** That is the W28-type physics, so the
    escaping-CR case is weaker in grey than the shock–cloud interaction case.
- **The SNR–MC setup has no self-consistent magnetic field.** Item 15 is hydro with
  `naima`'s default B. Obliquity-dependent injection η(θ_Bn) is **not implemented**: the finder
  gives the shock normal, and the upstream B has to be read and combined with it. That is new but
  small code.
- **The frozen adaptive time step.** `RATIONALE.md`'s survey records an O(1) forward-mode
  error (8.58 against converged FD 6.78) in `astronomix`'s *old* cosmic-ray test
  ([`LITERATURE_what_gradients_buy.md` §5](LITERATURE_what_gradients_buy.md)). The grey module
  has not been measured on this point.
- **Shock-finder reselection.** Item 16's first attempt gave ~20 % AD-vs-FD through a live DSA
  rollout of a weakening, near-stationary shock; D2 gave 0.1 % for an expanding one. Niklas
  records the stationarity explanation as a **working hypothesis**.
- **CR energy deficit against the two-fluid Sedov reference.** At 240³, `E_cr/E_SN` is 0.0675
  against 0.140, from the finder's area and Mach under-count and the unresolved early phase. A
  methods paper must report it, not hide it.
- **The FD (WENO) scheme carries no CRs.** The paper is FV-only.
- **Provenance.** The code is on a fork, not upstream. The paper needs it merged into
  `leo1200/astronomix` or tagged, with figures carrying a provenance row on the D-017 / D-031
  pattern.
- **Code home of the emission operators.** Leonard moved the differentiable observation operators
  (X-ray, NEI, synchrotron, radio, XRISM) to a separate repository, `observistix`
  (`astronomix` commit `61b7612`). Niklas's γ-ray operators sit in `_cosmic_rays_grey/`. Where
  they live is an open decision (§10) that should be settled before the paper cites a module path.

---

## 2. The science case, tested against the five conditions

The survey's regime
([`LITERATURE_what_gradients_buy.md` §1](LITERATURE_what_gradients_buy.md)) has five conditions:

- **C1:** many unknowns;
- **C2:** identifiable;
- **C3:** through the dynamics;
- **C4:** inside the predictability horizon;
- **C5:** low simulator stochasticity.

The candidate designs:

| Design | P | C1 | C2 | C3 | C4 | C5 | Verdict |
|---|---|---|---|---|---|---|---|
| SNR, global (κ, η, α, E_cut, n_cloud, K_ep) | ~6 | ✗ | likely | partly: α, E_cut, K_ep are observable-only | ✓ | ✓ | **Control row.** FD, a grid or SBI does it; AREPO + a grid does it. No gradient win to claim |
| SNR–MC, **η over the shock surface** (HEALPix η(Ω) or binned η(θ_Bn)) | 10²–10³ | ✓ | **the question** | ✓: injection → transport → CR-modified compression → target | ✓ (expanding shock; measure) | ✓ | **Lead case** |
| SNR–MC, **3D cloud density field** δ(x), with a CO-column constraint | 10³–10⁵ | ✓ | the question; CO fixes the column, the dynamics may fix depth | ✓: shock–cloud timing sets when each parcel is lit | needs measuring (shock–cloud KH/RT) | fixed seed, or the field is the latent | **Second science case**, after G2 |
| CWB, global (Ṁ₁, Ṁ₂, v∞, e, i, η, κ) | ~7–10 | ✗ | ✓ (Niklas's SBI) | ✓ | ✗ over many orbits (unstable collision region) | ✗ (his turbulence parameter) | **Not Paper 1.** Gradient vs SBI comparison, §8 |
| CWB, mass-loss history Ṁ(φ) over the orbit, from a phase-resolved γ-ray light curve | 50–200 | ✓ | the question; one light curve is little data | ✓ | ✗ across periastron; one passage may work | ✗ | Speculative; §8 |

**The paper's question, in one sentence:** *how many independent combinations of the
CR-acceleration efficiency over the shock, and of the target gas the shock runs into, does a γ-ray
map of an SNR–cloud system constrain — with Fermi-LAT today and with CTAO — and how much of that
information arrives through the dynamics rather than through the emission formula?*

Why this question and not "fit IC 443":

- **It is honest about C2.** The deliverable is the effective number of constrained parameters,
  computed with `mimirax`'s Fisher diagnostics. A small number is a result, not a failure: it
  says one-zone fits are not leaving information on the table.
  - Those diagnostics are `fisher_information`, `effective_parameters` and the SVD route
    `effective_parameters_from_jacobian`, in `mimirax/diagnostics/degeneracy.py` on `origin/main`.
- **It needs `Jᵀ`.** Forming the prior-preconditioned data-misfit Hessian matrix-free, by Lanczos,
  costs one tangent plus one adjoint solve per iteration. Its cost scales with the informed rank k,
  not with P. That is the scalable PDE-inverse-problem result (Isaac et al. 2015, survey §1 C1),
  transplanted here.
  - A non-differentiable code has `J v` by finite differences but no `Jᵀ w`.
  - It has to build J column by column: P + 1 runs.
- **It is the same physics AREPO is used for.** Pais et al. (2018, 2020) and Winner et al.
  (2020) study the obliquity dependence of acceleration in SNRs with suites of forward runs. They
  reach their conclusions by comparing discrete models with the data (§11.2). The comparison is
  natural and can be stated without a speed claim.
- **The CR-acceleration question is live.** PIC simulations predict efficient ion acceleration at
  quasi-parallel shocks (Caprioli & Spitkovsky 2014). Whether a γ-ray map can see that is a
  question an observer cares about independent of our method.

---

## 3. What the paper may and may not claim

**May claim**, once the gates in §6 pass:

- The first grey two-moment CR-MHD with DSA injection and hadronic + leptonic emission that is
  differentiable end to end and runs on GPUs. This is subject to the novelty pass in §11.
- Verified gradients of γ-ray maps with respect to:
  - transport parameters;
  - injection parameters, with the reselection caveat measured;
  - target-gas fields.
- The informed rank of Fermi-LAT-like and CTAO-like maps for the chosen fiducial, with the
  dynamics-versus-post-processing split.
- A twin reconstruction of η(Ω) (and, if G2 passes, of δ(x)) with a Laplace posterior.

**May not claim:**

- **Any speed comparison with AREPO or another production code**, in either direction. None has
  been measured ([`RATIONALE.md` §6](../RATIONALE.md)).
- **A spectral prediction.** The SED shape is assumed in grey.
- **A fit to IC 443, W44 or any real source**, unless a real-data section is added and passes its
  own gate. The plan here is twin experiments with real instrument responses.
- **A gradient result through live DSA without its AD-vs-FD number at the same setup.**
- **"First" anything** until the novelty pass (§11) has read the closest prior art.
- **That careful finite differences overstate the information in double precision.**
  [`RATIONALE.md` §6](../RATIONALE.md) records they were measured not to. If the forecast runs in
  single precision on GPU, measure FD-vs-AD Fisher there. That is where the probe found
  overstatement.

---

## 4. Paper outline (the Overleaf skeleton)

Working titles (pick one; the track name waits on O-15):
- *Differentiable cosmic-ray magnetohydrodynamics on GPUs: how much do γ-ray maps of supernova
  remnants constrain cosmic-ray acceleration?*
- *Through the shock: gradients of γ-ray maps with respect to cosmic-ray acceleration and the gas
  it illuminates.*

1. **Introduction.**
   - SNR–cloud systems as hadronic laboratories (the pion bump).
   - How parameters are constrained today:
     - one-zone SED fits;
     - forward grids of CR-MHD runs, with AREPO's CR work cited and its richer physics conceded.
   - The gap: map-level inference with many parameters and the dynamics in the loop.
   - Contributions.
2. **Method.**
   - 2.1 Grey two-moment CR-MHD in `astronomix`: equations as discretised, the FV scheme, HLLC +
     hybrid CR entropy, ETD2RK relaxation, realizability cap, anisotropic transport.
   - 2.2 DSA injection: the shock finder, η(M) and the new η(θ_Bn), smooth regularisations.
   - 2.3 Emission: pion decay, synchrotron + IC, `K_ep`, the linearity trick that reduces a 3D map
     to one spectral integral times a field, line-of-sight projection.
   - 2.4 Instrument model: PSF, exposure, background and Poisson likelihood, from public
     Fermi-LAT and CTAO responses. Possibly through `observistix`.
   - 2.5 Differentiation: reverse mode with checkpointing, the time-step treatment (G0),
     where gradients are deliberately stopped.
3. **Verification.** One table condensed from ladder items 1–19
   (test | metric | measured | expected | verdict), with the limitations stated:
   - Mach bias;
   - `E_cr` deficit against two-fluid Sedov;
   - cross-field leak;
   - streaming open.
4. **Gradient verification.**
   - AD vs FD per parameter class, in fp64 and fp32.
   - Before and after the `dt` treatment.
   - Live-DSA reselection, measured on an expanding and on a stalled shock.
   - The predictability horizon: tangent growth against SNR age.
5. **Information content of γ-ray maps.**
   - The fiducial SNR–MC system: magnetised, η(θ_Bn) on, one dense cloud.
   - Prior-preconditioned Gauss–Newton spectrum.
   - Effective parameters per tier and per instrument.
   - Adjoint sensitivity maps.
   - **The dynamics-versus-post-processing ablation.**
6. **Twin reconstruction.**
   - MAP of η(Ω) from a mock CTAO map, then a Laplace posterior.
   - Coverage on a few seeds.
   - If G2 passes, the joint η(Ω) + δ(x) case with a CO-like column constraint.
7. **Discussion.**
   - What grey cannot do and spectral CRs will (Girichidis).
   - Escaping CRs.
   - What a production code does better.
   - The CWB outlook.
   - Cost, labelled as measured on our hardware only.
8. **Conclusions.**

Appendices: the full ladder table, the shock-finder bias model (from the shock-finder thesis, if
its results land in time), the instrument-response details.

---

## 5. Figures

| # | Figure | Exists? |
|---|---|---|
| 1 | Pipeline schematic: simulation → CRs → emission → instrument → likelihood, with the tangent and adjoint arrows | no |
| 2 | The fiducial SNR–MC system: slices of ρ, \|B\|, `e_cr`, `n·e_cr` and the projected γ-ray map at two energies | partly (item 15 / item 11 figures, hydro only) |
| 3 | Verification panel: two-fluid shock tube, Sedov partition, pion and IC SEDs against `naima` | yes, in `pytests/cosmic_rays_grey/pics/`; must be regenerated at the paper commit |
| 4 | AD vs FD by parameter class and rollout length, with the `dt` treatment before and after | partly (items 16–17, Phase D) |
| 5 | Horizon: tangent norm against time, at two resolutions | no |
| 6 | Adjoint maps: ∂(counts in a region)/∂η(Ω) on the shock, and ∂/∂δ(x) in the cloud | no |
| 7 | Gauss–Newton eigenvalue spectrum against the prior's cut: Fermi-LAT-like versus CTAO-like, per tier | no |
| 8 | **Information through the dynamics**: the same spectrum with the hydro response frozen | no |
| 9 | Twin reconstruction: true, MAP and posterior σ of η(Ω) | no |

---

## 6. Gates, in order

| Gate | Pass condition | If it fails |
|---|---|---|
| **G0 — the time step** | Re-measure the frozen-`dt` error on the grey module. Then do one of two things. **Either** make the step schedule a fixed sequence taken from the fiducial run, for both AD and FD, and report the schedule's own modelling error (difference to the adaptive run) as a systematic. **Or** differentiate through `dt`. The derivative must agree with converged FD at ≤ 1 % on the item-17 setup | ROADMAP says no `astronomix` gradient paper ships. Nothing downstream starts |
| **G1 — gradient validity** | AD vs FD at three step sizes, for κ∥, κ⊥, global η, 8–16 η(Ω) directions and 8 random δ(x) directions, on the fiducial. ≤ 1 % in fp64; fp32 reported | Drop the failing class from §5–6. Transport and target fields alone still carry the paper |
| **G2 — horizon** | Tangent growth over the SNR age stays below a stated factor (set before running) at two resolutions | Shorten the window to the shock–cloud encounter. Drop δ(x) and keep η(Ω) |
| **G3 — information** | The Gauss–Newton spectrum is computed at k ≥ 50 with converged leading eigenvalues | None; the effective-parameter count is the result |
| **G4 — twin** | MAP recovers η(Ω) within its Laplace σ on ≥ 3 of 4 seeds | Report the bias; the paper becomes §§1–5 |

**Resolution and hardware.**
- The item-15 SNR–MC run is 128³ and fits an 11 GB RTX 2080 Ti forward; 240³ needs 8.6 GB.
- A checkpointed adjoint at 128³ fits a 40–80 GB A100/H100 by extrapolation (labelled so; not
  measured).
- GPU selection uses `autocvd` (org rule).
- Accuracy benches can run on CPU; timing goes to TB's hardware (memory
  `gpu-access-from-sessions`).

---

## 7. How to word the comparison with AREPO

AREPO's CR modules (Pfrommer et al. 2017 and successors) are the reference forward models. They
carry physics this paper does not:
- spectrally resolved CRs and electrons;
- a moving mesh;
- years of validation.

The paper's claim is narrow and is about capability:
- `astronomix` returns `J v` and `Jᵀ w` of a γ-ray map with respect to any input field, at the
  cost of one or two simulations.
- That makes three things affordable:
  - a field-level MAP;
  - a matrix-free Fisher spectrum;
  - an adjoint sensitivity map.
- A finite-difference code needs P + 1 runs to build the same Jacobian, which is infeasible at
  P ≳ 10³. That is a count of runs, not a timing.
- Directional finite differences give `J v` but never `Jᵀ w`.

Suggested sentence: *"A non-differentiable code can probe how the map responds to a chosen
change of the inputs, but not which inputs a given feature of the map responds to; the latter is
what a low-rank Fisher analysis and a field-level reconstruction need."*

---

## 8. The colliding-wind binary: where it belongs

**Not the headline of Paper 1**, for three measured or structural reasons:
1. **Global parameters.** Niklas's existing pipeline already infers seven of them with neural
   posterior estimation from Hα image time series. The pipeline is `n-knoell/SBI-for-CWBs`,
   published as Knöll, Buck, Branca & Viterbo 2026, arXiv:2606.10762. A gradient adds little at P ~ 7
   (C1).
2. **Quasi-stationary shocks.** In the binary's frame the collision-region shocks stand still.
   That is exactly the regime of item 16's ~20 % live-DSA error (a hypothesis, but the risk is
   concrete).
3. **An unstable collision region.** Thin-shell and Kelvin–Helmholtz instabilities make a
   multi-orbit rollout chaotic (C4). The run also has a stochastic turbulence parameter (C5).

**What it is good for:** a second paper, presumably Niklas's, that does what the survey says
astronomy almost never does. It compares, at matched cost, SBI against gradient-based inference on
the same simulator for the same CWB. The comparison runs on Hα (existing) and γ-ray (new) light
curves. It extends to a mass-loss history Ṁ(φ) only if one periastron passage passes the G2
horizon test. This is a natural contribution to the measured-guide paper (O-16). §7's outlook
sentence in Paper 1 points to it.

---

## 9. Risks

- **The informed rank is tiny for Fermi-LAT-like maps.** Likely, given its PSF against SNR sizes.
  This is a result, but the paper's interest then rests on the CTAO forecast. Make sure CTAO's
  public responses are usable before committing (§11).
- **Grey spectral shape.** A referee will ask why not spectral. The answer: maps first; spectral
  CRs with Girichidis are Paper 2.
- **Scoop.** The novelty pass (§11.1) found nothing through a CR solver. Two 2026 papers sit close:
  - Mishra-Sharma et al., differentiable γ-ray sky inference over templates;
  - Zandinejad et al., a field-level CR density without transport.

  The second shows a NIFTy group already reconstructing 3D CR fields. Read both in full before
  the intro is written.
- **No citable CTAO response.** No DOI-bearing IRF or sensitivity paper was verified (§11.6). If
  none exists, the forecast cites the CTAO data release and the GPS paper, or falls back to a
  parametrised PSF stated as an assumption.
- **Code churn.** The branch is large (> 13 000 lines of logs alone) and unmerged. Freeze a paper
  commit early and carry provenance.
- **The shock-finder bias** (Mach ~10 % low at 64 cells per radius) propagates into η. State it as
  a calibration systematic. The shock-finder thesis may supply a correction.

---

## 10. Open decisions (proposed O-20)

**Proposed O-20: the first `astronomix` CR paper.** Settle before an Overleaf project exists.

1. **Lead and authors.**
   - Niklas leads by default.
   - Leonard, as `astronomix` lead (D-037).
   - The shock-finder thesis student, if the bias correction lands.
   - Girichidis now or at Paper 2.
2. **Track and number** under O-15.
3. **Code home.** The paper commit must be on `leo1200/astronomix` (a PR from the fork) or a
   tagged fork commit. The γ-ray operators go to `_cosmic_rays_grey/` or `observistix`.
4. **Scope of Paper 1:**
   - η(Ω) only;
   - η(Ω) + δ(x);
   - or + the CWB control.

   This file recommends η(Ω), with δ(x) behind G2 and the CWB as Paper 2.
5. **G0 route:** a frozen schedule with its systematic reported, or differentiating through `dt`.
   This choice is also Leonard's, since it touches `time_integration.py` for every `astronomix`
   user.

---

## 11. References

**How it was made.**
- A literature subagent searched on 2026-10-29.
- This session then re-checked, on the same day:
  - every DOI in §11.2–§11.5 through the Crossref API, matched on first author, journal, volume
    and pages;
  - the five new arXiv IDs in §11.1 through the arXiv API, matched on title and authors.
  - The `astronomix` methods paper (arXiv:2607.18176) and Isaac et al. 2015 were not re-checked:
    they were already verified in [`LITERATURE_what_gradients_buy.md`](LITERATURE_what_gradients_buy.md).
  - **Crossref dates Pfrommer et al. 2017 to 2016**, its online issue; cite it as 2017, MNRAS 465.
- Everything is **[M]** unless marked **[A]**: the subagent read abstracts, this session read
  metadata only.
- Run counts in the AREPO papers were **not** read: their abstracts do not state them. Do not
  quote a number of runs until the full texts are read.

### 11.1 Novelty — the nearest prior art

**Verdict: related but distinct.** No work was found that runs gradient-based, adjoint or Fisher
inference *through* a CR transport or CR-MHD simulation to fit γ-ray or radio data. Re-check before
submission. The nearest:

- **Mishra-Sharma, Slatyer, Sun & Wu 2026**, arXiv:2604.08648, *High-dimensional inference for
  the γ-ray sky with differentiable programming* [A].
  - It is a JAX inference over the Galactic-centre sky with about 40 parameters.
  - Its gradients go through **fixed emission templates**, not a CR solver.
  - It is the paper a referee will name first; the intro must say how we differ.
- **Zandinejad, Roth, Phan, Edenhofer, Frank et al. 2026**, arXiv:2605.22739, *A
  three-dimensional tomographic reconstruction of the Galactic cosmic-ray proton density* [A].
  - Field-level reconstruction with NIFTy.
  - The CR density is a free Gaussian-process field, with no transport or dynamics.
  - It is the closest analogue of §5's δ(x) tier, and the cleanest contrast for the
    "information through the dynamics" ablation.
- **Diao, Li, Grumitt & Mao 2024**, arXiv:2410.01136, `synax` [A]: differentiable synchrotron
  line-of-sight integration with HMC. There are no dynamics.
- **In-house, to cite and stand apart from:**
  - Storcks & Buck 2024, arXiv:2410.23093 (`jf1uids`);
  - the `astronomix` methods paper, arXiv:2607.18176;
  - **Knöll, Buck, Branca & Viterbo 2026, arXiv:2606.10762**: amortised SBI of seven CWB
    parameters from Hα image time series. It is not gradient-based and not γ-ray, and it is the
    baseline §8's Paper 2 compares against.

### 11.2 AREPO and the current baseline

- Pfrommer, Pakmor, Schaal, Simpson & Springel 2017, MNRAS 465, 4500 — doi:10.1093/mnras/stw2941.
  CR-MHD in AREPO.
- Pais, Pfrommer, Ehlert & Pakmor 2018, MNRAS 478, 5278 — doi:10.1093/mnras/sty1410.
  Obliquity-dependent acceleration in SNR blast waves [A].
- Pais, Pfrommer, Ehlert, Werhahn & Winner 2020, MNRAS 496, 2448 — doi:10.1093/mnras/staa1678.
  TeV morphology against the magnetic coherence scale, run by run [A].
- Pais & Pfrommer 2020, MNRAS 498, 5557 — doi:10.1093/mnras/staa2827. A suite of SN 1006 models;
  the best fit comes from a discrete suite [A].
- Winner, Pfrommer, Girichidis & Pakmor 2019, MNRAS 488, 2235 — doi:10.1093/mnras/stz1792.
  Spectral CR electrons (CREST).
- Winner, Pfrommer, Girichidis, Werhahn & Pais 2020, MNRAS 499, 2785 — doi:10.1093/mnras/staa2989.
  SN 1006 across bands, models chosen by comparison [A].
- Zabalza 2015, `naima`, PoS(ICRC2015)922 — doi:10.22323/1.236.0922, arXiv:1509.03319. One-zone
  MCMC: the spectrum-only baseline.

### 11.3 Observational anchors

- Ackermann et al. 2013, Science 339, 807 — doi:10.1126/science.1231160. The pion-decay signature
  in IC 443 and W44.
- Aharonian et al. (H.E.S.S.) 2008, A&A 481, 401 — doi:10.1051/0004-6361:20077765. The W 28 clouds.
- Gabici, Aharonian & Casanova 2009, MNRAS 396, 1629 — doi:10.1111/j.1365-2966.2009.14832.x.
  Clouds lit by escaping CRs.
- CTA Consortium (Abe et al.) 2024, JCAP 2024(10), 081 — doi:10.1088/1475-7516/2024/10/081,
  arXiv:2310.02828. Galactic Plane Survey prospects.
- CTA Consortium, *Science with the Cherenkov Telescope Array*, World Scientific —
  doi:10.1142/10986.
- **CWB, for §8:**
  - Abdo et al. 2010, ApJ 723, 649 — doi:10.1088/0004-637X/723/1/649 (η Car with Fermi).
  - Reitberger et al. 2015, A&A 577, A100 — doi:10.1051/0004-6361/201525726 (η Car orbital
    variability).
  - Martí-Devesa & Reimer 2021, A&A 654, A44 — doi:10.1051/0004-6361/202140451.
  - H.E.S.S. Collaboration 2020, A&A 635, A167 — doi:10.1051/0004-6361/201936761 (η Car at VHE).
  - Pshirkov 2016, MNRAS 457, L99 — doi:10.1093/mnrasl/slv205 (γ² Vel / WR 11 detected; WR 140
    upper limits only).
  - Martí-Devesa et al. 2020, A&A 635, A141 — doi:10.1051/0004-6361/202037462.
  - Martí-Devesa, Reimer & Reimer 2023, A&A 670, L6 — doi:10.1051/0004-6361/202245332 (Apep
    limits).

### 11.4 CWB γ-ray modelling (for §8)

- Reitberger, Kissmann, Reimer & Reimer 2014, ApJ 789, 87 — doi:10.1088/0004-637X/789/1/87. 3D
  hydro + particle transport for a generic WR+B system; no fitting [A].
- Grimaldo et al. 2019, ApJ 871, 55 — doi:10.3847/1538-4357/aaf6ee. DSA on an MHD background,
  typical systems [A].
- White et al. 2020, A&A 635, A144 — doi:10.1051/0004-6361/201937031. η Car, a semi-analytic
  two-zone model fitted to Fermi + NuSTAR. It is the nearest thing to inference from a γ-ray light
  curve, and it is analytic [A].
- **Nothing found** infers CWB parameters from γ-ray data through a hydro or MHD forward model.

### 11.5 Methods

- Jiang & Oh 2018, ApJ 854, 5 — doi:10.3847/1538-4357/aaa6ce.
- Thomas & Pfrommer 2019, MNRAS 485, 2977 — doi:10.1093/mnras/stz263.
- Kafexhiu, Aharonian, Taylor & Vila 2014, PRD 90, 123014 — doi:10.1103/PhysRevD.90.123014.
- Caprioli & Spitkovsky 2014, ApJ 783, 91 — doi:10.1088/0004-637X/783/2/91 (efficiency against
  obliquity).
- Kang & Ryu 2013, ApJ 764, 95 — doi:10.1088/0004-637X/764/1/95.
  - **Check** that `dsa_efficiency_kang_ryu_2013` follows this paper and not Kang, Ryu & Jones
    2012 (ApJ 756, 97, doi:10.1088/0004-637X/756/1/97).
- Isaac, Petra, Stadler & Ghattas 2015 — already DOI-checked in
  [`LITERATURE_what_gradients_buy.md`](LITERATURE_what_gradients_buy.md); cite from there.

### 11.6 Unverified — may not migrate into the paper

- **A CTAO instrument-response or sensitivity paper with a DOI.** None was found; the public IRFs
  may exist only as a data release. The forecast needs a citable response, so this is a real gap
  (§9).
- A refereed Balbo & Walter paper on η Car hydro + γ-ray modelling (proceedings only).
- Any WR 140 γ-ray detection.
- Run counts in the Pais and Winner papers.

---

## 12. Reconciliation with the drafted outline (2026-10-29)

**Where the outline lives.** TB supplied the group's drafted outline later the same day:
*Differentiable cosmic-ray physics on GPUs — Gradient-based inference of injection and transport
from non-thermal emission*. It is 4 pages, A&A format, with authors Knöll, Buck, Storcks and
others. The copy is at `~/Documents/Nexus/Studenten/Niklas/CRs/CRs_astronomix.pdf`, outside every
repository, and is not vendored here.

**The outline is the paper. This scope is advice on it.**

**Applied to the source.** §12.2–§12.4 were written into the outline's `main.tex` (same folder)
on 2026-10-29, as tagged `% REV(2026-10-29)` comments and in-place prose edits:
- the abstract;
- contributions item 4;
- Figs. 5–7 captions;
- a new dynamics-ablation figure.

The original is kept beside it as `main_before_revision_2026-10-29.tex`. The edited file compiles
with `aa.cls`: 4 pages, citations undefined because no `references.bib` exists yet.

It settles two of §10's open decisions:
- Niklas is lead author;
- the venue is A&A.

It also confirms one of this scope's recommendations: the CWB is not in Paper 1.

### 12.1 What already agrees

- Outline §§2–4 (method, validation, differentiating through the simulation) map one-to-one onto
  §4 items 2–4 here.
- The outline's "differentiation horizon" (its Fig. 4, §4.2) is this file's G1 + G2.
- Fig. 3 (emission against `naima`, plus the SNR–cloud pion bump) is §5's Fig. 3 here.
- Fig. 2's finder Mach ∝ N^−0.6 matches Niklas's 20 / 13 / 8 % at 48³ / 96³ / 224³. The
  shock-finder thesis plan derives the same p ≈ 0.6. Credit that work.

### 12.2 Claims in the outline's abstract that the code does not support yet (at `ac4e411`)

| Outline claim | State | Change |
|---|---|---|
| "collisional (hadronic and Coulomb) losses"; Eq. 1's Λ_coll; §2.4; §3.7 "calorimetric limit" | **Not implemented.** `PROGRESS.md` records "confirmed not implemented anywhere"; no loss code exists in the module | Implement before the paper. Hadronic losses matter most exactly in the dense cloud that lights up, so a pion map without them is energy-inconsistent there. Or drop §3.7 and Λ_coll |
| "field-aligned diffusion and streaming" | Streaming is implemented, but `DESIGN.md` keeps "physically correct CR streaming" **open** | Fix it, or say "diffusion" only and turn streaming off in every run the paper shows |
| "existing forward models cannot invert them from observations" | Too strong. Pais & Pfrommer 2020 and Winner et al. 2020 constrain field geometry and acceleration by comparing **suites** of AREPO runs with data (§11.2) | *"… constrain them only by comparing discrete suites of forward runs with the data."* |
| "End-to-end gradients are validated … with an explicitly characterised trust horizon" | No horizon has been measured yet. The frozen-`dt` gradient (G0) is untreated | Keep the sentence; make §4.1 state the `dt` treatment and §4.2 report it |
| "two-fluid shock tube to ∼10⁻³" | Supported: mean absolute errors 0.001–0.003 at 400 cells (`DESIGN.md`, item 6) | none |

### 12.3 The science section: where the gradient case has to go

The outline's two cases are both few-parameter:
- Case A recovers η and κ;
- Case B is a Fisher forecast on global parameters, by implication of Fig. 7's "parameter
  constraints".

That is §2's **control row**. A referee can reproduce both with finite differences or a grid of
runs. [`RATIONALE.md` §4](../RATIONALE.md) concedes that a double-precision Fisher forecast at
this size gains nothing from autodiff. As drafted, §5 demonstrates the pipeline but not what a
gradient buys. **Recommended split, which keeps both of the outline's cases:**

- **Case A, the SNR (recovery): keep it low-dimensional, and make it an SNR–cloud system.**
  - Item 15's setup already exists. The target gas makes the hadronic signal.
  - Optimisation through a live rollout is the fragile use of a gradient, and a few parameters
    keep it robust.
  - **Add two cheap things:**
    - the **finite-difference Fisher matrix beside the AD one** (expected to agree in fp64; say
      so, and report fp32);
    - the **dynamics-frozen ablation** (§0 item 7). It turns D-028 into a figure.
- **Case B, the Cygnus-like superbubble (forecast): this is where the parameter count goes up.**
  - A forecast needs only the local Jacobian at the fiducial. That is far more forgiving than an
    optimisation, so the high-dimensional case belongs here.
  - A superbubble has many sources, so the natural field is **per-source** injection
    efficiencies and wind/SN energies, plus κ per region, or η(θ_Bn) in bins: P ~ 10².
  - Fig. 7 then becomes the **prior-preconditioned Gauss–Newton spectrum and the effective number
    of constrained parameters** for Fermi-LAT, CTAO and joint. It is computed matrix-free (§2),
    with the global-parameter ellipses as its top-left corner.
  - **Gate:** the tangent must stay valid over the superbubble window (G2). Overlapping shocks
    there are the reselection regime (§1), so measure AD vs FD on the superbubble itself.
  - If it fails, the high-P forecast moves to the SNR–cloud η(Ω) of §2, and Case B keeps global
    parameters.
- **§5.4 "Value of spectral resolution" (optional).** In grey, energy bins inform only the
  *assumed* spectral parameters (α, E_cut), which do not pass through the dynamics. Either label
  it that way or move it to the Discussion as the motivation for spectral CRs.

### 12.4 What to add to the outline

- **§1, related work:**
  - Mishra-Sharma et al. 2026 (differentiable γ-ray sky over templates);
  - Zandinejad et al. 2026 (field-level CR density without transport);
  - the AREPO suites;
  - the in-house papers (§11.1).
- **§2.7 / §4.1:** the `dt` treatment, i.e. a frozen schedule with its systematic, or the
  derivative through `dt` (G0).
- **§3.3:**
  - the finder's Mach bias (~10 % at 64 cells per radius);
  - the `E_cr` deficit against the two-fluid Sedov reference (0.0675 against 0.140 at 240³).

  Report them as limitations, not footnotes.
- **§4.3:** AD vs FD through live injection on an expanding **and** on a stalled shock: Niklas's
  0.1 % and ~20 %. One sentence of explanation, flagged as a hypothesis.
- **Fig. 5:** replace the scalar ∂(γ-ray output)/∂η with an **adjoint sensitivity map**, i.e.
  which cells of the shock and cloud the observed flux responds to. It is the figure a
  finite-difference code cannot make.
- **Superbubble anchors** for Case B: the Fermi-LAT Cygnus cocoon and the LHAASO Cygnus bubble.
  **Unverified; not yet looked up.** No citation until DOI-checked.
- **Code availability:** a "version-pinned astronomix release" requires the branch merged into
  `leo1200/astronomix`, or a tagged fork commit (§10 item 3).
