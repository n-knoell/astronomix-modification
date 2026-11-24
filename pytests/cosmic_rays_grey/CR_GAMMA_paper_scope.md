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

**Update 2026-11-17: §13 (incl. §13.7 gate measurements: δ(x) GN spectrum dropped) supersedes the recommendation of §0 items 3–5, §2's "lead case" and
§12.3's Case B.** §13 re-checks the code facts against the branch at `e15df58`, argues why the
field-level η(Ω) case and the superbubble forecast are weaker than stated here, and gives the
recommended Paper 1 (methods + gradient verification + a low-dimensional SNR–cloud recovery with
an adjoint sensitivity map; the δ(x) Gauss–Newton spectrum as a stretch goal behind a cost gate).
§§1–12 are kept unchanged as the record of the 2026-10-29 scope.

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

---

## 13. Revised recommendation (2026-11-17)

**Basis.** The code facts below were re-read on `feature/cosmic-rays-grey` at `e15df58`
(2026-11-16) plus the uncommitted γ-ray proxy work (`GAMMA_RAY_PLAN.md` steps 1–4). Nothing in
§§1–12 was deleted. Where this section disagrees with them, this section holds.

### 13.1 Code facts that changed since `ac4e411`

| §1 / §12 statement | State at `e15df58` |
|---|---|
| `E_cr/E_SN` 0.0675 vs 0.140 (240³) | After DB7/DB8: item 7 gives 0.52 / 0.67 / 0.74 of the self-similar 0.1398 (`PROGRESS_PHASEB.md`). Still a deficit; much smaller |
| Mach bias ≈ 10 % at 64 cells per radius | Resolution study Q1: Mach error ~10 % / ~5 % at ~20 / ~60 cells per radius (`resolution_study/RESULTS.md`) |
| D2 live-DSA AD vs FD 0.10 % | Re-measured 6.9 % (2026-09-26, adaptive sampling) → 5.1e-3 (2026-10-10) → **2.1e-3 (2026-10-20)**. **Not re-measured** after DB8 (CR-aware velocity Mach), the continuous normal surface and `mach_sampling_at_shock`, which are all DSA defaults now |
| Pion emissivity normalisation | Fixed: `pion_decay_emissivity_per_ecr`, momentum power law, default α = 2.3; band maps and SEDs from stored states; α budget passes (> 99.99 % of injected energy at α < 2.05 in the SNR runs) |
| Obliquity η(θ_Bn) | Still **not implemented** (`GAMMA_RAY_PLAN.md` "Optional") |
| Hadronic / Coulomb losses | Still **not implemented** (no loss term in the module) |
| Instrument model | Still **none** |
| Superbubble (Case B) setup | **None** exists |
| `stop_gradient` on `dt` | Still at `time_integration.py:607–625` |
| Hardware | 10 × RTX 2080 Ti (11 GB) locally; H200s on compgpu12 (max 2 GPUs). The scope's "A100/H100 by extrapolation" can be measured now |

### 13.2 Where the science case of §2 / §12.3 is weaker than stated

1. **η(Ω) as a free field does not need the adjoint.** At η ≈ 0.1 the CR back-reaction on the
   shock is a few per cent, so the map is close to linear in the patch efficiencies:
   map ≈ Σᵢ ηᵢ × (response of patch i). That frozen Jacobian can be built in **one forward run** by
   tagging the CRs injected in each patch as a passive scalar. A non-differentiable code (AREPO
   included) can do the same. The "only a differentiable code has `Jᵀ`" argument (§2, §7) fails
   for η(Ω). The "information through the dynamics" ablation will most likely show a small
   effect. The informed rank of η(Ω) is set mainly by PSF and projection geometry, which can be
   estimated without a simulation. Physically, the question is η(θ_Bn) with 2–3 parameters
   (θ_crit, width), not a free field on the sphere.
2. **Grey + one global α makes band maps scaled copies of each other** (`GAMMA_RAY_PLAN.md`
   step 2). A CTAO TeV map then has no morphology the GeV map lacks. A "Fermi-LAT vs CTAO informed
   rank" comparison is only a PSF / photon-count comparison, and TeV morphology (set by
   energy-dependent escape) is exactly what grey cannot predict. Do not make this a headline.
3. **The cloud density field δ(x) is where autodiff genuinely wins.** δ enters both directly
   (map ∝ n_H e_cr) and through the dynamics (when and how strongly each parcel is shocked), with
   P ~ 10³–10⁵. The adjoint is needed, and the frozen-dynamics ablation separates the two paths
   cleanly. The risk is the gradient's validity window through the shock–cloud encounter
   (KH/RT, §6 G2), which has to be measured first.
4. **Case B (Cygnus-like superbubble) is not Paper 1.** Nothing exists. Termination-shock injection
   still changes by 20–30 % between 96 and 192 cells (resolution study, 2026-11-17 reruns), and
   overlapping shocks are the surface-reselection regime of item 16.
5. **Losses matter for an IC 443-like target.** t_pp ≈ 7e7 yr / n gives ~7e4 yr at n = 10³ cm⁻³,
   comparable to IC 443's age. The `GAMMA_RAY_PLAN.md` estimate ("irrelevant") holds for the
   2 kyr, n ≤ 100 demo only. If the outline keeps "calorimetric limit" and Λ_coll, implement them;
   otherwise drop both.
6. **Streaming:** drop it from the claims ("diffusion" only, streaming off in every run shown).
7. **G0 is a measurement, not a code change.** The `stop_gradient` on `dt` already returns the
   gradient for the realised step schedule; D2's 2e-3 agreement with FD shows that works on that
   setup. Report the schedule as a systematic; differentiating through `dt` is not worth it.

### 13.3 Recommended Paper 1

**Question.** *How well do γ-ray maps of an SNR–molecular-cloud system constrain CR injection and
transport, which cells of the shock and cloud does the observed flux respond to, and how much of
that information arrives through the dynamics?*

1. **Method + verification** (outline §§2–3). Exists; regenerate the figures at a frozen paper
   commit. Limitations stated in the text: Mach bias, `E_cr` deficit, cross-field leak, no
   streaming.
2. **Gradient verification** (outline §4):
   - G0 measured on the grey module (step schedule as a systematic);
   - AD vs FD through live DSA on an expanding **and** a stalled shock (D2 and item 16 setups),
     re-measured with the current DSA defaults;
   - the validity window (tangent growth vs SNR age) on the 3D SNR–cloud fiducial, two resolutions.
3. **Low-dimensional recovery on a magnetised SNR–cloud system** (outline Case A): η, κ,
   n_cloud and the obliquity parameters (θ_crit, width), P ≈ 5–6:
   - MAP + Laplace posterior from a mock map with a simple instrument model;
   - the **AD Fisher matrix beside the FD one** (fp64; fp32 reported);
   - the **frozen-dynamics ablation** (Fisher with the hydro response frozen vs live).
4. **One adjoint sensitivity map**: ∂(flux in a region)/∂(initial density, injection) per cell.
   One reverse pass; the figure a finite-difference code cannot make. This replaces the outline's
   scalar ∂(γ-ray output)/∂η (Fig. 5).
5. **~~Stretch goal: the δ(x) Gauss–Newton spectrum~~ — dropped 2026-11-17, fails all three gates (§13.7).** (effective number of constrained density
   modes, live vs frozen dynamics). Only if the cost gate (§13.5) and the validity window pass;
   otherwise it moves to the Discussion as the motivation for Paper 2.

**Dropped from Paper 1:** η(Ω) as a free field (point 1); the Fermi-vs-CTAO rank headline
(point 2); Case B superbubble (point 4); CWB (Paper 2, §8, unchanged); streaming (point 6).

### 13.4 New code needed

| Piece | Size | Needed for |
|---|---|---|
| Hadronic + Coulomb losses (sink + 1/6 gas heating, Pfrommer et al. 2017a; coefficients to verify) | ~1 day | item 3, if "calorimetric" stays |
| Obliquity η(θ_Bn) (`GAMMA_RAY_PLAN.md` "Optional"), opt-in, oblique planar test | ~1–2 days | item 3 |
| Magnetised SNR–cloud fiducial (item 15 + uniform B, MHD) | ~1 day + runs | items 3–5 |
| Instrument model: parametrised PSF + exposure + Poisson likelihood in JAX | ~2–3 days | items 3, 5 |
| Injection efficiency as a per-cell multiplier field (for the sensitivity map) | ~0.5 day | item 4 |

Roughly a week of code, before runs and gradient checks.

### 13.5 First step: cost of one checkpointed reverse pass

Everything in items 4–5 depends on what one reverse pass of the 3D SNR–cloud run costs. The
δ(x) Gauss–Newton spectrum needs ~100–200 tangent + adjoint solves; if one reverse pass at 128³
costs well over an hour on an H200, item 5 is not affordable and Paper 1 is items 1–4.
Measurement on the local 2080 Ti: see §13.6.

### 13.6 Measured: one checkpointed reverse pass on the item-15 SNR–cloud setup (2026-11-17)

**Setup.** Item 15 unchanged (E_SN = 1e51 erg, n = 1, T = 1e4 K, 100× cloud + two 10× clumps,
2 kyr, constant η = 0.1, current DSA defaults at `e15df58`), **float32**, one RTX 2080 Ti (11 GB).
Loss L = 1–10 GeV pion luminosity (α = 2.3) from inside a 1.5 pc disc around the cloud's projected
centre. One `jax.value_and_grad` w.r.t. (η, the full initial density field ρ₀), `BACKWARDS` mode
(equinox checkpointed while loop). Script, logs and gradient fields:
`/export/scratch/nknoell/paper_adjoint/` (`reverse_pass_cost.py`; nothing in the repo was changed).
**Nested remat** (`REMAT=1/2`) is a scratch monkeypatch: `jax.checkpoint` around the
injection stage and each RK stage interior (level 1), plus the per-axis reconstruction /
Riemann / CR-work pieces (level 2).

| N | steps | forward | reverse pass | reverse / forward | temp memory | note |
|---|---|---|---|---|---|---|
| 64 | 464 | 7.3 s | 120 s | 16× | 4.81 GiB | no remat, 20 checkpoints |
| 64 | 464 | 7.3 s | 91 s | 12.5× | 1.70 GiB | remat 1, 20 checkpoints; gradient = no-remat to 0.4 % (max) / 0.6 % (L2), fp32 |
| 112 | 1331 | 109 s | **1459 s (24 min)** | 13.4× | 7.89 GiB | remat 2, 8 checkpoints |
| 128 | 1686 | 205 s | — | — | 38.4 / 39.7 GiB | no remat, 5 / 20 checkpoints: **does not fit 11 GB** |
| 128 | 1686 | 205 s | — | — | 12.5 GiB (remat 1, 20 ckpt), 12.0 GiB (remat 2, 8 ckpt) | **OOM on 2080 Ti** (~10.4 GiB usable) |

**Reading.**
- Checkpoint storage is small (~1 GiB at 128³ for 20 checkpoints). The memory is the
  backward pass of a single time step. Nested remat cuts it ~3× and is *faster*, not slower,
  at 64³. Splitting the per-axis flux pieces (level 2) gained little, so the remaining peak is
  probably in the DSA injection stage (shock finder + walk + spread). Not yet confirmed: that
  needs an XLA buffer-assignment dump.
- **128³ on a 2080 Ti: not without further remat.** On an H200 it fits even without remat.
  Extrapolated, not measured: ~45 min per reverse pass at 128³ on a 2080 Ti-class GPU; on an
  H200 (~8× the memory bandwidth, and room for more checkpoints) plausibly 5–10 min.
- **Gauss–Newton cost (item 5), extrapolated:** 100–200 tangent + adjoint solves at 128³ ≈
  15–30 GPU-hours on one H200. That is feasible but not cheap. The tangent (forward-mode) cost
  through the checkpointed loop has **not** been measured.
- **Near-linearity in η, measured:** dL/dη · η / L = 0.97 (64³) and 0.95 (112³), i.e. the map is
  linear in the injection efficiency to 3–5 %. This supports §13.2 point 1.
- **Resolution:** L(112³) / L(64³) = 4.8. The cloud's γ-ray flux is far from converged between
  these two resolutions. Before any information-content number is quoted, the fiducial needs a
  resolution check of the observable itself.
- **Adjoint sensitivity map (first look):**
  `pytests/cosmic_rays_grey/pics/paper_adjoint/sensitivity_drho0_N112.png`. ∂L/∂ρ₀ is non-zero
  throughout the volume the blast wave has swept up. It is positive in and around the cloud, with
  alternating-sign shells where density changes the shock's arrival. This is a single fp32 pass,
  **not yet FD-checked**.

**Consequences for §13.3.** Items 1–4 are unaffected; the 112³ pass already gives item 4's figure
type. Item 5 needs the H200s (or remat of the injection stage plus ≤ 112³ on the 2080 Tis) and
a measured tangent-solve cost before it is committed to. *(Superseded in part by §13.7: the
density sensitivity of the CR map fails the FD check.)*

### 13.7 Measured: the three gates for the δ(x) Gauss–Newton spectrum (2026-11-17)

**Setup.** As §13.6, but with the cloud edge smoothed over a fixed **0.3 pc** instead of 2 cells, so
the fiducial does not change with N. Parameter field δ = ln ρ_init − ln ρ_fid, nonzero in the cloud plus
a 50 % margin. Test directions are Gaussian random fields with smoothing scale ℓ = 0.5 pc, normalised
to unit standard deviation inside the cloud. Observable: the full projected 1–10 GeV map, and L
(its sum over the 1.5 pc disc). Metrics are given pixel by pixel and after a Gaussian PSF of 0.44 pc
(≈ 1′ at 1.5 kpc) and 2 pc. fp32 throughout; 2080 Tis. Scripts: `gates.py`,
`diag_noise.py`, `diag_split.py` and `plot_gates.py` in `/export/scratch/nknoell/paper_adjoint/`;
data in `data/gates/`. Figure: `pics/paper_adjoint/gates_gauss_newton.png`.

**Gate 1 — tangent solve: cost passes; agreement with the simulator fails.**

*Cost and consistency:*

| N | JVP / forward | JVP memory | VJP / forward | dot test ⟨w, Jv⟩ vs ⟨Jᵀw, v⟩ |
|---|---|---|---|---|
| 64 | 2.20 | 0.44 GiB | 12.5 (remat 1) | 3.4e-5, 5.3e-6 |
| 112 | 2.09 | 2.31 GiB | 13.4 (remat 2, 8 ckpt) | 1.1e-5, 3.3e-5 |
| 128 | 2.18 | 3.63 GiB | — | — |
| 160 | 2.14 | 6.55 GiB | — | — |

- Forward mode works through the whole grey CR + DSA path, using the `FORWARDS` while loop (equinox's
  checkpointed loop has no forward mode).
- A JVP costs ~2.1 forward runs and fits a 2080 Ti up to at least 160³. A GN product (JVP + VJP) is
  therefore ~15 forward runs, and the VJP dominates.
- JVP and VJP are the same linear operator to 1e-5.
- The η tangent reproduces the reverse-mode value (dL/dη·η/L = 0.975 / 0.948 / 0.946 / 0.947).

*JVP vs central FD of the actual simulator, along a GRF direction (ε = 1e-2; stable for
ε = 3e-2 … 3e-3):*

| N | image (pixels) | image (2 pc PSF) | L | ‖Ju‖/‖F₀‖ |
|---|---|---|---|---|
| 64 | 1.00 | 1.06 | 1.87 | 0.020 |
| 112 | 0.52 | 0.33 | 0.33 | 0.93 |
| 128 | 0.80 | 0.74 | 0.75 | 1.02 |
| 160 | 0.29 | 0.19 | 0.11 | 1.00 |

*Diagnosis at 64³:*

- **Not GPU noise.** Repeat runs differ by 1e-5 (relative).
- **The difference is local to the cloud.** For ε ≥ 1e-3, 94–98 % of |F(εu) − F₀| lies inside the cloud disc.
- **It is a jump, not a smooth response.** |ΔF|/ε is 4–7× the AD tangent at every ε from 1e-3 to 1e-1.
- **The hydro is differentiable; the CR part is not.** Same runs, same direction:

  | Projected observable | AD vs FD |
  |---|---|
  | ρ | 0.1 % |
  | ρ² | 0.1 % |
  | p_gas | 3.6 % |
  | e_cr | 62 % |
  | ρ·e_cr (the pion map) | 100 % |

- **No single CR switch removes it.** Pion-map error with each switch group off:

  | Variant | Error |
  |---|---|
  | baseline | 1.00 |
  | adaptive sampling off | 0.87 |
  | continuous-normal surface / sampling-at-shock off | 0.88 |
  | hybrid entropy off | 1.00 |
  | flux cap off | 1.00 |
  | guard off | 1.00 |
  | pressure-based Mach | 0.91 |
  | KR13 η(M), continuous at M_min | 0.99 |
  | all of the above off | 0.87–0.92 |

**Reading.** The remaining non-differentiability is the shock finder itself: its zone masks and the
discrete choice of shock/deposit cells along each normal. AD treats these as constants. In the
simulator, a density perturbation flips them, moving injection between cells across the cloud's
100× density edge. The jump term is O(1) of the response at 64–128³. It falls to ~0.2 (PSF) /
0.1 (L) at 160³, but not monotonically (128³ is worse than 112³), so it cannot be extrapolated to
"fine at production resolution". **The η derivative is unaffected** (η multiplies the injection after
all discrete choices), which is why D2 and §13.6 passed.

**Gate 2 — validity window: fails.** The linear prediction σJu is compared with the actual ΔF for
+σ perturbations, σ_lnρ = 0.1 … 1.5 (two GRF seeds); error measured on the 2 pc PSF image:

| N | Seed | σ = 0.1 | 0.3 | 0.6 | 1.0 | 1.5 |
|---|---|---|---|---|---|---|
| 64 | 0 and 1 | ≥ 0.8 | ≥ 0.8 | ≥ 0.8 | ≥ 0.8 | ≥ 0.8 |
| 112 | 0 | 0.39 | 0.20 | 0.17 | 0.47 | 1.22 |
| 112 | 1 | 1.8 | 2.2 | 4.4 | 3.0 | 3.5 |

- Linearising in ρ instead of ln ρ does not help.
- At 112³ the response is large (σ = 0.1 already changes L by −5 % / −14 %) and in these runs negative
  for every perturbation. The smooth cloud sits near a maximum of L, so the first-order term alone
  misdescribes it.
- There is no σ range in which the linearisation holds for both seeds. **With a molecular-cloud prior
  (σ_lnρ ~ 1) the Gauss–Newton spectrum of δ(x) would describe the AD operator, not the simulator.**

**Gate 3 — resolution: fails.** L in the 1.5 pc disc [1e33 erg/s]:

| N (cloud radius in cells) | 48 (2.4) | 64 (3.2) | 80 (4) | 96 (4.8) | 112 (5.6) | 128 (6.4) | 160 (8) | 192 (9.6) |
|---|---|---|---|---|---|---|---|---|
| edge 0.3 pc | 0.139 | 0.161 | 0.270 | 0.582 | 0.751 | 0.885 | 1.226 | 1.638 |
| edge 2 cells (item 15) | 0.112 | 0.186 | 0.658 | 0.799 | 0.889 | 0.918 | 1.037 | 1.164 |

- With the fixed-width edge, L still grows ~linearly in N at 192³; the 2-cell series flattens only
  because its cloud edge sharpens with N.
- The total map grows 4.3 → 7.1 over the same range.
- The cloud-response derivatives also drift with N. From FD at ε = 0.1, d ln L/dε is:

  | N | Uniform ln ρ shift in the cloud | Dipole (front/back) |
  |---|---|---|
  | 48–64 | −0.02 … 0.03 | −0.01 … 0.03 |
  | 112 | 0.13 | 0.12 |
  | 192 | 0.40 | 0.06 |

- This is the known non-convergence of inviscid shock–cloud problems at ≲ 100 cells per cloud radius
  (Klein, McKee & Colella 1994). At ≤ 10 cells per radius no information-content number about the
  cloud's density structure is resolution-independent.

**Verdict on item 5.** Do not run the δ(x) Gauss–Newton spectrum. It fails gates 1–3 independently:
1. The derivative w.r.t. density misses the shock finder's discrete jumps.
2. The response is non-linear at prior-sized amplitudes.
3. The fiducial is unconverged.

Fixing (1) would need a smoothed shock finder (soft masks, soft cell selection), which is a research
project of its own. Fixing (3) needs ≳ 50–100 cells per cloud radius, i.e. a cloud-zoom setup, not a
20 pc box. Compute cost was *not* the blocker: tangent solves are cheap (2×).

**Consequences for the rest of §13.3.**
- **Item 4 (sensitivity map w.r.t. ρ₀):** the §13.6 figure is the AD operator and is *not* what the
  simulator does (FD disagrees at O(1)). Use an **η(x) (per-cell injection) sensitivity map** instead:
  η enters after the discrete choices, its tangent is validated (0.95–0.97 linear, AD = reverse), and
  it needs the 0.5-day per-cell η field of §13.4.
- **Item 3 (low-dim recovery):** n_cloud is a density parameter, so its AD derivative is suspect in the
  same way. Validate AD vs FD for n_cloud (and θ_crit, if the obliquity switch is discrete) before
  using the AD Fisher; otherwise use FD for those entries. η and κ are not affected by this finding;
  κ is untested.
- **Item 2 (gradient verification)** gains a concrete, publishable negative result: AD through a
  shock-finder-based DSA is exact for injection parameters and biased at O(1) for parameters that
  move the shocks. That belongs in the paper as the limitation, with the split table above.
