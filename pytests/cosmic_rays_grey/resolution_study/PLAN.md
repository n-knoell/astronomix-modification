# Resolution study: shock finding and DSA in the Sedov blast and the CWB (plan, 2026-11-04)

Status: plan only, nothing run yet. Work through the steps in order and tick
the boxes. Each step lists what it does, which existing code it reuses, what
it writes, and when it counts as done.

- Folder: `pytests/cosmic_rays_grey/resolution_study/`. This file, the scripts
  and `RESULTS.md` (written as steps finish) live here; figures go to `pics/`.
- Raw data: `/export/scratch/nknoell/resolution_study/`. All raw-output env
  vars (`CR_PHASEB_RAW`, `CR_PHASEB_DB7_RAW`, `CR_PHASEB_DB4_RAW`) point
  there, so no Phase-B scratch file gets overwritten.
- Code state: the current branch with the 2026-11-04 DSA default:
  - continuous-normal surface with our zones, `column_area_weights`;
  - adaptive sampling with extend;
  - velocity Mach, CR-aware (DB8).

  The commit hash is recorded at the top of `RESULTS.md`. The code is not
  changed during the study; a bug found along the way is fixed, logged, and the
  affected steps are rerun.
- GPUs: only on GPU (RTX 2080 Ti, 11 GB). Pin by UUID, check occupancy right
  before the launch, verify PID→UUID afterwards. Plotting and numpy analysis
  run on CPU.

---

## 1. Questions the study answers

- **Q1 Hydro.** At which resolution do the shock positions and speeds and the
  post-shock states that the finder reads converge, and at what order?
  - Sedov: R(t), Ṙ(t), shell compression.
  - CWB: contact and termination-shock stand-off, pre-shock wind profile.
- **Q2 Finder.** How do detection (surface completeness, false positives),
  shock area and Mach number (bias, per-cell scatter, orientation artefacts)
  depend on resolution? Measure resolution in the cells across the length
  that matters (`R/dx`, shock-to-source distance / `dx`), not just `N`.
- **Q3 DSA.** How do the injected energy flux D, the injection rate and the
  CR energy E_cr(t) converge? How much of the error is finder-Mach error
  amplified by the flux's Mach dependence? How much is history: the
  unresolved early blast, and the deposit surplus in the smeared shell
  (DB9)?
- **Q4 Mach range.** Does the answer change across the Mach range DSA cares
  about? In the M ≈ 2–15 range the efficiency ζ(M) (KR13) and δ(M) are
  steep; above M ≈ 30 both saturate and Mach errors barely matter.
- **Q5 Practical rule.** What is the minimum resolution, in cells per shock
  radius or per stand-off distance, for a given accuracy of M, D and E_cr?
  For example, 5% / 10% / 10%. This rule is the deliverable that goes into
  the report and into the setup of production runs (M7, SNR, CWB).

---

## 2. Design principles (why the study is set up this way)

1. **Expect first order at best.** Shock-capturing schemes converge at most at
   first order in L1 once a shock is present; the shock's local error
   contaminates the smooth flow (Engquist & Sjögreen 1998). Any jump-based
   finder reads a numerical shock that is 2–3 cells wide whatever `N` is.
   - So expect M, D and E_cr errors ∝ (dx / L)^p with p ≈ 1, where L is the
     length the jump has to be read against:
     - Sedov: the post-shock gradient scale ~0.1 R, and the curvature R;
     - CWB: shock-to-injection-sphere distance and shocked-layer width.
   - Measure the observed order p from three or more grids instead of assuming
     it. Report a grid convergence index (GCI; Roache) only where p is stable.
2. **Measure resolution in the cells across the physical scale.** The Sedov
   blast is self-similar, so one run sweeps `R/dx` in time.
   - If a metric is controlled by resolution alone, plotting it against
     `R/dx` collapses all `N` onto one curve.
   - Where the curves do not collapse, history matters (the unresolved
     start, accumulated E_cr) and the metric must be read at fixed `t`.
   - Both views are produced for every metric.
3. **Use the exact reference where one exists; otherwise make one.**
   - Sedov, p₀ → 0: the self-similar Sedov and two-fluid (constant-ζ)
     solutions (`astronomix/test_setups/reference_solutions/sedov_two_fluid.py`,
     B7.1). With CRs, the self-similarity is approximately kept (Pfrommer et
     al. 2017).
   - True Mach number, any setup with an unshocked upstream at rest:
     M_true(t) = Ṙ(t) / c₀ from the run's own shock trajectory. This holds
     even with counter-pressure, where no analytic solution exists.
   - CWB: Chevalier & Clegg (1985) wind upstream of the termination shocks.
     `expected_cwb_mach.py` evaluates it at the shock distance measured in
     the run. Ram-pressure balance gives the stand-off
     (`stagnation_point_fractions`).
   - Where no reference exists (CWB DSA): Richardson extrapolation from three
     or more grids with a constant refinement ratio.
4. **Follow the error chain, not only the end result.** The same metric of
   the CR energy can be right for the wrong reasons; B7.2 found an injection
   deficit offset by a deposit surplus. Each step decomposes:
   hydro → detection / area → Mach → flux → injection → deposit and CR
   transport. This reuses B7.3's split `D = A_inj · ⟨f⟩` and
   `D / D_meas = (A_inj / 4πR²)(⟨f⟩ / f_meas)`, and the adiabatic-kernel
   prediction of E_cr from the run's own injection rate
   (`cr_sedov_injection_rate.py`).
5. **Separate the finder's resolution from the hydro's.** On top of the
   native runs, block-average (restrict) the finest state onto the coarse
   grids (2×, 4× …; conservative variables averaged, then converted to
   primitives) and run the finder there.
   - Native coarse vs restricted fine = the hydro's share of the error, at
     the same finder resolution.
   - Restricted fine vs exact = the finder's own share. It is cheap: finder
     only, no extra runs.
6. **Cover the Mach range DSA cares about.**
   - The standard Sedov setup (`p₀ = 1e-4`) has M ≈ 150–1400 over the run,
     where δ(M) and KR13 ζ(M) have saturated. A Mach error there enters D
     only as ~(1 + 3ΔM/M), so Mach errors are hardly tested.
   - The study therefore adds a counter-pressure Sedov blast whose Mach
     number falls from ~30 to ~2.5 within the box (Section 3.1, Step S4).
     Its errors are compared with the predicted amplification
     `d ln D / d ln M = 3 + d ln δ/d ln M + d ln ζ/d ln M`, which is large
     for KR13 at M = 2–5.
7. **Fix every numerical parameter in cells, and vary it only in the
   sensitivity step (Step X1).**
   - Kept fixed: CFL, limiter, `dsa_area_normal_smoothing` = 2 cells,
     `dsa_shock_sampling_max_steps` = 15, `mach_min`, smoothing of the
     initial blast (`SMOOTH_CELLS`), wind injection radius = `N/32` cells.
   - With the injection radius fixed at `N/32` cells, it is fixed in physical
     units too, so the CWB stays one physical problem.
   - Never use a fixed `num_injection_cells`: that changes the physical
     problem, because the wind Mach at the shock grows as N^(2/3) (memory /
     FIXES_TODO 2026-10-14).
8. **The CWB converges in statistics, not point by point.**
   - Colliding-wind shells develop thin-shell and Kelvin–Helmholtz
     instabilities whose growth depends on resolution (Lamberts et al. 2011;
     Kissmann et al. 2016).
   - Pointwise L1 differences between grids therefore need not converge.
     Use integrated or averaged metrics instead:
     - per shock (wind-1 termination shock, wind-2 termination shock, outer
       forward shocks);
     - over fixed physical regions (apex cone, wings);
     - time-averaged over a quasi-steady window.
9. **The physics stays the same across `N`.** Hold fixed: same t_end, ζ
   model, diffusion coefficient, `v_red`, floors and options.
   - Some physical scales are not resolved at any `N` here. The CWB's CR
     diffusion length κ / v_sh ≈ 1.2e-3 box is below one cell even at 256³.
   - Converged results then converge to the "κ below grid scale" limit; say
     so in the write-up instead of hiding it.

---

## 3. Setups and resolution ladders

### 3.1 Sedov blast with CRs

| Setup | Source | Purpose | Mach range |
|---|---|---|---|
| **S-PE**: point explosion | `cr_sedov_taylor._run_sedov` (E = ρ₀ = 1, p₀ = 1e-4, t_end 0.07, R_EXPLOSION 0.05, ζ = 0.1) | production-like start, includes the unresolved early blast | ~150–1400 |
| **S-SS**: self-similar start | `cr_sedov_self_similar._run` (R₀ = 0.15, two-fluid initial state) | the scheme's own error without start-up memory; exact reference at all t | ~170–760 |
| **S-CP**: counter-pressure blast (new) | as S-SS / S-PE, but `p₀` raised (below) | the DSA-relevant M ≈ 2.5–30, KR13 and constant ζ | ~2.5–30 |

S-CP sizing:
- In Sedov, with E = ρ₀ = 1 and γ = 5/3, `Ṙ ≈ 0.567 R^(-3/2)`.
- For M(R = 0.42) ≈ 2.5 this needs c₀ ≈ 0.83, i.e. **p₀ ≈ 0.4**.
- Then M(R = 0.08) ≈ 30, and the ambient energy inside R = 0.42 is ~20% of
  E, so the blast leaves the Sedov solution. That is intended: M_true comes
  from Ṙ.
- Step S0 refines `p₀` and t_end with a cheap 64³ run.
- Start: the self-similar state at R₀ = 0.08, where the counter-pressure is
  still negligible (p₀ / p_post ≈ 1e-3). It needs ≥ 4 cells at the coarsest
  grid, so N ≥ 48.

Resolution ladder: **N = 32, 48, 64, 96, 128, 192, 256**.
- It holds two refinement-ratio-2 chains (32-64-128-256, 48-96-192) for
  Richardson, and spans R/dx ≈ 2–110 over the run.
- 256³ with CRs fits on 11 GB; the CWB 256 runs with CRs have run before.
  Step S0 checks memory.
- Cost estimate from the regression timings:
  - `cr_sedov_taylor` 160³ ≈ 10 min for control + DSA;
  - 256³ ≈ 6.5× that ≈ 1.1 h per run pair;
  - the whole Sedov ladder, all setups: ~15–20 GPU-h, spread over the idle
    GPUs.

Runs per N and setup:
- control (ζ = 0, CR module on);
- constant ζ = 0.1;
- for S-CP also KR13 (`DSA_EFFICIENCY_KANG_RYU_2013`).

Snapshots: ≥ 30 per run, evenly spaced in log t, so that Ṙ comes from a
smooth fit and the `R/dx` collapse is well sampled. Save the full state only
at ~6 snapshots; the others keep reduced diagnostics (Section 4).

### 3.2 CWB with CRs

Setup: `pytests/shock_finder3D/_cwb_setup.run_cwb` defaults, as in
`cwb_shock_finder.py` and the 2026-11-04 comparison (`build_cwb` returns the same
setup unrun, for segmented integration).
- Two O stars (50 / 40 M☉) with unequal winds: Ṁ 6.5e-7 / 6.5e-8 M☉/yr, v 2200 / 1850 km/s,
  momentum ratio η ≈ 0.084.
- Separation 0.4 box, injection radius `N/32` cells (fixed in physical units).
- ζ = 0.1, κ = 1 code unit, `v_red` = 1000, unsplit HLLC + RK2, no dual energy,
  no wind temperature floor; the pre-shock Mach number on the apex is therefore
  ~10 (Chevalier & Clegg at r_s / R_inj). With dual energy + 1e4 K floor it would
  be ~150 (FIXES_TODO round 23); that variant is a different upstream, not part of
  the ladder (decision 6, Section 8).
- Stationary stars (no orbit), t_end = 1.1 × wind crossing time.

Resolution ladder: **N = 64, 96, 128, 192, 256**.
- `N` must be a multiple of 32, which makes the injection radius 2, 3, 4, 6,
  8 cells.
- Chains: 64-128-256 (r = 2) and 96-192.
- At 64³ the injection sphere is 2 cells. It is expected to be outside the
  asymptotic range and is kept only to show where convergence starts.
- The binding constraint is the wind-2 side. Star 2 to the contact is
  ≈ 0.225 × separation ≈ 0.09 box, i.e. 11.5 cells at 128³. That leaves
  only ~5 cells between the injection sphere and the wind-2 termination
  shock.
- Cost: 128³ = 330 s (DSA) + 400 s (control) on one 2080 Ti. 256³ ≈ 16× ≈
  1.5–2 h per run, about 5 GPU-h for the ladder.

Runs per N: control (ζ = 0) and constant ζ = 0.1. KR13 is optional (Step C4),
since the CWB shocks are at M ≈ 5–150.

---

## 4. Metrics

Every metric is recorded per snapshot, per N and per run. Plots use two
views: against `R/dx` (or the stand-off in cells), and at fixed time.

**Hydro**
- Sedov: R(t) (front and peak fits, B7.3 `_shock_radius`), Ṙ(t) from a
  smoothing fit, ξ, α (R ∝ t^α), shell peak ρ_max / 4ρ₀, shell width in
  cells, L1 of ρ, P_tot and P_cr against the reference (S-SS).
- CWB, along the binary axis and averaged over a cone of a few degrees:
  - positions of the wind-1 TS, contact and wind-2 TS, against the
    ram-pressure stand-off;
  - upstream wind ρ, v, T against Chevalier & Clegg;
  - effective Ṁ through spheres around each star, against nominal. The EI
    wind is known to inject only ~36% of Ṁ; this needs to be checked as
    independent of N.

**Finder**
- Surface completeness: the fraction of solid angle (HEALPix-like μ–φ bins)
  with ≥ 1 surface cell within ±2 dx of the true shell.
- False positives: surface cells farther than 3 dx from any true shock.
- Area: `A_inj / 4πR²` (weighted) and `A_cells / 4πR²`. Per-|n|_inf-bin area
  and energy per area, the grid-artefact metric (`sedov_mach_optin.py`
  tooling).
- Mach: M_finder / M_true for injecting cells, area-weighted median, the
  16–84% half-width, and the orientation dependence (axis vs body diagonal).
  Also the walk length (median / max steps).
- CWB: Mach on the axis for each termination shock, against Chevalier &
  Clegg at the measured shock distance. Mach distribution per shock.

**DSA**
- Instantaneous flux:
  `D_true = 4πR² ζ(M_true) δ(M_true) ρ₀ Ṙ³ / 2` (code convention `gas_rh`),
  split into the area and flux factors.
- Realized injection rate vs nominal (cap and positivity losses).
- E_cr(t) / E_cr,ref(t); for S-PE also the fraction injected before
  `R/dx = 8`.
- Adiabatic-kernel prediction vs the run (deposit surplus, DB9).
- X_cr in the shell window λ ∈ [0.7, 0.9], peak P_cr ratio, shock slowdown
  ξ_dsa / ξ_ctrl.
- CWB: E_cr(t), total and per shock region (split at the contact as in
  `dsa_cwb.py`, plus the outer forward shocks). Time-averaged injection rate
  per shock over the quasi-steady window.
- CWB semi-analytic expectation for the termination-shock injection:
  `L_exp = ζ ∫ δ(M_n) ½ ρ v_n³ dA` over the shock surface measured at the
  finest N, with the Chevalier & Clegg upstream and v_n the normal component
  of the wind velocity. It is the best available target, with the shape
  error stated.

**Convergence statistics**
- For every scalar metric Q: Q(N), the error against the reference where one
  exists, and the observed order p from each r = 2 chain.
- Richardson extrapolate Q_∞ and GCI (safety factor 1.25 with ≥ 3 grids),
  only where p lies in [0.5, 2.5] and both chains agree. Otherwise report
  "not in the asymptotic range" and the range spanned.

---

## 5. Work plan

### Step 0: infrastructure (no physics yet)

- [x] **0.1 Folder layout.**
  - `resolution_study/{PLAN.md, RESULTS.md, pics/, common.py}`.
  - Raw data in `/export/scratch/nknoell/resolution_study/{sedov,cwb}/N{n}/`.
  - `common.py`:
    - GPU pin check;
    - raw-data paths;
    - the reduced-diagnostics writer (one `.npz` per run, every Section 4
      metric per snapshot);
    - code-state stamp (git hash + `git diff --stat` hash) written into
      every output;
    - the injected-CR-energy accumulator (realized injection integrated
      between snapshots, total and per region) for `C3_Ecr_per_Lwind_vs_N.png`.
- [x] **0.2 Launcher.** A copy of the `finder_v2_comparison/regress` launcher
  (`launch.sh` refusing busy GPUs, `verify.sh` PID→UUID, `dispatch.sh` over a
  GPU list), logging to `/export/scratch/nknoell/resolution_study/logs/`.
- [x] **0.3 Memory and time probe.**
  - One S-PE DSA run at 256³ for 10 steps, and one CWB 256³ DSA run for 10
    steps.
  - Record peak GPU memory and s/step; extrapolate the wall time.
  - If 256³ does not fit, the ladders end at 224³ (CWB: 224 = 7 × 32) or
    192³; record that in `RESULTS.md`.
- [x] **0.4 Restriction operator.**
  - `restrict(state, factor)`: block-average ρ, momentum, total energy and
    e_cr, then convert to primitives.
  - Unit test: restricting a uniform state and a linear profile is exact, and
    mass and energy are conserved.
- [x] **0.5 1D spherical reference?** (no: injection is Cartesian only)
  - Check whether the grey-CR module, DSA injection and the finder run in 1D
    `SPHERICAL` geometry. The CR module has no `SPHERICAL` code path today,
    so probably not.
  - If yes: high-resolution 1D runs (N = 4096–16384) are the reference for
    S-CP.
  - If no: the S-CP reference is the Richardson-extrapolated 3D result, plus
    M_true from Ṙ, which needs no reference. Do not build 1D spherical CR
    support for this study without asking first.
- [x] **0.6 Analysis skeleton.** `analyse.py` (CPU) reads the reduced
  `.npz` files and makes the standard plots:
  - metric vs `R/dx`, all N overlaid;
  - metric vs N at fixed t, with the fitted slope;
  - observed-order table.

  Test it on the existing B7.3 raw data (48/96/160) before any new runs.

**Done when:** a 48³ S-PE run goes through launch → reduced `.npz` →
`analyse.py` plots end-to-end, and the memory/time table is in `RESULTS.md`.

### Sedov

- [x] **S0 S-CP sizing (64³, 1 run).** (done 2026-11-04: p₀ = 0.6, R₀ = 0.08, t_end = 0.07; RESULTS.md)
  - Choose p₀ and t_end so that M_true falls from ≥ 25 to ≤ 3 before
    R = 0.42, and the blast stays inside the box.
  - Check that the self-similar start at R₀ = 0.08 is still valid
    (p₀ / p_post ≲ 1e-2).
  - Write the chosen values into `RESULTS.md`.
- [x] **S1 Hydro and finder on control runs (all N, S-PE + S-SS + S-CP,
  ζ = 0).**
  - Run the finder with the DSA settings on every snapshot.
  - Answers Q1 and most of Q2, without feedback from injection.
  - Figures:
    - `S1_mach_vs_Rdx.png`: M_finder / M_true vs R/dx, all N; collapse test;
    - `S1_area_vs_Rdx.png`;
    - `S1_orientation.png`: per-|n|_inf bins vs N;
    - `S1_completeness.png`;
    - `S1_hydro.png`: R, Ṙ, ρ_max/4ρ₀, shell width in cells.
- [x] **S2 Restriction test (CPU-cheap, finder on GPU).**
  - Restrict the 256³ (or finest) control snapshots to 128 / 64 / 32 and run
    the finder.
  - Split the Mach and area errors at each coarse N into a hydro share and a
    finder share.
  - Figure: `S2_error_split.png`.
- [x] **S3 Live DSA, constant ζ = 0.1 (S-PE + S-SS, all N).**
  - E_cr(t), D(t) and their split, deposit surplus, X_cr, P_cr profiles,
    slowdown.
  - S-SS has the exact reference at every t; S-PE shows how much the
    unresolved start costs.
  - Report the fraction of the final E_cr injected while R/dx < 8, per N.
  - Figures:
    - `S3_Ecr_vs_t.png`;
    - `S3_D_split_vs_Rdx.png`;
    - `S3_convergence.png`: error vs N, slopes, both chains.
- [x] **S4 Mach sweep with DSA (S-CP, constant ζ and KR13, all N).**
  - M_finder / M_true and D / D_true as functions of M_true and R/dx, as 2D
    maps.
  - Compare the measured D error with the predicted amplification
    `(d ln D / d ln M) · ΔM/M`.
  - Answers Q4.
  - Figures:
    - `S4_mach_error_map.png`;
    - `S4_flux_amplification.png`;
    - `S4_Ecr_KR13_vs_const.png`.
- [x] **S5 Sedov summary.** Fill the Sedov rows of the Q5 table:
  - cells per R needed for |ΔM/M| < 5%, |ΔD/D| < 10% and |ΔE_cr/E_cr| < 10%;
  - for constant ζ and for KR13, at high and low M.

### CWB

- [x] **C0 Preconditions (64³ and 128³ control runs).**
  - Measure the effective Ṁ of both winds against N. The EI wind is known to
    inject ~36% of Ṁ (memory, unfixed). If the fraction depends on N, the
    CWB study measures the wind model, not the finder: stop and ask the user
    whether to fix the injection first.
  - Determine the quasi-steady window: apex positions and injection rate
    within ±2% over the window, e.g. [0.6, 1.0] t_end.
  - `test_cwb_shock_finder_correctness` fails on mirror symmetry because the
    winds are unequal since the setup changed. That is a stale test, not a
    study blocker; note it.
- [x] **C1 Hydro and finder on control runs (N = 64 … 256).**
  - Stand-offs, upstream profiles against Chevalier & Clegg, axis Mach per
    termination shock against Chevalier & Clegg at the measured distance.
  - Mach distribution, area and completeness per shock.
  - Figures:
    - `C1_axis_profiles.png`: all N overlaid, Chevalier & Clegg dashed;
    - `C1_mach_per_shock.png`;
    - `C1_standoff_vs_N.png`.
- [x] **C2 Restriction test (CWB).** As S2, on the finest control state; the
  hydro/finder error split per shock.
- [x] **C3 Live DSA, ζ = 0.1 (N = 64 … 256).**
  - E_cr(t), total and per shock region.
  - Time-averaged injection rate per shock against L_exp; Richardson for the
    totals.
  - CR energy maps in the orbital plane per N, as ratios to the finest.
  - Figures:
    - `C3_Ecr_vs_t.png`;
    - `C3_injection_per_shock_vs_N.png`;
    - `C3_maps.png`;
    - `C3_convergence.png`;
    - `C3_Ecr_per_Lwind_vs_N.png`: injected CR energy per total wind
      luminosity `L_w = (Ṁ₁ v₁² + Ṁ₂ v₂²) / 2` as a function of N.
      - Two panels: (a) the cumulative fraction `E_cr,inj(t_end) / (L_w t_end)`;
        (b) the quasi-steady rate `⟨dE_cr,inj/dt⟩ / L_w`, time-averaged over
        the C0 window, error bar = its scatter.
      - Total and per shock (wind-1 TS, wind-2 TS, outer forward shocks).
      - Normalised once with the nominal and once with the effective `L_w`
        measured in C0 (the EI wind's Ṁ deficit enters here).
      - Horizontal references: the Richardson value and `L_exp / L_w`.
      - "Injected" means the time integral of the injection source term
        (realized, after the cap), not the CR energy left in the box, so
        adiabatic losses and outflow through the boundary do not enter. The
        box `E_cr / (L_w t)` is shown alongside as a dashed line.
      - Needs a per-step (or per-snapshot-interval) accumulator of the
        realized injection, total and per region. Step 0.1's diagnostics
        include it.
- [x] **C4 (optional) KR13 on the CWB** (128 and 256 only): checks whether
  the M ≈ 5–15 parts of the wind-2 shock change the conclusions.
- [x] **C5 CWB summary:** the Q5 rows in cells per stand-off distance and per
  injection radius.

### Cross-cutting

- [x] **X1 Sensitivity to cell-based numerical parameters**, at one middle N
  (96 Sedov S-CP, 128 CWB), changing one at a time:
  - `dsa_shock_sampling_max_steps` 8 / 15 / 25;
  - `dsa_area_normal_smoothing` 1 / 2 / 3;
  - `mach_min` 1.3 / 2;
  - CFL ×0.5;
  - limiter (MINMOD vs the default).

  Decides whether any cell-based parameter should instead scale with N, or
  with a physical length, for convergence.
- [x] **X2 Old surface as a control** (finder-only, on the S1/C1 control
  snapshots): the segment-argmin surface with 1/|n|_inf.
  - Shows whether the 2026-11-04 default changes the convergence rate, or
    only the constant.
  - Cheap: no new runs.
- [x] **X3 Write-up.**
  - `RESULTS.md`: per step the numbers, figures, observed orders and the Q5
    table.
  - Update `PROGRESS_PHASEB.md` with a log entry and new DB items if
    something must be decided, the memory, and, on request, the report
    section (WRITE_REPORT.md conventions).

Suggested order for GPU use:
1. Step 0;
2. S0;
3. the S1/S3 runs together (control + DSA at the same time);
4. C0;
5. C1/C3 together;
6. S4;
7. the cheap analysis steps S2, C2, X2 while big runs are queued;
8. X1;
9. X3.

Total ≈ 25–35 GPU-h, i.e. about 2 days on 3–4 idle 2080 Ti.

---

## 6. Expected outcomes and checks against earlier results

Before trusting new numbers, check them against what is already known:
- **Sedov S-PE E_cr / E_cr,ref:** registry values ~0.52 / 0.66 / 0.72 at
  48 / 96 / 160, now with the continuous-normal surface.
- **Sedov S-SS:** within 5% / 2% / 0.5% of the reference. The injection
  deficit (0.77 / 0.87 / 0.92) is offset by the deposit surplus (B7.2).
- **Velocity Mach / true:** ~0.88 / 0.94 / 0.94 (B7.3). The study should
  show this as a function of R/dx; at 48³ late times, R/dx ≈ 20.
- **CWB:** apex Mach ≈ 10 on the wind-1 side (Chevalier & Clegg at r_s / R_inj).
  E_cr changed by 0.995 between surfaces at 128³.
- **Self-similar collapse:** if M_finder / M_true vs R/dx does not collapse
  across N in S-SS, the Mach error has a time- or history-dependent part.
  Investigate before going on.

---

## 7. Literature this plan builds on

Shock finding and Mach numbers:
- Schaal & Springel 2015: shock finding on a moving mesh. Numerical shocks
  are broadened, which makes accurate Mach numbers hard, and they matter for
  DSA. [arXiv:1407.4117](https://arxiv.org/pdf/1407.4117)
- Schaal et al. 2016: Mach numbers in SPH; particle noise gives inaccurate
  Mach numbers. [arXiv:1604.07401](https://pdf.arxiv.org/pdf/1604.07401)
- Beck, Dolag & Donnert 2016: geometrical on-the-fly shock detection in SPH,
  validated on shock tubes of rising Mach number.
  [arXiv:1511.07444](https://ar5iv.labs.arxiv.org/html/1511.07444)
- Vazza et al. 2011: code comparison of shock statistics from 64³ to 512³.
  Not converged at the best resolution, most of all in grid codes; the
  finder method adds its own resolution dependence.
  [arXiv:1106.2159](https://arxiv.org/pdf/1106.2159)
- Vazza, Brunetti & Gheller 2009: Mach numbers from Rankine–Hugoniot jumps
  along the axes in ENZO. [arXiv:0805.4214](https://web3.arxiv.org/pdf/0805.4214)
- Dubois et al. 2019: RAMSES CR shock finder, M_min ≈ 1.5, pressure jumps,
  probing up to 4 cells away for very strong shocks; accurate shock locations
  and Mach numbers on several tests.
  [arXiv:1907.04300](https://arxiv.org/pdf/1907.04300)

CR acceleration at blast waves:
- Pfrommer et al. 2017: CR physics on a moving mesh. DSA at resolved shocks;
  CR acceleration does not significantly break Sedov self-similarity
  (constant effective γ). [arXiv:1604.07399](https://pdf.arxiv.org/pdf/1604.07399)
- Pais et al. 2018: CR acceleration and SN blast dynamics. Appendix
  convergence: shock radius converged at 50³ apart from the earliest times,
  eccentricity only at 200³; a resolution ladder similar to the one here.
  [arXiv:1805.00128](https://arxiv.org/pdf/1805.00128)
- Böss et al. 2023 (CRESCENDO): on-the-fly spectral CRs with SPH shock
  injection; the dissertation has injection-efficiency shock-tube tests.
  [arXiv:2207.05087](https://arxiv.org/pdf/2207.05087),
  [thesis](https://edoc.ub.uni-muenchen.de/33280/1/Boess_Ludwig_Maximilian.pdf)
- Kang & Ryu 2013 (KR13 ζ(M)); Chevalier 1983 (self-similar CR-modified
  blast). Already implemented and cited in the code
  (`dsa_efficiency_kang_ryu_2013`, `sedov_two_fluid.py`).

Colliding-wind binaries:
- Lamberts, Fromang & Dubus 2011: high-resolution unstable colliding winds;
  instabilities dominate long-term behaviour and are costly to resolve.
  Statistical convergence is the realistic target.
  [arXiv:1109.1434](https://www.arxiv.org/abs/1109.1434)
- Lamberts et al. 2012: RAMSES CWB; the double-shock structure needs high
  resolution for its instabilities.
  [arXiv:1212.4045](https://ar5iv.labs.arxiv.org/html/1212.4045)
- Kissmann et al. 2016: MHD CWB; the resolution needed near the stars
  conflicts with the wind-collision region.
  [arXiv:1609.01130](https://arxiv.org/pdf/1609.01130)
- Reitberger et al. 2014: high-energy particle transport in 3D hydro CWB
  models. [arXiv:1401.1323](https://arxiv.org/pdf/1401.1323)
- Pittard, Romero & Vila 2021: DSA in CWBs. Mach and Alfvénic Mach vary
  strongly along and between the oblique shocks, so efficiency varies
  non-linearly: report per shock and per region, not only totals.
  [arXiv:2104.07399](https://arxiv.org/pdf/2104.07399)
- Pittard 2009: 3D colliding-wind models.
  [PDF](https://astro.swarthmore.edu/~cohen/projects/CEN1/AAS2010/papers/Pittard_MNRAS2009.pdf)
- Chevalier & Clegg 1985 (thermally driven wind; `expected_cwb_mach.py`).

Convergence methodology:
- Engquist & Sjögreen 1998: with shocks, even formally high-order schemes
  are only first order; the shock error contaminates smooth regions.
  Excerpted via [Wang 1998](https://www.math.umd.edu/~tadmor/references/files/Wang_WC_L1_optimaility_SIMA98.pdf).
  Modern schemes drop to first order in the shock-influence region:
  [arXiv:2304.10960](https://arxiv.org/pdf/2304.10960).
- Roache's GCI and observed order from ≥ 3 grids, with p = 1 assumed
  conservatively across discontinuities.
  [arXiv:1705.09062](https://arxiv.org/pdf/1705.09062),
  [Phillips & Roy 2014](https://www.aoe.vt.edu/content/dam/aoe_vt_edu/people/faculty/cjroy/Publications-Articles/re_jfe-phillips-2014.pdf).

---

## 8. Decisions for the user (defaults in bold; the plan proceeds with them unless told otherwise)

1. Counter-pressure Sedov blast (S-CP) for the DSA-relevant Mach range:
   **include**.
2. Largest grid: **256³** if Step 0.3 fits it, otherwise 224 / 192.
3. CWB wind injection (~36% of Ṁ): **measure in C0; fix it first only if it
   depends on N**.
4. KR13 on the CWB (C4): **optional, after C3**.
5. CR diffusion in the CWB stays at the physical κ = 1 (below grid scale):
   **keep** and state the limit. A variant with κ resolved over a few cells
   would test the CR transport, not the finder.
6. CWB upstream: **`run_cwb` defaults (no dual energy, no wind floor; apex M ≈ 10)**,
   consistent with all earlier CWB + DSA results; the dual-energy + 1e4 K variant
   (apex M ≈ 150) only on request.
