# Cosmic Rays in `astronomix`: Phase B Injection Test Plan (Review and Extension)

**Goal.** Strengthen the Phase B injection tests of `astronomix_CR_implementation_plan.md`
(Sec. 4, ladder items 7 and 8). As in the Phase A plan, there are three steps, applied to
each item in turn:

1. **Review.** Go through the existing pytest scripts and the figures they produce, and check
   the results against what the physics and the literature say should happen.
2. **Extend.** Add tests from the literature that check the same implementation from a
   different angle, and write the most promising ones down concretely enough to implement.
3. **Fix.** Where a result disagrees with the expected physical outcome, propose a fix, with
   literature backing where possible. Check back with the user before changing the existing
   schemes.

**Scope.**
- Included: ladder item 7 (CR Sedov-Taylor blast: thermal/CR/kinetic energy partition) and
  ladder item 8 (DSA efficiency vs Kang & Ryu 2013 / Caprioli & Spitkovsky 2014 as a function
  of Mach number).
- Excluded: items 9-12 (integration tests). Item 9 Test A (`test_cr_dsa_shock_jump`) is
  referred to where it overlaps.
- Excluded: CR streaming (as in Phase A).
- The SN/wind CR source terms belong to Phase B in the plan's Sec. 1 but have no ladder item.
  One optional test is listed (B7.8) because item 7's energy partition applies to them too.
- The shock finder (`astronomix/shock_finder3D/`) is shared with the CWB work. Finder changes
  proposed here need the same care as in `pytests/shock_finder3D/FIXES_TODO.md`.

**Basis of this review (2026-10-13, branch `feature/cosmic-rays-grey`, HEAD `c2b197a` plus the
uncommitted working tree, i.e. HLLC + hybrid CR entropy as the grey-CR default).**
- Scripts: `pytests/cosmic_rays_grey/cr_sedov_taylor.py` (item 7) and
  `pytests/cosmic_rays_grey/cr_dsa_mach_dependence.py` (item 8).
- Code: `cr_grey_injection.py` (`inject_crs_at_shocks`, `_spread_over_post_shock_cells`,
  `dsa_efficiency_kang_ryu_2013`), `shock_finder3D/_energy_dissipation.py`
  (`calculate_thermal_energy_flux`), `_iteration_level_updates.py` and `time_integration.py`
  (injection ordering and `dt`).
- Logs: `DESIGN.md` ("Resolved: DSA shock injection", "Open: CR-row advection undershoots ...
  and DSA finder false positives", "Decided: CR shock partition"), `PROGRESS.md` (2026-08-24,
  2026-08-26, 2026-09-26, 2026-10-02 "Known limitation", 2026-10-07/08) and
  `PROGRESS_PHASEA.md`.
- Figures in `pics/07_sedov_taylor/` and `pics/08_dsa_mach_dependence/`. The 48^3 Sedov SVG
  (23 MB, more than 1e5 referenced elements) could not be rasterized, so its numbers come from
  the docstrings and logs. The 224^3 figures date from 2026-10-08 (energy scheme, HLL, before
  the HLLC + hybrid default).
- The tests were **not** re-run for this review.

**Number sources.**
- *(fig.)*: read off a committed figure.
- *(log)*: taken from DESIGN.md, PROGRESS.md, PROGRESS_PHASEA.md or a docstring.
- *(calc.)*: computed for this plan with the self-similar two-fluid Sedov solver in
  Appendix A, or by hand from the formulas quoted.

**Status legend.**
- ✅ Correct physics, and the gate is meaningful.
- ⚠️ The result passes, but the physics check is incomplete or the gate is weak.
- ❌ The result contradicts the physical expectation.

---

## 0. Cross-cutting findings (apply to both items)

| # | Finding | Consequence | Action |
|---|---|---|---|
| Y1 | **No physical reference in either test.** Item 7 gates total-energy conservation twice (the "partition identity" is the difference of two conserving runs) and a band `0.01 < E_cr/E < 0.3`. Item 8 gates conservation, a linear scale knob (CS14/KR13 = 0.5) and a formula cross-check. | How much CR energy is injected, where it sits and how it changes the blast are not tested. The self-similar reference (Sec. 7.2) predicts `E_cr/E` = 0.143 (item 7) and 0.274 (item 8, KR13 strong blast) *(calc.)*. The code gives 0.041 and 0.083 at 48^3 *(log)*, i.e. **0.29x and 0.30x**, and both pass. | B7.1-B7.3, B8.1. |
| Y2 | **Efficiency convention.** KR13 define `eta(M)` relative to the *kinetic* energy flux `0.5 rho_1 u_s^3` (KR13 eq. 8; also Ryu et al. 2003, Kang & Jones 2007, Vazza et al. 2016). The code multiplies `eta_KR13` by the finder's *dissipated* flux `delta(M) 0.5 rho_1 u_s^3`. Pfrommer et al. (2017) and Dubois et al. (2019) use the dissipated-flux convention, but with their own normalization (`zeta`; Dubois rescales KR13 by 1/0.225). | The code's "KR13" injects `delta(M)` times KR13's own fraction: `delta` = 0.15 / 0.32 / 0.46 / 0.53 / 0.56 at M = 2 / 3 / 5 / 10 / strong *(calc.)*. Strong-shock CR yield is 0.12 instead of KR13's ~0.2 of the kinetic flux. Already flagged in PROGRESS.md 2026-10-02; the user decision is pending. | Decision DB1. |
| Y3 | **CR transport in the DSA tests is an undamped free wave.** Items 7 and 8 use the defaults `reduced_streaming_speed = 1`, `diffusive_relaxation = False`. CRs then propagate at `v_red/sqrt(3) = 0.577` relative to the gas. At t = 0.07 the post-shock gas leaves the shock at `V_s/x ≈ 0.54` in the shock frame *(calc.)*, so freshly injected CRs can catch up with the shock and run ahead of it. | Neither the tightly coupled limit (which the analytic references assume) nor a diffusive regime. Item 9 Test A found the same constraint (`v_red < 2 sqrt(3)` at M = 2). | Reference comparisons run at `v_red = 0` (`F_cr ≡ 0`, as in item 6). Production transport is a separate, tracked variant (DB2). |
| Y4 | **The integrated `E_cr` is dominated by the least-resolved phase.** In the self-similar solution, ~39% of the CR energy at time t sits in gas that was shocked while `R_s < R(t)/4`, and ~22% in gas shocked while `R_s < R(t)/10` *(calc., Appendix A)*. At 48^3 and t = 0.07, `R/4` is ~5 cells. The explosion region `R_EXPLOSION = 0.05` (0.13 R, 2.4 cells) is never shocked at all. | A test that gates time-integrated `E_cr` mostly measures early-time under-resolution. It cannot converge faster than the early blast is resolved. | Gate instantaneous or Lagrangian-resolved quantities (B7.2, B7.3). Start from the self-similar state instead of a point explosion. |
| Y5 | **Finder Mach bias enters directly.** The finder reads 0.80 / 0.87 / 0.92 of the true Mach number at 48^3 / 96^3 / 224^3 (PROGRESS.md 2026-10-02) *(log)*. The dissipated flux scales as `delta(M) M^3`, and KR13's rising part amplifies the error. | The item-8 weak-blast coverage check uses finder *medians* and was calibrated at 48^3. It fails at 96^3 and 224^3 because the medians move *towards* the true values, so the gate rewards the bias. | B8.1/B8.2 measure Mach and efficiency against known shocks. Decision DB4 (correct the bias or quote resolution limits). |
| Y6 | **Single resolution, no baselines.** Both items are calibrated at 48^3 only. `E_cr/E` rises from 0.043 to 0.079 between 48^3 and 224^3 (item 7, energy scheme) *(log)* and is not converged. Neither item is in `baselines.json` (Phase A X1); the docstrings carry a history of re-measurements instead. | — | Resolution series 48 / 96 / 160 (/ 224) for each new test. Register items 7 and 8 in `baselines.json`. |
| Y7 | **Scheme interplay is untested at shocks with injection.** The grey-CR default is HLLC + hybrid CR entropy. Weber, Thomas & Pfrommer (2023) find that with CR acceleration the entropy formulation is less stable and gets the shock speed clearly wrong at low resolution, while the energy formulation is accurate. The hybrid was regression-checked for items 7/8 only through `E_cr` totals (0.0411 vs 0.0401 for the energy scheme) *(log)*. | A shock-speed error with injection would not be seen. | B7.4 runs hybrid vs energy scheme against an exact solution. |
| Y8 | **Silent injection limiters.** Three things reduce injection without being counted: the positivity cap (`max_available`); the RH velocity-consistency check (`dsa_velocity_consistency = 0.5`, which also rejects the not-yet-formed blast in its first steps); and `dsa_mach_min`. In addition, `dt` is re-estimated *after* the lump-sum injections, but DSA injects `flux * dt_pre`. If a SN/wind injection in the same step shrinks `dt`, the DSA amount does not match the step it belongs to. | Under-injection cannot be attributed. The `dt` mismatch only matters with other lump-sum injections (SILCC-type runs, CWB), not in items 7/8. | Diagnostic counters (B7.7). Inject at the final `dt`, or rescale (fix proposal F3). |
| Y9 | **FV, Cartesian, uniform grid only** (`area/volume = 1/dx`). | — | Out of scope. Record as a limitation. |

---

## 7. Ladder item 7: CR Sedov-Taylor blast (thermal/CR/kinetic partition)

> Plan: "CR Sedov-Taylor blast: thermal/CR/kinetic energy partition."

### 7.1 What exists

`cr_sedov_taylor.py::test_cr_sedov_taylor`, with figures `pics/07_sedov_taylor/cr_sedov_taylor_test.svg`
(48^3, 2026-10-13) and `..._224.svg` (2026-10-08).
- **Setup.** 3D Cartesian FV, N = 48 (`CR_NUM_CELLS`), unit box, open boundaries, HLLC + MINMOD
  (+ hybrid CR entropy by default).
  - `rho = 1`, `P_amb = 1e-4`, `E = 1` deposited thermally in a tanh-tapered sphere with
    `R_EXPLOSION = 0.05` and a 2-cell taper. `t_end = 0.07`.
  - `gamma = 5/3`, `gamma_cr = 4/3`. Default CR transport (`v_red = 1`, no relaxation; Y3).
  - DSA with the constant model, `dsa_efficiency = 0.1`, `dsa_mach_min = 1.3`, adaptive
    sampling (15 steps), velocity consistency 0.5, injection spread over the post-shock ray
    weighted by `max(p - p_pre, 0)`.
- **Runs.** A control run (`dsa_efficiency = 0`) and a DSA run.
- **Gates.**
  - No NaN.
  - Containment: `r_shock < 0.47`, where `r_shock` is the outermost cell with a density
    departure above 1%.
  - Control `max e_cr ≤ 100 minimum_e_cr`.
  - Total-energy conservation of each run, below 1e-3.
  - `0.01 < E_cr/E < 0.3`.
  - "Partition identity": `(E_th + E_kin)_ctrl - (E_th + E_kin)_dsa = E_cr,dsa` to 1e-2.
- **Measured** *(log)*: `E_cr/E` = 0.0411 (hybrid default; 0.0401 energy scheme), identity
  3.6e-6, conservation 1.3e-5 (48^3). At 224^3 (energy scheme, HLL, 2026-10-08): `E_cr/E`
  0.079, conservation 1.1e-4.

### 7.2 Physical expectation

**Tightly coupled CRs, constant efficiency: a self-similar two-fluid blast.**
- If a fixed fraction `zeta` of the dissipated energy goes into CRs at a strong shock, and the
  CRs are then adiabatic and tightly coupled, the blast stays self-similar.
  - Chevalier (1983) gives the two-fluid self-similar solution with a constant post-shock CR
    fraction.
  - Pfrommer et al. (2017, Sec. 4.2) show in AREPO that CR acceleration "does not
    significantly break the self-similarity" of the Sedov-Taylor solution.
  - Pais et al. (2018) and Dubois et al. (2019, Sec. 4.4) reproduce this.
- The code's convention (`eta` times the dissipated flux) is Pfrommer's `zeta`. With
  `e_cr,2 = zeta e_diss` and `e_th,2 = (1 - zeta) e_diss` at a strong shock, the post-shock
  ratio is `X_cr,0 = P_cr/P_th = zeta / (2 (1 - zeta))` (Pfrommer et al. 2017).
- **Shock compression.** With a fixed partition, the strong-shock jump gives
  `x = (Gamma_e + 1)/(Gamma_e - 1)`. Here `Gamma_e = 1 + P/e` is the *caloric* index of the
  post-shock mixture, not the sound-speed index `gamma_eff`. The result is x = 4.16
  (`zeta` = 0.1), 4.35 (0.211) and 5.0 (0.5) *(calc.)*.
  - Pfrommer et al. (2017) eq. 67 quote 4.6 for `zeta = 0.5`, using `gamma_eff = 14/9`.
  - Their own exact App. B solution gives 4.74 at M = 9.56, which is consistent with a
    strong-shock limit of 5.0, not 4.6.
  - Use App. B or Appendix A of this plan, not eq. 67. *(Verify against the paper before
    citing.)*
- **After the shock**, every element keeps `P_th rho^{-5/3}` and `P_cr rho^{-4/3}`. So
  `X_cr ∝ rho^{-1/3}` rises steeply into the rarefied interior (Pfrommer et al. 2017, eq. 64).

**Reference values** (Appendix A solver, `rho_0 = E = 1`, `R = xi_0 t^{2/5}`; *calc.*):

| `zeta` | `X_cr,0` | x | `xi_0` | `R(0.07)` | `E_kin/E` | `E_th/E` | `E_cr/E` | `dE_inj/dt * t/E` |
|---|---|---|---|---|---|---|---|---|
| 0 (classic) | 0 | 4.000 | 1.1516 | 0.3975 | 0.2828 | 0.7172 | 0 | 0 (dissipation 0.458) |
| 0.1 (item 7) | 0.056 | 4.158 | 1.1289 | 0.3897 | 0.2673 | 0.5898 | **0.1429** | 0.0425 |
| 0.211 (item 8 strong, code convention) | 0.134 | 4.354 | 1.1066 | 0.3820 | 0.2531 | 0.4731 | **0.2738** | 0.0835 |
| 0.375 (item 8 strong, KR13's kinetic convention, `0.211/delta`) | 0.300 | 4.693 | 1.0775 | 0.3719 | 0.2362 | 0.3330 | 0.4308 | 0.1357 |
| 0.5 (Pfrommer 2017 / Dubois 2019 tests) | 0.5 | 5.000 | 1.0578 | 0.3651 | 0.2258 | 0.2455 | 0.5287 | 0.1704 |

- **Solver checks.** The `zeta = 0` row reproduces Sedov: `xi_0` = 1.1517, `E_kin/E` = 0.28.
  The gas-only dissipation rate 0.458 E/t agrees with the independent "~0.46 E/t" estimate in
  PROGRESS.md 2026-10-07.
- **Pais et al. (2018) eq. 16**, `gamma(zeta) = (4/3)(8 zeta + 5)/(9 zeta + 4)`, gives an
  effective single-fluid index of 1.578 for `zeta = 0.1` and 1.512 for 0.211. That is an
  alternative, coarser check of `xi_0`.
- **Shell profile, `zeta = 0.1`** (`lambda = r/R`, `r_0/R` = Lagrangian origin; *calc.*):

  | `lambda` | 0.98 | 0.95 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
  |---|---|---|---|---|---|---|---|
  | `r_0/R` | 0.92 | 0.82 | 0.68 | 0.47 | 0.32 | 0.21 | 0.13 |
  | `X_cr` | 0.060 | 0.067 | 0.080 | 0.106 | 0.137 | 0.181 | 0.249 |
  | `P_cr/P_2` | 0.049 | 0.044 | 0.040 | 0.038 | 0.042 | 0.050 | 0.064 |

  At t = 0.07: `V_s = 2.23`, `P_2 = 3.77` and `P_cr,2 = 0.198` *(calc.)*.

### 7.3 Step-by-step check of script and figure

1. **Conservation: ✅.** 1.3e-5 at 48^3 and 1.1e-4 at 224^3, the CR-free control level
   *(log)*. The floor problem of 2026-10-07 went away with the finder fix.
2. **"Partition identity": ⚠️, not a partition test.** Both runs conserve total energy, and
   injection moves energy from gas to CRs. So the identity holds for *any* injected amount,
   location or timing. It is a second conservation check, and calling it the item-7 physics
   gate overstates what it tests.
3. **Control against classic Sedov: ✅.**
   - 224^3 *(fig.)*: `E_th` 0.725 and `E_kin` 0.276, against 0.717 and 0.283 (kinetic 2.5%
     low, the usual shock smearing).
   - Central pressure about 1.15 against `0.31 P_2` = 1.15 for `gamma = 5/3`.
   - Pressure peak at r ≈ 0.39 against `R` = 0.3975. The test's `r_shock` = 0.414 is the outer
     foot of the smeared shock, not the shock position.
4. **CR energy: ❌ about 3.5x too low at 48^3, about 1.8x at 224^3.**
   - `E_cr/E` = 0.041 (48^3) and 0.079 (224^3, energy scheme) against 0.143 *(log vs calc.)*.
   - Even if injection only starts outside the explosion region (`r_0 > 0.13 R`, which removes
     ~24% of the reference), the bound is 0.109. 48^3 then still reaches only 0.38 of it.
   - The `[0.01, 0.3]` band cannot see this.
   - Causes, all logged and none quantified separately:
     - (a) the finder's Mach/flux bias (Y5);
     - (b) early-time under-injection while the blast spans only a few cells, including
       rejections by the velocity check (Y4, Y8);
     - (c) CR free-wave transport (Y3);
     - (d) the instantaneous rate itself: rate·t ≈ 0.015-0.019 against 0.0425 at 48^3, i.e.
       0.35-0.45x *(log, 2026-10-07)*. So even the late, resolved injection is too low at 48^3.
   - Point (d) shows that resolution is not the whole story at late times. B7.3 separates
     (a) and (d).
5. **Radial profile (224^3, 2026-10-08): ❌ wrong CR distribution** *(fig.)*.
   - Interior (r = 0.2-0.33, `lambda` ≈ 0.5-0.85): `P_cr` ≈ 0.07-0.08 against a reference of
     0.14-0.24. That is 0.3-0.5x, which shows early under-injection.
   - At the shock: the `P_cr` spike reaches ≈ 0.30 where `P_gas` ≈ 3.1, i.e. `X_cr` ≈ 0.1
     against `X_cr,0` = 0.056. That is **about 1.8x too high in the shocked layer**.
   - Dubois et al. (2019, Sec. 4.2) find that depositing into the shock cell overestimates the
     post-shock CR energy by about 2x, and deposit up to four cells downstream (minimum
     `|div u|`).
   - Our spread weights by `p - p_pre` along the ray. Possible causes of the spike:
     - (a) deposit into cells still being compressed, where `-P_cr div u` then adds energy;
     - (b) Y3 pile-up of CR waves at the shock;
     - (c) a spike that is just smearing, against a `P_gas` peak that is 18% low (3.1 vs 3.77).
   - B7.2 decides between these.
6. **Shock slowdown: ⚠️, not checked.** CR loading should shrink `R` by 2.0% for
   `zeta = 0.1` (0.4 cells at 48^3, 1.7 cells at 224^3) *(calc.)*. This is the dynamical
   signature that Pfrommer et al. (2017, Fig. 3) and Dubois et al. (2019, Fig. 9) use.
7. **Coverage: ⚠️.**
   - Single resolution (Y6) and a single `zeta`.
   - No isotropy check. Pfrommer et al. (2017) report radial "fingers" where converging rays
     inject twice; Dubois et al. (2019) a π/2-periodic grid pattern. Our nearest-cell ray
     deposit is prone to both.
   - No test of `dt`/CFL independence or of the sampling knobs.
8. **Figure: ⚠️.** The bar chart shows energies but no reference. The profile panel plots
   `P_cr` and `P_gas` without the self-similar curves and without `X_cr`. The 48^3 SVG is
   23 MB and unrenderable: rasterize the scatter, as done for 224^3.

**Verdict: ✅ for energy conservation and the control blast. ❌ for the CR partition.** The
amount is ~3.5x low at 48^3, the interior is under-loaded and the shocked layer is over-loaded,
and none of the existing gates can see it.

### 7.4 Complementary tests from the literature

**B7.1 Self-similar two-fluid Sedov reference (infrastructure + self-test).**
- *Reference.* Chevalier (1983); Pfrommer et al. (2017, Sec. 4.2, eqs. 64-67); Sedov (1959) for
  `zeta = 0`.
- *Build.* `test_setups/reference_solutions/sedov_two_fluid.py`, from the Appendix A sketch.
  It replaces the placeholder `sedov_solver.py`. Outputs:
  - `xi_0`, the profiles `U, G, P_th, P_cr (lambda)`, the energy fractions and
    `dE_inj/dt`;
  - the Lagrangian map `r_0(lambda)`.

  The solver integrates the four similarity ODEs (continuity, momentum, two entropies) from the
  strong-shock state with the given `zeta` inward to `lambda = 0.02`.
- *Gates (pure-Python pytest).*
  - `zeta = 0`: `xi_0 = 1.15167 ± 1e-4`, `E_kin/E = 0.2828 ± 1e-3`, and `xi_0 = 1.0328` for
    `gamma = 7/5` in a gas-only variant.
  - Energy closure of the integral below 1e-4.
  - Mass `3 ∫ G lambda^2 dlambda = 1 ± 1e-3`.
  - `X_cr(1) = zeta/(2(1-zeta))`.
- *Cross-check.* Against Pfrommer et al. (2017) Fig. 3: `X_cr ≈ 5 (delta/1e-3)^{-1/3}` for
  `zeta = 0.5`; and the `gamma = 7/5` shock radius (`alpha = 0.851`), which should agree within
  about 3% (it is an approximation in their paper).

**B7.2 Sedov from the self-similar state, tightly coupled CRs (the main item-7 physics test).**
- *Reference.* B7.1. Starting from the self-similar profile at `t_0`, instead of a point
  explosion, follows the Pakmor et al. (2016) practice of starting from an analytic state. It
  removes the unresolved early phase (Y4).
- *Setup.*
  - The item-7 box and ambient gas. Initial state: the B7.1 solution at `R_0 = 0.15`
    (7 cells at 48^3), including `e_cr`.
  - `v_red = 0` (`F_cr ≡ 0`, the tightly coupled limit of the reference).
  - `zeta` ∈ {0.1, 0.211}, the constant model.
  - Run to `R ≈ 0.42`, before the boundary. N = 48, 96, 160 (224 tracked).
  - HLLC + hybrid (default) and the energy scheme (Y7).
- *Expectation.* All profiles stay on the self-similar solution, and the energy fractions stay
  constant in time (B7.1 table).
- *Measure.*
  - `E_cr/E`, and its drift `d(E_cr/E)/dt`.
  - The shell-averaged `X_cr(lambda)` and `P_cr/P_2` at `lambda` ∈ [0.7, 0.95], where all gas
    was shocked during the run.
  - The `P_cr` spike height and width at the shock.
  - L1 of `rho`, `P_tot` and `P_cr` against the reference, by radius.
- *Gates.*
  - Tracked first.
  - Proposed after DB4: `|E_cr/E_ref - 1| < 0.1` and `|X_cr/X_ref - 1| < 0.2` for
    `lambda` ∈ [0.7, 0.9] at N ≥ 96, with monotone convergence in N.
  - The shocked-layer spike `X_cr,max / X_cr,0` decreasing with N.
- *Catches.*
  - Under- or over-injection that is not due to early times.
  - The shocked-layer overshoot of 7.3 step 5 (deposit location vs Y3).
  - Hybrid-scheme errors with injection.

**B7.3 Instantaneous injection rate against the analytic rate (cheap, no code change).**
- *Reference.* The B7.1 table: `dE_inj/dt = zeta * 0.425 E/t` for `zeta = 0.1` (the dissipation
  rate scales with `zeta` as in the table).
- *Setup.*
  - Save snapshots of the existing item-7 runs and of the B7.2 runs, about 10 per run.
  - On each snapshot, call the pure functions `find_shocks_pfrommer` and `inject_crs_at_shocks`
    with `dt = 1`, as the item-8 cross-check does. The injected `sum(delta e_cr) dV` is the
    instantaneous rate.
  - Also record the finder's `thermal_energy_flux` integrated over the shock surface (the
    dissipation rate seen by the finder), against the analytic 0.458 E/t (`zeta = 0`).
- *Expectation.* At `t ≥ 0.03`, `rate * t / E` approaches the analytic constant. The ratio
  `finder dissipation / analytic` exposes the Mach bias (Y5) directly.
- *Gates.*
  - Tracked: ratio per N.
  - Target: at least 0.9 at N = 160, and an order of about 0.6 in N, matching the finder's
    Mach convergence (PROGRESS.md 2026-10-02).
- *Why.* This isolates late-time injection from early-time history (7.3 step 4d). It costs
  nothing beyond saved states.

**B7.4 Planar shock tube with CR acceleration against the exact Riemann solution.**
- *Reference.* Pfrommer et al. (2017), App. B (thermal gas + injection) and App. C (composite
  with pre-existing CRs + injection), Table 1. Reproduced by Dubois et al. (2019, Sec. 4.3.2).
  - `rho = 1 | 0.125`; `P_5 = 63.499` (th) or `51.516` with `X_cr = 2 | 1` (CR);
    `P_1 = 0.1`; `zeta = 0.5`.
  - Compression `x_s` = 3.88 → 4.74 (th, with injection). The Mach number drops from 10.00 to
    9.56 (both cases).
  - 100 cells (Pfrommer), `t = 0.35`.
- *Build.* Extend `pfrommer_riemann_solver.py` with App. B eqs. 83-95 (one unknown `x_s`) and
  App. C eqs. 96-110 (unknowns `x_s`, `x_r`). Reduce to the existing composite solver for
  `zeta = 0`.
- *Setup.*
  - 1D, `v_red = 0`, `zeta` ∈ {0.1, 0.5}, N = 100 to 3200.
  - Default scheme (HLLC + hybrid) against the energy scheme (HLL and HLLC).
  - Spread deposit on, and off (surface cell only, `dsa_adaptive_shock_sampling = False`).
- *Expectation.* Shock speed, compression and plateau `P_th`, `P_cr` match the exact solution,
  with L1 order about 1 for the discontinuous solution.
  - Weber et al. (2023): an entropy-type CR variable gets the shock speed wrong with injection.
  - Dubois et al. (2019): a shock-cell deposit gives a post-shock `P_cr` about 2x too high, and
    a few-cells-downstream deposit gives a `P_th` overshoot just behind the shock.
- *Gates.*
  - Shock position within 1 dx at t.
  - Post-shock plateau `P_cr` and `P_th` within 2% at N ≥ 400.
  - `M_finder` within 1% of 9.56 (1D finder accuracy, Dubois et al. 2019 Sec. 4.3.1: < 0.1%
    with 4-cell sampling).
  - Scheme comparison tracked.
- *Relation to item 9 Test A.* Test A checks M ≈ 2 with `zeta = 0.1` against an RH bracket on
  the injected flux. B7.4 adds an exact solution, strong shocks, the shock-speed change and the
  scheme comparison.
- *Catches.* Wrong injected amount at a resolved planar shock, shock-speed errors of the
  default scheme, and the deposit-location bias.

**B7.5 Shock slowdown (effective adiabatic index).**
- *Reference.* B7.1: `R_dsa/R_ctrl = xi_0(zeta)/xi_0(0)` = 0.980 (`zeta = 0.1`), 0.961
  (0.211), 0.918 (0.5) *(calc.)*. Pfrommer et al. (2017, Fig. 3), Pais et al. (2018, eq. 16)
  and Dubois et al. (2019, Fig. 9).
- *Setup.* B7.2 runs, plus point-explosion runs at `zeta = 0.5` (the published value, which
  gives the largest signal: 8.2%). Fit `R(t) = xi t^{2/5}` over at least 8 snapshots from the
  peak of the shell-averaged density or pressure profile, using sub-cell (parabolic) peak
  location.
- *Gates.* Ratio within 0.5% of the reference at N ≥ 96 (B7.2 IC) and within 1% for the point
  explosion at `zeta = 0.5`. The fitted exponent is 0.400 ± 0.005.

**B7.6 Isotropy of the deposit (grid imprint, double injection).**
- *Reference.* Pfrommer et al. (2017, Sec. 4.2): radial fingers from converging rays. Dubois
  et al. (2019, Sec. 4.4): π/2-periodic pattern.
- *Setup.* B7.2 at N = 96. Bin the shell (`lambda` ∈ [0.7, 1]) into angular bins: about 48
  equal-area bins, or simply axis, face-diagonal and body-diagonal cones of 10°.
- *Measure.* `E_cr` per solid angle, normalized by its mean. The same for `E_th` as a control:
  the hydro has its own grid imprint.
- *Gates.* Tracked. Target: angular max/min of `E_cr` within 1.2 and no worse than 2x the
  `E_th` anisotropy.

**B7.7 Robustness: `dt`, sampling knobs, silent limiters.**
- *Setup.* B7.2 at N = 96, varying one at a time:
  - `C_cfl` = 0.2, 0.4, 0.8;
  - `dsa_shock_sampling_max_steps` = 8, 15, 25;
  - `dsa_velocity_consistency` = 0.3, 0.5, 0.7;
  - `dsa_mach_min` = 1.3 and 2.25.
- *Diagnostics (new, opt-in).* Per step: the number of cells and energy clipped by the
  positivity cap, the cells rejected by the velocity check, and the injected energy. The
  simplest form is a returned diagnostic from a test-side wrapper around
  `inject_crs_at_shocks`; nothing changes in the hot path.
- *Gates.*
  - `E_cr` varies by less than 2% with `C_cfl` (injection is `flux * dt`, so it must be
    rate-consistent).
  - Less than 5% across the sampling knobs at the resolved shock.
  - The positivity cap removes less than 1e-3 of the injected energy.
- *Also check (Y8).* A combined SN + DSA step where the re-estimated `dt` is smaller than
  `dt_pre`. The DSA energy should equal `flux * dt_final`. This is fix proposal F3.

**B7.8 (optional) SN source term: composite blast with direct CR deposit.**
- *Reference.* Girichidis et al. (2016) / Pfrommer et al. (2017, Sec. 3.2) convention:
  `zeta_SN = 0.1` of `E_SN` deposited as CRs. A 1D spherical high-resolution run of the same
  code serves as reference (no self-similar solution: the CR fraction is not set at the shock).
- *Setup.* The item-7 IC with `sn_cr_fraction = 0.1` deposited through `_inject_supernovae`,
  DSA off, `v_red = 0`. Compare against a thermal-only deposit.
- *Gates.*
  - The deposited CR energy equals `0.1 E_SN` to round-off.
  - Energy conservation.
  - `E_cr(t)` follows the adiabatic-loss curve of the reference within 5%.
  - `R(t)` slowdown sign and size (tracked).

---

## 8. Ladder item 8: DSA efficiency vs Mach number (KR13 / CS14)

> Plan: "DSA efficiency vs Kang & Ryu (2013) / Caprioli & Spitkovsky (2014) as a function of
> Mach number, using the PR #4 finder output."

### 8.1 What exists

`cr_dsa_mach_dependence.py`, with figures `pics/08_dsa_mach_dependence/cr_dsa_mach_dependence_test{,_48,_96,_224}.svg`.
1. **`test_dsa_efficiency_kang_ryu_2013_shape`** (pure function):
   - `eta = 0` below M = 2;
   - monotone;
   - continuous within 5% at M = 5 and within 1% at M = 15;
   - plateau 0.211.
2. **`test_cr_dsa_mach_dependence`**. The item-7 strong blast with KR13 (`mach_scale = 1`) and
   "CS14-like" (`mach_scale = 0.5`), plus control runs into `P_amb` = 0.02 ... 0.4 (weak
   blasts). Gates:
   - no NaN, containment, control `E_cr` at floor level;
   - conservation below 1e-3;
   - `0.001 < E_cr/E < 0.5`;
   - CS14/KR13 `E_cr` ratio within 8% of 0.5;
   - exact formula cross-check (finder + fit applied by hand against `inject_crs_at_shocks`)
     below 1e-5, on every control state;
   - weak-blast finder medians spanning below 2.6 and above 8.
- **Measured** *(log)*:
  - KR13 0.0826 and CS14-like 0.0432 (48^3, hybrid default);
  - ratio error 4.3% (48^3), 5.9% (96^3), 7.9% (224^3);
  - cross-check 1.3e-7.
- **Figure (48^3)** *(fig.)*:
  - Left: the bar chart.
  - Right: the KR13 fit with the control runs' shock cells. Clusters at M ≈ 2.5 (`eta` ≈ 0.002),
    3.7 (0.02), 5.0 (0.08-0.11), 8.2 (0.18), 10 (0.195) and about 140 (plateau). The exact
    Sedov Mach number of the strong blast, 176, is marked.

### 8.2 Physical expectation

- **Kang & Ryu (2013)**, ApJ 764, 95.
  - `eta(M_s) = [e_c,2 - e_c,0 (rho_2/rho_0)^{gamma_c}] u_2 / (0.5 rho_0 u_s^3)`, i.e. the
    CR energy flux relative to the **kinetic** flux (eq. 8). `delta(M_s)` is the gas analogue.
  - `eta` saturates at about 0.2 for `M_s ≳ 10`, varies by about 10% with model parameters, and
    is below 0.01 for `M_s ≲ 3` without pre-existing CRs.
  - Pre-existing CRs raise the weak-shock efficiency (`eta'` up to 0.1).
  - For `beta_P ~ 1` (ISM-like), the strong-shock `eta` drops to about 0.12 (Kang 2012).
  - KR13 publish curves (their Fig. 4), not a closed-form fit.
- **Caprioli & Spitkovsky (2014a)**, ApJ 783, 91 (hybrid PIC, M = 5-50, `theta` = 0-80°).
  10-20% of the bulk kinetic energy goes into ions at quasi-parallel shocks (`theta ≲ 45°`);
  acceleration is ineffective at quasi-perpendicular shocks.
- **"CS14 = 0.5 KR13" (Vazza et al. 2016)** holds only for M ~ 5-10 *quasi-parallel* shocks.
  For random obliquities (only ~0.3 of shocks have `theta ≤ 45°`), Vazza et al. rescale KR13
  by 0.15, not 0.5.
- **Ryu, Kang & Ha (2019)**, ApJ 883, 60 (weak quasi-parallel ICM shocks): no proton DSA below
  `M_s ≈ 2.25`; `eta` ~ 1e-3 to 1e-2 for 2.25 ≤ M ≤ 5. That is lower than KR13 near M = 5.
- **Pfrommer et al. (2017).** `zeta ≈ 0.1` of the *dissipated* energy for `theta ≲ 45°` and
  `M ≥ M_crit ≈ 3`; zero otherwise.
- **Pais et al. (2018, eq. 7) and Dubois et al. (2019, eq. 20).**
  `zeta(theta) = (zeta_0/2) [tanh((theta_crit - theta)/delta) + 1]`, with
  `theta_crit = π/4` and `delta = π/18`. Pais et al. give `zeta_0 ≈ 0.15` from PIC.
  - Dubois et al. use `eta_0 xi_KR13(M, X_cr)/0.225` times the dissipated flux, interpolating
    KR13 in `X_cr` (their App. B, Table 1).
  - For a sphere in a uniform field, the area-averaged `<zeta>/zeta_0` = 0.302 *(calc.)*.
    Dubois et al. measure 0.302 in a random field.

### 8.3 Step-by-step check of script and figures

1. **Fit shape: ✅ internally, ⚠️ unverified against KR13.**
   - Spot values *(calc.)*: `eta(3)` = 0.0060, `eta(5)` = 0.100, `eta(10)` = 0.196, plateau
     0.211.
   - These agree with KR13's text (below 0.01 at M ≲ 3; about 0.2 at M ≳ 10) and with KR13's
     `P_c,2/(rho_0 u_s^2)` = 0.076 at M = 5 (`eta ≈ 6 x 0.076/x ≈ 0.11`).
   - The docstring attributes the coefficients to Vazza et al. (2016) and Girichidis et al.
     (2022). Vazza et al. (2016) contains no closed-form fit, so the source of the coefficients
     is unverified. The test checks continuity and monotonicity, not agreement with KR13's
     data.
2. **Convention: ❌ (Y2).** The code's realized KR13 efficiency relative to the kinetic flux is
   `delta(M) eta_KR13` *(calc.)*:

   | M | 3 | 5 | 10 | strong |
   |---|---|---|---|---|
   | `delta(M) eta_KR13` | 0.0019 | 0.046 | 0.105 | 0.119 |
   | KR13 | 0.006 | 0.10 | 0.196 | 0.211 |

   That is a factor of 1.8-3.2 below KR13 by construction.
3. **CS14/KR13 ratio gate: ⚠️, tests a linear knob.** 0.52 ± 0.04 is expected for any
   implementation that multiplies by `mach_scale`. The physics ("CS14-like") is not tested, and
   the name hides the missing obliquity dependence (Vazza: 0.15 for random obliquity).
4. **Formula cross-check: ✅ for code consistency, ⚠️ for physics.** It proves that
   `inject_crs_at_shocks` applies the fit to the finder's output. It cannot detect a wrong Mach
   number, a wrong flux or a wrong convention, because the reference uses the same inputs.
5. **Mach coverage panel: ❌ physically, ⚠️ as a gate** *(fig. + log)*. The points sit on the
   KR13 curve at the *finder's* Mach number (0.80x the true one at 48^3):

   | blast `P_amb` | true M | finder M | `eta(true)` | `eta(finder)` |
   |---|---|---|---|---|
   | 0.08 | 6.3 | 5.07 | ≈ 0.15 | ≈ 0.10 |
   | 0.16 | 4.53 | 3.67 | ≈ 0.06 | ≈ 0.02 |
   | 0.4 | 3.0 | 2.47 | 0.006 | ≈ 0.002 |

   Weak-shock efficiency is therefore up to ~3.5x too low at 48^3 (PROGRESS.md: about 6x
   including the flux). The coverage gate (median below 2.6) only passes *because* of the bias
   and fails at 96^3/224^3 (Y5).
6. **Strong-blast `E_cr`: ❌.** KR13 0.083 at 48^3 *(log)* against the self-similar 0.274
   (code convention) or 0.431 (KR13's own convention) *(calc.)*. This is the same deficit as
   item 7 (0.30x), plus Y2.
7. **Pre-existing CRs and the composite Mach number: ⚠️, untested.**
   - The finder computes M and the dissipated flux from the *thermal* pressure jump and
     `c_pre(gas)`.
   - Pfrommer et al. (2017) and Dubois et al. (2019, eq. 16) use the composite Mach number for
     a thermal + CR upstream.
   - KR13's weak-shock `eta` depends on the upstream `X_cr`, which the code ignores (Dubois et
     al. interpolate it).
   - Upstream CRs are the normal case in SNR-cloud and SILCC-type runs.
8. **Sub-critical shocks: ⚠️.** KR13 is zero below M = 2. Ryu et al. (2019) put the threshold
   at 2.25 and Pfrommer et al. (2017) use 3. The constant model injects down to
   `dsa_mach_min = 1.3`, where the plasma literature expects no proton DSA.
9. **Coverage: ⚠️.** No planar test, no oblique shocks, a single resolution (Y6) and no
   obliquity.

**Verdict: ✅ for implementation consistency (the fit is applied as written). ❌ as a physics
test of item 8.** The realized efficiency against Mach number is never measured. It sits 2-3x
below KR13 by convention (Y2) and another ~3x below at weak shocks from the finder bias (Y5),
and the coverage gate rewards that bias.

### 8.4 Complementary tests from the literature

**B8.1 Realized efficiency `eta_eff(M)` at planar shocks (the actual item-8 measurement).**
- *Reference.*
  - KR13 eq. 8 for the definition.
  - B7.4's exact App. B solution, with `zeta` replaced by the model value at the true Mach
    number. The Mach number depends on `zeta` through the shock speed (10.00 → 9.56 at
    `zeta = 0.5`), so solve the fixed point `zeta = eta_model(M_1(zeta))`, a 1D root find.
- *Setup.*
  - 1D shock tubes (or the item-9 RH piston) with true M ∈ {1.8, 2.0, 2.25, 2.5, 3, 4, 5, 7,
    10, 15, 30, 100}.
  - `v_red = 0`, KR13 and constant models, N = 400 and 1600.
  - Measure in the shock frame: `eta_eff = e_cr,2 (u_2 relative to the shock) / (0.5 rho_1 u_s^3)`
    on the post-shock plateau, with `u_s` from a fit to the shock position.
- *Expectation.*
  - `eta_eff = delta(M) eta_KR13(M)` with the current convention, or `eta_KR13(M)` under DB1
    option (a).
  - Exactly 0 for M < 2 (KR13) or below `dsa_mach_min` (constant model).
- *Gates.*
  - `|eta_eff/eta_ref - 1|` < 5% for M ≥ 3 at N = 1600.
  - Below 20% for 2.25 ≤ M < 3 (steep part).
  - `E_cr = 0` to floor level for M = 1.8.
  - The figure plots `eta_eff(M_true)` over the KR13 curve, with the finder's Mach number as a
    second marker. This replaces the current right panel.
- *Why.* It is the plan's item 8 literally, measured where the Mach number is known exactly. It
  is cheap: 24 1D runs.

**B8.2 Finder Mach number and injection at shocks oblique to the grid.**
- *Reference.* Dubois et al. (2019, Sec. 4.3.1): Sod at M = 10, 100 and 1000; 2-cell sampling
  underestimates M; 4-cell sampling is accurate to < 0.1% in 1D. Schaal & Springel (2015):
  shock-finder validation on Sedov and shock tubes.
- *Setup.*
  - Planar M = 3, 5 and 10 shocks along the grid, at 26.6° and 45° (2D), and along
    `(1,1,1)/sqrt(3)` (3D).
  - Periodic in the transverse directions, N = 64 to 256 across.
  - `v_red = 0`, KR13.
- *Measure.* The finder's M against the truth; the injected energy per unit shock area per unit
  time against the B8.1 value along the grid; the scatter over the surface.
- *Gates.* Tracked. Target: angle dependence of the injected flux below 5% at N = 128. The 1D
  finder error stays below 1% (M ≤ 100).
- *Why.* It separates the geometric part of the Sedov Mach bias (curvature, oblique sampling)
  from the post-shock smearing part. The Sedov blast mixes both.

**B8.3 The KR13 fit against published data.**
- *Reference.* KR13 Fig. 4 (`X_cr = 0` models, `eta` against `M_s`), and Dubois et al. (2019)
  App. B Table 1 (KR13 `xi(M, X_cr)` renormalized by 0.225). Both should be digitized or
  transcribed into `test_setups/reference_solutions/dsa_efficiency_data.py`.
- *Gates.* The fit lies within max(10%, 0.005) of the data for M ∈ [2.5, 100]. Record the
  `X_cr = 0.05` curve as the size of the pre-existing-CR effect.
- *Action.* Pin down the source of the coefficients and fix the docstring citation.

**B8.4 Mach-resolved injection in the weak Sedov blasts against a 1D spherical reference.**
- *Reference.* The converged 1D spherical reruns of PROGRESS.md 2026-10-02 (4096 cells, Mach
  from the shock speed and from the pressure jump agreeing within 1%). They are not in the repo
  yet: commit them as a reference module plus cached data.
- *Setup.* The existing weak blasts (`P_amb` = 0.02 ... 0.4) with KR13, N = 48, 96 and 160.
  Use B7.3's snapshot method for the instantaneous injection rate.
- *Expectation.* `rate(t) = eta_model(M_ref(t)) * Ediss_ref(t)` from the 1D reference.
- *Gates.*
  - Tracked ratio per blast and N.
  - Replace the median coverage gate by "the finder's median Mach number lies within
    [0.75, 1.05] of the reference", which passes at every N and catches a regression in either
    direction.

**B8.5 Shocks into CR-loaded gas (composite Mach number, re-acceleration).**
- *Reference.*
  - Pfrommer et al. (2017) Table 1, cases "CR" and "CR+inj" (`X_cr` = 2 | 1, M = 10.00 →
    9.56), App. C. Reproduced by Dubois et al. (2019, Sec. 4.3.2).
  - Weak shocks M = 2.5 and 3 with upstream `X_cr` ∈ {0.05, 0.5, 1} (the KR13 regime where
    pre-existing CRs matter).
  - Phase A A6.2 tube A (`X_cr = 5`, M 2.59).
- *Setup.* 1D, `v_red = 0`, default scheme (the hybrid makes the adiabatic CR compression
  exact, so any remaining error is DSA's).
- *Measure.*
  - The finder's M against the exact composite M.
  - The dissipated flux against Pfrommer eq. 96 (`e_diss` excludes the adiabatic CR
    compression).
  - The post-shock `P_cr` split into its adiabatic and injected parts.
- *Gates.* Plateau within 2% of App. C at N = 1600. A finder Mach error above 2% is a finding
  for DB4 (composite Mach option in the finder).

**B8.6 Obliquity-dependent efficiency (needs a new feature, DB3).**
- *Reference.*
  - Caprioli & Spitkovsky (2014a); Pais et al. (2018, eq. 7); Dubois et al. (2019, eq. 20 and
    Sec. 4.3.3, 4.4); Vazza et al. (2016).
  - `theta` is the angle between the shock normal (finder `shock_direction`) and the upstream
    B.
- *Tests.*
  - (a) Dubois et al. (2019) Sec. 4.3.3: Sod with a dynamically negligible field at
    `theta_B` = 30°, 45°, 60°. Expected `zeta/zeta_0` = 0.953 / 0.500 / 0.047 *(calc.)*, with a
    gate on the post-shock `P_cr` within 3%.
  - (b) Sedov in a uniform weak field (3D MHD, N = 96): `E_cr/E_cr(isotropic)` = 0.302 for a
    self-similar shell *(calc.)*, gated within 10%, with polar caps.
  - (c) Optional: the Pais et al. (2018) ellipticity `epsilon = 0.25 ± 0.02` at `zeta_0 = 0.5`,
    `beta = 1`, 200^3. Check the memory budget first: items 11/15 at 256^3 did not fit 11 GB.
- *Why.* Without obliquity, "CS14-like" overestimates the mean efficiency by about 3x for
  random fields. Pfrommer, Pais and Dubois all make the efficiency obliquity-dependent.

**B8.7 Sub-critical shocks.**
- *Setup.* Folded into B8.1 (M = 1.8, 2.0, 2.25), plus the constant model at M = 1.5 and 2.
- *Expectation.* No injection below the threshold. With the hybrid default, no CR gain from the
  scheme either: this was A6.3's M 2-3 issue, which is now fixed.
- *Gate.* `E_cr` at floor level. Document which threshold the constant model uses after DB5.

---

## 9. Fix proposals (for discussion; nothing implemented)

- **F1 (Y2, DB1): make the efficiency convention explicit.** Add a
  `dsa_efficiency_reference` option: `DISSIPATED` (current, Pfrommer/Dubois `zeta`) or
  `KINETIC` (KR13/Vazza/Ryu 2003 `eta`). With `KINETIC`, inject `eta * f_kin`, limited to the
  dissipated flux (`eta/delta ≤ 1` holds for KR13 at every M: 0.019 at M = 3, 0.38 at strong
  *(calc.)*). Default as decided. Tests B8.1/B8.3 follow the choice.
- **F2 (7.3 step 5, Y3): deposit location.** If B7.2 shows the shocked-layer overshoot with
  `v_red = 0`, switch the spread weight from `p - p_pre` to Dubois et al. (2019)'s rule:
  deposit at the first post-shock cell with minimum `|div u|`, up to 4 cells. Otherwise exclude
  cells with `div u < 0` beyond the surface cell. Literature: Dubois et al. (2019, Sec. 4.2);
  Pfrommer et al. (2017, eqs. 43-44).
- **F3 (Y8): `dt` consistency.** Inject DSA after the post-injection `dt` re-estimate, or scale
  the DSA amount by `dt_final/dt_pre`. Only matters with SN/wind in the same step.
- **F4 (Y5, DB4): Mach bias.** Options, all in the shared finder:
  - (a) Mach number from the RH density or velocity jump, which is better resolved than the
    pressure peak (PROGRESS.md 2026-10-02);
  - (b) a calibrated correction against B8.2;
  - (c) no change, but quote item 7/8 results only as converged trends.

  Any finder change needs a re-run of the shock_finder3D suite and the CWB checks.

---

## 10. Execution order and deliverables

**Priority 1** (cheap, decides DB1/DB4, no code change in the module):
1. **B7.1** self-similar reference module with its self-test.
2. **B7.3** instantaneous injection rate from snapshots of the existing runs. Splits the
   item-7 deficit into finder flux vs early time.
3. **B7.4 and B8.1** planar exact-solution tests in 1D, including the scheme comparison (Y7)
   and the realized `eta_eff(M)`.
4. **Y6** register items 7 and 8 in `baselines.json`.

**Priority 2:**
1. **B7.2** Sedov from the self-similar state, `v_red = 0`, resolution series (the new item-7
   physics gate).
2. **B7.5** shock slowdown.
3. **B8.3** fit vs KR13 data, **B8.5** CR-loaded upstream, **B8.2** oblique planar shocks.
4. Decisions DB1-DB5 with the results above.

**Priority 3:**
1. B7.6 isotropy, B7.7 robustness and counters.
2. B8.4 weak blasts against the 1D spherical reference (commit it).
3. B8.6 obliquity (after DB3), B8.7, B7.8.

**Infrastructure** (build once):
- `sedov_two_fluid.py` (B7.1) and the injection extension of `pfrommer_riemann_solver.py`
  (B7.4, B8.1, B8.5).
- A test-side helper: run `find_shocks_pfrommer` + `inject_crs_at_shocks` on saved states and
  return the rate, the finder's dissipation and the limiter counters (B7.3, B7.7, B8.4).
- A self-similar initial-condition builder on the Cartesian grid (B7.2, B7.5, B7.6).
- Digitized KR13 / Dubois App. B data (B8.3) and the 1D spherical reference (B8.4).
- Shared shock-position fit (sub-cell peak, `R ∝ t^{2/5}` fit).

**Per-test conventions** (as in the existing suite and the Phase A plan):
- One pytest file per item and topic, with figures in `pics/07_sedov_taylor/` and
  `pics/08_dsa_mach_dependence/` that overlay the reference and state the measured numbers in
  the title. Rasterize scatter plots with more than 1e5 points.
- Docstrings say what physics each gate checks and where its threshold comes from.
- `@pytest.mark.slow` for 3D runs. Run every test on a free GPU, UUID-pinned, including the 1D
  ones. Raw data goes to `/export/scratch/nknoell`.

---

## 11. Decisions for the user

- **DB1: efficiency convention (Y2).** Apply KR13's `eta` to the kinetic flux, as KR13 define
  it (F1 `KINETIC`; strong-shock CR yield about 1.8x higher)? Or keep the dissipated-flux
  convention and renormalize as Dubois et al. (2019)? Or keep it as is and document that
  "KR13" means `delta(M) eta_KR13`?
- **DB2: transport in the item-7/8 gates (Y3).** Proposal: hard gates at `v_red = 0` against
  the analytic references; the production transport (`v_red`, relaxation, `kappa`) as tracked
  variants.
- **DB3: obliquity-dependent efficiency.** Implement `zeta(theta)` (Pais et al. 2018 / Dubois
  et al. 2019) for MHD runs before the emission science uses DSA? Without it, "CS14-like" is
  not CS14 for oblique shocks.
- **DB4: Mach bias.** Correct it in the finder (F4 a/b), or accept it and gate only resolution
  trends? B7.3 and B8.2 quantify what is at stake.
- **DB5: sub-critical threshold for the constant model.** `dsa_mach_min` = 1.3 (current, the
  finder default), 2.25 (Ryu et al. 2019) or 3 (Pfrommer et al. 2017)?
- **DB6: item-7 test IC.** Replace the point explosion by the self-similar initial state (B7.2)
  as the gated case, keeping the point explosion as a tracked smoke test?

---

## 12. Bibliography (verify volumes and pages on ADS before citing in a paper)

**Directly used for the tests above**
- Kang, H. & Ryu, D. (2013), ApJ 764, 95. DSA at cosmological shocks; `eta(M_s)` and
  `delta(M_s)` relative to the kinetic flux (eq. 8, Fig. 4); `eta ≈ 0.2` for M ≳ 10.
  arXiv:1212.3246.
- Caprioli, D. & Spitkovsky, A. (2014a), ApJ 783, 91. Hybrid simulations of ion acceleration:
  10-20% at quasi-parallel shocks, inefficient at quasi-perpendicular ones. arXiv:1310.2943.
- Pfrommer, C., Pakmor, R., Schaal, K., Simpson, C. M. & Springel, V. (2017), MNRAS 465, 4500.
  DSA in AREPO (Sec. 3.1.2, eqs. 39-44); shock tubes with acceleration (Sec. 4.1, Table 1,
  App. B/C); Sedov with acceleration (Sec. 4.2, eqs. 64-67, Figs. 2-3). arXiv:1604.07399.
- Pais, M., Pfrommer, C., Ehlert, K. & Pakmor, R. (2018), MNRAS 478, 5278. Obliquity-dependent
  acceleration `zeta(theta)` (eq. 7); `gamma_eff(zeta)` (eqs. 12, 15, 16); Sedov in a uniform
  field, ellipticity 0.25 ± 0.02. arXiv:1805.00128. *(Check the volume.)*
- Dubois, Y., Commerçon, B., Marcowith, A. & Brahimi, L. (2019), A&A 631, A121. Shock finder
  and CR injection in RAMSES; Sod Mach-recovery tests (Sec. 4.3.1); Pfrommer App. B
  reproduction (4.3.2); obliquity Sod (4.3.3); Sedov (4.4); KR13 table (App. B).
  arXiv:1907.04300. *(Check the volume.)*
- Weber, M., Thomas, T. & Pfrommer, C. (2023), MNRAS. Energy vs entropy formulation for CR
  hydrodynamics; with acceleration the energy formulation is more accurate for the shock speed.
  arXiv:2205.12288. *(Check the volume.)*
- Vazza, F., Brüggen, M., Wittor, D., Gheller, C., Eckert, D. & Stubbe, M. (2016), MNRAS 459,
  70. KR13 with the kinetic-flux convention; "CS14 ≈ 0.5 KR13" for quasi-parallel M ~ 5-10,
  and 0.15 for random obliquity. arXiv:1603.02688.
- Ryu, D., Kang, H. & Ha, J.-H. (2019), ApJ 883, 60. DSA at weak quasi-parallel ICM shocks; no
  proton DSA below M ≈ 2.25; `eta` ~ 1e-3 to 1e-2 for 2.25-5. arXiv:1905.04476.
- Ryu, D., Kang, H., Hallman, E. & Jones, T. W. (2003), ApJ 593, 599. Thermalization
  efficiency `delta(M)` (used by `calculate_thermal_energy_flux`).
- Kang, H. & Jones, T. W. (2007), Astropart. Phys. 28, 232. Earlier `eta(M)`, `delta(M)`
  (`eta → 0.55`).
- Kang, H. (2012), JKAS 45, 127. DSA with magnetic field amplification and Alfvénic drift;
  `eta ≈ 0.12` for `beta_P ~ 1`. arXiv:1209.5203.
- Chevalier, R. A. (1983), ApJ 272, 765. Self-similar blast waves with CR pressure (two-fluid,
  constant post-shock CR fraction). *(Not verified online for this plan; check on ADS.)*
- Sedov, L. I. (1959), *Similarity and Dimensional Methods in Mechanics*. `xi_0 = 1.1517`
  (`gamma = 5/3`).
- Schaal, K. & Springel, V. (2015), MNRAS 446, 3992. Shock finding on a moving mesh, Mach
  number tests. arXiv:1407.4117.

**Background**
- Diesing, R. & Caprioli, D. (2018), PRL 121, 091101. CR acceleration (~10%) boosts SNR
  momentum deposition 2-3x in the radiative stage. arXiv:1804.09731. Relevant once cooling and
  DSA are combined (items 11/12).
- Girichidis, P. et al. (2016), ApJL 816, L19. SN CR injection fraction 0.1 (B7.8).
- Pfrommer, C., Springel, V., Enßlin, T. A. & Jubelgas, M. (2006), MNRAS 367, 113. Composite
  Riemann problem (basis of `pfrommer_riemann_solver.py`).
- Phase A plan bibliography (Gupta et al. 2021; Semenov et al. 2021; Kudoh & Hanawa 2016) for
  the shock-partition background.

---

## Appendix A: self-similar two-fluid Sedov solver (sketch used for the *(calc.)* numbers)

Similarity variables: `lambda = r/R(t)`, `R = xi_0 (E t^2/rho_0)^{1/5}`, `V_s = (2/5) R/t`,
`u = V_s U`, `rho = rho_0 G`, `P_i = rho_0 V_s^2 Pi_i`. With `a = U - lambda` the ODEs are:
- continuity: `G U' + a G' = -2 U G/lambda`;
- momentum: `a U' + (Pi_th' + Pi_cr')/G = 1.5 U`;
- entropy, for each `i`: `-gamma_i a G'/G + a Pi_i'/Pi_i = 3`.

Strong-shock boundary at `lambda = 1`:
- `X = zeta/(2(1-zeta))`, `f = X/(1+X)`;
- `k = 2.5 (1-f) + 4 f`, `x = 2k - 1`;
- `U = 1 - 1/x`, `G = x`, `Pi = 1 - 1/x` split as `(1-f, f)`.

Energy normalization: `xi_0^5 = 25/(16 π I)` with
`I = ∫ (G U^2/2 + 1.5 Pi_th + 3 Pi_cr) lambda^2 dlambda`.

```python
import numpy as np
from scipy.integrate import solve_ivp
gth, gcr = 5/3, 4/3

def rhs(lam, y):
    U, G, Pt, Pc = y
    a = U - lam
    A = np.array([[G, a, 0, 0], [a, 0, 1/G, 1/G],
                  [0, -gth*a/G, a/Pt, 0], [0, -gcr*a/G, 0, a/Pc]])
    return np.linalg.solve(A, [-2*U*G/lam, 1.5*U, 3.0, 3.0])

def solve(zeta):                      # zeta > 0 (use 1e-6 for the classic case)
    X = zeta/(2*(1-zeta)); f = X/(1+X); x = 2*(2.5*(1-f) + 4*f) - 1; P = 1 - 1/x
    s = solve_ivp(rhs, [1.0, 0.03], [1-1/x, x, P*(1-f), P*f], method="LSODA",
                  rtol=1e-8, atol=1e-12)
    lam, (U, G, Pt, Pc) = s.t, s.y
    I = [-np.trapezoid(w*lam**2, lam) for w in (0.5*G*U**2, 1.5*Pt, 3.0*Pc)]
    xi0 = (25/(16*np.pi*sum(I)))**0.2
    e_int2 = (1 - 1/x)*(1.5*(1-f) + 3*f)              # e_int,2 / (rho_0 V_s^2)
    rate = zeta*4*np.pi*xi0**5*0.4**3*e_int2/x         # dE_inj/dt in units of E/t
    return xi0, np.array(I)/sum(I), rate
```

The Lagrangian origin of a shell is `r_0/R = (m(lambda)/M)^{1/3}`, with
`m(lambda) = 3 ∫_0^lambda G l^2 dl`. The B7.1 module should integrate to smaller `lambda`,
handle `zeta = 0` exactly, and return profiles for overlays.

---

*Basis: committed scripts and figures at `c2b197a` plus the uncommitted working tree
(2026-10-13). Numbers marked (fig.) were read off the SVGs by eye. (log) numbers are from the
module logs, (calc.) numbers from Appendix A. Re-measure them when the tests are next run, and
record them in the baseline registry.*
