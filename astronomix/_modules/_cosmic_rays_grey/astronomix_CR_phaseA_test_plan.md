# Cosmic Rays in `astronomix`: Phase A Unit/Analytic Test Plan (Review and Extension)

**Goal.** Strengthen the Phase A unit/analytic tests of `astronomix_CR_implementation_plan.md`
(Sec. 4, items 1, 2, 3, 4, 6). The plan has two parts, applied to each item in turn:

1. **Review.** Go through the existing pytest scripts and the figures they produce, and check
   the results against what the physics and the literature say should happen.
2. **Extend.** Add standard tests from the literature that check the same implementation
   from a different angle, and write the most promising complementary tests down concretely enough to implement.
3. **Fix.** If the test results are in disagreement with the expected physical outcome, propose a sensible fix for the affected
   implementation (if possible, literature backed). Check back with the user before impleneting fixes to the existing schemes.

**Scope.** Only Phase A items 1, 2, 3, 4 and 6. Item 5 (1D streaming) is excluded on request.
The FD-vs-AD gradient check (`cr_gradient_check.py`, plan item 16) was part of the kickoff
prompt but is not a Phase A ladder item, so it is also excluded.

**Basis of this review (2026-10-07, branch `feature/cosmic-rays-grey`, HEAD `79b1d77`).**
- Scripts in `pytests/cosmic_rays_grey/`: `cr_advection.py`, `cr_adiabatic_compression.py`,
  `cr_anisotropic_diffusion_oblique.py`, `cr_anisotropic_ring.py`, `cr_diffusion_rate.py`,
  `cr_isotropic_diffusion_convergence.py` and `cr_shock_tube.py`.
- The reference solver `astronomix/test_setups/reference_solutions/pfrommer_riemann_solver.py`.
- `DESIGN.md` and `PROGRESS.md`.
- The **committed** figures in `pytests/cosmic_rays_grey/pics/`. The tests were not re-run
  for this review. Numbers marked *(fig.)* were read off the committed SVGs, either from the
  plotted data paths or by eye.

**Status legend.**
- ✅ Correct physics, and the gate is meaningful.
- ⚠️ The result passes, but the physics check is incomplete or the gate is weak.
- ❌ The result contradicts the physical expectation.

---

## 0. Cross-cutting findings (apply to every item)

| # | Finding | Consequence | Action |
|---|---|---|---|
| X1 | **Stale calibration numbers in docstrings.** Item 1's docstring gives L2 0.011 and peak loss 0.06; the figure gives 0.0060 and 0.037. Item 4's docstring gives errors 3.6e-3, 9.2e-4, 2.7e-4, 1.4e-4 with pairwise orders 1.95/1.78/0.94; the figure gives 1.7e-3, 5.4e-4, 1.9e-4, 1.3e-4 with pairwise orders 1.66/1.47/0.59. T1b and T2b quote numbers from after fix step 2 or 3, but the figures show the code after step 4. | A reader cannot tell which scheme a gate was calibrated against. | Store measured baselines in one place, either a `baselines.json` next to `pics/` written by each test or a table at the end of this file. Docstrings then refer to it. |
| X2 | **Loose single-number gates.** Item 1's gates are 8x (L2) and 4x (peak) above the measured values. Item 6's mean-absolute-error gate is 3-10x above the measured errors and averages over the whole box. | A regression that makes the result several times worse still passes. | Gate on (a) convergence order, (b) error at the finest N within about 1.5x of the baseline, and (c) physically localized quantities such as plateau values, front positions and invariants. |
| X3 | **Mostly one resolution.** Items 1, 2, 6 and the oblique item-3 test run at a single N. | The order of accuracy is unknown, and a model error cannot be told apart from a discretization error. | Run every analytic test at 3-4 resolutions (doubling N each time). |
| X4 | **The production code path is not exercised in items 1, 2 and 6.** These run with `diffusive_relaxation=False`, so the realizability cap, the monotonicity guard and the optical-depth-reduced HLL speed are all inactive. Production (M7) runs with relaxation on, the cap on and the **guard off** (DESIGN.md, "Known limitation", 2026-10-06). The ring gates (3a/3b) pass only with the guard **on**. | Tests and production use different code paths, so the green results do not cover production. | Every item gets a variant in the production configuration (relaxation on, cap on, guard on and off). Where that variant is expected to fail, it is tracked as a *measured metric*, not a hard gate. |
| X5 | **Mixed precision.** The ring and diffusion-rate tests use float64. The others use the float32 default. | Small invariants (round-off conservation, 1e-4-level floors) cannot be checked in float32. | Run analytic convergence tests in float64. Add one float32 smoke run per item. |
| X6 | **Gas feedback contaminates transport tests.** In the item-4 and T1 bumps, the CR pressure sets the gas in motion, and `-P_cr div v` work removes about 1e-5 of the CR energy (DESIGN.md, follow-up audit). | Pure transport errors are mixed with physical coupling. | Reuse the ring tests' trick of a heavy, static gas (`rho = 1e4`, CR values times `E_SCALE = 1e-7`, a gate on `max|u| t < 0.1 dx`) for pure-transport tests. |
| X7 | **FV only.** The plan asks for both schemes, but CR transport is FV-only (DESIGN.md, "FD limitation"). | — | Out of scope here. Plan item 20 (FV vs FD consistency) will reuse the tests below once FD carries `e_cr`/`F_cr`. |

---

## 1. Ladder item 1: pure CR advection

> Plan: "Pure CR advection (translation of an `e_cr` blob): shape/amplitude preserved."

### 1.1 What exists

`cr_advection.py::test_cr_advection`, with figure `pics/01_advection/cr_advection_test.svg`.
- **Setup.** 1D FV, N = 512, periodic boundaries. Gas at rest: `rho = 1`, `P = 1`, `u = 0`.
  `gamma_cr = 4/3`, `v_red = 1`, `diffusive_relaxation = False`.
- **Initial state.** Gaussian `e_cr` with amplitude 1e-3, `sigma = 0.05` (25.6 cells per
  sigma), on a CR-free background. `F_cr = lambda e_cr`, with
  `lambda = v_red sqrt(gamma_cr - 1) = 0.577`.
- **Run.** `t_end = 1/lambda = 1.732`, exactly one box crossing.
- **Gates.** L2/amp < 0.05 and peak loss < 0.15, plus a NaN check.

### 1.2 Physical expectation

- With `u = 0` and no relaxation, the CR subsystem is the linear hyperbolic system
  `d_t e + d_x F = 0`, `d_t F + v_red^2 (gamma_cr - 1) d_x e = 0`. Its characteristic speeds are
  `±v_red sqrt(gamma_cr - 1) = ±v_red/sqrt(3)`: the two-moment closure with Eddington factor
  1/3, in the zero-scattering limit (Jiang & Oh 2018, Sec. 2-3).
- A pure right-moving eigenmode translates rigidly. After one period the exact solution equals
  the initial condition.
- The flux keeps its eigenvector, `F/e = lambda` everywhere.
- With a CR-free background, the feedback onto the gas is second order in the amplitude.

### 1.3 Step-by-step check of script and figure

1. **Eigenmode and speed: ✅.** The residual is almost symmetric about x = 0.5 *(fig.)*: a
   central dip of -3.8% of amp and two side lobes of +2% at x ≈ 0.47 and 0.54. A phase error
   would show up as an antisymmetric residual. This one is not antisymmetric, so the
   propagation speed is right, consistent with PROGRESS.md's centroid check (lambda matches to
   5 decimal places).
2. **Error shape: ✅, as expected for the scheme.** The dip-plus-lobes pattern is the usual
   clipping of a smooth extremum by a TVD limiter (minmod) in a second-order MUSCL-HLL scheme.
   L2/amp = 0.0060 and peak loss = 3.7% *(fig.)* are plausible at 25 cells per sigma over 512
   cells of travel.
3. **What "advection" means here: ⚠️.** The plan's "translation of an `e_cr` blob" is CR
   *advection by the gas* (tightly coupled CRs). The test instead propagates a free CR wave
   through gas **at rest**. The `u_n e_cr` / `u_n F_cr` advective fluxes, added later in the
   item-2 session (DESIGN.md, "Correction"), are therefore never tested in isolation. The
   Galilean run of the 2026-10-05 audit (u = 0/1/3, D/kappa = 1.0042/1.0051/1.0069,
   centroid at `u t` to 1e-6) was never committed. DESIGN.md says so: "Not covered by any
   committed test."
4. **Missing checks: ⚠️.**
   - `F_cr/e_cr = lambda` is not checked at `t_end`.
   - Only the right-moving family is tested.
   - Only 1D.
   - Only one resolution, so no convergence order.
   - The gates are loose (X2), and the docstring numbers are stale (X1).
5. **Production path: ⚠️.** With relaxation off, the cap, the guard and the `R(tau)` HLL
   speed are all bypassed (X4).

**Verdict: ✅ for the transport eigenstructure. ⚠️ as a test of the plan's item 1.** Advection
by the gas, pressure balance at CR/thermal contacts, and multi-D are untested.

### 1.4 Complementary tests from the literature

**A1.1 Passive CR blob advected by the gas.** This is the test the plan actually asks for.
- *Reference.* The standard scalar-advection test (e.g. Stone et al. 2008, Athena test suite;
  Toro 2009). Pfrommer et al. (2017) treat advective CR transport as the baseline transport
  mode.
- *Setup.*
  - 1D and 2D (diagonal) periodic box, uniform `u0` (1D: `u0 = 1`; 2D: `(1, 1)`), uniform
    `rho` and `P_th`.
  - `e_cr` Gaussian, plus a top-hat case, of small amplitude. Use the heavy-gas trick (X6) so
    the CRs stay passive.
  - Run (a): `v_red = 0`, so `F_cr ≡ 0` exactly, as in item 6. Run (b): relaxation on with
    `kappa → 0`.
  - One crossing time.
- *Expectation.* `e_cr(x, t) = e_cr(x - u0 t, 0)`. For the Gaussian: second-order L1 convergence
  away from extrema, and a global order of at least about 1.5 with minmod. For the top-hat: no
  new extrema (TVD) and L1 order of about 0.5-0.67.
- *Gates.*
  - L1 convergence order over N = 64 to 512.
  - Centroid at `u0 t` to better than 1e-3 dx.
  - `min e_cr` at or above the initial minimum.
  - `max|F_cr| = 0` in run (a).
- *Catches.* Errors in the `u_n e_cr` / `u_n F_cr` flux pieces, and in the Riemann wave speed
  `|u| + max(c, v_cr)` when advection dominates.

**A1.2 Diffusion in a moving fluid (Galilean invariance).**
- *Reference.* Jiang & Oh (2018), Sec. 4.1.4, "moving fluid": the analytic diffusion solution
  with `x → x - v t`. The JO18 two-moment scheme requires `V_m ≫ |v|` (their Sec. 2 and 5.2).
- *Setup.* Commit the 2026-10-05 audit run: 1D, `kappa = 0.02`, `v_red = 8`, N = 256/512,
  `u = 0, 1, 3`. Extend it as DESIGN.md step 5a planned, to `u = 1, 2, 3 × v_red sqrt(gamma_cr - 1)`,
  i.e. above the CR signal speed.
- *Expectation.* D/kappa and the centroid drift are the same as at `u = 0`. Because our `F_cr`
  is the flux relative to the gas, the `V_m ≫ |v|` restriction of JO18 should not apply. This
  test is exactly what checks that claim, and M7's hottest gas exceeds `v_red`.
- *Gates.*
  - `|D(u)/D(0) - 1| < 0.01`.
  - Centroid `x_c(t) - u t` constant to within 1e-6 (relative).
  - Monotone convergence of D/kappa in N at every u.

**A1.3 CR-thermal contact discontinuity in pressure balance.**
- *Reference.* Thomas, Pfrommer & Pakmor (2021), Sec. 4.3, "magnetized contact
  discontinuity". The classic multi-component pressure-oscillation problem (Abgrall 1996;
  Karni 1994).
- *Setup (TPP21).*
  - Domain x ∈ [-0.5, 0.5], 256 cells. `rho = 1` on both sides, `u = 0`, and `B = e_y`
    (perpendicular; also run hydro-only).
  - `[P_th, P_cr]` = `[1/8, 7/8]` on the left and `[3/4, 1/4]` on the right, so
    `P_th + P_cr = 1` everywhere.
  - `t = 1`. Run (a) is static. Run (b) is advected with `u0 = 1` in a periodic box, one crossing.
  - CR transport: `v_red = 0` (exact pressure balance must hold). Then `v_red = 10` with
    relaxation (TPP21's case). With diffusion, the CR side of the jump smooths out, and
    `P_th` must follow so the total stays in balance.
- *Expectation.* The total pressure `P_th + P_cr` stays at 1 and `u` stays at 0 (run a) or
  `u0` (run b) to round-off or truncation level. TPP21 show that a "naive" Riemann treatment
  smears the contact spuriously.
- *Gates.*
  - `max|P_th + P_cr - 1| < 1e-3` for `v_red = 0`, with convergence in N.
  - `max|u - u0| < 1e-3`.
  - Contact width growth matches the scalar-advection value (A1.1).
- *Catches.* `e_cr` is evolved separately from the gas, with a non-conservative coupling. This
  is the typical setup that produces pressure and velocity blips at contacts. The existing
  item-6 test only touches this through a single contact, averaged over the box.

**A1.4 Convergence and eigenvector version of the existing free-wave test.**
- *Reference.* The linear-wave convergence method of Stone et al. (2008).
- *Setup.*
  - The existing test at N = 64 to 1024, for the right-moving family, the left-moving
    family, and a superposition of the two.
  - A 2D plane-wave eigenmode along `(cos θ, sin θ)` at θ = 30° and 45°: wavelength equal to
    the box diagonal, periodic.
- *Expectation.* L1 order of about 2 for a sinusoidal eigenmode with no extrema clipping
  issue; about 1.5 for the Gaussian with minmod. `F/e = ±lambda` is preserved. The 2D
  error is independent of θ to within about 10%.
- *Gates.* Order at least 1.8 (sinusoid) and at least 1.4 (Gaussian). `max|F - lambda e|/amp` at
  most the L1 error. An anisotropy gate on the 2D error.

**A1.5 Free-streaming limit in the production configuration.**
- *Reference.* Telegrapher limit of the two-moment system (JO18, Sec. 3; TPP21, Sec. 4.5). The
  mode `e = e0 + δ cos(kx)` obeys `s^2 + ν s + c^2 k^2 = 0`, with `c^2 = (gamma_cr - 1) v_red^2`
  and `ν = (gamma_cr - 1) v_red^2 / kappa`.
- *Setup.* The A1.4 sinusoid with `diffusive_relaxation=True`, the cap on, and the guard on and
  off. Choose `kappa` so that `ν ≪ c k` (ballistic regime).
- *Expectation.* The wave propagates at `±c` (to `O(ν^2/c^2 k^2)`) and its amplitude decays as
  `exp(-ν t/2)`, which is exact for the linear system.
- *Gates.* The fitted decay rate matches `ν/2` within 2%, and the phase speed matches `c`
  within 1%. With the guard on, the fitted rate must not exceed the guard-off rate by more than
  2%: the guard must not dissipate smooth waves, because a sinusoid has `ψ` well below the
  onset.

---

## 2. Ladder item 2: adiabatic compression, `e_cr ∝ rho^gamma_cr`

### 2.1 What exists

`cr_adiabatic_compression.py::test_cr_adiabatic_compression`, with figure
`pics/02_adiabatic_compression/cr_adiabatic_compression_test.svg`.
- **Setup.** 1D FV, N = 512, open boundaries, `rho0 = 1`, `p0 = 0.01`, `e_cr0 = 1`
  (`P_cr = 1/3`, so the CRs dominate: `P_cr/P_th ≈ 33`). Velocity `v = -0.2 (x - 0.5)`,
  `v_red = 0.1`, relaxation off, `t = 1`.
- **Gate.** In the inner 80% of the box, the pointwise `e_cr/e_cr0` must match
  `(rho/rho0)^{4/3}` to within 5e-3. A sanity check requires more than 5% compression.

### 2.2 Physical expectation

- For a Lagrangian element, the first law gives `d e_cr/dt = -(e_cr + P_cr) div v`. So
  `K_cr ≡ P_cr rho^{-gamma_cr}` is constant along a fluid element and `e_cr ∝ rho^{gamma_cr}`
  in any dimension. This is the standard two-fluid CR hydrodynamics result (Drury & Völk 1981;
  Pfrommer et al. 2017, Sec. 2).
- For a *ballistic* homologous squeeze (pressure negligible), `rho(t) = rho0/(1 - α0 t)`, which
  gives 1.25 at t = 1, and `e_cr = e_cr0 (1 - α0 t)^{-gamma_cr}`, which gives 1.347.
- With dynamically important pressure, the combined sound speed is
  `c_eff^2 = (gamma P_th + gamma_cr P_cr)/rho`, about 0.68^2 here. That pressure decelerates
  the inflow.

### 2.3 Step-by-step check of script and figure

1. **Pointwise invariant: ✅.** The interior cells lie on the `gamma_cr` curve, and the wrong
   pre-fix exponent `gamma_cr - 1` is far off. The maximum relative error is 1.55e-3 *(fig.;
   the docstring says about 1e-3)*.
2. **Global dynamics: ⚠️, not compared with anything.** The central compression is
   `rho/rho0 ≈ 1.145` *(fig.)*, not the ballistic 1.25. That is physically expected because
   the CR pressure dominates and `c_eff t_end ≈ 0.68` exceeds the half-box. Pressure waves from
   the open boundaries therefore reach the center, and the flow is not homologous (the module
   docstring says this too). The *magnitude* of the deceleration, i.e. whether `-∇P_cr` has
   the right size, is not checked. The test only checks that `K_cr` is carried with the flow.
3. **Unexplained oscillations: ⚠️, possible ❌.** At x ≈ 0.28-0.36 and 0.64-0.72 *(fig.)*,
   both `rho` and `e_cr` show wave trains of about ±3% with wavelength about 0.02 (about 10
   cells), in phase. They do not break the invariant because the two oscillate together. But a
   minmod-limited second-order scheme should not ring like this behind smooth or weakly
   steepened compression fronts. Possible causes:
   - (a) the converging flow steepens into weak shocks, and the TVD scheme rings at the
     interaction with the open-boundary waves;
   - (b) the operator-split CR sources (`-∇P_cr`, `-P_cr div v`) are not TVD-compatible with
     the flux update;
   - (c) an open-boundary artefact.

   If these are weak shocks, the non-conservative `-P_cr div v` is also *expected* to break
   `K_cr` conservation there (Gupta et al. 2021; Semenov et al. 2021). The 1.55e-3 maximum
   error may be exactly that.
4. **Coverage: ⚠️.**
   - Compression only, no expansion.
   - 1D only. The multi-D `div v` sum and the `u_n e_cr` flux in y and z are untested here.
   - `gamma_cr = 4/3` only, so a hard-coded 4/3 would not be caught.
   - Relaxation off (X4).

**Verdict: ✅ for the local adiabatic invariant. ⚠️ for the dynamics.** The oscillations are
an open diagnostic item.

### 2.4 Complementary tests from the literature

**A2.1 Ballistic homologous compression and expansion with an analytic time series.**
- *Reference.* The analytic homologous flow, i.e. the adiabatic invariant in Lagrangian form
  (as above). The Zel'dovich-pancake adiabatic stage of Semenov et al. (2021) checks the same
  invariant in a cosmological setting.
- *Setup.*
  - 1D and 3D. `p0 = e_cr0 = 1e-6 rho0 α0^2 L^2`, so pressure is negligible over
    `t_end = 1/(2α0)`.
  - `v = -α0 (x - x_c)`, with expansion `α0 < 0` as the second case.
  - Open boundaries. The boundary perturbations travel only `c t ≪ L`, so the core stays
    homologous.
  - `gamma_cr ∈ {4/3, 1.4}`.
- *Expectation.* In the core, `rho = rho0 (1 - α0 t)^{-d}` and
  `e_cr = e_cr0 (1 - α0 t)^{-d gamma_cr}` (d = dimension). `K_cr` is constant.
- *Gates.*
  - Core `rho(t)` and `e_cr(t)` match the analytic values to within 1e-3, with
    second-order convergence in the time-series error.
  - Expansion does not hit the `minimum_e_cr` floor.
  - `gamma_cr = 1.4` gives exponent 1.4.

**A2.2 CR-modified linear sound wave (tightly coupled).**
- *Reference.*
  - The effective sound speed `c_eff^2 = (gamma P_th + gamma_cr P_cr)/rho` of the composite
    fluid (Pfrommer et al. 2006, App. B; Pfrommer et al. 2017). The code uses the same
    expression in `grey_cr_fast_speed`.
  - The linear-wave convergence method of Stone et al. (2008).
- *Setup.*
  - Periodic 1D and 2D (oblique). `v_red = 0`, so `F_cr ≡ 0`.
  - Eigenvector: `δrho/rho = ε`, `δu = c_eff ε`, `δP_th = gamma P_th ε`,
    `δe_cr = gamma_cr e_cr ε`, with `ε = 1e-6`.
  - Scan `X_cr = P_cr/P_th ∈ {0, 0.5, 2, 10}`. Run one period, N = 32 to 512.
- *Expectation.* The wave returns to its initial state after one period `L/c_eff`, with L1
  order 2. A wrong coupling strength appears as a phase error that does not converge away.
- *Gates.*
  - L1 order ≥ 1.8 for every variable.
  - Fitted phase speed matches `c_eff` within 1e-4 at the finest N.
  - The eigenvector ratio `δe_cr/(e_cr δrho/rho)` matches `gamma_cr` within 1e-3.
- *Catches.* A wrong `-∇P_cr` or `-P_cr div v` magnitude, and errors in splitting the sources.
  This is the dynamic check that the existing item-2 test lacks.

**A2.3 Smooth-flow equivalence: CR-dominated fluid vs `gamma = 4/3` hydro.**
- *Reference.* An exact identity. For smooth flow with `v_red = 0` and `P_th → 0`, the
  two-fluid system is identical to single-fluid hydro with `gamma = gamma_cr`. In
  Pfrommer et al. (2006, App. B) the composite EOS reduces to a single power law, and the
  repository's reference solver was validated against this limit, but not the code.
- *Setup.*
  - (a) The existing item-2 squeeze, with `P_th = 1e-6 P_cr`.
  - (b) A large-amplitude simple wave, stopped before the shock-formation time.
  - Compare against a pure-hydro run with `gamma = 4/3` and `P = P_cr` from the same
    initial conditions.
- *Expectation.* The two runs agree to truncation error, with second-order convergence of the
  difference.
- *Gate.* The difference converges at an order of at least 1.8.
- *Catches.* This test tells which origin the item-2 oscillations (Sec. 2.3, step 3) have:
  - If the hydro run shows the same ringing, it comes from the FV scheme or the open boundaries.
  - If not, it comes from the CR coupling.

**A2.4 Two-moment linear dispersion relation (sound waves + CR waves + diffusion).**
- *Reference.* Thomas, Pfrommer & Pakmor (2021), Sec. 4.5, the "telegrapher's equation" test
  with damping rate and phase velocity vs k. Background: `rho = 1`,
  `P_th = 0.01/(gamma - 1)`, `P_cr = 0.01/(gamma_cr - 1)`, `c = 1`, `kappa = 1/3`. They scan
  `k = 2π · 2^n` and also `kappa ∈ [1e-4, 1]` at fixed k. The transition is at
  `k_mfp ≈ c/(2 sqrt(3) kappa)`.
- *Our reference.* Linearize **our** equations. TPP21's flux is lab-frame and includes
  Alfvén-wave physics, so their curves serve only as qualitative cross-checks. For a periodic
  1D state `(δrho, δu, δP_th, δe, δF) ∝ exp(ikx + st)`, with `u0 = 0`:

  ```
  s δrho + i k rho0 δu = 0
  s rho0 δu + i k δP_th + i k (gamma_cr - 1) δe = 0
  s δP_th + i k gamma P_th0 δu = 0
  s δe + i k gamma_cr e0 δu + i k δF = 0
  s δF + i k (gamma_cr - 1) v_red^2 δe = -ν δF
  ```

  Here `ν = (gamma_cr - 1) v_red^2 / kappa`. The roots `s(k)` are the eigenvalues of a 5x5
  matrix; compute them with numpy. Check the limits:
  - `kappa → 0`: `c_eff` sound waves.
  - `kappa → ∞`: gas sound waves plus undamped CR waves at `±v_red/sqrt(3)`.
  - In between: damped modified sound waves, and the diffusive CR mode
    `s ≈ -kappa k^2`.
- *Setup.* Excite a single eigenmode at amplitude 1e-6 and fit `Re s` and `Im s` from the
  time series. Use N = 4096 (as TPP21) or fewer. Scan k and kappa.
- *Gates.* For `k dx ≤ 2π/16`, the fitted damping rate and phase speed match within 2%. Above
  that, record the deviation as numerical dissipation (as TPP21 do).
- *Catches.* The coupling of item 2 and the relaxation of item 4 at the same time, in the
  regime between tightly coupled and free streaming. This regime is where production runs
  operate, and no current test covers it.

---

## 3. Ladder item 3: anisotropic diffusion oblique to the grid (Sharma & Hammett 2007)

### 3.1 What exists

Three files cover this item.

1. **`cr_anisotropic_diffusion_oblique.py`** (the committed item-3 gate), figure
   `pics/03_anisotropic_diffusion/cr_anisotropic_diffusion_oblique_test.svg`.
   - 2D MHD, 128², uniform B at 30°, **projection only (no relaxation)**.
   - Bump with `sigma = 0.03`, `v_red = 1`, `t = 0.3`.
   - Gate: the ratio of perpendicular to parallel second-moment growth is below 0.1
     (measured 0.019; the isotropic control gives 1.0).
2. **`cr_anisotropic_ring.py`**, figures `pics/03_anisotropic_diffusion/cr_ring_*.svg`.
   - 3a: the Sharma & Hammett (2007) Sec. 7.1 ring (`kappa_par = 0.01`, `t = 200`, N = 50 to
     400). Gates: undershoot ≤ 1e-3 of the jump, and `kappa_perp,num/kappa_par` decreasing
     with N.
   - 3b: S&H Fig. 7 positivity variant.
   - 3c: Jiang & Oh (2018) Sec. 4.1.5 against their eq. 28. Gate: L1 decreasing with N.
   - "Cap alone": a documented failing configuration. Note: this test is collected by pytest
     and fails by design. Mark it `xfail` so it does not count as a CI failure.
3. **`cr_diffusion_rate.py`**, T2a-f and T3 (2D/3D measured `D_par`, `D_perp`), figures
   `pics/03_anisotropic_diffusion/cr_diffusion_rate_{2d,3d,out_of_plane*}_test.svg`.

### 3.2 Physical expectation

- With `kappa_perp = 0`, energy moves only along field lines: the flux is
  `F = -kappa_par b (b·∇e)`. Every field line evolves as an independent 1D diffusion problem.
- **Ring at t = 200.** The ring relaxes to 10 + 2/12 = 10.1667; outside it stays 10. Limited
  schemes keep the minimum at or above the initial minimum (S&H 2007). Published
  `kappa_perp,num/kappa_par` at N = 50/100/200/400:
  - asymmetric MC (FLASH): 0.0127 / 0.0040 / 0.0015 / 6.8e-4;
  - symmetric MC: 0.0072 / 8.4e-4 / 2e-4 / 6.5e-5;
  - van Leer: 0.0238 / 0.0104 / 0.0038 / 0.0013.
- **Ring at early times.** Compare with the erfc solution (JO18 eq. 28; Pakmor et al. 2016).
  Published L1 orders: JO18 report `N^-0.7` (t = 0.1 and 0.26); Pakmor et al. (2016) report
  `N^-0.55` at t = 10 and `N^-0.7` at t = 200.
- **Uniform oblique field, projection only.** Two pulses travel along ±b at
  `v_red/sqrt(3) = 0.577`, i.e. 0.173 from the center at t = 0.3.

### 3.3 Step-by-step check of scripts and figures

1. **Oblique projection test: ✅ physically.**
   - The isotropic ring radius *(fig.)* is about 0.17, the expected 0.173.
   - The anisotropic pulses sit at (0.35, 0.415) and (0.65, 0.585) *(fig.)*. That is a distance
     of 0.172 along the 30° line, as expected.
   - Leak ratio 0.019, against 1.0 for the isotropic control.

   **⚠️, but it is a wave test, not a diffusion test.** The diffusion question S&H address
   (numerical cross-field diffusion of a *diffusing* field) is answered by the ring and T2/T3,
   not by this file. DESIGN.md (literature check) already concludes that "step 3 is what closes
   item 3 as planned". The ladder item-3 gate should point at the ring tests.
2. **Ring 3a, large perpendicular numerical diffusion: ⚠️, ❌ at N ≤ 100.**
   - Measured `kappa_perp,num/kappa_par` = 0.041 / 0.020 / 0.0086 / 0.0033 at N = 50/100/200/400.
     That is 3.2-4.9x the asymmetric-MC row and 1.7-2.6x the van Leer row. It is the worst of
     all published schemes in the comparison at every N. The order is 1.0-1.4; S&H's MC order
     is about 1.5-2.
   - At N = 50 and 100 the ring has smeared into a filled disc *(fig.)*. The perpendicular
     spreading length `sqrt(2 kappa_perp,num kappa_par t)` is about 0.40 (N = 50) and 0.28
     (N = 100), larger than the ring width of 0.2. Physically the result is wrong at those
     resolutions.
   - At N = 400, 43% of the ring excess has still left the ring by t = 200.
   - The gate only asks for "decreasing with N", so a much worse scheme would also pass.
3. **Ring 3a/3b monotonicity: ✅ with the guard on, but production runs guard off.**
   - Undershoots are 0 / 1.1e-5 / 0 / 0 of the jump.
   - 3b *(fig.)*: the N = 100 minimum dips to 0.09989 at t ≈ 0.1; at N = 200 it stays exactly
     at 0.1.
   - With the guard off (production, X4), 3a undershoots by 2-3% and 3b goes negative (-0.09
     and -0.13). The committed gates do not describe the production scheme.
4. **Ring 3c vs JO18 eq. 28: ⚠️, minor.**
   - The numerical peak is about 10.5 against an exact value of about 10.6 at N = 256
     *(fig.)*. The radial edges are smeared over several cells.
   - The L1 order is about 0.5, against JO18's 0.7 and Pakmor's 0.55. Somewhat below JO18's
     own scheme.
   - The gate is again only "decreasing".
5. **T2/T3 rates: ✅ for the parallel rate, ⚠️ for the perpendicular one.**
   - `D_par/kappa` = 1.126 / 1.052 / 1.016 at N = 64/128/256 (0°), and about 1.015 at 30° and
     45° *(fig.)*. It converges to 1.
   - `D_perp,num/kappa` at 30° and 45°: about 0.05 → 0.025 → 0.011-0.014, first order. The
     grid-aligned value is about 9e-5, a physical floor from gas advection driven by the bump's
     own pressure.
   - 3D in-plane agrees with 2D to 7e-5 kappa. For the diagonal field, `D_perp ≈ D_par - 1`,
     i.e. an isotropic numerical diffusivity of about 0.005 `v_red dx`.
   - Physically important: at oblique angles, `D_perp,num ∝ v_red dx`, so it *grows* with the
     `v_red` that M7 now needs (1e4 km/s).
   - Docstring numbers for T2b are stale (X1).

**Verdict: ✅ for parallel transport. ⚠️/❌ for perpendicular numerical diffusion.** Its size
is above every published second-order scheme, and no gate bounds it. The production
configuration (guard off) is not gated.

### 3.4 Complementary tests from the literature

**A3.1 Sovinec steady-state test (direct measurement of `kappa_perp,num`).**
- *Reference.*
  - Sovinec et al. (2004, JCP).
  - Used for CR transport by Dubois & Commerçon (2016, RAMSES) and Pakmor et al. (2016,
    Sec. 3.4), who report that numerical perpendicular diffusion decreases "almost with
    second order".
- *Setup.*
  - Domain `[-0.5, 0.5]^2`, `e_cr = 0` on the boundary.
  - Source `Q = Q0 cos(πx) cos(πy)` with `Q0 = 2π^2`.
  - Field `B_x = cos(πx) sin(πy)`, `B_y = -sin(πx) cos(πy)`. The field lines are contours of
    `ψ = cos(πx) cos(πy)`, so `b·∇e = 0` for the exact solution.
  - `kappa_par = 1`; `kappa_perp = 0` or a small value. Run to steady state.
  - Alternative without Dirichlet boundaries: a periodic box `[-1, 1]^2` with a constant offset
    `e_off > Q0/(2π^2 kappa_perp)`. The steady solution is then `e = e_off + ψ Q0/(2π^2 kappa_perp)`.
- *Expectation.* `e(0, 0) = Q0 / (2π^2 (kappa_perp + kappa_perp,num))`, so
  `kappa_perp,num = 1/e(0,0)` for `kappa_perp = 0`.
- *Gates.* `kappa_perp,num/kappa_par` decreasing with an order of at least 1.5 over N = 32 to
  256, and an absolute value of at most a stated budget at N = 128 (see decision D1).
  Repeat with `kappa_perp/kappa_par = 1e-3`: `e(0,0)` must match the formula within 5%.
- *Needs.* A test-only `e_cr` source hook, e.g. a callable source term in
  `_time_integrator_sources`, or an injection callback. Also the heavy-static-gas trick (X6).
- *Why.* It is steady, needs no time integral of S&H's eq. 39, and gives one unambiguous
  number per N that compares directly with three published codes.

**A3.2 Anisotropic Gaussian: full-profile comparison with the analytic Green's function.**
- *Reference.* Pakmor et al. (2016), Sec. 3.1 (2D Gaussian, `kappa = 0.01`, t = 0.1 → 0.2,
  second order), extended to an oblique field as in Dubois & Commerçon (2016).
- *Setup.* Uniform `b` at θ ∈ {0°, 30°, 45°}, and `(1,1,1)/sqrt(3)` in 3D. `kappa_perp ∈ {0,
  0.01 kappa_par}`. Static heavy gas.
- *Expectation (2D).*
  `e = A sigma0^2/(sigma_par sigma_perp) exp(-s_par^2/(2 sigma_par^2) - s_perp^2/(2 sigma_perp^2))`,
  with `sigma_par^2 = sigma0^2 + 2 kappa_par t` and `sigma_perp^2 = sigma0^2 + 2 kappa_perp t`.
- *Gates.* L1 order of at least 1.8 against the full profile (not only second moments), at
  every θ.
- *Why.* T2 measures second moments only, which are blind to shape errors such as the square
  grid imprint visible on the pulses in the oblique-test figure, or skewness.

**A3.3 Absolute gates and early-time ring.**
- *Reference.* S&H (2007) Tables 1-4. Pakmor et al. (2016) at t = 10 (erfc solution, order 0.55).
- *Action.*
  - Add an absolute gate to 3a: `kappa_perp,num/kappa_par` at most the S&H van Leer value at
    each N, as a first target that the current scheme fails by 1.7-2.6x (decision D1).
  - Add the t = 10 erfc comparison with an order gate of at least 0.5.
  - Add an order gate of at least 0.5 to 3c (JO18: 0.7).

**A3.4 Production-configuration tracking (guard off).**
- *Action.* Run 3a/3b/3c and A3.1 with the guard off. Record the undershoot, `e_min` and
  `kappa_perp,num` as tracked metrics: an `xfail` with numbers, or a non-failing report written
  to the baselines (X1).
- *Why.* This makes the "known limitation" visible in CI. It also gives the future
  `v_red`-independent guard (DESIGN.md option "2") and the limited cross-field flux (option 2
  of the known limitation) a ready benchmark.

**A3.5 `v_red` dependence of the cross-field numerical diffusion.**
- *Reference.* JO18, Sec. 5.2: results must be checked for convergence in `V_m`.
- *Setup.* T2c (30°, 45°) and A3.1 at `v_red` = 4, 8, 16, 32 (fixed kappa, fixed N).
- *Expectation.* For the current scheme, `D_perp,num ∝ v_red dx` (Sec. 3.3, step 5). A scheme
  whose cross-field error is controlled by `kappa` would have `D_perp,num` independent of `v_red`.
- *Gate.* For now, record the measured slope `d ln D_perp,num / d ln v_red`. This quantifies
  the M7 risk directly: production `v_red` is 10x the earlier value.

---

## 4. Ladder item 4: isotropic diffusion vs the analytic Gaussian Green's function

### 4.1 What exists

1. **`cr_isotropic_diffusion_convergence.py`**, figure
   `pics/04_isotropic_diffusion/cr_isotropic_diffusion_convergence_test.svg`.
   - 1D, relaxation on, `v_red = 8`, `kappa = 0.02`, `sigma0 = 0.02`, `t = 0.24`,
     N = 128 to 1024.
   - Gates: L2 least-squares order of at least 1.3 over N = 128-512, and L2 below 2e-4 at
     N = 1024.
2. **`cr_diffusion_rate.py` T1a-c**, figure `pics/04_isotropic_diffusion/cr_diffusion_rate_1d_test.svg`.
   - T1a: D independent of dt (spread below 0.02).
   - T1b: D/kappa → 1 under refinement.
   - T1c: stiff regime, optical depth 7.4 per cell, `ν dt ≈ 3`.

### 4.2 Physical expectation

- **Diffusion limit.** `e(x,t) = A sigma0/sigma(t) exp(-(x-x0)^2/(2 sigma^2))`, with
  `sigma^2 = sigma0^2 + 2 kappa t`. The integral of `e` is conserved for a passive CR population.
- **Two-moment model.** With finite `v_red`, the model is a telegrapher equation:
  `e_tt + ν e_t = c^2 e_xx`, with `c^2 = (gamma_cr - 1) v_red^2` and `ν = c^2/kappa` (JO18
  Sec. 3; TPP21 Sec. 4.5). For a Fourier mode with `F(0) = 0`, the slow root is
  `s ≈ -kappa k^2 - kappa^2 k^4/ν`, and the slow-mode amplitude is `≈ 1 - kappa k^2/ν`.
  So the model's deviation from pure diffusion scales as `kappa k^2/ν ∝ v_red^-2`.
- **Scheme.** Second order in space for this smooth problem. D does not depend on dt (implicit
  per-stage relaxation).

### 4.3 Step-by-step check of scripts and figures

1. **Profile: ✅.** At N = 1024, the profile lies on the analytic curve *(fig.)*.
2. **Convergence: ⚠️.**
   - L2 = 1.69e-3, 5.4e-4, 1.94e-4, 1.29e-4 at N = 128/256/512/1024 *(fig., from the SVG data
     path)*. Pairwise orders 1.66 / 1.47 / 0.59; least-squares order 1.56 over the first three.
   - Fix step 4 lowered the coarse-grid errors (docstring: 3.6e-3 at N = 128). The model floor
     of about 1.2e-4 is now reached earlier, so the measured "order" mixes discretization with
     the model floor. The order-1.3 gate is therefore fragile: a further improvement at coarse
     N would *lower* the measured order.
3. **Floor scaling: ⚠️, inconsistent with theory.**
   - The docstring says the floor "halves when `v_red` doubles" (1.2e-4 → 6.7e-5).
   - Leading-order telegrapher theory (Sec. 4.2) predicts a factor of 4 (`∝ v_red^-2`).
   - A `v_red^-1` component points to another error source, possibly the HLL CR dissipation
     at intermediate optical depth, the initial `F = 0` transient, or gas coupling (X6).
   - Unexplained.
4. **T1 rates: ✅.**
   - T1a *(fig.)*: D/kappa = 1.0042 at all three dt.
   - T1b *(fig.)*: 1.0155 / 1.0042 / 1.0012, second order. The docstring's 1.029 / 1.0073 /
     1.0020 values are pre-step-4 (X1).
   - T1c: the stiff regime is now relaxation-unlimited.
5. **Coverage: ⚠️.**
   - The diffusion-vs-Green's-function test is 1D only. Multi-D isotropy (the amplitude
     `∝ (sigma0/sigma)^d`, diagonal symmetry) is only tested through anisotropic T2/T3.
   - There is no steady-state test of the fixed point `F = -kappa ∇e`.
   - Energy is not conserved to round-off because of gas coupling (X6).

**Verdict: ✅ for the diffusion rate and dt-independence. ⚠️ for the convergence
measurement** (floor-limited, fragile gate, unexplained `v_red^-1` scaling).

### 4.4 Complementary tests from the literature

**A4.1 Exact two-moment reference: Fourier/telegrapher solution of the discrete initial data.**
- *Reference.* Telegrapher dispersion (JO18 Sec. 3; TPP21 Sec. 4.5). DESIGN.md step 5a
  already uses the spectral-diffusion reference `e(k,t) = e(k,0) exp(-kappa k^2 t)` for M7.
- *Setup.*
  - FFT the discrete initial `e`, `F = 0`. Evolve every mode exactly with the 2x2 linear
    system `(e_k, F_k)` (eigenvalues `s± = (-ν ± sqrt(ν^2 - 4 c^2 k^2))/2`), then inverse FFT.
  - Use the existing item-4 parameters, plus a single-mode test `e = e0 + δ cos(kx)` with k
    across `k_mfp` (overdamped and underdamped).
  - Static heavy gas (X6), float64.
- *Expectation.* The model floor disappears from the error, so the error is purely
  discretization. Second-order convergence down to the finest N. The single-mode decay rate
  matches `s_slow(k)`.
- *Gates.*
  - L2 order of at least 1.8 over N = 128 to 2048 against the telegrapher reference.
  - Separately, the telegrapher-vs-diffusion difference scales as `v_red^-2`; this check
    closes Sec. 4.3, step 3.

**A4.2 `v_red` scan of the model floor.**
- *Setup.* The existing item-4 run at fixed N = 2048, with `v_red` = 4, 8, 16, 32 and
  `kappa` fixed.
- *Expectation.* The error against the diffusion Green's function scales as `v_red^-2`
  (Sec. 4.2). The error against A4.1's reference is independent of `v_red` at fixed N, apart
  from the HLL dissipation's dependence on `R(tau)`.
- *Gate.* Fitted slope `-2 ± 0.3`, or a documented explanation of a different slope.

**A4.3 Multi-D isotropic Gaussian.**
- *Reference.* Pakmor et al. (2016), Sec. 3.1: 2D, `kappa = 0.01`, Gaussian evolved from
  `t0 = 0.1` to `0.2`, second-order convergence reported.
- *Setup.* 2D and 3D, `anisotropic_transport=False`, relaxation on, N = 32 to 256. Also with
  the Gaussian center offset to a cell corner.
- *Expectation.* `e = A (sigma0/sigma)^d exp(-r^2/(2 sigma^2))`, rotationally symmetric.
- *Gates.*
  - L1 order of at least 1.8.
  - Ratio of the second moments along the diagonal and along the axes within 1e-3 of 1.
  - Amplitude exponent d.

**A4.4 Steady-state diffusion with a source (fixed-point test).**
- *Reference.* A standard steady-state check of a diffusion operator. It is the isotropic
  analogue of the Sovinec test (A3.1).
- *Setup.* Periodic 1D/2D, source `Q = Q0 sin(kx)`, static heavy gas, relaxation on. Run to
  steady state at `v_red` = 4, 8, 16.
- *Expectation.* `e = e_off + Q0/(kappa k^2) sin(kx)` and `F = -kappa ∇e`, *independent of
  `v_red` and dt* (the implicit relaxation has `F = -K ∇e` as an exact fixed point, DESIGN.md
  follow-up audit).
- *Gates.* Second-order convergence in N. The `v_red` and dt dependence of the steady state is
  below 1e-6 relative.
- *Needs.* The same source hook as A3.1.

**A4.5 Passive energy conservation.**
- *Setup.* The item-4 run with heavy static gas (X6), float64, periodic.
- *Gate.* `|∫e dV(t) - ∫e dV(0)| / ∫e dV(0) < 1e-12`. The current drop of 1.06e-5 is physical
  gas work and should disappear with this setup.

---

## 6. Ladder item 6: two-fluid CR-modified shock tube

### 6.1 What exists

`cr_shock_tube.py::test_cr_shock_tube`, with figure `pics/06_shock_tube/cr_shock_tube_test.svg`.
- **Setup.** Sod-like: `rho = 1 | 0.125`, `P_th = 0.6 | 0.06`, `P_cr = 0.4 | 0.04` (40% CR
  pressure on both sides). `gamma = 5/3`, `gamma_cr = 4/3`. N = 400, open boundaries, t = 0.2.
  `v_red = 0`, so `F_cr ≡ 0` exactly (tightly coupled).
- **Reference.** The new semi-analytic composite-EOS Riemann solver
  (`pfrommer_riemann_solver.py`). It was validated, with ad hoc scripts that are not committed,
  in the `P_cr = 0` and `gamma_cr = gamma` limits to about 2e-7.
- **Gates.** Mean absolute error below 1e-2 per variable (measured 1e-3 to 3e-3), and
  `max|F_cr| < 1e-6`.

### 6.2 Physical expectation

- **Reference solution.** The solution of Pfrommer, Springel, Enßlin & Jubelgas (2006,
  MNRAS 367, 113), App. B. *(The test and DESIGN.md cite "Pfrommer, Enßlin & Jubelgas 2006";
  fix the citation to the four-author paper.)*
  - The CRs are compressed **adiabatically through the shock**: `P_cr2/P_cr1 = (rho2/rho1)^{gamma_cr}`,
    i.e. `K_cr` is unchanged. The thermal gas takes up all the dissipation.
  - The contact discontinuity keeps `P_th + P_cr` and `u` continuous, while `P_th` and `P_cr`
    jump individually.
- **Known numerical issue.** The two-fluid system with a separate, non-conservative CR energy
  equation (`-P_cr div v`) has **no unique weak solution at shocks**. Gupta, Sharma & Mignone
  (2021) show that post-shock CR quantities depend on reconstruction, time stepping, CFL and
  discretization. Semenov, Kravtsov & Diemer (2021) find spurious CR energy generation at
  numerically resolved shocks: about 20% excess post-shock CR pressure for `gamma_cr = 4/3`
  in their strong-shock test. The error grows with the shock compression.
- **Fixes in the literature.** Each removes the ambiguity:
  - a conservative CR entropy or "CR number" variable `n ∝ e_cr^{3/4}` (Kudoh & Hanawa 2016,
    who reproduce Pfrommer et al. 2006's solution; Semenov et al. 2021; Gupta et al. 2021's
    "Et+Scr");
  - a prescribed shock energy partition (Gupta et al. 2021).
- **This code.** It evolves `e_cr` separately, with operator-split `-∇P_cr` / `-P_cr div v`
  sources (DESIGN.md). That is the class of scheme Gupta et al. find to be ambiguous.

### 6.3 Step-by-step check of script and figure

1. **Wave pattern: ✅.** Rarefaction head and tail, the plateau values `rho*` (0.45) and
   `u*` (0.88), the contact position (x ≈ 0.675) and the shock position (x ≈ 0.86) all match
   the reference *(fig.)*. The contact is smeared over about 3-4 cells, as usual for HLL with
   minmod.
2. **Post-shock partition: ⚠️, the signature of the expected non-conservative error.** On the
   shocked plateau (0.68 < x < 0.86) *(fig.)*:
   - `e_cr` ≈ 0.295 against the reference 0.290, about **+2%**;
   - `P_th` ≈ 0.200 against 0.203, about **-1.5%**.

   The sign is the one Semenov et al. (2021) and Gupta et al. (2021) predict: CRs gain energy at
   the expense of the gas inside the numerically smeared shock. For this weak, Sod-like shock
   the size is a few percent; it should grow with Mach number. **The mean absolute error over
   the box (1e-3 to 3e-3) hides it,** because the plateau covers only about 18% of the box. The
   test therefore cannot see the main known failure mode of the scheme class.
3. **Unknown convergence of the plateau error: ⚠️.** If the plateau error converges to a
   nonzero value as N → ∞, it is the non-uniqueness error, not truncation error. That decides
   whether a scheme change is needed before Phase B. DSA injection (items 7-8) adds CR energy
   *at exactly these shocks*, so a scheme-dependent baseline partition directly biases
   injection efficiencies.
4. **Contact in pressure balance: ✅ by eye.** No visible `P_th + P_cr` or `u` blip at
   x ≈ 0.675 *(fig.)*. Not quantified. See A1.3.
5. **Regime: ⚠️.** `v_red = 0` realizes Pfrommer's tightly coupled assumption exactly, but
   the production path (relaxation, cap, guard, `R(tau)` HLL) is never shock-tested (X4).
6. **Reference-solver validation: ⚠️.** The ad hoc limit checks (2e-7) are not committed, so a
   regression in the reference would go unnoticed. The solver has not been cross-checked against
   any published case.

**Verdict: ✅ for the wave structure. ⚠️ for the thermal/CR partition at the shock**, which
is the item's main physics content and is not gated.

### 6.4 Complementary tests from the literature

**A6.1 Post-shock CR-entropy diagnostic and convergence study.**
- *Reference.* Gupta, Sharma & Mignone (2021), Sec. 3-5 (method dependence of the post-shock
  CR entropy `P_cr/rho^{4/3}`). Semenov et al. (2021), shock-tube tests.
- *Setup.* The existing item-6 initial conditions, varying one thing at a time:
  - N = 100, 200, 400, 800, 1600, 3200;
  - `C_cfl` = 0.1, 0.4, 0.8;
  - limiter (minmod, MC, van Leer);
  - Riemann solver (HLL, HLLC).
- *Measure.*
  - The plateau-averaged `K_cr,2/K_cr,1 - 1` (exact value 0) and `P_th,2/P_th,2,ref - 1`.
  - The shock and contact positions against the reference (to within dx).
  - L1 order per variable (expected about 1 for discontinuous solutions; Toro 2009).
- *Gates.* Tracked first. Promote to hard gates after decision D2. A converging plateau error
  (`→ 0` with N, independent of CFL and limiter) can be gated. A non-converging one needs a
  scheme decision.

**A6.2 Gupta et al. (2021) shock tubes A and B.**
- *Reference.* Gupta, Sharma & Mignone (2021), Table 2. `gamma = 5/3`, `gamma_cr = 4/3`,
  1000 cells.
  - **A:** left `rho = 1, P_th = 2, P_cr = 1`; right `rho = 0.2, P_th = 0.02, P_cr = 0.1`;
    `t = 0.1`.
  - **B (strong, CR-dominated):** left `rho = 1, P_th = 6.7e4, P_cr = 1.3e5`; right
    `rho = 0.2, P_th = 2.4e2, P_cr = 2.4e2`; `t = 1e-4`.
- *Expectation.* The adiabatic-CR reference from `pfrommer_riemann_solver.py`. Gupta et al.
  show visible method dependence of the post-shock CR entropy here.
- *Gate.* As in A6.1. Case B also stresses positivity and the CR acoustic speed in the CFL.

**A6.3 Mach-number scan.**
- *Reference.*
  - Pfrommer et al. (2006), App. B: eight shock tubes with `X_cr = P_cr/P_th = 2` (left) and
    1 (right), `rho = 1 | 0.2`, and the low pressure adjusted for `M ∈ {1.4, 2, 3, 6, 10, 30,
    60, 100}`. Take the exact values from App. B.
  - Pfrommer et al. (2017), Sec. 4.1: composite case without acceleration, `M ≈ 10`,
    `rho 1 : 0.125`, `P 63.499 : 0.1`, 100 cells.
- *Expectation.* Semenov et al. (2021) find that the CR overproduction "strongly increases for
  more compressible shocks".
- *Measure.* `K_cr` plateau error vs M at two resolutions.
- *Why.* Phase B DSA uses these shocks. This scan gives the error budget per Mach number that
  item 8's `eta(M)` validation needs. It also exercises the reference solver over a published
  parameter range, and the plots can be compared with Pfrommer et al. 2006's Fig. B1.

**A6.4 Committed reference-solver regression tests.**
- *Setup.* Turn the ad hoc validation into pytest cases:
  - `X_cr = 0` against `riemann_solver._exact_riemann_ideal_gas` (Sod with gamma = 1.4 and
    5/3, and reverse Sod), over the full profile;
  - `gamma_cr = gamma` against the summed-pressure single-gamma solution;
  - `K_cr` continuous across the shock and the rarefaction in the composite case (an identity
    of the reference).
- *Cross-check.* Kudoh & Hanawa (2016) reproduce Pfrommer et al. (2006)'s solution. Compare
  against one of their published cases, if their parameters can be read off.
- *Gates.* Agreement to 1e-6; the identity holds to 1e-8.

**A6.5 Conservation in the shock tube.**
- *Setup.* The item-6 run with reflecting walls on both ends (or a periodic double tube),
  float64.
- *Expectation.* Mass, momentum and **total** energy `E_gas + e_cr` are conserved to
  round-off. The coupling `-v·∇P_cr` (gas) and `-P_cr div v` (CRs) sums to a divergence
  (DESIGN.md, product rule). `e_cr` alone is not conserved, and the partition is what A6.1
  measures.
- *Gate.* Relative drift below 1e-12 for each quantity.

**A6.6 Finite-`kappa` two-moment shock tube (production path).**
- *Reference.* Thomas, Pfrommer & Pakmor (2021), Sec. 4.2, with `v_a = 0`:
  - `rho = 1`, `u = 0`, `P_th = 1`, `P_cr = 1 | 0.3333`;
  - `c = 10`, `kappa = 1/300`;
  - 1024 cells, x ∈ [-5, 5], `t = 2`.

  With `v_a = 0`, TPP21 recover the adiabatic two-fluid solution.
- *Setup.* Our `v_red = 10`, `kappa = 1/300`, relaxation on, cap on, guard on and off.
  - Here `kappa/u_s` is below dx, so the CR precursor is unresolved and the result should
    approach the `v_red = 0` solution away from the shock.
  - Scan `kappa` upward until the precursor is resolved (several cells). The resolved-precursor
    regime belongs to item 9 (CR-modified shock structure) and is out of scope.
- *Gates.* In the unresolved regime, the plateau values match the `v_red = 0` run within the
  A6.1 error. The guard on and off agree within 1%.

**A6.7 Remedy evaluation (a decision, not a test).**
If A6.1 to A6.3 confirm a non-converging, method-dependent partition, the options from the
literature are:
- (a) an entropy-like conservative CR variable (Kudoh & Hanawa 2016; Semenov et al. 2021);
- (b) Gupta et al. (2021)'s robust unsplit discretization, with interface `u` and `P` taken
  from the HLL states;
- (c) a prescribed shock partition, which fits Phase B, where DSA prescribes the CR fraction
  at shocks anyway.

A6.1 to A6.3 then serve as the acceptance tests for whichever option is chosen (decision D2).

---

## 7. Execution order and deliverables

**Priority 1** (cheap, highest diagnostic value; mostly reuses existing setups):
1. **A6.1.** Plateau `K_cr` diagnostic and convergence. A few minutes of 1D runs. Decides D2.
2. **A2.3 and A2.2.** Equivalence control and CR sound wave. These explain the item-2
   oscillations and check the size of the coupling.
3. **A1.1 and A1.3.** Gas advection and the pressure-balance contact. Together they close the
   plan's actual item-1 intent.
4. **A4.1.** Exact telegrapher reference. It removes the fragile order gate and resolves the
   `v_red^-1` floor question (with A4.2).
5. **X1 and X2.** Baseline registry, and tightening of the existing gates.

**Priority 2:**
1. A3.1 Sovinec test and A4.4 steady-state test. Both need the source hook, so build it once.
2. A3.2 anisotropic Gaussian.
3. A6.2, A6.3 and A6.4.
4. A1.2 Galilean (commit the audit run).
5. A2.4 dispersion relation.

**Priority 3:**
1. A2.1 (3D homologous).
2. A1.4 and A1.5.
3. A3.3, A3.4 and A3.5 (absolute gates, guard-off tracking, `v_red` scan).
4. A4.3 and A4.5.
5. A6.5 and A6.6.

**Infrastructure** (shared, build first where needed):
- A test-only `e_cr` source hook (A3.1, A4.4).
- A static-heavy-gas helper (X6). Factor it out of `cr_anisotropic_ring.py`.
- `linear_theory.py` in `test_setups/reference_solutions/`, with the 5x5 dispersion matrix
  (A2.4), the eigenvectors (A1.4, A2.2) and the spectral telegrapher solver (A4.1).
- A convergence-fit helper (least-squares order, pairwise orders) shared by all tests.
- `baselines.json` (X1).

**Per-test conventions** (as in the existing suite):
- One pytest file per item, each with a figure in its topic subfolder `pics/<NN_topic>/` that overlays the reference, shows the
  convergence plot, and states the measured numbers in the title.
- Docstrings say what physics each gate checks, and where its threshold comes from (a
  literature value or a measured baseline).
- Long runs are marked `@pytest.mark.slow`.

---

## 8. Decisions for the user

- **D1: perpendicular numerical diffusion budget.** Should item 3 gate an *absolute*
  `kappa_perp,num/kappa_par`? The proposal is to reach at most the S&H van Leer row (ring,
  eq. 39) and an order of at least 1.5 in the Sovinec test. The current scheme fails this by
  1.7-2.6x. The physically relevant target for M7 is `kappa_perp,num ≪ kappa_perp,phys`
  (= 0.01 `kappa_par`) at M7's resolution and `v_red`.
- **D2: shock partition.** If A6.1 shows a non-converging partition error, should we change the
  CR energy formulation (A6.7 a/b) before Phase B results are used, or accept it and use a
  prescribed shock partition (A6.7 c)?
- **D3: production vs library defaults in gates.** Should hard gates follow the library
  default (guard on) or production (guard off)? The proposal is hard gates on the library
  default, plus tracked metrics for production (A3.4).

---

## 9. Bibliography (verify volumes and pages on ADS before citing in a paper)

**Directly used for the tests above**
- Jiang, Y.-F. & Oh, S. P. (2018), ApJ 854, 5. Two-moment CR transport. Sec. 4.1.4:
  diffusion in a static and a moving fluid. Sec. 4.1.5: anisotropic ring, eq. 28. arXiv:1712.07117.
- Thomas, T., Pfrommer, C. & Pakmor, R. (2021), MNRAS 503, 2242. Two-moment CR hydrodynamics
  on a moving mesh. Sec. 4.2: CR shock tubes. Sec. 4.3: contact discontinuity. Sec. 4.5:
  telegrapher dispersion test. arXiv:2010.11960.
- Sharma, P. & Hammett, G. W. (2007), JCP 227, 123. Ring test (Sec. 7.1), positivity (Fig. 7),
  Tables 1-4. arXiv:0707.2616.
- Pakmor, R., Pfrommer, C., Simpson, C. M., Kannan, R. & Springel, V. (2016), MNRAS 462, 2603.
  Gaussian diffusion (Sec. 3.1), ring at t = 10 and 200 (Sec. 3.2), Sovinec test (Sec. 3.4).
  arXiv:1604.08587.
- Sovinec, C. R. et al. (2004), JCP 195, 355. Steady-state anisotropic-diffusion test.
- Dubois, Y. & Commerçon, B. (2016), A&A 585, A138. Anisotropic CR/heat diffusion in RAMSES
  (Sovinec, ring, Gaussian tests). arXiv:1509.07037.
- Parrish, I. J. & Stone, J. M. (2005), ApJ 633, 334. Origin of the circular-field ring test.
- Pfrommer, C., Springel, V., Enßlin, T. A. & Jubelgas, M. (2006), MNRAS 367, 113.
  App. B: Riemann problem for a CR + thermal composite; Mach-scan shock tubes.
  arXiv:astro-ph/0603483.
- Pfrommer, C., Pakmor, R., Schaal, K., Simpson, C. M. & Springel, V. (2017), MNRAS 465, 4500.
  CR shock tubes with and without acceleration (Sec. 4.1, Table 1). arXiv:1604.07399.
- Gupta, S., Sharma, P. & Mignone, A. (2021), MNRAS 502, 2733. Non-uniqueness of two-fluid CR
  equations at shocks; shock tubes A and B (Table 2). arXiv:1906.07200.
- Semenov, V. A., Kravtsov, A. V. & Diemer, B. (2021/2022). Entropy-conserving scheme for
  non-thermal energies; spurious CR generation at shocks. arXiv:2107.14240 (check the journal
  reference on ADS).
- Kudoh, Y. & Hanawa, T. (2016), MNRAS 462, 4517. Conservative CR-MHD with CR "number"
  `∝ e_cr^{3/4}`; reproduces the Pfrommer et al. (2006) solution. arXiv:1608.03206.
- Stone, J. M. et al. (2008), ApJS 178, 137. Athena test suite; linear-wave convergence method.
- Toro, E. F. (2009), *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed.,
  Springer. Exact Riemann solver and error behaviour at discontinuities.
- Abgrall, R. (1996), JCP 125, 150; Karni, S. (1994), JCP 112, 31. Pressure oscillations at
  multi-component contacts.
- Drury, L. O'C. & Völk, H. J. (1981), ApJ 248, 344. Two-fluid CR hydrodynamics and the
  adiabatic CR invariant.

**Background**
- Hanasz, M., Strong, A. W. & Girichidis, P. (2021), Living Rev. Comput. Astrophys. 7, 2.
  Review of CR propagation simulations. arXiv:2106.08426.
- Ruszkowski, M. & Pfrommer, C. (2023), A&A Rev. 31, 4. CR feedback review. arXiv:2306.03141.

---

*Basis: committed scripts and figures at `79b1d77` (2026-10-07). Numbers marked (fig.) were
read off the SVGs, from the data paths where possible, otherwise by eye. Re-measure them when
the tests are next run, and record them in the baseline registry (X1).*
