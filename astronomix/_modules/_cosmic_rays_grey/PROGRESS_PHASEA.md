# Phase A test plan -- PROGRESS

Status tracker for `astronomix_CR_phaseA_test_plan.md` (review + extension of the Phase A ladder
items 1, 2, 3, 4, 6). IDs below are the plan's. The plan itself is a frozen review at `79b1d77`;
do not edit its "(fig.)" numbers, record new measurements here (and in the baseline registry, X1).

Scope rule: anything that changes the scheme (fixes proposed under the plan's "Fix" step, the
outcomes of D1-D3) is logged in `PROGRESS.md` / `DESIGN.md` as usual and only linked from here.
Fixes to existing schemes need the user's go-ahead before they are implemented.

Status: `todo` / `running` / `done` / `blocked` / `dropped`.

## Status table

| ID | Prio | Status | Test file | Key result | Date |
|---|---|---|---|---|---|
| **Item 1: advection** | | | | | |
| A1.1 passive blob advected by gas | 1 | todo | | | |
| A1.2 Galilean invariance of diffusion | 2 | todo | | | |
| A1.3 CR-thermal contact in pressure balance | 1 | todo | | | |
| A1.4 free-wave convergence + eigenvector | 3 | todo | | | |
| A1.5 free-streaming limit, production config | 3 | todo | | | |
| **Item 2: adiabatic compression** | | | | | |
| A2.1 ballistic homologous compression/expansion | 3 | todo | | | |
| A2.2 CR-modified linear sound wave | 1 | todo | | | |
| A2.3 CR-dominated fluid vs gamma=4/3 hydro | 1 | todo | | | |
| A2.4 two-moment linear dispersion relation | 2 | todo | | | |
| **Item 3: anisotropic diffusion** | | | | | |
| A3.1 Sovinec steady state | 2 | todo | | | |
| A3.2 anisotropic Gaussian, full profile | 2 | todo | | | |
| A3.3 absolute gates + early-time ring | 3 | todo | | | |
| A3.4 production (guard off) tracking | 3 | todo | | | |
| A3.5 `v_red` dependence of `D_perp,num` | 3 | todo | | | |
| **Item 4: isotropic diffusion** | | | | | |
| A4.1 exact telegrapher reference | 1 | todo | | | |
| A4.2 `v_red` scan of the model floor | 1 | todo | | | |
| A4.3 multi-D isotropic Gaussian | 3 | todo | | | |
| A4.4 steady-state diffusion with source | 2 | todo | | | |
| A4.5 passive energy conservation | 3 | todo | | | |
| **Item 6: shock tube** | | | | | |
| A6.1 post-shock `K_cr` diagnostic + convergence | 1 | done | `cr_shock_tube_partition.py` | `K_cr,2` +1.06% at every N (100-3200), 0.82-1.21% over methods: **not converging** | 2026-10-08 |
| A6.2 Gupta et al. (2021) tubes A and B | 2 | done | `cr_shock_tube_partition.py` | A (CR-dominated, M 2.6): `K_cr,2` +6.3%, `P_th,2` -10.5%, spurious 19% of e_th; B (M 9.9): +10.6% | 2026-10-08 |
| A6.3 Mach-number scan (Pfrommer et al. 2006) | 2 | done | `cr_shock_tube_partition.py` | `K_cr,2` error 0.4% (M 1.4) -> saturates 11.2% (M >= 30); spurious/e_th peaks 4.2% at M 3, ~M^-2 above | 2026-10-08 |
| A6.4 reference-solver regression tests | 2 | todo | | | |
| A6.5 conservation in the shock tube | 3 | done | `cr_shock_tube_partition.py` | mass <= 2e-16, E_gas + e_cr <= 5e-14 (reflecting box, both schemes) | 2026-10-08 |
| A6.6 finite-`kappa` two-moment shock tube | 3 | todo | | | |
| A6.7 remedy evaluation (decision) | -- | done | DESIGN.md | option a (CR entropy, Semenov et al. 2021 eq. 8 + Et+Scr energy bookkeeping); design note written | 2026-10-08 |
| **Cross-cutting** | | | | | |
| X1 baseline registry, stale docstrings | 1 | todo | | | |
| X2 tighten loose gates | 1 | todo | | | |
| X3 multi-resolution runs | -- | todo | | covered per test | |
| X4 production-config variants | -- | todo | | covered per test | |
| X5 float64 for analytic tests | -- | todo | | covered per test | |
| X6 static heavy gas for pure transport | -- | todo | | via helper below | |
| X7 FV only | -- | dropped | -- | out of scope (plan item 20) | |
| Mark ring "cap alone" test `xfail` (Sec. 3.1) | -- | todo | `cr_anisotropic_ring.py` | | |
| Fix Pfrommer et al. 2006 citation (Sec. 6.2) | -- | todo | `cr_shock_tube.py`, DESIGN.md | | |

## Decisions

| ID | Question | Status | Outcome |
|---|---|---|---|
| D1 | Absolute `kappa_perp,num/kappa_par` budget for item 3? | open | |
| D2 | Shock partition: change CR energy formulation or prescribe it? | **done: option a, default on** (2026-10-09) | Conservative CR entropy row synced into `e_cr` every RK stage; design note in DESIGN.md "Resolved: conservative CR entropy at shocks", awaiting review, not implemented |
| D3 | Hard gates on library default (guard on) or production (guard off)? | open | |

## Infrastructure

- [ ] Test-only `e_cr` source hook (A3.1, A4.4)
- [ ] Static-heavy-gas helper, factored out of `cr_anisotropic_ring.py` (X6)
- [ ] `test_setups/reference_solutions/linear_theory.py`: 5x5 dispersion matrix, eigenvectors,
      spectral telegrapher solver (A1.4, A2.2, A2.4, A4.1)
- [ ] Convergence-fit helper: least-squares and pairwise orders
- [ ] `baselines.json` next to `pics/` (X1)

## Log (newest first)

### 2026-10-09: M7 A/B (cr_entropy off vs on) -> real difference -> default on

- **Runs:** `m7_girichidis_pilot.py both --res=0.5` (62.5 pc, 250 Myr, MHD + self-gravity,
  guard off), seeds 42 and 43, v_red = 1000 km/s, flag off and on.
  - Instrumented with `/export/scratch/nknoell/phaseA/m7ab/m7_budget_ab.py`: the 2026-10-08
    budget wrapper plus a log of the per-call CR->gas transfer (two calls per step, from the
    MHD gas half-steps).
  - Analysis: `ab_analyze.py`. Data: `/export/scratch/nknoell/phaseA/m7ab/out/`.
  - The pilot got a `--cr-entropy` switch for these runs; since the default flip it is
    `--energy-scheme` (opt-out). Entropy runs carry `_crent` in the suffix.

  | v_red = 1000 | seed 42 off -> on | seed 43 off -> on | seed scatter (off / on) |
  |---|---|---|---|
  | E_cr(250 Myr) | 4.27e11 -> 3.48e11 (-19%) | 4.71e11 -> 3.22e11 (-31%) | -- |
  | E_cr / injected (250 Myr) | 1.41 -> 1.15 | 1.56 -> 1.07 | 0.15 / 0.08 |
  | cumulative CR->gas transfer | +4.7e10 | +5.3e10 | -- |
  | eta(1 kpc), late | 0.28 -> 0.38 | 0.22 -> 0.40 | 0.06 / 0.02 |
  | v_out late (km/s) | 3.1 -> 3.9 | 3.1 -> 4.2 | 0 / 0.3 |
  | H_gas late (pc) | 351 -> 358 | 336 -> 360 | 15 / 2 |
  | mean mass flux at abs(z) = 1 kpc, 150-250 Myr (code) | -2.3 -> -1.4 | -3.1 -> -1.3 | -- |

  - **Transfer time series (seed 42):** gas -> CR up to ~100 Myr (-6.8e9; the energy scheme
    under-predicts CR energy in expansions, as in the 1-2-3 test), then CR -> gas (+5.4e10)
    once infall and compression take over.
  - **Reading:**
    - The flag removes most of M7's CR-energy excess over injection (1.4-1.6x -> 1.07-1.15x).
      The rest is the open-boundary inflow found on 2026-10-08, which the flag does not touch.
    - It raises the late outflow loading by 35-80%, well beyond the seed scatter. Net infall
      at 1 kpc halves.
    - Disc thickness is unchanged within the scatter.
  - **Decision (user's rule):** a real difference, so `CosmicRayGreyConfig.cr_entropy`
    defaults to True. Opt-outs:
    - CWB `_cwb_setup.py` (split MUSCL scheme);
    - `cr_streaming_1d`, `cr_energy_budget` and `cr_mhd_energy_budget` (streaming);
    - `cr_shock_tube_partition.run_shock_tube` keeps `cr_entropy=False` as its default, so
      A6.1-A6.3 keep measuring the energy scheme as tracked controls.
- **v_red = 1e4 (production setting), seed 42** (~77 / 93 min):

  | | flag off | flag on |
  |---|---|---|
  | E_cr(250 Myr) | 5.23e11 | 3.48e11 (-33%) |
  | E_cr / injected | 1.75 | 1.16 |
  | cumulative CR->gas transfer | -- | +3.7e10 |
  | CR energy advected in through the z boundaries (cumulative) | +1.70e11 | +0.71e11 |
  | eta(1 kpc), late | 0.21 | 0.44 |
  | v_out late (km/s) | 3.3 | 4.4 |
  | H_gas late (pc) | 280 | 323 (+15%) |
  | mean mass flux at abs(z) = 1 kpc, 150-250 Myr (code) | -2.3 | -0.35 |
  | e_cr(abs(z) > 1.5 kpc) / e_cr(abs(z) < 250 pc), late | 0.70 | 0.59 |

  - Same direction as at v_red = 1000, and stronger.
  - The CR energy entering through the open z boundaries (the 2026-10-08 finding) drops by 60%:
    with less spurious shock CR energy there is less infall and a less CR-filled upper box.
    So part of that "boundary artefact" was itself driven by the energy scheme's shock error.
  - The remaining E_cr/injected = 1.16 is the boundary inflow that is left; the diode-boundary
    option of 2026-10-08 is still open.
- **Re-baseline of items 7, 8, 10, 11 and 15** (scratch copies, flag forced off vs the new
  default; items 11 and 15 at 128^3; `/export/scratch/nknoell/phaseA/rb2_{on,off}/`). All pass
  in both.
  - Figures in `pics/{07,08,10,11,15}_*` regenerated with the default; docstrings carry both
    sets of numbers.
  - The four tests now print their calibrated metrics (one `print` per metric before its gate,
    no gate changes).

  | item | metric | off (energy scheme) | on (default) |
  |---|---|---|---|
  | 7 Sedov 48^3 | E_cr/E_tot; partition identity | 0.0434; 3.3e-5 | 0.0414; 4.6e-5 |
  | 8 DSA vs Mach 48^3 | KR13 / CS14 E_cr/E_tot | 0.0877 / 0.0457 | 0.0836 / 0.0436 |
  | 10 wind bubble | E_cr/E_tot; partition identity | 0.0438; 4.4e-4 | 0.0552; 3.2e-4 |
  | 11 clumpy SNR 128^3 | E_cr/E_tot uniform / clumpy; clumpy effect | 0.0570 / 0.0532; -6.7% | 0.0545 / 0.0511; -6.3% |
  | 15 pion bump 128^3 | hotspot; SED slope; suppression | 1.0 cell; 0.120; 0.949 | identical |

  The wind bubble goes up: the energy scheme loses CR energy in adiabatic expansion (cold
  rarefaction test -9%), and the shocked wind expands for most of the run. The shock-dominated
  tests go down by 4-5%.
- **Not re-baselined:** the other CR tests also run with the new default, and they pass in the
  flag-on suite. Their committed figures and docstring numbers still show the energy scheme:
  items 1-4, 6 and 9, Phase D, and the gradient checks. Item 9 Test A improved: 0.13-0.15% ->
  0.08-0.13%.

### 2026-10-09: CR entropy step 3 (gas-positivity safeguard, dual-energy interplay)

- **Code (uncommitted):**
  - `CosmicRayGreyParams.cr_entropy_max_thermal_drain` (0.5): the transfer may take at most
    this fraction of the gas thermal energy above the floor per step; the rest stays in
    `e_cr`.
  - With gas `dual_energy`, cells whose pressure comes from the gas entropy take
    `e_cr := e(s_cr)` without a transfer (`evolve_state._dual_energy_entropy_cells`, factored
    out of `_dual_energy_select`), and the gas entropy row is re-synced afterwards.
  - The `NotImplementedError` for `dual_energy` is removed.
- **Why:** in a cold, CR-dominated double rarefaction (Einfeldt 1-2-3, P_th/P_cr = 1e-2 ...
  1e-6) the energy scheme turns ~9% of the CR energy into start-up gas heat. `cr_entropy`
  took it back, more than the gas had, and went NaN.
- **New test** `cr_shock_tube_partition.test_cr_entropy_cold_rarefaction` (periodic, P_th
  1e-4) passes:

  | | energy scheme | cr_entropy |
  |---|---|---|
  | max \|K_cr/K_cr,0 - 1\| in the rarefaction, no / with dual energy | 9.1% / 7.4% | 0.89% / 0.75% |
  | energy drift (both hit the gas pressure floor at start-up) | 4.6e-4 | 2.1e-4 |

  - Known trade-off: at P_th/P_cr = 1e-6 with dual energy, the start-up region takes ~1e-4 of
    the CR energy as heat, 8x the energy scheme. Without dual energy `cr_entropy` halves the
    spurious heating.
- **Flag off:** bitwise identical to HEAD `54c4655` for 1D relaxation, 2D MHD anisotropic,
  and `dual_energy` with and without CRs.
- **Flag on, full CR suite** (`/export/scratch/nknoell/phaseA/regress_step3_on/`, 3D test with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`):
  - 48 pass. Failures: the 3 streaming tests (rejected by design) and the ring cap-alone test
    (fails by design).
  - Every printed metric is identical to the step-2 flag-on run (items 8 and 9, Phase D kappa
    and injection, T3, ring 3a/3b/3c), so the safeguard and the dual-energy rule do not act in
    any existing test.

### 2026-10-08: CR entropy step 2 (diffusion carried by s_cr) + full-suite reruns

- **Scope decision (user):** streaming is out of scope. The CR model is the diffusion limit of
  Jiang & Oh (2018), so `cr_entropy` with `streaming=True` raises.
- **Code (uncommitted):**
  - `cr_grey_sources.cr_entropy_closure_source`, called per axis in each RK stage, adds the
    closure (diffusion) part of the `e_cr` change to `s_cr` with the cell weight
    `(gamma_cr - 1) rho^(1 - gamma_cr)`;
  - the shared helper `cr_grey_transport.cr_passive_row_flux`;
  - `cr_entropy` validation only when `grey_cosmic_rays` is on: a CR-free config with the flag
    set raised before, which the flag-on suite found in `cr_divergence_b_preservation`.
- **Unit checks:**
  - Static heavy gas, 2D MHD anisotropic diffusion: flag on vs off agree to 3e-14.
  - Stratified static rho: differences 1.0e-3 / 2.5e-4 / 8.0e-5 at N = 32 / 64 / 128. This is
    HLL contact smearing, which mixes gas of different `K_cr`, so it is truncation error.
- **Flag off is still bitwise identical** to HEAD `54c4655` (1D relaxation shock tube, 2D MHD
  anisotropic).
- **Flag-off regression of the full CR suite** (step-1 code; step 2 is bitwise-equal with the
  flag off):
  - 50/52 pass.
  - `test_cr_ring_flux_cap_alone` fails by design (plan Sec. 3.1, to be marked xfail).
  - `test_cr_phase_d_injection_efficiency_inference` crashed on GPU index 2 with
    `CUDA_ERROR_ILLEGAL_ADDRESS` (the known flaky card). Rerun on GPU 0 it passes, identical to
    HEAD: AD-FD 9.04e-4, mach_scale 1.06358.
  - Items 11 and 15 OOM at their default 256^3 / 300^3 and pass at 128^3.
- **Flag-on run of the full CR suite** (default forced on in the runner,
  `/export/scratch/nknoell/phaseA/regress_step2_on/`):
  - 43 pass, plus divergence-B after the validation fix.
  - Rejected by design: item 5, `cr_energy_budget` and `cr_mhd_energy_budget` (they use
    streaming).
  - Unchanged failure: the ring cap-alone test.
  - `test_cr_anisotropic_3d` OOMs at 128^3 float64 with the default 75% preallocation; rerun
    with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` it passes, with D_par/kappa 1.02607 and D_perp
    0.01662 at N = 128, identical to flag off to 5 digits. The flag-off run is already at the
    limit: XLA warns it cannot get below 9.15 GiB.
- **Flag off vs on:**

  | test | flag off | flag on |
  |---|---|---|
  | item 8 (48^3) E_cr/E_tot KR13 / CS14 | 0.0877 / 0.0457 | 0.0836 / 0.0436 (-4.7%) |
  | item 9 Test A rho/u/P err; Test B du/dx | 0.13/0.15/0.15%; 26% | 0.09/0.13/0.08%; 20% |
  | ring 3a kappa_perp,num/kappa_par (N 50-400) | 4.08e-2 ... 3.26e-3 | 4.00e-2 ... 3.26e-3 |
  | Phase D kappa | 0.01974 | 0.01974 |
  | Phase D injection mach_scale; AD-FD | 1.06358; 9.0e-4 | 1.06371; 2.6e-3 |
  | divergence-B (CR iso / aniso) | 6.1e-15 / 7.1e-15 | 6.7e-15 / 6.4e-15 |
  | T3 3D diffusion, N 128: D_par/kappa, D_perp | 1.02607, 0.01662 | 1.02607, 0.01662 |

  The item-8 drop is the first measurement on a 3D shock problem: ~5% of the DSA test's
  CR energy was spurious shock gain from the energy scheme.

### 2026-10-08: CR entropy implementation step 1 -- post-shock K_cr exact to round-off

- **Code (uncommitted):** `CosmicRayGreyConfig.cr_entropy` (default off), the
  `cosmic_ray_entropy_index` row, the s_cr flux in `hll._grey_cr_hll_rows`,
  `cr_grey_sources.cr_entropy_sync` / `cr_entropy_to_energy` in
  `_evolve_gas_state_unsplit`, and config validation. Details in DESIGN.md
  "Resolved: conservative CR entropy at shocks" -> Status.
- **Design correction:** one sync per hydro step after the operator-split CR sources, not per
  RK stage. The CR sources are not applied inside the stages, and one sync makes
  `e_cr = e(s_cr)` exact after the RK average.
- **Acceptance (new tests in `cr_shock_tube_partition.py`, CPU float64):**
  - `test_cr_entropy_shock_partition` passes; figure
    `pics/06_shock_tube/cr_shock_tube_cr_entropy_test.svg`.

    | case | energy scheme dK_cr,2 | cr_entropy dK_cr,2 | cr_entropy dP_th,2 |
    |---|---|---|---|
    | item 6, N 1600, C_cfl 0.1 / 0.4 / 0.8 | 1.21 / 1.06 / 0.83% | -7e-14 / -2e-14 / -1e-14 | -3e-6 / -5e-5 / -1e-4 |
    | Gupta A, N 2000 | 6.35% | 1.5e-14 | -8e-5 |
    | Gupta B, N 4000 (18-cell window) | 10.6% | -7.3e-6 | -9.6e-4 |
    | M = 2 / 10 / 100, N 1600 | 2.6 / 10.6 / 11.2% | 2e-14 / 6e-10 / -1e-9 | -1e-4 / -5.5e-4 / -5.7e-4 |

  - Full scan (51 runs, `/export/scratch/nknoell/phaseA/ent/`): every resolved run has
    |dK_cr,2| <= 2e-9. Spurious CR energy per e_th is <= 1e-4 (was up to 19%). The CFL,
    limiter and Riemann-solver dependence is gone. What remains are plateau errors <= 1e-3
    (truncation at the smeared shock, ~0.4 dx shock offset). At N = 400 the M >= 30 windows
    still see the smeared contact (up to 1%), as before.
  - `test_cr_shock_tube_energy_conservation` (A6.5) passes: reflecting box, 3 t_end, both
    schemes; mass <= 2.2e-16, total energy <= 4.5e-14.
- **Flag off is bitwise unchanged:** a 1D relaxation shock tube and a 2D MHD anisotropic
  transport run give identical final states with the committed code (HEAD `54c4655`
  worktree) and the working tree (max |diff| = 0).
- **Regression of the full CR suite (flag off; GPUs pinned by UUID, from a scratch copy so
  committed figures are not touched):** see the next entry.
- **Not yet (step 2):** the closure-flux and streaming sources on s_cr. Until then
  `cr_entropy` is only correct with `reduced_streaming_speed = 0`; streaming and dual_energy
  raise `NotImplementedError`.

### 2026-10-08: D2 -> option a; design note written (no code)

- User picked A6.7 option a. Design note: DESIGN.md "Resolved: conservative CR entropy at shocks
  (Phase A plan D2, option a; design note 2026-10-08)".
- **Core of the design:**
  - Add a CR entropy row `s_cr = P_cr rho^(1 - gamma_cr)`, advected as a passive
    density-like row (same pattern as the gas `dual_energy` entropy row).
  - Closure-flux divergence and the streaming loss enter `s_cr` as cell-weighted sources.
  - At the end of every RK stage, `e_cr := e(s_cr)`, and the difference goes to the gas
    thermal energy, so `E_gas + e_cr` stays exact.
  - No shock detection.
- Checked against the paper text: Semenov et al. 2021 eq. 8 and Sec. 2.2 (CRs always by the
  entropy equation; total energy kept via the Kudoh & Hanawa / Gupta "Et+Scr" route).
- Next: user review of the note, then implementation step 1 (row + advection + sync; check
  A6.1 at `v_red = 0` and A6.5).

### 2026-10-08: A6.2 + A6.3 done -- error grows with Mach, saturates at ~11% in K_cr

- **Tests:** `test_cr_shock_tube_gupta` and `test_cr_shock_tube_mach_scan` in
  `pytests/cosmic_rays_grey/cr_shock_tube_partition.py` (the module now takes any `ShockTube`);
  figures `pics/06_shock_tube/cr_shock_tube_{gupta,mach_scan}_test.svg`. All three tests pass
  on CPU (16 s / 23 s / 124 s).
- **Setups, from the papers** (text extracted from the arXiv PDFs):
  - Gupta et al. (2021) Table 2: A = L {1, 0, 2, 1}, R {0.2, 0, 0.02, 0.1}, t = 0.1; B = L {1, 0,
    6.7e4, 1.3e5}, R {0.2, 0, 240, 240}, t = 1e-4; 1000 cells.
  - Pfrommer et al. (2006) Sec. 5.2: rho 1 | 0.2, X_cr 2 | 1, P_th,L = 1e5 (gamma - 1); the
    right pressure is not tabulated, so it is solved for the Mach number (composite upstream
    sound speed) with our reference solver. t lets the shock travel 0.35.
  - Cross-check: the reference's post-shock P_cr/P for B is 0.028; Gupta et al. quote ~0.03.
    B is essentially Pfrommer's M = 10 tube (M 9.89, P_R 240 vs 238.7).
- **Metric change (applies to A6.1 too):** the shocked-plateau window is now the shock side,
  55-85% of the way from contact to shock. In the strong tubes the smeared contact carries the
  left side's 63x (B) to 6600x (M = 100) higher K_cr into the contact half, which at low N gave
  spurious +100% to +850% "errors" (contact diffusion, A1.3, not the shock). A6.1 numbers are
  unchanged (+1.056%). Added the spurious CR energy per post-shock thermal energy,
  `dP_cr/(gamma_cr-1) / (P_th,2/(gamma-1))`, which is on the scale of a DSA efficiency.
- **A6.3 (N = 3200; N = 1600 the same to 3 digits; C_cfl 0.4):**

  | M | compression | dK_cr,2 | dP_th,2 | spurious / e_th |
  |---|---|---|---|---|
  | 1.4 | 1.62 | +0.41% | -0.53% | 0.91% |
  | 2 | 2.37 | +2.56% | -2.12% | 3.76% |
  | 3 | 3.09 | +5.99% | -2.34% | 4.23% |
  | 6 | 3.73 | +9.61% | -0.87% | 1.60% |
  | 10 | 3.90 | +10.6% | -0.33% | 0.61% |
  | 30 | 3.99 | +11.1% | -0.02% | 0.07% |
  | 100 | 4.00 | +11.2% | +0.01% | 0.006% |

  - `C_cfl` 0.1 / 0.8 at N = 1600: +15% / -18% relative change at every M (e.g. M = 10: 12.1 /
    8.6%). No crashes up to M = 100 in float64.
  - In the strong limit the spurious CR energy is a fixed ~11% of the *adiabatically
    compressed upstream* CR energy, so it scales with the upstream CR pressure, not with the
    dissipation; per thermal energy it falls as M^-2 above M ~ 6.
- **A6.2:**
  - Tube A (upstream X_cr = 5, M 2.59): dK_cr,2 +6.35%, P_cr,2 +9.9%, **P_th,2 -10.5%**,
    spurious **19%** of e_th, identical at N = 250 / 1000 / 4000. C_cfl 0.1 / 0.8: 6.6 / 6.0%;
    MC 6.36%, van Albada 6.29%, HLLC 6.34%.
  - Tube B (M 9.89): plateau is 15 cells at N = 1000 (window 4 cells), so N <= 1000 is not
    resolved. N = 2000 / 4000: dK_cr,2 10.6%, spurious 0.62%.
- **Reading for D2 / Phase B:**
  - The error is largest, relative to the gas, at weak shocks in CR-loaded gas (M ~ 2-3): there
    it is 4-19% of the post-shock thermal energy. KR13's DSA efficiency is 0 below M = 2.25, so
    there the scheme injects CR energy that DSA would not.
  - At M >= 10 it is <= 0.6% of e_th, small next to DSA efficiencies of ~10%, but still a
    systematic +11% on the adiabatic CR part.
  - It depends on CFL by about +-15-18%, so it also varies with the time step in production.
- Raw data and scan scripts: `/export/scratch/nknoell/phaseA/a62_a63/` (`summary.json`).

### 2026-10-08: A6.1 done -- the post-shock partition error does not converge (-> D2)

- **Test:** `pytests/cosmic_rays_grey/cr_shock_tube_partition.py`, figure
  `pics/06_shock_tube/cr_shock_tube_partition_test.svg`. Item-6 initial conditions, float64,
  plateau means over the middle 50% of each plateau (fixed in x). Passes (~1 min on CPU).
- **Shocked plateau, HLL + minmod, `C_cfl` 0.4:**

  | N | dK_cr,2 | dP_th,2 | dP_cr,2 | drho_2 | dK_cr,L (control) | shock offset |
  |---|---|---|---|---|---|---|
  | 100 | +1.05e-2 | -4.2e-3 | +9.0e-3 | -1.1e-3 | +5.4e-4 | +0.23 dx |
  | 400 | +1.06e-2 | -7.5e-3 | +1.34e-2 | +2.1e-3 | -3.4e-4 | +0.06 dx |
  | 1600 | +1.06e-2 | -7.5e-3 | +1.35e-2 | +2.2e-3 | -9.3e-5 | -0.72 dx |
  | 3200 | +1.06e-2 | -7.6e-3 | +1.35e-2 | +2.2e-3 | -4.8e-5 | -1.70 dx |

  - The CR entropy of the shocked gas is 1.06% too high at every N. The left star plateau
    (smooth rarefaction only) converges at order 0.85 (N >= 200), so this is the shock, not
    transport or the reference.
  - Sign as Semenov et al. (2021) predict: CRs gain, gas loses. The plan's by-eye "+2% e_cr"
    from the figure is +1.35% in P_cr when measured.
  - The softer post-shock mixture makes the shock slower: a fixed ~5e-4 lag in x (i.e. growing
    in cells) and a +0.2% denser plateau.
- **Method dependence** (N = 400; identical at N = 1600): `C_cfl` 0.1 / 0.4 / 0.8 -> 1.21 / 1.06 /
  0.82%; MC limiter 0.99%; van Albada 1.04%; HLLC 1.02%.
- **Shock strength:** compression 1.94, composite Mach ~1.6. A6.3 measures the Mach dependence.
- **Gates (tracked, until D2):** `|dK_cr,2| < 2%` (upper bound only), control order > 0.7,
  control error < 1e-3 at N = 1600.
- **Conclusion:** the Gupta et al. (2021) non-uniqueness is present: the partition is a scheme
  property, not truncation error. D2 needs a user decision. A6.2/A6.3 (stronger shocks) would
  size the problem for Phase B DSA before choosing between A6.7 a/b/c.
- Raw scan data: `/export/scratch/nknoell/phaseA/a61/` (`summary.json`).

### 2026-10-08: tracker created

Nothing from the plan implemented yet. Starting with A6.1.
