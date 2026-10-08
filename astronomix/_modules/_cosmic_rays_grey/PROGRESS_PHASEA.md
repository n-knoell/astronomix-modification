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
| A6.5 conservation in the shock tube | 3 | todo | | | |
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
| D2 | Shock partition: change CR energy formulation or prescribe it? | **decided: option a** (2026-10-08) | Conservative CR entropy row synced into `e_cr` every RK stage; design note in DESIGN.md "Open: conservative CR entropy at shocks", awaiting review, not implemented |
| D3 | Hard gates on library default (guard on) or production (guard off)? | open | |

## Infrastructure

- [ ] Test-only `e_cr` source hook (A3.1, A4.4)
- [ ] Static-heavy-gas helper, factored out of `cr_anisotropic_ring.py` (X6)
- [ ] `test_setups/reference_solutions/linear_theory.py`: 5x5 dispersion matrix, eigenvectors,
      spectral telegrapher solver (A1.4, A2.2, A2.4, A4.1)
- [ ] Convergence-fit helper: least-squares and pairwise orders
- [ ] `baselines.json` next to `pics/` (X1)

## Log (newest first)

### 2026-10-08: D2 -> option a; design note written (no code)

- User picked A6.7 option a. Design note: DESIGN.md "Open: conservative CR entropy at shocks
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
