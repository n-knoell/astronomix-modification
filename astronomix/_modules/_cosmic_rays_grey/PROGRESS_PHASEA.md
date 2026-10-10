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
| A1.1 passive blob advected by gas | 1 | done | `cr_advection_by_gas.py` (1D + 3D) | as expected: 1D Gaussian order 1.51, top-hat 0.66, TVD; 3D diagonal order 1.56 / 1.55, round to 4e-5, passive | 2026-10-09 |
| A1.2 Galilean invariance of diffusion | 2 | todo | | | |
| A1.3 CR-thermal contact in pressure balance | 1 | done (tracked) | `cr_contact_pressure_balance.py` (1D + 3D, hydro + MHD) | **balance not kept**: max abs(P_tot - 1) ~1e-2, order ~0.5-0.75, spurious u ~1e-2; cause: gas energy HLL-diffused, e_cr not. Fix proposed | 2026-10-09 |
| A1.4 free-wave convergence + eigenvector | 3 | todo | | | |
| A1.5 free-streaming limit, production config | 3 | todo | | | |
| **Item 2: adiabatic compression** | | | | | |
| A2.1 ballistic homologous compression/expansion | 3 | todo | | | |
| A2.2 CR-modified linear sound wave | 1 | done; fixed by the stage-wise coupling | `cr_adiabatic_coupling.py` | **fails with the operator-split coupling**: order 1.09 / 0.68 / -0.09 at P_cr/P_th 0.5 / 2 / 10, an instability at 8-12 cells, unstable above C_cfl ~0.2. Stage-wise coupling prototype: order 1.85-1.87, stable at C_cfl 0.8 | 2026-10-09 |
| A2.3 CR-dominated fluid vs gamma=4/3 hydro | 1 | done; fixed by the stage-wise coupling | `cr_adiabatic_coupling.py` | item-2 ringing is **not** in gamma=4/3 hydro (0.0012), only in the operator-split two-fluid run (0.36 at N 2048). Stage-wise: 0.0026, and the simple wave matches hydro at order 1.85 | 2026-10-09 |
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
| Mark ring "cap alone" test `xfail` (Sec. 3.1) | -- | done | `cr_anisotropic_ring.py` | `pytest.mark.xfail(strict=True)` when pytest is importable (the venv has none); standalone runs print XFAIL; verified with pytest 9.1.1 from a scratch install | 2026-10-09 |
| Fix Pfrommer et al. 2006 citation (Sec. 6.2) | -- | todo | `cr_shock_tube.py`, DESIGN.md | | |

## Decisions

| ID | Question | Status | Outcome |
|---|---|---|---|
| D1 | Absolute `kappa_perp,num/kappa_par` budget for item 3? | open | |
| D2 | Shock partition: change CR energy formulation or prescribe it? | **open again** (2026-10-09) | option a implemented as opt-in `cr_entropy` (default off after the 2026-10-09 analysis: the energy-conserving transfer drains CR-dominated gas); candidate: shock-only transfer (Semenov-style) |
| D3 | Hard gates on library default (guard on) or production (guard off)? | open | |

## Infrastructure

- [ ] Test-only `e_cr` source hook (A3.1, A4.4)
- [ ] Static-heavy-gas helper, factored out of `cr_anisotropic_ring.py` (X6)
- [ ] `test_setups/reference_solutions/linear_theory.py`: 5x5 dispersion matrix, eigenvectors,
      spectral telegrapher solver (A1.4, A2.2, A2.4, A4.1)
- [ ] Convergence-fit helper: least-squares and pairwise orders
- [ ] `baselines.json` next to `pics/` (X1)

## Log (newest first)

### 2026-10-10: P2 implemented in the repo (HLLC only, opt-in via `riemann_solver=HLLC`)

- **Code:**
  - `cr_grey_transport.cr_total_pressure_flux` (predicate: grey CRs, FV, unsplit, HLLC or
    HLLC-LM, not MHD) and `CR_TOTAL_PRESSURE_SIGNAL_SPEED_FACTOR = 1.1`;
  - `hll._hllc_wave_speeds` (factored out of `_hllc_solver`; total pressure + widened speeds)
    and `hll.cr_hllc_face_velocity`;
  - `_hllc_solver` adds P_cr to the momentum and gas-energy fluxes;
  - `cr_grey_sources.cr_face_velocity_work` (pdv per axis, in the flux loop of
    `_evolve_gas_state_unsplit_inner`);
  - `cr_grey_feedback_sources` drops -grad P_cr / -v.grad P_cr / -P_cr div v under the
    predicate.
  - HLL, AM-HLLC, split, MHD: unchanged (source coupling).
  - DESIGN.md "Resolved: total-pressure HLLC flux".
- **Test:** `cr_contact_pressure_balance.py` now runs the hydro cases with HLLC.
  - Gates: max abs(P_tot - 1) and max abs(u - u0) below 1e-10 at every N.
  - The 1D HLL runs and the 3D MHD runs (HLL) stay as tracked bounds; the docstring is
    rewritten.
  - 1D measured with the repo code: HLLC static <= 4.4e-15, advected <= 1.2e-13; HLL
    identical to before (5.9e-3 / 2.0e-3).
- **Verification** (CPU, HLLC forced as default solver, `p2_repo/forced_hllc/`): matches the
  prototype P2 + HLLC to the 3rd-4th digit, identically for 5 of 12 tests.
  - A2.3 shows the same order-gate failure as the prototype (two-fluid equals hydro). Not
    relevant as configured (A2.3 runs HLL).
- **Full test set as configured** (`/export/scratch/nknoell/phaseA/p2_repo/logs/`; 1D files
  on CPU this time, from now on GPU only):
  - All files pass. `test_cr_ring_flux_cap_alone` "fails" as before: it is the strict xfail,
    and the direct runner does not apply the mark.
  - The ring and the MHD energy budget (HLL / MHD, untouched by P2) are identical to the
    previous run: MHD+CR energy error 1.022e-10.
  - HLL-configured tests are unchanged (A1.3's HLL runs are bit-identical to before).
  - A1.3 (HLLC): 1D static <= 3.3e-16, advected <= 8.6e-14; 3D slab 2.2e-16 / 2.7e-14, bubble
    3.3e-16. MHD (HLL) unchanged at 3.3e-2 / 9.2e-3.
  - HLLC-configured tests, old -> new:
    - item 7 Sedov: E_cr/E_tot 0.0434 -> 0.0401, energy error unchanged at 1.3e-5;
    - item 8 DSA Mach: KR13 0.0876 -> 0.0810, CS14 0.0457 -> 0.0423, ratio 0.5215 -> 0.5217,
      Ms unchanged;
    - item 18 energy budget: 1e-10 (same), E_cr(t_end) 3.517e-2 -> 3.332e-2;
    - item 10 wind bubble: R_2 error 1.2% -> 1.6%, interior P 23.5% -> 22.7%, E_cr/E_tot
      0.0438 -> 0.0430;
    - item 15 pion bump: unchanged to the printed digits;
    - item 11 clumpy SNR (128^3 scratch copy as before; 256^3 never fits 11 GB):
      - E_cr/E_tot uniform 0.0570 -> 0.0530, clumpy 0.0532 -> 0.0494;
      - clumpy-vs-uniform -6.67% -> -6.72%;
      - energy errors unchanged;
    - Phase D injection inference: recovered mach_scale 1.0638 -> 1.0634; AD-vs-FD 2.6e-3 ->
      5.1e-3.
  - Memory: P2 adds +355 MB (+3.7%) of XLA temporaries at 256^3 (9.50 -> 9.86 GB, measured with
    `memory_analysis`). It comes from recomputing the HLLC wave speeds for the face velocity;
    passing S* out of the solver would avoid it.
  - Stale docstrings or figures (quoted E_cr fractions): items 7, 8, 10, 11, 18, Phase D
    (figures regenerated except item 11).

### 2026-10-09: A1.3 fix prototype P2 (total pressure in the gas Riemann flux, pdv) -- recommended

- **Prototype** (scratch `a1/p2_patch.py`, monkeypatch, hydro only; MHD falls back to repo code;
  Gupta, Sharma & Mignone 2021 "Eg+Ecr Unsplit-pdv"):
  - P_tot = P_th + P_cr in the normal momentum flux and in the gas energy flux u (E_g + P_tot);
  - the HLLC star state uses P_tot; signal speeds are the existing sqrt(c_th^2 + c_cr^2) x 1.1;
  - -grad P_cr, -v . grad P_cr and the centred -P_cr div v sources are removed. Instead
    +-P_cr,i (v_{i+1/2} - v_{i-1/2}) / dx goes on E_g / e_cr, with the face velocities from the
    same Riemann solve (HLLC: S*; HLL: Gupta Eq. 29);
  - e_cr / F_cr rows are unchanged (mass flux x upwind + closure). Under HLL only, e_cr also gets
    the HLL dissipation; otherwise HLL cannot balance a static contact.
  - Logs: `/export/scratch/nknoell/phaseA/p2/{base,base_hllc,p2_hllc,p2_hll}/`, `p2/gpu/`.
- **A1.3 contact:**
  - P2 + HLLC and P2 + HLL both reach round-off: 1D static 5e-15, advected 1e-13.
  - 3D oblique slab (static 2e-16, advected 3e-14) and bubble (3e-16), hydro, P2 + HLLC.
  - Before: 5.9e-3 / 2.0e-3 (1D N 512) and 2.6e-2 / 9.8e-3 / 2.2e-2 (3D N 64).
- **P2 + HLLC vs today (HLL, stage-wise) on the CPU set:**
  - Unchanged to 4 digits: item 1, item 3 oblique leak, item 4 (D/kappa, convergence), A1.1.
    CR transport in gas at rest is untouched, unlike P1 and P2 + HLL, which degrade it like P1.
  - A2.2: the coupled acoustic mode now behaves exactly like single-fluid hydro. The L1 error is
    independent of P_cr/P_th (2.03e-2 ... 4.31e-4, order 1.86, eigenvector 3e-10).
  - A2.3:
    - The two-fluid run now *equals* gamma 4/3 hydro: difference 2.4e-5 -> 4.6e-7 (N 64-512), was
      8.4e-4 -> 1.9e-5. Its error vs exact (1.49e-3 ... 3.1e-5) equals hydro's.
    - The squeeze ringing drops to the hydro level: 0.0014 vs hydro 0.0012, was 0.0026.
    - The "two-fluid vs hydro order >= 1.8" gate fails (1.70) only because the difference
      flattens at N 1024; gate needs changing.
  - A6.1 post-shock K_cr error 1.27% -> 0.42% (3x smaller). Still method-dependent: C_cfl 0.8
    0.19%, MC 0.47%, so the partition is still non-unique.
  - Item 2 (e_cr ~ rho^gamma_cr): 9.8e-4 -> 8.8e-4.
  - Item 6 shock tube MAE: similar (e_cr 3.7e-3 -> 3.0e-3).
  - Item 9: within tolerance.
  - The only failing gates are "decreasing with N" in A1.3, now at round-off, and the A2.3
    order gate.
- **GPU tests (HLLC as configured):**
  - Item 18 energy budget: 1.0e-10, same as before. P2 is exactly conservative: the momentum has
    no source, and the pdv terms cancel between E_g and e_cr.
  - E_cr ends lower: item 7 E_cr/E_tot 0.0434 -> 0.0401, item 8 KR13 0.0876 -> 0.0810 (ratio
    unchanged), item 18 E_cr(t_end) -5%. This matches the smaller spurious CR gain at shocks
    seen in A6.1.
  - Item 7 and item 8 Ms unchanged.
  - Item 10 wind bubble passes: R_2 error 1.2% -> 1.6%, interior P 23.5% -> 22.7%, E_cr/E_tot
    0.0438 -> 0.0430.
- **Not covered:**
  - MHD (needs P_cr in the MHD solver's total pressure, as in TPP21's HLLD);
  - HLL users keep a choice: P2 + HLL balances contacts but diffuses CR at the gas speed;
  - M7, other GPU tests;
  - Gupta Eq. 30's face-averaged P_cr (cell P_cr used).

### 2026-10-09: A1.3 fix prototype P1 (e_cr with the gas HLL dissipation) -- trade-off not acceptable

- **Prototype** (scratch `a1/hll_cr_patch.py`, monkeypatch of `hll._grey_cr_hll_rows`): the
  advective part of the e_cr / F_cr flux becomes the HLL flux
  `(S_R u_L q_L - S_L u_R q_R + S_L S_R (q_R - q_L)) / (S_R - S_L)` with the gas rows' speeds,
  instead of mass flux x upwind q/rho. 13 tests on CPU, all pass. Logs:
  `/export/scratch/nknoell/phaseA/hllcr/`.
- **Gains:**
  - A1.3 static contact: max abs(P_tot - 1) at N 64 / 512 2.9e-2 / 5.9e-3 -> 3.8e-3 / 1.0e-3,
    max abs(u) at 512 3.4e-3 -> 2.5e-4.
  - The advected slab does *not* improve (2.0e-3 -> 2.5e-3).
  - Item 2, A2.2, A2.3, item 6 and item 9 unchanged.
- **Costs** (the CR energy now gets numerical diffusion at the gas sound speed even in gas at
  rest):
  - item 1 free CR wave: L2/amp 6.0e-3 -> 1.4e-2, peak loss 3.7% -> 7.8%;
  - item 3 oblique anisotropic: cross-field leak ratio 0.022 -> 0.036 (+65%);
  - item 4: D/kappa - 1 0.0155 / 0.0042 / 0.0012 -> 0.0202 / 0.0053 / 0.0015, convergence L2 at
    N 128 1.7e-3 -> 2.3e-3;
  - A1.1 Gaussian L1 at 512 2.1e-3 -> 2.6e-3;
  - A6.1 dK_cr,2 1.27% -> 1.39%.
- **Verdict:** not adopted. It trades a ~1e-2 contact pressure error for worse CR transport,
  above all anisotropic, which M7 relies on.
- **Better candidate (P2, not prototyped):** contact-preserving gas side instead of a
  more-diffused CR side.
  - HLLC with the total pressure P_th + P_cr in the momentum flux and the star state (Gupta
    et al. 2021 Option 1/2), with e_cr kept on the mass flux.
  - At a contact HLLC then neither diffuses the gas energy nor sees a pressure jump, so the
    balance is exact. Today's HLLC sees only the gas pressure, treats the contact as a shock
    tube, and was *worse* in A1.3 (static 4.7e-2 vs HLL 2.9e-2 at N 64).
  - Needs P_cr moved from the momentum source into the flux (only for HLLC), which changes the
    CR/gas coupling structure: larger, user decision.
- **Literature check of P2 (2026-10-09):**
  - **Supported.** P_cr in the gas Riemann flux, with the effective sound speed
    sqrt(gamma P_th/rho + gamma_cr P_cr/rho), is the standard in:
    - Gupta, Sharma & Mignone 2021 (all options; HLL);
    - Kudoh & Hanawa 2016 (HLL/Roe);
    - Pfrommer et al. 2017 (AREPO);
    - Thomas, Pfrommer & Pakmor 2021, a *two-moment* scheme. Its Eq. 2 has P_tot in the
      HLLD flux; only the parallel part b grad_par P_cr is handled as a source. Its contact
      test (Sec. 4.3, B perpendicular) holds to machine precision.
  - **Refinement needed.** The gas energy must change too.
    - Gupta Eq. 23: e_g flux (e_g + P_th + P_cr) v with source +P_cr div v, and e_cr source
      -P_cr div v ("pdv").
    - Ours is "vdp" on the gas side (-v . grad P_cr, centered) plus a centered -grad P_cr in
      momentum. Gupta Fig. 3 shows exactly vdp failing the *advected* pressure-balance mode,
      because the derivative of the discontinuous P_cr creates spurious disturbances. That
      explains why P1 fixed the static contact but not the advected slab.
  - **Gupta details:**
    - div v from the Riemann-solver interface velocities (their Eq. 29), the same value in
      both energy rows;
    - signal speeds widened by phi = 1.1 for robustness.
  - **Kudoh & Hanawa caveat.** Evolving a nonlinear CR variable (E_cr^(3/4)) gives spurious
    sound waves at advected contacts. Pdv with a linear e_cr does not. Our e_cr is linear, so
    this is fine.
  - **Not fixed by P2.** The shock partition stays non-unique (Gupta); A6 needs re-measuring.

### 2026-10-09: A1.1 and A1.3 in 1D and 3D

- **A1.1 (`cr_advection_by_gas.py`):** passive CR blob (amplitude 1e-6 on 1e-7, gas
  rho = P = 1), one crossing, v_red = 0 (F_cr exactly 0), float64.
  - 1D: Gaussian L1/amp 4.7e-2 ... 2.1e-3 (N 64-512), order 1.51; top-hat order 0.66, no new
    extrema (1e-11); centroid <= 0.012 dx; max abs(u - u0) 1.3e-7. A relaxation variant
    (v_red 1, kappa 1e-6) is identical to 4 digits.
  - 3D, along (1, 1, 1): Gaussian 6.8e-3 / 2.3e-3 / 7.9e-4 (N 32 / 64 / 128), order 1.56 / 1.55;
    second moments along / across the diagonal 1.00004 / 1.00000 / 1.00000; centroid
    <= 0.025 cells. Top-hat sphere (N 64): min -5.8e-9, max 0.98.
  - Both pass. The advective `u_n e_cr` flux behaves as expected for HLL + minmod.
- **A1.3 (`cr_contact_pressure_balance.py`):** rho = 1, [P_th, P_cr] = [1/8, 7/8] | [3/4, 1/4],
  v_red = 0, float64, t = 1. **The pressure balance is not kept.**

  | case | max abs(P_tot - P_ref), N = 64 -> 512 (1D) / 32 -> 128 (3D) | max abs(u - u0) |
  |---|---|---|
  | 1D static | 2.9e-2 ... 5.9e-3 | 1.3e-2 ... 3.4e-3 |
  | 1D advected | 6.8e-3 ... 2.0e-3 | 7.4e-3 ... 1.7e-3 |
  | 3D oblique slab, static | 4.0e-2 / 2.6e-2 / 1.7e-2 | 5.6e-3 ... 4.6e-3 |
  | 3D oblique slab, advected | 1.4e-2 / 9.8e-3 / 6.1e-3 | 1.6e-2 ... 1.1e-2 |
  | 3D MHD slab (incl. B^2/2), static / advected | 4.6e-2 ... 2.1e-2 / 1.4e-2 ... 6.7e-3 | <= 1.6e-2 |
  | 3D CR bubble, static | 4.4e-2 / 2.2e-2 / 1.6e-2 | 2.4e-2 ... 1.2e-2 |

  - In 1D the static contact is far worse early on: 0.16 and |u| 0.055 at t = 0.01.
  - HLLC is no better. A pure-gas density contact at uniform P is exact (1e-15) with both
    solvers.
  - MHD: uniform B stays exact without a contact and with B along the normal. A tangential B
    is compressed by the spurious flows (max abs(dB) up to 0.21 next to the static contact).
    1D FV MHD is not supported, which is why the magnetized case is 3D only.
  - Diffusive variant (v_red 10, kappa 1/300): CRs diffuse out of the contact, so the
    imbalance (~1.5e-2) is physical; reported only.
- **Cause:** the gas energy row gets the HLL dissipation `S_L S_R (E_R - E_L) / (S_R - S_L)`,
  and E jumps at the contact. The e_cr row only moves with the mass flux (zero here).
  - P_th smears over 2 cells within t = 1e-3 while P_cr stays sharp.
  - u grows ~t^2 (3.2e-5 at 1e-4, 3.2e-3 at 1e-3), so a growing imbalance, not an initial
    force.
- **Gates:** tracked bounds (measured + margin, decreasing with N); all 4 tests pass.
  Figures `pics/01_advection/cr_{advection_by_gas,contact_pressure_balance}_{1d,3d}_test.svg`.
- **Proposed fix (literature-backed, needs the user's OK):** give the e_cr row the same HLL(C)
  dissipation as the gas energy row (Gupta et al. 2021 Et+Ecr: the CR energy inside the Riemann
  flux, so all energy components smear by the same linear operator and P_tot stays constant).
  It must be checked against item 2's e_cr ~ rho^gamma_cr consistency, which motivated the
  mass-flux upwinding.

### 2026-10-09: re-baseline for the stage-wise coupling (docstrings + figures)

- **A6 scans repeated** with the new coupling (`/export/scratch/nknoell/phaseA/a61_sw`,
  `a62_a63_sw`).
  - A6.1 dK_cr,2 1.27% at every N; C_cfl 0.1 / 0.4 / 0.8 1.26 / 1.27 / 1.31% (was 1.21 / 1.06 /
    0.82%).
  - Gupta A / B 6.83 / 12.9% (was 6.35 / 10.6%).
  - Mach scan saturates at 13.6% (was 11.2%); C_cfl sensitivity -2 / +5% (was +15 / -18%).
  - The smooth control now converges at order ~1.1 (dK_L 3.4e-9 at N = 3200).
  - `cr_shock_tube_partition.py`: the module tables were rewritten (old values in brackets),
    and the measured values in the test docstrings, including the cr_entropy tests, updated.
    The squeeze no longer rings in any variant; the global mode's gas-entropy error there is
    1.5e-3, against 0.34 before.
- **Dated "Re-measured 2026-10-09 with the stage-wise CR coupling" notes**, with the
  operator-split values in brackets, in 12 test docstrings: items 2, 6, 7, 8, 9, 10 and 11, the
  energy budget, the MHD energy budget, div B, the gradient check and Phase D injection.
  - The older per-argument "Calibrated observed error" lines are left as history; the notes
    supersede them.
- **Figures regenerated** (from `regress_coupling/`, flag defaults, items 11 and 15 at 128^3):
  `pics/02` (item 2, plus the new A2.2 / A2.3 figures), `06` (item 6 and the four A6 /
  cr_entropy figures), `07`, `08`, `09` (jump, precursor), `10`, `11`, `16_17` (rollout
  stability), `18` (both energy budgets + fields png), `19`, and `phase_d_inference` (injection).
- **Unchanged and therefore not touched:** items 1, 3, 4, 5 and 15, the ring tests, diffusion
  rate T1-T3, and Phase D kappa (identical to the printed digits).
- **Not regenerated** (made by separate scripts / higher resolutions):
  `cr_modified_shock_structure_resolution.svg`, the `*_48/_96/_224.svg` variants of items 7/8.

### 2026-10-09: stage-wise CR coupling implemented (user OK); A2.2/A2.3 committed as tests; regression running

- **Code (uncommitted):**
  - `_time_integrator_sources.cr_grey_feedback_sources` (new) and
    `_time_integrator_sources(..., include_cr=True)`;
  - `evolve_state._gravity_source_presolve(..., include_cr=True)`;
  - in the unsplit scheme the CR sources are added per stage in
    `_evolve_gas_state_unsplit_inner` (before the positivity floor), and the operator-split
    presolve/apply covers gravity only.
  - DESIGN.md "Resolved: stage-wise CR-gas coupling".
- **The repo code reproduces the scratch prototype exactly** (A2.2 orders 1.85-1.87; A2.3a order
  1.86, two-fluid vs hydro 1.85; squeeze ringing 0.0026/0.0027; item 2 ringing 0.0014; A6
  numbers identical).
- **New test file** `pytests/cosmic_rays_grey/cr_adiabatic_coupling.py`, both tests pass:
  - `test_cr_linear_sound_wave` (A2.2): order >= 1.8, phase error < 1e-4, eigenvector error
    < 1e-3 (measured 1e-8), C_cfl 0.8 error < 1e-2 eps;
  - `test_cr_hydro_equivalence` (A2.3): order >= 1.8 vs exact and vs hydro, squeeze ringing
    <= 3x hydro.
  - Figures `pics/02_adiabatic_compression/cr_{linear_sound_wave,hydro_equivalence}_test.svg`.
- **Regression:** full CR suite (scratch copy `regress_coupling/`; GPUs through
  `scheduler.py`, which waits for free GPUs, as other users occupy most of them; 1D tests on
  CPU) plus M7 with the new coupling (tag `stagewise`: v_red 1000 seeds 42 and 43, v_red 1e4).
  Results below when done.
- **CPU results (all pass; operator-split -> stage-wise):**
  - item 1 L2/amp 6.035e-3 -> 6.034e-3;
  - item 2 invariant 1.52e-3 -> 9.8e-4, max rho/rho0 1.167 -> 1.145 (ringing gone);
  - item 4 order 1.555 -> 1.554;
  - item 6 MAE rho/u/P_th/e_cr 2.50/3.15/1.38/2.93e-3 -> 3.00/4.12/1.81/3.68e-3 (gate 1e-2; the
    post-shock partition error is a bit larger, as in A6.1: 1.06 -> 1.27%);
  - item 9 Test A 0.13-0.15% -> 0.17-0.22%, Test B du/dx 26% -> 21%;
  - A2.2/A2.3, item 5 and the shock-partition tests pass.
- **M7, v_red 1000 (cr_entropy off), operator-split -> stage-wise:**
  - seed 42: E_cr(250) 4.270e11 -> 4.286e11, E_cr/inj 1.410 -> 1.416, compression work
    2.83e10 -> 2.87e10, halo P_th 3.78e-2 -> 3.78e-2, eta 0.28 -> 0.27, H_gas 351 -> 351;
  - seed 43: 4.706e11 -> 4.701e11, 1.557 -> 1.555, 4.10e10 -> 4.08e10, 3.66e-2 -> 3.65e-2,
    0.22 -> 0.24, 336 -> 339.
  - All changes are below the seed scatter. M7's dt (hot gas, v_red) is far below the coupling
    oscillation timescale, so the operator-split error was small there.
  - **Correction:** the ~10% smooth-expansion CR error seen in the cr_entropy A/B is *not* the
    time coupling. It is more likely spatial (central -P_cr div u vs upwind advection in strong
    expansion). Open.
- **M7, v_red 1e4, seed 42, operator-split -> stage-wise:** E_cr(250) 5.231e11 -> 5.259e11, E_cr/inj
  1.746 -> 1.755, compression work 5.66e10 -> 5.75e10, boundary advection 1.70e11 -> 1.72e11, halo
  P_th 2.70e-2 -> 2.68e-2, eta 0.21 -> 0.20, v_out 3.3 -> 3.3, H_gas 280 -> 283. Also unchanged
  within ~1%, so the stage-wise coupling does not alter the M7 results.
- **GPU suite (stage-wise coupling, flag defaults):** everything passes except
  - `test_cr_ring_flux_cap_alone`: the expected failure (the standalone runner ignores xfail);
  - `test_cr_entropy_cold_rarefaction`: the opt-in cr_entropy test, re-calibrated (below).

  Metrics, operator-split -> stage-wise:
  - DSA item 8 KR13 / CS14 0.0877 / 0.0457 -> 0.0876 / 0.0457;
  - item 7 E_cr/E_tot 0.0434 -> 0.0434, partition identity 3.3e-5 -> 3.0e-6;
  - item 10 E_cr/E_tot 0.0438 -> 0.0438, identity 4.4e-4 -> 1.8e-4, Weaver checks identical;
  - item 11 0.0570 / 0.0532 -> same, identity 6-7e-5 -> 2.5-2.7e-5, clumpy effect -6.67% -> same;
  - item 15 identical;
  - div B 6.1 / 7.1e-15 -> 7.5 / 8.0e-15;
  - energy budgets 1.0e-10 / 9.8e-11 -> 1.0e-10 / 1.0e-10;
  - Phase D kappa 0.01974 -> 0.01974;
  - ring, diffusion rate T1-T3, gradient checks: pass.
- **`test_cr_entropy_cold_rarefaction` re-calibrated** (opt-in cr_entropy):
  - With the stage-wise coupling the energy scheme's start-up error drops: K_cr 9.1% -> 5.3%,
    energy drift 4.6e-4 -> 3.8e-8 (no more floor hits). cr_entropy without dual energy:
    0.89% -> 6.3e-5.
  - With dual energy: 0.75% -> 2.2% (energy scheme 2.6%). Less spurious start-up heat means the
    safeguard limits more.
  - New gates: 1e-3 / 3e-2 (no dual / dual), better than the energy scheme, energy drift < 1e-4.
- **Stale docstring numbers** (calibrated with the operator-split coupling; gates still pass):
  items 2, 6 and 9, A6.1-A6.3 (energy scheme now 1.27% at item 6), and the cr_entropy tests'
  "measured" values. To re-baseline once the user wants it (figures too).

### 2026-10-09: A2.2 + A2.3 -- the operator-split CR coupling is unstable; stage-wise coupling (Gupta et al. 2021 "Unsplit-pdv") fixes it

- **Ring cap-alone test** marked `xfail(strict=True)` (pytest optional; verified with pytest
  9.1.1 from a scratch install).
- **A2.2** (periodic 1D, v_red = 0, float64, eps = 1e-6 eigenmode, one period):
  - Coupling strength is right: phase speed within ~2e-5 of c_eff, eigenvector ratio
    dP_cr/P_cr / (drho/rho) = gamma_cr to 2e-4.
  - The error converges only at order 1.09 / 0.68 / -0.09 for P_cr/P_th = 0.5 / 2 / 10, against
    1.86 without CRs.
  - The error is a wave train at a fixed 8-12 cells, independent of N, growing exponentially
    in time (P_cr/P_th = 10, N = 1024: 0.08 eps at T/2, 26 eps at T).
  - It scales with C_cfl: first order in dt for C_cfl <= 0.2, unstable above (C_cfl 0.8: 2.7e2
    eps at P_cr/P_th = 2, 1e5 eps at 10).
  - Cause: `-grad P_cr` / `-P_cr div u` are presolved from the pre-step state and applied once
    after the whole RK2. That is a forward-Euler step of an oscillatory subsystem, which the
    HLL dissipation only partly damps.
- **A2.3:**
  - (a) Simple wave (gamma = 4/3, A = 0.1, t = 0.5 t_shock), P_th = 1e-6 P_cr vs gamma = 4/3
    hydro vs the exact characteristic solution: L1 two-fluid vs exact *grows* with N
    (order -0.31, 1.5e-2 at N = 1024); hydro order 1.87.
  - (b) Item-2 squeeze in the CR-only limit: ringing 0.080 / 0.355 (N 512 / 2048); hydro
    gamma = 4/3 0.0012. So the item-2 ringing comes from the coupling, not from the gas scheme or
    the boundaries.
- **Prototype stage-wise coupling** (scratch monkeypatch: the same source, evaluated on each
  RK stage's start state with the stage dt, inside `_evolve_gas_state_unsplit_inner`; no
  post-RK application). This is Gupta, Sharma & Mignone (2021)'s Et+Ecr "Unsplit-pdv", the
  method they found least sensitive to numerical details.
  - A2.2: order 1.85 / 1.86 / 1.87, C_cfl 0.8 stable (3.9e-4 eps at P_cr/P_th = 10).
  - A2.3a: two-fluid vs exact 7.1e-6 at N = 1024 (order 1.86), vs hydro order 1.85.
    Needs the stage's pressure floor after the source (NaN at N <= 128 otherwise, P_th = 1e-6
    P_cr).
  - A2.3b: ringing 0.0025.
  - Item 2 (v_red = 0, N = 2048): ringing 0.30 -> 0.0013; max abs(K_cr - 1) 1.2e-2 -> 2.3e-6;
    max abs(K_th - 1) 9.4e-2 -> 1.4e-7. Both fluids are adiabatic in smooth flow.
  - A6 shock partition: unchanged in kind (item 6 1.26-1.31% over C_cfl 0.1-0.8, Gupta A 6.8%,
    M 10 12.9%), less CFL-dependent. As Gupta et al. say, the post-shock partition still
    depends on the method.
  - Energy conservation is kept (the source pair telescopes at every stage).
- **Proposal (user decision; fix to an existing scheme):**
  - Move the CR-grey feedback sources from the operator-split presolve/apply into the RK stages
    of `_evolve_gas_state_unsplit` (the well-balanced-gravity path already does this for
    gravity), before the stage's positivity floor.
  - Then re-check items 1-11, 15, 18 and M7. The M7 smooth-flow CR error (~10% of the CR
    injection) should shrink, which may also change the cr_entropy picture.

### 2026-10-09: M7 with the shock-only transfer -- no drain, but CR energy is created in the expanding halo

Runs: `--cr-entropy=shocks`, `--res=0.5`, 250 Myr; the off and global runs are the earlier ones.
Analysis: `/export/scratch/nknoell/phaseA/m7ab/ab3_analyze.py`. Late = 150-250 Myr.

| run | mode | E_cr(250) | E_cr/inj | energy created (discarded) | CR adv. in through z | halo median P_th | eta | v_out | H_gas |
|---|---|---|---|---|---|---|---|---|---|
| v_red 1000, seed 42 | off | 4.27e11 | 1.41 | -- | 9.9e10 | 3.8e-2 | 0.28 | 3.1 | 351 |
| | global | 3.48e11 | 1.15 | -- | 7.7e10 | 7.3e-5 | 0.38 | 3.9 | 358 |
| | shocks | 5.72e11 | 1.89 | 3.6e10 (12% of CR inj.) | 1.7e11 | 3.6e-2 | 0.25 | 3.1 | 345 |
| v_red 1000, seed 43 | off | 4.71e11 | 1.56 | -- | 1.3e11 | 3.7e-2 | 0.22 | 3.1 | 336 |
| | global | 3.22e11 | 1.07 | -- | 6.3e10 | 6.9e-5 | 0.40 | 4.2 | 360 |
| | shocks | 5.04e11 | 1.67 | 3.3e10 (11%) | 1.3e11 | 3.5e-2 | 0.26 | 3.0 | 367 |
| v_red 1e4, seed 42 | off | 5.23e11 | 1.75 | -- | 1.7e11 | 2.7e-2 | 0.21 | 3.3 | 280 |
| | global | 3.48e11 | 1.16 | -- | 7.1e10 | 3.3e-6 | 0.44 | 4.4 | 323 |
| | shocks | 6.21e11 | 2.07 | 4.3e10 (14.5%) | 2.0e11 | 2.8e-2 | 0.24 | 3.3 | 323 |

- **No drain:** shock-only keeps the halo gas intact, and eta, v_out and H_gas are close to
  the energy scheme. H_gas differs by -6 / +31 / +43 pc against a seed scatter of ~15 pc, so
  the v_red 1e4 difference is not settled with one seed.
- **But energy is created:** 11-15% of the CR injection, ~1-1.3% of the total SN energy
  (1e51 erg thermal + 1e50 erg CR per SN). In the expanding halo the energy scheme turns
  ~10% of the CR energy into spurious gas heat. Shock-only resets the CRs to the exact adiabat
  but leaves that heat in the gas, so the energy is counted twice.
  - The CR budget gets worse (E_cr/inj 1.67-2.07 vs 1.41-1.75 off), and so does the boundary
    inflow.
- **Side result:** in M7's smooth expansion the energy scheme's own CR error is ~10% of the
  CR injection, numerically moved into gas heat. That is larger than its shock error.
- **Verdict for M7:** neither entropy variant beats the energy scheme as is. Global drains the
  thin gas; shock-only double-counts.
  - The literature's complete version (Semenov et al. 2021, Sec. 2.2) also evolves the gas by
    its entropy away from shocks, so the spurious gas heat would not be kept either. Both
    fluids would then be adiabatic in smooth flow, and the energy-scheme exchange would be
    dropped as truncation error.
  - That needs gas dual energy with the "entropy except in shock zones" selection (ours uses
    entropy only in cold, supersonic cells). User decision.

### 2026-10-09: cr_entropy shock-only transfer (opt-in) -- 1D results; M7 runs queued

- **Code (uncommitted):**
  - `CR_ENTROPY_TRANSFER_GLOBAL` / `CR_ENTROPY_TRANSFER_SHOCKS`;
  - `CosmicRayGreyConfig.cr_entropy_transfer` and `cr_entropy_shock_dilation` (2);
  - `CosmicRayGreyParams.cr_entropy_shock_threshold` (0.5);
  - `cr_grey_sources.cr_entropy_shock_mask` (Gupta et al. 2021 detector, N-D);
  - hook in `_evolve_gas_state_unsplit` through the existing `gas_energy_cells`;
  - M7 pilot `--cr-entropy=shocks` (suffix `_crentshock`).
  - Flag off: bitwise identical to `54c4655` (four setups).
- **New test** `test_cr_entropy_shock_only` passes (CPU, about 1 min).
- **1D results (off / global / shocks):**

  | test | off | global | shocks |
  |---|---|---|---|
  | post-shock dK_cr,2: item 6 (C_cfl 0.1 / 0.4 / 0.8) | 1.21 / 1.06 / 0.83% | <= 4e-14 | <= 4e-14 |
  | post-shock dK_cr,2: Gupta A / B, M 3 / 10 / 100 | 6.4% / 10.6%, 6.0 / 10.6 / 11.2% | <= 7e-6 | <= 5e-6 |
  | plateau dP_th,2 (worst) | up to 10.5% | 9.6e-4 | 1.9e-3 |
  | closed-box energy drift (item 6 / Gupta A) | 4e-14 / 2e-14 | 4e-14 / 3e-14 | -9.9e-5 / +8.3e-5 |
  | item 2 at N 512 / 2048: rho ringing | 0.044 / 0.27 | 0.136 / 0.34 | 0.019 / 0.17 |
  | item 2 at N 512 / 2048: max abs(K_th - 1) | 1.5e-3 / 0.09 | 0.34 / 0.82 | 4.5e-4 / 0.019 |
  | cold rarefaction: max abs(K_cr - 1); energy drift | 9.1e-2; 4.6e-4 | 8.9e-3; 2.1e-4 | 7e-15; 2.7e-3 |
  | finite-kappa tube (v_red 10, kappa 1/300): dK_cr,2 | 7.65% | 7.86% | 7.66% |

  The finite-kappa row is CR diffusion across the shock, so the adiabatic reference does not
  apply. The variants agree, so the shock-only mode does no harm with diffusion.
- **M7:** three runs queued with `launch_when_free.sh` (v_red 1000 seeds 42 and 43, v_red 1e4
  seed 42, all `--cr-entropy=shocks`). It waits for a GPU with no compute processes: another
  user had all ten busy. The wrapper now logs the CR loss and the gas gain of the transfer
  separately, so the discarded energy is measured. Analysis: `ab3_analyze.py`.

### 2026-10-09: cr_entropy default reverted to off (user decision)

- `CosmicRayGreyConfig.cr_entropy = False` again; flag docstring updated.
- The M7 pilot is back to the opt-in `--cr-entropy` (suffix `_crent`).
- Opt-outs added for the default (CWB `_cwb_setup.py`, `cr_streaming_1d`, `cr_energy_budget`,
  `cr_mhd_energy_budget`) reverted to `a5c0968`.
- Figures of items 1-4, 6-11 and 15 back to their flag-off versions: items 7, 8, 10, 11 and 15
  from `a5c0968`, the rest from HEAD (never committed with the flag on). The re-baseline
  notes are removed from the 13 test docstrings. The metric prints added for the re-baseline
  stay; they do not depend on the flag.
- Kept: the `cr_entropy` implementation (steps 1-3), its tests in `cr_shock_tube_partition.py`
  (flag set explicitly), and the log entries below as history.
- DESIGN.md section renamed back to "Open: conservative CR entropy at shocks".

### 2026-10-09: is the cr_entropy default worth it? Analysis -> the energy-conserving transfer is harmful in CR-dominated gas

- **Literature:**
  - Gupta, Sharma & Mignone (2021), Sec. 5.1-5.2: their "Et+Scr" (total energy + CR entropy,
    gas = remainder; our scheme family) "fails to maintain the pressure balance mode"
    (spurious waves).
  - Kudoh & Hanawa (2016) reduce those waves only with resolution or extra diffusion. Gupta et
    al. also argue that constant CR entropy across shocks is not physically justified (DSA).
  - Semenov, Kravtsov & Diemer (2021, Sec. 2.2) always follow CRs by entropy but take the gas
    thermal energy from the total energy only in detected shock zones (gas entropy elsewhere),
    and accept that strict energy conservation is lost.
- **Item 2 resolution study** (P_cr/P_th = 33; off / on / shock-only prototype = transfer only
  in Gupta-type shock zones, div u < 0 and total-pressure jump >= 0.5, widened 2 cells):
  - Ringing in rho: N 256 0.017 / 0.033 / 0.011; 512 0.044 / 0.136 / 0.019; 1024 0.20 / 0.42 /
    0.037; 2048 0.27 / 0.34 / 0.17.
  - Gas entropy error max abs(K_th - 1): N 512 1.5e-3 / 0.34 / 4e-4; N 2048 0.09 / 0.82 / 0.02.
  - CR entropy error: N 2048 9.9e-3 / 2.9e-2 / 1.6e-3.
  - The ringing is a pre-existing energy-scheme instability that grows with N. "on" amplifies
    it and wrecks the gas entropy; shock-only is best on every metric at every N.
- **Shock-only prototype on the shock tests:** post-shock K_cr exact (<= 1e-10) like "on";
  plateau P_th errors <= 1.1e-3. Energy no longer exact: drift 9e-5 in the reflecting box, 2.7e-3
  in the cold rarefaction (K_cr there exact).
- **M7 (late, >= 150 Myr):**
  - 80-85% of the volume has P_cr/P_th > 30, the halo median is 110-200 with the flag off.
  - With the flag on, the halo median P_th collapses by x1.9e-3 (v_red 1000) and x1.2e-4
    (v_red 1e4), and T_halo with it. Halo rho and e_cr change little; the disc is unchanged.
  - The transfer drains the expanding halo gas. The 50%-per-step safeguard does not limit the
    cumulative drain.
  - **So the M7 A/B differences (eta x1.4-2, less infall, E_cr -19..-33%) cannot be credited to
    the shock fix.**
- **Conclusion:**
  - The energy-conserving variant of cr_entropy (current default) is not acceptable for
    CR-dominated gas, which is M7's halo.
  - The shock benefit itself is real (A6.1-A6.3; DSA tests ~5%), but needs a transfer
    restricted to shocks (Semenov-style), not the global one.
  - Recommendation to the user: revert the default to off now. Optionally implement and test
    the shock-only variant (N-D shock mask, e.g. the existing `_dual_energy_shock_mask`), then
    redo item 2, A6 and the M7 A/B before deciding again.

### 2026-10-09: re-baseline of items 1-4, 6 and 9 for the cr_entropy default

- **How:** flag off / on from the same code. Items 1 and 2, the oblique item-3 test, the
  isotropic item-4 test and item 6 were rerun (`/export/scratch/nknoell/phaseA/rb3_{on,off}/`);
  these five tests now print their gated metrics. Ring, diffusion rate (T1-T3) and item 9 reuse
  the step-1 (off) and step-3 flag-on suite runs (same code paths). All pass in both. Figures in
  `pics/01-04, 06, 09` regenerated; docstrings carry both sets of numbers.

  | item | metric | off | on |
  |---|---|---|---|
  | 1 advection | L2/amp; peak loss | 6.035e-3; 3.715e-2 | 6.036e-3; 3.715e-2 |
  | 2 adiabatic compression | invariant error; ringing amplitude in rho | 1.52e-3; 0.044 | 1.89e-3; **0.136** |
  | 3 oblique | leak ratio | 2.182e-2 | 2.182e-2 |
  | 3 ring 3a, N 50 / 400 | kappa_perp,num/kappa_par | 4.08e-2 / 3.26e-3 | 4.00e-2 / 3.26e-3 |
  | 3/4 diffusion rate T1-T3 | all D/kappa | -- | identical to printed digits |
  | 4 isotropic | L2 at N 128-1024; order | 1.69e-3 ... 1.27e-4; 1.555 | same; 1.555 |
  | 6 shock tube | MAE rho / u / P_th / e_cr | 2.50 / 3.15 / 1.38 / 2.93e-3 | 2.50 / 3.45 / 1.20 / 2.49e-3 |
  | 9 modified shock | Test A rho/u/P err; Test B du/dx | 0.13/0.15/0.15%; 26.1% | 0.09/0.13/0.08%; 19.7% |

- **Finding (item 2): the flag triples the wave-train ringing in strongly CR-dominated
  compression.** Diagnosis (1D, item-2 setup, N = 512):

  | precision | P_th0 | v_red | rho ringing off -> on | max abs(K_cr - 1) off -> on | P_th ringing off -> on |
  |---|---|---|---|---|---|
  | f32 and f64 (identical) | 0.01 | 0.1 | 0.044 -> 0.136 | 1.5e-3 -> 2.1e-3 | 7.9e-4 -> 5.6e-3 |
  | f64 | 0.01 | 0 | 0.056 -> 0.212 | 1.7e-3 -> 3e-14 | 1.0e-3 -> 7.8e-3 |
  | f64 | 0.3 | 0.1 | 0.0004 -> 0.0002 | 5.5e-4 -> 5.6e-4 | 2.0e-4 -> 1.1e-4 |

  - The ringing exists in both schemes (the open item-2 diagnostic, plan Sec. 2.3 / A2.3).
  - With P_cr/P_th = 33 the energy scheme's truncation-level CR error (~1.5e-3 of e_cr) is
    handed to the thin gas to keep total energy exact: ~50% of P_th in the ringing zone. That
    amplifies the wave trains, the same trade-off as the cold rarefaction.
  - At P_cr/P_th ~ 1 the flag reduces the ringing.
  - **Needs a user decision:** accept it as a known limitation for P_cr/P_th >> 1, or look for
    a remedy (e.g. limit the transfer per step relative to the gas thermal energy in both
    directions, at the price of exactness at shocks in very CR-dominated gas).

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
  "Open: conservative CR entropy at shocks" -> Status.
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
