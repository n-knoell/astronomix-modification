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
| A1.3 CR-thermal contact in pressure balance | 1 | done; fixed for HLLC by P2 (total-pressure HLLC flux) | `cr_contact_pressure_balance.py` (1D + 3D, hydro + MHD) | HLLC + P2: **exact** (gate 1e-10; 1D <= 1e-13, 3D hydro / MHD <= 6e-14). HLL (default, no P2): still ~1e-2 at N 64, order ~0.5-0.75, tracked bounds (1D). Cause was the vdp source coupling at the discontinuous P_cr (Gupta+21 Fig. 3) | 2026-10-10 |
| A1.4 free-wave convergence + eigenvector | 3 | todo | | | |
| A1.5 free-streaming limit, production config | 3 | todo | | | |
| **Item 2: adiabatic compression** | | | | | |
| A2.1 ballistic homologous compression/expansion | 3 | todo | | | |
| A2.2 CR-modified linear sound wave | 1 | done; fixed by the stage-wise coupling | `cr_adiabatic_coupling.py` | **fails with the operator-split coupling**: order 1.09 / 0.68 / -0.09 at P_cr/P_th 0.5 / 2 / 10, an instability at 8-12 cells, unstable above C_cfl ~0.2. Stage-wise coupling prototype: order 1.85-1.87, stable at C_cfl 0.8 | 2026-10-09 |
| A2.3 CR-dominated fluid vs gamma=4/3 hydro | 1 | done; fixed by the stage-wise coupling; under the 2026-10-13 default (HLLC) two-fluid vs hydro order 1.70 (gate split, 1.6) with a 20x smaller gap | `cr_adiabatic_coupling.py` | item-2 ringing is **not** in gamma=4/3 hydro (0.0012), only in the operator-split two-fluid run (0.36 at N 2048). Stage-wise: 0.0026, and the simple wave matches hydro at order 1.85 | 2026-10-09 |
| A2.4 two-moment linear dispersion relation | 2 | todo | | | |
| **Item 3: anisotropic diffusion** | | | | | |
| A3.1 Sovinec steady state | 2 | done (with A3.5); led to the van Leer CR limiter + guard retune (2026-10-13, misfire fixed) | `cr_sovinec.py` | **since 2026-10-13 (van Leer, default):** 3.4e-3 / 1.1e-3 / 2.2e-4 / 3.5e-5 at 16-128 cells per unit length, orders 1.6-2.7; guard on = guard off within 0.4%. Before: `kappa_perp,num/kappa_par` (MINMOD, guard off) 2.0e-2 / 8.9e-3 / 3.4e-3 / 1.0e-3 at 16-128 cells per unit length, orders 1.2-1.7; MC 1.2e-3 ... 1.1e-5 (orders 1.9-2.6); field-aligned / projected speeds 2-4x worse in thick cells; additive to physical `kappa_perp` (1e-2: 2.0 / 0.91 / 0.32 / 0.095 of it); steady-state anchor agrees to <= 3.2%; guard on: leak 2x, symmetry broken | 2026-10-13 |
| A3.2 anisotropic Gaussian, full profile | 2 | todo | | | |
| A3.3 absolute gates + early-time ring | 3 | todo | | | |
| A3.4 production (guard off) tracking | 3 | todo | | | |
| A3.5 `v_red` dependence of `D_perp,num` | 2 (with A3.1) | done | `cr_sovinec.py` (`test_sovinec_vred_scan`) | leak = HLL dissipation: ~`v_red` in thin cells (slope 0.5-1.1 at `tau` <= 0.5), `v_red`-independent at `tau` >~ 2 (slope <= 0.09 face-normal) | 2026-10-12 |
| **Item 4: isotropic diffusion** | | | | | |
| A4.1 exact telegrapher reference | 1 | done (1D); F_cr relaxation fixed (ETD2RK, 2026-10-10) | `cr_telegrapher_reference.py`, `linear_theory.py` | Gaussian vs telegrapher order 1.93 (N 128-2048), pairwise 2.00 to N 4096 (1.98e-6), no floor; item-4 floor = model (1.08e-4, slope -2.008 in v_red); single modes second order (1.89-2.00 at 1024 -> 2048), C_cfl-independent; stage-wise relaxation (old) was first order, kept as a comparison test | 2026-10-10 |
| A4.2 `v_red` scan of the model floor | 1 | done; finding: MINMOD makes CR diffusion first order in optically thick cells; limiter: option a (MINMOD kept, 2026-10-12), superseded 2026-10-13 by van Leer on the CR rows | `cr_telegrapher_reference.py` (`test_vred_scan_*`) | error vs diffusion = floor (v_red^-2) + scheme error; slope -1.975 at N 8192 (gate -2 +- 0.3), -1.31 at N 2048 (scheme error ~v_red overtakes floor near v_red 16: the old "v_red^-1 floor", Sec. 4.3 step 3 closed); MINMOD scheme error ~v_red dx^2 (thin) / ~kappa dx, first order (tau >~ 2); MC: second order, 15x smaller, ~v_red-independent; HLLC identical to HLL; limiter: option a, MINMOD kept (relative error 0.6-2.4% in thick cells, << D1/D2) | 2026-10-12 |
| A4.3 multi-D isotropic Gaussian | 3 | todo | | | |
| A4.4 steady-state diffusion with source | 2 | done | `cr_steady_state_source.py` (1D + 2D) | steady state dt-independent to <= 4e-13 (C_cfl 0.4/0.2/0.1; ETD2RK and stage-wise; thin and thick); MINMOD second order in thin cells (2.00 at N 512->1024, all v_red), first order thick (1.14 at tau 3.3->1.7); error ~v_red^0.92 at fixed N; MC 2.1-2.4, 4-20x smaller; v_red dependence converges at 1.99; 2D diagonal mode 1.99 | 2026-10-12 |
| A4.5 passive energy conservation | 3 | todo | | | |
| **Item 6: shock tube** | | | | | |
| A6.1 post-shock `K_cr` diagnostic + convergence | 1 | done; fixed by the default (D2) | `cr_shock_tube_partition.py` | `K_cr,2` +1.27% at every N (100-3200; stage-wise coupling), 1.26-1.31% over C_cfl/limiter, HLLC + P2 0.42%: **not converging**, method-dependent. **Default since 2026-10-13 (HLLC + hybrid): 8.6e-5 at N 200-1600**, 3.9e-4 at C_cfl 0.8 | 2026-10-13 |
| A6.2 Gupta et al. (2021) tubes A and B | 2 | done; fixed by the default (D2) | `cr_shock_tube_partition.py` | stage-wise coupling, HLL: A (CR-dominated, M 2.6): `K_cr,2` +6.83%, `P_th,2` -11.3%, spurious 21% of e_th; B (M 9.9): +12.9% (N >= 2000). Operator-split values (2026-10-08): +6.35% / -10.5% / 19% / +10.6%. **Default (HLLC + hybrid, N 4000): A +1.3e-4, spurious 3.7e-4; B +7.2e-6** | 2026-10-13 |
| A6.3 Mach-number scan (Pfrommer et al. 2006) | 2 | done; fixed by the default (D2) | `cr_shock_tube_partition.py` | stage-wise coupling, HLL: `K_cr,2` error 0.48% (M 1.4) -> saturates 13.6% (M >= 30); spurious/e_th peaks 5.0% at M 3, ~M^-2 above; C_cfl 0.1 / 0.8: -2 / +5% relative. Operator-split (2026-10-08): 0.4% -> 11.2%, peak 4.2%. **Default (HLLC + hybrid, N 3200): <= 1.3e-4 at every M**, spurious <= 2.3e-4 | 2026-10-13 |
| A6.4 reference-solver regression tests | 2 | todo | | | |
| A6.5 conservation in the shock tube | 3 | done | `cr_shock_tube_partition.py` | mass <= 2e-16, E_gas + e_cr <= 5e-14 (reflecting box, both schemes) | 2026-10-08 |
| A6.6 finite-`kappa` two-moment shock tube | 3 | todo | | | |
| A6.7 remedy evaluation (decision) | -- | done | DESIGN.md | option a (CR entropy, Semenov et al. 2021 eq. 8 + Et+Scr energy bookkeeping); design note written | 2026-10-08 |
| **Cross-cutting** | | | | | |
| X1 baseline registry, stale docstrings | 1 | done | `pytests/cosmic_rays_grey/cr_baselines.py`, `baselines.json` | one registry for the 12 Phase A files (28 tests, 118 metrics); docstrings point to it; refresh with `CR_UPDATE_BASELINES=1`; GPU reruns reproduce it to its 4-digit rounding | 2026-10-12 |
| X2 tighten loose gates | 1 | done | the 12 Phase A files | errors at the finest N within 1.5x of the registry (was up to 8-10x), orders within 0.1-0.15, localized quantities: item 1 `F/e = lambda` and the speed, item 6 plateau values and front positions (box mean was blind to the partition); fixed-order item-4 gate replaced by per-N errors (A4.1 gates the order) | 2026-10-12 |
| X3 multi-resolution runs | -- | todo | | covered per test | |
| X4 production-config variants | -- | todo | | covered per test | |
| X5 float64 for analytic tests | -- | todo | | covered per test | |
| X6 static heavy gas for pure transport | -- | running | `cr_static_gas.py` | helper used by A4.4, A3.1; ring test not migrated yet | 2026-10-12 |
| X7 FV only | -- | dropped | -- | out of scope (plan item 20) | |
| Mark ring "cap alone" test `xfail` (Sec. 3.1) | -- | done | `cr_anisotropic_ring.py` | `pytest.mark.xfail(strict=True)` when pytest is importable (the venv has none); standalone runs print XFAIL; verified with pytest 9.1.1 from a scratch install | 2026-10-09 |
| Fix Pfrommer et al. 2006 citation (Sec. 6.2) | -- | todo | `cr_shock_tube.py`, DESIGN.md | | |

## Decisions

| ID | Question | Status | Outcome |
|---|---|---|---|
| D1 | Absolute `kappa_perp,num/kappa_par` budget for item 3? | open; data from A3.1 (2026-10-12), default changed 2026-10-13 | measured (Sovinec, 16-128 cells per unit length): van Leer (default since 2026-10-13) 3.4e-3 ... 3.5e-5, MINMOD 2e-2 ... 1e-3, MC ~3x below van Leer; S&H ring N 50-400: 0.015 ... 0.00061 |
| D2 | Shock partition: change CR energy formulation or prescribe it? | **decided 2026-10-13** (user): HLLC + hybrid CR entropy is the grey-CR default | hybrid = energy scheme everywhere, `e_cr := e(s_cr)` only in shock zones (Gupta et al. 2021 detector); post-shock `K_cr` ~1e-4 (HLL energy scheme 0.4-13.6%); exact energy, contacts, mixing; M7 not validated (sets its scheme explicitly) |
| D3 | Hard gates on library default (guard on) or production (guard off)? | open; the A3.1 guard misfire (2026-10-12) is fixed by the five-face sensor (2026-10-13) | A3.1 gates guard on and off since 2026-10-13 (equal within 0.4%) |
| D4 | P2 (total-pressure HLLC) with gravity: (a) exclude gravity runs or (b) well-balanced P_th + P_cr reconstruction? | **decided 2026-10-14** (user): (a), automatic | `cr_total_pressure_flux` is False with self-gravity or an external potential; those runs keep the source coupling. CR-supported column core Mach 0.0098 / 0.0138 (N_z 128 / 256) vs 0.137 with P2 |

## Infrastructure

- [x] Test-only `e_cr` source hook (A4.4, A3.1 steady-state anchor): inside the RK stage RHS, not
      in `_time_integrator_sources` (spec in the 2026-10-12 A3.1/A4.4 log entry). Done 2026-10-12:
      `CosmicRayGreyParams.test_e_cr_source` (see log)
- [ ] Static-heavy-gas helper, factored out of `cr_anisotropic_ring.py` (X6): `cr_static_gas.py`
      written 2026-10-12 (A4.4, A3.1 use it); `cr_anisotropic_ring.py` still has its own copy
- [ ] `test_setups/reference_solutions/linear_theory.py`: 5x5 dispersion matrix, eigenvectors,
      spectral telegrapher solver (A1.4, A2.2, A2.4, A4.1). Telegrapher solver done 2026-10-10
      (1D/2D/3D, verified against expm to 1e-15); the rest open.
- [ ] Convergence-fit helper: least-squares and pairwise orders
- [ ] `baselines.json` next to `pics/` (X1)

## Log (newest first)

### 2026-10-14: D4 decided -- P2 off with gravity (option a, automatic)

- **Code:** `cr_grey_transport.cr_total_pressure_flux` returns False when
  `gravity_config.self_gravity or gravity_config.external_potential`. All callers (`hll`,
  `evolve_state`, `_time_integrator_sources`) use the predicate, so gravity runs get the source
  coupling (-grad P_cr, -v.grad P_cr, centred -P_cr div v) with HLLC, as before 2026-10-10.
  `resolve_auto_defaults` is unchanged (HLLC + hybrid stays the default; the hybrid also works
  with the source coupling).
- **Check** (scratch `/export/scratch/nknoell/m7_rerun/scripts/cr_hydrostatic_check.py`: the
  `stratified_hydrostatic_column.py` potential, 4 x 4 x N_z, P_cr = 0.9 P_tot, v_red = 0, HLLC,
  t = 4, float64; max core |v| / sqrt(gamma C2)):

  | | N_z 128 | N_z 256 |
  |---|---|---|
  | CR, new rule (no P2, hybrid entropy) | 0.0098 | 0.0138 |
  | CR, P2 forced on | 0.137 | -- |
  | thermal only | 0.110 | -- |

  Same as 2026-10-10 (0.011 / 0.015, P2 0.147 / 0.142, thermal 0.110).
- **Affected:** only runs with grey CRs and gravity (M5, M6, M7); no committed pytest combines
  them, so the regression suite is unchanged.

### 2026-10-13: HLLC + hybrid CR entropy as the grey-CR default (user decision) -- regression done, all pass

- **Code (uncommitted):**
  - `simulation_config.RIEMANN_SOLVER_AUTO` (the new default of `riemann_solver`) and
    `resolve_auto_defaults`, called by `finalize_config` and `get_registered_variables`;
  - `CosmicRayGreyConfig.cr_entropy = None` (auto) and
    `cr_entropy_transfer = CR_ENTROPY_TRANSFER_SHOCKS`.
- **Resolution checked** (`auto_default_check.py`):
  - no CRs -> HLL;
  - CRs -> HLLC with the hybrid on;
  - CRs + streaming -> HLLC, hybrid off;
  - split -> HLL, hybrid off;
  - explicit settings kept; explicit `cr_entropy=True` with streaming raises.
- **Tests pinned to their calibrated scheme:**
  - A6.5 and the cold rarefaction: HLL + global transfer;
  - A6.1-A6.3: the energy scheme with HLL stays as the control (dashed), the default is the
    new main curve (solid), with gates `abs(dK_cr,2) < 1e-3` and registry metrics `default_*`.
- **Regression** (update mode, every grey-CR test file, GPU only): split over two clusters,
  `/export/scratch/nknoell/phaseA/dflt` (A6 partition, item 1, item 2, item 9, streaming, item 4,
  gradient, Phase D, oblique, div B) and `/export/scratch/nknoell/phaseA/hllc_hybrid_update`
  (everything, incl. items 3, 7, 8, 10, 11, 13-15, 18, ring, Sovinec `SOVINEC_VARIANTS=8`).
  Logs in each `logs/`. Overrides as in earlier full runs: `test_cr_anisotropic_3d` with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`, pion bump `CR_NUM_CELLS=128`, clumpy medium at 128^3
  (scratch copy).
  - **All pass except** (first pass):
    - `test_cr_hydro_equivalence` (A2.3a): two-fluid vs hydro order **1.70 < 1.8**, also with
      both runs on HLLC (so the solver mismatch was not the cause). Two-fluid vs exact keeps
      order 1.87 (8.3e-6 at N 1024), the same as hydro. **Not the hybrid:** with
      `cr_entropy=False` forced (verified: 5 instead of 6 state rows) the result is identical
      to all printed digits (`hllc_hybrid_update/diag_hydro_eq/diag.log`, GPU 0, 2026-10-13
      17:49), so the shock mask never fires in this wave. With HLL in both runs it was 1.85,
      so the 1.70 comes from HLLC (two-fluid with the total-pressure flux vs gamma = 4/3
      hydro HLLC are different discretizations). The gap itself is 20x smaller than with HLL
      (2.4e-7 vs 4.9e-6 at N 1024), squeeze ringing 0.0027 -> 0.0015. **Resolved (user):**
      separate gate `min_order_vs_hydro = 1.6` (vs exact stays 1.8); rerun on GPU 0 passes,
      registry entry and figure updated (`/export/scratch/nknoell/phaseA/hydro_eq_rerun`).
    - `test_cr_ring_flux_cap_alone`: the strict xfail, as designed.
  - **Fixed during the run** (test-side pins, scheme unchanged): the test-only `e_cr` source
    hook does not support `cr_entropy`, so `cr_steady_state_source.py` and `cr_sovinec.py`
    set `cr_entropy=False` (static gas, nothing to gain); `cr_energy_budget.py` sets
    `cr_entropy=False` in both runs, since the streaming run cannot use it and the comparison
    must be scheme against scheme; item 6 front gate 1.0 -> 1.5 cells (shock +1.03 cells, a
    fixed start-up offset in cells, first order in x).
  - **Better with the default:**
    - item 6 shocked plateau: `e_cr` 1.7e-2 -> 3.5e-4, `P_th` 9.3e-3 -> 2.9e-4, `rho`
      3.8e-3 -> 3.7e-4;
    - A6.1-A6.3: `abs(dK_cr,2)` <= 1.3e-4 for every tube and N >= 200 (energy scheme
      0.48-13.6%), spurious/e_th <= 3.7e-4 (was up to 21%); 3D Sedov shell 1.4e-2 -> 1.9e-4;
    - item 2 invariant error 9.8e-4 -> 8.8e-4.
  - **Worse with the default** (all within their gates):
    - item 9 Test A post-shock errors 0.17 / 0.21 / 0.22% -> 0.30 / 0.49 / 0.53% (gate 2%),
      Test B du/dx 24.2% -> 32.9% (gate 40%); HLLC and the hybrid not separated;
    - A2.2 linear wave at `C_cfl = 0.8`: 3.9e-4 ... 1.1e-3 -> 1.45e-3 eps (finest N at
      `C_cfl = 0.4` unchanged, 4.3e-4);
    - item 6 left star `P_th` 3.2e-4 -> 1.4e-3, mean `u` error 4.1e-3 -> 4.5e-3;
    - A6.3 left-plateau control `dK_cr,L` up to -2.3e-4 at M >= 30 (energy scheme <= 7e-7).
  - **Unchanged** (registry entries identical or at the 1e-4 relative noise level): items 3, 4,
    A4.1/A4.2/A4.4, A1.1, A1.3, Sovinec, ring. Items 7, 8, 10, 11, 13-15, 18, 19, gradient
    checks and Phase D have no registry entries; they pass.
- **Merged:** `baselines.json` (15 entries re-recorded; the two entries both clusters
  recorded agree exactly; `_meta` updated by hand); 42 figures copied into `pics/` (all but
  `cr_hydro_equivalence_test.svg`, added after its rerun). Docstrings re-measured: `cr_shock_tube_partition.py`
  (default-scheme results), `cr_adiabatic_coupling.py`, `cr_modified_shock_structure.py`,
  `cr_adiabatic_compression.py`, `cr_advection.py`, `cr_shock_tube.py`.

### 2026-10-13: D2 restarted -- literature check, measurements, and a hybrid prototype that fixes the partition (no repo code change)

Refined plan and literature: DESIGN.md "Open: conservative CR entropy at shocks", "Refined plan
(2026-10-13)". Scratch scripts: `d2_measure.py` (A6.1-A6.3 tubes), `d2_expansion.py`,
`d2_mixing.py`, `d2_hybrid_patch.py` (monkeypatch) in the session scratchpad. Logs:
`/export/scratch/nknoell/phaseA/d2/logs/`. 1D, float64, GPU 0 (UUID-pinned, free at launch).

**1. Energy scheme, HLLC + P2 (Gupta et al. 2021's "Unsplit-pdv" family) vs HLL**
(`dK_cr,2` = post-shock CR entropy error; spurious = spurious CR energy / post-shock `e_th`;
N = 1600 for the Mach scan; HLL = the current default).

| tube | HLL | HLLC + P2 | HLLC + P2, spread over C_cfl 0.1-0.8 and MINMOD / MC / van Leer |
|---|---|---|---|
| item 6 (M 1.64) | +1.27% / 1.5% | +0.42% / 0.5% | 0.19-0.47% |
| Gupta A (M 2.59) | +6.83% / 20.6% | +2.59% / 7.6% | 2.22-2.94% |
| Gupta B (M 9.89) | +12.9% / 0.8% | +6.19% / 0.4% | 5.95-7.06% |
| M 1.4 / 2 / 3 | 0.48 / 2.98 / 7.09% | 0.12 / 1.12 / 3.10% | M 2: 0.76-1.26% |
| M 10 / 100 | 12.9 / 13.6% | 6.2 / 6.6% | M 10: 5.96-7.22% |

HLLC + P2 halves the error and narrows the method spread, but it is neither small nor
method-independent. HLL with van Leer (gas limiter) equals HLL with MINMOD within 1%.

**2. The M7 "smooth-expansion CR error" is not in the energy scheme.**
- 1D homologous expansion (`v = x - 1/2`, t = 0.5, open boundaries), `P_cr / P_th` = 100, inner
  half of the box:
  - subsonic: `|dK_cr|` 1.5e-5 -> 1.5e-8 (N 64 -> 1024), second order;
  - supersonic (edge Mach ~10): `|dK_cr|` 8.6e-6 -> 3.4e-8;
  - identical for HLL and HLLC.
  - The gas carries the error instead: `|dK_th|` 0.32 -> 1.2e-3 in the supersonic case,
    second order. That is the same absolute error as a gas-only run at the total pressure
    (3.1e-3 -> 1.2e-5), on a thermal energy 100x smaller (a dual-energy issue, not D2).
- **Mixing at a pressure-balanced density contact** (periodic, slab rho 1 in 0.1 or 0.5,
  `P_th = P_cr = 0.5`, u = 1, one crossing; exact: nothing changes). `dE_cr / E_cr`:

  | contrast | energy scheme HLL | HLLC + P2 | `cr_entropy` global | `cr_entropy` shocks (as implemented) |
  |---|---|---|---|---|
  | 10, N 128 / 512 | -1.4e-3 / -5.7e-4 | 1e-15 | +4.8e-2 / +2.0e-2 | +4.3e-2 / +1.7e-2, total energy +2.7e-2 / +1.1e-2 |
  | 2, N 128 / 512 | -2.0e-5 / -7.9e-6 | 1e-15 | +4.2e-3 / +1.7e-3 | +4.1e-3 / +1.7e-3 |

  Advecting `s_cr` conservatively and taking `e_cr = e(s_cr)` creates CR energy wherever the
  grid mixes gas of different density (Jensen: the mixed `e(s)` exceeds the mixed `e`). This
  is first order in dx, and both entropy modes also break the contact pressure balance by
  2-5% (Gupta et al.'s Et+Scr finding).
  - **This explains the 2026-10-09 M7 results**: shock-only created 11-15% of the CR
    injection in the halo, and global moved CR energy out of the thin gas. The energy scheme
    was the more accurate side there, not the less accurate one.
- The current shock-only mode is also wrong with HLLC + P2: `K_cr` is exact, but `P_th,2` is
  +2.3-4.3% at M >= 6 and the shock is +5 cells ahead (with HLL: <= 0.3%, shock within 1 cell).

**3. Hybrid prototype (monkeypatch of `cr_entropy_to_energy`, shock-only mode):**
- **Inside the shock mask** (Gupta et al. detector), `e_cr := e(s_cr)` and the difference goes
  to the gas.
- **Outside it**, the energy-scheme `e_cr` is kept.
- `s_cr` is re-synced from `e_cr` every step, so it acts as a one-step auxiliary and mixing
  errors cannot accumulate. Total energy is exact by construction.

| | HLL + hybrid | HLL + hybrid, C_cfl 0.8 | HLLC + P2 + hybrid | HLLC + P2 + hybrid, MC |
|---|---|---|---|---|
| `dK_cr,2`, all 11 tubes | 0.15-0.31% | 0.21-0.41% | <= 0.013% | <= 0.007% |
| spurious / `e_th` (max) | 0.69% (M 1.4) | 0.90% | 0.03% | 0.01% |
| shock offset | <= 0.85 cells | <= 0.94 | <= 0.89 | <= 0.55 |

- Mixing test: identical to the energy scheme (HLLC: `dE_cr` 1e-15, `P_tot` exact; HLL
  -5.7e-4), total energy 1e-14.
- Literature anchors:
  - Gupta et al. (2021): a cell-wise, energy-conserving redistribution in detected shock
    zones, with the energy scheme elsewhere.
  - Ryu et al. (1993) / Bryan et al. (1995): a dual-energy selection by shock detection.
  - Semenov et al. (2021): CRs adiabatic through shocks.
  - DSA stays finder-based in energy form, so Weber et al.'s (2023) entropy-injection
    oscillations do not arise; `s_cr` is re-synced after injection.
**4. Implemented as opt-in (user decision: "implement opt in + test", without M7).**
- **Code:** `cr_entropy_to_energy(..., shock_cells)`; `CR_ENTROPY_TRANSFER_SHOCKS` is now the
  hybrid, and its docs are updated. Default unchanged (`cr_entropy=False`).
- **Tests**, run in the scratch copy `/export/scratch/nknoell/phaseA/d2/tests` on GPU 0
  (UUID-pinned; logs `d2/logs/h_*.log`):
  - `cr_shock_tube_partition.py`: all 9 tests pass.
    - `test_cr_entropy_shock_only` was rewritten for the hybrid. Two gates were corrected on
      the way: the HLL gain is 4x (measured 5.7x for item 6); the rarefaction comparison
      excludes the edge shock and is bounded at 1e-5 (measured 4.8e-7).
    - Shocks: HLLC 9e-6 ... 1.3e-4, HLL 1.5-2.4e-3.
    - Closed box: drift 3e-14.
    - Squeeze: identical to the energy scheme.
    - Mixing: `dE_cr` 1e-14 and `P_tot` 6e-14, as the energy scheme (global +2.3e-2).
  - `test_cr_entropy_hybrid_sedov_3d` (new; 3D Sedov into a uniform CR background, exact
    `K_cr` = const): shell mean `abs(dK_cr)` HLLC 1.4e-2 -> 1.9e-4, HLL 3.5e-2 -> 2.1e-4, so
    the mask works at every shock orientation.
  - With the hybrid forced on (`hybrid_side_run.py`; registry printed, not gated: no misses),
    everything passes:
    - item 2 (compression, linear sound wave, hydro equivalence);
    - item 9 (all 3);
    - A1.3 1D + 3D (contacts stay exact with HLLC, so the mask has no false positives there);
    - all 5 gradient checks;
    - item 7: `E_cr/E_tot` 0.0411 (energy scheme 0.0401), identity 3.6e-6;
    - item 8: KR13 0.0826, CS14-like 0.0432.
  - Flag off: untouched code path (the mask argument is None).
- **Open:** M7 (excluded for now), the default decision, and whether `CR_ENTROPY_TRANSFER_GLOBAL`
  (mixing error) should stay.

### 2026-10-13: CR-row limiter switched to van Leer, guard retuned (user decisions; uncommitted)

Scheme change, logged in DESIGN.md "Decided: CR diffusion accuracy and the slope limiter"
(status 2026-10-13); summary and regression here.

- **Option c (user): MC on the CR rows + guard retune.** Implemented as `cr_limiter`
  (CR rows only, gas keeps `SimulationConfig.limiter`), `cr_guard_wide_sensor` (five-face
  shape sensor, fixes the A3.1 misfire) and `cr_guard_minmod_blend` (CR slopes blended towards
  MINMOD in guarded cells).
- **MC rejected** after the update-mode regression (`/export/scratch/nknoell/phaseA/optc/logs/u_*.log`):
  ring 3a/3b keep a smooth undershoot of ~1e-3 of the jump (1.4e-3 / 1.1e-3 / 1.1e-3 at N = 50
  / 100 / 200) inside the ring, invisible to any shape sensor; sensor thresholds 0.30-0.35
  did not help and 0.30 / 0.60 brought the Sovinec misfire back
  (`guard_c/logs/p2_*.log`).
- **Literature check (user: "literature backed", little time):** JO18 Sec. 4.1 (Athena++) and
  Rosdahl et al. 2025 (RAMSES) use van Leer for two-moment CRs; S&H 2007: slope-limited schemes
  keep the ring minimum. FCT / algebraic flux correction and transverse-flux limiting judged
  not worth the effort (DESIGN.md). **User chose van Leer** (`VAN_LEER = 6`, new limiter
  constant; defaults `cr_limiter = VAN_LEER`, blend and wide sensor on, ring gates back to 1e-3,
  Sovinec anchor pinned to MINMOD / guard off so its run length stays bounded).
- **Regression + re-baseline** (update mode, scratch copy `/export/scratch/nknoell/phaseA/optc2`,
  logs `optc2/logs/u_*.log`, seven free GPUs, ~75 min): every test passes;
  `test_cr_ring_flux_cap_alone` fails as designed (strict xfail). Files not rerun
  (`cr_adiabatic_*`, `cr_advection.py`, `cr_shock_tube*.py`) never set
  `diffusive_relaxation`, so the CR re-reconstruction does not run there.
  Check-mode rerun (registry gated) of all rerun files except ring and Sovinec (only one
  free GPU): 41/41 pass, values reproduce the registry (`optc2/logs/c_quick.log`).

| Metric (finest N unless stated) | MINMOD (before) | van Leer (now) |
|---|---|---|
| Ring 3a undershoot / jump, N 50 / 100 / 200 / 400 | 0 / 1.1e-5 / 0 / 0 | 0 / 5.0e-6 / 6.5e-6 / 1.4e-5 |
| Ring 3a `kappa_perp,num / kappa_par`, N 50 / 100 / 200 / 400 | 0.041 / 0.020 / 0.0087 / 0.0033 | 0.015 / 0.0054 / 0.0017 / 0.00061 (S&H van Leer 0.024 / 0.010 / 0.0038 / 0.0013) |
| Ring 3a L1, N 400 | 0.027 | 0.010 |
| Ring 3b undershoot / jump, N 100 / 200 | 1.1e-5 / 0 | 4.7e-6 / 6.7e-6 |
| Ring 3c (JO18) L1, N 64 / 128 / 256 | 1.99e-2 / 1.40e-2 / 1.02e-2 | 1.33e-2 / 8.87e-3 / 6.21e-3 |
| Sovinec `kappa_perp,num / kappa_par` (`kappa_perp` = 0), 16 / 32 / 64 / 128 per unit | 2.0e-2 / 8.9e-3 / 3.4e-3 / 1.0e-3 | 3.4e-3 / 1.1e-3 / 2.2e-4 / 3.5e-5 (MC 1.2e-3 ... 1.1e-5) |
| Sovinec, `kappa_perp` = 1e-2: leak / `kappa_perp` | 2.0 / 0.91 / 0.32 / 0.095 | 0.41 / 0.11 / 0.022 / 0.0033 |
| T1 `|D/kappa - 1|` N 1024; stiff | 1.2e-3; 1.8e-2 | 2.1e-4; 4.7e-4 |
| T2c `D_perp/kappa` 45 deg; T2d ratio error | 1.5e-2; 4.0e-2 | 1.0e-3; 4.4e-3 |
| T3 diagonal `D_perp/kappa`, N 128 | 1.6e-2 | 1.4e-3 |
| Item 4 L2, N 128 / 256 / 512 / 1024 | 1.67e-3 / 5.22e-4 / 1.81e-4 / 1.19e-4 | 4.71e-4 / 1.62e-4 / 1.14e-4 / 1.09e-4 (model floor) |
| A1.1 relaxation variant L1 Gaussian / top-hat, N 512 | 2.1e-3 / 2.2e-2 | 7.9e-4 / 1.4e-2 |

- Guard misfire fixed: with the guard on, every Sovinec variant (MINMOD, MC, van Leer) gives
  the guard-off value within 0.4% (2026-10-12: 2x leak, broken symmetry).
- Cap alone (guard off = plain van Leer): 3a undershoot 2.2e-2 / 3.1e-2, 3b `e_min` -0.0044 /
  -0.0085 at N = 100 / 200 (MINMOD: 1.9e-2 / 2.3e-2, -0.0027 / -0.0041). The guard stays needed.
- Tests that set the limiter explicitly (telegrapher, steady state) now pass it as
  `cr_limiter` too; their values moved by <= 4% (guard blend / wide sensor).
- Registry: 66 metrics changed, 16 new (Sovinec van Leer variants); diff against the
  pre-switch registry in `optc2/baselines_before.json`.
- Docstrings updated: `cr_sovinec.py` (module), `cr_anisotropic_ring.py` (module, 3a, 3b, 3c,
  cap alone), `cr_diffusion_rate.py` (module, T3), `cr_isotropic_diffusion_convergence.py`,
  `cr_advection_by_gas.py`, `cr_telegrapher_reference.py`.

### 2026-10-12: limiter / guard side run on the existing anisotropic tests (no code change)

Question (after A3.1): what does MC, with and without the monotonicity guard, do on the other
anisotropic tests, in particular the discontinuous rings the guard was built for? Variants
MINMOD / guard off, MC / guard on, MC / guard off; the default (MINMOD / guard on) is the
registry. Tests unchanged: a scratch driver overrides `limiter` in every `SimulationConfig` and
the guard thresholds (2.0 / 3.0) in every `CosmicRayGreyParams`; registry neither checked nor
written; figures to `/export/scratch/nknoell/phaseA/a3_variants/pics`, logs in `.../logs`. Free
GPUs 4 / 5 / 7, about 85 min each.

**Smooth tests (`cr_diffusion_rate.py`, oblique), finest resolution:**

| quantity | MINMOD (guard on = off) | MC (guard on = off) |
|---|---|---|
| T2b `|D_par / kappa - 1|`, 0 / 30 / 45 deg | 1.5e-2 / 1.4e-2 / 1.5e-2 | 1.4e-4 / 1.3e-4 / 1.3e-4 |
| T2c `D_perp / kappa`, 30 / 45 deg (N 64 / 128 / 256 at 45) | 1.1e-2 / 1.5e-2 (7.0e-2 / 3.6e-2 / 1.5e-2) | 1.1e-4 / 1.1e-4 (1.7e-3 / 3e-4 / 1e-4; 1e-4 = the T2a floor) |
| T2d `Delta D_perp / kappa_perp` | 1.0405 | 0.9999 |
| T2e out-of-plane `D_x / target` (N 128) | 1.07 | 1.0007 |
| T2f `D_num` (wave speed test) | 3.4e-4 | 3.2e-5 |
| T3 3D, N 128: in-plane `D_perp`, diagonal `D_perp` | 1.66e-2, 1.57e-2 | 1.5e-4, 1.5e-4 |
| oblique leak ratio | 2.18e-2 | 1.51e-2 |

The guard changes nothing on these smooth Gaussians (MINMOD guard off = registry to all printed
digits; MC on = off) -- it does not misfire here, unlike the Sovinec `psi` extrema. Everything
passes; the `D_perp` error drops 100x, consistent with A3.1.

**Rings** (S&H 3a: patch 12 on 10, jump 2, `kappa_perp,num / kappa_par` at N 50 / 100 / 200 / 400;
3b: patch on 0.1, jump 9.9; JO 3c: L1 at N 64 / 128 / 256):

| variant | 3a `kappa_perp,num / kappa_par` | 3a leaked | 3a worst undershoot / jump (any snapshot) | 3b `e_min` (N 100 / 200) | 3c L1 |
|---|---|---|---|---|---|
| MINMOD, guard on (default) | 4.08e-2 / 1.98e-2 / 8.66e-3 / 3.26e-3 | 0.74 ... 0.43 | 0 | 0.09989 / 0.09999 | 1.99e-2 / 1.40e-2 / 1.02e-2 |
| MINMOD, guard off | 3.33e-2 / 1.83e-2 / 8.31e-3 / 3.20e-3 | 0.73 ... 0.43 | 2.2e-2 ... 3.1e-2 | -0.0027 / -0.0041 | 1.82e-2 / 1.35e-2 / 1.01e-2 |
| MC, guard on | 1.01e-2 / 3.27e-3 / 1.13e-3 / 4.65e-4 | 0.53 ... 0.13 | 1.4e-3 / 1.2e-3 / 2.8e-3 / 6.1e-3 | 0.0874 / 0.0758 (1.3e-3 / 2.5e-3 of the jump) | 1.22e-2 / 8.25e-3 / 5.51e-3 |
| MC, guard off | 9.02e-3 / 3.17e-3 / 1.11e-3 / 4.61e-4 | 0.53 ... 0.13 | 2.3e-2 ... 4.6e-2 | -0.0038 / -0.0049 | 1.09e-2 / 8.10e-3 / 5.47e-3 |
| S&H Tables 1-4 (comparison) | asym. MC 1.3e-2 ... 6.8e-4; sym. MC 7.2e-3 ... 6.5e-5; van Leer 2.4e-2 ... 1.3e-3 | | | | |

- **MC improves the rings too:** `kappa_perp,num` 4-7x lower (from worse than every S&H scheme to
  between their van Leer and symmetric MC), leaked fraction 0.43 -> 0.13 at 400^2, JO L1 1.6-1.9x
  lower. Final-time profiles stay monotone (3a `e_min` >= 9.9975 at `t = 200`).
- **But MC + guard is not monotone:** transient undershoots 1.2e-3 ... 6.1e-3 of the jump in 3a
  (growing with N) and 1.3e-3 / 2.5e-3 in 3b, above the 1e-3 gate (3a and 3b fail). `e_cr` stays
  positive (3b `e_min` 0.076 > 0). The guard is tuned for MINMOD (strictly monotone there).
- **Without the guard both limiters undershoot by 1-5% of the jump and go negative in 3b**; the
  guard is needed with either limiter. The guard barely affects the leak (MINMOD 3a 4.08e-2 vs
  3.33e-2 at N 50, equal from N 200; MC equal), so the Sovinec guard misfire (smooth thick
  extrema) does not show in these tests.
- **Consequence for the limiter decision:** MC is a large accuracy gain on every anisotropic test
  (smooth 50-100x, rings 2-7x), at the price of a small transient non-monotonicity (<= 0.6% of the
  jump, positive) with the current guard. Options: (a) keep MINMOD; (b) MC and relax the 3a/3b
  undershoot gates to ~1e-2 of the jump (positivity kept); (c) MC and retune the guard for MC
  (scheme change, then rerun 3a/3b and Sovinec). User decision.
- Driver bug, fixed: the guard-off override replaced `CosmicRayGreyParams` with a function, so
  T2d (`CosmicRayGreyParams._fields`) errored in the guard-off runs; rerun with a proxy (T2d above).

### 2026-10-12: A3.1 + A3.5 Sovinec test -- done (uncommitted); field-aligned wave speeds do not help, MC does, the guard misfires

`pytests/cosmic_rays_grey/cr_sovinec.py`, 4 tests (3 gated + figure
`pics/03_anisotropic_diffusion/cr_sovinec_test.svg`); static-gas helper `cr_static_gas.py`
(X6; the ring test still has its own copy); logs `/export/scratch/nknoell/phaseA/a31/logs/`
(`*_r2.log`, `*_r3.log`); baselines registered (2 test ids). Setup and measurement as in the spec
entry below and the file docstring: periodic `[-1, 1]^2`, `kappa_par = 1`, `v_red = 96` (`tau` =
2 / 1 / 0.5 / 0.25 at 16 / 32 / 64 / 128 cells per unit length; the `psi` wavelength is 2), decay
rate of the `psi` mode fitted over `t` in [0.5, 2].

New library option (test variants, default unchanged and bitwise identical to HEAD):
`CosmicRayGreyConfig.cr_wave_speed_model` = `CR_WAVE_SPEED_FACE_NORMAL` (default, `R(tau(kappa_n))`),
`CR_WAVE_SPEED_FIELD_ALIGNED` (Athena++-like `sqrt(b_n^2 R_par^2 + (1 - b_n^2) R_perp^2)`),
`CR_WAVE_SPEED_PROJECTED` (TPP21 eq. 44, `|b_n|`, no `tau` reduction).

**`kappa_perp,num / kappa_par` at `kappa_perp = 0`** (guard off unless marked):

| variant | 16 | 32 | 64 | 128 | orders |
|---|---|---|---|---|---|
| MINMOD, face-normal (default) | 2.01e-2 | 8.92e-3 | 3.38e-3 | 1.02e-3 | 1.18 / 1.40 / 1.73 |
| MINMOD, field-aligned | 4.06e-2 | 1.57e-2 | 3.92e-3 | 8.27e-4 | 1.37 / 2.00 / 2.25 |
| MINMOD, projected | 8.26e-2 | 2.00e-2 | 4.22e-3 | 8.41e-4 | 2.05 / 2.24 / 2.33 |
| MC, face-normal | 1.23e-3 | 3.39e-4 | 6.56e-5 | 1.09e-5 | 1.85 / 2.37 / 2.59 |
| MC, field-aligned | 2.38e-3 | 5.00e-4 | 7.39e-5 | 9.26e-6 | 2.25 / 2.76 / 3.00 |
| MINMOD, face-normal, guard on | 4.34e-2 | 1.55e-2 | 4.07e-3 | 1.07e-3 | 1.49 / 1.93 / 1.93 |
| MC, face-normal, guard on | 4.06e-3 | 4.98e-4 | 7.51e-5 | 1.14e-5 | 3.03 / 2.73 / 2.72 |

**`kappa_perp,num / kappa_perp` at `kappa_perp = 1e-2`** (normalized by an isotropic run with the
same limiter, Dubois & Commerçon): default 2.03 / 0.91 / 0.32 / 0.095; field-aligned 3.6 / 1.4 /
0.34 / 0.073; MC 0.155 / 0.038 / 0.0068 / 0.0011; MC field-aligned 0.25 / 0.052 / 0.0073 /
0.00084; guard on 3.97 / 1.33 / 0.37 / 0.099 (MINMOD), 0.41 / 0.053 / 0.0077 / 0.0011 (MC). This
equals the `kappa_perp = 0` row divided by 1e-2 to within 2-7%: the numerical cross-field
diffusivity adds to the physical one. The isotropic reference itself converges at first order
(error 7.1e-2 / 3.2e-2 / 1.5e-2 / 7.1e-3; A4.2's MINMOD-in-thick-cells result).

- **The literature proposal (field-aligned or projected HLL speeds, Athena++ / TPP21) does not
  improve the leak where it matters.** In thick cells (`tau` >= 1, M7 has `tau ~ 3`) both leak 2-4x
  more than the face-normal default; they win only in thin cells (19% less at `tau` 0.25). Reason:
  for an oblique face the face-normal `R(tau_n)` with `kappa_n = kappa_par b_n^2` scales as
  `b_n^2 / tau`, the field-aligned speed as `|b_n| / tau`. Keep the default.
- **MC cuts the leak 16x (16 cells per unit length) to 94x (128)**, second to third order; the
  MINMOD default stays below the plan's order 1.5 except at the finest pair (first order in thick
  cells, as in A4.2). This is the A3.1 side run the limiter decision (option a, MINMOD kept) asked
  for: MINMOD is fine for isotropic diffusion (0.6-2.4%), not for the cross-field leak.
- **D1 numbers:** for `kappa_perp,num <= 0.1 kappa_perp` at `kappa_perp / kappa_par = 1e-2` the
  default needs ~128 cells per unit length (256 per wavelength of the mode), MC ~32 (64 per
  wavelength). At 16 per unit (32 per wavelength) the default leak is 2x the physical
  `kappa_perp`.
- **A3.5, `v_red` scan** (`kappa_perp = 0`, MINMOD, guard off; `v_red` 24 ... 384):

  | cells/unit, wave speed | `tau` | `kappa_perp,num / kappa_par` | local slopes `d ln / d ln v_red` |
  |---|---|---|---|
  | 32, face-normal | 0.25 ... 4 | 4.98e-3 / 7.19e-3 / 8.92e-3 / 9.44e-3 / 9.45e-3 | 0.53 / 0.31 / 0.08 / 0.00 |
  | 32, field-aligned | 0.25 ... 4 | 4.25e-3 / 8.92e-3 / 1.57e-2 / 1.98e-2 / 2.00e-2 | 1.07 / 0.82 / 0.33 / 0.01 |
  | 64, face-normal | 0.12 ... 2 | 1.31e-3 / 2.25e-3 / 3.38e-3 / 4.28e-3 / 4.55e-3 | 0.78 / 0.59 / 0.34 / 0.09 |
  | 64, field-aligned | 0.12 ... 2 | 8.57e-4 / 1.83e-3 / 3.92e-3 / 7.27e-3 / 9.33e-3 | 1.10 / 1.10 / 0.89 / 0.36 |

  The leak is HLL dissipation: proportional to `v_red` in thin cells (slope -> 1), independent of
  `v_red` once `tau >~ 2` (it saturates at the `kappa`-limited `R v_red ~ kappa / dx`). At M7's
  `tau ~ 3` a lower `v_red` does not reduce it.
- **Steady-state anchor** (source `Q0 psi`, default variant): steady-state rate vs decay rate
  -0.8% / +3.2% (`kappa_perp = 0`, 32 / 64 per unit), -0.5% / +0.15% (`kappa_perp = 1e-2`). The
  decay rate is the published steady-state measure.
- **Finding: the monotonicity guard misfires on smooth extrema in thick cells** (details in the
  file docstring). MINMOD clips the `psi` extrema, the shape sensor reads the flat top as a step
  (0.47 > onset 0.45) and raises `R` towards 1. Effect: the leak doubles at 16-32 cells per unit
  length (3.3x with MC at 16), odd symmetry broken by 0.1-0.2 of `A` (round-off, ~1e-10, with the
  guard off), rate drift 0.2; isotropic `kappa = 0.01`, `tau = 200` (smoke run): decay 4.6x too
  fast (1.07x off). Production M7 runs guard off; the library default is guard on (D3). A fix
  changes the scheme -> user decision.
- Gates: antisymmetry <= 1e-9 (guard off; round-off up to 1.8e-10 after ~1e4 steps), rate drift
  <= 0.05 at `kappa_perp > 0` (measured <= 1.3e-2); at `kappa_perp = 0` the leak is not a
  single-mode decay (drift up to 0.34 for MC at 16 per unit, 0.13 for MINMOD field-aligned at 64;
  projection residual 0.07-0.29), so the drift is printed only and the anchor checks the rate;
  default `kappa_perp,num` decreasing with N; finest-pair order (registry, margin 0.15); every
  value within 1.5x of the registry; anchor mismatch <= 10%; gas at rest (< 0.1 cell).
- Launch note: the scripts' own `autocvd` call overrides a UUID pin in `CUDA_VISIBLE_DEVICES`
  (four pinned jobs landed on one GPU); launch through a wrapper that no-ops `autocvd`.
- Verified: gating rerun (no `CR_UPDATE_BASELINES`, `gate_*.log`) passes all four tests and
  reproduces every registered value exactly (74 metrics).

### 2026-10-12: A4.4 steady-state diffusion with a source -- done (uncommitted)

`pytests/cosmic_rays_grey/cr_steady_state_source.py`, 5 tests (4 gated + figure
`pics/04_isotropic_diffusion/cr_steady_state_source_test.svg`); logs
`/export/scratch/nknoell/phaseA/a44/logs/`; baselines registered (3 test ids, 37 metrics).

- Setup: `Q = Q0 sin(2 pi x)` through `test_e_cr_source`, `kappa = 0.1` (5x item 4, so the
  slowest mode relaxes at `kappa k^2 = 3.95` and a steady state costs <= 2.6e5 steps at N 1024,
  `v_red` 32), amplitude 0.5 on offset 1, static heavy gas, float64, start from the cell-averaged
  analytic state, run 22 e-folds. Errors: L2 / amplitude against the analytic cell averages.
- **Fixed point (spec item 4 confirmed):** `C_cfl` 0.4 vs 0.2 / 0.1 differ by <= 3.7e-13 of the
  amplitude (`e_cr` and `F_cr`), thin (`v_red` 8, N 128, `tau` 0.21) and thick (`v_red` 32, N 32,
  `tau` 3.3), with ETD2RK and with the old stage-wise relaxation alike; 2D: 3e-14.
- **Steady-state error (MINMOD), N 32 ... 1024:**

  | `v_red` | N 32 | N 128 | N 1024 | orders |
  |---|---|---|---|---|
  | 4 | 2.02e-2 | 1.28e-3 | 1.99e-5 | 1.98, 2.00 ... |
  | 8 | 3.51e-2 | 2.38e-3 | 3.68e-5 | 1.89, 1.99, 2.00 ... |
  | 16 | 4.76e-2 | 4.43e-3 | 7.01e-5 | 1.54, 1.89, 1.98, 2.00 ... |
  | 32 | 4.91e-2 | 7.75e-3 | 1.36e-4 | 1.14, 1.53, 1.87, 1.97, 2.00 |

  Second order once `tau <~ 0.4`, first order at `tau` 3.3 -> 1.7; at N 1024 the error grows
  as `v_red^0.92`. Same structure as A4.2's transient scheme error: it is the HLL dissipation
  in the steady operator, not a time error.
- **MC** (`v_red` 32, N 32 ... 256): 1.18e-2 / 2.74e-3 / 5.62e-4 / 1.09e-4, orders 2.10 / 2.29 /
  2.37, MINMOD / MC = 4.2 / 8.1 / 13.8 / 19.5.
- **`v_red` dependence** `L2(e(32) - e(4))`: 2.9e-2 ... 1.16e-4, orders 0.75 / 1.41 / 1.84 /
  1.97 / 1.99: a discretization error, gone at second order (the plan's "independent of
  `v_red`" holds only in the limit).
- **2D diagonal mode** `sin(k (x + y))`, `v_red` 8, N 32 / 64 / 128: 3.46e-2 / 9.38e-3 / 2.36e-3,
  orders 1.88 / 1.99 (about the 1D error at the same N; the 2D path with two `F_cr` rows works).
- Drift over the last 10%: 6e-11 to 2e-10, not round-off: nothing balances `grad P_cr` on the
  heavy gas, which accelerates at ~1e-11; its `-P_cr div u` work explains the drift
  (estimate, not separately measured). It is identical at every `C_cfl`, so the dt gate is
  unaffected; drift gate 1e-9.
- Gates: dt <= 1e-10; thin order >= 1.8; thick order in [0.8, 1.3]; MC order >= 1.8; `v_red`
  difference order >= 1.8, thin slope vs `v_red` in [0.8, 1.2]; 2D order >= 1.8; registry 1.5x.

### 2026-10-12: test-only `e_cr` source hook implemented (uncommitted)

- `CosmicRayGreyParams.test_e_cr_source`: `Q(x)` on the interior grid (the shape of the state
  passed to `time_integration`), or None (default, off). Added as `dt * Q` to the `e_cr` row of
  the stage RHS in `evolve_state._evolve_gas_state_unsplit_inner`, right after
  `cr_grey_feedback_sources`, so it is inside every Heun / ETD2RK stage and in both gas
  half-steps of FV MHD. Padded with zeros to the ghost halo (overwritten by the boundary handler).
  `NotImplementedError` with the split scheme (`_evolve_gas_state_split`) and with `cr_entropy`.
- Verification (scratch script, float64, GPUs 3/5/6 idle at launch, PIDs checked by UUID):
  - hook off: 1D isotropic relaxation (N 128) and 2D MHD anisotropic + relaxation (64^2,
    30 deg) **bitwise identical** to HEAD 420cdee (git worktree).
  - uniform `Q = q0`: total `e_cr` grows by `q0 t` to 6e-14 (1D) and 9e-16 (2D MHD, so both
    Strang half-steps apply it once each, dt/2 + dt/2).
  - A4.4 preview, 1D: `Q = sin(2 pi x)` (cell averages), `kappa = 0.1`, `v_red = 8`, N 128
    (`tau` 0.21), from the analytic state to `t = 30/(kappa k^2)`: the steady states at
    C_cfl 0.4 and 0.2 differ by **3.1e-14** of the amplitude (dt-independent fixed point, as
    the spec predicts); both 2.4e-3 from the analytic profile (scheme error, the A4.4 subject).
  - split scheme: raises as intended.
- Not yet: the A4.4 test file itself (`v_red` scan, N convergence, baselines).

### 2026-10-12: second literature check of the A3.1 spec -- Riemann dissipation and limiters

Question: does the A3.1 / A3.5 spec (entry below) promise an improvement of the cross-field leak?
Answer: it is the right instrument, not a fix; the literature names two cheap levers it can A/B.
The spec entry below is updated accordingly.

**Other two-moment codes (what was read).**
- Athena++ CR module (the JO18 code; `src/cr/integrators/cr_transport.cpp`, `cr_flux.cpp`, read
  from the public repo). The HLLE speed of the CR rows is computed in the field-aligned frame,
  `v_diff = v_red/sqrt(3) (R(tau_par), R(tau_perp), R(tau_perp))` with
  `R = sqrt((1 - exp(-tau'))/tau')`, `tau' = 3/2 (taufact sigma dx)^2`, then rotated into the lab
  frame and `|.|` per axis: at `kappa_perp = 0` the face-normal speed is ~`|b_n| R(tau_par)`,
  capped at `v_red/sqrt(3)`. Same speed on the `e_cr` and `F_cr` rows.
- Thomas, Pfrommer & Pakmor (2021, MNRAS 503, 2242), eq. 44: `S = c sqrt(gamma_cr - 1)
  max(|b_n,L|, |b_n,R|)`, local Lax-Friedrichs, no optical-depth reduction (stiff sources are
  integrated separately). Sec. 4.3 / Fig. 5: without the `|b_n|` weighting ("naive" solver) the
  perpendicular contact diffuses. Sec. 4.4 (wedge on a ring field, 128^2-512^2): numerical
  diffusion "increases with the speed of light", decreases with resolution; qualitative only.
- Rosdahl et al. (2025, arXiv:2509.19447, RAMSES): scalar `tau` reduction of the signal speed
  (their eqs. 17-19), van Leer limiter, `|F| <= c e` cap; which diffusivity enters `tau` for
  anisotropic transport is not stated (not resolved). Looped-field test (Sec. 3.3, Fig. 3):
  "some diffusion perpendicular" at 512^2, less at 1024^2; no number.
- CR-MHD two-moment framework (2025, arXiv:2509.04387): optical-depth-dependent HLLE speed from
  the summed scattering rate, no cross-field test.
- Not checked: RAMSES-MCR (arXiv:2602.21147); Hopkins (2017, arXiv:1602.07703) seen, not read.

**Comparison with ours (`cr_wave_speed_reduction`, `R(n . K . n)`, `n . K . n = kappa_par b_n^2`
at `kappa_perp = 0`; constants in `tau` aside).**
- Thin regime (`tau_par << 1`, M7 along B: 0.16-0.32): ours ~`v_red/sqrt(3)` at any oblique
  face, Athena++ / Thomas `|b_n|` times that: ours is `1/|b_n|` larger (1.41x at 45 deg).
- Thick regime (`R ~ 1/tau`): ours `~ b_n^2 kappa_par/dx`, Athena++ `~ |b_n| kappa_par/dx`: ours
  is `|b_n|` smaller.
- So the field-aligned weighting is at most a ~1.4x lever at 45 deg, and only in thin cells.
  Any per-face scalar dissipation on `e_cr` jumps still diffuses across B at oblique angles;
  a change of order would need transverse-gradient limiting (Sharma & Hammett 2007), a scheme
  change for D1.
- Limiter: JO18 (Sec. 4.1.5, van Leer per direction) and Rosdahl et al. use van Leer; ours is
  MINMOD, which A4.2 found first order in thick cells (MC 15x smaller). Prediction (not
  measured): the leak is mostly HLL dissipation x MINMOD clipping, largest at the O-point
  extremum; the A3.5 `v_red` scan decides it.

**Confirmed as planned.** No two-moment CR paper runs the Sovinec test (all published Sovinec
numbers are implicit / semi-implicit solvers). The A3.5 merge matches Thomas et al.'s
`c_red` dependence. The in-stage source hook is the standard well-balanced placement (the
other codes integrate their stiff sources implicitly / IMEX, no counterexample).

**Spec changes (applied to the entry below).** Decay rate marked unpublished, with the
`kappa_perp = 0` null-space caveat; the steady-state anchor is required at both `kappa_perp`
values. MINMOD and MC as equal main variants; a test-only Athena++-style wave-speed flag as a
second axis (2x2 matrix); expectations lowered to order ~1 with a `v_red`-dependent leak.
Correction: `kappa_n` at 45 deg is `kappa_par/2`, not ~`kappa_par`.

### 2026-10-12: A3.1 / A3.5 / A4.4 and the source hook -- spec revised after a literature check (not started)

Updated the same day after a second literature check (Riemann dissipation and limiters in other
two-moment codes; entry above). Changed: the decay-rate caveat at `kappa_perp = 0` (item 3), the
leak attribution (item 5), the variants (last section).

The plan's A3.1, A3.5 and A4.4 (Sec. 3.4 / 4.4) were checked against the papers that define or
use the Sovinec test, and the hook design against the code. The plan file stays frozen; this
entry is the spec to implement. Nothing is implemented yet (user: "don't start yet").

**Literature (what the plan gets right).**
- Sovinec et al. (2004, JCP 195, 355, Sec. 4.2): unit square [-0.5, 0.5]^2, homogeneous Dirichlet
  `T = 0`, source `Q = 2 pi^2 cos(pi x) cos(pi y)` "to drive the lowest eigenmode", field tangent
  to the contours of `psi = cos(pi x) cos(pi y)`, "an extremely large mass density prevents MHD
  motions". Finite `chi_perp = 1`, `chi_par/chi_perp` = 1e3, 1e6, 1e9; error measure
  `|1/T(0,0) - 1|` = `chi_perp,num/chi_perp`. Run to steady state (semi-implicit FE).
- Sharma & Hammett (2011, arXiv:1008.0861, Sec. 3.2): the same with `chi_perp = 0`,
  `chi_perp,num = 1/T(0,0)` from the asymptotic maximum; close to second order, independent of
  the time step up to 1e4x the explicit limit (semi-implicit).
- Pakmor et al. (2016, Sec. 3.4): the plan's field and source, `kappa_par = 1`, `kappa_perp = 0`,
  outflow boundaries with zero outside, fixed mesh, no hydro; `< 1e-2` at 16^2, "almost second
  order".
- Dubois & Commerçon (2016, **Sec. 3.6**; the plan gives no section): Dirichlet `T = 0`,
  `kappa_iso = 1`, `kappa_Sp = 100` (`kappa_par = 99`), 16^2-256^2, `kappa_num ~ dx^2`, and the
  normalization `kappa_num = T_iso(0,0)/T_meas(0,0) - 1` with `T_iso` from an isotropic run on the
  same grid (after Sharma & Hammett 2007).
- No two-moment CR paper runs this test (Jiang & Oh 2018 and Thomas, Pfrommer & Pakmor 2021
  checked). All published numbers come from implicit / semi-implicit diffusion solvers, so they
  compare as `kappa_perp,num/kappa_par`, read off their figures (Pakmor Fig. 10, S&H 2011 Fig. 6,
  D&C Fig. 7).

**Corrections to the plan.**
1. *Finite-`kappa_perp` gate.* The plan's "`kappa_perp/kappa_par = 1e-3`: `e(0,0)` within 5%"
   needs `kappa_perp,num < 5e-5 kappa_par`; the ring test gives 3e-3 to 4e-2, so it would fail by
   ~100x. Replace with Sovinec's measure `kappa_perp,num/kappa_perp` at `kappa_perp/kappa_par =
   1e-2` (M7 / Girichidis et al. 2016) and 1e-3, tracked; it answers D1 directly (is the leak
   small against the physical `kappa_perp`?). Use the D&C isotropic normalization to cancel the
   Laplacian's own discretization error.
2. *Boundaries: use the periodic variant.* `FIXED_BOUNDARY` writes one uniform state per boundary
   face and raises `NotImplementedError` for MHD on the conservative path; the Sovinec field varies
   along the boundary. The periodic box [-1, 1]^2 (four Sovinec cells, `Q = Q0 psi`,
   `e = e_off + psi/kappa_tot`) is exactly the Dirichlet problem with odd-reflection ghost cells
   (the standard second-order FV Dirichlet) if (a) N is divisible by 4, so the separatrices
   x, y = +-1/2 are cell faces, and (b) the scheme is odd-symmetric in `e - e_off` (MINMOD, the
   HLL rows, the shape-based guard: yes; realizability cap: inactive at a large offset; to check:
   the guard's absolute smoothing `eps` must stay far below the `E_SCALE`-scaled jumps). Check
   (b) in the run: `e = e_off` on the separatrix faces to round-off. `b b^T` is unchanged when
   `psi -> -psi`, so the diffusion tensor is the same in all four cells.
   - Plan typo: the offset must exceed `Q0/(2 pi^2 (kappa_perp + kappa_perp,num))`, not
     `Q0/(2 pi^2 kappa_perp)` (undefined at `kappa_perp = 0`).
   - Field nulls: B = 0 at the O-points (where `e(0,0)` is measured) and at the X-points
     (+-1/2, +-1/2). With N divisible by 4 all of them are cell corners, so no cell centre sits on
     a null; `b_field_floor = 1e-10` << `B0 pi dx` for `B0 = 1e-5` (ring-test value).
   - Measurement: the amplitude from a projection onto the cell-averaged `psi`, not the corner
     value at (0,0).
3. *Cost (estimate, not measured).* With `kappa_par = 1`, Courant 0.4 on `v_red/sqrt(3)` and
   fixed `tau = (gamma_cr - 1) v_red dx / kappa_par`, reaching steady state to 1e-3
   (`t ~ 0.35/kappa_tot`, `kappa_tot = kappa_perp + kappa_perp,num`) takes
   `~1.5 tau / (kappa_tot dx^2)` steps: ~6e5 at 64 cells per unit length and ~4e6 at 128
   (256^2 periodic box) for M7-like `tau ~ 3`; more with `kappa_perp = 0`. The published runs are
   implicit and do not have this cost.
   - **Primary measurement: decay rate (our variant, not published).** `psi` is an exact
     eigenfunction of the continuous anisotropic operator (`b . grad psi = 0`):
     `div(K grad psi) = -2 pi^2 kappa_perp psi`. For `kappa_perp > 0` it is the lowest one
     (Rayleigh quotient >= the isotropic value). For `kappa_perp = 0` the continuous operator has
     a null space (every `f(psi)` is steady), so the fitted rate only measures how fast the
     discrete operator damps the `psi` projection, and the profile may drift towards other
     `f(psi)` shapes: record the residual of the projection. Start
     from `e = e_off + A psi`, `F = -K grad e`, no source, and fit `lambda = 2 pi^2 kappa_tot` from
     the projected amplitude after the parallel / two-moment transients (~L/v_red, ~1/(kappa_par
     k^2)). About 0.5/lambda of run time instead of ~7/lambda, roughly 10x cheaper. Telegrapher
     correction `kappa_tot k^2 / nu` is negligible.
   - **Anchor (required, not optional):** one steady-state run with the source hook at 32 and 64
     cells per unit length, at `kappa_perp = 0` and at `kappa_perp/kappa_par = 1e-2`; the
     decay-rate and steady-state `kappa_tot` must agree within 10% (both should see the same
     lowest discrete mode; at `kappa_perp = 0` this is the only check of that). If they do not
     agree, the steady state is the reference (it is the published measure).
   - Resolutions 16-128 cells per unit length (Pakmor and D&C start at 16^2); 256 per unit length
     (512^2 periodic) is not affordable.
4. *A4.4 gate.* The plan says the steady state is independent of `v_red` and dt to 1e-6. Only the
   dt part holds (exact fixed point of the stage-wise source and the ETD2RK relaxation). The
   `e_cr`-row HLL dissipation `R(tau) v_red/sqrt(3) (q_R - q_L)` is part of the discrete steady
   operator, so the steady state carries the A4.2 scheme error (~`v_red dx^2` thin, ~`kappa dx`
   thick with MINMOD). New gates: dt-independence to round-off (~1e-10, float64); the `v_red`
   dependence converges with N (rate tracked against A4.2's).
5. *Merge A3.5 into A3.1.* The same `v_red` dependence in A3.1 is A3.5's quantity. A `v_red` scan
   at fixed N splits `kappa_perp,num` into a part from the HLL dissipation (depends on `tau`; the
   face-normal `kappa_n = kappa_par b_n^2` in `cr_wave_speed_reduction` is `kappa_par/2` on both
   axes at 45 deg) and a part from the gradient projection in the relaxation target (independent
   of `v_red`). That tells which part a fix has to target.
   - Prediction (second check, not measured): the HLL part dominates, through limiter clipping.
     The projection uses centred cell gradients, second order in smooth regions (Sharma & Hammett
     2007), while MINMOD x HLL is first order in thick cells (A4.2) and clips at the O-point
     extremum where the amplitude peaks. This would explain the ring's order ~1.0-1.4.
   - Metric: `d ln kappa_perp,num / d ln v_red` at 32 and 64 cells per unit length,
     `v_red` = 4, 8, 16, 32 (at fixed `kappa_par`, so `tau` scales with `v_red`).

**Source hook (spec).**
- Placement: in the stage RHS of the unsplit FV scheme, next to `cr_grey_feedback_sources` in
  `evolve_state._evolve_gas_state_unsplit_inner`, not in `_time_integrator_sources` (for the
  unsplit FV scheme that is the operator-split presolve). By my analysis, an operator-split
  source leaves an O(dt) bias in the discrete steady state, whereas an in-stage source keeps the
  exact, dt-independent fixed point (both Heun stages vanish when `L(e) + Q = 0`). The A4.4 dt
  gate tests this.
- Form: a time-constant `Q(x)` array (cell-averaged, spatial shape) on the `e_cr` row only, as in
  Jiang & Oh (2018) and Thomas et al. (2021); carried in the params (pytree, jit-friendly,
  differentiable; default None = off). In FV MHD (Strang split) it acts in each gas half-step.
- Test-only scope for now; no change to existing defaults (bitwise check against HEAD with it off).

**Other points for the A3.1 test file.**
- Heavy static gas as in the ring test (`rho = 1e4`, `E_SCALE = 1e-7`, `B0 = 1e-5`), float64,
  2D MHD, `anisotropic_transport=True` (requires MHD). Recheck the `max|u| t < 0.1 dx` gate for
  the longer runs (estimate: displacement ~5e-7, fine).
- Expected order ~1.0-1.4 (ring 3a), below the literature's ~2: order >= 1.5 is a tracked metric
  until D1, not a hard gate. The literature's second order comes from implicit diffusion
  solvers without a Riemann dissipation term; for this scheme order ~1 and a `kappa_perp,num`
  that grows with `v_red` (Thomas et al. 2021, Sec. 4.4) are the expected baseline.
- Variants (all at the same N and `v_red` set, decay rate; reported side by side):
  - limiter: MINMOD (default) and `DOUBLE_MINMOD` (MC) as equal main variants, not a side run.
    JO18 and Rosdahl et al. (2025) use van Leer; the repo has none, MC is the nearest.
  - `e_cr`-row HLL wave speed: the current `R(n . K . n)` and the Athena++ (JO18 code) form:
    `(R(tau_par), R(tau_perp), R(tau_perp)) v_red/sqrt(3)` rotated into the lab frame, absolute
    value per axis (~`|b_n| R(tau_par)` at `kappa_perp = 0`; `tau` constant as in our
    `cr_wave_speed_reduction`). Optional third: Thomas et al. (2021) eq. 44, `|b_n| v_red/sqrt(3)`
    without the `tau` reduction. A test-only opt-in flag, default unchanged (bitwise check);
    adopting it as default would be a scheme change and needs the user's OK. Expected gain at
    most ~1.4x at 45 deg in the thin regime (M7 along B), a loss in thick cells.
  - guard on and off (A3.4 / D3).
  - The 2x2 limiter x wave-speed matrix is the minimum; guard on/off only for the default pair.

### 2026-10-12: X1/X2 -- baseline registry and tightened gates

- **Registry (X1):** `pytests/cosmic_rays_grey/cr_baselines.py` + `baselines.json`, keyed by
  `<file>::<test>` and metric, with the measurement date.
  - Checks: `upper` (errors: `<= 1.5x` baseline, optional absolute floor for round-off-sized
    metrics; a value far below prints a NOTE), `lower` (orders: `>= baseline - margin`),
    `close` (physical quantities, two-sided).
  - The physics gates (literature orders, invariants) stay in the tests; the registry adds
    "no worse than measured".
  - Refresh: `CR_UPDATE_BASELINES=1` records instead of comparing (physics gates still apply);
    review the json diff before committing.
  - Docstrings keep their dated history and point to the registry for the gated values.
- **Gates (X2), per file** (the 12 Phase A files of items 1, 2, 3, 4, 6 and A1-A6 tests):
  - `cr_advection.py`: L2 and peak loss 1.5x (were 8x / 4x). New localized checks:
    `max |F/(lambda e) - 1|` in the pulse (3.0e-4) and the `e_cr` first moment (the speed).
  - `cr_advection_by_gas.py`, `cr_contact_pressure_balance.py` (HLL tracked bounds),
    `cr_adiabatic_coupling.py`: finest-N errors and orders against the registry.
  - `cr_adiabatic_compression.py`: invariant error 1.5x (was 5x), and max `rho/rho0` within 1%
    (the size of the `-grad P_cr` deceleration, previously unchecked).
  - Item 3: oblique leak ratio 1.5x (was 4.6x) and the parallel spread within 2%; ring 3a
    `kappa_perp,num/kappa_par` at every N and L1 at N = 400; ring 3c L1 at every N and order;
    T1-T3 rates (e.g. T1b `|D/kappa - 1|` 1.2e-3 against a 0.02 gate) and the oblique
    `D_perp` (T2c 30/45 deg, T3 diagonal and in-plane), so numerical cross-field diffusion can
    no longer grow unnoticed (its absolute budget is still D1).
  - Item 4: `cr_isotropic_diffusion_convergence.py` drops the fragile order gate (plan
    Sec. 4.3; the order is gated without a floor in A4.1) for per-N errors;
    `cr_telegrapher_reference.py` moves its mode baselines into the registry and adds the
    Gaussian per N and the A4.2 scheme errors.
  - Item 6: `cr_shock_tube.py` now gates plateau averages on the left star and shocked
    plateaus (windows from the reference) and the shock / contact positions (< 1 cell; the box
    mean stays as a loose sanity bound). Shocked plateau: `e_cr` +1.68%, `P_th` -0.93%,
    `rho` 0.38%, `u` 0.08% (the A6.1 partition, D2); left star plateau <= 0.22%.
    `cr_shock_tube_partition.py`: A6.1-A6.3 partition errors at 1.5x per case. The old fixed
    bounds were already ~1.5x for A6.1 and Gupta A / B, but A6.3 had one bound (0.2) for every
    Mach number: 41x at M 1.4, 6.7x at M 2.
- **Finding (item 1 speed):** the first moment of `e_cr` ends -0.074 cells behind after one
  period (512 cells, 1.4e-4 of the distance; float32 and float64 alike). Not transport: at
  `amp = 1e-5` it is 1.4e-3 cells (speed exact to ~3e-6). At the test's `amp = 1e-3` the CR
  pressure moves the gas and the coupling is second order in `amp`. Tracked (+- 0.02 cells);
  the docstring's "speed to 5 decimals" is qualified.
- **Runs (GPU):** update mode `/export/scratch/nknoell/phaseA/x1x2/logs/`, check mode
  `x1x2_check/logs/`: all tests of the 11 non-ring files pass in both (106 registry checks,
  reproduced to the 4-digit rounding). Ring file (update mode only, ~70 min): 3a / 3b / 3c
  pass and match 2026-10-05 (3a `kappa_perp,num/kappa_par` 0.0408 / 0.0198 / 0.0087 / 0.0033,
  3c L1 1.99e-2 / 1.40e-2 / 1.02e-2, orders 0.51 / 0.45); the cap-alone test fails as its
  strict xfail expects. Not rerun in check mode (the check path is the same generic code,
  exercised by the 106 checks above).

### 2026-10-12: limiter decision for CR diffusion -- option a (MINMOD kept)

- **HLLC check:** HLL, HLLC and AM-HLLC share `hll._grey_cr_hll_rows` for the CR rows; the A4.2
  points rerun with HLLC match HLL to all printed digits (`a42/logs/rs_*.log`).
- **van Albada:** `limiters._van_albada_limiter` adds an unscaled `3 dx` to the squared slopes,
  so for e_cr in code units it is the unlimited central slope (why it matched MC); not a
  candidate for CR rows.
- **Size:** relative to the bump RMS (0.084) MINMOD costs 0.009% at v_red 8 / N 2048 (model
  error 0.13%) and 2.4 / 1.2 / 0.6% at v_red 64 / N 128 / 256 / 512 (MC 0.08% at N 256).
- **Decision (user):** option a, MINMOD kept. Revisit only via an MC side run in A3.1/A3.2
  (perpendicular leakage). Next: X1/X2.

### 2026-10-11: A4.2 `v_red` scan -- floor explained, MINMOD limits CR diffusion accuracy

- **Tests:** `cr_telegrapher_reference.py` gained 4 A4.2 tests. They use item 4's Gaussian at
  v_red 4-64 and N 128-8192 (ETD2RK, float64, C_cfl 0.4), against diffusion and the exact
  telegrapher solution. The file runs 9/9 pass (GPU, `a42/logs/test_run1.log`). Figure:
  `pics/04_isotropic_diffusion/cr_telegrapher_vred_scan_test.svg`.
- **Floor:** the error vs diffusion is the model floor (exactly v_red^-2) plus the scheme
  error.
  - Slope -1.975 at N 8192 (plan gate -2 +- 0.3 passes).
  - At the plan's N 2048 it is -1.31, because the scheme error (4.2e-6 ... 3.0e-5, ~v_red)
    overtakes the floor near v_red 16. This closes plan Sec. 4.3 step 3 (the "v_red^-1
    floor").
- **Scheme error:** spatial; C_cfl 0.1 and 0.4 agree to < 1%.
  - MINMOD, thin cells: ~v_red dx^2.
  - MINMOD, thick cells (tau >~ 2): ~kappa dx, first order, v_red-independent.
  - Cause: minmod's O(dx^2) face jumps times the HLL dissipation R(tau) c.
  - MC / van Albada: second order everywhere, 15x smaller at v_red 8 / N 2048, and nearly
    v_red-independent (the plan's expectation).
- **Production:** M5-M7 run MINMOD; M7 at v_red 1e4 km/s has tau ~ 3. DESIGN.md "Decided: CR
  diffusion accuracy and the slope limiter": option a (2026-10-12).

### 2026-10-11: MHD energy-budget cavity change explained (streaming leak across B)

- **What changed:** the ETD2RK switch moved the printed cavity widths of `cr_mhd_energy_budget.py`
  (along / across B 0.086 / 0.164 -> 0.102 / 0.195). The cause is the *old* scheme.
- **Mechanism:**
  - Its RK average kept 1/2 of F_perp (relaxation factor -> 1/2 at a -> inf).
  - The isotropic streaming target is set once per step, so the second MHD gas half-step
    streamed that remnant across B.
- **Verification:** stage-wise + a B-projection at the end of each half-step reproduces ETD2RK
  exactly, and with streaming off both integrators agree.
- **Out of scope:** streaming is outside Phase A (item 5 excluded); it appears only because
  ladder item 18's MHD test turns it on. The remaining leak (stage 1 uses the unprojected
  streaming target) is left for a later streaming phase.
- Details: DESIGN.md "F_cr relaxation time accuracy: ETD2RK", follow-up. Figure:
  `pics/18_energy_budget/cr_mhd_cavity_investigation.png`.

### 2026-10-10 (night, 3): ETD2RK F_cr relaxation implemented (user decision)

- **New option** `CosmicRayGreyConfig.flux_relaxation_integrator`:
  - default `CR_RELAXATION_ETD2RK`, implemented in `cr_grey_sources.cr_flux_relaxation_etd2rk`:
    ETD2RK on the F_cr rows, Heun on all other rows, the realizability cap after the stage and
    after the step;
  - `CR_RELAXATION_STAGEWISE_IMPLICIT` keeps the old per-stage backward Euler.
- **Rewired:** `_evolve_gas_state_unsplit` uses it. Its `rhs` stays explicit with ETD2RK.
- **A4.1:** the strict xfail is now a regular test, and it passes (orders 2.00 / 1.95 / 1.89 / 1.95).
  - New `test_cr_relaxation_integrators` (m = 24): the old scheme has order 0.69; ETD2RK is
    5.5x more accurate at N 2048.
  - The baselines were re-measured: modes 3-7x smaller at N 2048 and C_cfl-independent; Gaussian
    7.94e-6 at N 2048 (gate 1.2e-5), pairwise 2.00 to N 4096.
  - The figures were regenerated.
- **Regression:** all 12 CR test files that use the relaxation pass (GPU, logs
  `/export/scratch/nknoell/phaseA/regress_etd/logs/`):
  - telegrapher, item 4, diffusion rate, ring, oblique, gradient check, modified shock,
    kappa inference, div B, advection by gas, contact balance, MHD energy budget.
  - Changed docstring values: modified shock Test B (du/dx 23.0%, dP_cr/dx 1.30%, dF_cr/dx
    0.47%), gradient-check guard (2.1e-6), div B (6.8e-15 aniso), item 4 (L2 1.67e-3 ...
    1.19e-4).
  - Diffusion-rate values are unchanged to 4 digits.
  - Ring: matches to 3-4 digits; the cap-alone test is still the strict xfail.
  - MHD energy budget: 1.27e-10 [1.05e-10], gate 1e-9. The printed (ungated) cavity widths of
    the anisotropic run changed: along B 0.102 [0.086], across B 0.195 [0.164] at N 128
    (explained 2026-10-11, entry above).
- **Not rerun:** M5-M7 (production setups).

### 2026-10-10 (night, 2): second-order F_cr relaxation coupling -- literature and prototypes

- **Literature:**
  - Our scheme is PLUTO's "IMEX1" (Melon Fuksman & Mignone 2019).
  - Arepo's two-moment CR solver (Thomas+ 2021) is formally first order too.
- **IMEX-SSP2(2,2,2) (Pareschi & Russo 2005) is not a good fix here.** Its cell F_cr is wrong in
  the stiff regime (final layer: -2.3x / -62x at nu dt = 10 / 100), and it is unstable with
  upwind fluxes at large nu dt.
- **Strang and integrating-factor splittings lose the diffusion limit** (D_eff ~ nu dt / 2 x
  kappa).
- **Recommended: ETD2RK on the F_cr rows, Heun on the rest.** Prototyped by scratch monkeypatch
  (GPU):
  - single modes become second order (1.89-2.00 at 1024 -> 2048, C 0.4);
  - 3-7x smaller error at N 2048;
  - nearly C_cfl-independent;
  - the item-4 Gaussian improves to 7.9e-6 at N 2048 (order 2.0);
  - stiff case (kappa 1e-3) unchanged.
- A Heun-compatible GSA IMEX works equally well, but its stages overshoot.
- Side finding: at kappa 1e-3 (dx >> kappa / c) all schemes converge at first order. This is
  spatial (HLL dissipation), not temporal.
- Details are in DESIGN.md, "F_cr relaxation time accuracy: ETD2RK". The user chose ETD2RK
  (entry above).

### 2026-10-10 (night): A4.1 exact telegrapher reference (1D) -- relaxation is first order in time

- **New:** `test_setups/reference_solutions/linear_theory.py`. It is the exact spectral solution of
  the isotropic two-moment system in gas at rest, on a periodic box, in 1D/2D/3D: the per-mode 2x2
  propagator, in cancellation-free form, with a series near critical damping. It matches
  `scipy.linalg.expm` per mode to 1e-15 (including an exactly critical mode, a 3D oblique mode and a
  3D slab vs 1D) and d'Alembert for nu -> 0 to 7e-14.
- **Test:** `pytests/cosmic_rays_grey/cr_telegrapher_reference.py`. It uses item 4's parameters,
  float64 and static heavy gas (rho 1e4, P 1e-4, e_cr = 1e-7 (0.1 + profile); max abs(u)
  2.7e-12). Figures are in `pics/04_isotropic_diffusion/cr_telegrapher_reference_*`. Logs:
  `/export/scratch/nknoell/phaseA/a41/logs/`. Result: 3 pass, 1 strict xfail.
  - **Model floor:** telegrapher - diffusion at t = 0.24 is 4.41e-4 / 1.08e-4 / 2.69e-5 /
    6.71e-6 / 1.68e-6 at v_red 4 ... 64, slope -2.008. At v_red 8 this *is* item 4's ~1.2e-4
    floor.
  - **Gaussian vs telegrapher:** 1.66e-3 ... 3.42e-6 (N 128 ... 4096), least-squares order 1.86
    (128-2048), pairwise 1.74 / 1.92 / 1.91 / 1.79 / 1.55. There is no floor. The error vs
    diffusion at N 4096 is 1.026x the model floor. Gate: order >= 1.8 (plan), N 2048 <= 1.5e-5.
  - **Single modes** m = 2, 8, 18, 24, 48 (k/k_c 0.11 ... 2.61), max-in-time error over three
    e-folds:
    - at C_cfl 0.4 (default), first order at fine N (pairwise 0.4-1.0 at 1024 -> 2048);
    - at C_cfl 0.1, ~2nd order, 4-5x smaller at N 2048 (e.g. m 24: 4.7e-3 vs 9.4e-4).
    - A C_cfl scan at N 1024 shows the error ~linear in C_cfl above ~0.1. Tracked bounds; the
      plan's "second order" expectation is `test_cr_single_mode_second_order`,
      xfail(strict).
- **Plan question "v_red^-1 floor" (Sec. 4.3) answered:**
  - The model floor scales as v_red^-2 exactly.
  - At fixed N the scheme error against the telegrapher reference *grows* with v_red: N 2048
    gives 1.2e-5 / 1.0e-5 / 1.6e-5 / 3.0e-5 at v_red 4 / 8 / 16 / 32 (prototype,
    `a41/logs/proto_vred.log`). This is pre-asymptotic: tau = (gamma_cr - 1) v_red dx / kappa.
  - Together they mimicked a slower decrease of the item-4 floor. A4.2 makes this the test.
- **Finding: the F_cr relaxation is first order in time** (not a test artefact):
  - `evolve_state` applies `F <- F / (1 + nu dt)` at the end of each SSP-RK2 stage and
    averages the stages. Pure relaxation then gets the step factor
    (1 + 1/(1+a)^2)/2 = 1 - a + 1.5 a^2 (a = nu dt), vs exp(-a) = 1 - a + a^2/2. That is a local
    error a^2, a global error O(nu dt) = O(dx) at fixed C_cfl.
  - A steady, gradient-slaved F_cr stays an exact fixed point, so the diffusion limit is fine
    (Gaussian at t_end: order 1.86). Transients and every mode not << k_c see the error.
  - M7 estimate (kappa 1e28, v_red 1000 km/s, dx 31 pc), if dt is set by the CR signal speed:
    nu dt ~ 0.2, and scales below 2 pi / k_c ~ 700 pc are damped waves. So it can matter in
    production.
  - Fixed the same night with ETD2RK (log entry above; DESIGN.md "F_cr relaxation time
    accuracy: ETD2RK").

### 2026-10-10 (late): tracker rows A1.3 and A6.1 updated

- A1.3: done; exact with HLLC + P2 (hydro and MHD); HLL (the default) keeps the tracked bounds.
- A6.1: the row now shows the stage-wise values (1.27%, from the 2026-10-09 re-baseline) and the
  HLLC + P2 method variant (0.42%, `p2_repo/logs/cr_shock_tube_partition.log`). The test
  docstring said HLLC 1.24% and now quotes both. A6.2/A6.3 run HLL only, so they are unchanged.

### 2026-10-10 (late): stale docstring numbers re-baselined to HLLC + P2

- Test docstrings whose quoted values came from the stage-wise source coupling now carry the
  HLLC + P2 values, with the 2026-10-09 values in brackets. Sources are the as-configured runs in
  `p2_repo/logs/` and `p2_mhd/logs/`:
  - item 7 `cr_sedov_taylor.py`: E_cr/E_tot 0.0401 [0.0434], partition identity 1.0e-6 [3.0e-6];
    the inline calibration comment is extended too;
  - item 8 `cr_dsa_mach_dependence.py`: KR13 0.0810, CS14 0.0423, ratio 0.5217 (4.34%);
  - item 10 `cr_wind_bubble.py`: E_cr/E_tot 0.0430, R_2 1.6%, interior P 22.7%, identity 1.4e-4,
    control vs DSA 6.2e-6;
  - item 11 `cr_snr_clumpy_medium.py` (N 128): uniform 0.0530, clumpy 0.0494, -6.72%, identity
    2.3e-5 / 2.6e-5;
  - item 18 `cr_energy_budget.py`: E_cr(t_end) 0.03332 [0.03517];
    `cr_mhd_energy_budget.py`: energy 1.05e-10, single precision 2.88e-9, E_cr(t_end) 0.0405
    [0.04295];
  - Phase D `cr_phase_d_injection_efficiency_inference.py`: AD-vs-FD 5.1e-3, mach_scale 1.06340.
- Docstring text only; no gates changed (all still pass with margin). The item 11 figure is not
  regenerated: the committed file is 256^3, which does not fit 11 GB.
- X1 "stale docstrings" part is done for P2; the baseline registry is still todo.

### 2026-10-10 (evening): remaining MHD CR tests with HLLC + P2 forced -- no change

- **Setup:** the files that leave the solver at the HLL default were run with HLLC as the
  `SimulationConfig` default (P2 active), all tests per file, figures off:
  - `cr_anisotropic_ring.py`;
  - `cr_anisotropic_diffusion_oblique.py`;
  - `cr_diffusion_rate.py` (its 2D/3D cases are MHD);
  - `cr_divergence_b_preservation.py`.
  - Logs: `/export/scratch/nknoell/phaseA/p2_mhd_forced/logs/`. Compared with today's
    as-configured (HLL) run in `p2_repo/logs/`.
- **Results:**
  - All pass, as before; the flux-cap-alone ring test is the strict xfail, as before.
  - Metrics agree to the printed 3-4 digits:
    - oblique leak ratio 2.1818e-2 both;
    - ring 3a L1/Linf/kappa_perp,num identical, 3c L1 2.00e-2 -> 1.99e-2;
    - T2/T3 rates identical except the last digit of T2c and T3 D_z (9.5e-5 -> 1.02e-4).
  - The spurious velocities (1e-11 ... 1e-9) are 5-20% larger with HLLC; still negligible.
  - div B stays at round-off: 5.9-7.1e-15 vs 6.8-8.0e-15.
- **Conclusion:** these tests measure CR transport through static or uniform gas, where P2 does
  nothing (consistent with the hydro sweep). Among the MHD tests, only A1.3 (contact exact) gains.
  `cr_mhd_energy_budget` is unchanged in quality: energy at round-off, and E_cr shifts by -5.7%
  with no reference to judge it.

### 2026-10-10 (afternoon): P2 extended to MHD; M7 changes -- gravity is not well-balanced with P2

- **Code:** `cr_total_pressure_flux` no longer excludes MHD.
  - The FV MHD scheme is a Strang split (gas half-step / `magnetic_update` / gas half-step);
    the gas half-steps are ordinary HLLC solves on the gas-only state.
  - A1.3's 3D MHD slabs now run with HLLC and the round-off gates.
  - DESIGN.md is updated.
- **A1.3 MHD** (`/export/scratch/nknoell/phaseA/p2_mhd/logs/`): static 2.2e-16, advected 5.7e-14
  (N 64). Before: 3.3e-2 / 9.2e-3.
- **M7 A/B** (`m7ab/logs/v1000_s4{2,3}_p2.log`, `--res=0.5`, MHD + HLLC; vs yesterday's
  `stagewise`, identical code minus P2; `m7ab/p2_analyze.py`):

  | | s42 stage-wise | s42 P2 | s43 stage-wise | s43 P2 |
  |---|---|---|---|---|
  | E_cr(250) / injected | 1.42 | 2.14 | 1.56 | 2.38 |
  | CR advective inflow, open z | +1.0e11 | +2.6e11 | +1.3e11 | +3.1e11 |
  | compression work | +2.9e10 | +8.9e10 | +4.1e10 | +1.1e11 |
  | eta(1 kpc) | 0.27 | 0.10 | 0.24 | 0.11 |
  | H_gas | 351 | 347 | 339 | 335 |

  The divergence starts at ~100 Myr, and the injection history is identical.
- **Cause (tested):** a CR-supported hydrostatic column (`stratified_hydrostatic_column.py`
  potential, P_cr = 0.9 P_tot, v_red = 0, HLLC, t = 4; scratch `a1/cr_hydrostatic.py`).
  Max core Mach:
  - thermal-only 0.110 / 0.109 (N_z 128 / 256);
  - CR, source coupling 0.011 / 0.015;
  - CR, P2 0.147 / 0.142.
  - The FV gravity source (SIMPLE_SOURCE) is not well-balanced against the Riemann flux (known
    for thermal support: Mach ~0.1, resolution-independent; M0b docstring).
  - The centred -grad P_cr source balances the centred gravity source to O(dx^2). With P2,
    the CR pressure goes through the HLLC flux and inherits the thermal residual.
  - M7's halo is CR-supported. So P2 adds spurious halo flows, more CR-loaded infall through
    the open boundaries, and less outflow.
- **Consequence:** P2 trades contact balance (exact) for hydrostatic balance of CR-supported
  gas (10x worse) as long as gravity is not well-balanced. Proposed (needs user decision):
  - (a) exclude gravity runs from P2 (M7 back to the source coupling); or
  - (b) extend the well-balanced FV gravity reconstruction (opt-in, gas pressure only) to
    P_th + P_cr and use it with P2.
- **Not rerun:** `m7_frozen_field_probe.py` (exploratory, 16 per-variant runs, would overwrite
  its baseline raw data; uniform static gas, P2 only changes the gas response).
- **M7 v1e4 seed 42** (stage-wise -> P2): E_cr/inj 1.76 -> 3.04, CR inflow +1.7e11 -> +4.5e11,
  compression work +5.8e10 -> +1.6e11, eta 0.20 -> 0.073, H_gas 283 -> 248 pc. Same direction as
  v1000, stronger.
- **`cr_mhd_energy_budget.py`** (MHD + HLLC, now P2): PASS, MHD+CR energy error 1.05e-10 (was
  1.02e-10), mass 2.8e-15.

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
    (figures regenerated except item 11). Docstrings re-baselined later the same day (see above).

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
