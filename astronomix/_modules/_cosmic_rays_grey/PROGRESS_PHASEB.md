# Phase B test plan -- PROGRESS

Status tracker for `astronomix_CR_phaseB_test_plan.md` (review + extension of the Phase B ladder
items 7 and 8, DSA injection). IDs below are the plan's. The plan is a frozen review (2026-10-13);
do not edit its "(calc.)/(log)/(fig.)" numbers, record new measurements here (and in the baseline
registry `pytests/cosmic_rays_grey/baselines.json`).

Scope rule: anything that changes the scheme (fix proposals F1-F4, the outcomes of DB1-DB6) is
logged in `PROGRESS.md` / `DESIGN.md` as usual and only linked from here. Fixes to existing
schemes need the user's go-ahead before they are implemented.

Status: `todo` / `running` / `done` / `blocked` / `dropped`.

## Status table

| ID | Prio | Status | Test file | Key result | Date |
|---|---|---|---|---|---|
| **Item 7: CR Sedov-Taylor blast** | | | | | |
| B7.1 self-similar two-fluid Sedov reference | 1 | done | `reference_solutions/sedov_two_fluid.py`, `cr_sedov_two_fluid_reference.py` | Sedov `xi_0` 1.15167 / 1.03278 (5/3, 7/5), energy integral <= 4e-5, plan table to 1e-3; code convention (`"gas_rh"`, `zeta` 0.1): `E_cr/E` 0.1398, injection 0.0416 E/t, finder dissipation 0.416 E/t | 2026-10-14 |
| B7.2 Sedov from the self-similar state, `v_red = 0` | 2 | todo | | | |
| B7.3 instantaneous injection rate vs analytic | 1 | done; findings for DB4 + new area under-count (Y10) | `cr_sedov_injection_rate.py` | late-time injection rate 0.43 / 0.53 / 0.60 of the reference at N 48 / 96 / 160 (order ~0.3), `E_cr/E` 0.29 / 0.39 / 0.45. Finder dissipation / simulated shock = area 0.85 / 0.87 / 0.84 (= `<|n|_inf>` 0.831, resolution-independent) x flux per cell 0.45 / 0.59 / 0.65 (= `(M_fin/M_true)^3`, Mach ratio 0.76 / 0.84 / 0.87). Cap: 0 | 2026-10-14 |
| B7.4 planar shock tube with injection vs exact (Pfrommer et al. 2017 App. B/C) | 1 | done; new finding Y11 (DB8) | `cr_shock_tube_injection.py`, `pfrommer_riemann_solver.py` (`injection=`) | default scheme at the gas-pressure fixed point (N >= 400): `v_s` 8e-4, `P_th` -0.1 / -0.7%, `P_cr` +2.7% (deposit inside the smeared shock), finder M -0.5 / -1.8%. Vs the true Mach number: `P_cr` -2.7% / **-23%** (`zeta` 0.1 / 0.5), finder M -2.2 / -9%. Energy scheme `P_cr` +6 to +16% | 2026-10-14 |
| B7.5 shock slowdown | 2 | todo | | | |
| B7.6 isotropy of the deposit | 3 | todo | | | |
| B7.7 robustness: `dt`, sampling knobs, limiter counters | 3 | todo | | | |
| B7.8 (optional) SN source term, composite blast | 3 | todo | | | |
| **Item 8: DSA efficiency vs Mach number** | | | | | |
| B8.1 realized `eta_eff(M)` at planar shocks | 1 | done | `cr_dsa_efficiency_planar.py` | vs the gas-pressure fixed point: KR13 within 3% for M >= 3, -5.6% at M 2.25; constant +0.3 ... +2.8%. Vs the true Mach number: KR13 **-9%** at strong shocks (-10% at M 5), constant -1 ... -3%. N 400 = N 1600 to 1e-3 | 2026-10-14 |
| B8.2 finder Mach and injection at oblique shocks | 2 | todo; partly answered by the DB7 check | `cr_sedov_shock_area.py` | Sedov: finder M / M_true varies only 0.75-0.78 (N 48) / 0.85-0.88 (N 160) from axis to diagonal; the area factor is `|n|_inf` per cell | 2026-10-14 |
| B8.3 KR13 fit vs published data | 2 | todo | | | |
| B8.4 weak blasts vs 1D spherical reference | 3 | todo | | | |
| B8.5 shocks into CR-loaded gas | 2 | todo | | | |
| B8.6 obliquity-dependent efficiency (needs DB3) | 3 | blocked (DB3) | | | |
| B8.7 sub-critical shocks | 3 | partly done (in B8.1) | `cr_dsa_efficiency_planar.py` | KR13: no injection at M 1.8 / 2 (`E_cr/E_th` ~ 1e-10, floor). Constant model injects above `dsa_mach_min` 1.3 (DB5); M = 1.5 not run | 2026-10-14 |
| **Cross-cutting** | | | | | |
| Y6 register items 7 and 8 in `baselines.json` | 1 | done | `cr_sedov_taylor.py`, `cr_dsa_mach_dependence.py` | N 48 / 96 / 160, 18 + 54 metrics. Item 7 `E_cr/E` 0.0411 / 0.0544 / 0.0626 = 0.294 / 0.389 / 0.448 of the self-similar reference (= B7.3); item 8 KR13 0.0826 / 0.1075 / 0.1224 = 0.315 / 0.410 / 0.466. Item-8 Mach coverage gate restricted to N <= 48 (Y5) | 2026-10-14 |

## Decisions

| ID | Question | Status | Outcome |
|---|---|---|---|
| DB1 | Efficiency convention: KR13 `eta` on the kinetic or the dissipated flux? | open | |
| DB2 | CR transport in the item-7/8 gates (`v_red = 0` hard, production tracked)? | open | |
| DB3 | Obliquity-dependent efficiency for MHD runs? | open | |
| DB4 | Finder Mach bias: correct it (F4 a/b) or accept and gate trends? | open; B7.3 data (2026-10-14) | Mach ratio 0.76 / 0.84 / 0.87 (N 48 / 96 / 160), order 0.57 -> 0.38; enters the injection cubed (0.45 / 0.59 / 0.65); 0.95 would need N ~ 1400 |
| DB5 | Sub-critical threshold of the constant model (1.3 / 2.25 / 3)? | open | |
| DB6 | Item-7 gated IC: self-similar state instead of the point explosion? | open | |
| DB7 (new) | Area under-count of oblique shock surfaces (Y10): correct the per-cell area by `1/|n|_inf`? | open; B7.3 + DB7 check (2026-10-14): Y10 confirmed with exact normals (1.01-1.05), local normals over-correct (1.06-1.13) | spherical shock: 0.84-0.87 of the dissipation at every N; with the finder's normals the correction gives 1.07-1.10 (over-corrects, noisy normals). B8.2 (planar oblique) gives the exact factor |
| DB8 (new) | CR back-reaction on the finder Mach number (Y11): the finder reads M from the **gas**-pressure jump, which the injected CRs lower. (a) Mach from the total-pressure jump (composite RH), (b) from the density/velocity jump, or (c) accept? | **decided 2026-10-14: (c) accept for now** (user); document that the realized efficiency is lower, no finder change. Revisit with DB4/B8.5 | Resolved planar shocks settle at the self-consistent fixed point. Injected / intended flux (exact): 0.95 at `zeta` 0.1 (finder M -1.7%), 0.77 at 0.5 (-7.4%), 0.89 for KR13 at strong shocks (-3.8%). Measured vs the true-M reference: -2.7% / -23% / -9% (partly offset by the +2.7% deposit excess). Resolution-independent. In the item-7 Sedov run (`zeta` 0.1) ~2% of the Mach ratio, ~5% of the injection |

## Infrastructure

- [x] `test_setups/reference_solutions/sedov_two_fluid.py` (B7.1): self-similar two-fluid Sedov,
      conventions `"total"` (Pfrommer `zeta`) and `"gas_rh"` (the code's), profiles in physical
      units, Lagrangian map. Replaces nothing yet: the placeholder `sedov_solver.py` is untouched
      (it has no users).
- [x] Snapshot analysis helper for the finder/injection rate on saved states (B7.3): in
      `cr_sedov_injection_rate.py` (`_analyse`, `_derived`); factor out when B7.7/B8.4 need it.
      Snapshot states are kept in `/export/scratch/nknoell/phaseB/b7_3/` (`CR_B73_REUSE=1`
      reanalyses them without rerunning).
- [x] Injection extension of `pfrommer_riemann_solver.py` (B7.4, B8.1, B8.5): `ShockInjection(zeta, convention, mach,
      mach_min)` via `injection=` (default None = unchanged), conventions `"total"` (Pfrommer et al. 2017; reproduces
      their Table 1) and `"gas_rh"` (the code's) with `mach="true"` or `"gas_pressure"`, `zeta` may be a function of M
      (KR13 fixed point); `pfrommer_shock_structure` returns star states, shock speed, Mach numbers and fluxes. Fixed on the
      way: the Hugoniot bracket now starts at `rho_K` and the mass flux has an acoustic limit (infinitesimal left shocks of
      RH-state tubes failed before). Existing users (`cr_shock_tube.py`, `cr_shock_tube_partition.py`) unaffected.
- [ ] Self-similar initial-condition builder on the Cartesian grid (B7.2, B7.5, B7.6)
- [ ] Digitized KR13 / Dubois App. B data (B8.3), 1D spherical reference (B8.4)
- [ ] Shared shock-position fit: first version in `cr_sedov_injection_rate.py`
      (`_shock_radius`: front midpoint + pressure peak of shell averages; `_derived`: fit
      `R = R_off + xi t^{2/5}` to both, mean `xi`). Front and peak `xi` agree to 1.9% / 1.2% /
      0.4% at N 48 / 96 / 160

## Log (newest first)

### 2026-10-20: DB4 = (a) velocity-jump Mach, DB7 = smoothed-normal area weights: implemented (step 3, uncommitted)

The user decided on 2026-10-19: DB7 smoothed-normal weights, DB4 option (a).

**Code.**
- `shock_finder3D/pfrommer_shock_finder.py`: new `mach_estimator="pressure"|"velocity"`. The default stays `"pressure"` (the finder tests, CWB finder diagnostics and non-DSA callers are unchanged). `"velocity"` takes the normal-velocity jump between the adaptive walk's own post/pre sample points (`_shock_zones.sample_along_direction`, a new helper that reads fields at the walk's points). It requires `mach_sampling_adaptive`.
- **Guard (found in this step):** a velocity-Mach cell must also show the pressure jump at the same points, `M_p > mach_min` and `M_p − 1 ≥ f (M_u − 1)` with `f = mach_velocity_consistency` (0.5). This is the converse of the existing velocity check.
  - Without it, B8.2 (2,1,0), M = 10, ζ = 0.1, N = 64 went NaN at the last step.
  - Cause: 4026 zone cells next to the box faces, whose walk clamps instead of wrapping. Their normals were reversed and the pressure *fell* across them (p_pre 57 vs 1.4, M_p = 1), but they had a velocity difference: M_u 2.4, flux 780 from the far side's sound speed. That drained them to the pressure floor, and the next hydro step gave NaN everywhere (via dt).
  - With velocity Mach or area weights alone the drain stayed just below 100%.
  - The guard removes these cells. No Sedov cell is affected: B7.3 is identical to 4 digits with and without the guard.
- **Gradient fix (found in this step):** the velocity projection onto the normal uses `stop_gradient(shock_direction)`.
  - Without it the Phase-D inference rollout (`cr_phase_d_injection_efficiency_inference.py`) had a NaN AD gradient. The cause: the derivative of `-grad T/|grad T|` is singular in uniform gas, and a masked-out cell turns that into 0·inf = NaN, the same issue as the walk's FIXES_TODO round 24.
  - With it: AD −0.4812 vs FD −0.4802 (relative error 2.1e-3; 5.1e-3 before this change). Forward results are unchanged.
- `shock_finder3D/_surface_area.py` (new): `surface_area_weights` = 1/|n|∞ of `-grad T` of the Gaussian-smoothed temperature (σ in cells, separable, edge-padded; 1 in 1D).
- `CosmicRayGreyConfig`: `dsa_mach_from_velocity=True`, `dsa_area_weighting=True`, `dsa_area_normal_smoothing=2.0`. `dsa_shock_finder_kwargs` passes the estimator; `inject_crs_at_shocks` multiplies the per-surface-cell amount by the weight before the post-shock spread.
- `pfrommer_riemann_solver.py`: `ShockInjection(mach="velocity")`, the fixed point at the velocity-jump Mach number (explicit, since the jump is fixed by `(p*, rho*)`), plus a `mach_gas_velocity` output.

**Tests adapted.**
- B7.3: D, flux ratio and area use the injection's weight; new metric `dsa_area_ratio_inj`; figure panel (b) title.
- B7.4: `FINDER_MACH = "velocity"`; the gates compare with that fixed point ("fp" metrics, previously "gp"); the self-test checks the velocity convention.
- B8.1: the same reference switch. The KR13 M = 2 tube is now tracked only: KR13 is 1.6e-4 at M = 2 and some cells read M_u slightly > 2, giving E_cr/E_th = 1.7e-8.
- Item 8: the formula cross-check applies the weight; the weak-blast coverage bound goes 2.6 → 3.0 (the velocity estimator reads 2.69 for the weakest blast at N = 48).
- `_finder_estimators.py`: the closure follows the finder's estimator.

**Regression** (scratch copy `/export/scratch/nknoell/phaseB/db47_update/`, `CR_UPDATE_BASELINES=1`, logs `logs/r2/`; raw data `/export/scratch/nknoell/phaseB/db47/`).

B7.3 late means, ζ = 0.1, before → after:

| N | Mach ratio | area (injection) | injection/ref | E_cr(t_end)/ref | E_cr run / kernel prediction |
|---|---|---|---|---|---|
| 48 | 0.763 → 0.876 | 0.851 → 1.054 | 0.431 → 0.766 | 0.294 → 0.515 | 1.19 → 1.20 |
| 96 | 0.841 → 0.935 | 0.874 → 1.072 | 0.527 → 0.861 | 0.389 → 0.672 | 1.12 → 1.12 |
| 160 | 0.869 → 0.943 | 0.841 → 1.028 | 0.600 → **0.918** | 0.448 → 0.740 | 1.11 → 1.10 |

The plan's B7.3 target (≥ 0.9 at N = 160) is met.

| Test | Result |
|---|---|
| Item 7 | E_cr/E 0.0720 / 0.0940 / 0.1035 = 0.515 / 0.672 / 0.740 of the self-similar 0.1398 (was 0.041 / 0.054 / 0.063); passes |
| Item 8 | KR13 E_cr/E 0.149 / 0.193 / 0.211 = 0.57 / 0.73 / 0.80 of 0.2625 (N 48 / 96 / 160); CS14/KR13 ratio error 1.6 / 2.7 / 3.2% (was 4.6 / 6.4 / 7.6% vs the hard 8%, the Y6 watch item); passes at all three |
| B8.2 (real injection, N = 128) | injected flux per area / exact 0.989–1.002 (ζ = 0) and 1.010–1.039 (ζ = 0.1) at all 5 orientations and M = 3, 10. Angle dependence < 2% (target < 5%; was 43%). No NaN at N = 64 or 128 |
| B7.4, ζ = 0.1 | vs velocity fixed point: v_s −2e-4, P_th −0.15%, P_cr +2.6% (the deposit-inside-shock excess), ρ +0.09%, finder Mach −0.4 to −0.7%, L1 order 0.89; passes. vs the true Mach number: P_cr +6.6%, rate +6.3% |
| B7.4, ζ = 0.5 | **fails** vs the velocity fixed point: v_s +0.57%, P_th +7.4%, P_cr −14%, ρ −2.8%, rate −11%, finder Mach −2.8 to −6%, L1 order 0.38. The sim stays between the true-Mach and the velocity fixed point (vs true: P_cr +8%, rate +6.2%) |
| B7.4 CR tube | M_f 13.26 vs fixed point 13.62 |
| B8.1 (N = 1600), vs velocity fixed point | KR13 +1.4 to +2.3% (M ≥ 7), −1.0 to −3.4% (M 3–5), −3.2 / −5.2% (M 2.5 / 2.25); constant ζ −1.5 to +2.6%; passes |
| B8.1, vs the true Mach number | KR13 +11% at strong shocks (was −9%), +7.3% at M 5; constant ζ +0.3 to +6.6% |
| Item 10, wind bubble | E_cr/E_tot 0.043 → 0.108 (2.5x; no exact DSA reference); hard gates pass |
| Item 11, clumpy SNR (128³ scratch) | uniform E_cr 465, clumpy 455: −2.2% (was −6.7%); **fails** the "clumpy differs by ≥ 5%" gate |
| Pion bump (128) | passes (SED slope 0.120, low-energy suppression 0.949) |
| CWB, N = 128 (`cwb_db47.py`) | E_cr 5.68e-5 → 8.00e-5 (E_cr/E_tot 0.23% → 0.33%) although the velocity Mach median is lower (4.9 vs 6.2) |
| Modified shock structure, two-fluid reference, energy budget, MHD energy budget, gradient checks (5) | pass |
| Phase-D injection-efficiency inference | passes: AD-vs-FD 2.1e-3, recovered mach_scale 1.0626 (was 1.0634) |
| shock_finder3D | synthetic smeared shock and Sedov correctness (256) / convergence pass; `analytic_wind_zone` not run (`pytest` missing in the venv; wind-module tests only) |

- **Registry and figures.** `baselines.json` was merged into the repo from the scratch copy (196 stale `gp` keys of B7.4/B8.1 removed; no test partially refreshed), and the regenerated figures of the passing DSA tests were copied in (not item 11's). Docstrings of B7.3, B7.4, B8.1 and Phase D were updated.
- **Decided (user, 2026-10-20):** (1) B7.4 ζ = 0.5 tracked only: hard gates for ζ = 0.1 only (`GATED_ZETAS`); rerun in compare mode passes. (2) Item 11's clumpy effect tracked only: `cr_snr_clumpy_medium.py` now takes `CR_NUM_CELLS` (default 256) and registers `N128_clumpy_effect` −0.0224 and E_cr/E_tot uniform 0.0924 / clumpy 0.0903 in `baselines.json`; passes at 128, figure copied. (3) CWB checked (below); wind bubble accepted without a closer look.
- **CWB split** (`shock_finder3D/cwb_split.py` in the scratch copy, N = 128, ζ = 0.1, logs `logs/r3/`): E_cr old 5.67e-5, area weights only 6.15e-5 (**+8%**), velocity Mach only 7.04e-5 (**+24%**), both 8.01e-5 (+41% = 1.08 x 1.24 x 1.05 interaction). At a fixed final state the velocity estimator raises the summed dissipated flux by 14% (2.18e4 → 2.49e4): it reads lower in the many weak cells (hence the lower median) but higher in the strong ones that carry the flux (flux-weighted Mach 7.23 → 7.34; 7.19 → 7.60 at the new run's state), still below the expected pre-shock M ~ 10 (Chevalier & Clegg at r_s/R_inj). Flux-weighted area weight 1.07–1.11 (cell mean 1.21). Flux within 4 cells of the box faces 0.2–0.6%: no boundary artefact. The increase is the same correction as in Sedov, not a new effect.
- **CWB figures regenerated** with the new defaults (`pytests/shock_finder3D/figures/cwb_cr/cwb_cr_{influence,orbital_slice}_{64,128}.png`, `cwb_default/cwb_shocked_cells_{2d,3d}_{64,128}.png`; the 128 ones replace stale 2026-08 versions). N = 256 not possible (CR-on run ~13.8 GiB > 11 GB); the `*_256` figures are still from 2026-09. The shocked-cell/Mach panels show `run_cwb`'s own finder call (pressure Mach), not the DSA velocity Mach.
- **Found while checking the figures (pre-existing, not from DB4/DB7):** the white patches in the orbital-slice maps (also in the 2026-10-02 version) are cells with **negative e_cr**: N = 64, 40554 of 262144 cells (15%), min −4.6e-6 (max e_cr ~3e-3), in star 2's shocked tail, although `minimum_e_cr` = 1e-10 should floor them. Not fixed; open for the user.
- **B7.4 ζ = 0.5, why:** in the smeared CR-loaded shock the sampled velocity jump is smaller than the exact post-shock one, so the fixed-point picture does not describe the sim at very high ζ.
- **Item 11, a plausible reading:** part of the old clumpy effect was the area under-count, since a corrugated shock has more oblique surface cells.

### 2026-10-19: B8.2 oblique planar shocks = DB4/DB7 fix plan step 2 (diagnostic, uncommitted)

`pytests/cosmic_rays_grey/cr_oblique_planar_shocks.py` (new, no gates).
- **Setup.** 3D periodic box with a RH slab along normals (1,0,0), (2,1,0), (1,1,0), (2,1,1), (1,1,1). M = 3 and 10, ζ = 0 and 0.1, N = 64 and 128.
- **Measurement.** Interior cube with a 10-cell margin, because the finder walk clamps at the box faces. True area = the exact plane–cube intersection.
- **True Mach.** Initial RH M for ζ = 0; for ζ = 0.1, M_init × (tracked speed ζ=0.1 / ζ=0). The front tracking alone reads 1–4% high.
- **Shared code.** Estimators moved to `_finder_estimators.py`; `cr_sedov_mach_estimators.py` uses it and reproduces its N=48 numbers to 1e-7.
- **Output.** Raw data in `/export/scratch/nknoell/phaseB/b8_2/`; figure `pics/08_dsa_mach_dependence/cr_oblique_planar_shocks.png`.

Results at N = 128 (means over 4 snapshots):

| k | \|n\|∞ | cells/area | area, finder normal | area, Gauss σ=2 | finder normal error | injection now (probe, ζ=0) | velocity + σ=2, ζ=0 | velocity + σ=2, ζ=0.1 |
|---|---|---|---|---|---|---|---|---|
| (1,0,0) | 1.000 | 1.000 | 1.000 | 1.000 | 0° | 0.99 | 0.99 | 1.03 |
| (2,1,0) | 0.894 | 0.895 | 1.05 | 1.005 | 5.2–5.7° | 0.88–0.89 | 0.995 | 1.01–1.02 |
| (1,1,0) | 0.707 | 0.707 | 1.000 | 1.000 | 0° | 0.70–0.71 | 0.996–0.999 | 1.02 |
| (2,1,1) | 0.816 | 0.816 | 1.07–1.08 | 1.006 | 3–4° | 0.81–0.82 | 0.992–0.994 | 1.02 |
| (1,1,1) | 0.577 | 0.577 | 1.000 | 1.000 | 0° | 0.57–0.58 | 0.995–0.996 | 1.00 |

- **DB7 confirmed exactly.** Surface cells per area equal \|n\|∞ to 4 digits. The current injection therefore depends on angle by 43% (plan target < 5%).
- **Weighting fix.** 1/\|n\|∞ with σ = 2 smoothed-temperature normals gives an angle dependence < 1% (ζ = 0) and < 3% (ζ = 0.1).
- **Finder normals on planar shocks.** -∇T is exact for the symmetric orientations but off by 3–6° for (2,1,0) and (2,1,1), a stencil effect on the staircased front, so it over-weights by 5–8%. σ = 2 cuts the error to 0.5°.
- **Mach, ζ = 0, N = 128.** Pressure 0.994–0.999 and velocity 0.992–0.998: both fine on planar shocks.
- **Mach at N = 64 oblique (2,1,0)/(2,1,1), M = 10.** The medians drop to 0.85–0.91 for both estimators (wide per-cell scatter; mean flux with exact weights still 1.00–1.03). Not looked into further.
- **ζ = 0.1.** Pressure reads 0.983–0.994 (the DB8 fixed point); velocity reads 1.000–1.010 (CR-modified compression).
- **Exact 1D references** (`pfrommer_shock_structure`, injection at the true Mach number), flux from the velocity jump vs from the pressure jump:

  | ζ | M | flux(M_u) / exact | flux(M_p) / exact |
  |---|---|---|---|
  | 0.1 | 10 | +3.7% | −5.5% |
  | KR13 (0.21) | ≥ 10 | +7–8% | −11 to −13% |

  The velocity error is smaller, of opposite sign, and grows with ζ. A ζ-aware correction (effective γ from P_cr/P_th, or v_s = Δu r/(r−1) from the density jump) would remove it but was not tried.
- **Conclusion for DB4/DB7.** Planar shocks barely distinguish the two Mach estimators; the difference comes from the Sedov shell (step 1). The area weighting is needed at every non-aligned shock, and needs smoothed normals.

### 2026-10-19: DB4/DB7 fix plan step 1, Mach estimators + smoothed normals (diagnostic, uncommitted)

`pytests/cosmic_rays_grey/cr_sedov_mach_estimators.py` (new, no gates): it repeats the finder's adaptive walk on the B7.3 snapshots and compares estimators at the same post-/pre-shock sample points. Raw data and logs are in `/export/scratch/nknoell/phaseB/db4/`; the figure is `pics/07_sedov_taylor/cr_sedov_mach_estimators.png`. The finder flux is reproduced exactly (closure 1.00000).

Late means for the DSA run (ζ = 0.1); the control run (ζ = 0) agrees to about 1–2%:

| N | M_p | M_u | M_T | √(M_p M_T) | M_ptot | flux/area p | flux/area u | area exact | area finder | area box5 | area Gauss σ=2 | inj/ref now | inj/ref u + σ=2 | inj/ref u + exact |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 48 | 0.764 | 0.876 | 1.128 | 0.931 | 0.774 | 0.448 | 0.656 | 1.021 | 1.104 | 1.097 | 1.049 | 0.431 | 0.778 | 0.757 |
| 96 | 0.841 | 0.936 | 1.091 | 0.957 | 0.854 | 0.592 | 0.809 | 1.049 | 1.118 | 1.116 | 1.068 | 0.527 | 0.880 | 0.865 |
| 160 | 0.869 | 0.943 | 1.035 | 0.947 | 0.886 | 0.653 | 0.839 | 1.010 | 1.071 | 1.070 | 1.024 | 0.600 | 0.941 | 0.928 |

(Mach columns are median M_est / M_true; area columns are Σ dx²/|n|∞ / 4πR².)

- **Velocity jump (F4a).** It roughly halves the Mach deficit. The flux factor rises from 0.45/0.59/0.65 to 0.66/0.81/0.84, but stalls between N = 96 and 160 at a ~6% Mach deficit (still the smeared, falling post-shock profile). The largest velocity along the walk equals the velocity at the pressure-peak point.
- **Temperature jump.** It overshoots (1.13 → 1.03) and converges from above. √(M_p M_T) is 0.93–0.96 but not monotone in N: an accidental cancellation, not recommended.
- **Gas + CR pressure.** Only +1–2% over the gas-pressure Mach, so DB8 is a minor effect at ζ = 0.1.
- **Normals.**
  - The finder's -∇T normal is tilted *systematically* away from the grid axes: ⟨|n|∞ − |n_exact|∞⟩ = −0.06/−0.05/−0.046, median angle 7.4/6.0/5.4°.
  - Averaging over 3³ or 5³ surface boxes does not help (the error is coherent, not noise).
  - -∇T of a Gaussian-smoothed T (σ = 2 cells) cuts the tilt to −0.02/−0.015/−0.011, giving an area within 1.3–3% of the exact-normal weighting.
  - The exact-normal area itself is 1.01–1.05, consistent with the ±2% R_true uncertainty.
- **Combined (velocity Mach + σ = 2 weights).** Implied injection/ref is 0.78/0.88/0.94, against 0.43/0.53/0.60 now.
- **Caveats.**
  - This is a reanalysis of states evolved with the current injection. A run with the new estimator injects more CR and modifies the shock; the γ = 5/3 velocity formula then overestimates by a few percent.
  - Planar accuracy (B7.4/B8.1) and B8.2 oblique shocks are not yet checked: that is step 2.
  - No finder change has been made; DB4/DB7 are open for the user.

### 2026-10-19: B7.3 figure redesign + adiabatic-memory check (uncommitted)

`cr_sedov_injection_rate.py`: new `_plot` (every panel a ratio to the reference: radius, finder
Mach, finder dissipation vs area x Mach^3, injection, `E_cr` with kernel prediction, late-time
summary bars) and `_adiabatic_kernel` / `_kernel_prediction`. The kernel follows shells through
the two-fluid reference: a shell keeps `K ~ (R_s/R)^(3/4)` of its injected CR energy
(Bell 2015, `P^(1/4)` in the isobaric interior; local exponent 0.38-0.28), not `1/R`
(homologous; would give `E_cr/E` 0.104). New hard gate: kernel closure on the reference
(1.00001, tol 1e-3). Rerun N 48/96/160 on GPU (UUID-pinned): all 42 metrics reproduce.

- Adiabatic losses do **not** explain the `E_cr` deficit: the run holds 1.19 / 1.12 / 1.11 x
  the `E_cr` predicted from its own injection rate with the reference kernel (losses not too
  large). The extra gap between `E_cr` (0.29/0.39/0.45) and late injection (0.43/0.53/0.60) is
  memory: 53% of the reference `E_cr(t)` was injected before 0.1 t, and the run is worst early.
- Not resolved: the 11-19% surplus (deposit inside the compressing shock suspected, unverified);
  a direct `P_cr div u` budget on snapshots leaves 10-24% of the injection unaccounted for
  (central differences across the shock too crude to tell).
- 2-page explainer: `reports/sedov_injection_rate/main.pdf`.

### 2026-10-14: DB7 check, shock area with exact normals (done; diagnostic)

`pytests/cosmic_rays_grey/cr_sedov_shock_area.py` (reanalyses B7.3's saved snapshots, DSA run,
`zeta = 0.1`, t >= 0.03; no new simulation, < 1 min per N on GPU), figure
`pics/07_sedov_taylor/cr_sedov_shock_area.png`, raw numbers `/export/scratch/nknoell/phaseB/db7/`.
No gates or baselines yet.

| N | 48 | 96 | 160 |
|---|---|---|---|
| cell count `n_surf dx^2 / 4 pi R^2` (= B7.3 area) | 0.851 | 0.874 | 0.841 |
| re-weighted `1/|n|_inf`, **exact** radial normal | **1.021** | **1.049** | **1.010** |
| same, finder normal (`-grad T`) | 1.104 | 1.118 | 1.071 |
| same, `-grad rho` / `-grad P` | 1.122 / 1.103 | 1.130 / 1.114 | 1.075 / 1.064 |
| median angle finder normal vs radial | 7.4 deg | 6.0 deg | 5.4 deg |
| dissipation re-weighted (exact) / simulated shock's | 0.459 | 0.624 | 0.663 |
| `(M_fin / M_true)^3` (B7.3) | 0.445 | 0.595 | 0.655 |

- **Y10 confirmed.** With the exact normal the `1/|n|_inf` weight restores the sphere area to
  1-5% at every N (0.84-0.87 without it). Per obliquity bin the surface-cell density follows
  `|n|_inf` (0.63 ... 0.98); at N 160 every bin is within 4%. The leftover excess sits in the
  near-axis bin (`|n|_inf` > 0.95: 1.09 / 1.07 / 1.02 against 0.975) and shrinks with N.
- **Closure.** With the area corrected exactly, the remaining dissipation deficit equals the
  Mach-only factor `(M_fin/M_true)^3` to 3 / 5 / 1%. The late-time Sedov deficit is therefore
  area x Mach^3 with the area factor now derived per cell, not matched to a sphere average.
- **A per-cell correction with local normals over-corrects** by 7-12%: the normals scatter by a
  median of 5-7 deg around the radial direction, and the mean of `1/|n|_inf` grows with that
  scatter. `-grad P` is slightly better than the finder's `-grad T`, `-grad rho` slightly worse;
  none is good enough as is. A fix (F-proposal for DB7) needs smoother normals or a different
  area estimate. DB7 stays open (user decision).
- **The Mach bias barely depends on obliquity**, and is largest for shocks *along* the axes:
  finder M / M_true 0.75 (axis) to 0.78 (diagonal) at N 48, 0.85 to 0.88 at N 160. So it is not
  an oblique-sampling effect; it shrinks with resolution (likely post-shock smearing behind the
  Sedov shock; the 1D planar finder in B8.1 is within 1% of its fixed point). This answers
  B8.2's "geometric vs smearing" question for the Sedov blast approximately (B8.2 itself not done).

### 2026-10-14: Y6 items 7 and 8 registered in `baselines.json` (done)

Both item tests record per resolution (`CR_NUM_CELLS`, metric suffix `_N<cells>`) at N = 48, 96
and 160: item 7 18 metrics, item 8 54. Update runs plus gate-mode reruns (N 48 / 96 for both
items, 160 for item 7) on 2080 Tis, all pass; gate-mode values reproduce the registry to float32
noise. Item 7 at N 160: ~18 min; item 8 at N 160 (8 blasts): ~60 min. Logs in
`/export/scratch/nknoell/phaseB/y6/`.

| N | 48 | 96 | 160 |
|---|---|---|---|
| item 7 `E_cr/E` (/ self-similar 0.1398) | 0.0411 (0.294) | 0.0544 (0.389) | 0.0626 (0.448) |
| item 7 `r_shock` (outermost 1% density cell) | 0.460 | 0.433 | 0.419 |
| item 7 conservation (control / DSA) | 1.3e-5 / 1.3e-5 | 3.2e-5 / 3.2e-5 | 6.0e-5 / 5.8e-5 |
| item 8 KR13 `E_cr/E` (/ self-similar 0.2625, `zeta` 0.211 `"gas_rh"`) | 0.0826 (0.315) | 0.1075 (0.410) | 0.1224 (0.466) |
| item 8 CS14-like `E_cr/E` | 0.0432 | 0.0572 | 0.0658 |
| item 8 CS14/KR13 ratio error (hard gate 8%) | 4.6% | 6.4% | **7.6%** |
| strong-blast finder median M / exact Sedov 176 | 0.795 | 0.865 | 0.901 |
| weak-blast finder medians (`P_amb` 0.02 ... 0.4) | 9.94 ... 2.48 | 10.82 ... 2.67 | 11.27 ... 2.77 |

- Item 7 matches B7.3's time-integrated `E_cr/E` (0.29 / 0.39 / 0.45) exactly: same runs.
- **Item-8 coverage gate changed (test only).** At N 96 the weakest blast's median was 2.67, above
  the gate's 2.6, as the plan predicted (Y5: the bounds are 48^3 finder medians, which move
  towards the true Mach numbers with N). The gate now applies at N <= 48 only; above, the
  medians are tracked in the registry. B8.4 replaces it with a gate against a 1D reference.
- **Watch:** the CS14/KR13 ratio error grows with N and is at 7.6% of the hard 8% tolerance at
  N 160 (the time-integrated feedback described in the item-8 docstring). It will likely fail at
  224^3. Not changed.
- The conservation errors grow with N (more float32 steps) but stay below the registry floor
  1e-4, so those gates are floor-bound at every N.

### 2026-10-14: session end, resume point

Done today (all uncommitted): B7.1, B7.3, B7.4, B8.1 (+ B8.7 partly), solver extension of
`pfrommer_riemann_solver.py`. DB8 decided: (c). All four test files pass in gate mode
(`baselines.json`: additions only). Y6 done the same day (entry above). **Next:** priority 2
(B7.2, B7.5, B8.3, B8.5, B8.2). Open decisions for the user: DB1-DB7.


### 2026-10-14: B8.1 realized efficiency at planar shocks (done; B8.7 partly)

`pytests/cosmic_rays_grey/cr_dsa_efficiency_planar.py` (`test_kr13_numpy_copy`: numpy;
`test_cr_dsa_efficiency_planar`: GPU, ~40 min), figure
`pics/08_dsa_mach_dependence/cr_dsa_efficiency_planar.png`, raw values
`/export/scratch/nknoell/phaseB/b8_1/b8_1.npz`, 132 tracked metrics.

**Setup.** Tubes from the gas-only RH state at M = 1.8 ... 100 (a single shock without
injection), KR13 and constant `zeta = 0.1`, N 400 and 1600, default scheme, `v_red = 0`.
`eta_eff = e_cr,2 (v_s - u_2) / (rho_1 v_s^3 / 2)` on the plateau. Reference: B7.4's exact
solution with `zeta = eta(M)` solved as a fixed point with the shock, at the true Mach number and
at the gas-pressure Mach number (Y11).

**Results** (N = 400 and 1600 agree to 1e-3):

| M | KR13 vs fixed point | KR13 vs true M | constant vs fixed point / true M |
|---|---|---|---|
| 1.8, 2 | no injection (`E_cr/E_th` ~ 1e-10) | | -1.4% / -1.9% (1.8) |
| 2.25 | -5.6% | -5.7% | +0.3% / -1.2% |
| 3 | -2.2% | -2.6% | +1.3% / -1.4% |
| 5 | -1.0% | -10.3% | +2.2% / -2.2% |
| 10 | +2.5% | -8.6% | +2.6% / -2.7% |
| >= 15 | +2.8 ... +2.9% | -8.8% | +2.7% / -2.9% |

- All plan gates pass against the fixed point (5% for M >= 3, 20% for 2.25 <= M < 3, floor at M
  <= 2). The deviation is the net of the deposit excess (B7.4) and the 1D finder reading
  0.45-1.0% below the fixed point's Mach number; the latter is amplified on KR13's steep part
  (`eta ~ M^5.3`: -5.6% at M 2.25). The plan's "< 0.1%" 1D finder accuracy is not reached.
- Against the true Mach number KR13 loses **9%** at strong shocks through Y11 (`zeta = 0.211`
  lowers the finder's M by 3.8%), and 10% at M = 5, where the fixed point falls onto KR13's
  weak piece (finder 4.89). The constant model (`zeta = 0.1`) loses 1-3%.
- These are the realized efficiencies in the code's convention `delta(M) eta(M)` (DB1). Under
  KR13's own kinetic-flux definition the strong-shock target is 0.211 (dotted curve in the
  figure), against 0.108 realized.
- B8.7: KR13 has no injection at M 1.8 and 2. The constant model injects at every M above
  `dsa_mach_min = 1.3` (DB5 open); the plan's constant-model point at M = 1.5 was not run.

### 2026-10-14: B7.4 planar shock tube with injection vs exact (done)

`pytests/cosmic_rays_grey/cr_shock_tube_injection.py` (`test_injection_riemann_reference`: numpy,
~25 s; `test_cr_shock_tube_injection`: GPU, ~20 min), figure
`pics/07_sedov_taylor/cr_shock_tube_injection.png`, raw profiles in
`/export/scratch/nknoell/phaseB/b7_4/`, 202 tracked metrics in `baselines.json`.

**Reference.** The plan asked for Pfrommer et al. (2017) App. B/C. Instead of coding their
equations one by one, the existing composite Riemann solver got a general post-shock split: mass,
momentum and total energy jump as before, `P_cr,2 = P_cr,1 x^4/3 + e_inj/3`. With their
`"total"` convention it reproduces Table 1 exactly (M 10.00 -> 9.555, `x_s` 3.884 -> 4.741; CR
tube composite M 10.00 -> 9.559), which pins down their setup (`P_5 = 51.516` and `P_1 = 0.1` are
total pressures). Flux continuity through the shock holds to 1e-10 for every convention, and the
strong-shock limit equals B7.1's state to 1e-4.

**Setup.** Thermal tube (`rho 1 | 0.125`, `P 63.499 | 0.1`), box [0, 10], `t = 0.35`, float64,
`v_red = 0`, constant `zeta` 0.1 and 0.5, the code's convention. Default scheme (HLLC + hybrid,
adaptive sampling with the spread deposit) at N = 100-3200; energy scheme with HLLC and HLL and
the surface-cell deposit at N = 400 and 1600; the CR tube (`X_cr = 2 | 1`) at `zeta = 0.5`,
tracked. Plateau = 55-85% from contact to shock; shock speed and injection rate from 6 snapshots.

**New finding Y11: the finder's Mach number feels the injected CRs.** The finder reads M from the
jump of the *gas* pressure. Injection removes thermal pressure behind the shock, so the finder
reads a lower M, injects less, and the shock settles at a self-consistent fixed point
(`ShockInjection(..., mach="gas_pressure")`). The code sits exactly there:

| `zeta` | M true | M at fixed point | finder M (N 400-3200) | injected / nominal (exact) | `P_cr` vs true-M ref | vs fixed point |
|---|---|---|---|---|---|---|
| 0.1 | 9.913 | 9.748 | 9.70 | 0.948 | -2.7% | +2.7% |
| 0.5 | 9.600 | 8.893 | 8.73-8.75 | 0.766 | **-23%** | +2.7% |

Not resolution-dependent. It grows with `zeta`; for KR13 at strong shocks (`zeta` 0.211) it is
-9% (B8.1). In the Sedov blast (B7.3, `zeta = 0.1`) it explains ~2% of the 0.87 Mach ratio.
Against the true-M reference the `zeta = 0.5` shock is 0.8% too fast (less CR pressure). Logged as
DB8; nothing changed. **DB8 decided (user, 2026-10-14): option (c), accept and document;** the B7.4/B8.1
gates stay against the gas-pressure fixed point, the true-Mach deviations are tracked in `baselines.json`.

**Against the fixed point** (default scheme, identical for N = 400-3200): shock speed within
8e-4, `P_th` -0.14% / -0.70%, density +0.1% / +0.3%, `P_cr` **+2.7%** for both `zeta` and the box
CR energy growth +2.6% / +2.3% over the exact injected flux. The excess is the deposit location:
the spread deposit puts CRs into the smeared shock, where the gas is still compressed, so they
gain adiabatic energy the exact solution does not have (Dubois et al. 2019 discuss the same
bias). L1 density order 0.88 / 0.77 (N 200-3200). The plan's gates were met except `P_cr` within
2% (gate set at 5%) and finder M within 1% of 9.56 (that number is Pfrommer's convention; the
code's finder is -0.5% / -1.8% from its own fixed point, gate 3%).

**Schemes** (plateau `P_cr` vs fixed point, `zeta` 0.1 / 0.5): default +2.7 / +2.7%, energy
HLLC +6.3 / +8.3%, energy HLL +10 / +16% (`P_th` -4.2%, shock speed -0.44% at 0.5). The energy
scheme adds the Phase A spurious CR shock heating on top. Weber et al. (2023)'s shock-speed error
of entropy-type CR variables with injection does not appear with the hybrid (speed 8e-4).
Surface-cell deposit (`dsa_adaptive_shock_sampling = False`): finder M ~6 instead of ~9.7,
`P_cr` -47% / -45%.

**CR tube** (`zeta = 0.5`, tracked; B8.5): against its gas-pressure fixed point `P_cr` +2.3%
(injected part +2.6%), `P_th` -0.7%; the finder reads 11.52-11.55 against the fixed point's
11.74 and the composite Mach number 9.70 (the gas-only Mach number of a CR-loaded upstream is
not the composite one; B8.5 / DB4).


### 2026-10-14: B7.3 instantaneous injection rate (done)

`pytests/cosmic_rays_grey/cr_sedov_injection_rate.py`, figure
`pics/07_sedov_taylor/cr_sedov_injection_rate.png`, raw data and snapshot states in
`/export/scratch/nknoell/phaseB/b7_3/`, 42 tracked metrics (14 per N) in `baselines.json`.
GPU runs (2080 Ti, UUID-pinned): N 48 ~3 min, 96 ~6 min, 160 ~20 min for both runs.

**Method.** The item-7 runs unchanged (default scheme HLLC + hybrid, `v_red = 1`, adaptive
sampling), ten snapshots each. On every snapshot: `find_shocks_pfrommer` and
`inject_crs_at_shocks` exactly as in the run. Reference: `sedov_two_fluid`, code convention.
The shock radius and speed come from fitting `R = R_off + xi t^{2/5}` (t >= 0.03) to the
shell-averaged front midpoint and to the pressure peak (they bracket the shock by about +-1 cell
at 48^3). A first version that used only the pressure peak put the shock 5% inside and biased
`M_true` by ~8% (`V ~ R^-3/2`); a version with `dR/dt` of the front was too noisy.

**Results** (late-time means, t >= 0.03; DSA run unless stated):

| N | `xi/xi_0` ctrl / DSA | finder `D` / simulated shock | area `A_cells/4 pi R^2` | flux per cell | `(M_fin/M_true)` (cubed) | injection / ref. | `E_cr/E` / ref. at t_end |
|---|---|---|---|---|---|---|---|
| 48 | 1.009 / 1.025 | 0.381 | 0.851 | 0.448 | 0.763 (0.445) | 0.431 | 0.294 |
| 96 | 0.990 / 1.004 | 0.518 | 0.874 | 0.592 | 0.841 (0.594) | 0.527 | 0.389 |
| 160 | 1.005 / 1.018 | 0.549 | 0.841 | 0.653 | 0.869 (0.655) | 0.600 | 0.448 |

- **The deficit is fully accounted for by two factors.** At late times the finder's dissipation
  equals (surface area) x (flux per cell), and the flux per cell equals `(M_fin/M_true)^3` to
  0.3%. With `delta(M)` saturated at M ~ 150-500, the dissipated flux `~ v^3 ~ M^3`, so the finder's
  Mach bias enters the injection cubed. Positivity cap: 0 (|loss| < 2e-7, float32 noise). The
  instantaneous rate is therefore also too low at late times (plan 7.3 step 4d confirmed), not
  only early.
- **Mach bias** (DB4): ratio 0.76 / 0.84 / 0.87, deficit order 0.57 then 0.38. At order ~0.45,
  a ratio of 0.95 needs N ~ 1400. (PROGRESS.md 2026-10-02 quoted 0.80 / 0.87 / 0.92 at 48 / 96 /
  224 from a different, single-snapshot estimate.)
- **New finding Y10: the injection under-counts the area of oblique shocks.** The cell count of
  the shock surface is 0.84-0.87 of `4 pi R^2 / dx^2` at every N, which is `<|n|_inf>` = 0.831,
  the sphere average of `max_i |n_i|`. One surface cell per grid line along the dominant axis
  gives `|n|_inf` cells per `dx^2` of shock area. `inject_crs_at_shocks` assigns every cell the
  area `dx^2` (area/volume = `1/dx`), so it under-counts the dissipation of a shock with normal
  `n` by `|n|_inf` (0.71 at 45 deg in 2D, 0.58 along a body diagonal). This does **not**
  converge. Re-weighting each cell by `1/|n|_inf` with the finder's normals gives 1.07-1.10
  instead of 1, so the normals (temperature gradient) are too noisy for a per-cell correction
  as is. B8.2 (planar shocks at fixed angles) will give the exact factor. Logged as DB7, no fix
  implemented. Consequence: the plan's B7.3 target (`D/D_ref` >= 0.9 at N 160) is unreachable
  without it; the cap is 0.83 even with an exact Mach number.
- **Time-integrated `E_cr`** converges at order ~0.2 and is lower than the instantaneous rate
  (0.45 vs 0.60 at N 160) because the rate is even lower early on (smaller `R/dx`).
- **Blast dynamics** are fine: control `xi` within 1% of Sedov at all N. The DSA blast's
  `xi` is 0.4-2.5% above the `zeta = 0.1` reference, as expected for a blast that carries
  less CR energy than the reference (`xi` 1.1517 at `zeta` = 0).
- **Early times** (t < 0.03 at N 48): `A_cells` is 2-3x the sphere area and the median surface
  Mach is 0 (most flagged surface cells are below `dsa_mach_min`); the area split is meaningful
  only at late times.

**Implications.** Item 7's `E_cr/E` = 0.041 at 48^3 (0.29 of the reference) decomposes into
area (0.85) x Mach^3 (0.45) at late times and an extra early deficit. Convention (DB1) and
transport (DB2) are separate, not yet measured factors. For B7.2 (self-similar IC) the same
factors apply unless DB4/DB7 change the finder or the injection.

### 2026-10-14: B7.1 self-similar two-fluid Sedov reference (done)

`astronomix/test_setups/reference_solutions/sedov_two_fluid.py`, self-test
`pytests/cosmic_rays_grey/cr_sedov_two_fluid_reference.py` (pure numpy, < 1 s; figure
`pics/07_sedov_taylor/cr_sedov_two_fluid_reference.png`).

- **Solver.** The plan's Appendix A sketch (linear 4x4 system in `U, G, Pi_th, Pi_cr`, LSODA) did
  not finish in minutes when integrated to `lambda = 1e-3`: `G ~ lambda^4.5` makes it stiff and
  ill-conditioned. Rewritten in `U, ln G, ln Pi_i` against `ln lambda` with the entropy
  equations eliminated (explicit, DOP853): all cases in < 0.1 s. `zeta = 0` is solved exactly
  (CR equation dropped).
- **Two conventions.** `"total"` (Pfrommer et al. 2017's `zeta`: `e_cr,2 = zeta e_int,2`) and
  `"gas_rh"` (the code's: CR flux = `zeta` x the gas-only RH dissipated flux
  `delta rho_0 V_s^3/2`, which the two-fluid shock exceeds slightly). For `zeta = 0.1`,
  `"gas_rh"` = effective `"total"` 0.098: `E_cr/E` 0.1398 instead of 0.1429, injection rate
  0.0416 instead of 0.0425 E/t, finder dissipation 0.4159 E/t. The plan's 7.2 table uses
  `"total"`; the code comparisons (B7.3 on) use `"gas_rh"`.
- **Self-test results** (all pass):
  - classic limit: `xi_0` 1.151666 (5/3; ref. 1.15167) and 1.032777 (7/5; ref. 1.0328),
    `E_kin/E` 0.2828 / 0.2187, gas dissipation 0.4583 E/t;
  - energy integral (energy flux through every co-moving sphere, Sedov's first integral)
    <= 4.1e-5 of the post-shock flux, the residual sits in the core (`lambda < 0.01`, `G < 1e-9`);
    mass closure <= 3.4e-7; `profiles()` integrates back to `E` to 2.3e-5;
  - boundary `X_cr,0 = zeta/(2(1-zeta))` and `x = 2k-1` exact; `zeta = 1e-6` vs 0: `xi_0` 2.4e-7;
  - plan Sec. 7.2 table reproduced to < 1e-3 for all five `zeta` (regression values in the test);
  - Pais et al. (2018) eq. 16 single-fluid `gamma_eff`: `xi_0` 1.1-2.0% below the two-fluid one
    (gate 3%).
  - `X_cr = X_cr,0 (G/x)^(-1/3)` holds to 1e-14, but this is structural in the log form, not an
    independent check.
- **Not done** from the plan's B7.1 cross-checks: Pfrommer et al. (2017) Fig. 3 numbers and their
  `gamma = 7/5` approximation (`alpha = 0.851`) were not compared (no digitized data).
- **Physics note** (figure): for `zeta > 0` the core is CR-dominated (`P_th -> 0`,
  `P_cr` ~ const), since the early-shocked, rarefied gas keeps `P_th ~ rho^5/3` and
  `P_cr ~ rho^4/3`. Plan Y4 fraction confirmed: 39.4% of `E_cr` (`zeta = 0.1`) sits in gas
  shocked while `R_s < R/4`.
- The placeholder `reference_solutions/sedov_solver.py` has no users and is left alone.
