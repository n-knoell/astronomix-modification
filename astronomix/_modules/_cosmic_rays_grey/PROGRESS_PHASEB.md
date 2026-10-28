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
| B7.2 Sedov from the self-similar state, `v_red = 0` | 2 | done (tracked); new finding DB9 | `cr_sedov_self_similar.py`, `sedov_two_fluid.py` (`cell_averages`) | E_cr/E end 0.951 / 0.982 / 0.996 of the reference (N 48 / 96 / 160; point explosion 0.52 / 0.67 / 0.74) = injection 0.77 / 0.87 / 0.92 x deposit surplus (injected part +18 / +14 / +11%, under-compressed shell); passive adiabatic losses right to 4 / 2 / 1% | 2026-10-21 |
| B7.3 instantaneous injection rate vs analytic | 1 | done; findings for DB4 + new area under-count (Y10) | `cr_sedov_injection_rate.py` | **Since DB4 + DB7 (2026-10-20):** late-time injection 0.766 / 0.861 / 0.918 of the reference at N 48 / 96 / 160, Mach ratio 0.876 / 0.935 / 0.943, injection area 1.054 / 1.072 / 1.028, `E_cr/E` end 0.515 / 0.672 / 0.740 of the reference (point explosion). Before (2026-10-14): injection 0.43 / 0.53 / 0.60 = area 0.85 / 0.87 / 0.84 (`<|n|_inf>` 0.831) x `(M_fin/M_true)^3` (pressure Mach ratio 0.76 / 0.84 / 0.87). Cap: 0 | 2026-10-20 |
| B7.4 planar shock tube with injection vs exact (Pfrommer et al. 2017 App. B/C) | 1 | done; new finding Y11 (DB8, reopened 2026-10-27) | `cr_shock_tube_injection.py`, `pfrommer_riemann_solver.py` (`injection=`) | **Since DB4 (2026-10-20), velocity-jump Mach:** `zeta` 0.1 vs the velocity fixed point (N >= 400): `v_s` -2e-4, `P_th` -0.15%, `P_cr` +2.6% (deposit inside the smeared shock), finder M -0.4 to -0.7%; **vs the true Mach number `P_cr` +6.6%, rate +6.3%**. `zeta` 0.5 (tracked only): `P_cr` -14% vs the velocity fixed point, +8% vs true. DB9 flag: `P_cr` -0.8% (fixed point), +3.0% (true). Before (gas-pressure Mach, 2026-10-14): `P_cr` +2.7% vs its fixed point, -2.7% / **-23%** vs true (`zeta` 0.1 / 0.5). Energy scheme `P_cr` +6 to +16% | 2026-10-21 |
| B7.5 shock slowdown | 2 | done (B7.2 runs; point explosion at ζ 0.5 not run) | `cr_sedov_self_similar.py` | ξ_dsa / ξ_ctrl over the reference 0.9998 / 1.0015 / 0.9984 (ζ 0.1), 0.9959 / 1.0001 / 1.0017 (ζ 0.211); α 0.407 / 0.404 / 0.403 (estimator offset) | 2026-10-21 |
| B7.6 isotropy of the deposit | 3 | todo | | | |
| B7.7 robustness: `dt`, sampling knobs, limiter counters | 3 | todo | | | |
| B7.8 (optional) SN source term, composite blast | 3 | todo | | | |
| **Item 8: DSA efficiency vs Mach number** | | | | | |
| B8.1 realized `eta_eff(M)` at planar shocks | 1 | done | `cr_dsa_efficiency_planar.py` | **Since DB4 (2026-10-20):** vs the velocity fixed point KR13 within 3.4% for M >= 3, -5.2% at M 2.25; constant -1.5 ... +2.6%. **Vs the true Mach number: KR13 +11% at strong shocks (+7.3% at M 5), constant +0.3 ... +6.6%** (DB8). Before (gas-pressure Mach): KR13 -9% at strong shocks, constant -1 ... -3% vs true. N 400 = N 1600 to 1e-3 | 2026-10-20 |
| B8.2 finder Mach and injection at oblique shocks | 2 | done (diagnostic, no gates) | `cr_oblique_planar_shocks.py`, `cr_sedov_shock_area.py` | N 128, 5 orientations, M 3/10: with DB4 (a) + DB7 weights the injected flux per area is 0.989-1.002 (ζ 0) / 1.010-1.039 (ζ 0.1) of exact, angle dependence < 2% (was 43%) | 2026-10-20 |
| B8.3 KR13 fit vs published data | 2 | todo | | | |
| B8.4 weak blasts vs 1D spherical reference | 3 | todo | | | |
| B8.5 shocks into CR-loaded gas | 2 | done (planar, diagnostic; DB8 runs) | DB8 scratch scripts (`cr_shock_tube_injection.py` `run_tube` / `reference`) | composite-RH tubes `P_cr,1 / P_th,1` = 0.05 / 0.5 / 1, M 2.5-10, KR13 + `zeta` 0.1, N 1600, vs the true gas Mach number: gas-only velocity Mach `P_cr` **-17 to -57%** at M 2.5-3 (X 0.5-1; the stiffer CR-loaded gas compresses less), CR-aware Mach (DB8, now default) -0.4 to +4.1% (X 0.5), +0.1 to +10.3% (X 1). Shock speed <= 5e-4 | 2026-10-27 |
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
| DB4 | Finder Mach bias: correct it (F4 a/b) or accept and gate trends? | **decided 2026-10-19** (user): (a) Mach from the velocity jump (`mach_estimator="velocity"`, `dsa_mach_from_velocity=True`) | Sedov Mach ratio 0.876 / 0.935 / 0.943 (N 48 / 96 / 160; pressure jump 0.76 / 0.84 / 0.87); log 2026-10-20 |
| DB5 | Sub-critical threshold of the constant model (1.3 / 2.25 / 3)? | open | |
| DB6 | Item-7 gated IC: self-similar state instead of the point explosion? | open | |
| DB7 (new) | Area under-count of oblique shock surfaces (Y10): correct the per-cell area by `1/|n|_inf`? | **decided 2026-10-19** (user): yes, with the normal of the Gaussian-smoothed temperature (`dsa_area_weighting=True`, `dsa_area_normal_smoothing=2`) | Sedov injection area 1.05 / 1.07 / 1.03; B8.2 angle dependence < 2%; log 2026-10-20 |
| DB8 (new) | CR back-reaction on the finder Mach number (Y11): the finder reads M from the **gas**-pressure jump, which the injected CRs lower. (a) Mach from the total-pressure jump (composite RH), (b) from the density/velocity jump, or (c) accept? | decided 2026-10-14: (c) accept for now (user). Reopened 2026-10-27 (velocity Mach since DB4 reads high: +6.6% at `zeta` 0.1, +11% KR13, and low with CR-loaded upstream). **Decided 2026-10-27** (user): CR-aware inversion of the velocity jump, a switch that is on by default (`dsa_mach_cr_aware=None` = on with adaptive sampling + velocity Mach); gas Mach convention (KR13) kept; DB9 keeps the gas-only `rho_RH`; log 2026-10-27 | planar vs true M: `zeta` 0.1 +2.6% (was +6.6%), 0.5 -1.7% (+8.0%), KR13 strong +2.1 / +2.4% (+11%), CR-loaded upstream (B8.5) -0.4 to +10% (-17 to -57%); Sedov injection 3.5% lower (N 48 / 96) |
| DB9 (new) | Deposit surplus (B7.2): CRs deposited into the smeared, under-compressed shock shell later lose less to expansion than CRs injected at the RH post-shock density. (a) accept and document (it offsets part of the injection deficit), (b) density-consistent deposit: scale each cell's share by `min(1, (rho/rho_RH)^(gamma_cr - 1))`, the withheld energy stays in the gas, or (c) other? | **decided 2026-10-21** (user): (b), opt-in (`dsa_density_consistent_deposit=True`, default off); implemented + regression run 2026-10-21 | with the flag: Sedov E_cr / prediction 0.985 / 0.990 / 0.990, E_cr/E end 0.84 / 0.90 / 0.93 (injection deficit no longer offset), planar P_cr -0.8%. Before: injected-part surplus +18 / +14 / +11% (Sedov N 48 / 96 / 160), +2.6% planar (B7.4). Scratch prototype of (b): planar -0.8%, Sedov -13 / -7 / -5% (over-corrects where the shell is unresolved); log 2026-10-21 |

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
- [x] Self-similar initial-condition builder on the Cartesian grid (B7.2, B7.5, B7.6): `SedovTwoFluidSolution.cell_averages` (2026-10-21), `n_sub^3` sub-samples per cell averaging the conserved densities, optional core density floor
- [ ] Digitized KR13 / Dubois App. B data (B8.3), 1D spherical reference (B8.4)
- [ ] Shared shock-position fit: first version in `cr_sedov_injection_rate.py`
      (`_shock_radius`: front midpoint + pressure peak of shell averages; `_derived`: fit
      `R = R_off + xi t^{2/5}` to both, mean `xi`). Front and peak `xi` agree to 1.9% / 1.2% /
      0.4% at N 48 / 96 / 160

## Log (newest first)

### 2026-10-27: DB8 reopened + decided: CR-aware velocity-jump Mach number, default on (implemented; regression running; uncommitted)

**Why reopened.** Since DB4 the finder inverts the normal-velocity jump with the gas-only Rankine-Hugoniot relation, `du / c_1 = 2 (M - 1/M) / (gamma + 1)`. The CRs injected at the shock compress the gas further, so `du` and with it the Mach number read high; CR-loaded upstream gas is stiffer and compresses less, so the Mach number reads low. Exact steady planar shocks (scratch `steady.py`, reference solver's `_hugoniot_rho_star` / `_post_shock_partition` with a root-find for the mass flux), injected / intended flux at the velocity fixed point, `X = P_cr,1 / P_th,1`:
- X = 0: `zeta` 0.1 +2.4 to +3.9%, KR13 +2.6 to +8.6% (strong shocks), `zeta` 0.5 +14 to +23%.
- X > 0, weak shocks: KR13 M 3 0.73 (X 0.5), 0.52 (X 1); `zeta` 0.1 M 2.25 0.54 (X 1). The old gas-pressure estimator is worse still.

**Method.** In units `rho_1 = c_th,1 = 1` with the unknown `s = rho_1 / rho_2` and `M = w / (1 - s)` (`w` the measured jump), mass and momentum leave one energy equation: gas enthalpy, upstream CRs compressed adiabatically (`P_cr,2 = P_cr,1 s^-gamma_cr`) and the injected specific energy `zeta(M) delta(M) M^2 / 2` (the code's `gas_rh` convention). It has exactly one root on `[max(0.1, 1 - w), 1)` (checked on a 4000-point scan for X 0-2, `zeta` 0 / 0.1 / 0.5 / KR13, M 1.5-100); bisection recovers the true M to 3e-15. The Mach number stays the gas one (`v_1 / c_th,1`, KR13's definition); only the inversion changes.

**Implementation.**
- `CosmicRayGreyConfig.dsa_mach_cr_aware: Optional[bool] = None`: on wherever it applies (`dsa_adaptive_shock_sampling` and `dsa_mach_from_velocity`), off elsewhere (`simulation_config.resolve_auto_defaults`, as `cr_entropy`); `True` raises where it does not apply.
- `cr_grey_injection.cr_aware_velocity_mach`: 60 bisection steps in a `lax.while_loop` (a fixed-length `fori_loop` is unrolled by XLA and kept ~250 full-grid temporaries: 7.5 GiB at 160^3). The final step is a stop-gradient Newton step that changes only the derivative (implicit-function theorem), so AD through `dsa_efficiency_mach_scale` / `dsa_efficiency` stays exact; the value is the bisection root. Inputs away from shocks are replaced by harmless stand-ins so masked gradients stay finite.
- `inject_crs_at_shocks`: samples `e_cr` at the walk's pre-shock point (`sample_along_direction` with the new `ShockFinderResult.pre_shock_steps`), `X = (gamma_cr - 1) e_cr / P_pre`, recomputes M, `valid = M > dsa_mach_min` and the flux `delta(M) rho_pre (M c_pre)^3 / 2` (`thermal_energy_flux_from_pre_state`); the efficiency is evaluated at that M. Cells the finder already rejected (gas-only M <= `dsa_mach_min`) stay rejected.
- DB9 (`dsa_density_consistent_deposit`) keeps `rho_RH` from the finder's gas-only Mach number. With the CR-aware compression it over-corrected (B7.4 `zeta` 0.1 -1.7%, 0.5 -10.8%); with the gas-only one -0.8% and -6.0% (N 1600 and 400 alike).
- Cost: one 160^3 injection call 1.17 GiB temporaries vs 0.98 GiB; full 160^3 B7.2 step identical (11.16 GB, the peak is elsewhere).

**Checks** (repo code, `/export/scratch/nknoell/phaseB/db8_impl/`): all 32 planar cases identical (to the printed 5 digits, difference 0) to the scratch prototype, both on and forced off; forced off = the previous default exactly. AD: `d dE_cr / d dsa_efficiency_mach_scale` and `/ d dsa_efficiency` on a smeared M 3 shock (X 0 and 1) agree with central differences to 1e-13 ... 5e-10.

**Planar results** (N 1600, `P_cr` from injection vs the reference at the TRUE Mach number; gas-only velocity Mach -> CR-aware):

| case | gas-only | CR-aware | CR-aware + DB9 |
|---|---|---|---|
| B7.4 thermal tube `zeta` 0.1 | +6.6% | **+2.6%** | -0.8% |
| B7.4 thermal tube `zeta` 0.5 | +8.0% | **-1.7%** (N 400: -2.0%) | -6.0% |
| B8.1 RH tube `zeta` 0.1, M 2.25-30 | +2.5 ... +6.7% | **+0.2 ... +2.6%** | |
| B8.1 KR13 M 10 / 30 | +11.0 / +11.6% | **+2.1 / +2.4%** | -1.7% (M 10) |
| B8.1 KR13 M 5 | +7.4% | -1.3% | |
| B8.1 KR13 M 2.25 / 3 | -5% / -1.2% | -5% / -1.6% (resolution, not the estimator) | -4.4% (M 3) |
| B8.5 X 0.5 KR13 M 2.5 / 3 | -36.9 / -27.0% | +4.1 / -0.4% | |
| B8.5 X 0.5 `zeta` 0.1 M 2.5 | -17.3% | +0.4% | |
| B8.5 X 1 KR13 M 2.5 / 3 | -57.2 / -47.2% | +10.3 / +0.5% | |
| B8.5 X 1 `zeta` 0.1 M 2.5 / 3 | -35.8 / -21.6% | +0.1 / +0.9% | |
| B8.5 X 0.05, all | -7.1 ... +10.9% | -2.4 ... +2.6% | |
| Pfrommer CR tube `zeta` 0.1 / 0.5 | +5.6 / +3.6% | +2.6 / -5.0% | |

Shock speed within 5e-4 everywhere (Pfrommer CR tube `zeta` 0.5: 1.7e-3). The remaining +2.6% at `zeta` 0.1 is the deposit inside the smeared shock (B7.4).

**Sedov (B7.2, `zeta` 0.1, gas-only -> CR-aware -> CR-aware + DB9 with the CR-aware `rho_RH`):** E_cr/E end / ref 0.951 -> 0.936 -> 0.828 (N 48), 0.982 -> 0.966 -> 0.882 (N 96); realized late injection / ref 0.755 -> 0.727 -> 0.562, 0.858 -> 0.828 -> 0.697; E_cr / kernel 1.11 -> 1.09, 1.08 -> 1.06. `zeta` 0.211: 0.965 -> 0.934, 0.992 -> 0.960. The old high reading hid ~3.5% of the under-resolution deficit (DB4/DB7, B7.3). 160^3 not run: the B7.2 setup (10 returned 3D snapshots) needs 11.2 GB in either version and no longer fits; the 2026-10-21 runs used a GPU (`GPU-97e5a2a9`) no longer in the machine.

**Regression** (DSA tests only, update mode against a scratch copy of the registry, `/export/scratch/nknoell/phaseB/db8_impl/reg/`): running.

**Not covered:** the CWB (`pytests/shock_finder3D`, tracked +41% split) and M7 runs use DSA too; their tracked numbers move with this default and were not rerun.


### 2026-10-21: DB9 = (b) density-consistent deposit, opt-in (implemented; regression run; uncommitted)

**Implementation.**
- `CosmicRayGreyConfig.dsa_density_consistent_deposit` (default `False`; needs `dsa_adaptive_shock_sampling`, else `ValueError`). Each post-shock cell's share of the spread injection is multiplied by `min(1, (rho_cell / rho_RH)^(gamma_cr - 1))`, `rho_RH = rho_pre (gamma+1) M^2 / ((gamma-1) M^2 + 2)` with the finder's Mach number and the surface cell's sampled pre-shock density. The withheld energy stays in the gas (it is simply not diverted).
- `ShockFinderResult.pre_shock_density` (new field, adaptive sampling only): the `rho_pre` of the walk's pre-shock sample point, which the finder already used for the flux.
- `cr_grey_injection._spread_over_post_shock_cells`: optional `density`, `post_shock_density`, `gamma_cr` keywords (no factor without them; existing positional callers unchanged).
- Tests (tracked only, no new hard gates): B7.2 run `z0.1_dcd` (+ metric `inj_realized_over_nominal`), B7.4 scheme `"default, density-consistent deposit"` (N 400 / 1600, both zeta); 42 + 32 new metrics in `baselines.json`. B7.2 figure: dash-dot curves in (a), two "(d.c.)" bars in (f).

**Check against the scratch prototype** (repo code, no patching): B7.4 to <= 3e-4 (the prototype assumed `rho_pre` = 0.125), Sedov N = 48 to 4 digits; energy conservation 1e-14.

**Results** (regression run, `/export/scratch/nknoell/phaseB/db9/`):

| N | E_cr/E end / ref (default) | E_cr / kernel prediction (default) | deposited / nominal | X_cr shell / ref (default) | P_cr peak / ref (default) | L1 P_cr (default) | slowdown / ref |
|---|---|---|---|---|---|---|---|
| 48 | **0.843** (0.951) | **0.985** (1.113) | 0.778 | 0.818 (1.063) | 0.600 (0.779) | 0.169 (0.118) | 1.003 |
| 96 | **0.900** (0.982) | **0.990** (1.081) | 0.848 | 0.889 (1.064) | 0.712 (0.859) | 0.113 (0.080) | 1.004 |
| 160 | **0.931** (0.996) | **0.990** (1.060) | 0.885 | 0.923 (1.094) | 0.805 (0.916) | 0.083 (0.060) | 1.000 |

- B7.4 thermal tube vs the velocity fixed point (N 400 / 1600): `zeta` 0.1 `P_cr` **-0.86 / -0.83%** (default +2.6%), rate -0.75 / -0.79%, `P_th` +0.06%, `v_s` < 2e-4; vs the true Mach number `P_cr` +3.0% (default +6.6%). `zeta` 0.5 `P_cr` -17.3 / -16.9% (default -14%), vs true M +3.7 / +4.3% (default +8%).

**Reading.**
1. **The surplus is gone.** With the flag the run holds what its own injection predicts with the reference's adiabatic losses to 1-1.5% at every N (default +6 to +11%); the planar plateau is within 1% of its fixed point.
2. **What remains is the injection deficit, now visible.** E_cr/E, the shell X_cr / P_cr and the L1 of P_cr are further from the reference than with the default deposit, and converge with the injection rate (0.76 / 0.86 / 0.91): the default deposit's agreement was the two errors cancelling (log entry below). The remaining deficit is the finder Mach number and area (DB4/DB7, B7.3).
3. **No side effects on the blast.** Shock radius, slowdown (B7.5) and exponent unchanged to < 0.5%; total energy conserved as before.

**Regression (flag off, everything else as committed + the B7.2 entry below).** Full grey-CR suite, update mode against a scratch copy of the registry, then diffed: all 853 pre-existing metrics identical to the registry's 4 digits except roundoff-level ones (conservation errors ~1e-5, item-7 partition identity 3e-7 -> 2e-6, B7.3 cap loss ~1e-8, item-8 cross check ~2e-7; all far below their gate floors 1e-4 / 1e-3 / 1e-6). Every test passes except `test_cr_ring_flux_cap_alone`, the strict xfail (identical numbers to the 2026-10-20 run). Only the new keys were merged into `baselines.json`; existing entries untouched.

### 2026-10-21: B7.2 Sedov from the self-similar state + B7.5 shock slowdown (done; tracked, uncommitted)

`pytests/cosmic_rays_grey/cr_sedov_self_similar.py` (new); initial-state builder `SedovTwoFluidSolution.cell_averages` in `sedov_two_fluid.py` (new method).

**Setup.**
- The item-7 box, ambient gas and `T_END` = 0.07. The blast starts from the self-similar solution at `R_0` = 0.15 (`t_start = t_0` ≈ 0.0061-0.0067, i.e. the reference's own clock), so the final state is directly comparable with item 7 / B7.3.
- Cell averages of the conserved densities from 4³ sub-samples per cell. The initial energy is the reference's to 1e-3, and E_cr/E to 7e-4.
- Core density floored at 0.02 ρ0. The self-similar G is 1e-9 at λ = 0.05, and the sound speed there would collapse dt; the point-explosion runs have ρ_min 0.015-0.14. Pressure and velocity are kept.
- `v_red = 0`, constant efficiency, code convention `"gas_rh"`.
- Runs per N: control (ζ = 0), and ζ = 0.1 and 0.211 (0.211 = item 8's KR13 strong-blast E_cr/E 0.2625), each with the default scheme (HLLC + hybrid) and the energy scheme (`cr_entropy=False`).
- N = 48 / 96 / 160 took 2 / 12 / 82 min (5 runs each).
- Raw data `/export/scratch/nknoell/phaseB/b7_2/`; runs and diagnostics `/export/scratch/nknoell/phaseB/b7_2_run/` (logs in `logs/`).

**Results, ζ = 0.1, default scheme** (energy scheme in brackets; late = t ≥ 0.03, end = t 0.07):

| N | E_cr/E end / ref | point explosion (B7.3) | injection / ref (end) | finder Mach / true (end) | E_cr / kernel prediction (end) | xi / ref | slowdown / ref (B7.5) |
|---|---|---|---|---|---|---|---|
| 48 | **0.951** [0.931] | 0.515 | 0.768 | 0.896 | 1.113 | 1.0104 | 0.9998 |
| 96 | **0.982** [0.979] | 0.672 | 0.872 | 0.941 | 1.081 | 1.0050 | 1.0015 |
| 160 | **0.996** [1.003] | 0.740 | 0.920 | 0.962 | 1.060 | 1.0009 | 0.9984 |

- ζ = 0.211: E_cr/E end 0.965 / 0.992 / 1.008 [0.949 / 0.996 / 1.022], slowdown / ref 0.9959 / 1.0001 / 1.0017.
- Control ξ / Sedov: 1.0106 / 1.0035 / 1.0025. Exponent α 0.4069 / 0.4037 / 0.4028 (the front/peak estimator offset; 0.400 expected).
- Shell window λ ∈ [0.7, 0.9] at t_end, ζ = 0.1 default: X_cr / ref 1.063 / 1.064 / 1.094, P_cr / (ρ0 V² Π_cr,ref) 1.014 / 1.017 / 1.038.
- Post-shock P_cr peak / reference 0.78 / 0.86 / 0.92: the smeared shell, no deposit spike (plan 7.3 step 5 not seen).
- L1 (ζ = 0.1 default): ρ 0.316 / 0.237 / 0.181, P_tot 0.137 / 0.099 / 0.076, P_cr 0.118 / 0.080 / 0.060.
- Energy conservation ≤ 1e-4 in every run.

**Reading.**
1. **Item 7's E_cr deficit is the point-explosion start.** From the self-similar state, the same scheme keeps E_cr/E within 5 / 2 / 0.5% of the reference. Plan Y4 is confirmed: the time-integrated E_cr of item 7 measures the unresolved early blast.
2. **But it is right for two compensating reasons.** The injection rate is only 0.77 / 0.87 / 0.92 of the reference, from the finder Mach number cubed and the area (as B7.3). The run nevertheless holds 6-11% more E_cr than its own injection rate predicts with the reference's adiabatic losses (the B7.3 "surplus", now also seen without the early phase).
3. **The two schemes agree.** Default and energy scheme differ by ≤ 2% (more at N = 48 for the default, more at N = 160 for the energy scheme). Weber et al.'s (2023) shock-speed problem of entropy schemes with injection does not appear: ξ agrees to 2e-4.
4. **B7.5 is passed.** The shock slowdown matches the reference to < 0.4% at every N and both ζ.

**Surplus split** (`b72_split.py`, scratch; same setup, kernel prediction from each run's own injection; run / prediction at t_end):

| N | (A) passive: ζ 0.1 state, no injection | (B) injection only: ζ 0 Sedov state, ζ 0.1 injection |
|---|---|---|
| 48 | 1.039 [1.036] | 1.180 [1.164] |
| 96 | 1.016 [1.020] | 1.138 [1.153] |
| 160 | 1.009 [1.014] | 1.109 [1.133] |

The adiabatic losses of CRs already present are right (to 1-4%, converging). The surplus comes with fresh deposits.

**Mechanism: deposit into the under-compressed numerical shell** (`b72_deposit.py`, scratch). Late snapshots of the ζ = 0.1 runs, the injection probed with a tiny dt:

| N | deposit-weighted λ | shell density peak / ρ_2 (RH, 4.15 ρ0) | compression to the simulated peak `<(ρ_pk/ρ)^(1/3)>` | to the RH density `<(ρ_2/ρ)^(1/3)>` | observed (B) |
|---|---|---|---|---|---|
| 48 | 0.91-0.93 | 0.49-0.53 | 1.04-1.05 | 1.28-1.34 | 1.18 |
| 96 | 0.95-0.96 | 0.61-0.65 | 1.03 | 1.18-1.22 | 1.14 |
| 160 | 0.97-0.98 | 0.69-0.73 | 1.03 | 1.14-1.16 | 1.11 |

- **The mechanism.** The numerical shell never reaches the RH density. A CR packet deposited at ρ < ρ_2 keeps `(ρ_f/ρ)^(1/3)` of its energy per unit mass later, against `(ρ_f/ρ_2)^(1/3)` in the reference, so it loses less to expansion.
- **Bounds.** The observed gain lies between the two bounds, closer to the RH one, and converges with them.
- **Planar shocks.** The same effect is B7.4's +2.6% P_cr excess (deposit inside the smeared shock).

**Prototype of a density-consistent deposit** (scratch only, monkeypatched; repo unchanged; `b72_proto.py`, `b74_proto.py`):
- **What it changes.** Each post-shock cell's share is multiplied by `min(1, (ρ_cell/ρ_RH)^(1/3))`, with `ρ_RH = ρ_pre x(M_finder)` (gas-only RH compression; prototype shortcut: ρ_pre = ambient). The withheld energy stays in the gas.
- **Planar B7.4 thermal tube, ζ = 0.1, vs the velocity fixed point.** P_cr +2.75 / +2.57% → **−0.84 / −0.80%** (N 400 / 1600). The rate goes +2.5% → −0.75%, and P_th, ρ and v_s get closer too. Vs the true Mach number, P_cr goes +6.6% → +3.0%.
- **Planar, ζ = 0.5 (tracked only).** −14% → −17% vs the velocity fixed point, +8% → +4% vs the true Mach number.
- **Sedov.**
  - Full run / prediction: 0.985 / 0.990 / 0.990 (was 1.11 / 1.08 / 1.06).
  - Injection-only part: 0.87 / 0.93 / 0.95. It over-corrects where the shell is unresolved, since the deposited CRs still get part of the compression later, but converges faster than the current deposit (+18 / +14 / +11%).
  - E_cr/E end 0.84 / 0.90 / 0.93: the injection deficit is then no longer offset.
- **Not done in the prototype.** A real version needs the pre-shock density from the finder walk (`sample_along_direction` with the pre-shock steps, not in `ShockFinderResult` yet).
- **Open for the user:** DB9 (decided 2026-10-21: (b), opt-in; see the entry above).

**Notes.**
- **jit cache.** Monkeypatching `_iteration_level_updates.inject_crs_at_shocks` after a first run in the same process has no effect: the time loop is already compiled for the same static config. The first planar prototype comparison was void for that reason and was rerun in a fresh process.
- **Stale B7.3 states.** `/export/scratch/nknoell/phaseB/b7_3/` held pre-DB4/DB7 summaries. The current `b7_3_N*.npz` (from `cap_floor/raw`, 2026-10-20) were copied in, and the old ones kept as `pre_db47_*.bak`. The `states_eff*.npz` there are still pre-DB4/DB7 (do not use `CR_B73_REUSE=1` on them).
- **Gates.** Hard gates only for the initial state, conservation and the radius analysis; everything else is registered in `baselines.json` (165 metrics, N 48 / 96 / 160). The plan's proposed hard gates (|E_cr/E_ref − 1| < 0.1 at N ≥ 96) pass now, but wait for DB9: with (b), N = 96 would sit at 0.90.

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
- **Negative e_cr, diagnosis (2026-10-20, user asked).** The floor acts once per step before the hydro update, so the final state shows one step's undershoot. One-step decomposition (N = 64): F_cr drives it (F_cr zeroed: 0 negative cells; with it 42150); the split CR source plays no part. The negative cells start the step at the floor with |F_cr| ~ 7e5 v_red e_cr: CWB runs without `diffusive_relaxation`, nothing relaxes F_cr, and the realizability cap |F_cr| <= v_red e_cr (`cr_flux_realizability_cap`) is only applied in the unsplit path with relaxation/anisotropic transport on; 29% of cells are non-realizable. Per-step CR energy budget (N = 64, closes to 1e-10; scratchpad `cwb_budget.py`, logs `logs/r4/`): injected 1.08e-4, boundary outflow −1.04e-4, **floor creates +1.93e-5 (74% of the final E_cr 2.60e-5)**. Prototype fix = apply the cap together with the per-step floor: 0 negative cells (N = 64 and 128), floor energy 2e-10, outflow −6.3e-5, injection unchanged, E_cr 2.60e-5 → 4.45e-5 (N = 64) and 8.01e-5 → 1.33e-4 (N = 128); capping also after every split sweep changes E_cr by < 0.5%. Even with the cap 59% of the injected CR energy leaves the box: without relaxation the CRs free-stream at v_red = 1000 (> the wind speed), so the box E_cr is set by the numerical v_red. Proposal made to the user.
- **Negative e_cr, fix implemented (2026-10-20, user approved).** `_iteration_level_continuous_updates` applies `cr_flux_realizability_cap` right after the e_cr floor whenever grey CRs with allocated F_cr rows are active and `flux_realizability_cap` is on (default), in every scheme. The interface/stage caps of the unsplit relaxation path are unchanged. Full grey-CR regression (update mode, scratch `/export/scratch/nknoell/phaseB/cap_floor/`, 11 jobs, Sedov / Mach dependence / Sedov injection rate at N = 48/96/160, clumpy and pion bump at 128): every test passes except the strict-xfail `test_cr_ring_flux_cap_alone` (the harness ignores the marker; values unchanged: 3a e_min 9.955595 / 9.937447, 3b −0.0044 / −0.0085). 16 of 704 baseline values moved, all within their gates: DSA conservation errors (≤ 0.8%, under the 1e-4 floor), Sedov partition identity, Sedov cap loss (~1e-8, under the 1e-3 floor), cr_advection centroid −0.0777 → −0.0706 cells (tol 0.02) and flux-ratio error +1.5%; no CR energy fraction or Mach baseline changed at 4 digits. Relaxation runs are unaffected by construction (already capped at the end of the step). Registry merged. CWB figures (64, 128) regenerated: 0 negative cells, DSA E_cr 4.44e-5 (N = 64, E_cr/E_tot 0.33%) and 1.33e-4 (N = 128, 0.54%), matching the prototype. The missing CR diffusion / split-path relaxation in CWB (free streaming at v_red) remains open.
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
- **CWB with CR diffusion, probe + CR energy budget (2026-10-20, user asked).** Why no diffusion: the CWB CR setup only switched on DSA and kept the default pure two-moment wave mode; `diffusive_relaxation` needs the unsplit RK2 scheme and CWB runs split (unsplit crashed 2026-09-19, FIXES_TODO.md item 4). The unsplit scheme now runs CWB cleanly (N = 64 and 128, no NaN, no negative e_cr). Scratch copy of the setup with `unsplit`/`kappa` arguments: `/export/scratch/nknoell/phaseB/cwb_diffusion/` (`_cwb_setup_diff.py`, `probe.py`, `figs.py`, `budget.py`, logs). Units: kappa code unit = 4.98e20 cm^2/s; kappa = 1e-3 (5e17 cm^2/s, Bohm-like for GeV protons in B ~ 0.1 G) gives a diffusion length << dx over T_END, so CRs ride with the gas; galactic 3e28 cm^2/s would be free streaming again. Figures (unsplit, kappa = 1e-3, cr_entropy default on): `pytests/shock_finder3D/figures/cwb_cr_unsplit_kappa0.001/` (64, 128): CRs confined to the collision shell and its outflow arms (P_cr/P_gas ~ 0.1 there), instead of filling the box. Per-step E_cr budgets (all close to <= 3e-10; inject / adiabatic work / CR-entropy shock transfer / boundary outflow / E_cr end):
  - N = 64 split, no diffusion (current): 1.07e-4 / ~0 / - / -6.27e-5 (58% of injected) / 4.44e-5
  - N = 64 unsplit, no diffusion: 1.16e-4 / +8.1e-6 / +3.68e-5 / -9.03e-5 (78%) / 7.11e-5; with cr_entropy off: 1.15e-4 / +9.9e-6 / - / -7.03e-5 (61%) / 5.51e-5
  - N = 64 unsplit kappa = 1: 1.18e-4 / +1.0e-5 / -3.37e-5 / -2.87e-5 (24%) / 6.58e-5
  - N = 64 unsplit kappa = 1e-3: 1.20e-4 / -7.6e-6 / -3.12e-5 / -2.37e-5 (20%) / 5.77e-5; cr_entropy off: 1.14e-4 / +4.1e-6 / - / -3.73e-5 (33%) / 8.04e-5
  - N = 128 split, no diffusion: 2.33e-4 / +8.99e-5 (split centred source, a gain; ~0 at N = 64, unexplained) / - / -1.89e-4 (81%) / 1.33e-4
  - N = 128 unsplit kappa = 1e-3: 2.40e-4 / -4.93e-5 / -2.62e-5 / -5.60e-5 (23%) / 1.08e-4; cr_entropy off: 2.36e-4 / -4.40e-5 / - / -6.57e-5 (28%) / 1.26e-4
  Reading: diffusion cuts the boundary outflow from 60-80% to 20-33% of the injected energy, as intended. Smaller kappa gave less box E_cr only because of the hybrid CR-entropy shock transfer: with confined CRs it moves 11-26% of the injected energy to the gas, without diffusion it added 32%; with it off, kappa = 1e-3 keeps the most (8.0e-5 vs 5.5e-5 at N = 64). Adiabatic losses in the expanding shell grow with resolution (-4.4e-5 to -4.9e-5 at N = 128, ~20% of injected). Open (user decision): adopt unsplit + kappa for CWB (which kappa); whether the shock-zone transfer is right where DSA injects into the same cells; the split-source CR gain at N = 128.
- **CWB moved to unsplit + CR diffusion, kappa = 1e28 cm^2/s (2026-10-20, user decision).** `_cwb_setup.run_cwb(unsplit=True, diffusion_coefficient=CR_DIFFUSION_COEFFICIENT)` (1e28 cm^2/s = 2.007e7 code units). In this box that is the free-streaming limit (nu T_END ~ 3e-5). N = 64 runs (E_cr 7.76e-5, 0 negatives; figures in `figures/cwb_cr/`, `figures/cwb_default/` regenerated). **N = 128 goes NaN** (step 745, in the hydro step), and so does unsplit without diffusion (step 2046); kappa = 1 and 1e-3 code run clean at 128. First NaN cells (scratch `nanfind.py`): CR-dominated low-density gas, e.g. rho 4e-10, P_gas ~1e-9 vs e_cr 1.6e-5, |v| ~ 1370 > v_red, near the box edge; the no-diffusion case P_cr ~ 20 P_gas. Free-streaming CRs fill the low-density wind and push it with the full -grad P_cr (the grey model is the diffusion limit, so the coupling assumes kappa << v_red L; here kappa / (v_red L) ~ 2e4). The 128 figures in `figures/cwb_cr/` and `figures/cwb_default/` are NaN output and must be regenerated once a working setting is chosen. Entropy shock transfer at kappa = 1e28, N = 64: on: transfer +4.03e-5 (a CR gain of 35% of injected, taken from the gas), E_cr 7.77e-5; off: E_cr 6.02e-5 (work +1.56e-5, outflow -7.04e-5).
- **CWB default kappa = 1 code unit (4.98e20 cm^2/s), 2026-10-20 (user decision; entropy shock transfer on).** Figures `figures/cwb_cr/` and `figures/cwb_default/` (64, 128) regenerated: N = 64 DSA E_cr 6.59e-5 (0.49% of total), N = 128 1.50e-4 (0.61%), 0 negative cells, no NaN; CRs confined to the shocked shell and its outflow arms. Budget N = 128: inject 2.52e-4, adiabatic work +3.9e-5, entropy transfer -6.0e-5, boundary outflow -8.3e-5 (33% of injected), E_cr end 1.49e-4. The `*_256.png` in `figures/cwb_cr/` are the user's 17:51 run with the old split, no-diffusion setup.
