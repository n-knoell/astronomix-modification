# shock_finder3D (ours) vs shock_finder_v2 (2026-11-03)

Both finders run on the same saved states: Sedov N = 64/128/256 (`_sedov_setup`)
and the current default CWB at N = 128 (`_cwb_setup.run_cwb`, unsplit, CR + DSA
0.1). Every zone x surface x Mach-sampler combination is evaluated
(`compare_finders.py`, figures/summary from `plot_comparison.py`; states and
results in `/export/scratch/nknoell/finder_v2_comparison/`). v2 is imported
unchanged via a module alias, since it expects `astronomix._physics_modules._shock_finder`.
Surface cells within 6 cells of the box are excluded (at N = 64 the Sedov shell
is only 2-3 cells from the faces on the axes).

## Findings

Sedov: spread of ln M split into sky-bin medians (systematic, i.e. grid
orientation) and spread within a bin (random). Pressure Mach unless noted.

| N=256 | M / M_Sedov | sys | rand | local | surface cells | DSA area weight |
|---|---|---|---|---|---|---|
| ours (walk+extend) | 0.935 | 0.008 | 0.004 | 0.002 | 145656 | 0.99 |
| ours DSA (velocity) | 0.967 | 0.007 | 0.006 | 0.006 | 145656 | 0.99 |
| v2 default | 0.918 | 0.015 | 0.020 | 0.020 | 195480 | 1.36 |
| v2 profile-aware | 0.940 | 0.007 | 0.003 | 0.001 | 195480 | 1.36 |
| v2 surface + our walk | 0.936 | 0.008 | 0.004 | 0.002 | 195480 | 1.36 |

CWB 128, local (neighbour) spread of ln M: ours 0.023, ours velocity 0.027,
v2 default 0.081, v2 profile-aware 0.010, v2 surface + our walk 0.032.

* **v2 does have the larger scatter, and the cause is its default Mach sampler,
  not its surface.** Swapping one stage at a time (`sedov_stage_attribution.png`),
  the zones and the surface change nothing in the Mach spread. The v2 sampler
  (pressure 0.25 cells past the first zone exit) increases it 2-5x. It samples where the zone
  mask ends, which is still inside the numerically smeared jump on both sides
  (`sedov256_sampling.png`: sigma(ln p_post) 0.040 vs 0.019, the pre sample
  sits up to 0.2 dex above ambient, distances spread over 2-4 cells), so M
  depends on where the boolean zone edge falls relative to the ramp. It is also
  biased low (0.918 vs 0.935) and is the source of *more* orientation
  dependence (sys 0.015 vs 0.008). Our walk continues to the plateau, and v2's
  `profile_aware_mach=True` (max/min pressure 0.25-1.5 cells past the exit)
  fixes it too.
* **The "grid artefacts" in ours are the surface-cell layout, not the Mach
  numbers** (`sedov_surface_coverage.png`). Our surface is exactly one cell per
  dominant-axis column, so cells per unit shock area scale with |n|_inf
  (up to 1.7x more on the axes than on the diagonals). That is what the 3D scatter plots
  show. The DSA area weight 1/|n|_inf (DB7) corrects it exactly: 0.99 of
  4 pi R^2. v2's continuous-normal local-minimum surface is uniform on the sky
  to ~5% but has 1.34 cells per column (34% of columns hold two). With our
  1/|n|_inf weight it would over-inject by 36%, and with unit weight by 11%.
  The Mach orientation dependence that remains (sys ~0.008, lower on the axes)
  is the same for both surfaces, so it comes from the hydro state.
* Zones: v2's continuous-normal criterion 3 changes the zone mask slightly. It
  matters only for the v2 sampler (with our zones that sampler is noisier, rand 0.028),
  not for ours.
* Cost: v2 full finder 0.16 s vs ours 1.33 s at 256^3 (our surface runs an
  N-step scan per cell). v2 profile-aware full finder runs out of memory at
  256^3 on an 11 GB GPU (13.3 GB requested); its stages run separately.

## Verdict

For DSA, ours is better as it stands: lower scatter, less bias, and an exact
area normalisation. v2 default should not be used for injection. v2's
profile-aware sampler matches our walk (slightly lower local noise) but has no
velocity Mach number (DB4/DB8). v2's surface looks nicer but would need a new
area weighting before it can be used for injection.

## Figures (`pics/`)

* `sedov{64,128,256}_sky_and_orientation.png`: Mach sky maps + M vs |n|_inf
* `sedov_stage_attribution.png`: one-stage swaps ours -> v2 and cross combinations
* `sedov{64,128,256}_sampling.png`: where p_pre / p_post are sampled
* `sedov_surface_coverage.png`: surface-cell density on the sky + area weight
* `sedov_resolution.png`: median and 16-84% vs N
* `{sedov*,cwb128}_slice.png`: mid-plane surface cells coloured by Mach
* `cwb128_mach_hist.png`: CWB Mach distribution + neighbour scatter
* `summary.txt`: all variants and metrics

# Part 2: right physics for DSA, and v2 + area weighting (2026-11-03)

Scripts: `dsa_physics.py` (Sedov B7.3 snapshots, N = 48/96/160, control and
zeta = 0.1 runs, t >= 0.03, against the simulated shock's own dissipation
D_meas from B7.3's fit), `dsa_planar.py` (B8.2 planar oblique shocks, N = 64,
exact area), `dsa_cwb.py` (CWB 128, against ours), `dsa_timing.py` (fully
jitted, RTX 2080 Ti), `plot_dsa.py` -> `pics/dsa_physics_and_cost.png`,
`pics/dsa_cwb128_map.png`, `pics/dsa_cwb128.txt`, `pics/dsa_timing.txt`.

Pipelines: ours (production DSA path); "hybrid" = v2 zones + v2 surface +
our walk with the velocity Mach number and the CR-aware flux, i.e. a drop-in
for `find_shocks_pfrommer` (`dsa_physics.find_shocks_hybrid`); v2 default /
profile-aware with their own pressure Mach and flux. Area weights for the v2
surface: unit, ours (1/|n|_inf), and **column** (1/|n|_inf divided by the
number of surface cells in the same zone segment of the dominant-axis column,
`dsa_physics.column_area_weights`).

| | Sedov N=160 D/D_meas | angular spread | planar area (5 normals) | planar D, M=10 | CWB vs ours | inject 160^3 / 256^3 |
|---|---|---|---|---|---|---|
| ours | 0.855 | 3.0% | 0.997-1.022 | 0.92-1.01 | 1 | 252 / 1452 ms |
| hybrid + column w | 0.856 | 3.0% | 0.997-1.028 | 0.92-1.01 | 0.989 | 84 / 348 ms |
| hybrid + unit w | 0.958 | 5.6% | 0.94-1.22 | 0.93-1.15 | 1.036 | |
| hybrid + 1/\|n\|_inf | 1.19 | 13% | 1.0-1.99 | 0.98-1.95 | 1.111 | |
| v2 default + column w | 0.675 | 4.2% | as column | 0.75-0.95 | 0.742 (wind-1 shock 0.57) | |
| v2 profile + column w | 0.704 | 4.4% | as column | 0.96-1.09 | 0.809 (wind-1 shock 0.61) | |

* Right physics: ours and the hybrid with column weights inject the same
  (within 1% on Sedov and planar shocks, 1.5% on the CWB). Neither has a grid
  artefact in the *injected energy*: the axis/diagonal ratio is 0.98 at 160^3
  for both (ours 1.10 at 48^3, hybrid 1.03). The remaining Sedov deficit
  (0.855) is the velocity-Mach deficit cubed (M/M_true = 0.94), the same for both.
* v2's own Mach samplers are the wrong physics for DSA: Mach low, so
  injection 20-45% low, most at the strongest shocks. v2's surface with
  unit or our weight over-injects, orientation-dependent (unit: up to +22% at
  (2,1,0); 1/|n|_inf: up to 2x on the body diagonal).
* Speed: our surface search is O(N^4) (23 / 174 / 1135 ms at 96 / 160 /
  256), v2's is O(N^3) (0.3 / 0.6 / 3.5 ms). Hybrid injection is 2.2x / 3.0x /
  4.2x faster. A 20-step Sedov + CR interval: 1.8 s (no DSA) / 4.5 s (ours) /
  3.0 s (hybrid) at 96^3, and 7.3 / 24.0 / 12.9 s at 160^3. The remaining
  cost is our walk (48 / 200 ms at 160 / 256).
* Side finding: B8.2's last snapshot (t_end) has no clean upstream plateau
  for |k| >= sqrt(5) (next periodic slab 1-2 cells ahead, p up to 6x ambient),
  so all finders misread it, and B8.2's averages include it.

# Part 3: opt-in mode, CR regression, Sedov and CWB tests (2026-11-04)

The hybrid is in the code, off by default:
`find_shocks_pfrommer(..., continuous_normal=True)` (zones and surface along
the continuous normal, `_shock_zones.py` / `_shock_surface.py`; our walk is
unchanged), `column_area_weights` (`_surface_area.py`), and
`CosmicRayGreyConfig(dsa_continuous_normal_surface=True)` switches DSA to
both (only with adaptive sampling, the default; ignored otherwise). The
injection and the two tests that recompute it use `dsa_area_weights`
(`cr_grey_injection.py`). With the flag off the injection equals HEAD to
float32 rounding (<= 2.3e-7 relative in 40 of ~170k cells, CWB 128 + Sedov
128); with it on it equals the Part 2 prototype exactly.

Scripts: `optin_tests.py` (the Sedov / CWB tests, default vs the flag),
`plot_optin.py` -> `pics/optin_cwb128.png`, `pics/optin_sedov256.png`,
`pics/optin_regression.txt`; the tests' own figures are in
`pics/optin_tests/<case>_<mode>__*.png`. Data and logs in
`/export/scratch/nknoell/finder_v2_comparison/{optin,regress}/`.

**CR regression** (all of `pytests/cosmic_rays_grey` except M7, flag forced
on wherever DSA runs, gated against `baselines.json`): 86 PASS, 12 FAIL.

* 1 pre-existing: `test_cr_ring_flux_cap_alone` (no DSA, fails on HEAD too).
* 11 DSA-energy gates, all the same effect: Sedov-type blasts end with more
  CR energy. `cr_sedov_taylor` cr_fraction +5.5 / +8.3 / +6.0% (N = 48 / 96 /
  160), `cr_dsa_mach_dependence` +5 to +8%, `cr_sedov_injection_rate`
  E_cr_end +5 / +8 / +6%, `cr_sedov_self_similar` N48 shell CR +3%,
  `cr_snr_clumpy_medium` uniform +14%, clumpy +9%. `dsa_area_ratio` (the
  *unweighted* surface-cell count) +35-39% is by design (more cells per
  column); its weighted counterpart `area_ratio_inj` moves <= 0.6%.
* Cause: the very first, unresolved stage of the blast. Default vs flag time
  series of `cr_sedov_injection_rate` (extra default run, raw_default/):
  at t = 0.007 (R = 8 / 16 cells at N = 48 / 96) the flag already has
  E_cr x1.13 / x1.23, its injection rate is x1.12 / x1.09 (higher velocity
  Mach, 0.90 vs 0.86 of true at N96, smaller spurious area); from t = 0.014
  on the rates agree within 1-3% and the E_cr ratio decays (1.054 / 1.083
  at t_end). Sedov injects equal energy per log t, so the early excess
  persists. Both are below the self-similar reference there (D/D_ref 0.69 vs
  0.62, 0.86 vs 0.79), the flag less so; E_cr_end/ref goes 0.714 -> 0.757 at
  N160, i.e. towards the reference, not past it.
* Everything else passes, incl. the planar DSA tests
  (`cr_dsa_efficiency_planar`, `cr_shock_tube_injection`,
  `cr_oblique_planar_shocks`), `cr_sedov_shock_area`, wind bubble, pion bump,
  MHD energy budget.

**Sedov 256** (`sedov_shock_finder`, hydro, finder on the final state): PASS
in both modes. Median M / M_Sedov 0.935 vs 0.936; 145656 vs 195480 surface
cells; total DSA area / 4 pi R^2 1.021 in both; per-orientation area within
-7.5..+5% (flag) vs -5..+9% (default).

**CWB 128** (control + zeta = 0.1 live DSA): E_cr at t_end 0.997 of default
(star-2 side 0.998, wind-1 side 0.877 of a small share); thermal / kinetic
energies equal to 4e-4. The e_cr map shifts by about a cell across the
wind-1 shock (+-40% stripes in the ratio, same total). Wall time on the same
GPU incl. compile: control 602 -> 401 s, DSA 540 -> 337 s (-33 / -38%; the
control runs the DSA path with zeta = 0). `test_cwb_shock_finder_correctness`
fails in both modes on the mirror-symmetry assert (stagnation point at
x ~ +0.08, pre-existing).

Adopting the flag as default would need a re-baseline of the 11 gated DSA
metrics above (energies +5-8% on Sedov, +9-14% on the SNR).

## Part 3b: DSA Mach numbers and grid artefacts on the Sedov shell (2026-11-04)

`sedov_mach_optin.py` runs the production DSA finder in both modes on the
same B7.3 control snapshots (zeta = 0, N = 96 / 160, t = 0.007 / 0.035 /
0.07) and keeps the Mach number the injection uses (velocity Mach,
CR-aware); `plot_sedov_mach_optin.py` -> `pics/sedov_mach3d_N96.png`,
`pics/sedov_mach3d_N160.png` (3D + sky map, coloured by M / M_true),
`pics/sedov_mach3d_test256.png` (the `sedov_shock_finder` run, its pressure
Mach), `pics/sedov_mach_stats.png`, `pics/sedov_mach_stats.txt`.

| N, t | median M/M_true def / flag | scatter def / flag | E per area, axis / diagonal def / flag |
|---|---|---|---|
| 96, 0.035 | 0.921 / 0.922 | 0.024 / 0.021 | 1.77 / 1.11 |
| 96, 0.07 | 0.947 / 0.944 | 0.020 / 0.019 | 1.67 / 1.01 |
| 160, 0.007 | 0.929 / 0.934 | 0.024 / 0.022 | 2.00 / 1.07 |
| 160, 0.035 | 0.939 / 0.939 | 0.015 / 0.015 | 1.09 / 0.98 |
| 160, 0.07 | 0.947 / 0.947 | 0.014 / 0.014 | 1.29 / 1.06 |

(scatter: area-weighted half 16-84% width / median; axis / diagonal: injected
energy per exact shell area in the |n|_inf > 0.95 vs < 0.63 bins.)

* Mach scatter is not higher with the flag: equal or lower in 5 of 6
  snapshots (N96 t = 0.007: 0.039 vs 0.037).
* Grid pattern in the Mach number: both show the same cubic +-1.5% pattern
  (axes low, face diagonals high), set by our walk, not by the surface. The
  default additionally has a dip at the body diagonals (-3.5% at N160, -5%
  at N96), which the flag removes.
* Injected energy per unit area: the default under-injects at the body
  diagonals (-24% at N160, t = 0.07) and has +-15% bin-to-bin structure;
  with the flag -7% and +-9%. Part 2's coarse 8 x 16 sky bins (0.98 for both)
  averaged this out.
* The test's own pressure Mach (N256) is 2.2% low in bands along the three
  coordinate planes through the blast centre, in both modes; the DSA velocity
  Mach has no such band (+0.5%).

Why `sedov_mach3d_test256.png` shows bands along the coordinate planes and
the N96/N160 figures do not (`sedov_mach_crosscheck.py`, median Mach on the
plane cells / rest; axis, face- and body-diagonal / overall median):

| state | estimator | plane/rest def / flag | axis def / flag | body-diag def / flag |
|---|---|---|---|---|
| test N256 t=0.1 | pressure (test call) | 0.978 / 0.979 | 0.985 / 0.984 | 1.014 / 1.010 |
| test N256 t=0.1 | velocity (DSA call) | 1.037 / 1.035 | 0.991 / 0.989 | 1.014 / 1.006 |
| CR control N160 t=0.07 | pressure (test call) | 0.987 / 0.989 | 0.980 / 0.978 | 0.984 / 1.008 |
| CR control N160 t=0.07 | velocity (DSA call) | 1.005 / 1.005 | 0.990 / 0.986 | 0.960 / 0.993 |

The two figure sets differ in the run (hydro test run, N256, t = 0.1 vs CR
control run, N160, t = 0.07) and in the Mach estimator (test: pressure;
DSA: velocity). The plane bands are in the simulated gas: on the three
planes through the blast centre the 256 test run's post-shock shell has
about 10% lower peak density and 4% lower peak pressure than elsewhere,
so every estimator reads them (pressure -2%, velocity +4%) and the flag,
which changes only which cells form the surface, cannot remove them.
The N160 run has them weaker (pressure -1.3%, velocity +0.5%). The cubic
+-1-2% pattern (axes low) is in all four cases and both modes (our walk);
the flag removes the default's body-diagonal dip.

# Part 4: not the default yet (2026-11-04)

A trial switch of the default (`dsa_continuous_normal_surface=True`, 219
registry metrics re-measured) passed all 18 affected tests, but was reverted
after `cr_oblique_planar_shocks.py`'s injection probe showed a failure mode:

| B8.2 N64, injection / exact (zeta 0 / 0.1) | M 3 old | M 3 flag | M 10 old | M 10 flag |
|---|---|---|---|---|
| (1,0,0) | 0.965 / 0.996 | same | 0.969 / 1.003 | same |
| (2,1,0) | 0.955 / 0.975 | 0.954 / 0.974 | 0.875 / 0.891 | 0.912 / 0.913 |
| (1,1,0) | 0.951 / 0.971 | 0.951 / 0.971 | 0.911 / 0.927 | 0.914 / 0.930 |
| (2,1,1) | 1.039 / 1.065 | 1.046 / 1.071 | 0.984 / 0.946 | 0.986 / 1.005 |
| (1,1,1) | 1.015 / 1.044 | 1.036 / 1.066 | 0.966 / 0.998 | **1.177 / 1.208** |

`planar_inj_check.py`: the injection pipeline is neutral (inj / zeta D equal
in both modes); the amount D = sum(flux w) agrees within 2% except in the last
snapshot (+82%). There the next periodic compression sits ~3 cells upstream.
The continuous-normal zones leave a one-cell gap to its zone (old: two), the
adaptive walk runs 5-8 steps (old: 2) and takes the pre-shock sample inside
the compression: at the same front cells the median Mach number drops to 7.0
(old 10.0) but the mean flux is 2.6x (pre-shock density and sound speed
high). Shocks with a compression a few cells upstream (clumpy media,
SN-driven turbulence, reverse shocks) are exposed. Untested fixes: our
dominant-axis zones with the continuous-normal surface, or stopping the walk
at a zone gap. The mode stays opt-in.

# Part 5: our zones + continuous-normal surface, now the DSA default (2026-11-04)

Fix for Part 4: `find_shocks_pfrommer(continuous_normal=True)` now changes only
the surface (phase 3). The zones (criterion 3) are always ours, along the
dominant axis, so the one-cell zone gap that the walk crossed is gone. The v2
zone sampler was removed. `dsa_continuous_normal_surface` defaults to `True`.

**B8.2 planar shocks** (N 64, injection / exact, `zeta` 0 / 0.1; old = segment surface):

| case | old | v2 zones (Part 4) | our zones |
|---|---|---|---|
| (1,1,1) M 10 | 0.966 / 0.998 | **1.177 / 1.208** | 0.972 / 1.002 |
| (2,1,1) M 10 | 0.984 / 0.946 | 0.986 / 1.005 | - / 0.889 |
| (2,1,0) M 10 | 0.875 / 0.891 | 0.912 / 0.913 | 0.800 / 0.848 |
| everything else | | | same as old within 1-2% |

Per snapshot (`planar_inj_check2.py`), D = sum(flux w) agrees with the old
surface within 1% everywhere, except in the last snapshot of the |k| >= sqrt 5
runs (D 0.68-0.81 of old). In those snapshots the next periodic slab is 1-2
cells ahead and every surface misreads the shock: the old one reads a median
M of 6.4 / 4.35 instead of ~10 and finds half the surface cells. These are
contaminated snapshots, not a failure mode. The test passes.

**Sedov.**
- Late-time injection rate: `inj_over_ref`, `D_over_ref` and `area_ratio_inj`
  within 1.2% of old.
- `E_cr_over_ref_end`: +5.4 / +2.3 / +1.0% (N 48 / 96 / 160). The v2 zones gave
  +5.4 / +8.3 / +6.0%.
- Plain cell count `area_ratio` 1.19 / 1.20 / 1.14, against 0.85 for old. This
  is expected: several cells per line, which `column_area_weights` divides out.
- Orientation, N 160 at t 0.07 (`sedov_mach_stats.png`):
  - energy per exact area at the body diagonals 0.958, against 0.762 for old;
  - all |n|_inf bins within 0.955-1.086, against 0.76-1.16 for old;
  - body-diagonal Mach dip gone;
  - per-cell scatter equal (0.014).

**CWB 128** (`optin_cwb128.png`):
- E_cr is 0.995 of old (star-2 side 0.994; wind-1 side 1.17 of a 0.3% share).
- DSA run 328 s against 540 s; control 397 s against 602 s, same GPU.
- `test_cwb_shock_finder_correctness` fails on mirror symmetry, as it does
  with the old surface (pre-existing).

**Sedov 256 finder test:** PASS, median M / Sedov 0.936 against 0.935 for old.

**DSA regression** (flag forced, jobs 41-51, HEAD registry,
`optin_regression.txt`):
- All physics gates pass. 7 registry gates fail in 5 tests:
  - `cr_sedov_taylor` and `cr_dsa_mach_dependence`: CR fraction +5.4% at N 48
    (+2.3% at N 96, +1.0% at N 160);
  - `cr_sedov_injection_rate`: plain `dsa_area_ratio` +35-39% (diagnostic),
    plus `E_cr_over_ref_end` at N 48;
  - `cr_snr_clumpy_medium`: e_cr / e_tot +3.3%;
  - `cr_sedov_self_similar` N 48: shell ratios +3%.
- `L1_p_cr` improves by 3-8%.
- These 5 tests were re-baselined from the run's logs (2026-11-04; `cr_sedov_self_similar` N 160 not re-measured).
- `cr_sedov_shock_area`, `cr_sedov_mach_estimators` and
  `cr_oblique_planar_shocks` request the segment surface explicitly; they
  study DB7's weight.
