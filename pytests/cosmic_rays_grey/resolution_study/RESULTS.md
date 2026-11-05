# Resolution study: results (see PLAN.md)

Code state: branch `feature/cosmic-rays-grey`, HEAD `c70df4d` (the user's commit of the
continuous-normal DSA surface default + re-baselined registry) plus the uncommitted
`_cwb_setup.build_cwb` split; every output file carries `stamp_git_head` /
`stamp_git_diff_sha1`. From C0 on (2026-11-05) also the `_wind_ei3D` normalisation fix.

## Summary: answers to Q1-Q5 (2026-11-05)

- **Q1 Hydro.** Sedov radius and speed are converged to < 1% at every N used (the front
  estimator leads by a constant ~0.9 cells); the shell compression is far from resolved
  (3.15 of 4 at 224³). CWB: the wind needs R_inj ≥ 3-4 cells to follow Chevalier & Clegg;
  the wind-2 stand-off and the contact still move by ~5% between 160³ and 192³.
- **Q2 Finder.** Detection is complete at every N, with false positives only at 32³. The Mach
  number is a universal function of the cells per shock radius,
  |ΔM/M| ≈ 0.55 (R/dx)^-0.6, for every Sedov setup, Mach number (2.5-760) and ζ model,
  always low. The finder's own share is first order (ΔM/M ≈ −(2-3) dx/R; S2); on the
  Sedov blast the coarse hydro partly cancels it, on the CWB it adds an error of the same
  size (C2). Area over-count +2.5% (strong) / +0.5% (S-CP) at ~90 cells per radius;
  orientation artefacts < per-bin noise. The old and new DSA surfaces agree within 1% on
  the Sedov blast (X2). No cell-based parameter carries the error (X1).
- **Q3 DSA.** The injection error is the Mach error times A = d ln(ζ δ M³)/d ln M plus the
  area excess, with no other loss term (S3, S4: single-Mach prediction within 1-3%).
  Strong shocks: 0.89-0.92 of the reference at ~90 cells per radius, order ~0.6. E_cr
  profits from the deposit surplus (+3.5% at 90 cells, decaying). Point-explosion starts
  carry a fixed E_cr deficit from the unresolved deposit phase that decays only like
  (t_onset/t)^0.3 (S3); use a resolved start for E_cr statements.
- **Q4 Mach range.** Contrary to PLAN.md's expectation, Mach errors matter at every Mach
  number: A = 3 even when ζ and δ saturate (the flux is ∝ v_s³). A = 4 at M = 3 (constant
  ζ); with KR13 A = 5 for 5 < M < 15 and 9-12 for 2 < M < 5, which makes KR13 injection
  at moderate Mach numbers by far the hardest quantity to converge.
- **Q5 Practical rule** (cells per shock radius, Sedov; per stand-off distance, CWB):

  | quantity | 10% | 5% |
  |---|---|---|
  | Mach number (any M) | ~20 | ~60 |
  | injection, constant ζ, strong shock | ~80 | (~300, extrap.) |
  | injection, constant ζ, M ≈ 3 | ~90-130 | – |
  | injection, KR13, 2 < M < 15 | hundreds (20% needs ~75-170) | – |
  | E_cr, constant ζ, strong, resolved start | ~55-60 | – |
  | CWB total injection (constant ζ) | R_inj ≥ 3 cells, wind-2 stand-off ≥ 9 cells | R_inj ≥ 4, stand-off ≥ 12 |

  The CWB total converges faster than its parts (per-shock M50 still ±4% at 16 cells of
  stand-off) and carries two non-converging biases: the wind-1 apex Mach over-read
  (+17-25%) and an injection 1.5-1.6× the exact-upstream estimate.
- **Found on the way:** the `_wind_ei3D` Ṁ / L normalisation bug (fixed, C0); a 2× CR
  energy surplus over the injected energy in `cr_wind_bubble` (open); KR13 CWB 192³ goes
  non-finite (open, C4).

## Step 0: infrastructure (2026-11-04)

Files:
- `common.py`: paths, `require_pinned_gpu`, `code_stamp`, `run_segments` (segmented
  integration with a host callback per snapshot), `InjectionAccumulator` (realized
  injection per step via `jax.debug.callback`, total + 16^3 block map), `restrict`
  (conservative block average), `save_npz`.
- `setups.py`: `build_sedov_pe` (item-7 point explosion, p_0 / KR13 selectable),
  `build_cwb` (`_cwb_setup.build_cwb`, new: `run_cwb` split into build + run, unchanged
  behaviour).
- `sedov_run.py`: segmented Sedov run with B7.3's per-snapshot analysis + accumulator.
- `step0_probe.py`: restriction tests, memory / time probes, segmented-vs-continuous check.
- `analyse.py` (CPU): metric vs R/dx, metric vs N at fixed t, observed orders,
  M_true from the trajectory.
- `launcher/`: copies of `launch.sh` / `verify.sh` / `dispatch.sh`
  (live in `/export/scratch/nknoell/resolution_study/`; jobs in `jobs/<tag>.sh`, logs in `logs/`).

### Accumulator and segmentation checks

- First version of the accumulator baked the target instance into the compiled loop at
  trace time: a second run of the same setup in one process reported nothing. Fixed:
  the loop calls a class-level dispatcher that forwards to the instance active at run
  time. The recorded step time is the time at the end of the step.
- Segmented vs continuous (32^3 S-PE, to t = 0.004, step by step): identical steps up
  to the segment end, which shortens one step and shifts the later step boundaries by
  ~0.2 dt. In this unresolved phase (R/dx ~ 3) the first injecting step then moves by
  one step and E_cr(0.004) differs by 28% (1.31e-4 vs 0.94e-4): injection switches on
  in lumps of 2-7e-5 per step while the shell is 3 cells across. Not a segmentation
  error, but a property of the early blast that S-PE inherits; quantified on 48^3 to
  T_END in `segcheck` below.
- `segcheck` (48^3 S-PE, ζ = 0.1, to T_END = 0.07; one segment vs ten): L1 differences
  of the final ρ / p / e_cr 1.7e-5 / 1.1e-4 / 1.3e-3; E_cr(T_END) and the accumulated
  injection agree to 1.0e-4 and 1.4e-4 (272 vs 278 steps). No recompilation after the
  first segment (later segments 0.14-0.34 s at 48^3). E_cr in the box at T_END = 0.774
  of the injected energy (adiabatic losses since injection).
- `restrict`: uniform and linear states exact, mass / momentum / energy / e_cr
  conserved (round-off) for factors 2 and 4: pass.
- End-to-end 48^3 S-PE (control + ζ = 0.1, 30 log-spaced snapshots, `sedov_run.py`):
  48 s each including compilation. B7.3's analysis restricts the radius search to
  r > 0.1 and takes the median Mach over all surface cells, so before t ≈ 0.01 it reports
  R ≈ 0.13 and M = 0 while CRs are already injected (from t = 0.0018, R/dx ≈ 6). Hence the
  new `sedov_diag.py` (front radius at any R > 2 dx, Mach quantiles over the injecting
  cells), used by `sedov_study.py` from S0 on.

### 0.3 Memory and time (RTX 2080 Ti, 11 GB, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`)

| run (DSA on) | fits | peak / failing allocation | s / step | full run |
|---|---|---|---|---|
| CWB 192^3 | yes | 7.2 GB peak (run + finder analysis) | 0.40 | ~5400 steps → ~36 min |
| CWB 224^3 | **no** | 8.7 GiB single allocation | | |
| CWB 256^3 | **no** | 12.1 GiB single allocation | | |
| Sedov 256^3 | **no** | 11.0 GiB single allocation | | |

So the CWB ladder ends at **192^3**: **N = 64, 96, 128, 160, 192** (multiples of 32,
injection radius 2-6 cells), r = 2 chains 64-128 and 96-192 (no three-grid r = 2 chain;
observed orders from the generalized three-grid formula). Sedov: see the 224 / 192 probes
below. (The 256^3 CWB figures of earlier rounds were hydro-only controls.)

### 0.5 1D spherical reference

Not available: `inject_crs_at_shocks` supports Cartesian geometry only
(`cr_grey_injection.py`, "Curvilinear/spherical geometry is not supported here"); the
finder has 1D spherical gradients but the injection does not. Per plan: S-CP reference
= Richardson-extrapolated 3D + M_true from Rdot. Not built.

### 0.6 Analysis skeleton on the B7.3 data (old surface, 10 snapshots, 48/96/160)

`pics/step0_b73_vs_Rdx.png`, `pics/step0_b73_mach_vs_N.png`, `pics/step0_b73_orders.txt`.
- M_true from the trajectory fit agrees with B7.3's self-similar fit to 0.2% (48, 96)
  and 2% (160, late snapshots): usable for S-CP.
- `mach_ratio` and `area_ratio_inj` collapse onto one curve against R/dx; the Mach
  ratio plateaus at ~0.94 from R/dx ~ 30 on (no further convergence 96 -> 160).
- `D_over_meas` does **not** collapse: 48^3 sits at ~0.69 where 96^3 reads ~0.82 at
  the same R/dx. Something besides R/dx sets the flux at 48^3 (to be resolved in S1/S3).
- Observed orders from three grids with ratios 2 / 1.67 are mostly undefined (the
  differences change sign): as expected, 10 snapshots and non-constant ratios are not
  enough, hence the r = 2 chains of the plan.

### Sedov memory (DSA on)

| run | fits | peak | s / step | full run (to T_END, upper) |
|---|---|---|---|---|
| Sedov 192^3 | yes | 5.7 GB | 0.36 | ~6 min (+ ~14 s finder analysis per snapshot) |
| Sedov 224^3 | yes | 8.3 GB | 0.56 | ~14 min (+ ~14 s per snapshot) |
| Sedov 256^3 | no | 11.0 GiB allocation | | |

Sedov ladder: **N = 32, 48, 64, 96, 128, 192, 224** (r = 2 chains 32-64-128 and 48-96-192;
224 as the finest grid).

## S0: counter-pressure blast sizing (2026-11-04)

`s0_sizing.py`, `pics/S0_sizing.png`, `pics/S0_sizing.txt`. 64^3, self-similar start at
R_0 = 0.08 (5 cells), controls at p_0 = 0.25 / 0.4 / 0.6 and KR13 at 0.4, run until the
front passes 0.44; M_true = Rdot / c_0 from the front trajectory.

| p_0 | t(R = 0.42) | M_true at R_0 | M_true at R = 0.42 | p_0 / p_post at R_0 | finder M / M_true (median) |
|---|---|---|---|---|---|
| 0.25 | 0.0730 | 40.7 | 3.76 | 5.6e-4 | 0.878 |
| 0.4 | 0.0720 | 32.2 | 3.05 | 8.9e-4 | 0.881 |
| 0.4, KR13 | 0.0765 | 30.0 | 2.93 | 1.1e-3 | 0.865 |
| 0.6 | 0.0706 | 26.4 | 2.58 | 1.3e-3 | 0.880 |

**Chosen: p_0 = 0.6, R_0 = 0.08, t_end = 0.07** (front at R ≈ 0.42; same t_end as S-PE / S-SS):
M_true falls from ~26 to ~2.6, i.e. through KR13's steep range (2-15) for most of the run;
the counter-pressure is 1e-3 of the post-shock pressure at the start, so the self-similar
initial state is valid. The front runs slightly ahead of the p_0 = 0 Sedov trajectory
(the ambient thermal energy adds to the blast), as expected.

First look (64^3): finder median Mach / M_true rises from 0.65 at R/dx ≈ 6 to 0.91 at
R/dx ≈ 18 and then falls slowly to 0.88-0.90 as M_true drops towards 2.6, the same for all
p_0 (so it is set by R/dx and M, not by p_0 itself); KR13 reads ~1-2% lower while
R/dx < 12.

## S1: hydro and finder on the control runs (2026-11-04)

`sedov_study.py` (ζ = 0, 30 log-spaced snapshots, states kept at snapshots 9/14/19/24/29),
`s1_analyse.py`; `pics/S1_mach_vs_Rdx.png`, `pics/S1_area_vs_Rdx.png`, `pics/S1_hydro.png`,
`pics/S1_summary.txt`. N = 32-224 (S-CP 48-224). M_true,ref: exact Sedov for S-PE / S-SS,
the 224^3 trajectory for S-CP; M_true,own: the run's own front trajectory (local quadratic
fits of ln R(ln t), ±5 snapshots; a global cubic biased the end-point slope by up to +10%).
GPUs 0, 1, 3, 9 (idle at launch; pins verified). Wall: 224^3 S-PE / S-SS ~25 min,
S-CP 61 min (the hot ambient medium sets a small time step).

**Mach number (what DSA sees, against the truth), at fixed times:**

| N | R/dx at t = 0.07 | S-PE 0.035 / 0.07 | S-SS 0.035 / 0.07 | S-CP 0.035 / 0.07 (M_true 3.6 / 2.5) |
|---|---|---|---|---|
| 32 | 12.7 | 0.861 / 0.875 | 0.859 / 0.874 | |
| 48 | 19 | 0.884 / 0.896 | 0.891 / 0.904 | 0.881 / 0.914 |
| 64 | 25 | 0.894 / 0.912 | 0.909 / 0.921 | 0.898 / 0.930 |
| 96 | 38 | 0.914 / 0.935 | 0.930 / 0.939 | 0.922 / 0.951 |
| 128 | 51 | 0.929 / 0.946 | 0.940 / 0.950 | 0.934 / 0.962 |
| 192 | 76 | 0.953 / 0.955 | 0.952 / 0.961 | 0.948 / 0.973 |
| 224 | 89 | 0.960 / 0.958 | 0.957 / 0.964 | 0.953 / 0.977 |

- **The Mach error is a function of R/dx alone** (S-SS, S-CP: all N collapse onto one curve
  once a start-up transient of ~1.5x in radius after the self-similar start has passed):
  M_finder / M_true ≈ 0.85 at R/dx = 10, 0.90 at 20, 0.93-0.94 at 40, 0.96-0.98 at 90. The
  finder always reads low; the deficit shrinks like (R/dx)^-p with p ≈ 0.6 (S-SS, M 170-760)
  to 0.8 (S-CP, M 26 → 2.5): slower than first order, and not converged at 224^3 (−2 to −4%).
- The own-shock ratio (finder only, against the shock the grid actually has) is within
  ±1% of the reference ratio for S-SS / S-CP after the transient: the grid's shock speed is
  right, the deficit is the finder's reading of the smeared jump, not slow hydro.
- S-PE: late times as S-SS; early (t ≲ 0.01, R/dx < 10 at N ≤ 64) up to −30% plus the
  departure of the point explosion from Sedov.
- **Per-cell scatter** (16-84% half width / median) also collapses: 5-9% at R/dx ≈ 6, 3% at
  20, 1.5% at 50, 1% at 90 (∝ (R/dx)^-0.75). **Orientation (grid) artefact** in the Mach
  number: spread over |n|_inf bins 9% (S-PE 32^3) → 2-4% (R/dx ≳ 25) → ~2% at 224^3.

**Shock surface and hydro:**
- Completeness 1.00 (every solid-angle bin has surface cells near the front) at all N
  after the start; false positives only at 32^3 S-PE (1.8% at t = 0.07).
- Weighted area A_inj / 4πR_ref² at t = 0.07: 1.14 / 1.07 / 1.05 / 1.04 / 1.035 / 1.028 /
  1.026 (S-PE, N = 32 ... 224); S-SS 1.075 → 1.025; S-CP 1.04 → 1.005. Early, unresolved
  S-PE blasts over-count strongly (3.9 at 32^3, t = 0.01: several surfaces in the smeared
  deposit). The ~2.5% excess at 224^3 (S-PE / S-SS) is not yet converged (p ≈ 0.5-1).
- Front radius (half maximum of the density excess) runs ahead of the reference by a
  constant ~0.9 cells (R_err × R/dx = 0.80 / 0.81 / 0.77 / 0.79 / 0.80 / 0.87 / 0.92 cells,
  S-PE): a first-order estimator offset, not a shock-speed error.
- Peak compression at t = 0.07: 1.93 → 3.15 (S-PE, 32 → 224; strong-shock limit 4) and
  1.96 → 2.45 (S-CP, M = 2.5: limit 2.7): the shell is far from resolved even where the
  Mach number is within 4%; the adaptive walk reads the jump past the smeared shell.

## S2: hydro vs finder share of the error (+ S1 orientation) (2026-11-04)

`s2_restrict.py` (GPU; block-averages the 192^3 control states kept at snapshots
9/14/19/24/29 onto 96 / 64 / 48 / 32 and runs `sedov_diag` there; flux per |n|_inf bin of
every kept state), `s2_analyse.py`; `pics/S2_error_split.png`, `pics/S1_orientation.png`,
`pics/S2_summary.txt`.

**Mach number: the finder's own error is first order; the hydro partly cancels it.**
- Restricted fine states (finder at coarse N on a well-resolved flow) read the Mach number
  *lower* than the native coarse runs at every N and time: finder share −18% / −14% / −10% /
  −7% at R/dx ≈ 13 / 19 / 25 / 38 (S-SS), total −13% / −10% / −8% / −6%. The coarse run's
  own smearing *raises* the reading (hydro share +7% → +1%).
- Finder share × R/dx ≈ −2.3 to −2.9 (S-PE / S-SS, M 170-760) and −1.8 to −2.3 (S-CP,
  M 7 → 2.5) over R/dx = 10-40: **ΔM/M ≈ −(2-3) dx/R**, i.e. first order in the cells per
  radius with a coefficient that grows slowly with R/dx (S1 at 224^3: total × R/dx ≈ −3.2).
  Interpretation: the adaptive walk takes its post-shock sample a few cells behind the
  front, where the steep Sedov post-shock profile has already fallen; a sharper flow at the
  same dx makes that worse, a numerically broader shell (coarse hydro) flattens the profile
  and partly compensates. As the hydro share decays faster (≈ +7% → +1% while the finder
  share goes −18% → −7%), the total approaches the finder's first-order deficit from above,
  which is why S1's apparent order is only 0.6-0.8.
- The point explosion adds a large early hydro share (S-PE 32^3 at R/dx < 6: total −62%,
  finder −34%): the unresolved deposit, not the finder.

**Area:** the finder share is a slowly decaying over-count, +3-5% (S-PE / S-SS) and
+1-2% (S-CP) at R/dx = 20-40; the hydro adds +1-2%. In S-PE the early (R/dx < 15) area
excess of +100-600% is entirely hydro (several surfaces in the smeared deposit; the
restricted fine states read +6-20%).

**Orientation (grid artefacts) at t = 0.07** (`S1_orientation.png`): dissipated flux per
unit exact area per |n|_inf bin within 0.89-1.11 at all N, no trend with N (bin-to-bin noise
of ±3-5% at 224^3, the body-diagonal bin covers 3% of the sphere); the Mach number per bin
has a small systematic pattern, diagonal bins +1% and axis bin −1% (S-SS / S-CP; −6% axis at
32^3 S-PE), shrinking to ±1% by R/dx ≈ 40 and not further. No grid artefact larger than the
per-bin noise remains in the dissipated flux with the continuous-normal surface.

## S3: live DSA, constant ζ = 0.1, S-PE and S-SS (2026-11-04)

`sedov_study.py --zeta 0.1` at N = 32 … 224 (30 snapshots, t_end = 0.07), `s3_analyse.py`
(reference: `solve_sedov_two_fluid(0.1, "gas_rh")`, f_cr = 0.1398, injection t/E = 0.0416);
`pics/S3_Ecr_vs_t.png`, `pics/S3_D_split_vs_Rdx.png`, `pics/S3_convergence.png`,
`pics/S3_summary.txt`.

**Injection collapses in R/dx and is a Mach-number deficit.** S-SS late injection / reference
(realized = nominal to < 0.3%: the accumulator agrees with ζ D on every snapshot):

| N | R/dx (t_end) | injection / ref | area | M50 / M_true | surplus (E_cr / kernel pred.) |
|---|---|---|---|---|---|
| 32 | 12.5 | 0.653 | 1.090 | 0.860 | 1.118 |
| 64 | 25 | 0.774 | 1.060 | 0.909 | 1.082 |
| 128 | 50 | 0.854 | 1.041 | 0.940 | 1.050 |
| 192 | 75 | 0.884 | 1.032 | 0.951 | 1.039 |
| 224 | 87 | 0.893 | 1.029 | 0.955 | 1.035 |

- The split closes: injection ≈ area × (M50/M_true)³ (strong shocks: δ → const, A = 3),
  e.g. 224³: 1.029 × 0.955³ = 0.896 vs 0.893. All N fall on one curve against R/dx after a
  start-up transient of a factor ~1.3 in radius.
- Observed order of the injection deficit ≈ 0.55-0.6 (pairs vs exact), the Mach order of S1
  times 1: the injection error is 3 × the Mach error, minus the area excess.
- Deposit surplus (E_cr over the adiabatic-kernel prediction from the run's own injection,
  B7.2's "deposit surplus") 1.12 → 1.035, also converging; it partly cancels the
  injection deficit in E_cr.
- S-SS E_cr / E_cr,ref = 0.92 (32³) → 0.987 (224³) at t = 0.07, but ~half of E_cr at t_end
  is the exact initial CR content (start at R = 0.15; kernel weight (t_start/t)^0.3 ≈ 0.48),
  so E_cr understates the injection error. The asymptotic (memory-free) estimate
  injection × surplus is 0.73 / 0.84 / 0.90 / 0.918 / 0.924 (32 / 64 / 128 / 192 / 224).

**S-PE: the unresolved start costs a fixed, slowly decaying E_cr deficit.** Injection is
zero until the front leaves the R = 0.05 deposit (R/dx ≈ 4.5 at 32³, ≈ 16 at 224³: for
N ≥ 96 the onset is set by the deposit radius, not by dx), with early area excesses up to
25× (S2: hydro). Late injection / ref then matches S-SS within the oscillations (0.914 at
224³). But E_cr / E_cr,ref = 0.41 / 0.59 / 0.70 / 0.744 (32 / 64 / 128 / 224) at t = 0.07,
0.66 at t = 0.035 at 224³: the missing early injection decays only like (t_onset / t)^0.3
(kernel memory), (1 − 0.66)·2^−0.3 = 0.28 vs measured 0.26. Fraction of the final E_cr
injected while R_front < 8 cells: 26% (32³), 11% (48³), 5% (64³), 0 for N ≥ 96.
**Use S-SS (or a resolved, self-similar start) for E_cr convergence statements;** S-PE's
E_cr is a property of the initial deposit.

## S4: Mach sweep with DSA (S-CP, ζ = 0.1 and KR13) (2026-11-04)

`sedov_study.py --setup cp --p0 0.6` (`--zeta 0.1` / `--kr13`), N = 48 … 224, M_true
26 → 2.5; truth = each model's finest trajectory. `s4_analyse.py`;
`pics/S4_mach_error_map.png`, `pics/S4_flux_amplification.png`,
`pics/S4_Ecr_KR13_vs_const.png`, `pics/S4_summary.txt`.

**Mach number:** the error map is a function of R/dx alone; at fixed N the deficit grows
toward high M only because high M is early (small R). At M = 10 / 5 / 3 (224³, R/dx = 33 /
53 / 78): 0.924 / 0.940 / 0.958 (ζ = 0.1), 0.916 / 0.940 / 0.962 (KR13). CRs in the shock
(KR13 has 2× the CR pressure) change the finder reading by < 1%.

**Flux amplification A = d ln(ζ δ M³)/d ln M** turns ΔM/M into an injection error
A ΔM/M. A = 3.0 (strong) → 4.1 (M = 3) for constant ζ (δ's weak-shock drop); for KR13
A ≈ 3.4 at M = 10, ≈ 5 for 5 < M < 15, and ≈ 9-12 for 2 < M < 5 (ζ ∝ M^5.3 there). The
KR13 fit has a 2.5% step at M = 5 (0.0999 → 0.1028); the A spike at M = 5 in the figure is
that step in the finite difference, and nominal/exact steps by ~+0.15 when M_true crosses
5 while M_finder is still below 5.

| N = 224 | M_true = 10 | 5 | 3 | realized/exact (late) |
|---|---|---|---|---|
| ζ = 0.1: nominal / exact (single-Mach prediction) | 0.770 (0.780) | 0.809 (0.810) | 0.849 (0.842) | 0.841 |
| KR13 | 0.723 (0.737) | 0.580 (0.586) | 0.688 (0.680) | 0.655 |
| N = 96, ζ = 0.1 | 0.667 (0.679) | 0.710 (0.724) | 0.751 (0.754) | 0.745 |
| N = 96, KR13 | 0.605 (0.618) | 0.408 (0.426) | 0.505 (0.510) | 0.479 |

- The single-Mach prediction area × g(M50)/g(M_true), g = ζ δ M³, reproduces the measured
  nominal/exact within 1-3% at every N and M: **the injection error is the Mach error times
  A, plus the small area excess; there is no other loss term.**
- Total injected energy (accumulator) converges at order 0.45-0.65 (ζ) and 0.4-0.55 (KR13);
  224³ / Richardson extrapolation = 0.79-0.84 (ζ = 0.1) and 0.69-0.75 (KR13), consistent
  with the late realized/exact ratios 0.84 / 0.66.
- E_cr(t_end) / finest differs by < 1% between 192³ and 224³ for both models, but this is
  not convergence: E_cr(t_end) is dominated by the initial CRs of the S-CP start (0.139 /
  0.260 at t_start vs 0.12 / 0.14 injected). Use the injected energy.

## S5: Sedov Q5 rows (2026-11-04)

`s5_summary.py`: every snapshot of every N pooled against R/dx (start-up transient
dropped), `pics/S5_q5.png`, `pics/S5_summary.txt`. "measured": every pooled point beyond
this R/dx is within tolerance; "fit": power law over R/dx ≥ 15 (* = extrapolated beyond
the finest point, R/dx ≈ 90 here, 35-40 for S-CP at M ≥ 8).

| quantity | case | p | 10% | 5% | 2% |
|---|---|---|---|---|---|
| ΔM/M | all setups, control and DSA, M 2.5-760 | 0.55-0.7 | R/dx ≈ 18-25 | ≈ 55-65 (CP M ≥ 8: 58*-85*) | ≈ 170*-300* |
| ΔD/D (injection), ζ = 0.1 | strong (S-SS/S-PE, M > 100) | 0.5-0.6 | ≈ 70-82 | | 20%: ≈ 27-33 |
| | S-CP M ≥ 8 / M ≤ 4 | 0.45 / 0.64 | ≈ 210* / 91-133* | | 20%: 45* / 46 |
| ΔD/D, KR13 | S-CP M ≥ 8 / M ≤ 4 | 0.41 / 0.55 | ≈ 400* / 600* | | 20%: 74* / 170* |
| ΔE_cr/E_cr, ζ = 0.1 | S-SS (injection × surplus) | 0.65 | ≈ 55-60 | | 20%: ≈ 20 |
| | S-CP (realized injection) | 0.59 | ≈ 160* | | 20%: ≈ 50 |
| ΔE_cr/E_cr, KR13 | S-CP (realized injection) | 0.48 | ≈ 1000* | | 20%: ≈ 240* |

Reading:
- **The Mach number is a universal function of R/dx**: |ΔM/M| ≈ 0.55 (R/dx)^−0.6, the same
  for every setup, Mach number (2.5-760), constant ζ / KR13 / control. 5% needs ~60 cells
  per shock radius, 10% ~20.
- The injection error is A × the Mach error (S4), so the required resolution scales like
  (A / tol)^(1/p): with p ≈ 0.6, 10% in D for constant ζ at strong shocks needs ~80 cells
  per radius; at weak shocks (A ≈ 4) ~100-130; with KR13 at 2 < M < 15 (A = 5-12) it is out
  of reach (hundreds of cells per radius) and only the 20-35% level is achievable at
  R/dx ≈ 50-100.
- E_cr profits from the deposit surplus (+3-4% at R/dx ≈ 80, decaying) and is within 10%
  for constant ζ at strong shocks from R/dx ≈ 55-60; this is partly a cancellation of two
  first-order errors.
- Everything is sub-first-order (p ≈ 0.5-0.7), although the finder's own error is first
  order (S2): the hydro share converges with the opposite sign and slower.

## C0: wind preconditions, and the wind-injection fix (2026-11-05)

`cwb_study.py` (64 / 96 / 128 / 160 / 192, control and ζ = 0.1, 20 snapshots, t_end =
`_cwb_setup` default; R_inj = N/32 cells), `c_analyse.py`; `pics/C0_winds.png`,
`pics/C_summary.txt`.

**The first CWB pass failed the C0 precondition** (decision 3): the effective Ṁ of both
winds was 0.33 / 0.54 / 0.67 / 0.76 of nominal at 64 / 96 / 160 / 192. Cause: `_wind_ei3D`
(FV path) divided Ṁ and L by the analytic 4/3 π R_inj³ but deposits only into cells with
r ≤ R_inj − dx/2, whose volume is 0.36 / 0.57 / 0.69 / 0.77 of it (measured/mask agree to
1-9%; the FD path `_wind_ei3D_source` already normalised by the mask volume). **Fixed on
user request (2026-11-05):** `_wind_ei3D` now normalises by the discrete mask volume per
source. All CWB runs were redone; the unfixed set is kept in
`RAW/cwb_unfixed/`. `analytic_wind_zone.py` passes (7/7); `cr_wind_bubble` was
re-baselined (see X3 / its docstring). The wind-flux estimator now uses the mean flux
density over the 1.5 R_inj shell × 4π r² (the raw cell count of that shell deviates from
4π r²/dx² by up to 5% at 3 cells).

After the fix: Ṁ / nominal = 0.951 / 0.971 / 0.986 / 0.973 / 0.994 and L / L_w = 0.94-0.99
(64 → 192), the same for both winds; the remaining 1-5% is mostly the estimator at 3-9
cells radius, with no systematic N trend. **Precondition met.** Window t/t_end ≥ 0.6: the
axis positions move by ≤ 2.3% of the separation and the injection rate by ≤ 5% (at
64-192).

`test_cwb_shock_finder_correctness` (mirror symmetry) is stale since the winds became
unequal (PLAN.md C0); not run.

## C1: CWB hydro and finder, control runs (2026-11-05)

`pics/C1_axis_profiles.png`, `pics/C1_standoff_mach_vs_N.png`, `pics/C_summary.txt`.

| N | R_inj (cells) | x_TS1 | x_c | x_TS2 | M_TS1 axis (finder / CC85 at r_s) | M_TS2 axis | M50 wind-1 TS [16, 84] | M50 wind-2 TS |
|---|---|---|---|---|---|---|---|---|
| 64 | 2 | +0.070 | +0.094 | +0.102 | 8.95 / 10.4 | – / 5.0 | 3.42 [2.70, 5.11] | 2.30 |
| 96 | 3 | +0.063 | +0.073 | +0.105 | 11.5 / 10.2 | 3.1 / 4.9 | 4.73 [3.86, 6.54] | 3.37 |
| 128 | 4 | +0.066 | +0.080 | +0.106 | 12.8 / 10.3 | 5.3 / 4.9 | 5.24 [4.45, 7.03] | 4.13 |
| 160 | 5 | +0.068 | +0.080 | +0.110 | 12.4 / 10.4 | 4.7 / 4.7 | 5.60 [4.83, 7.23] | 4.39 |
| 192 | 6 | +0.073 | +0.084 | +0.118 | 12.3 / 10.5 | 4.8 / 4.4 | 5.79 [5.04, 7.34] | 4.56 |

(stars at x = ∓0.2; ram-pressure contact with nominal Ṁ, v_inf: x_c = +0.110.)
- The upstream winds follow Chevalier & Clegg once R_inj ≥ 3-4 cells (local |u|/c_s on the
  axis; 64³ is ~25% slow). Hence pre-shock M ≈ 10 (wind 1) and ≈ 4.5 (wind 2) at every N.
- The wind-2 stand-off and the contact still move outward with N (x_TS2 0.102 → 0.118,
  x_c 0.073 → 0.084 from 96 to 192): the colliding region is not converged at 192³
  (wind-2 stand-off ≈ 16 cells at 192³).
- **The finder over-reads the apex Mach number of the wind-1 shock** by 17-25% (axis
  surface cell 12.3-12.8 vs CC85 10.3-10.5 and the local upstream |u|/c_s ≈ 9.5) at
  N ≥ 128; the wind-2 apex is within ±10%. (This is the known CWB velocity-Mach excess,
  Phase B 2026-10-20: "+41% split vel 1.24 × area 1.08", accepted.)
- The area-weighted M50 per shock is dominated by the oblique wings (mean |r̂·n| ≈ 0.35 /
  0.44) and rises with N without levelling off: +3.4% (wind 1) and +4% (wind 2) from 160 to
  192.

## C2: CWB restriction test (2026-11-05)

`c2_restrict.py` (GPU; the 192³ control last state block-averaged by 2 / 3 / 4, ζ = 0.1
settings, R_inj = N/32 cells, so the restricted state carries the fine run's wind
structure), output `RAW/cwb/C2_control.npz`.

| N | M50 wind 1: native / restricted / 192 | split total = hydro × finder | M50 wind 2 | split | nominal inj. wind 1 / wind 2 (native, restricted; 192: 0.194 / 0.078) |
|---|---|---|---|---|---|
| 96 | 4.72 / 5.32 / 5.90 | −20% = −11% × −10% | 3.19 / 3.75 / 4.26 | −25% = −15% × −12% | 0.164 / 0.078, 0.203 / 0.075 |
| 64 | 3.46 / 4.61 / 5.90 | −41% = −25% × −22% | 2.26 / 3.28 / 4.26 | −47% = −31% × −23% | 0.194 / 0.045, 0.215 / 0.061 |

Unlike the Sedov blast (S2: the coarse hydro *raises* the reading and partly cancels the
finder deficit), the CWB's hydro share has the same sign and size as the finder share: a
coarse run's wind is slower (CC85 acceleration zone under-resolved at R_inj = 2-3 cells)
and its colliding region sits elsewhere. The finder's own share (−10% / −12% at 96 =
wind-1 stand-off ≈ 26 cells, wind-2 ≈ 8 cells) is comparable to the Sedov value at the same
cells per radius. Injection per shock reacts less than M50³ because the wind-1 area and
the obliquity distribution compensate.

## C3: live DSA on the CWB, ζ = 0.1 (2026-11-05)

`c_analyse.py`, `c_expect.py` (GPU; semi-analytic reference); `pics/C3_Ecr_vs_t.png`,
`pics/C3_injection_per_shock_vs_N.png`, `pics/C3_Ecr_per_Lwind_vs_N.png`,
`pics/C_summary.txt`.

Injected CR energy per total wind luminosity L_w = Σ Ṁ v_inf² / 2:

| N | E_inj(t_end)/(L_w t_end) nominal / measured L_w | window rate / L_w | wind-1 TS | wind-2 TS | ambient | box E_cr/(L_w t_end) | L_exp / L_w |
|---|---|---|---|---|---|---|---|
| 64 | 3.58e-3 / 3.81e-3 | 4.11e-3 | 3.27e-3 | 0.83e-3 | 0 | 1.95e-3 | 2.30e-3 |
| 96 | 4.32e-3 / 4.48e-3 | 4.63e-3 | 3.13e-3 | 1.50e-3 | 0 | 2.37e-3 | 2.62e-3 |
| 128 | 4.45e-3 / 4.53e-3 | 4.94e-3 | 3.34e-3 | 1.60e-3 | 0 | 2.62e-3 | 3.08e-3 |
| 160 | 4.59e-3 / 4.72e-3 | 5.18e-3 | 3.49e-3 | 1.69e-3 | 0 | 2.81e-3 | 3.01e-3 |
| 192 | 4.58e-3 / 4.62e-3 | 5.25e-3 | 3.67e-3 | 1.59e-3 | 0 | 3.11e-3 | 3.40e-3 |

- The total quasi-steady rate rises −22% / −12% / −6% / −1.4% below the 192³ value (64 / 96
  / 128 / 160): a total within ~5% from N = 128 (R_inj = 4 cells, wind-2 stand-off ≈ 11
  cells, wind-1 ≈ 35 cells). The three-grid Richardson is undefined (non-monotone
  differences), so the converged value is not pinned better than "≈ 5.3-5.5e-3 L_w".
- Per shock it is less converged: wind 1 still +5% from 160 to 192, wind 2 −6%; the total
  is partly a cancellation. No injection at shocks into the ambient gas (the outer
  bow shocks stay below mach_min or out of the box within t_end).
- Measured vs nominal L_w now differ by ≤ 6% (64³) and ≤ 1% (192³): the fix removed the
  dominant resolution dependence of the normalisation.
- **Semi-analytic reference** (`c_expect.py`: exact CC85 upstream with the measured Ṁ, L
  on the run's own injecting TS cells, the finder's normals and area weights, δ(M_n)):
  nominal injection / L_exp = 1.56 / 1.72 / 1.63 (wind 1, 128 / 160 / 192) and 1.56 /
  1.60 / 1.51 (wind 2), N-independent within the scatter, i.e. **a ~1.5-1.6× excess that
  does not converge away**. For wind 1, part of it is the Mach over-read (finder M_n 6.22
  vs CC85 5.59 at 192³, ×1.37 through A ≈ 3.3); for wind 2 the Mach number agrees (4.44 vs
  4.56) and the excess is in the flux. Caveat: L_exp ∝ |r̂·n|³ δ(M|r̂·n|) with mean
  |r̂·n| ≈ 0.35-0.45, so a small tilt of the finder normals toward the tangent lowers L_exp
  strongly; the reference is not decisive on its own. Consistent in size with the Phase B
  CWB excess (2026-10-20, +41%, accepted).
- E_cr in the box is 0.55-0.68 of the injected energy (adiabatic losses + outflow) and
  converges more slowly (+11% from 160 to 192).

## X1: sensitivity to cell-based parameters (S-CP, N = 96, ζ = 0.1) (2026-11-05)

`sedov_study.py --cr/--mach-min/--cfl/--limiter`, `x1_analyse.py`; `pics/X1_summary.txt`,
`pics/X1_sensitivity.png`. Change vs the baseline in units of the baseline's own
resolution error (1 − x; injected: vs 224³):

| variant | M/M_t @M10 | @M3 | inj/exact @M10 | @M3 | realized/exact late | injected |
|---|---|---|---|---|---|---|
| baseline | 0.878 | 0.930 | 0.667 | 0.751 | 0.745 | 0.1049 |
| sampling max steps 8 / 25 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| area normal smoothing 1 / 3 | +0.01 / 0 | 0 / 0 | +0.03 / −0.02 | +0.03 / −0.01 | +0.02 / −0.01 | +0.08 / −0.03 |
| mach_min 2 | +0.03 | +0.02 | +0.08 | 0 | **−0.40** | −0.06 |
| CFL 0.2 | 0 | 0 | 0 | 0 | 0 | 0 |
| limiter VAN_LEER | 0 | +0.09 | −0.05 | +0.04 | +0.02 | −0.02 |

- The adaptive walk is never longer than 3 steps at 96³ (5 at 224³ S-SS), so the step cap
  is irrelevant; CFL has no effect; the area smoothing moves ≤ 3% of the error; the limiter
  ≤ 9% (weak-shock Mach). **No cell-based parameter carries the resolution error; none
  needs to scale with N.**
- mach_min = 2 removes the injection of the late S-CP (M_true → 2.5 is read as ~2.3): a
  physical threshold, to be compared with the *finder's* low reading near M ≈ 2.

## X2: old segment-argmin surface vs the continuous-normal default (2026-11-05)

`x2_old_surface.py` / `x2_cwb.py` (GPU, finder only on the kept control states),
`x2_analyse.py`; `pics/X2_old_vs_new_surface.png`, `pics/X2_summary.txt`.

- **Sedov (all three setups, all N):** M50 within ±0.01, area within ±0.01 (S-PE 32³:
  1.215 vs 1.136), D_old / D_new = 0.996-1.004 for N ≥ 96 (0.96 at 32³). Fitted orders of
  the Mach error: 0.52 / 0.66 / 0.76 (old) vs 0.48 / 0.63 / 0.74 (new). The default
  changes neither the convergence rate nor the constant on a clean spherical shock.
- **CWB (control last states):** wind 1 identical (injection within 0.1%; +6% at 64³);
  wind 2: the old surface has 3-14% more area and 3-9% more injection (64 → 192, shrinking).
  The DB10 zone-gap over-count (+70% in the finder_v2 comparison) does not appear in these
  states.

## C4: KR13 on the CWB (2026-11-05)

`cwb_study.py --kr13` at 128³ and 192³.
- **128³** (window means, vs ζ = 0.1 at 128³): injection rate / L_w 7.11e-3 vs 4.94e-3
  (×1.44); wind-1 TS ×1.80 (its oblique wings have M = 5-15, ζ_KR13 = 0.10-0.21), wind-2 TS
  ×0.69 (M50 ≈ 4.2, ζ_KR13(4) ≈ 0.05); hydro unchanged (x_c, x_TS2 within 1%), E_cr in the
  box / injected 0.59 (as ζ = 0.1). With S4's amplification (A ≈ 5-12 for 2 < M < 15) and
  the CWB's ~10-20% Mach deficit at 128³, the KR13 CWB injection is far from converged at
  128³; the 192³ partner run is needed for a convergence statement and failed:
- **192³ goes non-finite** between t/t_end = 0.10 and 0.15 (while the two termination
  shocks form and collide, M_TS ≈ 15-17, ζ_KR13 = 0.211), reproducibly at the same time with
  C_cfl = 0.4 and 0.2: not a time-step issue. ζ = 0.1 at 192³ and KR13 at 128³ run
  stably. **Open robustness finding** (logged in PROGRESS_PHASEB.md); no KR13 convergence
  statement for the CWB.

## C5: CWB Q5 rows (2026-11-05)

Stand-off distances (window mean, star to termination shock on the axis), in cells:

| N | R_inj | wind-1 stand-off | wind-2 stand-off | total injection / 192³ (window rate) | M50 wind 1 / wind 2 vs 192³ (control) |
|---|---|---|---|---|---|
| 64 | 2 | 17 | 6 | 0.78 | 0.59 / 0.50 |
| 96 | 3 | 25 | 9 | 0.88 | 0.82 / 0.74 |
| 128 | 4 | 34 | 12 | 0.94 | 0.91 / 0.91 |
| 160 | 5 | 43 | 14 | 0.99 | 0.97 / 0.96 |
| 192 | 6 | 52 | 16 | 1 | 1 / 1 |

- **The wind itself** needs R_inj ≥ 3-4 cells to reach the Chevalier & Clegg profile
  (64³, R_inj = 2: ~25% slower upstream on the axis).
- **Total CR injection within ~5%** from R_inj = 4 cells, wind-2 stand-off ≈ 12 cells, wind-1
  ≈ 34 cells (N = 128); within ~12% from R_inj = 3 (N = 96). Per shock the error is larger
  (±5-6% between 160 and 192), and the total partly cancels.
- **Shock Mach numbers per termination shock are not converged at 192³**: M50 still +3-4%
  between 160 and 192 (wind-2 stand-off 14 → 16 cells). This is consistent with the Sedov
  rule (S5: ~20 cells per radius for 10%, ~60 for 5%), with the stand-off taking the role
  of the shock radius; with the CWB's additional hydro share (C2) of the same size.
- **Two biases do not converge away** and have to be carried as systematic: the finder
  over-reads the wind-1 apex Mach number by 17-25% (N ≥ 128), and the injection is
  1.5-1.6× the exact-upstream estimate L_exp at every N (caveat: L_exp is sensitive to the
  normals; C3).
- KR13: no statement (C4).

## Gallery figures (2026-11-05)

Visual summary of the study, ζ = 0.1, final state of each run (one column per N).
`gallery_extract.py` (GPU) re-runs the finder with the DSA settings on the stored final
states and writes `RAW/gallery/{sedov,cwb}_N<n>.npz` (mid-plane slices, Sedov shell
profiles, every injecting surface cell with position / Mach / w / injection);
`gallery_plot.py` (CPU) draws:

- `pics/G1_sedov_gallery.png` (S-PE, N 32 / 64 / 128 / 224, R/dx 12-87): mid-plane ρ
  and e_cr, injecting surface in 3D (upper hemisphere) coloured by M / M_ref (two-fluid
  Sedov, ζ = 0.1, diverging around 1), and for all seven N: shell-mean ρ and e_cr
  against the exact two-fluid solution, area-weighted Mach quantiles against R/dx with
  the study fit. Median M / M_ref 0.875 / 0.910 / 0.939 / 0.948. Visible in 3D: the
  32³ surface has 20-30% low patches (face centres, triangle at the body diagonal) that
  shrink to rings by 128³; at 224³ a ~1-cell seam along the x = 0 / y = 0 planes reads
  higher than its surroundings (orientation artefact, not quantified here). e_cr
  peaks above the reference just behind the front from 128³ on (the deposit surplus of S3).
- `pics/G2_cwb_gallery.png` (N 64 / 96 / 128 / 192, R_inj 2-6 cells): orbital-plane ρ
  and e_cr, injecting cells in 3D in the view of `cwb_shocked_cells_3d_<N>.png`
  (box 0-1, default perspective, `hot`, Mach 2.5-10), for all five N:
  binary-axis ρ and e_cr, per-shock median Mach (16-84%) against R_inj in cells. Wind-1
  M50 3.6 → 6.2 and wind-2 2.3 → 4.4 at t_end, wind 1 still rising at 6 cells; the
  wind-1 apex reads ~10 at every N.
