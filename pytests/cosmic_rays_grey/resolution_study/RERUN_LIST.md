# Resolution study: runs to redo after the changes of 2026-11-16

Two library changes of 2026-11-16 alter what some of the stored runs would give today:

- **A, entropy fix** (DB13, committed in 5febc9c): the hybrid CR-entropy transfer no longer
  drains gas heat in a CR precursor. It acts wherever CRs sit ahead of a shock, i.e. in runs
  with CR diffusion.
- **B, finder option** (`dsa_sampling_at_shock`, default on, uncommitted): the upstream state
  is read at the shock instead of 2-4 cells ahead, and the pre-shock walk no longer runs down
  background gradients. It acts wherever the gas ahead of a shock is not uniform: winds
  (r^-2 upstream of a termination shock) and, weakly, CR precursors.

Details and measurements: `PROGRESS_PHASEB.md`, log entries "2026-11-16" (DB13) and
"2026-11-16 (finder)". Raw tree: `RAW = /export/scratch/nknoell/resolution_study`.
Job scripts: `RAW/jobs/`. Times are the measured wall times of the stored runs on one GPU
(2080 Ti for N <= 192, H200 for N >= 256; some H200 times include the host-RAM stalls of
2026-11-13).

## Status (2026-11-17)

Local part done with code e15df58 (B committed there): all N <= 192 reruns of W (23), CWB
(9: zeta 0.1, controls, KR13 128) and S-SS diffusion (16), 2026-11-16 23:33 - 2026-11-17 04:15
on 8 local 2080 Ti, all rc 0 (logs `RAW/logs/<tag>.log`, old logs `RAW/logs/pre20261116/`,
queue `RAW/logs/rerun20261116_queue.txt`). Old states in `RAW/{wind,cwb}_pre20261116/`,
`RAW/sedov/{ss,cp_p0.6}_pre20261116/`. S-CP probe: keep (Sec. 4). Quick comparisons:
`RAW/w_rerun_compare.py`, `RAW/cp_probe_compare.py`. Open: the H200 runs (N >= 256) on
compgpu12, optional DB12 retries, then the re-analysis (Sec. 6).

H200 part started 2026-11-17 10:32 on compgpu12 (code e15df58): `launcher/gpu_lanes.py --max 2`
on `RAW/jobs/queue_rerun20261116_h200.txt` (18 jobs: S-SS 4, CWB 2, W 8, then the DB12
probes 4; `C4_192` left out = same run as `H_cwb_kr13_N192`). Old done markers in
`RAW/logs/done_pre20261116/`, old logs in `RAW/logs/pre20261116/`, events in
`RAW/logs/queue_events.txt`, dispatcher log `RAW/logs/gpu_lanes_rerun20261116.log`.
Re-analysis chained after it (`launcher/rerun_reanalysis_chain.sh`, pid 3068503, log
`RAW/logs/rerun20261116_reanalysis.log`): aborts if a core rerun failed but left an output
file; GPU job `R_rerun20261116_gpu` (gallery extracts, `c2_restrict`, `x2_cwb`, `c_expect`),
then old figures -> `RAW/pics_pre_rerun20261116/`, then the CPU scripts s1-s5, c_analyse,
d_analyse, d_plots, gallery_plot, gallery_d_plot; marker `logs/done/R_rerun20261116_reanalysis_all`.
Still by hand: the RESULTS.md / PROGRESS_PHASEB.md text, the C3 mass-flux reference.

## Summary

| Group | Runs | Why | Verdict | GPU time |
|---|---|---|---|---|
| Wind bubble W, all DSA runs | 28 + 3 variants | B (A: -0.2%) | **rerun** | 21.7 h (2080 Ti) + 18.7 h (H200) |
| CWB, zeta 0.1 / KR13 / control | 6 + 1 + 6 | B, A | **rerun** | 2.2 h (2080 Ti) + 0.5 h (H200) |
| Sedov S-SS with diffusion | 15 + 1 variant + 3 (H200) + 1 failed | A | **rerun** | 1.6 h (2080 Ti) + ~4 h (H200) |
| Sedov S-CP with diffusion | 30 + 1 variant + 8 (H200) | A, B (not measured) | **probe first** (2 runs, 0.4 h) | 5.8 h (2080 Ti) + 13.7 h (H200) if needed |
| Sedov advective (S-PE, S-SS, S-CP), all N | ~75 | none | **keep** | 0 |
| One-off probes (S0, X1, X2, DB11, step0) | | none / historical | **keep** | 0 |

Evidence for each verdict is in the sections below. The numbers come from the regression
tests of the same setups (old value -> with A -> with A and B) and from the finder run on
stored states.

## Before starting

1. Decide and commit the code state the reruns use. B is uncommitted. DB14 (CWB apex Mach
   number, mass-flux deficit) is open; if it leads to another finder change, the CWB and W
   runs would have to be done again. Running the Sedov groups first avoids that risk.
2. The run scripts overwrite `RAW/<setup>/N<n>/<name>.npz`. Keep the old files, e.g.
   `cp -a RAW/wind RAW/wind_pre20261116` (same for `cwb`, `sedov/ss`, `sedov/cp_p0.6`); they
   are the "before" reference for every comparison below.
3. N <= 192 on the local 2080 Ti nodes (`RAW/launch.sh`, `RAW/dispatch.sh`), N >= 256 on
   compgpu12 (`launcher/gpu_lanes.py --max 2`, NATIVE_JAX, `--gpu-diag` as in the `H_*.sh`
   scripts of 2026-11-13).

## 1. Wind bubble W: rerun all DSA runs

Evidence (regression tests `cr_wind_bubble`, `cr_wind_bubble_diffusion`, 128^3):

| quantity | before | with A | with A and B |
|---|---|---|---|
| advective: E_cr / (L_w t_end) | 0.0675 | 0.0675 | 0.0625 (-7.4%) |
| lambda ~0.3: E_cr / (L_w t_end) | 0.1020 | 0.1018 | 0.0900 (-11.6%) |
| E_cr diffusive / advective | 1.512 | 1.508 | 1.441 |

The forward shock (uniform ambient gas) is unchanged; the change is the injection at the
termination shock and in the interior, where the old upstream sample lay towards the star.
The W results that depend on it: E_cr / injected (RESULTS.md D1, 1.31 / 1.34 / 1.50 vs 0.92),
the total-injection convergence, the "4-11% more injection with diffusion" statement.

| Jobs | N | Scripts | Time |
|---|---|---|---|
| adv | 64, 96, 128, 160, 192 | `D_w_adv_N*.sh` | 1.8 h |
| lambda 0.03 | 64, 96, 128, 160, 192 | `D_w_lam0.03_N*.sh` | 6.1 h |
| lambda 0.1 | 64, 96, 128, 160, 192 | `D_w_lam0.1_N*.sh` | 6.1 h |
| lambda 0.3 | 64, 96, 128, 160, 192 | `D_w_lam0.3_N*.sh` | 6.2 h |
| variants: 2 x v_red at 96; injection off after t_end / 2 at 128 (adv, lambda 0.1) | 96, 128 | `D_w_lam0.1_vx2_N96.sh`, `D_w_adv_off05_N128.sh`, `D_w_lam0.1_off05_N128.sh` | 1.4 h |
| adv, lambda 0.03 / 0.1 / 0.3 (H200) | 256, 320 | `H_w_adv_N*.sh`, `H_w_lam*_N*.sh` | 18.7 h |

Not needed again: `h200check_lam0.1` (overlap check of the two machines; redo only if the
H200 runs are redone on a different software stack).

## 2. CWB: rerun the constant-zeta ladder, KR13 and the controls

Evidence (finder on the stored final states, option off -> on):

| quantity (wind 1 / wind 2) | 128^3 | 192^3 |
|---|---|---|
| dissipated flux at the termination shocks | 0.93 / 0.74 | 0.94 / 0.78 |
| upstream density / exact (median) | 1.000 / 1.003 -> 0.963 / 0.957 | 1.009 / 1.032 -> 0.995 / 1.003 |
| mass flux / (Mdot Omega / 4 pi) | 0.80 / 1.01 -> 0.76 / 0.82 | 0.84 / 1.03 -> 0.82 / 0.88 |
| apex Mach / Chevalier & Clegg | 1.33 / 0.87 -> 1.38 / 0.97 | 1.11 / 1.08 -> 1.10 / 1.13 |

These are static re-readings; a live run also feeds the lower injection back into the flow.
The CWB runs use kappa = 1 (a thin precursor), so A acts as well (size not measured).

| Jobs | N | Scripts | Time |
|---|---|---|---|
| zeta 0.1 | 64, 96, 128, 160, 192 | `C_dsa_64_96.sh`, `C_dsa_128.sh`, `C_dsa_160.sh`, `C_dsa_192.sh` | 1.0 h |
| control (zeta 0) | 64, 96, 128, 160, 192 | `C_ctrl_*.sh` | 1.0 h |
| KR13 | 128 | `C4_128.sh` | 0.15 h |
| zeta 0.1, control (H200) | 256 | `H_cwb_z0.1_N256.sh`, `H_cwb_control_N256.sh` | 0.5 h |

The controls inject nothing, so their flow does not change; they are listed because their
per-snapshot finder diagnostics (Mach number, dissipated flux) were written at run time with
the old finder. If only the final state matters, re-analysing the stored controls is enough
(`control.npz` keeps the last state), also for the 320^3 / 384^3 controls.

Optional, as a DB12 probe with both changes in: the runs that went non-finite
(`H_cwb_z0.1_N320.sh`, `H_cwb_z0.1_N384.sh`, `H_cwb_kr13_N192.sh`, `H_cwb_kr13_N256.sh`,
`C4_192.sh`; 0.1-0.5 h each until they fail). Whether A or B changes them is not known.

Not needed: the CWB diffusion series `D_cwb_lam*` (excluded from the study on 2026-11-06)
and `E_check_cwb` / `h200check_z0.1`. `c2_restrict.py` (C2) and `x2_cwb.py` (X2) are analyses
of the control last states, not runs; they are recomputed in the re-analysis (Section 6).

## 3. Sedov S-SS with diffusion: rerun

Evidence (full reruns with A, lambda 0.03): the finder Mach number of the stored runs is
biased high by the drained precursor (M50 / M_true at the end 1.10 at 192^3; 0.94 with A,
equal to the advective run within 0.3%); E_cr and the injected energy are unchanged (0.1952
vs 0.1953). B changes the corrected run by -0.1% in M50 (128^3). Everything in RESULTS.md D2 /
D4 about the S-SS Mach number with diffusion (DB13) rests on the stored runs. The failed
384^3 lambda 0.03 run (M84 growing from snapshot 1 on) looked like the same effect.

| Jobs | N | Scripts | Time |
|---|---|---|---|
| lambda 0.03 | 48, 64, 96, 128, 192 | `D_ss_z0.1_lam0.03_N*.sh` | 0.6 h |
| lambda 0.1 | 48, 64, 96, 128, 192 | `D_ss_z0.1_lam0.1_N*.sh` | 0.6 h |
| lambda 0.3 | 48, 64, 96, 128, 192 | `D_ss_z0.1_lam0.3_N*.sh` | 0.3 h |
| 2 x v_red | 96 | `D_ss_z0.1_lam0.1_vx2_N96.sh` | 0.1 h |
| lambda 0.03, 0.1 (H200) | 256 | `H_ss_z0.1_lam0.03_N256.sh`, `H_ss_z0.1_lam0.1_N256.sh` | ~1.5 h |
| lambda 0.1 (H200) | 384 | `H_ss_z0.1_lam0.1_N384.sh` | 1.2 h |
| lambda 0.03 (H200), failed before | 384 | `H_ss_z0.1_lam0.03_N384.sh` | ~1.2 h |

Already available with A (not B): `review_20261115/drain0/raw_study/sedov/ss/N128/z0.1_lam0.03_repo.npz`
and `N192/z0.1_lam0.03_H9.npz` under `/export/scratch/nknoell/`; they can serve as a
cross-check of the first two reruns.

## 4. Sedov S-CP with diffusion: probe first

Not measured. A should act less than in S-SS: the precursor's CR pressure is of the order
of the gas pressure here (p_0 = 0.6, M 2.5-30), not 1e3 times it. B acts through the
precursor's density gradient (S-SS: -0.1% in Mach). KR13 amplifies Mach changes at M 2.5-4
(planar: 0.1% in Mach is 1% in efficiency).

Probe (0.4 h): `D_cp_z0.1_lam0.1_N128.sh` and `D_cp_kr13_lam0.1_N128.sh`, compared with the
stored files (finder M50 over time, injected energy, E_cr at the end). Rerun the group if any
of them moves by more than the study's 128^3 -> 192^3 difference for that quantity; keep it
otherwise and note the measured difference in RESULTS.md.

**Probe done 2026-11-17: keep.** All four quantities move by less than the 128³ → 192³ step
(largest: KR13 M50 1.5% vs 5.7%); numbers in RESULTS.md D2. The probe runs are in
`RAW/sedov/cp_probe_20261117/N128/`; `RAW/sedov/cp_p0.6/` keeps the stored files (one code
version per ladder).

| Jobs (only if the probe says so) | N | Scripts | Time |
|---|---|---|---|
| zeta 0.1: lambda 0.03, 0.1, 0.3 | 48, 64, 96, 128, 192 | `D_cp_z0.1_lam*_N*.sh` | 2.9 h |
| KR13: lambda 0.03, 0.1, 0.3 | 48, 64, 96, 128, 192 | `D_cp_kr13_lam*_N*.sh` | 2.9 h |
| 2 x v_red | 96 | `D_cp_z0.1_lam0.1_vx2_N96.sh` | 0.1 h |
| zeta 0.1 and KR13: lambda 0.03, 0.1 (H200) | 256, 384 | `H_cp_z0.1_lam*_N*.sh`, `H_cp_kr13_lam*_N*.sh` | 13.7 h |

## 5. Keep: Sedov advective runs

S-PE (control, zeta 0.1; 32-512), S-SS (control, zeta 0.1; 32-384), S-CP (control, zeta 0.1,
KR13; 48-512), the S-CP 96^3 variants `x1_*`, `S2_control`, `X2_old_surface`, `z0.1_vred0`.

Evidence: the regression tests of the same setups moved by at most 5e-4 with A and 5e-3 with
A and B (`cr_sedov_taylor` = S-PE at 48 / 96 / 160: 2e-4; `cr_sedov_self_similar` = S-SS at
48 / 96: 5e-4 with A, up to 4.5e-3 with B, only at zeta 0.211 and 48^3; `cr_dsa_mach_dependence`
incl. KR13 and counter-pressure blasts: 7e-4; `cr_sedov_injection_rate`: 4e-4). The finder
on the stored S-SS states reads the same Mach quantiles to four digits with B. The S-PE runs
have CRs ahead of the shock late in the run (placeholder v_red = 1, the DB11 caveat already
in RESULTS.md); `cr_sedov_taylor` covers that phase and did not move.

## 6. After the reruns

- Full re-analysis as on 2026-11-15 (`launcher/reanalysis_watch.sh`: `gallery_extract`,
  `gallery_d_extract`, then `s1`-`s5`, `c_analyse`, `d_analyse`, `d_plots`, `gallery_plot`,
  `gallery_d_plot`). The analysis scripts call the finder with the DSA options of the
  current code, so they must not be run on a mix of old and new CWB / W states.
- CWB only: also `c2_restrict.py` (C2) and `x2_cwb.py` (X2) on the new control states.
- RESULTS.md sections to rewrite: C1-C5 and the CWB rows of the summary (Q1-Q5), X2 (CWB
  part), D1 (W), D2 / D4 for S-SS (and S-CP if rerun); the CWB, wind and S-SS diffusion
  figures and the galleries G2-G4 (G5, G6 if S-CP is rerun). The expected-luminosity
  reference of C3 (`c_expect.py`, along the finder normal) is the quantity review item 3
  found unreliable; replace it with the mass-flux closure when C3 is redone.
- PROGRESS_PHASEB.md: close DB13 for the study (S-SS statements), update DB12 if the
  non-finite runs were retried, DB14 with the live CWB numbers.
