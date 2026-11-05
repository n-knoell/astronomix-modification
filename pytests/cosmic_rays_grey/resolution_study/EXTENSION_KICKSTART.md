# Kickstart: extend the resolution study to higher N on the H200s (all existing setups)

Revised 2026-11-06. Hand this file to a fresh session on the H200 node (compgpu12, 8× H200, 141 GB,
shared with other users).

**Task:** add finer grids to the existing ladders, for every setup of the study, advective and with CR
diffusion. Run them, then re-run the analyses. Do not redo the existing runs and do not change
physics code. If something fails or looks wrong, stop and report it to the user; do not fix it on
your own.

## Read first

1. `RESULTS.md`:
   - the summary and Q5 table at the top (advective study);
   - section **"Extension D"**, subsection **D4** (the local summary of the diffusion runs, and its
     "Input for D5" list).
2. `PLAN.md`:
   - Sections 3-5 (setups, metrics, steps S0-S5 / C0-C5 / X1-X2);
   - **Section 9** (extension D: λ = κ / (v_sh R), the setups W / S-SS / S-CP, `v_red` rule, box margin).
3. `../../../astronomix/_modules/_cosmic_rays_grey/PROGRESS_PHASEB.md`: the open DB items **DB12**
   (CWB non-finite at 192³ around t/t_end 0.10-0.15) and **DB13** (finder over-reads S-SS with a CR
   precursor).

## Before the first job (must do)

- **Code state.** The runs depend on files that were uncommitted on 2026-11-06:
  - `resolution_study/`: `common.py`, `setups.py`, `sedov_study.py`, `cwb_study.py`, and the new
    `wind_study.py`, `d0_sizing.py`, `D0_sizing.json`, `d_analyse.py`, `d_plots.py`,
    `gallery_d_extract.py`, `gallery_d_plot.py`, `launcher/mk_d_jobs.py`, `launcher/mk_h200_jobs.py`.
  - Check `git status`. If they are still uncommitted, ask the user to commit before launching,
    so that every output's `stamp_git_head` / `stamp_git_diff_sha1` points at a recorded state.
  - Without `--kappa` / `--vred` the advective code paths are unchanged.
- **Same file systems.** `/export/home/nknoell` (compnfs1) and `/export/scratch/nknoell` (compnfs6)
  are NFS-shared with compgpu4, so code, raw data, job scripts and logs are the same files. **Home is
  97% full (3.3 GB free on 2026-11-06):** nothing large goes to home. Raw output and logs go to
  `RAW = /export/scratch/nknoell/resolution_study/` only; figures to `pics/` are small.
- **Backend.** All builders pin `backend=NATIVE_JAX`: `setups.py` (all three) and
  `wind_study.build_wind`. On CC ≥ 8.0, `OPTIMAL_BACKEND` would otherwise pick PALLAS. The FV Pallas
  kernel is off for HLLC / grey CRs anyway. Don't change this.
- **GPU rules (user decisions):**
  - at most **2 H200s at once**;
  - only GPUs with no compute process of anyone;
  - pin by UUID (`common.require_pinned_gpu`);
  - `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` (set in every job script);
  - everything runs on GPUs, nothing on CPU queues except numpy / matplotlib analysis.

## Mechanics

- **Job scripts and queues.** They are written by `python launcher/mk_h200_jobs.py` (already run on
  2026-11-06; rerun only if you change something). This writes `RAW/jobs/H_*.sh`, the queue files
  `RAW/jobs/queue_h200_{0,1,2,3}.txt`, and the cost table `pics/H200_queue_estimate.txt`. Phase 0 also
  reuses the existing `E_check_*` / `E_probe_*` scripts.
- **Dispatcher:**

      setsid nohup python launcher/autoqueue.py RAW/jobs/queue_h200_<k>.txt --max 2 > RAW/logs/autoqueue_h200_<k>.log 2>&1 &

  `autocvd(interval=5)` picks a free GPU. The dispatcher waits until the job's python holds that GPU,
  checks the PID → UUID pin, and kills and requeues the job if the GPU is wrong or shared.
  - Events: `RAW/logs/queue_events.txt`. Per-job logs: `RAW/logs/<tag>.log`. Exit codes:
    `RAW/logs/done/<tag>`.
  - The queue file is re-read every loop, so tags can be appended while it runs.
  - Start a phase only when the previous one has been checked.
- **Output tree** (the analysis scripts glob it, so keep it):
  - `sedov/{pe,ss,cp_p0.6}/N<n>/{control,z0.1,kr13}.npz`;
  - `sedov/{ss,cp_p0.6}/N<n>/{z0.1,kr13}_lam<λ>.npz`;
  - `cwb/N<n>/{control,z0.1,kr13}.npz`;
  - `wind/N<n>/{adv,lam<λ>}.npz`.
- **Cost.** `pics/H200_queue_estimate.txt` gives 2080 Ti hours, scaled ∝ N⁴ from the finest existing
  run of the same kind. Totals: phase 1 ~151 h, phase 2 ~181 h, phase 3 ~43 h. Peak memory is about
  6.8 GB × (N/192)³: ~16 GB at 256, ~31 at 320, ~54 at 384, ~86 at 448, ~129 at 512. Divide the hours
  by the H200 speed-up measured in phase 0. With 2 GPUs, expect a few days of wall time for phases 1-2.

## Phase 0: checks and probes (`queue_h200_0.txt`)

| tag | what | pass if |
|---|---|---|
| `E_check_sedov` | S-PE 128³ ζ 0.1 → `sedov/pe/N128/h200check_z0.1.npz` | M50(t), injected, E_cr(t) match `sedov/pe/N128/z0.1.npz` (2080 Ti) to ≲ 1e-3 relative |
| `E_check_cwb` | CWB 128³ ζ 0.1 → `cwb/N128/h200check_z0.1.npz` | same vs `cwb/N128/z0.1.npz` (per-shock M50, injection, E_cr) |
| `H_check_wind` | W 128³ λ 0.1 → `wind/N128/h200check_lam0.1.npz` | same vs `wind/N128/lam0.1.npz` (E_cr / injected, M50_fs, precursor) |
| `H_check_cp_lam` | S-CP 128³ ζ 0.1 λ 0.1 → `sedov/cp_p0.6/N128/h200check_z0.1_lam0.1.npz` | same vs `.../z0.1_lam0.1.npz` |
| `E_probe_sedov` | `step0_probe.py memory sedov 256 / 384 / 448 / 512` → `RAW/step0/memory_sedov_N<n>.json` | records peak memory and s/step, OOM recorded, not raised |
| `E_probe_cwb` | `step0_probe.py memory cwb 256 / 320 / 384` | same |

- The 2080 Ti probe results for 256³ were renamed `*_2080ti.json`. Don't overwrite them.
- From the checks' wall time over the old wall time, get the **H200 speed-up**. Write it with the
  check results into `RESULTS.md` (new section "H200 extension: phase 0").
- **512³ decision:** only if the sedov 512 probe peaks below ~120 GB. Otherwise drop phase 3's 512 jobs.
- **If any check differs by more than ~1e-3, stop and report.** Different GPUs reorder floating-point
  sums, so expect ~1e-5 to 1e-4.

## Phase 1: advective / hydro ladders (`queue_h200_1.txt`, 27 jobs, N ascending)

Same flags as the original study (`--snapshots 30`; control runs keep states 9 14 19 24 29 for the
S2 restriction, DSA runs keep 29):

- S-PE (`pe`): control + ζ 0.1 at 256, 384, 448.
- S-SS (`ss`): control + ζ 0.1 at 256, 384. These are also the λ = 0 anchors of phase 2.
- S-CP (`cp`, p₀ 0.6): control + ζ 0.1 + KR13 at 256, 384, 448. These are also the λ = 0 anchors.
- CWB: control + ζ 0.1 at 256, 320, 384 (N multiple of 32; R_inj = N/32 cells).
- W advective (`wind_study.py --vred 0`, name `adv`) at 256, 320: the anchors of the W diffusion series.

The r = 2 chains for observed orders are 96-192-384, 128-256 (-512), 64-128-256 (CWB), and 160-320 (CWB, W).

## Phase 2: CR diffusion, extension D step D5 (`queue_h200_2.txt`, 18 jobs)

κ and `v_red` come from `D0_sizing.json` (λ fixed per setup; do not resize). λ 0.03 / 0.1 / 0.3 is
the precursor length κ / v_sh per shock radius.

- **W** (wind bubble): λ 0.03 / 0.1 / 0.3 at 256, 320.
  - Answers whether λ 0.03 resolves its precursor (needs κ / v_sh ≳ 3 cells, i.e. R/dx ≳ 100, N ≈ 300).
  - Answers whether E_cr / injected converges at λ 0.03 and 0.3 (still rising at 192³: 1.31 / 1.50).
- **S-CP** ζ 0.1 and KR13, λ 0.03 / 0.1 at 256, 384: E_cr / (E_cr0 + injected) was not converged by
  192³; KR13 is the most Mach-sensitive.
- **S-SS** ζ 0.1, λ 0.03 / 0.1 at 256, 384: **also the DB13 probe.** Does the finder's over-read at
  M ≈ 250 with a precursor (+15% at 192³ λ 0.03, grid-following patches) grow, saturate or vanish with N?
- Not in the queue: λ 0.3 for Sedov (it stops at R 0.26 by the box margin; converged enough locally),
  and the **CWB with diffusion (excluded by the user 2026-11-06**: non-finite at 192³ for all λ, DB12).

## Phase 3: optional (`queue_h200_3.txt`)

- S-PE / S-CP ζ 0.1 at 512³: only after the phase-0 memory decision.
- CWB KR13 at 192 and 256 (DB12: does the non-finite state at t/t_end 0.10-0.15 recur at 256³?). Record
  the outcome, don't debug.

## Analysis after the runs (CPU)

Most loaders glob `N*`. Update the hard-coded resolution lists:

| script | what to update |
|---|---|
| `s1_analyse.py` | `CHAINS` (add 96-192-384, 128-256, 224-448) |
| `gallery_extract.py` | `SEDOV_NS`, `CWB_NS` |
| `gallery_plot.py` | `SEDOV_SHOW` / `CWB_SHOW` (pick 4 columns) |
| `d_plots.py` | `N_ALPHA` (add 256 / 320 / 384) |
| `gallery_d_plot.py` | `N` (the gallery N; re-extract first with `gallery_d_extract.py --n 384` and `--n 320`, GPU) |
| `d0_sizing.py` | do not rerun (it would change κ) |

Then rerun:
- `s1_analyse.py` … `s5_summary.py`, `c_analyse.py`, `gallery_extract.py` (GPU) + `gallery_plot.py`;
- `d_analyse.py` (all), `d_plots.py`, `gallery_d_extract.py` (GPU) + `gallery_d_plot.py`.

Write the new numbers into `RESULTS.md`:
- new section "H200 extension";
- update the Q5 table and the D4 table only where the new grids change them, and say so;
- add a log entry in `PROGRESS_PHASEB.md`.

The user commits; you don't.

## Questions the new grids should answer

- **Advective.**
  - Does `|ΔM/M| ≈ 0.55 (R/dx)^-0.6` hold beyond R/dx ≈ 90?
  - Does strong-shock injection reach 5% (extrapolated ~300 cells per radius)?
  - KR13 moderate-Mach injection: "hundreds of cells", really?
- **CWB.** The wind-1 median Mach still rises at R_inj = 6 cells (3.6 → 6.2). The apex over-read
  (+17-25%) and the injection at 1.5-1.6× the exact-upstream estimate did not converge by 192³. Do
  they converge, or are they systematic?
- **Diffusion (D5).**
  - Precursor rule κ / v_sh ≳ 3 cells at λ 0.03.
  - Convergence of E_cr / injected (W λ 0.03 / 0.3) and of the Sedov CR energy budget.
  - DB13 trend with N.
- **DB12** at 256³ (phase 3).

## Out of scope

- CWB with CR diffusion (excluded).
- CR streaming.
- Fixing DB12 / DB13 (report only).
- Any physics-code change.
