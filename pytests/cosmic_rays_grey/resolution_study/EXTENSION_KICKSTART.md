# Kickstart: extend the resolution study to higher N (H200)

Read first: `RESULTS.md` (summary + Q5 table at the top), `PLAN.md` (steps S0-S5, C0-C5, X1-X2).
Do not redo the existing study. Add finer grids to the same ladders and re-fit.

## State to carry over
- Code: branch `feature/cosmic-rays-grey`, HEAD `c70df4d` + **uncommitted** changes the runs
  depend on: `_stellar_wind/stellar_wind.py` (`_wind_ei3D` Mdot/L fix, C0),
  `pytests/shock_finder3D/_cwb_setup.py` (`build_cwb` split), and this directory (untracked).
  Commit or copy these first, or the new runs will not match the old ones.
- Raw data: `/export/scratch/nknoell/resolution_study/` (18 GB). The analysis scripts read
  it via `RAW` in `common.py` (env `RES_STUDY_RAW`); `analyse.py` hardcodes the path.
  New runs must land in the same tree (`sedov/<setup>/N<n>/<name>.npz`, `cwb/N<n>/<name>.npz`).
- Old grids: Sedov N = 32, 48, 64, 96, 128, 192, 224 (setups `pe`, `ss`, `cp_p0.6`);
  CWB N = 64, 96, 128, 160, 192 (`control`, `z0.1`; `kr13` only at 128).
  Limit on the 11 GB RTX 2080 Ti: Sedov 256³ and CWB 224³ did not fit.

## Must check before production
1. **Backend.** On CC >= 8.0 GPUs `OPTIMAL_BACKEND` resolves to **PALLAS**; every old run
   used **NATIVE_JAX**. Either pin `SimulationConfig(backend_config=BackendConfig(backend=NATIVE_JAX))`
   in `setups.py` (and `_cwb_setup.build_cwb`), or rerun one overlap grid (e.g. Sedov `pe`
   128 z0.1, CWB 128 z0.1) and show it matches the old file (M50, E_cr, injection) before
   mixing data.
2. Re-run the Step-0 probe for the new N: `python step0_probe.py memory sedov|cwb <N>`
   (peak memory, s/step, wall-time estimate; OOM recorded, not raised). Rough scaling from
   the old peaks (∝ N³): Sedov 224³ 8.3 GB → 448³ ~66 GB, 512³ ~100 GB; CWB 192³ 7.2 GB →
   384³ ~58 GB. H200 has 141 GB.
3. Keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` and pin GPUs by UUID (`require_pinned_gpu`).
   The launcher scripts live in `launcher/` and expect `jobs/<tag>.sh`.

## Suggested extension
- Sedov: add 256, 384, 448 (512 if it fits) for `pe` z0.1 + control, and `cp_p0.6` z0.1 / kr13
  (the KR13 moderate-Mach rows extrapolate to "hundreds of cells" today; these grids test it).
  Usage: `python sedov_study.py --setup pe --n 384 --zeta 0.1` (`--kr13`, `--setup cp`).
  Keep r = 2 chains (e.g. 96-192-384, 128-256-512) for observed orders.
- CWB: add 256, 320, 384 for `control` + `z0.1`, and retry `kr13` at 192/256.
  Usage: `python cwb_study.py --n 256 --zeta 0.1` (`--kr13`).
- Then rerun the analysis scripts (`s1_analyse.py` … `s5_summary.py`, `c_analyse.py`,
  `gallery_extract.py` + `gallery_plot.py` with the new N in `SEDOV_NS` / `CWB_NS`).

## Open questions the new grids should answer
- Does `|ΔM/M| ≈ 0.55 (R/dx)^-0.6` hold beyond R/dx ≈ 90? Does strong-shock injection reach
  5% (extrapolated ~300 cells per radius)?
- CWB: wind-1 median Mach still rises at R_inj = 6 cells (3.6 → 6.2); the apex over-read
  (+17-25%) and the injection at 1.5-1.6× the exact-upstream estimate did not converge by 192³.
  Do they converge, or are they systematic?
- DB12: KR13 CWB 192³ goes non-finite at t/t_end 0.10-0.15 (also at C_cfl 0.2). Check whether
  it recurs at 256³ and 384³.
- Out of scope here: DB11 (`cr_wind_bubble` v_red, user decision open), CR streaming.
