# Task: short LaTeX report on the CR / shock-finder work

Audience: an expert in cosmic rays, CR diffusion and shock finding. They want to know
what the code does, whether it works, and what doesn't. Not a paper: max ~12 pages
including figures (hard cap 15), excluding bibliography.

## Deliverable
- New LaTeX project in `reports/cr_status_report/` (main.tex, sections/, figures/, refs.bib,
  README.md). Build with `~/bin/tectonic -X compile main.tex`; it must compile without errors.
- Produce a complete first draft, not just an outline.
- Text and figures from `reports/shock_mach_summary/` may be reused or condensed.

## State to report
- Describe the code at the current HEAD of `feature/cosmic-rays-grey` (put the commit hash on
  the title page). If uncommitted changes matter, say so explicitly.
- "Successful runs" = the current defaults. State the scheme for each topic (Riemann solver,
  CR limiter, time integrator, guard settings). Mention non-default variants only when they
  explain a result.
- Sources of truth, in priority order: `PROGRESS.md` (top section), `PROGRESS_PHASEA.md`,
  `DESIGN.md`, `pytests/shock_finder3D/FIXES_TODO.md`, `pytests/cosmic_rays_grey/baselines.json`.
  Check numbers against the code/test output where feasible; if sources disagree, flag it
  rather than picking one silently.

## Structure
0. Executive summary (<=1 page): one table, one row per topic/test:
   status (works / partial / broken / open decision) + one-line reason.
Then, for each topic below:
1. Method: what is implemented, with the governing equations as discretised in the code.
   Give each equation a code reference (file:function). Name the scheme, integrator and
   limiter, and cite the literature it follows.
2. Figures: at most 3 per topic. Each caption states the script, resolution and what to look at.
3. Result: a table of metric | measured | expected (analytic or literature) | tolerance | verdict,
   then a short list of shortcomings, known limitations and open decisions.

## Topics and scope
1. 3D shock finder: algorithm (Pfrommer 2017 / Schaal & Springel 2015), the adaptive
   sampling fix, Sedov validation incl. the convergence sweep, the Mach underestimate at low
   resolution, and the CWB verdict (brief).
2. Phase A of the grey-CR plan = test-plan ladder items 1, 2, 3, 4, 6 plus the cross-cutting
   items (stage-wise coupling, HLLC/hybrid entropy D2, limiter choice, ETD2RK F_cr relaxation,
   guard). Excludes streaming (item 5): mention in one sentence that it is out of scope.
3. DSA: ladder items 7-8 (injection at shocks, Mach dependence of efficiency) and 9 (modified
   shock).
- Out of scope, do not include at all: ladder items 10-15, Phase D, and the M5-M7
  production runs.

## Figures
- Use existing PNGs from `pytests/*/pics/` and `pytests/shock_finder3D/figures/` if they were
  made with the current defaults. Otherwise list the stale ones and ask before re-running
  (GPU only, see conventions).

## Rules
- Real references only, as BibTeX entries checked against ADS. Mark any you cannot verify with
  \textcolor{red}{[unverified]}.
- Separate what a test verified from what is expected but untested.
- Units: cgs; notation consistent across sections (define symbols once).
- Report open decisions as open; don't resolve them in the text.
