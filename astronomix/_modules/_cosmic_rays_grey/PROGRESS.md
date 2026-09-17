# Grey two-moment cosmic rays -- PROGRESS

Status tracker for `pytests/shock_finder3D/astronomix_CR_implementation_plan.md`. Read this
first when picking the work back up; `DESIGN.md` in this directory is the target design, this
file is what's actually done against it.

## Where things stand (2026-09-17, later: ladder item 14 -- synchrotron + IC SED vs. naima, DONE)

**Item 14 turned out not to actually be blocked by the plan's "electron treatment" design
decision (Sec. 6) after all** -- checked explicitly before starting, same scoping logic as item
13: that decision only gates *deriving* an electron population from the CR-grey/proton simulation
state, not validating the emission formula against a directly-specified, known one (item 14's own
literal wording: "for a known electron population"). Proceeded without it, deferring it (and
line-of-sight integration to a map) to item 15, same as item 13 did for the analogous proton-side
questions.

**New module**: `astronomix/_modules/_cosmic_rays_grey/cr_grey_emission_leptonic.py` --
`GreyElectronSpectrum` (mirrors `cr_grey_emission.GreyProtonSpectrum` exactly) plus JAX ports of
`naima.radiative.Synchrotron` (Aharonian, Kelner & Prosekin 2010) and `naima.radiative.
InverseCompton`'s isotropic-thermal-seed case (Khangulyan, Aharonian & Kelner 2014) -- only the
isotropic-thermal-seed IC case, covering naima's CMB/FIR/NIR presets plus custom diluted
gray-bodies; anisotropic/tabulated seeds deferred.

**New pytest**: `pytests/cosmic_rays_grey/cr_synchrotron_ic_emission.py`. Validated against naima
for the naima-tutorial-style ECPL electron population (amplitude `1e33`/eV at `E_0=1` TeV,
`alpha=2.0`, `e_cutoff=100` TeV), across 60 photon energies log-spaced 1 ueV-1 TeV, restricted to
energies within `1e-30` of each spectrum's own peak (both fall off exponentially past cutoff;
comparing floating-point noise `>100` orders of magnitude below the peak isn't meaningful): **max
relative error `4.7e-14`** (synchrotron, 100 uG) and **`3.7e-8`** (IC on CMB) -- both comfortably
inside the `1e-6` tolerance. Re-verified across several field strengths/spectral shapes and all
three seed presets plus a diluted 20 K gray body: every case `<4e-8`. Differentiability smoke
check (both channels): finite, nonzero gradients throughout.

**Two real bugs caught, both the same underlying mistake (a positivity floor sized for the wrong
quantity's scale) manifesting two different ways:**
1. **Synchrotron's forward values were wrong by ~15-20 orders of magnitude, silently (no crash,
   no NaN) -- both curves looked SED-shaped, just at the wrong scale.** `jnp.maximum(x, _TINY)`
   with this module's shared `_TINY=1e-30` (sized for `cr_grey_emission.py`'s dimensionless
   quantities) clobbered `cs1_1`/`e_c_erg`, two CGS-erg-scale quantities that legitimately sit at
   `~1e-17` to `~1e-51`. Fixed by flooring `B` itself (the only genuinely-zero-able input) once,
   at the top, instead of the derived erg-scale products.
2. **Inverse Compton's forward values were already correct (`~1e-14` agreement) but `jax.grad`
   gave `NaN`.** `1.0 - (1.0 - 1e-30)` rounds to exactly `1.0` in float64 -- the intended clip was
   silently a no-op, so `1-z` could hit an exact `0.0` in the unphysical (masked-to-zero-output)
   region, and `jnp.where` differentiating both branches let `0 * NaN = NaN` contaminate the whole
   gradient despite a perfectly correct forward pass. Fixed by flooring `(1-z)` directly rather
   than deriving it from a clipped `z`.

**General lesson**: `_TINY=1e-30` is only safe for dimensionless, order-unity quantities -- any
new physics formula with its own natural scale needs its own appropriately-scaled floor. Full
write-up: DESIGN.md's "Resolved: synchrotron + inverse-Compton emission (ladder item 14, Phase C,
2026-09-17)" section.

**Next up: item 15 (end-to-end SNR-molecular-cloud case, now unblocked -- items 13 and 14 both
done) -- or item 12's M6 clumpiness gap, per user direction.**

## Where things stand (2026-09-17, ladder item 13 -- pion-decay SED vs. naima, DONE. Phase C started)

Ladder item 12 (SILCC-ISM project, M0a-M6) is done, with one flagged, deliberately-left-open gap
(M6's clumpiness comparison, see the 2026-09-16 entry below -- user chose to leave it and move on
rather than spend more GPU time closing it). **Item 13 starts Phase C** (differentiable emission
operators, plan Sec. 3): "Pion-decay SED for a known proton population vs naima."

**New module**: `astronomix/_modules/_cosmic_rays_grey/cr_grey_emission.py` -- a JAX
reimplementation of `naima.radiative.PionDecay`'s analytic (non-lookup-table) implementation of
Kafexhiu et al. (2014)'s p-p -> pi0 -> gamma parametrization, plus `GreyProtonSpectrum` (an
exponential-cutoff power-law proton spectrum matching naima's `ExponentialCutoffPowerLaw`
parameter-for-parameter -- the plan's grey "particle-spectrum object"). `naima` was installed
into the dev venv (`pip install naima`) as a validation-only dependency (not declared in
`pyproject.toml`, matching this module's `pytest` non-dependency).

**New pytest**: `pytests/cosmic_rays_grey/cr_pion_decay_emission.py`. Validated directly against
naima (`useLUT=False`, forcing naima's own analytic formula rather than a cached fit to it) for
the naima-tutorial-style ECPL proton population (amplitude `1e36`/eV at `E_0=1` TeV, `alpha=2.1`,
`e_cutoff=30` TeV, `nh=2.5`/cm^3), across 40 photon energies log-spaced 100 MeV-100 TeV: **max
relative error `3.4e-13`, median `5.0e-15`** -- essentially floating-point-exact agreement, not
just "close." Re-verified across all four Monte Carlo high-energy models (Pythia8/Geant4/SIBYLL/
QGSJET), with and without the nuclear-enhancement factor, and several different spectral shapes:
every case agrees to `<5e-12`. Also a differentiability smoke check (`jax.grad` of the total
summed photon rate w.r.t. every spectrum field plus gas density: finite and nonzero everywhere
tested).

**One real bug caught before the comparison ever ran, not obvious in advance:** the first
version silently returned `NaN` for every photon energy. Root cause: JAX's float32 default
overflows (`>3.4e38`) given this ladder item's own physical scale -- a *total* particle-count
proton-spectrum amplitude (`~1e36`/eV, the plan's own "known proton population" convention, not
a volumetric density) integrated over a `~1.2` GeV-`10` PeV proton-energy grid -- well before
precision would matter at all. Fixed via `jax.config.update("jax_enable_x64", True)` in the
pytest (same fix `cr_gradient_check.py`, item 16, already uses for an unrelated reason). Worth
remembering for item 15: any future code path carrying a *total* (not volumetric) CR
particle-count normalization needs `jax_enable_x64` too.

**Deliberately deferred, not attempted this ladder item:** deriving a spectrum's normalization
from a simulation cell's local `e_cr` field, and line-of-sight integration to a map (item 15's
job); the leptonic (electron) emission operators (item 14 -- turned out not to actually need the
plan's still-open "electron treatment" design decision, Sec. 6, resolved either; see the
2026-09-17 "later" entry above). Full design write-up, including the exact regime-priority
reconstruction needed to port naima's numpy-boolean-mask-assignment code into
`jax.grad`-compatible `jnp.where` form: DESIGN.md's "Resolved: pion-decay emission (ladder item
13, Phase C, 2026-09-17)" section.

**Ladder item 12's SILCC-ISM project is done except one flagged gap (M6, below); items 13 and 14
(Phase C) are now both done too -- see the 2026-09-17 "later" entry above for item 14. Next up:
item 15 (end-to-end SNR-cloud case, now unblocked), or closing item 12's M6 clumpiness gap, per
user direction.**

## Where things stand (2026-09-16, SILCC-ISM project M6 -- Simpson et al. (2016) SN-placement comparison, done; one real comparison gap flagged, not yet closed)

M6 was left as an open "and/or" in the original plan (MHD + anisotropic CR diffusion, or Simpson
et al. (2016)'s SN-placement comparison, or both) -- user picked the SN-placement branch only.

**New infrastructure**: `SNDrivingConfig.density_weighted_placement` (default `False`, fully
backward-compatible) + a new `_draw_density_weighted_site` helper in `_modules/_sn_driving/
sn_driving.py` -- Simpson et al. (2016)'s "density peak" placement mode, per-cell trigger
probability `propto rho ** SNDrivingParams.sn_density_weighting_power` (default `1.5`, their own
local-SFR proxy `sfr_i ~ m_i/t_ff,i ~ rho_i^1.5` on this codebase's fixed-volume grid), implemented
as `jax.random.categorical` sampling over eligible (interior, z-range-restricted) cells rather than
`_inject_supernovae`'s existing continuous-uniform draw. `interior_mask` construction was hoisted
to the top of `_inject_supernovae` (previously computed only for the post-site deposit-weight
normalization) so both placement modes can share it and the density-weighted draw never lands on a
ghost cell.

**New script**: `pytests/stratified_ism/m6_sn_placement_comparison.py` -- reuses M5 attempt 3's
exact box/cooling/CR-grey-transport setup (kappa_perp=1e26 cm^2/s, reduced_streaming_speed=300
km/s) **at M5 attempt 3's own resolution (32x256)**, deliberately decoupled from
`m5_cr_driven_outflow.py`'s own current state (that script now holds an unrun, 8x-resolution
"attempt 4" per a separate, still-pending user request) -- only `density_weighted_placement=True`
changes. Two new diagnostics added, targeting Simpson et al.'s specific qualitative claims (not
metrics from the paper itself, independently chosen): `_midplane_clumpiness` (std(log10(rho)) at
the midplane face-on slice, a "smooth vs. clumpy" proxy) and `_outflow_pressure_budget` (mean
P_gas/P_cr/ram-pressure at the outflow reference height, a "pressure-driven vs. ballistic" proxy).
Smoke-tested (8x64, short t_end, boosted SN rate) before the real run -- clean, no crashes, density
-weighted draw visibly landing near the high-density midplane as expected.

**Real run: completed with no NaNs through the full t_end (7.39e6 yr, all 40 snapshots).**
Late-time (last quarter of valid snapshots) comparison, M6 (density-weighted) vs. M5 attempt 3
(random, already on record, not rerun):

| Quantity | M6 (density-weighted) | M5 attempt 3 (random) | Simpson et al. (2016) |
|---|---|---|---|
| Mass-loading eta | 7.77 | 6.16 | comparable between modes |
| Outflow velocity | 21.3 km/s | 5.83 km/s | both reach >50 km/s (their units/scale) |
| H_gas | 28.54 pc | 28.41 pc | -- |
| H_cr | 115.97 pc | 86.81 pc | -- |
| Midplane clumpiness | 0.467 (time-varying, 0-0.75) | **not measured** | density-weighted -> smoother |
| P_cr vs. P_ram at outflow height | P_cr=52.2 > P_ram=33.0 | **not measured** | density-weighted -> pressure-driven (P_cr >~ P_ram) |

**Two real findings, not just "it ran":**
1. **Mass-loading is genuinely comparable between the two placement modes (7.77 vs. 6.16, same
   order of magnitude)** -- directly matches Simpson et al.'s own headline finding that mass
   loading is placement-mode-insensitive even though the driving mechanism differs.
2. **The outflow pressure budget confirms the "pressure-driven" signature on its own terms**: CR
   pressure (52.2) exceeds ram pressure (33.0) at the reference height in the density-weighted run
   -- consistent with Simpson et al.'s claim that density-weighted placement drives a
   pressure-dominated (not ballistic/kinetic-dominated) outflow. This comparison doesn't need M5's
   own number to be meaningful (P_cr vs. P_ram is evaluated within the same run).

**One real, honest gap: the clumpiness comparison is incomplete.** `_midplane_clumpiness` and
`_outflow_pressure_budget` were built for this milestone and never retroactively run against M5
attempt 3's own states -- and M5's snapshot states were never persisted to disk (only diagnostics/
plots), so there is no saved data to compute them from after the fact. M6's own clumpiness is
itself far from uniformly low (0 early, rising to a peak ~0.75 around snapshots 10-11 as density
builds up between triggers, settling to ~0.3-0.5 in the tail) -- a real, self-regulating
feedback-loop signature (high density -> high trigger probability -> disruption -> lower density)
worth noting on its own, but without M5's own number there is no way to confirm Simpson et al.'s
specific "smoother than random placement" claim quantitatively from this run alone. **Not yet
closed** -- would need a fresh run of M5's exact random-placement config with these same two new
diagnostics added (another ~43 min run at this resolution), not yet done, paused to report and
ask the user rather than spend more GPU time unprompted.

**Also worth flagging: outflow velocity differs a lot more than mass-loading (21.3 vs. 5.83 km/s,
~3.6x) and the raw per-snapshot v_out for M6 spikes considerably higher still (up to 35.98 km/s at
snapshot 37)** -- Simpson et al.'s own framing ("both reach significant velocities above 50 km/s")
doesn't map cleanly onto a single-number ratio like this given our very different absolute box
scale from theirs; reported as an observation, not matched against a specific literature number.

Diagnostic plots: `pics/m6_sn_placement_comparison.svg` (6-panel time series: midplane density,
peak T, outflow velocity, mass loading, clumpiness, pressure budget), `pics/
m6_structure_girichidis_fig1_style.svg` (same Girichidis-2016-style structure plot as M5, labeled
for this milestone's config).

## Where things stand (2026-09-16, SILCC-ISM project M5 -- first real CR-grey-transport run, blocked by a new NaN mechanism, root cause confirmed via instrumentation)

New script `pytests/stratified_ism/m5_cr_driven_outflow.py`, M4's exact stratified-column +
SN-driving + delayed-cooling + subcycled-cooling stack with grey two-moment CR transport turned
on for the first time in this project (`sn_cr_fraction=0.1`, `diffusive_relaxation=True`) --
the code-comparison anchor against Girichidis et al. (2016), ApJL 816, L19
(arXiv:1509.07247). Design decisions discussed with the user before implementing: reuse M4's
box/SN-rate unchanged rather than build a literature-scale box (cost); leave the FV self-gravity
bias (still open) undiagnosed rather than fix it first.

**Attempt 1: completed cleanly (no NaNs, all 40 snapshots, ~82 min), but the diagnostic numbers
are meaningless -- CR pressure over-diffused to a flat profile.** Used `kappa_parallel=1e28
cm^2/s` (the paper's along-field-lines value) as an isotropic stand-in for the anisotropic
transport this milestone's grey model can't represent. Result: mass-loading eta=1.09e-2 (paper:
order unity), v_out=2.37 km/s (paper: 10-50 km/s), H_gas=84.27, **H_cr=nan** -- traced directly
from the pressure-profile plot: CR pressure varies <0.1 dex across the entire 300 pc box. Root
cause, computed after the fact: diffusion length `sqrt(kappa*T_end) ~= 495 pc` vastly exceeds
this box's own 300 pc height (Girichidis's real box is ~13x taller), so CR pressure fully
homogenizes well before `t_end`. `kappa_parallel` isotropically applied includes the *vertical*
(confining) direction, where the paper's own `kappa_perp=1e26` (100x smaller) is what actually
keeps a real gradient in their much bigger box.

**Attempt 2: recalibrated to `kappa_perp=1e26 cm^2/s` (diffusion length ~49.5 pc, close to
`H_SCALE`) -- NaN'd instead of completing.** `reduced_streaming_speed` scaled down from 3000 to
300 km/s alongside kappa (100x each) specifically to hold the relaxation rate `nu =
reduced_streaming_speed^2/diffusion_coefficient` -- and hence the explicit stability bound this
source term imposes on `_cfl_time_step` -- fixed at attempt 1's already-tractable cost. Result:
clean through snapshot 4 (t=739440 yr, `n_H`(mid) already dropping 9.45->5.6, a disruption
starting), then the state silently corrupted (all-zero, t=0) between snapshots 4 and 5, with an
actual `jnp.isnan` only appearing at snapshot 39 (a preallocated-buffer artifact: once
`time`/`dt` go NaN, the driving while-loop's `time < t_end` condition is permanently False, so
the loop halts and every unwritten snapshot slot keeps its zero-initialized default -- this is
the same signature the M4 NaN investigations documented, not a new bug in the loop itself).

**This attempt 2 failure also caught a real bug in the new diagnostic script itself, fixed same
session:** `run_m5`'s per-snapshot loop only checked literal `jnp.isnan(state)`, so the 34
zero-filled (not NaN) snapshots silently passed as "valid" data -- feeding a degenerate all-zero
array into `_girichidis_fig1_style_plot`'s `LogNorm` and crashing matplotlib's colorbar deep in
its callback chain. Fixed: corruption detection now also flags `jnp.all(density == 0)` (density
is never exactly 0 given the unconditional positivity floor), and the plot function has a
defensive early-return guard against degenerate (all non-positive) density/e_cr fields.

**Root cause of the attempt-2 NaN, confirmed via instrumentation (user explicitly asked to
confirm the mechanism before picking a fix, rather than guess) -- same technique as every other
investigation in this project: temporary `jax.debug.print` behind `jax.lax.cond`, gated to the
failure window (`time+dt` in `[0.70, 1.05]` code units), inserted at the end of
`time_stepping/time_integration.py`'s `_step` (right after the hydro evolve), reproducing the
exact same PRNG/dt trajectory via an unmodified rerun of the identical config. Fully reverted
after (`grep -n DEBUG` on the file returns nothing, `git status` on it is clean).** The trace
shows a completely clean causal chain:

1. **t=0.700 to ~0.866 (code): quiet baseline.** `rho_min~3.30e-3` essentially flat,
   `cs_max` (gas sound speed) slowly drifting 16.8->15.2 km/s, `e_cr_max~250-270`, `dt~3.6-3.9e-4`
   -- ordinary background state, nothing building up.
2. **t~0.87-0.88 (~t=850,000-860,000 yr): a fresh SN trigger fires.** `cs_max` jumps
   15.2 -> 1291 km/s (~85x) and `e_cr_max` jumps 248 -> 4094 (~16x) within a handful of steps --
   the thermal and CR deposits landing simultaneously, exactly `_inject_supernovae`'s signature.
   `dt` drops correspondingly (3.9e-4 -> 1.2e-4) as the CFL bound tightens. **This is the first
   time in this project `reduced_streaming_speed` has been directly observed exceeded by a real
   gas signal speed during an actual run** -- 300 km/s vs. a peak `cs_max` of 1291 km/s, a ~4.3x
   overshoot -- confirming the accepted trade-off flagged in the module's own docstring before
   this run.
3. **t~0.880-0.890: the hot bubble decays** (`cs_max` 1291 -> 781 -> 319 km/s, `e_cr_max`
   4094 -> 3634 -> 3463) while `rho_min` stays flat through t=0.885, **then collapses sharply**
   (3.30e-3 -> 1.83e-3 -> ... -> 6.4e-4 over the final ~15 captured steps, an order of magnitude
   in well under 1e-3 code time units) **with `p_min` following it down through zero to
   -5.5e-3** -- negative pressure, in a cell still ~20x *above* this run's own density floor
   (`MIN_DENSITY_CODE~3.25e-5`), so the floor was never even close to engaging. The very next
   step (outside the captured window) is presumably where this negative-pressure cell's Riemann
   solve turns it into `NaN` (matches every earlier NaN in this project once a negative pressure
   feeds HLLC's sound-speed calculation) -- domain-wide corruption follows, then the
   preallocated-buffer zero-fill described above.

**This is the same near-vacuum negative-pressure FV-unsplit-solver fragility M4's own
investigation found and partially fixed (2026-09-15, "exact NaN mechanism confirmed" entry) --
but appearing here for the first time via CR-grey's own dynamics, not SN driving's thermal
deposit alone.** Whether the proximate trigger is specifically the CR-pressure-gradient source
term evacuating this cell faster than the existing per-stage positivity clamp can compensate for,
or whether `reduced_streaming_speed=300` being undercut by `cs_max=1291` degrades the shared
gas/CR HLL wave-speed bound (DESIGN.md's still-open ladder-item-9 Finding 1, 2026-08-27) enough to
corrupt the Riemann solve directly, is **not yet distinguished** -- both candidate mechanisms are
consistent with everything captured so far; separating them would need a second round of
per-stage instrumentation inside `_evolve_gas_state_unsplit_inner` itself (the technique M4's
deepest-dive session used). **Paused here to report findings and decide the next step with the
user**, per this module's established pattern, rather than keep digging or guess at a fix.

**Root cause fully confirmed (2026-09-16, same session, user asked to instrument one level
deeper before picking a fix): the CR-grey/gravity operator-split source term
(`_apply_gravity_source` in `_finite_volume/_state_evolution/evolve_state.py`) has NO positivity
floor, unlike the RK2-internal hydro path M4 already protects (`_evolve_gas_state_unsplit_inner`'s
floor, 2026-09-15).** Instrumented `_evolve_gas_state_unsplit` directly (two conditional
`jax.debug.print`s, `_dbg_time` threaded through as a temporary optional kwarg from `_step` ->
`_evolve_state_fv` -> `_evolve_gas_state_unsplit`, gated to the same `[0.70, 1.05]` window,
reproducing the identical PRNG/dt trajectory again): "PRE-SOURCE" (right after the RK2 hydro
stages, which *do* go through `_evolve_gas_state_unsplit_inner`'s floor) vs. "POST-SOURCE" (right
after `_apply_gravity_source` adds the pre-computed source term, which carries self-gravity *and*
-- since this run has no self-gravity active but does have CR-grey on -- the CR pressure-gradient/
adiabatic-work/relaxation terms from `_time_integrator_sources`). Result, unambiguous: `rho_min`
is bit-identical pre- vs. post-source at all 596 captured steps (the source term never touches
mass, as expected) but `p_min` **diverges sharply starting exactly where the earlier trace's
`p_min` first crashed** -- and at the very last captured step (t=0.895606), PRE-SOURCE
`p_min=3.62e-2` (still comfortably positive) while POST-SOURCE `p_min=-5.55e-3` (negative) -- the
first and only sign flip anywhere in the whole 596-step trace, happening *at* the source-term
addition, not before it. The RK2 hydro floor is doing its job correctly throughout; the bug is
squarely `_apply_gravity_source` adding an unfloored source term to an already-thin-pressure
near-vacuum cell. Debug instrumentation (both this round and the earlier `_step`-level round)
fully reverted (`grep -n DEBUG` on both touched files returns nothing, `git status` on them is
clean).

**This is a genuine, previously-unknown gap in the FV solver** -- `_apply_gravity_source` is
shared by every self-gravity run in the codebase (not just CR-grey ones; the function predates
CR-grey and was extended to carry CR-grey feedback terms via the same code path, per the existing
"names/variable ('gravity_source') predate CR-grey and are now a misnomer" comment already in
`_evolve_gas_state_split`/`_evolve_gas_state_unsplit`). M4's own self-gravity-only runs apparently
never hit this exact failure (no report of it in any M4 investigation entry), but nothing about
the mechanism is CR-grey-specific -- a strong enough self-gravity source term alone, in a near-
enough-vacuum cell, could plausibly trigger the identical gap. Natural, minimal fix (not yet
implemented, paused for user decision): mirror `_evolve_gas_state_unsplit_inner`'s existing floor
pattern (`jnp.maximum` against `params.minimum_density`/`minimum_pressure`) inside
`_apply_gravity_source` itself, right after its own `primitive_state_from_conserved` call.

**Fix implemented and attempt 3 (2026-09-16, same session): DONE. M5 is complete.** Added the
same `jnp.maximum` positivity-floor pattern to `_apply_gravity_source`
(`_finite_volume/_state_evolution/evolve_state.py`), right after its own
`primitive_state_from_conserved` call, unconditional (not gated on
`config.positivity_config.per_stage_mode`, same reasoning as the existing
`_evolve_gas_state_unsplit_inner` floor). Reran the identical M5 config unchanged otherwise
(user picked this over also raising `reduced_streaming_speed`). **Result: completed with no NaNs
through the full `t_end` (7.39e6 yr, all 40 snapshots)** -- and unlike attempt 2, this run
actually reaches and passes through the SN-driven blow-out event that previously NaN'd (snapshot
4->5: `n_H`(mid) crashes 5.6 -> 0.44 cm^-3, then continues down to `~0.001-0.002` cm^-3 through
snapshot 13 with `max(T)` spiking to `1e7-4e8` K -- a real, violent, sustained disruption, in
the same character as M4's own thermal-only disruption cycles) and keeps going: a genuine
CR-and-thermal-driven outflow develops from snapshot 14 onward (`Mdot_out` nonzero, peaking
`~1.02e4` at snapshot 16 with `v_out=13.6` km/s), then the column re-collapses and re-disrupts
multiple more times through the rest of the run (matching M4's own qualitative "episodic
disruption/re-collapse" behavior, now with CR feedback also contributing).

**Late-time (last quarter of valid snapshots) comparison to Girichidis et al. (2016):**

| Quantity | This run (M5) | Girichidis et al. (2016) |
|---|---|---|
| Mass-loading factor eta | 6.16 | order unity |
| Outflow velocity v_out | 5.83 km/s | 10-50 km/s |
| H_gas | 28.41 code (pc) | -- |
| H_cr | 86.81 code (pc) | -- |
| **H_cr / H_gas** | **~3.1** | **> 1 (their headline qualitative finding)** |

Same order of magnitude on mass loading (6 vs. "order unity" -- high side but not wildly off
given the box/SN-rate/isotropic-transport approximations already accepted going in), same order
of magnitude but low on outflow velocity (5.8 vs. 10-50 km/s), and **the paper's central
qualitative claim -- CR pressure support extends well beyond the gas scale height, generating an
"extended atmosphere" -- is reproduced directly** (H_cr exceeds H_gas by ~3x at the final
snapshot, and H_cr climbs steadily and cleanly from `6.5` to `106` code (pc) across the whole run
with no NaNs, unlike attempt 2's H_cr which was never defined past the over-diffusion/NaN
problems). This is the first time in this project H_cr has been a real, finite, trustworthy
number.

**One remaining cosmetic gap, not a physics bug:** `H_gas`'s own e-folding scale-height fit
(`_pressure_scale_heights`) returns `nan` for several snapshots in the violently-perturbed second
half of the run (27, 28, 31, 32, 34-39) -- the horizontally-averaged gas-pressure profile
presumably stops being cleanly monotonic-decaying-from-midplane once the column is this
disrupted, so the simple 1/e-crossing search fails to find a target. `H_cr` never has this
problem (its profile stays smooth throughout). Late-time `H_gas` average above is a `nanmean`
over whichever tail snapshots did resolve; not revisited this session since it doesn't change the
overall conclusion (H_cr's own value is unambiguous).

**M5 is now complete: milestones M0a through M5 of ladder item 12 (SILCC-ISM project) are done.
Only M6 (optional MHD + anisotropic-diffusion stretch) remains, not started.** Diagnostic plots:
`pics/m5_cr_driven_outflow.svg` (time series), `pics/m5_pressure_profiles.svg` (final-snapshot
P_gas/P_cr vs. z), `pics/m5_structure_girichidis_fig1_style.svg` (edge-on density, face-on
density, midplane CR energy density -- Girichidis et al. 2016 Fig. 1 style, one column since this
project only has one feedback configuration to show).

**New this session, independent of the NaN: a Girichidis et al. (2016) Fig. 1-style structure
plot** (`_girichidis_fig1_style_plot` in the M5 script) -- edge-on density (x-z slice through box
center), face-on density (x-y slice at the midplane), and CR energy density in the midplane, one
column (this project only has one feedback configuration to show, not the paper's three-panel
comparison). Written to `pics/m5_structure_girichidis_fig1_style.svg` whenever a valid final
snapshot exists.

## Where things stand (2026-09-16, later: M1's naive fixed-point cooling solver replaced with Newton's method -- fixes silent divergence, does NOT eliminate all truncation error; two dependent regression tests updated)

User asked for a proposed solution to the M0a-M4 audit's item 2 (`update_temperature_implicit`'s
naive fixed-point iteration, still-open even after `subcycle_stiff_cooling`), then to implement
and verify it. Proposed and implemented Newton's method in place of the fixed-point map
(`_cooling.py`): `F(T) = T - T_old - dT/dt(T)*dt = 0`, solved via Newton steps using `jax.grad`
for the (diagonal, since cooling has no spatial coupling) Jacobian, with an overshoot guard
(`T_new` floored at `1e-2 * T_old`) against a pathological curve driving a step negative.

**Verified via direct comparison against the naive solver across a wide stiffness range (same
`n_H=10, T=1e4` K calibration point as M1's own test, varying `dt`):**

| dt/tau ratio | reference | naive (old) | Newton (new) |
|---|---|---|---|
| 14.5x (M1 calibration) | 5854.8 K | 10068.3 K (72.0% err) | 6951.0 K (18.7% err) |
| 100x | 160.5 K | 16832.4 K (10386% err) | 444.2 K (176.7% err) |
| 1000x | 160.5 K | 78324.0 K (48692% err) | 182.7 K (13.8% err) |
| 1e5x | 160.5 K | 6.84e6 K (4.26e6% err) | 160.7 K (0.13% err) |
| 1e7x | 160.5 K | 6.83e8 K (4.26e8% err) | 160.5 K (0.001% err) |

**Two real, honest findings, not just "fixed it":**
1. **The silent-divergence failure mode is genuinely gone.** Directly traced the Newton
   iteration (separate diagnostic script, not committed) at the 100x/1000x points: residual
   converges to `~1e-12` in ~12 iterations every time -- Newton is *not* failing to converge the
   way the old fixed-point did; it's finding the exact root of the implicit equation every time.
   The old solver, by contrast, doesn't just lose some accuracy at high stiffness -- it diverges
   to physically nonsensical values (millions of percent error, e.g. `683` million K at 1e7x).
2. **Newton does NOT eliminate truncation error at large dt, and this is expected, not a bug.**
   Even an *exactly*-solved single backward-Euler step is only first-order accurate in time -- at
   intermediate stiffness (~100-1000x the local cooling time) a single such step still carries
   real error, which is *worse* in this awkward middle zone than either the mild (<15x, near the
   naive solver's own old convergence radius) or extreme (>1e5x, where the equation approaches
   just finding the equilibrium root directly) regimes. This means `_cfl_time_step`'s `dt_cool`
   term and `CoolingConfig.subcycle_stiff_cooling` remain genuinely necessary for *accuracy*, not
   merely retained out of caution -- Newton's fix is complementary to them, not a replacement.
   `update_temperature_implicit_subcycled` (M4's mechanism) is unaffected either way: it uses its
   own adaptive explicit sub-stepping, not `update_temperature_implicit` internally.

**Fixing this surfaced two existing regression tests whose entire premise was "the naive solver
fails visibly at this exact stiff point" -- both updated, not just patched to pass:**
- `pytests/stratified_ism/ki_cooling_thermal_relaxation.py`'s `test_ki_cooling_cfl_guard`: Part 1
  now asserts Newton's error is both small in absolute terms (`<0.22`, calibrated with margin
  around the observed `0.187`) and a real, substantial improvement over the old naive solver's
  historical error (hardcoded `0.71966` from the original 2026-09-11 calibration, since that
  solver no longer exists standalone to call). Module-level and function docstrings rewritten to
  narrate old-failure-then-fix rather than presenting the failure as current behavior.
- `pytests/stratified_ism/sn_driving_with_cooling.py`'s `_naive_vs_guarded_check` /
  `test_cooling_guard_engages_for_sn_driving`: same physical calibration point (this test wires
  M1's exact state through this module's own config objects), same fix, `naive_failure_rel_err_min`
  renamed `newton_rel_err_max` throughout (default `0.22`, same as M1's). **This one was caught
  only by actually re-running the test** -- editing `_cooling.py` alone would have silently broken
  it, since its assertion is a near-exact duplicate of M1's, factored into a shared helper but not
  otherwise cross-referenced by any static check.

**Regression-verified after the fix, GPU-pinned, individually:** M1's `ki_cooling_thermal_relaxation.py`
(updated test, passes), M2's `stratified_column_thermal_collapse.py` (unchanged test, passes --
confirms Newton doesn't perturb this milestone's own real run), M3.5's `sn_driving_with_cooling.py`
(updated test, passes after the fix above). M3's `sn_driving_energy_conservation.py` not
re-checked -- confirmed via grep it never references `cooling_config` at all (cooling off by
default), so `update_temperature_implicit` is never called there. M0a/M0b don't use cooling
either. M4 itself not re-run (see above: its cooling path is architecturally independent of this
change).

## Where things stand (2026-09-16: M0a/M0b/M2/M3 regression check against the M4-added FV-unsplit positivity floor -- clean, no residual gap)

Before this, the positivity floor added in `_evolve_gas_state_unsplit_inner` during M4 (see the
2026-09-15 entry below) had only ever been checked as a no-op against M4's own script -- it was
never re-run against the earlier milestones' own pytests, despite being an *unconditional* change
to shared FV evolve code. User asked for this to be closed out before starting M5. Re-ran all
four of M0a/M0b/M2/M3's own pytests (`bc_smoke_test.py`, `stratified_hydrostatic_column.py`,
`stratified_column_thermal_collapse.py`, `sn_driving_energy_conservation.py`), GPU-pinned,
individually: **all four pass clean** -- no tracebacks, no assertion failures (these are plain
`assert`-based scripts, not pytest, so a silent clean exit at 100% progress is the pass signal),
consistent with the floor's default `minimum_density`/`minimum_pressure=1e-14` being far below
any density these tests' own physics reaches. This closes the one previously-unverified item from
the M0a-M4 unresolved-issues audit; the FV self-gravity "not well-balanced" bias (M0a/M0b, still
open) and `update_temperature_implicit`'s naive fixed-point solver (M1, still open, worked around
but not repaired by `subcycle_stiff_cooling`) remain the two real open gaps carried into M5.

## Where things stand (2026-09-15, later same day: SILCC-ISM project M4 -- DONE. Root cause of the stall was cooling stiffness, not the positivity floor at all; fixed via opt-in adaptive cooling subcycling; full run completes cleanly through `t_end`)

**Picked up exactly where the previous session paused (the problem-scale-floor run that stalled
at a fixed percentage, GPU pinned at ~100% but simulation time frozen) and continued
autonomously per explicit user instruction ("keep going ... until you accomplish the goal of M4
or hit the session usage limit").** Reproduced the stall directly (GPU-pinned rerun) with a new,
throttled `jax.debug.callback` probe added temporarily inside `_step` (removed again once done,
`grep -n DEBUG` on `time_stepping/time_integration.py` clean) printing `dt`/`rho_min`/`p_min`/
`speed_max` every 2000 steps or whenever `dt<1e-9`. **This immediately refuted the previous
session's leading hypothesis:** at the stall (`n=2000`, `t=0.689`, `dt=9.35e-7`), `rho_min=
3.44e-3` is ~100x *above* the positivity floor (`3.25e-5`), not pinned at it -- so "cells sitting
at the new cold floor" was not what was happening. Deriving the cell's temperature from its
captured `rho`/`p` gives `T~7591` K at `n_H~0.1 cm^-3` -- ordinary warm/diffuse post-blowout gas,
not floor-temperature gas.

**Root cause: `_cfl_time_step`'s cooling-time constraint (`dt_cool`, added at M1 as a guard
against `update_temperature_implicit`'s fixed-point iteration diverging for stiff `dt`) is a
*domain-wide minimum* -- one persistently fast-cooling cell anywhere pins the *entire*
simulation's `dt` to that cell's own local cooling time forever.** M4's SN-driven blow-out
continuously produces exactly this kind of cell (diffuse, recently-shocked-then-unshielded gas
in the K&I curve's fast-cooling regime), so once the run reaches that regime, `dt` collapses and
stays collapsed -- a real stall, not a transient dip, distinct from (and unrelated to) the
positivity-floor NaN mechanism this and the previous several sessions were fixing. The fix that
worked for the *NaN* (the local positivity floor in `_evolve_gas_state_unsplit_inner`,
problem-scale floor values) is correct and stays; it just wasn't the stall's cause.

**Fix: `CoolingConfig.subcycle_stiff_cooling` (new, opt-in, default `False` -- every existing
config/test is bit-for-bit unaffected).** When set, `update_pressure_by_cooling` calls a new
`update_temperature_implicit_subcycled` (`_cooling.py`) instead of the plain
`update_temperature_implicit`, and `_cfl_time_step` (`_timestep_estimator.py`) skips its
`dt_cool` term entirely (the stiffness it guards against is now handled internally instead of by
bounding the global hydro `dt`). Two implementation attempts, the first of which failed
instructively:

1. **First attempt (wrong): a `jax.lax.fori_loop` over a *fixed* number of sub-steps, sized once
   from the state's *initial* relaxation rate.** Validated well against the calibrated stiff case
   from `ki_cooling_thermal_relaxation.py`'s `test_ki_cooling_cfl_guard` (`n_H=10`, `T=1e4` K,
   `dt=14.5x` the local cooling time: subcycled gives `5939` K vs. the fine-ODE reference's
   `5855` K, 1.4% error, vs. the naive single-step call's `10068` K / 72% error) -- but a
   synthetic 1000x-more-extreme case (`dt=1000x` the initial cooling time, relaxing most of the
   way to a `~160` K equilibrium) diverged badly (`78324` K vs. reference `160.5` K): the
   *initial* rate underestimates how stiff the trajectory gets as `T` keeps dropping, so a fixed
   sub-step count under-resolves the later, stiffer part, and the fixed-point solve inside an
   under-resolved sub-step can itself diverge -- compounding across every subsequent sub-step
   since each starts from the previous (wrong) state.
2. **Second attempt (used): an adaptive `jax.lax.while_loop`** that re-derives the safe local
   sub-step (`dt_sub * rate <= 0.1`, from the *current* state, re-evaluated every sub-step) and
   accumulates elapsed time until it reaches the full `time_step`, using plain explicit
   (forward-Euler) sub-steps rather than chaining fixed-point solves -- once a sub-step is kept
   inside the stability target, explicit is simpler and avoids the fixed-point's own
   non-convergence risk entirely. Re-validated: mild case `5536` K (5.4% error, still far better
   than the naive 72%), extreme 1000x case `147.4` K (8.2% error, vs. the first attempt's ~490x
   overshoot) -- degrades gracefully instead of diverging. `max_substeps=20000` bounds worst-case
   cost; timed on an M4-grid-shaped (`32x32x256`) field: `0.002s` at the mild calibrated
   stiffness, `0.069s` at the 1000x ratio (close to what the real stalled cell implied), `1.85s`
   only at a deliberately-extreme synthetic `1e5x` stress test far beyond anything the real run
   demanded.

**Verified the default (non-subcycled) path is unaffected before spending GPU time on the real
run:** `ki_cooling_thermal_relaxation.py` (M1's cooling regression) and
`sn_driving_with_cooling.py` (M3.5's SN+cooling regression) both re-run clean (exit 0) with no
changes to their configs, since `subcycle_stiff_cooling` defaults to `False` everywhere except
the M4 script.

**Real M4 run (`subcycle_stiff_cooling=True` set in the M4 script's `CoolingConfig`, GPU-pinned,
progress bar monitored end-to-end, auto-killing any run stuck at a fixed percentage for 3+
minutes per this session's instructions): completed cleanly through the full `t_end` (`~7.39e6`
yr, all 40 snapshots), no NaNs, no stall.** A first re-run attempt (before this fix, with only
the previous session's problem-scale floor) reproduced the documented stall exactly (frozen at
9.1% for 3+ min, GPU pinned near 100%) and was killed per this session's instructions; the second
run, with `subcycle_stiff_cooling=True` added, sailed through the 9.1% and 34.4% stall points
from this and the previous session and ran to completion. The resulting trajectory shows genuine
repeated SN-driven disruption/re-collapse cycles -- midplane `n_H` swinging between `~20-35`
cm^-3 (collapsed) and `~0.001-0.4` cm^-3 (blown out) several times over the run, with coincident
`max(T)` spikes to `1e7-1e8` K at each disruption -- exactly the qualitative behavior M4 was
scoped to demonstrate (episodic SN driving able to disrupt and re-trigger collapse, not just
overcool/collapse monotonically like M2's baseline). Diagnostic plot:
`pytests/stratified_ism/pics/m4_stratified_column_sn_driving_delayed_cooling.svg`.

**Operational note for future GPU-pinned background runs:** an early launch attempt in this
session backgrounded an entire `cd && source && LOG=... && nohup ...` chain with a single
trailing `&` -- since `&` applies to the whole `&&`-chained list, the `LOG` variable assignment
happened inside the backgrounded subshell, not the calling shell, so a subsequent `tail "$LOG"`
in the calling shell failed (empty variable) even though the backgrounded run itself started
fine. This left an orphaned, unmonitored duplicate process running on the GPU for ~35 minutes
before being noticed and killed. Fix: assign shell variables like `LOG=...` on their own line
(so they execute in the foreground/current shell) before backgrounding only the actual long-running
command.

**M4 is now done, completing milestones M0a through M4 of the ladder's SILCC-ISM project (item
12); M5 (the CR-on code-comparison anchor against Girichidis et al. 2016: mass-loading factor,
outflow velocity, CR-vs-gas pressure scale height) and M6 (optional MHD + anisotropic-diffusion
stretch, Simpson et al.'s SN-placement comparison) remain, not yet started.**
`subcycle_stiff_cooling` is a generally-useful opt-in fix (not M4-specific) worth keeping in mind
for any future module that can locally drive the K&I curve into its fast-cooling regime for an
extended stretch, not just SN driving.

## Where things stand (2026-09-15 continued to session's end, SILCC-ISM project M4 -- the problem-scale positivity floor no longer NaNs quickly, but the run stalls/hangs instead -- a new, unresolved, unrelated-looking problem; session paused here)

**User picked "try a problem-scale floor directly" over first instrumenting to confirm the
1e-14-is-too-small hypothesis.** Left the floor mechanism in `_evolve_gas_state_unsplit_inner`
unchanged (still reads `params.minimum_density`/`minimum_pressure` generically) and instead
overrode those two `SimulationParams` fields from the M4 script itself, tied to this run's own
density scale rather than the codebase-wide `1e-14` default:

```
MIN_DENSITY_CODE = RHO_MIDPLANE_CODE * 1e-4       # ~3.25e-5 code units
MIN_PRESSURE_CODE = get_pressure_from_temperature(MIN_DENSITY_CODE, FLOOR_TEMPERATURE_CODE, ...)
```

(`FLOOR_TEMPERATURE_CODE` is this module's existing 10 K floor temperature, already used by K&I
cooling -- reused here so the floor represents "cold gas at floor density," not an arbitrary
number.) Printed both new values at run start for visibility. This is a pure `SimulationParams`
override in the M4 script; no further changes to `_evolve_gas_state_unsplit_inner` were needed,
since it already reads these fields generically.

**Result: inconclusive on the original bug, but NOT because it still NaNs quickly -- this attempt
did not reproduce the earlier fast NaN at all, but it also never finished.** The re-run (GPU 3, this
session) ran far longer than every earlier attempt -- all of which NaN'd within ~20-26 minutes of
wall time -- reaching over 70 minutes of continuous 97-100% GPU/CPU utilization with no sign of
being stuck (still actively computing, not a frozen process) before being terminated (see below).
**Separately, the user copied the identical script
(`m4_stratified_column_sn_driving_delayed_cooling_copy.py`, byte-identical, confirmed via `diff`)
and ran it themselves on a different GPU, watching the real progress bar directly (my own run's
progress was invisible due to a `| tail -200` piping mistake that buffers all output until the
process exits, an avoidable monitoring error worth not repeating) -- their run stalled at a fixed
34.4% progress, for a reason not yet diagnosed.** Two independent runs both failing to progress
normally (mine silently for 70+ minutes with no way to check its percentage, the user's own visibly
frozen at a specific point) is strong corroborating evidence this is the same real, reproducible
phenomenon, not a fluke of one run.

**Not a NaN -- a stall/hang, a qualitatively new failure mode distinct from every previous entry in
this investigation.** Not yet diagnosed at all. A plausible, not-yet-checked hypothesis: the
adaptive-`dt` CFL estimator may be driven pathologically small once a cell sits at the new floor
(e.g. if `config.source_term_aware_timestep` folds in a cooling-stiffness criterion, evaluated at
the floor's own cold, `FLOOR_TEMPERATURE_CODE=10` K state, which could sit on a numerically stiff
part of the K&I curve) -- producing a real adaptive loop that is *technically* still advancing but
via such enormous numbers of vanishingly small steps that it never completes in any practical wall
time, which would look exactly like "stuck at a fixed percentage" to a live progress bar without
ever being a true deadlock. **Not confirmed** -- no instrumentation was added to check this before
the session ended.

**Background task terminated by user request** (`kill` on the driving bash process and its python
child; confirmed via the task's own completion notification, exit code 144 = SIGTERM, and via
`nvidia-smi --query-compute-apps` showing no remaining process of ours). **Session paused here per
explicit user request ("document the findings for now, we will pick it up here later") -- do not
resume experimentation without the user restarting this thread.**

**Where a future session should pick this up:** the local floor mechanism itself
(`_evolve_gas_state_unsplit_inner`'s unconditional clamp, still in place, not reverted) is a real,
defensible fix for the negative-pressure-at-vacuum mechanism confirmed two entries above --
independent of whatever is causing this new stall, and worth keeping. The problem-scale floor
*values* (`MIN_DENSITY_CODE`/`MIN_PRESSURE_CODE` in the M4 script, also still in place) may or may
not be implicated in the stall -- this is exactly the open question. Suggested next steps, not yet
discussed with the user: (a) instrument `_estimate_cfl_dt`/`_step` (same debug-print technique used
throughout this investigation) to directly observe whether `dt` collapses toward some tiny,
non-recovering value once the floor first engages; (b) try a substantially less aggressive floor
(e.g. much closer to the codebase default, or scaled differently) to see whether the stall's onset
depends on how deep the floor sits; (c) check whether disabling `delayed_cooling` (isolating the
floor fix from the SN-driving/cooling interaction entirely) still stalls, to determine whether the
combination is required to reproduce it or whether the floor alone is enough.

## Where things stand (2026-09-15 continued still further, SILCC-ISM project M4 -- the local positivity floor fix had ZERO effect, likely because the generic minimum_density/minimum_pressure defaults are the wrong scale for this problem)

**User picked "implement the local floor fix" over continuing to test `split=SPLIT`.** Added an
unconditional positivity floor to `_evolve_gas_state_unsplit_inner` (`_finite_volume/
_state_evolution/evolve_state.py`), right after `primitive_state_from_conserved` at the end of the
per-axis loop -- `jnp.maximum(density, params.minimum_density)` /
`jnp.maximum(pressure, params.minimum_pressure)` -- exactly the point the debug instrumentation
identified as where the negative pressure first appears. Reverted the M4 script's `split=SPLIT`/
`time_integrator=MUSCL` experiment back to defaults (MINMOD, unsplit, RK2_SSP) for a clean test of
this fix in isolation.

**Result: no effect whatsoever.** Bit-for-bit identical to the original, completely unfixed
baseline through snapshot 13 (expected -- the floor is a no-op everywhere before the failure point,
confirmed by the debug instrumentation two entries above to occur strictly between snapshot 13 and
14), and **still NaNs at the identical snapshot 14.** Verified the code itself is correct and
reachable: the edit sits inside `_evolve_gas_state_unsplit_inner` (confirmed via `sed`/`grep` on the
function boundaries), the same exact call site the debug prints fired from in the investigation two
entries above, so this isn't a case of dead/unreached code.

**Most likely explanation, not yet independently confirmed: the floor VALUES are the wrong scale
for this problem, not the floor mechanism's placement.** `SimulationParams.minimum_density`/
`minimum_pressure` default to `1e-14` (`simulation_params.py`), and M4's own `SimulationParams(...)`
call never overrides them. M4's actual density scale is `RHO_MIDPLANE_CODE~0.325` code units (the
`n_H=10` calibration point) -- the negative-pressure cell found via instrumentation had
`rho~6.9e-4`, already tiny relative to ambient, but flooring it all the way down to `1e-14` (~25
trillion times smaller than the box's ambient density) doesn't recover anything physically
sensible -- it plausibly creates an even sharper vacuum-level density discontinuity against its
still-normal neighbors than the unfloored negative-pressure state did, likely reproducing
essentially the same category of Riemann-solver failure just as fast (or via a closely adjacent
pathway that happens to fail by the same snapshot). `1e-14` is a reasonable floor for whatever
regime the rest of the codebase's tests operate in, but not for this specific milestone's density
scale.

**Not yet independently confirmed** (would need the same debug-instrumentation technique, this time
checking whether the floor actually engages and what happens in the very next Riemann solve
afterward) **-- paused to report this negative-but-informative result and get direction on the
floor value itself**, rather than guess at a replacement value and burn another ~20 minutes of GPU
time on a blind retry.

## Where things stand (2026-09-15 continued yet another level, SILCC-ISM project M4 -- SPLIT+MUSCL ran cleanly but the result is inconclusive on the specific bug, and surfaced a different, likely pre-existing failure mode)

**With `time_integrator=MUSCL` added, `split=SPLIT` ran without any config error and reached a
materially different, farther trajectory than every earlier attempt.** Per-snapshot summary:
density rises smoothly and monotonically the entire run (`n_H_mid`: `10.0 -> 10.8` through snapshot
9, then `11.7 -> 12.3 -> 13.3 -> 18.4 -> 40.8 -> 48.5` through snapshot 15), with only a mild,
gradual temperature rise (`7.6e3 -> 1.86e4` K) -- nothing resembling the unsplit run's dramatic,
disruptive SN-driven spike/blow-out (which reached `1.3e7` K and crashed density from `35.8` down to
`0.43` before NaNing). This run instead NaNs at snapshot 16 (`t` between `2.77e6` and `~2.96e6` yr)
while density is still *climbing*, at `n_H~48+` -- the opposite regime (high, still-increasing
density, not a rarefied/blown-out one) from the mechanism root-caused two entries above.

**Not a clean confirmation either way on the original bug.** This run's own SN triggers landed at
different times/sites than the unsplit run's (expected: MUSCL/SPLIT numerics differ from MINMOD/
unsplit from step 1, so the whole dt/PRNG trajectory diverges immediately, not just after a
trigger) -- and this particular trajectory never produced a disruptive-enough SN event to blow the
column out to low density at all, so **the near-vacuum negative-pressure mechanism was never
exercised in this run one way or the other.** The eventual NaN, at high and still-rising density
with only mild heating, looks instead like this project's own separately-documented, pre-existing
issue: **M2's baseline finding that this exact box/resolution/IC undergoes genuine, unmitigated
thermal-gravitational runaway collapse with no equilibrium (see the 2026-09-11 M2 entry) -- a real
physics result, not a numerical bug, that predates SN driving entirely and was never claimed to be
fixed by any of delayed cooling, momentum injection, or this axis-splitting change.**

**Interpretation, not yet confirmed further:** `split=SPLIT` may well have avoided the specific
negative-pressure-at-vacuum mechanism (consistent with the code-path reasoning: primitives are
recovered once per axis rather than after combining all three, so no single evolve call can
"secretly" go negative from the combined effect the way the unsplit path did) -- but this run
doesn't prove it, since density never got low enough to test it. **Not yet decided how to proceed --
paused to report this ambiguous-but-promising result and get direction** rather than keep spending
GPU time chasing a specific low-density state without a plan (e.g. trying other random seeds under
SPLIT to see if any produces a real blow-out and, if so, whether it survives; or treating this as
good enough evidence to adopt SPLIT and separately tackle M2's own baseline collapse limit, which is
outside this bug's scope).

## Where things stand (2026-09-15 continued a level further, SILCC-ISM project M4 -- trying config.split=SPLIT (Strang) as a cheaper FV-only alternative to switching solver modes)

**User asked, before picking a fix scope, whether switching to FINITE_DIFFERENCE with a "split"
formalism would sidestep this entirely -- investigated directly rather than guessed.** Found: FD in
this codebase has no axis-split formalism at all (`_ssprk.py`'s `rhs` always accumulates all three
axes' flux divergence into one combined update per stage, structurally identical to FV's unsplit
path) -- but FD's SSPRK/LSRK4 integrators do call `_apply_stage_positivity` (using
`config.positivity_config.per_stage_mode`, and honoring `nan_safe`) before *and* after every single
stage, which is a real, working floor entirely absent from FV's `rk2_ssp`. So FD would plausibly
sidestep this specific failure category -- but at real cost specific to this milestone: the
`cooling_shield`/`delayed_cooling` gating built this session lives only inside
`_iteration_level_continuous_updates`'s FV-gated cooling block, not in FD's `_time_integrator_
sources`-based cooling path, so switching to FD would silently drop the very mitigation this
investigation exists to validate (would need porting, not just a config flip); SN driving has never
been exercised under FD; none of M2/M3/M3.5's validation carries over. User picked the cheaper,
FV-only alternative instead: **`config.split = SPLIT`** (Strang splitting, already in this
codebase), which recovers the primitive state once per axis (`_evolve_state_along_axis`) instead of
combining all three axes before ever recovering primitives -- directly targets the confirmed
mechanism without a solver-mode change.

**First attempt failed immediately with a real (non-NaN) error, not yet a physics result:**
`_reconstruct_at_interface_split` (the split path's reconstruction) only supports
`config.time_integrator == MUSCL`, and M4's config left `time_integrator` at its default
(`RK2_SSP`) -- `ValueError: Time integrator 0 not supported for split reconstruction. Only MUSCL is
supported.` Confirmed via `finalize_config` that this auto-correction only fires for `SPHERICAL`
geometry, not `CARTESIAN` (M4's geometry), so it has to be set explicitly. Added
`time_integrator=MUSCL` alongside `split=SPLIT`. Confirmed via grep this is purely a
reconstruction/predictor-step detail local to `_reconstruct_at_interface_split`
(`config.time_integrator == RK2_SSP` is only otherwise checked in the *unsplit* path) -- doesn't
interact with SN driving/cooling/gravity, which all run outside `_evolve_state_fv` regardless of
integrator. This exact `split=SPLIT` + `time_integrator=MUSCL` + external-potential-gravity
combination has never been exercised anywhere else in this codebase (grepped `pytests/` -- only one
unrelated prior `MUSCL` usage, in the CWB shock-finder work) -- genuinely new territory, though the
code-path analysis above gives good reason to expect it composes cleanly. Re-run in progress.

## Where things stand (2026-09-15 continued to the deepest level yet, SILCC-ISM project M4 -- exact NaN mechanism confirmed: the unsplit scheme's combined multi-axis update goes negative-pressure, and per_stage_mode positivity protection is dead code for the FV solver)

**User asked to instrument inside `_evolve_state_fv` itself. Added the same conditional
`jax.debug.print`-behind-`jax.lax.cond` pattern at three points inside
`_evolve_gas_state_unsplit_inner`** (`_finite_volume/_state_evolution/evolve_state.py`): right after
each axis's reconstruction (checking that axis's left/right interface density/pressure), right after
that axis's Riemann-solver flux (checking for NaN), and once after the full per-axis loop's combined
conserved-state is converted back to a primitive state. All three removed again after use (`grep -n
DEBUG` on the file returns nothing).

**Exact mechanism, now fully confirmed rather than hypothesized:**

1. **First bad value: a genuinely negative pressure (not yet NaN), appearing only *after* all three
   axes' contributions are combined -- with every individual axis's reconstruction and Riemann flux
   having been completely clean.** `min_rho=6.86e-4`, `min_p=-0.0604` (code units) right after
   `primitive_state_from_conserved` at the end of the per-axis loop, while every per-axis debug gate
   earlier in that same call stayed silent. This is the textbook multidimensional-unsplit-FV
   positivity failure: this solver's unsplit scheme accumulates the x, y, and z flux-divergence
   contributions into one conserved-state update before ever recovering/flooring the primitive state
   (`conservative_states += conserved_change` inside the `for axis in range(...)` loop, primitive
   recovery only once at the end) -- so even though no *single* axis's flux would have driven the
   cell negative on its own, the *sum* of all three can, in a genuinely near-vacuum cell.
2. **That negative-pressure state is not caught anywhere and feeds directly into the RK2_SSP
   scheme's next stage evaluation** (`rk2_ssp`'s corrector call to the same `rhs`/
   `_evolve_gas_state_unsplit_inner`). Reconstruction on the already-uniform bad cell just carries
   the negative pressure through unchanged (confirmed: the *next* call's per-axis debug gate fires
   on reconstruction too, with matching `min_p_l=min_p_r=-0.0532`, i.e. the same bad value, now
   flagged since it was already present in the input this time).
3. **The Riemann solver (HLLC), given that negative-pressure input, immediately produces `NaN`
   fluxes on all three axes** (`any_nan_flux=True` on every axis this same call) -- almost certainly
   `sqrt` of a negative argument in the sound-speed/wave-speed estimate, this codebase's own
   documented NaN-hint category. From this point on the entire domain is `NaN` (matches every
   downstream debug print and the outer per-snapshot log).

**Why `positivity_config.default_positivity_protection=True` (tried and found ineffective two
entries above) had *zero* effect is now fully explained, and it's a more fundamental gap than
"clamping was too late": `PositivityConfig.per_stage_mode` (the field `default_positivity_
protection` sets, meant to enforce positivity "inside every SSPRK/LSRK stage") is dead
configuration for this solver path.** Grepped the whole codebase: `per_stage_mode` is read only by
`astronomix/_finite_difference/_time_integrators/_ssprk.py` (the finite-difference SSPRK
integrator) -- **never** by `astronomix/_integrators/_explicit_rk.py`'s `rk2_ssp` (the generic RK2
driver `_evolve_gas_state_unsplit` actually uses) or anywhere in `_finite_volume/`. Enabling
`default_positivity_protection` was therefore a true no-op for any finite-volume run, not merely an
ineffective floor -- explaining the earlier bit-identical-run result completely (there was no
per-stage clamp of any kind to ever engage, not just one that engaged too late). `per_step_mode`
(the *other* field the same flag sets) is real and does run, but only once per full step in
`_iteration_level_continuous_updates`, *before* that same step's own hydro evolve -- so it could
only ever have clamped the *previous* step's output, not the intra-evolve negative pressure this
mechanism produces mid-RK-stage inside the current step.

**This reframes both earlier "ineffective fix" findings as actually being one and the same latent
gap, now precisely diagnosed and directly fixable**: the finite-volume unsplit solver path has *no*
positivity enforcement anywhere between combining the three axes' fluxes and the very next Riemann
solve that consumes the result -- not because positivity protection is the wrong tool for this
failure mode (as the earlier entries speculated), but because it was never wired up for this code
path at all. This is a genuine, previously-unknown gap in the finite-volume solver, independent of
SN driving, delayed cooling, or this milestone specifically -- any FV run (with or without gravity,
with or without SN driving) that reaches a sufficiently rarefied multidimensional state could hit
the same failure.

**Not yet fixed -- a genuine choice of fix scope, per this module's established pattern.** Candidate
directions, not yet discussed with the user: (a) a narrow, local fix -- clamp density/pressure to
`params.minimum_density`/`minimum_pressure` right after `primitive_state_from_conserved` inside
`_evolve_gas_state_unsplit_inner`, unconditionally (matches this file's existing per-axis
Strang-split counterpart, `_evolve_state_along_axis`, which already checkifies -- but does not
clamp -- pressure/density positivity at the same point); (b) the more general, "do what the config
already claims to do" fix -- actually wire `config.positivity_config.per_stage_mode` into the FV
unsplit path (and `rk2_ssp` generically), matching the FD SSPRK integrator's existing behavior, so
`default_positivity_protection=True` stops being a silent no-op for FV; (c) narrower still --
apply the fix only inside `rk2_ssp` itself (position-agnostic to split vs. unsplit, FV vs. FD),
since that's the one shared piece of code every affected caller goes through.

## Where things stand (2026-09-15 continued once more, SILCC-ISM project M4 -- VAN_ALBADA_PP silently rejected by a codebase-wide gravity guard; candidate (b) is a dead end as attempted)

**The `VAN_ALBADA_PP` re-run never actually tested `VAN_ALBADA_PP`.** Output is byte-for-byte
identical to *both* previous runs (MINMOD unprotected, MINMOD + positivity protection) -- and the
run's own stdout explains why, right after config finalization:

```
Curiously, in self-gravitating systems, the VAN_ALBADA limiters seem to cause crashes.
Setting MINMOD limiter for gravity.
```

Traced to `simulation_config.py`'s `finalize_config` (line ~836): `if config.gravity_config.gravity
and (config.limiter != MINMOD): ... config = config._replace(limiter=MINMOD)` -- an unconditional,
codebase-wide guard (pre-existing, not something this investigation added) that silently forces
`MINMOD` whenever gravity is active (`self_gravity` **or** `external_potential`), based on an
undocumented earlier finding that the `VAN_ALBADA`-family limiters are themselves unstable in
gravitating systems. M4's `_base_config()` sets `external_potential=True` (it's a stratified column
under an external potential), so `config.gravity_config.gravity` is `True` -- our `limiter=
VAN_ALBADA_PP` request was silently discarded before the run started, explaining the exact
bit-identical result.

**This means candidate (b) (a positivity-preserving limiter) is not merely untried but structurally
unavailable for M4 as configured** -- and, more broadly, for essentially this entire SILCC-ISM
project, since M2 onward all use an external potential. Forcing the limiter past this guard (there
is no config flag to opt back in -- it would need bypassing `finalize_config`'s override directly)
would mean overriding a documented prior finding that this exact combination (`VAN_ALBADA`-family +
gravity) is itself a known source of crashes, i.e. potentially trading this NaN for a different,
previously-observed one rather than fixing it.

**Not yet decided how to proceed -- genuinely reopened, since the two most standard stabilization
knobs (post-hoc positivity floor, positivity-preserving limiter) are now respectively proven
ineffective and structurally blocked.** Remaining candidates from the original menu: (c) reduce
`params.C_cfl` (untried); (d) instrument one level deeper inside `_evolve_state_fv` to pin the exact
substep before choosing further; new (e) deliberately bypass the gravity/limiter guard for a
one-off test run, accepting the risk the guard describes, purely to see whether `VAN_ALBADA_PP`
would have fixed *this* NaN (informative even if not adoptable as a permanent fix, given the guard).
Paused to check in with the user given this reopens the decision. Full details: this entry plus the
two entries above it.

## Where things stand (2026-09-15 continued yet further, SILCC-ISM project M4 -- positivity protection tried and RULED OUT; switching to a positivity-preserving limiter)

**User picked "enable positivity protection" (candidate (a) from the previous entry) as the first
thing to try.** Added `positivity_config=PositivityConfig(default_positivity_protection=True)` to
the M4 script's `_base_config()` (`limiter` left at `MINMOD`), re-ran end to end.

**Result: no effect whatsoever -- the run is byte-for-byte identical to the unprotected run,
snapshot by snapshot, all the way to the same NaN at the same snapshot 14.** Verified
programmatically (diffed all 40 per-snapshot summary lines from both runs' captured output: 0
differences). This is a real, informative negative result, not just "didn't help": if the
between-stage/between-step conserved-state floor had ever actually engaged anywhere in the run
(even far from the final crash, where it would be a harmless no-op numerically but could still, in
principle, shift floating-point rounding), the two runs could not be bit-identical. Bit-identical
through the *entire* trajectory is direct evidence the floor never had anything to clamp -- **the
bad value responsible for the eventual NaN is produced strictly *inside* a single RK stage's
reconstruction/Riemann-solve computation** (most likely: MINMOD reconstructs a negative pressure or
density at a cell face in the now-rarefied gas, and HLLC's sound-speed calculation immediately turns
that into a NaN via `sqrt` of a negative argument), **before the between-stage/between-step hard
floor -- which only clamps the already-computed, already-stored conserved state -- ever gets a
chance to act.** `positivity_config`'s post-hoc clamping is consequently the wrong category of tool
for this specific failure mode; removed again from the M4 script (reverted the config addition and
its now-unused `PositivityConfig` import) rather than leaving dead configuration in place.

**Switched instead to candidate (b): a positivity-preserving limiter.** `_base_config()`'s
`limiter` changed from `MINMOD` to `VAN_ALBADA_PP` (imported from
`astronomix.option_classes.simulation_config`, not re-exported at the top-level `astronomix`
package like `MINMOD` is). Unlike a post-hoc conserved-state floor, a positivity-preserving limiter
constrains the *reconstruction* itself so a face value can never go negative in the first place --
directly addresses the mechanism the bit-identical result above points to, rather than trying to
clean up after it. Re-run in progress; not yet confirmed to fix (or fail to fix) the NaN.

## Where things stand (2026-09-15 continued still further, SILCC-ISM project M4 -- new low-density NaN localized to the hydro evolve stage, not SN driving)

**User asked to keep digging into the new low-density NaN from the previous entry.** Used the same
instrumentation technique as every earlier mechanism in this investigation: temporary conditional
`jax.debug.print`s (behind `jax.lax.cond`, gated on "any NaN or non-positive density/pressure
present") inserted at three points in `time_integration.py`'s `_step` -- right after
`_iteration_level_injections`, right after `_iteration_level_continuous_updates`, and right after
the hydro evolve (`_evolve_state_fv`) -- plus one more in `sn_driving.py`'s `_inject_supernovae`,
gated on `triggered`, printing the trigger site and the local weight-averaged ambient density. All
four are temporary and have been fully removed again (confirmed via `grep -n DEBUG` on both files
returning nothing).

**Result: only 2 real SN triggers fired in this run before the NaN (not 3, unlike the previous
tapered-shield attempt), both early and both at near-ambient density** -- trigger 1 at site
`(36.26, 22.27, 142.57)` pc, ambient `n_H~9.90` (code density `0.32228`) -- **the identical site
and density to the previous tapered-shield attempt's own trigger 1**, direct confirmation that the
two runs are bit-identical up to the first trigger (expected: the uniform-vs-tapered shield
formula only changes behavior once a trigger actually fires, see the entry above). Trigger 2 fired
at a different site, `(15.22, 25.48, 146.85)` pc, also near-ambient (`n_H~10.9`, code density
`0.3537`) -- a different site/time than the tapered-shield run's own trigger 2, as expected once
the two runs' cooling-shield footprints diverge after trigger 1 and the dt/PRNG sequences drift
apart. **Neither trigger is anywhere near the eventual NaN in either time or footprint.**

**The NaN itself: first appears after the hydro evolve step, with injections and continuous
updates both clean on that same step.** At the failing step (`t=2.606834230245266` code,
`dt=1.158e-5` -- a very small, CFL-throttled step, and `~2.55e6` yr via
`sn_cooling_delay_time`'s own yr-per-code-time conversion, consistent with the outer per-snapshot
log's NaN window of `t~2.41e6` to `~2.59e6` yr from the previous entry), `_iteration_level_injections`
and `_iteration_level_continuous_updates` both produce a fully finite, positive-density/pressure
state -- the `[DEBUG after injections]`/`[DEBUG after continuous_updates]` gates never fire for this
step -- but `_evolve_state_fv` (the FV flux/reconstruction/Riemann-solve pipeline) turns that clean
input into an all-NaN state. **This rules out SN driving, delayed cooling, and K&I cooling as the
proximate cause of this specific NaN** -- the corruption happens entirely inside the shared hydro
evolve, acting on an ordinary (no fresh injection, no active shield) but very low-density state.

**Leading hypothesis, well-supported by everything gathered so far but not yet independently
confirmed by inspecting the evolve internals directly: a generic FV near-vacuum/positivity
fragility, not anything specific to SN driving.** By this point the midplane has undergone a real,
SN-driven blow-out (the previous entry's finding) down to `n_H_mid~0.43` and falling, well below
the box's initial ambient `n_H=10` -- a genuinely new, rarefied regime no earlier M4 attempt
(all of which failed via runaway *collapse* to *high* density) ever reached. This run's config
(`_base_config()` in the M4 script) sets `limiter=MINMOD` (not positivity-preserving) and never
touches `positivity_config` (`default_positivity_protection` defaults to `False`, so
`per_step_mode`/`per_stage_mode` both stay `POSITIVITY_NONE` -- confirmed by reading
`PositivityConfig`'s definition in `simulation_config.py`) -- i.e. **nothing in this run's
configuration actually prevents a reconstructed face state or a Riemann-solved cell from going
negative near vacuum.** This is exactly the failure signature `_raise_with_time_integration_hint`
in `time_integration.py` already anticipates and has a built-in hint message for (NaN ->
suggests enabling positivity protection, a positivity-preserving limiter, or a smaller CFL
number) -- this run does not go through that hinted path today (`config.runtime_debugging` is
`False` in the M4 script, so `checkify` never wraps the call and the hint is never printed), but
the underlying stabilization knobs it names are directly applicable candidates here.

**Not yet fixed or independently confirmed at the exact sub-step (which internal evolve substep --
reconstruction, the Riemann solve, or the conservative-to-primitive recovery -- first produces the
bad value) -- a genuine choice of which stabilization knob to try, per this module's established
pattern.** Candidate directions, not yet discussed with the user: (a) turn on
`positivity_config.default_positivity_protection=True` (the cheapest, most standard fix -- clamps
density/pressure post-stage/post-step, no change to the reconstruction/Riemann-solve scheme
itself); (b) switch `limiter` to a positivity-preserving one (e.g. `VAN_ALBADA_PP`, named directly
in the codebase's own NaN hint); (c) reduce `params.C_cfl` so steps stay further inside the
stability bound even as the column rarefies; (d) instrument one level deeper (inside
`_evolve_state_fv`) to pin down the exact substep before picking a fix, since (a)-(c) are
educated guesses from the failure signature, not a confirmed root cause at the sub-step level.

## Where things stand (2026-09-15 continued further, SILCC-ISM project M4 -- tapered-shield fix applied and re-run: the fixed mechanism is confirmed gone, but a new, later, different-regime NaN appears)

**User picked "uniform shield duration" (option (a) from the previous entry's three candidates):
every cell inside the footprint (`weight` above the existing `1e-3` numerical cutoff) now gets the
same, full, untapered `sn_cooling_delay_time`, instead of `weight`-scaled duration.** Implemented in
`sn_driving.py`'s `_inject_supernovae` (the `delayed_cooling` block) -- `cooling_shield =
jnp.maximum(cooling_shield, jnp.where(footprint_mask, triggered_delay, 0.0))` in place of the old
`footprint_weight * triggered_delay`. Docstrings updated in both `sn_driving.py` (module docstring's
"Delayed cooling" section and the inline comment at the block) and `sn_driving_options.py`
(`sn_cooling_delay_time`'s field docstring) to explain why duration is deliberately *not* tapered
the way the energy deposit is. Since `SNDrivingConfig.delayed_cooling` is a static (non-traced)
config flag defaulting to `False`, this block is never even traced for any existing test
(M3/M3.5, every CR-grey ladder item) -- confirmed via the jit static-argnames mechanism, not just
asserted -- so no regression re-run was needed for those.

**Re-ran the real M4 script (`m4_stratified_column_sn_driving_delayed_cooling.py`) end to end with
the fix.** Same calibration as the previous attempt (`sn_cooling_delay_time=2457.36` yr, 50x the
probed `t_cool=49.15` yr). Result: **the specific mechanism just fixed is confirmed gone -- no more
single-cell hot spike stranded next to already-cold gas immediately after a shield lifts** -- but
the run still goes to NaN, later, and via what looks like a materially different mechanism:

- Snapshots 0-6 (`t=0` to `1.11e6` yr): quiet collapse, `n_H`(midplane) creeping `10.0 -> 12.1`,
  matching M2's baseline exactly, no visible SN heating yet.
- Snapshots 7-8 (`t=1.30e6`, `1.48e6` yr): collapse accelerates sharply, `n_H`(midplane) `25.2 ->
  35.8` -- *denser than the previous attempt's own NaN point* (`n_H_mid=30.98` at the moment it
  NaN'd) -- and the run does **not** NaN here, direct evidence the fixed mechanism is no longer
  triggered even at comparable-or-higher density than before.
- Snapshot 9-10 (`t=1.67e6`, `1.85e6` yr): `n_H`(midplane) drops sharply `35.8 -> 11.1 -> 2.6`,
  `max(T)` at snapshot 10 ticks up to `1.001e4` K (modest, but a real, visible heating signature
  coincident with the density collapse reversing) -- consistent with a real SN trigger firing near
  peak density and driving a genuine blow-out/disruption of the collapsing midplane, i.e.
  delayed cooling's mitigation doing real work this time, not just failing silently.
- Snapshots 11-13 (`t=2.04e6` to `2.41e6` yr): `n_H`(midplane) continues falling `1.08 -> 0.64 ->
  0.43` -- the midplane is now substantially *rarefied* relative to the initial `n_H=10`, a
  qualitatively new regime no earlier M4 attempt reached (every prior attempt only ever showed
  monotonic collapse toward higher density before NaNing).
- Snapshot 14 (nominally `t~2.4-2.6e6` yr, the next snapshot after 13): domain-wide NaN, same
  corrupted-`dt`/`current_time` signature as every earlier M4 NaN (all subsequent snapshots read
  `t=0`, matching the pattern already seen in the momentum-injection investigation).

**Not yet root-caused.** This is a later failure, in a low-density/rarefied regime never reached
before, following what looks like a genuinely successful SN-driven disruption event -- a
structurally different situation from the tapered-shield mechanism just fixed (which failed via a
sharp hot/cold discontinuity right at a shield's end) or the original overcooling problem (thermal
inertness). Plausible candidate causes, none checked yet: another SN trigger landing in the now
very-low-density gas (where the same absolute injection radius/energy could be far more extreme
relative to the ambient state, or where `n_0**(-0.17)`-style density scaling elsewhere in this
module could misbehave at unexpectedly low density); the K&I cooling curve's own low-density
behavior (M1's known "silent non-convergence to 98 K" failure mode was found reachable via a stale-
`dt` pathway in M3.5's Finding 5 -- worth checking whether something analogous applies here even
post-fix); or an ordinary CFL/rarefaction numerical limit unrelated to SN driving at all, now that
the column has genuinely evacuated far below its initial density. No instrumentation has been added
yet to distinguish these -- the same forced-trigger/`jax.debug.print`-behind-`jax.lax.cond` technique
used for every earlier mechanism in this investigation would apply directly.

**Paused here to check in with the user before starting a new open-ended investigation into this
new mechanism**, per this module's established working pattern (see "Working pattern" section
below) -- the tapered-shield fix just applied is a genuine, confirmed improvement (a qualitatively
new, farther-progressing, real-blow-out outcome, not just the same failure moved later), but this is
a new bug, not a continuation of the one just closed out, and deserves its own explicit go-ahead
before consuming more GPU time on it.

## Where things stand (2026-09-15 continued (root-caused), SILCC-ISM project M4 -- delayed cooling's NaN mechanism found: the taper unshields the wrong cells first)

**Root-caused via the same instrumentation technique as the momentum-injection investigation:
temporary conditional `jax.debug.print`s (behind `jax.lax.cond`, so the exact PRNG/dt trajectory
from the entry below is reproduced bit-for-bit) in the unmodified real M4 run, gated on "a trigger
just fired" and "any cell is still shielded" -- both removed again once the mechanism was
confirmed.** Only **3** real SN triggers fired before the run NaN'd (not the `~16` statistically
expected over the full window -- ordinary Poisson variance, not a bug, since the run never reaches
`t_end`): `t=0.6876` code (`~672,300` yr), site `(36.26, 22.27, 142.57)` pc, local `n_H~9.91`
(essentially the calibration density) -- shield ran its course cleanly, no problem. `t=1.7949` code
(`~1,754,900` yr), site `(18.46, 15.98, 157.79)` pc, local `n_H~13.59` -- same, clean. `t=1.8879`
code (`~1,846,300` yr, matching snapshot 10's recorded `1.328e7` K spike almost exactly), site
`(13.48, 2.67, 100.13)` pc, local `n_H~3.78` -- **this is the trigger whose shield's end precedes
the NaN.**

**The calibration-mismatch half of the previous entry's hypothesis is directly refuted by the
data, in the opposite direction from what was guessed.** Trigger 3's local ambient density
(`n_H~3.78`) is *below*, not above, the `n_H=10` the delay was calibrated against -- and the K&I
cooling time computed directly at the last-remaining shielded cell's own state right before its
shield lifts (`T=1.523e7` K, `rho` code `0.1179`, i.e. `n_H~3.63`) is `~310` yr, comfortably
*shorter* than its `~2457` yr shield, not longer -- so the delay was, if anything, generous here,
not insufficient.

**The real mechanism, confirmed by tracing trigger 3's shield cell-by-cell to its end: the tapered
`cooling_shield = max(current, weight * sn_cooling_delay_time)` formula unshields the *coolest,
tapered-edge* cells first and leaves the single *hottest* cell shielded longest -- exactly
backwards from what numerical safety needs.** The shielded-cell count shrinks smoothly and
monotonically from `7248` cells right after the trigger down to a single cell by `t=1.890374` code
(`Δt~2435` yr later, matching the nominal `2457.36` yr delay for the peak-weight cell) -- and that
last cell's pressure barely moves the entire time (`2.6716e4 -> 2.5117e4` code, `<6%` over its
whole `~2435` yr shielded lifetime, confirming the shield mechanically does exactly what it's
supposed to for that cell). But every other cell in the tapered footprint, having received a
strictly *shorter* `weight * delay` by construction, exited its own shield well before this one --
and this vicinity's K&I equilibrium is only `T_eq(n_H~3.78) = 512` K, `T_eq(n_H~9.9) = 161` K --
so by the time the last cell's shield finally lifts, its neighbors have almost certainly already
relaxed back down toward a few hundred K (not independently re-traced this session, but a direct
consequence of their shorter shields plus a K&I cooling time that only gets faster at their higher
tapered-edge densities). **The result at the moment of unshielding: a `~1.5e7` K spike compressed
into what has narrowed to essentially one grid cell, immediately adjacent to gas already back near
ambient equilibrium -- a far sharper, less-resolved discontinuity than existed right after the
original, still-uniformly-hot injection.** This is a strong, mechanistically grounded candidate for
the domain-wide NaN that follows a few hundred steps later (once cooling's implicit solve, or the
hydro flux update consuming its output, has to handle that single extreme cell next to its already-
cold neighbors) -- not independently confirmed at the exact failing step (the debug gate, keyed on
"still shielded," went silent once the last cell's shield hit exactly `0` and only caught the
NaN several hundred steps later, by which point `dt`/`current_time` were already NaN too).

**Not a bug in the sense of an accounting error -- `_inject_supernovae`'s `max(current, weight *
delay)` formula does exactly what it says -- but a real, previously-unrecognized design flaw: the
same tapered `weight` that correctly scales the *energy* deposit (more energy at the site center,
tapering to zero) is reused, unchanged, to also scale the *shield duration* -- but duration and
energy don't want the same taper direction for numerical safety. The hottest cell is the one that
most needs to stay protected at least as long as its cooler neighbors, not shed its protection
last while everything around it has already cooled out from under it.** This finding supersedes
the calibration-mismatch half of the previous entry's hypothesis; the spatial/tapered-boundary
half was directionally right but the precise mechanism is now understood (it's not "frozen core
vs. collapsing exterior," it's "the taper strips the mild edge of its shield first and abandons
the extreme core alone").

**Not yet fixed -- a genuine design decision (how to restructure the shield's spatial/temporal
profile) with more than one defensible answer, per this module's established pattern of asking
rather than picking unilaterally.** Candidate directions, not yet discussed with the user: (a) give
every cell in the footprint the *same* (untapered, e.g. the center's) shield duration rather than
scaling duration by `weight`, so the whole footprint unshields together; (b) unshield gradually
by ramping the *cooling rate* back up over some tail window instead of a hard `shield>0` boolean
gate, so there's no single-step jump from "no cooling" to "full-rate cooling" anywhere; (c) widen
`sn_smooth_cells`/the injection footprint so the hottest region spans enough cells that even a
tapered shield leaves a resolved, multi-cell hot region rather than a single-cell remnant by the
time it unshields.

## Where things stand (2026-09-15 continued, SILCC-ISM project M4 -- delayed cooling tried on the real run: NaNs again, earlier than the unmitigated baseline)

**First real M4 run with `delayed_cooling=True` (this session, right after the isolated unit test
below) still NaNs -- but differently from every earlier M4 attempt, and with a real, positive
sign the mitigation itself is working as designed.** Ran
`pytests/stratified_ism/m4_stratified_column_sn_driving_delayed_cooling.py` end to end
(`T_END=2*T_DYN`, 40 snapshots, `sn_cooling_delay_time` auto-calibrated by the script's own
`_probe_post_shock_cooling_time` at `DELAY_MULTIPLIER=50` -> `2457.36` yr, `t_cool=49.15` yr at
the probed midplane post-shock state).

**Positive sign: unlike the original pure-thermal M4 attempt (2026-09-14, "no visible
energy-budget jump attributable to a supernova ... effectively thermally inert"), this run shows a
real, sharp heating event survive the cooling.** `max(T)` at snapshot 10 (`t=1848435.3` yr,
`n_H(midplane)=30.98`) jumps to `1.328e7` K from `~1.0-1.2e4` K at the surrounding snapshots --
direct evidence a real SN trigger fired and delayed cooling kept its heat from being immediately
radiated away, the mitigation doing what it was designed to do.

**But the whole domain (not just the injection footprint) is NaN by the very next snapshot (`t`
between `~1.85e6` and `~2.03e6` yr), and -- the important new number -- this happens roughly
*twice as fast* as the pre-existing unmitigated-collapse NaN window this exact box/resolution/IC
already has on record.** `stratified_column_thermal_collapse.py` (M2, identical `N_H_MIDPLANE=10`,
`H_SCALE=50` pc, `N_XY=32`, `N_Z=256` setup, SN driving off) established via its own Check 4 that
this setup NaNs from pure thermal-gravitational collapse alone somewhere between `1` and `2`
dynamical times, reaching `n_H_mid=21.65` at `t=1.0 T_DYN`. This M4 run instead NaNs at only
`~0.5 T_DYN`, already past `n_H_mid=30.98` -- denser, sooner, than the documented unmitigated
baseline at *twice* that time. **This is a different result from the original 2026-09-14
pure-thermal M4 attempt, which NaN'd "at a density and rough timescale comparable to M2's own
unmitigated collapse"** (i.e. consistent with just the baseline happening on schedule, SN driving
not visibly changing anything). Here, delayed cooling's real, working heat deposit appears to be
actively accelerating the local collapse toward NaN rather than merely failing to prevent it -- a
materially different, not yet root-caused, failure mode.

**Leading hypothesis, not yet confirmed:** `sn_cooling_delay_time` is calibrated once at setup
time by `_probe_post_shock_cooling_time`, against the *initial* ambient midplane density
(`RHO_MIDPLANE_CODE`, `n_H=10`) -- but `sn_z_min`/`sn_z_max` only restricts *where* a trigger can
land (within one scale height of the midplane), not *when*, so by the time a trigger actually
fires (observed here at `n_H_mid` already `~31`, well into the collapse), the true local
post-shock cooling time at that much higher ambient density is likely shorter than the value the
fixed delay was calibrated against (K&I net cooling generally scales faster than linearly with
density in this regime) -- so a delay tuned for `n_H=10` may substantially over-shield a trigger
landing in already-collapsed, much denser gas. Separately (not mutually exclusive): the shield's
spatially-tapered footprint (full delay at the site center, decaying to zero at the tapered edge,
per-cell, Eulerian/non-advecting) holds a sharp boundary in place -- interior cells frozen at
pre-cooling pressure while the surrounding gas keeps collapsing normally underneath/around it for
the shield's full duration -- which could itself be numerically destabilizing independent of the
calibration mismatch. Neither half of this hypothesis has been directly checked yet -- no
per-step/per-site instrumentation was added to this run. The technique that resolved the
momentum-injection investigation and this milestone's own unit test (below) would apply directly:
force a trigger and step through the shield's decay by hand, this time watching the tapered
boundary cells specifically, and compare the probe's calibration density against the actual local
density at a real trigger site late in the collapse.

**Not yet decided how to proceed -- a genuine design decision, per this module's established
pattern of asking rather than picking unilaterally.**

## Where things stand (2026-09-15, SILCC-ISM project M4 -- delayed cooling implemented and unit-tested, tried as an alternative to momentum injection)

**User picked "delayed cooling" (a per-cell K&I cooling shutoff near a fresh injection, e.g.
Thacker & Couchman 2000 / Stinson et al. 2006) instead of continuing to root-cause momentum
injection's unresolved third NaN mechanism (see the 2026-09-14 (continued) entry below) -- tried
as an alternative mitigation for the same overcooling finding, not a fix for that NaN.** Rationale:
delayed cooling keeps the plain thermal deposit as-is (no large velocity kick), so it sidesteps
that mechanism's catastrophic-cancellation/third-NaN issues entirely rather than needing to
understand them. `momentum_injection`'s code is left in place (default off, still available).

**Implementation: a new persistent per-cell "cooling shield" field, threaded through the loop
exactly like the OU forcing field.** New `SNDrivingConfig.delayed_cooling` +
`SNDrivingParams.sn_cooling_delay_time` (`sn_driving_options.py`). At a trigger, each cell's
shield is extended to `max(current_shield, weight * sn_cooling_delay_time)` (the same tapered
`weight` as the energy deposit, so the shield duration itself tapers off toward the footprint's
edge) -- `_inject_supernovae` now takes and returns `cooling_shield`. `_iteration_level_continuous_
updates`'s cooling block gates on `cooling_shield > 0` (shielded cells keep their pre-cooling
pressure exactly; unshielded cells cool/heat normally) and decays the whole field by `dt` (floored
at 0) every step, regardless of whether cooling itself is on. This needed a persistent carry across
steps that isn't part of the advected hydro state -- the same shape of problem the OU forcing field
already solves via `LoopState` -- so `cooling_shield` was added to `LoopState` (default `None`,
like `forcing`) and threaded through every place `forcing`/`nbody_state` already go: `_step`,
`_build_initial_loop_state` (+ new `_seed_cooling_shield`), `_run_segment`,
`_time_integration_to_disk`, Orbax's `_orbax_storage.py` (`LoopCheckpoint`/`save_loop_checkpoint`/
`load_loop_checkpoint`), and `setup_helpers/restart.py` -- so a disk-checkpoint restart doesn't
silently drop the shield field, unlike leaving it half-wired would have. It is an Eulerian
(grid-fixed), not Lagrangian (fluid-comoving) field -- it does not advect with the flow -- a
documented simplification, defensible since the shield timescale is short relative to both a
cell's flow-crossing time and the time between triggers in M4's regime.

**Verified inert when off:** re-ran the full M3/M3.5 regression
(`sn_driving_energy_conservation.py`, `sn_driving_with_cooling.py`) after every threading change --
both pass unchanged, confirming the new carry-through is a byte-identical no-op at the default
`delayed_cooling=False`.

**A real bug found and fixed via this module's own isolated unit test (forced-trigger technique,
same as the momentum-injection investigation used), before it could contaminate the real M4
run:** the raw tanh `weight` used for the footprint never reaches exact `0.0` far from the site
(just an exponentially small tail) -- but the cooling gate is a hard `cooling_shield > 0.0`
boolean, so an un-floored tail spuriously suppressed cooling for one step across the *entire*
domain, not just near the footprint (caught directly: a control cell far from the site showed
`cooling_shield ~ 6.7e-8 > 0` and identical pressure before/after an otherwise-cooling-active
step). Fixed in `_inject_supernovae` by flooring the weight to exactly `0.0` below a `1e-3` cutoff
before multiplying by the delay time, so `cooling_shield` itself is exactly `0.0` outside a
well-defined footprint and the boolean gate is a correct proxy everywhere.

**Isolated unit test (not a committed pytest -- a scratch verification, mirroring the "forced
trigger" technique used for the momentum-injection root-causing) confirms the mechanism end to
end**, at M1's own stiff calibration point (`n_H=10 cm^-3`, `T=1e4` K, K&I `t_cool~3438` yr in this
setup): (1) a forced trigger (`sn_rate` set astronomically high so `trigger_probability` clips to 1
regardless of the PRNG draw) sets a tapered, spatially-localized shield (max `~0.066` code units at
the site vs. the nominal `sn_cooling_delay_time=0.070`, bounded by, not equal to, the nominal value
since no discrete grid cell sits exactly at the tanh's center); (2) a cooling call at the shielded
site leaves its pressure *exactly* unchanged (`rtol=1e-10`), the shield decays by exactly `dt`, and
an unshielded far cell is genuinely acted on by cooling/heating at the same call (K&I nets heating
below the thermally-unstable branch, so the sign isn't guaranteed -- what matters is that it's
*not* a no-op, proving the shielded-cell result is a real gate); (3) once the shield fully decays
to `0` (floored, confirmed exactly), cooling/heating resumes acting on that cell. **Mechanism
verified as designed; not yet tried against the real M4 stratified-column run** (the full-run
attempt, calibrating `sn_cooling_delay_time` against the ~60 yr midplane post-shock cooling time
measured in the 2026-09-14 M4 entry below, is the natural next step).

## Where things stand (2026-09-14 continued, SILCC-ISM project M4 -- momentum injection implemented, third NaN partially investigated, root cause still open)

**User picked "inject momentum directly" (Kim & Ostriker 2015) to fix the overcooling finding
below.** New `SNDrivingConfig.momentum_injection` + `SNDrivingParams.sn_momentum_coefficient`/
`sn_momentum_density_reference` (`astronomix/_modules/_sn_driving/sn_driving.py`/
`sn_driving_options.py`): a radial velocity kick over the same tapered footprint as the thermal
deposit, normalized so the mass-weighted radial momentum equals K&O15's fitted
`p_terminal = 2.8e5 Msun*km/s * (sn_energy/1e51 erg) * n_0**(-0.17)` (`n_0` the *local*,
weight-averaged pre-injection ambient density -- this column's density varies too much for a
single fixed reference to be appropriate). Verified M3/M3.5 still pass bit-identically (neither
sets `momentum_injection`).

**Second real finding: pure momentum injection (zero thermal deposit) NaN'd even faster than the
pure-thermal version it replaced.** Root-caused via an isolated unit test (forced trigger, same
technique as the overcooling investigation): peak velocity kick ~693 km/s in cold ambient gas
(`T~160` K, `c_s~1-2` km/s) leaves the conserved total energy completely kinetic-dominated --
recovering the tiny internal energy back out of `E_total - 0.5*rho*v^2` is a catastrophic-
cancellation computation that reliably produces negative/NaN pressure. **Fixed** with a new
`SNDrivingParams.sn_momentum_thermal_floor_fraction` (started at `0.01`): a small nonzero fraction
of the normal thermal deposit retained alongside the momentum kick, purely for numerical
stability (not a physical model) -- confirmed via the same unit test: no more negative pressure,
Mach number at the peak-kick cell drops from undefined/effectively-infinite to `~15.8`,
post-kick temperature `~82,254` K (sensible "warm shell" scale, not pathologically cold).

**Third real finding, not yet fixed: even with the thermal floor, the full run still NaNs -- but
via a genuinely different, later mechanism.** Fine-grained snapshot diagnostics (150 snapshots
over the first `~1` Myr) show: a real trigger fires cleanly at `t=0.672` Myr (`T_max` jumps to
`4.59e4` K, `|v|_max` jumps to `~467` km/s, exactly as expected); the following `~29` closely-
spaced snapshots show **smooth, physically sensible relaxation** -- `T_max` declining monotonically
toward a `~2.51e4` K plateau, `|v|_max` declining smoothly toward `~457` km/s, the hot/fast region
visibly advecting through adjacent grid cells (consistent with a real, moving shock, not a stuck
degenerate cell) -- then the *very next* snapshot abruptly NaNs, with no gradual runaway visible
beforehand. This does not match either of the first two findings' signatures (no catastrophic
early spike, no periodic-edge proximity established) -- likely a Riemann-solver/reconstruction
edge case specific to the steep velocity gradient a kick this size produces, or a CFL/dt-recompute
subtlety specific to velocity-dominated (vs. thermal-dominated) perturbations, but **not yet
root-caused**.

**User asked to keep root-causing. Investigation continued, found a real methodological trap plus
strong (but not fully conclusive) evidence on the mechanism, then paused again -- see below.**

**Methodological trap found: `return_snapshots`'s `dt_max` cap, and even just changing `t_end`
under `exact_end_time=True`, silently perturb the real dt sequence and therefore which PRNG draws
`_inject_supernovae` consumes and when -- meaning "the same run at finer snapshot resolution" is
not actually the same run.** Concretely: the original checkpoint (`NUM_SNAPSHOTS=40`,
`T_END=2*T_DYN~=7.56`) has `dt_max` cap `~=0.189`, well above the natural CFL dt (`~0.0118`) --
uncapped, this is the "authentic" dt-schedule. Every "zoom in" attempt used a shorter `T_END` with
more snapshots, capping `dt_max` below the natural value and changing the step sequence (confirmed
directly: a 150-snapshot rerun found a *different* trigger, at a different site/time, than the
authentic 40-snapshot run). Even a plain `t_end`-truncated replay (no snapshots at all, just
`exact_end_time=True` clamping `dt` near the requested endpoint) turned out to perturb the
sequence too -- a truncated run targeting a time *after* the real trigger found *no* trigger at
all in the same window. **Lesson for future sessions: to faithfully replay a specific adaptive-dt
run's exact trajectory, do not change `num_snapshots`, `return_snapshots`, or `t_end` relative to
the original -- any of these can silently change the dt sequence and hence which stochastic events
fire.** The only reliable way to isolate a specific step found here: extract the exact per-step
`time`/`dt` values from a debug trace of the *unmodified* original config (via temporary
`jax.debug.print` in `_step`/`_inject_supernovae`, removed after use), then separately replay a
single step at that exact known `dt` via `config.fixed_timestep=True`/`num_timesteps=1` starting
from a state obtained by running the *original, unmodified* config out to the exact target `t_end`
(exact_end_time's clamp reliably lands on a requested `t_end` regardless of intermediate step
boundaries, unlike snapshot-based bisection).

**Using that clean, unperturbed debug trace (original 40-snapshot config), found the exact trigger
and the exact suspect step:** trigger fires during the step starting at `t=0.676070`, site
`[36.26, 22.27, 142.57]` pc -- **only 1.24 pc from the x=37.5 pc periodic edge**, well inside the
6 pc injection radius. Post-injection `dt` correctly collapses to `~1.28e-6` (the fix working as
intended) and recovers smoothly over ~30 tiny steps up to `~7.9e-6`, with `T_max`/`|v|_max` both
*declining* the whole time (no runaway). The very next step's incoming state (estimated `dt` before
that step's own injection check) is already NaN -- so the corruption happens during the hydro
evolve of the step starting at `t=0.6877708538752518`, `dt=7.931805785265837e-06`.

**Attempted to isolate that exact step by taking a clean state at the prior boundary and
single-step-replaying just that one evolve -- it came back completely clean (no NaN, sane
pressure, `|v|_max~3.6`), NOT reproducing the crash.** Root cause: this "clean state" extraction
itself used a `t_end`-truncated replay (see the methodological trap above) that took 60 steps to
reach the same real time the authentic run took ~77 steps to reach, meaning the trigger never
fired in this replay at all -- confirmed by zero `triggered=True` in its own 60-step debug trace.
The single-step "replay" was therefore replaying an untriggered ambient step, not the real suspect
step. **This specific isolation attempt is inconclusive, not negative** -- it didn't rule out the
hypothesis, it just failed to test it due to the trap above.

**Directly tested the leading hypothesis (periodic x-wrap momentum collision, M3.5 finding-3-style
but via velocity convergence instead of thermal pile-up) by manually computing the momentum-kick
field at the exact real trigger site, bypassing the PRNG entirely.** Result: **not confirmed, and
actually weighs against this specific mechanism.** The vx profile along the site's x-column is
smooth throughout: one sign flip exactly at the injection site (expected, symmetric outward push)
and a second sign flip at the antipodal point on the periodic ring (`x~17.5` pc, `37.5 - 36.26 +
17.5`-ish), which is the real collision point a periodic radial kick must produce -- but the
injection weight there is negligible (`~3e-5`, since the 6 pc radius is far short of the ~18.75 pc
antipodal distance), so that collision is numerically harmless at this specific site/radius. No
sudden multi-hundred-km/s discontinuity between adjacent cells was found anywhere in the profile.
**This doesn't rule out periodic wrapping in general** (a site much closer to the edge, or a larger
radius, could still make the antipodal collision land inside significant weight) -- it just rules
it out for *this specific* trigger.

**Status: paused again, third time this investigation, to check in with the user** -- per this
module's own established working pattern (see "Working pattern established across this module's
sessions" below). Genuine progress made (the exact trigger/step identified, one strong hypothesis
tested and weighed against, a real methodological pitfall documented for future sessions) but the
precise failure mechanism is still not pinned down. All temporary debug instrumentation
(`jax.debug.print` calls added to `time_stepping/time_integration.py`'s `_step` and
`_modules/_sn_driving/sn_driving.py`'s `_inject_supernovae`) has been removed; both files are back
to only the permanent momentum-injection changes from earlier in this entry. M4 itself remains an
exploratory scratchpad script, not a committed pytest.

## Where things stand (2026-09-14, SILCC-ISM project M4 -- BLOCKED on a real overcooling finding, not started as a committed pytest yet)

**First M4 attempt (M0b+M1+M3 combined, CR off) reuses M2's exact stratified+cooling column
(`stratified_column_thermal_collapse.py`: 37.5x37.5x300 pc, `h=50` pc, `N_XY=32`/`N_Z=256`,
one-shot local-equilibrium IC) with episodic SN driving added.** Two design decisions made with
the user before implementing, both grounded in real numbers, not guessed:

1. **SN rate:** literature SILCC-project areal rates (Girichidis/Walch et al., SILCC-II/III
   papers, ~1-1.5e-4 SNe/Myr/pc^2 over a 500x500 pc box) scaled down to M2's much smaller
   37.5x37.5 pc footprint give only ~1 SN every ~6 Myr -- comparable to or longer than M2's own
   collapse timescale (NaN by 1-2 dynamical times, `T_DYN~3.7` Myr, without SN driving). User
   picked: keep M2's box, deliberately boost the rate (~10x the area-scaled literature value,
   targeting ~8 expected triggers per `T_DYN`) rather than widen the box toward literature scale
   (much more expensive) -- a documented small-box compensation, consistent with this project's
   "target scaling relations, not exact numbers" design decision (#4 in DESIGN.md's SILCC-ISM
   section).
2. **SN site placement:** `_inject_supernovae` draws sites uniformly over the *whole* box; since
   this column's density falls off sharply away from the midplane, most random sites would land
   in tenuous, numerically-expensive envelope gas rather than the astrophysically relevant
   midplane. User picked: add a new, backward-compatible `SNDrivingParams.sn_z_min`/`sn_z_max`
   z-restriction (default +/-inf = unrestricted, bit-identical to today's formula at that default
   -- verified M3/M3.5 still pass unchanged after this change), and M4 restricts sites to +/-1
   scale height (`|z-z0|<50` pc) around the midplane.

**Injection radius calibrated via a pre-run probe** (mirroring M3.5's own calibration practice)
across the column's actual density range (midplane `n_H=10` down to envelope `~0.1`): picked
`sn_injection_radius=6` pc (~5.1 grid cells) as a compromise between numerical gentleness
(`T_hot~8.1e6` K at the midplane -- comparable order of magnitude to M3.5's own successful
`n_H=100`/`4` pc calibration point) and keeping the footprint a modest fraction (~16% of the
half-width) of the 37.5 pc box.

**Result: the first full run NaN'd within the checkpoint window (`T_END=2*T_DYN`), at a density
(`n_H_max~197`) and rough timescale comparable to M2's own unmitigated collapse -- SN driving did
not visibly change the outcome.** Snapshot instrumentation (fine time resolution, both a 40- and a
100-snapshot rerun) ruled out the two more mundane explanations first: (1) the domain-wide (not
localized) nature of the NaN rules out a periodic x/y-wrap self-collision (M3.5's own still-open
finding 3) as the cause; (2) an isolated unit test, calling `_inject_supernovae` directly with
M4's exact config and a forced (guaranteed) trigger, deposited *exactly* the expected thermal
energy (`50291442.16` code units, matching `sn_energy` to full precision) -- ruling out a bug in
the injection function or the new z-range code.

**Root cause, confirmed with a direct calculation: real, severe "overcooling."** Domain-total
energy tracked at fine time resolution (both a coarse 40-snapshot full run and a fine
100-snapshot short run, different PRNG step sequences via different snapshot-driven `dt_max`
caps) never showed *any* visible energy-budget jump attributable to a supernova, despite ~1-5.6
expected triggers across the two runs combined. Directly computed the K&I cooling time at the
midplane injection's own post-shock state (`n_H=10`, `T_hot~8.1e6` K, using `ki_net_rate`):
**`~60.5` yr** -- roughly 200-3000x shorter than any reasonable snapshot cadence (`~11,600` yr at
finest) and a tiny fraction of a single hydro dynamical time (`T_DYN~3.7e6` yr). SN driving's
correctly-deposited thermal energy is radiating away via K&I cooling almost immediately after
each real trigger, before it can do meaningful PdV work or drive turbulent support -- so at this
resolution/density/injection-radius combination, SN driving is effectively thermally inert, and
the column collapses essentially as in M2's baseline. **This is real, well-documented ISM-
simulation physics** (the classic "overcooling problem," e.g. Katz 1992), not a numerical bug --
confirmed via an independent, from-first-principles cooling-time calculation, not just inferred
from the run's behavior.

**Not yet fixed -- a genuine design decision with more than one defensible answer, per this
module's established pattern, presented to the user rather than picked unilaterally.** Standard
literature mitigations: inject momentum directly instead of (or alongside) thermal energy (e.g.
Kim & Ostriker 2015's terminal-momentum prescription); temporarily suppress cooling near a fresh
injection site for some delay time; increase the injection radius/resolution until the immediate
post-shock cooling time is no longer catastrophically short (this project's `radius=6` pc,
`N_XY=32` combination is likely too under-resolved for this to work without a much bigger
radius/box, per the earlier radius-calibration table); or boost `sn_energy` non-physically far
beyond `1e51` erg so enough survives cooling to matter. **No shared simulation code changed this
session beyond the `sn_z_min`/`sn_z_max` addition** (`_sn_driving/sn_driving_options.py`,
`sn_driving.py`) -- M4 itself is not yet a committed pytest, still an exploratory script in this
session's scratchpad.

## Where things stand (2026-09-14, stale-dt-at-injection bug -- FIXED)

**M3.5's Finding 5 (stale `dt` reused across injection/cooling/hydro-evolve on a triggering
step) is fixed, before starting M4 as recommended below.** Discussed four candidate approaches
with the user first (global unconditional fix; the same split but gated behind an opt-in flag;
each injection module reporting its own CFL requirement with a full adaptive retry loop; other) --
**user picked the global, unconditional fix.** One correction surfaced during that discussion
worth recording: the bug is *not* CR-triggered (M3.5, where it was found, and M4, which it
blocks, both run CR off) -- an earlier framing of "gate behind a CR flag" would not have covered
the actual failing case, since the real trigger condition is "an episodic lump-sum injection
module is active" (SN driving today; wind and CR-DSA shock injection structurally share the same
risk). A second correction: the recompute point can't sit between `_iteration_level_updates` and
the hydro evolve as PROGRESS.md's earlier sketch assumed -- cooling itself, which runs *inside*
that function right after SN injection, was directly implicated in Finding 5 (cooling alone at
the stale `dt` drove cells to unphysical `98` K), so the split has to be internal to that
function, not at its outer boundary.

**Implementation:** `astronomix/_modules/_iteration_level_updates.py`'s single
`_iteration_level_updates` is now two jitted functions: `_iteration_level_injections` (stellar
wind, SN driving, and -- newly regrouped here, previously last in the old function -- grey-CR
diffusive shock injection: all three are lump-sum/discontinuous updates to `primitive_state`) and
`_iteration_level_continuous_updates` (cooling, the neural-net/CNN correctors, viscosity,
turbulent forcing, frame tracking, CR streaming/anisotropic-transport corrections, the positivity
floor -- unchanged relative order among themselves). `time_stepping/time_integration.py`'s `_step`
now: estimates `dt` from the pre-injection state (factored into a new `_estimate_cfl_dt` helper,
byte-for-byte the same FV/FD x mhd/hydro x source-term-aware branching that used to be inlined),
applies `_iteration_level_injections` at that `dt`, then -- only `if not config.fixed_timestep`
and at least one of `wind_config.stellar_wind` / `sn_driving_config.sn_driving` /
`cosmic_ray_grey_config.diffusive_shock_acceleration` is active (a static, config-level condition)
-- calls `_estimate_cfl_dt` a second time on the post-injection state and takes
`jnp.minimum(old_dt, new_dt)` (never relaxes the snapshot/end-time clamps already applied, only
tightens), before running `_iteration_level_continuous_updates` and the hydro evolve at the final
`dt`. Configs using none of the three lump-sum modules compile to exactly the same single-estimate
path as before this change -- zero behavior change for the rest of the test suite.
`config.fixed_timestep` (M3's own exact-PRNG-replay energy-conservation technique) is deliberately
exempted, since re-estimating there would desync the replay from the simulation's actual `dt`
sequence.

**Reordering risk check (CR-DSA injection moved out of the continuous phase):** grepped every
`pytests/cosmic_rays_grey/*.py` config for `diffusive_shock_acceleration=True` combined with
`cooling`/`turbulent_forcing`/`neural_net_force`/`cnn_mhd_corrector`/`diffusion`/`frame_tracking`
-- none of items 7-11's tests ever combine DSA injection with any of the modules now running after
it (this module's consistent "isolate one thing per test" convention), so this reordering changes
none of their computed results; not independently re-run this session (deferred to the general
regression pass below) but the config-level argument is airtight given the existing test suite's
composition.

**Known, deliberately accepted residual gap, documented in a code comment (`time_integration.py`,
next to the N-body advance):** N-body's RK4 advance must run *before* `_iteration_level_injections`
(multi-source stellar wind needs the freshly-advanced orbit position for its injection sites this
same step), so it necessarily uses the pre-recompute `dt` even on a step where SN driving/wind/
CR-DSA later shrinks it -- a currently-unexercised combination (no CR-grey ladder test uses
`nbody`) analogous in spirit to the standing Lax-Friedrichs CR-dissipation gap.

**Regression suite re-run against this change (2026-09-14), all pass:** every CR-grey ladder
pytest that exercises the moved/split code paths -- items 3, 5, 7, 8, 10, 11
(`cr_anisotropic_diffusion_oblique.py`, `cr_streaming_1d.py`, `cr_sedov_taylor.py`,
`cr_dsa_mach_dependence.py`, `cr_wind_bubble.py`, `cr_snr_clumpy_medium.py`) -- plus both
SN-driving tests (`sn_driving_energy_conservation.py`, M3's `fixed_timestep` exact-replay check;
`sn_driving_with_cooling.py`, M3.5's own test, closest to the original bug). Run individually
(not in a shared batch) via the GPU workaround on a pinned free device to avoid cross-run GPU
memory contention -- an early batched run produced one apparent OOM (`cr_snr_clumpy_medium.py`)
and one apparent hang (`cr_wind_bubble.py` hit a 590s timeout at 76% progress) that both turned
out to be contention/insufficient-timeout artifacts from running heavy jobs back-to-back on one
GPU, not real regressions -- both passed cleanly in isolation with more time.

**M3.5's closing test redesigned (2026-09-14) to exercise the real stochastic trigger, per
explicit user request as direct evidence the fix works.** `test_sn_injection_with_cooling_stability`
(`pytests/stratified_ism/sn_driving_with_cooling.py`) no longer builds the SN deposit into the
initial condition -- it now runs the actual `SNDrivingConfig` stochastic trigger
(`_run_stochastic_injection_with_cooling`, replacing the deleted `_run_injection_with_cooling`)
under genuine adaptive (not `fixed_timestep`) CFL stepping, the exact mid-run mechanism Finding 5
traced the original NaN to. Same ambient setup as the deterministic tests (`n_H=100`, 4 pc fixed
injection radius, 40 pc box, 128 cells, open boundaries, cooling on, CR-grey off), over the same
`T_END_YEARS=3e4`. New `EXPECTED_N_TRIGGERS=2` / `SN_RATE_CODE = EXPECTED_N_TRIGGERS / T_END_CODE`
constants (kept modest, not M3's 5-9, since each real trigger now correctly forces a run of tiny
post-injection steps -- that's the fix working as intended -- so more triggers means real added
wall-clock cost).

**Result at the default `random_seed=42` (fixed, fully reproducible): no NaNs -- direct,
positive evidence the fix resolves the original bug via the actual failing mechanism, not just via
the sidestep the test used before today.** Calibrated numbers from this run: peak temperature at
t_end `9083.2` K, `9041.4` K above ambient equilibrium (`41.77` K) -- proves at least one real
trigger fired, since this box has no other heat source -- comfortably below the
single-deposit-reference-scaled ceiling (`5x` the raw single-deposit `T_hot=2,592,683` K
`=12.96e6` K). Mass conserved to `9.14e-6` relative error -- looser than M3's near-machine-precision
periodic-box identity, and *correctly so*: this test's real random site can land close enough to
the open boundary for its shock to reach the edge and carry real mass out within `T_END_YEARS`,
unlike M3's periodic box or this module's old deterministic test (deliberately box-centered, kept
away from any edge) -- confirmed via a diagnostic probe that boundary-face density shifts by
`~0.07%` from ambient at t_end, consistent with a real edge-reaching shock, not an accounting bug.
`mass_conservation_tol` recalibrated from a stale `1e-9` (copied from the old deterministic test's
different physical regime) to `1e-4` (~11x margin over the observed value); `max_temperature_
ceiling_kelvin` widened from `1.5x` to `5x` the raw single-deposit reference (to allow, without
asserting, for the small chance two of the `EXPECTED_N_TRIGGERS` random sites land close enough to
locally superpose); new `min_peak_temperature_excess_kelvin=1000` K check added (replacing the old
"cooling reduces peak T from the known raw value" check, which doesn't transfer cleanly to
multiple, randomly-timed triggers) as the test's real evidence a trigger fired. Full narrative:
that file's own module docstring (canonical record, updated in place).

## Where things stand (2026-09-12, SILCC-ISM project M3.5 -- DONE, SN driving + cooling; a real bug found and reported, not fixed)

**Milestone M3.5 (SN driving + K&I cooling combined, still uniform/unstratified) is done, after a
long investigation that found five real things, the last of which is a genuine architectural bug
in shared time-stepping code, reported here rather than fixed.** New pytest
`pytests/stratified_ism/sn_driving_with_cooling.py`, three tests: `test_sn_blast_evolves_cleanly_pre_cooling`,
`test_cooling_guard_engages_for_sn_driving`, `test_sn_injection_with_cooling_stability`. Full
narrative and all numbers are in that file's own module docstring (the canonical record); this
entry summarizes it.

**Finding 1: the instant of SN injection is *not* where cooling stiffness (gap #3) shows up.** A
single SN deposit (`E_SN=1e51` erg into `n_H=1 cm^-3` ambient gas, any astrophysically-reasonable
injection radius) produces `T_hot ~2.3e9` K -- so hot that K&I's `Lambda(T)` has already saturated,
making the cooling time *longer* than the pure-hydro CFL `dt` (`~1.7e5` yr vs. `~11` yr) -- the
opposite of stiff. This is the textbook reason the early adiabatic/free-expansion Sedov-Taylor
phase (item 11's own cooling-free SNR test lives here) is adiabatic in the first place.

**Finding 2: this project's own affordable box/resolution doesn't spontaneously reach a
genuinely cooling-stiff state either.** Evolved a single SN deposit hydro-only (no cooling) and
scanned `T_EVOLVE_YEARS in {2000, 5000, 10000, 20000, 30000}`, tracking the cell actually driving
the cooling-CFL constraint (via `dtemperature_dt`, not a guessed cell). The compressed cell's
density grows (`n_H` up to `~20.4` at `2e4` yr) but its temperature never drops below `~1e5-1e7` K
in this window, and `dt_hydro/t_cool` never exceeds `~1.0` across the whole scan (need `>10x` to
count as stiff) -- the interior stays too hot for too long, in this 20 pc periodic box, for the
shell's cooling time to ever overtake the interior's own sound-crossing constraint.

**Finding 3: a real, randomly-sited explosion in a periodic box NaN'd the solver via a
periodic-edge wrap, at two resolutions.** The first full-run test (`NUM_CELLS=48`, all-periodic
BCs, matching M3's own convention) NaN'd reproducibly (cooling on *and* off, ruling out cooling)
within `~50-100` yr of a real trigger firing. Direct bisection traced it to a trigger landing at
grid index `x=5` of `48` (`~2.3` pc from the edge), whose `~2.5` pc footprint wraps the periodic
seam -- the ghost-cell handler mirrors a second copy of the `~1e8-1e9` K blast onto itself, an
extreme, poorly-resolved head-on collision the HLLC solver can't handle. Doubling resolution to
`NUM_CELLS=96` only reduced the failure rate (halves the wrap danger zone) -- a second,
independent random trigger reproduced the same near-instant NaN there too.

**Finding 4: switching periodic -> open boundaries did not fix it either.** The *exact same*
explosion (boundary type cannot change the PRNG trigger sequence) NaN'd again under open
boundaries at `NUM_CELLS=96`, ruling out "periodic wrap" as the sole mechanism -- an open
boundary's own interior-only weight renormalization can make a near-edge deposit *more*
concentrated when part of the tapered sphere falls outside the domain. Root cause reframed as the
raw injection magnitude itself (`T_hot ~1e8-1e9` K at `n_H=1`) being numerically extreme near any
edge, any handling. Fix (user-directed): ambient density raised 100x to `n_H=100 cm^-3` (M1's
other calibration point, `T_eq~41.8` K) and the injection radius decoupled from grid spacing,
fixed at a physical 4 pc -- a direct calibration scan picked this to land `T_hot` at `~2.6e6` K
instead of `~1e9` K (`~1000x` reduction). Box widened to 40 pc, `NUM_CELLS` raised to 128 (both
user-directed).

**Finding 5, the deepest one: even the 1000x-gentler injection still NaN'd, and root-causing it
found a genuine architectural bug in the shared time-stepping code -- unrelated to cooling, edges,
or injection magnitude.** The new hot cell was `~30+` cells from any domain edge (rules out edges);
cooling-on vs. cooling-off NaN'd at the identical step with near-identical pressure (rules out
cooling); every synthetic single-explosion reproduction attempted -- box-centered, off-grid-offset,
and the *exact* off-center coordinates the real failure used -- evolved cleanly for `10000+` yr
(rules out "any explosion at this energy/location is just fragile"). The common thread: only the
real, stochastic, *mid-run* `_inject_supernovae` path ever failed. Reading
`astronomix/time_stepping/time_integration.py` confirmed why: each step's `dt` is computed from the
primitive state *before* `_iteration_level_updates` runs, and that same already-fixed `dt` is then
used by `_inject_supernovae`, `update_pressure_by_cooling`, *and* the hydro flux evolve for that
same step -- so on the triggering step, cooling and the hydro update both apply a `dt` sized for
the calm pre-explosion state to the freshly-injected, extremely hot cell. Measured directly: the
pre-injection ambient `dt` (`~41281` yr) was `~441824x` larger than the correctly-sized
post-injection `dt` (`~0.093` yr). Applying cooling alone at the stale `dt` didn't NaN but drove
some cells to an unphysical `98` K (M1's known silent-non-convergence failure mode, confirmed
reachable via this exact pathway); the subsequent hydro flux step at the same stale `dt` is the
most likely proximate source of the actual NaN. Lowering `EXPECTED_N_TRIGGERS` from 4 to 1 (an
intermediate attempt at working around it within this test) *still* NaN'd -- consistent with the
bug firing from any single real stochastic injection, not requiring overlapping explosions.

**This is a real, open bug, flagged for a future dedicated session -- not fixed here (user
confirmed: report, don't fix now).** It affects the shared time-stepping code
(`astronomix/time_stepping/time_integration.py`'s per-step `dt` computation order relative to
`_iteration_level_updates`), not just SN driving -- any future module that injects a large,
localized perturbation mid-run (not via the initial condition) would hit the same gap. Not touched
this session per explicit user direction; `astronomix/_modules/_iteration_level_updates.py` and
`astronomix/time_stepping/time_integration.py` are unmodified.

**Closing test design (redesigned around Finding 5):** since the stochastic mid-run trigger
mechanism can't be exercised reliably right now, and the bug is about that mechanism (not gap #3
itself), the closing test returns to the same safe pattern every earlier point-injection ladder
item already uses: build the SN deposit into the *initial condition*, not via `SNDrivingConfig`'s
trigger, so `time_integration`'s first `dt` is already correctly sized for the hot state.
`test_sn_injection_with_cooling_stability` (`n_H=100`, `NUM_CELLS=128`, 40 pc box, `T_END=3e4` yr,
item 11's own radiative-phase-onset estimate): NaN-free; mass conserved to `1.4e-16` relative error
(tol `1e-9`); peak temperature drops from the raw-injection `2,592,683` K to `5002.8` K at t_end
(a `99.8%` drop, tol `>1%` margin >>1) while staying far below the `1.5x` raw-injection runaway
ceiling; minimum temperature at t_end matches the ambient equilibrium (`41.77` K) exactly, as
expected for far-field cells untouched by the blast.

**No shared simulation code was changed for this milestone** (the bug above is reported, not
fixed) -- `_iteration_level_updates.py`, `time_stepping/time_integration.py`, `sn_driving.py`, and
the M1/M3 cooling/SN-driving modules are all exactly as M3 left them.

**Next session should pick up at M4** (per the roadmap: M0b+M1+M3 combined, CR off, flagged
highest-risk step) -- but first, strongly consider whether to fix Finding 5's stale-`dt`-at-injection
bug before attempting M4, since M4 combines SN driving with a real, much longer, much more
elaborate run where the same bug would very likely resurface (M4 cannot rely on M3.5's workaround
of avoiding the stochastic trigger mechanism entirely, since M4's whole point is repeated SN
driving over time). A plausible fix shape (not implemented, not verified): recompute/re-clamp `dt`
after `_iteration_level_updates`'s injection step, before the hydro evolve consumes it -- or have
`_inject_supernovae` report the post-injection state's own CFL requirement back to the loop.

## Where things stand (2026-09-12, SILCC-ISM project M3 -- DONE, episodic SN driving)

**Milestone M3 (new episodic SN-driving module, validated in isolation via an exact
energy-conservation check) is done.** New module `astronomix/_modules/_sn_driving/`
(`sn_driving.py`'s `_inject_supernovae`, `sn_driving_options.py`'s `SNDrivingConfig`/
`SNDrivingParams`), wired into `_iteration_level_updates.py` before cooling (gap #4, per the
plan file's already-decided ordering). New pytest
`pytests/stratified_ism/sn_driving_energy_conservation.py`.

**Mechanism:** each step draws one Bernoulli trial with probability `sn_rate * dt` (thinned-
Poisson approximation, valid for `sn_rate * dt << 1` -- at most one supernova per step) and, if
it fires, a uniformly random site in the box. Deposits the same tanh-tapered spherical footprint
items 7/11 use: a thermal part added to gas pressure, plus -- when the CR-grey model is active --
`sn_cr_fraction` (default 0.1, Girichidis et al. 2016) of the total dumped directly into `e_cr`
(gap #4's design decision #2: not routed through `inject_crs_at_shocks`/DSA). `sn_cr_fraction` is
forced to 0 (all-thermal) whenever `registered_variables.cosmic_ray_e_active` is False, so total
energy conservation never depends on whether CR-grey is on -- verified directly (see below).

**Real bug found and fixed while building the validation test:** the first version computed the
taper weight (and its normalizing sum) over the full *ghost-padded* array
`_iteration_level_updates` operates on. For a site drawn near a periodic edge this double-counts
real domain volume in the normalization (a ghost cell mirrors a real interior cell on the far
side, so both get weight) and then silently discards the ghost cells' share of the deposit the
next time the boundary handler refreshes them from the interior -- losing a fraction of
`sn_energy` whenever a site landed near an edge. Calibrated mismatch before the fix: `~0.125` out
of `0.45` total injected over 9 triggers -- large and systematic, not noise. Fixed by restricting
the weight (both the normalization sum and the actual `.add()` deposit) to the interior cells
only, relying on the standard boundary handler to refresh the ghost cells afterward. This is a
genuinely new failure mode relative to every earlier point-injection ladder item (7/10/11), which
all use *open* boundaries with the explosion kept well away from the domain edge by construction
-- periodic-box point injection at arbitrary random sites had never been exercised in this
codebase before M3.

**Test design (exact, not statistical energy-conservation check):** a uniform 3D Cartesian
*periodic* box (`NUM_CELLS=32`), no gravity/cooling/turbulent forcing, `config.fixed_timestep=True`
(`NUM_TIMESTEPS=400`, `T_END=0.05`) so every step's `dt` -- and hence the trigger probability -- is
identical and known in advance, and `config.random_seed`'s default key is the *only* per-step
PRNG consumer (turbulent forcing, the other one in this codebase, is off). This lets the test
independently replay the exact `jax.random.split`/`bernoulli` sequence outside the simulation
(`_replica_trigger_count`) and know `N_triggers` exactly, then check
`E_total(t_end) - E_total(0) == N_triggers * sn_energy` to near machine precision -- valid
regardless of how complicated the subsequent shock evolution gets, since a periodic box with no
gravity/cooling has no other channel to gain or lose energy. Same `fixed_timestep`-based technique
ladder item 9 used to instrument the per-step `e_cr` budget (see the 2026-09-09 entry below).
Requires `jax_enable_x64` (this directory's now-familiar float32-round-off lesson, see M0a) --
without it the accumulated round-off over 400 fixed steps blew the tolerance by ~5 orders of
magnitude and, more subtly, could shift which step the `exact_end_time` clamp lands on.

**Calibrated numbers:** at `NUM_CELLS=32`, `SN_RATE=100`, `T_END=0.05` (expected count 5, observed
`N_triggers=9` at the default seed): CR-active run's energy-conservation relative error `~7.2e-10`
(tol `1e-6`, >1000x margin); CR-inactive run (verifies the `sn_cr_fraction`-forced-to-0 fallback)
`~7.8e-15` (essentially exact -- one fewer floating-point exchange path than the CR-active run).
CR-active run's final `E_cr / (N_triggers * sn_energy) ~= 0.096` vs. the raw injected
`sn_cr_fraction = 0.1` -- not an exact match, and not expected to be: once deposited, `e_cr`
exchanges energy with the gas via the always-active adiabatic-work coupling (verified correct back
in ladder items 1/2), so some drift away from the raw injected fraction is expected physics.

**No existing simulation code changed except the new gated `_iteration_level_updates.py` branch**
(off by default, `SNDrivingConfig.sn_driving = False`) and a new `needs_geometric_centers` gate in
`simulation_helper_data.py`, both no-ops for every existing config -- confirmed via
`shock_tube1D.py` and `cr_advection.py` re-runs, unaffected.

**Next: milestone M3.5** (SN driving + cooling combined, still uniform/unstratified -- isolates
the freshly-injected-hot-ejecta-vs-explicit-cooling stiffness risk, gap #3, in a 2-module
combination before it's buried inside M4's 4-way combination). See the roadmap in
`/export/home/nknoell/.claude/plans/memoized-discovering-scone.md` for M3.5 through M6.

## Where things stand (2026-09-11, SILCC-ISM project M2 -- DONE, rescoped to collapse onset)

**Milestone M2 is done, but not as originally scoped.** The plan's goal ("M0b + M1 combined --
stratified hydrostatic column with cooling active; verify a physically sensible two-phase-structured
quasi-equilibrium") turned out to be unreachable: **a plain external-potential-confined column with
K&I cooling and no additional support (turbulence/SN driving/magnetic pressure) does not have a
dynamically stable, meaningfully-stratified equilibrium at all** -- it undergoes genuine,
physically real thermal-gravitational runaway condensation instead. This was established via four
independent checks (all user-directed, one AskUserQuestion round each as the investigation
progressed) before rescoping the milestone (user-confirmed) to verify the collapse's *onset* is
real, self-consistent physics rather than chasing an equilibrium that isn't there. New pytest
`pytests/stratified_ism/stratified_column_thermal_collapse.py`. Full design rationale:
DESIGN.md's "Resolved (rescoped): SILCC-ISM project M2" section -- this entry covers the
investigation's numbers.

**Check 1 (naive IC): a single-temperature isothermal column (M0b's own IC style, `T=8000` K
everywhere) collapses immediately.** By `t=2` vertical dynamical times (`h/c_s,warm ~= 3.70e6` yr
at `h=50` pc, `T_warm=7500` K -- `t_dyn ~= 3.70e6` yr), the midplane (`n_H,0=10` cm^-3, matching
`ki_cooling_thermal_relaxation.py`'s own CNM calibration point) runs from `n_H=10` to `n_H=195.5`
cm^-3, `T` from 8000 K to 47.6 K (still *above*, i.e. still actively approaching, its new local
equilibrium of 33.3 K -- not yet settled). Mass conservation degrades to `~1.1%` (vs. `~1e-6` to
`1e-14` in every prior non-collapsing test in this project). By `t=5` dynamical times, the run is
NaN everywhere. The initial hypothesis was that starting `~100-250x` out of local thermal
equilibrium at the midplane (T=8000 K vs. T_eq~33-160 K there) was driving an avoidable violent
transient.

**Check 2 (well-equilibrated IC): rules that hypothesis out.** Built a "coarse local-equilibrium"
IC instead -- density from a warm-isothermal (`T=7500` K) barometric guess, but temperature set
*per cell* from the K&I equilibrium at that cell's guessed local density (so the midplane starts
at `T=160.6` K, essentially exactly its true equilibrium `T_eq(10)=160.5` K). **Made no
qualitative difference**: the midplane still ran away to `n_H=202.9`, `T=55.6` K by `t=2` dynamical
times, NaN by `t=5`. (A fully self-consistent hydrostatic+K&I-equilibrium profile was also
attempted via fixed-point iteration on the coupled ODE `d(ln rho)/dz = -mu*g(z)/T_eq(n_H(rho))*k_B`
-- this diverges outright even with strong under-relaxation (tried alpha down to 0.05, still
diverges within a few iterations), an independent confirmation that the underlying continuous
problem is itself unstable, not just a numerical-iteration artifact.)

**Check 3 (weaken gravity): does not find a stable, still-stratified regime.** Scanned `C_pot`
(the external potential's depth, `phi=C_pot*log(cosh((z-z0)/h))`) at `{1.0, 0.3, 0.1, 0.03, 0.01}`
of its reference value (set via a warm `T=7500` K reference), at fixed `n_H,0=10`, `t_end in
{2,5} t_dyn`. Factors `1.0` down to `0.1` all still collapse (`n_H` at midplane reaching
189-831 by `t=5` dynamical times, mass errors `13-28%`). Factors `0.03`/`0.01` do avoid collapse
(`n_H_mid` only reaches 12.9/11.1) -- but only because a shallower potential also flattens the
*initial* density profile it's used to construct: at those factors the edge density is `8.05`/
`9.37` cm^-3, barely below the midplane's `10` -- i.e. avoiding collapse this way also erases the
very density contrast (needed: `~100x`, warm envelope to cold core) this milestone is meant to
demonstrate. Weakening gravity trades away the stratification rather than stabilizing it.

**Check 4 (resolution scan) -- the decisive one.** Reran the well-equilibrated-IC setup (Check 2)
at higher resolution (`N_XY=32`, `N_Z=256`, vs. the earlier `N_XY=16`, `N_Z=128`). At `t=1`
dynamical time the higher-resolution run looks genuinely good (`n_H_mid=21.65` from `10`, a modest,
physically coherent factor; `T_mid=91.5` K vs. its own new local equilibrium `T_eq(21.65)=88.0` K
-- tracking closely, not noise) -- but by `t=2` dynamical times it is *already* NaN, **faster**
than the low-resolution run (which took until `t=5`). A resolution increase making the collapse
happen *sooner*, not converging toward a stable answer, is the textbook signature of an
**unregularized thermal instability**: Field (1965)'s classical linear result is that the ISM
thermal instability's growth rate is unbounded at short wavelength without a regularizing
mechanism (thermal conduction, turbulence, magnetic tension -- none implemented in this module yet).
Refining resolution further would only resolve smaller, faster-growing modes, never converge.

**Conclusion (user-confirmed rescoping):** this is real physics, not a bug or an artifact to
eliminate -- and it previews exactly why the roadmap's M3 (episodic SN driving/turbulence) exists
as the mechanism expected to resist this runaway. The "stable two-phase profile" claim is deferred
to M4/M5, once that additional physics is present to provide it. M2 instead verifies the
collapse's *onset* (up to a defensible, calibrated cutoff time) is real, self-consistent physics.

**Committed setup and numbers (2026-09-11), `stratified_column_thermal_collapse.py`:** same
physical setup as Check 4's higher-resolution run (`CodeUnits(1 pc, 1 Msun, 1 km/s)`, `h=50` pc,
domain `z0 +/- 3h`, `n_H,0=10` cm^-3, `N_XY=32`, `N_Z=256`, one-shot local-equilibrium IC),
`t_end=1.0` dynamical time (`~3.70e6` yr) -- comfortably inside the clean, NaN-free window
(collapse reaches NaN between 1 and 2 dynamical times at this resolution). Calibrated checks, all
passing with real margin: NaN-free; mass conserved to `2.33e-3` (tol `5e-3`, >2x margin); midplane
`n_H` grows `10.00 -> 21.65` (factor 2.165, tol `>1.5`, >1.4x margin) while tracking its own new
local equilibrium to `3.98%` (tol `8%`, >2x margin) -- confirms the K&I cooling correctly drives
the collapsing gas, not some other numerical effect; envelope stays quiescent (`n_H` grows only
`1.14x`, tol `<1.3`; T tracks its equilibrium to `0.091%`, tol `1%`, >10x margin). The diagnostic
plot (temperature vs. the K&I equilibrium curve evaluated at the final density) shows the final
profile tracking the equilibrium curve closely across the *entire* column, not just at the two
calibration points -- a strong visual confirmation the physics is behaving correctly throughout.

**No shared simulation code was touched for this item** -- purely a new test exercising existing
infrastructure (M0b's stratified-potential pattern, M1's K&I cooling curve) in a new combination,
so items 1-11 and M0a/M0b/M1 are unaffected by construction (not re-run this session).

**Next: milestone M3** (episodic SN-driving module, validated in isolation via an exact
energy-conservation differential check, same style as items 7/10/11) -- per the roadmap, M3's
turbulence/repeated-energy-injection is also the physics expected to eventually let a future
milestone (M4) reach the genuinely stable, turbulence-supported multiphase quasi-equilibrium that
a plain gravity+cooling column (this M2 investigation now shows directly) cannot reach on its own.

## Where things stand (2026-09-11, SILCC-ISM project M1 -- DONE, K&I two-phase net-cooling curve)

**Milestone M1 ("new K&I-style two-phase net-cooling table + builder function, validated via an
isolated thermal-relaxation test") is done.** New cooling-curve type `KOYAMA_INUTSUKA_NET_COOLING`
(`astronomix/_modules/_cooling/cooling_options.py`/`_cooling_tables.py`/`_cooling.py`), new pytest
`pytests/stratified_ism/ki_cooling_thermal_relaxation.py`, new independent reference solver
`astronomix/test_setups/reference_solutions/koyama_inutsuka_equilibrium.py` (plain numpy/scipy, no
astronomix/JAX). Full design rationale (why a new analytic dispatch branch rather than the existing
`PIECEWISE_POWER_LAW` table, and why the mu_e/mu_H compensation is exact, not an approximation):
DESIGN.md's "Resolved: SILCC-ISM project M1" section -- this entry covers the session's numbers and
the three real findings that reshaped the milestone along the way.

**Finding 1: the plan's "two stable phases from a range of initial temperatures" assumption
doesn't hold at fixed density -- checked directly, not assumed.** `ki_bracket(T)` (the K&I curve's
dimensionless bracket) is monotonically increasing across `T=10` to `1e5` K with no local hump
(checked by direct evaluation before writing any test code) -- the equilibrium condition at fixed
`n_H` (`bracket(T)=1/n_H`) has exactly one root, always thermally stable under Field's isochoric
criterion. Since `update_pressure_by_cooling` only ever touches pressure (never density), a
fixed-density box structurally cannot explore bistability regardless of initial `T`. The real
K&I "S-curve" instead lives in the **pressure-density plane**: a finer scan (`n_H=0.05` to `50`
cm^-3, 60 log-spaced points) shows `T_eq(n_H)` decreasing monotonically (`n_H=0.05 -> T_eq=7979`
K; `n_H=50 -> T_eq=55.7` K) while `P_eq(n_H)=n_H*k_B*T_eq(n_H)` is genuinely non-monotonic: rises
to a local maximum `P_eq/k_B~4950` K/cm^3 near `n_H~0.93-1.05` cm^-3, falls to a local minimum
`P_eq/k_B~1597` K/cm^3 near `n_H~8.6` cm^-3, then rises again -- any pressure in `[1597, 4950]`
K/cm^3 admits three equilibrium densities (cold/dense, unstable, warm/diffuse), which is where
genuine bistability shows up, but only once density can respond to pressure imbalance (M2 onward),
not in this milestone's isolated, no-gravity, fixed-density scope. The committed test instead
checks: (a) convergence to the unique correct root from both below and above at 3 representative
densities (`n_H={0.3, 1.0, 10.0}` cm^-3, spanning `T_eq~{6724, 5006, 161}` K), and (b) that
`P_eq(n_H)` traced from the simulation's own converged temperatures is non-monotonic -- the
defining two-phase signature, not just "some curve that happens to cool".

**Finding 2 (a real test-setup bug, caught by the independent-reference cross-check): missing a
`mu_H` factor when building the initial density silently relaxes to the wrong equilibrium.** This
codebase's cooling module defines `n_H = density/mu_H` (`get_particle_number_density`'s
convention) -- every existing CR-grey pytest builds density as `n_H*m_p` (matching
`cr_snr_clumpy_medium.py`/`cr_wind_bubble.py`'s own convention) because none of them ever touch the
cooling module's `n_H`-dependent machinery, so this had never mattered before. The first M1 run
built density this same (wrong, for cooling) way and got a **clean, NaN-free, spatially-uniform**
result that nonetheless relaxed to `6923.05` K instead of the intended `n_H=0.3` equilibrium
(`6723.82` K) -- traced directly to the actual simulated `n_H` being `0.3/mu_H=0.228` cm^-3 (whose
independently-computed equilibrium, `6923.12` K, matches the wrong simulated result to `<0.001%`,
confirming the exact mechanism). Fixed by building density as `n_H*mu_H*m_p`. This is exactly why
the test cross-checks against an independent reference rather than only asserting "did it converge
to some value" -- a self-consistent-looking wrong answer would otherwise have passed silently.

**Finding 3 (gap #3, confirmed not assumed): `update_temperature_implicit`'s naive fixed-point
iteration fails at a genuinely stiff dt, silently -- fixed with a new CFL term.** The plan's
expectation was that the module's default `IMPLICIT_COOLING` integrator would handle an
arbitrarily large `dt` gracefully; checked directly at a deliberately stiff point (`n_H=10 cm^-3`,
`T=1e4` K -- e.g. freshly-shocked or freshly-injected hot ejecta suddenly at high density;
instantaneous cooling time `~3438` yr) and found **false**: a single implicit step at `dt=5e4` yr
(~14.5x the local cooling time) gives `10068.32` K vs. the independently-integrated true `5854.83`
K (rel. err `0.72`) -- `jax.lax.while_loop`'s `cond_fun` just stops at `max_iter=50` regardless of
whether `tol=1e-6` was reached, silently returning a wrong temperature with no error or warning.
Fixed by adding a new cooling-aware `dt_cool` term to
`astronomix/_finite_volume/_timestep_estimation/_timestep_estimator.py`'s `_cfl_time_step`
(mirroring the existing `dt_visc`/`dt_relax` pattern: `dt_cool = C_cfl /
max_over_grid(|dT/dt|/max(T, floor_temperature))`), gated on `config.cooling_config.cooling` so
every pre-existing non-cooling test is provably unaffected (the branch is simply never entered).
Verified the fix directly: at the same stiff state, the guarded `_cfl_time_step` returns
`1.407e-3` code time vs. the pure-hydro `3.274e-3` (ratio `0.430`), and the guarded value matches
`C_cfl *` the independently-computed instantaneous cooling time to `~1.4e-15` (exact at float64,
as expected for a single uniform cell -- both are the same calculation by construction).
Repairing the fixed-point iteration itself (e.g. Newton's method) would be a separate, larger
change to shared cooling code -- out of scope here; the new CFL term is what prevents the hydro
loop from ever handing the existing solver a dt this stiff in the first place.

**Setup and calibrated numbers (2026-09-11):** 1D Cartesian FV (`HLLC`/`MINMOD`), `NUM_CELLS=8`
(purely local physics, uniform ICs -- confirmed the converged final state is spatially uniform to
machine precision, i.e. genuinely zero hydro dynamics, a true isochoric-relaxation test), box 1 pc,
`CodeUnits(1 pc, 1 Msun, 1 km/s)`, `X=0.76`/`Z=0.02` (module defaults), `floor_temperature=10` K.
Six `(n_H, T_init factor)` runs (`n_H={0.3,1.0,10.0}` cm^-3 x factor `{0.8,1.2}`, i.e. below/above
each density's own true equilibrium), `t_end` chosen generously above each density's
ODE-estimated 1%-convergence time (`n_H=0.3`: `6e6` yr; `n_H=1.0`: `3e7` yr; `n_H=10`: `5e5` yr) --
all cheap (~10-2200 CFL-limited steps per run, well under a minute each on CPU/native-JAX
fallback, no GPU with compute capability >=8.0 was available this session). Final calibration, all
six runs: density conserved exactly (rel. err `0.0` at float64); `eq_rel_err` (sim vs. true T_eq)
ranges `1.5e-5` to `3.4e-3` (worst case `n_H=10`, committed tol `0.01` has >2.9x margin);
`ode_rel_err` (sim vs. independent ODE trajectory at the same t_end) ranges `6.0e-8` to `1.2e-3`
(same worst case, committed tol `0.005` has >4x margin). Simulated `P_eq/k_B` at
`n_H={0.3,1.0,10.0}` is `{2017, 5006, 1605}` -- matches the independent reference scan to <1% and
confirms the required non-monotonic shape (`P_eq(1.0) > P_eq(0.3)` and `P_eq(1.0) > P_eq(10.0)`).

**No pre-existing test was re-run this session for regression** beyond a spot-check of a few
representative pytests (`cr_advection.py`, `cr_sedov_taylor.py`, `bc_smoke_test.py`) after touching
the shared `_timestep_estimator.py` file, chosen because the new `dt_cool` branch is gated on
`config.cooling_config.cooling` (default `False`, untouched by any of items 1-11 or M0a/M0b) so it
is structurally a no-op for every pre-existing test -- these spot-checks exist to catch an import-
or syntax-level regression, not a physics one.

**Next: milestone M2** (M0b + M1 combined -- stratified hydrostatic column with cooling active;
verify a physically sensible two-phase-structured quasi-equilibrium). Read the plan file
(`/export/home/nknoell/.claude/plans/memoized-discovering-scone.md`) and this entry before starting
-- M0b's own flagged FV-gravity well-balancedness gap (previous entry below) is still open and
unrelated to M1, but both will be simultaneously active in M2's setup.

## Where things stand (2026-09-11, ladder item 12 -- SILCC-ISM project started, M0a+M0b DONE)

**Ladder item 12 ("reproduce a published grey CR-ISM result") has been scoped as its own
multi-session project, not a single ladder item** -- unlike items 7-11 (each a new pytest reusing
existing machinery), a SILCC-style stratified box needs real new infrastructure (a stratified
external potential, an ISM cooling curve, an episodic SN-driving module) that doesn't exist yet.
User explicitly chose the full-reproduction scope over a scoped-down single-session anchor or
deferring to item 13. Full milestone roadmap, verified infrastructure facts, and design decisions
are written up in a plan file this session (`/export/home/nknoell/.claude/plans/
memoized-discovering-scone.md`) -- **read that file in full before picking this project back
up**; this entry summarizes it plus this session's M0a/M0b results.

**Target papers** (verified via web search, not just citation text): **Girichidis et al. 2016**
(ApJL 816, L19; arXiv:1509.07247) -- primary target, SILCC-project turbulent/magnetized/
self-gravitating/SN-driven stratified box, CR injected *directly at each SN site* as a fixed 10%
of SN thermal energy (`E_CR=1e50` erg, `E_SN=1e51` erg per SN) -- note this matches this module's
existing `dsa_efficiency=0.1` convention by coincidence, not design. **Simpson et al. 2016** (ApJL
827, L29; arXiv:1606.02324) -- secondary/stretch, AREPO stratified-column SN-only-vs-SN+CR /
density-peak-vs-random-placement comparison.

**Verified infrastructure facts worth preserving (so future milestones don't re-investigate):**
- Gravity (`astronomix/_modules/_gravity/`): external potential is a pure user-supplied
  `jnp.array` field (`params.gravitational_potential`), no functional-form registry, no built-in
  stratified/disc potential -- `pytests/self_gravity/external_potential.py`'s Plummer-sphere
  pattern is the template. Wired for both FD and FV.
- **Cooling already works with FV** -- corrects a wrong sub-report generated mid-investigation
  that claimed FD-only. Directly verified: `_iteration_level_updates.py:108-112`, gated
  `config.solver_mode == FINITE_VOLUME`, calls the same `update_pressure_by_cooling` the FD path
  uses. No porting needed for a future milestone. Cooling has **zero test coverage anywhere in
  the repo** and its Townsend exact-integration path is explicitly broken (own code comment) --
  only explicit/implicit Euler stepping usable; **no cooling-time CFL constraint exists**
  (`_timestep_estimator.py` has `dt_visc`/`dt_relax` but nothing for cooling) -- a real risk for a
  future milestone near a thermally-unstable cooling curve, flagged not yet investigated.
- **Self-gravity is deferred, and more strongly justified than first assumed**: the FFT Poisson
  solver (`_poisson_solver.py`) has its own code TODO stating a mix of periodic and non-periodic
  boundaries is not yet supported -- exactly the periodic-xy/open-z combination this whole project
  needs. Self-gravity would need real Poisson-solver work before it could even attempt this
  project's boundary setup; a fixed external stratified potential sidesteps this entirely.
- Turbulent-forcing's carried-PRNG-key pattern (`LoopState.key`, split every step in
  `_turbulent_forcing.py`) is the right reusable precedent for a future episodic SN-driving
  module's stochastic trigger/placement -- must split deliberately per consumer so SN triggering
  and turbulent forcing don't correlate if both draw from the loop's key in the same step.
- Direct CR injection at the SN site (Girichidis' convention) is the chosen design for a future
  SN-driving module, **not** a reuse of `inject_crs_at_shocks`/DSA -- avoids compounding item 11's
  own logged, unresolved DSA-injection resolution-non-convergence finding (see the entry below) in
  a much bigger, more expensive setup. This does **not** sidestep item 9's Finding 1
  (`reduced_streaming_speed` sharing the HLL Riemann solver's wave-speed bound) -- that's general
  CR-grey transport machinery, still an open risk once a future milestone turns on
  diffusion/streaming for directly-injected CRs.

**M0a (BC smoke test) DONE.** New `pytests/stratified_ism/bc_smoke_test.py`: validates the mixed
periodic-xy/open-z boundary combination in isolation (uniform box, no potential, a tanh-tapered
overdense blob advecting under a uniform bulk `v_x=0.5`, `t_end=3.0` -- one full periodic wrap,
settling at the analytically expected `x=0.75` box units). Originally calibrated at `N=32` (mass
conserved to `~1.19e-6` rel. error, x-centroid abs. error `~0.021`, y/z drift `~1e-6`); rerun this
session at `N=128` per the user's request. That rerun surfaced a real precision issue at the
default float32: mass conservation failed (`~5.6e-5` against the `1e-5` tolerance) from float32
round-off accumulation (more cells feeding `jnp.sum`, more CFL-limited time steps at finer `dx`),
not a BC bug -- fixed by enabling `jax_enable_x64` (now on by default in this file, same as
`stratified_hydrostatic_column.py`/`external_potential.py`). Final calibration at `N=128`,
float64: mass conserved to `~2.15e-14` relative error (tol tightened to `1e-9`, >1e4x margin),
x-centroid abs. error `~4.9e-5` (tol tightened to `1e-3`, >20x margin), y/z drift `~1e-15` (tol
tightened to `1e-5`). No BC bugs found -- the general hydro BC handler's per-axis independence
(flagged as lower-risk than gravity's BC handling during this session's investigation) held up as
expected; the only real issue was the float32/float64 precision choice.

**M0b (stratified hydrostatic column) DONE, with a real flagged finding.** New
`pytests/stratified_ism/stratified_hydrostatic_column.py`: generalizes `external_potential.py`'s
Plummer pattern to a vertical potential `phi(z) = C_pot * log(cosh((z-z0)/h))` (smooth everywhere,
chosen over a literal `phi(z)=g0*|z|` linear form to avoid a midplane source-term kink -- a
deliberate deviation from this session's plan, which had floated the linear form first), whose
matching isothermal hydrostatic density is the classic Spitzer (1942) `rho(z) = rho0 *
sech^2((z-z0)/h)` profile when `C_pot = 2*c_s^2`. Originally calibrated at `N_XY=8`, `N_Z=64`
(core Mach `~0.096`, core density drift `~0.023`, tols `0.15`/`0.04`); rerun this session at
`N_XY=64`, `N_Z=512` (`h=1.0`, `z0=4.0`, open z-boundaries sit 4 scale heights out, density there
`~1.4e-3` of midplane, FV+HLLC, `t_end=4.0` ~5 vertical dynamical times) per the user's request.
Final calibration at `N_XY=64`: core Mach number `~0.1057` (tol `0.15`, >1.4x margin), core
relative density drift `~0.0255` (tol `0.04`, >1.5x margin) -- both existing tolerances held
without needing adjustment.

**The real finding**: these residuals are substantially larger than expected from a "tiny
round-off residual" equilibrium test, and -- checked directly across three resolutions (`N_XY=8`:
mach `~0.096`, drho `~0.023`; `N_XY=16`: mach `~0.101`, drho `~0.023`; `N_XY=64`: mach `~0.1057`,
drho `~0.0255`) -- **do not shrink with resolution** across an 8x range in `N_XY` (64x in total
cell count). This rules out ordinary truncation error. Root cause identified by reading the
gravity module directly:
`_gravitational_source_term_along_axis` (`_gravity.py:299`), the *only* FV-mode gravity coupling,
uses a simple non-conservative `rho*a`/`rho*v*a` source (its own docstring: "finite-volume
self-gravity supports only the SIMPLE_SOURCE coupling") -- unlike the FD path's
`SECOND_ORDER_CONSERVATIVE`/`FOURTH_ORDER_CONSERVATIVE` options, which build the energy source
from the density fluxes to stay consistent with the flux-based pressure-gradient discretization.
A non-conservative source added on top of an unmodified Riemann-solver flux is a classic
"not well-balanced" FV scheme combination -- exactly the kind of resolution-independent systematic
residual observed here. A third check (halving `C_pot`) was inconclusive/confounded (it also
changes the equilibrium profile's steepness, not just the gravity magnitude in isolation) and
wasn't pursued further, per this module's practice of bounding exploratory digging.

**This is a real, open gap for future milestones**, not fixed here (out of scope for a smoke-test
milestone): M2 onward builds hydrostatic structure on top of this FV gravity coupling, and if
scale-height measurements need to be precise, this ~10% Mach / ~2% density floor either needs to
be accepted as a documented systematic bias, or a well-balanced FV gravity coupling needs to be
implemented first. Flagged here rather than guessed at further.

**Full milestone roadmap** (see the plan file for complete detail): M1 (K&I-style cooling curve +
cooling-CFL stability check), M2 (M0b+M1 combined), M3 (episodic SN-driving module, isolated
energy-conservation check), M3.5 (SN driving + cooling combined, catches cooling-stiffness-vs-hot-
ejecta risk before the big combination), M4 (stratification+cooling+SN-driving combined, CR off --
flagged highest-risk step), M5 (M4 + CR on -- the actual code-comparison anchor: mass-loading
factor, outflow velocity, CR-vs-gas pressure scale height vs. Girichidis 2016, at a
scaling/qualitative level given this project's necessarily coarser box/resolution than the papers'
own), M6 (optional stretch: MHD + anisotropic diffusion, Simpson et al.'s SN-placement comparison).
**Next session should pick up at M1**, after rereading the plan file and this entry.

## Where things stand (2026-09-11, ladder item 11 follow-up -- resolution check, 48^3 vs 128^3)

**Not yet resolved, logged for later.** After item 11 (below) was calibrated and committed at
its documented `NUM_CELLS=48`, the user reran the same test with `NUM_CELLS=128` (all four
`{uniform, clumpy} x {control, DSA}` runs, everything else unchanged) and asked whether the
result was physically valid and an improvement. Numbers below for the 128^3 run were decoded
directly from the run's own diagnostic plot (`cr_snr_clumpy_medium_test.svg`'s vector bar-chart
paths and embedded density-slice raster) rather than from captured stdout, since the run's raw
printed diagnostics weren't saved -- precise enough for this comparison (cross-checked against
the plot's own axis-tick calibration) but a future resolution-scan session should capture the
printed numbers directly instead of re-deriving them from the SVG.

**What holds up well at 128^3 (both conservation checks get *tighter*, not looser):**
- Total energy conservation: ~2e-5 relative error across all four runs (all ~5031.8 code-energy
  units), vs. the 48^3 calibration's ~1.5e-6 -- both comfortably inside the committed
  `conservation_tol=1e-3`.
- The energy-partition identity (thermal+kinetic lost = CR gained): ~8e-5 relative error in both
  media at 128^3, vs. ~6.5e-4 (uniform) / ~2e-4 (clumpy) at 48^3 -- tighter at higher resolution,
  as expected (less numerical diffusion contaminating the bookkeeping).
- The clumpy-DSA run's density slice shows a physically sensible, still-mostly-spherical shock
  shell with a real, visible bow-shaped distortion where it overruns a clump caught in the
  z-mid slice -- the qualitative picture item 11 was built to check for.

**What does not hold up: the Check-2 "clumps-matter" signature and the absolute `E_cr` numbers
are resolution-sensitive, not converged.**
- `E_cr` (uniform-DSA / clumpy-DSA): `122.82 / 100.82` at 48^3 (**-17.9%**, the number item 11's
  `clumpy_effect_min=0.05` was calibrated against, originally with a >3.5x margin) vs.
  `255.30 / 240.72` at 128^3 (**-5.7%**) -- barely above that same 5% floor, almost no margin
  left.
- More fundamentally: `E_cr` as a *fraction* of the (resolution-independent, by construction)
  total energy budget roughly doubled between resolutions -- `2.44%` (48^3) to `5.07%` (128^3) at
  the same `dsa_efficiency=0.1`/`dsa_mach_min=1.3`. Since the total energy budget itself cannot
  change with `NUM_CELLS` (set purely by `E_SN`, `P_AMBIENT`, `BOX_SIZE` in
  `inject_crs_at_shocks`-independent code), this means the *absolute* amount of CR energy
  `inject_crs_at_shocks` injects over a full run is not resolution-converged.
- Root cause not identified -- reasoned through the `delta_e_cr_density = efficiency *
  thermal_energy_flux / grid_spacing * dt` formula (`cr_grey_injection.py`) by hand two different
  ways and got two different, equally plausible, opposite-sign explanations: (a) 48^3 under-
  resolves the shock jump that `find_shocks_pfrommer`'s Rankine-Hugoniot-based
  `thermal_energy_flux` depends on, so 48^3 *under*-injects and 128^3 is closer to converged; or
  (b) the injection formula's `1/grid_spacing` normalization (calibrated for a nominally
  one-cell-thick shocked shell) doesn't fully cancel the growth in flagged shock-surface-cell
  count at finer resolution, so 128^3 *over*-injects. Dimensional analysis alone doesn't
  distinguish these -- would need actual instrumentation (shocked-cell counts and per-cell
  `thermal_energy_flux` values at both resolutions, the same style of per-step budget
  instrumentation used to crack item 9's Finding 2) to settle it. Not attempted this session --
  flagged instead of guessed, per this module's usual practice.
- **This affects more than item 11**: every "X% of energy went to CR" number quoted for items 7,
  8, and 10 (Sedov ~5.3%, wind bubble ~1.6%, item 11's own 2.44%/5.07% above) was calibrated at a
  single, specific `NUM_CELLS` and should now be read as resolution-specific, not as a converged
  physical prediction, until this is actually investigated.

**Next step (user-scheduled, not yet run):** the user plans to rerun the same test at
`NUM_CELLS=256` next. When that happens, extend the table above with the 256^3 numbers (`E_cr`
uniform-DSA/clumpy-DSA, the fractional clumps-effect, and the `E_cr`/`E_total` fraction) and
check whether the trend (`2.44% -> 5.07%` so far) is converging, still growing, or oscillating --
that will narrow down which of the two hypotheses above is more likely, or rule out both. Ideally
capture the run's real printed diagnostics (shocked-cell count, per-cell flux) directly this time
rather than decoding the plot. Item 11's committed test itself is unaffected for now (its
`NUM_CELLS=48` default and calibrated tolerances are unchanged and still self-consistent at that
resolution) -- this entry is a heads-up flag, not a blocker on the ladder.

## Where things stand (2026-09-11, ladder item 11 -- DONE, SNR into uniform/clumpy medium)

**Ladder item 11 ("SNR expanding into a uniform then a clumpy medium") is done.** New pytest
`pytests/cosmic_rays_grey/cr_snr_clumpy_medium.py`. Three design decisions were resolved with the
user up front via concrete options (same pattern as items 4/6/8/9/10): **(1) clumpy-medium
generation** -- discrete spherical clumps (generalizing the Sedov test's own tanh-taper `weight`
pattern to off-center positions), not a turbulent lognormal field (the codebase's
`create_turb_field` Gaussian-random-field generator exists but is currently unused anywhere and
would have added unvalidated spectral-parameter choices with no precedent); **(2) uniform
baseline** -- a new physically-scaled SNR setup (real `E_SN=1e51` erg, `n=1 cm^-3`, `T=1e4` K ISM,
astropy units + `CodeUnits`, matching item 10's move away from item 7's toy code-unit setup), not
a reuse of item 7's setup as-is; **(3) validation** -- the established item-7/9/10 differential
CR energy-partition identity (Check 1) PLUS a second "clumps-matter" signature check (Check 2)
comparing clumpy-DSA vs. uniform-DSA `E_cr`, since no analytic reference solution exists for a
CR-DSA shock breaking through an inhomogeneous medium.

**Setup (calibrated 2026-09-11):** 3D Cartesian, non-MHD, FV, `NUM_CELLS=48`, box 20 pc physical
(`CODE_UNITS` base length 5 pc), `t_end=1000` yr (well inside the adiabatic Sedov-Taylor phase for
these parameters -- free-expansion ends after a few hundred yr, radiative-phase onset is
`~3e4` yr, and there's no cooling term in this module anyway). Explosion: `E_SN=1e51` erg
deposited as a smooth pressure bump at the box center (tanh-tapered over `SMOOTH_CELLS=2`,
generalizing item 7's formula to physical units). Clumpy medium: three spherical clumps
(`CLUMP_RADIUS=1` pc, density contrast 10x ambient, tanh-tapered over 2 cells), centered 3 pc from
the explosion along three deliberately non-axis-aligned directions (to break spherical symmetry)
-- **density-only** perturbation, gas pressure held uniform everywhere (including inside clumps),
per the user-confirmed design: this gives clumps a lower local sound speed (higher local Mach
number) than the surrounding ambient gas when the shock arrives, a concrete checkable signature.
DSA: `dsa_efficiency in {0, 0.1}` (control/DSA), `dsa_mach_min=1.3`, matching items 7/10.

**Results, all four runs (uniform/clumpy x control/DSA):**
- **Containment:** forward-shock max radius 1.431 (uniform) / 1.210 (clumpy) code units vs.
  domain half-width 2.0 -- comfortable margins (`containment_margin=0.4` calibrated).
- **Total-energy conservation** (new check, not present in earlier items' clumpy analog since this
  is the first item with an inhomogeneous initial density): identical initial energy budget
  (`E_total_initial ~= 5031.59` code units) across *all four* runs, because ambient **pressure**
  (not density) sets the initial thermal energy and pressure is uniform even in the clumpy
  configuration -- a useful internal-consistency confirmation that the clump construction is
  density-only as intended. Observed conservation error `~1.5e-6`, `tol=1e-3`.
- **Check 1 (energy-partition identity), per medium:** uniform rel. err `~6.5e-4`, clumpy rel.
  err `~2e-4`, both well inside `tol=1e-2`.
- **Check 2 (clumps-matter signature):** `E_cr` = 122.82 (uniform-DSA) vs. 100.82 (clumpy-DSA)
  code units -- a clear **17.9% decrease**, `tol=0.05` (>3.5x margin). See the calibration note
  below for why this is a decrease, not the naively-expected increase.

**A real physical finding, not a bug (calibration note, matches this test's module docstring):**
the initial hypothesis was that clumps' lower local sound speed (fixed pressure, 10x density)
would *raise* the local shock Mach number and so *increase* total DSA injection. The
locally-elevated-Mach part is confirmed directly in the data (clumpy run's minimum
shock-surface Mach number is `~1.9-2.0` vs. uniform's `~1.35`, same `dsa_mach_min=1.3` threshold
in both). But the *net*, whole-surface effect on total injected `E_cr` goes the other way: the
clumps' extra inertia measurably slows the overall forward shock (max shock radius ~15% smaller
at the same `t_end`) and the shock finder flags `~11%` fewer total shock-surface cells
(`num_shocks`: 1809->1610-1618) -- so less total thermal energy is processed through the whole
shock surface within a fixed `t_end`, and less is available to divert into CR at fixed
`dsa_efficiency`. This net decrease wins over the per-clump Mach increase. Not investigated
further (out of scope for this item -- it's a real prediction of the injection model as built,
not a numerical artifact: NaN checks pass, energy is conserved to `~1.5e-6`, and the effect is
robust/reproducible in the single calibrated configuration run). Worth keeping in mind for Phase
C's later SNR-cloud emission work (item 15) -- a clumpy ambient medium may plausibly *reduce*
total CR power output relative to a uniform medium of the same mean density, not increase it, at
least in this regime (moderate 10x density contrast, `t_end` short enough that only a few clumps
are engulfed).

**No shared simulation code was touched for this item** -- purely a new test exercising existing
infrastructure (`inject_crs_at_shocks`/`find_shocks_pfrommer`, the CR-grey feedback sources) in a
new physical setup, same pattern as item 10, so items 1-10 are unaffected by construction (not
re-run this session).

**Next: ladder item 12** (reproduce a published grey CR-ISM result as a code-comparison anchor,
e.g. a SILCC-style stratified box, Girichidis et al. 2016; Simpson et al. 2016) -- the ladder's
integration/physical group (items 9-11) is now fully complete.

## Where things stand (2026-09-10, ladder item 10 -- DONE, wind-blown bubble)

**Ladder item 10 ("wind-blown bubble with CR pressure") is done.** New pytest
`pytests/cosmic_rays_grey/cr_wind_bubble.py`. Three design decisions were resolved with the user
up front (asked via concrete options, same pattern as items 4/6/8/9): **(1) geometry** -- 3D
Cartesian (not the cheaper 1D spherical alternative also on the table), matching
`examples/stellar_wind/stellar_wind_3d.py`'s existing precedent; **(2) CR injection** -- reuse
items 7/8's shock-DSA machinery (`inject_crs_at_shocks`/`find_shocks_pfrommer`) at the wind's
forward shock, *not* a new dedicated "wind CR source term" (the plan Sec. 2's other, distinct
injection channel) -- so no new injection code was needed, only a new test exercising the
existing mechanism in a new physical setup; **(3) validation** -- "Weaver baseline + differential
CR check" (lighter-weight, reuses item 7's pattern) over deriving a whole new CR-modified
analytic wind-bubble solution (the item-9-style heavier option).

**Closed a real, previously-untested gap: `astronomix/_modules/_stellar_wind/weaver.py`
(the classic Weaver et al. 1977 analytic wind-bubble solution) existed in this repo already but
had never been exercised by any pytest.** No plain-hydro wind-module test existed at all before
this session (only an uncalibrated example script,
`examples/stellar_wind/stellar_wind_3d.py`, MHD/FD). Calibrated setup (single EI-scheme wind
source, 3D Cartesian FV non-MHD, `NUM_CELLS=64`, box 3.6 pc physical, `t_end=1e4` yr, `M_star=40
Msun`, `v_inf=2000` km/s, ambient `n=2 cm^-3`/`P/k_B=3e4 K cm^-3`, `num_injection_cells=4`):
the control run's forward-shock radius (found via `find_shocks_pfrommer`, same mechanism used for
DSA injection) matches Weaver's analytic `R_2(t)` to **<1%** (0.42% observed); the shocked-wind
interior pressure plateau matches Weaver's analytic value to ~19%. The inner (wind-termination)
shock is deliberately *not* checked quantitatively against Weaver's `R_1(t)` -- confirmed
directly that at this resolution the shock finder's innermost detection sits right at the wind
source term's own injection-region edge (`num_injection_cells * grid_spacing`), not the
physically resolved termination shock; a resolution limitation of a small
(`num_injection_cells=4`) source region, not a bug, and out of scope to chase further for this
item.

**A real, pre-existing finding surfaced while calibrating, not fixed (unrelated to CR-grey,
out of scope):** the wind module's 3D thermal-energy-injection scheme (`_wind_ei3D`,
`astronomix/_modules/_stellar_wind/stellar_wind.py`) normalizes injected power by the *nominal*
spherical injection volume (`4/3 * pi * (num_injection_cells * grid_spacing)**3`) but the actual
per-cell injection mask uses a conservative radius shrunk by half a grid cell (`dist <=
injection_radius - grid_spacing / 2`). At small `num_injection_cells` this is a large fractional
volume mismatch: verified directly that at `num_injection_cells=4` the shrunk-vs-nominal volume
ratio is `(3.5/4)**3 ~= 0.67`, and the measured total injected energy (final thermal+kinetic+CR
minus initial ambient thermal, over the ~1e4 yr run) came out at `~66%` of the nominal `L_w *
t_end` wind luminosity -- matching this predicted ratio almost exactly. Because of this, the new
test does **not** assert an absolute energy budget against the nominal `L_w * t_end` formula (it
would fail by a wide, resolution-dependent margin that has nothing to do with CR-grey
correctness). Used two budget checks insensitive to this bias instead: (a) the item-7-style
differential CR-partition identity (thermal+kinetic energy diverted from the control run equals
the DSA run's `E_cr`) -- holds to ~2.4e-4 relative error; (b) a tight self-consistency check that
the control and DSA runs' own *realized* total energy agree with each other (both draw from the
same biased budget; `dsa_efficiency` only redirects ~1.6% of it into `e_cr`) -- holds to ~4e-6
relative error. `E_cr` is ~1.6% of the DSA run's own realized total energy at
`dsa_efficiency=0.1`, `dsa_mach_min=1.3`, forward-shock Mach ~2.9 -- clearly nonzero, physically
bounded, and lower than item 7's Sedov blast (~5.3%) as expected for a much weaker shock. This
volume-normalization bias is worth flagging as a real limitation of `_wind_ei3D` if the wind
module is ever revisited (e.g. for Phase C SNR/wind emission work, ladder item 11), especially at
small `num_injection_cells` -- not investigated further here since it's pre-existing and outside
this module's scope.

**No shared simulation code was touched for this item** -- purely a new test exercising existing
infrastructure (`_wind_ei3D`, `inject_crs_at_shocks`/`find_shocks_pfrommer`, the CR-grey feedback
sources) in a new physical setup, so items 1-9 are unaffected by construction (not re-run this
session, since nothing they depend on changed).

**Next: ladder item 11** (SNR expanding into a uniform then a clumpy medium) -- Phase B/C's
integration-ladder items 9-10 are now both complete.

## Where things stand (2026-09-09, ladder item 9 -- DONE, via a moving-shock redesign)

**Ladder item 9 is now done.** Picking back up from the same-day entry below (Finding 2's
mechanism understood, three candidate fixes identified, user asked to choose before
implementing): discussed the tradeoffs of all three candidates with the user (physical
accuracy, whether multi-cell injection is inherently less accurate, implementation
difficulty) and the user chose **(c)**, redefining the test around a genuinely moving shock
rather than one pinned motionless in the lab frame -- confirmed as not just the easiest
option but plausibly the most physically correct one, since "steady CR-driven flow" in the
DSA literature means steady *in the shock's own comoving frame* (gas flows through a fixed
precursor profile, each parcel spending a finite time in it), not a shock nailed to one grid
cell while CR energy accumulates in it forever.

**Confirmed directly (ad hoc script, not committed) that this fixes Finding 2.** Built a
Rankine-Hugoniot "piston" IC -- an exact steady-shock jump launched at a chosen nonzero,
constant velocity into a quiescent ambient medium, so it genuinely propagates rather than
needing an inflow boundary condition -- with `diffusive_shock_acceleration` and
`diffusive_relaxation` both on. Fitted shock velocity from several snapshots matches the
nominal value to <0.1% (confirms genuinely constant-velocity propagation, not drift), and
peak `e_cr` at the shock **saturates to a bounded value** (e.g. `0.0100 -> 0.0119 -> 0.0125
-> ... -> 0.0133`, flattening) instead of growing without bound the way the stationary-shock
IC did -- the core mechanism (each cell only hosts the actively-injecting shock surface
briefly before the shock advances to the next one) works exactly as the item-8-and-earlier
"always expanding into fresh gas" pattern already relies on.

**Calibrating the redesign surfaced a further, previously-unappreciated interaction with
Finding 1 (still open, not fixed here).** The reference precursor ODE
(`cr_precursor_ode_rhs`, built the prior session) only holds once `F_cr`'s relaxation rate
`nu = reduced_streaming_speed^2 / diffusion_coefficient` is fast relative to the precursor's
own advective formation rate `omega ~ v_shock / precursor_width`. Working through both rate
expressions shows `nu / omega = (gamma_cr - 1) * (reduced_streaming_speed / v_shock)^2` --
independent of `diffusion_coefficient` entirely -- so a resolvable, genuinely quasi-steady
precursor requires `reduced_streaming_speed` several times the shock velocity, *regardless*
of how `diffusion_coefficient` is chosen. That is almost exactly Finding 1's own
already-documented bad regime (large `reduced_streaming_speed` relative to local gas signal
speeds). Measured directly: raising `reduced_streaming_speed` from 4x to 8x the shock
velocity improves the precursor-ODE match (median rel. err on `dP_cr/dx`/`dF_cr/dx` drops
from ~230% to ~20%, as expected from a deeper quasi-steady margin) but *worsens* the subshock
jump match (RH-jump errors on density/velocity/pressure stay at 24-42% at both settings, and
`F_cr`'s jump error grows from ~4% to ~148%) and further degrades the measured shock Mach
(2.0 nominal -> 1.4, then -> 1.3, right at `dsa_mach_min`); at 16x the shock velocity, the
shock degrades below `dsa_mach_min` and disappears entirely by the run's end. **A single run
cannot quantitatively validate both reference formulas (`cr_precursor_ode_rhs` and
`modified_rankine_hugoniot_with_cr_injection`) at once without first fixing Finding 1** --
confirmed with the user, who chose to split into two independent tests (same "independent
layers" pattern ladder item 8 already used) rather than force one compromise
`reduced_streaming_speed` onto both, loosen tolerances past the point of being diagnostic, or
tackle Finding 1's Riemann-solver fix as a prerequisite.

**Also found and fixed a real bug in the new reference solver's root-finding**
(`modified_rankine_hugoniot_with_cr_injection`, uncommitted since the prior session, so fair
game to fix directly): with `injected_energy_flux > 0`, `residual(r=1) = +injected_energy_flux`
rather than exactly `0` (the plain-gas case's exact trivial root), which shifts a vestige of
that trivial root to a spurious crossing very close to `r=1`. The naive two-point
`brentq(residual, 1+eps, r_max)` then fails outright whenever the probed endpoints have equal
sign (both positive, straddling the real root in between) -- hit immediately once a
genuinely CR-precursor-modified upstream state (not the pristine ambient state the original
degenerate-limit validation used) was fed in. Fixed by scanning for sign changes across
`(1, r_max)` and taking the one closest to `r_max` (the real, well-separated, entropy-
producing compression root; the near-1 artifact is confined to a narrow neighborhood of `r=1`
by construction). Re-verified this doesn't regress the original validation: the degenerate
(zero `p_cr`/injection) limit still matches the ideal-gas RH ratio to ~1e-14 across Mach
1.5-7, and the previously-failing nonzero case now converges to a root where mass flux, total
momentum flux, and total energy flux (gas + CR) are all exactly conserved (matches to machine
precision), confirming it's the physical root, not a second spurious one.

**New pytest: `pytests/cosmic_rays_grey/cr_modified_shock_structure.py`**, two independent
test functions sharing one moving-shock IC helper:
- `test_cr_dsa_shock_jump` (**Test A**): `diffusive_shock_acceleration` only (no
  `diffusive_relaxation`), `reduced_streaming_speed=1.0` (Finding-1-safe, matches the local
  pre-shock sound speed). Validates `modified_rankine_hugoniot_with_cr_injection` against the
  simulation's own converged post-shock state (density/velocity/pressure match to <1.1%,
  `F_cr` to ~21% -- looser because, without `diffusive_relaxation`, nothing pins `F_cr` to a
  fixed post-shock plateau the way the strong-RH-conserved gas fields are; the reference
  value is only exact as an instantaneous jump condition right at the subshock). Also serves
  as this module's permanent regression guard for Finding 2 (asserts peak `e_cr` has
  plateaued, not drifted, in the second half of the run).
- `test_cr_precursor_ode` (**Test B**): `diffusive_relaxation` on, `reduced_streaming_speed=8.0`
  (deep quasi-steady, accepting the Finding-1-degraded, but still `> dsa_mach_min`, shock this
  requires). Confirms `mass_flux`/`momentum_flux` are genuinely close to constant across the
  sampled precursor window (<0.5% spread, comfortably inside the 2% tolerance) before
  cross-checking the simulation's own finite-difference precursor gradients against
  `cr_precursor_ode_rhs` (median rel. err ~20-41% across the three quantities, calibrated
  tolerances documented in the test's own docstring with the reasoning above).

Full regression: all 9 pre-existing `pytests/cosmic_rays_grey/*.py` scripts (ladder items
1-8 + `cr_gradient_check.py`) still pass unchanged; the new test passes with the tolerances
above.

**Finding 1 remains open** (unaffected by this session -- see DESIGN.md's "Open questions").
This session's own calibration work (above) is a second, independent illustration of its
cost, beyond the original Mach-degradation measurement: it directly limits how tightly a
resolved shock combined with a fast CR-diffusion relaxation rate can be validated, which may
be relevant again for Phase C wind/SNR work (ladder items 10-11) if those also combine a
resolved shock with diffusive CR transport.

**Next: ladder item 10** (wind-blown bubble with CR pressure, the first Phase C
emission-adjacent target) -- Phase A/B's core ladder (items 1-9) is now complete.

## Where things stand (2026-09-09, ladder item 9 -- Finding 2's budget instrumented, still blocked)

**Item 1 of the 2026-08-27 "Next steps" list (instrument the per-step `e_cr` budget at the
shock cell) is done.** Ad hoc script, not committed, per this module's established practice:
reproduced Finding 2's setup (stationary Mach-2.5 shock, ideal-gas Rankine-Hugoniot IC,
`N=1600`, `reduced_streaming_speed=1.5`, `dsa_efficiency=0.1`, `diffusive_shock_acceleration`
only, no `diffusive_relaxation`) and drove it one fixed `dt` at a time (`fixed_timestep` +
`return_snapshots` with `num_snapshots == num_timesteps`, `dt` picked below the CFL estimate so
every recorded snapshot is exactly one internal step apart -- confirmed the `SnapshotData` this
produces really is one state per `_step()` call, not a coarser sample). At the tracked cell, each
step's actual `e_cr` change was decomposed as `injected` (`inject_crs_at_shocks`'s own delta) +
`adiabatic_work * dt` (`cr_adiabatic_work_source`'s `-P_cr * div(v)`, evaluated explicit-Euler on
the pre-step state) + `residual` (everything else, dominated by advection) -- this closes exactly
to the measured actual change every step (to float32 precision), so the decomposition itself has
no accounting bug.

**The box-model estimate broke down for three compounding reasons, not "weak advective
removal" (the original hypothesis):**

1. **`cr_adiabatic_work_source`'s `-P_cr * div(v)` term (always active, not gated by
   `diffusive_relaxation`) is a genuine, substantial, self-reinforcing source at the shock's
   finite-width numerical transition zone -- proportional to the CR pressure already sitting
   there, so it does not saturate on its own.** Measured directly: by `t~=0.056` in this repro,
   `adiabatic_work * dt` (`8.8e-3`) is the same order of magnitude as `injected` (`1.4e-2`) at the
   active cell, not a small correction. This is the *same* physical term ladder item 2 verified
   gives the correct `P_cr propto rho^gamma_cr` adiabatic invariant for a fluid parcel that
   transits a compression zone once -- but at a near-stationary shock the Eulerian compression
   zone never releases the cell sitting in it, so nothing bounds the term the way a finite
   transit time would.
2. **The shock finder's detected surface cell is not perfectly static, even for an exact
   Rankine-Hugoniot stationary IC -- it creeps downstream by exactly one grid cell at a time.**
   Confirmed directly: printing `shock_surface_cells`/`mach_numbers` in a 12-cell window each
   step around two observed transitions (cell 799->800 at step 1915 of 1968, `t~=0.0507`; an
   earlier 799->800-equivalent transition at `t~=0.0137` in a longer run) shows a single flagged
   cell (`identify_shock_surface`'s "one cell of max compression per zone" design) jumping
   discretely, with the Rankine-Hugoniot Mach number at the flagged cell *decreasing* at each
   new site (`2.17 -> 2.04` across the observed jump) -- consistent with the shock continuously
   losing energy (to DSA injection itself, and to Finding 1's HLL-wave-speed-inflation
   dissipation), so whatever compression-peak criterion the finder tracks slowly drifts, and
   snaps to a new cell once the peak crosses a cell boundary.
3. **Because the drift is real relocation, not injection re-appearing from nothing, each newly
   adopted surface cell inherits a head start of CR pressure already built up there via ordinary
   advection from its former-injection-site neighbor, on top of which direct DSA injection then
   begins.** A single-fixed-cell box model implicitly assumes a static injection site starting
   from `e_cr=0`, which undercounts this cross-cell hand-off.

None of this means advection is failing to remove `e_cr` -- the measured `residual` (net
advective divergence) is large and *negative* (strong outward removal) through most of the run
once a cell is actively injecting (e.g. `injected=1.4e-2`, `adiabatic_work*dt=8.8e-3`,
`residual=-1.9e-2` around `t~=0.056`) -- removal is working about as hard as a simple estimate
would predict; it is just outpaced by the *combination* of direct injection, the self-reinforcing
compressive term, and the cross-cell hand-off above, none of which a naive box model accounts for.

**Still blocked, not fixed.** This explains the mechanism but does not yet suggest an obviously
correct fix -- candidates not yet evaluated: (a) make `cr_adiabatic_work_source` (or DSA
injection) aware of residence time / only apply the compressive term away from actively-injecting
shock cells; (b) have `inject_crs_at_shocks` (or the finder) spread injection/detection over the
whole resolved shock *zone* (`identify_shock_zones`'s ~3-4-cell criterion) rather than a single
peak cell, so the "single fixed cell, no memory of predecessor" box-model mismatch in point 3
above goes away by construction; (c) revisit whether a genuinely *stationary* shock (as opposed
to the always-expanding shocks every other DSA test in this module uses) is a physical
configuration this injection scheme is meant to support at all -- the plan's own ladder item 9
wording ("1D steady CR-driven flow") may need to mean *slowly advecting*, not perfectly at rest,
in which case the test design itself (not the injection code) is what should change. Next
session: pick one of (a)/(b)/(c) with the user before implementing, since each is a real
design decision with more than one defensible answer (same category of choice ladder items 2, 4
and 8 all stopped to ask about).

## Where things stand (2026-08-27, ladder item 9 -- BLOCKED, two real findings)

**Ladder item 9 ("1D steady CR-driven flow / CR-modified shock structure") is not done and the
pytest was never written.** Scope agreed with the user up front: a self-consistent DSA-driven
precursor (run a shock with `diffusive_shock_acceleration` + `diffusive_relaxation` both on
together for the first time, let a diffusive CR precursor build upstream self-consistently,
validate against a new semi-analytic reference). That reference solver was built and
independently validated, but setting up the actual simulation surfaced two real, previously-
latent numerical findings that block the test as designed. Both are documented here rather than
worked around silently, per the user's explicit direction to stop and write them up.

**What's built and trustworthy:** new
`astronomix/test_setups/reference_solutions/cr_modified_shock_structure.py` (plain numpy/scipy,
same convention as `pfrommer_riemann_solver.py`), derived directly from this module's actual
coupled equations (not textbook forms) by tracing through `grey_cr_flux_terms`,
`cr_pressure_gradient_source`, `cr_adiabatic_work_source`, `cr_flux_relaxation_source`'s
quasi-steady limit, and `inject_crs_at_shocks`/`calculate_thermal_energy_flux` directly:

1. `cr_precursor_ode_rhs` -- the steady-state ODE (`du/dx`, `dP_cr/dx`, `dF_cr/dx`) the smooth
   precursor upstream of a subshock must satisfy, given the two exact first integrals `mass_flux
   = rho*u` and `momentum_flux = rho*u^2 + P_gas + P_cr`. Key derivation subtlety: `e_cr`'s actual
   coded flux is the bare `u*e_cr + F_cr` (`grey_cr_flux_terms`), *not* the enthalpy-like `u*P_cr`
   term a textbook two-fluid treatment would use -- using the wrong (enthalpy) form gives the
   *wrong* adiabatic exponent when cross-checked against ladder item 2's already-verified
   `P_cr propto rho^gamma_cr` invariant; the bare-advection form is the one that's actually
   consistent with this codebase.
2. `modified_rankine_hugoniot_with_cr_injection` -- the (mass, momentum, energy) jump across the
   subshock itself, given the CR energy flux DSA injects there. Key derivation result: `P_cr` is
   continuous through the subshock (collisionless, doesn't thermalize on the gas's collisional
   length scale, and diffusion doesn't create a discontinuity in a quantity it's smoothing), which
   makes it cancel out of the momentum jump entirely, leaving the *ordinary single-fluid*
   mass+momentum Rankine-Hugoniot relations for `(rho, u, P_gas)` -- but gas energy loses exactly
   the DSA-injected flux (a "radiative-shock-like" jump with a known energy sink), solved via a
   1D root-find over the compression ratio.

**Validation done before trusting either function** (ad hoc scripts, not committed, per this
module's established practice -- see ladder item 6's Pfrommer-solver validation for the
precedent):
- `modified_rankine_hugoniot_with_cr_injection` reduces *exactly* to the standard ideal-gas
  Rankine-Hugoniot jump (density/pressure ratio formulas) when `P_cr`/injection are zeroed, across
  Mach 1.5-10 (`~1e-8` agreement, root-finder precision).
- For a general (nonzero `P_cr`, nonzero injection) case, mass flux, total momentum flux, *and*
  total energy flux `E = (1/2) rho u^3 + gamma_gas/(gamma_gas-1) P_gas u +
  gamma_cr/(gamma_cr-1) P_cr u + F_cr` are all exactly conserved across the computed jump (checked
  to `~1e-9`) -- a genuine, non-trivial cross-check, since `F_cr`'s post-shock value was derived
  from the *individual* CR-only energy equation, independently of the gas-only jump used for
  `(rho2, u2, P_gas2)`, so exact total-energy conservation confirms the two pieces are mutually
  consistent rather than hard-coded to agree.
- `cr_precursor_ode_rhs`: linearized the ODE around the exact upstream fixed point `(u1, 0, 0)`,
  found the growing eigenmode (`F_cr = -[u1/(gamma_cr-1)] * P_cr`, growth rate `u1 /
  (diffusion_coefficient*(gamma_cr-1))`), and numerically integrated (`scipy.integrate.solve_ivp`,
  tight tolerances) from a small perturbation along that exact direction. Total energy flux `E`
  (same formula as above) stays constant to `~1.6e-7` relative error along the resulting
  trajectory even after `u` drops 30% and `P_cr` grows four orders of magnitude -- again a
  non-trivial check, since `du/dx` (from the gas equation) and `dF_cr/dx` (from the CR equation)
  were derived independently and only *have* to sum to zero net change in `E` if both are correct.

**Finding 1: `reduced_streaming_speed` shares the Riemann-solver wave-speed bound with the gas,
which measurably degrades shock-capturing whenever it exceeds local gas signal speeds --
independent of resolution, independent of whether any CR is actually present.** `grey_cr_fast_speed`
always contributes `reduced_streaming_speed` to `c = jnp.maximum(c_gas, grey_cr_fast_speed(...))`
in `hll.py`'s `S_L`/`S_R` estimates (`registered_variables.cosmic_ray_e_active` gates this, not
whether `e_cr` is actually nonzero anywhere). Reproduced directly: a stationary Mach-4 shock (exact
gas-only Rankine-Hugoniot IC, `grey_cosmic_rays=True` but zero `e_cr` everywhere the whole run) is
measured by `find_shocks_pfrommer` at Mach 3.81 with `reduced_streaming_speed=1.0` (the default),
degrading monotonically to Mach 1.64 at `reduced_streaming_speed=8.0` (item 4's calibrated value) --
**with `max|e_cr| = 0.0` confirmed throughout**, so this is a pure Riemann-solver-accuracy effect,
not real CR physics. Ruled out grid resolution as an explanation: 4x finer (`N=800` to `N=3200`)
gave essentially identical degradation (Mach 1.6414 vs. 1.6468), which rules out a simple
"numerical diffusion width `~ wave_speed * dx`" picture (that would improve with resolution).
Mechanism: HLL's flux formula has a term `S_L*S_R*(U_R-U_L)/(S_R-S_L)` that does *not* vanish even
when `F_L = F_R` exactly (the definition of a stationary shock's RH condition) -- this term's
magnitude scales with `|S_L*S_R|`, so an inflated, state-independent wave-speed floor directly
adds spurious dissipation to an otherwise-exactly-resolved discontinuity, and this is a property
of the HLL flux formula itself (not fixed by refining the grid). This wasn't visible in any prior
ladder item because none combined a genuine propagating/standing shock with a
`reduced_streaming_speed` large compared to the local gas signal speed at the same time (item 4's
own large-`v_red` calibration used a smooth Gaussian diffusion test, no shock; items 7/8's
Sedov-Taylor DSA tests used the default `reduced_streaming_speed=1.0`, comparable to or below the
blast's own speeds). **Not fixed** -- would require giving the Riemann solver separate gas/CR
wave-speed bounds instead of one shared `max(...)`, a real design change to shared FV code, out of
scope to do unprompted. Practical consequence for any future shock+diffusion test in this module:
keep `reduced_streaming_speed` close to (not many times above) the local gas `|u|+c`, and get a
fast relaxation rate `nu = reduced_streaming_speed^2 / diffusion_coefficient` via a small
`diffusion_coefficient` instead of a large `reduced_streaming_speed`.

**Finding 2: sustained, repeated DSA injection at a *stationary* shock cell grows without bound,
confined to a fixed few-cell width -- present with or without `diffusive_relaxation`.** Set up
per Finding 1's mitigation (`reduced_streaming_speed=1.5`, small `diffusion_coefficient=0.05`,
`N=1600`, a Mach-2.5 stationary-shock IC): `diffusive_shock_acceleration` alone (no
`diffusive_relaxation`, undamped `F_cr` wave equation, pure advective removal) still shows the
peak `P_cr` growing roughly exponentially in time (`t=0.05: 2.1e-5`, `t=0.12: 8.3e-5`, `t=0.2:
4.0e-4`, `t=0.4: 1.6e-2`) while its *spatial width stays pinned at exactly 5 cells the entire
time* -- i.e. this is not a genuinely spreading diffusive precursor slowly saturating, it's
amplitude growing at a fixed location. Adding `diffusive_relaxation` back (the original item-9
config) shows the same fixed-width/growing-amplitude pattern, just slower. A simple box-model
estimate (roughly constant injection rate at roughly constant measured Mach, `~5.6%` of the cell's
`e_cr` advected out per step at this resolution/CFL) predicts convergence to a finite steady value
via a geometric series -- not the observed unbounded growth -- so either the actual advective
removal of `e_cr` right at a strong, near-stationary discontinuity is weaker than that estimate
(a plausible HLL-flux effect, not yet confirmed), or there is some other discrete-injection
feedback not yet isolated. Measured shock Mach stayed essentially flat (`1.7188 -> 1.7143`)
across this entire growth, ruling out "the growing `P_cr` is progressively weakening the shock,
which increases the DSA efficiency, which is a real physical runaway" as the mechanism (that would
require Mach to visibly change, and it barely does at these still-small `P_cr` magnitudes,
`< 0.4%` of `P_gas` at the largest tested amplitude). **Root mechanism not found** -- the user
asked to stop debugging further and log the finding rather than continue an open-ended
investigation, given how much runtime/effort the session had already spent (see Finding 1 above
for the other real finding from the same investigation). This is new territory: every existing
DSA test (`cr_sedov_taylor.py`, `cr_dsa_mach_dependence.py`) has a shock that is *always expanding
into fresh gas*, so no single cell there ever receives more than a handful of injection events --
a genuinely *stationary* shock receiving *sustained* repeated injection has never been exercised
in this module before, and this looks like a real gap in either the injection mechanism or its
interaction with the FV solver at a discontinuity, not a test-design mistake.

**Next steps for ladder item 9, in order:**
1. Instrument the actual per-step `e_cr` budget at the shock cell directly (injected amount vs.
   advected-out amount vs. actual `e_cr` after the step) to find exactly where the simple box-model
   estimate in Finding 2 breaks down -- this is the concrete, bounded next action, more useful than
   another hypothesis-and-rerun cycle at this point (same lesson `FIXES_TODO.md`'s CWB item 4
   converged on after a similar number of ruled-out guesses).
2. Once Finding 2 is understood/fixed, re-attempt the item-9 test using Finding 1's mitigation
   (modest `reduced_streaming_speed`, small `diffusion_coefficient`, correspondingly higher
   resolution to keep the precursor resolved) and the already-validated reference solver above.
3. Consider whether Finding 1 (shared gas/CR wave-speed bound degrading shock-capturing) deserves
   a separate, scoped fix independent of item 9 -- it plausibly affects the *accuracy* (not just
   this new test) of any past or future ladder item that combines a resolved shock with
   `reduced_streaming_speed` notably above local gas speeds; no existing ladder item's assertions
   are known to depend on shock sharpness in a way this would break, but it hasn't been checked
   item-by-item.

## Where things stand (2026-08-26, ladder item 8 -- Mach-dependent DSA efficiency)

**Implementation done, smoke-verified; a dedicated calibrated pytest (matching
item 7's `cr_sedov_taylor.py` rigor) is not yet written** -- that's the actual
next step, not a fresh ladder item.

- New `dsa_efficiency_kang_ryu_2013` in `cr_grey_injection.py`: a piecewise
  fit (weak/intermediate/strong-shock pieces, `Ms<2 -> 0`, plateau `0.211` for
  `Ms>15`) to Kang & Ryu (2013)'s kinetic DSA simulation results, in the form
  standard across the cluster-shock CR literature (Vazza et al. 2016;
  CRESCENDO, Girichidis et al. 2022) -- KR13 itself only publishes
  tables/figures, not a closed-form fit. Verified by hand: continuous across
  both piece boundaries (~2% at Ms=5, <1% at Ms=15) and consistent with KR13's
  own stated asymptote (eta -> ~0.2 for Ms gtrsim 10). Caprioli & Spitkovsky
  (2014) has no independent fit either; implemented as the literature-standard
  approximation, half of KR13 (`dsa_efficiency_mach_scale=0.5`), rather than a
  separate function.
- New `CosmicRayGreyConfig.dsa_efficiency_model` (`DSA_EFFICIENCY_CONSTANT`
  default | `DSA_EFFICIENCY_KANG_RYU_2013`) and
  `CosmicRayGreyParams.dsa_efficiency_mach_scale` (default 1.0). Added as an
  opt-in mode specifically so item 7's calibrated `cr_sedov_taylor.py` and its
  tolerances are untouched -- user confirmed this design (both "which
  model(s)" and "how to wire it in") when asked, since it's a genuine
  API/physics decision with more than one defensible answer.
- Differentiability guard: the intermediate piece divides by `Ms**4`; a naive
  implementation is NaN-safe forward (masked by `jnp.where`) but NaN-*unsafe*
  backward (jnp.where differentiates every branch; an unselected `1/0**4` at
  `Ms=0`, the common "no shock here" value, has infinite local gradient,
  contaminating the total via `0 * inf = nan`) -- same category of bug this
  module has hit before (`cr_pressure_speed_floor`). Fixed by flooring the
  Mach number used *inside* the discardable branches (`ms_safe =
  jnp.maximum(ms, 1.0)`) rather than the one driving branch selection.
  Verified directly: `jax.grad` through the function at `Ms=0` gives exactly
  `0.0`, not NaN.
- Smoke-tested standalone (N=32 Sedov setup, not the calibrated N=48 test):
  no NaN; KR13 gives `E_cr ~= 2.7%` of total energy (vs. item 7's flat-10%
  model's ~5.3% -- lower is expected, since KR13's efficiency only nears 10%
  around Ms~5 and this blast's shock weakens over time); the CS14-like
  scale=0.5 run gives almost exactly half that (`~1.3%`), confirming the
  scale wiring.
- Full regression: all 7 pre-existing ladder pytests + `cr_sedov_taylor.py`
  (item 7) still pass.

**Found and fixed an unrelated but real regression while verifying this**:
item 7's `cr_sedov_taylor.py` started going to all-NaN, traced to a shock-finder
change from an unrelated CWB debugging session earlier the same day (the
"walk to shock-zone edge" fix to `get_post_pre_shock_values` /
`_shock_mach.py` / `pfrommer_shock_finder.py` -- `find_shocks_pfrommer` is
shared infrastructure between the CWB and CR-grey work). That fix had only
ever been verified against synthetic profiles, never a real run. Bisected and
reverted it (back to `HEAD`); see
`pytests/shock_finder3D/FIXES_TODO.md` item 1a for the full account. Not a
CR-grey bug, but worth knowing this module's tests can be broken by changes
elsewhere that touch the shock finder.

**Ladder item 8 is now fully done**, including the dedicated pytest
(`pytests/cosmic_rays_grey/cr_dsa_mach_dependence.py`), written same-day
(2026-08-26). Two independent layers, per the design question above:

1. `test_dsa_efficiency_kang_ryu_2013_shape` -- pure-function checks on the
   fit itself (zero below Ms=2, continuous at both piece boundaries,
   monotonic, asymptotes to 0.211). No simulation; exercises the whole Mach
   range a real blast can't guarantee sampling.
2. `test_cr_dsa_mach_dependence` -- reuses `cr_sedov_taylor.py`'s exact
   physical setup with three configs (control/KR13/CS14-like): no NaN,
   energy conservation, `E_cr` bounded and nonzero (calibrated ~4.8% KR13 /
   ~2.4% CS14-like at NUM_CELLS=48 -- comparable to, modestly below, item
   7's flat-10% model's ~5.3%; the control run's shock-surface cells span
   Ms~3-29, mean~11 at t=0.07, so most sit well above KR13's Ms~5
   "reaches 10%" point -- the total is dominated by the shock's whole,
   Mach-decreasing-over-time history, not this final snapshot, so no tight
   a priori match to the flat model is expected), the CS14-like/KR13 `E_cr` ratio close to 0.5
   (calibrated ~1.9% deviation -- **not** near-machine-precision, since
   these are two independent nonlinear time integrations: diverting
   different amounts of thermal energy into CRs feeds back into the gas
   pressure and hence the shock's subsequent trajectory), and an **exact**
   formula cross-check (the main check): call `find_shocks_pfrommer` and
   `dsa_efficiency_kang_ryu_2013` directly on the control run's final state,
   independently compute the expected `delta_e_cr`, and compare against
   `inject_crs_at_shocks`'s actual output on the same state -- matches to
   ~8e-8 (float32 precision), proving the injection code genuinely uses the
   documented formula rather than something else.

Two tolerance-calibration mistakes worth remembering if this test is ever
revisited: initially assumed the CS14/KR13 ratio would hold near machine
precision (wrong -- see the feedback explanation above; fixed by loosening
to 5%) and assumed the formula cross-check would hold to 1e-9 (wrong for
float32; fixed to 1e-5, still >100x margin over the observed ~8e-8). Neither
was a real bug, both were the test's own expectations being initially too
strict.

Full regression (all 7 pre-existing ladder pytests + `cr_sedov_taylor.py` +
this new test) passes.

## Where things stand (2026-08-24, ladder item 7 -- CR-DSA Sedov-Taylor blast)

**Ladder item 7 (CR Sedov-Taylor blast: thermal/CR/kinetic energy partition) passes for
real**, the first Phase B ladder item. Built fresh against `e_cr`/`F_cr` and the real PR #4
finder, per the consolidation decision below -- there was no old-model code to adapt.

- New `astronomix/_modules/_cosmic_rays_grey/cr_grey_injection.py`
  (`inject_crs_at_shocks`): runs `astronomix.shock_finder3D.pfrommer_shock_finder.
  find_shocks_pfrommer` every step, diverts `dsa_efficiency` of each shock-surface cell's
  `thermal_energy_flux` (already zero everywhere except those cells) from gas thermal pressure
  into `e_cr`. Injects at **every** detected shock-surface cell, not just "the strongest
  shock" the retired old model's 1D-only injection was limited to -- the N-D finder already
  returns a full per-cell boolean surface array, so no reduction to a single shock is needed.
  Cartesian/uniform-grid only (`area/volume = 1/grid_spacing`), matching every grey-CR ladder
  test to date. Energy-conserving by construction: `e_cr` gains exactly what gas pressure
  loses (`delta_e_cr * (gamma_gas - 1)`), density/velocity untouched.
- New config/params: `CosmicRayGreyConfig.diffusive_shock_acceleration` (off by default),
  `CosmicRayGreyParams.dsa_efficiency` (0.1 default, matches the retired old model's
  identical default), `dsa_start_time` (0.0, ad-hoc guard against spurious pre-shock
  detections, same purpose as the old model's identical knob), `dsa_mach_min` (1.3, matches
  `find_shocks_pfrommer`'s own default).
- Wired into `_iteration_level_updates.py` as a discrete once-per-step correction (same
  pattern as `streaming_flux_target`/`anisotropic_flux_projection`/the retired old model's
  injection), gated on `registered_variables.cosmic_ray_e_active and
  config.cosmic_ray_grey_config.diffusive_shock_acceleration`. No `_time_integrator_sources.py`
  changes -- DSA injection is a discrete detection+deposit operation, not a continuous PDE
  source term.
- **Test design** (`pytests/cosmic_rays_grey/cr_sedov_taylor.py`): a 3D Cartesian point
  explosion (N=48, `t_end=0.07`, matches `pytests/shock_finder3D/_sedov_setup.py`'s physical
  setup), run twice -- `dsa_efficiency=0` (control, DSA code path live but injecting nothing)
  and `dsa_efficiency=0.1`. Rather than compare to an analytic self-similar Sedov profile
  (this module's `cr_adiabatic_compression.py` already found that kind of global-profile
  comparison unreliable near open boundaries), the test checks a *differential* identity:
  since injection is energy-conserving by construction, `(E_thermal + E_kinetic)` lost by the
  DSA run relative to the control run must equal the DSA run's `E_cr` gained. Calibrated:
  this identity holds to ~2.1e-5 relative error (`tol=1e-2`, >450x margin); each run's own
  total-energy conservation (initial ambient + `E_EXPLOSION` vs. final
  thermal+kinetic+CR) independently holds to ~1.7e-5 (`tol=1e-3`); `E_cr` is ~5.3% of the
  total energy budget at `dsa_efficiency=0.1` (asserted in `[0.01, 0.3]`); the control run's
  `E_cr` is exactly `0.0` (confirms `dsa_efficiency` is a genuine off switch, not just "code
  path never called"); the shock front stays at `r~=0.455` vs. domain half-width `0.5`
  (asserted `< 0.47`, so the energy-budget checks aren't silently passing because energy
  already left through the open boundaries). `NUM_CELLS=48` was picked after confirming
  runtime cost empirically (~34s including JIT compile at N=48; ~24s at N=24) -- continuous
  per-step shock-finding turned out to be cheap enough that a true 3D setup (matching the
  plan's literal "Sedov-Taylor" wording) was affordable without needing a cheaper 2D
  substitute.
- Smoke-tested the mechanism standalone before committing to the full test (N=24 and N=48,
  `dsa_efficiency=0.1`): no NaNs, `e_cr` becomes nonzero and physically plausible in magnitude,
  `p_gas` floors correctly at the ambient value away from the shock.

Full regression: all 7 pre-existing `pytests/cosmic_rays_grey/*.py` scripts (ladder items 1-6
+ `cr_gradient_check.py`) still pass; `pytests/hydrodynamics/shock_tube1D.py` (FD + FV Pallas)
still passes; `import astronomix` clean.

**Next: ladder item 8** (DSA efficiency vs. Kang & Ryu 2013 / Caprioli & Spitkovsky 2014 as a
function of Mach number) only needs to replace `cr_grey_injection.py`'s scalar
`dsa_efficiency` with a function of `sf_result.mach_numbers` -- the injection mechanics built
here don't need to change.

## Where things stand (2026-08-24, old `_cosmic_rays` (`n_cr`) model retired)

**Resolved the consolidation open question (PROGRESS's own "next step" from the last session)
in favor of retiring the old model**, not keeping both. Grounds: the old `n_cr` model was
1D-only (explicit `NOTE`s in its own source), had **zero test coverage** anywhere in the repo
(no pytest ever turned on `cosmic_rays=True`/`diffusive_shock_acceleration=True`), and its DSA
injection (`inject_crs_at_strongest_shock`) called a legacy 1D-only shock finder
(`astronomix/shock_finder/shock_finder.py::find_shock_zone`) that is a *different, superseded*
module from the actual PR #4 finder the plan's ladder item 8 means
(`astronomix/shock_finder3D/pfrommer_shock_finder.py::find_shocks_pfrommer` -- genuinely N-D,
independently tested against Sedov/CWB setups). So "already wired up" was true only against a
shock finder Phase B was never going to use anyway. The registry also allowed both
`cosmic_ray_n_active` and `cosmic_ray_e_active` to be enabled simultaneously with no guard,
which would have double-counted CR pressure (old model folds `P_cr` into the shared
`pressure_index`; grey model tracks it as independent state). User confirmed this direction
explicitly (real design decision, not a bug fix) -- see DESIGN.md's "Consolidating with the old
`_cosmic_rays` model" note for the full writeup.

**Removed:** `astronomix/_modules/_cosmic_rays/` (whole module: `cosmic_ray_options.py`,
`cr_fluid_equations.py`, `cr_injection.py`) and `astronomix/shock_finder/` (the legacy 1D finder
-- its only consumer was the old model's injection code). Un-wired everywhere it touched shared
FV code: `CosmicRayConfig`/`CosmicRayParams` and their fields on
`SimulationConfig`/`SimulationParams`; `cosmic_ray_n_index`/`cosmic_ray_n_active` on
`RegisteredVariables` (and the matching re-index line in `evolve_state.py`'s
`_split_gas_and_magnetic_state`); the `*_with_crs` branches in `_fluid_equations/_equations.py`
(`primitive_state_from_conserved`/`conserved_state_from_primitive`), `_fluxes.py` (`_euler_flux`)
and `total_quantities.py` (`calculate_internal_energy`/the total-energy diagnostic);
`speed_of_sound_crs` branches in `reconstruction.py`, `hll.py` (both solver functions) and
`_timestep_estimator.py` (`get_wave_speeds`); the DSA-injection block (and its `shock_criteria`
import) in `_iteration_level_updates.py`; the `diffusive_shock_acceleration` gate in
`simulation_helper_data.py`; the Pallas-gate exclusion in `_pallas_evolve.py`; and the
`cosmic_ray_pressure` parameter on `construct_primitive_state`/`_assemble_primitive_state`
(never called from outside the module being removed).

**Verified clean:** `import astronomix` succeeds; repo-wide grep confirms no remaining
references to `CosmicRayConfig`/`CosmicRayParams`/`cosmic_ray_n_index`/`cosmic_ray_n_active`/
`inject_crs_at_strongest_shock`/`speed_of_sound_crs`/`gas_pressure_from_primitives_with_crs`/
`total_energy_from_primitives_with_crs`/`total_pressure_from_conserved_with_crs`/
`astronomix.shock_finder` (the legacy one) anywhere in `astronomix/` or `pytests/`. Full
regression: all 7 `pytests/cosmic_rays_grey/*.py` ladder-item scripts (items 1-6 +
`cr_gradient_check.py`) still pass identically; `pytests/hydrodynamics/shock_tube1D.py` (FD +
FV Pallas paths) still passes; `pytests/mhd/alfven_wave3D.py` (plain 3D MHD, exercises the
`evolve_state.py` split/join helpers this change touched) still passes.

**Next: Phase B's DSA injection (ladder item 8) should be built fresh against `e_cr`/`F_cr`,
targeting `find_shocks_pfrommer` from the start** -- there is no old-model code left to adapt,
and there shouldn't have been (see above).

## Where things stand (2026-08-21, ladder item 6 -- Phase A ladder complete)

**Ladder item 6 (two-fluid CR-modified shock tube) passes for real, and
this closes out Phase A's core ladder (items 1-6).** Scope decision made
with the user up front: build a real Pfrommer et al. (2006)-style
semi-analytic two-fluid Riemann solver (not a same-code high-resolution
self-consistency check), since the composite (two-adiabatic-index) EOS has
no closed-form rarefaction integral and no existing reference in this repo.

- New `astronomix/test_setups/reference_solutions/pfrommer_riemann_solver.py`
  (plain numpy/scipy, not JAX -- one-shot reference generator, not hot
  path): generalizes the existing single-gamma exact Riemann solver
  (`riemann_solver.py`) to `P(rho) = P_th,ref*(rho/rho_ref)^gamma_th +
  P_cr,ref*(rho/rho_ref)^gamma_cr`. Shock jump solved via a general-EOS
  Hugoniot (energy-jump) relation with the CR component staying on its own
  adiabat through the shock (collisionless, doesn't thermalize) while the
  thermal component absorbs the RH-consistent remainder; rarefaction-fan
  velocity via numerically-integrated Riemann invariant (no closed form for
  a composite EOS). Both nested inside the usual outer star-region-pressure
  bisection.
- **Validated the new solver independently before trusting it**, since
  there's no existing CR-composite-EOS reference in this repo to check
  against: with CR pressure zeroed on both sides it must reduce to the
  ordinary single-gamma problem, checked against the trusted
  `_exact_riemann_ideal_gas` across the *entire* sampled profile (not just
  star-region p*/u*) for three cases (classic Sod gamma=1.4, Sod at
  gamma=5/3, and a reversed-Sod case exercising the left-shock branch
  instead of the more common left-rarefaction) -- matched to ~2e-7. A
  second check with `gamma_cr=gamma_th` (nonzero, unequal CR pressure each
  side; composite EOS degenerates to one power law in the *combined*
  pressure) also matched to ~2e-7, and confirmed the CR pressure fraction
  per side stays constant through shock and rarefaction as physically
  expected when both components share an adiabatic index. This cross-check
  caught two real sign-error bugs during development (rarefaction-fan
  left/right family signs swapped; shock-speed sampling formula's signed
  mass-flux convention backwards) -- neither would have been obvious from
  inspection alone; both are exactly the kind of bug an independent
  numerical check like this is for.
- **Test design: `reduced_streaming_speed = 0`, not "small enough".**
  Pfrommer's solution assumes CRs move exactly with the gas (tightly
  coupled, no independent flux). `v_red=0` makes `grey_cr_flux_terms`'s
  `F_cr` equation collapse to homogeneous advection of zero initial data --
  `F_cr` stays *exactly* 0 for the whole run (verified: `max|F_cr|=0.0`),
  realizing Pfrommer's assumption exactly rather than approximately. This
  is a genuine physics insight specific to the two-moment closure's
  structure, not a workaround -- it only works because `F_cr`'s pressure-
  driving term is proportional to `v_red^2` while the momentum/energy
  feedback sources (`cr_pressure_gradient_source`/`cr_adiabatic_work_source`)
  are `v_red`-independent, so setting `v_red=0` removes exactly the piece
  of physics Pfrommer's model doesn't have, and nothing else.
- Result on the real (non-degenerate) case: `gamma_th=5/3`, `gamma_cr=4/3`,
  40% CR pressure fraction both sides (`P_th=0.6/0.06`, `P_cr=0.4/0.04`,
  otherwise classic Sod `rho`/positions), 400 cells, `t_end=0.2` -- mean
  absolute errors ~0.001-0.003 across density/velocity/pressure/`e_cr`,
  comfortably under `tol=1e-2` and comparable to the plain hydro
  `shock_tube1D.py` test's own HLL/minmod numerical-diffusion level. This
  specific left/right split is this session's own reasonable choice in
  Pfrommer's spirit, not a literal reproduction of a table from their
  paper -- only the *method* was independently validated (above).

Re-ran `cr_advection.py` and `cr_streaming_1d.py` as a regression spot
check (this item added no changes to the simulation code itself, only new
test-setup/reference-solver files, so full regression risk was low) --
both still pass.

## Where things stand (2026-08-21, ladder item 5)

**Ladder item 5 (1D streaming) passes for real.** Implemented both
previously-stubbed pieces:

1. **`streaming_flux_target` (`cr_grey_transport.py`, new function).** Per
   axis: `F_cr,axis = -sign(dP_cr/dx_axis) * reduced_streaming_speed *
   e_cr`, using the already-implemented `regularized_streaming_sign`.
   Applied once per full step in `_iteration_level_updates` as a discrete
   correction that overwrites `F_cr` -- the same pattern
   `anisotropic_flux_projection` (item 3) already established, not a new
   stiff relaxation-rate source. Deliberately does **not** require
   `config.mhd` (isotropic per-axis, direction from the local `∇P_cr`
   directly) -- matches the plan's "1D streaming" staging; combining with
   `anisotropic_transport` is documented as untested (streaming applied
   first, B-projection second) rather than assumed correct.
2. **`cr_streaming_heating_source` (`cr_grey_sources.py`, filled in).**
   `Gamma = reduced_streaming_speed * |dP_cr/dx|` (regularized-sign
   version), subtracted in full from `e_cr`, added to gas thermal scaled by
   `streaming_heating_efficiency`. Distinct from and additive to the flux
   target above (one is a conservative spatial redistribution, this is a
   genuine non-conservative CR-to-gas energy transfer).

**A calibration false alarm, worth recording so it isn't re-debugged:** the
first attempt used a periodic cosine `e_cr` profile (to exercise both
streaming directions in one test) and saw huge (~200%) `F_cr` errors and
sign flips. Root cause: `regularized_streaming_sign`'s `tanh` transition,
which is narrow in *gradient* space, mapped to an even narrower region in
*position* space near that profile's gradient zero-crossings (curvature
there is what sets the map) -- under one grid cell wide at the tested
resolution, i.e. an effectively-discontinuous, unresolved sign flip right at
the pressure extrema. Confirmed not a code bug by checking
`streaming_flux_target` in isolation at `t=0` (matched the analytic formula
exactly, per-cell) and after a single timestep (the large deviation was
already present, not something that accumulated over many steps). Fixed the
*test*, not the code: switched to a linear, open-boundary ramp (constant-sign
gradient everywhere, `regularized_streaming_sign` stays saturated), which
matches the plan's own "linear e_cr gradient" suggestion anyway. Verified:
`F_cr` vs. `streaming_flux_target` bulk rel. err ~2e-4; measured gas-heating
rate vs. the analytic `dP/dt = (gamma - 1) * efficiency * Gamma` bulk rel.
err ~7e-5 (the `(gamma - 1)` factor converts the conserved-energy-row rate
the source function adds into the primitive-pressure rate actually
measured -- tripped me up during calibration before I accounted for it).

Re-ran ladder items 1-4 and `cr_gradient_check.py` after wiring
`streaming_flux_target` into `_iteration_level_updates.py` (a shared file) --
all still pass identically (`config.cosmic_ray_grey_config.streaming`
defaults `False`, and none of items 1-4/the gradient check turn it on).

## Where things stand (2026-08-21, ladder item 4 + gradient check)

**Ladder item 4 (isotropic diffusion convergence) passes for real**, and
`cr_gradient_check.py` (plan item 16, run alongside per DESIGN.md's "next
steps") passes too. Both required real fixes, not just filling in a stub:

1. **No diffusion limit existed to test against.** `grey_cr_flux_terms` (as
   built for items 1-3) is a pure undamped wave equation -- a localized
   `e_cr` bump propagates rigidly, confirmed explicitly in item 3's own note
   below. Added `cr_grey_sources.cr_flux_relaxation_source` (Jiang & Oh
   2018's scattering term): `-nu * F_cr`, `nu = reduced_streaming_speed^2 /
   diffusion_coefficient` (`diffusion_coefficient` = new
   `CosmicRayGreyParams` field `kappa`). At quasi-steady balance this relaxes
   `F_cr ~= -kappa * grad(P_cr)`, recovering Fick's law with diffusion
   coefficient `kappa * (gamma_cr - 1)` for `e_cr`. Gated behind a new
   `CosmicRayGreyConfig.diffusive_relaxation` flag, **off by default** -- items
   1-3 use no flag changes and are unaffected. Wired into
   `_time_integrator_sources.py` (same pattern as `streaming`'s gate) and
   into `_cfl_time_step` (`_finite_volume/_timestep_estimation/
   _timestep_estimator.py`) as a new `dt < C_CFL / nu` branch, mirroring the
   existing `config.diffusion` viscous `dt_visc` constraint -- this is a
   real, deliberate re-introduction of a parabolic-like explicit CFL
   constraint (the plan's "no stiff *implicit/global* solve" goal is
   preserved; this stays fully explicit, just with its own stability bound).
   User confirmed this direction (add the physical relaxation term, over
   redefining the test as a numerical-diffusion check or deferring item 4)
   when asked, since it's a real physics/API decision, not a bug fix.
   Parameters (`reduced_streaming_speed=8`, `diffusion_coefficient=0.06`)
   were picked via an ad hoc calibration script (not committed) to put the
   relaxation rate deep enough in the quasi-steady regime
   (`nu ~= 1067 >> 1/(D/sigma0^2) ~= 50`) that the telegrapher-vs-diffusion
   model bias is negligible at the tested resolutions. Verified: L2 error vs.
   the analytic Gaussian Green's function shrinks monotonically (0.0089 ->
   0.0042 -> 0.0020 -> 0.0010 for N = 128/256/512/1024), total `e_cr`
   conserved to 6 significant figures at every resolution, convergence order
   ~1.0-1.1 (a least-squares fit across all 4 points; capped near first order
   by the sources' explicit-Euler operator splitting, not spatial
   truncation -- not a red flag). Test asserts `order >= 0.7`, comfortable
   margin below the observed ~1.0-1.1.
2. **`grey_cr_fast_speed` had an infinite-gradient singularity at `e_cr = 0`,**
   found while building `cr_gradient_check.py` (not by the diffusion work
   directly). Its CR-pressure acoustic-mode term (added ladder item 2) was
   `sqrt(jnp.maximum(x, 0.0))`; `sqrt`'s derivative is `+inf` at `x = 0`,
   and every ladder-item test so far (1, 3, 4) deliberately uses a CR-free
   background (`e_cr = 0` outside a localized pulse) for exactly the reasons
   documented in `cr_advection.py` -- so this singularity sits at almost
   every grid cell in every test run once you differentiate through it. JAX's
   `0 * inf = NaN` rule turned this into a `NaN` gradient the first time
   reverse-mode AD was actually run through the FV/CR-grey path at all (no
   prior test in this repo differentiates through FV --
   `pytests/differentiability/sensitivity.py` only covers FD). Fixed with a
   floor added in quadrature under the sqrt (not a `jnp.maximum` on the
   result), new `CosmicRayGreyParams.cr_pressure_speed_floor` (default
   `1e-10`), same "prefer smooth regularization" convention as
   `b_field_floor`. Verified: AD gradient of `sum(e_cr_final**2)` w.r.t.
   `reduced_streaming_speed` now matches a central finite difference to
   `rel_err ~= 9.5e-4` (was `NaN` before the fix). Also needed
   `differentiation_mode = BACKWARDS` in the test config -- plain
   reverse-mode AD does not work through `time_integration`'s adaptive-dt
   `jax.lax.while_loop` at all (forward or CR-related; this repo's
   checkpointed-adaptive-loop backend, dispatched on `differentiation_mode`,
   is what makes it work) -- unrelated to CR-grey specifically but necessary
   plumbing to get `cr_gradient_check.py` running.

Re-ran ladder items 1-3 after both fixes (the `_time_integrator_sources.py`/
`_cfl_time_step` wiring and the `grey_cr_fast_speed` floor both touch shared
CR-grey code paths) -- all three still pass identically (`diffusive_relaxation`
defaults off, and the speed-floor change is a sub-`1e-10`-scale perturbation
to an already-conservative CFL bound).

## Where things stand (2026-08-20, ladder item 3)

**Ladder item 3 (oblique anisotropic diffusion) passes for real**, and `anisotropic_flux_projection`
is implemented (was a dangling, uncalled stub). This required first fixing a real, pre-existing,
CR-unrelated bug in the FV MHD Strang split, found because ladder item 3 is the first test in this
module to combine `mhd=True` with `grey_cosmic_rays=True` at all.

1. **`evolve_state.py`'s magnetic-field split assumed B is always the trailing 3 state rows.**
   `_evolve_state_fv` hardcoded `primitive_state[-3:, ...]` as "the magnetic field" (with an
   explicit `# WARNING` comment saying so) to split gas and B for the Strang-split magnetic
   update. `registered_variables.py`'s FV branch allocates `magnetic_index` *before*
   `cosmic_ray_e_index`/`cosmic_ray_flux_index`, so with both `mhd` and `grey_cosmic_rays` on, B is
   no longer last -- the slice grabbed `(B_z, e_cr, F_cr_x)` as "the magnetic field" and
   mislabelled real `B_z` as gas, corrupting both halves of the split. Symptom: CR transport went
   completely inert with `mhd=True` (`e_cr` frozen to 6 significant figures, `F_cr` stuck at
   ~1e-9 instead of the ~1e-4 a matching `mhd=False` run reached) -- diagnosed by comparing
   otherwise-identical `mhd=True`/`mhd=False` runs and noticing wall-clock time was comparable
   (ruling out a CFL/dt collapse) while the state was essentially unchanged (ruling out "just
   slow", pointing at a flux/indexing bug instead). This is a **general bug, not CR-specific** --
   it would equally corrupt `wind_density` or the old `cosmic_ray_n` model combined with MHD, just
   never triggered before since no prior test combined MHD with anything registered after
   `magnetic_index`.

   User's explicit direction (asked, since the fix could go two ways with different blast radius):
   fix `evolve_state.py` to locate the magnetic rows by their actual registered indices, not
   reorder the registry to keep the old slice valid. Implemented as
   `_split_gas_and_magnetic_state`/`_join_gas_and_magnetic_state` (new helpers in `evolve_state.py`,
   general-purpose, not under `_cosmic_rays_grey/`) -- gather the real `magnetic_index.x/.y/.z`
   rows into `magnetic_field`, gather everything else (order-preserving) into `gas_state`, and
   re-index every other `RegisteredVariables` field (density/velocity/momentum/pressure/energy/
   wind_density/cosmic_ray_n/cosmic_ray_e/cosmic_ray_flux/interface_magnetic_field) past however
   many removed B-rows sit before each one, via `_shift_index_past_removed_rows`/
   `_shift_field_past_removed_rows`. Verified: plain 3D MHD (`pytests/mhd/alfven_wave3D.py`, no
   CR) still passes with identical L1 errors (FV native and FD Pallas paths); with the fix, a
   `mhd=True` + `grey_cosmic_rays=True` isotropic run now matches a `mhd=False` run to 6
   significant figures (`max e_cr` at `t=0.3`: `1.054278e-04` vs `1.054263e-04`) -- strong
   confirmation, since the isotropic CR closure doesn't depend on B at all and the two runs should
   (and now do) agree almost exactly.

2. **`anisotropic_flux_projection` cannot be called from `grey_cr_flux_terms`.** Original design
   (see the now-superseded note in a prior revision of that function's docstring) assumed it would
   hook into the per-interface flux computation, projecting F_cr onto B right where the isotropic
   closure reads it. But `grey_cr_flux_terms` runs inside the FV MHD Strang split's *gas-only*
   Riemann solve -- exactly where fix #1 above deliberately strips the magnetic rows out of the
   state array for that half-step. B is therefore structurally unavailable at that call site;
   confirmed by hitting `AttributeError: 'int' object has no attribute 'x'` (`registered_variables_gas.magnetic_index`
   is `-1` there) the first time the anisotropic run actually reached that code path. Fixed by
   moving the projection to `astronomix._modules._iteration_level_updates` (alongside the existing
   `e_cr` positivity floor) -- it runs once per full step, before the hydro update, on the full
   (unsplit) primitive state where B is genuinely present, overwriting `F_cr` with its B-projected
   value. `grey_cr_flux_terms` itself needed no anisotropy-aware branch at all once this was in
   place: by the time it reads `F_cr` mid-step, `F_cr` is already B-aligned if
   `anisotropic_transport` is on.

See "Verified (2026-08-20, ladder item 3)" below for the numerical result (isotropic run: an
expanding ring, perp/parallel variance-growth ratio ~1.0; anisotropic run: two pulses along ±B,
ratio ~0.03) and the plot.

## Where things stood (2026-08-20, ladder item 2)

**Ladder item 2 (adiabatic compression) passes for real**, but getting there required fixing two
real, pre-existing bugs in the transport layer that ladder item 1 never exercised (its CR
background was zero, so both the bulk-advection path and the CR-pressure momentum coupling were
no-ops all session). Both are described in full in `grey_cr_transport.py`'s docstrings and in
`DESIGN.md`'s "Correction" note; short version:

1. **Missing bulk advection.** `grey_cr_flux_terms` moved `e_cr`/`F_cr` only via the relative
   flux (`F_cr`), never via the gas velocity `u_n` -- a deliberate, previously-signed-off choice
   ("e_cr advected by F_cr, not by gas velocity"). Working the adiabatic-compression ladder test
   by hand (a homologous squeeze, exact Euler solution) showed this gives the wrong exponent
   (`e_cr ~ rho^(gamma_cr - 1)` instead of the plan's `rho^gamma_cr`) and -- more importantly --
   means CRs in a uniformly-flowing wind (`div(v) = 0`) are never swept downstream at all,
   breaking the Phase C wind/SNe science target. Fixed by adding the generic `u_n * q`
   bulk-advection piece to both CR rows (user confirmed this direction explicitly, over patching
   only the source term, when asked -- see `DESIGN.md`).
2. **Missing CR-pressure acoustic mode in the CFL bound.** With bulk advection alone the flux was
   correct but the *first* real run (nonzero, non-negligible `e_cr` under real momentum coupling)
   went to NaN. Root cause (isolated via clean-subprocess source-term zeroing -- an in-process
   monkeypatch attempt gave a false "transport, not sources" answer due to JAX JIT-cache reuse
   across calls with the same static args, a trap worth remembering): `cr_pressure_gradient_source`
   (`-grad(P_cr)` on momentum) and `cr_adiabatic_work_source` (`-P_cr div(v)` on `e_cr`) form a
   genuine coupled gas+CR acoustic mode with speed
   `sqrt(c_gas^2 + gamma_cr (gamma_cr - 1) e_cr / rho)` (linearized dispersion relation, derived
   by hand -- a real, stable wave, not a physical instability), and nothing in
   `grey_cr_fast_speed` ever accounted for it (it only returned `reduced_streaming_speed`, a
   completely independent quantity). For the test's `e_cr0 = 1.0` this combined speed (~0.68) is
   ~5x what the old bound (`max(c_gas, v_red) = 0.129`) provided, so CFL was silently violated.
   Fixed by widening `grey_cr_fast_speed` to `reduced_streaming_speed + sqrt(gamma_cr (gamma_cr -
   1) e_cr / rho)` (sum, not the tighter `sqrt(a^2+b^2)`, to keep it a simple, easily-conservative
   widening of an already-"never-too-small" bound). No call-site changes needed -- `e_cr`/`rho`
   were already available via `primitive_state`.

Both fixes verified not to perturb ladder item 1 (re-ran `cr_advection.py` after each -- passes
identically, as expected since its CR background is zero) or the existing hydro suite
(`shock_tube1D.py`, FD + FV Pallas path, still passes). See "Verified (2026-08-20 session)" below
for the numerical calibration that pinned down `cr_adiabatic_compression.py`'s tolerances.

## Where things stood (2026-08-19)

**Branch:** `feature/cosmic-rays-grey` (off `shock-finder`, which is off `main`).
**Latest commit:** `789a277` (merge of `main`) as of session start; this session's work
(params-threading + ladder item 1) is uncommitted on top.

**Ladder item 1 (pure CR advection) passes for real.** `DESIGN.md` was reviewed and signed off
with the user (summary + explicit confirmation, no changes requested) -- the plan's "stop and
show it to me before implementing" gate is cleared. The `params`-threading scaffold gap is
resolved. `grey_cr_flux_terms`/`grey_cr_fast_speed` (transport) and
`cr_pressure_gradient_source`/`cr_adiabatic_work_source` (feedback -- needed too, see below) are
implemented and numerically verified; `pytests/cosmic_rays_grey/cr_advection.py::test_cr_advection`
passes. `anisotropic_flux_projection` and `cr_streaming_heating_source` are still
`NotImplementedError` (not needed until ladder items 3 and 5 respectively).

Three real bugs were found and fixed along the way (none introduced by this session's own
earlier edits -- all pre-existing in the scaffold or the surrounding FV code, surfaced by
actually exercising the CR-grey path end-to-end for the first time):
1. The Pallas fast-path gate didn't exclude `grey_cosmic_rays` configs.
2. FV's operator-split source application (`_gravity_source_presolve`/`_apply_gravity_source`,
   which carries the CR-grey feedback calls) was gated on `config.gravity_config.gravity` alone,
   making CR-grey feedback dead code for every gravity-off run -- i.e. every Phase A ladder test.
3. The default `config.split == UNSPLIT` branch of `_cfl_time_step` had zero CR-grey (or even
   old-model) wave-speed awareness, so any `reduced_streaming_speed` exceeding the gas sound
   speed silently violated the CFL condition and blew up to NaN instead of erroring.
See "What's done" and "Verified" below for details.

## What's done

- `astronomix/_modules/_cosmic_rays_grey/`: `DESIGN.md`, `cosmic_ray_grey_options.py`
  (`CosmicRayGreyConfig`/`Params`), `cr_grey_fluid_equations.py` (real:
  `pressure_from_e_cr`/`e_cr_from_pressure`), `cr_grey_transport.py` (stubs:
  `grey_cr_flux_terms`, `grey_cr_fast_speed`, `anisotropic_flux_projection`; real:
  `regularized_streaming_sign`), `cr_grey_sources.py` (stubs: `cr_pressure_gradient_source`,
  `cr_adiabatic_work_source`, `cr_streaming_heating_source`).
- Registry: `cosmic_ray_e_index`/`cosmic_ray_flux_index` allocated in
  `registered_variables.py`'s FV branch. Verified: a 1D non-MHD config gets
  `cosmic_ray_e_index=3`, `cosmic_ray_flux_index=4`, `num_vars=5` as expected.
- Wired (all reachable only when `grey_cosmic_rays=True`): `_fluid_equations/_fluxes.py`
  (`_euler_flux`), `hll.py` (`_hll_solver`/`_hllc_solver`/`_am_hllc_solver`), `reconstruction.py`
  (`_reconstruct_at_interface_split`), `_timestep_estimator.py` (`get_wave_speeds`),
  `_iteration_level_updates.py` (`e_cr` positivity floor -- this one branch is real code, not a
  stub), `_time_integrator_sources.py` (feedback source calls).
- `pytests/cosmic_rays_grey/`: 7 skeleton scripts (one per Phase A ladder item + the
  differentiability check), each `def test_x(): raise NotImplementedError(...)` +
  `if __name__ == "__main__":`. Matches the repo's plain-script test convention (no `import
  pytest` -- see Environment notes below for why that matters).
- **`params` threading (this session).** `SimulationParams` is now a real parameter of
  `_euler_flux`, `_hll_solver`/`_hllc_solver`/`_am_hllc_solver`, `_lax_friedrichs_solver`,
  `_riemann_solver` (the dispatcher), `_reconstruct_at_interface_split` and `get_wave_speeds`,
  and every call site up through `evolve_state.py`/`_pallas_evolve.py`/`_timestep_estimator.py`/
  `_weno.py` (FD's `_euler_flux` call) was updated to pass it. `cr_grey_transport.py`'s
  `grey_cr_flux_terms`/`grey_cr_fast_speed`/`anisotropic_flux_projection` now take `params:
  SimulationParams` too, and the module-level `gamma_cr`/`reduced_streaming_speed` constants are
  gone -- nothing reads them anymore. Convention: `params` inserted immediately after `config` in
  every signature (mirrors the pre-existing `_reconstruct_at_interface_unsplit` pattern).
  `cr_grey_sources.py` already took full `params: SimulationParams` from the original scaffold,
  so nothing changed there.
- **Pallas-gate fix.** `_pallas_evolve.py`'s `_fv_pallas_evolve_supported` only excluded the
  *old* `_cosmic_rays` module (`config.cosmic_ray_config.cosmic_rays`), not `grey_cosmic_rays`.
  Since `backend_config.backend` defaults to `OPTIMAL_BACKEND`, which resolves to `PALLAS` on any
  compute-capability >= 8.0 GPU (e.g. this cluster's A100s), a CR-grey run on such hardware would
  have silently taken the fused Pallas kernel path -- which has zero `e_cr`/`F_cr` awareness --
  instead of the native path DESIGN.md and the stubs assume, producing silently wrong physics
  rather than a `NotImplementedError`. Added a `config.cosmic_ray_grey_config.grey_cosmic_rays`
  guard alongside the existing one.
- **Operator-split-source gating fix (`evolve_state.py`, both `_evolve_gas_state_split` and
  `_evolve_gas_state_unsplit`).** `_gravity_source_presolve`/`_apply_gravity_source` are the
  *only* FV callers of `_time_integrator_sources`, which is also where the CR-grey feedback calls
  (`cr_pressure_gradient_source`/`cr_adiabatic_work_source`/`cr_streaming_heating_source`) live.
  Both were gated on `config.gravity_config.gravity` alone in every dimensionality/split branch,
  so CR-grey feedback was silently dead code whenever gravity was off. New local
  `apply_operator_split_sources = config.gravity_config.gravity or
  registered_variables.cosmic_ray_e_active` gates both now. While fixing this, also found the
  unsplit path's trailing apply-gate read `... and config.time_integrator`, and `RK2_SSP == 0`
  (see `simulation_config.py`) -- so that clause was **always False** or (since the branch above
  it already requires `time_integrator == RK2_SSP`) not doing anything; `_apply_gravity_source`
  was never actually called there, for gravity *or* CR-grey. Removed the dead clause. No existing
  test exercises FV-unsplit self-gravity (the one gravity pytest, `self_gravity/jeans_waves.py`,
  uses `FINITE_DIFFERENCE`), so nothing currently passing depended on the old (broken) behavior.
- **UNSPLIT CFL fix (`_timestep_estimator.py::_cfl_time_step`).** The `config.split == UNSPLIT`
  branch (the default -- `SimulationConfig.split` defaults to `UNSPLIT`) computed its wave-speed
  bound purely from `speed_of_sound(rho, p, gamma)`, with no `grey_cr_fast_speed` call at all
  (unlike the `else`/SPLIT branch's `get_wave_speeds` calls). Found by testing
  `reduced_streaming_speed` values above the gas sound speed: the run went to NaN almost
  immediately, because dt was being set without ever seeing the CR subsystem's actual signal
  speed. Added the same `jnp.maximum(c, grey_cr_fast_speed(...))` widening used elsewhere, gated
  on `registered_variables.cosmic_ray_e_active`. (The old `_cosmic_rays` model has this same gap
  in the UNSPLIT branch -- not touched, out of scope for this module.)

## Verified (this session)

- `import astronomix` -- clean after the params-threading refactor.
- `pytests/hydrodynamics/shock_tube1D.py::test_shock_tube1D` -- still passes with CR off,
  including the FD path and the FV Pallas fast path (confirmed via the printed "OPTIMAL_BACKEND:
  using the PALLAS backend" line) -- the new gated branches and the Pallas-gate fix are both true
  no-ops for existing physics.
- A standalone FV run with `riemann_solver=HLL` (not exercised by `shock_tube1D.py`, which only
  covers HLLC/FD) against the analytic Sod solution: no NaNs, sane density/pressure ranges, mean
  relative density error ~0.9% (well under the suite's `tol=1e-2`) -- confirms the params
  threading through `_hll_solver`/`_riemann_solver`/`_euler_flux` didn't perturb actual physics,
  not just that it runs.
- A minimal 1D config with `grey_cosmic_rays=True` reached `grey_cr_fast_speed`'s
  `NotImplementedError` end-to-end through `time_integration` -> `_cfl_time_step` ->
  `get_wave_speeds`, with `num_vars=5`, `cosmic_ray_e_index=3`, `cosmic_ray_flux_index=4` as
  expected, confirming the params-threaded plumbing before any physics was filled in (this probe
  is now obsolete -- the stub it targeted is implemented -- but is why the transport/CFL wiring
  gaps above were caught before the ladder-item-1 test was even written).
- **`pytests/cosmic_rays_grey/cr_advection.py::test_cr_advection` passes.** Implementation and
  calibration notes (see that file's docstring for the full derivation):
  - `grey_cr_flux_terms`: isotropic-closure two-moment flux
    (`d(e_cr)/dt + d(F_cr)/dx = 0`, `d(F_cr)/dt + d(v_red^2 (gamma_cr-1) e_cr)/dx = 0`), with the
    `(gamma_cr - 1)` factor matching the same `P_cr` used for gas-momentum coupling (not an
    independent constant) -- this is the standard radiation-M1-style isotropic Eddington-tensor
    closure Jiang & Oh (2018) is based on.
  - `grey_cr_fast_speed`: returns the un-scaled `reduced_streaming_speed` (the true eigenvalue of
    the system above is smaller, `reduced_streaming_speed * sqrt(gamma_cr - 1)`; the un-scaled
    value is used everywhere as a conservative CFL/Riemann safety bound, standard
    reduced-speed-of-light practice).
  - `cr_pressure_gradient_source`: `-grad(P_cr)` on momentum *and* the matching `-v . grad(P_cr)`
    work-rate on gas total energy (`pressure_index`) -- the second term isn't in DESIGN.md's
    one-line bullet but is required for energy conservation (verified below) and mirrors
    `_gravity._gravitational_source_term_along_axis`'s momentum/energy pairing exactly.
  - `cr_adiabatic_work_source`: `-P_cr * div(v)` on `e_cr`, exactly as DESIGN.md states.
  - **Numerical verification done during implementation** (ad hoc scripts, not committed as
    tests): with a genuine CR-free background (`e_cr` background = 0, only the pulse), the
    numerical propagation speed matches the analytic eigenvalue
    `reduced_streaming_speed * sqrt(gamma_cr - 1)` to 5 decimal places and total `e_cr` is
    conserved to 6, across resolutions 256-2048 and `reduced_streaming_speed` in [1, 20] -- this
    is what pinned down the flux/wave-speed formulas above as correct rather than guessed.
  - **With a nonzero background `e_cr` (i.e. a small pulse on top of an ambient CR population),
    momentum/energy feedback is real and non-negligible** (order pulse-amplitude, not
    amplitude-squared, since `-P_cr * div(v)` doesn't vanish against a spatially uniform
    background pressure the way a naive "small perturbation -> negligible coupling" assumption
    would suggest) -- total (gas + CR) energy was checked to be conserved to ~7e-5 relative in
    that regime, consistent with the exact local-conservation identity
    `d(E_gas)/dt + d(e_cr)/dt + div(flux terms) = 0` the two source functions were derived to
    satisfy (product-rule identity `div(P_cr v) = v.grad(P_cr) + P_cr div(v)`). This is why
    `cr_advection.py`'s test deliberately uses a CR-free background -- ladder item 6
    (`cr_shock_tube.py`) is where the coupling itself gets tested.
- All 7 `pytests/cosmic_rays_grey/*.py` skeletons other than `cr_advection.py` still import and
  run to their intended `NotImplementedError` (unaffected by this session's implementation work).

## Verified (2026-08-20, ladder item 3)

- `pytests/cosmic_rays_grey/cr_anisotropic_diffusion_oblique.py::test_cr_anisotropic_diffusion_oblique`
  passes. Setup: 128x128 2D box, `mhd=True`, uniform B at 30 degrees to the grid, periodic BCs,
  localized `e_cr` Gaussian bump (`amp=1e-3`, `sigma=0.03`) on a uniform gas at rest, `F_cr=0`
  initial, `reduced_streaming_speed=1`, run to `t_end=0.3`; isotropic and anisotropic closures run
  on the identical IC for direct contrast.
  - With no bulk flow and no relaxation/damping term in the two-moment system (pure hyperbolic
    wave, not diffusive), an at-rest localized bump does not diffuse smoothly -- it propagates as
    a genuine wave. Isotropic closure: the bump expands as a clean ring (matches the standard
    even-dimension wave-equation fundamental solution). Anisotropic (B-projected) closure: the
    bump splits into exactly two pulses moving along +/-B -- both clearly visible in the
    committed plot (`pics/cr_anisotropic_diffusion_oblique_test.svg`).
  - Diagnostic: `e_cr`-weighted second moment of the distribution, resolved parallel/perpendicular
    to B, tracked as *growth* relative to the initial (isotropic Gaussian) moment to normalize out
    the pulse's own width. Isotropic run: perp/parallel growth ratio ~1.00 (spreads equally in
    both directions, as expected). Anisotropic run: ratio ~0.03 (parallel growth statistically
    identical to the isotropic run -- confirms along-B transport isn't suppressed -- while
    perpendicular growth is ~30x smaller). Committed test asserts the anisotropic ratio
    `< 0.1`, comfortable margin above the observed ~0.03 and far below the isotropic ~1.0.
  - Getting a working `mhd=True` + `grey_cosmic_rays=True` config at all required the
    `evolve_state.py` fix above first -- initial attempts (before that fix) showed CR transport
    completely inert under MHD, diagnosed by comparing wall-clock time (comparable between
    `mhd=True`/`False`, ruling out a CFL/dt collapse) against the frozen state (ruling out "just
    slow").
- Re-ran `cr_advection.py` and `cr_adiabatic_compression.py` after the `evolve_state.py` and
  `_iteration_level_updates.py` changes -- both still pass (neither uses `mhd` or
  `anisotropic_transport`, so unaffected in principle; re-run anyway since `evolve_state.py`
  fix #1 touches the shared FV evolution path).
- Re-ran `pytests/mhd/alfven_wave3D.py::test_alfven_wave_convergence` (plain 3D MHD, no CR) --
  passes with identical L1 errors (FV native-JAX and FD Pallas paths) to before the
  `evolve_state.py` fix, confirming the re-indexed split/join is a no-op when nothing is
  registered after `magnetic_index` (the common case today).
- Re-ran `shock_tube1D.py::test_shock_tube1D` (non-MHD) -- still passes, confirming the new
  `_split_gas_and_magnetic_state`/`_join_gas_and_magnetic_state` helpers (only reachable when
  `config.mhd` and `dimensionality > 1`) are a no-op for non-MHD configs.

## Verified (2026-08-20 session)

- `pytests/cosmic_rays_grey/cr_adiabatic_compression.py::test_cr_adiabatic_compression` passes.
  Setup: 512-cell 1D box, open boundaries, homologous squeeze
  `v(x,0) = -alpha0 (x - x_center)` with `alpha0 = 0.2`, uniform `rho0=1`, `p0=0.01` (gas),
  `e_cr0=1` (CR, deliberately *not* small relative to gas pressure -- this is what exercises the
  CR-pressure acoustic mode above), run to `t_end=1`.
  - The idealized analytic homologous solution (uniform density, pure function of time) breaks
    down under open BCs: the zero-gradient ghost-cell approximation to the nonzero analytic edge
    velocity perturbs the density profile across most of the domain by `t_end`, not just a thin
    edge layer (checked: density varies smoothly but non-trivially, ~1.08 at the edges to ~1.15
    at the center, not flat). Comparing against the analytic formula directly gave ~8-11% error
    even in a nominal "interior" window -- not tight enough to be a good test.
  - The *pointwise* scaling relation `e_cr/e_cr0 ~= (rho/rho0)^gamma_cr`, evaluated using the
    simulation's own `rho` at each cell, is far more robust (a local thermodynamic identity,
    independent of the density profile's global shape): max relative error is 0.54% over the
    *entire* domain including boundary-adjacent cells, and drops to 0.08% with a 30% edge margin.
    The wrong (pre-advection-fix) exponent `gamma_cr - 1` disagrees with the same data by ~15% --
    a wide, unambiguous separation. The committed test uses a 10% edge margin and `tol=5e-3`.
  - Confirmed the fix is necessary, not just sufficient: reverted `grey_cr_flux_terms` to the
    pre-session (no bulk advection) formula in an isolated script and reran the identical
    setup -- stable up to `t_end=0.5` (vs. NaN by `t_end~0.09-0.095` for the *buggy pre-CFL-fix*
    version at the same `e_cr0`), consistent with the old formula lacking the acoustic-mode
    coupling this test's `e_cr0` triggers in the *new* formula; not re-verified against the wrong
    exponent numerically beyond the `t_end=1` comparison above, since by that point both the
    advection and CFL fixes were already in place together.
- Re-ran `cr_advection.py::test_cr_advection` after each fix (both the advection-flux change and
  the CFL widening) -- passes identically both times, as expected (its CR background is zero, so
  neither change is triggered).
- Re-ran `shock_tube1D.py::test_shock_tube1D` (CR off, FD + FV Pallas path) -- still passes,
  confirming neither fix perturbs non-CR physics.
- **JIT-cache trap, worth remembering:** an in-process attempt to isolate "is the instability from
  the source terms or the transport flux" by monkeypatching `cr_pressure_gradient_source`/
  `cr_adiabatic_work_source` to return zero *in the same Python process* as an unpatched
  "baseline" run gave a false negative (still NaN) -- because `_time_integrator_sources`'s jitted
  caller had already been traced and cached against the *original* functions on the first
  (baseline) call, and reassigning the module-level name afterward doesn't invalidate that cache
  entry when the static args (`config`, `registered_variables`) compare equal across calls. Redid
  the same isolation as two calls in two separate subprocesses -- got the correct (opposite)
  answer. Any future "patch a function and rerun in the same process" debugging in this codebase
  should go through a fresh subprocess instead.

## Known gap carried forward (not fixed this session, flagged deliberately)

- `_lax_friedrichs_solver` (`_lax_friedrichs.py`) now takes `params` (mechanical, needed for the
  `_euler_flux` signature change) but its dissipation coefficient `alpha` is still gas-only --
  unlike `_hll_solver`/`_hllc_solver`/`get_wave_speeds`, it does not widen `alpha` with
  `grey_cr_fast_speed` when `registered_variables.cosmic_ray_e_active`. DESIGN.md's "known
  scaffold gap" list never named Lax-Friedrichs (only `_euler_flux`, `_hll_solver`/`_hllc_solver`,
  `_reconstruct_at_interface_split`, `get_wave_speeds`), and `LAX_FRIEDRICHS` isn't the default
  `riemann_solver` (`HLL` is), so this doesn't block Phase A ladder work with default settings --
  but picking `LAX_FRIEDRICHS` explicitly with CR-grey on would currently under-dissipate the CR
  subsystem. Flagged in the function's docstring; fix alongside the ladder items if/when a test
  needs that solver.

## Next steps, in order

1. ~~Review `DESIGN.md`'s physics content~~ -- done (summary reviewed and confirmed with the
   user; no changes requested).
2. ~~Resolve the "known scaffold gap"~~ -- done (params threading, see above).
3. ~~`cr_advection.py` (ladder item 1)~~ -- done, passes (see "Verified" above).
4. ~~`cr_adiabatic_compression.py` (ladder item 2)~~ -- done, passes; required fixing two real
   gaps (missing bulk advection in `grey_cr_flux_terms`, missing CR-pressure acoustic mode in
   `grey_cr_fast_speed`'s CFL bound) -- see "Where things stand" and "Verified" above.
5. ~~`cr_anisotropic_diffusion_oblique.py` (item 3)~~ -- done, passes; required implementing
   `anisotropic_flux_projection` (was a dangling stub), plus a general (non-CR-specific) fix to
   the FV MHD Strang split in `evolve_state.py` -- see "Where things stand" and "Verified" above.
   The projection is applied once per step in `_iteration_level_updates`, not inside
   `grey_cr_flux_terms` -- B is structurally unavailable at that call site (see point 2 above).
6. ~~`cr_isotropic_diffusion_convergence.py` (item 4)~~ -- done, passes; required adding the
   previously-missing `F_cr` relaxation term (`cr_grey_sources.cr_flux_relaxation_source`, gated
   behind new `diffusive_relaxation` config flag, off by default) since the two-moment system as
   built for items 1-3 has no diffusion limit at all -- see "Where things stand (2026-08-21)"
   above. User confirmed this direction explicitly when asked (real physics/API decision).
7. ~~`cr_gradient_check.py` (item 16, FD-vs-AD)~~ -- done, passes; required fixing a real
   infinite-gradient singularity in `grey_cr_fast_speed` at `e_cr = 0` (new
   `cr_pressure_speed_floor` param) -- see "Where things stand (2026-08-21)" above. This is the
   first time any test in this repo has differentiated through the FV solver path at all (prior
   differentiability coverage, `pytests/differentiability/sensitivity.py`, is FD-only), so this
   fix likely matters beyond CR-grey too -- worth a heads-up to whoever next tries `jax.grad`
   through an FV run for any reason, CR or not, since `differentiation_mode = BACKWARDS` is also
   required for FV/CR-grey reverse-mode AD to work at all (adaptive-dt `while_loop`, unrelated to
   CR specifically -- see that item's note above).
8. ~~`cr_streaming_1d.py` (item 5)~~ -- done, passes; required implementing both
   `streaming_flux_target` (new, `cr_grey_transport.py`) and `cr_streaming_heating_source`
   (`cr_grey_sources.py`) -- see "Where things stand (2026-08-21, ladder item 5)" above.
   Note: streaming's interaction with `cr_flux_relaxation_source` (item 4's diffusive relaxation)
   was not directly exercised -- both are independent, additively-gated terms (different config
   flags), which is physically reasonable (bulk streaming + residual scattering/diffusion is the
   standard combined picture) but untested in combination.
9. ~~`cr_shock_tube.py` (item 6)~~ -- done, passes; required building a new semi-analytic
   two-fluid Riemann solver (`astronomix/test_setups/reference_solutions/
   pfrommer_riemann_solver.py`) since no composite-EOS reference existed in this repo -- see
   "Where things stand (2026-08-21, ladder item 6)" above. **Phase A's core ladder (items 1-6) is
   now complete.** This is where the next session should pick up on remaining Phase A items.
9. FD transport is out of scope until the WENO-eigensystem extension (see DESIGN.md's "FD
   limitation") gets separately scoped -- don't attempt it inside a ladder-item pass.
10. ~~Revisit `DESIGN.md`'s open question on consolidating with the older
    `astronomix/_modules/_cosmic_rays/` (`n_cr`) model~~ -- done (2026-08-24): retired the old
    model rather than keeping both -- see "Where things stand (2026-08-24, old `_cosmic_rays`
    (`n_cr`) model retired)" above for the full rationale and what was removed.
11. ~~`cr_sedov_taylor.py` (item 7, Phase B)~~ -- done (2026-08-24), passes; required building
    `cr_grey_injection.py`'s `inject_crs_at_shocks` fresh against `find_shocks_pfrommer` -- see
    "Where things stand (2026-08-24, ladder item 7)" above.
12. ~~Ladder item 8 (Mach-dependent DSA efficiency)~~ -- done (2026-08-26), including the
    dedicated `pytests/cosmic_rays_grey/cr_dsa_mach_dependence.py`: `dsa_efficiency_kang_ryu_2013`
    + `dsa_efficiency_model` / `dsa_efficiency_mach_scale` config, see "Where things stand
    (2026-08-26, ladder item 8)" above for the full test design and the two tolerance-calibration
    lessons.
13. **Ladder item 9 (1D steady CR-driven flow / CR-modified shock structure) is BLOCKED, not
    done** -- see "Where things stand (2026-08-27, ladder item 9)" above for the full account.
    The semi-analytic reference solver (`cr_modified_shock_structure.py`) is built and
    independently validated and can be reused once unblocked. **This is where the next session
    should pick up**, in order: (a) instrument the per-step `e_cr` budget at a stationary shock
    cell to find why sustained DSA injection there grows without bound instead of saturating
    (Finding 2); (b) once that's understood, re-attempt the item-9 test with Finding 1's
    mitigation (modest `reduced_streaming_speed`, small `diffusion_coefficient`, higher
    resolution); (c) separately consider whether Finding 1 (the shared gas/CR Riemann-solver
    wave-speed bound measurably degrading shock-capturing) warrants its own fix, independent of
    item 9, since it plausibly affects shock accuracy in any CR-grey config with
    `reduced_streaming_speed` well above local gas speeds.
14. `evolve_state.py`'s `_split_gas_and_magnetic_state`/`_join_gas_and_magnetic_state` fix (general,
    not CR-specific) is worth a heads-up to whoever owns the MHD module / other in-flight MHD work,
    since it changes behavior (from silently wrong to correct) for any future combination of `mhd`
    with `wind_density`, not just grey CR (the old `cosmic_ray_n` model this originally also
    applied to has since been retired -- see item 10 above).
15. **Note: item 13 above is stale** -- ladder item 9 was actually resolved (2026-09-09, via a
    moving-shock redesign) and ladder item 10 (wind-blown bubble) completed (2026-09-10) after
    item 13 was written; see this file's dated entries above (newest-first) for the current
    status, not item 13's text.
16. ~~Ladder item 11 (SNR expanding into a uniform then a clumpy medium)~~ -- done (2026-09-11);
    see "Where things stand (2026-09-11, ladder item 11 -- DONE, SNR into uniform/clumpy medium)"
    above for the full design, calibrated numbers, and the clumps-matter signature's (counter-
    intuitive) sign. The ladder's integration/physical group (items 9-11) is now fully complete.
17. Ladder item 12 (SILCC-ISM project: reproduce a published grey CR-ISM result as a
    code-comparison anchor, Girichidis et al. 2016 / Simpson et al. 2016 style stratified box) --
    **milestones M0a through M4 done (2026-09-15); M5 (the CR-on code-comparison anchor) and M6
    (optional MHD/anisotropic-diffusion stretch) remain, not yet started.** M4, the
    longest-running milestone so far (episodic SN driving + K&I cooling combined, evolved to `t_end~
    7.39e6` yr), needed three real fixes before it would run stably: the tapered-shield NaN in
    `delayed_cooling` (root-caused and fixed), the FV unsplit-solver near-vacuum positivity floor
    (root-caused and fixed), and -- the fix that finally let the full run complete -- the new
    opt-in `CoolingConfig.subcycle_stiff_cooling`, which stopped M1's own `dt_cool` CFL guard
    (a domain-wide minimum) from stalling the whole simulation whenever one persistently
    fast-cooling diffuse cell exists anywhere in the domain. See "Where things stand (2026-09-15,
    later same day: SILCC-ISM project M4 -- DONE...)" above and DESIGN.md's "Resolved: SILCC-ISM
    project M4" section for the full account. **Note: M2, M3.5, and M4 all run with CR-grey off
    (`cosmic_ray_grey_config` default) -- only M3's own pytest turns `grey_cosmic_rays=True` on,
    and only in one of its two compared configs, for a narrow cr-on/cr-off energy-conservation
    cross-check, not sustained CR transport within the stratified-column setting.** These
    milestones built and stress-tested the stratified-column + cooling + SN-driving *scaffold*
    the eventual CR-ISM comparison needs; M5 (actually running grey CR transport/feedback
    throughout this setting and comparing against the literature result) and M6 are the
    remaining, not-yet-started milestones.
18. ~~Ladder item 12 (SILCC-ISM project)~~ -- done (2026-09-16), M0a through M6, with one
    deliberately-left-open gap (M6's clumpiness comparison vs. M5 -- M5's states were never saved
    to disk, so this needs a fresh M5-random-placement rerun with M6's diagnostics added to close;
    user chose to leave it and move on rather than spend more GPU time unprompted). See the
    2026-09-16 "M6" and "M5" entries above for the full account.
19. ~~Ladder item 13 (pion-decay SED for a known proton population vs. naima, Phase C's first
    item)~~ -- done (2026-09-17): new `cr_grey_emission.py` (JAX port of naima's Kafexhiu et al.
    (2014) implementation) validated to floating-point precision against naima itself. See the
    2026-09-17 entry above and DESIGN.md's "Resolved: pion-decay emission (ladder item 13, Phase
    C, 2026-09-17)" section for the full account, including the float32-overflow NaN bug caught
    along the way and the naima-numpy-mask-to-jnp.where regime-priority reconstruction.
20. ~~Ladder item 14 (synchrotron + IC SED for a known electron population vs. naima)~~ -- done
    (2026-09-17): new `cr_grey_emission_leptonic.py` (JAX ports of naima's Synchrotron/
    InverseCompton implementations) validated against naima to `<4e-8` relative error across
    several spectral shapes/field strengths/seed fields. Turned out not to actually need the
    plan's "electron treatment" decision (Sec. 6) resolved first -- see the 2026-09-17 "later"
    entry above and DESIGN.md's "Resolved: synchrotron + inverse-Compton emission (ladder item
    14, Phase C, 2026-09-17)" section, including two real bugs caught (a positivity floor sized
    for the wrong quantity's scale, in two different guises).
21. **Next: item 15 (end-to-end SNR-molecular-cloud case, now unblocked -- items 13 and 14 both
    done) -- needs the electron-treatment decision (Sec. 6) resolved for real this time, since
    item 15 actually has to derive an electron population from the CR-grey/proton state, not just
    validate a formula against a known one; or close item 12's M6 clumpiness gap (18, above) --
    whichever the user prioritizes.**

## Environment notes (so the next session doesn't have to rediscover these)

- The venv with `jax`/astronomix installed for this user is `~/venv_grav_source`
  (`source ~/venv_grav_source/bin/activate`) -- not on `PATH` by default, no system `python`.
- That venv does **not** have `pytest` installed, and it isn't a declared dependency in
  `pyproject.toml` either. This is why the `pytests/cosmic_rays_grey/*.py` skeletons (like the
  rest of `pytests/<category>/*.py`, except `test_reproduce_paper.py`) deliberately avoid
  `import pytest` -- they're plain scripts, runnable standalone.
- `autocvd(num_gpus=1)` (called at import time by every `pytests/*.py` script) waits
  indefinitely for a GPU to read as free by utilization. On this shared cluster it never does,
  so it hangs. Workaround: set `CUDA_VISIBLE_DEVICES` to a specific free GPU *and*
  monkeypatch `autocvd.autocvd` to a no-op before importing the target script/module, e.g.:
  ```python
  import os
  os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # or whichever GPU the user names
  import autocvd
  autocvd.autocvd = lambda *a, **k: [0]
  # NOW import/run the astronomix script/module
  ```
  Ask the user which GPU to pin if they haven't said; check `nvidia-smi` for a sane default.
