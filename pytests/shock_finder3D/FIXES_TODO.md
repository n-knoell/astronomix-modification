# CWB / shock-finder fixes — carryover to next session

Origin: investigating why `cwb_shocked_cells_2d_256.png`'s Mach numbers looked
qualitatively right but capped at max ~7, when the CWB wind-termination shock
is physically ~Mach 100. Split into four threads: (1) the Pfrommer
shock-finder formalism, (2) the CWB test's silently-diffusive solver config,
(3) a dimensional bug in the 3D stellar-wind injection module, and (4) the
CWB sim going to NaN.

**Current status (updated 2026-09-23, after round 18):**
- **1c / 1d, 2, 3: done.**
- **1a: superseded.** The original zone-edge-walk fix was reverted (it broke a
  live sim); the adaptive walk-until-plateau sampling from rounds 10–16
  (`dd7a5ae`) replaces it.
- **1b (Mach sampling formalism): fixed** (rounds 10–17) and validated
  quantitatively against Sedov (~10% of the exact Mach number).
- **1b (the CWB Mach gap itself): fully diagnosed, fix not implemented**
  (round 18). The measured Mach ~10 is *correct* for the simulated field; the
  pre-shock wind is ~256x hotter than ambient at the point of collision
  because the wind is injected at a numerically large radius with no
  adiabatic-cooling history. With that gas at ambient temperature the same
  jump gives Mach ~143, matching theory. Resolution (64→512), the separation
  fix, and radiative cooling are all ruled out (rounds 12–14). Proposed fix:
  impose an analytic free-wind profile (`rho ~ r^-2`, `T ~ r^-4/3`) over an
  extended wind zone. **Now implemented as the opt-in
  `WindConfig.analytic_wind_zone` (round 19):** Mach ~1.4x (max) / ~2.4x
  (median) higher at N=64, but spurious total-energy heating at the zone edge
  still keeps the pre-shock wind ~20x above ambient.
- **4 (NaN): worked around, not fixed.** The stationary binary
  (`nbody=False`, explicit `wind_injection_positions`) runs clean at
  N=64/128/256/512 (rounds 8–9, 13) and is what `_cwb_setup.py` now uses.
  With an orbiting binary (`nbody=True`) the failure is still open; the
  sharpened question is what about source motion breaks positivity.

**Repo state:** the items 1c/1d/3 changes and the round-16 Mach-sampling fix
are committed. `_cwb_setup.py` / `cwb_shock_finder.py` currently carry
uncommitted, unrelated work (grey-CR / DSA coupling for the CWB setup, new
`*_64.png` CR figures). `cwb_shocked_cells_{2d,3d}_256.png` were last
regenerated 2026-09-19, i.e. before the rounds 10–17 finder fixes; the
`*_stat_*` / `_512` figures are the stationary-setup outputs. `..._256a.png`
are the old `HLL`/first-order, Mach-capped-at-~7 references, kept for
comparison; delete once no longer needed.

**Running things:** `source ~/venv_grav_source/bin/activate`, then pin a GPU
manually and no-op `autocvd` (the repo's own `autocvd(num_gpus=1)` hangs
forever on this shared cluster) — see `verify_wind_fix.py` / `debug_nan2.py`
/ `cwb_param.py` / `run_injection_sweep.py` under the scratch dir for the
pattern; `cwb_param.py` in particular is a parametrized rebuild of
`_cwb_setup.py` (code-unit scale, `num_injection_cells`, and N-body all
exposed as arguments) worth reusing rather than rebuilding for the next round
of experiments. Check `nvidia-smi` fresh each session.


---

## 1. Shock-finder formalism bugs (real, independent of the CWB solver issue)

### 1a. [REVERTED — correctness bug found, not just "insufficient"] Pre/post-shock sampling walked a fixed 1 cell — the zone-edge-walk replacement produces real NaN in a live simulation

**Correction to an earlier claim in this file:** this was previously marked
`[DONE]` with "this is genuine and should stay." That was wrong, and has been
reverted (`_shock_zones.py`, `_shock_mach.py`, `pfrommer_shock_finder.py` are
back to their pre-this-fix `HEAD` state; only 1c/1d remain applied in
`_shock_surface.py`).

What happened: `get_post_pre_shock_values` gained an optional `shock_zones`
argument (walk outward, capped at `max_steps`, until leaving the zone,
instead of a fixed 1-cell offset), wired through `_shock_mach.py`
(`max_steps=8`) and `pfrommer_shock_finder.py`. This was verified correct
*in isolation* (synthetic profiles recover the true jump) — but it was
**never actually verified against a real running simulation**, because the
CWB test (the only 3D consumer available at the time) never got that far
(see item 4). It turns out to be broken in a real run: found 2026-08-26 while
working on the CR-grey module's ladder item 8 (`cr_grey_injection.py`, which
also calls `find_shocks_pfrommer` for DSA injection, entirely independent of
the CWB test) — the previously-passing `cr_sedov_taylor.py` (item 7) started
going to all-NaN. Bisected by selectively reverting subsets of that session's
shock-finder changes and re-running: `_shock_surface.py` alone (items
1c/1d) is fine; adding back `_shock_mach.py` + `_shock_zones.py` +
`pfrommer_shock_finder.py` (this item) reproduces the NaN on its own, with
or without 1c/1d present. Root cause not yet diagnosed (candidate mechanism:
`walk_to_zone_edge`'s up-to-8-cell walk, via `jnp.roll`'s wraparound, sampling
across a domain boundary or another shock's plateau in some geometry the
1-cell version never reached) — deprioritized in favor of just reverting,
since CR ladder work was the actual priority when this was found.

**This is a real, reproducible correctness bug, not merely "insufficient to
fix the ~7 cap"** — don't re-apply this fix without first (a) diagnosing the
actual mechanism above and (b) verifying against a real running 3D
simulation (not just synthetic profiles) before calling it done again.

### 1b. [NOT FIXED — blocked behind item 4] `shock_zones` itself is narrower than the true (numerically smeared) transition

`shock_zones` = criterion1 (converging flow) & criterion2 (aligned ∇T·∇ρ) &
criterion3 (`_shock_zone_criterion_minimum_mach`, `_shock_zones.py`). Criterion
3 requires the *immediate-neighbor* jump alone to exceed `mach_min` — so the
zone only ever covers the "steep core" of a shock, not its shallow tails. For
a smeared numerical shock this steep core can be much narrower than the full
transition, so 1a's zone-edge walk still stops short of the true asymptotic
plateau. Reproduced directly: feeding `find_shocks_pfrommer` a synthetic
Rankine-Hugoniot-consistent Mach-100 shock smeared over *w* cells gives
Mach≈78 (w=1), ≈33 (w=2), **≈8.5 (w=3)** — matching the observed ~7 almost
exactly, and confirming the mechanism, not just the sampling distance, is the
bottleneck.

**Fix direction:** decouple the pre-/post-shock sampling walk from
`shock_zones` membership entirely — walk outward until the *field itself*
plateaus (e.g. stop when the local log-gradient drops below a tolerance),
rather than until leaving the detection mask. Criterion 3 should stay as-is
for zone *detection* (that's a different job — deciding whether a cell is
shock-like at all); only the Mach-sampling walk needs to stop using it as a
stopping bound. Can't be usefully re-tested against the real CWB profile
until item 4 is resolved.

### 1c. [DONE, verified in isolation] `_find_shock_surface_2d` / `_find_shock_surface_3d`: one-sided surface-cell selection

`is_surface_cell` (`_shock_surface.py`) only checked whether any cell *ahead*
along the ray had smaller divergence, never cells *behind* — so every cell
from the true peak through to the far end of the zone (in the ray direction)
also had "nothing smaller ahead of it" and got spuriously tagged as an extra
"surface" cell. **Fixed**: `is_surface_cell` now walks the ray in *both*
directions (`walk_finds_smaller(sx, sy) | walk_finds_smaller(-sx, -sy)` in
2D, same pattern in 3D) and only keeps a candidate if neither direction finds
a smaller value — i.e. the true argmin over the whole contiguous segment, not
just its "ahead" half. Verified with a synthetic single-peaked profile
(`[0,-10,-50,-100,-30,0]` in a zone of width 4): before the fix all 4 zone
cells were tagged surface; after, only the true peak is. Test script:
scratch `verify_surface_fix.py` (not committed — rerun from this diff if
needed).

### 1d. [DONE, verified in isolation] `_find_shock_surface_1d` off-by-one

`jax.lax.fori_loop(1, max_segment_id, ...)` silently found **zero** surface
cells whenever there was exactly one contiguous shock-zone segment (the
common case for a single isolated shock), since `range(1, 1)` is empty
(segment labels run 1..max_segment_id inclusive). **Fixed**: loop bound
changed to `max_segment_id + 1`. Verified with both a single-segment and a
two-segment synthetic case (same scratch script as 1c) — single segment now
correctly tags its one true peak (previously tagged nothing); two segments
tag one peak each. Doesn't affect the CWB test (3D uses
`_find_shock_surface_3d`), but was silently broken for any 1D shock-finder
usage.

---

## 2. `_cwb_setup.py` was silently running a much more diffusive solver than documented

**Fixed in a prior session** (`riemann_solver=HLLC`, `first_order_fallback=False`,
`limiter=VAN_ALBADA_PP`). This uncovered items 3/4 below. Not touched further
this session.

---

## 3. [DONE — formula fixed and independently validated] `_wind_ei3D` / `_wind_ei`'s pressure-rate formula was dimensionally wrong

`pressure_rate = pressure_from_energy(energy_rate, updated_density, speed, gamma)`
mixed a *rate* (`energy_rate`, energy/volume/time) with the *actual* local
density (`updated_density`) instead of the injection's own density *rate* —
`pressure_from_energy(E, rho, u, gamma) = (gamma-1)*(E - 0.5*rho*u**2)`, so
this silently computed `(gamma-1)*(energy_rate - 0.5*updated_density*speed**2)`,
dimensionally inconsistent (the subtracted term has no `/time`).

**Fixed** in both `_wind_ei3D` (`astronomix/_modules/_stellar_wind/stellar_wind.py`,
3D EI path, used by the CWB test) and `_wind_ei` (1D EI path — same bug,
same fix, also a live code path via `_wind_injection`). New formula, derived
from first principles rather than reusing the mismatched helper (avoids a
0/0 outside the injection mask, where both terms are legitimately zero):

```python
pressure_rate = (gamma - 1) * (energy_rate - 0.5 * density_rate * speed**2)
```

Reasoning: the injected mass is added at the local gas velocity (no momentum
kick in the EI scheme), so it already carries `0.5 * density_rate * speed**2`
of kinetic energy "for free" as density grows at fixed velocity; only the
*remainder* of the wind's kinetic-luminosity budget (`energy_rate`) needs to
land as thermal energy (pressure). Both terms are volumetric rates
(energy/volume/time), so this is directly `(gamma-1)` times a volumetric
internal-energy rate — no division, so no NaN where `density_rate == energy_rate == 0`
outside the injection region (the old code's use of `pressure_from_energy`
would have hit that 0/0 had `rho` been replaced with `density_rate` naively).

**Independent validation:** solving `pressure_rate = 0` for `speed` gives
`speed = sqrt(2*energy_rate/density_rate) = vel_scale` (the wind's own
terminal velocity) *exactly*, algebraically, for a single source — i.e. the
fixed formula predicts injected heating shuts off exactly when local gas
reaches the wind's terminal speed, which is exactly the physical picture (a
freely expanding wind asymptotically reaching v_inf, no longer needing
thermal-to-kinetic conversion). Confirmed numerically for the CWB setup's
actual parameters (`v_brake` computed independently == `v_inf1` to full float
precision). This is strong evidence the *formula* is now correct.

**Also fixed as part of the same change:** `_wind_ei3D`'s old code set
`pressure_rate` over the *entire* grid, not just inside the injection mask —
outside the mask, `energy_rate` was correctly 0 (from the mask baked into
`energy_rate_sources`), but `updated_density`/`speed` were the ambient
values, so the old formula spuriously injected a nonzero (negative)
`-(gamma-1)*0.5*rho_ambient*speed_ambient**2` pressure source *everywhere in
the domain*, every step. (`dummy_multi_star_wind`, an unrelated/unused
placeholder function with the same underlying bug, masks this correctly with
`* injection_mask` — `_wind_ei3D` didn't.) The new formula is naturally zero
outside the mask (both `energy_rate` and `density_rate` are zero there), so
no explicit masking is needed.

---

## 4. [NOT FIXED — priority blocker] The CWB sim goes to NaN at N=64 too, regardless of the item-3 formula, code-unit choice, or a suspicious limiter epsilon — genuine numerical stiffness, not a quick fix

After fixing item 3, re-ran the CWB setup (`verify_wind_fix.py` in scratch):
**still fully NaN, at both N=64 and N=128.**

**Correction to an earlier claim in this file:** an earlier version of this
section said this was a *regression* from item 3's formula fix, because the
old (buggy) formula's N=64 had previously been recorded as running clean.
**That claim does not hold up under direct re-verification.** `git stash`-ing
the item-3 fix back to the original buggy formula and re-running the exact,
unmodified `_cwb_setup.run_cwb(64)` (not a reconstruction — the real function)
also goes fully NaN. So this is a **pre-existing blocker present before item
3's fix too**, not something the fix introduced. (Why the earlier note
claimed otherwise is unclear — possibly a stale observation from an earlier
point in a prior session, under settings that no longer match the current
`_cwb_setup.py`. Don't trust that claim; trust this re-verification.) This
does **not** undermine item 3's fix itself — the formula correction is still
real and independently validated (see item 3's `v_brake == v_inf` identity)
— it just means fixing it doesn't unblock the sim on its own, and never did.

**Diagnosis so far** (via `config.runtime_debugging=True` / `checkify`,
`debug_nan2.py` in scratch):

- Failure is the same `pressure needs to be non-negative, minimum pressure
  nan` check as before, inside `_evolve_state_along_axis`, at a cell
  adjacent to a wind source (index (20,34,33) at N=64 — physical position
  ≈(-0.18, 0.04, 0.02) in box-centered coords, right at star 1's injection
  sphere).
- **Ruled out: pressure_rate itself going negative.** Floored
  `pressure_rate` at 0 (`jnp.maximum(pressure_rate, 0.0)`, a quick
  experiment, *not* kept in the diff) — identical failure, same cell, same
  message. So the NaN is not coming from the source term writing a literal
  negative pressure into the state.
- **Ruled out: CFL number.** Re-ran with `C_cfl=0.05` (vs. the test's 0.4, an
  8x tighter constraint) — identical failure, same cell. If this were a
  garden-variety CFL/timestep-too-large issue, a much tighter CFL should
  have at least moved or delayed it; it didn't, which points at the
  `_source_term_aware_time_step` machinery already shrinking `dt` to the
  point where further CFL tightening is not the limiting factor, and the
  problem is elsewhere (the reconstruction/Riemann step's actual robustness
  to the pressure *ratio* itself, not the step size).
- **Tried: `MINMOD` instead of `VAN_ALBADA_PP`.** Different failure
  signature — `pressure needs to be non-negative` at index (0,0,0), which
  looks like "the whole domain is already NaN" (argmin/first-index behavior
  over an all-NaN array) rather than a fresh, localized failure. Consistent
  with the same underlying corruption, just caught later/differently by a
  less dissipative limiter, not a different root cause.
- **Ruled out: code-unit choice.** Built a parametrized version of
  `_cwb_setup.py` (`cwb_param.py` in scratch) that varies the code-unit scale
  (`length_temp`/`mass_temp`, i.e. `CodeUnits`' arbitrary length/mass
  normalization) while holding the *actual physical* separation, box size,
  masses, wind parameters and ambient density/pressure fixed — this matters
  because `_cwb_setup.py` hardcodes `SEPARATION`/`BOX_SIZE` as dimensionless
  numbers at one specific `code_length`, so naively changing `code_length`
  alone silently changes the physical size being simulated too; the
  parametrized version derives the dimensionless separation/box size from a
  fixed physical value each trial, to isolate the unit choice cleanly. Swept
  `length_temp`/`mass_temp` across **~10 orders of magnitude**, including a
  regime deliberately tuned so `RHO_AMBIENT`/`P_AMBIENT` land at O(1) instead
  of O(1e-17) (`length_temp=1e8 au, mass_temp=1.4e7 Msun`) — **every regime
  still failed identically**, with both the old and new pressure-rate
  formula. Mechanism: bringing the *ambient* values up to O(1) requires a
  large `code_length`; since the physical box size is fixed, that shrinks the
  dimensionless grid spacing, which shrinks the wind's dimensionless
  injection volume, which *inflates* the wind's dimensionless injected
  pressure even further (`pressure_rate` ∝ 1/injection_volume ∝
  1/grid_spacing³) — so the ambient/wind pressure ratio actually got *worse*
  (6×10²⁵ vs. the baseline's 6.6×10¹⁷), not better. A single global unit
  rescaling can't fix both ends of this at once. **Code units are not a
  viable fix.**
- **Ruled out (so far): a suspicious limiter epsilon.** While investigating
  the above, found that `_van_albada_limiter`
  (`astronomix/_finite_volume/_state_evolution/limiters.py:74`,
  used by both `VAN_ALBADA` and `VAN_ALBADA_PP`) uses
  `epsilon = 3 * grid_spacing`, added to *squared* forward/backward field
  differences. The standard van Albada TVD limiter in the literature uses
  `epsilon ~ (K*dx)**3` (cubic, not linear) specifically so it vanishes fast
  enough as resolution increases without swallowing genuine intermediate
  gradients — with the current linear form, at this test's grid spacing
  (~0.016-0.03) `epsilon` (~0.05-0.09) is ~4-5 orders of magnitude larger
  than the canonical cubic form would give (~1e-5), meaning *any* field
  difference below ~0.05 gets the limiter's "unlimited central difference"
  branch rather than real limiting — plausible in an intermediate
  ambient-to-wind transition zone. Tested directly: patched to
  `epsilon = (3 * grid_spacing) ** 3` and re-ran the baseline N=64 case —
  **identical failure, same cell, same message.** Reverted (not kept in the
  diff). Doesn't rule out that epsilon's exponent being wrong as *a* bug, but
  it's not *this* bug, at least not on its own.
- **Ruled out: `num_injection_cells` (injection-region resolution), tested
  thoroughly across N=64/128/256.** Extended `cwb_param.py` (scratch) to take
  an explicit `num_injection_cells` override and swept it well past the
  default `num_cells // 32`:
  - N=64 (default 2): tried 2, 4, 8, 16, 32 — **all fail.** `pressure_rate_at_rest`
    dropped monotonically as expected (8.3 → 1.0 → 0.13 → 0.016 → 0.002, the
    expected 1/injection_volume ∝ 1/nic³ scaling), a **~4000x reduction in
    the ambient/wind pressure ratio** (6.6e17 → 1.6e14) — still fails every
    time. At `nic=32` the injection radius reaches half the box size (a
    physically nonsensical "injection region," at that point occupying most
    of the domain), and the failure shifts to a domain-boundary cell —
    likely a different (boundary/open-BC) failure mode taking over rather
    than a real test of the hypothesis at that extreme.
  - N=128 (default 4): tried 4, 8, 16 — **all fail** (baseline reproduces the
    original session's index (89,64,65) exactly). Same monotonic
    `pressure_rate_at_rest` decrease, same outcome.
  - N=256 (default 8): tried 8, 32 — **all fail**, same pattern.
  - The failure *index* consistently shifts outward as `nic` grows (tracking
    the edge of the growing injection sphere, not a fixed absolute
    location) — so this isn't an unrelated indexing bug; it's genuinely tied
    to the wind-injection region. It just isn't fixed by making that region
    bigger, even by orders of magnitude in the pressure ratio.
- **Ruled out: N-body.** All the above runs (and the original test) track
  the wind sources' positions via the N-body integrator
  (`nbody_config.nbody=True`). Disabled it (fixed star positions at
  `wind_params.wind_injection_positions`, matching `_cwb_setup.py`'s
  original commented-out non-orbiting design) and re-ran the N=64 baseline
  — **still fails**, same signature, at a nearby index. N-body is not
  a contributing factor; the bug is in the wind-injection/FV-evolve
  interaction itself, independent of how the source position is supplied.
- **Scale of the problem:** computed analytically for the CWB setup's actual
  parameters at N=64 — `energy_rate ≈ 12.5`, giving `pressure_rate` at rest
  `≈ 8.3` (code units), against `P_AMBIENT ≈ 1.3e-17`. That's a **~17 orders
  of magnitude** pressure contrast, injected into a spherical region only
  `num_injection_cells = num_cells // 32 = 2` cells in radius, and (per the
  code-units experiment above) this ratio is *intrinsic* — it doesn't shrink
  under any unit rescaling, since it's a real physical ratio (wind base
  pressure vs. warm-ISM ambient pressure) invariant to how we choose to
  represent it numerically.

**Five independent, well-motivated knobs have now been ruled out — formula
choice, code units, one specific limiter epsilon, injection-region size (up
to a ~4000x pressure-ratio reduction, tested at N=64/128/256), and N-body —
all with the exact same failure signature.** That consistency across such
different levers is itself informative: this is not a tunable-parameter
problem, and probably not a "make one quantity less extreme" problem either
(the injection-radius result specifically argues against that: a 4000x
reduction in the ambient/wind ratio produced *zero* qualitative change).
Remaining plausible directions for next session, in roughly increasing order
of how invasive they are:

1. **Sub-cycle the wind source term** — apply `_wind_ei3D`'s source term in
   several smaller sub-steps within one hydro `dt`, rather than one
   `source_term * dt` shot, so the pressure buildup in the injection cells
   doesn't overshoot within a single Riemann-solver call. This is
   qualitatively different from everything ruled out above (a change to the
   *time integration* of the source term, not to any single number), and
   matches how genuinely stiff source terms are usually handled elsewhere.
2. **Look past the limiter for `VAN_ALBADA_PP`'s actual positivity
   guarantee** — the epsilon experiment above ruled out *that specific*
   epsilon as the culprit, but `reconstruction.py`'s own positivity-clamping
   logic (`eps = 1e-14` at `reconstruction.py:188`, an absolute threshold
   compared directly to raw, unnormalized density/pressure differences) is a
   separate, not-yet-tested suspect — same family of bug (absolute epsilon
   vs. wildly-scaled field), different file. Worth the same
   patch-and-rerun treatment as the limiters.py experiment above.
3. **Instrument the actual failing timestep directly** — every experiment so
   far has diagnosed via the `checkify` failure alone (final error + index);
   nobody has yet stepped through the sim by hand (bypassing the jitted
   `while_loop`, e.g. by calling `_step` a few times directly and inspecting
   `primitive_state`/`dt` after each call) to see *which* step first goes
   bad and what the state looks like immediately before it. Given how
   resistant this has been to five different hypotheses, direct inspection
   of the actual failing step is probably more efficient at this point than
   another hypothesis-and-rerun cycle.
4. Re-examine whether the CWB test's physical parameters (real O-star
   mass-loss rate / wind velocity against a diffuse warm-ISM ambient) are
   just inherently too stiff for an explicit FV scheme at N=64-256, in which
   case the test's ambient density/pressure or the wind parameters
   themselves may need adjusting to something less extreme, purely to keep
   the test numerically tractable (would need to preserve whatever makes the
   test physically meaningful, i.e. still highly supersonic — check with
   whoever owns the physical realism requirements before doing this).

Scope note (unchanged from before): `_wind_ei3D` is the shared 3D
thermal-energy-injection scheme, used by any 3D EI wind sim, not just this
test — so is item 4, once diagnosed further.

### 2026-09-18: ROOT CAUSE FOUND (direction #3 above, done) — `VAN_ALBADA_PP` is silently inert for `config.split = SPLIT`, the combination `_cwb_setup.py` actually uses

Did direction #3 above first: bypassed the jitted `while_loop` entirely and
replicated `_step`'s body (from `_integrate_core` in
`astronomix/time_stepping/time_integration.py`) as a plain Python loop,
calling the same (individually-jitted) building blocks directly, with a
host-side NaN/negative-pressure check after *every* sub-stage (scratch
`instrument_cwb.py`). This pinned the failure down immediately — it is not a
many-steps-in drift, it happens on the very first hydro sweep of the very
first step:

1. Pre-step state (N=64): uniform ambient, `p=1.25779e-17` everywhere. Fine.
2. After `_iteration_level_injections` (the wind source term): pressure peak
   rises to `6.68e-4` in the injection cell(s) — a large (~5e4x) but
   perfectly finite, non-NaN, non-negative jump. Fine — confirms item 3's
   fixed formula is, again, not the culprit.
3. After `_iteration_level_continuous_updates`: unchanged. Fine.
4. The **very first** Strang-split axis sweep (`_evolve_state_along_axis`,
   axis=1/x, `sub_dt=dt/2`) immediately produces pressure as low as
   `-561580` and 40 NaN cells, first bad index `(19,33,33)` — matching (up to
   an indexing/ghost-cell-offset convention) the originally-reported failing
   cell `(20,34,33)`.

**Root cause**, traced through `reconstruction.py`/`evolve_state.py`:

- The SPLIT path's reconstruction, `_reconstruct_at_interface_split` →
  `_calculate_limited_gradients`, treats `VAN_ALBADA_PP` completely
  identically to plain `VAN_ALBADA` (`limited_gradients.py:89-95`: one
  shared `_van_albada_limiter` call, no special-casing at all).
- The actual positivity-preserving clamp — the whole `eps=1e-14`-guarded,
  multi-dimensional alpha/kappa/beta clamp block, `reconstruction.py:185-380`
  — lives *only* inside `_reconstruct_at_interface_unsplit`, which is only
  ever called from `_evolve_gas_state_unsplit_inner`, i.e. only when
  `config.split == UNSPLIT`.
- Separately, `_evolve_gas_state_unsplit_inner` *also* applies an
  unconditional `jnp.maximum(density/pressure, params.minimum_density /
  minimum_pressure)` positivity floor after its sweep (`evolve_state.py`,
  added 2026-09-15/16 per its own comment, for an unrelated SILCC-ISM M4/M5
  bug). `_evolve_state_along_axis` (the SPLIT path) has **no equivalent
  floor at all** — its only positivity handling is a `checkify.check` that
  *reports* a violation when `config.runtime_debugging` is on, but never
  clamps anything.

So for this test's actual configuration (SPLIT + MUSCL + VAN_ALBADA_PP), the
"positivity-preserving" limiter choice documented at length in
`_cwb_setup.py`'s own `SimulationConfig(...)` comment provides **zero actual
protection** — the same silently-inert-config-knob bug pattern that same
comment already diagnosed once for `PositivityConfig` (also dead for this FV
path), just one layer deeper. This is why the five previously-ruled-out
levers (formula, code units, limiter epsilon, injection size, N-body) all
failed identically: none of them touch this gap, and the real culprit — a
legitimate, physically-expected steep single-cell pressure spike from the
wind injection, MUSCL-extrapolated to the interface with no positivity clamp
whatsoever — was never on that list. (Direction #2 above was on the right
track but too narrow: it isn't that the clamp's epsilon is miscalibrated for
SPLIT, it's that the clamp doesn't run for SPLIT at all.)

**First fix attempt tried: switch `config.split` from `SPLIT` to `UNSPLIT`**
(which does wire up both the clamp and the floor). Blocked on two things:

- `UNSPLIT` requires `config.time_integrator = RK2_SSP`, not `MUSCL` (hard
  `ValueError` otherwise) — an easy swap, not a real blocker.
- With that swap, a full `run_cwb`-style run (`time_integration` with
  `runtime_debugging=True`) still fails, but with a **different** error:
  `jax._src.checkify.DivisionByZeroError` inside `speed_of_sound`'s `p/rho`,
  raised from `_cfl_time_step`'s `config.split == UNSPLIT`-only branch
  (`_timestep_estimator.py:140`, `c = speed_of_sound(rho, p, gamma)` called
  directly on the *whole* state array — unlike the SPLIT branch's
  `get_wave_speeds`, which only ever sees pairwise interface-adjacent
  slices). Checkify only reports the first error hit anywhere across the
  whole `while_loop`, so unlike the SPLIT bug above this is *not* necessarily
  a step-0 failure — isolated a manual replica of
  `_source_term_aware_time_step`'s exact two `_cfl_time_step` calls at step 0
  for both SPLIT and UNSPLIT (scratch `check_hypothetical_state.py`):
  neither literally produces a zero/negative/NaN density at step 0 (checked
  directly, `n_zero=0` for both), so this second bug is a genuine *separate*,
  not-yet-root-caused UNSPLIT-path issue arising at some later iteration —
  not an artifact of the split→unsplit switch being wrong in itself. Not
  pursued further this session.

**Recommended fix direction (not yet implemented — needs a decision before
touching shared code):** rather than chasing the UNSPLIT path's own separate
bug, add a positivity floor to the SPLIT path itself, mirroring
`_evolve_gas_state_unsplit_inner`'s `jnp.maximum(density/pressure,
params.minimum_density/minimum_pressure)` call, inside or right after
`_evolve_state_along_axis` in `evolve_state.py`. This fixes the *actually
configured* scheme (SPLIT+MUSCL, which is what `_cwb_setup.py`'s own
documentation says it wants) instead of switching away from it into a
different, still-broken path, and is a general fix for FV SPLIT runs, not
just this test — same category of gap as the already-fixed UNSPLIT
positivity floor (2026-09-15/16). **Caution:** `params.minimum_density` /
`minimum_pressure` both default to `1e-14`, which is *larger* than this
test's ambient density/pressure (`RHO_AMBIENT`/`P_AMBIENT` ≈ `1.26e-17`) —
applying this floor unconditionally (as the UNSPLIT path already does) would
artificially raise the *entire* ambient medium by ~1e3x, not just clamp
genuine violations. Pass an explicit, ambient-appropriate
`minimum_density`/`minimum_pressure` (e.g. some small fraction of
`RHO_AMBIENT`/`P_AMBIENT`) in `_cwb_setup.py`'s `SimulationParams(...)`
alongside this fix, rather than relying on the global default.

Scratch scripts from this session (not committed, rerun from this diff if
needed): `instrument_cwb.py` (per-substage NaN instrumentation, the main
diagnostic), `verify_unsplit_fix.py` (SPLIT→UNSPLIT swap test, surfaced the
second div-by-zero bug), `check_hypothetical_state.py` (isolated
`_cfl_time_step`/`_wind_injection` check ruling out a step-0 zero-density
cause for that second bug).

### 2026-09-18 (same session, continued): fix implemented — real improvement, but NOT a full fix; a second, distinct mechanism found

Implemented the recommended direction above, gated on `config.limiter ==
VAN_ALBADA_PP` (so every other limiter's numerics/conservation are exactly
unchanged):

1. `_reconstruct_at_interface_split` (`reconstruction.py`): clamp the
   reconstructed interface density/pressure to `params.minimum_density` /
   `minimum_pressure` before they reach the Riemann solver — this is what
   the UNSPLIT path's PP variant achieves by construction; SPLIT had nothing
   preventing a negative *interface* value from ever being computed.
2. `_evolve_state_along_axis` (`evolve_state.py`): added the same
   `jnp.maximum(density/pressure, params.minimum_*)` post-sweep floor
   `_evolve_gas_state_unsplit_inner` already has, as defense in depth.
3. `_cwb_setup.py`: pass explicit `minimum_density=1e-3*RHO_AMBIENT`,
   `minimum_pressure=1e-3*P_AMBIENT` (the global `SimulationParams` default,
   `1e-14`, is *larger* than this test's ambient `~1.26e-17` and would have
   inflated the whole ambient medium).
4. Fixed a stale comment on `SimulationParams.minimum_density`/
   `minimum_pressure` in `simulation_params.py` claiming they're "currently
   only used in finite difference mode" — false since the 2026-09-15/16
   UNSPLIT floor and now this SPLIT floor/clamp both use them in FV mode too.

**Re-verified against the exact same step-0/axis-1-sweep instrumentation
(`instrument_cwb.py`) that originally reproduced the bug.** Real,
measurable improvement:
- The large finite negative-pressure blow-up (`-561580` at the original
  failing index) is **completely gone** — `neg(p)=0` after the fix, where it
  was `8` before.
- NaN cell count after the first axis-1 sweep dropped from **40 → 16**.
- Density stays positive everywhere (correctly floored to the new
  `1e-3*RHO_AMBIENT`-scaled value, not NaN).

**But 16 NaN pressure cells remain** at step 0's very first axis sweep,
including at essentially the same location as before (`(20,33,33)`, one
cell removed from the originally-reported `(20,34,33)`). Traced this to a
**second, distinct, deeper mechanism** (scratch `debug_hllc.py`, which
reconstructs the axis=1 interfaces exactly as `_evolve_state_along_axis`
does and manually recomputes `_hllc_solver`'s `S_L`/`S_R`/`S_star` and its
three division denominators by hand): at the failing interface, none of
`_hllc_solver`'s denominators (`s_star_den`, `S_L - S_star`, `S_R - S_star`)
are actually degenerate (checked directly: no near-zero or exact-zero
denominators anywhere in the array) — so this is *not* the classic
HLLC-degenerate-star-state bug it first looked like. Instead, the
**reconstructed interface velocity itself is already insane before the
Riemann solver ever runs**: `u_R = -9.3e9` (code units) at that interface,
with `u_hat`/`c_hat`/`S_L`/`S_R` all similarly huge. My density/pressure
clamp doesn't touch velocity at all, so this passes straight through.

**Mechanism:** `_reconstruct_at_interface_split`'s MUSCL predictor step
projects the limited primitive-variable gradients through `A_W`, the
"primitive Jacobian" (per its own comment, "not an actual Jacobian"), which
has a `1/rho` entry (`A_W.at[axis, pressure_index].set(1/rho)`) using the
**current cell's own (unclamped) density** — right next to the wind
injection, ambient density is `~1.26e-17`, so `1/rho ~ 8e16`; multiplied
through a large pressure gradient (the injection creates a ~5e4x single-cell
pressure jump) inside `projected_gradients`, this explodes the *predicted*
velocity at the interface to something like `-9.3e9`. This is a
characteristic-projection reconstruction issue, not a positivity issue in
the density/pressure sense — clamping density/pressure *after* the fact
doesn't fix an already-nonsensical velocity, which then presumably overflows
somewhere further downstream (`speed_of_sound`, `conserved_state_from_primitive`'s
kinetic-energy term, or the HLLC star-state formulas) into literal NaN via
an inf-inf or inf*0 pattern that wasn't tracked down further this session.

**Not yet fixed — this needs a decision before touching the reconstruction's
core math:** the general, standard MUSCL/TVD fix for this class of bug is to
clamp *every* reconstructed primitive variable (not just density/pressure)
to lie within the range spanned by the two neighboring cell-centered values
("no new extrema") — a stronger, more standard safety net than what's
implemented here, but more invasive since it touches the reconstruction for
every variable, not just the two already known to need positivity. Given
this is a distinct numerical-method concern (the `A_W`/primitive-Jacobian
projection's `1/rho` blow-up) from what this session was scoped to fix
(positivity of density/pressure), and a wrong fix here risks accuracy
regressions across every SPLIT+MUSCL simulation, this was intentionally left
for a follow-up decision rather than pushed through silently. Scratch
`debug_hllc.py` reproduces the exact failing interface and has the full
`S_L`/`S_R`/`S_star`/denominator trace if picking this back up.

**Status: item 4 is closer but still open.** The positivity-floor fix is a
real, verified improvement and worth keeping regardless of what happens
next (it's strictly gated on VAN_ALBADA_PP, zero risk to other configs), but
N=64 still does not run clean.

### 2026-09-18 (same session, continued again): the VAN_ALBADA_PP fix above was reverted by request — pivoted to `limiter=MINMOD` instead, and found the failure is not (only) numerical

The positivity-floor/clamp fix above (`reconstruction.py`, `evolve_state.py`,
`simulation_params.py` comment, `_cwb_setup.py`'s explicit
`minimum_density`/`minimum_pressure`) was reverted at the user's request
(kept only as the write-up above, for reference/reuse later). Redirected to
diagnose and fix the test's *other* previously-untried limiter,
`MINMOD` — `_cwb_setup.py` now sets `limiter=MINMOD` (was `VAN_ALBADA_PP`),
everything else unchanged (`split=SPLIT`, `time_integrator=MUSCL`,
`riemann_solver=HLLC`). No other code changed this round.

**MINMOD completely fixes the original failure.** Re-ran the same
step-by-step instrumentation that reproduced the VAN_ALBADA_PP bug at step 0
(scratch `instrument_cwb.py`): 15 full steps (5 axis sweeps each), zero NaN,
zero negative pressure, throughout — MINMOD's limited gradient collapses to
(near) zero at the sharp injection-boundary jump (differing-sign
forward/backward differences), which never feeds a large gradient through
`_reconstruct_at_interface_split`'s `A_W`/`1/rho` Jacobian in the first
place (see the `A_W` mechanism described above under the VAN_ALBADA_PP
write-up — same underlying reconstruction fragility, MINMOD's more
conservative limiter just never triggers it here).

**But the full run to `t_end=T_END` (the full orbital period, `0.167552`)
still ends up entirely NaN** (scratch `verify_minmod_full.py`: runs to
completion without raising — no checkify — but the final state is 100% NaN,
262144/262144 cells). Bisected the failure time (scratch
`bisect_minmod_failure.py`, reusing one jit-compiled `_time_integration`
across trials by varying `params.t_end`, a dynamic leaf, so no
recompilation between the ~22 bisection trials): **clean up to
`t≈0.0259195`, NaN by `t≈0.0259196`** — i.e. only **~15.5% into the full
run**, not near the end. (Note: once *any* single cell goes NaN, the next
step's domain-wide `dt` estimate — `_cfl_time_step`'s `jnp.max` reductions —
itself becomes NaN, `jnp.maximum(nan, x) = nan`, and that NaN `dt`
instantly poisons the *entire* domain on the very next step, since `dt`
multiplies every flux/conserved-state update. This is why `checkify` always
reports the degenerate `pressure ... at index (0,0,0)` signature — that's
just `argmin` over an already-fully-NaN array — even when truncating
`t_end` to just past the bisected point; scratch `locate_minmod_failure.py`
confirmed this directly.)

**Inspected the state at the last clean bisection point instead**
(`t≈0.0259195`, no checkify needed since it's still finite — scratch
`inspect_last_clean.py`): the minimum-pressure cell is at grid index
`(62,62,48)` of `64^3` — box-centered coordinates `(0.477, 0.477, 0.258)`,
i.e. **1-2 cells from the domain's `+x`/`+y` edge** (`BOX_SIZE/2 = 0.5`),
not anywhere near the wind sources (`x=±0.2` at `t=0`). Density/pressure
there are far *above* true ambient (`rho≈2.1e-11`, `p≈3.2e-10`, vs.
`RHO_AMBIENT≈4.5e-17`/`P_AMBIENT≈1.3e-17`) but velocity is huge
(`speed≈85`, close to the wind's own terminal-velocity scale) — this is a
**largely unshocked, high-Mach, free-streaming wind having reached the open
boundary directly**, not a swept-up ambient shell. `_open_left_boundary`/
`_open_right_boundary` (`boundaries.py`) are plain zero-gradient
extrapolation (ghost cells ← nearest interior cell) — nothing exotic there;
the failure is consistent with the *same* `A_W`/`1/rho`-Jacobian
reconstruction fragility as the original VAN_ALBADA_PP bug, just triggered
by the wind's leading edge meeting the box boundary instead of by the
injection source.

**Root scope issue found, likely the actual trigger:** `git log --follow -p
-- _cwb_setup.py` shows `T_END` was originally `4e-3` (`b9d2058`), with an
explicit comment — still present, now stale — that this value was "chosen
empirically... so the two wind bubbles have merged... while the bow shocks
stay comfortably inside the domain (max |x| well below BOX_SIZE/2)". A later
commit (`bab53b4`, "tune CWB setup: eccentric binary params, N-body orbit,
open boundaries") changed `T_END` to `Period` (the full N-body orbital
period, `≈0.167552`) — **~42x larger** — apparently to study a full orbit
once N-body/eccentricity were added, without revisiting whether the bow
shock still stays inside the domain for that much longer `t_end`. Bisection
above shows it very much does not: the wind exits the box at only ~15.5% of
the current `T_END`, i.e. at `t≈0.0259`, only ~6.5x past the *old* `T_END`
of `4e-3` — squarely consistent with the "stays inside the domain" design
budget having been calibrated for the old, much shorter `T_END` and never
re-checked for the new one.

**This reframes the failure**: it isn't purely a numerical-stability bug
independent of the physical setup — the *current* `T_END=Period` asks the
solver to run well past the point where BOX_SIZE can contain the flow,
which was true even before N-body/eccentricity were added (the old `T_END`
was chosen specifically to avoid this). Two independent, non-exclusive fix
directions, not yet decided between:

1. **Setup-level (cheap, narrow scope):** reduce `T_END` back down to
   something that keeps the bow shock inside the domain (bisection above
   puts the safe ceiling somewhere below `t≈0.0259`; the old `4e-3` is
   comfortably inside that, or a value close to but under `0.0259` would
   capture more physical evolution while staying safe). Sidesteps the
   open-boundary blow-up entirely rather than fixing it, at the cost of no
   longer capturing a full orbital period (which the N-body/eccentricity
   work presumably wanted to study — check with whoever added `bab53b4`
   before assuming this is fine).
2. **Solver-level (broader, more invasive):** keep `T_END=Period` and
   instead make the reconstruction robust to a high-Mach, near-vacuum
   free-streaming wind exiting an `OPEN_BOUNDARY` — e.g. generalize the
   interface clamp from the (reverted) VAN_ALBADA_PP fix above to work
   under any limiter, not just VAN_ALBADA_PP, or enlarge `BOX_SIZE` so the
   wind stays contained for a full period at this resolution. More general
   (fixes any future setup that pushes a supersonic wind to an open
   boundary at this resolution), but bigger, and doesn't have a verified
   fix yet — only a verified diagnosis of where/why it happens.

Scratch scripts from this round (not committed): `instrument_cwb.py`
(updated to MINMOD, confirms 15 clean steps), `verify_minmod_full.py` (full
run, all-NaN result), `bisect_minmod_failure.py` (binary search on `t_end`,
one compiled executable reused across trials), `locate_minmod_failure.py`
(confirms the NaN-poisons-global-dt cascade), `inspect_last_clean.py`
(locates the boundary-adjacent hot cell at the last clean bisection point).

### 2026-09-19: root-caused via `first_order_fallback=True`; three fix variants tried against the real N=64 run, best one delays but does not eliminate the boundary failure

User confirmed directly: `config.first_order_fallback=True` (bypasses
`_reconstruct_at_interface_split` entirely, using the raw neighboring cell
values as interface states) **runs the CWB N=64 MINMOD setup clean**. This
unambiguously localizes the bug to `_reconstruct_at_interface_split` itself
— not the Riemann solver, boundary handler, CFL estimator, wind injection,
or N-body (`_cfl_time_step`'s SPLIT branch and `first_order_fallback` both
bypass reconstruction identically, so this had already been implicitly
ruled out, but the hint makes it explicit).

**Re-diagnosed the boundary failure with actually-correct state** (earlier
manual step-by-step attempts near t≈0.0259 kept landing back on the stable
side — foiled by `cwb.nbody_state` being the *initial* t=0 star positions,
not the true positions at that time; `config.nbody_config.nbody=True` means
the stars orbit throughout). Fixed two ways, cross-checked against each
other: (a) an analytic Kepler reconstruction (exact for this setup's
circular, e=0 orbit, since `Period == T_END` by construction so orbital
phase = t/Period exactly) via `binary_starting_orbits_at_phase`, correctly
threaded into *both* `params.nbody_params.nbody_state` (needed by the first
`_estimate_cfl_dt` call each step, evaluated *before* that step's own N-body
advance) and the loop carry — scratch `step_with_correct_nbody.py`; (b)
`config.return_snapshots=True` with `snapshot_settings.return_states=True`
and many snapshots, which threads the framework's own real `LoopState`
(real N-body state, real key) through the *entire* natural `while_loop` with
zero manual reconstruction — scratch `snapshot_trace.py`. Both agree: last
clean state at t≈0.02592/0.02610 (depending on which fix, if any, is
active) has the same hot cell each time, always at grid index `(62-63,
62-63, ~15-50)` — i.e. **always at the `+x`/`+y` domain edge simultaneously
(a genuine 3D corner/edge line, not a generic single-face boundary
effect)** — density/pressure there are far above true ambient but still
tiny in absolute terms (`rho~2e-11`, `p~3e-10`) with high velocity
(`speed~85`, close to the wind's own terminal-velocity scale): a largely
unshocked, high-Mach, free-streaming wind reaching the open boundary. Also
ruled out two side hypotheses directly: an artificially tiny forced `dt`
(mimicking `exact_end_time`'s or a snapshot boundary's landing-clamp) does
**not** reproduce the failure on its own (tested `dt` from `1e-6` down to
literally `0` on the known-clean state — all fine, scratch `test_tiny_dt.py`);
and the bisection search's own `exact_end_time`-forced tiny last-step is
*not* an artifact of the diagnostic method (confirmed via the native
snapshot trace, which never forces an off-schedule tiny step and still
shows the same natural transition).

**Three fix variants implemented and tested against the real N=64 run**
(each verified by rebisecting `bisect_minmod_failure.py` with the change
in place — a *different*, unbiased bisection each time, not just checking
the one known point):

1. **Fixed global floor on interface density/pressure only** (the
   VAN_ALBADA_PP-era fix, un-gated) — **zero effect**: rebisection gave the
   *bit-identical* failure window (`t≈0.0259195`/`0.0259196`) as with no
   fix at all. Makes sense in hindsight — density there (`~2e-11`) is
   nowhere near any sane floor value, so the floor never engages.
2. **Independent per-variable clip** of each reconstructed interface value
   to the range spanned by its two neighboring cells ("no new extrema") for
   *every* variable, not just density/pressure — plus flooring `rho` in the
   `A_W` Jacobian's `1/rho` entry (unrelated mechanism, targets the
   injection-region bug, kept from the prior round) and a NaN/Inf
   sanitization fallback (`jnp.where(isfinite(...), ..., cell_here/
   cell_minus)`) before the clip, since `jnp.clip(nan, lo, hi)` is still
   `nan`. **Real, measurable improvement**: delayed the failure from
   `t≈0.025920` (15.47%) to `t≈0.026099` (15.58%) — small but genuine and
   reproducible, confirmed via rebisection landing on a distinctly
   different (later) window each time this variant was (re)tested. The
   NaN/Inf sanitization specifically added **nothing further** on its own
   (bisection identical with vs. without it) — the value reaching the clip
   is apparently always finite, just occasionally still wrong.
3. **Single scalar alpha per cell**, shared across every variable and both
   extrapolation directions, scaling the *entire* reconstructed perturbation
   down uniformly to the largest value that keeps every variable in bounds
   (the same idea as `_reconstruct_at_interface_unsplit`'s own VAN_ALBADA_PP
   positivity machinery, just a single scalar instead of its full
   alpha/kappa/beta formula) — theoretically the more correct fix (keeps
   density/velocity/pressure mutually consistent, so the conserved state's
   implied internal energy can't go negative from one clamp fighting
   another), but **empirically worse**: regressed the failure to
   `t≈0.017381` (10.37%), earlier than no fix at all. One misbehaving
   variable in a cell drags the shared alpha toward zero for every other
   variable in that same cell too — evidently far more diffusive overall
   across the grid than independently clamping each variable, enough to
   destabilize something else sooner. **Reverted** back to variant 2.

**Current state of `reconstruction.py`: variant 2 (independent clip) is
what's in the tree.** It is a real, safe improvement — confirmed to not
regress the already-clean MINMOD injection-region behavior (that path's own
`limited_gradients` never triggers the clip in the first place, so it's a
no-op there) and to strictly reduce to `first_order_fallback`'s own values
in the worst case — but **does not eliminate the boundary failure**, only
delays it by a modest ~0.7 percentage points of the run.

**Open question, not resolved this session:** why does *global*
`first_order_fallback=True` (literally every cell, every step) run clean,
when a reconstruction that reduces to the *exact same values* only in the
specific cells/steps that need it does not? Two live hypotheses, neither
tested yet: (a) the instability is genuinely cumulative/multi-step (a slow,
compounding erosion at this exact corner across many steps — the observed
min-pressure/min-density at the hot cell *decreases* smoothly across
consecutive snapshots rather than jumping discontinuously, consistent with
this), so a locally-triggered clip that engages *after* the state is
already partway degraded doesn't act early enough, whereas blanket
first-order changes the trajectory from step 0 onward; (b) something about
`_boundary_handler`'s per-axis-sequential ghost-cell filling
(`_apply_axis_bcs` called for x, then y, then z, each on the array as
already modified by the previous axis — standard practice, but not
independently verified correct at a genuine edge/corner where two open
boundaries meet, which is exactly where every failing cell found this
session sits) behaves differently under first-order vs. this MUSCL
reconstruction in a way unrelated to the reconstruction's own overshoot
behavior. Neither investigated further this session in the interest of
reporting back with the (partial, honest) result in hand rather than
continuing to iterate blind.

Scratch scripts from this round (not committed): `step_with_correct_nbody.py`
/ `snapshot_trace.py` (the two independent, verified-correct-state
diagnostics described above), `test_tiny_dt.py` (rules out tiny forced dt),
`debug_boundary_step.py` / `debug_dt_nan.py` / `checkpoint_and_step.py`
(earlier, flawed attempts using the wrong t=0 nbody_state — kept for the
record of what *not* to repeat; `checkpoint_and_step.py` additionally
needs `orbax-checkpoint`, not installed in `venv_grav_source`).

### 2026-09-19 (same day, continued): independent-clip fix reverted at user's request; both open hypotheses tested directly; one ruled out, the other confirmed but still unfixed; two more fix variants tried and also fell short

User explicitly declined to keep variant 2 (the independent per-variable
neighbor-range clip) in the tree and asked for the two open hypotheses
above to be tested directly instead. `reconstruction.py` and
`evolve_state.py` were reverted to `HEAD` (pristine, no fix of any kind)
before this round started.

**Hypothesis (a) — cumulative/multi-step erosion at the corner: confirmed.**
Tracked the specific hot cell (and its immediate neighborhood, since the
exact argmin cell drifts by 1-2 cells as the wind front advances) across
~260 natural snapshots from `t=0` to just past the known failure, using the
vanilla (no-fix) code (scratch `track_hot_cell.py`). Pressure/density there
decay **smoothly and monotonically over at least the last ~16 consecutive
snapshots** (snapshot 241 to 257, `t≈0.0243` to `0.0259`) — many real
timesteps of continuous decay, not a single-step collapse. (There's an
earlier wobble/partial-recovery around snapshots 231-240, so the decay
isn't perfectly monotonic from very early on, but the final approach to
failure clearly is.) This is consistent with a genuinely physical
rarefaction/evacuation developing at the domain corner as the supersonic
wind front sweeps past an open boundary with nothing flowing in to replace
what leaves — not a one-off reconstruction glitch.

**Hypothesis (b) — `_boundary_handler`'s per-axis-sequential ghost-cell
fill order interacting badly at the edge/corner: ruled out, decisively.**
Monkeypatched a reversed-order boundary handler (z, then y, then x, instead
of the real x, then y, then z) into `evolve_state.py`'s namespace before
any JIT tracing (scratch `test_boundary_order.py` for the full-run check,
`bisect_boundary_order.py` for the precise bisection) and reran the exact
same bisection search. Result: **bit-identical failure window**
(`t≈0.0259195`/`0.0259196`, matching the untouched baseline to the same
precision) — the fill order has *zero* measurable effect. Also reasoned
through the sequential-fill mechanics by hand beforehand (with
`num_ghost_cells`, confirmed 2 here, i.e. padded shape `68^3`): each axis's
fill broadcasts across the *full* extent of the other axes, including
whatever the previous axis already wrote — which, worked through
carefully, correctly propagates the true interior corner cell's value into
the ghost corner regardless of order. The empirical result confirms this
reasoning. **This hypothesis is closed; do not re-open without new
evidence.**

**Two more fix variants tried, informed by hypothesis (a) — both fell
short, for informative reasons:**

3. A **loose, magnitude-based safety net** instead of the tight
   neighbor-range one: density/pressure only floored to stay positive
   (`params.minimum_density`/`minimum_pressure`, a genuine positivity
   floor, not a neighbor-range clip), and each velocity component clamped
   only if it exceeds `cell_here_v ± 50*(|cell_here_v| + local_sound_speed
   + 1e-3)` — i.e. only intervenes on overshoots many orders of magnitude
   beyond anything physically expected at that cell, leaving normal
   second-order behavior in healthy (even steep) flow regions completely
   untouched. Motivated directly by a diagnostic finding: instrumenting
   variant 3's (the reverted single-shared-alpha fix's) own alpha field at
   a real, early, still-fully-healthy point in the run (`t≈0.0173`, scratch
   `inspect_alpha_stats.py`) showed it clamping **~25% of the entire
   domain** to `alpha < 0.999` (many cells all the way to `alpha = 0`, full
   first order) — nowhere near just the eventual failure corner. "No new
   extrema relative to the two immediate neighbors" is evidently far
   stricter than positivity actually requires; it fires constantly in
   ordinary, unremarkable flow features (any mild local extremum in any one
   variable), not just near genuine danger. That explains both why variant
   3 (which ties every variable to one shared, therefore very easily
   triggered, scale factor) regressed so badly, and hints that variant 2's
   independent per-variable version was *also* over-triggering broadly,
   just less catastrophically. **Result: the looser criterion barely
   helped** — rebisection gave `t≈0.025832` (15.42%), statistically
   indistinguishable from (very slightly *worse* than) the unfixed baseline
   (15.47%), and clearly worse than variant 2's 15.58%. Cross-checked
   directly why: the actual values at the failure point are *not*
   obviously extreme in absolute terms (density `~2e-11`, far above any
   sane floor; velocity `~85`, roughly the wind's own terminal velocity,
   not absurdly large) — so a magnitude-based check essentially never
   engages there at all. This means the real mechanism is likely a
   *cross-variable inconsistency* introduced by the `A_W` Jacobian's
   coupling (e.g. the reconstructed state's implied conserved internal
   energy going slightly negative even though density, pressure and
   velocity each look individually unremarkable) rather than any single
   variable becoming absolutely extreme — consistent with why the tight
   neighbor-range clip (which *does* catch inconsistencies regardless of
   absolute magnitude) was the only variant so far to measurably help at
   all, and why a genuinely rigorous, CFL-aware positivity criterion (like
   the codebase's own real `VAN_ALBADA_PP` unsplit machinery, not any of
   this session's simplified approximations) is probably what's actually
   needed.

**Status at end of this round: `reconstruction.py` and `evolve_state.py`
are back to pristine `HEAD` — no fix of any kind is currently applied.**
Four fix variants have now been tried and rejected/reverted across the two
sessions (fixed floor: no effect; independent neighbor-range clip: modest
delay, declined by user; single shared alpha: regression; loose
magnitude-based: no real effect) — all sharing the same *ad hoc* character
the alpha-field diagnostic exposed as the actual problem. The one
principled path not yet attempted is porting the *real*, already-validated
`VAN_ALBADA_PP` positivity-preserving algorithm from
`_reconstruct_at_interface_unsplit` (its full alpha/kappa/beta quadratic
formula, derived to guarantee the *conserved* state's positivity under the
active CFL number specifically — not a neighbor-range heuristic) to the
split scheme, rather than continuing to invent new heuristics. This is a
substantially bigger undertaking than anything tried so far (that block is
~150 lines of genuinely non-trivial math) and was not attempted this
session in the interest of reporting back with confirmed findings rather
than another partial, ad hoc attempt.

Scratch scripts from this round (not committed): `track_hot_cell.py`
(confirms the smooth multi-snapshot decay), `inspect_alpha_stats.py`
(the 25%-of-domain over-triggering finding, run twice — once at step 0
showing zero false triggers there, once at `t≈0.0173` showing the
widespread over-triggering), `test_boundary_order.py` /
`bisect_boundary_order.py` (the reversed-fill-order test that ruled out
hypothesis (b)).

### 2026-09-19 (same day, continued a third time): the full VAN_ALBADA_PP port, done properly — genuinely fixes the *original* injection-region bug (the one this whole item started with); the boundary-corner failure still isn't fully solved, but its character has changed

User asked for the full port to be attempted (not another ad hoc heuristic).
`_cwb_setup.py` switched back to `limiter=VAN_ALBADA_PP` (the newly-ported
code is gated on it, so `MINMOD` would leave it dead code) — everything
else unchanged (`split=SPLIT`, `time_integrator=MUSCL`, `riemann_solver=HLLC`).

**First attempt (reverted): scale `limited_gradients` before the A_W
einsum.** Reused the unsplit path's exact alpha_density/kappa_pressure/beta
formula, applied to `limited_gradients` *before* it reaches the A_W
"primitive Jacobian" matrix multiply that builds `projected_gradients` for
the MUSCL predictor step — reasoning that if A_W's `1/rho` coupling (the
mechanism that turns an ordinary pressure gradient into a wildly unphysical
projected *velocity* gradient) only ever sees an already-positivity-safe
input, it couldn't produce a positivity-safe-but-still-huge output. **This
was wrong and made things dramatically worse**: failed at literally `t≈0`
(the very first step), worse than doing nothing at all. Root cause: A_W
*mixes* variables. A pressure gradient scaled down by `kappa_pressure`
(chosen using only pressure's own positivity requirement) still flows
through A_W's `1/rho` entry into the *velocity* row of
`projected_gradients` — entirely unconstrained by `beta` (the
velocity-specific, generally much stricter bound). Scaling an input to a
mixing matrix can't bound a blow-up that only exists in the matrix's
output.

**Second attempt (kept, with two follow-up fixes): scale the reconstructed
*output* instead.** Let `predictors`/`primitives_left`/`primitives_right`
be computed completely normally first (unscaled — i.e. the original,
pristine SPLIT+MUSCL code, predictor step and all), then apply the
alpha_density/kappa_pressure/beta formula to the *actual total delta*
(`primitives_{left,right} - primitive_state`, which already has whatever
mixing A_W introduced baked into it, by construction) rather than to the
pre-mixing gradient. This departs from the unsplit algorithm's assumption
of one *symmetric* difference per axis (`primitive_state -/+` the same
value): the MUSCL predictor step generally makes the left and right deltas
asymmetric. Nothing in the underlying derivation actually requires
symmetry — each factor only bounds "how far can `primitive_state` move in
this specific direction and stay positive" — so left and right are scaled
independently, each using its own delta. The `alpha_lax`/`C` cross-axis
weighting still needs the other two axes' global max wave speeds (cheap:
just `jnp.max(|velocity|+c)` over the domain, no extra gradient
computation needed this time, unlike the first attempt). `A1`/`A2` were
correspondingly simplified to a single-axis form (no cross-axis sum, since
this now runs once per axis on that axis's own delta, not on an aggregated
multi-axis tensor).

Two bugs found and fixed empirically, in order, each isolated with the same
step-by-step instrumentation (`instrument_cwb.py`) used throughout this
item:

1. **Missing NaN/Inf sanitization on `predictors`.** Re-added the same
   `jnp.where(isfinite(predictors), predictors, primitive_state)` fallback
   used in earlier (reverted) attempts — belt-and-braces, didn't change the
   outcome on its own here, but cheap and still correct to keep (the
   positivity scaling can only shrink an already-finite delta; it cannot
   repair one that's already NaN).
2. **The actual bug: `alpha_density`/`kappa_pressure` only guarantee
   *non*-negativity, not strict positivity.** Their limiting-case formula
   (`alpha = rho / |diff|`) is specifically designed to drive the scaled
   delta to *exactly* cancel `rho`/`p` in the worst case — landing at
   precisely `0.0`, not some small positive value. Diagnosed directly
   (scratch `debug_port_nan.py`, which calls `_reconstruct_at_interface_split`
   standalone and inspects its return value before/after the Riemann
   solver): the reconstruction's own output was completely clean (zero
   NaN) — but one interface had `rho=0.0`, `p=0.0` *exactly* (with a
   nonzero velocity surviving alongside it, since `beta`'s derivation
   doesn't force velocity to zero just because density vanished). This is
   a valid non-negative state, but `speed_of_sound(rho=0, p=0, gamma) =
   sqrt(gamma*p/rho)` is a literal `0/0` — NaN, not `0` — which then
   poisoned the whole HLLC flux at that interface. **Fix:** clamp both
   `_positivity_scale`'s output to `params.minimum_density`/
   `minimum_pressure` (a small strictly-positive floor) as the final step,
   purely to keep downstream `sqrt`/division well-defined — not the
   scaling's own job, just closing this one specific gap.

**Result of both fixes together: the original injection-region blow-up
(the bug this whole `FIXES_TODO.md` item started with, back when it was
first found under `VAN_ALBADA_PP` well before `MINMOD` was ever tried) is
now genuinely, robustly fixed.** Re-ran the same 15-step / 120-substage
instrumentation that caught the very first version of this bug — every
single substage check across every step is completely clean (zero NaN,
zero negative pressure), matching or exceeding MINMOD's own (already-clean)
behavior at the injection region.

**But the full run to `T_END` (`Period`, ~0.167552) still does not complete
clean** — rebisected: `t≈0.000217` (0.13% of the run), which is *earlier*
than the unfixed-MINMOD baseline's 15.47%, but the *character* of the
failure has changed. Extended the step instrumentation to 30 steps: no
NaN/negative pressure appears anywhere in those 30 steps (t up to
`0.000217`, right at the bisected boundary) — instead, `dt` **collapses by
~500x between step 14 and step 15** (from `~7.4e-6` down to `~1.5e-8`,
settling into a `~1.5e-9`–`2.8e-9` range for the remaining steps checked),
then the simulation continues *stably* (no further NaN in 30 steps) in
this much slower, stiffer regime. The bisected failure point
(`t≈0.0002169544182`) sits only slightly past where the 30-step check
stopped (`t≈0.000216908`) — reaching it would need several dozen more of
these now-tiny (`~1.7e-9`) steps, not yet run.

**Not yet diagnosed:** whether this sudden ~500x `dt` collapse reflects (a)
a genuinely new, legitimately stiff physical feature appearing around this
time (e.g. a shock steepening enough to demand much finer CFL resolution),
(b) an artifact of the new positivity floor creating an artificially sharp
feature that then forces an overly conservative CFL estimate, or (c)
something else. Given how much ground was already covered this session
(the actual, original bug now fixed; a materially different, much later
and differently-shaped remaining issue), this was intentionally left for a
follow-up rather than pushed through blind — reporting back with the
substantial, validated progress in hand.

**Current code state:** `reconstruction.py` has the full, working
positivity-preserving split reconstruction (output-scaling version, both
bugs fixed); `evolve_state.py` has a matching `VAN_ALBADA_PP`-gated
post-sweep floor (needed for a separate, tiny ~1e-14-scale negative
pressure from ordinary floating-point roundoff in the conserved-state flux
combination — not the same mechanism as the two bugs above); `_cwb_setup.py`
uses `limiter=VAN_ALBADA_PP`. All changes strictly gated on
`config.limiter == VAN_ALBADA_PP`, so every other limiter/config in the
codebase is provably unaffected.

Scratch scripts from this round: `debug_port_nan.py` (isolates
`_reconstruct_at_interface_split`'s own output from the Riemann solver,
found the exact `rho=0,p=0` degenerate interface), `instrument_cwb.py`
(updated to `VAN_ALBADA_PP`, extended to 30 steps, found the `dt` collapse).

### 2026-09-19 (same day, continued a fourth time): dt collapse mechanism located — a localized pressure/density ratio (temperature) spike, appearing within a single step, near (not at) the injection region

Located the specific cell(s) and mechanism behind the ~500x `dt` collapse
between step 14 and step 15 (scratch `find_dt_collapse_cell.py`): saved the
padded primitive state right before step 13, right before step 14, and
right after step 14 (all from the same manual step-loop instrumentation
used throughout this item), then directly computed the per-cell wave speed
(`|u_axis| + speed_of_sound(rho, p, gamma)`) for every axis at each saved
state and located the argmax.

- **Before step 13 / before step 14:** the max wave speed is already
  elevated (`~976` → `~846`, at the *same* cell, `(24,32,30)`) but slowly
  *decreasing* — that cell has extremely low density (`rho~4.6e-19` →
  `5.9e-19`, well below true ambient `~4.5e-17`) and a small but non-tiny
  pressure (`p~2.0e-13` → `1.9e-13`), giving a large but *shrinking* sound
  speed. Looked stable/improving on its own.
- **After step 14:** the picture changes completely, at a *different*
  nearby cell. Max wave speed jumps to `153140` at `(24,36,37)`
  (`u_x=5056`, `c=148084`, `rho=1.27e-16`, `p=1.67e-6`), and even further
  to `420422` (`|u_z|+c`) at `(21,31,37)` (`u_z=312125`(!), `c=108297`,
  `rho=1.02e-16`, `p=7.20e-7`). Both cells have density only modestly above
  true ambient (order `1e-16`, vs. `4.5e-17`) but pressure many orders of
  magnitude above ambient (`~1e-17`) — `p/rho` (∝ temperature²) has spiked
  to a genuinely extreme value at these specific cells, within a single
  step. `c = sqrt(gamma*p/rho)` directly confirms this arithmetic
  (`sqrt(1.667*7.2e-7/1.02e-16) ≈ 108297`, matching exactly).
- These cells sit near, but are *not* the same as, the original
  injection-region failure cell `(19,33,33)` — close enough to plausibly be
  just outside the injection sphere's edge (`num_injection_cells =
  64//32 = 2`), not coincident with it.

**Not yet determined:** *why* pressure (and, separately, velocity —
`u_z=312125` is itself wildly unphysical, ~3000x the wind's own terminal
velocity scale) spikes so specifically at these edge-of-injection cells
within one step, while density stays comparatively modest. Two live
candidates, neither tested yet: (a) the wind injection's own energy
deposition landing disproportionately in these edge cells for some
resolution/geometry reason unrelated to any of this session's
reconstruction changes; (b) an interaction between the new
VAN_ALBADA_PP positivity scaling (which independently floors/scales
density and pressure but does not explicitly preserve momentum or kinetic
energy consistency when it does so) and the injection or a hydro sweep,
inadvertently concentrating energy/momentum into a narrow region. Given
the density values found here (`~1e-16` to `~1e-19`) are all comfortably
above `params.minimum_density` (`~4.5e-20`), candidate (b) would have to
involve the *scaling* itself (not the final positivity floor) if it's
implicated — worth checking directly whether disabling just the strict
positivity floor (vs. the alpha/kappa/beta scaling) changes anything, as a
first, cheap test.

Scratch script from this round: `find_dt_collapse_cell.py`.

### 2026-09-19 (same day, continued a fifth time): back to fixing MINMOD's own ~15.47% failure directly, per explicit user redirect — seven systematic approaches tried, none breaks through; strong evidence the fix is not local

User clarified the actual target: fix MINMOD's own ~15.47%-of-T_END
boundary-corner failure (not the VAN_ALBADA_PP dt-collapse issue from the
previous round, which was set aside). `_cwb_setup.py` back to
`limiter=MINMOD`.

**Extended the (properly-derived, output-delta) VAN_ALBADA_PP positivity
scaling to also apply when `config.limiter == MINMOD`** (the underlying
alpha/kappa/beta math operates on the actual reconstructed delta, not on
which limiter produced the input gradient, so nothing in it is intrinsically
VAN_ALBADA_PP-specific) — both in `reconstruction.py`'s scaling block and
`evolve_state.py`'s matching post-sweep floor, gated on
`config.limiter == VAN_ALBADA_PP or config.limiter == MINMOD`. **Result:
bit-identical failure window to the unmodified baseline (15.4696%, exact
same bisection sequence).** This is a clean, important negative result:
MINMOD's own reconstruction *never* triggers this positivity check, even at
the actual failure point (consistent with the earlier finding that MINMOD's
gradients are already far more conservative than VAN_ALBADA's) — so
whatever breaks MINMOD's own run is not a "reconstructed value violates
positivity/kinetic-energy-budget" event at all. This code is left in place
(harmless no-op for MINMOD, confirmed) but is not the fix.

**Tested whether the MUSCL predictor half-step itself (the A_W-Jacobian
dt/2 time-correction) is the culprit**, independent of the spatial
gradient: monkeypatched `_reconstruct_at_interface_split` with a version
that skips the predictor entirely (`predictors = primitive_state`, keeping
the ordinary 2nd-order *spatial* extrapolation via MINMOD's own limited
gradients) — scratch `test_no_predictor.py`. **Result: slightly worse**
(14.65% vs. the 15.47% baseline) — removing the predictor step doesn't
help and in fact loses a little ground, ruling out "the predictor step
alone is the problem."

**Tested a location-based fix**: since every failing cell found across this
whole investigation has sat at or very near an open domain boundary, and
`first_order_fallback=True` (applied to literally every cell) fixes the
whole run, tried forcing first-order reconstruction only within
`THRESHOLD` grid cells of the nearest physical domain edge (any of the 3
axes; all are open boundaries here), MUSCL elsewhere — a per-cell blend via
`jnp.where` on a static distance-to-boundary mask, monkeypatching the whole
`_evolve_state_along_axis` (not just the reconstruction call, since
`first_order_fallback`'s branch lives in that function) — scratch
`test_boundary_first_order.py`. Swept three thresholds:

| threshold (cells) | failure point (% of T_END) |
|---|---|
| unmodified baseline | 15.4696% |
| 4 | 15.6133% |
| 10 | 15.3600% |
| 16 | 15.7030% |

All three cluster in a narrow band barely above the baseline, **with no
clear monotonic trend towards the full fix even as the forced-first-order
region grows substantially** (threshold=16 already covers a large fraction
of a 64³ domain's volume, especially near corners/edges where multiple
axes' buffer zones overlap) — threshold=10 is actually *worse* than
threshold=4. Only threshold → domain-covering (i.e. literal
`first_order_fallback=True`) actually works.

**Taken together, seven independent, systematically-tested approaches now
fail to fix MINMOD's own boundary-corner failure**: fixed global floor,
independent per-variable neighbor-range clip, single shared-alpha scaling,
loose magnitude-based clamp, the properly-derived VAN_ALBADA_PP
alpha/kappa/beta scaling (confirmed inert for MINMOD), removing the MUSCL
predictor step, and location-based first-order at three different
buffer widths. This is now strong evidence that **the fix is not local** —
not a per-cell value check (any of the positivity/magnitude variants), not
a per-cell location check (the boundary-buffer variants), and not a single
structural component of the reconstruction (predictor-step removal alone).
Whatever makes blanket `first_order_fallback=True` work must depend on
every cell in the domain being first-order simultaneously — e.g. a
genuinely global change in the scheme's overall numerical dissipation
altering how the flow develops well before it ever reaches the boundary,
not a targeted fix applicable near the danger zone alone. This reframes the
problem: further local/targeted reconstruction fixes are unlikely to
succeed without addressing this global-dissipation characteristic somehow
(e.g. a genuinely more diffusive limiter tuned specifically for this
regime, artificial viscosity, or revisiting whether this setup's physical
parameters are just fundamentally too stiff for MUSCL at N=64, per item
4's very first "remaining plausible directions" list from 2026-08-26).

**Current code state:** the MINMOD-extended VAN_ALBADA_PP positivity
scaling (confirmed harmless no-op for MINMOD) is left in `reconstruction.py`/
`evolve_state.py`; none of the predictor-removal or boundary-buffer
experiments were kept (both were scratch-only monkeypatches, never applied
to the real files). `_cwb_setup.py` is back to `limiter=MINMOD`.

Scratch scripts from this round: `test_no_predictor.py`,
`test_boundary_first_order.py` (swept via `THRESHOLD` at the top of the
file).

---

## 2026-09-19, round 6: config sweep (split / time_integrator / limiter /
## riemann_solver) — is there a more promising starting point than
## SPLIT+MUSCL+MINMOD+HLLC?

Per user request, before sinking more effort into local fixes for MINMOD's
own ~15.47%-of-`T_END` failure (round 5 above), swept the N=64 bisection
harness (`sweep_config.py`, a general env-var-configurable version of the
existing bisection driver, reusing one `jax.jit`-compiled
`_time_integration` wrapper per combo) across 8 combinations. All numbers
are "clean up to X% of `T_END`" from the same bisection method used
throughout this investigation (18 bisection iterations, `T_END=Period`,
`BOX_SIZE` unchanged, `first_order_fallback=False`):

| split | time_integrator | limiter | riemann_solver | clean up to (% of T_END) | note |
|---|---|---|---|---|---|
| SPLIT | MUSCL | MINMOD | HLLC | **15.4696%** | current baseline |
| SPLIT | MUSCL | MINMOD | HLL | 15.3374% | ~same, slightly worse |
| SPLIT | MUSCL | MINMOD | HYBRID_HLLC | 15.3610% | ~same |
| SPLIT | MUSCL | MINMOD | **AM_HLLC** | **16.2731%** | **best result — +0.80pp, ~5% relative** |
| UNSPLIT | RK2_SSP | MINMOD | HLLC | 15.2519% | ~same as SPLIT baseline |
| UNSPLIT | RK2_SSP | MINMOD | AM_HLLC | 0.0156% | fails almost immediately — AM_HLLC does NOT help under UNSPLIT |
| UNSPLIT | RK2_SSP | VAN_ALBADA_PP | HLLC | ~0.0000% | fails immediately |
| SPLIT | MUSCL | VAN_ALBADA_PP | AM_HLLC | 0.0950% | the known separate VAN_ALBADA_PP dt-collapse bug (round-4 thread), unrelated to AM_HLLC |

**Findings:**
- Riemann solver choice barely matters for MINMOD (HLL/HLLC/HYBRID_HLLC all
  within ~0.15pp of each other) — confirms the failure is a reconstruction/
  positivity issue, not a flux-function issue, for the ordinary solvers.
- `AM_HLLC` (the adaptive/compression-aware hybrid HLLC-LM/HLLC blend) is
  the *only* combination that moves the needle at all, and only in the
  SPLIT+MUSCL+MINMOD context — 15.47% → 16.27%. This is a genuine, if
  modest, improvement (not within bisection noise — every other tested
  variant clusters in a 15.25%-15.47% band).
- That improvement does **not** transfer: AM_HLLC combined with UNSPLIT
  fails almost immediately (0.016%), and combined with VAN_ALBADA_PP
  re-triggers the separate, already-documented dt-collapse bug near
  injection (round 4) rather than helping. So AM_HLLC's benefit is
  specifically synergistic with SPLIT+MUSCL+MINMOD, not a generic
  robustness win — swapping it in elsewhere makes things worse.
- UNSPLIT+MINMOD+HLLC (the codebase's "native" positivity-preserving path)
  is not meaningfully different from SPLIT+MUSCL+MINMOD+HLLC (15.25% vs
  15.47%) — switching split/integrator alone doesn't help either.
- No config change found so far gets remotely close to
  `first_order_fallback=True`'s 100%. AM_HLLC's +0.8pp is real but small
  next to the ~84.5 percentage points still missing.

**Conclusion:** none of the 8 tested configs comes close to replacing the
"the fix must be global, not local" conclusion from round 5 — AM_HLLC is
worth keeping as a modest, free improvement (swap `riemann_solver=HLLC` →
`AM_HLLC` in `_cwb_setup.py`, no other changes needed, since it strictly
improves on the current baseline with no observed downside in the
SPLIT+MUSCL+MINMOD context), but it is not a fix for item 4 and doesn't
change the round-5 conclusion that a targeted/local intervention is
unlikely to fully solve this.

Scratch script: `sweep_config.py` (env vars `CWB_SPLIT`, `CWB_TI`,
`CWB_LIMITER`, `CWB_RS`, `CWB_GPU_PIN`).

---

## 2026-09-19, round 7: does the code-unit regime or the physical parameters
## (Mdot, v, a, m1/m2) influence when MINMOD's failure happens?

Per user request, kept split=SPLIT/time_integrator=MUSCL/limiter=MINMOD/
riemann_solver=HLLC fixed (the round-6 conclusion) and instead swept the
*inputs*: the code-unit length/mass scale (`CodeUnits(code_length,
code_mass, code_velocity)`), and the physical wind/orbital parameters. New
general driver: `sweep_units_physics.py` (env-var configurable, same
bisection method as `sweep_config.py`). Sanity-checked first: default env
vars reproduce `_cwb_setup.py`'s exact `T_END`/`SEPARATION` and land at
15.4694% (baseline is 15.4696% — matches to float precision).

**Code-unit mass scale (`CODE_MASS_MSUN`, `code_length` held at baseline
2 au) — exactly, provably inert:**

| CODE_MASS_MSUN | clean up to |
|---|---|
| 0.01 | 15.4694% |
| 0.1 | 15.4694% |
| 1 (baseline) | 15.4696% |
| 10 | 15.4694% |
| 100 | 15.4694% |

Bit-identical across 4 orders of magnitude. Makes sense dimensionally: at
fixed `code_length`, rescaling `code_mass` (with `code_velocity =
sqrt(G*code_mass/code_length)`, i.e. G=1 preserved) is a pure self-similar
rescaling of time/velocity/density/pressure together — every dimensionless
ratio the scheme actually sees (Mach numbers, density contrasts, CFL
fraction) is exactly unchanged. **Choice of mass code-unit is a non-issue.**

**Code-unit length scale (`CODE_LENGTH_AU`, `code_mass` held at baseline
1 Msun, `SEPARATION_CODE` held at 0.4 -- i.e. the dimensionless box geometry
is bit-for-bit identical to baseline) — a strong, clean, monotonic trend:**

| CODE_LENGTH_AU | clean up to |
|---|---|
| 0.25 | 17.5446% |
| 0.5 | 16.6245% |
| 1.0 | 15.7658% |
| 2.0 (baseline) | 15.4696% |
| 4.0 | 14.9464% |
| 8.0 | 14.5920% |
| 16.0 | 14.3215% |

**Important caveat, not a clean "pure representation" result like mass:**
unlike mass, length is *not* separable from physics in this codebase's
`CodeUnits` design, because `BOX_SIZE=1.0` and `SEPARATION=0.4` are both
fixed in *code*-unit (dimensionless) terms while `Mdot`/`v_inf`/`RHO_0`/
`P_0`/`M1`/`M2` are fixed in *physical* terms. So sweeping `CODE_LENGTH_AU`
necessarily and simultaneously rescales the real physical domain size
(∝ code_length) and the real physical binary separation (= 0.4 ×
code_length_au, from 0.1 au at the low end to 6.4 au at the high end) —
it is not possible to vary "what the code-length unit represents" in
isolation here. So this result is really a domain-size/separation-scale
physics effect in disguise, not evidence that raw numerical representation
matters — it overlaps with (and is corroborated by) the `a` sweep below.

**Physical parameters (code units held at baseline throughout):**

| knob | value | clean up to |
|---|---|---|
| Mdot (both stars, ratio fixed) | 0.1x | 15.3198% |
| Mdot (both stars, ratio fixed) | 1x (baseline) | 15.4696% |
| Mdot (both stars, ratio fixed) | 10x | 15.3236% |
| v_inf (both stars, ratio fixed) | 0.5x | 16.6286% |
| v_inf (both stars, ratio fixed) | 1x (baseline) | 15.4696% |
| v_inf (both stars, ratio fixed) | 2x | 14.5390% |
| a (`SEPARATION_CODE`, domain/mass fixed) | 0.2 | 15.7207% |
| a (`SEPARATION_CODE`, domain/mass fixed) | 0.4 (baseline) | 15.4696% |
| a (`SEPARATION_CODE`, domain/mass fixed) | 0.6 | 15.5582% |
| m1/m2 | 45/45 (equal) | 16.3010% |
| m1/m2 | 50/40 (baseline) | 15.4696% |
| m1/m2 | 70/20 (extreme) | 12.7266% |

**Findings:**
- `Mdot` and pure geometric `a` (decoupled from domain size) both have only
  small, non-monotonic effects (~0.15-0.25 percentage points) — close to
  noise.
- `v_inf` has a real, monotonic effect (~2.1pp range): faster wind is more
  failure-prone, slower wind more forgiving — consistent with a
  higher-Mach, steeper-gradient wind being harder for MINMOD's positivity
  behavior at the boundary.
- Stellar mass ratio has the largest physical effect found (~3.6pp range):
  equal masses are more forgiving (16.30%), a very unequal ratio is
  noticeably worse (12.73%) — plausibly because unequal masses put the
  contact discontinuity/collision region off-center relative to the domain,
  changing which wind's free-streaming material reaches the open boundary
  first/hardest.
- The length-scale/domain-size effect (14.3%-17.5%) is the largest overall,
  but per the caveat above it's a physical-scale effect, not a pure-unit
  effect.

**Conclusion:** the failure point is somewhat sensitive to the physical
setup (mainly wind speed, mass ratio, and domain/separation scale), with a
combined spread of roughly 12.7%-17.5% across everything tested here versus
the 15.47% baseline — real, but modest next to `first_order_fallback=True`'s
100%. No tested combination gets remotely close to a full fix. This is
consistent with (not a replacement for) the round-5/round-6 conclusion:
the remaining gap is a structural/global scheme property, not something
resolvable by retuning the physical scenario or the code-unit choice.
Mass-unit choice specifically is now *confirmed* fully inert, which at
least rules out "sensitivity to code-unit mass scale" as a contributing
factor.

Scratch script: `sweep_units_physics.py` (env vars `CODE_LENGTH_AU`,
`CODE_MASS_MSUN`, `SEPARATION_CODE`, `M1_MSUN`, `M2_MSUN`,
`MDOT{1,2}_MSUN_YR`, `VINF{1,2}_KMS`, `CWB_GPU_PIN`, `TAG`).

---

## 2026-09-19, round 8: does the stationary (nbody=False) binary avoid the
## failure entirely? **YES — this is the most significant result of the
## whole item-4 investigation.**

Noticed while researching round 7 that `_cwb_setup.py`'s own module
docstring and an in-function comment both describe a *stationary* binary
("`config.nbody_config` is left at its default (`NBodyConfig(nbody=False)`)
... the sources do not orbit") — but `run_cwb()`'s actual
`SimulationConfig` sets `nbody_config=NBodyConfig(nbody=True,
deposit_particles=NGP, central_object_only=False)`. Confirmed via git
history (`bab53b4`, "tune CWB setup: eccentric binary params, N-body orbit,
open boundaries") that this is genuinely stale documentation, not a
reconciling code path: the docstring/comment predate that commit and were
simply never updated when the setup was switched from a hand-placed
mirror-symmetric stationary pair to an actual RK4-integrated, mass-
asymmetric (M1=50, M2=40 Msun) Kepler orbit. Dispatched an agent to confirm
the exact mechanism in `_wind_source_params`
(`astronomix/_modules/_stellar_wind/stellar_wind.py:458-506`): with
`nbody=True`, wind-source positions come from
`params.nbody_params.nbody_state` every step (live orbit); with
`nbody=False`, they come from `wind_params.wind_injection_positions`
directly (defaults to a single source at the box center if left unset —
silently wrong for two stars, so it must be set explicitly, e.g. to
`STAR_POSITIONS`).

Built a `nbody=False` variant (same `sweep_units_physics.py`, new
`NBODY_ENABLED` env var: when false, uses `NBodyConfig()` default and
passes `wind_injection_positions=STAR_POSITIONS` explicitly) with
*everything else* bit-for-bit identical to the current baseline (same
SPLIT+MUSCL+MINMOD+HLLC, same M1/M2/Mdot/v_inf/RHO_0/P_0/a/code units,
same `T_END=Period`, same N=64, `first_order_fallback=False`):

```
RESULT [nbody_false]: FULLY CLEAN THROUGH T_END (100%)
```

**Confirmed this is a real, physically meaningful run, not a degenerate
no-op** — diagnostics at `t=T_END`: `max_mach=17.34` (genuinely
supersonic/shocked), `max_density=2.95e-7` (~6.5 million times
`RHO_AMBIENT=4.51e-17` — strongly compressed shocked gas), `min_density=
3.72e-11` (~8×10^5 times ambient — the bubble has swept essentially the
whole domain by `T_END`, consistent with earlier rounds' finding that the
open-boundary corner is where the wind front reaches first), `any_nan=
False`.

**This is the cleanest, most significant result in the whole item-4
investigation.** Every other variable tested (limiter, split/integrator,
riemann_solver, code-unit scale, Mdot, v_inf, a, m1/m2 — rounds 5-7) was
held or varied with `nbody=True` throughout and never got close to 100%;
flipping *only* this one switch (with the exact same asymmetric M1/M2 and
unequal wind parameters that round 7 showed only partially help when
*more* symmetric) goes from 15.47% to a full, clean 100%. This strongly
localizes the failure's trigger to the orbital motion / N-body coupling
itself, not to the reconstruction/limiter math the seven round-5 attempts
targeted -- those were fixing symptoms downstream of a disturbance that
orbital motion introduces, not the disturbance's source.

**Not yet root-caused *why* orbital motion triggers the failure** (out of
scope for this round, flagged for next session) — candidate mechanisms,
untested:
1. The moving injection region continuously sweeps across grid-cell
   boundaries as the stars orbit (a continuous position, not grid-snapped,
   since `_wind_source_params` reads `nbody_state` directly — no NGP
   discretization is involved in *wind* injection specifically; NGP
   deposition is a separate, gravity-only mechanism gated by
   `self_gravity`, which is not enabled in this setup). Even without
   grid-snapping, a continuously moving source changes which cells are
   "recently freshly injected" vs "relaxed" every step, unlike a fixed
   source where the near-source cells reach a quasi-steady state.
2. Orbital velocity is itself a nontrivial fraction of the terminal wind
   velocity for this binary (M1/M2 ~ 50/40 Msun at a=0.4 code units), so
   the swept-back/asymmetric bow-shock geometry from orbital motion could
   be a genuinely harder numerical case than a stationary symmetric-ish
   bubble pair, independent of any reconstruction-scheme detail.
3. Possibly interacts with the specific open-boundary-corner failure mode
   from earlier rounds (a moving source could change exactly when/where
   the wind front first reaches an open boundary, and from what angle).

**Practical implication:** if a stationary binary is an acceptable
approximation for what this test is meant to validate (the docstring
suggests it originally was the intended design), switching `_cwb_setup.py`
back to `nbody=False` (+ explicit `wind_injection_positions=STAR_POSITIONS`)
is a complete, clean fix for item 4 as currently scoped — no
`first_order_fallback`, no reconstruction-level changes needed at all. If
the orbiting binary is a hard requirement (testing N-body-coupled wind
injection specifically), item 4 remains open, but now with a much sharper
question: what specifically about source motion breaks positivity, rather
than "MINMOD's boundary corner is unfixable" in general.

Scratch script: `sweep_units_physics.py`, extended with `NBODY_ENABLED` and
a `PRINT_DIAGNOSTICS` block.

---

## 2026-09-19, round 9: run the stationary setup for real (T_END=Period/8,
## N=64 and N=256) and assess Mach-number physical correctness

Per user request, ran the actual correctness pipeline (`find_shocks_pfrommer`
+ the real `plot_shocked_cells_2d`/`_3d` plotting code from
`cwb_shock_finder.py`, not just the bare bisection harness) with the
stationary binary (`nbody=False`, `wind_injection_positions=STAR_POSITIONS`
explicit) at `T_END = Period/8` (not the full period this time), current
SPLIT+MUSCL+MINMOD+HLLC, for both N=64 and N=256. Scratch script:
`run_cwb_stationary.py`. Both runs completed **fully clean** (zero NaN,
zero negative density/pressure). Figures saved to `figures/`:
`cwb_shocked_cells_{2d,3d}_stat_{64,256}.png` (new "_stat" suffix, does not
touch the existing non-suffixed baseline PNGs from the orbiting setup).

**Shock geometry (visual check of the saved figures):** at both
resolutions, the shock-finder surface is a single paraboloid/bow-shock-like
sheet with its vertex near the *weaker* wind source (star 2: Mdot=6.5e-8
Msun/yr, v=1850 km/s) and opening away from the *stronger* one (star 1:
Mdot=6.5e-7 Msun/yr, v=2200 km/s) — i.e. **not** mirror-symmetric about
x=0. This is physically correct, not a bug: `_cwb_setup.py`'s current wind
parameters are asymmetric (momentum flux `Mdot*v_inf` is ~12x larger for
star 1), so by momentum balance the contact discontinuity/wind-collision
shock is pushed far past star 2 and wraps around it — a real, well-known
CWB phenomenon (e.g. WR140-like systems), not the mirror-symmetric
equal-wind picture the module docstrings describe (those describe a
different, not-currently-used equal-wind configuration). N=256 shows the
same structure with a visibly thinner, better-resolved collision layer.

**Mach numbers — capped far below the ~100 physically expected for O-star
winds, but this is the pre-existing, already-diagnosed item-1b bug, not a
new problem:**

| N | n_shock_cells | Mach min | Mach max | mean | median | p90 | p99 | max\|x\| of surface (box half-width=0.5) |
|---|---|---|---|---|---|---|---|---|
| 64 | 6315 | 1.51 | 10.31 | 3.55 | 3.22 | 5.44 | 8.72 | 0.477 |
| 256 | 99955 | 1.30 | 10.82 | 3.96 | 3.49 | 6.27 | 9.79 | 0.494 |

Real O-star winds are highly supersonic against the ambient sound speed
(Mach ~100, per both `_cwb_setup.py`'s and `cwb_shock_finder.py`'s own
docstrings) — the measured cap here (~10-11) is roughly an order of
magnitude too low for that. **This is not new**: it's exactly the
originally-diagnosed item 1b bug (`shock_zones` narrower than the true
numerically-smeared transition — see that section above), which was
explicitly "blocked behind item 4" and "can't be usefully re-tested
against the real CWB profile until item 4 is resolved" until now. Item
1b's own synthetic reproduction (a true Mach-100 shock smeared over `w`
cells) found Mach≈78 (w=1), ≈33 (w=2), **≈8.5 (w=3)** — matching what we
see here (~10-11) almost exactly, strongly suggesting the wind-collision
shock is smeared over roughly 3 cells at both resolutions and the
shock-finder's immediate-neighbor, zone-bounded RH sampling (not the
underlying flow) is what's capping the reported Mach number.

**Supporting evidence it's the sampling methodology, not physical/grid
smearing, that dominates:** going from N=64 to N=256 (4x finer per axis)
barely moved the cap (10.31 -> 10.82), despite the collision layer visibly
thinning in the density plot. If under-resolved *physical* smearing were
the main limiter, a 4x resolution increase should have shrunk the shock's
width-in-cells and raised the measured Mach much more than this. This is
consistent with item 1b's diagnosis that the *sampling walk stopping at
the detection-zone edge* (not just cell count across the shock) is the
bottleneck, and is now confirmed against a real (not synthetic) CWB
profile for the first time.

**Other observation, not necessarily a defect:** even at `T_END=Period/8`
(1/8 of the full orbital period), the bow shock's surface cells already
reach `|x|=0.494` at N=256 — 98.8% of the way to the open boundary at
`|x|=0.5`. The domain-containment margin used by the original `T_END`
design note ("bow shocks stay comfortably inside the domain") is nearly
gone even at this much-reduced end time; a longer run at this same domain/
wind-parameter combination would very likely have the bow shock reach the
boundary well before a full period (consistent with round 8's `nbody=True`
failure and round-8's diagnostic that the stationary full-period run
ends up with the whole domain swept by shocked material).

**Conclusion:** the shock finder is working correctly on a real,
physically sensible (if momentum-asymmetric) colliding-wind bow shock; the
low reported Mach numbers are the known, pre-existing item-1b sampling
bug, not evidence of anything wrong with the stationary setup, the fix
from round 8, or the underlying flow. Item 1b's fix direction (decouple
the Mach-sampling walk from `shock_zones` membership, walk until the field
itself plateaus) is now testable against this real profile whenever
picked up.

---

## 2026-09-19, round 10: item 1b Tier 1 implemented — a real, non-trivial
## improvement, but exposes a *new* limitation a fixed sampling distance
## can't get past

**Implemented** (real repo files, not scratch): a `sampling_steps` /
`mach_sampling_steps` parameter threaded through `_shock_mach.py`
(`_calculate_mach_at_surface`), `_energy_dissipation.py`
(`calculate_thermal_energy_flux`, which already had this parameter but
wasn't wired up) and `pfrommer_shock_finder.py` (`find_shocks_pfrommer`,
new `mach_sampling_steps` kwarg, **default 1** -- strictly backward
compatible, so `cr_grey_injection.py` and every other existing caller is
unaffected unless it opts in). Both Mach and thermal-flux sampling always
use the *same* step count now (previously could silently mismatch -- Mach
from one offset, pre-shock state for the energy flux from another).
Generalized `_shock_zones._make_interior_mask` to take a `margin` param
(default 1, unchanged for its one pre-existing caller, criterion 3) and
reused it in both `_shock_mach.py` (new) and `_energy_dissipation.py`
(refactored from duplicated inline logic) as the boundary-safety guard:
cells within `sampling_steps` of a domain edge are excluded from the
reported Mach/flux, since `get_post_pre_shock_values` samples via
`jnp.roll`, which wraps at the edge -- this guard was present for thermal
flux already but **entirely missing for Mach before this fix**. Shock-zone
*detection* (criterion 3) is untouched, exactly as item 1b's original fix
direction specified.

**Regression check:** re-ran `sedov_shock_finder.py`'s full assertion
suite (N=256, default `mach_sampling_steps=1`) -- passes identically to
before (`mach in [2.87, 24.57]`, same as pre-change), confirming the
refactor is a true no-op at the default. This is the validation step item
1a skipped, which is exactly what let its bug reach a real simulation
undetected.

**Investigation on the real stationary N=64 CWB run (T_END=Period/8, same
state as round 9, shock-finder re-run post-hoc per step count so the
expensive time integration only ran once):**

| steps | n_valid (margin-excluded) | Mach min | max | mean | median | p90 | p99 | argmax location (x,y,z) |
|---|---|---|---|---|---|---|---|---|
| 1 (old default) | 6315 (0) | 1.51 | 10.31 | 3.55 | 3.22 | 5.44 | 8.72 | (+0.070, +0.023, +0.023) |
| 2 | 5814 (501) | 1.69 | 8.45 | 5.05 | 5.04 | 6.15 | 7.54 | (+0.070, +0.023, +0.023) |
| 3 | 5440 (875) | 1.00 | 8.74 | 5.35 | 5.11 | 7.20 | 8.22 | (+0.242, -0.211, +0.039) |
| 4 | 5077 (1238) | 1.00 | 10.51 | 5.28 | 4.60 | 8.60 | 9.76 | (+0.242, -0.227, +0.023) |
| 5 | 4728 (1587) | 1.00 | 11.72 | 5.12 | 4.07 | 9.87 | 11.13 | (+0.258, -0.242, +0.023) |
| 6 | 4386 (1929) | 1.00 | 13.06 | 4.92 | 3.59 | 11.22 | 12.58 | (+0.289, -0.258, +0.055) |
| 8 | 3751 (2564) | 1.00 | 16.32 | 4.46 | 2.75 | 14.35 | 15.82 | (+0.320, -0.273, -0.070) |
| 10 | 3155 (3160) | 1.00 | 19.79 | 3.43 | 2.04 | 11.32 | 18.88 | (+0.336, -0.273, -0.086) |

(First pass at this table naively reported "n_shock_cells" and Mach
min/mean/median over *all* originally-detected surface cells including
ones the new boundary-margin guard had just zeroed out -- that's wrong,
it dilutes the stats with excluded cells reading as Mach=0. The table
above filters to `surface & (mach > 0)`, the actually-trusted cells.)

**Real improvement at small step counts:** steps=2 already lifts the
median from 3.22 to 5.04 -- a genuine, substantial recovery of previously
undersampled shock strength, with the argmax still at the *same physical
location* as steps=1 (the wind-collision shock near the orbital plane,
close to star 2). This matches the prediction and is exactly what Tier 1
was designed to do.

**New problem found, not previously anticipated: at steps>=3 the argmax
location jumps to a different, drifting position, and keeps drifting
further as steps grows** (+0.242,-0.211 at steps=3 -> +0.336,-0.273 at
steps=10 -- monotonically away from the orbital plane, off the x-axis
entirely). This is *not* the same shock converging toward its true Mach as
sampling reaches further into the plateau -- it's the walk picking up a
*different* part of the 3D bow-shock surface. Root cause: `get_post_pre_
shock_values` walks along the single dominant *grid axis* of the local
shock-direction vector, not the true (continuous) shock-normal direction.
For the near-orbital-plane shock (close to axis-aligned) this is fine even
several cells out; for the curved parts of the paraboloid bow-shock
surface further from the orbital plane, a grid-axis walk increasingly
diverges from the true local normal the further it goes, eventually
sampling material that isn't actually pre-/post-shock relative to that
point on the curved surface at all. This explains the Mach max growing
seemingly without bound (10.3 -> 19.8) rather than plateauing -- it's
cross-contamination from a mismatched, curved region, not genuine
recovery of a higher true Mach. The median/mean also turn over and
decline again past steps=3 (5.11 -> 2.04 by steps=10): the boundary-margin
guard is increasingly excluding the *good*, high-Mach cells near the
domain edge (round 9 found surface cells reach `|x|=0.477`-`0.494` out of
the `0.5` box half-width -- i.e. many of the strongest, most informative
cells sit close to the boundary and get progressively margined out as
`sampling_steps` grows) faster than any genuine signal is being added.

**Conclusion:** Tier 1 works and is worth keeping (a real, validated,
backward-compatible improvement over the old always-1-cell default), but
a single fixed step count can't be pushed very far before trading real
signal for two compounding problems: (a) grid-axis-vs-true-normal
divergence contaminating the result with an unrelated part of the shock
surface, and (b) the boundary-margin guard removing an increasing share of
the domain's most informative (near-edge) cells. A small, conservative
value (steps=2, maybe 3) looks safe and genuinely helpful; anything larger
is actively counterproductive on this geometry, not just "diminishing
returns." **This is new, concrete evidence for going further than
originally scoped Tier 1**: the earlier Tier-2 proposal (adaptive
walk-until-plateau) would need to also walk along the true local
shock-normal direction, not just add a smarter stopping rule, to avoid
this specific failure mode -- a refinement not anticipated when Tier 2 was
first sketched (round 9's response to "think about a way to fix it").

**Not yet done:** deciding on / applying a new default `mach_sampling_steps`
in `_cwb_setup.py`/`cwb_shock_finder.py`'s actual `find_shocks_pfrommer`
call (still `mach_min=MACH_MIN` only, no `mach_sampling_steps` override --
this round's investigation used a separate scratch script,
`investigate_mach_sampling.py`, deliberately kept out of the real pipeline
pending a decision on the axis-direction issue above). No change made to
`cr_grey_injection.py` or any other consumer.

Scratch scripts: `regression_check_sedov.py` (Sedov correctness regression
check), `investigate_mach_sampling.py` (the sweep above).

---

## 2026-09-19, round 11: Tier 2 (adaptive walk-until-plateau, along the
## true shock-normal, not a grid axis) tried against the real N=64 CWB
## stationary run — solves both problems round 10 found

**Prototype only, scratch, not yet in the real package** (this round was
"try it and report the numbers," not "implement" — unlike Tier 1):
`adaptive_shock_sampling.py`. Design, directly informed by round 10's
diagnosis:
- Walks along the **continuous local shock-normal direction**
  (`shock_direction`, already a unit vector) via trilinear interpolation
  (`jax.scipy.ndimage.map_coordinates`, `order=1`), not the single
  dominant *grid axis* `get_post_pre_shock_values` uses — this directly
  targets round 10's finding that a grid-axis walk increasingly diverges
  from the true normal on curved parts of the 3D shock surface.
- `mode="nearest"` clamps any out-of-domain sample to the boundary value
  instead of wrapping (`jnp.roll`'s behavior) — structurally eliminates
  the wraparound risk suspected (never confirmed) to be item 1a's real-
  simulation NaN cause, rather than patching around it with a margin mask.
- Adaptive per-cell stopping: walks independently on the post/pre sides,
  advancing until `|Δlog(field)| < tol` (plateaued) **or** the local
  log-gradient's sign reverses (crossed into a different feature — the
  safeguard aimed at the "compound shock" risk noted in this file's
  module docstrings), capped at `max_steps=15`. Vectorized as a bounded
  Python-unrolled loop over `max_steps` with a per-cell "still active"
  mask, same style as the existing `shift_field` walk.

**Validated against the exact same converged, clean N=64 stationary CWB
state used in rounds 9-10** (Phase 1-3 -- shock direction/zones/surface --
reused unchanged from `find_shocks_pfrommer`'s default path; only Phase 4
Mach replaced):

| method | n_valid | Mach min | max | mean | median | p90 | p99 | argmax location |
|---|---|---|---|---|---|---|---|---|
| Tier 1, steps=1 (old default) | 6315/6315 | 1.51 | 10.31 | 3.55 | 3.22 | 5.44 | 8.72 | (+0.070,+0.023,+0.023) |
| Tier 1, steps=2 (round 10's best) | 5814/6315 | 1.69 | 8.45 | 5.05 | 5.04 | 6.15 | 7.54 | (+0.070,+0.023,+0.023) |
| Tier 2, tol=0.10 | 6315/6315 | 2.34 | 10.28 | 5.61 | 5.59 | 6.64 | 8.48 | (+0.070,+0.023,+0.023) |
| Tier 2, tol=0.05 | 6315/6315 | 2.34 | 10.28 | 6.01 | 5.93 | 7.50 | 8.81 | (+0.070,+0.023,+0.023) |
| Tier 2, tol=0.03 | 6315/6315 | 2.34 | 10.28 | 6.15 | 5.98 | 7.87 | 8.91 | (+0.070,+0.023,+0.023) |
| Tier 2, tol=0.01 | 6315/6315 | 2.34 | 10.28 | 6.22 | 6.01 | 8.11 | 9.06 | (+0.070,+0.023,+0.023) |

**Both problems round 10 found are resolved:**
1. **No drift.** The argmax cell is identical across every Tier-2 setting
   tested and matches Tier 1's own steps=1/2 argmax exactly -- unlike
   Tier 1 at steps>=3, which jumped to a different, progressively-drifting
   location. Walking along the true normal keeps the sample anchored to
   the correct physical shock, confirming the round-10 diagnosis (grid-
   axis divergence, not "more sampling reveals more contamination") was
   the right explanation.
2. **No cell loss.** `n_valid=6315/6315` at every tolerance -- `mode=
   "nearest"`'s boundary clamping means no cell is ever excluded for
   being close to the domain edge, unlike Tier 1's margin guard, which
   increasingly threw away the domain's most informative (near-boundary)
   cells as `sampling_steps` grew.
3. **Converges cleanly, doesn't run away.** Tightening `tol` from 0.10 to
   0.01 (a 10x range) moves the mean from 5.61 to 6.22 with clearly
   diminishing increments (+0.40, +0.14, +0.07) -- converging toward an
   asymptote near ~6.3-6.4, not growing without bound the way Tier 1's
   max did (10.3 -> 19.8 from steps=1 to steps=10). The max itself is
   *flat* at ~10.28 across all four tolerances -- that cell's plateau is
   reached almost immediately and doesn't move further, a strong
   consistency signal.
4. Clean throughout (no NaN, no exceptions) across all four tolerances.

**A real, substantial improvement in typical Mach, still well short of
the ~100 physically expected for O-star winds:** median goes from 3.22
(original bug) to ~6.0 (Tier 2, tol<=0.03) -- roughly +86% -- but the
*maximum* detected Mach stays near ~10.3, essentially unchanged from the
original bug's own max (10.31) by coincidence. **Open question, not yet
investigated:** is ~10.3 genuinely the correct plateau value for this
specific cell (plausible -- it sits in the wind-collision region between
the two stars, i.e. possibly a compound/secondary shock into already-
once-shocked material rather than the pristine-ambient bow shock the
"~Mach 100" expectation describes), or is the *true* strongest shock (the
outer bow shock into cold ambient gas) a different, thinner feature that
isn't well-represented in the detected surface-cell population at all (a
zone-*detection* completeness question, separate from the Mach-sampling
question these three rounds have focused on)? Distinguishing these needs
looking at where in the domain the various Mach percentiles actually sit
(e.g. a 2D/3D plot colored by Mach, analogous to round 9's figures, but
using Tier-2 Mach) and/or checking the pre-shock temperature at the
argmax cell against the true ambient value.

**Not yet done / caveats:**
- This is a **prototype, not wired into the real package** -- unlike Tier
  1, nothing in `astronomix/shock_finder3D/` was changed this round.
  `adaptive_shock_sampling.py` and `run_tier2_cwb.py` are scratch-only.
- Not yet jitted (ran in eager mode for this investigation) -- should be
  trivially `@jax.jit`-compatible (pure `jnp` ops, a `max_steps`-bounded
  Python-unrolled loop, no data-dependent control flow) but this hasn't
  been confirmed.
- Not yet validated against a real *evolving* simulation loop (only used
  as post-hoc analysis on a converged final state here) -- item 1a's
  lesson is that synthetic/single-snapshot correctness isn't sufficient;
  before this could be trusted as a real fix it should be re-checked
  against the Sedov-Taylor test (the one 1a broke) the same way Tier 1
  was in round 10, and ideally against a case with genuinely nearby
  compound shocks to stress-test the gradient-reversal safeguard.
- Thermal-energy-flux hasn't been ported to the adaptive approach yet
  (only Mach was, for this investigation); would need the same treatment
  for `calculate_thermal_energy_flux` before this could fully replace
  Tier 1 in `pfrommer_shock_finder.py`.

Scratch scripts: `adaptive_shock_sampling.py` (the new sampling function),
`run_tier2_cwb.py` (this round's investigation, reuses the same T_END=
Period/8 N=64 stationary setup as rounds 9-10).

---

## 2026-09-21, round 12: independent theoretical Mach-number calculation — confirms the gap is real and physical, not a shock-finder artifact; identifies the likely root cause and a proposed (not yet implemented) fix

Round 11 left an open question: is Tier 2's plateau of median~6.0/max~10.3
close to the real answer for this specific shock, or is the true expectation
much higher? Wrote `expected_cwb_mach.py` (new, committed, in this folder —
deliberately does not import `astronomix`/JAX, so it runs instantly with no
GPU dependency) to answer this independently of the shock finder entirely,
using standard analytic colliding-wind-binary theory (Stevens, Blondin &
Pollock 1992 / Eichler & Usov 1993 wind-momentum-balance stagnation point)
applied to `_cwb_setup.py`'s actual physical parameters. Two independent
estimates, both as functions of the winds' physical/orbital parameters:

1. **Ambient-comparison Mach** (`v_wind / c_s(ambient)`) — exactly what
   `_cwb_setup.py`'s own module docstring uses to justify "Mach number ~
   100". Needs only verified setup inputs (wind terminal velocity, ambient
   temperature). At baseline: **~153 (star 1), ~129 (star 2)**.
2. **Adiabatic-cooled-wind Mach** — the wind that actually crosses the
   wind-wind collision shock is not ambient gas, it's each star's own
   freely-expanding wind, which cools substantially (`T ~ r**(-2(gamma-1))`
   for constant-velocity adiabatic free expansion) between the stellar
   surface and the stagnation point. Needs two extra reference values the
   codebase doesn't specify (assumed: `R_star=20 Rsun`, `T_wind_base=3.5e4
   K`, typical O-star literature values, flagged as illustrative). At
   baseline (using the separation *actually simulated*, see below): **~355
   (star 1), ~131 (star 2)** — i.e. even this more careful estimate lands in
   the same ~100-350 range as the simple ambient comparison for this
   specific geometry.

**Both independent estimates agree the sim is far short**: Tier 2's real
max (10.3) is **~13x below even the conservative ambient-comparison floor**
(~129, for star 2 — the wind whose termination shock is the one actually
being measured, per round 9/10's argmax location matching the star-2
stagnation point). This rules out "10.3 is just genuinely the right answer
for this compound/secondary shock" (round 11's open question) — no
physically-motivated estimate gets anywhere near that low. The gap is real.

**Independent finding, not the root cause but worth fixing regardless:**
`_cwb_setup.py` defines `SEPARATION_PHYSICAL = 20 au` and a docstring
describing "20 au from its companion", but this variable is never actually
used — the live code (`sep_in_au = 20` → `code_length = (sep_in_au/10) au =
2 au`, independent of the hardcoded `SEPARATION = 0.4`) means the separation
*actually simulated* is `0.4 * 2 au = 0.8 au`, **25x smaller** than
documented. Confirmed this doesn't explain the Mach gap on its own (the
adiabatic-cooled estimate at the *actual* 0.8 au separation, ~131-355, is
still >>10.3) — consistent with item 4's own "code units don't matter"
finding, generalized: rescaling a physical length scale alone doesn't change
any dimensionless ratio the scheme sees. Still worth fixing so the setup
simulates what its own documentation claims.

**Likely root cause, quantified directly by the same script**
(`cells_between_injection_and_shock`): at N=64 (the only resolution item 4
has gotten to run clean), there are only **~3.75 grid cells** between the
edge of the wind-injection sphere and the star-2 stagnation-point shock
(`num_injection_cells = num_cells // 32` is a fixed ~1/32-of-box fraction,
so this scales only because `dx` shrinks with `N`: ~7.5 cells at N=128, ~15
at N=256, ~30 at N=512). A 2nd-order finite-volume scheme cannot resolve a
temperature transition spanning the ~2-3 orders of magnitude adiabatic
cooling implies (see estimate 2 above) within 3-15 cells — numerical
diffusion mixes the freshly-injected (numerically much hotter, near-ambient
pressure-scale) wind with whatever it's near well before it reaches the
shock, so the *pre-shock* wind the simulation actually sees is far warmer
(far lower true Mach) than the real, physically-cooled wind. This is
consistent with, and extends, every relevant finding already in this file:
item 4's own "code units don't matter" result (this is a resolution effect,
not a scale effect), and item 1b/round 9-11's finding that the shock-finder
side of the problem is now essentially fixed (Tier 2 has no drift, no cell
loss, and converges cleanly) — the remaining gap is upstream of the shock
finder, in what physical state the wind is in by the time it reaches the
shock at all.

**Proposed fix (not yet implemented):** don't try to resolve the wind's
adiabatic cooling with brute-force uniform resolution — item 4's whole
saga shows the sim is already barely stable at N=64 and gets *harder*, not
easier, to keep clean at higher N, and per the cells-between-injection-
and-shock scaling above, resolving even ~2 more decades of cooling would
need N in the many hundreds to thousands (`~0.06 * N` cells available,
linear in N; getting comfortably past ~50-100 cells needs N~1000+, i.e.
~4000x the memory/compute of N=64). Instead: **impose the analytic
freely-expanding wind profile directly** (`rho(r) = Mdot/(4 pi r^2 v_inf)`,
pressure from the same adiabatic-cooling law `expected_cwb_mach.py` uses)
as an overwritten/reset state in a "wind zone" extending well beyond the
current tiny injection sphere (out to some fixed fraction of the stagnation
distance, chosen so several cells of *ordinary hydro* remain between the
wind-zone edge and the shock for the Riemann solver to actually capture the
jump) — every step or substep, not just once at t=0. Ordinary hydro/HLLC
still does the actual shock-capturing outside the wind zone; only the
pristine unshocked-wind interior, whose analytic solution is already known
exactly, is prescribed rather than rebuilt from a volumetric source term
each step. This is a standard technique in wind-bubble/CWB grid codes (an
analytically-prescribed "wind injection zone" larger than a single
source-term cell, refreshed every step) — it directly targets the diagnosed
mechanism (pre-shock wind is numerically too warm because there's no room
to develop the true cooling profile) without requiring the sim to run at a
resolution it currently cannot reach cleanly. More invasive than anything
tried in items 1b/4 so far (touches how the wind is represented in the
domain, not just the shock finder or the reconstruction), and would need
its own validation (in particular: does overwriting a growing wind-zone
each step interact badly with the N-body-tracked, moving source positions,
or with the open-boundary/positivity issues item 4 already found) — not
attempted this round, in the interest of reporting the (now well-quantified)
diagnosis and proposal rather than a rushed, unvalidated implementation.

**Secondary, cheaper, non-exclusive knob**: decouple `num_injection_cells`
from `N` (e.g. fix it at 2 regardless of resolution, instead of
`num_cells // 32`) — `expected_cwb_mach.py`'s resolution panel shows this
buys a modest extra margin "for free" (e.g. ~90 vs ~60 cells at N=1024) at
zero cost, by shrinking the injection sphere's *code-unit* radius as
resolution grows instead of holding it at a fixed ~1/32-of-box fraction.
Worth doing regardless of the wind-zone fix above, but not a substitute for
it — it only meaningfully closes the gap at resolutions the sim cannot
currently run stably.

Committed script: `expected_cwb_mach.py` (this folder) — prints the baseline
summary and writes `figures/expected_cwb_mach.png` (6 panels: ambient-Mach
vs. T_ambient and v_wind, stagnation-point location vs. mass-loss ratio,
adiabatic-ceiling vs. ambient-floor Mach vs. separation, resolution margin
vs. N, and Mach modulation over an eccentric orbit).

---

## 2026-09-21, round 13: re-checked the theory-vs-sim Mach gap against a real N=512, a=50 au run (the round-12 separation bug now fixed, T_END now derived from wind-crossing physics) — gap is unchanged, actually worse in relative terms

Since round 12, `_cwb_setup.py` was updated (separate work this session): the
separation bug is fixed (`code_length = sep_in_au / SEPARATION`, so the setup
now genuinely simulates `a=50 au`, not the old silently-0.8-au value),
`nbody=False` (stationary, matching round 8's finding), `num_cells=512`
(8x item 4's N=64 baseline), and `T_END` is now derived from a wind
boundary-crossing-time estimate rather than an arbitrary fraction of the
orbital period. The user ran this real N=512 setup and produced
`figures/cwb_shocked_cells_{2d,3d}_512_stat.png`; asked to check the Mach
agreement between the *current* physical parameters and this real result.

**No raw shock-finder data was saved from that run** (no `data/` output, no
printed percentiles) — only the two PNGs. Recovered the actual Mach range by
calibrating the plots' own colorbar pixel-by-pixel: sampled the tick-mark row
positions (labeled 2/4/6/8/10) in both figures, fit a linear pixel-to-value
map, and extrapolated to the colorbar's true top/bottom edge. **First attempt
was contaminated by suptitle/subplot-title text bleeding into the same pixel
column as the colorbar** (gave a spuriously high 2D-slice max of 12.1, higher
than the reported 3D whole-domain max — a logical impossibility that exposed
the bug); fixed by isolating the colorbar's own *contiguous* solid-color
pixel run (excluding the isolated 1-2px text-bleed blobs above it) as the true
top edge. Corrected reading, cross-validated two ways:
- 2D mid-plane (z=256) slice: Mach ∈ [1.29, 10.67].
- 3D whole-domain scatter (the authoritative one — `plot_shock_surface_3d`'s
  SCATTER mode auto-scales its colorbar to `mach[surface].min()/.max()` over
  the *entire* 3D array, unlike the 2D panel which explicitly fixes
  `vmin=MACH_MIN` and only shows one slice): Mach ∈ [1.31, 10.95] → reported
  as **~10.7** (splitting the difference between the two independent
  calibrations, which agree to ~2%).
- Both `vmin` readings land almost exactly on `MACH_MIN=1.3` (the shock-zone
  detection threshold, not a physical floor) — strong validation that the
  pixel-calibration method itself is correct.
- This run used the default `find_shocks_pfrommer` call in `_cwb_setup.py`
  (no `mach_sampling_steps` override) — i.e. Tier-1's `steps=1` backward-
  compatible default, not even Tier-1's own `steps=2` improvement (round 10),
  let alone the unmerged Tier-2 prototype (round 11).

**Theoretical expectation, recomputed for the current (now-correct) a=50 au
setup** (`expected_cwb_mach.py`, updated this round to drop the now-resolved
documented-vs-actual separation split): stagnation point unchanged (wind
momentum ratio only depends on `Mdot`/`v_inf`, not `a`) — star 2's shock
(the one actually measured, near the orbital plane close to star 2, per
rounds 9-10) sits at `r2 ≈ 11.24 au` (was `0.18 au` under the old bugged
separation). Ambient-comparison floor **unchanged** at ~129 (star 2), since
it doesn't depend on `a`. Adiabatic-cooled-wind ceiling **increased** to
**~2061** (star 2) / ~5594 (star 1) — up from round 12's ~131/~355 at the
old, artificially-tiny separation, since a real wind gets far more distance
(and hence far more adiabatic cooling) to travel before colliding at the
correct, larger separation.

**Result: the gap did not close — if anything it's worse in relative terms.**
Despite 8x more grid resolution (N=64→512, giving ~30 cells instead of ~3.75
between the injection sphere's edge and the shock — see
`cells_between_injection_and_shock`, confirmed this ratio is *not* affected
by the separation fix, only by `N`, exactly as round 12 predicted) and a
properly-scaled, 62x-larger physical domain, the measured max Mach barely
moved (~10.3 at N=64 → ~10.7 at N=512, a ~4% change) while the *theoretical*
ceiling grew substantially (because the fix gave the wind more real distance
to cool). Sim is now **~12x below the conservative ambient-comparison floor**
(unchanged from round 12's ~13x) and **~193x below the adiabatic-cooled
ceiling** (worse than round 12's ~30x at the old, bugged separation, precisely
*because* the separation fix raised the ceiling without raising the sim's
actual result).

**This is a third independent resolution level (N=64, 256, now 512, across
two different, unrelated sessions) all landing at the same ~10-11 max Mach**
— strong, repeated confirmation of round 12's core diagnosis: the bottleneck
is not shock-finder sampling (already fixed as far as tractable, Tier 1/2),
not the separation bug (now fixed, made no difference), and not resolvable by
brute-force grid refinement alone (8x more cells in exactly the zone that
matters barely moved the needle). The proposed fix from round 12 — impose the
analytic free-wind profile in an extended, per-step-refreshed wind zone
instead of relying on volumetric source terms + hydro to reconstruct the
cooling profile — remains the recommended next step; nothing this round
changes that recommendation, it only strengthens the evidence for it.

**Caveat on precision:** the ~10.7 figure is a calibrated pixel reading of a
PNG, not an exact computed value (no data file was saved from the run) —
good to roughly ±0.3-0.5 given the two independent calibrations' ~2%
agreement, but if a precise percentile breakdown (median, p90, p99, not just
the max) is needed for a future decision, the run should be repeated with the
actual `sf_result.mach_numbers` array saved or printed directly, the way
rounds 9-11 did.

Updated: `expected_cwb_mach.py` (`SEPARATION_BASELINE=50 au` replaces the
now-resolved documented-vs-actual split; `SIMULATED_MAX_MACH=10.7` from this
round's calibrated reading; resolution panel now marks both the round-12
(N=64) and round-13 (N=512) runs).

---

## 2026-09-21, round 14: is disabled cooling the actual cause? Enabled it (real fix, not scratch) — clean, controlled A/B test shows it makes no measurable difference at N=64

User's question: `_cwb_setup.py` runs pure adiabatic hydrodynamics (no
radiative cooling at all) — could *that*, not grid resolution, be why the
pre-shock wind never gets cold enough? Physically plausible and worth taking
seriously: unlike adiabatic (PdV) cooling, radiative cooling is a **local,
pointwise** process (rate ~ n^2 * Lambda(T)) that doesn't need a wide spatial
stencil to resolve — in principle it could cool the wind to a low
temperature right at/near the dense injection region, sidestepping round
12/13's "not enough cells between injection and shock" diagnosis entirely.
Real O-star CWBs are also known in the literature to span both adiabatic and
radiative regimes depending on wind/orbital parameters (the Stevens, Blondin
& Pollock 1992 cooling parameter chi), so this is a genuine, previously-
missing piece of physics, not just a numerical knob.

**Implemented** (real repo file, not scratch): `_cwb_setup.py` now wires up
the codebase's existing cooling module (`astronomix._modules._cooling`),
using `pytests/shock_finder3D/cooling_comparison2.py` (an old, `jf1uids`-era
draft script, API-incompatible with the current `astronomix` package but
useful as a guide) as a reference for which knobs to set:

- `SimulationConfig.cooling_config = CoolingConfig(cooling=True,
  cooling_method=IMPLICIT_COOLING, subcycle_stiff_cooling=True,
  cooling_curve_config=CoolingCurveConfig(cooling_curve_type=
  PIECEWISE_POWER_LAW))`.
- `SimulationParams.cooling_params = CoolingParams(hydrogen_mass_fraction=
  0.76, metal_mass_fraction=0.02 -- `CoolingParams`' own defaults --
  floor_temperature=<100 K in code units>, cooling_curve_params=
  schure_cooling(CODE_UNITS))` -- the Schure et al. (2009) + Dalgarno &
  McCray (1972) piecewise power-law curve, same one the draft used.
- **`IMPLICIT_COOLING` (Newton's method) chosen over the draft's
  `EXPLICIT_COOLING`**, and **`subcycle_stiff_cooling=True` is not
  optional**: `cooling_options.py`'s own docstring documents a prior
  incident (SILCC-ISM M4) where a single persistently-fast-cooling cell
  stalled a run indefinitely, because `_cfl_time_step`'s `dt_cool` term
  clamps the *global* hydro `dt` to the single fastest-cooling cell in the
  whole domain whenever `subcycle_stiff_cooling=False` (confirmed directly
  in `_timestep_estimator.py:312`). This setup's injection region is
  guaranteed to be exactly that single fastest-cooling cell (item 4's own
  "~17 orders of magnitude pressure contrast" finding), so running with the
  default (`subcycle_stiff_cooling=False`) would very likely reproduce that
  same stall.

**Verified safe first** (two smoke tests, not the real N=512 target -- that's
still the user's to run): N=32 completed in ~33s with zero NaN, but turned
out to be a degenerate case (`num_injection_cells = 32 // 32 = 1`; density
and pressure came back *exactly* uniform ambient everywhere -- no wind
injection happened at all at this resolution, unrelated to cooling). N=64
(the resolution used throughout rounds 9-13) completed cleanly in ~39s: zero
NaN, wind injection clearly present (density up to 1.7e6x ambient, pressure
up to 1.2e10x ambient), 6315 shock-surface cells found -- the same count
rounds 9-11 found at N=64 without cooling, a good sign nothing is
qualitatively broken.

**Clean, controlled A/B test (same exact setup, only `cooling_config.cooling`
toggled via a monkeypatch, both at N=64): essentially no difference.**

| | n_surface | Mach min | median | max |
|---|---|---|---|---|
| cooling ON | 6315 | 1.583 | 3.122 | 9.998 |
| cooling OFF | 6315 | 1.581 | 3.123 | 9.986 |

Differences are ~0.1%, within run-to-run float/compilation noise -- not a
real effect. **Cooling does not measurably change the shock Mach number at
N=64.**

**Why not, most likely:** cooling needs *time* to act, exactly like
adiabatic cooling needs *distance* -- and at N=64 there are only ~3.75 grid
cells (round 12/13's own number) between the injection sphere's edge and the
shock, i.e. only a very short transit time for a fluid parcel to spend in
that zone before crossing the shock. If that transit time is short compared
to the *local* cooling time even in the dense near-injection gas, cooling
simply doesn't have long enough to remove much thermal energy before the
parcel is already at the shock -- the same "not enough time/distance"
bottleneck as round 12/13's adiabatic-cooling diagnosis, just for a
different physical channel. This reframes the round-12/13 conclusion
slightly: it isn't specifically that adiabatic cooling is the missing
mechanism and radiative cooling would fix it -- it's that *no* thermal
process, adiabatic or radiative, gets enough time/distance to act in the
available few cells.

**Two secondary findings, neither the main conclusion but both worth
recording:**
1. **Achievable cooling floor is bounded by the curve's own table range,
   independent of the time argument above.** Schure's tabulated cooling rate
   is defined only for `T in [~6300 K, ~1.4e8 K]`; `_evaluate_piecewise_
   power_law` returns exactly `0.0` (no cooling, not an extrapolation)
   outside that range. Even with unlimited transit time, this specific curve
   could only cool the wind down to ~6300 K (giving `c_s ~ 9.3 km/s`,
   Mach ~200-236 for this setup's `v_inf`) -- a real, substantial
   improvement over the current ~10.7 if it could act, but nowhere near the
   ~2000+ adiabatic ceiling from round 13's idealized (unlimited-distance)
   estimate. A colder curve (extending the Dalgarno & McCray branch further,
   or a different table) would be needed to test the true ceiling.
2. **A mu-convention mismatch shifts the *effective* ambient temperature the
   cooling module sees.** `RHO_0`/`P_0` were originally defined assuming
   `mu=1` (`n = 2/cm^3` computed as `rho/m_p` with no ionization correction,
   giving the documented "T~1.5e4 K"). The cooling module's own
   `get_temperature_from_pressure` uses `mu~0.59` (from
   `hydrogen_mass_fraction=0.76`, appropriate for an actually-ionized
   plasma), so it recovers `T_ambient ~ 8850 K` for the *same* `(rho, P)`
   pair -- a real, ~40% systematic offset between "the temperature this
   setup's docstring claims" and "the temperature the cooling module
   actually computes for that gas", worth knowing about but not large enough
   to change this round's conclusion (8850 K is still comfortably inside the
   Schure table's range).

**Current code state:** cooling is now genuinely enabled in `_cwb_setup.py`
(not reverted) -- verified numerically safe (no NaN, no stall) and worth
keeping regardless of the null result, since it's real missing physics that
costs nothing now that `subcycle_stiff_cooling` avoids the stall risk, and
does not regress the (already small) N=64 shock-finder statistics. The
user's real target (N=512, a=50 au) has not been run with cooling yet --
that re-check, and specifically whether cooling's effect (if any) grows with
resolution the way it would need to for a resolution-dependent transit-time
argument, is the natural next step.

Scratch commands (not saved as scripts, run directly via `python3 -c`, not
committed): the N=32/N=64 smoke tests and the cooling-on/cooling-off A/B
comparison described above.

---

## 2026-09-21, round 15: where is Mach actually computed, and is the shock-finder formalism (not just physics) part of the gap? Re-verified item 1b's Tier-1 finding against the *current* setup — yes, and it's still not wired in, but it doesn't close the gap either

User asked where the shock Mach number is actually computed and whether the
shock-finder formalism itself could be the problem, given rounds 12-14 ruled
out resolution, the separation bug, and cooling as the explanation.

**Where it's computed:** `astronomix/shock_finder3D/pfrommer_shock_finder.py`'s
`find_shocks_pfrommer` (phase 4) calls `_shock_mach.py`'s
`_calculate_mach_at_surface`, which:
1. Samples pressure `sampling_steps` cells away from each shock-surface cell
   along the local shock direction, via `_shock_zones.py`'s
   `get_post_pre_shock_values` (walks the single *dominant grid axis* of the
   direction vector, `jnp.roll`-based, `sampling_steps` cells each way).
2. Computes `p_ratio = p_post / p_pre` (clamped >= 1).
3. Inverts the strong-shock Rankine-Hugoniot relation for `M`:
   `M = sqrt((p_ratio*(gamma+1) + (gamma-1)) / (2*gamma))`, `gamma=5/3`
   hardcoded (matches this setup's `GAMMA`, so not itself a live bug here).
4. Masks to shock-surface cells at least `sampling_steps` from any boundary
   (`_make_interior_mask`, since `jnp.roll` wraps).

`find_shocks_pfrommer` exposes `mach_sampling_steps` (default `1` -- literally
the worst case per item 1b/round 10's own characterization) as a public
kwarg, but **`_cwb_setup.py`'s actual call
(`find_shocks_pfrommer(state, config, registered_variables, helper_data,
mach_min=MACH_MIN)`) has never passed it** -- every single result reported
in rounds 9 through 14 (including this session's N=64/N=512, cooling on/off,
etc.) used the un-improved `sampling_steps=1` default, despite Tier 1 having
been implemented and validated back in round 10.

**Re-verified round 10's finding against the *current* setup** (a=50 au,
cooling on, N=64 -- not the old, bugged 0.8 au setup round 10 originally
tested): ran the real `run_cwb(64)` once, then called `find_shocks_pfrommer`
directly on the resulting state at several `mach_sampling_steps` values
(cheap -- no need to re-run the hydro):

| steps | n_valid | min | median | mean | p90 | max |
|---|---|---|---|---|---|---|
| 1 (current default) | 6315 | 1.583 | 3.122 | 3.496 | 5.478 | 9.998 |
| 2 | 5788 | 1.722 | 5.007 | 5.021 | 6.160 | 9.080 |
| 3 | 5413 | 1.000 | 5.095 | 5.323 | 7.172 | 8.931 |
| 4 | 5053 | 1.000 | 4.567 | 5.256 | 8.545 | 10.481 |
| 5 | 4705 | 1.000 | 4.037 | 5.102 | 9.751 | 11.825 |
| 8 | 3729 | 1.000 | 2.730 | 4.410 | 14.184 | 15.979 |

**Confirms round 10's exact pattern, on the new setup:** `steps=2` gives a
real, substantial median improvement (3.12→5.01, +60%, closely matching
round 10's own 3.22→5.04 on the old setup) at essentially the same max
(9.08 vs 9.998 -- a small *decrease*, from the boundary-margin exclusion
cutting the near-edge cells, exactly as round 10 diagnosed). Beyond
`steps=2-3`, `n_valid` keeps shrinking (boundary-margin loss) while `max`
climbs *without plateauing* (9.08→15.98) -- the same grid-axis-vs-true-normal
drift artifact round 10 found, not genuine signal (round 11's Tier-2
prototype, walking the true local normal instead of a grid axis, was built
specifically to fix this drift, but was never merged and isn't available in
the repo to re-test here).

**So: yes, the shock-finder formalism is measurably part of the picture --
but it is a secondary, low-risk lever, not the explanation for the gap's
size.** `steps=2` is a real, free, already-validated ~60% median
improvement sitting unused in `_cwb_setup.py` right now -- worth wiring in
regardless of anything else. But even at its best (`steps=2` or `3`), median
tops out at ~5.1 and max *decreases* to ~9.1 -- nowhere near closing the
~12x (ambient-floor) to ~190x (adiabatic-ceiling) gap from rounds 12-13, and
entirely consistent with (not a competing explanation to) round 14's
finding that the *physical* pre-shock state itself is the binding
constraint, independent of both resolution and cooling. Sampling-formalism
fixes cannot recover Mach signal that was never physically present in the
pre-shock state to begin with.

**Not yet decided/applied:** whether to set `mach_sampling_steps=2` as the
new default in `_cwb_setup.py`'s `find_shocks_pfrommer` call. It's a free
win with no observed downside at `steps=2` specifically, but changes every
historical Mach number this file has quoted (median comparisons across
rounds), and round 10 explicitly deferred this exact decision pending the
axis-direction limitation Tier 2 addresses -- left for the user to decide
rather than applied unprompted this round, since this round was scoped as
"investigate", not "fix".

Scratch command (not saved, run directly via `python3 -c`, not committed):
the `mach_sampling_steps` sweep described above, reusing one `run_cwb(64)`
call's output state across all six `find_shocks_pfrommer` calls.

---

## 2026-09-21, round 16: literature review + a real, committed fix to the shock-finder's Mach sampling — validated on a synthetic shock and Sedov, but confirms (doesn't overturn) that the CWB gap is physical, not a formalism bug

User pushed back on round 15 ("changing the step size is basically not a real
improvement... since max mach number stays roughly the same") and asked for
a literature-grounded investigation of `_calculate_mach_at_surface` itself,
with an implemented fix.

**Literature review** (`WebSearch`/`WebFetch`, arXiv full text via `pymupdf`
since `WebFetch` alone couldn't extract PDF text): read Schaal & Springel
(2015, MNRAS 446, 3992, https://arxiv.org/pdf/1407.4117) in detail — the
paper this codebase's own module docstring already cites as "Pfrommer et al.
2017 methodology" builds directly on. Traced the field's actual history:

- Quilis et al. (1998), Miniati et al. (2000): naive cell-by-cell jump
  conditions.
- **Ryu et al. (2003)**: introduced the two-step zone-then-surface method
  this codebase already implements (criteria 1-3, then max-compression
  cell) -- but for 3D, computed the Mach number along **each of the three
  coordinate axes separately** and took the *maximum*, not a single
  "dominant axis" pick.
- **Vazza et al. (2009)**: refined this to `M = sqrt(Mx^2+My^2+Mz^2)`
  ("minimizing projection effects"), and found the *velocity* jump gives
  less scatter than temperature in their code.
- **Skillman et al. (2008)**: avoided coordinate-splitting entirely via the
  local temperature-gradient direction (exactly this codebase's
  `shock_direction = -grad(T)/|grad(T)|`) plus a pre-/post-shock
  temperature-and-density cross-check against tangential discontinuities.
- **Schaal & Springel (2015)**, Sec. 2.3.3 (the load-bearing find): "rays
  are sent from each cell of the shock zone in the direction of the
  post-shock region... **When the first cell outside of the shock zone is
  reached, the post-shock temperature is recorded**, and the ray direction
  is reversed in order to find the pre-shock region." I.e. the walk
  distance is **not a fixed step count at all** -- it's however far the
  actual (per-cell-varying, ~3-4-cell) shock zone happens to extend at that
  point. This is the exact mechanism missing from this codebase's
  `sampling_steps` parameter (fixed count) and is *precisely* what item 1a
  attempted and had to revert (2026-08-26) -- but item 1a used the existing
  `jnp.roll`-based, grid-axis-only sampler, which is suspected (never
  confirmed) to have caused its NaN regression via wraparound. Also
  confirmed via Fig. 2 of the paper: the authors tested pressure/
  temperature/velocity/entropy jump estimators against ten shock-tube tests
  and found **temperature jump has the best accuracy** in AREPO (this
  codebase uses pressure -- noted as a possible secondary refinement, not
  pursued this round since the primary lever is clearly the sampling
  distance, not the choice of RH invariant).

**Implemented** (real repo files, `astronomix/shock_finder3D/`, not
scratch): `get_post_pre_shock_values_adaptive` (new function,
`_shock_zones.py`) -- walks along the **continuous local shock-normal**
(trilinear interpolation, `jax.scipy.ndimage.map_coordinates`) **until
leaving `shock_zones`** (the literal Schaal & Springel criterion, using the
zone mask `identify_shock_zones` already computes), with `mode="nearest"`
(clamps at the boundary) instead of `jnp.roll` (wraps) -- structurally
eliminating item 1a's suspected wraparound mechanism rather than
patching around it. Wired through as a strictly **opt-in** parameter
(`adaptive=False` default in `_calculate_mach_at_surface`,
`mach_sampling_adaptive=False` default in `find_shocks_pfrommer`) so every
existing caller (CR-grey, the fixed-step CWB/Sedov paths) is byte-for-byte
unaffected unless it explicitly asks for the new path -- learning directly
from item 1a's mistake, where an equivalent zone-aware walk was applied as
the *default* and silently broke an unrelated caller.

**Two real bugs found and fixed during validation, before this could be
trusted (exactly the discipline item 1a skipped):**

1. **OOM at realistic resolution.** The first implementation used a
   Python-unrolled loop (`for step in range(1, max_steps+1)`) for the walk.
   Passed a *synthetic* single-profile check fine, and even ran clean on the
   real N=64 CWB state fine -- but **crashed with `RESOURCE_EXHAUSTED`
   trying to allocate 11 GB** on the N=256 Sedov regression test: 15
   unrolled steps x 2 directions x 3 `map_coordinates` calls, each
   materializing full-grid intermediates, blew up the XLA compilation
   graph's memory footprint at realistic grid sizes. **Fixed** by rewriting
   the walk with `jax.lax.fori_loop` (traced once, executed by XLA's own
   `While` op) instead of unrolling -- re-verified bit-identical results on
   the synthetic test after the rewrite, and the N=256 Sedov run then
   completed cleanly. Lesson: a Python-unrolled JAX loop that runs fine at
   toy/small-N sizes can still be a real resource-usage bug at production
   grid sizes -- test the actual target resolution, not just N=64, before
   trusting a design.
2. **Boundary-margin/sampling-distance desync between Mach and
   thermal-energy flux.** `calculate_thermal_energy_flux`
   (`_energy_dissipation.py`) was not ported to the adaptive walk (still
   samples its own pre-shock density/pressure via the old fixed-step
   `get_post_pre_shock_values`), and `find_shocks_pfrommer` was reusing the
   same `mach_sampling_steps` value for both the adaptive walk's step cap
   *and* the flux calculation's fixed-step distance/boundary-margin size.
   With `mach_sampling_steps=15` (a sensible adaptive cap, but a huge
   fixed-step margin), this force-zeroed flux over a much wider boundary
   region than the now margin-free adaptive Mach output, breaking the
   Sedov test's "flux nonzero exactly at surface" invariant (confirmed
   failing before the fix). **Fixed**: `find_shocks_pfrommer` now passes a
   fixed `flux_sampling_steps=1` to the flux calculation whenever
   `mach_sampling_adaptive=True`, decoupling the two. This is a
   documented, not fully resolved, accuracy caveat (flux's own pre-shock
   sample point can now differ from the Mach's when adaptive is on) --
   porting flux to the adaptive walk too is future work, out of this
   round's scope (Mach was the actual target).

**Validation, in the order item 1a's postmortem said should have happened
the first time:**

1. **Synthetic smeared shock** (1D profile, exact Rankine-Hugoniot M=100
   jump smoothed over a logistic ramp of width `w` cells, matching item
   1b's original verification style): fixed-step collapses just as item 1b
   documented (`M(w=1)=84, M(w=2)=32, M(w=3)=14...` down to `M(w=12)=2.0`);
   **adaptive stays within ~12% of the true M=100 at every width tested, up
   to w=12** (`100.0, 99.7, 95.6, 92.6, 90.2, 88.5`). This is the core value
   proposition, cleanly demonstrated in isolation.
2. **Sedov-Taylor correctness test** (`sedov_shock_finder.py`, N=256 -- the
   exact test item 1a broke, re-run this time with adaptive mode explicitly
   enabled): every existing assertion passes for *both* modes (surface
   subset of zones, Rankine-Hugoniot bounds, `mach>=mach_min` on surface
   and `==0` off it, flux non-negative and nonzero exactly at surface, no
   NaN). Fixed-step result is **bit-identical** to before this round's
   changes (`mach in [2.87, 24.57], median=7.42`, confirming zero
   regression to the untouched default path). Adaptive result: **`mach in
   [6.44, 134.29], median=125.90`** -- a ~17x median improvement, recovering
   the physically-expected very-high Mach of a point explosion into cold
   ambient gas that the fixed-step default's narrow sampling was
   suppressing all along.
3. **The real, current CWB run** (N=64, a=50 au, cooling on -- reusing one
   `run_cwb(64)` state, calling `find_shocks_pfrommer` directly at several
   settings): adaptive gives **`n_valid=6315/6315` (no cell loss, unlike
   fixed-step's margin exclusion), median=4.29, max=10.36** -- essentially
   the same order of magnitude as the fixed-step default (median 3.12, max
   10.00), and **identical between a 15-step and a 30-step safety cap**
   (proving the walks are genuinely converging to a true zone exit, not
   truncated by the cap).

**Conclusion: this is a real, validated, literature-grounded fix to the
shock-finder formalism -- and it does not change the answer to the CWB
question.** The Sedov result proves the method genuinely recovers Mach
signal the fixed-step sampler was suppressing, when that signal is actually
present in the data (a 17x jump, matching the physically-expected value).
Applying the *identical*, now-trusted method to the real CWB data recovers
essentially the same ~10 plateau as before. Three independent, mutually
reinforcing lines of evidence now say the same thing: round 13's brute-force
resolution/separation tests, round 14's cooling A/B test, and now round 16's
best-practice literature method all agree the ~10 ceiling reflects the
actual pre-shock physical state in the simulation, not an artifact of how
that state is measured.

**Applied to the real pipeline**: `_cwb_setup.py`'s `find_shocks_pfrommer`
call now uses `mach_sampling_adaptive=True, mach_sampling_steps=15` (unlike
round 15's `mach_sampling_steps=2` recommendation, which was left
undecided -- this one is applied, since it's now been through the full
validation item 1a skipped, doesn't lose any near-boundary cells, and
converges cleanly). `find_shocks_pfrommer`'s own default remains
`mach_sampling_adaptive=False` (fully backward-compatible for every other
caller, e.g. `cr_grey_injection.py`).

**Not done / future work:** thermal-energy flux hasn't been ported to the
adaptive walk (see bug 2 above); the temperature-vs-pressure RH-invariant
question Schaal & Springel's Fig. 2 raises hasn't been tested in this
codebase; Ryu (2003)/Vazza (2009)'s multi-axis combination (`max` or
quadrature over per-axis Mach) was read about but not implemented, since the
continuous-normal approach (Skillman 2008's fix to the same grid-orientation
problem) was judged the more direct, already-partially-precedented (Tier 2,
round 11) choice.

Committed changes: `astronomix/shock_finder3D/_shock_zones.py` (new
`get_post_pre_shock_values_adaptive`), `_shock_mach.py`
(`_calculate_mach_at_surface`'s new `adaptive`/`shock_zones` params),
`pfrommer_shock_finder.py` (new `mach_sampling_adaptive` param, flux
sampling-steps decoupling), `pytests/shock_finder3D/_cwb_setup.py` (now
calls with adaptive mode on). Scratch commands (not saved, run directly,
not committed): the synthetic-profile sweep, both Sedov regression checks
(pre- and post-flux-fix), and the real-CWB adaptive sweep described above.

---

## 2026-09-21, round 17: what's the exact theoretical Sedov Mach number, and does round 16's fix actually match it? Yes, closely.

User noticed the committed `sedov_shocked_cells_3d_256.png` (pre-round-16,
fixed-step) showed a max of ~24 and asked whether that's realistic for this
setup's exact parameters (`_sedov_setup.py`: `T_END=0.1`, `E_EXPLOSION=1.0`,
`RHO_AMBIENT=1.0`, `P_AMBIENT=1e-4`, `GAMMA=5/3`).

**Theoretical answer, from the exact Sedov-Taylor self-similar solution**
(spherical/3D point explosion; `examples/scripts/forward/hydro/sedov_blast.py`
already uses `exactpack`'s `Sedov` solver for this exact comparison, but
`exactpack` isn't installed in this environment or reachable via pip here,
so computed directly from the closed-form self-similar formula instead --
same physics, same result): shock radius `R_s(t) = xi_0 * (E t^2/rho_0)^0.2`
with `xi_0 = 1.15167` (the standard tabulated spherical, `gamma=5/3`
constant, Sedov 1959/Taylor 1950 -- equivalently the energy-integral
constant `alpha = xi_0^-5 = 0.4935`), shock velocity
`v_s = 0.4 * R_s/t` (since `R_s ~ t^0.4`), Mach `M = v_s / c_0` with
`c_0 = sqrt(gamma * P_AMBIENT/RHO_AMBIENT)`. At `t=T_END=0.1`:
`R_s = 0.4585`, `v_s = 1.834`, `c_0 = 0.01291`, **`M = 142.1`**.

**Cross-validated the formula/constant two ways:**
1. `R_s=0.4585` vs. round 9's own independently-measured mean shock-surface
   radius at N=256 (`mean_r = 0.46272`, measured via the *density* profile,
   nothing to do with Mach sampling) -- agree to **0.92%**.
2. Applied round 16's adaptive fix (already validated as accurate on this
   exact test in round 16) to this exact run: **median Mach = 125.90, max =
   134.29** -- both within **6-11% of the theoretical 142.1**, using an
   *exact*, independently-derivable analytic benchmark (stronger than the
   CWB case's own necessarily-approximate physical estimates).

**Direct answer: no, the pre-round-16 figure's max of ~24.57 is not
realistic -- it underestimates the true shock strength by ~5.8x**, via the
exact same fixed-1-cell-sampling-inside-a-smeared-transition mechanism item
1b/round 16 diagnosed for the CWB case, now independently confirmed against
an exact analytic solution rather than only an order-of-magnitude physical
estimate. This is the single most convincing piece of evidence so far that
round 16's adaptive method is not just "different" but *quantitatively
accurate*: given a setup with a known-exact answer, it reproduces that
answer to ~10%, while the old default was off by ~6x.

Applied and regenerated: `pytests/shock_finder3D/figures/
sedov_shocked_cells_3d_256.png` and `sedov_shock_finder_correctness_256.png`
now reflect `_sedov_setup.py`'s new adaptive default (round 16) -- the 3D
figure now shows the shock sphere almost uniformly bright yellow/white
(Mach ~120-134) as the self-similar solution predicts, with a handful of
small dark (low-Mach) spots at specific angular positions -- grid-imprinting
artifacts where the Cartesian mesh's discretization interacts unfavorably
with the spherical symmetry, consistent with the surviving `min=6.44`
outlier and not a concern for the correctness test's own (unchanged,
resolution-relative) assertions.

---

## 2026-09-21, round 18: FINAL VERDICT on item 1b — Mach~10 is a correct measurement of a genuinely-too-warm simulated wind; root cause pinned down to the wind-injection module's thermal state, not a remaining code bug

User asked directly: given round 17 proved the shock finder itself is
accurate (Sedov, ~10% of the exact answer), is the CWB's ~10 Mach *actually*
correct, or is there still an undiagnosed bug somewhere else?

**Direct answer: ~10 is a correct measurement of what is actually in the
simulated field. The simulated field itself is not a correct representation
of a real O-star wind's pre-shock temperature.** Pulled the real density/
pressure/velocity profile along the ray through the max-Mach surface cell
(N=64, a=50 au, cooling on -- `_cwb_setup.run_cwb(64)`, sampled the
y=z=box-center axis line, which passes within 1-2 cells of the actual
argmax cell at index (37,32,33), Mach 10.36):

| x | rho | p | T=p/rho (pseudo-T) | \|v\| |
|---|---|---|---|---|
| 0.0703 (pre-shock) | 8.42e-8 | 3.76e-4 | **4467.8** | 770.5 |
| 0.0859 (post-shock) | 2.55e-7 | 3.08e-2 | 120749.3 | 247.5 |

`T_ambient = P_AMBIENT/RHO_AMBIENT = 17.45` (same pseudo-temperature
convention `_shock_mach.py` itself uses). **The gas immediately upstream of
this shock is at `T=4467.8`, i.e. 256x hotter than the true ambient ISM --
not remotely cold.** Its velocity (770.5, in code units) is close to the
wind's own terminal velocity, confirming this is genuinely the free-
streaming wind about to collide, not a numerically-drifted intermediate
state. This directly explains the modest Mach number with no remaining
mystery: `c_s(T=4467.8) = sqrt(gamma*4467.8) = 86.3`, `v/c_s = 8.93` --
matching the shock finder's own reported 10.36 for the true surface cell to
the expected precision (the sampled axis line is 1-2 cells off the exact
argmax cell). **The closing calculation**: if this same gas had genuinely
cooled all the way to the true ambient temperature (as the round-12/13
"ambient-comparison floor" estimate implicitly assumed), the *same*
velocity jump would register `M = 770.5/sqrt(gamma*17.45) = 142.9` --
matching that floor estimate (~129-153) almost exactly. **The entire gap
between measured (~10) and expected (~130-150) collapses to one number: the
pre-shock wind's temperature is ~256x higher than it should be at the point
of collision, not the shock jump itself being mismeasured.**

**Where this actually comes from (not a "bug" in the sense of code
disagreeing with its own equations -- every module checked works correctly
relative to its own inputs; a resolution/model-fidelity gap):** a real
O-star wind is launched hot near the photosphere (`~R_star`, a fraction of
an au) and cools substantially via adiabatic expansion by the time it
reaches a collision region many au away (round 12/13's adiabatic-ceiling
estimate: ~2 orders of magnitude in radius, hence ~2-3 orders of magnitude
cooling). In this simulation, the wind cannot be injected anywhere near the
true photospheric radius -- `num_injection_cells` sets a numerically
necessary injection sphere that is already a sizeable fraction of the
domain (at N=64, `injection_radius = 2*dx = 0.03125` code units, only
~3.75 grid cells short of the actual collision point). The injection
source term (`_wind_ei3D`, already the subject of item 3's earlier,
independently-validated fix) sets the *momentary* energy/momentum balance
needed to accelerate freshly-added mass to `v_inf` at that artificially
large radius -- it has no mechanism to account for the ~2-3 orders of
magnitude of adiabatic cooling a real wind would already have undergone
between the true photosphere and this numerically-necessary injection
radius. The wind is therefore injected, and stays, far hotter than the
physical wind it's meant to represent -- and (rounds 12-14) there is no
combination of grid resolution, code-unit/separation choice, or radiative
cooling available at any tractable resolution that gives the ~3-4 remaining
cells enough time or distance to close a 256x temperature gap on its own.

**This confirms item 1b is fully diagnosed, not merely "still open":**
- The shock-finder formalism (rounds 10, 11, 15, 16, 17): fixed and
  independently proven quantitatively accurate (Sedov, ~10% of exact).
- Grid resolution (rounds 12, 13): ruled out, 64->512 barely moves the
  number.
- The separation/code-unit bug (round 12/13 fix): ruled out, fixing it made
  the *relative* gap worse, not better.
- Radiative cooling (round 14): ruled out via a clean, controlled A/B test.
- **Root cause (this round): the wind-injection module deposits (and the
  simulation has no mechanism to subsequently cool) wind at a temperature
  ~256x too high for its distance from the star, because the injection
  radius is a numerically-necessary stand-in for the true (unresolvable)
  stellar photosphere, and the source term has no way to "pre-pay" the
  cooling history a real wind would have accumulated by that radius.**

**This is exactly round 12's proposed fix, now backed by a direct
measurement rather than an inference from cell-counting:** impose the
analytic free-wind density/temperature profile (`rho(r) ~ 1/r^2`,
`T(r) ~ r^-4/3` for adiabatic expansion from an assumed photospheric
base state) across an extended wind zone at every step, rather than letting
the source term alone set the injected gas's thermal state. This would
inject wind that is already as cold as it should be for its (unavoidably
large) numerical launch radius, sidestepping the resolution requirement
entirely rather than trying to satisfy it. Still not implemented (a
substantially larger undertaking than anything in items 1b/4, per round
12's own scoping) -- this round's contribution is closing out the
diagnosis with direct evidence, not implementing the fix.

Scratch command (not saved, run directly, not committed): the axis-profile
extraction and the closing Mach/temperature calculation described above,
reusing one `run_cwb(64)` call.

---

## 2026-09-23, round 19: analytic wind zone implemented as an opt-in option — cools the pre-shock wind ~15x and raises Mach ~1.4x (max) / ~2.4x (median), but the hydro solver re-heats the cold wind right at the zone edge

**Implemented (opt-in, default path unchanged):** `WindConfig.analytic_wind_zone`
(default `False`). When on, `_wind_ei3D` is followed by
`stellar_wind._analytic_wind_zone`. Every step, that function overwrites each
source's zone with `rho = Mdot/(4 pi r^2 v_inf)`, a radial `v_inf` (plus the
source's N-body velocity), and `p = rho T0 (r0/r)^(2(gamma-1))`. New
`WindParams` fields: `wind_base_radii`, `wind_base_temperatures`
(code p/rho), `wind_zone_stagnation_fraction` (default 0.5) and
`wind_zone_max_radius` (default inf; must be finite for a single source).
Zone radius = fraction x distance to the nearest ram-pressure stagnation
point, clipped to [EI injection radius, max radius]. `finalize_config`
rejects the flag unless the run is 3D + EI + FINITE_VOLUME + stellar_wind.
Unit tests: `pytests/shock_finder3D/analytic_wind_zone.py` (7 cases, all
pass on CPU). They check that flag-off equals plain `_wind_ei3D`
bit-for-bit, that in-zone cells match the analytic profile to 1e-5, that
out-of-zone cells are untouched, and that unsupported configs raise.
`_cwb_setup.run_cwb(..., analytic_wind_zone=False)` gained the opt-in, with
base state R0 = 20 Rsun and T0 = 3.5e4 K (the `expected_cwb_mach.py` values).

**N=64 A/B, current `_cwb_setup` (CR on, cooling off), both runs clean:**

| | Mach max | Mach median | Mach p90 | pre-shock T on axis (star-1 side) |
|---|---|---|---|---|
| plain EI | 24.65 | 4.36 | 9.71 | ~4600-6000 |
| analytic zone | 34.16 | 10.38 | 24.99 | ~325-620 |

(T_ambient = 17.45.) Star 1's zone reaches x ~ -0.045. Inside it, the wind
cools monotonically to T ~ 8 at the zone edge. **At the first hydro cell
past the edge, T jumps ~75x (8.2 -> 621) with no change in velocity and a
smooth density.** It then falls adiabatically to ~325 before the shock.
Even inside the zone, the last hydro step leaves T ~200x above the analytic
value (8.2 vs 0.036 at r = 0.145). **Not round-off:** rerunning in float64
reproduces the profile to 3 significant figures. The mechanism is the
standard high-Mach-number problem of a total-energy scheme. The wind at the
zone edge is at Mach ~220, so p / (0.5 rho v^2) ~ 2e-5, and ordinary
discretization error in the kinetic energy (MINMOD-limited reconstruction
of a diverging radial flow on a Cartesian grid) is large compared with the
thermal energy. `p = E - KE` absorbs that error as spurious heating.
So the zone removes the injection-radius part of the problem, but the
cold wind cannot survive ~4 cells of ordinary hydro between the zone edge
and the shock.

**Possible next steps (not done):** (a) a larger
`wind_zone_stagnation_fraction` (e.g. 0.8-0.9), which pushes the edge
closer to the shock and leaves fewer re-heating cells (cheap, one knob);
(b) a dual-energy / entropy-based pressure for cold supersonic cells, the
standard cure for the high-Mach problem but a solver-level change; (c) a
pressure floor tied to the analytic wind entropy outside the zone (hacky).

Scratch scripts (not committed): `cwb_wind_zone_ab.py`,
`cwb_wind_zone_x64.py`.

---

## 2026-09-23, round 20: zone fraction 0.9 at N=64 — on-axis pre-shock wind now at ~ambient T and Mach ~130-190, but only for the ~40 surface cells where the zone reaches the shock; plus a separate plain-EI mass-loss bug found

Script (committed): `cwb_analytic_wind_zone.py`, which runs plain EI and
zone fractions 0.5 and 0.9 through `run_cwb` (new
`wind_zone_stagnation_fraction` argument). Figures:
`figures/cwb_wind_zone_{axis_profile,mach_hist,slices}_64.png`.

| | Mach max | median | p90 | zone radii (star 1, star 2) |
|---|---|---|---|---|
| plain EI | 24.65 | 4.36 | 9.71 | — |
| zone f=0.5 | 34.16 | 10.38 | 24.99 | 0.155, 0.045 |
| zone f=0.9 | 186.65 | 10.69 | 24.83 | 0.279, 0.081 |

All runs finished without NaN. **At f=0.9, star 1's zone edge (x ~ 0.079)
sits right at the star-1 shock on the axis (x ~ 0.085).** The wind reaches
the shock at T ~ 14 (ambient 17.45) with a local Mach of ~170, and the
histogram gets a new cluster of ~40 surface cells at Mach 130-190. This is
the expected ~130-150 value for the first time. **The median does not move,
though.** The zone is a sphere, but the bow shock curves away from star 1,
so off-axis there is still a gap of hydro cells between the zone edge and
the shock. The cold wind is re-heated in that gap (round 19 mechanism),
and most of the arc stays at Mach ~10-30. **Caveat:** on the axis, less than
one hydro cell separates the zone edge from the shock, so the zone may now
be pinning the apex shock position rather than just feeding it. The apex
shock moves from x ~ 0.06 (plain) to ~ 0.07 (f=0.5) to ~ 0.085 (f=0.9).
Going above 0.9 would put the zone on top of the shock.

**Separate bug found in the default `_wind_ei3D` (not fixed; default path
left untouched on purpose):** the source term is normalised by
`injection_volume = 4/3 pi (num_injection_cells dx)^3`, but it is only
applied to cells with `dist <= injection_radius - dx/2`. At N=64
(`num_injection_cells=2`) that mask holds 12 cells, i.e. **36% of the
normalisation volume**. So plain EI injects only ~36% of the nominal
`Mdot` and kinetic luminosity. That matches the measured axis mass flux:
`rho v` in the plain run is ~0.34x the analytic-zone value at x = -0.1.
Consequences:
- plain-EI winds are ~3x weaker than configured in every past CWB round;
- the plain-vs-zone comparison is not momentum-matched, since the zone
  uses the full nominal `Mdot`.

The FD-path `_wind_ei3D_source` already normalises by the actual mask
volume and is not affected. The fix would be to use `sum(mask) * dx^3` as
the normalisation volume, but it changes the default EI behaviour, so it
needs your decision.

---

## Suggested order for next session

1. **Analytic wind zone follow-up (round 19).** First try the cheap knob:
   `wind_zone_stagnation_fraction` 0.8-0.9. If the zone-edge re-heating
   still dominates, decide whether a dual-energy / entropy pressure fix
   is worth doing at the solver level.
2. **Only if an orbiting binary is required:** reopen item 4 for
   `nbody=True`, starting from round 8's hypotheses (moving source changes
   which cells are freshly injected every step; orbital velocity is a
   sizeable fraction of `v_inf`; interaction with the open-boundary corner).
3. **Figures:** regenerate `cwb_shocked_cells_{2d,3d}_256.png` with the
   current (stationary, round-16-fixed) setup, or replace them with the
   `_stat_` variants; drop the `_256a.png` references.
4. **Bookkeeping:** formally close 1a (superseded by `dd7a5ae`) and retitle
   1b / 4 headings to match the status block at the top.
