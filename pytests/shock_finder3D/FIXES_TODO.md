# CWB / shock-finder fixes — carryover to next session

Origin: investigating why `cwb_shocked_cells_2d_256.png`'s Mach numbers looked
qualitatively right but capped at max ~7, when the CWB wind-termination shock
is physically ~Mach 100. Turned into two separate threads: (1) real bugs in
the Pfrommer shock-finder formalism, and (2) the CWB test's solver
configuration was silently unstable, and separately (3) a real, now-fixed
dimensional bug in the 3D stellar-wind injection module. None of that
unblocks (4): **the CWB sim does not run clean at any tested N (64/128/256)**
— a genuine numerical-stability problem, pre-existing (not caused by item
3's fix — see item 4), and resistant to five different well-motivated fixes
tried across two sessions. This is the priority blocker: nothing past it can
be properly re-validated until the sim runs cleanly.

**Repo state right now:** `_shock_surface.py` (items 1c/1d only — item 1a was
reverted, see below), `stellar_wind.py` (item 3) and `_cwb_setup.py` all have
uncommitted changes. `_shock_mach.py`, `_shock_zones.py` and
`pfrommer_shock_finder.py` are back to `HEAD` (item 1a's fix removed — it
turned out to break a real running simulation, see item 1a below; this was
found while working on an unrelated module, the CR-grey ladder, which shares
`find_shocks_pfrommer`). The tracked `cwb_shocked_cells_2d_256.png` /
`_3d_256.png` still show the *broken* (all-NaN → empty) result from an early
mid-investigation state — unchanged, since the sim still doesn't run clean
(item 4). `..._256a.png` are the pre-change (old `HLL`/first-order,
Mach-capped-at-~7) reference plots, kept for comparison; delete once no
longer needed.

**Running things:** `source ~/venv_grav_source/bin/activate`, then pin a GPU
manually and no-op `autocvd` (the repo's own `autocvd(num_gpus=1)` hangs
forever on this shared cluster) — see `verify_wind_fix.py` / `debug_nan2.py`
/ `cwb_param.py` / `run_injection_sweep.py` under the scratch dir for the
pattern; `cwb_param.py` in particular is a parametrized rebuild of
`_cwb_setup.py` (code-unit scale, `num_injection_cells`, and N-body all
exposed as arguments) worth reusing rather than rebuilding for the next round
of experiments. GPUs 2/3 were free and used so far; check `nvidia-smi` fresh
each session.

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

## Suggested order for next session

1. Investigate item 4 — nothing else can be tested until the sim runs.
   Formula choice, code units, one limiter epsilon, injection-region size,
   and N-body are all ruled out (two sessions' worth of testing); start with
   #3 under item 4 (instrument the actual failing step directly) rather than
   another blind hypothesis-and-rerun cycle, then sub-cycling if that
   doesn't pin it down.
2. Once N=128 (fast) then N=256 run clean with `riemann_solver=HLLC`,
   `first_order_fallback=False`: re-check the CWB Mach numbers. If item 1b's
   zone-width bottleneck is still capping Mach well below ~100, do the
   decoupled-plateau-walk fix described there.
3. Regenerate `cwb_shocked_cells_2d_256.png` / `_3d_256.png` for real, drop
   the `_256a.png` reference copies once no longer needed.
