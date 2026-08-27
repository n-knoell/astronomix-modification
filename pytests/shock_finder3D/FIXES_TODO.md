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
