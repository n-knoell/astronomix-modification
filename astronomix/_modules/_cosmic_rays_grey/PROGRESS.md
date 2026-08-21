# Grey two-moment cosmic rays -- PROGRESS

Status tracker for `pytests/shock_finder3D/astronomix_CR_implementation_plan.md`. Read this
first when picking the work back up; `DESIGN.md` in this directory is the target design, this
file is what's actually done against it.

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
8. Continue test-first, in plan-ladder order: `cr_streaming_1d.py` (item 5, needs
   `cr_streaming_heating_source` implemented, currently a stub) -> `cr_shock_tube.py` (item 6, the
   two-fluid CR-modified shock tube -- the real test of the momentum/energy feedback coupling).
   **This is where the next session should pick up.** Streaming's own relaxation-like structure
   (CR bulk transport at the streaming speed along B) may interact with or partially duplicate
   `cr_flux_relaxation_source`'s new scattering term -- worth checking Jiang & Oh (2018)'s
   combined streaming+scattering formulation before implementing item 5, rather than assuming
   they're fully independent.
9. FD transport is out of scope until the WENO-eigensystem extension (see DESIGN.md's "FD
   limitation") gets separately scoped -- don't attempt it inside a ladder-item pass.
10. Once Phase A's tests pass for real, revisit `DESIGN.md`'s open question on consolidating with
    the older `astronomix/_modules/_cosmic_rays/` (`n_cr`) model before starting Phase B
    (shock-finder DSA injection needs to target one CR model or the other).
11. `evolve_state.py`'s `_split_gas_and_magnetic_state`/`_join_gas_and_magnetic_state` fix (general,
    not CR-specific) is worth a heads-up to whoever owns the MHD module / other in-flight MHD work,
    since it changes behavior (from silently wrong to correct) for any future combination of `mhd`
    with `wind_density` or the old `cosmic_ray_n` model too, not just grey CR.

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
