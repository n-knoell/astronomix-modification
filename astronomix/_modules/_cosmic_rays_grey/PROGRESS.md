# Grey two-moment cosmic rays -- PROGRESS

Status tracker for `pytests/shock_finder3D/astronomix_CR_implementation_plan.md`. Read this
first when picking the work back up; `DESIGN.md` in this directory is the target design, this
file is what's actually done against it.

## Where things stand (2026-08-19)

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
4. Continue test-first, in plan-ladder order: `cr_adiabatic_compression.py` (item 2, `e_cr ~
   rho^gamma_cr` under uniform compression -- exercises `cr_adiabatic_work_source` under nonzero
   `div(v)`, already implemented) -> `cr_anisotropic_diffusion_oblique.py` (item 3, needs
   `anisotropic_flux_projection` implemented, currently a stub) ->
   `cr_isotropic_diffusion_convergence.py` (item 4) -> `cr_streaming_1d.py` (item 5, needs
   `cr_streaming_heating_source` implemented, currently a stub) -> `cr_shock_tube.py` (item 6,
   the two-fluid CR-modified shock tube -- the real test of the momentum/energy feedback coupling
   flagged above). Run `cr_gradient_check.py` (item 16, FD-vs-AD) alongside each as it becomes
   exercisable. **This is where the next session should pick up.**
5. FD transport is out of scope until the WENO-eigensystem extension (see DESIGN.md's "FD
   limitation") gets separately scoped -- don't attempt it inside a ladder-item pass.
6. Once Phase A's tests pass for real, revisit `DESIGN.md`'s open question on consolidating with
   the older `astronomix/_modules/_cosmic_rays/` (`n_cr`) model before starting Phase B
   (shock-finder DSA injection needs to target one CR model or the other).

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
