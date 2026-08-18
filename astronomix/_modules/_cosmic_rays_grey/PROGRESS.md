# Grey two-moment cosmic rays -- PROGRESS

Status tracker for `pytests/shock_finder3D/astronomix_CR_implementation_plan.md`. Read this
first when picking the work back up; `DESIGN.md` in this directory is the target design, this
file is what's actually done against it.

## Where things stand (2026-08-18)

**Branch:** `feature/cosmic-rays-grey` (off `shock-finder`, which is off `main`).
**Latest commit:** `fb1ad6a` -- "scaffold Phase A: grey two-moment cosmic-ray module".

**Phase A is scaffolded, not implemented.** Branch, module, DESIGN.md, config/params, and every
call site the transport/feedback physics needs are wired up and gated behind
`cosmic_ray_grey_config.grey_cosmic_rays` (default `False`, so nothing existing changed). Every
actual physics function still `raise NotImplementedError`. Nobody has reviewed `DESIGN.md`'s
physics content yet -- do that before or alongside filling in the stubs, per the plan's own
"stop and show it to me before implementing" step.

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
  (`_euler_flux`), `hll.py` (`_hll_solver`/`_hllc_solver`), `reconstruction.py`
  (`_reconstruct_at_interface_split`), `_timestep_estimator.py` (`get_wave_speeds`),
  `_iteration_level_updates.py` (`e_cr` positivity floor -- this one branch is real code, not a
  stub), `_time_integrator_sources.py` (feedback source calls).
- `pytests/cosmic_rays_grey/`: 7 skeleton scripts (one per Phase A ladder item + the
  differentiability check), each `def test_x(): raise NotImplementedError(...)` +
  `if __name__ == "__main__":`. Matches the repo's plain-script test convention (no `import
  pytest` -- see Environment notes below for why that matters).

## Verified (this session)

- `import astronomix` -- clean.
- `pytests/hydrodynamics/shock_tube1D.py::test_shock_tube1D` -- still passes with CR off
  (confirms the new gated branches are true no-ops).
- A minimal 1D config with `grey_cosmic_rays=True` reaches
  `grey_cr_fast_speed`'s `NotImplementedError` end-to-end (via `get_wave_speeds` in the CFL
  estimator, before ever reaching the Riemann solver) -- proves the wiring is real, not just
  that the stub exists.
- All 7 `pytests/cosmic_rays_grey/*.py` skeletons import and run to their intended
  `NotImplementedError`.
- NOT verified: the other stubs (`grey_cr_flux_terms`, `anisotropic_flux_projection`,
  `cr_pressure_gradient_source`, `cr_adiabatic_work_source`, `cr_streaming_heating_source`) --
  the CFL check above raises before the run ever reaches them. Worth a similar targeted check
  once each is filled in.

## Next steps, in order

1. Review `DESIGN.md`'s physics content (equations, closure, operator-split ordering) --
   nothing below should start until that's settled, since it's still a first draft.
2. Resolve the "known scaffold gap" in `DESIGN.md`: `params` isn't threaded through
   `_euler_flux`/`_hll_solver`/`_hllc_solver`/`_reconstruct_at_interface_split`/
   `get_wave_speeds`, so `cr_grey_transport.py` currently reads `gamma_cr`/
   `reduced_streaming_speed` off module-level constants instead of `CosmicRayGreyParams`. Decide
   whether to do that threading now (real signature changes across those 5 functions) or keep
   deferring it.
3. Implement test-first, in plan-ladder order (`pytests/cosmic_rays_grey/`): `cr_advection.py` ->
   `cr_adiabatic_compression.py` -> `cr_anisotropic_diffusion_oblique.py` ->
   `cr_isotropic_diffusion_convergence.py` -> `cr_streaming_1d.py` -> `cr_shock_tube.py`, with
   `cr_gradient_check.py` run continuously alongside each as it becomes exercisable.
4. FD transport is out of scope until the WENO-eigensystem extension (see DESIGN.md's "FD
   limitation") gets separately scoped -- don't attempt it inside a ladder-item pass.
5. Once Phase A's tests pass for real, revisit `DESIGN.md`'s open question on consolidating with
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
