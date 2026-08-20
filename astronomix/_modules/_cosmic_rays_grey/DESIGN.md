# Grey two-moment cosmic rays -- DESIGN

Phase A of `pytests/shock_finder3D/astronomix_CR_implementation_plan.md`. Draft for review
before the physics bodies get filled in (the plan's own kickoff prompt: "stop and show it to
me before implementing").

This module (`astronomix/_modules/_cosmic_rays_grey/`) is a **new, separate** model from the
existing `astronomix/_modules/_cosmic_rays/` (single advected scalar `n_cr`, polytropic closure
`P_cr = n_cr**gamma_cr`, folded into the total gas pressure/energy, with DSA shock injection
already wired up). The two coexist; consolidating them is a later decision, not Phase A.

## State variables

- `e_cr` -- grey (energy-integrated) cosmic-ray energy density. One scalar row,
  `registered_variables.cosmic_ray_e_index`.
- `F_cr` -- cosmic-ray flux, one component per spatial dimension (mirrors `velocity_index`'s
  `StaticIntVector` pattern), `registered_variables.cosmic_ray_flux_index`.

Both are tracked as **independent** state, separate from the gas thermal energy stored at
`pressure_index`/`energy_index`. This is the key structural difference from the old `n_cr`
model: there, CR pressure was recovered by subtracting it out of a *shared* total-pressure slot
(`_fluid_equations/_equations.py`'s `*_with_crs` functions). Here, gas primitive/conserved
conversion (`primitive_state_from_conserved`, `conserved_state_from_primitive`) needs **no**
CR-aware branch at all -- `e_cr`/`F_cr` behave like any other independent registered row for
that conversion (primitive == conserved, like `wind_density`). All CR <-> gas coupling instead
happens through explicit feedback source terms (see below), which is also what makes the
`-grad(P_cr)`/adiabatic-work/streaming-heating terms possible to reason about and test in
isolation from the gas EOS.

## Equations

- Closure: `gamma_cr = 4/3`, `P_cr = (gamma_cr - 1) * e_cr`
  (`cr_grey_fluid_equations.pressure_from_e_cr`).
- Transport: two-moment hyperbolic system (Jiang & Oh 2018; Thomas & Pfrommer 2019) --
  `e_cr` advected by `F_cr` (not by the gas velocity the way a passive scalar would be), `F_cr`
  relaxes toward the local CR pressure gradient at a *reduced* free-streaming speed
  `v_cr,red` (`CosmicRayGreyParams.reduced_streaming_speed`, a tunable accuracy/cost knob --
  see Open questions). This is what makes the system explicit/hyperbolic (no stiff parabolic
  solve, no global linear system) and keeps it smooth/differentiable.
- Feedback: `-grad(P_cr)` in gas momentum; `-P_cr * div(v)` adiabatic work on `e_cr`; streaming
  heating deposits CR energy into gas thermal. Positivity floor on `e_cr`
  (`CosmicRayGreyParams.minimum_e_cr`). CR fast speed enters the adaptive CFL as
  `max(u_gas + c_gas, v_cr_fast)`, not folded into a single effective sound speed (unlike the
  old model's `speed_of_sound_crs`) -- the two subsystems have genuinely different
  characteristics.
- Anisotropic transport: when `CosmicRayGreyConfig.anisotropic_transport`, project along the
  local B direction with a monotonicity-safe (Sharma & Hammett 2007) operator; isotropic
  otherwise.
- Streaming sign: regularized with `tanh` (`cr_grey_transport.regularized_streaming_sign`), not
  `min`/`max`/`sign`, to keep the adjoint finite (differentiability plan below).

## Operator-split ordering (traced from `astronomix/time_stepping/time_integration.py`)

Per physical step, both solver modes:

1. dt estimate (`_timestep_estimator.py`, FV: `_cfl_time_step`/`get_wave_speeds`; FD:
   `_cfl_time_step_fd*`) -- CR-grey fast speed folds in here (FV only for now).
2. N-body advance (independent of hydro).
3. `_iteration_level_updates` (discrete update on the primitive state, both modes) -- `e_cr`
   positivity floor goes here, alongside the existing density/pressure hard floor.
4. Hydro evolve:
   - **FV** (`_finite_volume/_state_evolution/evolve_state.py`): reconstruction -> Riemann
     solve -> flux differencing -> conserved update -> primitive recovery, per axis
     (Strang-split) or unsplit (RK2-SSP). Self-gravity enters as an operator-split source via
     `_time_integrator_sources`; CR-grey feedback sources are added the same way.
   - **FD** (`_finite_difference/_state_evolution/_evolve_state.py`): conserved state -> RK
     stage (SSPRK4/LSRK4) -> WENO interface flux -> divergence -> `_time_integrator_sources`
     added directly into *every* RK stage's RHS -> primitive recovery. CR-grey feedback sources
     would enter here identically to FD wind/cooling/viscosity/conduction -- but CR-grey
     **transport** (the flux/eigenspeed part) cannot, yet: see FD limitation below.

## Resolved: `params` threaded through the FV hot path

`_euler_flux`, `_hll_solver`/`_hllc_solver`/`_am_hllc_solver`, `_lax_friedrichs_solver`,
`_riemann_solver`, `_reconstruct_at_interface_split` and `get_wave_speeds` all now receive
`SimulationParams` (inserted immediately after `config` in each signature, mirroring the
pre-existing `_reconstruct_at_interface_unsplit` pattern), and every call site up through
`evolve_state.py`/`_pallas_evolve.py`/the FV `_timestep_estimator.py`/FD's `_weno.py` was updated
to pass it. `cr_grey_transport.py`'s `grey_cr_flux_terms`/`grey_cr_fast_speed`/
`anisotropic_flux_projection` now take `params` too and read `gamma_cr`/`reduced_streaming_speed`
from real `CosmicRayGreyParams` (`params.cosmic_ray_grey_params`) -- the module-level constants
this section used to describe are gone. `cr_grey_sources.py` already took full `params` from the
original scaffold and needed no change. See `PROGRESS.md` for what was verified.

While tracing this, found and fixed an unrelated pre-existing gap in the same hot path:
`_pallas_evolve.py`'s `_fv_pallas_evolve_supported` only excluded the old `_cosmic_rays` module,
not `grey_cosmic_rays` -- since the default backend (`OPTIMAL_BACKEND`) resolves to `PALLAS` on
compute-capability >= 8.0 GPUs, a CR-grey run there would have silently taken the fused Pallas
kernel (no `e_cr`/`F_cr` awareness) instead of the native path this design assumes. Now gated.

`_lax_friedrichs_solver`'s dissipation coefficient is still gas-only (not widened by the CR-grey
fast speed, unlike the HLL-family solvers and `get_wave_speeds`) -- a known, deliberately
un-fixed gap since `LAX_FRIEDRICHS` isn't the default `riemann_solver`; see `PROGRESS.md`.

Two more gaps surfaced while getting ladder item 1 to actually run (both pre-existing, both now
fixed -- see `PROGRESS.md` for the full writeup): FV's operator-split source application was
gated on `config.gravity_config.gravity` alone, so the CR-grey feedback calls wired into
`_time_integrator_sources` were dead code without gravity also on; and the default
`config.split == UNSPLIT` CFL branch had no CR-grey (or old-model) wave-speed awareness at all,
so `reduced_streaming_speed` values above the gas sound speed silently violated CFL and blew up.

## Resolved: transport flux and feedback-source formulas (ladder item 1)

`grey_cr_flux_terms`/`grey_cr_fast_speed` implement the isotropic-closure two-moment system
(Jiang & Oh 2018): `d(e_cr)/dt + d(u_n e_cr + F_cr)/dx = 0`,
`d(F_cr)/dt + d(u_n F_cr + v_red^2 P_cr)/dx = 0`, with `P_cr = (gamma_cr - 1) e_cr` the same
pressure used for the gas coupling below -- `grey_cr_fast_speed` returns the un-scaled
`reduced_streaming_speed` as a conservative CFL/Riemann bound *relative to the local advecting
velocity* `u_n` (the true eigenvalue relative to `u_n` is the smaller
`reduced_streaming_speed * sqrt(gamma_cr - 1)`); every call site adds `u_n` itself around this
(see hll.py/reconstruction.py/`_timestep_estimator.py`'s shared `|u| + max(c_gas, v_cr_fast)`
pattern), so `grey_cr_fast_speed` itself must not. `cr_pressure_gradient_source` adds
`-grad(P_cr)` to momentum *and* the matching `-v.grad(P_cr)` work-rate to gas total energy
(needed for energy conservation, not explicit in this doc's original one-line bullet);
`cr_adiabatic_work_source` adds `-P_cr div(v)` to `e_cr`. Together with the `u_n e_cr` advective
flux term above (which supplies the "volume dilution" piece, `-e_cr div(v)`, that a conservative
advective flux gives for free), these satisfy the local conservation law
`d(E_gas + e_cr)/dt + div(flux terms) = 0` exactly (the product-rule identity
`div(P_cr v) = v.grad(P_cr) + P_cr div(v)`), verified numerically to ~7e-5 relative, **and** the
adiabatic-compression invariant `e_cr ~ rho^gamma_cr` (plan Sec. 4, test 2; see `PROGRESS.md`).

**Correction (ladder item 2 session):** `grey_cr_flux_terms` originally had *no* `u_n * e_cr` /
`u_n * F_cr` advective piece at all -- `e_cr`/`F_cr` moved only via the relative flux terms
above, deliberately not by the gas velocity (see the now-superseded note in `cr_advection.py`'s
docstring, "e_cr advected by F_cr, not by the gas velocity"). Working through the
adiabatic-compression ladder test (a homologous squeeze, exact solution of the Euler equations)
by hand showed this was wrong on two counts: (1) it makes `cr_adiabatic_work_source`'s
`-P_cr div(v)` alone integrate to `e_cr ~ rho^(gamma_cr - 1)`, not the plan's `rho^gamma_cr`; (2)
more importantly, with `div(v) = 0` (e.g. inside a steady wind) the adiabatic-work source term is
identically zero and `F_cr`'s own dynamics only react to `P_cr` gradients, so a non-uniform CR
population in a uniformly-flowing wind would never be swept downstream at all -- silently
breaking the Phase C wind/SNe emission target. Fixed by adding the generic `u_n * q`
bulk-advection piece each row already gets from `_euler_flux` for every *other* registered
row, but which the CR rows previously discarded (`_euler_flux` `.set()`s them, not `.add()`s).
`cr_grey_sources.py`'s source terms were re-verified against this corrected flux and did not need
to change -- see `PROGRESS.md`.

## BC handling per scheme

- FV: inherits whatever `config.boundary_settings` already provides (open/reflective/periodic)
  for any registered row, including `e_cr`/`F_cr` -- no CR-specific BC code needed.
- FD: periodic-only for CR-grey (matches the plan's own staging -- FD doesn't have open BCs at
  all yet, CR or otherwise). No FD-specific BC work in Phase A.

## Differentiability plan

- No `e_cr`/`F_cr` floor via `jnp.where`/hard branching where avoidable; use smooth
  regularization (`tanh` for the streaming sign, as above).
- `NotImplementedError` stubs are the only "hard" thing about this scaffold -- once filled in,
  every transport/source function should be `jax.jit`-compatible and differentiable end-to-end,
  matching the pattern already used throughout the module (see `cr_grey_transport.py`,
  `cr_grey_sources.py`).
- Test 16 in the plan's ladder (finite-difference vs. autodiff gradient check) is scaffolded as
  `pytests/cosmic_rays_grey/cr_gradient_check.py`, gated to run once the transport stubs are
  filled in.

## FD limitation (why FD is registry/config-only in this scaffold)

The FD scheme's WENO reconstruction (`_finite_difference/_interface_fluxes/_weno.py`) does
characteristic decomposition against a **hardcoded eigensystem** (`_eigen_hydro.py`/
`_eigen_mhd.py`, fixed `N_char` = 5 (hydro) / 7 (MHD)) with no hook for extra registered scalars
-- confirmed by grep: FD's CFL estimator has zero CR references today, and both existing extra
FV-only fields (`wind_density`, `cosmic_ray_n`) are explicitly commented
"CURRENTLY ONLY IMPLEMENTED FOR FINITE VOLUME MODE" in `registered_variables.py`. Extending the
WENO eigensystem to carry `e_cr`/`F_cr` (or bolting on a separate operator-split flux/RHS path)
is real, separate follow-up work -- not attempted in this scaffold. `registered_variables.py`'s
FD branch is left with a comment pointing back here; `CosmicRayGreyConfig`/`Params` still apply
to FD (e.g. for a future periodic-only FD source-term path), but no FD state-array slots are
allocated yet. This matches the plan's own staging (Sec. 1: FD is CR-secondary until it gets
open BCs).

## Open questions (plan Sec. 6, restated for this module)

- **Reduced free-streaming speed**: `CosmicRayGreyParams.reduced_streaming_speed` currently
  defaults to a placeholder (`1.0`); pick the largest value that leaves wind/emission properties
  unchanged via a convergence study before Phase B/C.
- **Electron treatment**: fixed `K_ep` post-processing vs. separately-evolved grey electrons --
  decide before Phase C emission work; out of scope here.
- **FD open boundaries**: implement now (unblocks wind/SNe on FD) or defer -- see FD limitation
  above; currently deferred.
- **Spectral-interface contract** (Girichidis collaboration, Phase E): agree the `spectrum`
  object API early so Phase E is a swap, not a rewrite; not touched in Phase A.
- **Consolidating with the old `_cosmic_rays` model**: not decided. Candidates once both exist:
  keep both (different physics fidelity/cost tradeoff), or retire the old model once Phase B's
  DSA injection is re-targeted at `e_cr`/`F_cr`.
