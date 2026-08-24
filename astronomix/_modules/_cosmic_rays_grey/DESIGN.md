# Grey two-moment cosmic rays -- DESIGN

Phase A of `pytests/shock_finder3D/astronomix_CR_implementation_plan.md`. Draft for review
before the physics bodies get filled in (the plan's own kickoff prompt: "stop and show it to
me before implementing").

This module (`astronomix/_modules/_cosmic_rays_grey/`) was originally built as a **new,
separate** model alongside `astronomix/_modules/_cosmic_rays/` (single advected scalar `n_cr`,
polytropic closure `P_cr = n_cr**gamma_cr`, folded into the total gas pressure/energy). As of
2026-08-24, that older model has been **retired** -- see "Consolidating with the old
`_cosmic_rays` model" below -- and this grey two-moment model is now the only cosmic-ray model
in astronomix.

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
- Anisotropic transport: when `CosmicRayGreyConfig.anisotropic_transport` (requires `config.mhd`),
  `F_cr` is projected onto the local B direction (`cr_grey_transport.anisotropic_flux_projection`,
  implemented -- ladder item 3) once per full step, in `_iteration_level_updates` -- **not**
  inside `grey_cr_flux_terms`, because that function runs inside the FV MHD Strang split's
  gas-only Riemann solve, where the magnetic-field rows are structurally absent from the state
  array for that half-step (see "Resolved: anisotropic transport" below). A direct projection,
  not the full Sharma & Hammett (2007) monotonicity-preserving flux limiter -- see that function's
  docstring for the scope reasoning (the plan itself expects less limiting machinery to be needed
  for a hyperbolic, Riemann-solved two-moment system than for the diffusive/parabolic case S&H
  address). Isotropic (unprojected `F_cr`) otherwise, the default.
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

## Resolved: anisotropic transport, and a general FV+MHD bug it exposed (ladder item 3)

`anisotropic_flux_projection` (`cr_grey_transport.py`) is implemented: given the local B and
`F_cr`, returns `F_cr_parallel = (F_cr . b_hat) * b_hat`, `b_hat = B / sqrt(|B|^2 +
b_field_floor^2)` (a smooth, always-defined floor -- `CosmicRayGreyParams.b_field_floor` -- same
"prefer smooth regularization over min/max" philosophy as `regularized_streaming_sign`). It is
called from `astronomix._modules._iteration_level_updates` once per full step, before the hydro
update -- overwriting the `F_cr` state with its B-projected value -- **not** from within
`grey_cr_flux_terms`, which cannot reach B (see below). `F_cr`'s own pressure-driving term
(`v_red^2 * P_cr`, in `grey_cr_flux_terms`) stays isotropic, un-projected, each Cartesian
component driven by its own axis's gradient -- so within a single step the raw `F_cr` state can
pick up a small (order `dt`), non-accumulating perpendicular-to-B component before the *next*
step's projection removes it again. Verified (`PROGRESS.md`): a localized `e_cr` bump on a
30-degree oblique B (2D, MHD) shows a clean expanding ring for the isotropic closure vs. two
pulses moving along +/-B for the anisotropic one; perpendicular/parallel second-moment growth
ratio ~1.0 (isotropic) vs. ~0.03 (anisotropic).

**Why not inside `grey_cr_flux_terms`:** that function runs inside the FV MHD Strang split's
*gas-only* Riemann solve (`evolve_state._evolve_state_fv` splits `primitive_state` into a gas
sub-array and a magnetic sub-array for each half-step, evolves them separately, then rejoins) --
the magnetic-field rows are not part of the state array at that call site at all. Confirmed by
hitting `AttributeError: 'int' object has no attribute 'x'` the first time an anisotropic run
actually reached that code path (`registered_variables_gas.magnetic_index == -1` there).

**General (non-CR) bug this exposed and fixed:** `_evolve_state_fv` performed that gas/magnetic
split by hardcoding `primitive_state[-3:, ...]` as "the magnetic field" -- correct only when
nothing is registered *after* `magnetic_index`. `registered_variables.py`'s FV branch allocates
`magnetic_index` *before* `cosmic_ray_e_index`/`cosmic_ray_flux_index`, so with both `mhd` and
`grey_cosmic_rays` active the slice silently grabbed `(B_z, e_cr, F_cr_x)` as "the magnetic
field" and mislabelled real `B_z` as gas -- CR transport went completely inert under MHD as a
result (not specific to anisotropic transport -- the *isotropic* closure was equally broken).
This is a general bug (would equally affect `wind_density` or the old `cosmic_ray_n` model
combined with MHD), not something to fix inside this module. Fixed in
`astronomix/_finite_volume/_state_evolution/evolve_state.py` via
`_split_gas_and_magnetic_state`/`_join_gas_and_magnetic_state`, which locate the magnetic rows by
their actual `registered_variables.magnetic_index` positions and re-index every other registry
field around them, instead of assuming a fixed trailing position. Verified to be a no-op for
plain MHD (`pytests/mhd/alfven_wave3D.py` unchanged) and for non-MHD configs (`shock_tube1D.py`
unchanged). User confirmed this fix direction explicitly (over reordering the registry allocation
instead, which would have had a much wider blast radius on existing MHD index values) when asked
-- see `PROGRESS.md`.

## Resolved: F_cr relaxation term and the diffusion limit (ladder item 4)

As implemented for ladder items 1-3, `grey_cr_flux_terms` is a **purely
hyperbolic, undamped** two-moment system -- confirmed by ladder item 3's own
observation that a localized `e_cr` bump at rest propagates rigidly as a
wave, it does not spread. There is therefore no diffusion limit to converge
to as-is, contradicting this doc's closure section ("`F_cr` relaxes toward
the local CR pressure gradient") and the plan's "recovers anisotropic
diffusion + streaming in the appropriate limit". Added
`cr_grey_sources.cr_flux_relaxation_source` (Jiang & Oh 2018's scattering
term): `d(F_cr)/dt |_relax = -nu * F_cr`, `nu = reduced_streaming_speed^2 /
diffusion_coefficient` (`diffusion_coefficient`, a new `CosmicRayGreyParams`
field, is the physical CR diffusivity `kappa`). At a quasi-steady balance
against `grey_cr_flux_terms`'s pressure-driving flux term, this relaxes
`F_cr ~= -kappa * grad(P_cr)`, i.e. Fick's law with diffusion coefficient
`kappa * (gamma_cr - 1)` for `e_cr` itself.

Applied as a plain additive rate via the same explicit `source_term * dt`
composition as every other CR-grey source (no implicit/exact-exponential
treatment) -- gated behind a new `CosmicRayGreyConfig.diffusive_relaxation`
flag, **off by default**, so ladder items 1-3's already-verified undamped-wave
behavior is unchanged unless a config opts in. This reintroduces a genuine
parabolic-like CFL constraint (forward-Euler stability of `dF/dt = -nu*F`
needs `dt < 2/nu`), handled in `_cfl_time_step`
(`_finite_volume/_timestep_estimation/_timestep_estimator.py`) the same way
the existing `config.diffusion` viscous `dt_visc` constraint is -- a
`jnp.minimum(dt, C_CFL / nu)` branch gated on `diffusive_relaxation`.

Verified (`cr_isotropic_diffusion_convergence.py`): a narrow `e_cr` Gaussian
on a static uniform gas, run to a fixed `t_end`, matches the analytic 1D
diffusion Green's function with L2 relative error shrinking monotonically
(0.0089 -> 0.0010 across N = 128 -> 1024) and total `e_cr` conserved to 6
significant figures at every resolution; convergence order ~1.0-1.1
(capped near first order by the sources' explicit-Euler operator splitting,
not by spatial truncation). See `PROGRESS.md` for the parameter-choice
reasoning (`reduced_streaming_speed`, `diffusion_coefficient` picked so the
relaxation rate is deep enough in the quasi-steady regime that the
telegrapher-vs-diffusion model bias is negligible at the tested resolutions).

**A second, unrelated pre-existing bug surfaced while building the
FD-vs-AD gradient check (`cr_gradient_check.py`, plan item 16) alongside
this:** `grey_cr_fast_speed`'s `sqrt(jnp.maximum(x, 0.0))` (the CR-pressure
acoustic-mode contribution, added in ladder item 2) has an infinite gradient
exactly at `x = 0` -- which is the case in every cell of a CR-free background
(`e_cr = 0` outside a localized pulse, the setup every ladder-item test so
far deliberately uses for items 1/3/4). `sqrt(0)`'s derivative is `+inf`,
and JAX's `0 * inf = NaN` rule turned this into a `NaN` gradient through the
whole `time_integration` call the first time reverse-mode AD was actually
run through the FV/CR-grey path (no prior test in this repo differentiates
through FV -- `pytests/differentiability/sensitivity.py` only covers FD).
Fixed with a smooth floor added in quadrature under the sqrt, not a
`jnp.maximum` on the result (`CosmicRayGreyParams.cr_pressure_speed_floor`,
same "prefer smooth regularization" philosophy as `b_field_floor`):
`sqrt(max(x, 0) + speed_floor**2)`. Confirmed: `cr_gradient_check.py`'s AD
vs. central-finite-difference gradient of `sum(e_cr_final**2)` w.r.t.
`reduced_streaming_speed` now agree to ~0.1% (`rel_err ~ 9.5e-4`); NaN
before the fix, finite (and wrong) FD-only reference before the fix
confirmed it was AD-side, not a shared bug. This also required
`differentiation_mode = BACKWARDS` (the checkpointed adaptive-loop backend --
see `time_integration.py`'s dispatch) since plain reverse-mode AD does not
work through `jax.lax.while_loop`'s adaptive-dt trip count at all, forward
or CR-related.

## Resolved: streaming transport and streaming heating (ladder item 5)

`streaming_flux_target` (`cr_grey_transport.py`) and `cr_streaming_heating_source`
(`cr_grey_sources.py`) are implemented (both were `NotImplementedError` stubs).
Physical picture (Wiener et al. 2017; the "streaming-dominated,
always-at-equilibrium" limit): self-confined CRs stream down their own
pressure gradient at the reduced free-streaming speed, so `F_cr` is pinned
per axis to `F_cr,axis = -sign(dP_cr/dx_axis) * reduced_streaming_speed *
e_cr` (regularized sign via `regularized_streaming_sign`, already
implemented). Applied once per full step in `_iteration_level_updates` as a
discrete correction that **overwrites** `F_cr` -- the same "instantaneous
relaxation" pattern `anisotropic_flux_projection` (item 3) already uses, not
a new stiff relaxation-rate source term. Isotropic per-axis, **does not
require `config.mhd`** (unlike anisotropic transport) -- matches the plan's
own "1D streaming" staging for this item. When both `streaming` and
`anisotropic_transport` are enabled, `_iteration_level_updates` applies the
streaming target first and the B-projection second; this combination is a
documented, plausible, but **not separately verified** approximation (no
ladder item tests it).

The complementary, genuinely non-conservative loss (streaming does work
against the pressure gradient, converting `e_cr` into gas heat) is
`Gamma = reduced_streaming_speed * |dP_cr/dx|` (regularized-sign version),
subtracted in full from `e_cr` and added to gas thermal energy scaled by
`streaming_heating_efficiency` (`< 1` represents loss into channels this
grey model doesn't track, e.g. higher-frequency wave turbulence) --
distinct from, and additive to, `streaming_flux_target`'s conservative
spatial redistribution of `e_cr` (which alone moves energy around without
creating or destroying it).

Verified (`cr_streaming_1d.py`): a linear, open-boundary `e_cr` ramp
(constant-sign gradient, so `regularized_streaming_sign` stays saturated
everywhere -- see below for why a sign-changing profile was rejected)
reproduces both `F_cr ~= streaming_flux_target` (bulk rel. err ~2e-4) and
the analytic gas-heating rate `dP/dt = (gamma - 1) * efficiency * Gamma`
(bulk rel. err ~7e-5; the `(gamma - 1)` factor converts the
conserved-energy-row rate `cr_streaming_heating_source` adds into the
primitive-pressure rate actually measured).

**Rejected IC, worth remembering:** an initial calibration attempt used a
periodic cosine `e_cr` profile (sign-changing gradient, to exercise both
streaming directions in one test). `regularized_streaming_sign`'s `tanh`
transition width in *gradient* space, mapped back to *position* space near
a gradient zero-crossing, was narrower than one grid cell for that
profile's curvature at the tested resolution -- an unresolved,
effectively-discontinuous sign flip right at the pressure extrema, which
produced large (`~200%`), unphysical `F_cr` deviations from the target. Not
a code bug -- confirmed by isolating `streaming_flux_target`'s output at
`t=0` (matched the analytic formula exactly) and checking a single-timestep
run (deviations appeared immediately, from the correction itself hitting
the under-resolved transition, not from accumulated integration error).
Any future streaming IC with a sign-changing CR pressure gradient should
keep the zero-crossing's spatial width (set by
`|d(dP_cr/dx)/dx| / streaming_sign_regularization`) resolved by several grid
cells, or widen `streaming_sign_regularization` to match the grid.

## Resolved: two-fluid CR-modified shock tube (ladder item 6)

`pytests/cosmic_rays_grey/cr_shock_tube.py` passes against a new
semi-analytic two-fluid Riemann solver
(`astronomix/test_setups/reference_solutions/pfrommer_riemann_solver.py`,
Pfrommer, Enßlin & Jubelgas 2006) built for this item -- no existing code
covered a composite (gas + CR, two different adiabatic indices) equation of
state, so the repo's existing single-gamma exact Riemann solver
(`riemann_solver.py`) doesn't apply. Implemented in plain numpy/scipy
(root-finding + quadrature), not JAX, since it's a one-shot reference
generator, not part of the simulation hot path.

**Key test-design choice: `reduced_streaming_speed = 0`.** Pfrommer's
solution assumes CRs are tightly coupled to the gas (advected with it,
compressing adiabatically even through a shock, no independent flux).
Setting `reduced_streaming_speed = 0` makes this *exact*, not approximate:
`grey_cr_flux_terms`'s `F_cr` equation collapses to homogeneous advection of
zero initial data, so `F_cr` stays exactly `0` for the whole run (verified:
`max|F_cr| = 0.0`), leaving `e_cr` transported purely by bulk advection plus
the (v_red-independent) momentum/energy feedback sources -- precisely
Pfrommer's assumption, with no free "how small is small enough" tuning
parameter. This is a much cleaner test design than picking some small but
nonzero `v_red` and hoping the residual error stays below tolerance.

**Validation of the new solver** (ad hoc scripts, not committed): with CR
pressure zeroed on both sides, it reduces to the plain single-gamma problem
and was checked against `riemann_solver._exact_riemann_ideal_gas` across the
full profile for three cases (classic Sod gamma=1.4, Sod at gamma=5/3, and a
reversed-Sod case exercising the less-common left-shock branch) -- matched to
~2e-7. A second check set `gamma_cr = gamma_th` with nonzero, unequal CR
pressure per side (composite EOS degenerates to one power law in the summed
pressure) -- also matched to ~2e-7, and the per-side CR pressure fraction
stayed constant through both the shock and rarefaction as expected. Two real
sign-error bugs were caught by this cross-check during development (not
issues with the underlying method): the rarefaction-fan velocity integral
had left/right family signs swapped, and the shock-speed sampling formula
used the wrong sign convention for the signed mass flux. See
`cr_shock_tube.py`'s module docstring and the solver's own docstrings for
the corrected derivations.

**Result on the real (non-degenerate) two-fluid case** (`gamma_th=5/3`,
`gamma_cr=4/3`, 40% CR pressure fraction on both sides of an otherwise
classic Sod problem, 400 cells): mean absolute errors of ~0.001-0.003 across
density/velocity/pressure/`e_cr` (well under the `tol=1e-2` default,
comparable in magnitude to the plain hydro `shock_tube1D.py` test's own HLL/
minmod numerical-diffusion errors at similar resolution) -- no separate
literature table was reproduced (only the method was validated, via the
degenerate-limit checks above), so this is a genuine, independently-checked
comparison rather than a tuned-to-pass one.

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
- **Diffusion coefficient**: `CosmicRayGreyParams.diffusion_coefficient` (added ladder item 4)
  currently defaults to a placeholder (`1.0`) and `diffusive_relaxation` itself defaults to
  `False`; pick a physically-motivated value (and decide whether it should default on) once a
  real transport-coefficient target exists (Phase B/C), same open-question category as
  `reduced_streaming_speed` above -- the two jointly set the relaxation rate `nu`.
- **Electron treatment**: fixed `K_ep` post-processing vs. separately-evolved grey electrons --
  decide before Phase C emission work; out of scope here.
- **FD open boundaries**: implement now (unblocks wind/SNe on FD) or defer -- see FD limitation
  above; currently deferred.
- **Spectral-interface contract** (Girichidis collaboration, Phase E): agree the `spectrum`
  object API early so Phase E is a swap, not a rewrite; not touched in Phase A.
- **Consolidating with the old `_cosmic_rays` model**: resolved 2026-08-24 -- **retired**. The
  old `n_cr` model was 1D-only, had no test coverage anywhere in the repo (no pytest ever set
  `cosmic_rays=True`/`diffusive_shock_acceleration=True`), and its DSA injection
  (`inject_crs_at_strongest_shock`) targeted a legacy 1D-only shock finder
  (`astronomix/shock_finder/shock_finder.py::find_shock_zone`) that predates and is unrelated to
  the actual PR #4 finder the plan means (`astronomix/shock_finder3D/
  pfrommer_shock_finder.py::find_shocks_pfrommer`, genuinely N-D, independently tested against
  Sedov/CWB setups in `pytests/shock_finder3D/`). Keeping both would also have meant no guard
  against `cosmic_ray_n_active` and `cosmic_ray_e_active` both being on at once (double-counted
  CR pressure, since the old model folds `P_cr` into the shared `pressure_index`). Removed:
  `astronomix/_modules/_cosmic_rays/` (whole module), `astronomix/shock_finder/` (the legacy 1D
  finder, only consumer was the old model's injection code), `CosmicRayConfig`/`CosmicRayParams`
  and their fields on `SimulationConfig`/`SimulationParams`, `cosmic_ray_n_index`/
  `cosmic_ray_n_active` on `RegisteredVariables`, the `*_with_crs` branches in
  `_fluid_equations/_equations.py`/`_fluxes.py`/`total_quantities.py`, `speed_of_sound_crs` calls
  in `reconstruction.py`/`hll.py`/`_timestep_estimator.py`, the DSA-injection call in
  `_iteration_level_updates.py`, and the `cosmic_ray_pressure` param on
  `construct_primitive_state`. Phase B's DSA injection (ladder item 8) should target `e_cr`/`F_cr`
  using `find_shocks_pfrommer` fresh, not adapt the retired code.
