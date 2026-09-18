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
  implemented -- ladder item 3) once per full step, in `_iteration_level_continuous_updates` -- **not**
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
3. `_iteration_level_injections` (discrete lump-sum injections on the primitive state, both
   modes: stellar wind, SN driving, CR-DSA shock injection) -- **then, when any of those three
   is active, dt is re-estimated from the post-injection state** (`jnp.minimum` against step 1's
   estimate) before the next step, fixing the stale-dt-at-injection bug (see "Resolved:
   stale-dt-at-injection bug" below).
3b. `_iteration_level_continuous_updates` (discrete update on the primitive state, both modes) --
   cooling, the neural-net/CNN correctors, viscosity, turbulent forcing, frame tracking, CR
   streaming/anisotropic-transport corrections; `e_cr` positivity floor goes here, alongside the
   existing density/pressure hard floor.
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
called from `astronomix._modules._iteration_level_continuous_updates` once per full step, before the hydro
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
implemented). Applied once per full step in `_iteration_level_continuous_updates` as a
discrete correction that **overwrites** `F_cr` -- the same "instantaneous
relaxation" pattern `anisotropic_flux_projection` (item 3) already uses, not
a new stiff relaxation-rate source term. Isotropic per-axis, **does not
require `config.mhd`** (unlike anisotropic transport) -- matches the plan's
own "1D streaming" staging for this item. When both `streaming` and
`anisotropic_transport` are enabled, `_iteration_level_continuous_updates` applies the
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

## Resolved: DSA shock injection (ladder item 7, Phase B)

`pytests/cosmic_rays_grey/cr_sedov_taylor.py` passes. `cr_grey_injection.py`'s
`inject_crs_at_shocks` is the Phase B counterpart of the Phase A transport/
feedback functions above: it detects shocks every step with the real PR #4
finder (`astronomix.shock_finder3D.pfrommer_shock_finder.find_shocks_pfrommer`)
and diverts a `dsa_efficiency` fraction of each shock-surface cell's
Rankine-Hugoniot-predicted dissipated kinetic-energy flux from gas thermal
pressure into `e_cr`. See "Consolidating with the old `_cosmic_rays` model"
below for why this was built fresh rather than adapting the retired old
model's injection code (wrong shock finder, single-shock-only, no test
coverage).

Design choices, and why they weren't open questions worth asking about:

- **Injects at every detected shock-surface cell, not "the strongest
  shock".** The retired old model's 1D-only injection picked a single
  strongest shock because its legacy finder only supported 1D. The N-D
  `find_shocks_pfrommer` already returns a full per-cell boolean surface
  array covering every simultaneous shock in the domain -- restricting to
  one would require *adding* a reduction, not removing one, so using all
  detected cells is both the more general choice and the smaller diff.
- **Deposits directly into `e_cr` at the single-cell-thick `shock_surface`,
  not spread across the broader (3-4 cell) `shock_zones`.** The retired old
  model's own comments flagged its zone-spreading as a source of "effective
  over-injection" (`P_cr * div(v)` forces acting on CR pressure injected
  into cells that are still converging); depositing at the literal cell
  `thermal_energy_flux` is computed for avoids that failure mode entirely
  and needed no extra design decision.
- **`F_cr` is left untouched at injection** -- only `e_cr` and gas thermal
  pressure change. `F_cr` develops its own flux from the `e_cr` gradient
  injection creates via the existing (already-verified) two-moment
  transport equations; no other CR-grey source term (`cr_pressure_gradient_
  source`, `cr_adiabatic_work_source`) touches `F_cr` directly either, so
  this matches the established pattern rather than introducing a new one.
- **Placed in `_iteration_level_updates.py`'s `_iteration_level_injections`, not
  `_time_integrator_sources.py`.** DSA injection is a discrete detect-and-deposit operation (the
  shock finder is a discrete algorithm, not a smooth field), the same category as wind/SN-driving
  injection -- not a continuous RK-integrated PDE source term. Grouped with those (rather than
  with `streaming_flux_target`/`anisotropic_flux_projection`, which are also discrete
  per-step corrections but live in `_iteration_level_continuous_updates`) since, like wind/SN
  driving, it's a lump-sum addition to `e_cr` that the driver loop's post-injection dt
  re-estimate should account for -- see "Resolved: stale-dt-at-injection bug" below.
- **Cartesian, uniform-grid only** (`area / volume = 1 / grid_spacing`,
  mirroring the retired old model's identical non-spherical-branch formula).
  Matches every grey-CR ladder test's geometry to date; extend if a future
  item needs curvilinear/spherical support.

**Test design** (`cr_sedov_taylor.py`): a 3D Cartesian point explosion
(N=48, `t_end=0.07`), run at `dsa_efficiency=0` (control) and `0.1`. Checks
a *differential* energy-partition identity between the two runs (thermal+
kinetic energy the DSA run loses relative to the control run must equal the
CR energy it gained) rather than comparing to an analytic self-similar Sedov
profile -- `cr_adiabatic_compression.py` (ladder item 2) already found that
kind of global-profile comparison unreliable near open boundaries, and the
plan's own wording ("thermal/CR/kinetic energy partition") is naturally an
energy-budget check, not a spatial-profile one. Calibrated: the identity
holds to ~2.1e-5 relative error; each run's own total-energy conservation
holds independently to ~1.7e-5; `E_cr` is ~5.3% of the total energy budget.
Continuous per-step shock-finding turned out to be cheap enough (~34s
including JIT compile at N=48) that a true 3D setup was affordable without
needing a cheaper 2D substitute.

## Resolved: CR-modified shock structure (ladder item 9)

`pytests/cosmic_rays_grey/cr_modified_shock_structure.py` passes, closing out Phase A/B's core
ladder (items 1-9). As originally attempted (see "Open questions" below), a shock held exactly
stationary in the lab frame surfaced Finding 2 (sustained DSA injection at a cell that never
advects away grows `e_cr` without bound): the compressive `-P_cr * div(v)` term
(`cr_grey_sources.cr_adiabatic_work_source`) never releases the way it would for a fluid
parcel that transits a moving shock once (the finite-transit-time release ladder item 2
verified gives the correct `P_cr ~ rho^gamma_cr` invariant). This matches how DSA theory
itself frames a "steady shock": steady *in the shock's own comoving frame*, not a shock pinned
motionless in the lab frame while CR energy accumulates in the same cell forever.

**Resolved by redesigning the test around a genuinely moving shock** (confirmed with the
user): a Rankine-Hugoniot "piston" IC -- an exact steady-shock jump launched at a nonzero,
constant velocity into a quiescent ambient medium -- so each grid cell only hosts the
actively-injecting shock surface for the brief time the shock front is passing through it,
the same "always advancing into fresh gas" pattern `cr_sedov_taylor.py`/
`cr_dsa_mach_dependence.py` already rely on. Verified directly (ad hoc script, not committed):
fitted shock velocity from several snapshots matches the nominal value to <0.1%, and peak
`e_cr` saturates to a bounded value instead of growing without bound -- `test_cr_dsa_shock_jump`
turns this into a permanent regression guard (asserts the plateau, not just a single passing
run).

**A further interaction with Finding 1 surfaced while calibrating this redesign** (Finding 1
itself remains open -- see "Open questions" below, unaffected by this section). The reference
precursor ODE (`cr_precursor_ode_rhs`, in `astronomix/test_setups/reference_solutions/
cr_modified_shock_structure.py`) only holds once `F_cr`'s relaxation rate `nu =
reduced_streaming_speed^2 / diffusion_coefficient` is fast relative to the precursor's own
advective formation rate `omega ~ v_shock / precursor_width`; working through both rates shows
`nu / omega = (gamma_cr - 1) * (reduced_streaming_speed / v_shock)^2`, independent of
`diffusion_coefficient` -- so a resolvable precursor needs `reduced_streaming_speed` several
times the shock velocity *regardless* of `diffusion_coefficient`, which is almost exactly
Finding 1's own bad regime. Measured directly: raising `reduced_streaming_speed` from 4x to 8x
the shock velocity improves the precursor-ODE match but *worsens* the subshock-jump match and
further degrades the measured shock Mach; at 16x, the shock degrades below `dsa_mach_min` and
disappears. A single run cannot quantitatively validate both reference formulas at once
without first fixing Finding 1.

**Resolved by splitting into two independent tests** (confirmed with the user, same
"independent layers" pattern ladder item 8 already used), each run at the
`reduced_streaming_speed` its own reference formula actually needs:
- `test_cr_dsa_shock_jump` (Test A): `diffusive_shock_acceleration` only, `reduced_streaming_speed`
  close to the local gas sound speed (Finding-1-safe). Validates
  `modified_rankine_hugoniot_with_cr_injection` against the simulation's own converged
  post-shock state, and doubles as the Finding-2 regression guard.
- `test_cr_precursor_ode` (Test B): `diffusive_relaxation` on, `reduced_streaming_speed` well
  above the shock velocity (deep quasi-steady, accepting a Finding-1-degraded but still
  super-critical shock as the cost). Validates `cr_precursor_ode_rhs` against the simulation's
  own local precursor gradients, after confirming the two first integrals it assumes
  (`mass_flux`, `momentum_flux`) are genuinely near-constant across the sampled window.

**Also fixed a real root-finding bug** in `modified_rankine_hugoniot_with_cr_injection`
(uncommitted since the prior session, so fair game to fix directly): with nonzero
`injected_energy_flux`, the naive two-point `brentq` bracket can straddle a spurious root
near `r=1` (a vestige of the plain-gas case's exact trivial root, shifted off exactly 1 by the
injection term) and fail outright. Fixed by scanning for sign changes across `(1, r_max)` and
taking the one closest to `r_max`; re-verified the degenerate zero-injection limit and the
general nonzero-injection conservation checks (mass/momentum/total-energy flux, machine
precision) both still hold. See PROGRESS.md's 2026-09-09 entry for full calibration numbers
and tolerances.

## Resolved: wind-blown bubble with CR pressure (ladder item 10)

Test: `pytests/cosmic_rays_grey/cr_wind_bubble.py`. Three design decisions confirmed with the
user before implementing (same pattern as items 4, 6, 8, 9): 3D Cartesian geometry (not 1D
spherical); CR injection via the existing shock-DSA machinery (`inject_crs_at_shocks` at the
wind's forward shock) rather than a new dedicated wind-source CR channel; and validation against
the pre-existing (but previously untested) `Weaver` analytic solution plus an item-7-style
differential CR energy-partition check, rather than deriving a new CR-modified analytic
wind-bubble solution from scratch.

No new simulation code was needed -- this item is a new test of existing machinery
(`_wind_ei3D`, `inject_crs_at_shocks`/`find_shocks_pfrommer`, the CR-grey feedback sources) in a
new physical configuration, not a new physics implementation. The control run's forward-shock
radius matches Weaver's `R_2(t)` to <1%; a real, pre-existing (not CR-grey-related) energy-
normalization bias in the wind module's `_wind_ei3D` injection scheme (nominal injection volume
vs. the actually-masked, half-cell-shrunk volume) means an absolute energy-budget check against
the nominal wind luminosity is not meaningful at small `num_injection_cells` -- worked around by
using only differential (control-vs-DSA) energy checks, which are insensitive to this bias. See
PROGRESS.md's 2026-09-10 entry for full calibration numbers, tolerances, and the energy-bias
mechanism.

## Resolved: SNR expanding into a uniform then a clumpy medium (ladder item 11)

Test: `pytests/cosmic_rays_grey/cr_snr_clumpy_medium.py`. Three design decisions confirmed with
the user before implementing (same pattern as items 4, 6, 8, 9, 10): discrete spherical clumps
(generalizing the Sedov test's tanh-taper `weight` pattern to off-center positions) rather than a
turbulent lognormal density field (the repo's `create_turb_field` generator exists but is unused
elsewhere and would add unvalidated spectral choices); a new physically-scaled SNR setup
(`E_SN=1e51` erg, `n=1 cm^-3`, `T=1e4` K ISM) as the uniform baseline, rather than reusing item 7's
toy code-unit Sedov setup; and validation via the established item-7/9/10 differential CR
energy-partition identity plus a second "clumps-matter" signature check (clumpy-DSA vs.
uniform-DSA total `E_cr`), since no analytic reference solution exists for a CR-DSA shock breaking
through an inhomogeneous medium.

No new simulation code was needed -- like item 10, this is a new test of existing machinery
(`inject_crs_at_shocks`/`find_shocks_pfrommer`, the CR-grey feedback sources) in a new physical
configuration. The energy-partition identity holds to `~6.5e-4` (uniform) / `~2e-4` (clumpy)
relative error, and total energy is conserved to `~1.5e-6` across all four runs (uniform/clumpy x
control/DSA) -- identical because ambient *pressure*, not density, sets the initial thermal
energy budget, so the clumps' extra mass does not perturb it. The clumps-matter signature is a
clear, real effect (`E_cr` differs by `~18%` between uniform and clumpy media at fixed
`dsa_efficiency`/`dsa_mach_min`), but its **sign is a decrease, not the naively-expected
increase**: although clumps do locally raise the shock's Mach number as expected (lower sound
speed at fixed pressure), their extra inertia slows the *overall* forward shock enough (smaller
max shock radius, fewer total shock-surface cells) that less total thermal energy is processed
through the whole shock surface within a fixed `t_end`, and so less total CR energy is injected
overall. See PROGRESS.md's 2026-09-11 entry for the full calibration numbers and reasoning.

## Resolved (partially): SILCC-ISM project M0a/M0b (ladder item 12 start)

Ladder item 12 ("reproduce a published grey CR-ISM result") is scoped as its own multi-session
project (user's explicit choice, not a single ladder item) targeting Girichidis et al. 2016
(ApJL 816, L19) primarily and Simpson et al. 2016 (ApJL 827, L29) as a stretch cross-check. Full
milestone roadmap (M0a through M6) and design decisions live in this session's plan file
(`/export/home/nknoell/.claude/plans/memoized-discovering-scone.md`) and PROGRESS.md's 2026-09-11
entry -- summary here covers only what's done.

Two foundational milestones done this session, both new tests under `pytests/stratified_ism/`
(a new test category -- no existing one fits a stratified-box project):

- **M0a** (`bc_smoke_test.py`): validates the periodic-xy/open-z mixed boundary condition
  combination this whole project needs, in isolation (a tanh-tapered blob advecting under uniform
  bulk velocity, one full periodic wrap). Passed cleanly at its final calibration
  (`N=128`, float64: mass conserved to `~2.15e-14`, x-centroid error `~4.9e-5`) -- no BC bugs
  found, the general hydro BC handler's per-axis independence held up as expected. The one real
  issue hit along the way was a float32 precision problem (mass conservation failed at `N=128`
  under the default float32 before `jax_enable_x64` was enabled), not a BC bug.
- **M0b** (`stratified_hydrostatic_column.py`): generalizes `pytests/self_gravity/
  external_potential.py`'s Plummer-sphere pattern to a vertical potential
  `phi(z) = C_pot*log(cosh((z-z0)/h))`, whose matching isothermal hydrostatic density is the
  classic Spitzer (1942) `sech^2` profile. Passed its (loosened, honestly-calibrated) tolerances
  at its final calibration (`N_XY=64`, `N_Z=512`: core Mach `~0.1057`, core density drift
  `~0.0255`), but surfaced a **real, open gap**: the FV-mode gravity coupling
  (`_gravitational_source_term_along_axis`, `_gravity.py:299`) only supports a simple
  non-conservative source (unlike the FD path's flux-consistent conservative options), producing a
  resolution-independent equilibrium residual confirmed across three resolutions (`N_XY=8/16/64`:
  core Mach `~0.096`/`~0.101`/`~0.1057`, an 8x range in `N_XY`, 64x in total cell count) -- a
  "not well-balanced" FV scheme combination, not a bug in this test. Flagged for any future
  milestone (M2 onward) needing a tight hydrostatic baseline; not fixed here (out of scope for a
  smoke-test milestone). Full numbers and the resolution/steepness checks that established this:
  PROGRESS.md's 2026-09-11 entry.

## Resolved: SILCC-ISM project M1 (Koyama & Inutsuka two-phase net-cooling curve)

New cooling-curve type `KOYAMA_INUTSUKA_NET_COOLING` (`astronomix/_modules/_cooling/cooling_options.py`
/`_cooling_tables.py`/`_cooling.py`) implements Koyama & Inutsuka (2002, ApJ 564, L97 eq. 4,
corrected coefficients per Nagashima, Inutsuka & Koyama 2006) two-phase net heating/cooling:
`dU/dt = n_H*Gamma - n_H^2*Lambda(T)`, `Gamma=2e-26` erg/s,
`Lambda(T) = Gamma*[1e7*exp(-1.184e5/(T+1000)) + 1.4e-2*sqrt(T)*exp(-92/T)]`. New pytest
`pytests/stratified_ism/ki_cooling_thermal_relaxation.py`, with an independent (plain numpy/scipy)
reference solver at `astronomix/test_setups/reference_solutions/koyama_inutsuka_equilibrium.py`.

**Design decision (decided and documented, not asked -- became unambiguous once the existing
dispatch code was read carefully): a new analytic dispatch branch, not a `PIECEWISE_POWER_LAW`
table.** The existing table format stores `log10(Lambda)`, which cannot represent a curve that
changes sign -- and a sign change (net heating below the unstable branch, net cooling above it) is
exactly what a two-phase equilibrium needs. K&I's own curve is a closed-form fit anyway (not
tabulated data like Schure et al. 2009), so a dedicated `_cooling_rate` branch evaluating the
formula directly is both simpler and more faithful than forcing it into the table format (which
would also need a fabricated, physically meaningless Townsend `Y_table` for a curve Townsend
integration was never validated against anyway).

**The mu_e/mu_H bookkeeping is not an approximation.** Every downstream consumer
(`dtemperature_dt` et al.) assumes `dU/dt = -rho^2/(mu_e*mu_H)*cooling_rate(T,rho)` (an `n_e*n_H`-
shaped process). K&I's law has a different shape (`n_H^2*Lambda - n_H*Gamma`, no `n_e` at all).
Substituting `cooling_rate = (mu_e/mu_H)*Lambda(T) - mu_e*Gamma/rho` into the generic formula makes
every `mu_e` cancel algebraically, exactly reproducing `dU/dt = -n_H^2*Lambda(T) + n_H*Gamma`
regardless of the configured `hydrogen_mass_fraction`/`metal_mass_fraction` -- verified two
independent ways (symbolic re-derivation from the actual `dtemperature_dt` code, and a direct
numeric single-step comparison against a hand-derived physical `dT/dt`) before trusting it.
`gamma_heating_eff`/`lambda_scale_eff` bake this compensation in at build time (in
`koyama_inutsuka_cooling`, given the same X/Z the run's `CoolingParams` uses) so the runtime
dispatch branch's signature matches every other curve.

**Real finding #1: the plan's "two stable phases from a range of initial temperatures" assumption
was wrong for a fixed-density test.** `ki_bracket(T)` is monotonically increasing throughout
`T=10-1e5` K (checked directly, no local hump) -- the equilibrium condition at fixed `n_H` has
exactly one root, always thermally stable (Field's isochoric criterion). `update_pressure_by_cooling`
only ever touches pressure, so a fixed-density box structurally cannot explore bistability at all.
The real K&I "S-curve" lives in the **pressure-density plane**: `T_eq(n_H)` decreases with `n_H`,
so `P_eq(n_H) = n_H*k_B*T_eq(n_H)` is non-monotonic -- a local max near `n_H~1 cm^-3`
(`P_eq/k_B~4950` K/cm^3) and a local min near `n_H~8.6 cm^-3` (`P_eq/k_B~1597` K/cm^3); any
pressure in between admits three equilibrium densities. Genuine bistability only shows up once
density is free to respond to pressure imbalance (M2 onward), not in an isolated fixed-density
milestone. The committed test instead checks: unique-root convergence from both sides at 3
representative densities, plus the `P_eq(n_H)` non-monotonicity from the simulation's own
converged points (matches the independent reference to <1%).

**Real finding #2 (gap #3, confirmed not assumed): `update_temperature_implicit`'s naive
fixed-point iteration failed at a genuinely stiff dt, silently.** At `n_H=10 cm^-3`, `T=1e4` K
(instantaneous cooling time `~3438` yr), a single implicit step at `dt=5e4` yr (~14.5x that
cooling time) gave `10068` K vs. the true `5855` K -- `jax.lax.while_loop` just stopped at
`max_iter=50` regardless of `tol`, no error. First mitigated by adding a cooling-aware `dt_cool`
term to `_finite_volume/_timestep_estimation/_timestep_estimator.py`'s `_cfl_time_step`
(mirroring the existing `dt_visc`/`dt_relax` pattern: `C_cfl / max_over_grid(|dT/dt|/T)`, gated on
`config.cooling_config.cooling` so every existing non-cooling test is unaffected by construction),
which prevented the hydro loop from ever handing the solver a dt this stiff.

**Solver itself fixed 2026-09-16 (SILCC-ISM M4-audit follow-up): `update_temperature_implicit`
now solves the same backward-Euler equation via Newton's method** (`jax.grad` for the diagonal
Jacobian -- cooling has no spatial coupling, so a cell's `dT/dt` depends only on its own `T`).
Verified this eliminates the silent-divergence failure mode entirely (Newton's residual converges
to `~1e-12` in ~12 iterations at ratios where the old solver diverged to physically nonsensical
values, e.g. millions-of-percent error) -- but **does not eliminate all truncation error**: even
an exactly-solved single backward-Euler step is only first-order accurate, so real error persists
at intermediate stiffness (100-1000x the local cooling time, worse there than at either milder or
much more extreme ratios). `dt_cool` (above) and the separate `CoolingConfig.subcycle_stiff_cooling`
mechanism (SILCC-ISM M4, see below) therefore remain genuinely necessary for accuracy, not
superseded by the Newton fix. Two existing regression tests whose premise was "the naive solver
fails visibly here" had to be rewritten accordingly (M1's `test_ki_cooling_cfl_guard` and M3.5's
`test_cooling_guard_engages_for_sn_driving`, a near-duplicate check through this module's own
config path). Full before/after numbers across five stiffness ratios: PROGRESS.md's 2026-09-16
"naive fixed-point cooling solver replaced" entry.

**Real finding #3 (test-setup pitfall, worth remembering for future cooling tests): `n_H =
density/mu_H` is this codebase's convention** (`get_particle_number_density`), so building a
uniform box at a target `n_H` needs `density = n_H*mu_H*m_p`, not `density = n_H*m_p` (the
convention every CR-grey pytest uses, since none of them touch the cooling module's `n_H`-dependent
machinery). Missing `mu_H` silently builds the wrong density and relaxes cleanly to the wrong
`T_eq` (caught only because the test cross-checks against an independent reference, not just "did
it converge to *some* value").

Full exact numbers, all six calibrated (n_H, factor) combinations, and the CFL-guard calibration:
PROGRESS.md's 2026-09-11 M1 entry.

## Resolved (rescoped): SILCC-ISM project M2 (stratified column + K&I cooling: collapse, not equilibrium)

M2 combines M0b's stratified external-potential column with M1's K&I two-phase net-cooling curve
for the first time. The plan's original goal ("verify a physically sensible two-phase-structured
quasi-equilibrium") turned out to be unreachable in this setup -- established via four independent
checks (each a user-directed step in the investigation, per this module's "ask before picking a
direction" pattern for genuine design forks) -- and the milestone was rescoped (user-confirmed) to
verify the *onset* of a real thermal-gravitational collapse instead. New pytest
`pytests/stratified_ism/stratified_column_thermal_collapse.py`.

**The four checks, in order (full numbers: PROGRESS.md's 2026-09-11 M2 entry):**
1. A naive single-temperature isothermal IC (M0b's own style) collapses the midplane
   (`n_H`: 10 -> 195.5, `T`: 8000 -> 47.6 K by 2 dynamical times) and goes to NaN by 5.
2. A "coarse local-equilibrium" IC (density from a warm barometric guess, temperature set per-cell
   from the K&I equilibrium at that local density -- so the midplane starts essentially exactly at
   its true equilibrium) makes **no qualitative difference** -- same collapse, same NaN. Rules out
   "bad IC" as the cause. (A fully self-consistent hydrostatic+K&I-equilibrium profile, attempted
   via fixed-point iteration on the coupled ODE, diverges outright even under strong
   under-relaxation -- independent confirmation the underlying continuous problem is itself
   unstable, not a numerical-iteration artifact.)
3. Scanning the external potential's depth (`C_pot`) down to 10% of its reference value still
   collapses just as badly; only at 1-3% does collapse stop -- but only because that also flattens
   the *initial* density contrast away almost entirely (edge-to-midplane density ratio drops from
   the intended ~100x down to ~1.1-1.3x). Weakening gravity trades away the stratification this
   milestone needs, rather than stabilizing it.
4. **Decisive: a resolution scan.** The *same* well-equilibrated-IC setup, run at higher resolution
   (`N_Z`: 128 -> 256), collapses to NaN *faster* (between 1-2 dynamical times, vs. 2-5 at lower
   resolution) -- not slower or convergent. A resolution increase making a collapse happen sooner
   is the textbook signature of an **unregularized thermal instability** (Field 1965: the ISM
   thermal instability's growth rate is unbounded at short wavelength without a regularizing
   mechanism -- thermal conduction, turbulence, or magnetic tension, none implemented here).
   Refining resolution further would only resolve smaller, faster-growing modes, never converge.

**Conclusion:** a plain external-potential-confined column with K&I cooling and no additional
support genuinely has no dynamically reachable stable equilibrium at meaningful density contrast
-- real physics, not a bug, and a direct preview of why the roadmap's M3 (episodic SN
driving/turbulence) exists as the mechanism expected to resist this exact runaway. The "stable
two-phase profile" claim is deferred to M4/M5, once that additional physics can provide it.

**What the committed test actually checks:** the collapse's early, well-resolved onset (up to a
calibrated `t_end=1.0` dynamical time, safely inside the NaN-free window) is real, self-consistent
physics, not numerical noise -- substantial ongoing midplane condensation (`n_H` growth factor
`>1.5`), the midplane temperature tracking its own *new, shifting* local K&I equilibrium (not some
other numerical effect) to `<8%`, the envelope (which started near its own equilibrium) staying
quiescent, and mass conservation to `<5e-3` (all calibrated with real margin against the actual
observed run -- see PROGRESS.md for exact numbers). The diagnostic plot overlays the K&I
equilibrium curve evaluated at the final density against the actual final temperature profile,
showing them tracking closely across the *entire* column, not just at isolated points.

## Resolved: SILCC-ISM project M3 (episodic supernova-driving module)

New module `astronomix/_modules/_sn_driving/` (`SNDrivingConfig`/`SNDrivingParams`,
`_inject_supernovae`), wired into `_iteration_level_updates.py`'s `_iteration_level_injections`
before cooling -- resolving gap
#4's injection-ordering question in the direction the plan already anticipated (matching wind's
placement, so freshly-deposited hot ejecta isn't clipped by cooling before hydro ever advects it).
New pytest `pytests/stratified_ism/sn_driving_energy_conservation.py`.

**Mechanism:** each step draws one Bernoulli trial with probability `sn_rate * dt` (a standard
thinned-Poisson-process approximation -- valid whenever `sn_rate * dt << 1`, so at most one
supernova per step is the overwhelmingly common case and a second simultaneous trigger isn't
modeled) and, if it fires, a uniformly random site in the box ("random" placement, matching
Girichidis et al. 2016's own comparison case and Simpson et al. 2016's "random" mode -- "at
density peaks" is the M6 stretch cross-check, not implemented here). Deposits the same
tanh-tapered spherical footprint items 7/11 use, renormalized (Sedov-style) so the total energy
deposited is exactly `sn_energy` regardless of resolution: a thermal part added to gas pressure,
plus -- when the CR-grey model is active -- `sn_cr_fraction` (default 0.1, Girichidis' convention)
of the total dumped directly into `e_cr` (design decision #2: not routed through
`inject_crs_at_shocks`/DSA, so it doesn't compound item 9's still-open DSA resolution-
non-convergence finding). `sn_cr_fraction` is forced to 0 (all-thermal) whenever
`registered_variables.cosmic_ray_e_active` is False, so total energy conservation never depends on
whether CR-grey happens to be on.

**Periodic wraparound (needed for "random site anywhere in the box", genuinely new to this
codebase):** the site-to-cell distance uses the minimum-image convention along any axis whose
`config.boundary_settings` is periodic, so a site drawn near a periodic edge still gets its full,
undistorted footprint instead of being truncated by the domain edge.

**Real bug found and fixed (worth flagging for any future point-injection-in-a-periodic-box
work):** the first version computed the taper weight -- and, critically, its normalizing sum --
over the full *ghost-padded* array `_iteration_level_injections` actually operates on. For a site
near a periodic edge this double-counts real domain volume in the normalization (a ghost cell
mirrors a real interior cell on the far side, so both pick up weight), and then silently discards
the ghost cells' share of the deposit the next time the boundary handler refreshes them from the
interior -- quietly losing a fraction of `sn_energy` every time a site landed near an edge (a
large, systematic bias, not noise -- calibrated at `~0.125` lost out of `0.45` injected over 9
triggers). Fixed by restricting the weight -- both the normalization sum and the actual `.add()`
deposit -- to the interior cells only, and relying on the standard boundary handler to correctly
repopulate the ghost cells from the (now updated) interior afterward. Every earlier point-injection
ladder item (7/10/11) uses *open* boundaries with the explosion kept well inside the domain by a
containment check, so this exact failure mode had never been exercised before M3.

**Validation is an exact, not statistical, energy-conservation check** -- a genuinely different
validation style from every other feedback module in this codebase, made possible by two facts
specific to this setup: (1) in a periodic box with no gravity/cooling/open boundaries, a
conservative FV scheme's domain-integrated (thermal + kinetic + CR) energy can only change at a
supernova deposit, so `E_total(t_end) - E_total(0)` must equal exactly `N_triggers * sn_energy`
regardless of how complicated the subsequent shock evolution gets; (2) `N_triggers` itself can be
known exactly rather than just statistically, because `config.fixed_timestep=True` makes every
step's trigger probability identical and known in advance, and the SN-driving trigger is the only
per-step PRNG consumer active in this test (turbulent forcing, this codebase's other one, is off)
-- so the test replays the exact `jax.random.split`/`bernoulli` sequence outside the simulation and
compares against the real run's actual energy change. The same `fixed_timestep`-based technique
ladder item 9 used to instrument its per-step `e_cr` budget (2026-09-09). Calibrated relative error
`~7.2e-10` (CR-active) / `~7.8e-15` (CR-inactive) against a `1e-6` tolerance. Full numbers:
PROGRESS.md's 2026-09-12 entry.

## Resolved: SILCC-ISM project M3.5 (SN driving + K&I cooling combined) -- and a real bug found, since fixed (see below)

New pytest `pytests/stratified_ism/sn_driving_with_cooling.py` (three tests: a lightweight
pre-cooling sanity check, an M1-calibration-point guard-engagement regression check, and a
closing SN-injection-plus-cooling stability check). Gap #3's concern -- a hot cell arriving
*impulsively* at a density where cooling is genuinely stiff -- turned out not to be reachable the
way the plan first pictured it, and chasing that down surfaced a real, unrelated bug in the shared
time-stepping code. Five findings, in order (full numbers: PROGRESS.md's 2026-09-12 M3.5 entry;
canonical narrative: the pytest's own module docstring):

1. **The instant of SN injection is not cooling-stiff.** `E_SN=1e51` erg into `n_H=1 cm^-3`
   ambient gas gives `T_hot~2.3e9` K -- so hot K&I's `Lambda(T)` has saturated, making the cooling
   time *longer* than the pure-hydro CFL `dt`. This is why the early adiabatic/free-expansion phase
   (item 11's own cooling-free SNR test) is adiabatic in the first place.
2. **This project's affordable box/resolution doesn't spontaneously reach genuine stiffness
   either**, evolving a single deposit hydro-only across a `2000`-`30000` yr scan -- the
   persistently-hot interior keeps the domain-wide hydro `dt` tighter than the shell's cooling time
   throughout, `dt_hydro/t_cool` never exceeding `~1.0` (need `>10x` to count as stiff).
3. **A real, randomly-sited explosion in a periodic box NaN'd the solver via a periodic-edge
   wrap**, at two resolutions (`NUM_CELLS=48` then `96`) -- the ghost-cell handler mirrors a second
   copy of the `~1e8-1e9` K blast onto itself near a periodic seam, an extreme, poorly-resolved
   collision the HLLC solver can't handle.
4. **Switching periodic -> open boundaries did not fix it** -- the identical explosion (boundary
   type can't change the PRNG trigger sequence) NaN'd again, ruling out "periodic wrap" as the sole
   mechanism (open boundaries' own interior-only weight renormalization can concentrate a
   near-edge deposit further). Root cause reframed as the raw injection magnitude itself; fixed
   (user-directed) by raising ambient density 100x (`n_H=1 -> 100 cm^-3`) and decoupling the
   injection radius from grid spacing (fixed physical 4 pc), landing `T_hot` at `~2.6e6` K instead
   of `~1e9` K.
5. **Even the 1000x-gentler injection still NaN'd -- and root-causing it found a genuine
   architectural bug, unrelated to cooling, edges, or injection magnitude.** Every synthetic
   single-explosion reproduction (box-centered, off-grid-offset, even the exact real failure
   coordinates) evolved cleanly; only the real, stochastic, *mid-run* `_inject_supernovae` path
   ever failed. Traced to `astronomix/time_stepping/time_integration.py`: each step's `dt` is
   computed from the primitive state *before* `_iteration_level_updates` runs, and that same
   already-fixed `dt` is then used by `_inject_supernovae`, `update_pressure_by_cooling`, *and* the
   hydro flux evolve for that same step. On a triggering step, cooling and the hydro update both
   apply a `dt` sized for the calm pre-explosion state to the freshly-injected, extremely hot cell
   -- measured directly, the pre-injection ambient `dt` was `~441824x` larger than the
   correctly-sized post-injection `dt` at this state.

**This bug was real and open at the time -- reported here per explicit user direction rather than
fixed that session. It has since been fixed; see "Resolved: stale-dt-at-injection bug" below.**
It was a gap in shared code (`time_stepping/time_integration.py`'s per-step `dt`-computation order
relative to `_iteration_level_updates`), not specific to SN driving -- any future module injecting
a large, localized, mid-run perturbation (not via the initial condition) would have hit the same
gap.

**Closing test, redesigned around finding 5:** builds the SN deposit into the initial condition
(not via `SNDrivingConfig`'s trigger, sidestepping the bug since `time_integration`'s first `dt`
is then already correctly sized) and evolves with cooling active from `t=0` through item 11's
radiative-phase-onset timescale (`3e4` yr). Calibrated: NaN-free; mass conserved to `1.4e-16`
relative error; peak temperature drops `99.8%` from the raw-injection `2,592,683` K to `5002.8` K
at t_end while staying far below a `1.5x` raw-injection runaway ceiling; minimum temperature
matches the ambient equilibrium (`41.77` K) exactly.

## Resolved: stale-dt-at-injection bug (2026-09-14)

Before this fix, `time_stepping/time_integration.py`'s `_step` computed one `dt` per step from
the pre-injection state and reused it unchanged for every downstream module (injections, cooling,
forcing, the hydro evolve) -- so a lump-sum injection landing mid-step (SN driving's `E_SN` dump,
in particular) left the rest of that same step running at a `dt` sized for the calm state it just
blew past. See M3.5's writeup above for the discovery (a `~441824x` measured discrepancy between
the pre- and post-injection CFL estimate at the triggering step).

**Design discussion before implementing:** four candidate fix shapes were compared with the user
-- (1) a global, unconditional fix; (2) the same split gated behind an opt-in flag; (3) each
injection module reporting its own CFL requirement back to the loop, with a full adaptive
retry/reject substep; (4) other. **User picked (1).** Two things surfaced during that discussion
worth recording for future reference:

- **The bug is not CR-specific, and a CR-gated fix (an early framing of option 2) would not have
  covered the case that actually needs it.** M3.5 (where the bug was found) and M4 (which it
  blocks) both run with `cosmic_ray_grey_config` off -- the real trigger condition is "an episodic,
  lump-sum injection module is active" (SN driving today; wind and CR-DSA shock injection carry
  the same structural risk), not "CR is on."
- **Option (3)'s full adaptive retry was judged more machinery than the bug needs, and riskier.**
  `time_stepping/_time_loop.py` already runs the per-step body inside `fori_loop`/`while_loop`/
  `checkpointed_while_loop` (the last specifically built to keep reverse-mode AD working through an
  adaptive-trip-count loop); nesting a second, data-dependent-trip-count reject/retry loop inside
  that for one re-check is a real complexity and AD-risk increase for a bug that a single
  re-estimate (reusing the existing, already CR/MHD-aware `_cfl_time_step`/
  `_source_term_aware_time_step`/`_cfl_time_step_fd*` estimators) fully addresses.

**Implementation:** `_iteration_level_updates.py`'s single `_iteration_level_updates` function is
now two: `_iteration_level_injections` (stellar wind, SN driving, and CR-DSA shock injection --
all three lump-sum/discontinuous updates to `primitive_state`, previously scattered across the old
function's start and end) and `_iteration_level_continuous_updates` (cooling, the neural-net/CNN
correctors, viscosity, turbulent forcing, frame tracking, CR streaming/anisotropic-transport
corrections, the positivity floor -- unchanged relative order). `time_integration.py`'s `_step`
factors the pre-existing FV/FD x mhd/hydro x source-term-aware dt-estimation branch into a new
`_estimate_cfl_dt` helper, calls it once as before, applies `_iteration_level_injections`, then --
only when `not config.fixed_timestep` and at least one of `wind_config.stellar_wind` /
`sn_driving_config.sn_driving` / `cosmic_ray_grey_config.diffusive_shock_acceleration` is active
(a static, config-level condition -- configs using none of the three compile to exactly the same
single-estimate path as before) -- calls `_estimate_cfl_dt` again on the post-injection state and
takes `jnp.minimum(old_dt, new_dt)`, before running `_iteration_level_continuous_updates` and the
hydro evolve at the final `dt`. `config.fixed_timestep` (M3's exact-PRNG-replay energy-conservation
technique) is deliberately exempted, since re-estimating there would desync the replay from the
run's actual `dt` sequence.

**Why the recompute point had to move inside `_iteration_level_updates`, not just before the hydro
evolve** (correcting the earlier "plausible fix shape" sketch above): cooling itself, which used
to run right after SN injection inside the old combined function, was directly implicated in
Finding 5 above (cooling alone at the stale `dt` drove cells to an unphysical `98` K) -- so the
split has to separate injections from cooling/forcing/etc., not just from the hydro evolve.

**Reordering risk check (CR-DSA injection moved from last in the old function to grouped with
wind/SN driving):** every `pytests/cosmic_rays_grey/*.py` config using
`diffusive_shock_acceleration=True` was checked against `cooling`/`turbulent_forcing`/
`neural_net_force`/`cnn_mhd_corrector`/`diffusion`/`frame_tracking` -- none combine DSA injection
with any module now running after it, consistent with this module's "isolate one thing per test"
convention throughout items 7-11, so the reordering changes none of their computed results.

**Known, deliberately accepted residual gap:** N-body's RK4 advance must still run before
`_iteration_level_injections` (multi-source stellar wind needs the freshly-advanced orbit position
for its injection sites that same step), so it necessarily uses the pre-recompute `dt` even on a
step where SN driving/wind/CR-DSA later shrinks it -- a currently-unexercised combination (no
CR-grey ladder test uses `nbody`), flagged in a code comment analogous to the standing
Lax-Friedrichs CR-dissipation gap above.

**Regression suite re-run (2026-09-14), all pass:** items 3, 5, 7, 8, 10, 11 and both SN-driving
tests (M3, M3.5) -- run individually on a pinned GPU to avoid cross-run memory contention, which
produced two misleading artifacts (an apparent OOM and an apparent hang) in an earlier batched run
that both cleared on an isolated rerun. See `PROGRESS.md`'s 2026-09-14 entry for the full list.

**M3.5's closing test redesigned (2026-09-14, user-requested) to exercise the real stochastic
trigger directly, as evidence the fix works.** `test_sn_injection_with_cooling_stability` now runs
the actual `SNDrivingConfig` trigger under real adaptive CFL stepping (not the old
initial-condition workaround) -- same ambient setup, `EXPECTED_N_TRIGGERS=2`. Result at the default
seed: no NaNs, peak temperature `9041` K above ambient (proof a trigger fired), mass conserved to
`9.14e-6` (recalibrated from a stale `1e-9` -- this test's real random site can, unlike M3's
periodic box or the old box-centered deterministic test, land close enough to the open boundary
for real mass to leave the domain; confirmed via boundary-face density, not a bug). Full numbers:
PROGRESS.md's 2026-09-14 entry; full narrative: that pytest's own module docstring.

## Resolved: SILCC-ISM project M5 (CR-grey code-comparison anchor vs. Girichidis et al. 2016, 2026-09-16)

M5 turns grey two-moment CR transport on for the first time in this project's stratified-column
setting (M4's exact box/SN-driving/cooling stack, `sn_cr_fraction=0.1` and
`diffusive_relaxation=True` now active) and compares against Girichidis et al. (2016), ApJL 816,
L19 (arXiv:1509.07247)'s reported mass-loading factor, outflow velocity, and CR-vs-gas pressure
scale height. New script: `pytests/stratified_ism/m5_cr_driven_outflow.py`.

**Attempt 1 (kappa_parallel=1e28 cm^2/s isotropic): completed, but meaningless -- CR pressure
homogenized flat across the whole box** (diffusion length `sqrt(kappa*T_end)~=495` pc vs. this
box's own 300 pc height; the paper's own 100x-smaller kappa_perp is what confines *their* much
bigger box vertically). H_cr came back undefined.

**Attempt 2 (kappa_perp=1e26 cm^2/s, box-matched diffusion length ~50 pc; reduced_streaming_speed
scaled 3000->300 km/s to hold cost fixed): NaN'd.** Root-caused via two rounds of temporary debug
instrumentation (both fully reverted): a fresh SN trigger spikes gas sound speed to ~1291 km/s
(4.3x over `reduced_streaming_speed`) and CR energy 16x; shortly after, a cell's pressure goes
negative -- confirmed via PRE/POST-source-term tracing to happen for the first time immediately
after `_apply_gravity_source` (the operator-split source applicator carrying self-gravity *and*
CR-grey feedback) adds its update, not inside the already-floored RK2 hydro stages. **A genuine,
previously-unknown gap: `_apply_gravity_source` had no positivity floor**, unlike
`_evolve_gas_state_unsplit_inner`'s RK2-internal floor (M4, 2026-09-15). Fixed: the same
unconditional `jnp.maximum` floor pattern added to `_apply_gravity_source` too (see that
function's docstring in `evolve_state.py`) -- a general FV-solver fix, not CR-grey-specific (any
strong-enough self-gravity source term in a near-vacuum cell could plausibly have hit the same
gap).

**Attempt 3 (fix applied, config otherwise identical to attempt 2): DONE.** Completed with no
NaNs through the full run, including passing cleanly through the same SN-driven blow-out that
NaN'd attempt 2. Late-time comparison: mass-loading eta=6.16 (paper: order unity -- same order of
magnitude), outflow velocity v_out=5.83 km/s (paper: 10-50 km/s -- same order, low side), and
**H_cr/H_gas ~= 3.1 (paper's headline qualitative claim -- CR pressure support extends well
beyond the gas scale height -- reproduced directly)**. Full numbers, the SN-trigger/negative-
pressure trace, and the fix's exact code location: PROGRESS.md's 2026-09-16 M5 entries.

New diagnostics added for this milestone: `_mass_loading_and_outflow_velocity` (mass flux through
a horizontal plane one scale height above the midplane), `_pressure_scale_heights` (horizontally-
averaged P_gas/P_cr e-folding heights from the midplane), and `_girichidis_fig1_style_plot` (edge-
on density, face-on density, midplane CR energy density -- Girichidis et al. 2016 Fig. 1 style).

This completes milestones M0a through M5 of ladder item 12 (SILCC-ISM project). Only M6
(optional MHD + anisotropic-diffusion stretch) remains, not started -- would let a real
`kappa_perp`/`kappa_parallel` anisotropy replace this milestone's isotropic stand-in.

## Resolved: SILCC-ISM project M4 (SN driving + K&I cooling + long-run stability, 2026-09-15)

M4 combines M2's stratified column, M3's episodic SN driving and the delayed-cooling
overcooling mitigation (tapered-shield NaN root-caused and fixed earlier the same day) and the
FV unsplit-solver near-vacuum positivity floor (also fixed earlier the same day) -- but the real
run still stalled indefinitely (simulation time frozen while the GPU stayed pinned near 100%
utilization) rather than completing or NaN-ing. Root-caused via a throttled `jax.debug.callback`
probe (temporary, removed once done) inside `time_stepping/time_integration.py`'s `_step`: at the
stall, `rho_min` was ~100x *above* the positivity floor (ruling out "cells stuck at the cold
floor," the leading hypothesis when the previous session paused) and the stalled cell's derived
temperature (`~7591` K at `n_H~0.1 cm^-3`) was ordinary warm diffuse gas, not floor-temperature
gas.

**Actual mechanism: `_finite_volume/_timestep_estimation/_timestep_estimator.py`'s `_cfl_time_step`
has a cooling-time term (`dt_cool`, added at M1 as a guard against `update_temperature_implicit`'s
naive fixed-point iteration diverging for a stiff `dt` -- see "Resolved: SILCC-ISM project M1"
above) that is a *domain-wide minimum*.** M4's SN-driven blow-out continuously produces diffuse,
recently-shocked-then-unshielded cells sitting in the K&I curve's fast-cooling regime; once the
run reaches that regime, one such cell anywhere pins the *entire* simulation's `dt` to its own
local cooling time, indefinitely -- a real stall, mechanistically unrelated to the positivity-floor
NaN this and the previous several sessions were chasing.

**Fix: `CoolingConfig.subcycle_stiff_cooling`, new, opt-in (default `False` -- every existing
config/test unaffected).** When set: `update_pressure_by_cooling` (`_cooling.py`) calls a new
`update_temperature_implicit_subcycled` instead of the plain `update_temperature_implicit`, and
`_cfl_time_step` skips its `dt_cool` term entirely (the global hydro `dt` no longer needs to be
bounded by cooling stiffness, since the cooling update now absorbs it internally). Two designs
were tried:

1. **A fixed sub-step count, sized once from the state's *initial* relaxation rate
   (`jax.lax.fori_loop`, each sub-step a full `update_temperature_implicit` fixed-point solve).**
   Matched the calibrated M1 stiff-guard test case well (`5939` K vs. the fine-ODE reference's
   `5855` K, 1.4% error, vs. the naive single-step call's `10068` K / 72% error), but a
   1000x-more-extreme synthetic case (further from equilibrium, so the trajectory gets stiffer than
   its own starting rate predicts) diverged badly (`78324` K vs. a `160.5` K reference) -- the fixed
   count under-resolves the later, stiffer part of the trajectory, and an under-resolved fixed-point
   sub-step can itself fail to converge, compounding across every later sub-step.
2. **Used instead: an adaptive `jax.lax.while_loop`** re-deriving the safe local sub-step size
   (`dt_sub * rate <= 0.1`, from the *current* state, every sub-step) and accumulating elapsed time
   up to the full `time_step`, using plain explicit (forward-Euler) sub-steps rather than chained
   fixed-point solves -- once a sub-step is kept inside the stability target, explicit is simpler
   and sidesteps the fixed-point's own non-convergence risk. Re-validated: mild case `5536` K (5.4%
   error), 1000x-extreme case `147.4` K (8.2% error, vs. the fixed-count design's ~490x overshoot)
   -- degrades gracefully rather than diverging as the stiffness ratio grows.
   `max_substeps=20000` bounds worst-case cost; timed on an M4-grid-shaped (`32x32x256`) field:
   `0.002s` at the mild calibrated stiffness, `0.069s` at a 1000x ratio (close to what the real
   stalled cell implied), `1.85s` only at a deliberately-extreme `1e5x` synthetic stress test well
   beyond what the real run ever demanded.

Verified `ki_cooling_thermal_relaxation.py` (M1) and `sn_driving_with_cooling.py` (M3.5) both
still pass unchanged (`subcycle_stiff_cooling` defaults `False`) before spending GPU time on the
real run. With `subcycle_stiff_cooling=True` set in the M4 script's `CoolingConfig`, the real
run -- GPU-pinned, progress bar monitored end-to-end, any run stuck at a fixed percentage for 3+
minutes auto-killed -- completed cleanly through the full `t_end` (`~7.39e6` yr, all 40
snapshots), no NaNs, no stall, clearing both this and the previous session's documented stall
points. The resulting trajectory shows genuine repeated SN-driven disruption/re-collapse cycles:
midplane `n_H` swings between `~20-35` cm^-3 (collapsed) and `~0.001-0.4` cm^-3 (blown out)
several times over the run, with coincident `max(T)` spikes to `1e7-1e8` K at each disruption --
the qualitative behavior M4 was scoped to demonstrate (episodic SN driving able to disrupt and
re-trigger collapse, rather than M2's baseline monotonic overcooling collapse). Diagnostic plot:
`pytests/stratified_ism/pics/m4_stratified_column_sn_driving_delayed_cooling.svg`. Full numbers
and the two implementation attempts' detail: PROGRESS.md's 2026-09-15 (later same day) entry.

`subcycle_stiff_cooling` is a generally-useful fix, not M4-specific -- worth reaching for again in
any future module that can locally drive the K&I curve into its fast-cooling regime for an
extended stretch, not just SN driving.

## Resolved: pion-decay emission (ladder item 13, Phase C, 2026-09-17)

Ladder item 13 starts Phase C (differentiable emission operators, plan Sec. 3) -- the first ladder
item after the SILCC-ISM project (item 12, M0a-M6). Scope, per the plan's own wording ("Pion-decay
SED for a known proton population vs naima"): a standalone physics-formula check, decoupled from
any running simulation -- given an externally-specified proton spectrum and target gas density,
does this module's photon-spectrum formula match `naima`'s reference implementation of Kafexhiu et
al. (2014)? Wiring emission into the running simulation (a spectrum normalized from a cell's local
`e_cr`, line-of-sight integration to a map) is left to ladder item 15, per the plan's own staging.

**New module:** `astronomix/_modules/_cosmic_rays_grey/cr_grey_emission.py`. `GreyProtonSpectrum`
(a `NamedTuple`: `amplitude`, `e_0`, `alpha`, `e_cutoff`, `beta`) is the plan's grey "particle-
spectrum object" (Sec. 3: "normalization from e_cr, slope, cutoff") -- an exponential-cutoff
power law in total proton energy, deliberately matching naima's `ExponentialCutoffPowerLaw`
parameter-for-parameter so a naima particle distribution built with the same numbers gives the
same `J(E)`. `pion_decay_photon_spectrum` (and its naima-unit-convention sibling,
`pion_decay_photon_spectrum_per_ev`) is a line-for-line JAX port of `naima.radiative.PionDecay`'s
analytic (non-lookup-table) implementation of Kafexhiu et al. (2014): the inelastic p-p cross
section (Eq. 1), the low/mid/high-energy inclusive pi0-production cross sections and their
model-parameter tables (Tables IV, V, VII, all four Monte Carlo high-energy models), the
kinematic `Egmax`/`Xg` shape function (Eqs. 9-11), the ISM nuclear-enhancement factor (Sec. IV),
and naima's own log-log trapezoidal proton-spectrum integration (`trapz_loglog`) -- reimplemented,
not called, since `naima` is a validation dependency only (installed in the dev venv, not an
astronomix runtime dependency).

**Reimplementation strategy, since naima's own code uses numpy boolean-mask assignment
(`array[where(cond)] = ...`) for its five-or-so Tp/Egamma regime splits, which isn't
`jax.grad`-compatible as written:** every branch is evaluated unconditionally across the whole
input domain and combined with nested `jnp.where` in the same priority order naima's sequential
mask assignment implies (a later, narrower-`Tp` condition's branch wins over an earlier, wider
one wherever both apply -- confirmed by hand from naima's source that this, not the literal
per-regime boundaries the code comments suggest, is the *net* effective regime map, e.g. Table V's
nominal "20 < Tp <= 100" Geant4 regime is actually only used up to `Tp <= Etrans[hiEmodel]`
because the later `Tp > Etrans` assignment overwrites it for `Etrans < Tp <= 100`). Every
`sqrt`/`log`/fractional power in every branch is floored against invalid domains (negative
arguments from floating-point rounding right at a regime boundary, or from evaluating a branch's
formula outside the input range where it's actually valid) with a `1e-30` floor -- this module's
established "every branch must be finite everywhere so `jnp.where`'s backward pass doesn't
differentiate through a NaN in an unselected branch" pattern (`cr_pressure_speed_floor`,
`dsa_efficiency_kang_ryu_2013`). One analytically-derived case worth noting for future reference:
`Xg` (Eq. 11's kinematic shape variable) never needs a *lower*-bound clip in principle (`Yg`,
`Ygmax >= m_pi` for any positive energy, by the Y-substitution's own construction, so `Xg >= 0`
always) -- naima itself only clips the upper bound (`Xg > 1 -> 1`); this port clips both
defensively anyway, since the floors elsewhere can perturb `Xg` by a tiny amount right at the
edges.

**Validated against naima directly (`pytests/cosmic_rays_grey/cr_pion_decay_emission.py`),
essentially to floating-point precision, not just "close":** for the naima-tutorial-style ECPL
proton population (amplitude `1e36`/eV at `E_0=1` TeV, `alpha=2.1`, `e_cutoff=30` TeV) against a
`2.5`/cm^3 target density, across 40 photon energies log-spaced from 100 MeV to 100 TeV, with
`naima`'s own `useLUT=False` (forcing its analytic-formula path, not a cached fit to it, so the
comparison is against the same underlying physics rather than naima's numerics) -- **max relative
error `3.4e-13`, median `5.0e-15`**. Re-verified across all four Monte Carlo high-energy models
(Pythia8/Geant4/SIBYLL/QGSJET), with and without the nuclear-enhancement factor, and several
different `(alpha, e_cutoff)` spectral shapes: every case agrees to `<5e-12` relative error. This
level of agreement (machine precision, not merely "same order of magnitude") is itself evidence
the regime-priority reconstruction above is exactly right, not just approximately so -- a
transcription bug in any single regime boundary would show up as a sharp, larger disagreement
localized to that Tp/Egamma range, not a uniform floating-point-level residual across six decades
of photon energy.

**One real, non-obvious bug caught before the naima comparison ever ran:** the first version of
this comparison silently returned `NaN` for every photon energy. Root cause: `astronomix` (and
this reimplementation, importing only `jax.numpy`) defaults to JAX's 32-bit float precision, but
this ladder item's own physical quantities span an enormous dynamic range for an entirely mundane
reason -- the plan's chosen "known proton population" convention (a *total* particle-count
spectrum, e.g. amplitude `~1e36`/eV, not a volumetric density) combined with the proton-energy
integration grid spanning `~1.2` GeV to `10` PeV overflows `float32` outright (`>3.4e38`) partway
through the log-log trapezoidal integration, well before any precision concern would matter --
confirmed directly (isolated per-quantity NaN tracing) and fixed by enabling `jax_enable_x64`
in the pytest, the same fix `cr_gradient_check.py` (item 16) already uses for an unrelated
reason (FD-vs-AD precision). Worth remembering for ladder item 15 (the end-to-end SNR-cloud
case): any future code path that carries a *total* (not volumetric) particle-count normalization
alongside a wide proton-energy integration range will need `jax_enable_x64` too, not just
differentiability-sensitive tests.

**Differentiability smoke check** (the plan's own "gradients flow sim -> spectrum -> map" goal,
Sec. 3): `jax.grad` of the total summed photon rate with respect to every `GreyProtonSpectrum`
field plus the gas density is finite and nonzero everywhere tested -- a lighter check than ladder
items 16/17's dedicated FD-vs-AD gradient tests (not yet extended to the emission operators as of
this ladder item; item 13 itself only asks for the SED comparison).

**Deliberately deferred to later ladder items, not attempted here:** deriving a spectrum's
normalization from a simulation cell's `e_cr` (needs the still-open "spectral-interface contract"
design decision, plan Sec. 6, though for the grey phase specifically the mapping is just matching
`integral(E_kinetic * J(E) dE) = e_cr` for an assumed `alpha`/`e_cutoff` -- not attempted yet since
nothing in this ladder item needed it); line-of-sight integration to a map/SED (item 15); the
leptonic (electron) emission operators (item 14 -- turned out not to actually be blocked by the
"electron treatment" decision after all, see that item's own entry below: the decision only gates
*deriving* an electron population from the CR-grey/proton state, not validating the emission
formula against a directly-specified one). `naima` was added to the dev venv (`pip install naima`)
as a validation-only dependency, not declared in `pyproject.toml` (mirrors this module's `pytest`
non-dependency, per PROGRESS.md's "Environment notes").

## Resolved: synchrotron + inverse-Compton emission (ladder item 14, Phase C, 2026-09-17)

Ladder item 14: "Synchrotron + IC SED for a known electron population vs naima." Same scoping
logic as item 13 (previous section) applies and was checked explicitly before starting: the
plan's open "electron treatment" decision (Sec. 6: fixed `K_ep` post-processing vs. a
separately-evolved grey electron energy density) only gates *deriving* an electron population from
the CR-grey/proton simulation state -- it does not gate validating the emission formula itself
against a directly-specified, known electron spectrum, exactly as item 13 didn't need the
"spectrum normalization from e_cr" question resolved either. So item 14 proceeded without that
decision, deferring it (along with line-of-sight integration to a map) to item 15, same as item 13
did for the analogous proton-side questions.

**New module:** `astronomix/_modules/_cosmic_rays_grey/cr_grey_emission_leptonic.py`.
`GreyElectronSpectrum` mirrors `cr_grey_emission.GreyProtonSpectrum` exactly (same
exponential-cutoff-power-law fields), just for electrons. `synchrotron_photon_spectrum` is a JAX
port of `naima.radiative.Synchrotron`'s implementation of the Aharonian, Kelner & Prosekin (2010)
closed-form synchrotron-kernel approximation (`Gtilde`, AKP10 Eq. D7); `inverse_compton_photon_spectrum_planck`
ports `naima.radiative.InverseCompton`'s isotropic-thermal-seed IC cross section (Khangulyan,
Aharonian & Kelner 2014, Eq. 14, the `G34` kernel). Only the isotropic-thermal-seed IC case is
ported (naima's anisotropic and monochromatic/tabulated seed-field cases are deferred to whichever
later ladder item first needs them) -- covers naima's `"CMB"`/`"FIR"`/`"NIR"` presets plus any
custom (possibly diluted, i.e. non-blackbody-normalized) gray-body seed. As with item 13, naima
itself (not the plan's cited Blumenthal & Gould (1970)) is what's actually ported, since naima is
the plan's own chosen validation reference and these are the algorithms it actually runs.

**Validated against naima directly** (`pytests/cosmic_rays_grey/cr_synchrotron_ic_emission.py`):
for the naima-tutorial-style ECPL electron population (amplitude `1e33`/eV at `E_0=1` TeV,
`alpha=2.0`, `e_cutoff=100` TeV) across 60 photon energies log-spaced 1 ueV-1 TeV --
**synchrotron (100 uG field): max relative error `4.7e-14`; inverse Compton (CMB seed): max
relative error `3.7e-8`** (restricted to photon energies within `1e-30` of each spectrum's own
peak -- see the pytest's own "Tolerance note" docstring for why: both spectra fall off
exponentially past their cutoff, and comparing floating-point noise `>100` orders of magnitude
below the peak isn't physically meaningful, the same judgment call ladder item 8's tolerance
calibration made). Re-verified across several field strengths/spectral shapes (synchrotron) and
all three seed-field presets plus a custom diluted 20 K gray body (IC): every case agrees to
`<4e-8`. Also a differentiability smoke check (`jax.grad` of each channel's total summed photon
rate w.r.t. every spectrum field, `B`, and seed temperature/density: finite and nonzero
everywhere tested).

**Two real, non-obvious bugs caught before this comparison was fully trustworthy -- both were the
*same* underlying mistake (a positivity floor sized for the wrong quantity's scale) manifesting
two different ways, not two unrelated bugs:**

1. **Synchrotron's forward values were wrong by ~15-20 orders of magnitude, silently (no NaN, no
   crash) -- traced to `jnp.maximum(x, _TINY)` guards on `cs1_1`/`e_c_erg`, two CGS-erg-scale
   quantities that routinely sit at `~1e-17` to `~1e-51` for realistic photon energies/fields.**
   This module's shared `_TINY = 1e-30` (`cr_grey_emission.py`) is sized for the *dimensionless*
   quantities the hadronic-channel module actually uses it for -- reusing it here silently
   clobbered every legitimate small-but-nonzero `cs1_1`/`e_c_erg` value up to a photon energy
   where they finally exceeded `1e-30`, producing a completely different (wrong) curve shape, not
   an obviously-broken one (both curves fell with photon energy, just at the wrong absolute scale
   and rate) -- exactly the kind of bug a "does it look roughly SED-shaped" eyeball check would
   miss, and only the direct naima cross-check caught. **Fixed:** floor `B` itself (the only input
   quantity that can be legitimately exactly zero) once, at the top, instead of flooring either
   downstream erg-scale product -- `photon_energy_erg`/`gam` are never zero by construction, so
   `cs1_1`/`e_c_erg` need no floor of their own once `B` is floored.
2. **Inverse Compton's forward values were correct (matched naima to `~1e-14`) but `jax.grad`
   produced `NaN`.** Root cause: `z_safe = jnp.clip(z, _TINY, 1.0 - _TINY)` with the same
   `_TINY = 1e-30` -- at float64's ~16-digit precision, `1.0 - (1.0 - 1e-30)` rounds to *exactly*
   `1.0` (1e-30 is far below machine epsilon at scale 1), so the upper clip is silently a no-op
   and `1 - z_safe` can land on an exact `0.0` for any `z` at or above 1 (the unphysical
   `E_gamma >= E_electron` region, later masked to `0` in the *output* by this function's own
   `jnp.where`). `1/(1-z_safe)` then genuinely diverges there -- the forward value stays correct
   because the final `jnp.where` masks it, but `jnp.where` differentiates both branches
   (`cr_pressure_speed_floor`'s own category of gotcha), and `0 * NaN = NaN` in IEEE754
   contaminates the whole gradient even though that unphysical region's cotangent seed is exactly
   zero. **Fixed:** compute `one_minus_z = jnp.clip(1.0 - z, 1e-10, 1.0)` directly instead of
   deriving `(1-z)` from a clipped `z` -- avoids the float64 cancellation instead of trying to
   patch around it, at a floor scale (`1e-10`) that actually survives subtraction at this scale.
   (This also nudged the forward-value agreement from `~1e-14` to `~3.7e-8` -- still far inside
   this pytest's `1e-6` tolerance -- since the floor now genuinely clips a thin sliver of
   near-kinematic-edge phase space instead of computing it exactly via a cancellation that turned
   out to be numerically unstable to begin with.)

**General lesson for future ladder items, worth remembering:** this module's `_TINY = 1e-30`
"universal" floor constant (`cr_grey_emission.py`) is only safe for genuinely dimensionless
quantities of order unity or so -- any new physics formula with its own natural scale (CGS units,
a near-unity kinematic ratio, etc.) needs its *own* appropriately-scaled floor, not a blind reuse
of this one. Both bugs above were caught only by the direct naima cross-check (bug 1) and an
explicit gradient-finiteness check (bug 2) -- neither would have been visible from inspecting the
code or from a forward-value-only test in bug 2's case.

**Deliberately deferred, not attempted this ladder item:** naima's anisotropic and
monochromatic/tabulated-spectrum IC seed cases; Bremsstrahlung (mentioned in the plan as
"optional"); deriving an electron population from the CR-grey/proton state and line-of-sight
integration to a map -- both taken up by item 15 (below), which resolved the electron-treatment
decision (fixed `K_ep`) for real.

## Resolved: SNR-molecular-cloud pion-bump case (ladder item 15, Phase C's closing item, 2026-09-17)

Ladder item 15: "End-to-end SNR-molecular-cloud case reproducing a pion-bump source
morphology/SED (IC 443 / W44 analog; Ackermann et al. 2013)." This closes Phase C (items 13-15)
and is the first ladder item to turn a *running simulation's* local `e_cr` into an emission
map/SED rather than validating a formula against a directly-specified spectrum.

**Two design decisions, both discussed with the user via concrete options before starting (this
item's own genuine forks, not resolved by reading the surrounding code more carefully):**
1. **Electron treatment (finally exercised for real, plan Sec. 6): user picked the fixed
   `K_ep=0.01` ratio** (plan's own "simplest, good for first light" option) over building a
   separately-evolved grey electron energy density (substantial new Phase-A-scale infrastructure
   -- a new state variable with its own transport/loss physics -- not attempted). This resolves
   the decision for Phase C's purposes; a future stretch item could still revisit it.
2. **Setup scope: user picked reusing ladder item 11's SNR-into-clumpy-medium scaffold**
   (`pytests/cosmic_rays_grey/cr_snr_clumpy_medium.py`) over building a fresh IC443/W44-scale
   setup -- matches this project's established practice (M5/M6 also reused scaffolding rather
   than chasing literal literature parameters). The only change from item 11: one of the three
   clumps' density contrast is raised from item 11's ISM value (10x ambient) to a
   molecular-cloud-like value (100x ambient) -- the hadronic-emission target. A single run
   suffices (DSA on, clumpy medium) -- item 11 already validated the underlying energy-
   partition/clumps-matter dynamics.

**New library function**: `cr_grey_emission.proton_spectrum_normalized_to_energy` -- the plan's
"spectrum normalization from e_cr" (Sec. 3), deliberately deferred by items 13/14. Builds a
`GreyProtonSpectrum` (assumed shape: `alpha=2.0`, the standard test-particle strong-shock DSA
prediction; `e_cutoff=100` TeV, a typical assumed Galactic-SNR cutoff -- the plan's own "grey
caveat," shape assumed not predicted) whose total energy content (naima's own `Wp`/`We`
convention: `integral(E*J(E)dE)`) matches a given target -- verified directly against a
by-hand recomputation of that same integral, `<4e-12` relative error, broadcasting correctly over
an array of per-cell targets in one call.

**Key efficiency insight, not obvious in advance and what actually made a full-3D-grid emission
map tractable:** every per-cell pion-decay/synchrotron/IC computation is *linear* in that cell's
own spectrum amplitude -- the expensive part of each formula (the full Kafexhiu/AKP10/Khangulyan
integral over the proton/electron energy grid) depends only on the assumed spectral *shape*
(shared across every cell here), never the per-cell normalization. So each formula is evaluated
**once**, for a unit-amplitude spectrum, giving a photon-energy-only shape curve `K(E_gamma)`; the
full per-cell (or domain-summed, or line-of-sight-projected) map is then just `K(E_gamma) *
(per-cell amplitude * per-cell target density)`, an outer product rather than 128^3 independent
expensive integrals. This is a genuinely reusable pattern for any future ladder item wiring these
emission formulas to a real 3D field.

**New pytest**: `pytests/cosmic_rays_grey/cr_snr_molecular_cloud_pion_bump.py`. Single 128^3 run
(smaller than item 11's calibrated 256^3 -- item 15 doesn't need item 11's tight energy-partition
precision), same box/explosion/clump geometry as item 11, DSA on (`dsa_efficiency=0.1`,
`dsa_mach_min=1.3`), `T_END=1000` yr. **Completed cleanly, no NaNs**, forward shock contained
(`r_shock_max=1.127` code units vs. domain half-width `2.0`), total injected `E_cr=4.62e49` erg
(order-of-magnitude consistent with `~0.1 * E_SN=1e51` erg not yet fully processed through the
shock by `t_end`, same qualitative picture as item 11).

**Both checks passed, with real margin, not just "it ran":**
1. **Morphology**: the projected (line-of-sight-summed) pion-decay emission map's brightest pixel
   sits **3.16 grid cells** from the molecular cloud's own prescribed center -- well inside both
   the test's own 12-cell tolerance *and* the cloud's own physical radius (`~6.4` cells at this
   resolution) -- a tight, convincing confirmation that emission genuinely concentrates on the
   dense target, not just "somewhere in the general vicinity." Directly operationalizes the
   plan's own point (Sec. 3): "the multiphase/dense target gas *is* the signal."
2. **The pion bump**: the domain-total hadronic SED (`E^2 dN/dE`) at 50 MeV is suppressed
   **94.9%** below the naive power-law extrapolation of its own 3-30 GeV slope (measured slope
   `0.12`) -- a real, sharp spectral turnover, not a continuing power law, and comfortably past
   the test's own `50%` minimum. This is exactly Ackermann et al. (2013)'s headline finding
   (a low-energy break/turnover below the pi0-production kinematic threshold, ruling out a
   featureless power law) -- the first time ladder item 13's Kafexhiu et al. (2014) formula (there
   validated to floating-point precision against naima for a *hand-specified* spectrum) has been
   exercised end-to-end from a real simulation's own CR content.

**One honest caveat, not a bug:** the SED's overall peak (`E^2 dN/dE` maximized) lands at `~158`
GeV over this test's `20 MeV-500 GeV` grid, not the few-hundred-MeV-to-few-GeV peak IC443/W44's
own (softer, lower-cutoff) proton spectra show in Ackermann et al. (2013)'s actual data -- an
artifact of this item's own assumed `alpha=2.0`/`e_cutoff=100 TeV` shape choice (hard spectrum,
high cutoff) making `E^2 dN/dE` stay roughly flat-to-rising until that cutoff, not a claim that
this run quantitatively reproduces either specific SNR's spectrum. The physically meaningful,
literature-matching signature this ladder item actually checks for -- and found -- is the sharp
sub-100-MeV kinematic-threshold suppression (Check 2 above), which is a property of the pi0-decay
process itself (already validated exactly against naima in item 13), not of this item's own
assumed high-energy shape parameters.

**Leptonic channels, for context (not asserted on):** at the SNR's actual `B=3.24` uG field
(naima's own CMB-equipartition default, used since this setup carries no self-consistent magnetic
field), synchrotron is utterly negligible at gamma-ray energies (`sed_sync` sums to `~4e-130` over
the whole grid -- synchrotron from these electrons peaks at radio/X-ray energies, not gamma-ray,
at this field strength). Inverse Compton (on the CMB) is real but subdominant to pion decay at the
~3 GeV reference point checked (`E^2 dN/dE`: pion `9.96e44` vs. IC `2.98e44`, pion decay ~3.3x
brighter) -- consistent with the plan's framing that dense-target hadronic emission is expected to
dominate exactly where the dense cloud sits.

**Phase C (items 13-15) is now fully complete.** Next per the plan's own staging: Phase D
(gradient-based inference demo, Sec. 1) or extending ladder item 16's FD-vs-AD gradient check to
the injection/emission chain specifically (its own wording: "transport, then injection, then
emission" -- only the lighter smoke-check version was done inline in items 13/14/15) -- not yet
discussed with the user.

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
- **Shared gas/CR Riemann-solver wave-speed bound degrades shock-capturing** (found 2026-08-27,
  ladder item 9 investigation): `grey_cr_fast_speed`'s `reduced_streaming_speed` is combined with
  the gas sound speed via `max(...)` into one shared HLL wave-speed bound (`hll.py`), regardless of
  whether any `e_cr` is actually present. Reproduced: a stationary Mach-4 shock with zero `e_cr`
  the whole run is measured at Mach 3.81 by `find_shocks_pfrommer` with the default
  `reduced_streaming_speed=1.0`, degrading to Mach 1.64 at `reduced_streaming_speed=8.0` --
  resolution-independent (4x finer grid, same result), traced to HLL's flux formula having a term
  that doesn't vanish even at an exact stationary-shock RH state, scaling with the *chosen*
  wave-speed bound rather than the true local one. Not fixed (would need separate gas/CR
  wave-speed bounds in the Riemann solver, a real design change); see PROGRESS.md's 2026-08-27
  entry for the full mechanism and how it blocks ladder item 9.
- **Sustained DSA injection at a stationary shock grows without bound**: resolved 2026-09-09 --
  root mechanism was a stationary lab-frame shock never releasing the compressive
  `-P_cr * div(v)` term the way a fluid parcel crossing a genuinely moving shock would; fixed by
  redesigning ladder item 9's test around a shock that actually propagates. See "Resolved:
  CR-modified shock structure (ladder item 9)" above and PROGRESS.md's 2026-09-09 entry for the
  full account, including a further interaction with Finding 1 (below, still open) this
  redesign surfaced.
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
