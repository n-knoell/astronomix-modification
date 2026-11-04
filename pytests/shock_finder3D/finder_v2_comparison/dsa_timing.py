"""Cost of the DSA shock finding: ours vs shock_finder_v2 variants, fully jitted.

1. Finder (+ area weight) alone, on Sedov states at N = 96 / 160 (B7.3 DSA
   run, t = 0.07) and 256 (``generate_states.py``, hydro, CR rows zero):
   ours with the DSA kwargs + 1 / |n|_inf weight, the hybrid (v2 zones +
   surface, our walk) + column weight, v2 default / profile-aware + column
   weight, and the stages of ours (zones, surface, walk) separately.
2. One complete ``inject_crs_at_shocks`` (finder, weight, Mach/flux, spread,
   cap), ours vs hybrid + column weight, each in its own fresh trace.
3. A full simulation interval (``time_integration`` over ~20 CFL steps from
   the t = 0.07 snapshot) without DSA, with ours, with the hybrid: the DSA
   share of a time step.

Usage: python dsa_timing.py [96 160 256]
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CR_B73_REUSE", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "cosmic_rays_grey"))

import numpy as np
import jax
import jax.numpy as jnp

from astronomix import time_integration
from astronomix._modules._cosmic_rays_grey import cr_grey_injection
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_shock_finder_kwargs
from astronomix.shock_finder3D._gradients import _calculate_shock_direction
from astronomix.shock_finder3D._surface_area import surface_area_weights
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

from compare_finders import find_v2, our_surface, our_zones, v2_surface, v2_zones, MAX_STEPS
from dsa_physics import column_area_weights, find_shocks_hybrid

DATA_DIR = Path("/export/scratch/nknoell/finder_v2_comparison")
B73_DIR = Path("/export/scratch/nknoell/phaseB/b7_3")


def _timed(fn, repeats=5):
    jax.block_until_ready(fn())
    t0 = time.time()
    for _ in range(repeats):
        jax.block_until_ready(fn())
    return (time.time() - t0) / repeats


def _setup(n):
    """The B7.3 config at resolution n and a Sedov state for it."""
    os.environ["CR_NUM_CELLS"] = str(n)
    import importlib
    import cr_sedov_injection_rate as b73
    b73 = importlib.reload(b73)
    if (B73_DIR / f"states_eff0.1_N{n}.npz").exists():
        run = b73._run_with_snapshots(0.1)
        state = jnp.asarray(run["states"][-1])
        dt_probe = float(np.load(B73_DIR / f"b7_3_N{n}.npz")["dsa_dt_probe"][-1])
        return run, state, dt_probe
    # no CR snapshot at this resolution: the hydro Sedov state with zero CR rows
    # (the config build in _run_with_snapshots would rerun the blast, so build it here)
    data = np.load(DATA_DIR / f"sedov{n}.npz")
    from astronomix import (CARTESIAN, FINITE_VOLUME, HLLC, MINMOD, SimulationConfig, SimulationParams,
                            finalize_config, get_helper_data, get_registered_variables)
    from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
        CosmicRayGreyConfig, CosmicRayGreyParams)
    config = SimulationConfig(
        geometry=CARTESIAN, solver_mode=FINITE_VOLUME, riemann_solver=HLLC, limiter=MINMOD,
        dimensionality=3, num_cells=n, exact_end_time=True,
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True, diffusive_shock_acceleration=True),
    )
    rv = get_registered_variables(config)
    state = jnp.zeros((rv.num_vars, n, n, n), jnp.float32)
    for name, index in (("density", rv.density_index), ("pressure", rv.pressure_index),
                        ("vx", rv.velocity_index.x), ("vy", rv.velocity_index.y), ("vz", rv.velocity_index.z)):
        state = state.at[index].set(data[name])
    config = finalize_config(config, state.shape)
    params = SimulationParams(t_end=0.1, gamma=5.0 / 3.0, cosmic_ray_grey_params=CosmicRayGreyParams(
        dsa_efficiency=0.1, dsa_mach_min=b73.DSA_MACH_MIN))
    run = dict(config=config, params=params, registered_variables=rv, helper_data=get_helper_data(config))
    return run, state, None


def _with_cr_config(config, **changes):
    return config._replace(cosmic_ray_grey_config=config.cosmic_ray_grey_config._replace(**changes))


def _patched(fn):
    """Run fn with the hybrid finder + column weight patched into the injection."""
    holder = {}

    def finder(*args, **kwargs):
        holder["sf"] = find_shocks_hybrid(*args, **kwargs)
        return holder["sf"]

    def weights(pressure, density, sigma=2.0):
        sf = holder["sf"]
        return column_area_weights(pressure, density, sf.shock_surface_cells, sf.shock_zones)

    original = (cr_grey_injection.find_shocks_pfrommer, cr_grey_injection.surface_area_weights)
    cr_grey_injection.find_shocks_pfrommer = finder
    cr_grey_injection.surface_area_weights = weights
    try:
        return fn()
    finally:
        cr_grey_injection.find_shocks_pfrommer, cr_grey_injection.surface_area_weights = original


def main(resolutions):
    rows = []
    for n in resolutions:
        run, state, dt_probe = _setup(n)
        config, params = run["config"], run["params"]
        rv, hd = run["registered_variables"], run["helper_data"]
        cr_config = config.cosmic_ray_grey_config
        mach_min = params.cosmic_ray_grey_params.dsa_mach_min
        p, rho = state[rv.pressure_index], state[rv.density_index]
        timings = {}

        # 1. finder + weight, each one jitted function of the state
        def ours(s):
            sf = find_shocks_pfrommer(s, config, rv, hd, mach_min=mach_min, **dsa_shock_finder_kwargs(cr_config))
            return sf.thermal_energy_flux * surface_area_weights(s[rv.pressure_index], s[rv.density_index])

        def hybrid(s):
            sf = find_shocks_hybrid(s, config, rv, hd, mach_min=mach_min)
            return sf.thermal_energy_flux * column_area_weights(
                s[rv.pressure_index], s[rv.density_index], sf.shock_surface_cells, sf.shock_zones)

        def v2(profile):
            def f(s):
                res = find_v2(s, config, rv, hd, mach_min=mach_min, profile_aware_mach=profile)
                return res.thermal_energy_flux * column_area_weights(
                    s[rv.pressure_index], s[rv.density_index], res.shock_surface_cells, res.shock_zones)
            return f

        candidates = {"ours finder+w": ours, "hybrid finder+col": hybrid, "v2 finder+col": v2(False)}
        if n <= 160:  # the profile-aware v2 finder needs > 11 GB at 256^3
            candidates["v2prof finder+col"] = v2(True)
        for label, fn in candidates.items():
            timings[label] = _timed(lambda f=jax.jit(fn): f(state))

        # stages of ours
        direction = jax.jit(lambda s: _calculate_shock_direction(s[rv.pressure_index], s[rv.density_index], config))(state)
        zones_fn = jax.jit(lambda s, d: our_zones.identify_shock_zones(s, config, rv, hd, d, mach_min))
        timings["  stage: our zones"] = _timed(lambda: zones_fn(state, direction))
        zones = zones_fn(state, direction)
        our_surface_fn = jax.jit(lambda s, z, d: our_surface.identify_shock_surface(s, z, d, config, rv))
        timings["  stage: our surface"] = _timed(lambda: our_surface_fn(state, zones, direction))
        v2_surface_fn = jax.jit(lambda s, z, d: v2_surface.identify_shock_surface(s, z, d, config, rv))
        timings["  stage: v2 surface"] = _timed(lambda: v2_surface_fn(state, zones, direction))
        walk_fn = jax.jit(lambda d, p_, r_, z: our_zones.get_post_pre_shock_values_adaptive(
            d, p_, r_, z, max_steps=MAX_STEPS, extend_monotone=True,
            compressive=jnp.ones_like(z), return_sample_steps=True))
        timings["  stage: our walk"] = _timed(lambda: walk_fn(direction, p, rho, zones))

        # 2. one complete injection, fresh trace each (static config differs)
        raw_inject = cr_grey_injection.inject_crs_at_shocks.__wrapped__
        inject_ours = jax.jit(lambda s: raw_inject(s, config, params, rv, hd, 0.07, 1e-6))
        timings["inject ours"] = _timed(lambda: inject_ours(state))
        inject_hyb = jax.jit(lambda s: raw_inject(s, config, params, rv, hd, 0.07, 1e-6))
        timings["inject hybrid+col"] = _patched(lambda: _timed(lambda: inject_hyb(state)))

        # 3. a full simulation interval (~20 CFL steps), DSA off / ours / hybrid
        if dt_probe is not None:
            interval = params._replace(t_end=20 * dt_probe)
            run_cfg = config._replace(return_snapshots=False)
            for label, cfg, patch in (
                ("20 steps, no DSA", _with_cr_config(run_cfg, diffusive_shock_acceleration=False), False),
                ("20 steps, DSA ours", run_cfg, False),
                # different static config -> its own trace of the patched injection
                ("20 steps, DSA hybrid+col", _with_cr_config(run_cfg, dsa_area_normal_smoothing=2.0001), True),
            ):
                go = lambda c=cfg: time_integration(state, c, interval, rv)
                timings[label] = _patched(lambda: _timed(go, repeats=2)) if patch else _timed(go, repeats=2)

        for label, value in timings.items():
            rows.append(f"N={n:4d}  {label:28s} {value * 1e3:9.1f} ms")
            print(rows[-1], flush=True)
    (HERE / "pics" / "dsa_timing.txt").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [96, 160, 256])
