"""shock_finder3D (ours) vs shock_finder_v2 on the same Sedov / CWB states.

Both finders share Phase 1 (d_s = -grad T / |grad T|, identical
``_gradients.py``) and differ in three stages:

* zones (criterion 3): ours samples the immediate neighbour along the
  dominant grid axis, v2 interpolates one cell along the continuous normal;
* surface: ours keeps the argmin of div v over the whole zone segment along
  the dominant grid axis, v2 keeps local minima of div v interpolated one
  cell ahead/behind along the continuous normal, plus a sub-cell (parabolic)
  offset of the sampling origin;
* Mach sampling: ours walks in whole cells to the zone exit and on while the
  pressure keeps falling/rising (``mach_sampling_extend``, post sample at the
  pressure peak), with a pressure or (DSA default) velocity Mach number; v2
  samples 0.25 cells past the first zone exit (``profile_aware=False``,
  default) or the max/min pressure 0.25-1.5 cells past it
  (``profile_aware=True``).

Every zone x surface x sampler combination is evaluated, so the scatter and
the grid artefacts can be attributed to a stage. Results per variant go to
``/export/scratch/nknoell/finder_v2_comparison/results_<state>.npz``, figures
to ``pics/``.

Usage: python compare_finders.py [sedov64 sedov128 sedov256 cwb128 ...]
       python compare_finders.py --plot-only [...]
"""

import sys
import time
import types
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from astronomix import CARTESIAN, SimulationConfig, finalize_config, get_helper_data, get_registered_variables

# shock_finder_v2 imports itself as astronomix._physics_modules._shock_finder
import astronomix.shock_finder_v2 as _v2_package
sys.modules.setdefault("astronomix._physics_modules", types.ModuleType("astronomix._physics_modules"))
sys.modules["astronomix._physics_modules._shock_finder"] = _v2_package
from astronomix._physics_modules._shock_finder import _shock_zones as v2_zones
from astronomix._physics_modules._shock_finder import _shock_surface as v2_surface
from astronomix._physics_modules._shock_finder import _shock_mach as v2_mach
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer as find_v2

from astronomix.shock_finder3D import _shock_zones as our_zones
from astronomix.shock_finder3D import _shock_surface as our_surface
from astronomix.shock_finder3D._gradients import _calculate_shock_direction, _calculate_velocity_divergence
from astronomix.shock_finder3D._shock_mach import mach_from_pressure_samples, mach_from_velocity_jump
from astronomix.shock_finder3D.pfrommer_shock_finder import (
    _velocity_consistent,
    find_shocks_pfrommer as find_ours,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path("/export/scratch/nknoell/finder_v2_comparison")
PIC_DIR = Path(__file__).resolve().parent / "pics"
GAMMA = 5.0 / 3.0
MACH_MIN = 1.3
MAX_STEPS = 15  # our walk cap (dsa_shock_sampling_max_steps)

# Sedov reference (``_sedov_setup``): E = 1, rho = 1, p = 1e-4, t = 0.1
SEDOV_T = 0.1
SEDOV_XI0 = 1.15167  # gamma = 5/3
SEDOV_R = SEDOV_XI0 * (1.0 * SEDOV_T**2 / 1.0) ** 0.2
SEDOV_MACH = 0.4 * SEDOV_R / SEDOV_T / np.sqrt(GAMMA * 1e-4)

SAMPLERS = ("ours_p", "ours_u", "v2", "v2_profile")
SAMPLER_LABEL = {
    "ours_p": "ours walk+extend, pressure",
    "ours_u": "ours walk+extend, velocity (DSA)",
    "v2": "v2 exit+0.25, pressure",
    "v2_profile": "v2 profile-aware, pressure",
}


def load_state(name):
    data = np.load(DATA_DIR / f"{name}.npz")
    n = data["density"].shape[0]
    config = SimulationConfig(geometry=CARTESIAN, dimensionality=3, num_cells=n, box_size=1.0)
    rv = get_registered_variables(config)
    state = jnp.zeros((rv.num_vars, n, n, n), dtype=jnp.float32)
    state = state.at[rv.density_index].set(data["density"])
    state = state.at[rv.pressure_index].set(data["pressure"])
    state = state.at[rv.velocity_index.x].set(data["vx"])
    state = state.at[rv.velocity_index.y].set(data["vy"])
    state = state.at[rv.velocity_index.z].set(data["vz"])
    config = finalize_config(config, state.shape)
    return state, config, rv, get_helper_data(config)


def ours_sampler(state, config, rv, direction, zones, surface, estimator):
    """Our adaptive + extended walk (as find_shocks_pfrommer with the DSA
    kwargs), for an arbitrary zone/surface pair."""
    pressure = state[rv.pressure_index]
    density = state[rv.density_index]
    compressive = _calculate_velocity_divergence(state, config, rv) < 0
    p_post, p_pre, _, rho_pre, ex_post, ex_pre, steps_post, steps_pre = (
        our_zones.get_post_pre_shock_values_adaptive(
            direction, pressure, density, zones, max_steps=MAX_STEPS,
            extend_monotone=True, compressive=compressive, return_sample_steps=True,
        )
    )
    valid = surface & ex_post & ex_pre
    mach_p = mach_from_pressure_samples(p_post, p_pre, GAMMA)
    if estimator == "pressure":
        mach = mach_p
    else:
        valid = valid & _velocity_consistent(state, config, rv, direction, 0.5)
        velocity = [state[i] for i in (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)]
        u_post = sum(our_zones.sample_along_direction(v, direction, -steps_post) * d
                     for v, d in zip(velocity, direction))
        u_pre = sum(our_zones.sample_along_direction(v, direction, steps_pre) * d
                    for v, d in zip(velocity, direction))
        mach = mach_from_velocity_jump(u_post - u_pre, p_pre, rho_pre, GAMMA)
        valid = valid & (mach > MACH_MIN) & (mach_p > MACH_MIN) & (mach_p - 1.0 >= 0.5 * (mach - 1.0))
    return dict(mach=jnp.where(valid, mach, 0.0), p_post=p_post, p_pre=p_pre,
                d_post=steps_post, d_pre=steps_pre)


def v2_sampler(state, rv, direction, zones, surface, offsets, profile_aware):
    """v2's _calculate_mach_at_surface, also returning its samples."""
    pressure = state[rv.pressure_index]
    temperature = pressure / state[rv.density_index]
    sampler = (v2_zones.get_profile_aware_post_pre_shock_values if profile_aware
               else v2_zones.get_adaptive_post_pre_shock_values)
    p_post, p_pre, _, _, valid, d_post, d_pre = sampler(
        direction, zones, pressure, temperature, max_steps=8,
        center_offsets=direction * offsets[jnp.newaxis],
    )
    mach = mach_from_pressure_samples(p_post, p_pre, GAMMA)
    return dict(mach=jnp.where(surface & valid, mach, 0.0), p_post=p_post, p_pre=p_pre,
                d_post=d_post, d_pre=d_pre)


def evaluate(name):
    state, config, rv, helper_data = load_state(name)
    pressure = state[rv.pressure_index]
    density = state[rv.density_index]
    direction = _calculate_shock_direction(pressure, density, config)

    zones = {
        "ours": our_zones.identify_shock_zones(state, config, rv, helper_data, direction, MACH_MIN),
        "v2": v2_zones.identify_shock_zones(state, config, rv, helper_data, direction, MACH_MIN),
    }
    out = {"direction": np.asarray(direction, dtype=np.float32)}
    timing = {}
    for zone_name, zone in zones.items():
        out[f"zones_{zone_name}"] = np.asarray(zone)
        surfaces = {
            "ours": our_surface.identify_shock_surface(state, zone, direction, config, rv),
            "v2": v2_surface.identify_shock_surface(state, zone, direction, config, rv),
        }
        for surface_name, surface in surfaces.items():
            key = f"z{zone_name}_s{surface_name}"
            out[f"surface_{key}"] = np.asarray(surface)
            offset_options = {"off0": jnp.zeros(surface.shape, state.dtype)}
            if surface_name == "v2":
                offset_options["offv2"] = v2_surface.calculate_shock_surface_offsets(
                    state, surface, direction, config, rv)
            for sampler in SAMPLERS:
                for off_name, offsets in offset_options.items():
                    if sampler.startswith("ours") and off_name != "off0":
                        continue  # our walk has no sub-cell origin
                    if sampler == "ours_p":
                        res = ours_sampler(state, config, rv, direction, zone, surface, "pressure")
                    elif sampler == "ours_u":
                        res = ours_sampler(state, config, rv, direction, zone, surface, "velocity")
                    else:
                        res = v2_sampler(state, rv, direction, zone, surface, offsets, sampler == "v2_profile")
                    variant = f"{key}_{sampler}_{off_name}"
                    sel = np.asarray(surface)
                    idx = np.argwhere(sel)
                    out[f"{variant}__idx"] = idx.astype(np.int16)
                    for field, values in res.items():
                        out[f"{variant}__{field}"] = np.asarray(values)[sel].astype(np.float32)
                    print(f"[{name}] {variant}: surface={sel.sum()}, valid={(out[f'{variant}__mach'] > 0).sum()}",
                          flush=True)

    # consistency + runtime of the two complete finders
    for label, call in (
        ("ours_full_p", lambda: find_ours(state, config, rv, helper_data, mach_min=MACH_MIN,
                                          mach_sampling_adaptive=True, mach_sampling_steps=MAX_STEPS,
                                          mach_sampling_extend=True)),
        ("ours_full_u", lambda: find_ours(state, config, rv, helper_data, mach_min=MACH_MIN,
                                          mach_sampling_adaptive=True, mach_sampling_steps=MAX_STEPS,
                                          mach_sampling_extend=True, mach_velocity_consistency=0.5,
                                          mach_estimator="velocity")),
        ("v2_full", lambda: find_v2(state, config, rv, helper_data, mach_min=MACH_MIN)),
        ("v2_full_profile", lambda: find_v2(state, config, rv, helper_data, mach_min=MACH_MIN,
                                            profile_aware_mach=True)),
    ):
        try:
            jax.block_until_ready(call().mach_numbers)
            t0 = time.time()
            res = call()
            mach = np.asarray(jax.block_until_ready(res.mach_numbers))
            timing[label] = time.time() - t0
        except jax.errors.JaxRuntimeError as error:  # e.g. v2 profile-aware at 256^3 on 11 GB
            print(f"[{name}] {label}: failed ({str(error).splitlines()[0]})", flush=True)
            timing[label] = np.nan
            mach = np.zeros(1)
        out[f"{label}__mach_grid_nonzero"] = mach[mach > 0].astype(np.float32)
        print(f"[{name}] {label}: {timing[label]:.2f} s, valid={np.sum(mach > 0)}", flush=True)
    out["timing_labels"] = np.array(list(timing))
    out["timing_values"] = np.array(list(timing.values()))
    out["num_cells"] = np.array(state.shape[1])
    np.savez(DATA_DIR / f"results_{name}.npz", **out)


if __name__ == "__main__":
    args = sys.argv[1:]
    plot_only = "--plot-only" in args
    names = [a for a in args if not a.startswith("--")] or ["sedov64", "sedov128", "sedov256", "cwb128"]
    if not plot_only:
        for name in names:
            evaluate(name)
    from plot_comparison import make_all_plots
    make_all_plots(names)
