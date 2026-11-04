"""Which finder gives the right DSA injection? ours vs shock_finder_v2 (+ area weights).

On the saved B7.3 Sedov snapshots (``cr_sedov_injection_rate.py``,
``/export/scratch/nknoell/phaseB/b7_3/``: control run zeta = 0 and DSA run
zeta = 0.1, t >= 0.03), every pipeline's injection-relevant dissipation
``D = sum(flux * w) dx^2`` is compared with the simulated shock's own gas-only
Rankine-Hugoniot dissipation ``D_meas = 4 pi R^2 delta rho_0 V^3 / 2`` (R, V
from B7.3's self-similar fit), i.e. the quantity the injection is
``zeta`` times. D / D_meas splits into area (sum w dx^2 / 4 pi R^2) and flux.
The angular distribution of D over the sky (equal-area bins) measures grid
artefacts in the injected energy itself.

Pipelines (Mach number and flux as the injection would use them):

* ``ours``: production (find_shocks_pfrommer with dsa_shock_finder_kwargs,
  dsa_mach_and_flux, i.e. velocity Mach + CR-aware Hugoniot), smoothed-normal
  1 / |n|_inf weight (DB7);
* ``v2``: v2 default finder (pressure Mach 0.25 cells past the zone exit, its
  own flux);
* ``v2prof``: v2 with ``profile_aware_mach=True``;
* ``hyb``: v2 zones + v2 surface, our walk and velocity Mach, CR-aware flux.

each v2-surface pipeline with three area weights: ``unit`` (dx^2 per cell),
``ninf`` (ours, 1 / |n|_inf) and ``col`` (1 / |n|_inf divided by the number of
surface cells in the same zone segment of the cell's dominant-axis column, so
every column again counts dx^2 / |n|_inf once).

Usage (one resolution per process): CR_NUM_CELLS=96 python dsa_physics.py
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

from compare_finders import find_v2, our_zones, v2_surface, v2_zones, MAX_STEPS
from astronomix.shock_finder3D._data_structures import ShockFinderResult
from astronomix.shock_finder3D._gradients import _calculate_shock_direction, _calculate_velocity_divergence
from astronomix.shock_finder3D._shock_mach import mach_from_pressure_samples, mach_from_velocity_jump
from astronomix.shock_finder3D._energy_dissipation import thermal_energy_flux_from_pre_state
from astronomix.shock_finder3D._surface_area import _gaussian_smooth, surface_area_weights
from astronomix.shock_finder3D.pfrommer_shock_finder import _velocity_consistent, find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey import cr_grey_injection
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_mach_and_flux,
    dsa_shock_finder_kwargs,
    inject_crs_at_shocks,
)

from cr_sedov_injection_rate import LATE_TIME, NUM_CELLS, _run_with_snapshots

OUT_DIR = Path("/export/scratch/nknoell/finder_v2_comparison/dsa")
B73_DIR = Path("/export/scratch/nknoell/phaseB/b7_3")
GAMMA = 5.0 / 3.0
COLUMN_REACH = 3  # cells searched along the column on each side


def find_shocks_hybrid(primitive_state, config, registered_variables, helper_data, mach_min=1.3,
                       mach_sampling_steps=MAX_STEPS, mach_velocity_consistency=0.5, zones_from="v2", **_):
    """Drop-in for find_shocks_pfrommer with the DSA kwargs: v2 surface (on
    v2's or, with ``zones_from="ours"``, our zones), our adaptive + extended
    walk with the velocity Mach number."""
    rv = registered_variables
    pressure = primitive_state[rv.pressure_index]
    density = primitive_state[rv.density_index]
    direction = _calculate_shock_direction(pressure, density, config)
    zone_module = v2_zones if zones_from == "v2" else our_zones
    zones = zone_module.identify_shock_zones(primitive_state, config, rv, helper_data, direction, mach_min)
    surface = v2_surface.identify_shock_surface(primitive_state, zones, direction, config, rv)
    compressive = _calculate_velocity_divergence(primitive_state, config, rv) < 0
    p_post, p_pre, _, rho_pre, ex_post, ex_pre, steps_post, steps_pre = (
        our_zones.get_post_pre_shock_values_adaptive(
            direction, pressure, density, zones, max_steps=mach_sampling_steps,
            extend_monotone=True, compressive=compressive, return_sample_steps=True,
        )
    )
    mach_p = mach_from_pressure_samples(p_post, p_pre, GAMMA)
    normal = jax.lax.stop_gradient(direction)
    velocity = [primitive_state[i] for i in (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)]
    u_post = sum(our_zones.sample_along_direction(v, normal, -steps_post) * d for v, d in zip(velocity, normal))
    u_pre = sum(our_zones.sample_along_direction(v, normal, steps_pre) * d for v, d in zip(velocity, normal))
    mach = mach_from_velocity_jump(u_post - u_pre, p_pre, rho_pre, GAMMA)
    valid = (surface & ex_post & ex_pre
             & _velocity_consistent(primitive_state, config, rv, direction, mach_velocity_consistency)
             & (mach > mach_min) & (mach_p > mach_min)
             & (mach_p - 1.0 >= mach_velocity_consistency * (mach - 1.0)))
    mach = jnp.where(valid, mach, 0.0)
    flux = jnp.where(valid, thermal_energy_flux_from_pre_state(mach, p_pre, rho_pre, GAMMA), 0.0)
    return ShockFinderResult(
        shock_surface_cells=surface, shock_direction=direction, mach_numbers=mach,
        thermal_energy_flux=flux, shock_zones=zones, num_shocks=jnp.sum(surface, dtype=jnp.int32),
        shock_ids=surface.astype(jnp.int32), shock_zone_ids=zones.astype(jnp.int32),
        post_shock_steps=steps_post, pre_shock_pressure=p_pre, pre_shock_density=rho_pre,
        pre_shock_steps=steps_pre,
    )


def _smoothed_normal(pressure, density, sigma=2.0):
    """Gradient of the Gaussian-smoothed temperature, as surface_area_weights."""
    smooth = _gaussian_smooth(jax.lax.stop_gradient(pressure / density), sigma)
    grads = []
    for axis in range(3):
        pad = [(0, 0)] * 3
        pad[axis] = (1, 1)
        padded = jnp.pad(smooth, pad, mode="edge")
        n = smooth.shape[axis]
        grads.append(jax.lax.slice_in_dim(padded, 2, n + 2, axis=axis)
                     - jax.lax.slice_in_dim(padded, 0, n, axis=axis))
    return jnp.stack(grads)


def _shift(field, k, axis):
    """field[i + k] along axis, zero outside."""
    n = field.shape[axis]
    pad = [(0, 0)] * field.ndim
    pad[axis] = (max(-k, 0), max(k, 0))
    padded = jnp.pad(field, pad)
    return jax.lax.slice_in_dim(padded, max(k, 0), max(k, 0) + n, axis=axis)


@jax.jit
def column_area_weights(pressure, density, surface, zones):
    """1 / |n|_inf over the number of surface cells in the cell's zone segment
    along its dominant axis (within COLUMN_REACH cells)."""
    grads = _smoothed_normal(pressure, density)
    weight = surface_area_weights(pressure, density)
    dominant = jnp.argmax(jnp.abs(grads), axis=0)
    count = jnp.zeros(surface.shape, jnp.float32)
    for axis in range(3):
        c = surface.astype(jnp.float32)
        for sign in (1, -1):
            connected = jnp.ones(surface.shape, bool)
            for k in range(1, COLUMN_REACH + 1):
                connected = connected & _shift(zones, sign * k, axis)
                c = c + (connected & _shift(surface, sign * k, axis)).astype(jnp.float32)
        count = jnp.where(dominant == axis, c, count)
    return weight / jnp.maximum(count, 1.0)


def sky_anisotropy(pos, d_cell, r_true, f_meas, n_mu=8, n_phi=16):
    """Injected dissipation per unit shock area in equal-area sky bins, over
    f_meas: (bin map, robust spread, axis-bins / diagonal-bins ratio)."""
    r = np.linalg.norm(pos, axis=1)
    mu = pos[:, 2] / r
    phi = np.arctan2(pos[:, 1], pos[:, 0])
    b = (np.clip(((mu + 1) / 2 * n_mu).astype(int), 0, n_mu - 1) * n_phi
         + np.clip(((phi + np.pi) / (2 * np.pi) * n_phi).astype(int), 0, n_phi - 1))
    per_bin = np.bincount(b, weights=d_cell, minlength=n_mu * n_phi)
    per_bin = per_bin / (4 * np.pi * r_true**2 / (n_mu * n_phi)) / f_meas
    # orientation of each bin centre
    mu_c = (np.arange(n_mu) + 0.5) / n_mu * 2 - 1
    phi_c = (np.arange(n_phi) + 0.5) / n_phi * 2 * np.pi - np.pi
    mm, pp = np.meshgrid(mu_c, phi_c, indexing="ij")
    s = np.sqrt(1 - mm**2)
    ninf = np.max(np.abs(np.stack([s * np.cos(pp), s * np.sin(pp), mm])), axis=0).ravel()
    axis_like, diag_like = ninf > 0.9, ninf < 0.7
    ratio = per_bin[axis_like].mean() / per_bin[diag_like].mean()
    p16, p84 = np.percentile(per_bin, [16, 84])
    return per_bin.reshape(n_mu, n_phi), 0.5 * (p84 - p16) / np.median(per_bin), ratio


def _timed(fn, repeats=3):
    jax.block_until_ready(fn())
    t0 = time.time()
    for _ in range(repeats):
        out = jax.block_until_ready(fn())
    return out, (time.time() - t0) / repeats


def analyse(run_name, zeta):
    run = _run_with_snapshots(zeta)
    config, params = run["config"], run["params"]
    rv, hd = run["registered_variables"], run["helper_data"]
    cr_config, cr_params = config.cosmic_ray_grey_config, params.cosmic_ray_grey_params
    ref = np.load(B73_DIR / f"b7_3_N{NUM_CELLS}.npz")
    n = NUM_CELLS
    dx = config.grid_spacing
    centres = np.asarray(hd.geometric_centers) - 0.5
    results = {}
    for i_t, (t, state) in enumerate(zip(run["times"], run["states"])):
        if t < LATE_TIME:
            continue
        state = jnp.asarray(state)
        rho, p = state[rv.density_index], state[rv.pressure_index]
        r_true = float(ref[f"{run_name}_R_true"][i_t])
        d_meas = float(ref[f"{run_name}_D_meas"][i_t])
        m_true = float(ref[f"{run_name}_M_true"][i_t])
        f_meas = d_meas / (4 * np.pi * r_true**2)

        def ours():
            sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=cr_params.dsa_mach_min,
                                      **dsa_shock_finder_kwargs(cr_config))
            mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
            return sf, mach, flux, surface_area_weights(p, rho, sigma=cr_config.dsa_area_normal_smoothing)

        def hybrid():
            sf = find_shocks_hybrid(state, config, rv, hd, mach_min=cr_params.dsa_mach_min)
            mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
            return sf, mach, flux

        def v2(profile):
            return lambda: find_v2(state, config, rv, hd, mach_min=cr_params.dsa_mach_min,
                                   profile_aware_mach=profile)

        pipelines = {}
        (sf, mach, flux, w), t_ours = _timed(ours)
        pipelines["ours"] = (sf.shock_surface_cells, mach, flux, {"ninf": w}, t_ours)
        (sf_h, mach_h, flux_h), t_h = _timed(hybrid)
        pipelines["hyb"] = (sf_h.shock_surface_cells, mach_h, flux_h, sf_h.shock_zones, t_h)
        for name, profile in (("v2", False), ("v2prof", True)):
            res, t_v = _timed(v2(profile))
            pipelines[name] = (res.shock_surface_cells, res.mach_numbers, res.thermal_energy_flux,
                               res.shock_zones, t_v)
        w_ninf = surface_area_weights(p, rho, sigma=cr_config.dsa_area_normal_smoothing)
        for name in ("hyb", "v2", "v2prof"):
            surface, mach_v, flux_v, zones_v, t_v = pipelines[name]
            w_col, t_col = _timed(lambda: column_area_weights(p, rho, surface, zones_v))
            pipelines[name] = (surface, mach_v, flux_v,
                               {"unit": jnp.ones_like(w_ninf), "ninf": w_ninf, "col": w_col}, t_v + t_col)

        for name, (surface, mach_v, flux_v, weights, t_v) in pipelines.items():
            surface = np.asarray(surface)
            mach_v, flux_v = np.asarray(mach_v), np.asarray(flux_v)
            injecting = flux_v > 0
            pos = centres[injecting]
            for w_name, w in weights.items():
                w = np.asarray(w)
                key = f"{name}_{w_name}"
                d_cell = flux_v[injecting] * w[injecting] * dx**2
                sky, aniso, axis_diag = sky_anisotropy(pos, d_cell, r_true, f_meas)
                entry = results.setdefault(key, {k: [] for k in (
                    "t", "D_over_meas", "area_ratio", "flux_ratio", "mach_ratio", "aniso",
                    "axis_diag", "n_surface", "n_inject", "time", "sky")})
                area = float(np.sum(w[surface])) * dx**2 / (4 * np.pi * r_true**2)
                d_ratio = float(d_cell.sum()) / d_meas
                entry["t"].append(float(t))
                entry["D_over_meas"].append(d_ratio)
                entry["area_ratio"].append(area)
                entry["flux_ratio"].append(d_ratio / area)
                entry["mach_ratio"].append(float(np.median(mach_v[injecting])) / m_true)
                entry["aniso"].append(aniso)
                entry["axis_diag"].append(axis_diag)
                entry["n_surface"].append(int(surface.sum()))
                entry["n_inject"].append(int(injecting.sum()))
                entry["time"].append(t_v)
                entry["sky"].append(sky)
        print(f"[N={n} {run_name} t={t:.3f}] " + ", ".join(
            f"{k} {v['D_over_meas'][-1]:.3f}/{v['aniso'][-1]:.3f}" for k, v in results.items()), flush=True)

    # complete injection step (finder + weight + spread + cap) on the last snapshot,
    # ours vs the hybrid with column weights monkeypatched into the injection
    def inject():
        return inject_crs_at_shocks(state, config, params, rv, hd, current_time=float(t), dt=1e-5)

    _, t_inject_ours = _timed(inject)
    original = (cr_grey_injection.find_shocks_pfrommer, cr_grey_injection.surface_area_weights)
    zones_holder = {}

    def hybrid_finder(*args, **kwargs):
        sf = find_shocks_hybrid(*args, **kwargs)
        zones_holder["sf"] = sf
        return sf

    def hybrid_weights(pressure, density, sigma=2.0):
        sf = zones_holder["sf"]
        return column_area_weights(pressure, density, sf.shock_surface_cells, sf.shock_zones)

    cr_grey_injection.find_shocks_pfrommer = hybrid_finder
    cr_grey_injection.surface_area_weights = hybrid_weights
    try:
        _, t_inject_hyb = _timed(inject)
    finally:
        cr_grey_injection.find_shocks_pfrommer, cr_grey_injection.surface_area_weights = original

    flat = {"num_cells": n, "t_inject_ours": t_inject_ours, "t_inject_hyb": t_inject_hyb}
    for key, entry in results.items():
        for field, values in entry.items():
            flat[f"{key}__{field}"] = np.asarray(values)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"dsa_{run_name}_N{n}.npz", **flat)
    print(f"[N={n} {run_name}] injection step: ours {t_inject_ours:.3f} s, hybrid+col {t_inject_hyb:.3f} s",
          flush=True)


if __name__ == "__main__":
    for run_name, zeta in (("control", 0.0), ("dsa", 0.1)):
        analyse(run_name, zeta)
