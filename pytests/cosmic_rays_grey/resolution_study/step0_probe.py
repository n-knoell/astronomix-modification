"""Step 0 checks of the resolution study (PLAN.md 0.3 / 0.4 / segmented-run check). GPU only.

Usage (pinned: ``CUDA_VISIBLE_DEVICES=GPU-<uuid>``):

* ``python step0_probe.py restrict``: unit tests of ``common.restrict`` (uniform
  and linear states exact, mass / momentum / energy conserved to round-off).
* ``python step0_probe.py memory sedov|cwb <N>``: one DSA run for two short
  segments (compile + timed), then the finder + DSA flux + area weights on the
  final state. Peak GPU memory, s/step and an estimate of the full-run wall
  time; an out-of-memory error is recorded, not raised.
* ``python step0_probe.py segcheck``: 48^3 S-PE DSA run to ``T_END`` once in
  one segment and once in 10, both with the injection accumulator: final-state
  differences, accumulated injection vs box ``E_cr``, and whether later
  segments recompile.

Results: ``RAW/step0/<mode>[_<case>_N<n>].json`` and stdout.
"""

import json
import sys
import time
import traceback

import common
from common import RAW, InjectionAccumulator, peak_gpu_bytes, require_pinned_gpu, restrict, run_segments

import numpy as np
import jax
import jax.numpy as jnp

OUT = RAW / "step0"


def _dump(name, result):
    OUT.mkdir(parents=True, exist_ok=True)
    result = {**result, "stamp": common.code_stamp(), "gpu": require_pinned_gpu()}
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)


def check_restrict():
    from setups import build_sedov_pe
    state, config, params, hd, rv = build_sedov_pe(32, 0.1)
    gamma = params.gamma
    results = {}
    # uniform state: exact
    uni = jnp.ones_like(state).at[rv.velocity_index.x].set(0.3)
    r = restrict(uni, rv, gamma, 4)
    results["uniform_max_abs_err"] = float(jnp.max(jnp.abs(r - uni[:, :8, :8, :8])))
    # density linear in x, uniform v and p: block mean of the linear profile = its value at the block centre
    x = jnp.arange(32.0)
    lin = uni.at[rv.density_index].set(1.0 + 0.01 * x[:, None, None] * jnp.ones((32, 32, 32)))
    r = restrict(lin, rv, gamma, 2)
    expect = 1.0 + 0.01 * (2 * jnp.arange(16.0) + 0.5)
    results["linear_density_max_abs_err"] = float(jnp.max(jnp.abs(r[rv.density_index] - expect[:, None, None])))
    # random positive state: conservation (block sums times volume)
    key = jax.random.PRNGKey(0)
    rnd = state.at[rv.density_index].set(0.5 + jax.random.uniform(key, (32, 32, 32)))
    for i, k in zip((rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z), range(1, 4)):
        rnd = rnd.at[i].set(jax.random.normal(jax.random.PRNGKey(k), (32, 32, 32)))
    rnd = rnd.at[rv.pressure_index].set(0.1 + jax.random.uniform(jax.random.PRNGKey(9), (32, 32, 32)))
    rnd = rnd.at[rv.cosmic_ray_e_index].set(jax.random.uniform(jax.random.PRNGKey(7), (32, 32, 32)))

    def totals(s, f):
        rho = s[rv.density_index]
        v = [s[i] for i in (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)]
        e = s[rv.pressure_index] / (gamma - 1) + 0.5 * rho * sum(u * u for u in v)
        vol = float(f) ** 3
        return np.array([float(jnp.sum(rho)) * vol, *[float(jnp.sum(rho * u)) * vol for u in v],
                         float(jnp.sum(e)) * vol, float(jnp.sum(s[rv.cosmic_ray_e_index])) * vol])

    fine = totals(rnd, 1)
    for f in (2, 4):
        coarse = totals(restrict(rnd, rv, gamma, f), f)
        results[f"conservation_rel_err_f{f}"] = float(np.max(np.abs(coarse - fine) / np.maximum(np.abs(fine), 1.0)))
    results["pass"] = bool(results["uniform_max_abs_err"] < 1e-6 and results["linear_density_max_abs_err"] < 1e-5
                           and max(results["conservation_rel_err_f2"], results["conservation_rel_err_f4"]) < 1e-4)
    _dump("restrict", results)


def probe_memory(case, n):
    from setups import build_cwb, build_sedov_pe
    InjectionAccumulator.install(coarse_cells=16)
    result = {"case": case, "N": n}
    try:
        if case == "sedov":
            state, config, params, hd, rv = build_sedov_pe(n, 0.1)
            t_full = float(params.t_end)
            times = [2e-4, 4e-4]
        else:
            state, config, params, hd, rv = build_cwb(n, 0.1)
            t_full = float(params.t_end)
            times = [0.01 * t_full, 0.02 * t_full]
        acc = InjectionAccumulator().activate()
        walls, steps = [], []

        def on_snap(i, t, s, wall):
            walls.append(wall)
            steps.append(len(acc.steps))

        state = run_segments(state, config, params, rv, times, on_snap)
        n2 = steps[1] - steps[0]
        result.update(segment_walls=walls, steps=steps, peak_run_GB=peak_gpu_bytes() / 1e9,
                      s_per_step=walls[1] / max(n2, 1))
        _, _, dt = acc.series()
        result["dt_last"] = float(dt[-1])
        # finder + DSA diagnostics on the final state (the per-snapshot analysis)
        from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
        from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
            dsa_area_weights, dsa_mach_and_flux, dsa_shock_finder_kwargs)
        cr_config = config.cosmic_ray_grey_config
        t0 = time.time()
        sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=params.cosmic_ray_grey_params.dsa_mach_min,
                                  **dsa_shock_finder_kwargs(cr_config))
        mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
        w = dsa_area_weights(state, sf, cr_config, rv)
        jax.block_until_ready((mach, flux, w))
        result.update(analysis_s=time.time() - t0, peak_with_analysis_GB=peak_gpu_bytes() / 1e9)
        # rough full-run estimate: dt at this early time is a lower bound for the Sedov blast
        # (it decelerates), the CWB dt is set by the steady wind speed
        result["est_full_run_h_upper"] = (t_full / result["dt_last"]) * result["s_per_step"] / 3600
        result["fits"] = True
    except Exception as exc:  # out of memory and friends: recorded
        result.update(fits=False, error=f"{type(exc).__name__}: {str(exc)[:500]}",
                      peak_GB=peak_gpu_bytes() / 1e9)
        traceback.print_exc()
    _dump(f"memory_{case}_N{n}", result)


def segcheck():
    from setups import build_sedov_pe
    import cr_sedov_taylor as item7
    InjectionAccumulator.install(coarse_cells=16)
    out = {}
    finals = {}
    for name, times in (("continuous", [item7.T_END]), ("segmented", list(np.linspace(0.007, item7.T_END, 10)))):
        state, config, params, hd, rv = build_sedov_pe(48, 0.1)
        acc = InjectionAccumulator().activate()
        walls = []
        final = run_segments(state, config, params, rv, times, lambda i, t, s, w: walls.append(w))
        e_cr = float(jnp.sum(final[rv.cosmic_ray_e_index])) * config.grid_spacing**3
        finals[name] = np.asarray(final)
        out[name] = {"segment_walls": walls, "steps": len(acc.steps), "injected": acc.total(), "E_cr_box": e_cr,
                     "E_cr_box_over_injected": e_cr / acc.total()}
    a, b = finals["continuous"], finals["segmented"]
    rv_e = rv.cosmic_ray_e_index
    out["final_rel_diff"] = {
        "density_L1": float(np.abs(a[rv.density_index] - b[rv.density_index]).sum() / np.abs(a[rv.density_index]).sum()),
        "pressure_L1": float(np.abs(a[rv.pressure_index] - b[rv.pressure_index]).sum() / np.abs(a[rv.pressure_index]).sum()),
        "e_cr_L1": float(np.abs(a[rv_e] - b[rv_e]).sum() / np.abs(a[rv_e]).sum()),
        "E_cr_box": out["segmented"]["E_cr_box"] / out["continuous"]["E_cr_box"] - 1,
        "injected": out["segmented"]["injected"] / out["continuous"]["injected"] - 1,
    }
    w = out["segmented"]["segment_walls"]
    out["recompiles_after_first_segment"] = bool(max(w[1:]) > 5 * np.median(w[1:]))
    _dump("segcheck", out)


if __name__ == "__main__":
    print("GPU", require_pinned_gpu(), flush=True)
    mode = sys.argv[1]
    if mode == "restrict":
        check_restrict()
    elif mode == "memory":
        probe_memory(sys.argv[2], int(sys.argv[3]))
    elif mode == "segcheck":
        segcheck()
    else:
        raise SystemExit(__doc__)
