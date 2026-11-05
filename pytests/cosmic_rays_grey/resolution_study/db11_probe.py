"""DB11 diagnostic: where does the cr_wind_bubble DSA run's CR energy (~2x the injected) come from? GPU only.

Runs the cr_wind_bubble DSA setup in segments and records
* per step: the realized injection (``delta e_cr dx^3`` of ``inject_crs_at_shocks``) binned in radius,
  and a centred-difference estimate of the adiabatic work ``-P_cr div(v) dV dt`` on the pre-injection
  state, binned the same way;
* per snapshot: E_cr and the radial profiles (bin means) of rho, P_gas, e_cr, v_r.
``--off-after f`` sets ``dsa_efficiency = 0`` from ``f t_end`` on (do existing CRs gain energy?).

Output ``RAW/db11/<name>.npz``. Usage:
    python db11_probe.py --name def [--n 128] [--off-after 0.5] [--snapshots 24] [--vred 0] [--kappa 0.1] [key=value ...]
"""

import argparse
import sys

import common
from common import RAW, code_stamp, require_pinned_gpu, run_segments

import numpy as np
import jax
import jax.numpy as jnp

import astronomix._modules._iteration_level_updates as _ilu
import cr_wind_bubble as m
from astronomix import (CARTESIAN, FINITE_VOLUME, HLLC, MINMOD, SimulationConfig, SimulationParams,
                        WindParams, construct_primitive_state, finalize_config, get_helper_data,
                        get_registered_variables)
from astronomix.option_classes import WindConfig

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--n", type=int, default=m.NUM_CELLS)
ap.add_argument("--off-after", type=float, default=None)
ap.add_argument("--snapshots", type=int, default=24)
ap.add_argument("--vred", type=float, default=None, help="CosmicRayGreyParams.reduced_streaming_speed")
ap.add_argument("--kappa", type=float, default=None,
                help="CosmicRayGreyParams.diffusion_coefficient (code units); also sets diffusive_relaxation")
args, rest = ap.parse_known_args()
overrides = {} if args.kappa is None else {"diffusive_relaxation": True}
for item in rest:
    key, value = item.split("=")
    overrides[key] = {"0": False, "1": True}.get(value, value)

gpu = require_pinned_gpu()
cu = m.CODE_UNITS
N = args.n
dx = m.BOX_SIZE / N
NB = int(np.ceil(m.BOX_SIZE / 2 * np.sqrt(3) / dx)) + 1

config = SimulationConfig(
    progress_bar=False, geometry=CARTESIAN, solver_mode=FINITE_VOLUME, riemann_solver=HLLC,
    limiter=MINMOD, dimensionality=3, box_size=m.BOX_SIZE, num_cells=N, exact_end_time=True,
    wind_config=WindConfig(stellar_wind=True, num_injection_cells=m.NUM_INJECTION_CELLS * N // 128),
    cosmic_ray_grey_config=m.CosmicRayGreyConfig(grey_cosmic_rays=True, diffusive_shock_acceleration=True,
                                                 **overrides),
)
hd = get_helper_data(config)
rv = get_registered_variables(config)
shape = (N, N, N)
state = construct_primitive_state(
    config=config, registered_variables=rv,
    density=jnp.ones(shape) * m.RHO_AMBIENT.to(cu.code_density).value,
    velocity_x=jnp.zeros(shape), velocity_y=jnp.zeros(shape), velocity_z=jnp.zeros(shape),
    gas_pressure=jnp.ones(shape) * m.P_AMBIENT.to(cu.code_pressure).value,
)
config = finalize_config(config, state.shape)
mdot = m.M_DOT.to(cu.code_mass / cu.code_time).value
vinf = m.V_INF.to(cu.code_velocity).value
params = SimulationParams(
    t_end=m.T_END, gamma=m.GAMMA,
    wind_params=WindParams(wind_mass_loss_rates=jnp.array([mdot]), wind_final_velocities=jnp.array([vinf])),
    cosmic_ray_grey_params=m.CosmicRayGreyParams(
        dsa_efficiency=m.DSA_EFFICIENCY, dsa_mach_min=m.DSA_MACH_MIN,
        **({} if args.vred is None else {"reduced_streaming_speed": args.vred}),
        **({} if args.kappa is None else {"diffusion_coefficient": args.kappa})),
)
L_w = 0.5 * mdot * vinf**2

xc = (np.arange(N) + 0.5) * dx - m.BOX_SIZE / 2
X, Y, Z = np.meshgrid(xc, xc, xc, indexing="ij")
r = jnp.asarray(hd.r)
assert r.shape == shape and np.allclose(np.asarray(r), np.sqrt(X**2 + Y**2 + Z**2), atol=1e-6 * dx)
bin_index = jnp.clip((r / dx).astype(jnp.int32), 0, NB - 1).reshape(-1)
r_np = np.asarray(r)
bin_np = np.clip((r_np / dx).astype(int), 0, NB - 1).reshape(-1)
counts = np.bincount(bin_np, minlength=NB).astype(float)
gcr = 4.0 / 3.0


def binned(field):
    g = (field.shape[0] - N) // 2  # the time loop's state carries ghost cells
    if g:
        field = field[g:-g, g:-g, g:-g]
    return jnp.bincount(bin_index, weights=field.reshape(-1), length=NB)


steps = []
original = _ilu.inject_crs_at_shocks


def wrapped(primitive_state, config, params, registered_variables, helper_data, current_time, dt):
    out = original(primitive_state, config, params, registered_variables, helper_data, current_time, dt)
    i = registered_variables.cosmic_ray_e_index
    vol = config.grid_spacing**3
    inj = binned((out[i] - primitive_state[i]) * vol)
    v = registered_variables.velocity_index
    div_v = sum(jnp.gradient(primitive_state[c], config.grid_spacing, axis=a) for a, c in enumerate((v.x, v.y, v.z)))
    work = binned(-(gcr - 1.0) * primitive_state[i] * div_v * vol * dt)
    jax.debug.callback(lambda t, d, a, b: steps.append((float(t), float(d), np.asarray(a), np.asarray(b))),
                       current_time, dt, inj, work)
    return out


_ilu.inject_crs_at_shocks = wrapped

times = np.geomspace(0.02 * m.T_END, m.T_END, args.snapshots)
off = None if args.off_after is None else args.off_after * m.T_END
if off is not None:
    times = np.unique(np.append(times, off))
print(f"db11 {args.name} N {N} gpu {gpu} overrides {overrides} off_after {args.off_after} vred {args.vred} kappa {args.kappa}", flush=True)

snaps = []


def on_snapshot(k, t, s, wall):
    rho, p, ecr = (np.asarray(s[j]) for j in (rv.density_index, rv.pressure_index, rv.cosmic_ray_e_index))
    v = rv.velocity_index
    vr = sum(np.asarray(s[c]) * np.asarray(x) for c, x in zip((v.x, v.y, v.z), (X, Y, Z))) / np.maximum(r_np, 1e-30)
    prof = {q: np.bincount(bin_np, weights=f.reshape(-1), minlength=NB) / np.maximum(counts, 1)
            for q, f in (("rho", rho), ("p", p), ("ecr", ecr), ("vr", vr))}
    E_cr = float(ecr.sum() * dx**3)
    E_tot = float((p / (m.GAMMA - 1) + 0.5 * rho * sum(np.asarray(s[c])**2 for c in (v.x, v.y, v.z)) + ecr).sum() * dx**3)
    inj = sum(st[2].sum() for st in steps)
    snaps.append(dict(t=t, E_cr=E_cr, E_tot=E_tot, inj=inj, **prof))
    print(f"t/t_end {t / m.T_END:.3f}  E_cr {E_cr / (L_w * t):.4f} L t  injected {inj / (L_w * t):.4f} L t  "
          f"ratio {E_cr / max(inj, 1e-30):.3f}  ({wall:.0f} s)", flush=True)


# run_segments with the efficiency switched off after ``off``
t_prev = 0.0
p_cur = params
for k, t in enumerate(times):
    s_params = p_cur._replace(t_start=t_prev, t_end=float(t))
    state = run_segments(state, config, s_params, rv, [float(t)], lambda i, tt, s, w: on_snapshot(k, tt, s, w))
    if off is not None and t >= off - 1e-12 * m.T_END and p_cur is params:
        p_cur = params._replace(cosmic_ray_grey_params=params.cosmic_ray_grey_params._replace(dsa_efficiency=0.0))
    t_prev = float(t)

st_t = np.array([s[0] for s in steps]); order = np.argsort(st_t)
out = RAW / "db11"
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{args.name}.npz",
         step_t=st_t[order], step_dt=np.array([s[1] for s in steps])[order],
         step_inj=np.array([s[2] for s in steps])[order], step_work=np.array([s[3] for s in steps])[order],
         snap_t=np.array([s["t"] for s in snaps]), E_cr=np.array([s["E_cr"] for s in snaps]),
         E_tot=np.array([s["E_tot"] for s in snaps]), inj=np.array([s["inj"] for s in snaps]),
         **{f"prof_{q}": np.array([s[q] for s in snaps]) for q in ("rho", "p", "ecr", "vr")},
         r_bins=(np.arange(NB) + 0.5) * dx, dx=dx, L_w=L_w, t_end=m.T_END, N=N,
         final_state=np.asarray(state), stamp=str(code_stamp()), overrides=str(overrides))
print("saved", out / f"{args.name}.npz", flush=True)
