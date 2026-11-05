"""Shared infrastructure of the Sedov / CWB resolution study (PLAN.md Step 0).

* paths: raw data under ``RAW`` (``/export/scratch/nknoell/resolution_study``,
  override with ``RES_STUDY_RAW``), figures in ``PICS``; the Phase-B raw-output
  env vars are redirected below ``RAW`` so no Phase-B scratch file is touched.
* ``require_pinned_gpu``: the run must be pinned to one GPU by UUID
  (``CUDA_VISIBLE_DEVICES=GPU-...``) and JAX must see it.
* ``code_stamp``: git hash plus a hash of the working-tree diff, stored in
  every output file.
* ``run_segments``: integrates from snapshot time to snapshot time
  (``params.t_start`` / ``t_end``) and hands each state to a host-side
  callback, so no snapshot stack sits in GPU memory (256^3 with CRs).
* ``InjectionAccumulator``: wraps ``inject_crs_at_shocks`` inside the time
  loop and sums the realized injection (``delta e_cr dx^3`` after the cap) of
  every step on the host, in total and on a coarse block map, through
  ``jax.debug.callback``. No library code changes. Optionally (extension D) also the
  centred CR compression work ``-P_cr div(v) dV dt`` per step, binned by a given cell
  index (radial bins, or one bin).
* ``restrict``: conservative block average of a primitive state (PLAN.md
  principle 5).
"""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import autocvd

autocvd.autocvd = lambda *a, **k: [0]  # pinned by CUDA_VISIBLE_DEVICES, autocvd hangs here

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RAW = Path(os.environ.get("RES_STUDY_RAW", "/export/scratch/nknoell/resolution_study"))
PICS = HERE / "pics"
for _var, _sub in (("CR_PHASEB_RAW", "phaseb_raw"), ("CR_PHASEB_DB7_RAW", "phaseb_raw/db7"),
                   ("CR_PHASEB_DB4_RAW", "phaseb_raw/db4")):
    os.environ.setdefault(_var, str(RAW / _sub))
sys.path.insert(0, str(HERE.parent))  # pytests/cosmic_rays_grey
sys.path.insert(0, str(HERE.parents[1] / "shock_finder3D"))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import astronomix._modules._iteration_level_updates as _ilu  # noqa: E402
from astronomix import time_integration  # noqa: E402


def require_pinned_gpu():
    """The UUID this process is pinned to; raises unless pinned to one GPU that JAX uses."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd.startswith("GPU-") or "," in cvd:
        raise RuntimeError(f"pin one GPU by UUID (CUDA_VISIBLE_DEVICES={cvd!r})")
    devices = jax.devices()
    if devices[0].platform != "gpu":
        raise RuntimeError(f"JAX runs on {devices[0].platform}, not on the GPU")
    return cvd


def code_stamp():
    """Git hash, hash of ``git diff HEAD`` (uncommitted state) and the date."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout
    diff = git("diff", "HEAD")
    return {"git_head": git("rev-parse", "HEAD").strip(),
            "git_diff_sha1": hashlib.sha1(diff.encode()).hexdigest(),
            "git_diff_stat": git("diff", "--shortstat", "HEAD").strip(),
            "date": time.strftime("%Y-%m-%d %H:%M:%S")}


def peak_gpu_bytes():
    """Peak device memory in use so far (XLA allocator statistics)."""
    stats = jax.devices()[0].memory_stats() or {}
    return int(stats.get("peak_bytes_in_use", -1))


def run_segments(state, config, params, registered_variables, times, on_snapshot):
    """Integrate ``state`` from ``params.t_start`` through ``times`` (ascending),
    calling ``on_snapshot(index, time, state, wall_seconds)`` after each segment.

    ``config`` must have ``exact_end_time=True`` and ``return_snapshots=False``
    (``time_integration`` then returns the final state). The segment ends become
    step ends, otherwise the run is the continuous one (no hidden loop state: no
    forcing, no SN driving in this study). Returns the final state."""
    if not config.exact_end_time or config.return_snapshots:
        raise ValueError("run_segments needs exact_end_time=True and return_snapshots=False")
    t_prev = float(params.t_start)
    for i, t in enumerate(times):
        t = float(t)
        if t <= t_prev:
            raise ValueError(f"snapshot times must increase past t_start ({t} <= {t_prev})")
        start = time.time()
        state = time_integration(state, config, params._replace(t_start=t_prev, t_end=t),
                                 registered_variables)
        state.block_until_ready()
        jax.effects_barrier()  # flush the accumulator callbacks of this segment
        on_snapshot(i, t, state, time.time() - start)
        t_prev = t
    return state


class InjectionAccumulator:
    """Host-side sum of the realized DSA injection, step by step.

    ``install(coarse_cells)`` once per process, before the first
    ``time_integration`` call (the wrapper is traced into the time loop; a jit
    cache built before would keep the unwrapped call). ``coarse_cells`` (per
    axis; used where it divides N) sets the block-map resolution, ``None``
    keeps the total only. ``activate()`` on a new instance before each run: the
    compiled loop calls the class-level dispatcher, which forwards to whichever
    instance is active at run time (the jit cache is shared between runs of the
    same setup). Each step's injected energy is ``sum(delta e_cr) dx^3`` of
    ``inject_crs_at_shocks``' output, i.e. after the positivity cap, before the
    CRs are transported."""

    _original = None
    _active = None
    _coarse_cells = None
    _work = None

    @classmethod
    def install(cls, coarse_cells=None, work_bins=None):
        """``work_bins = (num_cells, bin_index, num_bins)`` (``bin_index``: int array over the
        N^3 interior cells) also accumulates the centred compression work
        ``-(gamma_cr - 1) e_cr div(v) dx^3 dt`` per bin and step (as ``db11_probe.py``; the
        estimate the DB11 budget used, not the scheme's own discrete work)."""
        if cls._original is not None:
            if coarse_cells != cls._coarse_cells:
                raise RuntimeError("InjectionAccumulator already installed with another coarse_cells")
            return
        cls._original = _ilu.inject_crs_at_shocks
        cls._coarse_cells = coarse_cells
        if work_bins is not None:
            n, idx, nb = work_bins
            cls._work = (int(n), jnp.asarray(idx, dtype=jnp.int32).reshape(-1), int(nb))

        def wrapped(primitive_state, config, params, registered_variables, helper_data,
                    current_time, dt):
            out = cls._original(primitive_state, config, params, registered_variables,
                                helper_data, current_time, dt)
            i = registered_variables.cosmic_ray_e_index
            delta = (out[i] - primitive_state[i]) * config.grid_spacing**config.dimensionality
            m = cls._coarse_cells
            if m is not None and config.dimensionality == 3 and delta.shape[0] % m == 0:
                f = delta.shape[0] // m
                coarse = delta.reshape(m, f, m, f, m, f).sum(axis=(1, 3, 5))
            else:
                coarse = jnp.zeros(())
            if cls._work is not None:
                n, idx, nb = cls._work
                g = (primitive_state.shape[1] - n) // 2  # the time loop's state carries ghost cells
                ps = primitive_state[:, g:g + n, g:g + n, g:g + n] if g else primitive_state
                v = registered_variables.velocity_index
                h = config.grid_spacing
                div_v = sum(jnp.gradient(ps[c], h, axis=a) for a, c in enumerate((v.x, v.y, v.z)))
                w = -(params.cosmic_ray_grey_params.gamma_cr - 1.0) * ps[i] * div_v * h**3 * dt
                work = jnp.bincount(idx, weights=w.reshape(-1), length=nb)
            else:
                work = jnp.zeros(())
            jax.debug.callback(cls._dispatch, current_time, dt, jnp.sum(delta), coarse, work)
            return out

        _ilu.inject_crs_at_shocks = wrapped

    @classmethod
    def _dispatch(cls, t, dt, total, coarse, work):
        if cls._active is not None:
            cls._active._add(t, dt, total, coarse, work)

    def __init__(self):
        self.reset()

    def reset(self):
        self.steps = []  # (t, dt, injected energy of the step)
        self.coarse = None
        self.work = []  # per step: binned compression work (only with work_bins)

    def activate(self):
        if InjectionAccumulator._original is None:
            raise RuntimeError("InjectionAccumulator.install() first")
        self.reset()
        InjectionAccumulator._active = self
        return self

    def _add(self, t, dt, total, coarse, work=None):
        self.steps.append((float(t), float(dt), float(total)))
        if work is not None and np.ndim(work) == 1:
            self.work.append((float(t), np.asarray(work, dtype=np.float64)))
        if np.ndim(coarse) == 3:
            c = np.asarray(coarse, dtype=np.float64)
            self.coarse = c if self.coarse is None else self.coarse + c

    def total(self):
        return float(sum(s[2] for s in self.steps))

    def work_series(self):
        """Arrays t (step start) and binned work (steps x bins), time-ordered; None without work_bins."""
        if not self.work:
            return None, None
        order = np.argsort([w[0] for w in self.work])
        return (np.array([self.work[k][0] for k in order]), np.stack([self.work[k][1] for k in order]))

    def series(self):
        """Arrays t (step start), dt, injected energy per step (time-ordered)."""
        a = np.array(sorted(self.steps), dtype=np.float64).reshape(-1, 3)
        return a[:, 0], a[:, 1], a[:, 2]


def restrict(primitive_state, registered_variables, gamma, factor):
    """Block-average a 3D primitive state by ``factor`` per axis, conservatively:
    mass, momentum and total gas energy are averaged and converted back; every
    other row (e_cr, F_cr, passive scalars) is volume-averaged as it is."""
    rv = registered_variables
    n = primitive_state.shape[1]
    if n % factor:
        raise ValueError(f"{n} cells not divisible by {factor}")
    m = n // factor

    def avg(field):
        return field.reshape(m, factor, m, factor, m, factor).mean(axis=(1, 3, 5))

    rho = primitive_state[rv.density_index]
    vel_idx = (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)
    vel = [primitive_state[i] for i in vel_idx]
    p = primitive_state[rv.pressure_index]
    energy = p / (gamma - 1.0) + 0.5 * rho * sum(v * v for v in vel)
    rho_c = avg(rho)
    mom_c = [avg(rho * v) for v in vel]
    vel_c = [mv / rho_c for mv in mom_c]
    p_c = (gamma - 1.0) * (avg(energy) - 0.5 * sum(mv * mv for mv in mom_c) / rho_c)
    out = jnp.stack([avg(primitive_state[i]) for i in range(primitive_state.shape[0])])
    out = out.at[rv.density_index].set(rho_c).at[rv.pressure_index].set(p_c)
    for i, v in zip(vel_idx, vel_c):
        out = out.at[i].set(v)
    return out


def save_npz(path, **arrays):
    """``np.savez`` with the code stamp, creating the directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = code_stamp()
    np.savez(path, **arrays, **{f"stamp_{k}": v for k, v in stamp.items()})
    return path
