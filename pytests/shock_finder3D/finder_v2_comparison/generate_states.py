"""Final states for the shock_finder3D vs shock_finder_v2 comparison.

Runs the shared Sedov blast (``_sedov_setup.run_sedov``) at N = 64, 128, 256
and the current default stationary CWB (``_cwb_setup.run_cwb``, unsplit, grey
CRs with DSA 0.1) at N = 128, and saves the hydro fields both finders read
(density, gas pressure, velocity; plus e_cr for the CWB) to
``/export/scratch/nknoell/finder_v2_comparison/``. ``compare_finders.py``
then runs both finders on exactly the same states.

Usage: python generate_states.py [sedov64 sedov128 sedov256 cwb128 ...]
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("/export/scratch/nknoell/finder_v2_comparison")


def _save(name, state, registered_variables, extra=None):
    rv = registered_variables
    fields = dict(
        density=np.asarray(state[rv.density_index], dtype=np.float32),
        pressure=np.asarray(state[rv.pressure_index], dtype=np.float32),
        vx=np.asarray(state[rv.velocity_index.x], dtype=np.float32),
        vy=np.asarray(state[rv.velocity_index.y], dtype=np.float32),
        vz=np.asarray(state[rv.velocity_index.z], dtype=np.float32),
    )
    fields.update(extra or {})
    np.savez(DATA_DIR / f"{name}.npz", **fields)
    print(f"saved {name}", flush=True)


def main(jobs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        if job.startswith("sedov"):
            from _sedov_setup import run_sedov
            run = run_sedov(int(job[len("sedov"):]))
            _save(job, run["state"], run["registered_variables"])
        elif job.startswith("cwb"):
            from _cwb_setup import run_cwb
            run = run_cwb(int(job[len("cwb"):]))
            rv = run["registered_variables"]
            _save(job, run["state"], rv,
                  extra=dict(e_cr=np.asarray(run["state"][rv.cosmic_ray_e_index], dtype=np.float32)))
        else:
            raise ValueError(job)


if __name__ == "__main__":
    main(sys.argv[1:] or ["sedov64", "sedov128", "sedov256", "cwb128"])
