"""Stationary CWB with a fixed 2-cell wind injection radius
(``run_cwb(N, cosmic_rays=False, num_injection_cells=2)``), pure hydro.

With the default ``num_cells // 32`` the injection radius is a fixed fraction of the
box, so the pre-shock Mach number (the Chevalier & Clegg 1985 thermal-wind value at
shock distance / injection radius) is ~10 at every N. With a fixed 2-cell radius it
should grow as N^(2/3): CC85 predicts ~11 / ~16 / ~28 on the star-1 apex at
N = 64 / 128 / 256 (FIXES_TODO.md, 2026-10-14).

Usage (one job per GPU, pin by UUID)::

    source ~/venv_grav_source/bin/activate
    cd pytests/shock_finder3D
    CUDA_VISIBLE_DEVICES=GPU-<uuid> PYTHONUNBUFFERED=1 setsid nohup python -u \
        _regen_fixed_injection.py 256 > /export/scratch/nknoell/cwb_separation/logs/ninj2_256.log 2>&1 < /dev/null &

Writes ``figures/cwb_fixed_injection/cwb_shocked_cells_{2d,3d}_ninj2_<N>.png`` and the
raw state to ``RAW_DIR``.
"""

import sys
from pathlib import Path

import autocvd

autocvd.autocvd = lambda *a, **k: [0]

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _cwb_setup import run_cwb  # noqa: E402
from cwb_shock_finder import plot_shocked_cells_2d, plot_shocked_cells_3d  # noqa: E402

RAW_DIR = Path("/export/scratch/nknoell/cwb_separation/data")
FIG_DIR = HERE / "figures" / "cwb_fixed_injection"
NUM_INJECTION_CELLS = 2


def main(num_cells):
    run = run_cwb(num_cells, cosmic_rays=False, num_injection_cells=NUM_INJECTION_CELLS)
    sf_result = run["sf_result"]
    surface = np.asarray(sf_result.shock_surface_cells)
    mach = np.asarray(sf_result.mach_numbers)
    m = mach[surface]
    p10, p50, p90 = np.percentile(m, [10, 50, 90])
    print(
        f"[cwb N={num_cells} ninj={NUM_INJECTION_CELLS}] surface cells={m.size}, "
        f"Mach median={p50:.2f}, p10={p10:.2f}, p90={p90:.2f}, max={m.max():.2f}, "
        f"nan={bool(np.isnan(np.asarray(run['state'])).any())}",
        flush=True,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    title = f"Stationary CWB, N={num_cells}, fixed {NUM_INJECTION_CELLS}-cell wind injection (pure hydro)"
    plot_shocked_cells_3d(run, num_cells, filename=f"cwb_shocked_cells_3d_ninj2_{num_cells}.png",
                          title=title, fig_dir=FIG_DIR)
    plot_shocked_cells_2d(run, num_cells, filename=f"cwb_shocked_cells_2d_ninj2_{num_cells}.png",
                          title=title, fig_dir=FIG_DIR)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rv = run["registered_variables"]
    np.savez_compressed(
        RAW_DIR / f"ninj2_N{num_cells}.npz",
        state=np.asarray(run["state"]), surface=surface, mach=mach,
        density_index=rv.density_index, pressure_index=rv.pressure_index,
        vx_index=rv.velocity_index.x, vy_index=rv.velocity_index.y, vz_index=rv.velocity_index.z,
        p_amb=run["params"].minimum_pressure * 1e3, rho_amb=run["params"].minimum_density * 1e3,
    )
    print(f"figures in {FIG_DIR}, raw state in {RAW_DIR}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
