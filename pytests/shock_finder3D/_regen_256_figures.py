"""Regenerate the N=256 shock-finder figures with the extended sampler
(``mach_sampling_adaptive=True, mach_sampling_extend=True``, already the
setting in ``_sedov_setup.py`` and ``_cwb_setup.py``).

Usage (one job per GPU, pin by UUID; the wrapper no-ops ``autocvd`` so the
pin is not overwritten)::

    source ~/venv_grav_source/bin/activate
    cd pytests/shock_finder3D
    CUDA_VISIBLE_DEVICES=GPU-<uuid> PYTHONUNBUFFERED=1 setsid nohup python -u \
        _regen_256_figures.py sedov > /export/scratch/nknoell/shock_finder_figs/sedov_256.log 2>&1 < /dev/null &
    CUDA_VISIBLE_DEVICES=GPU-<uuid> PYTHONUNBUFFERED=1 setsid nohup python -u \
        _regen_256_figures.py cwb > /export/scratch/nknoell/shock_finder_figs/cwb_256.log 2>&1 < /dev/null &

Outputs:

* ``sedov``: runs ``sedov_shock_finder.py``'s ``__main__`` (N=256) and
  overwrites ``figures/sedov/sedov_shock_finder_correctness_256.png`` and
  ``figures/sedov/sedov_shocked_cells_3d_256.png``. The 3D figure is written
  before the test's assertions, the correctness figure after them.
* ``cwb``: runs the current stationary CWB setup at N=256 as pure hydro
  (``run_cwb(..., cosmic_rays=False)``; with the CR module on, even as the
  ``dsa_efficiency=0`` control, the run needs ~13.8 GiB and does not fit an
  11 GB 2080 Ti) and overwrites
  ``figures/cwb_default/cwb_shocked_cells_{2d,3d}_stat_256.png``. Also saves the
  final state and the finder's surface/Mach fields to ``RAW_DIR``.

Both print surface-cell Mach statistics (median / p10 / p90 / max) to the log.
"""

import sys
from pathlib import Path

import autocvd

autocvd.autocvd = lambda *a, **k: [0]

import runpy  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RAW_DIR = Path("/export/scratch/nknoell/shock_finder_figs")
NUM_CELLS = 256


def _print_mach_stats(label, sf_result):
    surface = np.asarray(sf_result.shock_surface_cells)
    mach = np.asarray(sf_result.mach_numbers)[surface]
    if mach.size == 0:
        print(f"[{label}] no surface cells")
        return
    p10, p50, p90 = np.percentile(mach, [10, 50, 90])
    print(
        f"[{label}] surface cells={mach.size}, Mach median={p50:.2f}, "
        f"p10={p10:.2f}, p90={p90:.2f}, max={mach.max():.2f}"
    )


def regen_sedov():
    import _sedov_setup

    # print the finder statistics of the same run the test plots
    run_sedov = _sedov_setup.run_sedov

    def run_sedov_logged(num_cells):
        run = run_sedov(num_cells)
        _print_mach_stats(f"sedov N={num_cells}", run["sf_result"])
        return run

    _sedov_setup.run_sedov = run_sedov_logged
    runpy.run_path(str(HERE / "sedov_shock_finder.py"), run_name="__main__")


def regen_cwb():
    from _cwb_setup import run_cwb
    from cwb_shock_finder import plot_shocked_cells_2d, plot_shocked_cells_3d

    control = run_cwb(NUM_CELLS, cosmic_rays=False)
    sf_result = control["sf_result"]
    _print_mach_stats(f"cwb N={NUM_CELLS} hydro", sf_result)

    plot_shocked_cells_3d(control, NUM_CELLS, filename=f"cwb_shocked_cells_3d_stat_{NUM_CELLS}.png")
    plot_shocked_cells_2d(control, NUM_CELLS, filename=f"cwb_shocked_cells_2d_stat_{NUM_CELLS}.png")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW_DIR / f"cwb_hydro_{NUM_CELLS}.npz",
        state=np.asarray(control["state"]),
        shock_surface=np.asarray(sf_result.shock_surface_cells),
        mach_numbers=np.asarray(sf_result.mach_numbers),
    )
    print(f"saved raw state to {RAW_DIR}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in ("sedov", "cwb", "all"):
        raise SystemExit("usage: _regen_256_figures.py [sedov|cwb|all]")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if which in ("sedov", "all"):
        regen_sedov()
    if which in ("cwb", "all"):
        regen_cwb()
