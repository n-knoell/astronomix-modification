"""X2: the old segment-argmin surface (DB7 1/|n|_inf weights) as a finder-only control. GPU only.

Runs ``sedov_diag`` with ``dsa_continuous_normal_surface=False`` on the kept states
of the S1 control runs (snapshots 9 / 14 / 19 / 24 / 29, every N, setups pe / ss / cp).
The new-surface values of the same states are the ``native`` rows of
``S2_control.npz``, so the two differ only in the surface.

Output ``RAW/sedov/<subdir>/X2_old_surface.npz`` (one row per (N, snapshot)).

Usage: python x2_old_surface.py
"""

import common
from common import RAW, require_pinned_gpu, save_npz

import numpy as np
import jax.numpy as jnp

import cr_sedov_taylor as item7
from sedov_diag import analyse_snapshot
from s2_restrict import KEPT, SUBDIR, setup_for


def main():
    print("GPU", require_pinned_gpu(), flush=True)
    for setup, sub in SUBDIR.items():
        base = RAW / "sedov" / sub
        rows = []
        for path in sorted(base.glob("N*/control.npz"), key=lambda p: int(p.parent.name[1:])):
            run = np.load(path)
            n = int(run["N"])
            config, params, hd, rv = setup_for(setup, n)
            cr = config.cosmic_ray_grey_config._replace(dsa_continuous_normal_surface=False)
            config = config._replace(cosmic_ray_grey_config=cr)
            for i in KEPT:
                if f"state_{i}" not in run.files:
                    continue
                t = float(run["t"][i])
                d = analyse_snapshot(jnp.asarray(run[f"state_{i}"]), t, config, params, rv, hd,
                                     rho0=item7.RHO_AMBIENT, gamma=item7.GAMMA)
                d.update(N=n, snap=i)
                rows.append(d)
                print(f"{setup} N {n:3d} snap {i:2d} t {t:.4g} R/dx {d['R_front'] * n:6.1f} M50 {d['M50']:.3f} "
                      f"A_inj {d['A_inj']:.4f} D {d['D']:.4g}", flush=True)
        out = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        print("saved", save_npz(base / "X2_old_surface.npz", **out), flush=True)


if __name__ == "__main__":
    main()
