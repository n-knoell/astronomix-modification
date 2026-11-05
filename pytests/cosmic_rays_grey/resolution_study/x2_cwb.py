"""X2 (CWB part): the old segment-argmin surface on the CWB control last states. GPU only.

``cwb_study.analyse_cwb_snapshot`` with ``dsa_continuous_normal_surface=False`` (zeta = 0.1
settings) on the last state of every ``RAW/cwb/N*/control.npz``; the new-surface values are
the ``native`` rows of ``C2_control.npz``. Output ``RAW/cwb/X2_old_surface.npz``.

Usage: python x2_cwb.py
"""

import common
from common import RAW, require_pinned_gpu, save_npz

import numpy as np
import jax.numpy as jnp

from c2_restrict import SCALARS
from cwb_study import analyse_cwb_snapshot
from setups import build_cwb


def main():
    print("GPU", require_pinned_gpu(), flush=True)
    rows = []
    for path in sorted(RAW.glob("cwb/N*/control.npz"), key=lambda p: int(p.parent.name[1:])):
        run = np.load(path)
        n = int(run["N"])
        _, config, params, hd, rv = build_cwb(n, 0.1)
        cr = config.cosmic_ray_grey_config._replace(dsa_continuous_normal_surface=False)
        config = config._replace(cosmic_ray_grey_config=cr)
        d = analyse_cwb_snapshot(jnp.asarray(run["state_last"]), float(run["t"][-1]), config, params, rv, hd)
        row = {k: d[k] for k in SCALARS}
        row["N"] = n
        rows.append(row)
        print(f"old N {n:3d} M50 w1/w2 {d['M50_wind1']:.2f} {d['M50_wind2']:.2f} A w1/w2 {d['A_wind1']:.4e} "
              f"{d['A_wind2']:.4e} inj w1/w2/amb {d['inj_wind1']:.4e} {d['inj_wind2']:.4e} {d['inj_ambient']:.4e}",
              flush=True)
    out = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    print("saved", save_npz(RAW / "cwb" / "X2_old_surface.npz", **out), flush=True)


if __name__ == "__main__":
    main()
