"""DSA dissipation on the CWB 128 state: ours vs the v2-surface pipelines.

No exact answer here: the pipelines are compared with ours, in total and per
shell (wind-1 termination shock vs the bow/termination shock around star 2,
split at the contact x_c along the binary axis), and as an orbital-plane map.
Constant zeta, CR-aware Mach with the state's own e_cr (F_cr set to zero; the
finder and the flux do not read it).

Usage: python dsa_cwb.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from astronomix import (CARTESIAN, FINITE_VOLUME, HLLC, MINMOD, SimulationConfig, SimulationParams,
                        finalize_config, get_helper_data, get_registered_variables)
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import CosmicRayGreyConfig, CosmicRayGreyParams
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_mach_and_flux, dsa_shock_finder_kwargs
from astronomix.shock_finder3D._surface_area import surface_area_weights
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

from compare_finders import find_v2
from dsa_physics import column_area_weights, find_shocks_hybrid
from plot_comparison import SEQ_CMAP

DATA = Path("/export/scratch/nknoell/finder_v2_comparison/cwb128.npz")
MACH_MIN = 1.3


def main():
    data = np.load(DATA)
    n = data["density"].shape[0]
    config = SimulationConfig(
        geometry=CARTESIAN, solver_mode=FINITE_VOLUME, riemann_solver=HLLC, limiter=MINMOD,
        dimensionality=3, num_cells=n, box_size=1.0,
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True, diffusive_shock_acceleration=True),
    )
    rv = get_registered_variables(config)
    state = jnp.zeros((rv.num_vars, n, n, n), jnp.float32)
    for name, index in (("density", rv.density_index), ("pressure", rv.pressure_index),
                        ("vx", rv.velocity_index.x), ("vy", rv.velocity_index.y), ("vz", rv.velocity_index.z),
                        ("e_cr", rv.cosmic_ray_e_index)):
        state = state.at[index].set(data[name])
    config = finalize_config(config, state.shape)
    params = SimulationParams(gamma=5.0 / 3.0, cosmic_ray_grey_params=CosmicRayGreyParams(
        dsa_efficiency=0.1, dsa_mach_min=MACH_MIN))
    hd = get_helper_data(config)
    p, rho = state[rv.pressure_index], state[rv.density_index]

    sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=MACH_MIN,
                              **dsa_shock_finder_kwargs(config.cosmic_ray_grey_config))
    _, flux_o = dsa_mach_and_flux(state, sf, config, params, rv)
    sh = find_shocks_hybrid(state, config, rv, hd, mach_min=MACH_MIN)
    _, flux_h = dsa_mach_and_flux(state, sh, config, params, rv)
    v2d = find_v2(state, config, rv, hd, mach_min=MACH_MIN)
    v2p = find_v2(state, config, rv, hd, mach_min=MACH_MIN, profile_aware_mach=True)
    w_ninf = surface_area_weights(p, rho)
    col = lambda r: column_area_weights(p, rho, r.shock_surface_cells, r.shock_zones)
    pipelines = {
        "ours (1/|n|_inf)": (flux_o, w_ninf),
        "hybrid + column w": (flux_h, col(sh)),
        "hybrid + unit w": (flux_h, jnp.ones_like(w_ninf)),
        "hybrid + 1/|n|_inf": (flux_h, w_ninf),
        "v2 default + column w": (v2d.thermal_energy_flux, col(v2d)),
        "v2 profile + column w": (v2p.thermal_energy_flux, col(v2p)),
    }
    # contact along the binary axis: where u_x changes sign between the stars
    vx_axis = np.asarray(data["vx"][:, n // 2, n // 2])
    i_c = int(np.argmax((vx_axis[:-1] > 0) & (vx_axis[1:] <= 0) & (np.arange(n - 1) > n // 4)))
    x_index = np.arange(n)[:, None, None] * np.ones((1, n, n))
    dx = config.grid_spacing
    lines = [f"contact at x index {i_c} (box-centred x = {(i_c + 0.5) / n - 0.5:+.3f})"]
    ref = None
    maps = {}
    for name, (flux, w) in pipelines.items():
        d = np.asarray(flux * w) * dx**2
        total = d.sum()
        left, right = d[x_index <= i_c].sum(), d[x_index > i_c].sum()
        ref = ref or (total, left, right)
        lines.append(f"{name:24s} D/D_ours total {total / ref[0]:.3f}, wind-1 side {left / ref[1]:.3f}, "
                     f"star-2 side {right / ref[2]:.3f}, injecting cells {(d > 0).sum()}")
        maps[name] = d
    text = "\n".join(lines)
    print(text)
    (HERE / "pics" / "dsa_cwb128.txt").write_text(text + "\n")

    # orbital-plane map of the dissipation per cell column through |z| < 2 cells
    sl = slice(n // 2 - 2, n // 2 + 2)
    fig, axes = plt.subplots(1, len(maps), figsize=(3.1 * len(maps), 3.5), constrained_layout=True)
    vals = np.concatenate([m[:, :, sl].sum(axis=2).ravel() for m in maps.values()])
    vmax = np.percentile(vals[vals > 0], 99.5)
    for ax, (name, m) in zip(axes, maps.items()):
        im = ax.imshow(m[:, :, sl].sum(axis=2).T, origin="lower", extent=[-0.5, 0.5, -0.5, 0.5],
                       cmap=SEQ_CMAP, norm=LogNorm(vmax * 1e-3, vmax))
        ax.set_title(name, fontsize=8.5)
        ax.set_xlabel("x")
        ax.grid(False)
    axes[0].set_ylabel("y")
    fig.colorbar(im, ax=axes, shrink=0.85, label="dissipation weighted for DSA (code units)")
    fig.suptitle("CWB 128: where each pipeline would inject (|z| < 2 cells)")
    fig.savefig(HERE / "pics" / "dsa_cwb128_map.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
