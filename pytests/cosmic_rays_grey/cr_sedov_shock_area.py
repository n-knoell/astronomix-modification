"""
Shock-surface area of the item-7 Sedov blast (Phase B decision DB7, finding Y10).

B7.3 found that the finder's shock surface in the 3D Sedov blast has 0.84-0.87 times as many
cells as ``4 pi R^2 / dx^2`` at every resolution, close to ``<|n|_inf>`` = 0.831, the sphere
average of ``max_i |n_i|``. Hypothesis (Y10): a shock with normal ``n`` has ``|n|_inf`` surface
cells per ``dx^2`` of its area (one per grid line along the dominant axis), while
``inject_crs_at_shocks`` assigns every surface cell the area ``dx^2``. Re-weighting by
``1 / |n|_inf`` with the finder's own normals gave 1.07-1.10 instead of 1 (B7.3).

This diagnostic tests the hypothesis on B7.3's saved snapshots (DSA run, ``zeta = 0.1``, t >=
0.03; no new simulation):

1. Area re-weighted with the *exact* radial normal of each surface cell. 1 confirms Y10.
2. The same with the finder's normal (``-grad T``) and with ``-grad rho`` and ``-grad P``
   (central differences), to see whether a better normal estimate would make a per-cell
   correction usable.
3. Surface cells per unit area in bins of the exact ``|n|_inf``, against the prediction
   ``|n|_inf``.
4. The finder's median Mach number per ``|n|_inf`` bin over the simulated shock's Mach number
   (B7.3 fit): does the Mach bias depend on the shock's angle to the grid?
5. Closure: the dissipation re-weighted with the exact normals over the simulated shock's
   dissipation should equal the Mach-only factor ``(M_fin / M_true)^3`` of B7.3.

Needs ``/export/scratch/nknoell/phaseB/b7_3/`` (``cr_sedov_injection_rate.py`` with its snapshot
states). Run per resolution with ``CR_NUM_CELLS``; raw numbers go to
``/export/scratch/nknoell/phaseB/db7/``, the figure
(``pics/07_sedov_taylor/cr_sedov_shock_area.png``) overlays every resolution found there.
Diagnostic only: no gates yet.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import os
from pathlib import Path

# numerics
import numpy as np
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# shock finder
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_shock_finder_kwargs

# B7.3 snapshots and fits (reuse its saved states)
os.environ.setdefault("CR_B73_REUSE", "1")
import cr_sedov_injection_rate as b73
from cr_sedov_taylor import DSA_EFFICIENCY

NUM_CELLS = b73.NUM_CELLS
RAW_DIR = Path(os.environ.get("CR_PHASEB_DB7_RAW", "/export/scratch/nknoell/phaseB/db7"))
PICS_DIR = Path(__file__).resolve().parent / "pics" / "07_sedov_taylor"
# bins of the exact |n|_inf: 1/sqrt(3) (body diagonal) ... 1 (axis)
NINF_EDGES = np.array([1.0 / np.sqrt(3.0), 0.65, 0.75, 0.85, 0.95, 1.0])


def _ninf(vectors):
    """max_i |n_i| of unit vectors with the component axis first."""
    return np.max(np.abs(vectors), axis=0)


def _unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=0), 1e-30)


def _sphere_bin_fractions(rng=np.random.default_rng(0), samples=4_000_000):
    """Fraction of the unit sphere and mean |n|_inf per NINF_EDGES bin (Monte Carlo)."""
    n = _unit(rng.normal(size=(3, samples)))
    ninf = _ninf(n)
    idx = np.clip(np.digitize(ninf, NINF_EDGES) - 1, 0, len(NINF_EDGES) - 2)
    frac = np.bincount(idx, minlength=len(NINF_EDGES) - 1) / samples
    mean = np.bincount(idx, weights=ninf, minlength=len(NINF_EDGES) - 1) / np.maximum(
        np.bincount(idx, minlength=len(NINF_EDGES) - 1), 1)
    return frac, mean, float(ninf.mean())


def test_cr_sedov_shock_area():
    raw = np.load(b73.RAW_DIR / f"b7_3_N{NUM_CELLS}.npz")
    run = b73._run_with_snapshots(DSA_EFFICIENCY)
    config, params = run["config"], run["params"]
    rv, helper_data = run["registered_variables"], run["helper_data"]
    cr_config, cr_params = config.cosmic_ray_grey_config, params.cosmic_ray_grey_params
    dx = config.grid_spacing

    centres = np.moveaxis(np.asarray(helper_data.geometric_centers), -1, 0)
    flat = centres.reshape(3, -1)
    box_centre = 0.5 * (flat.min(axis=1) + flat.max(axis=1))
    r_vec = centres - box_centre[:, None, None, None]
    assert np.allclose(np.linalg.norm(r_vec, axis=0), np.asarray(helper_data.r), atol=1e-5 * dx)
    n_exact_all = _unit(r_vec)

    frac_sphere, ninf_bin_sphere, ninf_sphere = _sphere_bin_fractions()
    nbins = len(NINF_EDGES) - 1
    keys = ("t", "R_true", "M_true", "area_cells", "area_exact", "area_finder", "area_grad_rho",
            "area_grad_p", "angle_finder_med", "angle_grad_rho_med", "angle_grad_p_med",
            "D_exact_over_meas", "mach_ratio_cubed")
    out = {k: [] for k in keys}
    cells_per_area_bins, mach_ratio_bins = [], []

    times = run["times"]
    for i, (t, state) in enumerate(zip(times, run["states"])):
        if t < b73.LATE_TIME:
            continue
        assert np.isclose(raw["dsa_t"][i], t)
        R, M_true = float(raw["dsa_R_true"][i]), float(raw["dsa_M_true"][i])
        state_j = jnp.asarray(state)
        sf = find_shocks_pfrommer(
            state_j, config, rv, helper_data,
            mach_min=cr_params.dsa_mach_min, **dsa_shock_finder_kwargs(cr_config),
        )
        surface = np.asarray(sf.shock_surface_cells)
        mach = np.asarray(sf.mach_numbers)[surface]
        flux = np.asarray(sf.thermal_energy_flux)[surface]
        n_exact = n_exact_all[:, surface]
        n_finder = np.asarray(sf.shock_direction)[:, surface]
        rho = np.asarray(state[rv.density_index])
        p = np.asarray(state[rv.pressure_index])
        n_rho = _unit(-np.stack(np.gradient(rho, dx))[:, surface])
        n_p = _unit(-np.stack(np.gradient(p, dx))[:, surface])

        sphere = 4.0 * np.pi * R**2
        out["t"].append(float(t))
        out["R_true"].append(R)
        out["M_true"].append(M_true)
        out["area_cells"].append(surface.sum() * dx**2 / sphere)
        for name, n in (("exact", n_exact), ("finder", n_finder), ("grad_rho", n_rho),
                        ("grad_p", n_p)):
            out[f"area_{name}"].append(np.sum(1.0 / np.maximum(_ninf(n), 1e-6)) * dx**2 / sphere)
        for name, n in (("finder", n_finder), ("grad_rho", n_rho), ("grad_p", n_p)):
            cos = np.clip(np.sum(n * n_exact, axis=0), -1.0, 1.0)
            out[f"angle_{name}_med"].append(float(np.degrees(np.median(np.arccos(cos)))))
        D_exact = np.sum(flux / _ninf(n_exact)) * dx**2
        out["D_exact_over_meas"].append(D_exact / float(raw["dsa_D_meas"][i]))
        out["mach_ratio_cubed"].append(float(raw["dsa_mach_ratio"][i]) ** 3)

        # per obliquity bin: surface cells per sphere area, finder Mach over the true one
        idx = np.clip(np.digitize(_ninf(n_exact), NINF_EDGES) - 1, 0, nbins - 1)
        counts = np.bincount(idx, minlength=nbins)
        cells_per_area_bins.append(counts * dx**2 / (frac_sphere * sphere))
        mach_ratio_bins.append([np.median(mach[idx == b]) / M_true if np.any(idx == b) else np.nan
                                for b in range(nbins)])

    res = {k: np.asarray(v) for k, v in out.items()}
    res["cells_per_area_bins"] = np.asarray(cells_per_area_bins)
    res["mach_ratio_bins"] = np.asarray(mach_ratio_bins)
    res["ninf_bin_sphere"] = ninf_bin_sphere
    res["frac_sphere"] = frac_sphere
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RAW_DIR / f"db7_N{NUM_CELLS}.npz", **res)

    print(f"N = {NUM_CELLS}, DSA run, t >= {b73.LATE_TIME}; sphere <|n|_inf> = {ninf_sphere:.4f}")
    print("   t      A_cells A_exact A_finder A_gradrho A_gradp | angle to r_hat (deg) finder "
          "gradrho gradp | D_exact/D_meas (M ratio)^3")
    for j in range(len(res["t"])):
        print(f"  {res['t'][j]:.4f}  {res['area_cells'][j]:.4f}  {res['area_exact'][j]:.4f}  "
              f"{res['area_finder'][j]:.4f}   {res['area_grad_rho'][j]:.4f}    "
              f"{res['area_grad_p'][j]:.4f} |   {res['angle_finder_med'][j]:5.2f}   "
              f"{res['angle_grad_rho_med'][j]:5.2f}  {res['angle_grad_p_med'][j]:5.2f} |   "
              f"{res['D_exact_over_meas'][j]:.4f}        {res['mach_ratio_cubed'][j]:.4f}")
    print("late-time means:")
    for k in ("area_cells", "area_exact", "area_finder", "area_grad_rho", "area_grad_p",
              "angle_finder_med", "angle_grad_rho_med", "angle_grad_p_med",
              "D_exact_over_meas", "mach_ratio_cubed"):
        print(f"  {k}: {res[k].mean():.4f}")
    print("per |n|_inf bin (exact normal): bin, sphere fraction, predicted cells/area "
          "(<|n|_inf>), measured cells/area, finder M / M_true")
    for b in range(nbins):
        print(f"  [{NINF_EDGES[b]:.3f}, {NINF_EDGES[b + 1]:.3f}]  {frac_sphere[b]:.3f}  "
              f"{ninf_bin_sphere[b]:.4f}  {np.mean(res['cells_per_area_bins'][:, b]):.4f}  "
              f"{np.nanmean(res['mach_ratio_bins'][:, b]):.4f}")
    _plot()


def _plot():
    runs = sorted((int(p.stem.split("_N")[-1]), np.load(p)) for p in RAW_DIR.glob("db7_N*.npz"))
    fig, (ax_cells, ax_mach, ax_area) = plt.subplots(1, 3, figsize=(15, 4.5))
    centres = 0.5 * (NINF_EDGES[:-1] + NINF_EDGES[1:])
    for k, (n, d) in enumerate(runs):
        c = f"C{k}"
        ax_cells.plot(d["ninf_bin_sphere"], d["cells_per_area_bins"].mean(axis=0), "o-", color=c,
                      label=f"N = {n}")
        ax_mach.plot(centres, np.nanmean(d["mach_ratio_bins"], axis=0), "o-", color=c,
                     label=f"N = {n}")
        for m, (key, label) in enumerate((("area_cells", "cell count"),
                                          ("area_exact", "exact normal"),
                                          ("area_finder", r"finder normal ($-\nabla T$)"),
                                          ("area_grad_rho", r"$-\nabla\rho$"),
                                          ("area_grad_p", r"$-\nabla P$"))):
            ax_area.plot(n, d[key].mean(), "osD^v"[m], color=f"C{m}", label=label if k == 0 else None)
    line = np.array([NINF_EDGES[0], 1.0])
    ax_cells.plot(line, line, "k--", lw=1, label=r"prediction $|n|_\infty$")
    ax_cells.set_xlabel(r"$|n|_\infty$ of the exact normal (bin mean)")
    ax_cells.set_ylabel(r"surface cells $\times\,dx^2$ / shock area")
    ax_cells.set_title("Surface cells per area vs obliquity")
    ax_cells.legend(fontsize=8)
    ax_mach.set_xlabel(r"$|n|_\infty$ of the exact normal (bin centre)")
    ax_mach.set_ylabel(r"finder median $M$ / $M_{\rm true}$")
    ax_mach.set_title("Finder Mach number vs obliquity")
    ax_mach.legend(fontsize=8)
    ax_area.axhline(1.0, color="k", ls="--", lw=1)
    ax_area.set_xlabel("N (cells per axis)")
    ax_area.set_ylabel(r"$\sum dx^2 / |n|_\infty$ / $4\pi R^2$ (late-time mean)")
    ax_area.set_title("Shock area, per-cell weight from normal")
    ax_area.legend(fontsize=8)
    fig.suptitle(r"Item-7 Sedov (DSA run, $\zeta = 0.1$, $t \geq 0.03$): shock area "
                 r"and obliquity (DB7)")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_sedov_shock_area.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    test_cr_sedov_shock_area()
