"""
Alternative Mach-number estimators and smoothed-normal area weights on the item-7 Sedov blast
(Phase B decisions DB4 and DB7, step 1 of the fix plan: reanalysis only, no finder change).

B7.3 found the late-time DSA injection in the 3D Sedov blast at 0.43/0.53/0.60 (N = 48/96/160)
of the self-similar reference, the product of two finder factors:

* Mach (DB4): the finder reads the Mach number from the gas-pressure jump between the pressure
  peak behind the front and the pre-shock plateau. In the thin Sedov shell pressure falls steeply
  behind the front, so the smeared peak is low: ``M_fin / M_true`` = 0.76/0.84/0.87.
* Area (DB7): every surface cell counts as ``dx^2``, but an oblique shock has ``|n|_inf`` surface
  cells per ``dx^2`` of its area (sphere average 0.831, ``cr_sedov_shock_area.py``).
  ``1 / |n|_inf`` with exact normals restores the area; with the finder's normals it
  over-corrects (1.07-1.10), since noise near a grid axis always lowers ``|n|_inf``.

This diagnostic repeats the finder's adaptive walk on B7.3's saved snapshots (both runs: control
``zeta = 0`` and DSA ``zeta = 0.1``; t >= 0.03) and evaluates at the *same* post-/pre-shock
sample points:

1. Mach estimators against the simulated shock's Mach number ``M_true`` (B7.3 fit):
   ``p``    gas-pressure jump (the finder's, reproduced as a check);
   ``u``    normal-velocity jump, ``u_2 - u_1 = 2 c_1 (M^2 - 1) / ((gamma + 1) M)``;
   ``umax`` the same with the largest normal velocity along the post-shock walk;
   ``T``    temperature jump (inverse of the RH temperature ratio);
   ``PT``   geometric mean of ``p`` and ``T``;
   ``ptot`` gas + CR pressure jump (DSA run; the total-pressure shock of DB8).
   For each, the dissipated flux ``delta(M) rho_1 (M c_1)^3 / 2`` from the finder's pre-shock
   state, summed over the surface (``dx^2`` per cell) over the gas-only RH dissipation of the
   simulated shock, ``D_meas`` (B7.3): ``D / D_meas`` = area x Mach factor.
2. Area weights ``1 / |n|_inf`` with exact radial normals, the finder's normals, finder normals
   averaged over the surface cells in a 3^3 / 5^3 box, and ``-grad T`` of a Gaussian-smoothed
   (sigma = 1 cell) temperature, over ``4 pi R_true^2``.
3. Combined: ``D / D_meas`` with the velocity flux and the smoothed-normal weight, and the
   implied injection over the reference, ``D / D_meas x (xi / xi_0)^5`` (B7.3 decomposition).

Needs ``/export/scratch/nknoell/phaseB/b7_3/`` (``cr_sedov_injection_rate.py`` with its snapshot
states). Run per resolution with ``CR_NUM_CELLS``; raw numbers go to
``/export/scratch/nknoell/phaseB/db4/``, the figure
(``pics/07_sedov_taylor/cr_sedov_mach_estimators.png``) overlays every resolution found there.
Diagnostic only: no gates.
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
from scipy import ndimage

# plotting
import matplotlib.pyplot as plt

# shock finder
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_shock_finder_kwargs

# estimators at the finder's sample points, normal variants
from _finder_estimators import MACH_KEYS, finder_samples, ninf as _ninf, normal_variants, unit as _unit

# B7.3 snapshots and fits (reuse its saved states)
os.environ.setdefault("CR_B73_REUSE", "1")
import cr_sedov_injection_rate as b73
from cr_sedov_taylor import DSA_EFFICIENCY, GAMMA
from astronomix.test_setups.reference_solutions.sedov_two_fluid import solve_sedov_two_fluid

NUM_CELLS = b73.NUM_CELLS
RAW_DIR = Path(os.environ.get("CR_PHASEB_DB4_RAW", "/export/scratch/nknoell/phaseB/db4"))
PICS_DIR = Path(__file__).resolve().parent / "pics" / "07_sedov_taylor"
NORMAL_KEYS = ("exact", "finder", "box3", "box5", "gaussT", "gaussT2")


def _analyse_run(run, raw, prefix):
    config, params = run["config"], run["params"]
    rv, helper_data = run["registered_variables"], run["helper_data"]
    cr_config, cr_params = config.cosmic_ray_grey_config, params.cosmic_ray_grey_params
    kwargs = dsa_shock_finder_kwargs(cr_config)
    dx = config.grid_spacing

    centres = np.moveaxis(np.asarray(helper_data.geometric_centers), -1, 0)
    flat = centres.reshape(3, -1)
    r_vec = centres - (0.5 * (flat.min(axis=1) + flat.max(axis=1)))[:, None, None, None]
    n_exact_all = _unit(r_vec)

    out = {k: [] for k in ("t", "M_true", "n_valid", "D_meas")}
    for k in MACH_KEYS:
        out[f"mach_{k}"] = []      # median M_est / M_true over the valid surface cells
        out[f"D_{k}"] = []         # sum flux dx^2 / D_meas
    for k in NORMAL_KEYS:
        out[f"area_{k}"] = []      # sum dx^2 / |n|_inf / (4 pi R^2), all surface cells
        out[f"D_u_{k}"] = []       # velocity flux weighted by 1/|n|_inf, over D_meas
        out[f"D_p_{k}"] = []       # finder flux weighted by 1/|n|_inf, over D_meas
    out["area_cells"] = []
    for k in NORMAL_KEYS[1:]:
        out[f"angle_{k}"] = []     # median angle to the exact normal (deg)
        out[f"tilt_{k}"] = []      # mean |n|_inf - |n_exact|_inf (< 0: tilted away from axes)
    out["closure_p"] = []          # our D_p / the finder's own flux sum (must be 1)

    for i, (t, state) in enumerate(zip(run["times"], run["states"])):
        if t < b73.LATE_TIME:
            continue
        assert np.isclose(raw[f"{prefix}_t"][i], t)
        R, M_true = float(raw[f"{prefix}_R_true"][i]), float(raw[f"{prefix}_M_true"][i])
        D_meas = float(raw[f"{prefix}_D_meas"][i])
        sf = find_shocks_pfrommer(jnp.asarray(state), config, rv, helper_data,
                                  mach_min=cr_params.dsa_mach_min, **kwargs)
        surface = np.asarray(sf.shock_surface_cells)
        smp = finder_samples(state, sf, config, rv, kwargs, GAMMA, b73.GAMMA_CR)
        valid, M, flux = smp["valid"], smp["M"], smp["flux"]
        normals = {"exact": n_exact_all[:, surface], **normal_variants(state, sf, rv, dx)}
        valid_in_surface = valid[surface]

        sphere = 4.0 * np.pi * R**2
        out["t"].append(float(t))
        out["M_true"].append(M_true)
        out["n_valid"].append(int(valid.sum()))
        out["D_meas"].append(D_meas)
        out["area_cells"].append(surface.sum() * dx**2 / sphere)
        for k in MACH_KEYS:
            out[f"mach_{k}"].append(float(np.median(M[k])) / M_true)
            out[f"D_{k}"].append(float(flux[k].sum()) * dx**2 / D_meas)
        out["closure_p"].append(smp["closure_p"])
        for k in NORMAL_KEYS:
            w = 1.0 / np.maximum(_ninf(normals[k]), 1e-6)
            out[f"area_{k}"].append(float(w.sum()) * dx**2 / sphere)
            out[f"D_u_{k}"].append(float(np.sum(flux["u"] * w[valid_in_surface])) * dx**2 / D_meas)
            out[f"D_p_{k}"].append(float(np.sum(flux["p"] * w[valid_in_surface])) * dx**2 / D_meas)
        for k in NORMAL_KEYS[1:]:
            cos = np.clip(np.sum(normals[k] * normals["exact"], axis=0), -1.0, 1.0)
            out[f"angle_{k}"].append(float(np.degrees(np.median(np.arccos(cos)))))
            out[f"tilt_{k}"].append(float(np.mean(_ninf(normals[k]) - _ninf(normals["exact"]))))
    return {k: np.asarray(v) for k, v in out.items()}


def test_cr_sedov_mach_estimators():
    raw = np.load(b73.RAW_DIR / f"b7_3_N{NUM_CELLS}.npz")
    xi0 = {0.0: solve_sedov_two_fluid(0.0).xi_0,
           DSA_EFFICIENCY: solve_sedov_two_fluid(DSA_EFFICIENCY, convention="gas_rh").xi_0}
    res = {}
    for zeta, prefix in ((0.0, "control"), (DSA_EFFICIENCY, "dsa")):
        run = b73._run_with_snapshots(zeta)
        r = _analyse_run(run, raw, prefix)
        r["xi_ratio5"] = np.asarray((float(raw[f"{prefix}_xi_fit"]) / xi0[zeta]) ** 5)
        for k, v in r.items():
            res[f"{prefix}_{k}"] = v
        del run

        print(f"\nN = {NUM_CELLS}, {prefix} run (zeta = {zeta}), t >= {b73.LATE_TIME}, late means; "
              f"(xi/xi0)^5 = {float(r['xi_ratio5']):.4f}")
        print(f"  closure finder flux reproduced: {r['closure_p'].min():.5f} .. "
              f"{r['closure_p'].max():.5f}; valid cells per snapshot {r['n_valid'].min()}..")
        print("  estimator   median M/M_true   D/D_meas (dx^2 per cell)   (D/D_meas)/area_cells")
        for k in MACH_KEYS:
            print(f"  {k:9s}   {r[f'mach_{k}'].mean():.4f}            {r[f'D_{k}'].mean():.4f}"
                  f"                     {(r[f'D_{k}'] / r['area_cells']).mean():.4f}")
        print(f"  area: cells {r['area_cells'].mean():.4f}")
        print("  normal    area(1/|n|inf)  median angle to r_hat  D_p weighted  D_u weighted  "
              "inj/ref (u, weighted)  tilt <|n|inf - |n_ex|inf>")
        for k in NORMAL_KEYS:
            ang = r[f"angle_{k}"].mean() if f"angle_{k}" in r else 0.0
            tilt = r[f"tilt_{k}"].mean() if f"tilt_{k}" in r else 0.0
            print(f"  {k:8s}  {r[f'area_{k}'].mean():.4f}          {ang:5.2f}                 "
                  f"{r[f'D_p_{k}'].mean():.4f}        {r[f'D_u_{k}'].mean():.4f}        "
                  f"{(r[f'D_u_{k}'] * r['xi_ratio5']).mean():.4f}        {tilt:+.4f}")
        print("  per snapshot: t, M_p, M_u, M_umax, M_T, M_PT, M_ptot (median / M_true), "
              "area box3")
        for j in range(len(r["t"])):
            print("   " + "  ".join(f"{x:.4f}" for x in (
                r["t"][j], *(r[f"mach_{k}"][j] for k in MACH_KEYS), r["area_box3"][j])))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(RAW_DIR / f"db4_N{NUM_CELLS}.npz", **res)
    _plot()


def _plot():
    runs = sorted((int(p.stem.split("_N")[-1]), np.load(p)) for p in RAW_DIR.glob("db4_N*.npz"))
    if not runs:
        return
    ns = [n for n, _ in runs]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    labels = {"p": "gas pressure (finder)", "u": "velocity jump", "umax": "velocity, walk max",
              "T": "temperature", "PT": r"$\sqrt{M_p M_T}$", "ptot": "gas + CR pressure"}
    markers = dict(zip(MACH_KEYS, "osD^vP"))
    for k in (k for k in MACH_KEYS if k != "umax"):  # umax == u at every snapshot
        axes[0].plot(ns, [d[f"dsa_mach_{k}"].mean() for _, d in runs], marker=markers[k],
                     label=labels[k])
    axes[0].plot(ns, [d["control_mach_u"].mean() for _, d in runs], "s--", color="C1", alpha=0.6,
                 label=r"velocity jump, $\zeta = 0$")
    axes[0].set_ylabel(r"median $M_{\rm est} / M_{\rm true}$")
    axes[0].set_title("(a) Mach estimators")
    nlabels = {"exact": "exact radial", "finder": r"finder ($-\nabla T$)",
               "box3": r"finder, $3^3$ surface avg", "box5": r"finder, $5^3$ surface avg",
               "gaussT": r"$-\nabla T$, Gaussian $\sigma = 1$",
               "gaussT2": r"$-\nabla T$, Gaussian $\sigma = 2$"}
    axes[1].plot(ns, [d["dsa_area_cells"].mean() for _, d in runs], "k:", marker="x",
                 label=r"cell count ($dx^2$ each)")
    for m, k in enumerate(NORMAL_KEYS):
        axes[1].plot(ns, [d[f"dsa_area_{k}"].mean() for _, d in runs], marker="osD^vP"[m],
                     label=nlabels[k])
    axes[1].set_ylabel(r"$\sum dx^2 / |n|_\infty$ / $4\pi R^2$")
    axes[1].set_title(r"(b) Area with $1/|n|_\infty$ weights")
    combos = (("D_p", "finder now (p, cell count)", "o-"),
              ("D_u", "velocity, cell count", "s-"),
              ("D_p_gaussT2", r"pressure, $\sigma = 2$ weight", "D--"),
              ("D_u_gaussT2", r"velocity, $\sigma = 2$ weight", "^-"),
              ("D_u_exact", "velocity, exact weight", "v:"))
    for key, lab, style in combos:
        axes[2].plot(ns, [(d[f"dsa_{key}"] * d["dsa_xi_ratio5"]).mean() for _, d in runs], style,
                     label=lab)
    axes[2].set_ylabel(r"injection / reference = $D/D_{\rm meas} \times (\xi/\xi_0)^5$")
    axes[2].set_title(r"(c) Implied injection ($\zeta = 0.1$)")
    for ax in axes:
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("N (cells per axis)")
        ax.set_xticks(ns)
        ax.legend(fontsize=7)
    fig.suptitle(r"Item-7 Sedov, B7.3 snapshots ($t \geq 0.03$, late means): Mach estimators "
                 r"(DB4) and area weights (DB7), reanalysis only")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_sedov_mach_estimators.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    test_cr_sedov_mach_estimators()
