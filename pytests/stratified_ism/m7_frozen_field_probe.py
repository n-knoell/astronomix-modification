"""
M7 frozen-field probe: how much of M7's CR transport across B is numerical? (DESIGN.md
"Open: CR diffusion follow-up (audit 2026-10-05)", step 4.)

M7 (``m7_girichidis_pilot.py``) prescribes Girichidis et al. (2016)'s ``kappa_par = 1e28`` and
``kappa_perp = 1e26 cm^2/s`` at 33.3 pc. The ring tests (``cr_anisotropic_ring.py``) and T3
(``cr_diffusion_rate.py``) put the scheme's numerical cross-field diffusion at about 0.5-0.9 of
that ``kappa_perp`` for M7-like resolution, but in idealized fields. This probe measures it in
M7's own magnetic field.

**Setup.** M7's box and grid (2 x 2 x 5 kpc, 60 x 60 x 150, periodic in x/y, open in z), its
solver settings (FV, HLLC, minmod, MHD) and its CR parameters (``kappa_par``, ``v_red`` =
1000 km/s, anisotropic transport with the monotonicity guard). The B field is a frozen M7
snapshot (``snap_magnetic_*`` in M7's raw-data npz), scaled by ``B_SCALE`` so that ``B^2 / 2``
is far below the gas pressure (no Lorentz forces; ``b_hat`` unchanged; ``|B|`` stays far above
``b_field_floor``). Uniform static gas (``rho = P = 1``, sound speed 1.3 km/s, so the CR closure
speed sets dt). Four small CR deposits (peak ``e_cr`` 1e-3, ``e_cr`` floor 1e-20) with M7's SN profile (radius 40 pc, tanh edge 33.3 pc)
at the midplane, one per x-y quadrant, on a CR-free background. No gravity, cooling or SNe:
pure CR transport in a fixed field.

**Variants** per snapshot:
- ``A``: M7 grid, ``kappa_perp = 0`` (baseline).
- ``B``: M7 grid, ``kappa_perp = 1e26`` -> physical signal ``S = D_zz(B) - D_zz(A)``.
- ``C``: M7 grid, ``kappa_perp = 0``, guard off -> guard share ``G = D_zz(A) - D_zz(C)``.
- ``D``: 2x grid (120 x 120 x 300; B trilinearly interpolated, the deposits sampled from the
  same continuous profile), ``kappa_perp = 0`` -> ``E = D_zz(A) - D_zz(D)``. The numerical
  error on the M7 grid is ``E / (1 - 2^-p)`` with p ~ 1-1.4 (ring tests, T3).

All in float32, so ``D`` fits an 11 GB GPU (128^3 in float64 needs ~9.1 GB); the probe needs
percent-level precision only. The run advances in ``CHUNK_MYR`` chunks and takes moments in
between, instead of storing snapshots.

**Metrics.** ``D_zz = 0.5 d<z^2>/dt`` (e_cr-weighted, z from the midplane, fit over
``t >= FIT_START_MYR``), and the CR energy fractions above ``|z|`` = 300 / 500 pc at 1 and 3 Myr.

**Decision (DESIGN.md step 4).** If ``|E| / (1 - 2^-p) < 0.3 |S|``, M7 resolves ``kappa_perp``;
otherwise its cross-field transport is set by resolution. Also compare with the scheme the
reference paper used (FLASH, Yang et al. 2012: ~3-6x less numerical cross-field diffusion than
ours on the ring at equal N).

Usage::

    python m7_frozen_field_probe.py --snapshot=K --variant=A|B|C|D   # one run -> npz
    python m7_frozen_field_probe.py --summarize                     # table + figure

**Results (2026-10-05, RTX 2080 Ti, ~1 min per M7-grid variant, ~4 min per 2x variant).**
D_zz in cm^2/s; S = kappa_perp effect, G = guard share, E = A - D:

    field at   D_zz(A)   S         G          E
      0 Myr    1.46e26   7.8e25    1.46e26    1.5e25    (B = B_x: all vertical transport is the guard's)
     48 Myr    2.16e27   1.54e26   7.0e25     6.8e26
    144 Myr    2.35e27   1.64e26   -1.07e26   -2.0e26
    250 Myr    2.34e27   1.57e26   -1.23e26   -1.3e26

M7 does not resolve kappa_perp = 1e26: in the initial field the guard alone moves CRs
vertically ~1.9x as much as kappa_perp does (zero with the guard off), and in the evolved,
mostly vertical fields the 2x grid changes D_zz by 1.4-7.8x S. See DESIGN.md step 4.

**NOT a committed pytest** -- an exploratory script, like the M-series runs. Raw results go to
``/export/scratch/nknoell/m7_probe/``, the figure to ``pics/``.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import sys
import time
from pathlib import Path

# numerics
import numpy as np
from scipy.ndimage import map_coordinates

# units
import astropy.units as u

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix
from astronomix import BoundarySettings, BoundarySettings1D, CodeUnits
from astronomix import SimulationConfig, SimulationParams, get_helper_data
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FINITE_VOLUME,
    HLLC,
    MINMOD,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    SINGLE_PRECISION,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
)
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# ---- M7's units and parameters (m7_girichidis_pilot.py) ----
CODE_UNITS = CodeUnits(1 * u.pc, 1 * u.M_sun, 1 * u.km / u.s)
CODE_TIME = CODE_UNITS.code_time
MYR_CODE = float((1 * u.Myr).to(CODE_TIME).value)
KAPPA_PAR = float((1e28 * u.cm**2 / u.s).to(CODE_UNITS.code_length**2 / CODE_TIME).value)
KAPPA_PERP_PHYSICAL = float((1e26 * u.cm**2 / u.s).to(CODE_UNITS.code_length**2 / CODE_TIME).value)
V_RED = 1000.0  # km/s = code velocity
GAMMA_CR = 4.0 / 3.0
L_XY, L_Z = 2000.0, 5000.0  # pc
N_XY, N_Z = 60, 150
SN_INJECTION_RADIUS = 40.0  # pc
SN_SMOOTH_WIDTH = 2000.0 / 60  # pc: M7's 1 cell, kept physical on the 2x grid

# ---- probe settings ----
M7_DATA = Path("/export/scratch/nknoell/m7_pilot_data_both_mhd_sg_150.npz")
OUT_DIR = Path("/export/scratch/nknoell/m7_probe")
SNAPSHOTS = (0, 5, 15, 25)  # 0 / 48 / 144 / 250 Myr
B_SCALE = 1e-3  # max |B| 12.3 -> B^2/2 <= 8e-5 << P = 1; min |B| 1e-3 -> 1e-6 >> 1e-10 floor
E_AMPLITUDE = 1e-3  # peak e_cr: P_cr = 3e-4 << P = 1
# The always-on e_cr floor (1e-10 by default) would fill the CR-free 2e10 pc^3 box with ~10x
# the deposits' energy and swamp the moments (found in the first smoke run). Set it far below
# the deposits, and take moments of the excess above it.
E_FLOOR = 1e-20
DEPOSIT_XY = ((500.0, 500.0), (1500.0, 500.0), (500.0, 1500.0), (1500.0, 1500.0))
T_END_MYR = 3.0
CHUNK_MYR = 0.25
# v_red scan (DESIGN.md step 5a): finer early checkpoints, where a v_red dependence shows first
# (relaxation time kappa / ((gamma_cr - 1) v_red^2) = 0.095 Myr at 1000 km/s, 0.011 at 3000).
V_RED_SCAN = (1000.0, 3000.0, 10000.0, 30000.0)
SCAN_TIMES_MYR = (0.05, 0.1, 0.25, 0.5) + tuple(0.25 * k for k in range(3, 13))
FIT_START_MYR = 0.5  # skip the transient (1 / nu ~ 0.1 Myr)
Z_THRESHOLDS = (300.0, 500.0)
GUARD_OFF = dict(cr_guard_sensor_onset=2.0, cr_guard_sensor_full=3.0)
VARIANTS = {
    "A": dict(refine=1, kappa_perp=0.0, guard=True),
    "B": dict(refine=1, kappa_perp=KAPPA_PERP_PHYSICAL, guard=True),
    "C": dict(refine=1, kappa_perp=0.0, guard=False),
    "D": dict(refine=2, kappa_perp=0.0, guard=True),
    # Calibration (known answer): with the guard off and B = B_x (snapshot 0), D_zz(E) -
    # D_zz(C) must be kappa_perp exactly in the continuum.
    "E": dict(refine=1, kappa_perp=KAPPA_PERP_PHYSICAL, guard=False),
}
# Deposit shape (literature check, DESIGN.md step 4): M7 uses radius 40 pc with a 1-cell tanh
# edge (it overrides SNDrivingParams.sn_smooth_cells = 2 with 1). SILCC (Girichidis et al. 2018,
# Sec. 2.1) injects into a sphere of 800 Msun, or of radius 4 cells when the Sedov-Taylor
# radius is not resolved. --smooth-cells / --radius-cells override M7's values.
CELL = 2000.0 / 60


def _cell_centers(n, length):
    return (np.arange(n) + 0.5) * length / n


def _magnetic_field(snapshot, refine):
    """M7's B at ``snapshot``, scaled by ``B_SCALE``; trilinearly interpolated to the
    ``refine``-times finer grid (periodic in x/y, edge values in z)."""
    data = np.load(M7_DATA)
    b = np.stack([data[f"snap_magnetic_{c}"][snapshot] for c in "xyz"]).astype(np.float64)
    b *= B_SCALE
    if refine == 1:
        return b
    padded = np.pad(b, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="wrap")
    padded = np.pad(padded, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge")
    fine = [
        (np.arange(n * refine) + 0.5) / refine - 0.5 + 1.0  # +1: the padding
        for n in (N_XY, N_XY, N_Z)
    ]
    grid = np.meshgrid(*fine, indexing="ij")
    return np.stack([map_coordinates(component, grid, order=1) for component in padded])


def _tag(smooth_cells, radius_cells, v_red=None):
    tag = ""
    if v_red is not None:
        tag += f"_vred{v_red:g}"
    if smooth_cells is not None:
        tag += f"_sm{smooth_cells:g}"
    if radius_cells is not None:
        tag += f"_r{radius_cells:g}"
    return tag


def _run(snapshot, variant, smooth_cells=None, radius_cells=None, v_red=None):
    v = VARIANTS[variant]
    v_red_value = V_RED if v_red is None else v_red
    checkpoints = (
        tuple(CHUNK_MYR * (k + 1) for k in range(int(round(T_END_MYR / CHUNK_MYR))))
        if v_red is None else SCAN_TIMES_MYR
    )
    smooth_width = SN_SMOOTH_WIDTH if smooth_cells is None else smooth_cells * CELL
    radius = SN_INJECTION_RADIUS if radius_cells is None else radius_cells * CELL
    n_xy, n_z = N_XY * v["refine"], N_Z * v["refine"]
    config = SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        box_size=StaticFloatVector(L_XY, L_XY, L_Z),
        num_cells=StaticIntVector(n_xy, n_xy, n_z),
        exact_end_time=True,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        mhd=True,
        fv_mhd_alfven_cfl=True,
        numerical_precision=SINGLE_PRECISION,
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=True, diffusive_relaxation=True, anisotropic_transport=True
        ),
    )
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    centers = np.asarray(helper_data.geometric_centers)
    x, y, z = centers[..., 0], centers[..., 1], centers[..., 2] - 0.5 * L_Z

    e_cr = np.zeros((n_xy, n_xy, n_z))
    for x0, y0 in DEPOSIT_XY:
        distance = np.sqrt((x - x0) ** 2 + (y - y0) ** 2 + z**2)
        e_cr += 0.5 * (1.0 - np.tanh((distance - radius) / smooth_width))
    e_cr *= E_AMPLITUDE

    b = _magnetic_field(snapshot, v["refine"])
    state = np.zeros((registered_variables.num_vars, n_xy, n_xy, n_z), dtype=np.float32)
    state[registered_variables.density_index] = 1.0
    state[registered_variables.pressure_index] = 1.0
    for i, row in enumerate(registered_variables.magnetic_index):
        state[row] = b[i]
    state[registered_variables.cosmic_ray_e_index] = e_cr
    state = jnp.asarray(state)
    config = finalize_config(config, state.shape)

    cr_params = dict(
        gamma_cr=GAMMA_CR,
        reduced_streaming_speed=v_red_value,
        diffusion_coefficient=KAPPA_PAR,
        perpendicular_diffusion_coefficient=v["kappa_perp"],
        minimum_e_cr=E_FLOOR,
    )
    if not v["guard"]:
        cr_params.update(GUARD_OFF)
    def params_for(chunk_myr):
        return SimulationParams(
            t_end=chunk_myr * MYR_CODE,
            cosmic_ray_grey_params=CosmicRayGreyParams(**cr_params),
        )

    def moments(s):
        e_raw = np.asarray(s[registered_variables.cosmic_ray_e_index], dtype=np.float64)
        e = np.maximum(e_raw - E_FLOOR, 0.0)
        total = e.sum()
        return dict(
            total=total,
            e_min=float(e_raw.min()),
            z2=(e * z**2).sum() / total,
            above={h: e[np.abs(z) > h].sum() / total for h in Z_THRESHOLDS},
            max_speed=float(np.abs(np.asarray(s[registered_variables.velocity_index.x])).max()),
        )

    times, records = [0.0], [moments(state)]
    wall = time.time()
    for target in checkpoints:
        state = time_integration(state, config, params_for(target - times[-1]), registered_variables)
        times.append(target)
        records.append(moments(state))
        print(f"snapshot {snapshot} variant {variant} v_red {v_red_value:g}: t = {times[-1]:.2f} Myr, <z^2>^0.5 = "
              f"{np.sqrt(records[-1]['z2']):.1f} pc, |z|>300: {records[-1]['above'][300.0]:.3e}, "
              f"wall {time.time() - wall:.0f} s", flush=True)
    assert np.all(np.isfinite([r["z2"] for r in records])), "Probe produced NaNs."

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_DIR / f"probe_s{snapshot}_{variant}{_tag(smooth_cells, radius_cells, v_red)}.npz",
        times_myr=np.array(times),
        z2=np.array([r["z2"] for r in records]),
        total=np.array([r["total"] for r in records]),
        above_300=np.array([r["above"][300.0] for r in records]),
        above_500=np.array([r["above"][500.0] for r in records]),
        max_speed=np.array([r["max_speed"] for r in records]),
        e_min=np.array([r["e_min"] for r in records]),
        wall_s=time.time() - wall,
    )


def _d_zz(result):
    """``D_zz`` in cm^2/s from the ``<z^2>`` growth over ``t >= FIT_START_MYR``."""
    t = result["times_myr"]
    window = t >= FIT_START_MYR
    slope = np.polyfit(t[window], result["z2"][window], 1)[0]  # pc^2 / Myr
    return 0.5 * float((slope * u.pc**2 / u.Myr).to(u.cm**2 / u.s).value)


def _summarize(tag=""):
    times_myr = np.load(M7_DATA)["time_myr"]
    rows = []
    print(f"--- deposit variant '{tag or 'M7 default'}' ---")
    for snapshot in SNAPSHOTS:
        results = {}
        for variant in VARIANTS:
            path = OUT_DIR / f"probe_s{snapshot}_{variant}{tag}.npz"
            if path.exists():
                results[variant] = dict(np.load(path))
        if "A" not in results:
            continue
        d = {k: _d_zz(r) for k, r in results.items()}
        row = dict(snapshot=snapshot, t_myr=float(times_myr[snapshot]), d=d, results=results)
        rows.append(row)
        line = f"t = {row['t_myr']:5.1f} Myr: D_zz(A) = {d['A']:.3e}"
        if "B" in d:
            line += f", S = {d['B'] - d['A']:+.3e}"
        if "C" in d:
            line += f", G = {d['A'] - d['C']:+.3e}"
        if "E" in d and "C" in d:
            line += f", S(guard off) = {d['E'] - d['C']:+.3e}"
        if "D" in d:
            e = d["A"] - d["D"]
            line += f", E = {e:+.3e} (error at N: {e / (1 - 2**-1.0):+.2e} .. {e / (1 - 2**-1.4):+.2e})"
            if "B" in d:
                ratio = abs(e / (1 - 2**-1.2)) / abs(d["B"] - d["A"])
                line += f", |err| / |S| = {ratio:.2f} -> " + (
                    "resolves kappa_perp" if ratio < 0.3 else "set by resolution"
                )
        for k, r in results.items():
            line += f"\n    {k}: f(|z|>300) at 1 / 3 Myr = {np.interp(1.0, r['times_myr'], r['above_300']):.3e} / {r['above_300'][-1]:.3e}, max|u| {r['max_speed'].max():.1e}, wall {float(r['wall_s']):.0f} s"
        print(line)

    if not rows:
        print("No results yet.")
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(4.5 * len(rows), 4), squeeze=False)
    labels = {"A": "kappa_perp = 0", "B": "kappa_perp = 1e26", "C": "kappa_perp = 0, guard off", "D": "kappa_perp = 0, 2x grid"}
    for ax, row in zip(axes[0], rows):
        for i, (k, r) in enumerate(row["results"].items()):
            ax.plot(r["times_myr"], np.sqrt(r["z2"]), "o-", ms=3, color=f"C{i}", label=labels[k])
        ax.set_title(f"M7 field at {row['t_myr']:.0f} Myr")
        ax.set_xlabel("t (Myr)")
        ax.set_ylabel("rms height of CRs (pc)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    pics = Path(__file__).resolve().parent / "pics"
    pics.mkdir(exist_ok=True)
    fig.savefig(pics / f"m7_frozen_field_probe{tag}.svg")


def _summarize_vred(variant="B"):
    """v_red scan (DESIGN.md step 5a): per M7 field snapshot, CR rms height and the energy fraction
    above |z| = 300 pc at several times, D_zz, and the minimum e_cr, for each v_red in
    V_RED_SCAN; relative differences to the largest v_red."""
    times_myr = np.load(M7_DATA)["time_myr"]
    rows = []
    for snapshot in SNAPSHOTS:
        res = {}
        for v_red in V_RED_SCAN:
            path = OUT_DIR / f"probe_s{snapshot}_{variant}{_tag(None, None, v_red)}.npz"
            if path.exists():
                res[v_red] = dict(np.load(path))
        if not res:
            continue
        ref = res[max(res)]
        print(f"--- M7 field at {times_myr[snapshot]:.0f} Myr (variant {variant}); "
              f"differences relative to v_red = {max(res):g} km/s ---")
        print(f"{'v_red':>8s} " + " ".join(f"{'rms_z@' + f'{t:g}':>11s}" for t in (0.1, 0.5, 1.0, 3.0))
              + f" {'f300@1':>9s} {'f300@3':>9s} {'D_zz':>10s} {'e_min':>10s}")
        for v_red, r in sorted(res.items()):
            rms = [np.sqrt(np.interp(t, r["times_myr"], r["z2"])) for t in (0.1, 0.5, 1.0, 3.0)]
            rms_ref = [np.sqrt(np.interp(t, ref["times_myr"], ref["z2"])) for t in (0.1, 0.5, 1.0, 3.0)]
            f1, f3 = np.interp(1.0, r["times_myr"], r["above_300"]), r["above_300"][-1]
            f1r, f3r = np.interp(1.0, ref["times_myr"], ref["above_300"]), ref["above_300"][-1]
            d, dr = _d_zz(r), _d_zz(ref)
            print(f"{v_red:8g} " + " ".join(f"{a:6.1f}({(a / b - 1) * 100:+4.0f}%)" for a, b in zip(rms, rms_ref))
                  + f" {f1:9.2e} {f3:9.2e} {d:10.3e} {r['e_min'].min():10.2e}"
                  + f"   [f300 {((f1 / f1r - 1) * 100 if f1r else 0):+.0f}% / {((f3 / f3r - 1) * 100 if f3r else 0):+.0f}%, D_zz {(d / dr - 1) * 100:+.0f}%]")
        rows.append((snapshot, res))
    if not rows:
        print("No results yet.")
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(4.5 * len(rows), 4), squeeze=False)
    for ax, (snapshot, res) in zip(axes[0], rows):
        for i, (v_red, r) in enumerate(sorted(res.items())):
            ax.plot(r["times_myr"], np.sqrt(r["z2"]), "o-", ms=3, color=f"C{i}", label=f"v_red = {v_red:g} km/s")
        ax.set_xscale("log")
        ax.set_title(f"M7 field at {times_myr[snapshot]:.0f} Myr")
        ax.set_xlabel("t (Myr)")
        ax.set_ylabel("rms height of CRs (pc)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    pics = Path(__file__).resolve().parent / "pics"
    fig.savefig(pics / "m7_frozen_field_probe_vred_scan.svg")


if __name__ == "__main__":
    opts = dict(a.lstrip("-").split("=", 1) if "=" in a else (a.lstrip("-"), "1") for a in sys.argv[1:])
    smooth = float(opts["smooth-cells"]) if "smooth-cells" in opts else None
    radius = float(opts["radius-cells"]) if "radius-cells" in opts else None
    v_red_opt = float(opts["v-red-kms"]) if "v-red-kms" in opts else None
    if "summarize-vred" in opts:
        _summarize_vred(opts.get("variant", "B"))
    elif "summarize" in opts:
        _summarize(_tag(smooth, radius, v_red_opt))
    else:
        _run(int(opts["snapshot"]), opts["variant"], smooth, radius, v_red_opt)
