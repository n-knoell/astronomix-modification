"""
Compare M7 runs (``m7_girichidis_pilot.py``) with each other and with Girichidis et al. (2016,
ApJL 816, L19; 2018, MNRAS 479, 3042) from their saved raw data, on the observables the papers
quote:

- the heights enclosing 70% / 90% of the gas mass (2016: >~0.2 / >~1.5 kpc at 250 Myr; early on
  "up to ~400 pc" before the field tangles);
- the mass-loading factor at |z| = 1 kpc (2016: order unity with CRs, 0.1-0.2 without; 2018:
  <eta> = 0.7-1.4 for the CR runs);
- the outflowing gas at |z| = 1 kpc, weighted by mass flux: temperature (2016: warm, ~1e4 K;
  2018: only 3% of the outflow mass above 3e5 K with CRs), density (~1e-26 - 1e-24 g/cm^3) and
  speed (2018: 30-40 km/s; 2016: launching speeds <~100 km/s). Girichidis+18 average density and
  velocity over the outflowing cells *by volume*; the table gives both the mass-flux-weighted
  speed and that volume-weighted one, and the plot compares the volume-weighted one with their
  30-40 km/s.

Every run is analysed with the same code (M7's own temperature conversion and outflow-plane
definition), so differences between runs are not analysis artefacts. Caveats for the comparison
with the papers: M7's box is +-2.5 kpc high (theirs +-20 kpc), so mass above 2.5 kpc is lost and
z90 is biased low; M7's eta is instantaneous at 10 Myr snapshots (the papers average over 2 Myr
or 100 Myr); and M7 runs at 33.3 pc (2016: 15.6 pc, 2018: 3.9 pc).

Usage::

    python m7_compare_runs.py LABEL=path/to/m7_pilot_data_*.npz [LABEL=... ...]

Prints a table per run (early: <= 50 Myr, late: last quarter of the snapshots) and writes
``pics/m7_compare_runs.svg``. NOT a committed pytest -- an analysis script for the M-series runs.
"""

# general
import sys
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# M7's constants and temperature conversion (imported with the default M7 arguments; the
# module's top level only sets up constants, it does not run a simulation).
sys.argv = [sys.argv[0]] + [a for a in sys.argv[1:]]
_RUN_ARGS = sys.argv[1:]
sys.argv = ["m7_girichidis_pilot.py", "both", "--res=0.9375"]
import autocvd  # noqa: E402
autocvd.autocvd = lambda *a, **k: [0]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import m7_girichidis_pilot as m7  # noqa: E402
get_temperature_from_pressure = m7.get_temperature_from_pressure

KPC = 1000.0
HOT_KELVIN = 3e5


def _analyse(path):
    d = np.load(path)
    t = d["time_myr"]
    n_z = int(d["n_cells"][2])
    dx = float(d["dx_pc"])
    z_rel = (np.arange(n_z) + 0.5) * dx - float(d["z0_pc"])
    i_up, i_dn = int(np.argmin(np.abs(z_rel - KPC))), int(np.argmin(np.abs(z_rel + KPC)))
    rho_cgs = float(d["code_density_g_cm3"])
    rows = []
    for k in range(len(t)):
        rho = d["snap_density"][k].astype(np.float64)
        p = d["snap_pressure"][k].astype(np.float64)
        vz = d["snap_velocity_z"][k].astype(np.float64)
        z70, z90 = m7._mass_heights(rho, z_rel)
        flux_planes, w_t, w_rho, w_v, hot = 0.0, 0.0, 0.0, 0.0, 0.0
        n_out, sum_v_out = 0, 0.0
        for i, sign in ((i_up, 1.0), (i_dn, -1.0)):
            v_out = sign * vz[:, :, i]
            mask = v_out > 0
            flux = np.where(mask, rho[:, :, i] * v_out, 0.0)
            temp = np.asarray(get_temperature_from_pressure(
                rho[:, :, i], p[:, :, i], m7.X_H, m7.Z_METAL)) / m7.KI_PARAMS.code_temperature_per_kelvin
            flux_planes += flux.sum()
            w_t += (flux * np.log10(np.maximum(temp, 1.0))).sum()
            w_rho += (flux * np.log10(rho[:, :, i] * rho_cgs)).sum()
            w_v += (flux * v_out).sum()
            hot += flux[temp > HOT_KELVIN].sum()
            n_out += int(mask.sum())
            sum_v_out += float(v_out[mask].sum())
        mdot = flux_planes * dx**2
        eta = mdot / m7.SFR_CODE
        v_vol = sum_v_out / n_out if n_out else np.nan
        if flux_planes > 0:
            rows.append([t[k], z70, z90, eta, 10 ** (w_t / flux_planes), 10 ** (w_rho / flux_planes),
                         w_v / flux_planes, hot / flux_planes, v_vol])
        else:
            rows.append([t[k], z70, z90, eta, np.nan, np.nan, np.nan, np.nan, v_vol])
    return np.array(rows)


COLUMNS = ["z70 (pc)", "z90 (pc)", "eta(1 kpc)", "T_out (K)", "rho_out (g/cm3)", "v_out flux (km/s)",
           "hot frac", "v_out vol (km/s)"]


def main():
    runs = dict(a.split("=", 1) for a in _RUN_ARGS)
    results = {label: _analyse(path) for label, path in runs.items()}
    for window, name in ((lambda t: (t > 0) & (t <= 50.0), "early (0 < t <= 50 Myr)"),
                         (None, "late (last quarter)")):
        print(f"--- {name} ---")
        print(f"{'run':24s}" + "".join(f"{c:>19s}" for c in COLUMNS))
        for label, r in results.items():
            sel = window(r[:, 0]) if window else np.arange(len(r)) >= len(r) - max(1, len(r) // 4)
            means = []
            for j in range(1, r.shape[1]):
                col = r[sel, j]
                means.append(np.nanmean(col) if j not in (4, 5) else 10 ** np.nanmean(np.log10(col)))
            print(f"{label:24s}" + "".join(f"{m:19.3g}" for m in means))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, j, ylabel in zip(axes.flat, (1, 2, 3, 8), ("z70 (pc)", "z90 (pc)", "eta at 1 kpc",
                                                     "outflow speed at 1 kpc, volume-weighted (km/s)")):
        for i, (label, r) in enumerate(results.items()):
            ax.plot(r[:, 0], r[:, j], "o-", ms=3, color=f"C{i}", label=label)
        ax.set_xlabel("t (Myr)")
        ax.set_ylabel(ylabel)
    axes[0, 0].axhline(200, color="grey", ls=":", label="Girichidis+16: >~200 pc")
    axes[0, 1].axhline(1500, color="grey", ls=":", label="Girichidis+16: >~1.5 kpc")
    axes[1, 0].axhspan(0.7, 1.4, color="grey", alpha=0.2, label="Girichidis+18 CR runs")
    axes[1, 1].axhspan(30.0, 40.0, color="grey", alpha=0.2, label="Girichidis+18 CR runs (outflowing gas, 1 kpc)")
    for ax in axes.flat:
        ax.legend(fontsize=7)
    fig.tight_layout()
    pics = Path(__file__).resolve().parent / "pics"
    fig.savefig(pics / "m7_compare_runs.svg")


if __name__ == "__main__":
    main()
