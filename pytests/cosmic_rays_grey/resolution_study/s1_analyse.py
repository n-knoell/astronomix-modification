"""S1: hydro and finder on the control runs (zeta = 0), all N, S-PE / S-SS / S-CP. CPU only.

Reads ``RAW/sedov/{pe,ss,cp_p0.6}/N*/control.npz`` (``sedov_study.py``) and compares:

* M_true: Rdot / c_0 from (a) the run's own front trajectory ("own": the shock
  this grid actually has) and (b) the reference trajectory ("ref": the exact
  Sedov solution for S-PE / S-SS, the finest grid's trajectory for S-CP, where
  no analytic solution exists). M50 / M_true,own isolates the finder; M50 /
  M_true,ref is what DSA sees against the truth (finder + hydro).
* Mach vs R/dx (collapse test), Mach quantile width (per-cell scatter), Mach per
  |n|_inf bin (grid artefact), weighted area A_inj / 4 pi R_ref^2, completeness,
  false positives, front-radius error, peak compression and shell width in cells.
* Observed orders at fixed times from the chains (32, 64, 128), (48, 96, 192)
  and (64, 128, 224) (generalized three-grid Richardson) and, against the
  exact value, from pairs.

Figures ``pics/S1_*.png``, table ``pics/S1_summary.txt``.

Usage: python s1_analyse.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import INK, MUTED, PICS, RAW, _style, at_time, mach_true_from_trajectory, observed_order  # noqa: E402

XI0 = 1.15167  # Sedov, gamma 5/3
SETUPS = (("pe", "pe", "S-PE point explosion (p0 = 1e-4)"),
          ("ss", "ss", "S-SS self-similar start (p0 = 1e-4)"),
          ("cp", "cp_p0.6", "S-CP counter-pressure (p0 = 0.6)"))
CHAINS = ((32, 64, 128), (48, 96, 192), (64, 128, 224),
          (64, 128, 256), (96, 192, 384), (128, 224, 448))  # last three: H200 extension
TIMES = (0.01, 0.035, 0.07)


def load_setup(subdir):
    runs = {}
    for f in sorted((RAW / "sedov" / subdir).glob("N*/control.npz")):
        d = dict(np.load(f))
        runs[int(d["N"])] = d
    return dict(sorted(runs.items()))


def add_truth(runs, setup):
    """M_true (own and reference trajectory) and R_ref per snapshot."""
    n_fine = max(runs)
    fine = runs[n_fine]
    for n, d in runs.items():
        c0 = np.sqrt(d["gamma"] * d["p0"] / d["rho0"])
        d["c0"] = c0
        d["M_own"] = mach_true_from_trajectory(d["t"], d["R_front"], c0)
        if setup in ("pe", "ss"):
            d["R_ref"] = XI0 * d["t"] ** 0.4
            d["M_ref"] = 0.4 * d["R_ref"] / d["t"] / c0
        else:
            m_fine = mach_true_from_trajectory(fine["t"], fine["R_front"], c0)
            d["R_ref"] = np.interp(np.log(d["t"]), np.log(fine["t"]), fine["R_front"])
            d["M_ref"] = np.interp(np.log(d["t"]), np.log(fine["t"]), m_fine)
        d["Rdx"] = d["R_ref"] * n
        d["mach_own"] = d["M50"] / d["M_own"]
        d["mach_ref"] = d["M50"] / d["M_ref"]
        d["mach_width"] = 0.5 * (d["M84"] - d["M16"]) / d["M50"]
        d["area"] = d["A_inj"] / (4 * np.pi * d["R_ref"] ** 2)
        d["R_err"] = d["R_front"] / d["R_ref"] - 1
        d["ninf_spread"] = (np.nanmax(d["mach_ninf_bin"], axis=1) - np.nanmin(d["mach_ninf_bin"], axis=1)) / d["M50"]
    return runs


def _colors(ns):
    cmap = plt.get_cmap("Blues")
    return {n: cmap(0.35 + 0.6 * i / max(len(ns) - 1, 1)) for i, n in enumerate(ns)}


def grid_plot(data, panels, path, title, logx=True):
    """Rows = setups, columns = panels (key, label, reference line or None, ylim)."""
    fig, axes = plt.subplots(len(data), len(panels), figsize=(4.3 * len(panels), 3.1 * len(data)),
                             constrained_layout=True, squeeze=False)
    for row, (label, runs) in enumerate(data.items()):
        colors = _colors(list(runs))
        for col, (key, ylabel, ref, ylim) in enumerate(panels):
            ax = axes[row, col]
            for n, d in runs.items():
                ok = np.isfinite(d[key])
                ax.plot(d["Rdx"][ok], d[key][ok], "-o", ms=2.5, lw=1.4, color=colors[n], label=f"N = {n}")
            if ref is not None:
                ax.axhline(ref, color=MUTED, ls="--", lw=1)
            if logx:
                ax.set_xscale("log")
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_title(f"{label}: {ylabel}" if col == 0 else ylabel, fontsize=9, color=INK)
            ax.set_xlabel("R / dx (cells per shock radius)")
            _style(ax)
        axes[row, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(title, color=INK)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def orders(runs, key, exact):
    rows = []
    for t in TIMES:
        for chain in CHAINS:
            if not all(n in runs and runs[n]["t"].max() >= t - 1e-12 and runs[n]["t"].min() <= t for n in chain):
                continue
            q = [at_time(runs[n], key, t) for n in chain[::-1]]  # finest first
            p = observed_order([1.0 / n for n in chain[::-1]], q)
            rows.append((t, "-".join(map(str, chain)), "triple", p, q[::-1]))
        if exact is not None:
            ns = [n for n in runs if runs[n]["t"].min() <= t <= runs[n]["t"].max() + 1e-12]
            for a, b in zip(ns[:-1], ns[1:]):
                ea, eb = abs(at_time(runs[a], key, t) - exact), abs(at_time(runs[b], key, t) - exact)
                p = np.log(ea / eb) / np.log(b / a) if ea > 0 and eb > 0 else np.nan
                rows.append((t, f"{a}-{b}", "vs exact", p, [at_time(runs[a], key, t), at_time(runs[b], key, t)]))
    return rows


def main():
    data = {}
    for setup, subdir, label in SETUPS:
        runs = load_setup(subdir)
        if len(runs) >= 2:
            data[label] = add_truth(runs, setup)
    if not data:
        raise SystemExit("no S1 runs yet")
    PICS.mkdir(exist_ok=True)
    grid_plot(data, [("mach_ref", "M_finder / M_true (reference shock)", 1.0, (0.5, 1.1)),
                     ("mach_own", "M_finder / M_true (own shock)", 1.0, (0.5, 1.1)),
                     ("mach_width", "Mach half width (16-84%) / median", None, (0, 0.12)),
                     ("ninf_spread", "Mach spread over |n|_inf bins / median", None, (0, 0.3))],
              PICS / "S1_mach_vs_Rdx.png", "S1: DSA Mach number of the control runs vs cells per shock radius")
    grid_plot(data, [("area", "A_inj / 4 pi R_ref^2", 1.0, (0.8, 1.4)),
                     ("completeness", "completeness (solid-angle bins)", 1.0, (0.0, 1.05)),
                     ("false_pos_frac", "false-positive fraction", None, None)],
              PICS / "S1_area_vs_Rdx.png", "S1: shock surface area and detection vs cells per shock radius")
    grid_plot(data, [("R_err", "R_front / R_ref - 1", 0.0, None),
                     ("compression", "peak compression rho_max / rho_0", 4.0, (1, 4.5)),
                     ("shell_fwhm_cells", "shell FWHM (cells)", None, None),
                     ("walk_med", "median walk (cells)", None, None)],
              PICS / "S1_hydro.png", "S1: hydro of the control runs vs cells per shock radius")

    lines = ["# S1 summary: control runs, values at fixed times and observed orders",
             "# triple: generalized three-grid Richardson (NaN: differences change sign); vs exact: pairwise"]
    for label, runs in data.items():
        lines.append(f"\n## {label}: N = {list(runs)}")
        for key, exact in (("mach_ref", 1.0), ("mach_own", 1.0), ("area", 1.0), ("R_err", 0.0),
                           ("compression", None), ("mach_width", 0.0)):
            for t, grids, kind, p, q in orders(runs, key, exact):
                lines.append(f"{key:12s} t {t:<6g} {grids:12s} {kind:9s} p {p:6.2f}   values "
                             + " ".join(f"{v:.4f}" for v in q))
        lines.append("R/dx at t = 0.07: " + ", ".join(f"N{n} {at_time(d, 'Rdx', 0.07):.1f}" for n, d in runs.items()))
    text = "\n".join(lines)
    (PICS / "S1_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
