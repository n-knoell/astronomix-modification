"""X1: sensitivity of the S-CP (p0 = 0.6, zeta = 0.1) results at N = 96 to cell-based
numerical parameters, one at a time. CPU only.

Reads ``RAW/sedov/cp_p0.6/N96/x1_*.npz`` and the baseline ``N96/z0.1.npz``; truth as in S4
(the finest zeta = 0.1 run's trajectory). Per variant: M50 / M_true, nominal / exact
injection at M_true = 10 / 5 / 3, realized / exact (t >= 0.03) and the total injected
energy, each as the change against the baseline and in units of the baseline's own
error (1 - baseline), i.e. how much of the resolution error the parameter moves.

Table ``pics/X1_summary.txt``, figure ``pics/X1_sensitivity.png``.

Usage: python x1_analyse.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import INK, MUTED, PICS, RAW, _style  # noqa: E402
from s4_analyse import delta, load  # noqa: E402

N = 96
VARIANTS = (("z0.1", "baseline (steps 15, smoothing 2, mach_min 1.3, CFL 0.4, MINMOD)"),
            ("x1_steps8", "sampling max steps 8"), ("x1_steps25", "sampling max steps 25"),
            ("x1_smooth1", "area normal smoothing 1"), ("x1_smooth3", "area normal smoothing 3"),
            ("x1_machmin2", "mach_min 2"), ("x1_cfl0.2", "CFL 0.2"), ("x1_vanleer", "limiter VAN_LEER"))


def derive(d, fine):
    """M_ref / R_ref from the finest baseline trajectory, ratios as in s4_analyse.load (zeta = 0.1)."""
    d = dict(d)
    lt = np.log(d["t"])
    d["M_ref"] = np.interp(lt, np.log(fine["t"]), fine["M_ref"])
    d["R_ref"] = np.interp(lt, np.log(fine["t"]), fine["R_ref"])
    d["mach_ref"] = d["M50"] / d["M_ref"]
    c0 = fine["c0"]
    d["inj_true"] = 4 * np.pi * d["R_ref"] ** 2 * 0.1 * delta(d["M_ref"]) * 0.5 * d["rho0"] * (d["M_ref"] * c0) ** 3
    d["inj_ratio"] = d["inj_nominal"] / d["inj_true"]
    edges = np.concatenate([[float(d["t_start"])], d["t"]])
    cum = np.concatenate([[0.0], np.cumsum(d["acc_inj"])])
    cum_at = np.interp(edges, np.concatenate([[float(d["t_start"])], d["acc_t"]]), cum)
    rate = np.diff(cum_at) / np.diff(edges)
    d["real_ratio"] = rate / np.interp(np.sqrt(edges[:-1] * edges[1:]), d["t"], d["inj_true"])
    return d


def at_mach(d, key, m):
    order = np.argsort(d["M_ref"])
    return float(np.interp(np.log(m), np.log(d["M_ref"][order]), d[key][order]))


def main():
    runs = load("z0.1")
    fine = runs[max(runs)]
    base_dir = RAW / "sedov" / "cp_p0.6" / f"N{N}"
    data = {}
    for name, label in VARIANTS:
        f = base_dir / f"{name}.npz"
        if f.exists():
            data[name] = (label, derive(dict(np.load(f)), fine))
    if "z0.1" not in data or len(data) < 2:
        raise SystemExit("no X1 runs yet")
    base = data["z0.1"][1]
    late = lambda d: d["t"] >= 0.03
    metrics = [("M/M_t @M10", lambda d: at_mach(d, "mach_ref", 10)), ("M/M_t @M3", lambda d: at_mach(d, "mach_ref", 3)),
               ("inj/ex @M10", lambda d: at_mach(d, "inj_ratio", 10)), ("inj/ex @M3", lambda d: at_mach(d, "inj_ratio", 3)),
               ("real/ex late", lambda d: float(np.nanmean(d["real_ratio"][late(d)]))),
               ("injected", lambda d: float(d["acc_inj"].sum()))]
    ref224 = {k: f(fine) for k, f in metrics}
    lines = [f"# X1: S-CP zeta = 0.1 at N = {N}; value (change vs baseline in units of the baseline error 1 - x;",
             f"#     'injected': vs the {max(runs)}^3 value)",
             f"{'variant':36s} " + " ".join(f"{k:>20s}" for k, _ in metrics)]
    vals = {}
    for name, (label, d) in data.items():
        row = []
        vals[name] = {}
        for k, f in metrics:
            v, b = f(d), f(base)
            err = (ref224[k] - b) if k == "injected" else (1 - b)
            vals[name][k] = (v - b) / err
            row.append(f"{v:9.4f} ({(v - b) / err:+6.2f})")
        lines.append(f"{label[:36]:36s} " + " ".join(f"{r:>20s}" for r in row))
    lines.append(f"{max(runs)}^3 baseline                      " + " ".join(f"{ref224[k]:20.4f}" for k, _ in metrics))
    text = "\n".join(lines)
    (PICS / "X1_summary.txt").write_text(text + "\n")
    print(text)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name, (label, d) in data.items():
        style = dict(color=INK, lw=2) if name == "z0.1" else dict(lw=1)
        axes[0].plot(d["M_ref"], d["mach_ref"], label=label, **style)
        axes[1].plot(d["M_ref"], d["inj_ratio"], **style)
    axes[0].plot(fine["M_ref"], fine["mach_ref"], "--", color=MUTED, label=f"baseline {max(runs)}^3")
    axes[1].plot(fine["M_ref"], fine["inj_ratio"], "--", color=MUTED)
    for ax, title in zip(axes, ("M_finder / M_true", "nominal injection / exact")):
        ax.set_xscale("log")
        ax.set_xlabel("M_true")
        ax.set_title(title, fontsize=9, color=INK)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"X1: sensitivity to cell-based parameters, S-CP N = {N}", color=INK)
    fig.savefig(PICS / "X1_sensitivity.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
