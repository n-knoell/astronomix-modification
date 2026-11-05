"""S4: Mach sweep with live DSA (S-CP, p0 = 0.6, M_true ~26 -> 2.5), constant zeta = 0.1 and KR13. CPU only.

Reads ``RAW/sedov/cp_p0.6/N*/{z0.1,kr13}.npz``. For each model the reference
trajectory is that model's finest run (``s1_analyse.add_truth``), so M_true
includes the shock slowdown by the CRs. Per snapshot:

* M50 / M_true and the nominal injection sum(zeta(M) flux w) over the exact one,
  ``inj_true = 4 pi R^2 zeta(M) delta(M) rho_0 (M c_0)^3 / 2`` at M_true (delta: the
  finder's thermalization efficiency, Schaal & Springel 2015 eqs. 8-9);
* the prediction of a single-Mach error: area x [zeta delta M^3](M50) / [zeta delta M^3](M_true),
  and the amplification A(M) = d ln(zeta delta M^3) / d ln M (3 + d ln delta/d ln M +
  d ln zeta/d ln M), which turns a Mach error dM/M into a flux error A dM/M;
* realized injection (accumulator) over inj_true; E_cr(t).

Figures ``pics/S4_mach_error_map.png``, ``pics/S4_flux_amplification.png``,
``pics/S4_Ecr_KR13_vs_const.png``; table ``pics/S4_summary.txt``.

Usage: python s4_analyse.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import INK, MUTED, PICS, RAW, _style, at_time, observed_order  # noqa: E402
from s1_analyse import _colors, add_truth  # noqa: E402

GAMMA = 5.0 / 3.0
MODELS = (("z0.1", "constant zeta = 0.1"), ("kr13", "KR13 zeta(M)"))
SUBDIR = "cp_p0.6"


def delta(m, g=GAMMA):
    m = np.maximum(np.asarray(m, float), 1.0)
    m2 = m**2
    rj = (g + 1) * m2 / ((g - 1) * m2 + 2)
    pj = (2 * g * m2 - (g - 1)) / (g + 1)
    return np.maximum(2 / (g * (g - 1) * m2 * rj) * (pj - rj**g), 0.0)


def kr13(m):
    """Kang & Ryu 2013 fit as in cr_grey_injection.dsa_efficiency_kang_ryu_2013 (numpy copy)."""
    from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_efficiency_kang_ryu_2013
    return np.asarray(dsa_efficiency_kang_ryu_2013(np.asarray(m, np.float32)), float)


def zeta_of(model):
    return kr13 if model == "kr13" else (lambda m: 0.1 * np.ones_like(np.asarray(m, float)))


def amplification(model, m):
    f = lambda x: np.log(np.maximum(zeta_of(model)(x) * delta(x) * x**3, 1e-300))
    h = 1e-3
    return (f(m * np.exp(h)) - f(m * np.exp(-h))) / (2 * h)


def load(model):
    runs = {}
    for f in sorted((RAW / "sedov" / SUBDIR).glob(f"N*/{model}.npz")):
        d = dict(np.load(f))
        runs[int(d["N"])] = d
    if len(runs) < 2:
        return {}
    runs = add_truth(dict(sorted(runs.items())), "cp")
    z = zeta_of(model)
    for n, d in runs.items():
        M, R, c0 = d["M_ref"], d["R_ref"], d["c0"]
        d["inj_true"] = 4 * np.pi * R**2 * z(M) * delta(M) * 0.5 * d["rho0"] * (M * c0) ** 3
        d["inj_ratio"] = d["inj_nominal"] / d["inj_true"]
        g = lambda x: z(x) * delta(x) * x**3
        d["inj_pred"] = d["area"] * g(d["M50"]) / g(M)
        d["A"] = amplification(model, M)
        edges = np.concatenate([[float(d["t_start"])], d["t"]])
        cum = np.concatenate([[0.0], np.cumsum(d["acc_inj"])])
        cum_at = np.interp(edges, np.concatenate([[float(d["t_start"])], d["acc_t"]]), cum)
        rate = np.diff(cum_at) / np.diff(edges)
        inj_true_mid = np.interp(np.sqrt(edges[:-1] * edges[1:]), d["t"], d["inj_true"])
        d["real_ratio"] = rate / inj_true_mid
        d["injected"] = cum_at[1:]
    return runs


def main():
    data = {label: load(model) for model, label in MODELS}
    data = {k: v for k, v in data.items() if v}
    if not data:
        raise SystemExit("no S4 runs yet")
    PICS.mkdir(exist_ok=True)

    # Mach error map
    fig, axes = plt.subplots(1, len(data), figsize=(6 * len(data), 4.4), constrained_layout=True, squeeze=False)
    for ax, (label, runs) in zip(axes[0], data.items()):
        for n, d in runs.items():
            sc = ax.scatter(d["Rdx"], d["M_ref"], c=d["mach_ref"], cmap="Blues_r", vmin=0.75, vmax=1.0, s=22,
                            edgecolor="none")
        fig.colorbar(sc, ax=ax, label="M_finder / M_true")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("R / dx")
        ax.set_ylabel("M_true")
        ax.set_title(f"S-CP, {label}: Mach error over (R/dx, M_true), all N", fontsize=9, color=INK)
        _style(ax)
    fig.savefig(PICS / "S4_mach_error_map.png", dpi=140)
    plt.close(fig)

    # flux amplification
    fig, axes = plt.subplots(len(data), 3, figsize=(15, 3.6 * len(data)), constrained_layout=True, squeeze=False)
    for row, (label, runs) in enumerate(data.items()):
        colors = _colors(list(runs))
        for n, d in runs.items():
            axes[row, 0].plot(d["M_ref"], d["mach_ref"], "-o", ms=2.5, color=colors[n], label=f"N = {n}")
            axes[row, 1].plot(d["M_ref"], d["inj_ratio"], "-o", ms=2.5, color=colors[n])
            axes[row, 1].plot(d["M_ref"], d["inj_pred"], ":", lw=1, color=colors[n])
            axes[row, 2].plot(d["M_ref"], d["A"], "-", color=MUTED)
        titles = ("M_finder / M_true", "nominal injection / exact (dotted: single-Mach prediction)",
                  "amplification A = d ln(zeta delta M^3)/d ln M")
        for ax, title in zip(axes[row], titles):
            ax.set_xscale("log")
            ax.set_xlabel("M_true")
            ax.set_title(f"{label}: {title}" if ax is axes[row, 0] else title, fontsize=9, color=INK)
            _style(ax)
        for ax in axes[row, :2]:
            ax.axhline(1.0, color=MUTED, ls="--", lw=1)
        axes[row, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("S4: Mach error and its amplification into the injected flux (S-CP)", color=INK)
    fig.savefig(PICS / "S4_flux_amplification.png", dpi=140)
    plt.close(fig)

    # E_cr
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    styles = {"constant zeta = 0.1": "-", "KR13 zeta(M)": "--"}
    for label, runs in data.items():
        colors = _colors(list(runs))
        for n, d in runs.items():
            axes[0].plot(d["t"], d["E_cr"], styles[label], color=colors[n], lw=1.3,
                         label=f"{label}, N = {n}" if n in (min(runs), max(runs)) else None)
        ns = np.array(list(runs))
        e_end = np.array([runs[n]["E_cr"][-1] for n in ns])
        axes[1].plot(ns, e_end / e_end[-1], styles[label] + "o", color=INK, ms=4, label=label)
    axes[0].set_xlabel("t")
    axes[0].set_title("E_cr(t)", fontsize=9, color=INK)
    axes[1].set_xlabel("N")
    axes[1].set_title("E_cr(t_end) / finest", fontsize=9, color=INK)
    axes[1].axhline(1.0, color=MUTED, ls="--", lw=1)
    for ax in axes:
        _style(ax)
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle("S4: CR energy of the counter-pressure blast", color=INK)
    fig.savefig(PICS / "S4_Ecr_KR13_vs_const.png", dpi=140)
    plt.close(fig)

    lines = ["# S4: S-CP with DSA. Values at M_true = 10 / 5 / 3 (interpolated in M) and at t_end"]
    for label, runs in data.items():
        lines.append(f"\n## {label}")
        lines.append("N    | R/dx@M10 M/M_t inj/exact pred  A | R/dx@M5 M/M_t inj/exact pred  A | "
                     "R/dx@M3 M/M_t inj/exact pred  A | real/exact(late) | E_cr(end)/finest")
        e_fine = runs[max(runs)]["E_cr"][-1]
        for n, d in runs.items():
            order = np.argsort(d["M_ref"])
            parts = []
            for m in (10.0, 5.0, 3.0):
                if d["M_ref"].min() <= m <= d["M_ref"].max():
                    f = lambda k: float(np.interp(np.log(m), np.log(d["M_ref"][order]), d[k][order]))
                    parts.append(f"{f('Rdx'):6.1f} {f('mach_ref'):.3f} {f('inj_ratio'):.3f} {f('inj_pred'):.3f} "
                                 f"{f('A'):4.1f}")
                else:
                    parts.append(" " * 33)
            late = d["t"] >= 0.03
            lines.append(f"{n:4d} | " + " | ".join(parts) + f" | {np.nanmean(d['real_ratio'][late]):.3f}"
                         f"            | {d['E_cr'][-1] / e_fine:.4f}")
    # E_cr(t_end) is dominated by the initial CRs of the S-CP start; the injected
    # energy (accumulator) is the converging quantity
    lines.append("\n# total injected energy (accumulator) and Richardson extrapolation from grid triples")
    for label, runs in data.items():
        ns = sorted(runs)
        inj = {n: float(runs[n]["acc_inj"].sum()) for n in ns}
        lines.append(f"{label}: " + ", ".join(f"N{n} {inj[n]:.4f}" for n in ns))
        for trip in ((96, 128, 192), (128, 192, 224), (96, 192, 224)):
            if not all(n in inj for n in trip):
                continue
            h, q = [1.0 / n for n in trip[::-1]], [inj[n] for n in trip[::-1]]
            p = observed_order(h, q)
            if np.isfinite(p):
                ext = q[0] + (q[0] - q[1]) / ((h[1] / h[0]) ** p - 1)
                lines.append(f"  {trip}: p = {p:.2f}, extrapolated {ext:.4f}, finest / extrapolated "
                             f"{inj[max(ns)] / ext:.3f}")
    text = "\n".join(lines)
    (PICS / "S4_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
