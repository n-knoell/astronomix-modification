"""S3: live DSA with constant zeta = 0.1, S-PE and S-SS, all N. CPU only.

Reads ``RAW/sedov/{pe,ss}/N*/z0.1.npz`` (``sedov_study.py``) and compares with
the self-similar two-fluid reference (``solve_sedov_two_fluid(0.1, "gas_rh")``,
B7.1):

* E_cr / E against f_cr (constant in time for the reference), against t and R/dx;
* the realized injection (``InjectionAccumulator``, binned per snapshot interval)
  and the finder's nominal injection zeta D against the reference injection rate
  (both x t / E), with D's split into area (A_inj / 4 pi R_ref^2) and Mach
  (M50 / M_true,ref);
* the deposit surplus: E_cr of the run over the adiabatic-kernel prediction from
  its own injection (B7.3 ``_adiabatic_kernel`` / ``_kernel_prediction``);
* history: the fraction of the injected energy, and of the final E_cr (kernel-
  weighted), injected while the run's front radius is < 8 cells;
* convergence of E_cr(t) / E_cr,ref and of the late injection ratio with N
  (observed orders from the chains of S1, pairwise against the exact value).

Figures ``pics/S3_Ecr_vs_t.png``, ``pics/S3_D_split_vs_Rdx.png``,
``pics/S3_convergence.png``; table ``pics/S3_summary.txt``.

Usage: python s3_analyse.py
"""

import common  # noqa: F401  (sys.path, autocvd patch)

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import INK, MUTED, PICS, RAW, _style, at_time, observed_order  # noqa: E402
from s1_analyse import CHAINS, XI0, _colors  # noqa: E402
import cr_sedov_injection_rate as b73  # noqa: E402
from astronomix.test_setups.reference_solutions.sedov_two_fluid import solve_sedov_two_fluid  # noqa: E402

ZETA = 0.1
E = 1.0
SETUPS = (("pe", "S-PE point explosion"), ("ss", "S-SS self-similar start"))
TIMES = (0.035, 0.07)
RDX_EARLY = 8.0


def load(setup, ref, kernel):
    runs = {}
    for f in sorted((RAW / "sedov" / setup).glob("N*/z0.1.npz")):
        d = dict(np.load(f))
        n = int(d["N"])
        t = d["t"]
        xi = ref.xi_0
        d["R_ref"] = xi * t**0.4
        d["Rdx"] = d["R_ref"] * n
        c0 = np.sqrt(d["gamma"] * d["p0"] / d["rho0"])
        d["M_ref"] = 0.4 * d["R_ref"] / t / c0
        d["ecr_ratio"] = d["E_cr"] / E / ref.f_cr
        # realized injection per snapshot interval (accumulator), x t / E
        edges = np.concatenate([[float(d["t_start"])], t])
        cum = np.concatenate([[0.0], np.cumsum(d["acc_inj"])])
        cum_at = np.interp(edges, np.concatenate([[float(d["t_start"])], d["acc_t"]]), cum)
        rate = np.diff(cum_at) / np.diff(edges)
        tmid = np.sqrt(edges[:-1] * np.maximum(edges[1:], 1e-12)) if edges[0] > 0 else 0.5 * (edges[:-1] + edges[1:])
        d["inj_real_ratio"] = rate * tmid / E / ref.injection_rate
        d["inj_nom_ratio"] = d["inj_nominal"] * t / E / ref.injection_rate
        d["D_ratio"] = d["D"] * t / E / ref.finder_dissipation_rate
        d["area"] = d["A_inj"] / (4 * np.pi * d["R_ref"] ** 2)
        d["mach"] = d["M50"] / d["M_ref"]
        d["injected"] = cum_at[1:]
        # adiabatic-kernel prediction from the realized injection rate
        d["E_cr_pred"] = b73._kernel_prediction(t, rate, d["E_cr"], kernel)
        d["surplus"] = d["E_cr"] / d["E_cr_pred"]
        # history: injection while the run's own front is < 8 cells across (before the
        # first snapshot: its radius there)
        acc_R = np.interp(d["acc_t"], t, d["R_front"]) * n
        early = acc_R < RDX_EARLY
        tot = d["acc_inj"].sum()
        d["early_inj_frac"] = d["acc_inj"][early].sum() / tot if tot > 0 else np.nan
        s, K, _ = kernel
        w = np.interp(d["acc_t"] / t[-1], s, K)
        d["early_ecr_frac"] = (d["acc_inj"] * w)[early].sum() / max((d["acc_inj"] * w).sum(), 1e-300)
        runs[n] = d
    return dict(sorted(runs.items()))


def main():
    ref = solve_sedov_two_fluid(ZETA, convention="gas_rh")
    kernel = b73._adiabatic_kernel(ref)
    data = {label: load(setup, ref, kernel) for setup, label in SETUPS}
    data = {k: v for k, v in data.items() if v}
    if not data:
        raise SystemExit("no S3 runs yet")
    PICS.mkdir(exist_ok=True)
    lines = [f"# S3: live DSA, zeta = {ZETA}; reference f_cr = {ref.f_cr:.4f}, injection t/E = "
             f"{ref.injection_rate:.4f}, D t/E = {ref.finder_dissipation_rate:.4f}"]

    # E_cr vs t and vs R/dx
    fig, axes = plt.subplots(len(data), 2, figsize=(10, 3.3 * len(data)), constrained_layout=True, squeeze=False)
    for row, (label, runs) in enumerate(data.items()):
        colors = _colors(list(runs))
        for n, d in runs.items():
            axes[row, 0].plot(d["t"], d["ecr_ratio"], "-o", ms=2.5, color=colors[n], label=f"N = {n}")
            axes[row, 0].plot(d["t"], d["E_cr_pred"] / E / ref.f_cr, ":", lw=1, color=colors[n])
            axes[row, 1].plot(d["Rdx"], d["ecr_ratio"], "-o", ms=2.5, color=colors[n])
        axes[row, 0].set_xscale("log")
        axes[row, 1].set_xscale("log")
        axes[row, 0].set_xlabel("t")
        axes[row, 1].set_xlabel("R / dx")
        axes[row, 0].set_title(f"{label}: E_cr / E_cr,ref (dotted: kernel prediction)", fontsize=9, color=INK)
        axes[row, 1].set_title(f"{label}: E_cr / E_cr,ref vs cells per radius", fontsize=9, color=INK)
        for ax in axes[row]:
            ax.axhline(1.0, color=MUTED, ls="--", lw=1)
            _style(ax)
        axes[row, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("S3: CR energy, live DSA (zeta = 0.1)", color=INK)
    fig.savefig(PICS / "S3_Ecr_vs_t.png", dpi=140)
    plt.close(fig)

    # D split vs R/dx
    panels = (("inj_real_ratio", "realized injection / reference"), ("D_ratio", "finder D / reference D"),
              ("area", "A_inj / 4 pi R_ref^2"), ("mach", "M50 / M_true,ref"))
    fig, axes = plt.subplots(len(data), 4, figsize=(17, 3.3 * len(data)), constrained_layout=True, squeeze=False)
    for row, (label, runs) in enumerate(data.items()):
        colors = _colors(list(runs))
        for col, (key, title) in enumerate(panels):
            ax = axes[row, col]
            for n, d in runs.items():
                ok = np.isfinite(d[key])
                ax.plot(d["Rdx"][ok], d[key][ok], "-o", ms=2.5, color=colors[n], label=f"N = {n}")
            ax.axhline(1.0, color=MUTED, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("R / dx")
            ax.set_title(f"{label}: {title}" if col == 0 else title, fontsize=9, color=INK)
            _style(ax)
        axes[row, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("S3: injection and its finder split vs cells per shock radius", color=INK)
    fig.savefig(PICS / "S3_D_split_vs_Rdx.png", dpi=140)
    plt.close(fig)

    # convergence
    fig, axes = plt.subplots(1, len(data), figsize=(5.5 * len(data), 3.8), constrained_layout=True, squeeze=False)
    for col, (label, runs) in enumerate(data.items()):
        ax = axes[0, col]
        ns = np.array(list(runs))
        for key, style, name in (("ecr_ratio", "-o", "|E_cr / E_cr,ref - 1|"),
                                 ("inj_real_ratio", "-s", "|injection / ref - 1| (late mean)")):
            if key == "ecr_ratio":
                err = np.array([abs(at_time(runs[n], key, 0.07) - 1) for n in ns])
            else:
                err = np.array([abs(np.mean(runs[n][key][runs[n]["t"] >= 0.03]) - 1) for n in ns])
            ax.loglog(ns, err, style, ms=4, color=INK if key == "ecr_ratio" else MUTED, label=name)
        n_ref = np.array([ns[0], ns[-1]], float)
        e0 = abs(at_time(runs[ns[0]], "ecr_ratio", 0.07) - 1)
        ax.loglog(n_ref, e0 * (n_ref / n_ref[0]) ** -1, ":", color=MUTED, lw=1, label="N^-1")
        ax.set_xlabel("N")
        ax.set_title(label, fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=7)
        _style(ax)
    fig.suptitle("S3: convergence at t = 0.07", color=INK)
    fig.savefig(PICS / "S3_convergence.png", dpi=140)
    plt.close(fig)

    for label, runs in data.items():
        lines.append(f"\n## {label}")
        lines.append("N     R/dx(end)  E_cr/ref(0.035)  E_cr/ref(0.07)  inj_real/ref(late)  inj_nom/ref(late)  "
                     "D/ref(late)  area(late)  mach(late)  surplus(end)  early_inj  early_Ecr")
        for n, d in runs.items():
            late = d["t"] >= 0.03
            lines.append(f"{n:4d}  {d['Rdx'][-1]:8.1f}  {at_time(d, 'ecr_ratio', 0.035):15.4f}  "
                         f"{at_time(d, 'ecr_ratio', 0.07):14.4f}  {np.mean(d['inj_real_ratio'][late]):18.4f}  "
                         f"{np.mean(d['inj_nom_ratio'][late]):17.4f}  {np.mean(d['D_ratio'][late]):11.4f}  "
                         f"{np.mean(d['area'][late]):10.4f}  {np.mean(d['mach'][late]):10.4f}  "
                         f"{d['surplus'][-1]:12.4f}  {d['early_inj_frac']:9.4f}  {d['early_ecr_frac']:9.4f}")
        for t in TIMES:
            for chain in CHAINS:
                if all(n in runs for n in chain):
                    q = [at_time(runs[n], "ecr_ratio", t) for n in chain[::-1]]
                    p = observed_order([1.0 / n for n in chain[::-1]], q)
                    lines.append(f"order E_cr/ref t {t:g} chain {chain}: p {p:.2f}")
    text = "\n".join(lines)
    (PICS / "S3_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
