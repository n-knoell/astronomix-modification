"""CWB analysis of the resolution study: C0 (winds, quasi-steady window), C1 (hydro and
finder, control runs), C3 (live DSA, zeta = 0.1). CPU only.

Reads ``RAW/cwb/N*/{control,z0.1}.npz`` (``cwb_study.py``).

* C0: Mdot and the mechanical luminosity (kinetic + enthalpy flux) through the
  1.5 R_inj spheres against the nominal Mdot and L_w = Mdot v_inf^2 / 2, per wind and N;
  the quasi-steady window: the last snapshots over which the axis positions and
  the injection rate stay within +-2% / +-5% of their window mean.
* C1: axis profiles (last snapshot); stand-offs x_TS1, x_c, x_TS2 (window mean) against
  the ram-pressure balance (nominal and measured Mdot); termination-shock Mach numbers
  on the axis against Chevalier & Clegg (1985) at r_s / R_inj; Mach quantiles per shock class.
* C3: E_cr(t); realized injection (accumulator) and nominal injection per shock class;
  ``C3_Ecr_per_Lwind_vs_N.png``: injected CR energy per total wind luminosity vs N,
  (a) cumulative E_inj(t_end) / (L_w t_end), (b) window-mean rate / L_w, total and per
  class, normalised with the nominal and with the measured L_w; the CR energy in the box
  dashed; a generalized three-grid Richardson value as reference where the order is
  in (0.3, 3); in (b) the semi-analytic L_exp / L_w of ``c_expect.py`` if present.

Figures ``pics/C0_winds.png``, ``pics/C1_axis_profiles.png``, ``pics/C1_standoff_mach_vs_N.png``,
``pics/C3_Ecr_vs_t.png``, ``pics/C3_injection_per_shock_vs_N.png``,
``pics/C3_Ecr_per_Lwind_vs_N.png``; table ``pics/C_summary.txt``.

Usage: python c_analyse.py
"""


import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from analyse import INK, MUTED, PICS, RAW, _style, observed_order  # noqa: E402

GAMMA = 5.0 / 3.0
CLASSES = ("wind1", "wind2", "ambient")
CLASS_LABEL = {"wind1": "wind-1 termination shock", "wind2": "wind-2 termination shock",
               "ambient": "shocks into ambient gas", "total": "total"}
WINDOW = 0.6  # quasi-steady window: t / t_end >= WINDOW (checked in C0)


def cc85_mach(x, g=GAMMA):
    """Chevalier & Clegg (1985) Mach number at r / R_inj = x > 1 (supersonic branch)."""
    if not np.isfinite(x) or x <= 1.0:
        return np.nan
    f = lambda m: (2 / (g - 1) * np.log(m) + (g + 1) / (2 * (g - 1)) * np.log(((g - 1) + 2 / m**2) / (g + 1))
                   - 2 * np.log(x))
    return brentq(f, 1.0 + 1e-9, 1e6)


def stagnation(mdot1, v1, mdot2, v2):
    """Distance of the ram-pressure balance point from star 1, as a fraction of the separation."""
    eta = mdot2 * v2 / (mdot1 * v1)
    return 1.0 / (1.0 + np.sqrt(eta))


def load(name):
    runs = {}
    for f in sorted(RAW.glob(f"cwb/N*/{name}.npz")):
        d = dict(np.load(f))
        n = int(d["N"])
        d["tf"] = d["t"] / d["t_end"]
        d["win"] = d["tf"] >= WINDOW
        lw = 0.5 * d["mdot_nominal"] * d["v_inf"] ** 2
        d["Lw_nominal"] = float(lw.sum())
        d["Lw_measured"] = float(np.mean(d["ltot1"][d["win"]] + d["ltot2"][d["win"]]))
        runs[n] = d
    return dict(sorted(runs.items()))


def _colors(ns):
    cmap = plt.get_cmap("Blues")
    return {n: cmap(0.35 + 0.6 * i / max(len(ns) - 1, 1)) for i, n in enumerate(ns)}


def richardson(ns, q):
    """Generalized three-grid Richardson value from the three finest grids (p in (0.3, 3)), else NaN."""
    if len(ns) < 3:
        return np.nan, np.nan
    n3 = ns[-3:][::-1]
    q3 = q[-3:][::-1]
    h = [1.0 / n for n in n3]
    p = observed_order(h, q3)
    if not (0.3 < p < 3):
        return np.nan, p
    return q3[0] + (q3[0] - q3[1]) / ((h[1] / h[0]) ** p - 1), p


def window_rate(d):
    """Realized injection rate (accumulator) averaged over the quasi-steady window."""
    t0 = WINDOW * d["t_end"]
    sel = d["acc_t"] >= t0
    return d["acc_inj"][sel].sum() / (d["t_end"] - t0)


def class_fractions(d):
    """Fraction of the nominal injection per class, window mean."""
    tot = sum(d[f"inj_{c}"][d["win"]].mean() for c in CLASSES)
    return {c: d[f"inj_{c}"][d["win"]].mean() / tot if tot > 0 else np.nan for c in CLASSES}


def class_cumulative(d):
    """Cumulative injected energy per class at t_end: the accumulator total split by the
    time-integrated nominal injection per class."""
    nominal = {c: np.trapezoid(d[f"inj_{c}"], d["t"]) for c in CLASSES}
    tot = sum(nominal.values())
    acc = d["acc_inj"].sum()
    return {c: acc * nominal[c] / tot if tot > 0 else np.nan for c in CLASSES}


def main():
    ctrl, dsa = load("control"), load("z0.1")
    if not ctrl and not dsa:
        raise SystemExit("no CWB runs yet")
    PICS.mkdir(exist_ok=True)
    lines = ["# CWB resolution study (C0 / C1 / C3)"]
    base = ctrl or dsa

    # ---- C0: winds and window
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    colors = _colors(list(base))
    lines.append("\n## C0: winds through the 1.5 R_inj spheres (window mean) and window stability")
    lines.append("run       N   Mdot1/nom Mdot2/nom  L1/Lw1  L2/Lw2  (kin+enth)  | window spread: x_ts1 x_c x_ts2 inj_rate")
    for label, runs in (("control", ctrl), ("z0.1", dsa)):
        for n, d in runs.items():
            w = d["win"]
            m1, m2 = d["mdot1"][w].mean() / d["mdot_nominal"][0], d["mdot2"][w].mean() / d["mdot_nominal"][1]
            lw = 0.5 * d["mdot_nominal"] * d["v_inf"] ** 2
            l1, l2 = d["ltot1"][w].mean() / lw[0], d["ltot2"][w].mean() / lw[1]
            spread = [np.nanmax(np.abs(d[k][w] - np.nanmean(d[k][w]))) / abs(d["stars"][0][0] - d["stars"][1][0])
                      for k in ("x_ts1", "x_contact", "x_ts2")]
            tot = sum(d[f"inj_{c}"] for c in CLASSES)
            irs = np.nanmax(np.abs(tot[w] / tot[w].mean() - 1)) if tot[w].mean() > 0 else np.nan
            lines.append(f"{label:8s} {n:4d}   {m1:8.3f} {m2:8.3f}  {l1:6.3f}  {l2:6.3f}               | "
                         f"{spread[0]:.3f} {spread[1]:.3f} {spread[2]:.3f} (of the separation)  {irs:.3f}")
            if label == "control":
                axes[0].plot(d["tf"], d["mdot1"] / d["mdot_nominal"][0], "-", color=colors[n], label=f"N = {n}")
                axes[0].plot(d["tf"], d["mdot2"] / d["mdot_nominal"][1], ":", color=colors[n])
                axes[1].plot(d["tf"], d["x_ts1"], "-", color=colors[n])
                axes[1].plot(d["tf"], d["x_contact"], "--", color=colors[n])
                axes[1].plot(d["tf"], d["x_ts2"], ":", color=colors[n])
    if dsa:
        for n, d in dsa.items():
            tot = sum(d[f"inj_{c}"] for c in CLASSES)
            axes[2].plot(d["tf"], tot / max(tot[d["win"]].mean(), 1e-300), "-", color=colors.get(n, INK))
    for ax, title in zip(axes, ("Mdot / nominal (solid wind 1, dotted wind 2)",
                                "axis: x_TS1 (solid), contact (dashed), x_TS2 (dotted)",
                                "nominal injection rate / window mean (DSA runs)")):
        ax.axvline(WINDOW, color=MUTED, ls="--", lw=1)
        ax.set_xlabel("t / t_end")
        ax.set_title(title, fontsize=9, color=INK)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("C0: wind injection and the quasi-steady window", color=INK)
    fig.savefig(PICS / "C0_winds.png", dpi=140)
    plt.close(fig)

    # ---- C1: axis profiles and stand-offs / Mach
    runs = ctrl or dsa
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), constrained_layout=True)
    colors = _colors(list(runs))
    for n, d in runs.items():
        x = d["x_axis"]
        rho, p, ux = d["ax_rho"][-1], d["ax_p"][-1], d["ax_ux"][-1]
        axes[0].semilogy(x, rho, color=colors[n], label=f"N = {n}")
        axes[1].semilogy(x, p, color=colors[n])
        axes[2].plot(x, np.abs(ux) / np.sqrt(GAMMA * p / rho), color=colors[n])
        axes[3].plot(x, d["ax_mach"][-1], "o", ms=3, color=colors[n])
    d = runs[max(runs)]
    r_inj = float(d["r_inj"])
    xs = np.linspace(d["stars"][0][0] + 1.05 * r_inj, 0.1, 200)
    axes[2].plot(xs, [cc85_mach((xx - d["stars"][0][0]) / r_inj) for xx in xs], "--", color=MUTED, lw=1,
                 label="Chevalier & Clegg (wind 1)")
    xs2 = np.linspace(-0.0, d["stars"][1][0] - 1.05 * r_inj, 200)
    axes[2].plot(xs2, [cc85_mach((d["stars"][1][0] - xx) / r_inj) for xx in xs2], ":", color=MUTED, lw=1,
                 label="Chevalier & Clegg (wind 2)")
    for ax, title in zip(axes, ("density", "pressure", "local Mach |u_x| / c_s", "finder Mach (surface cells)")):
        ax.set_xlabel("x (binary axis)")
        ax.set_title(title, fontsize=9, color=INK)
        _style(ax)
    axes[2].set_ylim(0, 20)
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("C1: binary-axis profiles at t_end (control runs)", color=INK)
    fig.savefig(PICS / "C1_axis_profiles.png", dpi=140)
    plt.close(fig)

    lines.append("\n## C1: stand-offs (window mean, x box-centred; stars at -0.2 / +0.2) and axis Mach numbers")
    lines.append("run       N  R_inj(cells) x_ts1    x_c      x_ts2   | M_ts1 finder / CC85(r_s/R) | M_ts2 finder / CC85 | "
                 "M50 wind1 [16,84] | M50 wind2 [16,84]")
    sep = 0.4
    for label, rr in (("control", ctrl), ("z0.1", dsa)):
        for n, d in rr.items():
            w = d["win"]
            x1, xc, x2 = (np.nanmean(d[k][w]) for k in ("x_ts1", "x_contact", "x_ts2"))
            r_inj = float(d["r_inj"])
            cc1 = cc85_mach((x1 - d["stars"][0][0]) / r_inj)
            cc2 = cc85_mach((d["stars"][1][0] - x2) / r_inj)
            m1, m2 = np.nanmean(d["mach_ts1"][w]), np.nanmean(d["mach_ts2"][w])
            q = lambda c: f"{np.nanmean(d[f'M50_{c}'][w]):6.2f} [{np.nanmean(d[f'M16_{c}'][w]):.2f},{np.nanmean(d[f'M84_{c}'][w]):.2f}]"
            lines.append(f"{label:8s} {n:4d} {r_inj * n:5.0f}       {x1:+.4f} {xc:+.4f} {x2:+.4f} | "
                         f"{m1:6.2f} / {cc1:6.2f}            | {m2:6.2f} / {cc2:6.2f}     | {q('wind1')} | {q('wind2')}")
        if rr:
            d = rr[max(rr)]
            nom = stagnation(d["mdot_nominal"][0], d["v_inf"][0], d["mdot_nominal"][1], d["v_inf"][1])
            lines.append(f"ram-pressure contact (nominal Mdot): x_c = {d['stars'][0][0] + nom * sep:+.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    for k, (key, style) in enumerate((("x_ts1", "-o"), ("x_contact", "--s"), ("x_ts2", ":^"))):
        ns = np.array(list(runs))
        axes[0].plot(ns, [np.nanmean(runs[n][key][runs[n]["win"]]) for n in ns], style, color=INK, ms=4, label=key)
    ns = np.array(list(runs))
    for k, style, cc_side in ((1, "-o", 0), (2, ":^", 1)):
        fm = [np.nanmean(runs[n][f"mach_ts{k}"][runs[n]["win"]]) for n in ns]
        xk = [np.nanmean(runs[n][f"x_ts{k}"][runs[n]["win"]]) for n in ns]
        cc = [cc85_mach(abs(x - runs[n]["stars"][cc_side][0]) / float(runs[n]["r_inj"])) for n, x in zip(ns, xk)]
        axes[1].plot(ns, fm, style, color=INK, ms=4, label=f"finder, TS{k}")
        axes[1].plot(ns, cc, style, color=MUTED, ms=4, mfc="none", label=f"CC85 at r_s/R_inj, TS{k}")
    for ax, title in zip(axes, ("axis positions (window mean)", "termination-shock Mach number on the axis")):
        ax.set_xlabel("N")
        ax.set_title(title, fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=7)
        _style(ax)
    fig.suptitle("C1: CWB stand-offs and axis Mach numbers vs N", color=INK)
    fig.savefig(PICS / "C1_standoff_mach_vs_N.png", dpi=140)
    plt.close(fig)

    # ---- C3
    if dsa:
        ns = np.array(list(dsa))
        colors = _colors(list(dsa))
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.8), constrained_layout=True)
        for n, d in dsa.items():
            cum = np.cumsum(d["acc_inj"])
            ax.plot(d["acc_t"] / d["t_end"], cum, "-", color=colors[n], label=f"injected, N = {n}")
            ax.plot(d["tf"], d["E_cr"], "--", color=colors[n])
        ax.set_xlabel("t / t_end")
        ax.set_title("injected CR energy (solid) and E_cr in the box (dashed)", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=7)
        _style(ax)
        fig.savefig(PICS / "C3_Ecr_vs_t.png", dpi=140)
        plt.close(fig)

        # per-shock injection vs N
        fig, ax = plt.subplots(1, 1, figsize=(6, 3.8), constrained_layout=True)
        rate = {n: window_rate(dsa[n]) for n in ns}
        for c, style in zip(CLASSES, ("-o", "-s", "-^")):
            ax.plot(ns, [rate[n] * class_fractions(dsa[n])[c] for n in ns], style, color=INK if c != "ambient" else MUTED,
                    ms=4, label=CLASS_LABEL[c])
        ax.set_xlabel("N")
        ax.set_title("window-mean realized injection rate per shock class", fontsize=9, color=INK)
        ax.legend(frameon=False, fontsize=7)
        _style(ax)
        fig.savefig(PICS / "C3_injection_per_shock_vs_N.png", dpi=140)
        plt.close(fig)

        # injected CR energy per wind luminosity
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
        lines.append("\n## C3: injected CR energy per total wind luminosity")
        lines.append("N    | cumulative E_inj/(L_w t_end): nominal L_w, measured L_w | window rate / L_w: nominal, measured | "
                     "per class (rate/L_w nominal) w1 w2 amb | E_cr box/(L_w t_end)")
        path = RAW / "cwb" / "c_expect.npz"
        expect = dict(np.load(path)) if path.exists() else None
        cum_nom, cum_meas, rate_nom, rate_meas, box = [], [], [], [], []
        per_class = {c: [] for c in CLASSES}
        per_class_cum = {c: [] for c in CLASSES}
        for n in ns:
            d = dsa[n]
            inj_tot = d["acc_inj"].sum()
            cum_nom.append(inj_tot / (d["Lw_nominal"] * d["t_end"]))
            cum_meas.append(inj_tot / (d["Lw_measured"] * d["t_end"]))
            rate_nom.append(rate[n] / d["Lw_nominal"])
            rate_meas.append(rate[n] / d["Lw_measured"])
            box.append(d["E_cr"][-1] / (d["Lw_nominal"] * d["t_end"]))
            fr, cc = class_fractions(d), class_cumulative(d)
            for c in CLASSES:
                per_class[c].append(rate[n] * fr[c] / d["Lw_nominal"])
                per_class_cum[c].append(cc[c] / (d["Lw_nominal"] * d["t_end"]))
            lines.append(f"{n:4d} | {cum_nom[-1]:.4e} {cum_meas[-1]:.4e} | {rate_nom[-1]:.4e} {rate_meas[-1]:.4e} | "
                         + " ".join(f"{per_class[c][-1]:.3e}" for c in CLASSES) + f" | {box[-1]:.4e}")
        for ax, (tot_nom, tot_meas, parts, title) in zip(
                axes, ((cum_nom, cum_meas, per_class_cum, "(a) cumulative: E_inj(t_end) / (L_w t_end)"),
                       (rate_nom, rate_meas, per_class, "(b) quasi-steady: <dE_inj/dt>_window / L_w"))):
            ax.plot(ns, tot_nom, "-o", color=INK, ms=5, label="total, nominal L_w")
            ax.plot(ns, tot_meas, "--o", color=INK, ms=5, mfc="none", label="total, measured L_w")
            for c, mk in zip(CLASSES, ("s", "^", "v")):
                ax.plot(ns, parts[c], "-" + mk, color=MUTED, ms=4, lw=1, label=CLASS_LABEL[c] + " (nominal L_w)")
            q_inf, p = richardson(list(ns), np.asarray(tot_nom))
            if np.isfinite(q_inf):
                ax.axhline(q_inf, color=INK, ls=":", lw=1, label=f"Richardson (p = {p:.2f})")
            lines.append(f"{title[:3]} Richardson of the total (nominal L_w): {q_inf:.4e}, p = {p:.2f}")
            if ax is axes[1] and expect is not None:
                l_exp = np.array([expect["L_exp"][expect["N"] == n].sum() / dsa[n]["Lw_nominal"] for n in ns])
                ax.plot(ns, l_exp, "-x", color=MUTED, lw=1.5, ms=6,
                        label="L_exp / L_w (exact CC85 upstream on the run's TS surface)")
                lines.append("(b) L_exp / L_w per N: " + ", ".join(f"N{n} {v:.4e}" for n, v in zip(ns, l_exp)))
            ax.set_xlabel("N (injection radius N/32 cells)")
            ax.set_title(title, fontsize=10, color=INK)
            _style(ax)
        axes[0].plot(ns, box, ":s", color=MUTED, ms=4, label="E_cr in the box / (L_w t_end)")
        for ax in axes:
            ax.legend(frameon=False, fontsize=7)
        fig.suptitle("C3: injected CR energy per total wind luminosity vs resolution (zeta = 0.1)", color=INK)
        fig.savefig(PICS / "C3_Ecr_per_Lwind_vs_N.png", dpi=140)
        plt.close(fig)

    text = "\n".join(lines)
    (PICS / "C_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
