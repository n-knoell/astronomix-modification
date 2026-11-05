"""Analysis skeleton of the resolution study (PLAN.md 0.6). CPU only (numpy / matplotlib).

Reads reduced per-run files with B7.3's key layout (``<name>_<metric>``, one
entry per snapshot) and makes the standard views:

* ``plot_vs_rdx``: metrics against ``R / dx`` (cells per shock radius), one
  line per N: the self-similar collapse test (PLAN.md principle 2).
* ``plot_vs_n``: metrics at fixed times against N, with the reference value
  and the error slope.
* ``observed_orders``: observed convergence order from every grid triple
  (generalized Richardson for non-constant refinement ratios) and, where the
  exact value is known, from every pair.
* ``mach_true_from_trajectory``: M_true = Rdot / c_0 from a smooth fit of
  log R(log t), without assuming self-similarity (S-CP).

Usage:
  python analyse.py --b73                  # test on the B7.3 raw data (48/96/160)
  python analyse.py --setup pe --name dsa  # study runs in RAW/sedov/pe/N*/dsa.npz
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

RAW = Path("/export/scratch/nknoell/resolution_study")
PICS = Path(__file__).resolve().parent / "pics"
B73 = Path("/export/scratch/nknoell/phaseB/b7_3")
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e4e4e4"


def load_runs(files, name):
    """{N: {metric: array}} from npz files holding ``<name>_<metric>`` keys."""
    runs = {}
    for f in files:
        d = np.load(f)
        prefix = f"{name}_"
        metrics = {k[len(prefix):]: d[k] for k in d.files if k.startswith(prefix)}
        if not metrics:
            continue
        n = int(metrics["N"]) if "N" in metrics else int(Path(f).stem.split("_N")[-1])
        runs[n] = metrics
    return dict(sorted(runs.items()))


def mach_true_from_trajectory(t, R, c0, t_min=None, half_window=5):
    """Rdot / c_0 with Rdot = (R / t) dlnR/dlnt from local quadratic fits of ln R in
    ln t over +-``half_window`` snapshots (one-sided at the ends). A single global
    polynomial biases the end-point slope (S1: up to +10% at t_end)."""
    t, R = np.asarray(t, float), np.asarray(R, float)
    lt, lr = np.log(t), np.log(R)
    sel = t >= (t_min if t_min is not None else t.min())
    idx = np.flatnonzero(sel)
    out = np.full_like(t, np.nan)
    for k in idx:
        lo = max(idx[0], k - half_window)
        hi = min(idx[-1], k + half_window)
        lo, hi = max(idx[0], hi - 2 * half_window), min(idx[-1], lo + 2 * half_window)
        coef = np.polyfit(lt[lo:hi + 1] - lt[k], lr[lo:hi + 1], 2)
        out[k] = R[k] / t[k] * coef[1] / c0
    return out


def _colors(ns):
    """One hue, light to dark with N (sequential: resolution is a magnitude)."""
    cmap = plt.get_cmap("Blues")
    return {n: cmap(0.35 + 0.6 * i / max(len(ns) - 1, 1)) for i, n in enumerate(ns)}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def plot_vs_rdx(runs, metrics, path, refs=None, title=""):
    refs = refs or {}
    colors = _colors(list(runs))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.6), constrained_layout=True)
    for ax, m in zip(np.atleast_1d(axes), metrics):
        for n, d in runs.items():
            if m not in d:
                continue
            ax.plot(d["R_true"] * n, d[m], "-o", ms=3, lw=1.6, color=colors[n], label=f"N = {n}")
        if m in refs:
            ax.axhline(refs[m], color=MUTED, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("R / dx (cells per shock radius)")
        ax.set_title(m, fontsize=10, color=INK)
        _style(ax)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def at_time(d, m, t):
    """Metric ``m`` at time ``t`` (linear in log t between snapshots)."""
    return float(np.interp(np.log(t), np.log(d["t"]), d[m]))


def observed_order(h, q):
    """Observed order p from three grids h1 < h2 < h3 (any ratios) and their values:
    (q2 - q1) / (h2^p - h1^p) = (q3 - q2) / (h3^p - h2^p). NaN if the differences
    change sign (oscillatory, not in the asymptotic range) or no root in (0.05, 6)."""
    (h1, h2, h3), (q1, q2, q3) = h, q
    e21, e32 = q2 - q1, q3 - q2
    if e21 == 0 or e32 == 0 or np.sign(e21) != np.sign(e32):
        return np.nan
    f = lambda p: e21 * (h3**p - h2**p) - e32 * (h2**p - h1**p)
    try:
        return brentq(f, 0.05, 6.0)
    except ValueError:
        return np.nan


def observed_orders(runs, metric, times, exact=None):
    """Rows (t, grids, p) from every grid triple, plus pairwise orders against ``exact``."""
    ns = sorted(n for n in runs if metric in runs[n])
    rows = []
    for t in times:
        q = {n: at_time(runs[n], metric, t) for n in ns}
        for i in range(len(ns) - 2):
            trip = ns[i:i + 3][::-1]  # finest first
            p = observed_order([1.0 / n for n in trip], [q[n] for n in trip])
            rows.append((t, f"{trip[2]}-{trip[1]}-{trip[0]}", "triple", p))
        if exact is not None:
            for a, b in zip(ns[:-1], ns[1:]):
                ea, eb = abs(q[a] - exact), abs(q[b] - exact)
                p = np.log(ea / eb) / np.log(b / a) if ea > 0 and eb > 0 else np.nan
                rows.append((t, f"{a}-{b}", "vs exact", p))
    return rows


def plot_vs_n(runs, metric, times, path, exact=None, title=""):
    ns = sorted(n for n in runs if metric in runs[n])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    tones = plt.get_cmap("Greys")
    for k, t in enumerate(times):
        c = tones(0.45 + 0.5 * k / max(len(times) - 1, 1))
        q = np.array([at_time(runs[n], metric, t) for n in ns])
        axes[0].plot(ns, q, "-o", ms=4, color=c, label=f"t = {t:g}")
        if exact is not None:
            axes[1].loglog(ns, np.abs(q - exact), "-o", ms=4, color=c)
    if exact is not None:
        axes[0].axhline(exact, color=MUTED, ls="--", lw=1, label="reference")
        n_ref = np.array([ns[0], ns[-1]], float)
        e0 = np.abs(at_time(runs[ns[0]], metric, times[-1]) - exact)
        for p, ls in ((1, ":"), (2, "-.")):
            axes[1].loglog(n_ref, e0 * (n_ref / n_ref[0]) ** -p, ls, color=MUTED, lw=1, label=f"N^-{p}")
        axes[1].legend(frameon=False, fontsize=8)
        axes[1].set_title(f"|{metric} - reference|", fontsize=10, color=INK)
    for ax in axes:
        ax.set_xlabel("N (cells per box length)")
        _style(ax)
    axes[0].set_title(metric, fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, color=INK)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b73", action="store_true", help="test on the B7.3 raw data")
    ap.add_argument("--setup", default="pe")
    ap.add_argument("--name", default="dsa")
    ap.add_argument("--tag", default=None, help="figure/table name prefix")
    args = ap.parse_args()
    PICS.mkdir(exist_ok=True)
    if args.b73:
        files, tag = sorted(B73.glob("b7_3_N*.npz")), args.tag or "step0_b73"
    else:
        files, tag = sorted((RAW / "sedov" / args.setup).glob(f"N*/{args.name}.npz")), args.tag or f"{args.setup}_{args.name}"
    runs = load_runs(files, args.name)
    if not runs:
        raise SystemExit(f"no runs in {files}")
    c0 = np.sqrt(5.0 / 3.0 * 1e-4)
    for n, d in runs.items():
        d["M_true_traj"] = mach_true_from_trajectory(d["t"], d["R_num"], c0, t_min=0.0)
        d["mach_ratio_traj"] = d["M_finder"] / d["M_true_traj"]
        d["M_traj_over_selfsim"] = d["M_true_traj"] / d["M_true"]
    metrics = ["mach_ratio", "area_ratio_inj", "D_over_meas", "E_cr_over_E"]
    plot_vs_rdx(runs, metrics, PICS / f"{tag}_vs_Rdx.png", refs={"mach_ratio": 1, "area_ratio_inj": 1, "D_over_meas": 1},
                title=f"{tag}: metrics vs cells per shock radius")
    late = [t for t in (0.035, 0.07) if all(runs[n]["t"].max() >= t - 1e-12 for n in runs)]
    plot_vs_n(runs, "mach_ratio", late, PICS / f"{tag}_mach_vs_N.png", exact=1.0, title=f"{tag}: DSA Mach / true")
    lines = [f"# {tag}: observed orders (triples: generalized Richardson; pairs: vs exact)",
             "metric               t       grids           kind       p"]
    for m, exact in (("mach_ratio", 1.0), ("area_ratio_inj", 1.0), ("D_over_meas", 1.0), ("E_cr_over_E", None)):
        for t, grids, kind, p in observed_orders(runs, m, late, exact):
            lines.append(f"{m:20s} {t:<7g} {grids:15s} {kind:10s} {p:6.2f}")
    lines.append("")
    lines.append("# M_true from the trajectory fit vs B7.3's self-similar fit (late snapshots)")
    for n, d in runs.items():
        sel = d["t"] >= 0.03
        lines.append(f"N {n}: M_traj / M_selfsim median {np.median(d['M_traj_over_selfsim'][sel]):.4f}, "
                     f"range {d['M_traj_over_selfsim'][sel].min():.4f}-{d['M_traj_over_selfsim'][sel].max():.4f}")
    text = "\n".join(lines)
    (PICS / f"{tag}_orders.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
