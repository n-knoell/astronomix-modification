"""X2: old segment-argmin surface vs the continuous-normal default on the same kept control
states. CPU only.

Reads ``RAW/sedov/<subdir>/X2_old_surface.npz`` (old) and the ``native`` rows of
``S2_control.npz`` (new) per setup; truth as in S1 (exact Sedov for pe / ss, the finest
trajectory for cp). Compares M50 / M_true, A_inj / 4 pi R^2 and D / D_true-proxy
(D_old / D_new) against R/dx, and the observed order of the Mach and area errors
(pairwise, error ~ (R/dx)^-p over the snapshots at the last kept time).

Figure ``pics/X2_old_vs_new_surface.png``, table ``pics/X2_summary.txt``.

Usage: python x2_analyse.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import INK, MUTED, PICS, RAW, _style  # noqa: E402
from s1_analyse import add_truth, load_setup  # noqa: E402

SETUPS = (("pe", "pe", "S-PE"), ("ss", "ss", "S-SS"), ("cp", "cp_p0.6", "S-CP"))


def truth(setup, sub):
    """(N, snapshot) -> (R_ref, M_ref) from the S1 control runs."""
    runs = add_truth(load_setup(sub), setup)
    return {(n, i): (d["R_ref"][i], d["M_ref"][i]) for n, d in runs.items() for i in range(len(d["t"]))}


def table(path, kind=None):
    d = dict(np.load(path))
    sel = np.ones(len(d["N"]), bool) if kind is None else d["kind"] == kind
    return {(int(n), int(i)): {k: d[k][j] for k in ("M50", "A_inj", "D", "R_front")}
            for j, (n, i) in enumerate(zip(d["N"], d["snap"])) if sel[j]}


def fit_order(x, err):
    ok = (err > 0) & np.isfinite(err) & (x >= 10)
    if ok.sum() < 3:
        return np.nan
    return -np.polyfit(np.log(x[ok]), np.log(err[ok]), 1)[0]


def main():
    fig, axes = plt.subplots(len(SETUPS), 3, figsize=(15, 3.4 * len(SETUPS)), constrained_layout=True, squeeze=False)
    lines = ["# X2: old (segment-argmin, 1/|n|_inf) vs new (continuous-normal, column weights) surface,",
             "#     same control states. p: power-law order of |error| vs R/dx (R/dx >= 10)"]
    for row, (setup, sub, label) in enumerate(SETUPS):
        old_p, new_p = RAW / "sedov" / sub / "X2_old_surface.npz", RAW / "sedov" / sub / "S2_control.npz"
        if not (old_p.exists() and new_p.exists()):
            continue
        tr = truth(setup, sub)
        old, new = table(old_p), table(new_p, "native")
        keys = sorted(k for k in old if k in new and k in tr)
        rdx = np.array([tr[k][0] * k[0] for k in keys])
        cols = {}
        for tag, tab in (("old", old), ("new", new)):
            cols[f"mach_{tag}"] = np.array([tab[k]["M50"] / tr[k][1] for k in keys])
            cols[f"area_{tag}"] = np.array([tab[k]["A_inj"] / (4 * np.pi * tr[k][0] ** 2) for k in keys])
        d_ratio = np.array([old[k]["D"] / new[k]["D"] for k in keys])
        for tag, mk in (("old", "s"), ("new", "o")):
            axes[row, 0].plot(rdx, cols[f"mach_{tag}"], mk, ms=4, mfc="none" if tag == "old" else None,
                              color=MUTED if tag == "old" else INK, label=f"{tag} surface")
            axes[row, 1].plot(rdx, cols[f"area_{tag}"], mk, ms=4, mfc="none" if tag == "old" else None,
                              color=MUTED if tag == "old" else INK)
        axes[row, 2].plot(rdx, d_ratio, "o", ms=4, color=INK)
        for ax, title in zip(axes[row], ("M50 / M_true", "A_inj / 4 pi R^2", "D_old / D_new")):
            ax.set_xscale("log")
            ax.axhline(1.0, color=MUTED, ls="--", lw=1)
            ax.set_xlabel("R / dx")
            ax.set_title(f"{label}: {title}", fontsize=9, color=INK)
            _style(ax)
        axes[row, 0].legend(frameon=False, fontsize=7)
        lines.append(f"\n## {label}")
        lines.append("N    snap  R/dx   M/M_t old  new   | area old  new   | D_old/D_new")
        for k, x, mo, mn, ao, an, dr in zip(keys, rdx, cols["mach_old"], cols["mach_new"], cols["area_old"],
                                             cols["area_new"], d_ratio):
            lines.append(f"{k[0]:4d} {k[1]:4d} {x:6.1f}   {mo:.3f}  {mn:.3f}   | {ao:.3f}  {an:.3f}  | {dr:.3f}")
        for q in ("mach", "area"):
            p_old = fit_order(rdx, np.abs(1 - cols[f"{q}_old"]))
            p_new = fit_order(rdx, np.abs(1 - cols[f"{q}_new"]))
            lines.append(f"order of |1 - {q}|: old {p_old:.2f}, new {p_new:.2f}")
    fig.suptitle("X2: old vs new DSA surface on the same control states", color=INK)
    fig.savefig(PICS / "X2_old_vs_new_surface.png", dpi=140)
    plt.close(fig)
    text = "\n".join(lines)
    (PICS / "X2_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
