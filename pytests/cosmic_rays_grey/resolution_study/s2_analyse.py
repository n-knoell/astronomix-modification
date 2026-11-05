"""S2: split the finder's resolution error into a hydro and a finder share (+ S1 orientation). CPU only.

Reads ``RAW/sedov/<subdir>/S2_control.npz`` (``s2_restrict.py``) and the S1 runs
(``s1_analyse.load_setup`` / ``add_truth`` for the reference M_true and R). At each
coarse N and kept snapshot, with q = M50 / M_true,ref (or A_inj / 4 pi R_ref^2):

* total error      q_native - 1           (what a run at that N gets)
* finder share     q_restricted - 1       (the finder at that N on a fine-grid flow)
* hydro share      q_native / q_restricted - 1

Figures ``pics/S2_error_split.png`` (error shares vs R/dx), ``pics/S1_orientation.png``
(dissipated flux per unit exact area and Mach number per |n|_inf bin vs N), table
``pics/S2_summary.txt``.

Usage: python s2_analyse.py
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyse import INK, MUTED, PICS, RAW, _style  # noqa: E402
from s1_analyse import SETUPS, add_truth, load_setup, _colors  # noqa: E402
from sedov_diag import NINF_EDGES  # noqa: E402

BIN_CENTRES = 0.5 * (NINF_EDGES[:-1] + NINF_EDGES[1:])


def rows_of(path, runs):
    d = dict(np.load(path))
    out = []
    for j in range(len(d["N"])):
        n, i, kind = int(d["N"][j]), int(d["snap"][j]), str(d["kind"][j])
        t = float(d["t"][j])
        ref_run = runs[max(runs)]
        M_ref = float(np.interp(np.log(t), np.log(ref_run["t"]), ref_run["M_ref"]))
        R_ref = float(np.interp(np.log(t), np.log(ref_run["t"]), ref_run["R_ref"]))
        out.append(dict(N=n, snap=i, kind=kind, t=t, Rdx=R_ref * n,
                        mach=d["M50"][j] / M_ref, area=d["A_inj"][j] / (4 * np.pi * R_ref**2),
                        width=0.5 * (d["M84"][j] - d["M16"][j]) / d["M50"][j],
                        flux_bin=d["flux_per_area_ninf_bin"][j], mach_bin=d["mach_ninf_bin"][j] / d["M50"][j]))
    return out


def main():
    data = {}
    for setup, subdir, label in SETUPS:
        path = RAW / "sedov" / subdir / "S2_control.npz"
        if path.exists():
            data[label] = rows_of(path, add_truth(load_setup(subdir), setup))
    if not data:
        raise SystemExit("no S2 output yet")
    PICS.mkdir(exist_ok=True)

    # error split
    fig, axes = plt.subplots(len(data), 2, figsize=(10, 3.2 * len(data)), constrained_layout=True, squeeze=False)
    lines = ["# S2: error split at the kept snapshots (q = M50 / M_true,ref resp. A_inj / 4 pi R_ref^2)",
             "setup                               quantity  N    snap  R/dx   total   finder  hydro"]
    for row, (label, rows) in enumerate(data.items()):
        native = {(r["N"], r["snap"]): r for r in rows if r["kind"] == "native"}
        restricted = {(r["N"], r["snap"]): r for r in rows if r["kind"] == "restricted"}
        ns = sorted({k[0] for k in restricted})
        colors = _colors(ns)
        for col, q in enumerate(("mach", "area")):
            ax = axes[row, col]
            for (n, i), rr in sorted(restricted.items()):
                ax.plot(rr["Rdx"], rr[q] - 1, "o", mfc="none", color=colors[n], ms=6)
                if (n, i) in native:
                    rn = native[(n, i)]
                    ax.plot(rn["Rdx"], rn[q] - 1, "o", color=colors[n], ms=6)
                    ax.plot([rn["Rdx"]] * 2, [rn[q] - 1, rr[q] - 1], "-", color=colors[n], lw=1)
                    lines.append(f"{label[:34]:34s}  {q:8s} {n:4d} {i:4d}  {rn['Rdx']:5.1f}  "
                                 f"{rn[q] - 1:+.3f}  {rr[q] - 1:+.3f}  {rn[q] / rr[q] - 1:+.3f}")
            for n in ns:
                ax.plot([], [], "o", color=colors[n], label=f"N = {n}")
            ax.plot([], [], "o", color=MUTED, label="native run")
            ax.plot([], [], "o", mfc="none", color=MUTED, label="restricted 192^3")
            ax.axhline(0, color=MUTED, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("R / dx")
            ax.set_title(f"{label}: " + ("M_finder / M_true - 1" if q == "mach" else "A_inj / 4 pi R^2 - 1"),
                         fontsize=9, color=INK)
            _style(ax)
        axes[row, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("S2: finder error at coarse N, native run (filled) vs restricted fine state (open)", color=INK)
    fig.savefig(PICS / "S2_error_split.png", dpi=140)
    plt.close(fig)

    # orientation (native kept states), S1
    fig, axes = plt.subplots(len(data), 2, figsize=(10, 3.2 * len(data)), constrained_layout=True, squeeze=False)
    for row, (label, rows) in enumerate(data.items()):
        native = [r for r in rows if r["kind"] == "native" and r["snap"] == 29]
        colors = _colors(sorted({r["N"] for r in native}))
        for r in sorted(native, key=lambda r: r["N"]):
            axes[row, 0].plot(BIN_CENTRES, r["flux_bin"], "-o", ms=3, color=colors[r["N"]], label=f"N = {r['N']}")
            axes[row, 1].plot(BIN_CENTRES, r["mach_bin"], "-o", ms=3, color=colors[r["N"]])
            lines.append(f"orientation {label[:30]:30s} N {r['N']:3d} t {r['t']:.3f}: flux/area per bin "
                         + " ".join(f"{v:.3f}" for v in r["flux_bin"]) + " | Mach/median "
                         + " ".join(f"{v:.3f}" for v in r["mach_bin"]))
        for ax, title in zip(axes[row], ("dissipated flux per exact area / shell mean", "Mach / median")):
            ax.axhline(1.0, color=MUTED, ls="--", lw=1)
            ax.set_xlabel("|n|_inf of the position (0.58 body diagonal, 1 axis)")
            ax.set_title(f"{label}: {title}", fontsize=9, color=INK)
            _style(ax)
        axes[row, 0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("S1: grid artefacts at t = 0.07 (native control runs)", color=INK)
    fig.savefig(PICS / "S1_orientation.png", dpi=140)
    plt.close(fig)
    text = "\n".join(lines)
    (PICS / "S2_summary.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
