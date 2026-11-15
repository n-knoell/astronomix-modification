"""H200 extension phase 0 (EXTENSION_KICKSTART.md): H200 reruns vs the 2080 Ti files. CPU, numpy only.

For each pair: per-snapshot series compared as max_t |a - b| / max_t |b| (scale-normalised, so a
near-zero early value does not dominate), the total realised injection (sum of ``acc_inj``), the
step count, the final kept state (L1 relative difference per variable) and the wall-time ratio
(the H200 speed-up). Pass if every compared quantity differs by <= 1e-3.

Output: ``pics/H200_phase0_check.txt`` and stdout.
Usage: python h200_phase0_check.py
"""

from pathlib import Path

import numpy as np

RAW = Path("/export/scratch/nknoell/resolution_study")
PICS = Path(__file__).resolve().parent / "pics"
TOL = 1e-3

PAIRS = (
    ("S-PE 128 z0.1", "sedov/pe/N128/h200check_z0.1.npz", "sedov/pe/N128/z0.1.npz",
     ("M16", "M50", "M84", "R_front", "inj_nominal", "E_cr", "E_th", "E_kin")),
    ("CWB 128 z0.1", "cwb/N128/h200check_z0.1.npz", "cwb/N128/z0.1.npz",
     ("M50_wind1", "M50_wind2", "M50_ambient", "inj_wind1", "inj_wind2", "inj_ambient", "mach_ts1",
      "mach_ts2", "x_ts1", "x_ts2", "E_cr", "E_th")),
    ("W 128 lam0.1", "wind/N128/h200check_lam0.1.npz", "wind/N128/lam0.1.npz",
     ("M50_fs", "M50_ts", "R_front", "inj_acc", "E_cr", "E_th", "prof_ecr")),
    ("S-CP 128 z0.1 lam0.1", "sedov/cp_p0.6/N128/h200check_z0.1_lam0.1.npz", "sedov/cp_p0.6/N128/z0.1_lam0.1.npz",
     ("M16", "M50", "M84", "R_front", "inj_nominal", "E_cr", "E_th", "prof_ecr")),
)


def rel_series(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return np.nan, f"shape {a.shape} vs {b.shape}"
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if np.any(nan_a != nan_b):
        return np.inf, f"NaN pattern differs ({nan_a.sum()} vs {nan_b.sum()})"
    ok = ~nan_b
    scale = np.max(np.abs(b[ok])) if ok.any() else 0.0
    if scale == 0.0:
        return float(np.max(np.abs(a[ok]))) if ok.any() else 0.0, ""
    return float(np.max(np.abs(a[ok] - b[ok])) / scale), ""


def main():
    lines, speedups, all_pass = [], [], True
    for label, new_f, old_f, keys in PAIRS:
        lines.append(f"== {label}: {new_f} vs {old_f}")
        if not (RAW / new_f).exists():
            lines.append("   MISSING new file")
            all_pass = False
            continue
        new, old = np.load(RAW / new_f, allow_pickle=True), np.load(RAW / old_f, allow_pickle=True)
        worst = 0.0
        for k in keys:
            r, note = rel_series(new[k], old[k])
            worst = max(worst, r if np.isfinite(r) else np.inf)
            lines.append(f"   {k:<14} {r:9.2e} {'' if r <= TOL else 'FAIL'} {note}")
        inj_n, inj_o = float(np.sum(new["acc_inj"])), float(np.sum(old["acc_inj"]))
        r = abs(inj_n - inj_o) / abs(inj_o)
        worst = max(worst, r)
        lines.append(f"   {'injected':<14} {r:9.2e} {'' if r <= TOL else 'FAIL'} ({inj_n:.6e} vs {inj_o:.6e})")
        steps_n, steps_o = len(new["acc_t"]), len(old["acc_t"])
        lines.append(f"   steps          {steps_n} vs {steps_o}")
        for s in sorted(set(k for k in new.files if k.startswith("state_")) & set(old.files)):
            a, b = new[s].astype(float), old[s].astype(float)
            l1 = np.abs(a - b).sum(axis=(1, 2, 3)) / np.maximum(np.abs(b).sum(axis=(1, 2, 3)), 1e-300)
            worst = max(worst, float(l1.max()))
            lines.append(f"   {s} L1 rel per var: {' '.join(f'{x:.1e}' for x in l1)}"
                         f" {'' if l1.max() <= TOL else 'FAIL'}")
        w_n, w_o = float(np.sum(new["wall"])), float(np.sum(old["wall"]))
        speedups.append((label, w_o / w_n))
        lines.append(f"   wall           {w_n / 3600:.3f} h (H200) vs {w_o / 3600:.3f} h (2080 Ti): speed-up {w_o / w_n:.2f}"
                     f"  (s/step {w_n / steps_n:.3f} vs {w_o / steps_o:.3f})")
        lines.append(f"   gpu {new['gpu']}  peak {float(new['peak_gpu_GB']):.1f} GB (old {float(old['peak_gpu_GB']):.1f})"
                     f"  head {new['stamp_git_head']} diff {new['stamp_git_diff_stat']}")
        passed = worst <= TOL
        all_pass &= passed
        lines.append(f"   -> worst {worst:.2e}: {'PASS' if passed else 'FAIL'}")
    if speedups:
        lines.append("speed-up (2080 Ti wall / H200 wall): "
                     + ", ".join(f"{l} {s:.2f}" for l, s in speedups)
                     + f"; median {np.median([s for _, s in speedups]):.2f}")
    lines.append(f"OVERALL {'PASS' if all_pass else 'FAIL'} (tol {TOL:g})")
    text = "\n".join(lines)
    print(text)
    (PICS / "H200_phase0_check.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
