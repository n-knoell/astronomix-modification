"""Extension D, step D0: sizing of the diffusion runs from the existing advective runs. CPU, numpy only.

For each setup and lambda = kappa / (v_sh R) (PLAN.md Section 9.3):

* shock speed v_sh(R) from the finest existing run (local log-slope of R(t), smoothed);
* stop radius R_stop = min(R_end, R_box / (1 + 3 lambda)) (box margin, 9.3 item 3; the wind
  bubble runs to t_end, its boundary loss is measured instead);
* kappa = lambda v_sh(R_stop) R_stop, rounded to 2 significant digits;
* v_red = 5 x the largest shock / upstream flow speed of the run (9.3 item 2);
* cost: per existing step dt_new^-1 = dt_old^-1 + 3 (v_red / sqrt 3) / (C_cfl dx), an upper
  bound (the CR signal speed added to the gas one), as a factor on the existing wall time.

CWB: the wind-2 termination shock (stationary): R = star 2 to x_ts2, u_1 = upstream wind
speed on the axis, averaged over the second half of the 192^3 run.

Output ``pics/D0_sizing.txt`` and ``D0_sizing.json`` (read by the job generator).
"""

import json
from pathlib import Path

import numpy as np

RAW = Path("/export/scratch/nknoell/resolution_study")
HERE = Path(__file__).resolve().parent
LAMBDAS = (0.03, 0.1, 0.3)
C_CFL = 0.4
VRED_FACTOR = 5.0


def sig2(x):
    return float(f"{x:.2g}")


def v_of_r(t, R):
    """v_sh(R) from a smoothed log-log fit of R(t) (quadratic in log t)."""
    ok = np.isfinite(R) & (R > 0)
    lt, lr = np.log(t[ok]), np.log(R[ok])
    c = np.polyfit(lt, lr, 2)
    slope = 2 * c[0] * lt + c[1]
    v = slope * R[ok] / t[ok]
    order = np.argsort(R[ok])
    return lambda r: float(np.interp(r, R[ok][order], v[order])), R[ok], v


def cost_factor(acc_dt, n, box, vred):
    s = vred / np.sqrt(3.0)
    dt_new = 1.0 / (1.0 / acc_dt + 3.0 * s / (C_CFL * box / n))
    return float(np.sum(acc_dt / dt_new) / len(acc_dt))


rows, plan = [], {}

# --- Sedov S-SS and S-CP (box 1, half-width 0.5) ---
for setup, sdir in (("ss", "ss"), ("cp", "cp_p0.6")):
    ref = np.load(RAW / "sedov" / sdir / "N192" / "z0.1.npz")
    f, Rs, vs = v_of_r(ref["t"], ref["R_front"])
    R_end = float(ref["R_front"][-1])
    v_max = float(np.max(vs))
    vred = sig2(VRED_FACTOR * v_max)
    walls = {}
    for n in (48, 64, 96, 128, 192, 224):
        p = RAW / "sedov" / sdir / f"N{n}" / "z0.1.npz"
        if p.exists():
            e = np.load(p)
            walls[n] = (float(e["wall"].sum()), cost_factor(e["acc_dt"], n, 1.0, vred))
    for lam in LAMBDAS:
        R_stop = min(R_end, 0.5 / (1 + 3 * lam))
        kappa = sig2(lam * f(R_stop) * R_stop)
        key = f"{setup}_lam{lam:g}"
        plan[key] = dict(setup=setup, lam=lam, kappa=kappa, vred=vred,
                         stop_radius=None if R_stop >= R_end else round(R_stop, 3),
                         v_sh_stop=f(R_stop), R_stop=R_stop,
                         hours={n: w * c / 3600 for n, (w, c) in walls.items()})
        rows.append((key, f"R {Rs.min():.3f}->{R_stop:.3f}", f(R_stop), kappa, vred, walls))

# --- wind bubble W (box 1.2, half-width 0.6; runs to t_end) ---
w = np.load(RAW / "db11" / "vred0.npz")
rb, rho, t = w["r_bins"], w["prof_rho"], w["snap_t"]
rho0 = float(rho[0][rb < 0.55][-1])
R = np.array([rb[np.where((rb < 0.6) & (rho[k] > 2 * rho0))[0].max()] if np.any((rb < 0.6) & (rho[k] > 2 * rho0))
              else np.nan for k in range(len(t))])
late = (t > 0.4 * float(w["t_end"])) & np.isfinite(R)
a = np.polyfit(np.log(t[late]), np.log(R[late]), 1)[0]
R_end = float(R[-1])
v_end = a * R_end / float(t[-1])
vred_w = 100.0  # 5 x v_inf (2000 km/s = 20 code), as the kappa scan and cr_wind_bubble_diffusion.py
dt_w = np.asarray(w["step_dt"])
cf = cost_factor(dt_w, 128, 1.2, vred_w)
for lam in LAMBDAS:
    kappa = sig2(lam * v_end * R_end)
    key = f"wind_lam{lam:g}"
    # wall of the advective N 128 probe run: 5989 steps; scale ~N^4 for the ladder
    plan[key] = dict(setup="wind", lam=lam, kappa=kappa, vred=vred_w, stop_radius=None, v_sh_stop=v_end,
                     R_stop=R_end, hours={n: 0.2 * cf * (n / 128) ** 4 for n in (64, 96, 128, 160, 192)})
    rows.append((key, f"R_fs {R_end:.3f} (t_end)", v_end, kappa, vred_w, {128: (720.0, cf)}))

# --- CWB, wind-2 termination shock ---
c = np.load(RAW / "cwb" / "N192" / "z0.1.npz")
x, star2 = c["x_axis"], float(c["stars"][1][0])
half = slice(len(c["t"]) // 2, None)
R2 = np.nanmean(star2 - c["x_ts2"][half])
u1 = []
for k in range(len(c["t"]))[half]:
    sel = (x > c["x_ts2"][k] + 2 * (x[1] - x[0])) & (x < star2 - 0.05)
    u1.append(np.max(np.abs(c["ax_ux"][k][sel])))
u1 = float(np.mean(u1))
vred_c = sig2(VRED_FACTOR * float(np.max(c["v_inf"])))
walls = {}
for n in (64, 96, 128, 160, 192):
    e = np.load(RAW / "cwb" / f"N{n}" / "z0.1.npz")
    # the existing runs already carry v_red = 1000: the factor is relative to that
    old = cost_factor(e["acc_dt"], n, 1.0, 0.0)
    walls[n] = (float(e["wall"].sum()), (813 + vred_c / np.sqrt(3)) / (813 + 1000 / np.sqrt(3)))
for lam in LAMBDAS:
    kappa = sig2(lam * u1 * R2)
    key = f"cwb_lam{lam:g}"
    plan[key] = dict(setup="cwb", lam=lam, kappa=kappa, vred=vred_c, stop_radius=None, v_sh_stop=u1, R_stop=R2,
                     lam_wind1=kappa / (813 * 0.2755),
                     hours={n: wv * cfac / 3600 for n, (wv, cfac) in walls.items()})
    rows.append((key, f"R_ts2 {R2:.3f}", u1, kappa, vred_c, walls))

lines = ["D0 sizing (PLAN.md Section 9): kappa = lambda v_sh R at the stop radius; v_red = 5 x max speed;",
         "hours = existing wall x cost factor (upper bound, 2080 Ti)",
         f"{'run':14s} {'range':24s} {'v_sh':>8s} {'kappa':>8s} {'v_red':>7s}  hours per N"]
for key, rng, v, kappa, vred, _ in rows:
    hrs = " ".join(f"{n}:{h:.2f}" for n, h in plan[key]["hours"].items())
    lines.append(f"{key:14s} {rng:24s} {v:8.3g} {kappa:8.3g} {vred:7.4g}  {hrs}")
total = sum(sum(p["hours"].values()) for p in plan.values())
lines.append(f"total (one zeta model, all N listed): {total:.1f} GPU-h; S-CP x2 for KR13")
text = "\n".join(lines)
print(text)
(HERE / "pics" / "D0_sizing.txt").write_text(text + "\n")
(HERE / "D0_sizing.json").write_text(json.dumps(plan, indent=1, default=float))
