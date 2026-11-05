"""C3 reference: semi-analytic injection at the CWB termination shocks. GPU (finder) + numpy.

For the final state of each ``RAW/cwb/N*/z0.1.npz`` run, the finder's injecting
surface cells of the wind-1 / wind-2 termination shocks (classified as in
``cwb_study.py``) are given the analytic upstream of a Chevalier & Clegg (1985)
thermally driven wind from star k with the *measured* Mdot_k and mechanical
luminosity L_k (window means through the 1.5 R_inj sphere, C0):

  M(r) = CC85 at r / R_inj,  v_inf = sqrt(2 L_k / Mdot_k),
  v(r) = v_inf / sqrt(1 + 2 / ((gamma - 1) M^2)),  rho(r) = Mdot_k / (4 pi r^2 v(r)),
  normal components with the finder's shock direction n: v_n = v |r_hat . n|,
  M_n = M |r_hat . n|,

  L_exp = zeta * sum delta(M_n) rho v_n^3 / 2 * w * dx^2   (w: the DSA area weight)

i.e. the run's own shock geometry and area weighting, but the exact upstream state
and Mach number. L_exp / (nominal injection with the finder's Mach and upstream
sampling) isolates the finder's Mach / sampling error on the CWB; L_exp itself
converges with the geometry only.

Output ``RAW/cwb/c_expect.npz`` (per N and class: L_exp, nominal injection, mean
|r_hat . n|, median M_n analytic and finder) and stdout.

Usage: python c_expect.py
"""

import common
from common import RAW, require_pinned_gpu, save_npz

import numpy as np
import jax.numpy as jnp

import _cwb_setup as cwb
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_efficiency_of_mach, dsa_mach_and_flux, dsa_shock_finder_kwargs)
from c_analyse import CLASSES, WINDOW, cc85_mach
from s4_analyse import delta
from sedov_diag import weighted_quantiles
from setups import build_cwb

GAMMA = 5.0 / 3.0


def main():
    print("GPU", require_pinned_gpu(), flush=True)
    rows = []
    for f in sorted(RAW.glob("cwb/N*/z0.1.npz")):
        d = np.load(f)
        n = int(d["N"])
        _, config, params, hd, rv = build_cwb(n, float(d["zeta"]))
        state = jnp.asarray(d["state_last"])
        cr_config, cr_params = config.cosmic_ray_grey_config, params.cosmic_ray_grey_params
        sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=cr_params.dsa_mach_min,
                                  **dsa_shock_finder_kwargs(cr_config))
        mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
        w = dsa_area_weights(state, sf, cr_config, rv)
        eff = dsa_efficiency_of_mach(cr_config, cr_params)(mach)
        mach, flux, w, eff = (np.asarray(a, dtype=np.float64) for a in (mach, flux, w, eff))
        inj = np.asarray(sf.shock_surface_cells) & (flux > 0)
        dx = 1.0 / n
        c = (np.arange(n) + 0.5) * dx - 0.5
        centres = np.meshgrid(c, c, c, indexing="ij")
        u = np.stack([np.asarray(state[i], np.float64) for i in (rv.velocity_index.x, rv.velocity_index.y,
                                                                 rv.velocity_index.z)])
        idx = np.argwhere(inj)
        d_s = np.asarray(sf.shock_direction, dtype=np.float64)[:, inj].T
        up = np.clip(np.rint(idx + 2.0 * d_s).astype(int), 0, n - 1)
        u_up = u[:, up[:, 0], up[:, 1], up[:, 2]].T
        pos = np.stack([centres[a][inj] for a in range(3)], axis=1)
        speed = np.linalg.norm(u_up, axis=1)
        stars = np.asarray(d["stars"])
        cos = [np.sum(u_up * (pos - s), axis=1) / np.maximum(speed * np.linalg.norm(pos - s, axis=1), 1e-30)
               for s in stars]
        cls = np.where(cos[0] >= cos[1], 0, 1)
        cls = np.where(speed < 0.3 * min(cwb.v_inf1, cwb.v_inf2), 2, cls)
        win = d["t"] / d["t_end"] >= WINDOW
        r_inj = float(d["r_inj"])
        e_nom = (eff * flux * w)[inj] * dx**2
        for k in (0, 1):
            sel = cls == k
            mdot = float(np.mean(d[f"mdot{k + 1}"][win]))
            lum = float(np.mean(d[f"ltot{k + 1}"][win]))
            v_inf = np.sqrt(2 * lum / mdot)
            rel = pos[sel] - stars[k]
            r = np.linalg.norm(rel, axis=1)
            M = np.array([cc85_mach(x) for x in r / r_inj])
            v = v_inf / np.sqrt(1 + 2 / ((GAMMA - 1) * M**2))
            rho = mdot / (4 * np.pi * r**2 * v)
            mu = np.abs(np.sum(rel / r[:, None] * d_s[sel], axis=1))
            Mn = M * mu
            zeta = float(d["zeta"])
            L_exp = float(np.sum(zeta * delta(Mn) * 0.5 * rho * (v * mu) ** 3 * w[inj][sel] * dx**2))
            row = dict(N=n, cls=k, L_exp=L_exp, inj_nominal=float(e_nom[sel].sum()), mu_mean=float(np.mean(mu)),
                       Mn_analytic=weighted_quantiles(Mn, w[inj][sel], (0.5,))[0],
                       M_finder=weighted_quantiles(mach[inj][sel], w[inj][sel], (0.5,))[0],
                       mdot=mdot, lum=lum, v_inf_eff=v_inf, n_cells=int(sel.sum()))
            rows.append(row)
            print(f"N {n:3d} {CLASSES[k]}: cells {row['n_cells']:6d} L_exp {L_exp:.4e} nominal {row['inj_nominal']:.4e} "
                  f"(ratio {row['inj_nominal'] / L_exp:.3f}) M_n analytic {row['Mn_analytic']:.2f} finder "
                  f"{row['M_finder']:.2f} <|r.n|> {row['mu_mean']:.3f} v_inf,eff/v_inf {v_inf / (cwb.v_inf1, cwb.v_inf2)[k]:.3f}",
                  flush=True)
    out = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    print("saved", save_npz(RAW / "cwb" / "c_expect.npz", **out))


if __name__ == "__main__":
    main()
