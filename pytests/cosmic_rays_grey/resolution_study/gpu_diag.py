"""GPU versions of the per-snapshot diagnostics (``--gpu-diag``; H200 extension, user 2026-11-13).

The same quantities as ``sedov_diag.analyse_snapshot``, the ``wind_study`` snapshot block and
``cwb_study.analyse_cwb_snapshot``, computed with ``jax.numpy`` on the GPU right after each
segment; only scalars and short 1D arrays (shell profiles, axis lines, quantile inputs) go to the
host. Reason: on the shared H200 node the host-side numpy reduction of every snapshot took
minutes (node memory pressure), the GPU idles meanwhile, and the user wants no CPU work.

Differences to the numpy versions: JAX runs in float32 here (x64 stays off, it would change the
simulation), so sums are float32 tree reductions instead of float64 ones (relative differences
~1e-6) and weighted quantiles use a float32 cumulative sum. Every grid-only quantity (shell bin
of each cell, completeness and |n|_inf bins, wind classes' radii) is computed once on the host
with the numpy code's own float64 arithmetic and uploaded as integer / float32 arrays, so the
bin assignment is identical.
"""

import numpy as np
import jax.numpy as jnp

from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_efficiency_of_mach, dsa_mach_and_flux, dsa_shock_finder_kwargs,
    inject_crs_at_shocks)
from sedov_diag import NINF_EDGES, SPHERE_BIN_FRACTIONS, shock_front


def weighted_quantiles_gpu(values, weights, qs):
    """``sedov_diag.weighted_quantiles`` on device arrays (1D)."""
    if values.size == 0:
        return [np.nan] * len(qs)
    order = jnp.argsort(values)
    v, w = values[order], weights[order]
    cum = jnp.cumsum(w) - 0.5 * w
    x = jnp.interp(jnp.asarray(qs, dtype=v.dtype) * jnp.sum(w), cum, v)
    return [float(a) for a in np.asarray(x)]


class ShellBins:
    """Radial shell bins of width ``width`` from 0 (``sedov_diag.shell_profiles`` binning):
    the bin index of every cell is computed once on the host (np.digitize, float64 edges),
    the per-snapshot bincounts run on the GPU."""

    def __init__(self, r_flat_host, edges):
        self.nb = len(edges)
        idx = np.digitize(r_flat_host, edges) - 1
        self.counts = np.bincount(idx, minlength=self.nb)[: self.nb - 1]
        self.centres = 0.5 * (edges[:-1] + edges[1:])
        self.idx = jnp.asarray(idx, dtype=jnp.int32)
        self._den = jnp.asarray(np.maximum(self.counts, 1), dtype=jnp.float32)

    def means(self, field):
        s = jnp.bincount(self.idx, weights=field.reshape(-1), length=self.nb)[: self.nb - 1]
        return np.asarray(s / self._den, dtype=np.float64)


class SedovGeometry:
    """Grid-only arrays of ``sedov_diag.analyse_snapshot``, once per run (host float64, uploaded)."""

    def __init__(self, helper_r, n, dx):
        r = np.asarray(helper_r)  # float32, as analyse_snapshot uses it
        self.shells = ShellBins(r.reshape(-1), np.arange(0.0, 0.5 * np.sqrt(3.0) + dx, 0.5 * dx))
        c = (np.arange(n) + 0.5) * dx - 0.5
        X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
        rr = np.maximum(r, 1e-12)
        ct, ph = Z / rr, np.arctan2(Y, X)
        bins = (np.clip(((ct + 1) / 2 * 10).astype(int), 0, 9) * 20
                + np.clip(((ph + np.pi) / (2 * np.pi) * 20).astype(int), 0, 19))
        self.cbins = jnp.asarray(bins.reshape(-1), dtype=jnp.int32)
        ninf = np.max(np.abs(np.stack([X, Y, Z])), axis=0) / rr
        self.nbin = jnp.asarray(np.digitize(ninf, NINF_EDGES[1:-1]), dtype=jnp.int8)


def analyse_snapshot_gpu(state, t, config, params, rv, helper_data, geom, rho0, gamma, gamma_cr=4.0 / 3.0):
    """``sedov_diag.analyse_snapshot`` on the GPU (same keys, same definitions)."""
    state = jnp.asarray(state)
    dx = float(config.grid_spacing)
    cr_config = config.cosmic_ray_grey_config
    cr_params = params.cosmic_ray_grey_params
    r = jnp.asarray(helper_data.r)
    rho = state[rv.density_index]
    p_gas = state[rv.pressure_index]
    e_cr = state[rv.cosmic_ray_e_index]
    u = jnp.stack([state[rv.velocity_index.x], state[rv.velocity_index.y], state[rv.velocity_index.z]])
    out = {"t": float(t)}

    # hydro
    sh = geom.shells
    R, R_peak, compression, width = shock_front(sh.centres, sh.means(rho), sh.counts, rho0, dx)
    out.update(R_front=R, R_peak_rho=R_peak, compression=compression, shell_fwhm_cells=width)

    # finder with the DSA settings
    sf = find_shocks_pfrommer(state, config, rv, helper_data, mach_min=cr_params.dsa_mach_min,
                              **dsa_shock_finder_kwargs(cr_config))
    mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
    w = dsa_area_weights(state, sf, cr_config, rv)
    eff = dsa_efficiency_of_mach(cr_config, cr_params)(mach)
    surface = jnp.asarray(sf.shock_surface_cells)
    inj = surface & (flux > 0)
    near = surface & (jnp.abs(r - R) <= 2.0 * dx)
    fw = flux * w
    n_surf = int(jnp.sum(surface))
    out.update(
        n_surf=n_surf, n_inj=int(jnp.sum(inj)),
        A_cells=float(n_surf) * dx**2, A_inj=float(jnp.sum(jnp.where(surface, w, 0.0))) * dx**2,
        D=float(jnp.sum(fw)) * dx**2, inj_nominal=float(jnp.sum(eff * fw)) * dx**2,
        false_pos_frac=float(jnp.sum(surface & (jnp.abs(r - R) > 3.0 * dx))) / max(n_surf, 1),
    )
    m16, m50, m84 = weighted_quantiles_gpu(mach[inj], w[inj], (0.16, 0.5, 0.84))
    out.update(M16=m16, M50=m50, M84=m84)
    if sf.pre_shock_steps is not None:
        steps = jnp.asarray(sf.pre_shock_steps)[inj]
        out.update(walk_med=float(jnp.median(steps)) if steps.size else np.nan,
                   walk_max=float(jnp.max(steps)) if steps.size else np.nan)
    else:
        out.update(walk_med=np.nan, walk_max=np.nan)
    # completeness over (cos theta, phi) bins
    occ = jnp.zeros(200, jnp.int32).at[geom.cbins].max(near.reshape(-1).astype(jnp.int32))
    out["completeness"] = float(jnp.sum(occ)) / 200
    # orientation (|n|_inf of the position direction)
    e_inj = eff * fw * dx**2
    nbin = geom.nbin.reshape(inj.shape)
    mach_bin, e_bin = np.full(5, np.nan), np.zeros(5)
    for k in range(5):
        sel = inj & (nbin == k)
        if bool(jnp.any(sel)):
            mach_bin[k] = weighted_quantiles_gpu(mach[sel], w[sel], (0.5,))[0]
        e_bin[k] = float(jnp.sum(jnp.where(sel, e_inj, 0.0)))
    area_bin = SPHERE_BIN_FRACTIONS * 4 * np.pi * R**2
    shell_mean = e_bin.sum() / area_bin.sum() if e_bin.sum() > 0 else np.nan
    out.update(mach_ninf_bin=mach_bin, inj_per_area_ninf_bin=(e_bin / area_bin) / shell_mean)

    # probe injection (realized, after the cap) with a CFL-sized dt (as sedov_diag: already on the GPU)
    p_cr = (gamma_cr - 1.0) * e_cr
    c_eff = jnp.sqrt((gamma * p_gas + gamma_cr * p_cr) / rho)
    signal = max(float(jnp.max(jnp.max(jnp.abs(u), axis=0) + c_eff)), cr_params.reduced_streaming_speed)
    dt_probe = params.C_cfl * dx / signal
    injected = inject_crs_at_shocks(state, config, params, rv, helper_data, current_time=t, dt=dt_probe)
    out["inj_probe"] = float(jnp.sum(injected[rv.cosmic_ray_e_index] - e_cr)) * dx**3 / dt_probe
    out["dt_probe"] = dt_probe

    # energies
    out.update(
        E_th=float(jnp.sum(p_gas)) / (gamma - 1.0) * dx**3,
        E_kin=float(jnp.sum(0.5 * rho * jnp.sum(u**2, axis=0))) * dx**3,
        E_cr=float(jnp.sum(e_cr)) * dx**3,
    )
    out["E_tot"] = out["E_th"] + out["E_kin"] + out["E_cr"]
    return out


class SedovProfiles:
    """``sedov_study``'s transport profiles (shell means of rho, p, e_cr, v_r in dx / 2 bins of
    ``r_cells``) on the GPU; returns ``(centres, [means])`` like ``shell_profiles``."""

    def __init__(self, r_cells, cc, dx):
        self.shells = ShellBins(r_cells.reshape(-1), np.arange(0.0, 0.5 * np.sqrt(3.0) + dx, 0.5 * dx))
        X, Y, Z = np.meshgrid(cc, cc, cc, indexing="ij")
        self.xyz = [jnp.asarray(a, dtype=jnp.float32) for a in (X, Y, Z)]
        self.inv_r = jnp.asarray(1.0 / np.maximum(r_cells, 1e-12), dtype=jnp.float32)

    def __call__(self, s, rv):
        vel = (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)
        ur = sum(s[c] * x for c, x in zip(vel, self.xyz)) * self.inv_r
        fields = [s[rv.density_index], s[rv.pressure_index], s[rv.cosmic_ray_e_index], ur]
        return self.shells.centres, [self.shells.means(f) for f in fields]


class WindGeometry:
    """Grid-only arrays of the ``wind_study`` snapshot block (radial bins of dx, r of each cell)."""

    def __init__(self, r_np, bin_np, nb, X, Y, Z):
        self.nb = nb
        self.counts = np.bincount(bin_np.reshape(-1), minlength=nb).astype(float)
        self.idx = jnp.asarray(bin_np.reshape(-1), dtype=jnp.int32)
        self._den = jnp.asarray(np.maximum(self.counts, 1), dtype=jnp.float32)
        self.r = jnp.asarray(r_np, dtype=jnp.float32)
        self.xyz = [jnp.asarray(a, dtype=jnp.float32) for a in (X, Y, Z)]
        self.inv_r = jnp.asarray(1.0 / np.maximum(r_np, 1e-30), dtype=jnp.float32)

    def profile(self, field):
        return np.asarray(jnp.bincount(self.idx, weights=field.reshape(-1), length=self.nb) / self._den,
                          dtype=np.float64)


def analyse_wind_snapshot_gpu(s, t, config, params, rv, hd, geom, dx, box, rho0, gamma, classes):
    """The ``wind_study`` snapshot block on the GPU: ``(row dict, [prof rho, p, ecr, vr])``."""
    cr_params = params.cosmic_ray_grey_params
    rho, p, ecr = s[rv.density_index], s[rv.pressure_index], s[rv.cosmic_ray_e_index]
    u = jnp.stack([s[c] for c in (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)])
    vr = (u[0] * geom.xyz[0] + u[1] * geom.xyz[1] + u[2] * geom.xyz[2]) * geom.inv_r
    prof = [geom.profile(f) for f in (rho, p, ecr, vr)]
    r_bins = (np.arange(geom.nb) + 0.5) * dx
    R, R_peak, compression, width = shock_front(r_bins, prof[0], geom.counts, rho0, dx, r_max=box / 2)
    d = {"t": float(t), "R_front": R, "R_peak_rho": R_peak, "compression": compression,
         "shell_fwhm_cells": width}

    sf = find_shocks_pfrommer(s, config, rv, hd, mach_min=cr_params.dsa_mach_min,
                              **dsa_shock_finder_kwargs(config.cosmic_ray_grey_config))
    mach, flux = dsa_mach_and_flux(s, sf, config, params, rv)
    w = dsa_area_weights(s, sf, config.cosmic_ray_grey_config, rv)
    eff = dsa_efficiency_of_mach(config.cosmic_ray_grey_config, cr_params)(mach)
    surface = jnp.asarray(sf.shock_surface_cells)
    inj = surface & (flux > 0)
    fs = jnp.abs(geom.r - R) <= 4.0 * dx
    ts = geom.r < 0.6 * R
    cls = jnp.where(fs, 0, jnp.where(ts, 1, 2))
    n_surf = int(jnp.sum(surface))
    d.update(n_surf=n_surf, n_inj=int(jnp.sum(inj)),
             false_pos_frac=float(jnp.sum(surface & (cls == 2))) / max(n_surf, 1))
    fw = flux * w
    for k, c in enumerate(classes):
        sel = inj & (cls == k)
        d[f"inj_{c}"] = float(jnp.sum(jnp.where(sel, eff * fw, 0.0))) * dx**2
        d[f"D_{c}"] = float(jnp.sum(jnp.where(sel, fw, 0.0))) * dx**2
        d[f"A_{c}"] = float(jnp.sum(jnp.where(sel, w, 0.0))) * dx**2
        d[f"M16_{c}"], d[f"M50_{c}"], d[f"M84_{c}"] = weighted_quantiles_gpu(mach[sel], w[sel], (0.16, 0.5, 0.84))
    dv = dx**3
    d.update(E_th=float(jnp.sum(p)) / (gamma - 1.0) * dv, E_kin=float(jnp.sum(0.5 * rho * jnp.sum(u**2, axis=0))) * dv,
             E_cr=float(jnp.sum(ecr)) * dv)
    d["E_tot"] = d["E_th"] + d["E_kin"] + d["E_cr"]
    return d, prof


class CwbGeometry:
    """Grid-only arrays of ``cwb_study.analyse_cwb_snapshot`` (host float64, uploaded)."""

    def __init__(self, n, dx, box, stars, sphere_radius):
        c = (np.arange(n) + 0.5) * dx - 0.5 * box
        self.c = c
        X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
        self.centres = [jnp.asarray(a, dtype=jnp.float32) for a in (X, Y, Z)]
        self.spheres = []
        for star in stars:
            d = np.stack([X - star[0], Y - star[1], Z - star[2]])
            r = np.sqrt(np.sum(d**2, axis=0))
            shell = np.abs(r - sphere_radius) < 0.5 * dx
            # the shell's cells only: their unit radial vectors (float64 on the host -> float32)
            sel = np.flatnonzero(shell.reshape(-1))
            dn = (d.reshape(3, -1)[:, sel] / np.maximum(r.reshape(-1)[sel], 1e-12)).astype(np.float32)
            self.spheres.append((jnp.asarray(sel, dtype=jnp.int32), jnp.asarray(dn), int(shell.sum())))


def analyse_cwb_snapshot_gpu(state, t, config, params, rv, hd, geom, stars, v_inf, sphere_radius, classes):
    """``cwb_study.analyse_cwb_snapshot`` on the GPU (same keys, same definitions)."""
    state = jnp.asarray(state)
    n = state.shape[1]
    dx = float(config.grid_spacing)
    gamma = float(params.gamma)
    cr_config, cr_params = config.cosmic_ray_grey_config, params.cosmic_ray_grey_params
    rho = state[rv.density_index]
    p = state[rv.pressure_index]
    e_cr = state[rv.cosmic_ray_e_index]
    u = jnp.stack([state[i] for i in (rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z)])
    out = {"t": float(t)}
    rho_f, p_f, u_f = rho.reshape(-1), p.reshape(-1), u.reshape(3, -1)
    for k, (sel, dn, count) in enumerate(geom.spheres, start=1):
        rs, ps, us = rho_f[sel], p_f[sel], u_f[:, sel]
        ur = jnp.sum(us * dn, axis=0)
        v2 = jnp.sum(us**2, axis=0)
        dv = 4 * np.pi * sphere_radius**2 / max(count, 1)
        mdot = float(jnp.sum(rs * ur)) * dv
        lkin = float(jnp.sum(0.5 * rs * v2 * ur)) * dv
        ltot = lkin + float(jnp.sum(gamma / (gamma - 1.0) * ps * ur)) * dv
        out.update({f"mdot{k}": mdot, f"lkin{k}": lkin, f"ltot{k}": ltot})

    sf = find_shocks_pfrommer(state, config, rv, hd, mach_min=cr_params.dsa_mach_min,
                              **dsa_shock_finder_kwargs(cr_config))
    mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
    w = dsa_area_weights(state, sf, cr_config, rv)
    eff = dsa_efficiency_of_mach(cr_config, cr_params)(mach)
    surface = jnp.asarray(sf.shock_surface_cells)
    inj = surface & (flux > 0)
    out.update(n_surf=int(jnp.sum(surface)), n_inj=int(jnp.sum(inj)),
               D=float(jnp.sum(flux * w)) * dx**2, inj_nominal=float(jnp.sum(eff * flux * w)) * dx**2)

    # binary axis: the four central lines (1D results to the host)
    j0, j1 = n // 2 - 1, n // 2

    def line(a, red):
        return np.asarray(red(a[:, j0:j1 + 1, j0:j1 + 1], axis=(1, 2)))

    ax_rho = line(rho, jnp.mean).astype(np.float64)
    ax_p = line(p, jnp.mean).astype(np.float64)
    ax_ux = line(u[0], jnp.mean).astype(np.float64)
    ax_ecr = line(e_cr, jnp.mean).astype(np.float64)
    ax_surf = line(surface, jnp.any)
    ax_mach = line(jnp.where(surface, mach, 0.0), jnp.max).astype(np.float64)
    x = geom.c
    between = (x > stars[0][0]) & (x < stars[1][0])
    sign = np.flatnonzero(between[:-1] & between[1:] & (ax_ux[:-1] > 0) & (ax_ux[1:] <= 0))
    x_c = float(x[sign[0]] + dx * ax_ux[sign[0]] / (ax_ux[sign[0]] - ax_ux[sign[0] + 1])) if sign.size else np.nan
    out["x_contact"] = x_c
    for k, (lo, hi, pick) in enumerate(((stars[0][0], x_c, -1), (x_c, stars[1][0], 0)), start=1):
        cand = np.flatnonzero(ax_surf & (x > lo) & (x < hi)) if np.isfinite(x_c) else np.zeros(0, int)
        if cand.size:
            i = cand[pick]
            out[f"x_ts{k}"] = float(x[i])
            out[f"mach_ts{k}"] = float(ax_mach[i])
        else:
            out[f"x_ts{k}"], out[f"mach_ts{k}"] = np.nan, np.nan
    out.update(ax_rho=ax_rho, ax_p=ax_p, ax_ux=ax_ux, ax_ecr=ax_ecr, ax_mach=ax_mach)

    # classify injecting cells by the gas two cells upstream along the shock direction
    idx = jnp.argwhere(inj)  # (cells, 3), eager
    d_s = jnp.asarray(sf.shock_direction)[:, inj].T
    up = jnp.clip(jnp.rint(idx + 2.0 * d_s).astype(jnp.int32), 0, n - 1)
    u_up = u[:, up[:, 0], up[:, 1], up[:, 2]].T
    pos = jnp.stack([geom.centres[a][inj] for a in range(3)], axis=1)
    speed = jnp.linalg.norm(u_up, axis=1)
    cos = []
    for star in stars:
        rel = pos - jnp.asarray(star, dtype=pos.dtype)
        cos.append(jnp.sum(u_up * rel, axis=1) / jnp.maximum(speed * jnp.linalg.norm(rel, axis=1), 1e-30))
    cls = jnp.where(cos[0] >= cos[1], 0, 1)
    cls = jnp.where(speed < 0.3 * min(v_inf), 2, cls)
    e_inj = (eff * flux * w)[inj] * dx**2
    w_inj, m_inj = w[inj], mach[inj]
    for k, name in enumerate(classes):
        sel = cls == k
        out[f"inj_{name}"] = float(jnp.sum(jnp.where(sel, e_inj, 0.0)))
        out[f"A_{name}"] = float(jnp.sum(jnp.where(sel, w_inj, 0.0))) * dx**2
        q = weighted_quantiles_gpu(m_inj[sel], w_inj[sel], (0.16, 0.5, 0.84))
        out[f"M16_{name}"], out[f"M50_{name}"], out[f"M84_{name}"] = q

    # energies
    dv = dx**3
    side1 = (geom.centres[0] <= x_c) if np.isfinite(x_c) else jnp.ones_like(rho, bool)
    E_cr = float(jnp.sum(e_cr)) * dv
    out.update(E_th=float(jnp.sum(p)) / (gamma - 1.0) * dv, E_kin=float(jnp.sum(0.5 * rho * jnp.sum(u**2, axis=0))) * dv,
               E_cr=E_cr, E_cr_side1=float(jnp.sum(jnp.where(side1, e_cr, 0.0))) * dv,
               E_cr_side2=float(jnp.sum(jnp.where(side1, 0.0, e_cr))) * dv)
    return out
