"""Per-snapshot Sedov diagnostics of the resolution study (PLAN.md Section 4, Sedov rows).

``analyse_snapshot`` reduces one state to scalars (plus a few fixed-length
arrays) without assuming self-similarity, so it serves S-PE, S-SS and S-CP:

* hydro: shell-averaged profiles (bins of dx/2) -> front radius (half maximum
  of the density excess on the outer flank), density-peak radius, peak
  compression rho_max / rho_0, shell width (FWHM of the density excess) in cells;
* finder with the DSA settings (``dsa_shock_finder_kwargs``): surface cells,
  completeness (fraction of 10 x 20 (cos theta, phi) bins holding a surface cell
  within 2 dx of the front), false positives (surface cells more than 3 dx off
  the front), area-weighted DSA Mach quantiles (16 / 50 / 84%) of the injecting
  cells, walk length;
* DSA: dissipated flux D = sum(flux w) dx^2, nominal injection
  sum(zeta(M) flux w) dx^2 (the model's own efficiency), weighted area A_inj,
  probe injection (one CFL-sized ``inject_crs_at_shocks`` call, after the cap);
* orientation: per |n|_inf bin of the cell's position (5 bins between the body
  diagonal and the axis), median Mach and nominal injection per unit exact area
  over its shell-mean (the grid-artefact metric of finder_v2_comparison);
* energies E_th, E_kin, E_cr, E_tot.

M_true needs the trajectory (``analyse.mach_true_from_trajectory``), so it is
computed afterwards, not here.
"""

import numpy as np
import jax.numpy as jnp

from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    dsa_area_weights, dsa_efficiency_of_mach, dsa_mach_and_flux, dsa_shock_finder_kwargs,
    inject_crs_at_shocks)

NINF_EDGES = np.array([1 / np.sqrt(3), 0.65, 0.75, 0.85, 0.95, 1.0])


def _sphere_bin_fractions(edges=NINF_EDGES, samples=2_000_000, seed=1):
    """Fraction of the unit sphere per |n|_inf bin (Monte Carlo, 1e-3 accurate)."""
    v = np.random.default_rng(seed).normal(size=(samples, 3))
    ninf = np.max(np.abs(v), axis=1) / np.linalg.norm(v, axis=1)
    counts, _ = np.histogram(ninf, bins=edges)
    return counts / samples


SPHERE_BIN_FRACTIONS = _sphere_bin_fractions()


def weighted_quantiles(values, weights, qs):
    if values.size == 0:
        return [np.nan] * len(qs)
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) - 0.5 * w
    return [float(np.interp(q * w.sum(), cum, v)) for q in qs]


def shell_profiles(r_flat, fields, dx):
    """Shell means of each field in radial bins of dx / 2; returns centres, means, counts."""
    edges = np.arange(0.0, 0.5 * np.sqrt(3.0) + dx, 0.5 * dx)
    idx = np.digitize(r_flat, edges) - 1
    counts = np.bincount(idx, minlength=len(edges))[: len(edges) - 1]
    means = [np.bincount(idx, weights=f, minlength=len(edges))[: len(edges) - 1] / np.maximum(counts, 1)
             for f in fields]
    return 0.5 * (edges[:-1] + edges[1:]), means, counts


def shock_front(centres, rho_mean, counts, rho0, dx, r_max=0.5):
    """Front radius (half maximum of the density excess, outer flank, linear
    interpolation), density-peak radius, peak compression and FWHM in cells."""
    valid = (centres > 2.0 * dx) & (centres < r_max) & (counts > 0)
    excess = np.where(valid, rho_mean - rho0, -np.inf)
    j = int(np.argmax(excess))
    half = 0.5 * excess[j]
    k = j
    while k + 1 < len(centres) and excess[k + 1] > half:
        k += 1
    h = centres[1] - centres[0]
    r_front = centres[k] + (excess[k] - half) / (excess[k] - excess[k + 1]) * h
    i = j
    while i - 1 >= 0 and excess[i - 1] > half:
        i -= 1
    r_inner = centres[i] - (excess[i] - half) / (excess[i] - excess[i - 1]) * h if i > 0 else centres[0]
    return float(r_front), float(centres[j]), float(rho_mean[j] / rho0), float((r_front - r_inner) / dx)


def analyse_snapshot(state, t, config, params, rv, helper_data, rho0, gamma, gamma_cr=4.0 / 3.0):
    """Reduced diagnostics of one Sedov snapshot (dict of floats / small arrays)."""
    state = jnp.asarray(state)
    dx = float(config.grid_spacing)
    cr_config = config.cosmic_ray_grey_config
    cr_params = params.cosmic_ray_grey_params
    r = np.asarray(helper_data.r)
    rho = state[rv.density_index]
    p_gas = state[rv.pressure_index]
    e_cr = state[rv.cosmic_ray_e_index]
    u = jnp.stack([state[rv.velocity_index.x], state[rv.velocity_index.y], state[rv.velocity_index.z]])
    out = {"t": float(t)}

    # hydro
    centres, (rho_m,), counts = shell_profiles(r.reshape(-1), [np.asarray(rho).reshape(-1)], dx)
    R, R_peak, compression, width = shock_front(centres, rho_m, counts, rho0, dx)
    out.update(R_front=R, R_peak_rho=R_peak, compression=compression, shell_fwhm_cells=width)

    # finder with the DSA settings
    sf = find_shocks_pfrommer(state, config, rv, helper_data, mach_min=cr_params.dsa_mach_min,
                              **dsa_shock_finder_kwargs(cr_config))
    mach, flux = dsa_mach_and_flux(state, sf, config, params, rv)
    w = dsa_area_weights(state, sf, cr_config, rv)
    eff = dsa_efficiency_of_mach(cr_config, cr_params)(mach)
    surface = np.asarray(sf.shock_surface_cells)
    mach, flux, w, eff = (np.asarray(a, dtype=np.float64) for a in (mach, flux, w, eff))
    inj = surface & (flux > 0)
    near = surface & (np.abs(r - R) <= 2.0 * dx)
    out.update(
        n_surf=int(surface.sum()), n_inj=int(inj.sum()),
        A_cells=float(surface.sum()) * dx**2, A_inj=float(w[surface].sum()) * dx**2,
        D=float((flux * w).sum()) * dx**2, inj_nominal=float((eff * flux * w).sum()) * dx**2,
        false_pos_frac=float((surface & (np.abs(r - R) > 3.0 * dx)).sum() / max(surface.sum(), 1)),
    )
    m16, m50, m84 = weighted_quantiles(mach[inj], w[inj], (0.16, 0.5, 0.84))
    out.update(M16=m16, M50=m50, M84=m84)
    steps = np.asarray(sf.pre_shock_steps)[inj] if sf.pre_shock_steps is not None else np.zeros(0)
    out.update(walk_med=float(np.median(steps)) if steps.size else np.nan,
               walk_max=float(steps.max()) if steps.size else np.nan)
    # completeness over (cos theta, phi) bins
    c = (np.arange(state.shape[1]) + 0.5) * dx - 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    rr = np.maximum(r, 1e-12)
    ct, ph = Z / rr, np.arctan2(Y, X)
    bins = (np.clip(((ct + 1) / 2 * 10).astype(int), 0, 9) * 20
            + np.clip(((ph + np.pi) / (2 * np.pi) * 20).astype(int), 0, 19))
    out["completeness"] = float(np.unique(bins[near]).size / 200)
    # orientation (|n|_inf of the position direction)
    ninf = np.max(np.abs(np.stack([X, Y, Z])), axis=0) / rr
    b = np.digitize(ninf, NINF_EDGES[1:-1])
    e_inj = eff * flux * w * dx**2
    mach_bin, e_bin = np.full(5, np.nan), np.zeros(5)
    for k in range(5):
        sel = inj & (b == k)
        if sel.any():
            mach_bin[k] = weighted_quantiles(mach[sel], w[sel], (0.5,))[0]
        e_bin[k] = e_inj[sel].sum()
    area_bin = SPHERE_BIN_FRACTIONS * 4 * np.pi * R**2
    shell_mean = e_bin.sum() / area_bin.sum() if e_bin.sum() > 0 else np.nan
    out.update(mach_ninf_bin=mach_bin, inj_per_area_ninf_bin=(e_bin / area_bin) / shell_mean)

    # probe injection (realized, after the cap) with a CFL-sized dt
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
