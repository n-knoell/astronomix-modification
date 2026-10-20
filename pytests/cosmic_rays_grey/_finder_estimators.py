"""
Shared helpers for the DB4/DB7 diagnostics (``cr_sedov_mach_estimators.py``,
``cr_oblique_planar_shocks.py``): alternative Mach estimators at the shock finder's own
post-/pre-shock sample points, and alternative shock normals for the ``1 / |n|_inf`` area weight.
Analysis only; the finder's own ``mach_estimator="velocity"`` and ``surface_area_weights``
(DB4 = a, DB7, 2026-10-20) are the ``u`` estimator and the ``gaussT2`` normal here.
"""

# numerics
import numpy as np
import jax.numpy as jnp
from scipy import ndimage

# shock finder internals
from astronomix.shock_finder3D._shock_zones import get_post_pre_shock_values_adaptive
from astronomix.shock_finder3D._shock_mach import mach_from_pressure_samples, mach_from_velocity_jump
from astronomix.shock_finder3D._energy_dissipation import thermal_energy_flux_from_pre_state
from astronomix.shock_finder3D._gradients import _calculate_velocity_divergence

MACH_KEYS = ("p", "u", "umax", "T", "PT", "ptot")
NORMAL_KEYS = ("finder", "box3", "box5", "gaussT", "gaussT2")


def unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=0), 1e-30)


def ninf(vectors):
    """max_i |n_i| of unit vectors with the component axis first."""
    return np.max(np.abs(vectors), axis=0)


def mach_from_temperature_ratio(t_ratio, gamma):
    """Inverse of the RH temperature ratio
    T2/T1 = (2 g M^2 - (g - 1)) ((g - 1) M^2 + 2) / ((g + 1)^2 M^2), a quadratic in M^2."""
    t_ratio = np.maximum(t_ratio, 1.0)
    a = 2.0 * gamma * (gamma - 1.0)
    b = 4.0 * gamma - (gamma - 1.0) ** 2 - (gamma + 1.0) ** 2 * t_ratio
    c = -2.0 * (gamma - 1.0)
    m2 = (-b + np.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return np.sqrt(np.maximum(m2, 1.0))


def _sample(field, coords):
    return ndimage.map_coordinates(field, coords, order=1, mode="nearest")


def finder_samples(state, sf, config, rv, finder_kwargs, gamma, gamma_cr, select=None):
    """Mach estimators and fluxes at the finder's sample points.

    Repeats the finder's adaptive walk (same call as ``find_shocks_pfrommer`` with
    ``mach_sampling_adaptive`` and ``mach_sampling_extend``), so the post-shock sample is the
    pressure peak and the pre-shock sample the start of the upstream plateau, and evaluates
    every estimator of ``MACH_KEYS`` there:
    ``p`` gas-pressure jump (the finder's), ``u`` normal-velocity jump, ``umax`` the largest
    normal velocity along the post-shock walk, ``T`` temperature jump, ``PT`` = sqrt(p T),
    ``ptot`` gas + CR pressure jump.

    Args:
        state: primitive state (numpy).
        sf: ``find_shocks_pfrommer`` result for ``state``.
        select: optional boolean field; only valid surface cells inside it are used.

    Returns:
        dict with ``valid`` (boolean field of the cells used), ``M`` and ``flux`` (dicts over
        ``MACH_KEYS`` of per-cell arrays), ``p_pre``, ``rho_pre`` and ``closure_p`` (our flux
        of the estimator the finder uses, ``finder_kwargs["mach_estimator"]``, over the finder's
        own flux at the same cells; must be 1).
    """
    surface = np.asarray(sf.shock_surface_cells)
    mach_fin = np.asarray(sf.mach_numbers)
    valid = surface & (mach_fin > 0)
    if select is not None:
        valid = valid & select

    state_j = jnp.asarray(state)
    compressive = _calculate_velocity_divergence(state_j, config, rv, None) < 0
    walk = get_post_pre_shock_values_adaptive(
        sf.shock_direction, state_j[rv.pressure_index], state_j[rv.density_index], sf.shock_zones,
        max_steps=finder_kwargs["mach_sampling_steps"], extend_monotone=True,
        compressive=compressive, return_sample_steps=True,
    )
    p_post, p_pre, rho_post, rho_pre = (np.asarray(w)[valid] for w in walk[:4])
    s_post, s_pre = (np.asarray(w)[valid] for w in walk[6:8])

    n = np.asarray(sf.shock_direction)[:, valid]
    origin = np.stack(np.nonzero(valid)).astype(float)
    pre_xyz = origin + s_pre * n
    post_xyz = origin - s_post * n

    state = np.asarray(state)
    vel = np.stack([state[rv.velocity_index.x], state[rv.velocity_index.y],
                    state[rv.velocity_index.z]])
    rho, p = state[rv.density_index], state[rv.pressure_index]
    p_cr = (gamma_cr - 1.0) * state[rv.cosmic_ray_e_index]

    # consistency: our trilinear samples reproduce the walk's
    assert np.allclose(_sample(p, post_xyz), p_post, rtol=1e-4)
    assert np.allclose(_sample(rho, pre_xyz), rho_pre, rtol=1e-4)

    def u_n(xyz):
        return np.sum(np.stack([_sample(vel[d], xyz) for d in range(3)]) * n, axis=0)

    u_pre = u_n(pre_xyz)
    u_post = u_n(post_xyz)
    kmax = int(np.ceil(s_post.max())) if s_post.size else 0
    u_max = np.max(np.stack([np.where(k <= s_post + 1e-6, u_n(origin - k * n), -np.inf)
                             for k in range(kmax + 1)]), axis=0) if s_post.size else u_post

    g = gamma
    M = {
        "p": np.asarray(mach_from_pressure_samples(p_post, p_pre, g)),
        "u": np.asarray(mach_from_velocity_jump(u_post - u_pre, p_pre, rho_pre, g)),
        "umax": np.asarray(mach_from_velocity_jump(u_max - u_pre, p_pre, rho_pre, g)),
        "T": mach_from_temperature_ratio((p_post / rho_post) / (p_pre / rho_pre), g),
    }
    M["PT"] = np.sqrt(M["p"] * M["T"])
    M["ptot"] = np.asarray(mach_from_pressure_samples(
        p_post + _sample(p_cr, post_xyz), p_pre + _sample(p_cr, pre_xyz), g))
    flux = {k: np.asarray(thermal_energy_flux_from_pre_state(M[k], p_pre, rho_pre, g))
            for k in MACH_KEYS}
    finder_flux = float(np.asarray(sf.thermal_energy_flux)[valid].sum())
    used = "u" if finder_kwargs.get("mach_estimator", "pressure") == "velocity" else "p"
    return dict(valid=valid, M=M, flux=flux, p_pre=p_pre, rho_pre=rho_pre,
                closure_p=float(flux[used].sum()) / max(finder_flux, 1e-300))


def normal_variants(state, sf, rv, dx, mode="nearest"):
    """Shock normals at the surface cells (component axis first), per ``NORMAL_KEYS``:
    the finder's ``-grad T / |grad T|``; that averaged over the surface cells in a 3^3 / 5^3
    box; ``-grad T`` of a Gaussian-smoothed temperature (sigma = 1, 2 cells). ``mode``: boundary
    handling of the filters (``"wrap"`` for periodic boxes)."""
    surface = np.asarray(sf.shock_surface_cells)
    n_fin = np.asarray(sf.shock_direction)
    mask = surface.astype(float)
    out = {"finder": n_fin[:, surface]}
    for name, size in (("box3", 3), ("box5", 5)):
        avg = np.stack([ndimage.uniform_filter(n_fin[d] * mask, size=size, mode=mode)
                        for d in range(3)])
        out[name] = unit(avg[:, surface])
    state = np.asarray(state)
    temp = state[rv.pressure_index] / state[rv.density_index]
    for name, sigma in (("gaussT", 1.0), ("gaussT2", 2.0)):
        smooth = ndimage.gaussian_filter(temp, sigma=sigma, mode=mode)
        if mode == "wrap":
            grad = np.stack([(np.roll(smooth, -1, d) - np.roll(smooth, 1, d)) / (2.0 * dx)
                             for d in range(3)])
        else:
            grad = np.stack(np.gradient(smooth, dx))
        out[name] = unit(-grad[:, surface])
    return out
