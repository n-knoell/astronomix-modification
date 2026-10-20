"""
Area weight of shock-surface cells (Phase B decision DB7, 2026-10-19).

A shock with unit normal ``n`` crosses one surface cell per grid line along its dominant axis,
i.e. ``|n|_inf`` surface cells per ``dx^(ndim-1)`` of its area (``|n|_inf = max_i |n_i|``).
Counting every surface cell as ``dx^(ndim-1)`` therefore under-counts an oblique shock's area by
``|n|_inf`` (sphere average 0.831 in 3D; exactly ``|n|_inf`` on planar shocks, B8.2), so each
cell represents ``dx^(ndim-1) / |n|_inf``. Ryu et al. (2003) use the orientation average 1.19
instead, which is exact only for random normals.

The weight needs an accurate normal: the finder's ``-grad T`` is off by 3-6 deg at planar shocks
whose normal is not a grid symmetry direction (e.g. (2,1,0)) and by 5-7 deg at the Sedov shell,
always tilted away from the grid axes, so ``1 / |n|_inf`` from it over-weights by 5-10%
(averaging it over neighbouring surface cells does not help, the error is coherent). ``-grad T``
of the temperature smoothed with a Gaussian of ``sigma = 2`` cells is within 0.5 deg at planar
shocks and gives areas within 0.6% (planar) and 1-3% (Sedov) of the exact-normal weighting
(``pytests/cosmic_rays_grey/cr_oblique_planar_shocks.py``, ``cr_sedov_mach_estimators.py``).
"""

# general
from functools import partial
import math

# jax
import jax
import jax.numpy as jnp


def _gaussian_smooth(field, sigma):
    """Separable Gaussian filter, truncated at 4 sigma, edge-padded at the domain boundary."""
    radius = max(int(math.ceil(4.0 * sigma)), 1)
    offsets = jnp.arange(-radius, radius + 1, dtype=field.dtype)
    kernel = jnp.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)
    for axis in range(field.ndim):
        n = field.shape[axis]
        pad = [(0, 0)] * field.ndim
        pad[axis] = (radius, radius)
        padded = jnp.pad(field, pad, mode="edge")
        smooth = jnp.zeros_like(field)
        for j in range(2 * radius + 1):
            smooth = smooth + kernel[j] * jax.lax.slice_in_dim(padded, j, j + n, axis=axis)
        field = smooth
    return field


@partial(jax.jit, static_argnames=["sigma"])
def surface_area_weights(pressure, density, sigma=2.0):
    """``1 / |n|_inf`` of the unit normal ``-grad T / |grad T|`` of the Gaussian-smoothed
    temperature ``T = P / rho`` (``sigma`` in cells), at every cell: the area of the shock that
    a surface cell represents, in units of ``dx^(ndim-1)``. Between 1 (grid-aligned normal) and
    ``sqrt(ndim)`` (body diagonal); 1 where the smoothed gradient vanishes and in 1D.

    Gradient by central differences on the edge-padded smoothed field (one-sided halves at the
    domain boundary). Only the direction enters, so the grid spacing is not needed.
    """
    if pressure.ndim == 1:
        return jnp.ones_like(pressure)
    temperature = jax.lax.stop_gradient(pressure / density)
    smooth = _gaussian_smooth(temperature, sigma)
    grads = []
    for axis in range(smooth.ndim):
        pad = [(0, 0)] * smooth.ndim
        pad[axis] = (1, 1)
        padded = jnp.pad(smooth, pad, mode="edge")
        n = smooth.shape[axis]
        grads.append(jax.lax.slice_in_dim(padded, 2, n + 2, axis=axis)
                     - jax.lax.slice_in_dim(padded, 0, n, axis=axis))
    grads = jnp.stack(grads)
    norm_2 = jnp.sqrt(jnp.sum(grads**2, axis=0))
    norm_inf = jnp.max(jnp.abs(grads), axis=0)
    tiny = jnp.finfo(grads.dtype).tiny
    return jnp.where(norm_inf > tiny, norm_2 / jnp.maximum(norm_inf, tiny), 1.0)
