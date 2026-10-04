"""
FFT-based Poisson solver for the gravitational potential.

Solves Poisson's equation for the gravitational potential in Fourier space.
Fully periodic domains use the spectral Green's function directly (with the
Jeans swindle subtracting the mean density). Non-periodic (open) boundaries use
the Hockney & Eastwood method, zero-padding the domain to twice its size and
convolving with the isolated-system Green's function. 3D domains periodic in
x/y and open in z combine the two (FFT in x/y, isolated convolution in z).
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp
from jax.numpy.fft import fftn, ifftn
from jax.sharding import PartitionSpec

try:  # the stable alias (newer jax)
    from jax import shard_map as _shard_map
except ImportError:  # pragma: no cover - older jax
    from jax.experimental.shard_map import shard_map as _shard_map

# astronomix helpers
from astronomix._pallas_helpers import _current_pallas_mesh

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig


def _active_x_sharded_mesh():
    """The active ``(mesh, x_axis_name)`` when tracing under a multi-device
    1D X-decomposed state mesh, else ``None``.

    ``time_integration`` sets the mesh contextvar around the JIT trace
    whenever the caller supplies a sharding, with the standard
    ``(VARAXIS, XAXIS, YAXIS, ZAXIS)`` axis layout.  Only a pure-X
    decomposition is supported by the slab-transposed FFT below; anything
    else falls back to the replicated ``fftn`` path.
    """
    mesh = _current_pallas_mesh()
    if mesh is None:
        return None
    names = tuple(mesh.axis_names)
    if len(names) < 4:
        return None
    x_name = names[1]
    if mesh.shape[x_name] <= 1:
        return None
    other_sharded = mesh.shape[names[0]] > 1 or any(
        mesh.shape[name] > 1 for name in names[2:]
    )
    if other_sharded:
        return None
    return mesh, x_name


def _periodic_poisson_3d_sharded(
    gas_density,
    grid_spacing: float,
    G,
    mesh,
    x_name,
):
    """Distributed periodic Poisson solve for an X-sharded 3D density.

    The SPMD partitioner replicates the operand of an FFT along its
    transform dimensions, so the plain ``fftn`` path costs *global*-grid
    memory and work per device.  This version keeps everything slab-local:
    FFT over (y, z) batched over the local X slab, an ``all_to_all``
    transpose of the slab decomposition from X to Y, the X-axis FFT, the
    Green's-function multiply on the local k-slab, and the mirrored
    inverse.  Per-device memory and FLOPs then stay proportional to the
    local shard, which is what makes self-gravity weak scaling flat.

    Mathematically identical to the replicated branch (same discrete
    transform, Green's function and regularisation); only the data layout
    differs.
    """
    num_cells_x, num_cells_y, num_cells_z = gas_density.shape
    num_slabs = mesh.shape[x_name]
    local_cells_y = num_cells_y // num_slabs

    # The state mesh uses *integer* axis names, which JAX's named
    # collectives (all_to_all, axis_index) would misread as positional
    # array axes.  Alias the same devices (same order, so no data movement)
    # under string names for the shard_map body.
    named_mesh = jax.sharding.Mesh(mesh.devices, ("vars", "x", "y", "z"))

    k_base_x = jnp.fft.fftfreq(num_cells_x, d=grid_spacing) * 2 * jnp.pi
    k_base_y = jnp.fft.fftfreq(num_cells_y, d=grid_spacing) * 2 * jnp.pi
    k_base_z = jnp.fft.fftfreq(num_cells_z, d=grid_spacing) * 2 * jnp.pi

    def _local_solve(density_block):
        # density_block: the local X slab, (num_cells_x / P, ny, nz).
        density_k = jnp.fft.fftn(density_block, axes=(1, 2))
        # Transpose the decomposition X -> Y (device p ends up with full X
        # and the p-th Y slab), then transform the now-local X axis.
        density_k = jax.lax.all_to_all(
            density_k, "x", split_axis=1, concat_axis=0, tiled=True
        )
        density_k = jnp.fft.fft(density_k, axis=0)

        # Green's function on this device's k-slab (same regularisation as
        # the replicated branch: the k = 0 mode maps to -1/(4 pi), harmless
        # because the mean density was already subtracted).
        slab_index = jax.lax.axis_index("x")
        k_local_y = jax.lax.dynamic_slice(
            k_base_y, (slab_index * local_cells_y,), (local_cells_y,)
        )
        k_squared = (
            k_base_x.reshape(-1, 1, 1) ** 2
            + k_local_y.reshape(1, -1, 1) ** 2
            + k_base_z.reshape(1, 1, -1) ** 2
        )
        k_squared = jnp.where(k_squared == 0, 1e-12, k_squared)
        greens_function = jnp.where(
            k_squared > 1e-12, -4 * jnp.pi * G / k_squared, -1 / (4 * jnp.pi)
        )
        potential_k = greens_function * density_k

        # Mirrored inverse: X-axis inverse FFT, transpose Y -> X back, then
        # the (y, z) inverse on the local X slab.
        potential_k = jnp.fft.ifft(potential_k, axis=0)
        potential_k = jax.lax.all_to_all(
            potential_k, "x", split_axis=0, concat_axis=1, tiled=True
        )
        potential = jnp.fft.ifftn(potential_k, axes=(1, 2))
        return jnp.real(potential)

    slab_spec = PartitionSpec("x", None, None)
    return _shard_map(
        _local_solve,
        mesh=named_mesh,
        in_specs=(slab_spec,),
        out_specs=slab_spec,
    )(gas_density)


def _periodic_xy_open_z(config: SimulationConfig) -> bool:
    """x and y periodic on both sides, z non-periodic on both sides (a
    stratified-box / disc-patch domain)."""
    bs = config.boundary_settings
    periodic = [
        b.left_boundary == PERIODIC_BOUNDARY and b.right_boundary == PERIODIC_BOUNDARY
        for b in (bs.x, bs.y)
    ]
    z_open = (
        bs.z.left_boundary != PERIODIC_BOUNDARY and bs.z.right_boundary != PERIODIC_BOUNDARY
    )
    return all(periodic) and z_open


def _poisson_periodic_xy_isolated_z(
    gas_density: FIELD_TYPE,
    grid_spacing: float,
    config: SimulationConfig,
    G: Union[float, Float[Array, ""]] = 1.0,
) -> FIELD_TYPE:
    """Potential of a density field periodic in x and y and isolated in z.

    FFT in x and y, then for each transverse wavenumber k a 1D free-space
    convolution in z (zero-padded to twice the length, Hockney & Eastwood)
    with the Green's function of ``phi'' - k^2 phi = 4 pi G rho``,
    ``-2 pi G exp(-k |dz|) / k``, integrated over the source cell (the k = 0
    mode is the slab potential ``2 pi G |dz|``, up to a constant). So a
    density uniform in x and y gets ``dphi/dz = 2 pi G (Sigma_below -
    Sigma_above)`` exactly, with no Jeans swindle and no periodic images in z.

    A ghost-padded field (the shape of a state channel inside the time loop)
    is handled as such: x/y ghost cells are dropped before the transform and
    refilled periodically afterwards; z ghost cells carry no mass (they lie
    outside the domain) but get the potential evaluated there, so the
    gravity source's centred differences at the outermost interior cells see
    the correct field.
    """
    g = config.num_ghost_cells
    num_cells = config.num_cells
    padded = gas_density.shape[0] == num_cells.x + 2 * g and g > 0
    density = gas_density
    if padded:
        density = density[g:-g, g:-g, :]
        density = density.at[:, :, :g].set(0.0).at[:, :, -g:].set(0.0)
    nx, ny, nz = density.shape

    kx = jnp.fft.fftfreq(nx, d=grid_spacing) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(ny, d=grid_spacing) * 2 * jnp.pi
    k = jnp.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)[:, :, None]

    # z offset (in cells) of each slot of the doubled grid, minimum image
    m = jnp.arange(2 * nz)
    m = jnp.where(m < nz, m, 2 * nz - m)[None, None, :]
    k_safe = jnp.where(k > 0, k, 1.0)
    half = 0.5 * k_safe * grid_spacing
    # cell-integrated exp(-k |z - z'|) / k over a source cell of width dx
    kernel_k = jnp.where(
        m == 0,
        2.0 * (1.0 - jnp.exp(-half)),
        2.0 * jnp.sinh(half) * jnp.exp(-k_safe * m * grid_spacing),
    ) / k_safe**2
    kernel_k = -2.0 * jnp.pi * G * kernel_k
    # k = 0: cell-integrated |z - z'| (|m| dx^2, dx^2 / 4 for the own cell)
    kernel_0 = 2.0 * jnp.pi * G * grid_spacing**2 * jnp.where(m == 0, 0.25, m.astype(density.dtype))
    kernel = jnp.where(k > 0, kernel_k, kernel_0)

    density_hat = jnp.fft.fft2(density, axes=(0, 1))
    density_hat = jnp.pad(density_hat, ((0, 0), (0, 0), (0, nz)))
    potential_hat = jnp.fft.ifft(
        jnp.fft.fft(density_hat, axis=2) * jnp.fft.fft(kernel, axis=2), axis=2
    )[:, :, :nz]
    potential = jnp.real(jnp.fft.ifft2(potential_hat, axes=(0, 1)))

    if padded:
        potential = jnp.pad(potential, ((g, g), (g, g), (0, 0)), mode="wrap")
    return potential


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["grid_spacing", "config"])
def _compute_gravitational_potential(
    gas_density: FIELD_TYPE,
    grid_spacing: float,
    config: SimulationConfig,
    G: Union[float, Float[Array, ""]] = 1.0,
) -> FIELD_TYPE:
    """
    Compute the gravitational potential using FFT to solve Poisson's equation for
    periodic and open boundaries (via the Hockney & Eastwood method).

    Args:
        gas_density: The gas density field.
        grid_spacing: The grid spacing.
        config: The simulation configuration.
        G: The gravitational constant.

    Returns:
        The gravitational potential.

    """

    # TODO: remove ghost cells in this computation, which currently treats the
    # padded field directly.

    dimensionality = config.dimensionality

    # The open-boundary branch is taken when any boundary is non-periodic. The
    # one mixed case supported is 3D periodic-x/y with open z (stratified
    # boxes, _poisson_periodic_xy_isolated_z); other mixes are treated as
    # fully open.
    non_periodic_boundaries = False

    if dimensionality == 1:
        if not (
            config.boundary_settings.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.right_boundary == PERIODIC_BOUNDARY
        ):
            non_periodic_boundaries = True
    elif dimensionality == 2:
        if not (
            config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY
        ):
            non_periodic_boundaries = True
    elif dimensionality == 3:
        if not (
            config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.z.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.z.right_boundary == PERIODIC_BOUNDARY
        ):
            non_periodic_boundaries = True

    if config.gravity_config.poisson_manual_open_boundaries:
        non_periodic_boundaries = True
    elif dimensionality == 3 and _periodic_xy_open_z(config):
        return _poisson_periodic_xy_isolated_z(gas_density, grid_spacing, config, G)

    # The Jeans swindle: only a meaningful periodic solution exists once the
    # (unphysical) mean density is removed, so subtract it for periodic domains.
    if not non_periodic_boundaries:
        gas_density = gas_density - jnp.mean(gas_density)

    if not non_periodic_boundaries:
        # -------------------------------------------------------------
        # ============= ↓ Periodic boundaries version ↓ ==============
        # -------------------------------------------------------------

        # Under a multi-device X-decomposed mesh, route the 3D solve
        # through the slab-transposed distributed FFT -- the plain fftn
        # below would be replicated per device by the SPMD partitioner
        # (global-grid memory and work on every GPU).  Requires the slab
        # counts to divide the decomposition; otherwise fall through.
        if dimensionality == 3:
            active = _active_x_sharded_mesh()
            if active is not None:
                mesh, x_name = active
                num_slabs = mesh.shape[x_name]
                divisible = (
                    gas_density.shape[0] % num_slabs == 0
                    and gas_density.shape[1] % num_slabs == 0
                )
                if divisible:
                    return _periodic_poisson_3d_sharded(
                        gas_density,
                        grid_spacing,
                        G,
                        mesh,
                        x_name,
                    )

        # Transform the density to Fourier space.
        density_k = fftn(gas_density)

        # Build the squared wavenumber magnitude for the active dimensionality.
        if dimensionality == 1:
            num_cells_x = gas_density.shape[0]
            k_base_x = jnp.fft.fftfreq(num_cells_x, d=grid_spacing) * 2 * jnp.pi
            k = k_base_x
            k_squared = k**2
        elif dimensionality == 2:
            num_cells_x, num_cells_y = gas_density.shape
            k_base_x = jnp.fft.fftfreq(num_cells_x, d=grid_spacing) * 2 * jnp.pi
            k_base_y = jnp.fft.fftfreq(num_cells_y, d=grid_spacing) * 2 * jnp.pi
            kx, ky = jnp.meshgrid(k_base_x, k_base_y, indexing="ij")
            k_squared = kx**2 + ky**2
        elif dimensionality == 3:
            num_cells_x, num_cells_y, num_cells_z = gas_density.shape
            k_base_x = jnp.fft.fftfreq(num_cells_x, d=grid_spacing) * 2 * jnp.pi
            k_base_y = jnp.fft.fftfreq(num_cells_y, d=grid_spacing) * 2 * jnp.pi
            k_base_z = jnp.fft.fftfreq(num_cells_z, d=grid_spacing) * 2 * jnp.pi
            kx, ky, kz = jnp.meshgrid(k_base_x, k_base_y, k_base_z, indexing="ij")
            k_squared = kx**2 + ky**2 + kz**2

        # Regularise the k = 0 mode to avoid dividing by zero; its Green's-
        # function value is the finite constant -1 / (4 pi). This is harmless
        # because the Jeans swindle above already removed the mean density, so
        # density_k[0] ~ 0.
        k_squared = jnp.where(k_squared == 0, 1e-12, k_squared)
        greens_function = jnp.where(
            k_squared > 1e-12, -4 * jnp.pi * G / k_squared, -1 / (4 * jnp.pi)
        )

        # Apply the Green's function in Fourier space and transform back.
        potential_k = greens_function * density_k
        gravitational_potential = jnp.real(ifftn(potential_k))

        return gravitational_potential

    else:
        # -------------------------------------------------------------
        # ====== ↓ Open boundaries (Hockney & Eastwood) version ↓ ====
        # -------------------------------------------------------------
        # TODO: check that this works for differing cell counts per dimension.
        #
        # (a) Extend the domain to twice the size in each dimension and embed
        #     the original density in the (0, ..., 0) corner; the zero padding
        #     is what makes the periodic FFT convolution behave as an isolated
        #     (open-boundary) one.
        original_shape = gas_density.shape
        extended_shape = tuple(2 * s for s in original_shape)

        extended_density = jnp.zeros(extended_shape, dtype=gas_density.dtype)
        slices = tuple(slice(0, s) for s in original_shape)
        extended_density = extended_density.at[slices].set(gas_density)

        # (b) Construct the Green's function on the extended grid.
        #
        # The Hockney-Eastwood prescription computes, for each dimension,
        #     pos = [0, 1, 2, ..., n-1, 2n - n, ..., 1] * grid_spacing,
        # i.e. pos = arange(2*n); pos = where(pos < n, pos, 2*n - pos), which
        # yields the minimum-image distances from a source placed at the origin.
        grids = []
        for s in original_shape:
            n = s
            extended_n = 2 * n
            pos = jnp.arange(extended_n)
            pos = jnp.where(pos < n, pos, 2 * n - pos)
            pos = pos * grid_spacing
            grids.append(pos)

        # Build the radial distance array r on the extended grid.
        if dimensionality == 1:
            r = grids[0]  # already nonnegative
        elif dimensionality == 2:
            x, y = jnp.meshgrid(grids[0], grids[1], indexing="ij")
            r = jnp.sqrt(x**2 + y**2)
        elif dimensionality == 3:
            x, y, z = jnp.meshgrid(grids[0], grids[1], grids[2], indexing="ij")
            r = jnp.sqrt(x**2 + y**2 + z**2)

        # Replace any zero distance with grid_spacing to avoid the singularity
        # at the origin.
        r_safe = jnp.where(r == 0, grid_spacing, r)

        # (c) Isolated (open-boundary) Green's function for each dimensionality.
        if dimensionality == 1:
            # 1D: solving phi'' = 4 pi G delta(x) gives phi = -2 pi G |x|.
            kernel = -2 * jnp.pi * G * r
        elif dimensionality == 2:
            # 2D: phi = -2 G log(r) (up to an additive constant).
            kernel = -2 * G * jnp.log(r_safe)
        elif dimensionality == 3:
            # 3D: the isolated potential is phi = -G / r.
            kernel = -G / r_safe

        # (d) FFT-convolve the extended density with the Green's function.
        density_k_ext = fftn(extended_density)
        kernel_k_ext = fftn(kernel)
        potential_ext = jnp.real(ifftn(density_k_ext * kernel_k_ext))

        # (e) Extract the portion of the potential covering the original grid;
        #     the grid_spacing**dim factor accounts for the discrete convolution
        #     measure.
        gravitational_potential = potential_ext[slices]
        return gravitational_potential * grid_spacing**dimensionality
