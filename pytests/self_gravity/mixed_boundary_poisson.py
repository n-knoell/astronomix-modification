"""
Poisson solve for a 3D box periodic in x/y and open (isolated) in z --
``_poisson_periodic_xy_isolated_z``, the self-gravity solve of the stratified
ISM boxes (M7). Before it existed, that boundary combination fell into the
fully isolated Hockney & Eastwood branch, i.e. the disc patch was treated as
an isolated 2 x 2 kpc tile instead of a periodic sheet.

Checks, for a Gaussian layer (sigma = 60 pc in a 2 x 2 x 5 kpc box, as M7):

1. **Uniform in x/y**: the solve must be exact, to round-off, for the
   piecewise-constant (cell-averaged) density: the centred difference of the
   exact potential between two cell centres is Gauss's law at the face,
   ``-2 pi G (Sigma_below - Sigma_above)``, plus ``-pi G dx (rho_{i+1} -
   rho_i) / 2`` from the field varying linearly inside each half cell. The difference to the
   continuous ``-2 pi G Sigma erf(z / (sqrt(2) sigma))`` is the sampling
   error of a 60 pc Gaussian on the grid (printed, not asserted).
2. **A transverse mode** ``rho = rho_layer(z) (1 + A cos(2 pi x / L))``: the
   cos(k x) part of phi must match ``-(2 pi G / k) int rho_k(z') exp(-k |z -
   z'|) dz'``, evaluated by independent fine quadrature.
3. **Ghost-padded call** (as inside the time loop): the potential on the
   interior must equal the unpadded result, and the x/y ghost cells must be
   periodic copies.

The mode check passes at relative error < 1e-2 at 32 x 32 x 80 (62.5 pc)
and converges with resolution (sampling of the layer).
"""

from autocvd import autocvd
autocvd(num_gpus=1)

import numpy as np
from scipy.special import erf

import jax
import jax.numpy as jnp

from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import OPEN_BOUNDARY, PERIODIC_BOUNDARY
from astronomix import BoundarySettings, BoundarySettings1D, GravityConfig
from astronomix import SimulationConfig, finalize_config
from astronomix.option_classes.simulation_config import StaticFloatVector, StaticIntVector
from astronomix._modules._gravity._poisson_solver import _compute_gravitational_potential

jax.config.update("jax_enable_x64", True)

G = 4.30091e-3  # pc (km/s)^2 / Msun
L_XY, L_Z = 2000.0, 5000.0
SIGMA = 60.0  # pc
SURFACE_DENSITY = 10.0  # Msun / pc^2
AMPLITUDE = 0.5


def _config(n_xy, n_z):
    config = SimulationConfig(
        geometry=CARTESIAN, solver_mode=FINITE_VOLUME, riemann_solver=HLLC, limiter=MINMOD,
        dimensionality=3,
        box_size=StaticFloatVector(L_XY, L_XY, L_Z),
        num_cells=StaticIntVector(n_xy, n_xy, n_z),
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        gravity_config=GravityConfig(self_gravity=True),
    )
    return finalize_config(config, (5, n_xy, n_xy, n_z))


def _layer(z_rel):
    return SURFACE_DENSITY / (np.sqrt(2 * np.pi) * SIGMA) * np.exp(-0.5 * (z_rel / SIGMA) ** 2)


def _errors(n_xy, n_z):
    config = _config(n_xy, n_z)
    dx = L_XY / n_xy
    x = (np.arange(n_xy) + 0.5) * dx
    z_rel = (np.arange(n_z) + 0.5) * dx - 0.5 * L_Z

    # 1. uniform layer: g_z at interior faces vs. the slab result
    rho = np.broadcast_to(_layer(z_rel), (n_xy, n_xy, n_z)).copy()
    phi = np.asarray(_compute_gravitational_potential(jnp.asarray(rho), dx, config, G))
    g_num = -(phi[0, 0, 1:] - phi[0, 0, :-1]) / dx
    z_face = 0.5 * (z_rel[1:] + z_rel[:-1])
    column = rho[0, 0] * dx
    below = np.cumsum(column)[:-1]
    g_gauss = -2 * np.pi * G * (below - (column.sum() - below)) - 0.5 * np.pi * G * dx * np.diff(rho[0, 0])
    err_slab = np.max(np.abs(g_num - g_gauss)) / np.max(np.abs(g_gauss))
    g_continuous = -2 * np.pi * G * SURFACE_DENSITY * erf(z_face / (np.sqrt(2) * SIGMA))
    err_sampling = np.max(np.abs(g_num - g_continuous)) / np.max(np.abs(g_continuous))
    xy_spread = np.ptp(phi, axis=(0, 1)).max() / np.ptp(phi)

    # 2. transverse mode k = 2 pi / L: compare the cos(kx) amplitude of phi
    k = 2 * np.pi / L_XY
    rho_mode = rho * (1.0 + AMPLITUDE * np.cos(k * x))[:, None, None]
    phi_mode = np.asarray(_compute_gravitational_potential(jnp.asarray(rho_mode), dx, config, G)) - phi
    amp_num = 2.0 / n_xy * np.sum(phi_mode[:, 0, :] * np.cos(k * x)[:, None], axis=0)
    z_fine = np.linspace(-0.5 * L_Z, 0.5 * L_Z, 400001)
    rho_k_fine = AMPLITUDE * _layer(z_fine)
    window = np.abs(z_fine) < 12 * SIGMA
    amp_exact = np.array([
        -2 * np.pi * G / k * np.trapezoid(rho_k_fine[window] * np.exp(-k * np.abs(zi - z_fine[window])), z_fine[window])
        for zi in z_rel
    ])
    err_mode = np.max(np.abs(amp_num - amp_exact)) / np.max(np.abs(amp_exact))

    # 3. ghost-padded call: interior identical, x/y ghosts periodic
    g = config.num_ghost_cells
    padded = np.pad(rho_mode, ((g, g), (g, g), (g, g)), mode="edge")
    phi_pad = np.asarray(_compute_gravitational_potential(jnp.asarray(padded), dx, config, G))
    interior = phi_pad[g:-g, g:-g, g:-g]
    err_pad = np.max(np.abs(interior - phi_mode - phi)) / np.ptp(phi)
    err_wrap = np.max(np.abs(phi_pad[:g] - phi_pad[-2 * g:-g]))  / np.ptp(phi)
    return err_slab, err_mode, err_pad, err_wrap, xy_spread, err_sampling


def test_mixed_boundary_poisson():
    results = {}
    for n_xy in (16, 32, 64):
        n_z = int(round(n_xy * L_Z / L_XY))
        results[n_xy] = _errors(n_xy, n_z)
        e = results[n_xy]
        print(f"{n_xy:3d} x {n_xy:3d} x {n_z:3d} (dx = {L_XY / n_xy:6.1f} pc): slab g_z err {e[0]:.1e} (vs. continuous layer {e[5]:.1e}), "
              f"mode err {e[1]:.2e}, padded-vs-bare {e[2]:.1e}, ghost wrap {e[3]:.1e}, "
              f"x/y variation of uniform-layer phi {e[4]:.1e}")
    e32 = results[32]
    assert results[32][1] < 1e-2, results[32]
    for e in results.values():
        assert e[0] < 1e-10, "slab g_z must match Gauss's law for the cell masses"
        assert e[4] < 1e-12, "a uniform layer must give a potential uniform in x/y"
        assert e[2] < 1e-12 and e[3] < 1e-12, e
    # convergence of the mode amplitude: halving dx reduces the error by > 3x
    assert results[64][1] < results[32][1] / 3, (results[32][1], results[64][1])


if __name__ == "__main__":
    test_mixed_boundary_poisson()
    print("mixed_boundary_poisson: passed")
