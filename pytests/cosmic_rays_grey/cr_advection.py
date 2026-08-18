"""
Pure CR advection pytest (Phase A ladder item 1).

Advects an `e_cr` blob in a uniform gas flow (no gradients in rho/p/v, so the
gas evolution is trivial) and checks that the blob's shape and amplitude are
preserved -- the most basic correctness check on the two-moment transport
before any coupling or diffusion physics is exercised.

TODO (fills in once astronomix._modules._cosmic_rays_grey.cr_grey_transport's
NotImplementedError stubs are implemented):
  - build a periodic 1D/3D SimulationConfig with cosmic_ray_grey_config.grey_cosmic_rays=True
  - initial condition: uniform rho/p/v, a smooth (e.g. Gaussian) e_cr bump, F_cr=0
  - run time_integration for one or more advection periods (box_size / |v|)
  - compare the final e_cr profile to the shifted initial profile (shape + amplitude)

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md for the state-variable and
equation definitions this test exercises.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

def test_cr_advection(tol: float = 1e-2):
    """Advect an e_cr blob and check shape/amplitude preservation.

    Args:
        tol: The maximum allowed relative error on the blob shape/amplitude.
    """
    raise NotImplementedError(
        "Phase A ladder item 1: pure CR advection. See this module's docstring "
        "and astronomix/_modules/_cosmic_rays_grey/DESIGN.md."
    )


if __name__ == "__main__":
    test_cr_advection()
