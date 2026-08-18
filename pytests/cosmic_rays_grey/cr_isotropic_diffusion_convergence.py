"""
Isotropic CR diffusion convergence pytest (Phase A ladder item 4).

Compares the two-moment isotropic-diffusion limit against the analytic
Gaussian Green's-function solution of the diffusion equation, at increasing
resolution, and checks the expected convergence order -- validates that the
reduced free-streaming speed (cr_grey_transport.grey_cr_fast_speed) is high
enough for the two-moment system to recover ordinary diffusion in this
regime (plan Sec. 2: "recovers anisotropic diffusion + streaming in the
appropriate limit").

TODO (fills in once the transport stubs are implemented):
  - build a SimulationConfig with cosmic_ray_grey_config.grey_cosmic_rays=True,
    anisotropic_transport=False (isotropic), at a few resolutions N
  - initial condition: a narrow Gaussian e_cr bump, uniform gas, F_cr=0
  - run time_integration for a fixed diffusion time
  - compare each resolution's e_cr profile to the analytic Gaussian
    Green's-function solution and fit the convergence order

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

def test_cr_isotropic_diffusion_convergence(min_order: float = 1.5):
    """Isotropic diffusion vs. the analytic Gaussian Green's function.

    Args:
        min_order: The minimum acceptable convergence order.
    """
    raise NotImplementedError(
        "Phase A ladder item 4: isotropic diffusion convergence vs. the "
        "analytic Gaussian Green's function. See this module's docstring and "
        "astronomix/_modules/_cosmic_rays_grey/DESIGN.md."
    )


if __name__ == "__main__":
    test_cr_isotropic_diffusion_convergence()
