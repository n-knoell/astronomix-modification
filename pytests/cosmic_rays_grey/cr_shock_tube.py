"""
Two-fluid CR-modified shock tube pytest (Phase A ladder item 6).

The single most diagnostic coupling test: a Sod-like shock tube where the
left/right states carry CR pressure, compared against the semi-analytic
two-fluid solution of Pfrommer et al. (2006). Exercises the full Phase A
stack together -- transport flux (cr_grey_transport.grey_cr_flux_terms), the
CR-grey wave speed in the Riemann solver/CFL, and the momentum feedback
(cr_grey_sources.cr_pressure_gradient_source) -- rather than any one piece in
isolation.

TODO (fills in once the full transport + feedback stack is implemented):
  - build a SimulationConfig with cosmic_ray_grey_config.grey_cosmic_rays=True
  - initial condition: Sod-like left/right (rho, v, p_gas, e_cr) states per
    Pfrommer et al. (2006)
  - run time_integration to the reference time
  - compare density/velocity/pressure/e_cr profiles against the semi-analytic
    two-fluid Riemann solution

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md, and
pytests/hydrodynamics/shock_tube1D.py for the analogous (non-CR) test's
structure and conventions.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

def test_cr_shock_tube(tol: float = 1e-2):
    """Two-fluid CR-modified shock tube vs. Pfrommer et al. (2006).

    Args:
        tol: The maximum allowed L1 error against the semi-analytic solution.
    """
    raise NotImplementedError(
        "Phase A ladder item 6: two-fluid CR-modified shock tube vs. "
        "Pfrommer et al. (2006). See this module's docstring and "
        "astronomix/_modules/_cosmic_rays_grey/DESIGN.md."
    )


if __name__ == "__main__":
    test_cr_shock_tube()
