"""
Adiabatic CR compression pytest (Phase A ladder item 2).

Applies a uniform squeeze (a converging velocity field with no shocks) to a
CR-loaded gas and checks that e_cr scales as rho^gamma_cr, matching the grey
closure P_cr = (gamma_cr - 1) * e_cr under adiabatic compression -- the basic
check that the -P_cr * div(v) work term (cr_grey_sources.cr_adiabatic_work_source)
is implemented correctly, in isolation from transport/diffusion.

TODO (fills in once cr_grey_sources.cr_adiabatic_work_source is implemented):
  - build a SimulationConfig with cosmic_ray_grey_config.grey_cosmic_rays=True
  - initial condition: uniform e_cr, a prescribed uniform converging velocity field
  - run time_integration for a known compression factor
  - assert e_cr_final / e_cr_initial ~= (rho_final / rho_initial)**gamma_cr

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

def test_cr_adiabatic_compression(tol: float = 1e-2):
    """Uniform squeeze: check e_cr ~ rho^gamma_cr.

    Args:
        tol: The maximum allowed relative error on the e_cr/rho scaling.
    """
    raise NotImplementedError(
        "Phase A ladder item 2: adiabatic compression e_cr ~ rho^gamma_cr. "
        "See this module's docstring and "
        "astronomix/_modules/_cosmic_rays_grey/DESIGN.md."
    )


if __name__ == "__main__":
    test_cr_adiabatic_compression()
