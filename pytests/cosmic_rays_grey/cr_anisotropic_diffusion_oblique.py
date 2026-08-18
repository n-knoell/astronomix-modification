"""
Oblique anisotropic CR diffusion pytest (Phase A ladder item 3).

Sets up a uniform B field at an oblique angle to the grid axes and checks
that CR energy diffuses/streams *only* along B, with no cross-field leak --
the Sharma & Hammett (2007) monotonicity-safe operator's defining test,
implemented in cr_grey_transport.anisotropic_flux_projection.

TODO (fills in once anisotropic_flux_projection is implemented):
  - build a SimulationConfig with cosmic_ray_grey_config.grey_cosmic_rays=True
    and anisotropic_transport=True, mhd=True, B at a fixed oblique angle
  - initial condition: a localized e_cr bump, uniform gas, uniform oblique B
  - run time_integration for a diffusion time
  - assert the e_cr distribution stays confined to the B-aligned strip (e.g.
    via the second moment perpendicular vs. parallel to B)

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

def test_cr_anisotropic_diffusion_oblique(tol: float = 1e-2):
    """Oblique-to-grid anisotropic diffusion: no cross-field leak.

    Args:
        tol: The maximum allowed cross-field leak fraction.
    """
    raise NotImplementedError(
        "Phase A ladder item 3: oblique anisotropic diffusion, Sharma & "
        "Hammett (2007). See this module's docstring and "
        "astronomix/_modules/_cosmic_rays_grey/DESIGN.md."
    )


if __name__ == "__main__":
    test_cr_anisotropic_diffusion_oblique()
