"""
1D CR streaming pytest (Phase A ladder item 5).

Sets up a 1D CR pressure gradient with streaming enabled and checks that the
CR population streams down the gradient at the configured (reduced)
streaming speed, and that the streaming-heating rate deposited into the gas
thermal energy matches the analytic rate -- validates
cr_grey_transport.regularized_streaming_sign and
cr_grey_sources.cr_streaming_heating_source together.

TODO (fills in once the streaming stubs are implemented):
  - build a SimulationConfig with cosmic_ray_grey_config.grey_cosmic_rays=True
    and streaming=True
  - initial condition: a linear e_cr gradient over an otherwise uniform gas
  - run time_integration for a short time
  - assert the measured F_cr matches
    -sign(grad P_cr) * reduced_streaming_speed * e_cr (regularized), and that
    the gas thermal-energy growth matches the analytic streaming-heating rate

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

def test_cr_streaming_1d(tol: float = 1e-2):
    """Recover the streaming speed and streaming-heating rate.

    Args:
        tol: The maximum allowed relative error on the streaming speed and
            heating rate.
    """
    raise NotImplementedError(
        "Phase A ladder item 5: 1D CR streaming speed and heating rate. See "
        "this module's docstring and "
        "astronomix/_modules/_cosmic_rays_grey/DESIGN.md."
    )


if __name__ == "__main__":
    test_cr_streaming_1d()
