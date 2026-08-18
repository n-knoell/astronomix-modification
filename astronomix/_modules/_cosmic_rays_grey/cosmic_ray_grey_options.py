"""
Configuration and parameter containers for the grey two-moment cosmic-ray
module.

``CosmicRayGreyConfig`` holds the static (compile-time) switches that turn the
grey two-moment CR physics on and off, while ``CosmicRayGreyParams`` holds the
runtime numerical values controlling the transport and feedback coupling.
Mirrors the split used by the older ``cosmic_ray_options.py``.
"""

# typing
from typing import NamedTuple


class CosmicRayGreyConfig(NamedTuple):

    #: main switch for the grey two-moment cosmic-ray model. Currently
    #: finite-volume only -- see DESIGN.md for why the finite-difference WENO
    #: reconstruction cannot yet carry e_cr/F_cr.
    grey_cosmic_rays: bool = False

    #: evolve F_cr projected along the local B direction (Sharma & Hammett
    #: 2007 monotonicity-safe operator) rather than isotropically.
    anisotropic_transport: bool = False

    #: turn on CR streaming (with tanh-regularized sign) and the associated
    #: streaming heating of the gas.
    streaming: bool = False


class CosmicRayGreyParams(NamedTuple):

    #: adiabatic index of the (grey) cosmic-ray population.
    gamma_cr: float = 4.0 / 3.0

    #: reduced free-streaming speed entering the two-moment closure (Jiang &
    #: Oh 2018). Open question (plan Sec. 6): pick the largest value that
    #: leaves wind/emission properties unchanged, via a convergence study.
    reduced_streaming_speed: float = 1.0

    #: positivity floor on e_cr.
    minimum_e_cr: float = 1e-10

    #: efficiency with which streaming losses heat the gas thermal energy.
    streaming_heating_efficiency: float = 1.0

    #: smoothing scale for the tanh-regularized streaming-sign function, in
    #: units of the streaming speed. Smaller is closer to sign(), but less
    #: adjoint-friendly.
    streaming_sign_regularization: float = 1e-2
