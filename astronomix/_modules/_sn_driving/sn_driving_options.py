"""
Configuration and parameter containers for episodic supernova driving.

``SNDrivingConfig`` holds the static on/off switch, while ``SNDrivingParams``
holds the tunable physical parameters (rate, energy, CR fraction, injection
footprint). Mirrors the split used by ``turbulent_forcing_options.py`` and
``cosmic_ray_grey_options.py``.
"""

# typing
from typing import NamedTuple


class SNDrivingConfig(NamedTuple):

    #: main switch for episodic supernova driving.
    sn_driving: bool = False


class SNDrivingParams(NamedTuple):

    #: expected number of supernovae per unit time, for the whole
    #: simulation domain (not per unit volume) -- a single Poisson-process
    #: rate, thinned to a per-step Bernoulli trial (see ``sn_driving.py``'s
    #: docstring for why this is the standard, dt-small approximation).
    sn_rate: float = 0.0

    #: total energy released per supernova, split into a thermal part
    #: ``(1 - sn_cr_fraction) * sn_energy`` (deposited as a gas pressure
    #: bump) and a cosmic-ray part ``sn_cr_fraction * sn_energy`` (deposited
    #: directly into e_cr), both with the same spatial footprint. Girichidis
    #: et al. (2016)'s fiducial values are sn_energy = 1e51 erg,
    #: sn_cr_fraction = 0.1 -- convert to code units for the actual run.
    sn_energy: float = 1.0

    #: fraction of sn_energy dumped directly into e_cr (Girichidis et al.
    #: 2016 convention). Forced to 0 (all-thermal) whenever
    #: registered_variables.cosmic_ray_e_active is False, so total energy
    #: conservation never depends on whether the CR-grey model is active.
    sn_cr_fraction: float = 0.1

    #: radius of the smoothly-tapered spherical injection region, in the
    #: same physical units as config.box_size (same convention as
    #: cr_grey_injection/cr_sedov_taylor's R_EXPLOSION).
    sn_injection_radius: float = 0.05

    #: tanh taper width, in units of grid cells (same convention as
    #: cr_sedov_taylor's SMOOTH_CELLS).
    sn_smooth_cells: float = 2.0
