"""
Configuration and parameter containers for episodic supernova driving.

``SNDrivingConfig`` holds the static on/off switches, while ``SNDrivingParams``
holds the tunable physical parameters (rate, energy, CR fraction, injection
footprint). Mirrors the split used by ``turbulent_forcing_options.py`` and
``cosmic_ray_grey_options.py``.
"""

# typing
from typing import NamedTuple


class SNDrivingConfig(NamedTuple):

    #: main switch for episodic supernova driving.
    sn_driving: bool = False

    #: inject the SNR's terminal momentum directly (Kim & Ostriker 2015),
    #: instead of most of the thermal part of ``sn_energy``, rather than
    #: relying on (under-resolved) thermal pressure to drive it. When on,
    #: the thermal gas-pressure deposit is scaled down to
    #: ``SNDrivingParams.sn_momentum_thermal_floor_fraction`` of its normal
    #: value (see that field's docstring for why a small nonzero floor is
    #: needed, not exactly zero) -- a radial velocity kick (see
    #: ``sn_driving.py``'s docstring) is added alongside it, and, when
    #: CR-grey is active, the usual ``sn_cr_fraction * sn_energy`` direct
    #: e_cr deposit is unaffected. Off by default (matches every existing
    #: SN-driving test's pure-thermal behavior unchanged). See PROGRESS.md's
    #: 2026-09-14 M4 entry for why this was needed: at SILCC-ISM project
    #: milestone M4's box/resolution, the K&I cooling time at a fresh
    #: injection's own post-shock state (~60 yr) is far shorter than a
    #: single hydro dynamical time, so pure-thermal injection is radiated
    #: away before it can do any work ("overcooling", Katz 1992) -- injected
    #: momentum is not subject to radiative cooling, so it survives.
    momentum_injection: bool = False

    #: temporarily suppress K&I cooling within a fresh injection's footprint
    #: for ``SNDrivingParams.sn_cooling_delay_time`` (a "cooling shutoff",
    #: e.g. Thacker & Couchman 2000 / Stinson et al. 2006's blastwave
    #: feedback) -- the other mitigation considered for the same overcooling
    #: finding ``momentum_injection`` addresses (see that field's docstring
    #: and PROGRESS.md's 2026-09-14 M4 entry), tried instead of it because it
    #: keeps the plain thermal deposit (no large velocity kick, so it avoids
    #: that mitigation's own catastrophic-cancellation/third-NaN issues
    #: entirely) rather than replacing the physics. Off by default (matches
    #: every existing SN-driving test's behavior unchanged). Needs a
    #: persistent per-cell "time remaining shielded" field, carried through
    #: the loop the same way as the OU forcing field -- see
    #: ``astronomix.time_stepping.time_integration.LoopState.cooling_shield``.
    delayed_cooling: bool = False

    #: draw each trigger's site from a density-weighted distribution over
    #: eligible cells (probability per cell proportional to
    #: ``rho ** SNDrivingParams.sn_density_weighting_power``) instead of
    #: uniformly at random over the eligible volume -- Simpson et al.
    #: (2016)'s "density peak" placement mode (SILCC-ISM project milestone
    #: M6, ladder item 12's optional stretch cross-check), approximating
    #: their local-star-formation-rate-weighted trigger probability
    #: (``sfr ~ rho^1.5`` on this codebase's fixed-volume grid -- see
    #: ``sn_driving.py``'s docstring for the derivation). Off by default
    #: (matches every existing SN-driving test's uniform-random placement
    #: unchanged); the two modes are mutually exclusive alternatives, not
    #: combinable. ``sn_z_min``/``sn_z_max`` still restrict eligibility the
    #: same way for both modes.
    density_weighted_placement: bool = False


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

    #: restrict the randomly-drawn site's z-coordinate to
    #: ``[sn_z_min, sn_z_max]`` (clipped to the domain); x/y stay uniform
    #: over the whole box. Defaults to +/-inf (no restriction, the whole
    #: box) -- every existing SN-driving test relies on this default, and
    #: the PRNG-to-site mapping is bit-identical to the unrestricted formula
    #: at these defaults, so leaving them unset changes nothing. Meant for a
    #: vertically-extended (stratified) box, where real supernovae should
    #: track the star-forming layer near the midplane rather than a whole
    #: tall column's tenuous envelope (SILCC-ISM project milestone M4).
    sn_z_min: float = float("-inf")
    sn_z_max: float = float("inf")

    #: Kim & Ostriker (2015) terminal-momentum coefficient
    #: (``2.8e5 M_sun km/s``, their eq. for the momentum a single SN
    #: ultimately deposits in the ISM, fit at their fiducial sn_energy),
    #: scaled by ``sn_energy / 1e51 erg`` and converted to code momentum
    #: units -- do this conversion once at setup time (matches
    #: ``CoolingCurveParams``'s ``code_temperature_per_kelvin``-style
    #: precomputed-conversion-factor pattern; keeps ``_inject_supernovae``
    #: free of unit-conversion machinery). K&O15's own fit has no explicit
    #: sn_energy dependence (fixed at their fiducial value) -- linear
    #: scaling with sn_energy is the simplest defensible extrapolation, not
    #: independently verified against simulations at other energies. Only
    #: read when ``SNDrivingConfig.momentum_injection`` is on.
    sn_momentum_coefficient: float = 0.0

    #: code density corresponding to ``n_H = 1 cm^-3`` (``mu_H * m_p`` in
    #: code density units) -- lets ``_inject_supernovae`` express the local
    #: ambient density as a dimensionless ``n_0`` for the
    #: ``sn_momentum_coefficient * n_0**(-0.17)`` density scaling, without
    #: needing astropy/physical constants inside jitted code. Only read
    #: when ``SNDrivingConfig.momentum_injection`` is on.
    sn_momentum_density_reference: float = 1.0

    #: fraction of the thermal energy (``(1 - sn_cr_fraction) * sn_energy``)
    #: still deposited as a gas-pressure bump alongside the momentum kick,
    #: when ``SNDrivingConfig.momentum_injection`` is on (0.0 = pure
    #: momentum, no thermal deposit at all). Purely a numerical
    #: stabilization, not a physical model of the shell's residual heat:
    #: zero thermal energy alongside a large velocity kick in cold gas
    #: leaves the conserved total energy completely kinetic-dominated, and
    #: recovering the (tiny, near-zero) internal energy back out of
    #: ``E_total - 0.5*rho*v^2`` is then a catastrophic-cancellation
    #: computation that reliably NaNs -- confirmed directly (a pure
    #: momentum-only M4 attempt NaN'd faster than the pure-thermal one it
    #: replaced; see PROGRESS.md's 2026-09-14 M4 entry). A small nonzero
    #: floor keeps the local sound speed non-negligible relative to the
    #: kick velocity without reintroducing the original overcooling problem
    #: (this floor's own deposited energy is a small fraction of the full
    #: sn_energy, so even fast K&I cooling only removes a proportionally
    #: small amount). Only read when ``SNDrivingConfig.momentum_injection``
    #: is on.
    sn_momentum_thermal_floor_fraction: float = 0.0

    #: duration (code time units) a fresh injection's footprint is shielded
    #: from K&I cooling, when ``SNDrivingConfig.delayed_cooling`` is on.
    #: Applied *uniformly* to every cell inside the footprint (``weight``
    #: above a small numerical cutoff) -- each such cell's shield is extended
    #: to ``max(current_shield, this)``, deliberately NOT tapered by the same
    #: ``weight`` that scales the thermal/CR energy deposit. An earlier,
    #: tapered-duration version unshielded the coolest, tapered-edge cells
    #: first and left the single hottest cell shielded longest, so by the
    #: time that cell's shield finally lifted its neighbors had already
    #: cooled back to their local K&I equilibrium -- producing a sharper,
    #: less-resolved discontinuity than the original injection and a real
    #: NaN in the M4 run (root-caused and fixed, see PROGRESS.md's
    #: 2026-09-15 entry). Should be set to a modest multiple of the immediate
    #: post-shock K&I cooling time at the injection site (measured ~60 yr for
    #: M4's midplane state, see PROGRESS.md's 2026-09-14 M4 entry) -- long
    #: enough for the deposit to do PdV work before cooling resumes, but not
    #: so long it suppresses cooling over a large fraction of the box between
    #: triggers. Only read when ``SNDrivingConfig.delayed_cooling`` is on.
    sn_cooling_delay_time: float = 0.0

    #: exponent applied to local density when drawing a trigger's site under
    #: ``SNDrivingConfig.density_weighted_placement`` (per-cell probability
    #: ``propto rho ** this``). Default ``1.5`` is Simpson et al. (2016)'s own
    #: local-SFR proxy ``sfr_i ~ m_i / t_ff,i ~ m_i * sqrt(rho_i)``, which
    #: reduces to ``rho_i^1.5`` on this codebase's fixed-volume grid (cell
    #: mass ``m_i = rho_i * cell_volume``, free-fall time ``t_ff ~
    #: rho^-0.5``) -- not independently re-derived, taken directly from their
    #: Sec. 2. Only read when ``SNDrivingConfig.density_weighted_placement``
    #: is on.
    sn_density_weighting_power: float = 1.5
