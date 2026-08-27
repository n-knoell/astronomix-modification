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


#: diffusive_shock_acceleration efficiency models (CosmicRayGreyConfig.
#: dsa_efficiency_model), see cr_grey_injection.py.

#: fixed, Mach-independent efficiency (CosmicRayGreyParams.dsa_efficiency).
#: Ladder item 7's model; still the default so existing configs/tests (e.g.
#: cr_sedov_taylor.py, whose energy-conservation tolerances are calibrated
#: against a fixed 0.1 efficiency) are unaffected.
DSA_EFFICIENCY_CONSTANT = 0

#: Kang & Ryu (2013) Mach-number-dependent efficiency (ladder item 8), scaled
#: by CosmicRayGreyParams.dsa_efficiency_mach_scale -- see
#: cr_grey_injection.dsa_efficiency_kang_ryu_2013 for the fit and sources.
#: Caprioli & Spitkovsky (2014) is the same functional shape at half the
#: efficiency (dsa_efficiency_mach_scale=0.5), per the literature's standard
#: implementation of CS14 as a rescaled KR13 (no independent closed-form fit
#: exists for CS14 itself) -- not a separate constant.
DSA_EFFICIENCY_KANG_RYU_2013 = 1


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

    #: turn on the F_cr scattering/relaxation term (Jiang & Oh 2018) that
    #: damps F_cr toward -diffusion_coefficient * grad(P_cr), recovering
    #: Fick's-law diffusion in the appropriate limit (ladder item 4). Off by
    #: default: without it, the two-moment system is a pure undamped wave
    #: equation (verified in ladder items 1-3) -- this flag exists so that
    #: behavior is unchanged for configs/tests that don't ask for it.
    diffusive_relaxation: bool = False

    #: turn on diffusive shock acceleration: inject cosmic-ray energy into
    #: e_cr at shocks detected by the Pfrommer shock finder (PR #4,
    #: find_shocks_pfrommer), see cr_grey_injection.py (ladder items 7-8).
    diffusive_shock_acceleration: bool = False

    #: which DSA efficiency model inject_crs_at_shocks uses -- one of
    #: DSA_EFFICIENCY_CONSTANT (default, CosmicRayGreyParams.dsa_efficiency)
    #: or DSA_EFFICIENCY_KANG_RYU_2013 (Mach-dependent, ladder item 8; see
    #: cr_grey_injection.dsa_efficiency_kang_ryu_2013).
    dsa_efficiency_model: int = DSA_EFFICIENCY_CONSTANT


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

    #: smooth floor on |B|, in the same units as the magnetic-field primitive
    #: variable, used by anisotropic_flux_projection's b_hat = B / sqrt(|B|^2
    #: + b_field_floor^2) -- keeps the unit vector well-defined and
    #: differentiable at B=0 instead of a hard jnp.maximum/where floor.
    #: Should be well below any physically relevant |B| for the problem.
    b_field_floor: float = 1e-10

    #: smooth floor on the CR-pressure-coupling contribution to
    #: grey_cr_fast_speed (sqrt(gamma_cr (gamma_cr - 1) e_cr / rho)) -- added
    #: in quadrature under the sqrt, same "prefer smooth regularization"
    #: philosophy as b_field_floor. Without it, sqrt(x) has an infinite
    #: gradient at x = 0, which is exactly the CR-free-background case
    #: (e_cr = 0 outside a localized pulse) -- confirmed to NaN reverse-mode
    #: AD through time_integration otherwise (see cr_gradient_check.py).
    cr_pressure_speed_floor: float = 1e-10

    #: physical CR diffusion coefficient kappa (length^2 / time), only used
    #: when CosmicRayGreyConfig.diffusive_relaxation is set. Sets the F_cr
    #: relaxation rate nu = reduced_streaming_speed^2 / diffusion_coefficient
    #: in cr_grey_sources.cr_flux_relaxation_source; at steady state this
    #: relaxes F_cr toward -diffusion_coefficient * grad(P_cr), so the
    #: resulting diffusion coefficient for e_cr itself is
    #: diffusion_coefficient * (gamma_cr - 1).
    diffusion_coefficient: float = 1.0

    #: fraction of each shock's dissipated kinetic-energy flux
    #: (find_shocks_pfrommer's thermal_energy_flux) diverted into e_cr
    #: instead of gas thermal energy, when
    #: CosmicRayGreyConfig.diffusive_shock_acceleration is set and
    #: dsa_efficiency_model == DSA_EFFICIENCY_CONSTANT (the default). Ignored
    #: under DSA_EFFICIENCY_KANG_RYU_2013, which uses dsa_efficiency_mach_scale
    #: instead. Mirrors the retired old model's identical
    #: diffusive_shock_acceleration_efficiency default.
    dsa_efficiency: float = 0.1

    #: multiplicative scale applied to the Kang & Ryu (2013) Mach-dependent
    #: efficiency fit (cr_grey_injection.dsa_efficiency_kang_ryu_2013), only
    #: used under dsa_efficiency_model == DSA_EFFICIENCY_KANG_RYU_2013. 1.0
    #: (default) reproduces KR13 itself; 0.5 approximates Caprioli &
    #: Spitkovsky (2014) for quasi-parallel shocks, per the literature's
    #: standard implementation of CS14 as half of KR13 (Vazza et al. 2016;
    #: no independent closed-form CS14 fit exists). CS14's stronger finding
    #: -- efficiency also depends heavily on shock obliquity, near-zero for
    #: quasi-perpendicular shocks -- is not modeled here, since
    #: find_shocks_pfrommer does not currently expose shock-normal-vs-B
    #: obliquity; this scale is a Mach-only approximation of CS14.
    dsa_efficiency_mach_scale: float = 1.0

    #: simulation time before which diffusive_shock_acceleration injects
    #: nothing -- an ad-hoc guard against spurious shock detections before a
    #: real discontinuity has formed (mirrors the retired old model's
    #: identical diffusive_shock_acceleration_start_time). 0.0 (default): no
    #: delay.
    dsa_start_time: float = 0.0

    #: minimum Rankine-Hugoniot Mach number find_shocks_pfrommer requires to
    #: flag a cell as a shock surface, passed through to
    #: cr_grey_injection.inject_crs_at_shocks. Matches
    #: find_shocks_pfrommer's own default.
    dsa_mach_min: float = 1.3
