"""
End-to-end SNR-molecular-cloud pion-bump pytest (plan Sec. 4, item 15 --
Phase C's closing integration item, IC 443 / W44 analog; Ackermann et al.
2013, Science).

Reuses ladder item 11's exact SNR-into-clumpy-medium scaffold
(``pytests/cosmic_rays_grey/cr_snr_clumpy_medium.py``: a physically-scaled
3D CR-DSA point explosion into a warm ISM with three discrete, tanh-tapered
overdense clumps) rather than building a fresh literature-scale setup --
**user-picked design choice**, matching this project's established practice
of reusing existing, already-validated scaffolding at feasible cost (M5/M6
also reused scaffolding rather than chasing literal literature parameters).
The only change from item 11: one of the three clumps' density contrast is
raised from item 11's ISM-clump value (10x ambient) to a molecular-cloud-
like value (100x ambient) -- the hadronic-emission target. Only a single
run is needed here (DSA on, clumpy medium) -- item 11 already validated the
energy-partition/clumps-matter physics this run's dynamics rely on.

**Electron treatment -- resolved for this ladder item (user-picked, plan
Sec. 6's previously-open decision):** a fixed electron/proton ratio
``K_EP = 0.01`` (plan's own quoted 0.01-0.02 range), deriving the local
electron population directly from the local proton population's spectral
*shape* (same ``alpha``/``e_cutoff``, only the amplitude scaled down) --
the plan's own "simplest, good for first light" option. This resolves the
decision for *this* item; it does not retroactively require redoing items
13/14, which validated the emission formulas against directly-specified
spectra and never needed this decision either way.

**Physics pipeline, new for this ladder item -- turning a running
simulation's local ``e_cr`` into emission maps/SEDs (the piece items 13/14
explicitly deferred):**

1. Run the SNR-DSA simulation; read off the final ``e_cr`` (CR-grey energy
   density) and ``rho`` (gas density) fields, convert to physical units via
   this script's own ``CodeUnits`` (same pattern as
   ``pytests/stratified_ism/m5_cr_driven_outflow.py``).
2. Per grid cell, assume a DSA-motivated proton spectral *shape*
   (``alpha=2.0``, the standard test-particle strong-shock prediction;
   ``e_cutoff=100`` TeV, a typical assumed Galactic-SNR proton cutoff --
   the plan's own "grey caveat": shape assumed, not predicted) and
   normalize its amplitude so the spectrum's total energy content matches
   that cell's ``e_cr`` energy density times the cell's volume
   (:func:`astronomix._modules._cosmic_rays_grey.cr_grey_emission.proton_spectrum_normalized_to_energy`,
   new for this ladder item, naima's own ``Wp``/``We`` convention).
3. **Key efficiency insight, not obvious in advance:** every per-cell
   pion-decay/synchrotron/IC computation is *linear* in that cell's own
   spectrum amplitude (the proton/electron energy-loss cross sections
   don't depend on the spectrum's normalization at all, only its shape,
   which is shared across every cell here). So the expensive part of each
   emission formula (the full Kafexhiu/AKP10/Khangulyan integral) is
   computed **once**, for a unit-amplitude spectrum, giving a
   photon-energy-only "shape" curve ``K(E_gamma)`` -- the full per-cell (or
   domain-summed, or line-of-sight-projected) map is then just
   ``K(E_gamma) * (per-cell amplitude * per-cell target density)``, an
   outer product, not 128^3 independent expensive integrals. This is what
   actually makes a full 3D grid's worth of emission maps computationally
   tractable here.

**Two concrete, checkable predictions, not just "it ran":**

1. **Morphology**: the projected (line-of-sight-summed) pion-decay
   emission map's brightest pixel should sit near the molecular cloud's
   own projected position -- the plan's own point (Sec. 3): "the
   multiphase/dense target gas *is* the signal," since emissivity is
   propto ``n_gas * n_CR,p``, so even CR energy density comparable to the
   surrounding shocked gas lights up far brighter wherever it overlaps the
   dense cloud.
2. **The pion bump**: the domain-total hadronic SED (``E^2 dN/dE`` vs.
   ``E_gamma``) should fall measurably *below* the naive power-law
   extrapolation of its own higher-energy slope once ``E_gamma`` drops
   below the pi0-production kinematic threshold region (Ackermann et al.
   2013's headline "pion-decay bump" signature) -- a real spectral
   turnover, not a continuing power law. This is exactly the physics
   ladder item 13 already validated (Kafexhiu et al. (2014)'s ``F(Tp,
   Egamma)`` shape function, to floating-point precision against naima) --
   item 15 is the first time it's exercised end-to-end from a real
   simulation's CR content rather than a hand-specified spectrum.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
# Same reasoning as items 13/14: a *total* particle-count spectrum
# amplitude, here per grid cell, spans a huge dynamic range and overflows
# JAX's float32 default well before precision would matter.
jax.config.update("jax_enable_x64", True)

# general
from pathlib import Path

# numerics
import numpy as np

# jax
import jax.numpy as jnp

# units
from astropy import units as u
import astropy.constants as c

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# astronomix containers
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import CodeUnits
from astronomix import SimulationConfig, SimulationParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

# astronomix modules
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_emission import (
    GreyProtonSpectrum,
    proton_spectrum_normalized_to_energy,
    pion_decay_photon_spectrum_per_ev,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_emission_leptonic import (
    GreyElectronSpectrum,
    synchrotron_photon_spectrum,
    inverse_compton_photon_spectrum_planck,
)
from astronomix.shock_finder3D.pfrommer_shock_finder import find_shocks_pfrommer

# ---- physical setup: item 11's scaffold, unchanged unless noted ----
GAMMA = 5.0 / 3.0
NUM_CELLS = 128  # item 11's own lower-cost variant; item 15 doesn't need item 11's tight
                 # energy-partition calibration precision, just a qualitative emission check.
DSA_MACH_MIN = 1.3
DSA_EFFICIENCY = 0.1

CODE_UNITS = CodeUnits(5 * u.parsec, 1 * u.M_sun, 100 * u.km / u.s)

E_SN = 1e51 * u.erg
N_AMBIENT = 1 / u.cm**3
T_AMBIENT = 1e4 * u.K
RHO_AMBIENT = N_AMBIENT * c.m_p
P_AMBIENT = N_AMBIENT * c.k_B * T_AMBIENT

BOX_SIZE_PHYS = 20.0 * u.pc
BOX_SIZE = BOX_SIZE_PHYS.to(CODE_UNITS.code_length).value

T_END_PHYS = 1000.0 * u.yr
T_END = T_END_PHYS.to(CODE_UNITS.code_time).value

R_EXPLOSION_PHYS = 2.0 * (BOX_SIZE_PHYS / NUM_CELLS)
SMOOTH_CELLS = 2.0

_CLUMP_DIRECTIONS_RAW = np.array(
    [
        [1.0, 0.3, 0.2],
        [-0.4, 1.0, -0.3],
        [0.2, -0.5, 1.0],
    ]
)
CLUMP_DIRECTIONS = _CLUMP_DIRECTIONS_RAW / np.linalg.norm(_CLUMP_DIRECTIONS_RAW, axis=1, keepdims=True)
CLUMP_DISTANCE_PHYS = 3.0 * u.pc
CLUMP_RADIUS_PHYS = 1.0 * u.pc
CLUMP_SMOOTH_CELLS = 2.0

#: item 11's original ISM-clump contrast, kept for the two non-cloud clumps.
ISM_CLUMP_CONTRAST = 10.0
#: new for this ladder item: one clump is a molecular-cloud-like density
#: enhancement -- the hadronic-emission target. Index into CLUMP_DIRECTIONS.
MOLECULAR_CLOUD_INDEX = 0
MOLECULAR_CLOUD_CONTRAST = 100.0

# ---- emission-model assumptions, new for this ladder item ----
CR_ALPHA = 2.0  # standard test-particle strong-shock DSA index
CR_E_CUTOFF_GEV = 1e5  # 100 TeV, a typical assumed Galactic-SNR proton cutoff
CR_E_0_GEV = 1.0
K_EP = 0.01  # fixed electron/proton ratio (plan's "simplest, good for first light")
B_FIELD_GAUSS = 3.24e-6  # naima's own Synchrotron default (CMB-equipartition ISM value)
T_CMB_K = 2.72548
U_CMB_ERG_CM3 = float((4.0 * c.sigma_sb / c.c * (T_CMB_K * u.K) ** 4).to(u.erg / u.cm**3).value)

GEV_TO_ERG = float((1.0 * u.GeV).to(u.erg).value)

# Photon energies spanning well below and above the pi0-production
# threshold region, in eV throughout (this script's common unit across the
# hadronic and leptonic channels).
PHOTON_ENERGY_EV = jnp.logspace(7.3, 11.7, 45)  # ~20 MeV to ~500 GeV


def _run_snr_molecular_cloud():
    """Run the single SNR-DSA-into-clumpy-medium simulation (one clump
    upgraded to molecular-cloud density); return the final state and
    diagnostics."""
    config = SimulationConfig(
        progress_bar=True,
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=NUM_CELLS,
        exact_end_time=True,
        cosmic_ray_grey_config=CosmicRayGreyConfig(grey_cosmic_rays=True, diffusive_shock_acceleration=True),
    )
    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    shape = (NUM_CELLS, NUM_CELLS, NUM_CELLS)
    dx = BOX_SIZE / NUM_CELLS
    cell_volume = dx**3

    rho_ambient_code = RHO_AMBIENT.to(CODE_UNITS.code_density).value
    p_ambient_code = P_AMBIENT.to(CODE_UNITS.code_pressure).value

    density = jnp.ones(shape) * rho_ambient_code

    axis = (jnp.arange(NUM_CELLS) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    box_center = BOX_SIZE / 2.0
    clump_distance_code = CLUMP_DISTANCE_PHYS.to(CODE_UNITS.code_length).value
    clump_radius_code = CLUMP_RADIUS_PHYS.to(CODE_UNITS.code_length).value
    clump_smooth_width = CLUMP_SMOOTH_CELLS * dx

    cloud_center_code = None
    for i, direction in enumerate(CLUMP_DIRECTIONS):
        cx = box_center + direction[0] * clump_distance_code
        cy = box_center + direction[1] * clump_distance_code
        cz = box_center + direction[2] * clump_distance_code
        if i == MOLECULAR_CLOUD_INDEX:
            cloud_center_code = (cx, cy, cz)
        contrast = MOLECULAR_CLOUD_CONTRAST if i == MOLECULAR_CLOUD_INDEX else ISM_CLUMP_CONTRAST
        dist = jnp.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
        weight = 0.5 * (1.0 - jnp.tanh((dist - clump_radius_code) / clump_smooth_width))
        density = density + rho_ambient_code * (contrast - 1.0) * weight

    zeros = jnp.zeros(shape)

    radius = helper_data.r
    r_explosion_code = R_EXPLOSION_PHYS.to(CODE_UNITS.code_length).value
    smooth_width = SMOOTH_CELLS * dx
    explosion_weight = 0.5 * (1.0 - jnp.tanh((radius - r_explosion_code) / smooth_width))
    e_sn_code = E_SN.to(CODE_UNITS.code_energy).value
    delta_p = e_sn_code * (GAMMA - 1.0) / (jnp.sum(explosion_weight) * cell_volume)
    gas_pressure = p_ambient_code + delta_p * explosion_weight

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        gas_pressure=gas_pressure,
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        cosmic_ray_grey_params=CosmicRayGreyParams(dsa_efficiency=DSA_EFFICIENCY, dsa_mach_min=DSA_MACH_MIN),
    )

    final_state = time_integration(initial_state, config, params, registered_variables)

    shock_result = find_shocks_pfrommer(
        final_state, config, registered_variables, helper_data, mach_min=DSA_MACH_MIN
    )
    surf = np.array(shock_result.shock_surface_cells)
    r_np = np.array(radius)
    r_shock_max = float(r_np[surf].max()) if surf.any() else float("nan")

    return dict(
        final_state=final_state,
        registered_variables=registered_variables,
        cell_volume_code=cell_volume,
        cloud_center_code=cloud_center_code,
        r_shock_max=r_shock_max,
    )


def _build_emission_maps(run):
    """Turn the run's final e_cr/rho fields into per-cell proton/electron
    spectra, then (via the linearity trick, see module docstring) the
    domain-total SEDs and a projected pion-decay morphology map."""
    rv = run["registered_variables"]
    state = run["final_state"]

    rho_code = state[rv.density_index]
    e_cr_code = state[rv.cosmic_ray_e_index]

    rho_phys = (rho_code * CODE_UNITS.code_density).to(u.g / u.cm**3).value
    e_cr_density_unit = CODE_UNITS.code_energy / CODE_UNITS.code_length**3
    e_cr_phys = (e_cr_code * e_cr_density_unit).to(u.erg / u.cm**3).value

    n_gas = rho_phys / c.m_p.to(u.g).value  # cm^-3, matches this setup's RHO_AMBIENT = N_AMBIENT * m_p

    dx_phys_cm = (BOX_SIZE_PHYS / NUM_CELLS).to(u.cm).value
    cell_volume_cm3 = dx_phys_cm**3

    total_energy_gev = jnp.asarray(e_cr_phys) * cell_volume_cm3 / GEV_TO_ERG

    proton_spectrum = proton_spectrum_normalized_to_energy(
        total_energy_gev, e_0=CR_E_0_GEV, alpha=CR_ALPHA, e_cutoff=CR_E_CUTOFF_GEV
    )
    proton_amplitude = proton_spectrum.amplitude  # (NUM_CELLS,)^3, particles/GeV

    photon_energy_gev = PHOTON_ENERGY_EV * 1e-9
    unit_proton_spectrum = GreyProtonSpectrum(amplitude=1.0, e_0=CR_E_0_GEV, alpha=CR_ALPHA, e_cutoff=CR_E_CUTOFF_GEV)
    k_pion = pion_decay_photon_spectrum_per_ev(unit_proton_spectrum, photon_energy_gev, 1.0)  # (n_E,)

    pion_prefactor = proton_amplitude * jnp.asarray(n_gas)  # (NUM_CELLS,)^3
    sed_pion = k_pion * jnp.sum(pion_prefactor)  # (n_E,), photons/(s eV), domain total

    # Electron population via a fixed K_ep ratio to the local proton
    # population's *shape* (same alpha/e_cutoff, amplitude scaled and
    # unit-converted GeV -> eV -- see module docstring).
    amplitude_e_ev = K_EP * proton_amplitude * 1e-9
    e_0_ev = CR_E_0_GEV * 1e9
    e_cutoff_ev = CR_E_CUTOFF_GEV * 1e9
    unit_electron_spectrum = GreyElectronSpectrum(amplitude=1.0, e_0=e_0_ev, alpha=CR_ALPHA, e_cutoff=e_cutoff_ev)
    k_sync = synchrotron_photon_spectrum(unit_electron_spectrum, PHOTON_ENERGY_EV, B_FIELD_GAUSS)
    k_ic = inverse_compton_photon_spectrum_planck(unit_electron_spectrum, PHOTON_ENERGY_EV, T_CMB_K, U_CMB_ERG_CM3)

    electron_total = jnp.sum(amplitude_e_ev)
    sed_sync = k_sync * electron_total
    sed_ic = k_ic * electron_total

    # Line-of-sight (z-axis) projected pion-decay morphology map -- the
    # spatial pattern doesn't depend on which photon energy the K_pion
    # scalar is evaluated at, so any fixed reference energy works; 1 GeV is
    # a typical Fermi-LAT reference band.
    projected_prefactor = jnp.sum(pion_prefactor, axis=2)  # (NUM_CELLS, NUM_CELLS)
    k_pion_1gev = float(pion_decay_photon_spectrum_per_ev(unit_proton_spectrum, jnp.asarray(1.0), 1.0))
    morphology_map = np.asarray(projected_prefactor) * k_pion_1gev

    return dict(
        sed_pion=sed_pion,
        sed_sync=sed_sync,
        sed_ic=sed_ic,
        morphology_map=morphology_map,
        n_gas=n_gas,
        e_cr_phys=e_cr_phys,
    )


def test_snr_molecular_cloud_pion_bump(
    hotspot_tolerance_cells: int = 12,
    bump_suppression_min: float = 0.5,
    domain_half_width: float = BOX_SIZE / 2,
    containment_margin: float = 0.4,
):
    """SNR-DSA-into-clumpy-medium run, then the pion-bump morphology/SED checks.

    Args:
        hotspot_tolerance_cells: Max allowed distance (grid cells) between
            the projected pion-decay map's brightest pixel and the
            molecular cloud's own projected center.
        bump_suppression_min: Minimum required fractional suppression of
            the actual low-energy SED value below the naive power-law
            extrapolation from the high-energy slope (Check 2, "the pion
            bump").
        domain_half_width: Half the box size (code units) -- for the
            shock-containment sanity check (item 11's own convention).
        containment_margin: Minimum gap the forward shock must stay within
            the domain (code units).
    """
    run = _run_snr_molecular_cloud()

    assert not bool(jnp.any(jnp.isnan(run["final_state"]))), "SNR-molecular-cloud run produced NaNs."
    assert run["r_shock_max"] < domain_half_width - containment_margin, (
        f"Forward shock (r={run['r_shock_max']:.4f}) is too close to the open boundary "
        f"(domain half-width {domain_half_width:.4f}) for the emission checks below to be meaningful."
    )

    maps = _build_emission_maps(run)

    assert bool(jnp.all(jnp.isfinite(maps["sed_pion"]))), "Pion-decay SED contains non-finite values."
    assert bool(jnp.all(jnp.isfinite(maps["sed_sync"]))), "Synchrotron SED contains non-finite values."
    assert bool(jnp.all(jnp.isfinite(maps["sed_ic"]))), "Inverse-Compton SED contains non-finite values."
    assert float(jnp.sum(maps["sed_pion"])) > 0.0, "Pion-decay SED is identically zero -- no CR-gas overlap at all."

    # --- Check 1: morphology -- projected pion-decay emission peaks near the cloud ---
    dx = BOX_SIZE / NUM_CELLS
    cloud_cx, cloud_cy, _ = run["cloud_center_code"]
    cloud_i = int(round(cloud_cx / dx - 0.5))
    cloud_j = int(round(cloud_cy / dx - 0.5))

    peak_i, peak_j = np.unravel_index(np.argmax(maps["morphology_map"]), maps["morphology_map"].shape)
    hotspot_dist = float(np.hypot(peak_i - cloud_i, peak_j - cloud_j))
    assert hotspot_dist <= hotspot_tolerance_cells, (
        f"Projected pion-decay emission peaks {hotspot_dist:.1f} cells from the molecular "
        f"cloud's own projected center ({cloud_i}, {cloud_j}) -- tolerance {hotspot_tolerance_cells} "
        f"cells. Peak at ({peak_i}, {peak_j})."
    )

    # --- Check 2: the pion bump -- low-energy suppression below the naive power-law
    #     extrapolation of the high-energy slope ---
    e_hi_1, e_hi_2 = 3e9, 3e10  # 3 GeV, 30 GeV, in eV -- well above the pi0-threshold region
    sed_e2 = PHOTON_ENERGY_EV**2 * maps["sed_pion"]  # E^2 dN/dE, the standard SED convention

    def _interp_log(x_query, x, y):
        log_y = jnp.interp(jnp.log(x_query), jnp.log(x), jnp.log(jnp.maximum(y, 1e-300)))
        return jnp.exp(log_y)

    sed_hi_1 = _interp_log(e_hi_1, PHOTON_ENERGY_EV, sed_e2)
    sed_hi_2 = _interp_log(e_hi_2, PHOTON_ENERGY_EV, sed_e2)
    slope = jnp.log(sed_hi_2 / sed_hi_1) / jnp.log(e_hi_2 / e_hi_1)

    e_lo = 5e7  # 50 MeV, below the pi0-production kinematic threshold region
    sed_lo_actual = _interp_log(e_lo, PHOTON_ENERGY_EV, sed_e2)
    sed_lo_naive = sed_hi_1 * (e_lo / e_hi_1) ** slope

    suppression = 1.0 - float(sed_lo_actual / sed_lo_naive)
    assert suppression > bump_suppression_min, (
        f"Pion-decay SED at 50 MeV ({float(sed_lo_actual):.3e}) is not suppressed enough below "
        f"the naive power-law extrapolation from the 3-30 GeV slope ({float(sed_lo_naive):.3e}) -- "
        f"suppression {suppression:.3f} <= min {bump_suppression_min}. This is the 'pion bump' "
        f"signature (Ackermann et al. 2013); its absence would mean the low-energy turnover this "
        f"ladder item is checking for isn't actually present."
    )

    # Diagnostic plots: multi-wavelength SED (left) and projected pion-decay morphology (right).
    fig, (ax_sed, ax_map) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax_sed.loglog(PHOTON_ENERGY_EV, PHOTON_ENERGY_EV**2 * maps["sed_pion"], "-", lw=2.5, color="C0", label="pion decay (hadronic)")
    ax_sed.loglog(PHOTON_ENERGY_EV, PHOTON_ENERGY_EV**2 * maps["sed_sync"], "--", lw=1.5, color="C1", label="synchrotron")
    ax_sed.loglog(PHOTON_ENERGY_EV, PHOTON_ENERGY_EV**2 * maps["sed_ic"], "-.", lw=1.5, color="C2", label="inverse Compton (CMB)")
    ax_sed.axvline(1.35e8, color="grey", ls=":", lw=1, label=r"$m_{\pi^0}/2$")
    ax_sed.set_xlabel(r"$E_\gamma$ [eV]")
    ax_sed.set_ylabel(r"$E_\gamma^2\,dN/dE\,dt$ [eV/s]")
    ax_sed.set_title("Domain-total multi-wavelength SED")
    ax_sed.legend(fontsize=8)

    extent = [0, BOX_SIZE, 0, BOX_SIZE]
    floor = max(float(maps["morphology_map"].max()) * 1e-6, 1e-300)
    im = ax_map.imshow(
        np.maximum(maps["morphology_map"], floor).T, origin="lower", extent=extent,
        norm=LogNorm(vmin=floor, vmax=maps["morphology_map"].max()), cmap="inferno",
    )
    fig.colorbar(im, ax=ax_map, label="pion-decay rate @ 1 GeV (proj., a.u.)")
    ax_map.scatter([cloud_cx], [cloud_cy], marker="x", color="cyan", s=80, label="molecular cloud center")
    ax_map.set_title("Projected pion-decay morphology (z-summed)")
    ax_map.set_xlabel("x (code units)")
    ax_map.set_ylabel("y (code units)")
    ax_map.legend(fontsize=8)

    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics"
    pics_dir.mkdir(exist_ok=True)
    fig.savefig(pics_dir / "cr_snr_molecular_cloud_pion_bump.svg")
    plt.close(fig)


if __name__ == "__main__":
    test_snr_molecular_cloud_pion_bump()
