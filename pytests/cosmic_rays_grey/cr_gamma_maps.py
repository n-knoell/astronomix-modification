"""
Gamma-ray proxy, step 2 (GAMMA_RAY_PLAN.md Sec. 3): pion-decay maps and SEDs
from stored simulation states, ``j_gamma = n_H e_cr K(E)``.

Checks (synthetic fields, no simulation):
  2a. ``hadronic_target_field`` code-unit conversion vs a direct astropy calculation;
  2b. band integral (default 64 pts/decade) vs a 10x finer grid;
  2c. map / SED consistency: summed pixel fluxes of a band map equal the band
      integral of the SED (exact identity), uniform sphere;
  2d. uniform sphere: projected map vs the analytic chord 2 sqrt(R^2 - r^2);
  2e. AD vs FD of a map-derived scalar w.r.t. alpha and one cell's e_cr.

Demo (if the stored item-11 states exist; nothing is rerun):
  SNR into a clumpy vs uniform medium (item 11, 240^3, DSA eta = 0.1, t = 1 kyr),
  E^2 F at d = 1.5 kpc for alpha = 2.3 (default), 2.0 and the Ackermann et al.
  (2013) IC 443 shape, and band maps 0.1-1 / 1-10 / 10-100 GeV.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
jax.config.update("jax_enable_x64", True)

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
import astropy.constants as c

from astronomix import CodeUnits
from astronomix._modules._cosmic_rays_grey.cr_grey_emission import (
    ACKERMANN13_IC443,
    ProtonMomentumSpectrum,
    _trapz_loglog,
    hadronic_target_field,
    pion_decay_band_energy_per_ecr,
    pion_decay_band_map,
    pion_decay_emissivity_per_ecr,
    pion_decay_sed,
)

PICS_DIR = Path(__file__).resolve().parent / "pics" / "gamma_step2_maps"
ITEM11_DIR = Path("/export/scratch/nknoell/phaseB/item11_hires/data")
ITEM11_FILES = {"clumpy": "N240_clumpy_eff0.1.npz", "uniform": "N240_uniform_eff0.1.npz"}
# item-11 setup (pytests/cosmic_rays_grey/cr_snr_clumpy_medium.py)
ITEM11_UNITS = CodeUnits(5 * u.parsec, 1 * u.M_sun, 100 * u.km / u.s)
ITEM11_BOX_PC = 20.0
ITEM11_X_H = 1.0  # the setup defines rho = n m_p with n meant as hydrogen
DISTANCE_CM = (1.5 * u.kpc).to_value(u.cm)  # IC 443

BANDS_GEV = ((0.1, 1.0), (1.0, 10.0), (10.0, 100.0))

UNITS_REL_TOL = 1e-12
BAND_REL_TOL = 1e-4
IDENTITY_REL_TOL = 1e-10
GRAD_REL_TOL = 2e-6


def _sphere_field(n=64, radius_frac=0.3, value=1.0):
    x = (np.arange(n) + 0.5) / n - 0.5
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.where(X ** 2 + Y ** 2 + Z ** 2 < radius_frac ** 2, value, 0.0)


def test_target_field_units():
    rng = np.random.default_rng(1)
    rho_code = rng.uniform(0.1, 10.0, (4, 4, 4))
    ecr_code = rng.uniform(1e-3, 1.0, (4, 4, 4))
    mine = np.asarray(hadronic_target_field(jnp.asarray(rho_code), jnp.asarray(ecr_code), ITEM11_UNITS, 0.71))
    n_h = (0.71 * rho_code * ITEM11_UNITS.code_density / c.m_p).to_value(u.cm ** -3)
    e_cr = (ecr_code * ITEM11_UNITS.code_energy / ITEM11_UNITS.code_length ** 3).to_value(u.erg / u.cm ** 3)
    rel = np.max(np.abs(mine / (n_h * e_cr) - 1.0))
    print(f"  2a  target field vs astropy: max rel. err {rel:.1e}")
    assert rel < UNITS_REL_TOL, rel
    return rel


def test_band_integral_convergence():
    worst = 0.0
    for spec in (ProtonMomentumSpectrum(), ProtonMomentumSpectrum(alpha=2.0), ProtonMomentumSpectrum(**ACKERMANN13_IC443)):
        for e1, e2 in BANDS_GEV + ((0.03, 0.1), (100.0, 1e4)):
            coarse = float(pion_decay_band_energy_per_ecr(e1, e2, spec))
            fine = float(pion_decay_band_energy_per_ecr(e1, e2, spec, n_per_decade=640))
            worst = max(worst, abs(coarse / fine - 1.0))
    print(f"  2b  band integral, 64 (default) vs 640 pts/decade: worst rel. diff {worst:.1e}")
    assert worst < BAND_REL_TOL, worst
    return worst


def test_map_sed_identity_and_sphere_chord():
    n, box_cm, rfrac = 64, 3.0e19, 0.3
    dx = box_cm / n
    w = jnp.asarray(_sphere_field(n, rfrac, 2.5e-10))
    worst_id = 0.0
    for e1, e2 in BANDS_GEV:
        imap = pion_decay_band_map(w, dx, e1, e2)
        flux_map = float(jnp.sum(imap)) * dx * dx / DISTANCE_CM ** 2
        E = jnp.logspace(np.log10(e1), np.log10(e2), int(np.ceil(64 * np.log10(e2 / e1))) + 1)  # same grid as the band integral
        e_dfde = pion_decay_sed(w, dx ** 3, E, distance_cm=DISTANCE_CM) / E  # E dF/dE
        flux_sed = float(_trapz_loglog(e_dfde, E))
        worst_id = max(worst_id, abs(flux_map / flux_sed - 1.0))
    print(f"  2c  sum of map pixel fluxes vs SED band integral: max rel. diff {worst_id:.1e}")
    assert worst_id < IDENTITY_REL_TOL, worst_id

    # chord profile: I(r) = (1/4pi) w * 2 sqrt(R^2 - r^2) * band
    imap = np.asarray(pion_decay_band_map(w, dx, 1.0, 10.0))
    band = float(pion_decay_band_energy_per_ecr(1.0, 10.0))
    x = ((np.arange(n) + 0.5) / n - 0.5) * box_cm
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y)
    R = rfrac * box_cm
    analytic = 2.5e-10 * 2.0 * np.sqrt(np.clip(R ** 2 - r ** 2, 0.0, None)) * band / (4 * np.pi)
    inner = r < 0.8 * R  # away from the stair-stepped limb
    rel_inner = np.max(np.abs(imap[inner] / analytic[inner] - 1.0))
    # discretisation: the chord through a voxelised sphere is off by up to ~1 cell
    tol = 1.5 * dx / (2.0 * np.sqrt(R ** 2 - (0.8 * R) ** 2))
    print(f"  2d  sphere map vs analytic chord (r < 0.8 R): max rel. err {rel_inner:.3f} (tol {tol:.3f})")
    assert rel_inner < tol, rel_inner
    return worst_id, rel_inner


def test_gradients_ad_vs_fd():
    n = 16
    w0 = jnp.asarray(_sphere_field(n, 0.35, 1.0)) * 1e-10
    dx = 1e18

    def scalar(alpha, ecr_scale):
        w = w0.at[n // 2, n // 2, n // 2].multiply(ecr_scale)
        s = ProtonMomentumSpectrum(alpha=alpha)
        return jnp.sum(pion_decay_band_map(w, dx, 1.0, 10.0, s) ** 2) / jnp.sum(pion_decay_band_map(w0, dx, 1.0, 10.0) ** 2)

    a0, s0 = jnp.asarray(2.3), jnp.asarray(1.0)
    ad = jax.grad(scalar, argnums=(0, 1))(a0, s0)
    worst = 0.0
    for i, (name, x0) in enumerate((("alpha", a0), ("e_cr(centre)", s0))):
        h = 1e-5 * float(x0)
        args_p = [a0, s0]; args_m = [a0, s0]
        args_p[i] = x0 + h; args_m[i] = x0 - h
        fd = (float(scalar(*args_p)) - float(scalar(*args_m))) / (2 * h)
        rel = abs(float(ad[i]) - fd) / abs(fd)
        worst = max(worst, rel)
        print(f"  2e  d/d{name:13s} AD={float(ad[i]):+.6e} FD={fd:+.6e} rel={rel:.1e}")
        assert np.isfinite(float(ad[i])) and rel < GRAD_REL_TOL, rel
    return worst


# ---------------------------------------------------------------- demo
def _load_item11(name):
    z = np.load(ITEM11_DIR / ITEM11_FILES[name])
    st = z["state"]
    w = hadronic_target_field(jnp.asarray(st[int(z["density_index"])]), jnp.asarray(st[int(z["cosmic_ray_e_index"])]),
                              ITEM11_UNITS, ITEM11_X_H)
    n = int(z["num_cells"])
    dx_cm = (ITEM11_BOX_PC / n * u.pc).to_value(u.cm)
    e_cr_code = float(z["E_cr"])
    e_cr_erg = (e_cr_code * ITEM11_UNITS.code_energy).to_value(u.erg)
    return w, dx_cm, n, e_cr_erg


def demo_item11(lines):
    if not all((ITEM11_DIR / f).exists() for f in ITEM11_FILES.values()):
        lines.append("item-11 demo skipped: stored states not found")
        return
    E = jnp.logspace(-1.5, 3, 80)
    i1 = int(np.argmin(np.abs(np.asarray(E) - 1.0)))
    shapes = {r"$\alpha=2.3$ (default)": ProtonMomentumSpectrum(),
              r"$\alpha=2.0$": ProtonMomentumSpectrum(alpha=2.0),
              "IC 443 shape (Ackermann+13)": ProtonMomentumSpectrum(**ACKERMANN13_IC443)}
    fig = plt.figure(figsize=(13, 7.6))
    ax_sed = fig.add_subplot(2, 3, (1, 4))
    seds, maps = {}, {}
    for case, ls in (("clumpy", "-"), ("uniform", "--")):
        w, dx, n, e_cr_erg = _load_item11(case)
        lines.append(f"item 11 {case} (N={n}, t=1 kyr): E_cr = {e_cr_erg:.3e} erg, "
                     f"sum(n_H e_cr dV) = {float(jnp.sum(w)) * dx ** 3:.3e} erg")
        for k, (label, s) in enumerate(shapes.items()):
            sed = np.asarray(pion_decay_sed(w, dx ** 3, E, s, distance_cm=DISTANCE_CM))
            seds[(case, label)] = sed
            ax_sed.loglog(np.asarray(E), sed, ls, color=f"C{k}", lw=2.0 if k == 0 else 1.3,
                          label=label if case == "clumpy" else None)
            L = float(pion_decay_band_energy_per_ecr(0.1, 100.0, s)) * float(jnp.sum(w)) * dx ** 3
            lines.append(f"    {label:30s} E^2F(1 GeV, 1.5 kpc) = {sed[i1]:.3e} erg cm^-2 s^-1, "
                         f"L(0.1-100 GeV) = {L:.3e} erg/s")
        maps[case] = [np.asarray(pion_decay_band_map(w, dx, e1, e2)) for e1, e2 in BANDS_GEV]
        # map/SED identity on the stored state (1-10 GeV band)
        E_b = jnp.logspace(0, 1, 65)  # same grid as the band integral (64/decade)
        flux_sed = float(_trapz_loglog(pion_decay_sed(w, dx ** 3, E_b, distance_cm=DISTANCE_CM) / E_b, E_b))
        flux_map = float(np.sum(maps[case][1])) * dx * dx / DISTANCE_CM ** 2
        lines.append(f"    map/SED identity on stored state (1-10 GeV): {flux_map / flux_sed - 1:+.1e}")
        del w
    for label in shapes:
        lines.append(f"    clumpy/uniform E^2F at 1 GeV, {label}: {seds[('clumpy', label)][i1] / seds[('uniform', label)][i1]:.3f}")
    ax_sed.plot([], [], "k-", label="clumpy"); ax_sed.plot([], [], "k--", label="uniform")
    ax_sed.set_xlabel(r"$E_\gamma$ [GeV]")
    ax_sed.set_ylabel(r"$E^2\,dF/dE$ at 1.5 kpc [erg cm$^{-2}$ s$^{-1}$]")
    ax_sed.set_title("Item-11 SNR (240$^3$, 1 kyr, DSA $\\eta$=0.1): pion decay", fontsize=10)
    ax_sed.legend(fontsize=8, loc="lower center")
    extent = [0, ITEM11_BOX_PC, 0, ITEM11_BOX_PC]
    panels = [("clumpy", k) for k in range(3)] + [("uniform", 1)]
    vmax = {k: max(maps["clumpy"][k].max(), maps["uniform"][k].max()) for k in range(3)}
    for pos, (case, k) in zip((2, 3, 5, 6), panels):
        ax = fig.add_subplot(2, 3, pos)
        e1, e2 = BANDS_GEV[k]
        im = ax.imshow(maps[case][k].T, origin="lower", extent=extent,
                       norm=LogNorm(vmax[k] * 1e-3, vmax[k]), cmap="magma")
        ax.set_title(f"{case}, {e1:g}-{e2:g} GeV, " + r"$\alpha=2.3$", fontsize=10)
        ax.set_xlabel("x [pc]"); ax.set_ylabel("y [pc]")
        fig.colorbar(im, ax=ax, label=r"$I$ [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]", shrink=0.85)
    fig.tight_layout()
    fig.savefig(PICS_DIR / "item11_sed_and_maps.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["Gamma-ray proxy step 2: maps and SEDs (cr_gamma_maps.py)", ""]
    print("2a"); r1 = test_target_field_units()
    print("2b"); r2 = test_band_integral_convergence()
    print("2c/2d"); r3, r4 = test_map_sed_identity_and_sphere_chord()
    print("2e"); r5 = test_gradients_ad_vs_fd()
    lines += [f"2a units vs astropy          {r1:.1e} (tol {UNITS_REL_TOL:.0e})",
              f"2b band 64 vs 640 pts/dec    {r2:.1e} (tol {BAND_REL_TOL:.0e})",
              f"2c map/SED identity          {r3:.1e} (tol {IDENTITY_REL_TOL:.0e})",
              f"2d sphere chord (r < 0.8 R)  {r4:.3f}",
              f"2e AD vs FD                  {r5:.1e} (tol {GRAD_REL_TOL:.0e})", ""]
    demo_item11(lines)
    text = "\n".join(lines)
    print(text)
    (PICS_DIR / "summary.txt").write_text(text + "\n")
