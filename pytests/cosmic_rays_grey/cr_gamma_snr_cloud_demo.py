"""
Gamma-ray proxy, steps 3 + 4 (GAMMA_RAY_PLAN.md Sec. 3): post-processing of the
runs made by /export/scratch/nknoell/gamma_step3/run_alpha_budget.py
(nothing is simulated here).

Step 3 -- alpha budget. Each run evaluated
``cr_grey_injection.dsa_injection_rate_by_slope`` (unit-checked in
``cr_dsa_slope_budget.py``) at evenly spaced times; integrating the per-bin
injection rate over time gives the injected CR energy per test-particle slope
alpha = (r + 2) / (r - 1). Pass criterion of the plan: >= 90 % of the injected
energy at alpha < 2.2, i.e. one global spectral slope is justified.
The interval before the first sample (t < t_end / NSNAP) is only covered by
the trapezoid from zero; it holds the strongest (alpha ~ 2) shock, so a missing
piece makes the alpha < 2.2 share conservative.

Step 4 -- demo on the cached 240^3 item-15 state (SNR + 100x molecular cloud,
2 kyr, DSA eta = 0.1, float32 run): pion-decay SED (alpha = 2.3 default; 2.0
and the Ackermann et al. 2013 IC 443 broken power law for comparison), the
item-15 checks (hotspot on the cloud, sub-100-MeV pion bump), IC on the CMB with
K_ep = 0.01 for context, and the 1-10 GeV map.
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

from astronomix import CodeUnits
from astronomix._modules._cosmic_rays_grey.cr_grey_emission import (
    ACKERMANN13_IC443,
    ERG_TO_GEV,
    M_P_GEV,
    ProtonMomentumSpectrum,
    hadronic_target_field,
    pion_decay_band_map,
    pion_decay_sed,
    proton_number_per_total_energy,
    proton_unit_kinetic_energy_gev,
)
from astronomix._modules._cosmic_rays_grey.cr_grey_emission_leptonic import (
    GreyElectronSpectrum,
    inverse_compton_photon_spectrum_planck,
)

DATA = Path("/export/scratch/nknoell/gamma_step3/data")
PICS_DIR = Path(__file__).resolve().parent / "pics" / "gamma_step3_4"
BUDGET_RUNS = [("item15", 128), ("item15", 192), ("item15", 240), ("item11u", 240)]
DEMO_STATE = DATA / "item15_N240_state.npz"
X_H = 1.0  # item 11/15 define rho = n m_p with n meant as hydrogen
DISTANCE_CM = (1.5 * u.kpc).to_value(u.cm)
K_EP = 0.01
T_CMB_K = 2.72548
U_CMB = 4.17e-13  # erg cm^-3 (a T_CMB^4)
BUDGET_CRITERION = 0.90  # share of injected energy at alpha < 2.2
# Ackermann et al. (2013) Fig. 2 peak of IC 443, E^2 F ~ 1.2e-10 erg cm^-2 s^-1 near ~1 GeV
# (value as read in PROGRESS.md 2026-10-04), used only to place a shape reference.
IC443_PEAK = 1.2e-10


# ------------------------------------------------------------------ step 3
def budget(case, n):
    z = np.load(DATA / f"{case}_N{n}_budget.npz")
    t, rates = z["t"], z["rates"]
    e_bins = np.trapezoid(rates, t, axis=0) * float(z["code_energy_erg"])
    edges = z["slope_edges"]
    share = e_bins / e_bins.sum()
    return dict(t=t * float(z["code_time_s"]) / 3.156e7, rates=rates, share=share, edges=edges,
                e_inj=e_bins.sum(), e_cr_end=float(z["e_cr"][-1]) * float(z["code_energy_erg"]),
                below_2p2=share[: list(edges).index(2.2) + 1].sum(),
                below_2p3=share[: list(edges).index(2.3) + 1].sum(), n_snap=len(t), wall=float(z["wall"]))


def step3(lines):
    lines += ["Step 3: alpha budget (share of time-integrated DSA injection per test-particle slope)"]
    results = {}
    for case, n in BUDGET_RUNS:
        f = DATA / f"{case}_N{n}_budget.npz"
        if not f.exists():
            lines.append(f"  {case} N={n}: missing")
            continue
        b = budget(case, n)
        results[(case, n)] = b
        lab = ["<2.05"] + [f"<{e:g}" for e in b["edges"][1:]] + [f">={b['edges'][-1]:g}"]
        lines.append(f"  {case} N={n} ({b['n_snap']} samples, wall {b['wall'] / 3600:.2f} h): "
                     f"E_inj = {b['e_inj']:.3e} erg, E_cr(end) = {b['e_cr_end']:.3e} erg; "
                     f"share alpha<2.2 = {b['below_2p2']:.4f}, <2.3 = {b['below_2p3']:.4f} "
                     f"-> {'PASS' if b['below_2p2'] >= BUDGET_CRITERION else 'FAIL'}")
        lines.append("      bins " + "  ".join(f"[{lo},{hi}):{s:.4f}" for lo, hi, s in zip(
            ["-inf"] + [f"{e:g}" for e in b["edges"]], [f"{e:g}" for e in b["edges"]] + ["inf"], b["share"])))
    if not results:
        return results
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, ((case, n), b) in enumerate(results.items()):
        tot = b["rates"].sum(axis=1)
        inst = np.where(tot > 0, b["rates"][:, : list(b["edges"]).index(2.2) + 1].sum(axis=1) / np.where(tot > 0, tot, 1), np.nan)
        a1.plot(b["t"], inst, color=f"C{k}", label=f"{case} {n}$^3$")
        cum = np.concatenate([[0], np.cumsum(0.5 * (tot[1:] + tot[:-1]) * np.diff(b["t"]))])
        a2.plot(b["t"], cum / cum[-1], color=f"C{k}", label=f"{case} {n}$^3$")
    a1.set_xlabel("t [yr]"); a1.set_ylabel(r"instantaneous share at $\alpha<2.2$"); a1.set_ylim(0, 1.05)
    a1.axhline(BUDGET_CRITERION, color="grey", ls=":", lw=1)
    a2.set_xlabel("t [yr]"); a2.set_ylabel("cumulative injected / total"); a2.set_xscale("log")
    for a in (a1, a2):
        a.legend(fontsize=8)
    fig.suptitle(r"DSA injection by test-particle slope $\alpha=(r+2)/(r-1)$", fontsize=11)
    fig.tight_layout()
    fig.savefig(PICS_DIR / "alpha_budget.png", dpi=140)
    plt.close(fig)
    return results


# ------------------------------------------------------------------ step 4
def step4(lines):
    if not DEMO_STATE.exists():
        lines.append("Step 4: demo state missing")
        return None
    z = np.load(DEMO_STATE)
    units = CodeUnits(float(z["code_length_pc"]) * u.pc, float(z["code_mass_msun"]) * u.M_sun,
                      float(z["code_velocity_kms"]) * u.km / u.s)
    st = z["state"]
    n = int(z["num_cells"])
    dx = (float(z["box_size_pc"]) / n * u.pc).to_value(u.cm)
    w = hadronic_target_field(jnp.asarray(st[int(z["density_index"])], dtype=jnp.float64),
                              jnp.asarray(st[int(z["cosmic_ray_e_index"])], dtype=jnp.float64), units, X_H)
    e_cr_cgs = (jnp.asarray(st[int(z["cosmic_ray_e_index"])], dtype=jnp.float64)
                * float((1.0 * units.code_energy / units.code_length ** 3).to_value(u.erg / u.cm ** 3)))
    E_cr_tot = float(jnp.sum(e_cr_cgs)) * dx ** 3
    lines += ["", f"Step 4: item 15 at {n}^3, t = {float(z['t_end_yr']):.0f} yr, float32 run; "
                  f"E_cr = {E_cr_tot:.3e} erg, sum(n_H e_cr dV) = {float(jnp.sum(w)) * dx ** 3:.3e} erg"]

    E = jnp.logspace(-1.7, 3, 120)
    En = np.asarray(E)
    shapes = {r"$\alpha=2.3$ (default)": ProtonMomentumSpectrum(),
              r"$\alpha=2.0$": ProtonMomentumSpectrum(alpha=2.0),
              "IC 443 fit shape (Ackermann+13)": ProtonMomentumSpectrum(**ACKERMANN13_IC443)}
    seds = {lab: np.asarray(pion_decay_sed(w, dx ** 3, E, s, distance_cm=DISTANCE_CM)) for lab, s in shapes.items()}
    i1 = int(np.argmin(np.abs(En - 1.0)))
    for lab, sed in seds.items():
        # item-15 check 2: E^2 dN/dE at 50 MeV vs extrapolation of the 3-30 GeV slope
        sl = np.polyfit(np.log10(En[(En > 3) & (En < 30)]), np.log10(sed[(En > 3) & (En < 30)]), 1)
        i50 = int(np.argmin(np.abs(En - 0.05)))
        supp = 1 - sed[i50] / 10 ** np.polyval(sl, np.log10(En[i50]))
        lines.append(f"  {lab:32s} E^2F(1 GeV, 1.5 kpc) = {sed[i1]:.3e}, peak at {En[np.argmax(sed)]:.2f} GeV, "
                     f"50 MeV suppression vs 3-30 GeV slope {supp:.3f}; IC 443 peak / ours = {IC443_PEAK / sed.max():.1f}")

    # IC on the CMB from K_ep-scaled primary electrons (same alpha=2.3 shape at E >> m_p; no cooling)
    s0 = ProtonMomentumSpectrum()
    amp_per_erg = ERG_TO_GEV / float(proton_unit_kinetic_energy_gev(s0))
    jp_10gev = float(proton_number_per_total_energy(s0, amp_per_erg, jnp.asarray(10.0)))  # 1/GeV per erg of e_cr
    amp_e_per_ev = K_EP * jp_10gev * 1e-9 * E_cr_tot
    elec = GreyElectronSpectrum(amplitude=amp_e_per_ev, e_0=1e10, alpha=2.3, e_cutoff=1e5 * M_P_GEV * 1e9)
    E_ev = En * 1e9
    ic = np.asarray(inverse_compton_photon_spectrum_planck(elec, jnp.asarray(E_ev), T_CMB_K, U_CMB))  # 1/(s eV)
    sed_ic = E_ev ** 2 * ic * 1.602e-12 / (4 * np.pi * DISTANCE_CM ** 2)
    lines.append(f"  IC on CMB (K_ep={K_EP}, alpha 2.3, no cooling): E^2F(1 GeV) = {sed_ic[i1]:.3e}; "
                 f"pion/IC at 1 GeV = {seds[list(shapes)[0]][i1] / sed_ic[i1]:.1f}")

    # map + item-15 check 1: hotspot vs cloud centre
    imap = np.asarray(pion_decay_band_map(w, dx, 1.0, 10.0))
    cc = np.asarray(z["cloud_center_code"])
    dxc = float(z["box_size"]) / n
    ci, cj = int(round(cc[0] / dxc - 0.5)), int(round(cc[1] / dxc - 0.5))
    pi_, pj = np.unravel_index(np.argmax(imap), imap.shape)
    dist = float(np.hypot(pi_ - ci, pj - cj))
    lines.append(f"  1-10 GeV map: brightest pixel {dist:.2f} cells from the cloud centre "
                 f"(cloud radius {1.0 / (float(z['box_size_pc']) / n):.1f} cells)")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5))
    for k, (lab, sed) in enumerate(seds.items()):
        a1.loglog(En, sed, color=f"C{k}", lw=2.2 if k == 0 else 1.3, label=f"pion decay, {lab}")
    ref = seds["IC 443 fit shape (Ackermann+13)"]
    a1.loglog(En, ref * IC443_PEAK / ref.max(), color="k", ls=":", lw=1.3,
              label="IC 443 fit shape at its observed peak (reference)")
    a1.loglog(En, sed_ic, color="C4", ls="--", lw=1.2, label=r"IC on CMB, $K_{ep}=0.01$")
    a1.set_ylim(1e-14, 5e-10)
    a1.set_xlabel(r"$E_\gamma$ [GeV]"); a1.set_ylabel(r"$E^2 dF/dE$ at 1.5 kpc [erg cm$^{-2}$ s$^{-1}$]")
    a1.set_title(f"Item-15 SNR + molecular cloud, {n}$^3$, {float(z['t_end_yr']):.0f} yr", fontsize=10)
    a1.legend(fontsize=7.5)
    L = float(z["box_size_pc"])
    vmax = imap.max()
    im = a2.imshow(imap.T, origin="lower", extent=[0, L, 0, L], norm=LogNorm(vmax * 1e-3, vmax), cmap="magma")
    a2.plot((cc[0] / dxc) * L / n, (cc[1] / dxc) * L / n, "c+", ms=12, mew=1.5, label="cloud centre")
    a2.legend(fontsize=8, loc="upper left")
    a2.set_xlabel("x [pc]"); a2.set_ylabel("y [pc]"); a2.set_title(r"1-10 GeV, $\alpha=2.3$", fontsize=10)
    fig.colorbar(im, ax=a2, label=r"$I$ [erg cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
    fig.tight_layout()
    fig.savefig(PICS_DIR / "item15_sed_and_map.png", dpi=140)
    plt.close(fig)
    return dict(dist=dist, seds=seds)


if __name__ == "__main__":
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["Gamma-ray proxy steps 3-4 (cr_gamma_snr_cloud_demo.py)", ""]
    res3 = step3(lines)
    res4 = step4(lines)
    text = "\n".join(lines)
    print(text)
    (PICS_DIR / "summary.txt").write_text(text + "\n")
