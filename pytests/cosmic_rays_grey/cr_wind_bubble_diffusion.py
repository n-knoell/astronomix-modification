"""
CR wind-blown bubble with CR diffusion: what diffusion does to the DSA-injected CRs.

Physics probe next to the ``cr_wind_bubble`` regression (which advects the CRs with the gas only,
``reduced_streaming_speed = 0``). Same setup (``cr_wind_bubble._run_wind_bubble``), run twice with
DSA (``dsa_efficiency = 0.1``): advective, and with the F_cr relaxation (``diffusive_relaxation``,
``diffusion_coefficient = KAPPA``, ``reduced_streaming_speed = V_RED``).

**Choice of kappa (scan of 2026-11-05, PROGRESS_PHASEB.md DB11).** 1 code unit of kappa here is
3 pc x 100 km/s = 9.3e25 cm^2/s. With the forward shock at v_sh ~ 0.72 code at t_end and
dx = 0.0094, the precursor length kappa / v_sh is 0.15 / 1.4 / 14 / 144 cells for kappa
1e-3 / 1e-2 / 1e-1 / 1. ``KAPPA = 0.1`` (9.3e24 cm^2/s) is the one value whose precursor is
both resolved and smaller than the bubble (R_fs ~ 0.41 = 43 cells): 1e-3 behaves like the
advective run, at 1 the CRs fill the box. ``V_RED = 100`` (10^4 km/s) is 5x the wind speed and
>> kappa / R_fs, so F_cr is relaxed to Fick's law; it makes the run ~3.4x more steps than the
advective one.

Scan results (N 128; E_cr and injected in L_w t_end; injected from
``resolution_study/db11_probe.py``'s accumulator, not measured here):

=========================  ======  ========  ===========
transport                  E_cr    injected  E_cr / inj.
=========================  ======  ========  ===========
advective (v_red 0)        0.0675  0.0711    0.95
kappa 1e-3                 0.0717  0.0727    0.99
kappa 1e-2                 0.0959  0.0748    1.28
kappa 1e-1                 0.1020  0.0778    1.31
kappa 1                    0.0798  0.0796    1.00
undamped wave (v_red 1)    0.1552  0.0768    2.02
=========================  ======  ========  ===========

CRs that diffuse ahead of the forward shock are compressed again when it overtakes them: the
two-fluid picture of re-acceleration. It peaks where the precursor is a few to ~15 cells wide;
without diffusion there is nothing ahead of the shock, and at kappa = 1 most CRs are far from it.
Caveat: the injection efficiency already stands for the full DSA at the shock, so re-compressing
freshly injected CRs at their own shock counts part of that acceleration twice. This probe
shows the size of that effect (+38% E_cr over the advective run at kappa 0.1); it does not
decide whether it is physical.

Checks (calibrated 2026-11-05):

1. Energy budget of the diffusion run: (E_total - E_ambient) / (L_w t_end) = 1.0003 (the CRs
   stay inside the box; at kappa = 1 it drops to 0.995 by boundary outflow).
2. CR precursor ahead of the forward shock, l = int_{R_fs} e_cr dr / e_cr(R_fs) (radial means):
   0.63 of the planar steady-state kappa / v_sh, with v_sh = (3/5) R_fs / t_end (Weaver); the
   spherical divergence and the box edge (20 cells beyond R_fs) shorten it. The advective run's
   l is 1.4 cells, the numerical shock width.
3. E_cr (diffusion) / E_cr (advective) = 1.51, the re-acceleration above (band, not accuracy).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cr_wind_bubble as wb

KAPPA = 0.1
V_RED = 100.0


def _radial_profiles(run, num_cells):
    """Radial bin means (bin width dx) of rho and e_cr inside the box half-width."""
    dx = wb.BOX_SIZE / num_cells
    r = np.asarray(run["radius"]).reshape(-1)
    index = (r / dx).astype(int)
    keep = r < wb.BOX_SIZE / 2
    counts = np.bincount(index[keep])
    centres = (np.arange(counts.size) + 0.5) * dx

    def mean(field):
        return np.bincount(index[keep], weights=np.asarray(field).reshape(-1)[keep]) / np.maximum(counts, 1)

    return centres, mean(run["rho"]), mean(run["e_cr"])


def _precursor(run, num_cells, rho_ambient):
    """Forward-shock radius (outermost bin with rho > 2 rho_0) and precursor length."""
    dx = wb.BOX_SIZE / num_cells
    r, rho, e_cr = _radial_profiles(run, num_cells)
    i = np.where(rho > 2.0 * rho_ambient)[0].max()
    return r[i], float(np.sum(e_cr[i:]) * dx / e_cr[i]), (r, rho, e_cr)


def test_cr_wind_bubble_diffusion(
    energy_budget_tol: float = 0.02,
    precursor_band: tuple = (0.4, 1.0),
    advective_precursor_max_cells: float = 3.0,
    reacceleration_band: tuple = (1.2, 2.0),
):
    """CR wind bubble with CR diffusion vs the advective run.

    Args:
        energy_budget_tol: Max relative error of the diffusion run's (E_total - E_ambient)
            against L_w t_end. Calibrated 2.8e-4.
        precursor_band: Allowed range of the measured precursor length over kappa / v_sh.
            Calibrated 0.63.
        advective_precursor_max_cells: Max precursor length of the advective run in cells (no
            transport ahead of the shock; calibrated 1.4, the shock width).
        reacceleration_band: Allowed range of E_cr (diffusion) / E_cr (advective). Calibrated
            1.51.
    """
    advective = wb._run_wind_bubble(wb.DSA_EFFICIENCY, reduced_streaming_speed=0.0)
    diffusive = wb._run_wind_bubble(wb.DSA_EFFICIENCY, reduced_streaming_speed=V_RED,
                                    diffusion_coefficient=KAPPA)
    for name, run in (("advective", advective), ("diffusive", diffusive)):
        assert not bool(jnp.any(jnp.isnan(run["final_state"]))), f"{name} run produced NaNs."

    n = wb.NUM_CELLS
    dx = wb.BOX_SIZE / n
    cu = wb.CODE_UNITS
    rho_ambient = wb.RHO_AMBIENT.to(cu.code_density).value
    p_ambient = wb.P_AMBIENT.to(cu.code_pressure).value

    # 1. energy budget
    l_wind = 0.5 * wb.M_DOT.to(cu.code_mass / cu.code_time).value * wb.V_INF.to(cu.code_velocity).value ** 2
    e_ambient = p_ambient / (wb.GAMMA - 1.0) * wb.BOX_SIZE**3
    budget = (diffusive["E_total"] - e_ambient) / (l_wind * wb.T_END)
    print(f"diffusive run: (E_total - E_ambient) / (L_w t_end) = {budget:.4f}")
    assert abs(budget - 1.0) < energy_budget_tol, f"energy budget {budget:.4f} off by more than {energy_budget_tol}"

    # 2. precursor
    r_fs, l_diff, prof_diff = _precursor(diffusive, n, rho_ambient)
    r_fs_adv, l_adv, prof_adv = _precursor(advective, n, rho_ambient)
    v_sh = 0.6 * r_fs / wb.T_END
    l_model = KAPPA / v_sh
    ratio = l_diff / l_model
    print(f"forward shock R_fs {r_fs:.3f} (advective {r_fs_adv:.3f}), v_sh {v_sh:.3f}; precursor "
          f"{l_diff / dx:.2f} cells = {ratio:.3f} kappa / v_sh ({l_model / dx:.2f} cells); advective "
          f"{l_adv / dx:.2f} cells")
    assert precursor_band[0] < ratio < precursor_band[1], (
        f"precursor length {ratio:.3f} kappa / v_sh outside {precursor_band}")
    assert l_adv / dx < advective_precursor_max_cells, (
        f"advective run has a {l_adv / dx:.2f}-cell CR precursor (no transport ahead of the shock expected)")

    # 3. re-acceleration
    gain = diffusive["E_cr"] / advective["E_cr"]
    print(f"E_cr diffusive / advective = {gain:.3f} (E_cr / (L_w t_end) {diffusive['E_cr'] / (l_wind * wb.T_END):.4f} "
          f"/ {advective['E_cr'] / (l_wind * wb.T_END):.4f})")
    assert reacceleration_band[0] < gain < reacceleration_band[1], (
        f"E_cr diffusive / advective {gain:.3f} outside {reacceleration_band}")

    # figure: P_cr and rho around the forward shock, with the planar precursor exp(-(r - R_fs) v_sh / kappa)
    fig, (ax_cr, ax_rho) = plt.subplots(1, 2, figsize=(12, 4.6))
    for label, (r, rho, e_cr), r_shock, colour in (("advective", prof_adv, r_fs_adv, "#2a2a2a"),
                                                   (f"kappa {KAPPA:g}", prof_diff, r_fs, "#2171b5")):
        sel = r > 0.6 * r_shock
        ax_cr.plot(r[sel], e_cr[sel] / 3.0 / p_ambient, color=colour, lw=2, label=label)
        ax_rho.plot(r[sel], rho[sel] / rho_ambient, color=colour, lw=2, label=label)
    r, _, e_cr = prof_diff
    ahead = r >= r_fs
    i = np.argmax(ahead)
    ax_cr.plot(r[ahead], e_cr[i] / 3.0 / p_ambient * np.exp(-(r[ahead] - r_fs) / l_model), color="#2171b5",
               lw=1, ls="--", label=r"planar $\exp(-(r - R_{\rm fs})\, v_{\rm sh} / \kappa)$")
    for ax in (ax_cr, ax_rho):
        ax.axvline(r_fs, color="#999999", lw=1, ls=":")
        ax.set_xlabel("r [code length]")
        ax.legend(frameon=False, fontsize=8)
    ax_cr.set_yscale("log")
    ax_cr.set_ylim(1e-3, None)
    ax_cr.set_ylabel(r"$P_{\rm cr} / p_0$ (radial mean)")
    ax_cr.set_title(f"CR pressure at t_end: precursor {ratio:.2f} kappa / v_sh")
    ax_rho.set_ylabel(r"$\rho / \rho_0$ (radial mean)")
    ax_rho.set_title(f"gas density; E_cr diffusive / advective {gain:.2f}")
    fig.tight_layout()
    pics_dir = Path(__file__).resolve().parent / "pics" / "10_wind_bubble"
    pics_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(pics_dir / "cr_wind_bubble_diffusion.svg", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    test_cr_wind_bubble_diffusion()
