"""
Realized DSA efficiency ``eta_eff(M)`` at planar shocks: Phase B test plan B8.1 (and B8.7's
sub-critical points), ``astronomix/_modules/_cosmic_rays_grey/astronomix_CR_phaseB_test_plan.md``
Sec. 8.4.

**What it measures.** Item 8's quantity, the efficiency the code realizes as a function of the
Mach number, where the Mach number is known exactly. Each tube starts from the gas-only RH state
of Mach number M (``rho = 1``, ``c = 1`` ahead; box [0, 1], diaphragm 0.25, run until the shock
has travelled 0.6), so without injection it is a single shock; with injection a weak left wave
and a contact appear, which the exact solution includes. M = 1.8 ... 100, KR13
(``dsa_efficiency_mach_scale = 1``) and the constant model (``zeta = 0.1``), N = 400 and 1600,
default scheme, ``v_red = 0``, float64. Measured on the shocked plateau (55-85% from contact to
shock) in the shock frame: ``eta_eff = e_cr,2 (v_s - u_2) / (rho_1 v_s^3 / 2)``, with ``v_s`` from
the snapshot fit.

**Reference.** ``pfrommer_riemann_solver.py`` with ``injection=ShockInjection(zeta(M),
"gas_rh", mach, 1.3)`` (B7.4) solves the fixed point ``zeta = eta(M(zeta))`` together with the
shock jump: ``mach="true"`` evaluates KR13 at the true Mach number (the intended
``delta(M) eta_KR13(M)``), ``mach="gas_pressure"`` at the Mach number of the gas-pressure jump,
which the injected CRs lower and which the finder reads (B7.4, finding Y11).

**Results (2026-10-14, GPU).** N = 400 and 1600 agree to 1e-3, so the deviations are not
truncation errors.

  ======  ==============  ==============  ===============================
  M       KR13 vs gas-p.  KR13 vs true    constant vs gas-p. / true
  ======  ==============  ==============  ===============================
  1.8, 2  no injection    (E_cr ~ 1e-10   -1.4% / -1.9% (1.8)
          (floor)         of E_th)
  2.25    -5.6%           -5.7%           +0.3% / -1.2%
  2.5     -3.7%           -3.8%           +0.7% / -1.2%
  3       -2.2%           -2.6%           +1.3% / -1.4%
  4       -1.3%           -3.9%           +1.9% / -1.9%
  5       -1.0%           -10.3%          +2.2% / -2.2%
  7       +1.9%           -7.7%           +2.5% / -2.5%
  10      +2.5%           -8.6%           +2.6% / -2.7%
  >= 15   +2.8 ... +2.9%  -8.8%           +2.7% / -2.9%
  ======  ==============  ==============  ===============================

- Against the gas-pressure fixed point: within 3% for M >= 3 (gate 5%) and -5.6% at M = 2.25
  (gate 20%). The net of two parts that partly cancel: an excess from the deposit inside the
  smeared shock (B7.4: CRs injected there gain adiabatic energy) and the 1D finder's Mach number below
  the fixed point: -0.45% at M = 2.25 (2.240), -1.0% (KR13) / -0.5% (constant) at M = 10, -0.8%
  / -0.6% at M = 100 (the flux at saturated ``delta`` goes as ``M^3``, so this is part of the
  strong-shock budget too). On KR13's steep weak part
  (``eta ~ M^5.3``) that becomes -5.6% at M 2.25. The plan's "< 0.1%" finder accuracy in 1D
  (Dubois et al. 2019, 4-cell sampling) is not reached.
- Against the true Mach number (Y11, DB8 decided 2026-10-14: accepted, finder unchanged): KR13 -9% at strong shocks (``zeta = 0.211`` lowers the finder's
  Mach number by 3.8%, which also drops the shock speed); at M = 5 the fixed point lands on KR13's
  steep weak piece (``M_finder`` 4.89 < 5), -10%. The constant model is -1.2% to -2.9%.
- Sub-critical (B8.7): KR13 injects nothing at M = 1.8 and 2 (``E_cr / E_th`` ~ 1e-10, the
  ``e_cr`` floor). The constant model injects at every M above ``dsa_mach_min = 1.3`` (DB5).
- The shock speed matches the fixed point to 5e-4 everywhere.

Gates: hard at N = 1600 against the gas-pressure fixed point (bounds in the
``test_cr_dsa_efficiency_planar`` signature); the true-Mach ratios and the finder's Mach numbers
are tracked in ``baselines.json``. Runtime ~40 min on one 2080 Ti (N = 1600 ~65 s per run).
Raw values: ``/export/scratch/nknoell/phaseB/b8_1/b8_1.npz``.

See ``PROGRESS_PHASEB.md`` in the module directory.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import os
from pathlib import Path

# numerics
import numpy as np

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# baseline registry
from cr_baselines import Baselines

# B7.4 machinery: 1D runs with DSA, exact injection Riemann reference, plateau metrics
from cr_shock_tube_injection import GAMMA, GAMMA_CR, Tube, metrics, reference, run_tube

# astronomix
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import dsa_efficiency_kang_ryu_2013
from astronomix.test_setups.reference_solutions.pfrommer_riemann_solver import (
    thermalization_efficiency,
)

MACHS = (1.8, 2.0, 2.25, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 30.0, 100.0)
RESOLUTIONS = tuple(int(n) for n in os.environ.get("CR_B81_RESOLUTIONS", "400,1600").split(","))
CONSTANT_ZETA = 0.1  # the item-7 constant model (CosmicRayGreyParams.dsa_efficiency default)
BOX = 1.0
X0 = 0.25
TRAVEL = 0.6  # the shock travels 0.6 of the box by t_end
PICS_DIR = Path(__file__).resolve().parent / "pics" / "08_dsa_mach_dependence"
RAW_DIR = Path(os.environ.get("CR_PHASEB_RAW", "/export/scratch/nknoell/phaseB")) / "b8_1"


def kr13(mach):
    """numpy copy of ``dsa_efficiency_kang_ryu_2013`` (checked against it in the test)."""
    ms = float(mach)
    if ms < 2.0:
        return 0.0
    if ms <= 5.0:
        return -5.95e-4 + 1.88e-5 * ms**5.334
    if ms <= 15.0:
        b = (-2.87, 9.67, -8.88, 1.94, 0.18)
        return sum(bn * (ms - 1.0) ** n for n, bn in enumerate(b)) / ms**4
    return 0.211


def rh_tube(mach):
    """Gas-only RH post-shock state | ambient (``rho = 1``, ``c = 1``): without injection the
    Riemann problem is a single right-going shock of Mach number ``mach``."""
    rho1, p1 = 1.0, 1.0 / GAMMA
    x = (GAMMA + 1.0) * mach**2 / ((GAMMA - 1.0) * mach**2 + 2.0)
    p2 = p1 * (2.0 * GAMMA * mach**2 - (GAMMA - 1.0)) / (GAMMA + 1.0)
    return Tube(left=(x * rho1, mach * (1.0 - 1.0 / x), p2, 0.0), right=(rho1, 0.0, p1, 0.0))


def eta_of(ref):
    """Reference efficiency: injected CR energy flux per incoming kinetic flux."""
    return ref["f_inj"] / (0.5 * ref["v_shock"] ** 3)


def measure(mach, num_cells, model):
    """One run; returns the measured and reference efficiencies and diagnostics."""
    tube = rh_tube(mach)
    t_end = TRAVEL / mach
    zeta = kr13 if model == "KR13" else CONSTANT_ZETA
    ref_true = reference(tube, zeta, "true", x0=X0, t_end=t_end)
    ref_gp = reference(tube, zeta, "gas_pressure", x0=X0, t_end=t_end)
    run = run_tube(tube, num_cells, 0.0 if model == "KR13" else CONSTANT_ZETA, box=BOX, x0=X0,
                   t_end=t_end, kr13_scale=1.0 if model == "KR13" else None)
    m = metrics(tube, run, ref_gp, box=BOX, x0=X0)
    x = run["x"]
    a, b = ref_gp["x_contact"], ref_gp["x_shock"]
    window = (x > a + 0.55 * (b - a)) & (x < a + 0.85 * (b - a))
    v_shock = ref_gp["v_shock"] * (1.0 + m["dv_shock"])
    u_2 = run["u"][window].mean()
    e_cr = run["p_cr"][window].mean() / (GAMMA_CR - 1.0)
    return dict(
        eta_eff=e_cr * (v_shock - u_2) / (0.5 * v_shock**3),
        eta_true=eta_of(ref_true),
        eta_gp=eta_of(ref_gp),
        mach_true=ref_true["mach_gas"],
        mach_gp_true=ref_gp["mach_gas"],
        mach_gp_finder=ref_gp["mach_gas_pressure"],
        mach_finder=run["mach_finder"],
        dv_shock=m["dv_shock"],
        dP_th=m["dP_th"],
        drho=m["drho"],
        e_cr_box=float(run["e_cr_t"][-1].sum() * (x[1] - x[0])),
        e_box=float(np.sum(run["p_th"] / (GAMMA - 1.0)) * (x[1] - x[0])),
    )


def test_kr13_numpy_copy():
    """The reference's ``kr13`` equals ``dsa_efficiency_kang_ryu_2013`` (no simulation)."""
    ms = np.concatenate([np.linspace(1.0, 20.0, 1901), [30.0, 100.0, 1e3]])
    jax_values = np.asarray(dsa_efficiency_kang_ryu_2013(jnp.asarray(ms)))
    numpy_values = np.array([kr13(m) for m in ms])
    assert np.max(np.abs(jax_values - numpy_values)) < 1e-6


def test_cr_dsa_efficiency_planar(
    max_strong=0.05,
    max_steep=0.20,
    max_subcritical=1e-8,
):
    """B8.1 / B8.7: realized ``eta_eff(M)`` at planar shocks, KR13 and the constant model.

    Hard gates at the finest N against the exact solution at the gas-pressure fixed point
    (``reference(..., "gas_pressure")``, ``zeta = eta(M_finder)``):

    Args:
        max_strong: ``|eta_eff / eta_ref - 1|`` for M >= 3 (plan B8.1).
        max_steep: The same for 2.25 <= M < 3, where KR13 rises as M^5.3 (plan B8.1).
        max_subcritical: KR13 at M = 1.8 and 2: box CR energy per box thermal energy
            (no injection below M = 2; plan B8.7).

    Tracked in ``baselines.json``: every ratio at every N, against both references (true Mach
    number and gas-pressure fixed point), and the finder's Mach number.
    """
    baselines = Baselines("cr_dsa_efficiency_planar.py::test_cr_dsa_efficiency_planar")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = []
    for model in ("KR13", "constant"):
        for n in RESOLUTIONS:
            for mach in MACHS:
                r = measure(mach, n, model)
                results[(model, n, mach)] = r
                ratio_gp = r["eta_eff"] / r["eta_gp"] - 1.0 if r["eta_gp"] > 0 else np.nan
                ratio_true = r["eta_eff"] / r["eta_true"] - 1.0 if r["eta_true"] > 0 else np.nan
                print(f"{model:8s} N {n:5d} M {mach:6.2f}: eta_eff {r['eta_eff']:.4e}, vs gas-p. "
                      f"ref {ratio_gp:+.4f}, vs true-M ref {ratio_true:+.4f}; M_finder "
                      f"{r['mach_finder']:.3f} (gas-p. ref {r['mach_gp_finder']:.3f}, true "
                      f"{r['mach_true']:.3f}); dv {r['dv_shock']:+.4f}")
                tag = f"{model}_N{n}_M{mach:g}"
                if r["eta_gp"] > 0:
                    baselines.close(f"{tag}_eta_over_gp", ratio_gp, abs_tol=0.01)
                    baselines.close(f"{tag}_eta_over_true", ratio_true, abs_tol=0.01)
                    baselines.close(f"{tag}_mach_finder_over_true",
                                    r["mach_finder"] / r["mach_true"] - 1.0, abs_tol=0.005)
                if n != max(RESOLUTIONS):
                    continue
                if r["eta_gp"] == 0.0:
                    frac = r["e_cr_box"] / r["e_box"]
                    if not frac < max_subcritical:
                        failures.append(f"{tag}: E_cr/E_th {frac:.2e} without injection")
                elif mach >= 3.0 and not abs(ratio_gp) < max_strong:
                    failures.append(f"{tag}: eta_eff/eta_ref - 1 = {ratio_gp:+.4f}")
                elif mach < 3.0 and not abs(ratio_gp) < max_steep:
                    failures.append(f"{tag}: eta_eff/eta_ref - 1 = {ratio_gp:+.4f}")
    np.savez(RAW_DIR / "b8_1.npz", **{
        f"{model}_N{n}_M{mach:g}_{k}": v for (model, n, mach), r in results.items()
        for k, v in r.items()})
    _plot(results)
    baselines.check()
    assert not failures, "\n".join(failures)


def _plot(results):
    """``eta_eff(M)`` over the model curves, and the ratios to both references."""
    n_fine = max(RESOLUTIONS)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    m_grid = np.geomspace(1.5, 150.0, 400)
    delta = thermalization_efficiency(m_grid, GAMMA)
    axes[0].loglog(m_grid, delta * np.array([kr13(m) for m in m_grid]), "k-",
                   label=r"$\delta(M)\,\eta_{KR13}(M)$ (code convention)")
    axes[0].loglog(m_grid, [kr13(m) for m in m_grid], "k:", lw=0.8,
                   label=r"$\eta_{KR13}(M)$ (KR13's kinetic-flux definition)")
    axes[0].loglog(m_grid, CONSTANT_ZETA * delta, "C1-", lw=1,
                   label=rf"$\delta(M)\,\zeta$, constant $\zeta$={CONSTANT_ZETA}")
    for model, c in (("KR13", "C0"), ("constant", "C1")):
        for n, mk in zip(RESOLUTIONS, ("o", "s", "^")):
            pts = [(results[(model, n, m)]["mach_true"], results[(model, n, m)]["mach_finder"],
                    results[(model, n, m)]["eta_eff"]) for m in MACHS]
            pts = np.array([p for p in pts if p[2] > 1e-6])
            filled = n == n_fine
            axes[0].loglog(pts[:, 0], pts[:, 2], mk, color=c, mfc=c if filled else "none",
                           label=f"{model}, N={n}, at true M")
            if filled:
                axes[0].loglog(pts[:, 1], pts[:, 2], "x", color=c, ms=5,
                               label=f"{model}, N={n}, at finder M")
            for ax, key in ((axes[1], "eta_gp"), (axes[2], "eta_true")):
                ms = np.array([results[(model, n, m)]["mach_true"] for m in MACHS
                               if results[(model, n, m)][key] > 0])
                ratio = np.array([results[(model, n, m)]["eta_eff"] / results[(model, n, m)][key]
                                  for m in MACHS if results[(model, n, m)][key] > 0])
                ax.semilogx(ms, ratio, mk + "-", color=c, mfc=c if filled else "none",
                            label=f"{model}, N={n}")
    axes[0].set(xlabel="true Mach number", ylabel=r"$\eta_{eff}=f_{cr}/(\rho_1 v_s^3/2)$",
                ylim=(1e-5, 0.5), title="realized efficiency at planar shocks (1D)")
    axes[0].legend(fontsize=7)
    for ax, title in ((axes[1], "vs exact solution, finder Mach from the gas-pressure jump"),
                      (axes[2], "vs exact solution at the true Mach number")):
        ax.axhline(1.0, color="grey", lw=0.8)
        ax.axhspan(0.95, 1.05, color="grey", alpha=0.15)
        ax.set(xlabel="true Mach number", ylabel=r"$\eta_{eff}/\eta_{ref}$", title=title,
               ylim=(0.75, 1.1))
        ax.legend(fontsize=7)
    fig.suptitle("B8.1: DSA efficiency at planar shocks (RH-state tubes, default scheme, "
                 r"$v_{red}=0$)")
    fig.tight_layout()
    PICS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PICS_DIR / "cr_dsa_efficiency_planar.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    test_kr13_numpy_copy()
    test_cr_dsa_efficiency_planar()
