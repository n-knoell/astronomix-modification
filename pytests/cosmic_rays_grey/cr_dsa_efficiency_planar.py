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
``delta(M) eta_KR13(M)``), ``mach="velocity"`` / ``"gas_pressure"`` at the gas-only reading of the
velocity / gas-pressure jump. Since 2026-10-27 (DB8, ``dsa_mach_cr_aware``) DSA inverts the
velocity jump with the CR-loaded Hugoniot, which recovers the true Mach number of a resolved
planar shock, so the hard gates compare with ``mach=DSA_MACH = "true"`` (the "fp" metrics now
equal the "true" ones). Before: the gas-only velocity jump (since DB4, 2026-10-19), which the
extra compression of a CR-loaded shock raises; until then the gas-pressure jump, which the
injected CRs lower (B7.4, finding Y11). ``test_cr_aware_mach_exact`` checks the inversion on
exact steady shocks with upstream CRs (``P_cr / P_th`` 0-2), no simulation: 5e-15.

**Results (2026-10-28, GPU, CR-aware velocity-jump Mach number and area weights).** N = 400 and
1600 agree to 1e-3, so the deviations are not truncation errors.

  ======  ==============  ===============================  ===============================
  M       KR13 vs true    constant vs true                 before DB8, vs true: KR13 / const.
  ======  ==============  ===============================  ===============================
  1.8     none (floor)    -1.5%                            none / +0.3%
  2       threshold       -0.6%                            threshold / +1.5%
  2.25    -5.2%           +0.2%                            -5.1% / +2.5%
  2.5     -3.2%           +0.6%                            -3.0% / +3.3%
  3       -1.6%           +1.3%                            -1.2% / +4.3%
  4       -0.9%           +1.9%                            +1.5% / +5.3%
  5       -1.3%           +2.2%                            +7.3% / +5.8%
  7       +1.5%           +2.4%                            +9.3% / +6.2%
  10      +2.0%           +2.5%                            +10.6% / +6.4%
  >= 15   +2.2 ... +2.3%  +2.6%                            +11.0% / +6.5 ... +6.6%
  ======  ==============  ===============================  ===============================

- Against the true Mach number (= DSA's fixed point): within 2.3% for M >= 3 (gate 5%) and -5.2%
  at M = 2.25 (gate 20%). What remains is the deposit inside the smeared shock (+2.6%, B7.4) and,
  on KR13's steep weak part (``eta ~ M^5.3``), the 1D finder's velocity jump reading slightly
  low (gas-only M_finder 0.3-0.7% below the reference's gas-only reading).
- Before DB8 the gas-only velocity jump of a CR-loaded shock read high (exact 1D: +2.6% in M at
  KR13's strong-shock ``zeta = 0.211``): KR13 +11% at strong shocks, the constant model +0.3% to
  +6.6% (with the gas-pressure Mach number until 2026-10-19: -9% and -1.2% to -2.9%).
- Sub-critical (B8.7): KR13 injects nothing at M = 1.8 (``E_cr / E_th`` ~ 1e-10, the ``e_cr``
  floor). M = 2 is KR13's threshold (1.6e-4 at M = 2, 0 below): the finder reads a hair above 2 in
  some cells, ``E_cr / E_th`` = 1.7e-8, tracked only. The constant model injects at every M above
  ``dsa_mach_min = 1.3`` (DB5).
- The shock speed matches the reference to 5e-4 everywhere.

Gates: hard at N = 1600 against the exact solution at DSA's Mach number (``DSA_MACH``; bounds in
the ``test_cr_dsa_efficiency_planar`` signature); the true-Mach ratios and the finder's gas-only
Mach numbers are tracked in ``baselines.json``. Runtime ~25 min on one 2080 Ti (N = 1600 ~65 s per run).
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
import warnings
from pathlib import Path

# numerics
import numpy as np
from scipy.optimize import brentq

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# baseline registry
from cr_baselines import Baselines

# B7.4 machinery: 1D runs with DSA, exact injection Riemann reference, plateau metrics
from cr_shock_tube_injection import (
    DSA_MACH, FINDER_MACH_KEY, GAMMA, GAMMA_CR, Tube, metrics, reference, run_tube,
)

# astronomix
from astronomix._modules._cosmic_rays_grey.cr_grey_injection import (
    cr_aware_velocity_mach,
    dsa_efficiency_kang_ryu_2013,
)
from astronomix.test_setups.reference_solutions import pfrommer_riemann_solver
from astronomix.test_setups.reference_solutions.pfrommer_riemann_solver import (
    ShockInjection,
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
    ref_fp = reference(tube, zeta, DSA_MACH, x0=X0, t_end=t_end)
    run = run_tube(tube, num_cells, 0.0 if model == "KR13" else CONSTANT_ZETA, box=BOX, x0=X0,
                   t_end=t_end, kr13_scale=1.0 if model == "KR13" else None)
    m = metrics(tube, run, ref_fp, box=BOX, x0=X0)
    x = run["x"]
    a, b = ref_fp["x_contact"], ref_fp["x_shock"]
    window = (x > a + 0.55 * (b - a)) & (x < a + 0.85 * (b - a))
    v_shock = ref_fp["v_shock"] * (1.0 + m["dv_shock"])
    u_2 = run["u"][window].mean()
    e_cr = run["p_cr"][window].mean() / (GAMMA_CR - 1.0)
    return dict(
        eta_eff=e_cr * (v_shock - u_2) / (0.5 * v_shock**3),
        eta_true=eta_of(ref_true),
        eta_fp=eta_of(ref_fp),
        mach_true=ref_true["mach_gas"],
        mach_fp_true=ref_fp["mach_gas"],
        mach_fp_finder=ref_fp[FINDER_MACH_KEY],
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


def _steady_shock(mach, x_cr, zeta):
    """Exact steady planar shock of gas Mach number ``mach`` into ``rho = 1``, ``c_th = 1``,
    ``P_cr = x_cr P_th`` with injection ``zeta`` (code convention, at the true Mach number).
    Returns the compression ``rho_2 / rho_1``."""
    p_th1 = 1.0 / GAMMA
    p_1 = p_th1 * (1.0 + x_cr)
    inj = ShockInjection(zeta, "gas_rh", "true", 1.3)

    def mass_speed(p_star):
        rho_star = pfrommer_riemann_solver._hugoniot_rho_star(
            p_star, 1.0, p_th1, x_cr * p_th1, GAMMA, GAMMA_CR, inj)
        return np.sqrt((p_star - p_1) / (1.0 - 1.0 / rho_star))

    hi = p_1 * (1.0 + 2.0 * mach**2)
    while mass_speed(hi) < mach:
        hi *= 2.0
    p_star = brentq(lambda p: mass_speed(p) - mach, p_1 * (1.0 + 1e-9), hi, xtol=1e-14, rtol=1e-13)
    return pfrommer_riemann_solver._hugoniot_rho_star(
        p_star, 1.0, p_th1, x_cr * p_th1, GAMMA, GAMMA_CR, inj)


def test_cr_aware_mach_exact(max_err=1e-9):
    """DB8: ``cr_aware_velocity_mach`` recovers the true gas Mach number of exact steady shocks
    with upstream CRs and injection from their velocity jump (no simulation). The gas-only
    reading of the same jump is printed for scale."""
    zetas = {"0.1": CONSTANT_ZETA, "0.5": 0.5, "KR13": kr13}
    worst = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x_cr in (0.0, 0.5, 1.0, 2.0):
            for name, zeta in zetas.items():
                for mach in (1.5, 2.25, 3.0, 5.0, 10.0, 100.0):
                    if mach / np.sqrt(1.0 + GAMMA_CR / GAMMA * x_cr) < 1.05:
                        continue  # composite Mach number ~1: no shock
                    x = _steady_shock(mach, x_cr, zeta)
                    w = mach * (1.0 - 1.0 / x)
                    # the finder's gas-only reading of this jump: 2 (M - 1/M) / (gamma + 1) = w
                    a = 0.25 * (GAMMA + 1.0) * w
                    mach_u = a + np.sqrt(a**2 + 1.0)
                    if name == "KR13":
                        eff = lambda m: dsa_efficiency_kang_ryu_2013(m)  # noqa: E731
                    else:
                        eff = lambda m, z=zeta: z * jnp.ones_like(m)  # noqa: E731
                    m_aware, x_aware = cr_aware_velocity_mach(
                        jnp.array([mach_u]), jnp.array([x_cr]), eff, 1.3, GAMMA, GAMMA_CR)
                    err = abs(float(m_aware[0]) / mach - 1.0)
                    worst = max(worst, err, abs(float(x_aware[0]) / x - 1.0))
                    if mach in (3.0, 10.0):
                        print(f"X {x_cr} zeta {name:4s} M {mach:5.1f}: gas-only {mach_u / mach - 1:+.4f},"
                              f" CR-aware {float(m_aware[0]) / mach - 1:+.1e}")
    print(f"worst relative error {worst:.1e}")
    assert worst < max_err


def test_cr_dsa_efficiency_planar(
    max_strong=0.05,
    max_steep=0.20,
    max_subcritical=1e-8,
):
    """B8.1 / B8.7: realized ``eta_eff(M)`` at planar shocks, KR13 and the constant model.

    Hard gates at the finest N against the exact solution at the fixed point of the Mach
    number DSA injects at (``reference(..., DSA_MACH)``: the true one since DB8, 2026-10-27):

    Args:
        max_strong: ``|eta_eff / eta_ref - 1|`` for M >= 3 (plan B8.1).
        max_steep: The same for 2.25 <= M < 3, where KR13 rises as M^5.3 (plan B8.1).
        max_subcritical: KR13 at M = 1.8 and 2: box CR energy per box thermal energy
            (no injection below M = 2; plan B8.7).

    Tracked in ``baselines.json``: every ratio at every N, against both references (true Mach
    number and the finder's fixed point), and the finder's Mach number.
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
                ratio_fp = r["eta_eff"] / r["eta_fp"] - 1.0 if r["eta_fp"] > 0 else np.nan
                ratio_true = r["eta_eff"] / r["eta_true"] - 1.0 if r["eta_true"] > 0 else np.nan
                print(f"{model:8s} N {n:5d} M {mach:6.2f}: eta_eff {r['eta_eff']:.4e}, vs finder-M "
                      f"ref {ratio_fp:+.4f}, vs true-M ref {ratio_true:+.4f}; M_finder "
                      f"{r['mach_finder']:.3f} (finder-M ref {r['mach_fp_finder']:.3f}, true "
                      f"{r['mach_true']:.3f}); dv {r['dv_shock']:+.4f}")
                tag = f"{model}_N{n}_M{mach:g}"
                # KR13 is zero below M = 2 but 1.6e-4 at M = 2: the tube at M = 2 sits on the
                # threshold, where the finder reads a hair below or above it (velocity jump, some
                # cells), so it injects nothing or ~1e-8 of E_th. Tracked only.
                threshold = model == "KR13" and mach == 2.0
                subcritical = not threshold and (r["eta_fp"] == 0.0 or (model == "KR13" and mach < 2.0))
                if not subcritical and not threshold:
                    baselines.close(f"{tag}_eta_over_fp", ratio_fp, abs_tol=0.01)
                    baselines.close(f"{tag}_eta_over_true", ratio_true, abs_tol=0.01)
                    baselines.close(f"{tag}_mach_finder_over_true",
                                    r["mach_finder"] / r["mach_true"] - 1.0, abs_tol=0.005)
                if n != max(RESOLUTIONS):
                    continue
                if threshold:
                    baselines.upper(f"{tag}_ecr_over_eth", r["e_cr_box"] / r["e_box"], floor=1e-6)
                elif subcritical:
                    frac = r["e_cr_box"] / r["e_box"]
                    if not frac < max_subcritical:
                        failures.append(f"{tag}: E_cr/E_th {frac:.2e} without injection")
                elif mach >= 3.0 and not abs(ratio_fp) < max_strong:
                    failures.append(f"{tag}: eta_eff/eta_ref - 1 = {ratio_fp:+.4f}")
                elif mach < 3.0 and not abs(ratio_fp) < max_steep:
                    failures.append(f"{tag}: eta_eff/eta_ref - 1 = {ratio_fp:+.4f}")
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
            for ax, key in ((axes[1], "eta_fp"), (axes[2], "eta_true")):
                ms = np.array([results[(model, n, m)]["mach_true"] for m in MACHS
                               if results[(model, n, m)][key] > 0])
                ratio = np.array([results[(model, n, m)]["eta_eff"] / results[(model, n, m)][key]
                                  for m in MACHS if results[(model, n, m)][key] > 0])
                ax.semilogx(ms, ratio, mk + "-", color=c, mfc=c if filled else "none",
                            label=f"{model}, N={n}")
    axes[0].set(xlabel="true Mach number", ylabel=r"$\eta_{eff}=f_{cr}/(\rho_1 v_s^3/2)$",
                ylim=(1e-5, 0.5), title="realized efficiency at planar shocks (1D)")
    axes[0].legend(fontsize=7)
    for ax, title in ((axes[1], f"vs exact solution at DSA's Mach number ({DSA_MACH})"),
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
