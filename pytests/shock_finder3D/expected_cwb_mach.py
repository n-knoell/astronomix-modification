"""Theoretical expected shock Mach number for the CWB test, vs. physical/
orbital parameters.

Standalone analytic companion to ``_cwb_setup.py`` -- deliberately does not
import ``astronomix`` (no JAX/GPU dependency, so it runs anywhere, instantly,
without the ``autocvd`` GPU-pinning dance the actual simulations need). It
answers the question FIXES_TODO.md's item 1b investigation keeps running
into: "the shock finder measures some Mach number on the real simulation --
but what Mach number should a real colliding-wind binary with these
parameters actually produce?" Round 13 re-ran this comparison against the
real N=512, a=50 au stationary run (``figures/cwb_shocked_cells_{2d,3d}_512_stat.png``)
and found the same ~11x gap as every earlier resolution (round 12's N=64):
increasing resolution 8x and fixing the separation bug (see below) did not
close it. See ``FIXES_TODO.md`` round 13 for the full writeup.

Two independent estimates are computed, both as functions of the winds'
physical (mass-loss rate, terminal velocity) and orbital (separation,
eccentricity, stellar mass) parameters:

1. **Ambient-comparison Mach** (``wind_mach_vs_ambient``): M = v_wind /
   c_s(ambient). This is exactly the estimate ``_cwb_setup.py``'s own module
   docstring uses to justify "Mach number ~ 100" -- it only needs the wind's
   terminal velocity and the ambient medium's temperature, both of which are
   fixed, verified inputs to the real simulation. It is a *lower-bound*
   sanity check, not a model of the wind-collision shock itself: it says
   nothing about how close to the stars the shock actually sits, or how cold
   the *pre-shock wind* (as opposed to the pre-shock ambient medium) is.

2. **Adiabatic-cooled-wind Mach** (``adiabatic_cooled_wind_mach``): the wind
   that actually crosses the wind-wind collision shock is not ambient
   material -- it is each star's own freely-expanding wind, which cools as
   it expands (``T ~ rho**(gamma-1) ~ r**(-2*(gamma-1))`` for adiabatic,
   constant-velocity free expansion) all the way from near the stellar
   surface out to the stagnation point set by ram-pressure balance between
   the two winds (``stagnation_point_fractions``, the standard Stevens,
   Blondin & Pollock 1992 / Eichler & Usov 1993 result). Because this cooling
   spans a large factor in radius, the *true* pre-shock wind sound speed can
   be much lower (and the true Mach number much higher) than estimate 1 --
   using the ambient temperature as a stand-in for the wind's own temperature
   badly overestimates the wind's true pre-shock sound speed. This estimate
   needs two extra reference values the codebase does not itself specify (a
   stellar radius and a wind base temperature) -- these are set once, below,
   to typical O-star literature values and clearly flagged; the point is the
   qualitative scaling and order of magnitude, not a precision calculation
   for these two specific stars.

Both estimates are evaluated at ``_cwb_setup.py``'s baseline parameters and
swept over separation, terminal velocity, ambient temperature, mass-loss
ratio, and orbital eccentricity/phase. A third, purely numerical, quantity
is also computed: how many grid cells actually sit between the edge of the
wind-injection sphere and the stagnation-point shock, at the resolutions
FIXES_TODO.md's item 4 investigation actually tested (N=64/128/256) -- this
connects the "why is the measured Mach so much lower than expected" question
directly to the amount of grid available to resolve the wind's cooling, which
is the basis for this file's proposed fix (see the module docstring in
``FIXES_TODO.md``'s next entry, and the printed summary at the bottom of
``main()``).
"""

# general
from pathlib import Path

# numerics
import numpy as np

# units
import astropy.constants as const
import astropy.units as u

# plotting
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

GAMMA = 5.0 / 3.0

# ---- baseline physical parameters (mirrors _cwb_setup.py exactly) ----
M1_BASELINE = 50 * u.M_sun
M2_BASELINE = 40 * u.M_sun
MDOT1_BASELINE = 6.5e-7 * u.M_sun / u.yr
MDOT2_BASELINE = 6.5e-8 * u.M_sun / u.yr
VINF1_BASELINE = 2200 * u.km / u.s
VINF2_BASELINE = 1850 * u.km / u.s
ECCENTRICITY_BASELINE = 0.0

RHO_AMBIENT_BASELINE = 2 * const.m_p / u.cm**3
T_AMBIENT_BASELINE = 1.5e4 * u.K
P_AMBIENT_BASELINE = (2 / u.cm**3) * const.k_B * T_AMBIENT_BASELINE

# _cwb_setup.py's actual, currently-simulated separation. (Round 12 found
# this used to be silently 25x smaller than documented -- SEPARATION_PHYSICAL
# = 20 au was set but never wired into code_length, so the setup actually ran
# at 0.8 au. Fixed directly in _cwb_setup.py since then: code_length is now
# derived as sep_in_au / SEPARATION, so the two agree exactly. Nothing left
# to reconcile here anymore -- kept as one constant.)
SEPARATION_BASELINE = 50 * u.au
SEPARATION_CODE = 0.4
BOX_SIZE_CODE = 1.0

# Reference values the *codebase* does not specify anywhere, needed only for
# the illustrative adiabatic-cooling estimate (see module docstring, estimate
# 2). Typical O-star literature values, not tuned to match anything.
R_STAR_ASSUMED = 20 * u.R_sun
T_WIND_BASE_ASSUMED = 3.5e4 * u.K

# The shock finder's default (Tier-1, mach_sampling_steps=1 -- _cwb_setup.py
# does not opt into any sampling improvement) global max Mach measured on the
# real N=512, a=50 au stationary CWB run (FIXES_TODO.md round 13), read off
# figures/cwb_shocked_cells_3d_512_stat.png's colorbar (no raw data was
# saved, so this is a calibrated-pixel reading, not an exact value -- see
# round 13's writeup for the calibration method and error bar). Essentially
# unchanged from round 12's N=64, a=0.8 au (bugged separation) measurement of
# ~10.3, despite 8x more grid resolution and the separation bug fix.
SIMULATED_MAX_MACH = 10.7


def stagnation_point_fractions(mdot1, v1, mdot2, v2):
    """Ram-pressure-balance stagnation point between two colliding winds.

    Standard result (Stevens, Blondin & Pollock 1992; Eichler & Usov 1993):
    with rho_i(r) = mdot_i / (4 pi r_i**2 v_i) for each freely-expanding
    wind, ram-pressure balance rho1 v1**2 = rho2 v2**2 at the stagnation
    point gives r1/r2 = sqrt(mdot2 v2 / (mdot1 v1)).

    Args:
        mdot1: Star 1's mass-loss rate.
        v1: Star 1's terminal wind velocity.
        mdot2: Star 2's mass-loss rate.
        v2: Star 2's terminal wind velocity.

    Returns:
        ``(frac1, frac2, eta)``: the stagnation point's distance from star 1
        and star 2 as a fraction of the separation (``frac1 + frac2 == 1``),
        and the wind momentum ratio ``eta = mdot2 v2 / (mdot1 v1)``.
    """
    eta = ((mdot2 * v2) / (mdot1 * v1)).decompose().value
    sqrt_eta = np.sqrt(eta)
    frac1 = 1.0 / (1.0 + sqrt_eta)
    frac2 = sqrt_eta / (1.0 + sqrt_eta)
    return frac1, frac2, eta


def ambient_sound_speed(rho_ambient, p_ambient, gamma=GAMMA):
    """Adiabatic sound speed of the ambient medium."""
    return np.sqrt(gamma * p_ambient / rho_ambient).to(u.km / u.s)


def wind_mach_vs_ambient(v_wind, c_s_ambient):
    """Estimate 1: wind terminal velocity over the ambient sound speed."""
    return (v_wind / c_s_ambient).decompose().value


def adiabatic_cooled_wind_mach(v_wind, r_shock, r0=R_STAR_ASSUMED, t0=T_WIND_BASE_ASSUMED, gamma=GAMMA):
    """Estimate 2: Mach number of the wind at the shock, cooled adiabatically.

    A freely-expanding wind at constant velocity has rho(r) ~ r**-2; if it
    carries no further heating or cooling past some base radius/temperature
    (r0, T0), adiabatic (constant-entropy) expansion gives
    T(r) = T0 * (r / r0)**(-2*(gamma - 1)).

    Args:
        v_wind: The wind's terminal velocity.
        r_shock: Distance from the star to the wind-collision shock.
        r0: Reference (wind base) radius -- see ``R_STAR_ASSUMED``.
        t0: Reference (wind base) temperature -- see ``T_WIND_BASE_ASSUMED``.
        gamma: Adiabatic index.

    Returns:
        The dimensionless Mach number ``v_wind / c_s(r_shock)``.
    """
    temperature_at_shock = t0 * (r_shock / r0).decompose().value ** (-2.0 * (gamma - 1.0))
    c_s_wind = np.sqrt(gamma * const.k_B * temperature_at_shock / const.m_p).to(u.km / u.s)
    return (v_wind / c_s_wind).decompose().value


def cells_between_injection_and_shock(num_cells, r_shock_code, box_size_code=BOX_SIZE_CODE):
    """Grid cells between the wind-injection sphere's edge and the shock.

    ``_cwb_setup.py`` sets ``num_injection_cells = num_cells // 32`` (a fixed
    ~1/32-of-the-box injection radius, independent of resolution), so this
    grows only because ``dx`` itself shrinks as ``num_cells`` grows -- it is
    *not* a "code units don't matter" invariant the way FIXES_TODO.md item 4
    found for the injection-region blow-up bug (that ratio held code_length,
    N *and* num_injection_cells/N all fixed; this one varies N with
    num_injection_cells/N pinned).

    Args:
        num_cells: Grid resolution per dimension.
        r_shock_code: Distance from the injecting star to the shock, in
            dimensionless code-length units.
        box_size_code: The box size in code-length units.

    Returns:
        The number of grid cells separating the injection sphere's edge from
        the shock (can be negative if the shock sits inside the injection
        sphere, which does not happen for any case considered here).
    """
    num_injection_cells = num_cells // 32
    injection_radius_code = num_injection_cells * box_size_code / num_cells
    dx_code = box_size_code / num_cells
    return (r_shock_code - injection_radius_code) / dx_code


def orbital_velocity(m1, m2, separation):
    """Relative Keplerian orbital velocity at a given instantaneous separation."""
    return np.sqrt(const.G * (m1 + m2) / separation).to(u.km / u.s)


def separation_at_true_anomaly(a, eccentricity, true_anomaly):
    """Instantaneous binary separation at a given true anomaly (Kepler orbit)."""
    return a * (1.0 - eccentricity**2) / (1.0 + eccentricity * np.cos(true_anomaly))


def main():
    # ---------------------------------------------------------------
    # ---- baseline summary ----
    # ---------------------------------------------------------------
    frac1, frac2, eta = stagnation_point_fractions(
        MDOT1_BASELINE, VINF1_BASELINE, MDOT2_BASELINE, VINF2_BASELINE
    )
    c_s_ambient = ambient_sound_speed(RHO_AMBIENT_BASELINE, P_AMBIENT_BASELINE)
    mach1_ambient = wind_mach_vs_ambient(VINF1_BASELINE, c_s_ambient)
    mach2_ambient = wind_mach_vs_ambient(VINF2_BASELINE, c_s_ambient)

    r1 = frac1 * SEPARATION_BASELINE
    r2 = frac2 * SEPARATION_BASELINE

    mach1_adiabatic = adiabatic_cooled_wind_mach(VINF1_BASELINE, r1)
    mach2_adiabatic = adiabatic_cooled_wind_mach(VINF2_BASELINE, r2)

    print("=" * 72)
    print("CWB baseline (_cwb_setup.py's current parameters, a=50 au)")
    print("=" * 72)
    print(f"wind momentum ratio eta = mdot2 v2 / (mdot1 v1) = {eta:.4f}")
    print(f"stagnation point: {frac1:.4f} of separation from star 1, "
          f"{frac2:.4f} from star 2")
    print(f"ambient sound speed c_s = {c_s_ambient:.3f}")
    print()
    print(f"{'':>28s}{'star 1':>16s}{'star 2':>16s}")
    print(f"{'v_inf':>28s}{VINF1_BASELINE:>16}{VINF2_BASELINE:>16}")
    print(f"{'distance to shock':>28s}{r1:>16.3f}{r2:>16.3f}")
    print(f"{'ambient-comparison Mach':>28s}{mach1_ambient:>16.1f}{mach2_ambient:>16.1f}")
    print(f"{'adiabatic-cooled Mach':>28s}{mach1_adiabatic:>16.1f}{mach2_adiabatic:>16.1f}")
    print()

    r2_code = frac2 * SEPARATION_CODE
    print("grid cells between the (star-2, weaker-wind, closer stagnation "
          "point) injection sphere's edge and the shock, vs. resolution "
          "(this ratio is invariant to the a=50 au fix -- see "
          "cells_between_injection_and_shock's docstring):")
    for num_cells in (64, 128, 256, 512, 1024):
        cells = cells_between_injection_and_shock(num_cells, r2_code)
        print(f"  N={num_cells:>5d}: {cells:6.2f} cells")
    print()
    print(f"shock finder measurement on the real N=512, a=50 au run "
          f"(FIXES_TODO.md round 13): global max Mach ~{SIMULATED_MAX_MACH} "
          f"(pixel-calibrated from cwb_shocked_cells_3d_512_stat.png -- no "
          f"raw data saved)")
    print(f"vs. ambient-comparison floor: ~{mach2_ambient:.0f} (star 2, the "
          f"closer/measured shock) -- sim is {mach2_ambient / SIMULATED_MAX_MACH:.0f}x "
          f"below even this conservative estimate, let alone the adiabatic "
          f"ceiling of ~{mach2_adiabatic:.0f}")
    print("=" * 72)

    # ---------------------------------------------------------------
    # ---- plots ----
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0): Mach vs ambient temperature
    ax = axes[0, 0]
    t_ambient_range = np.linspace(1e3, 5e4, 200) * u.K
    c_s_range = np.array([
        ambient_sound_speed(RHO_AMBIENT_BASELINE, (2 / u.cm**3) * const.k_B * t).value
        for t in t_ambient_range
    ]) * u.km / u.s
    ax.plot(t_ambient_range, wind_mach_vs_ambient(VINF1_BASELINE, c_s_range), label="star 1 (v=2200 km/s)")
    ax.plot(t_ambient_range, wind_mach_vs_ambient(VINF2_BASELINE, c_s_range), label="star 2 (v=1850 km/s)")
    ax.axvline(T_AMBIENT_BASELINE.value, color="grey", ls=":", label="baseline T_ambient")
    ax.axhline(SIMULATED_MAX_MACH, color="red", ls="--", label="sim max (N=512)")
    ax.set_xlabel("ambient temperature [K]")
    ax.set_ylabel("ambient-comparison Mach")
    ax.set_title("estimate 1: v_wind / c_s(ambient)")
    ax.legend(fontsize=7)
    ax.grid(True, ls=":", alpha=0.6)

    # (0,1): Mach vs wind terminal velocity
    ax = axes[0, 1]
    v_range = np.linspace(500, 3500, 200) * u.km / u.s
    ax.plot(v_range, wind_mach_vs_ambient(v_range, c_s_ambient), color="tab:purple")
    ax.axvline(VINF1_BASELINE.value, color="grey", ls=":", label="star 1 baseline")
    ax.axvline(VINF2_BASELINE.value, color="grey", ls="--", label="star 2 baseline")
    ax.axhline(SIMULATED_MAX_MACH, color="red", ls="--", label="sim max (N=512)")
    ax.set_xlabel("wind terminal velocity [km/s]")
    ax.set_ylabel("ambient-comparison Mach")
    ax.set_title("estimate 1: v_wind / c_s(ambient)")
    ax.legend(fontsize=7)
    ax.grid(True, ls=":", alpha=0.6)

    # (0,2): stagnation point fraction vs mass-loss-rate ratio
    ax = axes[0, 2]
    ratio_range = np.geomspace(0.01, 10, 200)
    frac2_range = []
    for ratio in ratio_range:
        _, f2, _ = stagnation_point_fractions(
            MDOT1_BASELINE, VINF1_BASELINE, ratio * MDOT1_BASELINE, VINF2_BASELINE
        )
        frac2_range.append(f2)
    ax.semilogx(ratio_range, frac2_range, color="tab:green")
    ax.axvline((MDOT2_BASELINE / MDOT1_BASELINE).decompose().value, color="grey", ls=":", label="baseline")
    ax.set_xlabel(r"$\dot{M}_2 / \dot{M}_1$")
    ax.set_ylabel("stagnation point: fraction of $a$ from star 2")
    ax.set_title("wind-momentum-balance stagnation point")
    ax.legend(fontsize=7)
    ax.grid(True, ls=":", alpha=0.6)

    # (1,0): adiabatic-cooled Mach vs separation
    ax = axes[1, 0]
    # _cwb_setup.py's own documented valid range ("upper limit: 80, lower
    # limit: 5" au).
    a_range = np.geomspace(5, 80, 200) * u.au
    r2_over_a = frac2
    mach2_adiabatic_range = [
        adiabatic_cooled_wind_mach(VINF2_BASELINE, r2_over_a * a) for a in a_range
    ]
    mach1_adiabatic_range = [
        adiabatic_cooled_wind_mach(VINF1_BASELINE, frac1 * a) for a in a_range
    ]
    ax.loglog(a_range, mach1_adiabatic_range, label="star 1, adiabatic ceiling")
    ax.loglog(a_range, mach2_adiabatic_range, label="star 2, adiabatic ceiling")
    ax.axhline(mach1_ambient, color="tab:blue", ls="--", alpha=0.6, label="star 1, ambient-comparison floor")
    ax.axhline(mach2_ambient, color="tab:orange", ls="--", alpha=0.6, label="star 2, ambient-comparison floor")
    ax.axhline(SIMULATED_MAX_MACH, color="red", ls="--", label="sim max (N=512)")
    ax.axvline(SEPARATION_BASELINE.value, color="grey", ls=":", label="current setup (a=50 au)")
    ax.set_xlabel("binary separation a [au]")
    ax.set_ylabel("Mach number")
    ax.set_title("estimate 2: adiabatic-cooled wind vs. estimate 1 floor")
    ax.legend(fontsize=6.5)
    ax.grid(True, which="both", ls=":", alpha=0.6)

    # (1,1): resolution margin vs N
    ax = axes[1, 1]
    n_range = np.array([32, 64, 96, 128, 192, 256, 384, 512, 768, 1024])
    cells_current_scaling = [cells_between_injection_and_shock(int(n), r2_code) for n in n_range]
    # alternative: decouple num_injection_cells from N (fixed at 2), to show
    # how much of the current scaling's margin loss is avoidable "for free"
    fixed_injection_cells = 2
    dx_code = BOX_SIZE_CODE / n_range
    injection_radius_fixed = fixed_injection_cells * dx_code
    cells_fixed_injection = (r2_code - injection_radius_fixed) / dx_code
    ax.plot(n_range, cells_current_scaling, "o-", label="current (num_injection_cells = N // 32)")
    ax.plot(n_range, cells_fixed_injection, "s--", label="fixed num_injection_cells = 2")
    ax.axvline(64, color="grey", ls=":", label="round-12 run (N=64)")
    ax.axvline(512, color="black", ls=":", label="round-13 run (N=512)")
    ax.set_xlabel("N (cells per dimension)")
    ax.set_ylabel("cells between injection edge and shock")
    ax.set_title("numerical resolution of the wind's cooling zone")
    ax.legend(fontsize=7)
    ax.grid(True, ls=":", alpha=0.6)

    # (1,2): Mach modulation over an eccentric orbit
    ax = axes[1, 2]
    true_anomaly = np.linspace(0, 2 * np.pi, 200)
    for eccentricity, style in [(0.0, "-"), (0.3, "--"), (0.6, ":")]:
        separations = separation_at_true_anomaly(SEPARATION_BASELINE, eccentricity, true_anomaly)
        machs = [
            adiabatic_cooled_wind_mach(VINF2_BASELINE, frac2 * sep) for sep in separations
        ]
        ax.plot(true_anomaly, machs, style, label=f"e={eccentricity}")
    ax.set_xlabel("true anomaly [rad]")
    ax.set_ylabel("star 2 adiabatic-cooled Mach")
    ax.set_title(f"orbital (eccentricity) modulation, a={SEPARATION_BASELINE}")
    ax.legend(fontsize=7)
    ax.grid(True, ls=":", alpha=0.6)

    fig.suptitle("Expected CWB wind-collision shock Mach number vs. physical/orbital parameters")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "expected_cwb_mach.png", dpi=150)
    plt.close(fig)
    print(f"wrote {FIG_DIR / 'expected_cwb_mach.png'}")


if __name__ == "__main__":
    main()
