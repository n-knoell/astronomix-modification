"""
Koyama & Inutsuka (2002) two-phase thermal-equilibrium reference solution.

Plain numpy/scipy (no JAX, no astronomix imports), working entirely in
physical cgs units, so this is a genuinely independent cross-check of
``astronomix._modules._cooling._cooling_tables.koyama_inutsuka_cooling`` /
``_cooling._cooling_rate``'s ``KOYAMA_INUTSUKA_NET_COOLING`` branch --
same "independent reference solver" pattern as ``pfrommer_riemann_solver.py``
and ``cr_modified_shock_structure.py``.

Net heating/cooling rate per unit volume (Koyama & Inutsuka 2002, ApJ 564,
L97, eq. 4, corrected coefficients per the erratum discussion in Nagashima,
Inutsuka & Koyama 2006, ApJ 652, 1331):

    L(n_H, T) = n_H * Gamma - n_H^2 * Lambda(T)
    Gamma = 2e-26 erg/s
    Lambda(T) = Gamma * [1e7 * exp(-1.184e5 / (T + 1000))
                         + 1.4e-2 * sqrt(T) * exp(-92 / T)]   erg cm^3 / s

Thermal equilibrium (``L = 0``, ``T > 0``) requires ``Gamma = n_H *
Lambda(T)``, i.e. ``bracket(T) = 1 / n_H`` where ``bracket`` is the
dimensionless factor multiplying Gamma in Lambda(T) above. For
``n_H ~ O(1) cm^-3`` this has three roots (the classic S-shaped curve): a
cold, thermally stable root (``d(n_H*Lambda)/dT > d(Gamma)/dT = 0``, i.e.
``bracket`` increasing), an unstable middle root (``bracket`` decreasing),
and a warm, thermally stable root (``bracket`` increasing again).
"""

# numerics
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

GAMMA_HEATING_CGS = 2e-26  # erg / s
GAMMA_SCALE_CGS = 2e-26  # erg cm^3 / s (same numeral, different physical role -- see module docstring)


def ki_bracket(temperature_kelvin):
    """Dimensionless bracket: Lambda(T) = GAMMA_SCALE_CGS * ki_bracket(T)."""
    T = np.asarray(temperature_kelvin, dtype=float)
    return (
        1.0e7 * np.exp(-1.184e5 / (T + 1000.0))
        + 1.4e-2 * np.sqrt(T) * np.exp(-92.0 / T)
    )


def ki_lambda(temperature_kelvin):
    """Lambda(T) in erg cm^3 / s."""
    return GAMMA_SCALE_CGS * ki_bracket(temperature_kelvin)


def ki_net_rate(n_h_cgs, temperature_kelvin):
    """Net volumetric rate L = n_H*Gamma - n_H^2*Lambda(T), erg / (cm^3 s)."""
    n_h = np.asarray(n_h_cgs, dtype=float)
    return n_h * GAMMA_HEATING_CGS - n_h ** 2 * ki_lambda(temperature_kelvin)


def find_equilibrium_temperatures(n_h_cgs, t_min=10.0, t_max=1.0e5, num_scan=20000):
    """Find all thermal-equilibrium temperatures (``L=0``) at a fixed density.

    Scans ``[t_min, t_max]`` in log space for sign changes of
    ``ki_net_rate(n_h_cgs, T)`` and brackets each with ``brentq``. At
    ``n_h_cgs ~ O(1)`` this returns three roots (cold-stable,
    unstable, warm-stable, in increasing T); at higher/lower density fewer
    roots may exist (the S-shape can lose the unstable branch entirely).

    Args:
        n_h_cgs: Hydrogen number density in cm^-3.
        t_min: Lower end of the scanned temperature range, in Kelvin.
        t_max: Upper end of the scanned temperature range, in Kelvin.
        num_scan: Number of log-spaced scan points used to bracket roots.

    Returns:
        A sorted list of equilibrium temperatures in Kelvin.
    """
    T_scan = np.logspace(np.log10(t_min), np.log10(t_max), num_scan)
    rate = ki_net_rate(n_h_cgs, T_scan)

    roots = []
    sign_changes = np.where(np.sign(rate[:-1]) * np.sign(rate[1:]) < 0)[0]
    for i in sign_changes:
        root = brentq(lambda T: ki_net_rate(n_h_cgs, T), T_scan[i], T_scan[i + 1], xtol=1e-6, rtol=1e-12)
        roots.append(root)

    return sorted(roots)


def is_thermally_stable(n_h_cgs, temperature_kelvin, delta_log_t=1e-4):
    """Return True if the equilibrium at ``temperature_kelvin`` is thermally stable.

    Field's criterion at constant density: stable iff ``d(n_H*Lambda(T) -
    Gamma)/dT > 0`` at the root, i.e. the net cooling rate ``-L`` increases
    with T (a positive perturbation is damped, not amplified).
    """
    T = temperature_kelvin
    dT = T * delta_log_t
    rate_minus = ki_net_rate(n_h_cgs, T - dT)
    rate_plus = ki_net_rate(n_h_cgs, T + dT)
    # L = -[cooling - heating]; stable iff L decreases through zero with T.
    return (rate_plus - rate_minus) < 0.0


K_B_CGS = 1.380649e-16  # erg / K
SEC_PER_YEAR = 3.15576e7


def integrate_isochoric_relaxation(
    n_h_cgs, temperature_kelvin_init, t_end_years, mean_molecular_weight, mean_molecular_weight_hydrogen,
    gamma=5.0 / 3.0,
):
    """Integrate the isochoric (fixed n_H) thermal-relaxation ODE.

    ``d(n_tot*k_B*T/(gamma-1))/dt = L(n_H, T)`` at fixed ``n_H`` (hence
    fixed ``rho = n_H * mu_H * m_p``), with the TRUE total particle number
    density ``n_tot = rho / (mu * m_p) = n_H * mu_H / mu`` -- note this is
    *not* ``n_H / mu`` (a subtle, easy-to-get-wrong point: ``mu`` and
    ``mu_H`` differ, so the two don't cancel). Pass the same ``mu``/``mu_H``
    the simulation under test actually uses (via
    ``astronomix._modules._cooling._cooling.get_effective_molecular_weights``)
    so this reference tracks the same heat-capacity convention.

    Args:
        n_h_cgs: Hydrogen number density in cm^-3 (held fixed).
        temperature_kelvin_init: Initial temperature in Kelvin.
        t_end_years: Integration end time, in years.
        mean_molecular_weight: Mean molecular weight (mu).
        mean_molecular_weight_hydrogen: Hydrogen mean molecular weight
            (mu_H = 1 / hydrogen_mass_fraction).
        gamma: Adiabatic index.

    Returns:
        A ``scipy.integrate.OdeResult`` with dense output; ``.sol(t)``
        gives the temperature (Kelvin) at any time (seconds) in range.
    """
    n_tot = n_h_cgs * mean_molecular_weight_hydrogen / mean_molecular_weight
    heat_capacity_per_volume = n_tot * K_B_CGS / (gamma - 1.0)  # erg / (cm^3 K)

    def rhs(t, T):
        return [ki_net_rate(n_h_cgs, T[0]) / heat_capacity_per_volume]

    t_end_sec = t_end_years * SEC_PER_YEAR
    return solve_ivp(
        rhs, [0.0, t_end_sec], [temperature_kelvin_init],
        method="LSODA", rtol=1e-10, atol=1e-6, dense_output=True,
    )
