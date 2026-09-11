"""
Configuration and parameter containers for radiative cooling.

Defines the integer tags that select a cooling-curve type and a cooling method
(explicit / implicit), together with the NamedTuples that carry the parameters
of each cooling curve and the overall cooling configuration.
"""

# typing
from typing import NamedTuple, Union
from types import NoneType
from jaxtyping import PyTree

# jax
import jax.numpy as jnp

# Cooling-curve type tags (select which Lambda(T) model is used).
SIMPLE_POWER_LAW = 1
PIECEWISE_POWER_LAW = 2
NEURAL_NET_COOLING = 3
NEURAL_NET_COOLING_WITH_DENSITY = 4
SIMPLE_MIXING_LAYER_COOLING = 5
KOYAMA_INUTSUKA_NET_COOLING = 6

# Cooling-method tags (how the temperature update is integrated in time).
EXPLICIT_COOLING = 1
IMPLICIT_COOLING = 2


class SimplePowerLawParams(NamedTuple):
    """Parameters of a single power-law cooling curve Lambda(T)."""

    factor: float = 1.0
    exponent: float = 1.0
    reference_temperature: float = 1e8


class PiecewisePowerLawParams(NamedTuple):
    """Tabulated parameters of a piecewise power-law cooling curve.

    The tables hold, per temperature bin, the curve value and slope plus the
    Townsend temporal-evolution coefficients (``Y_table``).
    """

    log10_T_table: jnp.ndarray = jnp.array([])
    log10_Lambda_table: jnp.ndarray = jnp.array([])
    alpha_table: jnp.ndarray = jnp.array([])
    Y_table: jnp.ndarray = jnp.array([])
    reference_temperature: float = 1e8


class CoolingNetConfig(NamedTuple):
    """Static configuration of a neural-network cooling curve."""

    network_static: Union[PyTree, NoneType] = None


class CoolingNetParams(NamedTuple):
    """Trainable parameters of a neural-network cooling curve."""

    network_params: Union[PyTree, NoneType] = None


class MixingCoolingParams(NamedTuple):
    """Parameters of the simple mixing-layer cooling model (Lancaster 2026)."""

    xi: float = 0.5  # xi = t_sh / t_coolmin
    mach_number: float = 0.5
    density_contrast: float = 10.0


class KoyamaInutsukaCoolingParams(NamedTuple):
    """Parameters of the Koyama & Inutsuka (2002) two-phase net-cooling curve.

    The net volumetric rate is ``n_H * Gamma - n_H^2 * Lambda(T)`` (heating
    from photoelectric grain heating minus collisional-line cooling); see
    ``_cooling_tables.koyama_inutsuka_cooling`` for the derivation of these
    already-unit-converted, already mu_e/mu_H-compensated fields, which let
    ``_cooling_rate``'s ``KOYAMA_INUTSUKA_NET_COOLING`` branch return a value
    that reproduces the exact net rate above once fed through the existing
    ``dtemperature_dt`` machinery -- see that builder's docstring for why the
    compensation factors are needed and why they introduce no approximation.
    Unlike every other curve here, the returned "cooling rate" can be
    negative (net heating) -- there is no positive-definite Lambda(T) table
    to speak of, so this curve type is not compatible with the (already
    broken/unused) Townsend temporal-evolution machinery.
    """

    gamma_heating_eff: float = 0.0  # mu_e * Gamma_code (n_H^1 term), code units
    lambda_scale_eff: float = 0.0  # (mu_e / mu_H) * Gamma_code (Lambda(T) prefactor), code units
    code_temperature_per_kelvin: float = 1.0  # code \tilde{T} per 1 K, to recover physical T
    # Koyama & Inutsuka (2002) eq. 4 (corrected form, see Nagashima, Inutsuka
    # & Koyama 2006): bracket(T) = t1_coeff*exp(-t1_exp_coeff/(T+t1_offset))
    # + t2_coeff*sqrt(T)*exp(-t2_exp_coeff/T), Lambda(T) = Gamma * bracket(T).
    t1_coeff: float = 1.0e7
    t1_exp_coeff: float = 1.184e5
    t1_offset: float = 1000.0
    t2_coeff: float = 1.4e-2
    t2_exp_coeff: float = 92.0


# Union of every cooling-curve parameter container; the active variant is
# selected by the cooling-curve type tag in CoolingCurveConfig.
COOLING_CURVE_TYPE = Union[
    SimplePowerLawParams, PiecewisePowerLawParams, CoolingNetParams, MixingCoolingParams, KoyamaInutsukaCoolingParams
]


class CoolingCurveConfig(NamedTuple):
    """Static configuration selecting the cooling-curve model."""

    cooling_curve_type: int = SIMPLE_POWER_LAW

    #: In case of neural the cooling the network architecture
    cooling_net_config: CoolingNetConfig = CoolingNetConfig()


class CoolingConfig(NamedTuple):
    """Top-level cooling configuration (activation, method and curve)."""

    cooling: bool = False
    cooling_method: int = IMPLICIT_COOLING
    cooling_curve_config: CoolingCurveConfig = CoolingCurveConfig()


class CoolingParams(NamedTuple):
    """Runtime cooling parameters (composition, temperature floor, curve)."""

    # NOTE: CURRENTLY ONLY POWER LAW COOLING
    hydrogen_mass_fraction: float = 0.76
    metal_mass_fraction: float = 0.02

    floor_temperature: float = 1e4

    cooling_curve_params: COOLING_CURVE_TYPE = SimplePowerLawParams()
