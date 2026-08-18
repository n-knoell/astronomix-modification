"""
Grey cosmic-ray equation of state.

Unlike the older single-scalar model (``_modules._cosmic_rays.cr_fluid_equations``),
the grey two-moment model tracks ``e_cr`` as its own independent state
variable rather than folding a CR pressure into the total gas pressure/energy
slot -- CR and gas-thermal energy are tracked separately (plan Sec. 2), and
the coupling between them is the explicit feedback source terms in
``cr_grey_sources.py``, not a change to the primitive/conserved recovery of
the gas state. So there is no ``*_with_crs`` counterpart to
``total_energy_from_primitives``/``total_pressure_from_conserved`` here.
"""

# jax
import jax
import jax.numpy as jnp

# typing
from typing import Union
from jaxtyping import Array, Float


@jax.jit
def pressure_from_e_cr(
    e_cr: Float[Array, "..."],
    gamma_cr: Union[float, Float[Array, ""]],
) -> Float[Array, "..."]:
    """Grey CR pressure from the CR energy density.

    ``P_cr = (gamma_cr - 1) * e_cr`` (plan Sec. 2).

    Args:
        e_cr: The cosmic-ray energy density.
        gamma_cr: The cosmic-ray adiabatic index (4/3 for a relativistic gas).

    Returns:
        The cosmic-ray pressure.
    """
    return (gamma_cr - 1) * e_cr


@jax.jit
def e_cr_from_pressure(
    p_cr: Float[Array, "..."],
    gamma_cr: Union[float, Float[Array, ""]],
) -> Float[Array, "..."]:
    """Inverse of :func:`pressure_from_e_cr`.

    Args:
        p_cr: The cosmic-ray pressure.
        gamma_cr: The cosmic-ray adiabatic index.

    Returns:
        The cosmic-ray energy density.
    """
    return p_cr / (gamma_cr - 1)
