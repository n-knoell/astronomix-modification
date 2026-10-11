"""
Linear-theory reference solutions for the grey two-moment CR tests (Phase A
test plan, ``astronomix/_modules/_cosmic_rays_grey/astronomix_CR_phaseA_test_plan.md``).

So far: the exact spectral solution of the isotropic two-moment CR system in
gas at rest (the "telegrapher" reference of plan item A4.1). The plan's other
linear-theory pieces (the 5x5 gas + CR dispersion matrix for A2.4, the
eigenvectors for A1.4 / A2.2) go here as well when those tests are built.

The model is the one in ``cr_grey_transport.grey_cr_flux_terms`` and
``cr_grey_sources.cr_flux_relaxation_update`` with ``u = 0`` and isotropic
diffusive relaxation (Jiang & Oh 2018, JO18):

    d(e)/dt + div(F)                  = 0
    d(F)/dt + v_red^2 grad(P_cr)      = -nu F,
    P_cr = (gamma_cr - 1) e,  nu = (gamma_cr - 1) v_red^2 / kappa.

With ``c^2 = (gamma_cr - 1) v_red^2`` each Fourier mode ``exp(i k . x)``
decouples into ``(e_k, G_k)`` with the longitudinal flux ``G_k = k_hat . F_k``:

    d(e_k)/dt = -i |k| G_k
    d(G_k)/dt = -i |k| c^2 e_k - nu G_k,

eigenvalues ``s_pm = (-nu +- sqrt(nu^2 - 4 c^2 k^2)) / 2`` (JO18 Sec. 3,
Thomas, Pfrommer & Pakmor 2021 Sec. 4.5). Modes with ``|k| < k_c = nu / (2 c)``
are overdamped; the slow root tends to the diffusion rate ``-kappa k^2`` for
``k << k_c``. Modes with ``|k| > k_c`` are damped waves (rate ``nu / 2``,
frequency ``sqrt(c^2 k^2 - nu^2 / 4)``). The transverse flux decays as
``exp(-nu t)``.

The 2x2 propagator is evaluated as ``exp(M t) = a0 I + a1 M`` with
``a1 = (exp(s_+ t) - exp(s_- t)) / (s_+ - s_-)`` and ``a0 = exp(s_+ t) - s_+ a1``,
with the roots in their cancellation-free form (``s_- = -nu/2 - q``,
``s_+ = c^2 k^2 / s_-``) and a series for ``a1`` near critical damping.
Verified against ``scipy.linalg.expm`` per mode (random data, an exactly
critical mode, t = 1e-4 ... 0.05: agreement to 1e-15) and, for ``nu -> 0``,
against d'Alembert's solution of the free wave (6.7e-14).

Plain numpy (a one-shot reference generator for tests, like the other modules
in this package), not jit- or grad-compatible.
"""

# numerics
import numpy as np


def telegrapher_rates(k, gamma_cr, reduced_streaming_speed, diffusion_coefficient):
    """The two roots ``(s_slow, s_fast)`` of a Fourier mode of the isotropic
    two-moment CR system (complex for ``|k| > k_c``).

    Args:
        k: Wavenumber(s) ``|k|``.
        gamma_cr: The CR adiabatic index.
        reduced_streaming_speed: ``v_red``.
        diffusion_coefficient: ``kappa`` (the ``e_cr`` diffusivity).

    Returns:
        ``(s_slow, s_fast)``, complex arrays of the shape of ``k``.
    """
    c2 = (gamma_cr - 1.0) * reduced_streaming_speed**2
    nu = c2 / diffusion_coefficient
    k = np.asarray(k, dtype=float)
    q = np.sqrt((0.25 * nu**2 - c2 * k**2).astype(complex))
    s_fast = -0.5 * nu - q
    s_slow = c2 * k**2 / s_fast
    return s_slow, s_fast


def telegrapher_critical_wavenumber(gamma_cr, reduced_streaming_speed, diffusion_coefficient):
    """``k_c = nu / (2 c)``: modes above it are damped waves, below it overdamped.

    Args:
        gamma_cr: The CR adiabatic index.
        reduced_streaming_speed: ``v_red``.
        diffusion_coefficient: ``kappa``.

    Returns:
        ``k_c``.
    """
    c = np.sqrt(gamma_cr - 1.0) * reduced_streaming_speed
    return 0.5 * c / diffusion_coefficient


def telegrapher_spectral_solution(
    e0,
    box_size,
    t,
    gamma_cr,
    reduced_streaming_speed,
    diffusion_coefficient,
    flux0=None,
):
    """Exact solution of the isotropic two-moment CR system in gas at rest on a
    periodic box, for the trigonometric interpolant of the given cell data.

    Args:
        e0: Initial ``e_cr`` on a uniform periodic grid, shape ``(N,)`` in 1D
            or ``(N, N[, N])`` in 2D/3D (cubic box).
        box_size: The box length.
        t: The time.
        gamma_cr: The CR adiabatic index.
        reduced_streaming_speed: ``v_red``.
        diffusion_coefficient: ``kappa``.
        flux0: Initial ``F_cr``, shape ``(d,) + e0.shape``; ``None`` for zero.

    Returns:
        ``(e, flux)``: ``e`` of the shape of ``e0`` and ``flux`` of shape
        ``(d,) + e0.shape``.
    """
    e0 = np.asarray(e0, dtype=float)
    dims = e0.ndim
    n = e0.shape[0]
    dx = box_size / n
    k1 = 2.0 * np.pi * np.fft.fftfreq(n, dx)
    k_vec = np.stack(np.meshgrid(*([k1] * dims), indexing="ij"))
    k_abs = np.sqrt(np.sum(k_vec**2, axis=0))
    k_hat = np.where(k_abs > 0, k_vec / np.where(k_abs > 0, k_abs, 1.0), 0.0)

    c2 = (gamma_cr - 1.0) * reduced_streaming_speed**2
    nu = c2 / diffusion_coefficient
    s_slow, s_fast = telegrapher_rates(k_abs, gamma_cr, reduced_streaming_speed, diffusion_coefficient)

    z = (s_slow - s_fast) * t
    small = np.abs(z) < 1e-3
    a1 = np.where(
        small,
        np.exp(s_fast * t) * t * (1.0 + z / 2.0 + z**2 / 6.0 + z**3 / 24.0),
        (np.exp(s_slow * t) - np.exp(s_fast * t)) / np.where(small, 1.0, s_slow - s_fast),
    )
    a0 = np.exp(s_slow * t) - s_slow * a1

    e_hat = np.fft.fftn(e0)
    if flux0 is None:
        g_hat = np.zeros_like(e_hat)
        transverse_hat = np.zeros((dims,) + e0.shape, dtype=complex)
    else:
        f_hat = np.stack([np.fft.fftn(np.asarray(f, dtype=float)) for f in flux0])
        g_hat = np.sum(k_hat * f_hat, axis=0)
        transverse_hat = f_hat - k_hat * g_hat
        if dims == 1:
            # In 1D the k = 0 mode of F has k_hat = 0; it only decays.
            transverse_hat = np.where(k_abs > 0, 0.0, f_hat)

    e_t = a0 * e_hat + a1 * (-1j * k_abs * g_hat)
    g_t = a1 * (-1j * k_abs * c2 * e_hat) + (a0 - a1 * nu) * g_hat
    flux_t = k_hat * g_t + transverse_hat * np.exp(-nu * t)

    e = np.real(np.fft.ifftn(e_t))
    flux = np.stack([np.real(np.fft.ifftn(f)) for f in flux_t])
    return e, flux
