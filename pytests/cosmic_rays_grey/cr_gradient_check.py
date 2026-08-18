"""
CR transport differentiability pytest (plan Sec. 4, item 16 -- "continuous,
from Phase A").

Finite-difference vs. autodiff gradient check on a small CR transport
problem: perturb a CR-relevant input (e.g. the initial e_cr amplitude or
reduced_streaming_speed), run a short simulation to a scalar cost, and
compare the AD gradient (jax.grad) against a centered finite-difference
gradient. Catches a floor/limiter/jnp.where silently killing the adjoint --
exactly the failure mode the plan's differentiability plan (tanh-regularized
streaming sign, no min/max/sign) is designed to avoid. See
pytests/differentiability/sensitivity.py for the general style this repo
uses for gradient-correctness pytests (there: AD vs. a closed-form analytic
gradient for the non-CR Euler equations; here: AD vs. finite differences,
since a closed-form CR gradient isn't available).

TODO (fills in once the transport stubs are implemented):
  - build a small SimulationConfig with cosmic_ray_grey_config.grey_cosmic_rays=True
  - define cost(reduced_streaming_speed) = sum(e_cr_final**2) (or similar)
  - ad_grad = jax.grad(cost)(reduced_streaming_speed)
  - fd_grad = (cost(x + h) - cost(x - h)) / (2 * h) for a small h
  - assert relative error between ad_grad and fd_grad is below tol

See astronomix/_modules/_cosmic_rays_grey/DESIGN.md's differentiability plan.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

def test_cr_gradient_check(tol: float = 1e-2):
    """Finite-difference vs. autodiff gradient check on a small CR problem.

    Args:
        tol: The maximum allowed relative error between the AD and
            finite-difference gradients.
    """
    raise NotImplementedError(
        "Plan Sec. 4 item 16: CR transport FD-vs-AD gradient check. See this "
        "module's docstring and "
        "astronomix/_modules/_cosmic_rays_grey/DESIGN.md."
    )


if __name__ == "__main__":
    test_cr_gradient_check()
