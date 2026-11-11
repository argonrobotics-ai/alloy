# SPDX-License-Identifier: Apache-2.0
"""Autodiff admission for `method="grad"` (SPEC §3.12: Newton's differentiable contacts where supported, else CMA).

Newton differentiates SolverSemiImplicit, SolverXPBD and SolverFeatherstone through `wp.Tape` when the model is
finalized with `requires_grad=True`; SolverMuJoCo (MuJoCo Warp) does not. A twin joins the autodiff path only with a
`grad_rollout` hook whose tape gradient passes `gradient_check` against central differences. At this Newton pin no
rigid twin passes (CPU probes, [internal note]):
- XPBD d(apex height)/d(restitution) through the tape is 0 (finite differences -12..+20);
- XPBD dx(T)/dmu for a box sliding on a 30 deg incline is -0.127 vs -0.682 by finite differences (-0.680 analytic);
- SemiImplicit's forward run is already wrong at 1.25 ms (x(T) 0.349 vs 0.256 analytic) and its gradients swing
  between -1622 and +45.
So `grad` runs CMA-ES and records the reason; `hybrid` refines CMA-ES with batched finite-difference Gauss-Newton
(alloy.fit.refine), which needs no autodiff.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from alloy.fit.problem import FitSetup

GRAD_SOLVERS: frozenset[str] = frozenset({"semi_implicit", "xpbd", "featherstone"})
GRAD_RTOL = 0.05          # tape vs central difference, relative
EVIDENCE = "[internal note]"


def unsupported_reason(setup: FitSetup) -> str:
    """Why autodiff cannot fit this setup at this pin (always a reason today; see the module docstring)."""
    if setup.solver not in GRAD_SOLVERS:
        return f"solver {setup.solver!r} is not differentiable in Newton (differentiable: {sorted(GRAD_SOLVERS)})"
    if not callable(getattr(setup.twin, "grad_rollout", None)):
        return (f"twin {type(setup.twin).__name__} has no grad_rollout hook; Newton's rigid-contact tape gradients "
                f"fail the finite-difference check at this pin ({EVIDENCE})")
    return f"no twin's grad_rollout has passed gradient_check at this pin ({EVIDENCE})"


def gradient_check(rollout: Callable[[float], tuple[float, float]], x: float, eps: float) -> tuple[bool, float, float]:
    """(ok, tape, fd): `rollout(x) -> (value, tape d value / dx)` against a central difference with step eps."""
    _, tape = rollout(x)
    fd = (rollout(x + eps)[0] - rollout(x - eps)[0]) / (2.0 * eps)
    ok = math.isfinite(tape) and math.isfinite(fd) and abs(tape - fd) <= GRAD_RTOL * max(abs(fd), 1e-12)
    return ok, tape, fd


__all__ = ["GRAD_SOLVERS", "unsupported_reason", "gradient_check", "GRAD_RTOL"]
