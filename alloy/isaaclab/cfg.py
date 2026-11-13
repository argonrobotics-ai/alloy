# SPDX-License-Identifier: Apache-2.0
"""Isaac Lab configuration for the Alloy solver (needs Isaac Lab 3.x with its Newton backend; without it the module
imports and ``AlloySolverCfg`` raises ``FeatureUnavailableError`` with the install line)."""

from __future__ import annotations

from ._require import ISAACLAB_ERROR, NEWTON_SOLVER_API, missing_getattr

# Isaac Lab 3.0: the solver config names a NewtonManager subclass; Isaac Lab main: a NewtonSolver subclass
CLASS_TYPE = "{DIR}.solver_api:AlloyNewtonSolver" if NEWTON_SOLVER_API else "{DIR}.manager:NewtonAlloyManager"

if ISAACLAB_ERROR is not None:
    __getattr__ = missing_getattr(__name__, ("AlloySolverCfg",))
else:
    from isaaclab.utils import configclass
    from isaaclab_newton.physics import MJWarpSolverCfg

    @configclass
    class AlloySolverCfg(MJWarpSolverCfg):
        """Alloy's rigid route as an Isaac Lab Newton solver: select it like any other ``NewtonCfg(solver_cfg=...)``."""

        class_type: type | str = CLASS_TYPE
        """Manager class for the Alloy solver."""

        solver_type: str = "alloy"

        rigid_step: float | str = "task"
        """Alloy's step inside one Isaac Lab physics step. ``"task"`` (default): the task's own step, as Isaac Lab's
        MuJoCo Warp takes it. ``"alloy"``: Alloy's robot rule, equal steps of at most 1/600 s (a 5 ms step runs as
        4 x 1.25 ms), finer and slower. A number: a cap in seconds."""

        health: str = "report"
        """Alloy's per-world health checks (``alloy.isaaclab.health``): ``"report"`` (default) checks every world every
        physics step on the GPU, counts the episodes that end INVALID or DEGRADED and prints a summary line; worlds are
        flagged, never frozen. ``"off"``: no checks."""

        health_report_every: int = 2000
        """Physics steps between checks for news: a summary line is printed when a world went INVALID since the last line
        (DEGRADED worlds, such as contacts deeper than 1 mm, are in the summary printed when the run ends). 0: only then."""

        health_speed_limit: float = 20.0
        """A body faster than this (m/s) makes its world INVALID (SPEED_FAIL), Alloy's limit for manipulation scenes.
        Raise it for tasks that are fast on purpose: an untrained cartpole policy at full force drives the cart past
        20 m/s inside its track."""

        noslip_iterations: int = 0
        """MuJoCo's noslip post-solve, as Alloy runs it for a gripper holding an object (0: off, the default; Alloy's
        own scenes use 100 there). Elliptic cone: Alloy's noslip scoped to the robot's contacts; pyramidal cone: MuJoCo
        Warp's own over every contact. It stops a held object creeping out of a grip at the cost of extra solver work."""

        noslip_tolerance: float = 1e-6
        """Convergence tolerance of the noslip post-solve."""
