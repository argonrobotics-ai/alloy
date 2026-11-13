# SPDX-License-Identifier: Apache-2.0
"""``AlloyNewtonSolver``: Alloy's solver for Isaac Lab after 3.0 (``main``), where each solver is a ``NewtonSolver``
class (``isaaclab_newton.physics.newton_solver``) rather than a ``NewtonManager`` subclass (Isaac Lab 3.0,
``alloy.isaaclab.manager``). Same solver, bridge, step rule and health checks; only the adapter differs. Needs Isaac
Lab; without it (or on Isaac Lab 3.0) the module imports and the name raises ``FeatureUnavailableError``."""

from __future__ import annotations

import atexit
from typing import Any

from ._require import ISAACLAB_ERROR, NEWTON_SOLVER_API, missing_getattr

if ISAACLAB_ERROR is not None or not NEWTON_SOLVER_API:
    __getattr__ = missing_getattr(__name__, ("AlloyNewtonSolver",))
else:
    import warp as wp
    from isaaclab_newton.physics.mjwarp_solver import MJWarpSolver
    # Isaac Lab's own Newton classes, as its solver base uses them (Alloy's bundled Newton is a separate copy)
    from isaaclab_newton.physics.newton_solver import CollisionPipeline, Contacts, eval_fk

    from .solver import AlloyRigidSolver, physics_line

    class AlloyNewtonSolver(MJWarpSolver):
        """Isaac Lab's MuJoCo Warp solver class with Alloy's solver in place of Newton's ``SolverMuJoCo``. Isaac Lab's
        own model, state, control and contacts stay Isaac Lab's; Alloy steps a bridged model sharing their arrays."""

        def build(self, model: Any, mode: wp.DeterministicMode, env_template: str | None) -> AlloyRigidSolver:
            deterministic = mode != wp.DeterministicMode.NOT_GUARANTEED
            if deterministic:
                self.cfg.disable_sensors = True   # as Isaac Lab's MJWarpSolver: the sensor stage cannot repeat
            solver = AlloyRigidSolver(model, self.cfg, deterministic=mode)
            self._health_steps, self._health_seen = 0, 0
            if solver.health is not None:
                atexit.register(lambda: print(solver.health.report().line(), flush=True))
            print(physics_line(solver), flush=True)
            return solver

        # Isaac Lab's own Newton must run on Isaac Lab's own model: collision, FK and contact buffers
        def create_collision_pipeline(self, **kwargs) -> CollisionPipeline:
            return CollisionPipeline(self.native.host_model, **kwargs)

        def eval_fk(self, state, world_mask, fk_mask) -> None:
            eval_fk(self.native.host_model, state.joint_q, state.joint_qd, state, fk_mask)

        def create_contacts(self) -> Contacts:
            host = self.native.host_model
            return Contacts(rigid_contact_max=self.native.get_max_contact_count(), soft_contact_max=0,
                            device=host.device, requested_attributes=host.get_requested_contact_attributes())

        def reset(self, state, world_mask) -> None:
            super().reset(state, world_mask)
            if self.native.health is not None:
                self.native.health.on_reset(world_mask)

        def check_step(self, captured: bool, debug: bool) -> None:
            super().check_step(captured, debug)
            health, every = self.native.health, int(getattr(self.cfg, "health_report_every", 0) or 0)
            if health is None or every <= 0:
                return
            self._health_steps += 1
            if self._health_steps % every == 0:
                summary = health.report()
                invalid = len(summary.invalid_now) + summary.episodes_invalid
                if invalid > self._health_seen:
                    self._health_seen = invalid
                    print(summary.line(), flush=True)
