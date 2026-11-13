# SPDX-License-Identifier: Apache-2.0
"""``NewtonAlloyManager``: the Isaac Lab Newton manager that steps physics on Alloy (needs Isaac Lab; without it the
module imports and the name raises ``FeatureUnavailableError`` with the install line)."""

from __future__ import annotations

from ._require import ISAACLAB_ERROR, NEWTON_SOLVER_API, missing_getattr

if ISAACLAB_ERROR is not None or NEWTON_SOLVER_API:   # main: alloy.isaaclab.solver_api.AlloyNewtonSolver instead
    __getattr__ = missing_getattr(__name__, ("NewtonAlloyManager",))
else:
    from isaaclab.physics import PhysicsManager
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics.mjwarp_manager import NewtonMJWarpManager

    from .cfg import AlloySolverCfg
    from .solver import AlloyRigidSolver, physics_line

    class NewtonAlloyManager(NewtonMJWarpManager):
        """Isaac Lab's MuJoCo Warp manager with Alloy's solver in place of Newton's ``SolverMuJoCo``."""

        @classmethod
        def _create_solver(cls, model, solver_cfg: AlloySolverCfg) -> AlloyRigidSolver:
            return AlloyRigidSolver(model, solver_cfg, deterministic=NewtonManager._deterministic_mode)

        @classmethod
        def _build_solver(cls, model, solver_cfg: AlloySolverCfg) -> None:
            NewtonManager._solver = cls._create_solver(model, solver_cfg)
            NewtonManager._use_single_state = True
            NewtonManager._needs_collision_pipeline = not solver_cfg.use_mujoco_contacts  # as Isaac Lab's MuJoCo Warp
            NewtonManager._supports_rigid_body_force_input = True
            print(physics_line(NewtonManager._solver), flush=True)

        _health_calls: int = 0
        _health_invalid_seen: int = 0

        @classmethod
        def _reset_solver_internals(cls, world_mask) -> None:
            super()._reset_solver_internals(world_mask)
            health = getattr(cls._solver, "health", None)
            if world_mask is not None and health is not None:
                health.on_reset(world_mask)

        @classmethod
        def _check_solver_status(cls) -> None:
            """Every ``health_report_every`` physics steps, one summary line if any world was flagged (one host read)."""
            health = getattr(cls._solver, "health", None)
            every = int(getattr(PhysicsManager._cfg.solver_cfg, "health_report_every", 0) or 0)
            if health is None or every <= 0:
                return
            NewtonAlloyManager._health_calls += 1
            if NewtonAlloyManager._health_calls % every == 0:
                summary = health.report()
                invalid = len(summary.invalid_now) + summary.episodes_invalid
                if invalid > NewtonAlloyManager._health_invalid_seen:   # only news: a world went INVALID since the last line
                    NewtonAlloyManager._health_invalid_seen = invalid
                    print(summary.line(), flush=True)

        @classmethod
        def _solver_specific_clear(cls) -> None:
            health = getattr(cls._solver, "health", None)
            if health is not None:
                try:
                    print(health.report().line(), flush=True)
                except Exception as err:  # the run is ending; never fail it over the summary
                    print(f"[INFO] [alloy] health summary unavailable: {err}", flush=True)
            super()._solver_specific_clear()

