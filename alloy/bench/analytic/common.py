# SPDX-License-Identifier: Apache-2.0
"""Shared pieces of the WP6 analytic twins: solver settings per law, the rigid twin base and small helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench import default_route as DR
from alloy.bench import solvers as SV
from alloy.bench.twin import DeviceTwinLog, WorldRanges, record_body_position, record_body_velocity
from alloy.bench.types import Condition
from alloy.core.defaults import GRAVITY

log = logging.getLogger("alloy.bench")

G = GRAVITY  # [m/s^2] Newton's default |g| (newton@ac7f7830:newton/_src/sim/builder.py:1750)
MAX_SAMPLES = 2000  # recorded columns per run; the stride grows for long full runs


def mujoco_law_settings() -> SV.SolverSettings:
    """MuJoCo as the A5 amendment states it (SPEC §5.3): elliptic cone, impratio 300, flat solimp imp 0.99."""
    return SV.SolverSettings("mujoco", {"cone": "elliptic", "impratio": 300.0, "iterations": 100,
                                        "ls_iterations": 50}, mujoco_solimp=(0.99, 0.99, 0.001, 0.5, 2.0))


class BaseTwin:
    """Default hooks for WP6 twins: stock solver construction from per-solver `SolverSettings`, no drive.

    Subclasses set `law_id`, `units`, optionally `settings` (solver name -> SolverSettings), `needs_contacts`,
    and implement `conditions`, `build_world`, `record`, `scalars`.
    """

    law_id: str = ""
    units: Mapping[str, str] = {}
    settings: Mapping[str, SV.SolverSettings] = {}
    needs_contacts: bool = True

    def __init__(self) -> None:
        self.model: Any = None
        self.solver_name = ""
        self.monitors: dict[str, Any] = {}
        self.ranges: WorldRanges | None = None
        self._first_body: wp.array | None = None
        self.worlds: list[Condition] = []
        self.samples: list[Mapping[str, float]] = []

    def build_world(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        """Track the world order, then build it (`build`)."""
        self.worlds.append(cond)
        self.samples.append(dict(sample))
        self.build(b, cond, sample)

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        raise NotImplementedError

    def param(self, key: str) -> np.ndarray:
        """Per-world value of condition parameter `key`, in world order."""
        return np.array([c.params[key] for c in self.worlds], dtype=np.float64)

    def world_gravity(self, cond: Condition) -> tuple[float, float, float] | None:
        """Per-world gravity vector (None = builder default, -9.81 m/s^2 along z)."""
        return None

    def settings_for(self, solver: str) -> SV.SolverSettings:
        """The law's own settings for `solver`, else stock settings; MuJoCo without law settings runs as Alloy routes
        it (default route: builtin kwargs, SCR-WP6-4 integrator for this model)."""
        if solver in self.settings:
            return self.settings[solver]
        if solver == "mujoco":
            return DR.routed_mujoco_settings(self.model)
        return SV.SolverSettings(solver)

    def settings_record(self, solver: str) -> dict[str, Any]:
        return self.settings_for(solver).describe()

    def make_solver(self, model: newton.Model, solver: str) -> newton.solvers.SolverBase:
        self.model = model
        self.solver_name = solver
        self.ranges = WorldRanges.bodies(model) if model.body_count else None
        self._first_body = self.ranges.start if self.ranges is not None else None
        self.on_model(model)
        return SV.make_solver(model, self.settings_for(solver))

    def make_pipeline(self, model: newton.Model, solver: str) -> newton.CollisionPipeline | None:
        return SV.make_pipeline(model, self.settings_for(solver)) if self.needs_contacts else None

    def on_model(self, model: newton.Model) -> None:
        """Hook: precompute indices once the model exists."""

    def drive(self, step: int, t: float, state: Any, control: Any) -> None:
        """Passive scenes apply nothing."""

    def record_stride(self, n_steps: int) -> int:
        return max(1, -(-n_steps // MAX_SAMPLES))

    # -- recording helpers -----------------------------------------------------------------------------------------
    def record_first_body(self, step: int, state: Any, log_: DeviceTwinLog, velocity: bool = False) -> None:
        """Position (`x y z`) and optionally COM velocity (`vx vy vz`) of each world's first body."""
        if self._first_body is None:
            return
        record_body_position(log_, "", state.body_q, self._first_body, step)
        if velocity:
            record_body_velocity(log_, "v", state.body_qd, self._first_body, step)


def condition_params(conds: tuple[Condition, ...], key: str) -> np.ndarray:
    return np.array([c.params[key] for c in conds], dtype=np.float64)


def set_body_velocities(model: newton.Model, state: Any, qd: np.ndarray) -> None:
    """Assign body twists (linear COM, angular; world frame) and sync generalized coordinates (`eval_ik`).

    Newton's stopping-distance test does the same for MuJoCo (newton@ac7f7830:newton/tests/
    test_rigid_friction_ramp.py:315-320): maximal-coordinate solvers read `body_qd`, reduced ones `joint_qd`.
    """
    state.body_qd.assign(np.asarray(qd, dtype=np.float32).reshape(-1, 6))
    newton.eval_ik(model, state, state.joint_q, state.joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def smoothstep(s: float) -> float:
    """C1 ramp 0 -> 1 on [0, 1] used to ease loads in (CON-51: stepped loads slip early)."""
    s = min(1.0, max(0.0, s))
    return s * s * (3.0 - 2.0 * s)


__all__ = ["BaseTwin", "G", "mujoco_law_settings", "set_body_velocities", "smoothstep", "condition_params"]
