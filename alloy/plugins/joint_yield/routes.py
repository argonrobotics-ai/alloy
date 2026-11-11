# SPDX-License-Identifier: Apache-2.0
"""Where a yielding joint's spring reference lives on each MuJoCo route, and the write of s0 + q_p into it before every
step (SCR-WP7-15 P2; the coordinator's ruling: MuJoCo's own per-world spring reference first, C route included).

- **MuJoCo Warp**: ``mjw_model.qpos_spring``, one row per world (Newton's ``_MJW_BATCHED_MODEL_FIELDS`` with
  ``separate_worlds``), read by MuJoCo Warp's passive force each step. A kernel writes it: graph-safe, no host work.
- **Alloy's MuJoCo C route** (``alloy.route.mjc_core``): one float64 ``MjModel`` per world, built from Newton's
  per-world rows at the first step and rebuilt after ``notify_model_changed``. Before each step the route's models are
  made current (``finalize_route``, then the pending rebuild the step would run anyway), and each world whose set
  changed since the last write (a yield, a rollback, a reset) takes s0 + q_p in float64; a rebuilt model list takes
  every world. Below yield nothing is written, so the C models keep their build values bit for bit.
- **Newton's own MuJoCo C path** (one model for every world): one world only.

Writing before every step (not only after a return) is what makes restores safe: snapshots, rollbacks and resets put
back q_p, and the next step's spring reference follows it; a ``notify_model_changed`` that re-syncs ``qpos_spring``
from the build attribute (``mujoco:dof_springref``) is overwritten the same way.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.joint_yield.kernels import write_springref

log = logging.getLogger("alloy.plugins.joint_yield")


def mujoco_solver(stepper: Any) -> Any:
    """The one ``SolverMuJoCo`` that steps the yielding joints; refuses coupled steppers and other solvers."""
    import alloy._vendor.newton.solvers as ns  # noqa: PLC0415

    from alloy.integration.stepper import entry_solvers  # noqa: PLC0415

    entries = entry_solvers(stepper)
    if len(entries) > 1:
        raise PluginError(f"joint yield runs on a plain MuJoCo stepper; this scene couples {sorted(entries)} (cloth, "
                          f"liquid or soft parts beside the robot), which joint yield does not support yet "
                          f"(docs/known_gaps/plasticity-metal.md): drop the deformable or the yield")
    found = [s for s in entries.values() if isinstance(s, ns.SolverMuJoCo)]
    if len(found) != 1:
        raise PluginError(f"joint yield needs the robot on a MuJoCo solver; the scene steps "
                          f"{[type(s).__name__ for s in entries.values()] or type(stepper).__name__}")
    return found[0]


class WarpSpringref:
    """MuJoCo Warp's per-world ``qpos_spring``, written by a kernel before each step."""

    def __init__(self, solver: Any, adr: np.ndarray, s0: wp.array, q_p: wp.array) -> None:
        rows = int(solver.mjw_model.qpos_spring.shape[0])
        if rows != s0.shape[0]:
            raise PluginError(f"joint yield: MuJoCo Warp keeps {rows} spring-reference row(s) for {s0.shape[0]} "
                              f"worlds (separate_worlds off): build the MuJoCo entry with separate_worlds=True")
        self.solver, self.s0, self.q_p = solver, s0, q_p
        self.adr = wp.array(adr.astype(np.int32), dtype=wp.int32, device=s0.device)

    def push(self) -> None:
        wp.launch(write_springref, dim=self.s0.shape, inputs=[self.adr, self.s0, self.q_p],
                  outputs=[self.solver.mjw_model.qpos_spring], device=self.s0.device)


class CRouteSpringref:
    """Alloy's MuJoCo C route: each world's float64 ``MjModel.qpos_spring`` where the set changed (module doc)."""

    def __init__(self, solver: Any, adr: np.ndarray, q_p: wp.array) -> None:
        for name in ("finalize_route", "_rebuild_if_stale", "worlds"):
            if not hasattr(solver, name):
                raise PluginError(f"joint yield: this MuJoCo C route has no {name!r}; the spring references would "
                                  f"not reach its per-world models")
        self.solver, self.adr, self.q_p = solver, adr, q_p
        self.s0: np.ndarray | None = None              # float64 build values, read from the first models built
        self._models: Any = None                       # the route's model list the last write went into
        self._written = np.zeros(q_p.shape, np.float32)

    def push(self) -> None:
        s = self.solver
        s.finalize_route()
        s._rebuild_if_stale()          # the rebuild the step would run first; a write before it would be lost
        if self.s0 is None:
            self.s0 = np.stack([np.asarray(m.qpos_spring[self.adr], np.float64) for m in s.worlds])
        q_p = self.q_p.numpy()         # a CPU array: a live view, compared and copied below
        if s.worlds is not self._models:
            changed = np.arange(len(s.worlds))
            self._models = s.worlds
        else:
            changed = np.flatnonzero((q_p != self._written).any(axis=1))
        for w in changed.tolist():
            s.worlds[w].qpos_spring[self.adr] = self.s0[w] + q_p[w].astype(np.float64)
        if changed.size:
            self._written[changed] = q_p[changed]


class TemplateSpringref:
    """Newton's own MuJoCo C path: its one model (one world)."""

    def __init__(self, solver: Any, adr: np.ndarray, s0: np.ndarray, q_p: wp.array) -> None:
        if s0.shape[0] != 1:
            raise PluginError(f"joint yield sets spring references per world, but Newton's MuJoCo C path steps one "
                              f"model for all {s0.shape[0]} worlds: use one world, Alloy's MuJoCo C route or MuJoCo "
                              f"Warp")
        self.solver, self.adr, self.s0, self.q_p = solver, adr, s0.astype(np.float64), q_p

    def push(self) -> None:
        self.solver.mj_model.qpos_spring[self.adr] = self.s0[0] + self.q_p.numpy()[0].astype(np.float64)


def springref_writer(solver: Any, adr: np.ndarray, s0: wp.array, q_p: wp.array) -> Any:
    """The writer for ``solver``'s route (module doc); ``s0`` is the float32 build row the kernels use."""
    from alloy.route.mjc_core import MjcCoreSolver  # noqa: PLC0415

    if not getattr(solver, "use_mujoco_cpu", False):
        return WarpSpringref(solver, adr, s0, q_p)
    if isinstance(solver, MjcCoreSolver):
        return CRouteSpringref(solver, adr, q_p)
    return TemplateSpringref(solver, adr, np.asarray(solver.mj_model.qpos_spring[adr], np.float64)[None], q_p)
