# SPDX-License-Identifier: Apache-2.0
"""The CPU route's float64 MuJoCo state as solver state: snapshots, checkpoints and re-runs restore it.

MuJoCo C keeps each world's float64 state between calls (``mjc_core_window``); Newton's ``State`` holds a float32 copy.
Snapshots, checkpoints and re-run restores copy ``State`` and the arrays of the solver's state adapter. Without this
module the route had none of its own, so a restore put the float32 state back, the route saw it differ from its last
output and pushed it into MjData (float32 -> float64), and the run went on from a state the unbroken run never had: a
re-run world, a kept world beside it and a resumed checkpoint each left the unbroken run.

``LiveState`` keeps, after every step, each world's ``mj_getState`` row (``SPEC``) and the float32 coordinates the
route wrote with it (its key), in Warp arrays registered as the route's state (``install``). A restore copies them back
with ``State``. At the next hand-in (the coupler's notify or the step), a world whose state differs from the route's
last output but equals the key bit for bit takes the row back (``mj_setState``) instead of a float32 push, the same
rule (``mjc_core_window.keyed``) as the window start and the settled start. Any other change (a user edit, a reset, a
coupler's write) differs from the key and is pushed from float32, as before. A reset forgets its worlds' rows.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
import warp as wp

from alloy.core.worldops import joint_coord_world, joint_dof_world
from alloy.route.mjc_core_window import SPEC, keyed

PREFIX = "mjc"


class LiveState:
    """Each world's float64 MuJoCo state after the route's last step, keyed by the coordinates it wrote (module doc)."""

    def __init__(self, solver: Any, nworld: int) -> None:
        model = solver.model
        size = int(mujoco.mj_stateSize(solver.mj_model, SPEC))
        self.rows = wp.zeros((nworld, size), dtype=wp.float64, device="cpu")
        self.valid = wp.zeros(nworld, dtype=wp.int32, device="cpu")
        self.q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device="cpu")
        self.qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device="cpu")
        self.coord_world = joint_coord_world(model) if model.joint_coord_count else None
        self.dof_world = joint_dof_world(model) if model.joint_dof_count else None

    def arrays(self) -> dict[str, wp.array]:
        """The arrays a snapshot copies (non-empty ones: a zero-length Warp array has no allocation)."""
        out = {f"{PREFIX}.state": self.rows, f"{PREFIX}.valid": self.valid}
        out.update({f"{PREFIX}.{k}": a for k, a in (("q", self.q), ("qd", self.qd)) if a.size})
        return out

    def world_maps(self) -> dict[str, wp.array]:
        """Item->world maps of the per-coordinate keys (the (nworld, ...) arrays need none)."""
        out = {}
        if self.coord_world is not None:
            out[f"{PREFIX}.q"] = self.coord_world
        if self.dof_world is not None:
            out[f"{PREFIX}.qd"] = self.dof_world
        return out

    def save(self, models: list[Any], datas: list[Any], state: Any) -> None:
        """After a step: every world's row and the coordinates the route wrote (``state`` is its output)."""
        rows = self.rows.numpy()
        for w, (m, d) in enumerate(zip(models, datas)):
            mujoco.mj_getState(m, d, rows[w], SPEC)
        if self.q.size:
            self.q.numpy()[:] = state.joint_q.numpy()
        if self.qd.size:
            self.qd.numpy()[:] = state.joint_qd.numpy()
        self.valid.numpy()[:] = 1

    def forget(self, worlds: np.ndarray) -> None:
        """A reset: these worlds' rows no longer continue anything."""
        self.valid.numpy()[np.asarray(worlds, dtype=np.int64)] = 0

    def restore(self, models: list[Any], datas: list[Any], state: Any, only: np.ndarray) -> np.ndarray:
        """Of the worlds ``only`` names (they differ from the route's last output), put the row back into each whose
        handed-in coordinates are its key bit for bit; returns those worlds."""
        pairs = [(now.numpy(), key.numpy(), None if world is None else world.numpy())
                 for now, key, world in ((state.joint_q, self.q, self.coord_world),
                                         (state.joint_qd, self.qd, self.dof_world))]
        worlds = np.flatnonzero(keyed(len(datas), only, pairs) & (self.valid.numpy() != 0))
        rows = self.rows.numpy()
        for w in worlds.tolist():
            mujoco.mj_setState(models[w], datas[w], rows[w], SPEC)
        return worlds


def install() -> None:
    """Register the route's state adapter (idempotent): the MuJoCo solver's arrays as registered for
    ``SolverMuJoCo`` (looked up at each call, so health's tracker arrays composed into it stay), then
    ``LiveState``'s."""
    from alloy._vendor import newton
    from alloy.core import registry
    from alloy.route.mjc_core import MjcCoreSolver

    if getattr(registry._REG.state.get(MjcCoreSolver), "_mjc_live", False):
        return
    base_cls = newton.solvers.SolverMuJoCo

    def arrays(solver: Any) -> dict[str, wp.array]:
        base = registry._REG.state.get(base_cls)
        out = dict(base(solver)) if base is not None else {}
        out.update(solver.live.arrays())
        return out

    def maps(solver: Any) -> dict[str, wp.array]:
        base = registry._REG.world_maps.get(base_cls)
        out = dict(base(solver)) if base is not None else {}
        out.update(solver.live.world_maps())
        return out

    arrays._mjc_live = True                       # type: ignore[attr-defined]
    registry.register_state_adapter(MjcCoreSolver, arrays, world_map=maps)


__all__ = ["LiveState", "PREFIX", "install"]
