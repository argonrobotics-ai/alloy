# SPDX-License-Identifier: Apache-2.0
"""SolverMuJoCo that pushes Newton coordinates into MuJoCo Warp only for worlds whose Newton state changed.

Newton's SolverMuJoCo copies `state_in.joint_q/joint_qd` into MuJoCo Warp's qpos/qvel at the start of every step and
back at the end (newton@ac7f7830 solver_mujoco.py:5021, kernels.py:927-1007: free joints rotate the world-frame angular
velocity into the body frame and back). In float32 that round trip is not exact: a box tumbling at 6.2 rad/s
(WP6's A4) loses |ΔL|/L = 1.28e-3 over 10 s at dt 1.25 ms under rk4, 2.57e-3 at 0.625 ms (linear in steps), against
8.1e-6 when MuJoCo Warp keeps its own state (`update_data_interval=0`) and 3.8e-6 for MuJoCo C in float64
([internal note], a4_precision.py).

`SyncedMuJoCo` runs with `update_data_interval=0` and does the push itself, masked: a world is pushed when any of its
coordinates differs (bitwise) from what the solver wrote at its last pull, so every external write (reset, rollback,
WP5's latch, WP10 topology, a world rebase, a test setting velocities) still reaches MuJoCo, and untouched worlds keep
MuJoCo's own state. The first step after construction or `notify_model_changed` pushes every world. All of it is
device work (graph-capturable). The MuJoCo C path (`use_mujoco_cpu`) keeps Newton's own push. Used for rk4 entries
(SCR-WP6-4, `integrator_route`).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

log = logging.getLogger("alloy.scene")


@wp.kernel
def _mark_changed(now: wp.array(dtype=wp.float32), last: wp.array(dtype=wp.float32),
                  world_of: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.bool)):
    i = wp.tid()
    w = world_of[i]
    if w >= 0 and now[i] != last[i]:
        mask[w] = True


@wp.kernel
def _reset_mask(primed: wp.array(dtype=wp.int32), mask: wp.array(dtype=wp.bool)):
    mask[wp.tid()] = primed[0] == 0


@wp.kernel
def _set_primed(primed: wp.array(dtype=wp.int32)):
    primed[0] = 1


def _index_worlds(starts: np.ndarray, count: int, joint_world: np.ndarray) -> np.ndarray:
    """World of every coordinate (or dof) from the per-joint start offsets (-1: not in a world)."""
    out = np.full(count, -1, np.int32)
    n = len(joint_world)
    bounds = np.append(np.asarray(starts[:n], np.int64), count)
    for j in range(n):
        out[bounds[j]:bounds[j + 1]] = joint_world[j]
    return out


class SyncedMuJoCo(newton.solvers.SolverMuJoCo):
    """SolverMuJoCo with a masked, change-driven Newton -> MuJoCo Warp push (module docstring)."""

    def __init__(self, model: newton.Model, **kw: Any) -> None:
        kw["update_data_interval"] = 0
        super().__init__(model, **kw)
        dev = model.device
        jw = model.joint_world.numpy().astype(np.int32)
        qs, ds = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        self._coord_world = wp.array(_index_worlds(qs, model.joint_coord_count, jw), dtype=wp.int32, device=dev)
        self._dof_world = wp.array(_index_worlds(ds, model.joint_dof_count, jw), dtype=wp.int32, device=dev)
        self._last_q = wp.zeros(model.joint_coord_count, dtype=wp.float32, device=dev)
        self._last_qd = wp.zeros(model.joint_dof_count, dtype=wp.float32, device=dev)
        self._mask = wp.zeros(max(model.world_count, 1), dtype=wp.bool, device=dev)
        self._primed = wp.zeros(1, dtype=wp.int32, device=dev)

    def notify_model_changed(self, flags: int) -> None:
        """A model change (anchors, randomized placements) may move a pose without touching joint_q: push all."""
        super().notify_model_changed(flags)
        if hasattr(self, "_primed"):            # SolverMuJoCo.__init__ may notify before our arrays exist
            self._primed.zero_()

    def step(self, state_in, state_out, control, contacts, dt: float, **kw: Any) -> None:
        if self.use_mujoco_cpu:
            self._update_mjc_data(self.mj_data, self.model, state_in)
            super().step(state_in, state_out, control, contacts, dt, **kw)
            return
        dev = self.model.device
        wp.launch(_reset_mask, dim=len(self._mask), inputs=[self._primed], outputs=[self._mask], device=dev)
        if self.model.joint_coord_count:
            wp.launch(_mark_changed, dim=self.model.joint_coord_count,
                      inputs=[state_in.joint_q, self._last_q, self._coord_world], outputs=[self._mask], device=dev)
        if self.model.joint_dof_count:
            wp.launch(_mark_changed, dim=self.model.joint_dof_count,
                      inputs=[state_in.joint_qd, self._last_qd, self._dof_world], outputs=[self._mask], device=dev)
        with wp.ScopedDevice(dev), self._scoped_mujoco_warp_execution():
            self._update_mjc_data(self.mjw_data, self.model, state_in, world_mask=self._mask)
        super().step(state_in, state_out, control, contacts, dt, **kw)
        wp.copy(self._last_q, state_out.joint_q)
        wp.copy(self._last_qd, state_out.joint_qd)
        wp.launch(_set_primed, dim=1, outputs=[self._primed], device=dev)


__all__ = ["SyncedMuJoCo"]
