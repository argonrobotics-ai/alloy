# SPDX-License-Identifier: Apache-2.0
"""The CPU route's coupled exchange windows: a restart pass continues each world from MuJoCo's float64 state at the
window start, as pass 0 did.

MuJoCo C keeps float64 ``qpos`` / ``qvel`` between calls. Newton's float32 state is pushed only into worlds whose
``joint_q`` / ``joint_qd`` differ from the route's last output (``NewtonViews.changed_worlds``). Pass 0 of an exchange
continues from float64. A restart pass (the coupler's ``iteration_restart``) hands back the window start, which
differs from the previous pass's output, so before this it was pushed from float32. A world that converged in pass 0
and is re-stepped frozen while a batch-mate iterates (``alloy.coupling.proxy_loop``, ``freeze.py``) then did not
reproduce its pass: its result depended on its batch-mates' pass counts. That broke D9 / CPL-9 in one process (4
worlds against their first 2) and CPU mode A's law (WP9's review of F16, a ball dropped into water: 1.7 mm after
0.3 s).

``WindowStart.save`` runs on the coupler's state distribution at a window start (a notify with state bits and no
restart). It keeps every world's float64 MuJoCo state (``mj_getState`` of ``SPEC``: MuJoCo's full physics state, its
warm start, the active equality flags and mocap poses; the route pushes ctrl and applied forces every call), and Newton's float32
state as handed in. ``restore`` runs on a
restart. It puts the float64 state back into every world whose handed-in state equals the saved one bit for bit, and
marks those worlds as the route's last output, so the step does not push float32 over them. A world that the window
start already pushed from float32 (its state differed from the last output when saved) is pushed again, the same way.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

# what mj_step reads from MjData besides what the route pushes every call (ctrl, qfrc_applied, xfrc_applied):
# time, qpos, qvel, act, delay history, plugin state (FULLPHYSICS), the warm start, eq_active and mocap (in MuJoCo's
# USER set in 3.16; WP2's review)
_S = mujoco.mjtState
SPEC = int(_S.mjSTATE_FULLPHYSICS | _S.mjSTATE_WARMSTART | _S.mjSTATE_EQ_ACTIVE | _S.mjSTATE_MOCAP_POS
           | _S.mjSTATE_MOCAP_QUAT)
STATE_BITS = 1 | 2 | 4 | 8          # newton.StateFlags JOINT_Q | JOINT_QD | BODY_Q | BODY_QD


def _host(a: Any) -> np.ndarray:
    return a.numpy() if hasattr(a, "numpy") else np.asarray(a)


def _get(m: Any, d: Any) -> np.ndarray:
    out = np.empty(mujoco.mj_stateSize(m, SPEC), dtype=np.float64)
    mujoco.mj_getState(m, d, out, SPEC)
    return out


def _same(now: np.ndarray, then: np.ndarray) -> np.ndarray:
    return (now == then) | (np.isnan(now) & np.isnan(then))


class WindowStart:
    """Each world's float64 MuJoCo state at the last window start (module docstring)."""

    def __init__(self, views: Any) -> None:
        self.views = views
        self.q: np.ndarray | None = None
        self.qd: np.ndarray | None = None
        self.rows: list[np.ndarray | None] = []

    def save(self, models: list[Any], datas: list[Any], state: Any) -> None:
        """At a window start: every world's MjData state, None for a world the next step pushes from float32."""
        self.q, self.qd = _host(state.joint_q).copy(), _host(state.joint_qd).copy()
        fresh = set(self.views.changed_worlds(state).tolist())
        self.rows = [None if w in fresh else _get(m, d) for w, (m, d) in enumerate(zip(models, datas))]

    def restore(self, models: list[Any], datas: list[Any], state: Any) -> np.ndarray:
        """At a restart: the saved float64 state back into each world whose handed-in state is the saved one; returns
        those worlds."""
        if self.q is None or self.views.last_q is None or self.views.coord_world is None:
            return np.empty(0, dtype=np.int64)
        now_q, now_qd = _host(state.joint_q), _host(state.joint_qd)
        if now_q.shape != self.q.shape or now_qd.shape != self.qd.shape:
            return np.empty(0, dtype=np.int64)
        moved = np.zeros(len(datas), dtype=bool)
        moved[np.unique(self.views.coord_world[~_same(now_q, self.q)])] = True
        if self.views.dof_world is not None:
            moved[np.unique(self.views.dof_world[~_same(now_qd, self.qd)])] = True
        worlds = np.array([w for w, row in enumerate(self.rows) if row is not None and not moved[w]], dtype=np.int64)
        for w in worlds.tolist():
            mujoco.mj_setState(models[w], datas[w], self.rows[w], SPEC)
        sel_q = np.isin(self.views.coord_world, worlds)
        self.views.last_q[sel_q] = now_q[sel_q]
        if self.views.dof_world is not None:
            sel_qd = np.isin(self.views.dof_world, worlds)
            self.views.last_qd[sel_qd] = now_qd[sel_qd]
        return worlds

    def clear(self) -> None:
        """Forget the window (a reset starts a new one)."""
        self.q = self.qd = None
        self.rows = []


__all__ = ["SPEC", "STATE_BITS", "WindowStart"]
