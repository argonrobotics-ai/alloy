# SPDX-License-Identifier: Apache-2.0
"""The CPU route's exchange window (``alloy.route.mjc_core_window``): a restart pass puts back each world's float64
MuJoCo state from the window start, where the handed-in state is the window start bit for bit, and marks it as the
route's last output so the step does not push float32 over it (WP9's review of F16: a world re-stepped frozen while a
batch-mate iterates must reproduce its pass). Unit level: real MjData, a stand-in for the route's views."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from alloy.route.mjc_core_window import SPEC, WindowStart

pytestmark = pytest.mark.cpu_route

XML = """<mujoco><worldbody><body pos="0 0 .2"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body>
</worldbody></mujoco>"""
W = 3


class Views:
    """Three worlds of one free body: 7 coordinates and 6 dofs each; ``changed`` is what changed_worlds returns."""

    def __init__(self) -> None:
        self.coord_world = np.repeat(np.arange(W), 7)
        self.dof_world = np.repeat(np.arange(W), 6)
        self.last_q = np.zeros(7 * W, np.float32)
        self.last_qd = np.zeros(6 * W, np.float32)
        self.changed = np.empty(0, np.int64)

    def changed_worlds(self, state):
        return self.changed


FIELDS = ("qpos", "qvel", "act", "qacc_warmstart", "mocap_pos", "mocap_quat", "plugin_state", "eq_active")


def _setup():
    m = mujoco.MjModel.from_xml_string(XML)
    datas = [mujoco.MjData(m) for _ in range(W)]
    for w, d in enumerate(datas):
        d.qpos[:] = [0.1 * w, 0, 0.2, 1, 0, 0, 0]
        d.qpos[0] += 1e-12                                 # float64 bits that float32 cannot carry
        d.qvel[:] = [0, 0, -1.0 - w, 0.3, 0, 0]
        d.time = 0.5
    state = SimpleNamespace(joint_q=np.concatenate([d.qpos for d in datas]).astype(np.float32),
                            joint_qd=np.concatenate([d.qvel for d in datas]).astype(np.float32))
    return [m] * W, datas, state


def test_the_saved_state_is_mujocos_full_physics_warm_start_and_equalities():
    import mujoco as mj

    S = mj.mjtState
    for bit in (S.mjSTATE_TIME, S.mjSTATE_QPOS, S.mjSTATE_QVEL, S.mjSTATE_ACT, S.mjSTATE_HISTORY, S.mjSTATE_MOCAP_POS,
                S.mjSTATE_MOCAP_QUAT, S.mjSTATE_PLUGIN, S.mjSTATE_WARMSTART, S.mjSTATE_EQ_ACTIVE):
        assert SPEC & int(bit), bit


def test_a_restart_puts_back_the_window_start_in_float64():
    models, datas, state = _setup()
    views = Views()
    views.changed = np.array([2])                          # world 2 is pushed from float32 at this window start
    win = WindowStart(views)
    win.save(models, datas, state)
    start = [{f: np.copy(getattr(d, f)) for f in FIELDS} for d in datas]
    for d in datas:                                        # pass 0 moves every world on
        d.qpos[0] += 0.01
        d.qvel[2] -= 0.01
        d.time = 0.51
    views.last_q[:] = 7.0                                  # the route's last output: pass 0's
    restored = win.restore(models, datas, state)
    assert restored.tolist() == [0, 1]
    for w in (0, 1):
        assert all(np.array_equal(getattr(datas[w], f), start[w][f]) for f in FIELDS) and datas[w].time == 0.5
        assert np.array_equal(views.last_q[7 * w:7 * w + 7], state.joint_q[7 * w:7 * w + 7])
    assert datas[2].time == 0.51 and np.all(views.last_q[14:] == 7.0)   # world 2: the step pushes float32 again


def test_a_world_whose_handed_in_state_moved_is_not_restored():
    models, datas, state = _setup()
    views = Views()
    win = WindowStart(views)
    win.save(models, datas, state)
    moved = SimpleNamespace(joint_q=state.joint_q.copy(), joint_qd=state.joint_qd.copy())
    moved.joint_qd[6 + 2] += 1.0                           # world 1's velocity is not the window start's
    assert win.restore(models, datas, moved).tolist() == [0, 2]
    win.clear()
    assert win.restore(models, datas, state).size == 0              # after a reset: nothing to put back
