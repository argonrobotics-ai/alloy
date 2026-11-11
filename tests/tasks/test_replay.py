# SPDX-License-Identifier: Apache-2.0
"""Newton SolverMuJoCo replay of recorded traces (SPEC §7 WP13 acceptance item 2) and its building blocks."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

from alloy.tasks.mjb_convert import load_recorded, write_scene
from alloy.tasks.newton_map import body_index, initial_to_newton
from alloy.tasks.replay import ReplayConfig, import_scene, recorded_noslip, replay_scene
from alloy.tasks.trace import QPOS, QVEL, build_trace, load_trace, save_trace
from alloy.testing.tasks import REPO, TRACES

GATE_M = 1e-3


def test_initial_state_matches_mujoco_kinematics(synth_model: mujoco.MjModel, tmp_path: Path) -> None:
    m = synth_model
    d = mujoco.MjData(m)
    rng = np.random.default_rng(3)
    d.qpos[:] = m.qpos0 + rng.normal(0.0, 0.05, m.nq)
    for k in np.nonzero(m.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE))[0]:
        a = int(m.jnt_qposadr[k])
        d.qpos[a + 3 : a + 7] /= np.linalg.norm(d.qpos[a + 3 : a + 7])
    mujoco.mj_kinematics(m, d)
    from alloy.tasks.mjmodel_info import element_columns

    columns = {QPOS: element_columns(m, 0), QVEL: element_columns(m, 1)}
    model = import_scene(write_scene(m, tmp_path / "robots", "synth"), "cpu")
    q, qd = initial_to_newton(model, columns, d.qpos.copy(), np.zeros(m.nv))
    state = model.state()
    state.joint_q.assign(q.astype(np.float32))
    from alloy._vendor import newton

    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_q = state.body_q.numpy()
    for name, b in body_index(model).items():
        mb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        assert np.abs(body_q[b, :3] - d.xpos[mb]).max() < 1e-5, name
        dot = abs(float(np.dot(body_q[b, 3:], np.roll(d.xquat[mb], -1))))
        assert dot > 1.0 - 1e-6, name


def test_saved_trace_round_trips(synth_recording: Path, tmp_path: Path) -> None:
    tt = build_trace(load_recorded(synth_recording), "h", None)
    save_trace(tmp_path / "t", tt)
    back = load_trace(tmp_path / "t")
    assert back.trace.sha256 == tt.trace.sha256 and back.trace.columns == tt.trace.columns
    assert np.array_equal(back.ref.pos, tt.ref.pos) and back.ref.names == ("block", "wedge_obj")


def test_synthetic_replay_tracks_recording(synth_recording_level: Path, tmp_path: Path) -> None:
    rec = load_recorded(synth_recording_level)
    xml = write_scene(rec.model, tmp_path / "robots", "synth")
    tt = build_trace(rec, "h", None)
    r = replay_scene(xml, tt, ReplayConfig(seconds=1.0))
    assert r.steps == 300 and r.overflow == 0
    assert r.max_point_err < GATE_M  # 0.2 mm block (pair margin zeroed by Newton) and ~0.5 mm wedge, 2026-11-01


def _replay(name: str, seconds: float, **cfg: object):  # noqa: ANN202
    tt = load_trace(TRACES / name)
    return tt, replay_scene(REPO / tt.meta["scene"], tt, ReplayConfig(seconds=seconds, **cfg))


def test_recorded_noslip_comes_from_the_vendored_scene() -> None:
    assert recorded_noslip(load_trace(TRACES / "t11")) == (50, 1e-10)  # every robot recording ran noslip 50
    assert recorded_noslip(load_trace(TRACES / "bin10")) == (0, 1e-6)


def test_t12_short_replay_within_gate() -> None:
    _, r = _replay("t12", 0.2)
    assert r.overflow == 0 and r.settings["noslip_iterations"] == 50
    assert r.max_point_err < GATE_M
    assert r.pos_err.max() == pytest.approx(1e-4, abs=3e-5)  # margins zeroed by Newton (0.1 mm), see attribution


@pytest.mark.slow
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("name", ["t12", "t12pre", "t14", "t11", "h1"])
def test_two_second_replay_within_gate(name: str) -> None:
    _, r = _replay(name, 2.0)
    assert r.overflow == 0
    assert r.max_point_err < GATE_M, (name, r.max_point_err, r.first_exceed(GATE_M))


@pytest.mark.slow
@pytest.mark.timeout(3600)
def test_t11_fails_without_noslip() -> None:
    tt, r = _replay("t11", 2.0, noslip=0)  # 2026-11-01: 1.607 mm at 1.835 s; CPU MuJoCo noslip-off 1.626 mm
    assert tt.meta["offline"]["attribution"]["noslip_off"]["max_point_err_m"] > GATE_M
    assert r.max_point_err > GATE_M
