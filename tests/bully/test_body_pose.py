# SPDX-License-Identifier: Apache-2.0
"""SB-7 cutting / cloth classes (coordinator 2026-11-05): the ``body_pose`` action kind moves kinematic bodies on a
straight line to the command over each control period, and a target's own S1 counters (``bully_monitors``) reach
the classifier through ``EpisodeRunner``."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.bully.actions import HookedPoseActions, bind_actions, frame_from_poses
from alloy.bully.direct import DirectSim
from alloy.bully.replay import EpisodeRunner, target_monitors
from alloy.bully.severity import world_cause
from alloy.bully.types import ActionSpace, BullyLimits, BullyTarget
from alloy.core.errors import BullyError

SUBSTEPS, PERIOD = 4, 0.02


class _HoldSolver:
    """Integrates nothing: the next state is the current one (a collider-only body, as a VBD blade is)."""

    def step(self, s0, s1, control, contacts, dt) -> None:
        for name in ("body_q", "body_qd"):
            wp.copy(getattr(s1, name), getattr(s0, name))

    def reset(self, state, world_mask=None) -> None:
        pass


class _CountingSim(DirectSim):
    """A DirectSim reporting one target row, ``inverted_tets``: world w counts w per period."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.counts = np.zeros(self.model.world_count)

    def step_period(self) -> None:
        super().step_period()
        self.counts += np.arange(self.model.world_count)

    def reset(self, world_mask=None) -> None:
        super().reset(world_mask)
        self.counts[:] = 0.0

    def bully_monitors(self) -> dict[str, np.ndarray]:
        return {"inverted_tets": self.counts.copy()}


def _model(worlds: int) -> newton.Model:
    b = newton.ModelBuilder()
    b.add_body(xform=wp.transform(wp.vec3(0.1, 0.2, 0.3), wp.quat_rpy(0.0, 0.0, 0.5)), label="blade")
    s = newton.ModelBuilder()
    s.replicate(b, worlds)
    return s.finalize(device="cpu")


def _sim(worlds: int, cls=DirectSim) -> DirectSim:
    return cls(_model(worlds), _HoldSolver(), control_dt=PERIOD, substeps=SUBSTEPS, collide=False, health=False)


def test_body_pose_moves_on_a_line_with_the_period_velocity_and_keeps_orientation():
    sim = _sim(2)
    q0 = sim.state.body_q.numpy().copy()
    trace = np.array([[[0.11, 0.2, 0.32], [0.12, 0.2, 0.34]], [[0.1, 0.18, 0.3], [0.1, 0.16, 0.3]]], np.float32)
    seen: list[np.ndarray] = []
    sim.add_step_observer(lambda s, n, c, k, dt: seen.append(s.body_q.numpy()[:, :3].copy()))
    bound = bind_actions(sim, sim.model, "body_pose", (0, 1, 2), trace)
    assert isinstance(bound, HookedPoseActions)
    bound.apply(0)
    sim.step_period()
    for k in range(SUBSTEPS):  # before substep k + 1 of 4: start + (k + 1) / 4 (command - start)
        np.testing.assert_allclose(seen[k], q0[:, :3] + (k + 1) / SUBSTEPS * (trace[:, 0] - q0[:, :3]), atol=1e-6)
    qd = sim.state.body_qd.numpy()
    np.testing.assert_allclose(qd[:, :3], (trace[:, 0] - q0[:, :3]) / PERIOD, atol=1e-4)
    np.testing.assert_allclose(qd[:, 3:], 0.0)
    np.testing.assert_allclose(sim.state.body_q.numpy()[:, 3:], q0[:, 3:], atol=1e-7)  # orientation kept
    bound.apply(1)
    sim.step_period()
    np.testing.assert_allclose(sim.state.body_q.numpy()[:, :3], trace[:, 1], atol=1e-6)
    np.testing.assert_allclose(sim.state.body_qd.numpy()[:, :3], (trace[:, 1] - trace[:, 0]) / PERIOD, atol=1e-4)
    bound.detach()
    assert bound._write not in sim.pre_substep


def test_body_pose_addresses_single_components():
    sim = _sim(1)
    q0 = sim.state.body_q.numpy().copy()
    bound = bind_actions(sim, sim.model, "body_pose", (2,), np.array([[[0.25]]], np.float32))  # z only
    bound.apply(0)
    sim.step_period()
    np.testing.assert_allclose(sim.state.body_q.numpy()[0, :3], [q0[0, 0], q0[0, 1], 0.25], atol=1e-6)
    with pytest.raises(BullyError):
        bound.apply(1)


def test_target_monitors_reach_the_classifier_and_reset():
    worlds, ticks = 3, 2
    target = BullyTarget(name="pose_counter", scene=None, make_sim=lambda w, d: _sim(w, _CountingSim),
                         actions=ActionSpace(low=np.zeros(3), high=np.full(3, 0.5), rate_hz=1.0 / PERIOD,
                                             max_delta_per_tick=np.full(3, 0.01), initial=np.array([0.1, 0.2, 0.3])),
                         action_kind="body_pose", action_index=(0, 1, 2), robot_bodies=(0,))
    runner = EpisodeRunner(target, worlds, "cpu", BullyLimits())
    trace = np.broadcast_to(np.array([0.1, 0.2, 0.3]), (ticks, 3))
    res = runner.run(trace)
    assert res.severity.tolist() == [0, 1, 1]
    assert res.monitors["inverted_tets"].tolist() == [0.0, 2.0, 4.0]
    assert world_cause(res.health, res.monitors, BullyLimits(), 2)[0] == "inverted_tets"
    assert runner.sim.pre_substep == []  # the pose hook is detached after the episode
    assert runner.run(trace).monitors["inverted_tets"].tolist() == [0.0, 2.0, 4.0]  # counts restart on reset


def test_target_monitors_absent_on_plain_sims():
    assert target_monitors(_sim(1)) == {}


class _FramedSim(DirectSim):
    """A DirectSim with a per-world action frame (cut_block_free's blade frame): world 1 turned 90 deg about z."""

    AXES = np.array([np.eye(3), [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])

    def bully_action_frame(self):
        start = self._initial["body_q"].numpy()[:, :3].astype(np.float64)
        return start, self.AXES[: self.model.world_count]


def test_framed_body_pose_commands_land_in_world_frame():
    sim = _sim(2, _FramedSim)
    origin, axes = sim.bully_action_frame()
    c = np.array([[0.004, 0.002, -0.001]], np.float32)
    trace = np.stack([c, c])  # the same frame command in both worlds
    bound = bind_actions(sim, sim.model, "body_pose", (0, 1, 2), trace)
    bound.apply(0)
    sim.step_period()
    want = origin + np.einsum("k,wkj->wj", c[0].astype(np.float64), axes)
    np.testing.assert_allclose(sim.state.body_q.numpy()[:, :3], want, atol=1e-6)
    assert not np.allclose(want[0] - origin[0], want[1] - origin[1])  # world 1 moved along its own turned axes
    with pytest.raises(BullyError):  # a frame addresses one body's three components in order
        bind_actions(sim, sim.model, "body_pose", (0, 1), trace[:, :, :2])


def test_face_normal_band_is_held_in_the_blade_frame():
    """Coordinator ruling 2026-11-06T14:17Z: the legal box in the blade frame holds face-normal motion to +-1 mm."""
    worlds, band = 2, 1.0e-3
    target = BullyTarget(name="framed_band", scene=None, make_sim=lambda w, d: _sim(w, _FramedSim),
                         actions=ActionSpace(low=np.array([-band, -0.02, 0.0]), high=np.array([band, 0.02, 0.02]),
                                             rate_hz=1.0 / PERIOD, max_delta_per_tick=np.full(3, 0.01),
                                             initial=np.zeros(3)),
                         action_kind="body_pose", action_index=(0, 1, 2), robot_bodies=(0,))
    runner = EpisodeRunner(target, worlds, "cpu", BullyLimits())
    origin, axes = runner.sim.bully_action_frame()
    runner.run(np.array([[0.005, 0.0, 0.003]]))  # asks 5 mm along the face normal: legalised to the 1 mm band
    moved = runner.sim.state.body_q.numpy()[:, :3] - origin
    normal = np.einsum("wj,wj->w", moved, axes[:, 0])
    np.testing.assert_allclose(normal, band, atol=1e-6)
    np.testing.assert_allclose(np.einsum("wj,wj->w", moved, axes[:, 2]), 0.003, atol=1e-6)


def test_frame_from_poses_reads_the_blade_convention():
    from alloy.testing.topology import blade_pose

    yaw = 0.4
    edge, down = np.array([-np.sin(yaw), np.cos(yaw), 0.0]), np.array([0.1, 0.0, -1.0]) / np.hypot(0.1, 1.0)
    pose = blade_pose(np.array([0.01, 0.02, 0.03]), edge, down)
    origin, axes = frame_from_poses(pose[None])
    np.testing.assert_allclose(origin[0], [0.01, 0.02, 0.03], atol=1e-7)
    np.testing.assert_allclose(axes[0, 1], edge, atol=1e-6)  # edge
    side = down - np.dot(down, edge) * edge
    np.testing.assert_allclose(axes[0, 2], side / np.linalg.norm(side), atol=1e-6)  # cutting side, square to the edge
    np.testing.assert_allclose(axes[0, 0], np.cross(axes[0, 1], -axes[0, 2]), atol=1e-6)  # face normal = y x z
    np.testing.assert_allclose(axes[0] @ axes[0].T, np.eye(3), atol=1e-6)
