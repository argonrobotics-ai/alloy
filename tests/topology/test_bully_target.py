# SPDX-License-Identifier: Apache-2.0
"""SB-7 bully targets ``cut_block_stroke`` and ``cut_block_free`` (alloy/topology/bully_target.py), CPU.

The pose writer here mirrors WP7's ``HookedPoseActions`` (a pre_substep hook): before substep k of n the blade is at
start + k/n (command - start), with linear velocity (command - start) / control_dt.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.bully.replay import EpisodeRunner
from alloy.bully.targets import _TARGETS, get_target
from alloy.bully.types import BullyLimits
from alloy.topology import bully_target as bt
from alloy.topology.blade_contact import quat_rotate

WORLDS = 2


class _PoseHook:
    def __init__(self, sim, initial: np.ndarray) -> None:
        self.sim, self.blade = sim, np.asarray(sim.blades.numpy())
        self.cmd = np.tile(np.asarray(initial, float), (sim.model.world_count, 1))
        self.start, self.k = None, 0
        sim.pre_substep.append(self)

    def __call__(self, s) -> None:
        if self.k == 0:
            self.start = s.body_q.numpy()[self.blade, :3].copy()
        bq, bqd = s.body_q.numpy(), s.body_qd.numpy()
        bq[self.blade, :3] = self.start + (self.k / self.sim.substeps) * (self.cmd - self.start)
        bqd[self.blade] = 0.0
        bqd[self.blade, :3] = (self.cmd - self.start) / self.sim.control_dt
        s.body_q.assign(bq)
        s.body_qd.assign(bqd)
        self.k = (self.k + 1) % self.sim.substeps


def _stroke(worlds: int = WORLDS, dx: tuple[float, ...] = (), monitor: bool = False):
    """The nominal stroke: straight down the seam at the rate limit, then 3 periods at the bottom. ``dx`` [m] moves
    the worlds' blades off the seam in the first period; ``monitor`` reads the rows every period, as the runner does."""
    target = bt.cut_block_stroke()
    acts = target.actions
    sim = target.make_sim(worlds, "cpu")
    hook = _PoseHook(sim, acts.initial)
    hook.cmd[:len(dx), 0] += np.asarray(dx)
    x0 = sim.state.particle_q.numpy().copy()
    while hook.cmd[0, 2] > acts.low[2] + 1e-12:
        hook.cmd[:, 2] = np.maximum(hook.cmd[:, 2] - acts.max_delta_per_tick[2], acts.low[2])
        sim.step_period()
        if monitor:
            sim.bully_monitors()
    for _ in range(3):
        sim.step_period()
    return sim, x0


def _cut(sim) -> np.ndarray:
    return sim.ctl.cut_done.numpy().reshape(sim.model.world_count, -1).sum(axis=1)


def test_cut_block_stroke_is_registered_with_a_body_pose_action() -> None:
    assert "cut_block_stroke" in _TARGETS
    t = bt.cut_block_stroke()
    a = t.actions
    assert (t.action_kind, t.action_index, t.robot_bodies) == ("body_pose", (0, 1, 2), (0,))
    np.testing.assert_allclose(a.initial, bt.start_origin())
    assert np.all(a.low <= a.initial) and np.all(a.initial <= a.high)
    assert a.high[0] - a.low[0] == pytest.approx(2.0 * bt.SEAM_BAND)                 # x stays on the seam
    np.testing.assert_allclose(a.max_delta_per_tick, bt.SPEED * bt.CONTROL_DT)


@pytest.fixture(scope="module")
def stroked():
    return _stroke()


def test_the_nominal_seam_stroke_cuts_and_reads_clean(stroked) -> None:
    sim, _ = stroked
    report = sim.health.poll()
    assert np.all(report.flags == 0), report.flags                                      # SC5-3's stroke, no S1
    assert np.all(_cut(sim) > 0)
    assert np.isfinite(sim.state.particle_q.numpy()).all()
    rows = sim.bully_monitors()
    assert set(rows) == {"cut_pool_overflow", "inverted_tets", "cut_face_deviation"}
    assert all(v.shape == (WORLDS,) for v in rows.values())
    assert all(np.all(rows[k] == 0) for k in ("cut_pool_overflow", "inverted_tets"))
    assert np.all(rows["cut_face_deviation"] <= 1e-6)                                  # opened in the swept plane
    blade_z = sim.state.body_q.numpy()[sim.blades.numpy(), 2]
    np.testing.assert_allclose(blade_z, bt.cut_block_stroke().actions.low[2], atol=1e-6)


def test_reset_restores_the_cut_state_of_the_masked_worlds_only() -> None:
    sim, x0 = _stroke()
    cut = _cut(sim)
    sim.reset(np.array([1, 0]))
    after = _cut(sim)
    assert after[0] == 0 and after[1] == cut[1] > 0
    q = sim.state.particle_q.numpy().reshape(WORLDS, -1, 3)
    np.testing.assert_array_equal(q[0], x0.reshape(WORLDS, -1, 3)[0])
    sim.reset()
    assert _cut(sim).sum() == 0
    np.testing.assert_array_equal(sim.state.particle_q.numpy(), x0)
    assert int(sim.ctl.bonds.intact.numpy().sum()) == int(sim.ctl.bonds.count)


def test_an_inverted_tet_is_counted_for_its_world_until_reset() -> None:
    sim = bt.cut_block_stroke().make_sim(WORLDS, "cpu")
    m = sim.model
    tet = int(np.flatnonzero(m.particle_world.numpy()[m.tet_indices.numpy()[:, 0]] == 1)[0])
    a, b, c, d = m.tet_indices.numpy()[tet]
    q = sim.state.particle_q.numpy()
    q[d] = q[a] - (q[d] - q[a])                                                       # mirror one vertex: det F < 0
    sim.state.particle_q.assign(q)
    flipped = sim.bully_monitors()["inverted_tets"]
    assert flipped[0] == 0 and flipped[1] >= 1                                         # every tet on that vertex
    sim.state.particle_q.assign(m.particle_q)                                       # back to rest
    np.testing.assert_array_equal(sim.bully_monitors()["inverted_tets"], flipped)     # the episode peak stays
    sim.reset()
    np.testing.assert_array_equal(sim.bully_monitors()["inverted_tets"], [0, 0])


def test_a_stroke_off_the_seam_reads_its_offset_as_the_face_deviation() -> None:
    """World 1's blade runs 0.8 mm off the x = 10 mm seam: it opens seam twins, whose rest points lie 0.8 mm from
    the plane its edge swept. World 0 cuts in the seam plane."""
    sim, _ = _stroke(dx=(0.0, 8.0e-4), monitor=True)
    assert np.all(_cut(sim) > 0)
    dev = sim.bully_monitors()["cut_face_deviation"]
    assert dev[0] <= 1e-6
    assert dev[1] == pytest.approx(8.0e-4, abs=1e-6)
    sim.reset(np.array([1, 0]))
    np.testing.assert_array_equal(sim.bully_monitors()["cut_face_deviation"], [0.0, dev[1]])


def test_cut_block_free_is_registered_with_a_face_first_band_in_the_blade_frame() -> None:
    assert "cut_block_free" in _TARGETS
    t = bt.cut_block_free()
    a = t.actions
    assert (t.action_kind, t.action_index, t.robot_bodies) == ("body_pose", (0, 1, 2), (0,))
    np.testing.assert_allclose([a.low[0], a.high[0]], [-1.0e-3, 1.0e-3])                  # ruling 2026-11-06T14:17Z
    np.testing.assert_array_equal(a.initial, 0.0)
    sim = t.make_sim(4, "cpu")
    origin, axes = sim.bully_action_frame()
    q = sim.state.body_q.numpy()[sim.blades.numpy()].astype(np.float64)
    np.testing.assert_allclose(origin, q[:, :3], atol=1e-7)
    for k, body_axis in enumerate(([1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0])):              # face normal, edge, cut side
        np.testing.assert_allclose(axes[:, k], quat_rotate(q[:, 3:7], np.array(body_axis)), atol=1e-6)
    bottom = origin[:, 2] - bt.EDGE_OFFSET                                                   # edge height at the start
    assert np.all(a.high[2] * -axes[:, 2, 2] >= bottom + 3.0e-3 - 1e-9)                    # 3 mm past the bottom
    assert bt.cut_block_stroke().make_sim(1, "cpu").bully_action_frame() is None             # world frame


def test_free_angles_are_bounded_and_belong_to_the_world_not_the_batch() -> None:
    ang = bt.free_angles(64)
    assert np.all(np.abs(ang[:, 0]) <= bt.FREE_YAW) and np.all(np.abs(ang[:, 1]) <= bt.FREE_PITCH)
    assert np.ptp(ang[:, 0]) > bt.FREE_YAW and np.ptp(ang[:, 1]) > bt.FREE_PITCH                  # spread out
    np.testing.assert_array_equal(bt.free_angles(1, first_world=37)[0], ang[37])
    np.testing.assert_allclose(bt.free_pose(bt.start_origin(), 0.0, 0.0)[3:],                   # = the stroke's pose
                               [0.0, 0.0, 0.0, 1.0], atol=1e-7)


def test_free_blades_have_their_angles_and_the_block_a_seam_on_every_cell_face() -> None:
    t = bt.cut_block_free()
    sim = t.make_sim(3, "cpu")
    per_world = np.bincount(sim.model.particle_world.numpy(), minlength=3)
    np.testing.assert_array_equal(per_world, 8 * np.prod(bt.CELLS))                       # every cell its own corners
    q = sim.state.body_q.numpy()[sim.blades.numpy()].astype(np.float64)
    edge = np.diff(sim.sweep.edges_at(sim.state.body_q.numpy()), axis=1)[:, 0]
    ang = bt.free_angles(3)
    np.testing.assert_allclose(edge / np.linalg.norm(edge, axis=1, keepdims=True),
                               np.stack([-np.sin(ang[:, 0]), np.cos(ang[:, 0]), 0 * ang[:, 0]], 1), atol=1e-6)
    up = quat_rotate(q[:, 3:7], np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(up[:, 2], np.cos(ang[:, 1]), atol=1e-6)                    # the lean
    alone = bt.cut_block_free(first_world=2).make_sim(1, "cpu")
    np.testing.assert_allclose(alone.state.body_q.numpy()[alone.blades.numpy()], q[2:3], atol=0)


def test_a_free_descent_runs_through_the_bully_runner_with_the_deviation_row() -> None:
    t = bt.cut_block_free()
    a = t.actions
    acts = np.tile(a.initial, (30, 1))
    acts[:, 2] = np.minimum(a.max_delta_per_tick[2] * np.arange(1, 31), a.high[2])           # along the cutting side
    runner = EpisodeRunner(t, 2, "cpu", BullyLimits())
    res = runner.run(acts)
    dev = res.monitors["cut_face_deviation"]
    opened = [int(((runner.sim.ctl.bonds.intact.numpy() == 0) & (runner.sim.ctl.bonds.world.numpy() == w)).sum())
              for w in range(2)]
    assert min(opened) > 0
    assert np.all(np.isfinite(dev)) and np.all(dev > 0.0)                                   # off-lattice cuts
    assert res.severity.shape == (2,)


def test_cut_block_stroke_cut_faces_collide_and_free_keeps_its_solver() -> None:
    """cut_block_stroke runs VBD self-contact with the seam filter (broken halves collide, welded twins do not);
    cut_block_free is unchanged until it is measured."""
    stroke = bt.cut_block_stroke().make_sim(1, "cpu")
    assert stroke.solver.particle_enable_self_contact
    assert stroke.ctl.seam_contact is not None
    assert "topology.contact_released" in stroke.ctl.state_arrays()                     # rolled back with the cut
    assert bt.cut_block_stroke().name == "cut_block_stroke"
    assert bt.cut_block_stroke(self_contact=False).name == "cut_block_stroke_noself"   # not the class's records
    assert stroke.ctl.config.cut_alignment == bt.CUT_ALIGNMENT                          # the blade cuts its own seam
    for name, self_contact in (("cut_block_stroke_anyseam", True), ("cut_block_stroke_noself_anyseam", False)):
        old = get_target(name)                                                           # the rule before, registered
        assert old.name == name == bt.cut_block_stroke(self_contact=self_contact, cut_alignment=None).name
    assert get_target("cut_block_stroke_anyseam").make_sim(1, "cpu").ctl.config.cut_alignment is None
    assert stroke.blade_contact.config.reentry                                          # and its re-entry case
    old = get_target("cut_block_stroke_noreentry")
    assert old.name == "cut_block_stroke_noreentry" and not old.make_sim(1, "cpu").blade_contact.config.reentry
    free = bt.cut_block_free().make_sim(1, "cpu")
    assert not getattr(free.solver, "particle_enable_self_contact", False)
    assert free.ctl.seam_contact is None
    assert free.ctl.config.cut_alignment == bt.CUT_ALIGNMENT and get_target("cut_block_free_anyseam").name == \
        "cut_block_free_anyseam" == bt.cut_block_free(cut_alignment=None).name


def _replay(target, worlds: tuple[int, ...]) -> tuple[EpisodeRunner, np.ndarray, dict[int, tuple[int, int]]]:
    """SB-7's seed-100000 batch worlds, each replayed alone on CPU: (severity, first-finding tick) per world."""
    from alloy.bully.actions import legalize
    from alloy.bully.stress import random_legal_traces

    sp = target.actions
    traces = random_legal_traces(target, 2048, 2.0, seed=100000)
    legal = legalize(np.asarray(traces, dtype=np.float64), sp.low, sp.high, sp.max_delta_per_tick, sp.initial).trace
    runner = EpisodeRunner(target, 1, "cpu", BullyLimits())
    out = {}
    for w in worlds:
        r = runner.run(legal[w:w + 1])
        out[w] = (int(r.severity[0]), int(r.first_step[0]))
    return runner, legal, out


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(1500)
def test_the_alignment_gate_clears_the_junction_s1_worlds_on_the_replays() -> None:
    """Worlds 3, 6, 10 and 11 of SB-7's seed-100000 batch: under the rule before (``cut_block_stroke_anyseam``) the
    edge also broke the y seam's bonds on the line where the seams meet, and that loose corner twin crept into the
    neighbouring piece (PEN_FAIL, the soft-in-soft check; 15 of 16 failures in worlds 0-59, CPU, 2026-11-09). With the
    gate they are clean. The class calibration: S1 615 -> 86 of 2,048 (one L4 each, f690c5e48's two arms).
    The bounds leave one world of slack each way: a replay is chaotic across machines (world 3 read clean under any
    seam on an n2 VM and S1 on host-1), and the gated class still fails about 4 % of episodes for other reasons."""
    worlds = (3, 6, 10, 11)
    _, _, before = _replay(get_target("cut_block_stroke_anyseam"), worlds)
    _, _, after = _replay(bt.cut_block_stroke(), worlds)
    s1 = [sum(sev == 1 for sev, _ in arm.values()) for arm in (before, after)]
    assert s1[0] >= 3 and s1[1] <= 1, f"any seam {before}, gated {after}"


@pytest.mark.law
@pytest.mark.compile_heavy
@pytest.mark.timeout(1500)
def test_colliding_cut_faces_clear_s1_worlds_on_the_replays() -> None:
    """SB-7's seed-100000 batch, worlds 1, 3, 5 and 8 (S1 on main and with gap #9's face plane), each replayed alone
    as WP15's triage does: with the cut faces colliding, fewer of them are S1 than without, and at most two (CPU,
    2026-11-08: 4 -> 2 on main 2df666b30, 4 -> 2 with the face plane 819e12b3d, 4 -> 1 with plane + break rule).
    World 1 replayed again after the runner's resets gives the same severity and first-finding tick (VBD self-contact
    keeps detection buffers and a friction history: an episode's S1 must not depend on the one before it). Both arms
    run the cutter those numbers were measured with (``cut_alignment=None``, before the alignment gate)."""
    _, _, before = _replay(bt.cut_block_stroke(self_contact=False, cut_alignment=None), (1, 3, 5, 8))
    runner, legal, after = _replay(bt.cut_block_stroke(cut_alignment=None), (1, 3, 5, 8))
    s1_before = [w for w, (sev, _) in before.items() if sev == 1]
    s1_after = [w for w, (sev, _) in after.items() if sev == 1]
    assert len(s1_after) < len(s1_before) and len(s1_after) <= 2, f"S1 without {s1_before}, with {s1_after}"
    again = runner.run(legal[1:2])            # world 1 after the runner's resets: the self-contact state does not leak
    assert (int(again.severity[0]), int(again.first_step[0])) == after[1]
