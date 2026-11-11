# SPDX-License-Identifier: Apache-2.0
"""VNodeCutter rollback state (alloy/topology/vnode_state.py; WP4 review of item 9 step 3, item 1), CPU.

A WorldSnapshot with the cutter's ``state_arrays`` / ``state_worlds`` as extras, saved before a stroke's first split
and restored after it, re-runs the stroke bit for bit: splits, model arrays and particle state. Restoring one world
leaves the other's splits as they are.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.core.snapshot import WorldSnapshot
from alloy.testing.topology import blade_pose
from alloy.topology.vnode_rig import VNodeSim, vnode_block

X_CUT = 8.3e-3                 # [m] as test_vnode_cut: 3.3 mm and 1.7 mm from the 5 mm grid planes
DT, SUBSTEPS = 0.016, 16


class _Descent:
    """Every world's blade straight down at 0.05 m/s (pose and velocity per substep), as test_vnode_cut's hook."""

    def __init__(self, sim: VNodeSim) -> None:
        self.sim, self.bodies = sim, np.asarray(sim.scene.blade_bodies)
        self.z0 = self.z1 = sim.scene.size[2] + 2.0e-3 + 0.02
        self.k = 0
        sim.pre_substep.append(self)

    def period(self) -> None:
        self.z0, self.z1 = self.z1, max(self.z1 - 0.05 * DT, -3.0e-3 + 0.02)
        self.sim.step_period()

    def __call__(self, s) -> None:
        bq, bqd = s.body_q.numpy(), s.body_qd.numpy()
        bq[self.bodies, 2] = self.z0 + (self.k / SUBSTEPS) * (self.z1 - self.z0)
        bqd[self.bodies] = 0.0
        bqd[self.bodies, 2] = (self.z1 - self.z0) / DT
        s.body_q.assign(bq)
        s.body_qd.assign(bqd)
        self.k = (self.k + 1) % SUBSTEPS


def _stroke():
    sc = vnode_block("cpu", worlds=2)
    m = sc.model
    m.soft_contact_ke = 3.0e3
    q, qs = m.joint_q.numpy(), m.joint_q_start.numpy()
    for w in range(2):
        q[qs[w]:qs[w] + 7] = blade_pose(np.array([X_CUT, 0.5 * sc.size[1], sc.size[2] + 2.0e-3 + 0.02]),
                                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]))
    m.joint_q.assign(q)
    sim = VNodeSim(sc, newton.solvers.SolverVBD(m, iterations=4), control_dt=DT, substeps=SUBSTEPS)
    return sim, _Descent(sim)


def _record(sim) -> dict[str, np.ndarray]:
    out = {k: v.numpy().copy() for k, v in sim.cutter.state_arrays().items()}
    out["particle_q"], out["particle_qd"] = sim.state.particle_q.numpy().copy(), sim.state.particle_qd.numpy().copy()
    return out


def _run(sim, hook, start: tuple[float, float, int], periods: int) -> dict[str, np.ndarray]:
    hook.z0, hook.z1, hook.k = start
    for _ in range(periods):
        hook.period()
    return _record(sim)


def test_a_rollback_across_a_split_reruns_bit_for_bit() -> None:
    sim, hook = _stroke()                                       # the edge 2 mm above the top, 0.4 mm beyond the lead
    assert int(sim.cutter.split.numpy().sum()) == 0                                    # saved before the first split
    keys = sim.cutter.state_arrays()
    assert {"topology.model.tet_materials", "topology.model.particle_mass", "topology.model.spring_stiffness",
            "topology.vnode.split", "topology.vnode.hub_side", "topology.vnode.work"} <= set(keys)
    assert set(sim.cutter.state_worlds()) == set(keys)                                # every array has a world map
    snap = WorldSnapshot(sim.model, sim.state, extra=keys, extra_world=sim.cutter.state_worlds())
    snap.save(sim.state)
    start = (hook.z0, hook.z1, hook.k)
    first = _run(sim, hook, start, 12)
    assert np.all(np.bincount(sim.cutter.tets.world.numpy()[sim.cutter.split.numpy() == 1], minlength=2) > 0)
    snap.restore(sim.state)
    assert int(sim.cutter.split.numpy().sum()) == 0                                    # the splits are undone too
    again = _run(sim, hook, start, 12)
    for k, v in first.items():
        np.testing.assert_array_equal(again[k], v, err_msg=k)


def _mask(model, worlds: list[int]) -> wp.array:
    a = np.zeros(int(model.world_count), np.int32)
    a[worlds] = 1
    return wp.array(a, dtype=wp.int32, device=model.device)


def _restore_world_1(cutter, model, state, snap, before: dict, after: dict) -> None:
    """Restore world 1 alone: every key's world-1 items equal ``before``, its world-0 items stay ``after``."""
    snap.restore(state, world_mask=_mask(model, [1]))
    for k, v in cutter.state_arrays().items():
        now, w = v.numpy(), cutter.state_worlds()[k].numpy()
        assert w.shape[0] == now.shape[0], k
        np.testing.assert_array_equal(now[w == 0], after[k][w == 0], err_msg=f"{k}: world 0 changed")
        np.testing.assert_array_equal(now[w == 1], before[k][w == 1], err_msg=f"{k}: world 1 not restored")


def _changed(before: dict, after: dict) -> set[str]:
    return {k for k in before if k in after and not np.array_equal(before[k], after[k])}


CHANGED: set[str] = set()          # keys some stroke below changed (test_every_key_is_exercised)


def test_restoring_one_world_keeps_the_other_worlds_splits() -> None:
    """Lead rule, every key (WP4 review A)."""
    sim, hook = _stroke()
    snap = WorldSnapshot(sim.model, sim.state, extra=sim.cutter.state_arrays(), extra_world=sim.cutter.state_worlds())
    snap.save(sim.state)
    before = _record(sim)
    after = _run(sim, hook, (hook.z0, hook.z1, hook.k), 12)
    CHANGED.update(_changed(before, after))
    _restore_world_1(sim.cutter, sim.model, sim.state, snap, before, after)


def _top2(worlds: int):
    from alloy.topology.twin_copies import KnifeForceTwinCopies

    twin = KnifeForceTwinCopies()
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for _ in range(worlds):
        b.begin_world()
        twin.build_world(b, twin.conditions(quick=True)[0], {"softness": 1.0, "cut_spring_ke": 1000.0})
        b.end_world()
    b.color()
    m = b.finalize(device="cpu")
    solver = twin.make_solver(m, "vbd")
    pipe = twin.make_pipeline(m, solver)
    return twin, m, solver, pipe, pipe.contacts(), (m.state(), m.state())


def _top2_run(top2, steps: int) -> dict[str, np.ndarray]:
    """``steps`` (even: the current state is states[0] again) TOP-2 steps from the start; every key and the state."""
    twin, m, solver, pipe, contacts, (s0, s1) = top2
    ctrl = m.control()
    for k in range(steps):
        twin.drive(k, k * 5.0e-4, s0, ctrl)
        pipe.collide(s0, contacts)
        solver.step(s0, s1, ctrl, contacts, 5.0e-4)
        s0, s1 = s1, s0
    out = {k: v.numpy().copy() for k, v in solver.cutter.state_arrays().items()}
    out["particle_q"], out["particle_qd"] = s0.particle_q.numpy().copy(), s0.particle_qd.numpy().copy()
    return out


def test_a_work_rule_rollback_reruns_bit_for_bit_per_world() -> None:
    """The work rule (TOP-2 on copies, 2 worlds) changes the hub phases, knife work and bonds the lead stroke never
    touches (WP4 review B). Saved at the start, run past the first cut (step 205), restored world 1 alone, then all,
    and re-run bit for bit. SolverVBD's own history (``particle_q_prev``, ``body_q_prev``, ...: the knife's friction
    reads it) goes in the snapshot too, as a stepper's ``alloy.coupling.adapters.vbd_state_arrays``; without it the
    re-run's welds differed at 1e-4."""
    from alloy.coupling.adapters import vbd_state_arrays, vbd_world_maps

    top2 = _top2(2)
    m, solver, s0 = top2[1], top2[2], top2[5][0]
    cut = solver.cutter
    vbd_maps = dict(vbd_world_maps(solver.vbd))                       # the solver's own history, per-world arrays
    vbd = {k: v for k, v in vbd_state_arrays(solver.vbd).items() if k in vbd_maps}
    snap = WorldSnapshot(m, s0, extra={**cut.state_arrays(), **vbd}, extra_world={**cut.state_worlds(), **vbd_maps})
    snap.save(s0)
    before = {k: v.numpy().copy() for k, v in cut.state_arrays().items()}
    first = _top2_run(top2, 230)
    assert np.all(np.bincount(cut.tets.world.numpy()[cut.split.numpy() == 1], minlength=2) > 0)
    CHANGED.update(_changed(before, first))
    _restore_world_1(cut, m, s0, snap, before, first)
    snap.restore(s0)
    again = _top2_run(top2, 230)
    for k, v in first.items():
        np.testing.assert_array_equal(again[k], v, err_msg=k)


def test_a_split_at_a_kinematic_hub_rolls_back_its_added_mass() -> None:
    """``added`` changes only when a clamped (kinematic) hub's tet splits."""
    from alloy.testing.topology import soft_kwargs
    from alloy.topology.tet_copies import add_tet_copies
    from alloy.topology.types import Blade
    from alloy.topology.vnode_cut import VNodeCutter

    b = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    x = np.array([[5, 0, 0], [5, 5, 0], [5, 5, 5], [10, 5, 5]], float) * 1e-3
    b.add_soft_mesh(vertices=[wp.vec3(*v) for v in x], indices=[0, 1, 2, 3], **soft_kwargs(k_mu=5e4, k_lambda=5e4))
    rec = add_tet_copies(b, [0], 1000.0)
    body = b.add_body(xform=wp.transform(wp.vec3(X_CUT, 0.0, 0.024), wp.quat_identity()), mass=0.0)
    b.add_shape_box(body, hx=0.0005, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    b.color()
    m = b.finalize(device="cpu")
    for name in ("particle_mass", "particle_inv_mass"):
        a = getattr(m, name).numpy()
        a[int(rec.hub[0, 0])] = 0.0
        getattr(m, name).assign(a)
    cut = VNodeCutter(m, rec, [Blade(body=body, shape=-1, edge_a=(0.0, -0.05, -0.02), edge_b=(0.0, 0.05, -0.02))],
                      1.6e-3)
    s = m.state()
    qd = s.body_qd.numpy()
    qd[body, 2] = -0.05
    s.body_qd.assign(qd)
    snap = WorldSnapshot(m, s, extra=cut.state_arrays(), extra_world=cut.state_worlds())
    snap.save(s)
    before = {k: v.numpy().copy() for k, v in cut.state_arrays().items()}
    cut.pre_step(s)
    CHANGED.update(_changed(before, {k: v.numpy().copy() for k, v in cut.state_arrays().items()}))
    assert cut.added.numpy()[0] > 0.0
    snap.restore(s)
    for k, v in cut.state_arrays().items():
        np.testing.assert_array_equal(v.numpy(), before[k], err_msg=k)


def test_every_key_is_exercised() -> None:
    """Each key changed in at least one case above, so the re-runs would catch it missing (WP4 review B). Run with
    the file. ``conflicts`` counts a hub split to two sides in one pass, which the canonical per-hub sides rule out:
    it stays 0 by construction."""
    if not CHANGED:
        pytest.skip("run with the cases above")
    sim, _ = _stroke()
    keys = set(sim.cutter.state_arrays()) - {"topology.vnode.conflicts"}
    assert keys <= CHANGED, sorted(keys - CHANGED)


def _free_descent_runner(worlds: int, ticks: int):
    from alloy.bully.replay import EpisodeRunner
    from alloy.bully.types import BullyLimits
    from alloy.topology import bully_target as bt

    t = bt.cut_block_free_copies()
    a = t.actions
    acts = np.tile(a.initial, (ticks, 1))
    acts[:, 2] = np.minimum(a.max_delta_per_tick[2] * np.arange(1, ticks + 1), a.high[2])
    return EpisodeRunner(t, worlds, "cpu", BullyLimits()), acts


def test_a_second_episode_replays_the_first() -> None:
    """EpisodeRunner resets the sim between runs. DirectSim's initial state alone put the blades back at their build
    pose 1 m up (make_sim poses them after the build) and the next tick pulled them down at 60 m/s: every world S1
    at step 0. VNodeSim.reset restores its construction-time snapshot instead."""
    runner, acts = _free_descent_runner(2, 8)
    first = runner.run(acts)
    x1 = runner.sim.state.particle_q.numpy().copy()
    again = runner.run(acts)
    np.testing.assert_array_equal(again.severity, first.severity)
    assert np.all(first.severity == 0)
    np.testing.assert_array_equal(runner.sim.state.particle_q.numpy(), x1)


def test_resetting_one_world_restores_its_start_only() -> None:
    runner, acts = _free_descent_runner(2, 8)
    sim = runner.sim
    x0, bq0 = sim.state.particle_q.numpy().copy(), sim.state.body_q.numpy().copy()
    runner.run(acts)
    x1, split1 = sim.state.particle_q.numpy().copy(), sim.cutter.split.numpy().copy()
    sim.reset(_mask(sim.model, [1]))
    pw, tw = sim.model.particle_world.numpy(), sim.cutter.tets.world.numpy()
    x = sim.state.particle_q.numpy()
    np.testing.assert_array_equal(x[pw == 1], x0[pw == 1])
    np.testing.assert_array_equal(x[pw == 0], x1[pw == 0])
    np.testing.assert_array_equal(sim.cutter.split.numpy()[tw == 0], split1[tw == 0])
    assert sim.cutter.split.numpy()[tw == 1].sum() == 0 and split1[tw == 1].sum() > 0
    np.testing.assert_array_equal(sim.state.body_q.numpy()[sim.blades.numpy()[1]], bq0[sim.blades.numpy()[1]])
