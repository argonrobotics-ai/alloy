# SPDX-License-Identifier: Apache-2.0
"""#10b P1 S1 laws (alloy.coupling.fine_pack; [internal note] notes/wp4-perworld-clock/DESIGN.md sections 10-11): a world
moves between two simulations of the same scene with different world counts bit for bit, and a moved world steps as it
does in its own pack (batch invariance across world counts, the fine pack's premise)."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

import alloy
from alloy.coupling.fine_pack import HEALTH_INDEX, TransferError, WorldTransfer, control_snapshot, snapshot
from alloy.integration.simulation import _health_arrays

W, E = 4, 2                         # the coarse pack's worlds; the world moved


def _box_on_cloth(worlds: int) -> alloy.Sim:
    """A box dropped on a cloth (MuJoCo + VBD under the Scheduler), each world's box at its own x and height."""
    sc = alloy.Scene(worlds=worlds, dt=0.01)
    sc.add_ground()
    sc.add_cloth("sheet", size=(0.3, 0.3), pos=(0.0, 0.0, 0.05), resolution=8, mass=0.05, self_contact=False)
    pos = np.array([[0.01 * w, 0.0, 0.10 + 0.005 * w] for w in range(worlds)])
    sc.add_box("box", size=(0.06, 0.06, 0.06), pos=pos, mass=0.2)
    return alloy.build(sc, device="cpu")


def _local(model, attr: str, i: int) -> int:
    """A global body / shape / particle index as its index among its own world's items (-1 stays)."""
    wa = getattr(model, attr).numpy()
    if not 0 <= i < wa.size:
        return i
    return int(np.searchsorted(np.flatnonzero(wa == wa[i]), i))


def _world_words(sim, w: int) -> dict[str, np.ndarray]:
    """Every slot's words of world ``w`` (copies): State, the stepper's arrays, health (the D20 energy reference: the
    worlds start at different heights; the records' item indices as world-local ones) and Control."""
    out = {}
    for snap, src in ((snapshot(sim), sim.state), (control_snapshot(sim), None)):
        snap.save(src)
        for s in snap.slots:
            iw = s.item_world.numpy()
            out[s.name] = s.store.numpy().reshape(-1, s.words)[iw == w].copy()
    row = _health_arrays(sim.health)["health.record"].numpy()[w:w + 1].copy()
    for f, attr in HEALTH_INDEX.items():
        row[f] = _local(sim.model, attr, int(row[f][0]))
    out["extra:health/health.record"] = np.frombuffer(row.tobytes(), np.uint8)
    peak = sim.health.start.peak_body.numpy()
    out["extra:health/start.peak_body"] = np.array([_local(sim.model, "body_world", int(peak[w]))])
    assert any(k.startswith("extra:health/") for k in out), sorted(out)
    return out


def _same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    return sorted(k for k in a if not np.array_equal(a[k], b[k]))


@pytest.fixture(scope="module")
def packs():
    coarse, fine = _box_on_cloth(W), _box_on_cloth(1)
    try:
        coarse.step(15)                                  # the box has landed on the cloth by 0.15 s
        yield coarse.engine, fine.engine
    finally:
        coarse.close()
        fine.close()


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_a_world_moved_out_and_back_is_bit_for_bit_itself(packs) -> None:
    coarse, fine = packs
    before = {w: _world_words(coarse, w) for w in range(W)}
    WorldTransfer(coarse, fine).copy(E, 0)
    assert _same(_world_words(fine, 0), before[E]) == []
    WorldTransfer(fine, coarse).copy(0, E)
    for w in range(W):
        assert _same(_world_words(coarse, w), before[w]) == [], w


@pytest.mark.law
@pytest.mark.timeout(1800)
def test_a_moved_world_steps_as_in_its_own_pack(packs) -> None:
    """World E moved into a 1-world pack and stepped one period at the coarse pack's dt_x and substeps (the coarse
    decisions pinned, its clock taken) ends bit for bit as world E stepped in the coarse pack."""
    coarse, fine = packs
    WorldTransfer(coarse, fine).copy(E, 0)
    fine.clock.ticks, fine.steps, fine.periods = coarse.clock.ticks, coarse.steps, coarse.periods
    for _ in range(3):
        coarse.step_period()
        rep = coarse.stepper.report()
        assert int(rep.rollbacks) == 0
        fine.stepper.choose_clock = lambda control_dt, last, rep=rep: (float(rep.dt_x), dict(rep.substeps))
        fine.step_period()
        assert _same(_world_words(fine, 0), _world_words(coarse, E)) == []
    z = coarse.state.body_q.numpy()[E, 2]
    assert z < 0.10                                      # resting on the cloth, not falling: contact in play


def test_a_randomized_world_moves_with_its_parameters() -> None:
    """Was ``test_a_randomized_world_is_refused_until_its_parameters_move`` (S1 moved state only and refused a world
    whose randomized mass differed from the slot's). S1b moves the per-world Model rows with the world
    (alloy.coupling.world_clock_params; its laws: test_world_clock_params.py): the slot holds world 2's mass and
    inertia and its state, and a transfer told not to move parameters (the way back) still refuses rows that differ."""
    def scene(worlds):
        sc = alloy.Scene(worlds=worlds, dt=0.01)
        sc.add_ground()
        sc.add_box("box", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.1), mass=0.2)
        sc.randomize("box", "density", 500.0, 1500.0)
        return alloy.build(sc, device="cpu", seed=3)
    a, b = scene(3), scene(1)
    try:
        mass = a.engine.model.body_mass.numpy()
        assert mass[2] != b.engine.model.body_mass.numpy()[0]
        with pytest.raises(TransferError, match="per-world model parameters differ"):
            WorldTransfer(a.engine, b.engine, params="same").copy(2, 0)
        WorldTransfer(a.engine, b.engine).copy(2, 0)
        assert b.engine.model.body_mass.numpy()[0] == mass[2]
        assert np.array_equal(b.engine.model.body_inertia.numpy()[0], a.engine.model.body_inertia.numpy()[2])
        assert _same(_world_words(b.engine, 0), _world_words(a.engine, 2)) == []
    finally:
        a.close()
        b.close()


def test_a_pack_carrying_vbd_contact_history_is_refused() -> None:
    """VBD's per-contact-row history (_prev_contact_*) has no world map: its rows follow the pack's contact build. A
    pack that carries it (here the source only: it is allocated lazily) fails closed (WP10's review)."""
    from alloy.integration.stepper import entry_solvers

    a, b = _box_on_cloth(2), _box_on_cloth(1)
    try:
        vbd = entry_solvers(a.engine.stepper)["vbd"]
        assert getattr(vbd, "_prev_contact_lambda", None) is None or vbd._prev_contact_lambda.size == 0
        vbd._prev_contact_lambda = wp.zeros(5, dtype=float, device="cpu")
        with pytest.raises(TransferError, match="without a per-world axis.*_prev_contact_lambda"):
            WorldTransfer(a.engine, b.engine)
    finally:
        a.close()
        b.close()


ARM = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="upper" pos="0 0 0.1"><joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
    <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="1"/>
    <body name="lower" pos="0.2 0 0"><joint name="elbow" type="hinge" axis="0 1 0" range="-2 2" damping="1"/>
      <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.015" mass="0.5"/></body></body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/><position joint="elbow" kp="100" kv="5"/></actuator>
</mujoco>"""


def _arm(worlds: int) -> alloy.Sim:
    sc = alloy.Scene(worlds=worlds, dt=0.01)
    sc.add_ground()
    sc.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    return alloy.build(sc, device="cpu")


@pytest.mark.law
@pytest.mark.timeout(900)
def test_a_moved_robot_world_takes_its_own_targets() -> None:
    """Control moves with the world (WP12's review): 3 worlds of a 2-actuator arm with their own targets; world 2
    moved into a 1-world pack carries its targets (MuJoCo ctrl rows mapped to worlds by their articulation) and steps
    2 periods bit for bit as in its own pack."""
    coarse, fine = _arm(3), _arm(1)
    try:
        coarse.set_targets({"shoulder": np.array([0.1, 0.4, 0.8]), "elbow": np.array([-0.3, 0.2, 0.6])})
        coarse.step(5)
        a, b = coarse.engine, fine.engine
        WorldTransfer(a, b).copy(2, 0)
        assert _same(_world_words(b, 0), _world_words(a, 2)) == []
        b.clock.ticks, b.steps, b.periods = a.clock.ticks, a.steps, a.periods
        for _ in range(2):
            a.step_period()
            rep = a.stepper.report()
            b.stepper.choose_clock = lambda control_dt, last, rep=rep: (float(rep.dt_x), dict(rep.substeps))
            b.step_period()
            assert _same(_world_words(b, 0), _world_words(a, 2)) == []
    finally:
        coarse.close()
        fine.close()


@pytest.mark.law
@pytest.mark.timeout(900)
def test_a_moved_world_keeps_its_own_start_overlap() -> None:
    """The start checks move with the world (WP12): in world 2 the upper of two stacked boxes starts 5 mm into the lower
    (START_PEN, a sticky per-world level ORed into every record; the ground is a global shape, not judged), the 1-world
    pack's world starts clear; moved and stepped, its record keeps START_PEN bit for bit as world 2's."""
    def scene(worlds):
        sc = alloy.Scene(worlds=worlds, dt=0.01)
        sc.add_ground()
        sc.add_box("a", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.03), mass=0.2)
        z = np.array([0.0905, 0.0905, 0.085][:worlds])
        sc.add_box("b", size=(0.06, 0.06, 0.06), pos=np.stack([np.zeros(worlds), np.zeros(worlds), z], 1), mass=0.2)
        return alloy.build(sc, device="cpu")
    coarse, fine = scene(3), scene(1)
    try:
        coarse.step(2)
        fine.step(2)
        a, b = coarse.engine, fine.engine
        level = a.health.start._start.numpy()
        assert level[2] > 0 and level[0] == 0 and b.health.start._start.numpy()[0] == 0
        WorldTransfer(a, b).copy(2, 0)
        b.clock.ticks, b.steps, b.periods = a.clock.ticks, a.steps, a.periods
        a.step_period()
        rep = a.stepper.report()
        b.stepper.choose_clock = lambda control_dt, last, rep=rep: (float(rep.dt_x), dict(rep.substeps))
        b.step_period()
        assert _same(_world_words(b, 0), _world_words(a, 2)) == []
    finally:
        coarse.close()
        fine.close()


def test_a_driven_object_is_refused() -> None:
    """A step observer (sim.drive's Drive) keeps its own clock, which is not moved (WP12): refused."""
    def scene(worlds):
        sc = alloy.Scene(worlds=worlds, dt=0.01)
        sc.add_ground()
        sc.add_box("pusher", size=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.1), kinematic=True)
        sc.add_box("block", size=(0.05, 0.05, 0.05), pos=(0.2, 0.0, 0.03), mass=0.1)
        sim = alloy.build(sc, device="cpu")
        sim.drive("pusher", velocity=(0.1, 0.0, 0.0), duration=1.0)
        return sim
    a, b = scene(2), scene(1)
    try:
        with pytest.raises(TransferError, match="step observers"):
            WorldTransfer(a.engine, b.engine)
    finally:
        a.close()
        b.close()


def test_a_transfer_between_different_scenes_is_refused() -> None:
    a = _box_on_cloth(2)
    sc = alloy.Scene(worlds=1, dt=0.01)
    sc.add_ground()
    sc.add_box("box", size=(0.06, 0.06, 0.06), pos=(0.0, 0.0, 0.1), mass=0.2)
    b = alloy.build(sc, device="cpu")
    try:
        with pytest.raises(TransferError, match="slot names differ"):
            WorldTransfer(a.engine, b.engine)
    finally:
        a.close()
        b.close()
