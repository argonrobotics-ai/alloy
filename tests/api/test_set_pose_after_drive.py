# SPDX-License-Identifier: Apache-2.0
"""``sim.set_pose`` of a driven kinematic object sticks (CPU): a drive writes its object's pose every step, so a
teleport after the drive had ended was undone at the next step, silently (found 2026-11-09 reviewing WP9's
FIX-LIST 6; ``sim.stop_drives()`` first was the only way). ``set_pose`` now removes the object's drive: an ended one
always, a running one when every world is teleported; one world under a running drive raises."""

from __future__ import annotations

import numpy as np
import pytest

import alloy

TO = (0.2, 0.1, 0.05)


@pytest.fixture(scope="module")
def sim():
    s = alloy.Scene(worlds=2, dt=0.01)
    s.add_ground()
    s.add_box("pusher", size=(0.03,) * 3, pos=(0.5, 0, 0.1), kinematic=True)
    s.add_sphere("ball", 0.01, pos=(-0.5, 0, 0.01), mass=0.01)          # a dynamic entry to step
    out = alloy.build(s, device="cpu")
    yield out
    out.close()


def _pusher(sim, world: int = 0) -> np.ndarray:
    return sim.state(["pusher"], world=world)["pusher.pos"]


def _drives(sim) -> list:
    return [o for o in sim.engine._observers if type(o).__name__ == "Drive"]


def test_a_teleport_after_an_ended_drive_sticks(sim) -> None:
    sim.reset()
    sim.drive("pusher", velocity=(0.1, 0, 0), duration=0.2)
    sim.step(30)                                                         # the drive ended at 0.2 s
    assert _pusher(sim)[0] == pytest.approx(0.52, abs=1e-5)
    sim.set_pose("pusher", pos=TO, world=0)                              # one world: the ended drive goes anyway
    sim.step(2)
    assert np.allclose(_pusher(sim, 0), TO, atol=1e-6) and not _drives(sim)
    assert _pusher(sim, 1)[0] == pytest.approx(0.52, abs=1e-5)           # world 1 holds where its drive left it


def test_a_teleport_of_every_world_stops_a_running_drive(sim) -> None:
    sim.reset()
    sim.drive("pusher", velocity=(0.1, 0, 0))                            # runs for ever
    sim.step(5)
    sim.set_pose("pusher", pos=TO)
    sim.step(3)
    assert np.allclose(_pusher(sim, 0), TO, atol=1e-6) and np.allclose(_pusher(sim, 1), TO, atol=1e-6)
    assert not _drives(sim)


def test_one_world_under_a_running_drive_raises_and_moves_nothing(sim) -> None:
    sim.reset()
    sim.drive("pusher", velocity=(0.1, 0, 0))
    sim.step(5)
    before = _pusher(sim, 0).copy()
    with pytest.raises(alloy.SimulationError, match="would be undone") as info:
        sim.set_pose("pusher", pos=TO, world=0)
    assert "stop_drives" in info.value.fix and info.value.field == "pusher"
    assert np.allclose(_pusher(sim, 0), before) and len(_drives(sim)) == 1
    sim.stop_drives()
