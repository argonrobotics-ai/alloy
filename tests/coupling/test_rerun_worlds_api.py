# SPDX-License-Identifier: Apache-2.0
"""``alloy.build(..., rerun_worlds=W_f)`` (#10b P1 S3, opt-in; [internal note] notes/wp4-perworld-clock/DESIGN.md): off by
default, the pack built with the simulation, and freed (its card directory and its model) when the simulation
closes."""

from __future__ import annotations

import gc
import os
import weakref

import numpy as np
import pytest

import alloy


def _scene(worlds: int) -> alloy.Scene:
    """A light box resting on a sheet (a coupled scene: VBD + MuJoCo under the Scheduler, which re-runs periods)."""
    sc = alloy.Scene(worlds=worlds, dt=0.01)
    sc.add_ground()
    sc.add_cloth("cloth", size=(0.2, 0.2), pos=(0, 0, 0.002), resolution=(6, 6), mass=0.03, thickness=0.004,
                 self_contact=False)
    z = np.full(worlds, 0.005 + 0.015 + 0.001 * np.arange(worlds) / max(worlds - 1, 1))
    sc.add_box("box", size=(0.03,) * 3, pos=np.stack([np.zeros(worlds), np.zeros(worlds), z], 1), mass=0.2)
    return sc


def test_fine_pack_is_off_by_default() -> None:
    sim = alloy.build(_scene(2), device="cpu")
    try:
        assert getattr(sim.engine, "fine_rerun", None) is None
    finally:
        sim.close()


@pytest.mark.timeout(300)
def test_the_fine_pack_is_built_with_the_simulation_and_freed_by_close() -> None:
    """The leak law: after ``close`` the pack's card directory is gone and its model is collected."""
    sim = alloy.build(_scene(3), device="cpu", rerun_worlds=1)
    fr = sim.engine.fine_rerun
    assert fr.slots == 1 and fr.sim is not None and fr.failed is None and fr.sim.worlds == 1
    workdir, model = fr.sim._workdir, weakref.ref(fr.sim.engine.model)
    assert workdir and os.path.isdir(workdir)
    sim.step(2)
    sim.close()
    assert fr.sim is None and not os.path.exists(workdir)
    gc.collect()
    assert model() is None


@pytest.mark.parametrize("bad", [-1, 2, 3, True, 1.0])
def test_rerun_worlds_takes_an_int_below_the_world_count(bad) -> None:
    """OSS-3: a value the user asked for never turns it off quietly; bool is not a count."""
    with pytest.raises(alloy.BuildError, match="rerun_worlds"):
        alloy.build(_scene(2), device="cpu", rerun_worlds=bad)
