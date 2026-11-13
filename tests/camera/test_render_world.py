# SPDX-License-Identifier: Apache-2.0
"""Law: a render of a world the sim does not have raises, as ``sim.state(world=)`` does (Gemini audit S2-001).

The render kernel reads its camera and per-world arrays at the image's world index without a bound: on a 2-world sim,
``world=2`` and ``world=99`` returned a blank sky image reported as ``ok``, and ``world=100000`` or a negative world
crashed the process (SIGSEGV). Now ``sim.render(world=)``, ``alloy render`` / the MCP render op and a snapshot of a
model raise ``SimulationError`` naming the valid range, before any render.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.api.errors import SimulationError
from alloy.camera import snapshot

LOOK = ((0.35, -0.35, 0.25), (0.0, 0.0, 0.06))
SAME_SCENE = 6.0   # sRGB levels, mean |world 0 - world 1|: the same scene, each world with its own sample noise


@pytest.fixture(scope="module")
def sim():
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_ground()
    scene.add_box("cube", size=(0.12, 0.12, 0.12), pos=(0.0, 0.0, 0.06), mass=1.0)
    s = alloy.build(scene, device="cpu")
    yield s
    s.close()


@pytest.mark.parametrize("world", [2, 99, 100_000, -1])
def test_a_render_of_a_world_outside_the_sim_raises(sim, world):
    with pytest.raises(SimulationError, match=r"outside 0\.\.1") as err:
        sim.render(world=world, width=32, height=24, spp=1, look_at=LOOK)
    assert err.value.fix and err.value.message == f"world {world} is outside 0..1"
    with pytest.raises(SimulationError, match=r"outside 0\.\.1"):          # a model snapshot: the renderer checks
        snapshot(sim.engine.model, world=world, width=32, height=24, spp=1, look_at=LOOK, device="cpu")


def test_the_render_op_replies_with_the_error_not_an_image(sim):
    from alloy.cli.registry import respond
    from alloy.cli.session import Session
    sess = Session()
    recipe = {"worlds": 2, "dt": 0.01, "objects": [{"add": "ground"}, {"add": "box", "name": "cube", "mass": 1.0,
                                                                        "size": [0.12] * 3, "pos": [0, 0, 0.06]}]}
    sid = respond(sess, {"op": "build", "scene": recipe, "device": "cpu"})["result"]["sim"]
    try:
        reply = respond(sess, {"op": "render", "sim": sid, "world": 2, "width": 32, "height": 24, "spp": 1})
        assert not reply["ok"] and reply["error"]["type"] == "SimulationError", reply
        assert "outside 0..1" in reply["error"]["message"]
    finally:
        respond(sess, {"op": "close", "sim": sid})


def test_each_world_of_the_sim_still_renders(sim):
    imgs = [sim.render(world=w, width=32, height=24, spp=4, look_at=LOOK) for w in (0, 1)]
    assert all(i.shape == (24, 32, 3) and i.dtype == np.uint8 for i in imgs)
    a, b = (i.astype(np.int32) for i in imgs)
    cube = [float((np.abs(i - i[0, 0]).max(-1) > 30).mean()) for i in (a, b)]   # unlike the sky in the top corner
    assert min(cube) > 0.05 and np.abs(a - b).mean() < SAME_SCENE, (cube, np.abs(a - b).mean())
