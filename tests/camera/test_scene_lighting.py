# SPDX-License-Identifier: Apache-2.0
"""Law: ``camera_scene(sim, lighting=...)``, given once, lights every later render of that sim, on every path.

The public way to set the camera's sun and sky: ``alloy.camera.Lighting`` (suns plus a constant sky colour) and
``Sun.from_euler`` passed to ``camera_scene``. It held only for the scene that call built: a camera bound to a scene
built another way (a varied scene's item looks or ``quality="full"`` on the full-material tracer, a camera on another
device, a scene rebuilt after a cut) drew the studio light again without a word, and every new lighting kept one more
device scene; on a sim built with ``processes=`` it raised an error about ``sim.index``. Now the sim keeps the
lighting: every later render without a look file uses it, until the next one (``Lighting.studio()`` is the default),
and a multi-process sim raises CameraError naming ``processes=None``.
"""
from __future__ import annotations

import numpy as np
import pytest

import alloy
from alloy.camera import Camera, Lighting, Sun, camera_scene

LOOK = ((0.45, -0.45, 0.32), (0.0, 0.0, 0.06))
DARK = Lighting([Sun.from_euler((1.0, 0.0, 2.4), 3.0)], (0.1, 0.11, 0.13))   # a darker sky, the sun moved
DARKER = 0.75   # mean image level against the default: the sky fills most of the frame

pytestmark = pytest.mark.timeout(600)


def _scene(material=None, worlds=1):
    s = alloy.Scene(worlds=worlds, dt=0.01)
    s.add_ground()
    s.add_box("blk", size=(0.12, 0.12, 0.12), pos=(0.0, 0.0, 0.06), mass=1.0, **({"material": material} if material
                                                                              else {}))
    return s


def _cam(name):
    return Camera(name, look_at=LOOK, width=32, height=32, spp=4)


def _mean(img) -> float:
    return float(np.asarray(img, np.float64).mean())


def _snap(sim, **kw):
    return sim.render(width=32, height=24, spp=4, look_at=LOOK, **kw)


@pytest.fixture
def sim():
    s = alloy.build(_scene(), device="cpu")
    yield s
    s.close()


def test_given_once_it_lights_every_later_camera_and_snapshot(sim):
    bound = _cam("bound")
    before, snap = sim.render(bound, seed=1), _snap(sim)
    sc = camera_scene(sim, lighting=DARK)
    assert _mean(sim.render(_cam("new"), seed=1)) < DARKER * _mean(before)
    assert _mean(sim.render(bound, seed=1)) < DARKER * _mean(before)          # a camera bound before it, too
    assert _mean(_snap(sim)) < DARKER * _mean(snap)
    assert camera_scene(sim, lighting=Lighting(list(DARK.suns), DARK.world_color)) is sc   # equal: the same scene
    camera_scene(sim, lighting=Lighting.studio())
    assert np.abs(sim.render(bound, seed=1).astype(int) - before.astype(int)).mean() < 1.0


def test_it_holds_on_another_device_key_after_a_rebuild_and_keeps_one_scene(sim):
    from alloy.camera.camera import _SCENES, _forget
    camera_scene(sim, lighting=DARK, device="cpu")
    model = sim.engine.model
    for light in (Lighting([Sun.from_euler((0.2 * k, 0.0, 1.0), 2.0)], (0.2, 0.2, 0.2)) for k in range(4)):
        camera_scene(sim, lighting=light, device="cpu")
    assert len(_SCENES[model]) == 1                                            # not one more scene per lighting
    camera_scene(sim, lighting=DARK)
    _forget(camera_scene(sim, device="cpu"))                                   # as after a cut or a tear
    assert camera_scene(sim, device="cpu").sd.lighting.key() == DARK.key()


def test_a_varied_scene_and_full_quality_take_it_too():
    varied = alloy.build(alloy.vary(_scene("wood_beech"), 1, seed=0)[0], device="cpu")   # item looks: full tracer
    plain = alloy.build(_scene(), device="cpu")
    try:
        before, full = sim_render(varied), _snap(plain, quality="full")
        camera_scene(varied, lighting=DARK)
        camera_scene(plain, lighting=DARK)
        assert _mean(sim_render(varied)) < DARKER * _mean(before)
        assert _mean(_snap(plain, quality="full")) < DARKER * _mean(full)
    finally:
        varied.close()
        plain.close()


def sim_render(sim):
    return sim.render(_cam("v"), seed=1)


def test_a_sim_stepped_in_worker_processes_says_how_to_set_it():
    from alloy.camera.errors import CameraError
    with alloy.build(_scene(worlds=2), device="cpu", processes=2) as split:   # one world each
        with pytest.raises(CameraError, match="processes=None"):
            camera_scene(split, lighting=DARK)
