# SPDX-License-Identifier: Apache-2.0
"""Laws: a Gym policy camera renders the scene a video of the sim shows (TA-2 F2 / F3, TRAIN-FIX).

The policy camera renders the fast tracer's scene of the sim (``camera_scene(sim, quality="fast")``), the one
``Camera.render(sim)`` uses, not the bare engine model:

- a scanned object: the policy image IS ``Camera.render(sim)``, bit for bit, with the scan's photo texture;
- ``alloy.vary`` looks (PickPlace-v0, ``vary=True``): the policy image is the fast render of the sim, bit for bit,
  and that scene's surface colours are the looks ``Camera.render(sim)`` draws on the full-material tracer: the mean
  albedo images agree within ``ALBEDO_GAP`` (linear, worst channel). Albedo, not the colour image: the two tracers
  light and tone-map differently (the colour-image means differ by 20-32 sRGB levels with the looks right);
- the default ground and the first object differ in colour (GripperLift: the gripper's own colours used to leave
  the ground and the cube one palette blue);
- on an empty texture cache (a fresh install), the varied envs (PickPlace, Pour, DrawerOpen) draw no white floor: at
  most ``NEAR_WHITE_MAX`` of an image's pixels near white, and the bin, bowl and cabinet stand out from the floor
  (AUDIT-BENCH AB-policycam-1: the white placeholder colour of textured presets drew 72-95 % of those images white);
- mutant: the policy camera back on the engine model fails the first two laws.

Measured on the CPU, PickPlace with some colour maps cached / an empty cache: albedo gap 0.043 / 0.054 (the full
tracer's albedo adds its specular reflectance, about 0.04), engine-model mutant 0.39 / 0.35. Empty cache, near-white
share / share unlike the floor (bound): PickPlace 0 / 0.127 (0.06), Pour 0 / 0.056 (0.03), DrawerOpen
0.025 / 0.716 (0.40; 48 % on the release before the camera fix, AUDIT-BENCH); with the white placeholder again (mutant)
the near-white shares were 0.837, 0.945 and 0.715.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from alloy.camera import camera_scene
from alloy.camera.from_newton import GROUND_RGB
from alloy.camera.item_looks import items_look
from alloy.envs import VectorEnv, make
from alloy.envs.tasks import gripper_lift_scene, gripper_lift_task
from tests.camera.scan_fixture import TILE, quadrants, tile_card

ALBEDO_GAP = 0.08  # linear albedo, worst channel of the image mean: the policy scene's vs Camera.render(sim)'s
SCAN_MIN = 40.0   # sRGB levels: mean |image - mutant image| over the scan tile's pixels must exceed this
NEAR_WHITE_MAX = 0.20  # share of an env image's pixels at 250 or more in every channel, empty texture cache
UNLIKE_LEVELS = 30     # sRGB levels (worst channel) from the floor's median colour: a pixel that is not floor

pytestmark = pytest.mark.timeout(900)   # CPU renders, and the full-material tracer's first CPU build


def _bare(monkeypatch) -> None:
    """The mutant: the policy camera renders the bare engine model again."""
    monkeypatch.setattr(VectorEnv, "_camera_source", lambda self: self.sim.engine.model)


def _albedo(cam, source) -> np.ndarray:
    """(worlds, H, W, 3) linear albedo (first-hit surface colour) of ``source`` through ``cam``."""
    return cam.render(source, output="all").albedo.numpy().astype(np.float64)


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    """Worst-channel difference of the mean colour of two (worlds, H, W, 3) batches."""
    return float(np.abs(a.reshape(-1, 3).mean(0) - b.reshape(-1, 3).mean(0)).max())


@pytest.fixture(scope="module")
def tile_env(tmp_path_factory):
    scene = gripper_lift_scene(2)
    scene.roots.append(str(tile_card(tmp_path_factory.mktemp("scan"), image=quadrants(32))))
    scene.add_object("tile", TILE, pos=(0.07, -0.07, 0.0))
    env = VectorEnv(scene, gripper_lift_task(camera=True), worlds=2, device="cpu", backend="numpy")
    yield env
    env.close()


def test_a_scanned_object_shows_its_photo_texture_in_the_policy_image(tile_env, monkeypatch):
    env = tile_env
    obs, _ = env.reset(seed=0)
    video = env.cameras["front"].render(env.sim).numpy()            # what Camera.render(sim) and a video show
    assert np.array_equal(obs["front"], video)
    assert any(m.texture >= 0 for m in camera_scene(env.sim, quality="fast", device="cpu").sd.meshes)
    _bare(monkeypatch)
    bare = env.reset(seed=0)[0]["front"]
    diff = np.abs(bare.astype(np.float64) - video).max(-1)
    tile = diff > 0                                                # the tile's pixels: the only part that changes
    assert tile.mean() > 0.01 and diff[tile].mean() > SCAN_MIN, (tile.mean(), diff[tile].mean())


@pytest.mark.compile_heavy
def test_vary_looks_reach_the_policy_image(monkeypatch):
    env = make("PickPlace-v0", num_envs=2, device="cpu", backend="numpy", obj="cube_beech_30", vary=True, camera=True)
    try:
        assert items_look(env.sim) is not None                    # the variant gave its items looks
        policy = env.reset(seed=0)[0]["front"]
        cam, fast = env.cameras["front"], camera_scene(env.sim, quality="fast", device="cpu")
        assert np.array_equal(policy, cam.render(fast).numpy())
        video = _albedo(cam, env.sim)                              # full-material tracer: the item looks
        assert _gap(_albedo(cam, fast), video) <= ALBEDO_GAP, _gap(_albedo(cam, fast), video)
        _bare(monkeypatch)
        bare = env.reset(seed=0)[0]["front"]
        assert not np.array_equal(bare, cam.render(fast).numpy())
        assert _gap(_albedo(cam, env.sim.engine.model), video) > ALBEDO_GAP
    finally:
        env.close()


def test_the_default_ground_and_the_first_object_differ_in_colour():
    env = make("GripperLiftCamera-v0", num_envs=1, device="cpu", backend="numpy")
    try:
        model = env.sim.engine.model
        labels, colors = list(model.shape_label), model.shape_color.numpy()
        ground, cube = labels.index("ground"), labels.index("cube")
        assert np.allclose(colors[ground], colors[cube])           # what the assembly leaves: one palette blue
        sd = camera_scene(env.sim, quality="fast", device="cpu").sd
        rgb = {n: tuple(np.round(c, 6)) for n, c in zip(sd.inst_name, sd.inst_color)}
        assert rgb["ground"] == GROUND_RGB, rgb
        assert np.abs(np.subtract(rgb["cube"], rgb["ground"])).max() > 0.1, rgb   # apart, not just unequal
    finally:
        env.close()


@pytest.fixture
def empty_texture_cache(tmp_path, monkeypatch):
    """A fresh install: no render texture cache (and no item looks remembered from a cached one)."""
    import alloy.camera.item_looks as IL
    monkeypatch.setenv("ALLOY_RENDER_ASSETS", str(tmp_path))
    monkeypatch.setattr(IL, "_MEMO", {})
    return tmp_path


def _unlike_floor(env, img: np.ndarray) -> tuple[np.ndarray, float]:
    """(the floor's median colour in world 0's image ``img``, share of the image's pixels unlike it)."""
    fast = camera_scene(env.sim, quality="fast", device="cpu")
    ids = env.cameras["front"].render(fast, output="instance_id").numpy()[0]
    floor = np.median(img[ids == 1 + list(fast.sd.inst_name).index("ground")], 0)
    return floor, float((np.abs(img - floor).max(-1) > UNLIKE_LEVELS).mean())


def _camera_env(name: str, kw: dict):
    """One camera world of env ``name``; a skip when DrawerOpen's cabinet does not build here (its meshes not fetched,
    as in ``test_library_tasks.test_drawer_open_cpu``)."""
    if name != "DrawerOpen-v0":
        return make(name, num_envs=1, device="cpu", backend="numpy", camera=True, **kw)
    from alloy.api import catalog
    from alloy.envs.library_tasks import CABINET
    if not catalog.can_place(CABINET):
        pytest.skip(f"{CABINET} is not fetched here (alloy fetch {CABINET})")
    try:
        return make(name, num_envs=1, device="cpu", backend="numpy", camera=True, **kw)
    except Exception as exc:  # noqa: BLE001 - its door / handle meshes may be missing from the store
        pytest.skip(f"cabinet does not build here: {exc}")


@pytest.mark.compile_heavy
@pytest.mark.parametrize("name, kw, unlike_min", [("PickPlace-v0", {"obj": "cube_beech_30"}, 0.06),
                                                  ("Pour-v0", {}, 0.03), ("DrawerOpen-v0", {}, 0.40)])
def test_varied_envs_draw_no_white_floor_without_the_texture_cache(name, kw, unlike_min, empty_texture_cache, caplog):
    caplog.set_level(logging.INFO, logger="alloy.camera.item_looks")    # the env logs the missing textures as it builds
    env = _camera_env(name, kw)
    try:
        img = env.reset(seed=0)[0]["front"][0].astype(np.int32)
        white = float((img >= 250).all(-1).mean())
        floor, unlike = _unlike_floor(env, img)
        assert white <= NEAR_WHITE_MAX and floor.max() < 250, (white, floor)
        assert unlike >= unlike_min, (unlike, floor)          # the bin / bowl / cabinet stands out from the floor
        assert "python -m alloy.looks --fetch-textures" in caplog.text
    finally:
        env.close()
