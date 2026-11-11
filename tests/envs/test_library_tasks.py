# SPDX-License-Identifier: Apache-2.0
"""Library tasks on CPU: registry and make() options, PickPlace (with a built-in library cube so no fetch is needed),
PegInsert's held start, DrawerOpen when its cabinet is fetched, and Pour's one-clock episodes (slow)."""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import REGISTRY, EnvError, make
from alloy.envs.library_tasks import GRIP, peg_tip

NEW = ("PickPlace-v0", "PegInsert-v0", "DrawerOpen-v0", "Pour-v0")


def _placeable(card: str) -> bool:
    from alloy.api import catalog

    return catalog.can_place(card)


def test_registry_has_library_tasks():
    assert set(NEW) <= set(REGISTRY)
    with pytest.raises(EnvError, match="no environment named"):
        make("PickPlce-v0")


@pytest.mark.compile_heavy
def test_pick_place_cpu():
    env = make("PickPlace-v0", num_envs=2, device="cpu", backend="numpy", obj="cube_beech_30", vary=False)
    try:
        obs, _ = env.reset(seed=0)
        assert env.action_names() == [f"{GRIP}/x", f"{GRIP}/y", f"{GRIP}/z", f"{GRIP}/close"]
        assert obs.shape == (2, env.single_observation_space.shape[0]) and "obj.com[2]" in env.obs_names()
        s = env.state
        assert np.all(np.abs(s["obj.com"][:, 2] - 0.015) < 0.003)            # settled on the table
        for _ in range(5):
            obs, r, term, trunc, info = env.step(np.zeros((2, 4), np.float32))
        assert set(info["terms"]) == {"reach", "held", "lift", "carry", "in_bin"} and not info["success"].any()
        assert np.all(info["terms"]["reach"] > 0.05) and np.isfinite(obs).all()
    finally:
        env.close()


@pytest.mark.compile_heavy
def test_peg_insert_starts_held_cpu():
    env = make("PegInsert-v0", num_envs=2, device="cpu", backend="numpy", precise=False, vary=False)
    try:
        env.reset(seed=0)
        s = env.state
        assert np.allclose(s[f"{GRIP}/close.q"], 0.0345, atol=1e-4)            # fingers just short of the peg
        start = s["peg.com"].copy()
        assert np.allclose(start[:, :2], s[f"{GRIP}/palm.pos"][:, :2], atol=1e-3)   # moved with the gripper
        keep = np.zeros((2, 4), np.float32)
        keep[:, 3] = 1.0
        for _ in range(10):
            env.step(keep)
        s = env.state
        assert np.all(start[:, 2] - s["peg.com"][:, 2] < 0.015)                  # held, not dropped
        assert np.allclose(peg_tip(s)[:, 2], s["peg.com"][:, 2] - 0.03, atol=2e-3)
    finally:
        env.close()


@pytest.mark.compile_heavy
def test_drawer_open_cpu():
    from alloy.envs.library_tasks import CABINET

    if not _placeable(CABINET):
        pytest.skip(f"{CABINET} is not fetched here (alloy fetch {CABINET})")
    try:
        env = make("DrawerOpen-v0", num_envs=2, device="cpu", backend="numpy", vary=False)
    except Exception as exc:  # noqa: BLE001 - its door / handle meshes may be missing from the store
        pytest.skip(f"cabinet does not build here: {exc}")
    try:
        env.reset(seed=0)
        obs, r, term, trunc, info = env.step(np.zeros((2, 4), np.float32))
        assert set(info["terms"]) == {"reach", "open"} and np.allclose(info["terms"]["open"], 0.0, atol=1e-3)
    finally:
        env.close()


@pytest.mark.slow
def test_pour_resets_one_world_at_a_time_cpu():
    """Since SPH's snapshot keeps each world's particles apart (cd2f831ce), a liquid scene resets one world alone."""
    env = make("Pour-v0", num_envs=2, device="cpu", backend="numpy", volume=20e-6, vary=False, max_steps=50)
    try:
        assert not env.together
        env.reset(seed=0)
        start = env.state["water.points"].copy()
        for _ in range(3):
            obs, r, term, trunc, info = env.step(np.zeros((2, 2), np.float32))
        assert "in_bowl" in info["terms"]
        moved = env.state["water.points"].copy()
        assert not np.array_equal(moved[1], start[1])
        env.reset(options={"reset_mask": np.array([True, False])})
        now = env.state["water.points"]
        assert np.array_equal(now[0], start[0]) and np.array_equal(now[1], moved[1])
    finally:
        env.close()


def test_the_bowl_carriage_has_no_camera_only_geom():
    """A geom that collides with nothing is a visual shape to the importer, and a model with one shows none of its
    collision shapes: the bowl's boxes were hidden from every camera that way (audit, 2026-11-10)."""
    import xml.etree.ElementTree as ET

    from alloy.envs.soft_tasks import bowl_carriage

    root = ET.fromstring(bowl_carriage())
    assert not [g for g in root.iter("geom") if g.get("contype") == "0" or g.get("conaffinity") == "0"]
    carriage = next(b for b in root.iter("body") if b.get("name") == "carriage_x")
    assert float(carriage.find("inertial").get("mass")) == 0.2 and len(list(root.iter("geom"))) == 5


@pytest.mark.slow
def test_the_pour_camera_shows_the_bowl_and_the_water_cpu():
    """Pour-v0's image was a blank floor (mean 218, std 2.5 of 255): the bowl hidden, the water never drawn. Now the
    bowl is in view from the start (4.6 % of the pixels over 30 levels from the floor's colour; 0 with it hidden),
    the image changes as the water pours, an image handed out is not overwritten by the next render, and the camera
    does not change what is simulated."""
    kw = {"num_envs": 1, "device": "cpu", "backend": "numpy", "vary": False, "start_offset": 0.0}
    plain, cam = make("Pour-v0", **kw), make("Pour-v0", camera=True, **kw)
    try:
        state, _ = plain.reset(seed=0)
        obs, _ = cam.reset(seed=0)
        first, kept = obs["front"], obs["front"].copy()
        assert first.shape == (1, 64, 64, 3) and first.dtype == np.uint8
        unlike = float((np.abs(first[0] - np.median(first[0].reshape(-1, 3), 0)).max(-1) > 30).mean())
        assert unlike > 0.02, f"the bowl is not in the image ({100 * unlike:.1f} % of the pixels unlike the floor)"
        np.testing.assert_array_equal(obs["state"], state)
        zero = np.zeros((1, 2), np.float32)
        for _ in range(25):                                       # 0.5 s: the stream reaches the bowl after 0.2 s
            state = plain.step(zero)[0]
            obs = cam.step(zero)[0]
            np.testing.assert_array_equal(obs["state"], state)
        np.testing.assert_array_equal(first, kept)
        changed = float((np.abs(obs["front"].astype(int) - kept.astype(int)).max(-1) > 12).mean())
        assert changed > 0.005, f"the water is not drawn: {100 * changed:.2f} % of the pixels changed"
    finally:
        plain.close()
        cam.close()
