# SPDX-License-Identifier: Apache-2.0
"""The gantry library envs' gripper stalls on a wall instead of pushing a finger through it (``WALL_SAFE``).

Under uniform random actions the delta targets of the x, y, z servos (kp 3000 N/m) run on while a finger rests on a
wall, so the servo force grows with the error. A finger is 30 g, and MuJoCo scales a contact's stiffness by the mass
behind it: at MuJoCo's default 0.02 s a finger sank 3-6 mm into the bin's 6 mm wall at 25-55 N, past half the wall,
and the world went INVALID. ``WALL_SAFE`` caps each slide servo at 100 N and gives the finger contacts a 2.5 ms
time constant (two physics steps).

Measured (CPU, 256 worlds, 250 random control steps = 5 s, fingers toggled every 0.5 s, as the audit's repro):
- as shipped: 15 of 256 PickPlace worlds INVALID (seed 0; bin walls and ground, deepest 6.1 mm and 11 mm);
- a 25 N cap alone: 0 INVALID, but the scripted pick-and-place fell from 64 to 53 of 64 worlds (the cap slows the
  carry); 30-50 N caps alone left 1-5 worlds INVALID at 3.1-3.8 mm (the soft finger contact, not the force);
- ``WALL_SAFE``: 0 of 256 on PickPlace (seeds 0, 1, 2), DrawerOpen and PegInsert (seed 0), deepest contact 1.4-3.0 mm;
  scripted examples unchanged at 64 worlds: pick-and-place 64 of 64, drawer 64 of 64, peg 64 of 64 (63 as shipped).
One L4 (CUDA), the same repro: ``WALL_SAFE`` 0 of 256 on PickPlace (seeds 0, 1), DrawerOpen and PegInsert, as shipped
15 of 256 on PickPlace; scripted pick-and-place 62 and 64 of 64 and 254 of 256 (as shipped 61 and 60 of 64, 249 of
256); peg and drawer 64 of 64 both ways.
"""

from __future__ import annotations

import numpy as np
import pytest

from alloy.envs import library_tasks
from alloy.envs.robots import COUPLING_TIMECONST, WALL_SAFE, gantry_gripper

FINGER_SOLREF = f'condim="4" solref="{COUPLING_TIMECONST:g} 1"/>'


def test_wall_safe_caps_the_slides_and_stiffens_the_fingers():
    plain, safe = gantry_gripper(), gantry_gripper(**WALL_SAFE)
    assert "forcerange" not in plain and FINGER_SOLREF not in plain
    cap = WALL_SAFE["max_force"]
    for axis in "xyz":
        assert f'joint="{axis}" kp="3000" kv="200" forcelimited="true" forcerange="{-cap:g} {cap:g}"/>' in safe
    assert safe.count("forcerange") == 3                           # the slides only: yaw and close are not capped
    assert safe.count(FINGER_SOLREF) == 2
    assert safe.replace(FINGER_SOLREF, 'condim="4"/>').replace(
        f' forcelimited="true" forcerange="{-cap:g} {cap:g}"', "") == plain


@pytest.mark.parametrize(("option", "value"), [("max_force", 0.0), ("max_force", -5.0),
                                               ("finger_timeconst", 0.0), ("finger_timeconst", -0.01)])
def test_a_bad_option_raises_naming_it(option, value):
    with pytest.raises(ValueError, match=option):
        gantry_gripper(**{option: value})


def test_the_gantry_library_envs_build_their_gripper_wall_safe(monkeypatch):
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs)
        return gantry_gripper(*args, **kwargs)

    monkeypatch.setattr(library_tasks, "gantry_gripper", spy)
    for build in (library_tasks.pick_place_scene, library_tasks.peg_insert_scene, library_tasks.drawer_open_scene):
        build(1)
    assert len(seen) == 3
    assert all(kw.get("max_force") == WALL_SAFE["max_force"] for kw in seen)
    assert all(kw.get("finger_timeconst") == WALL_SAFE["finger_timeconst"] for kw in seen)


def _random_invalid(worlds: int, seed: int, steps: int = 250) -> int:
    """Worlds of PickPlace-v0 that go INVALID under uniform random actions, fingers toggled every 0.5 s."""
    import alloy.envs

    env = alloy.envs.make("PickPlace-v0", num_envs=worlds, device="cpu", seed=seed, autoreset="disabled",
                          backend="numpy")
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    ever = np.zeros(worlds, bool)
    for t in range(steps):
        a = rng.uniform(-1.0, 1.0, (worlds, 4)).astype(np.float32)
        a[:, 3] = -1.0 if (t // 25) % 2 == 0 else 1.0
        ever |= np.asarray(env.step(a)[4]["invalid"], bool).reshape(-1)
    return int(ever.sum())


@pytest.mark.slow
@pytest.mark.compile_heavy
@pytest.mark.timeout(3600)
def test_random_actions_leave_no_pick_place_world_invalid_cpu():
    assert _random_invalid(256, seed=0) == 0
