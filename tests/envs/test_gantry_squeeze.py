# SPDX-License-Identifier: Apache-2.0
"""Gap #2 (gaps-now-1108, SC4): ``gantry_gripper(squeeze=...)`` squeezes a thin part (CPU, MuJoCo only).

The close servo (kp 1500 N/m) aimed at the fingers' meeting point (``FINGER_CLOSED``) presses a part of thickness t
with only kp x t / 2. On ClothFold's 3.5 mm cloth ridge that is 2.6 N, and 43 of 64 worlds let the ridge slide out
of the fingertips during the fold ([internal note] scr/SC4.md, gap #2). ``squeeze`` lets the target run past the meeting
point, so the stopped fingers press about kp x ``squeeze`` more.

Rig: the gripper closes on a 3 mm plate (90 x 3 x 60 mm, 0.35 kg, 3.4 N) standing on a pedestal, then lifts 5 cm.
Finger and plate friction 0.3, so the grip holds while 2 x 0.3 x squeeze force > 3.4 N:
- default: 2 x 0.3 x 1500 x 1.5 mm = 1.4 N, so the plate stays on the pedestal;
- squeeze 10 mm: 2 x 0.3 x 1500 x 11.5 mm = 10 N, so the plate rises with the palm.
Measured (main 4522df750 + this change, CPU): default plate 0.0 mm, palm 49.8 mm; squeeze 10 mm plate 48.1 mm,
palm 48.6 mm; health ok in both."""

from __future__ import annotations

import pytest

from alloy.envs.robots import FINGER_CLOSED, gantry_gripper

MU = 0.3                 # finger and plate friction (MuJoCo takes the larger of the two)
PLATE_KG = 0.35


def _lift(squeeze: float) -> tuple[float, float, bool]:
    """(plate rise, palm rise, health ok) [m] after closing on the plate and lifting the gripper 5 cm."""
    import alloy

    sc = alloy.Scene(dt=0.02)
    sc.add_ground()
    sc.add_box("stand", size=(0.04, 0.04, 0.10), pos=(0.0, 0.0, 0.05), static=True)
    sc.add_box("plate", size=(0.09, 0.003, 0.06), pos=(0.0, 0.0, 0.1305), mass=PLATE_KG, friction=MU)
    sc.add_robot("gripper", mjcf=gantry_gripper(friction=MU, squeeze=squeeze), pos=(0.0, 0.0, 0.2))
    sim = alloy.build(sc, device="cpu")
    sim.run(0.6, targets={"gripper/close": (0.0, FINGER_CLOSED + squeeze)})
    sim.run(0.2)
    s0 = sim.state(world=0)
    sim.run(0.5, targets={"gripper/z": (0.0, 0.05)})
    sim.run(0.3)
    s1 = sim.state(world=0)
    rise = float(s1["plate.pos"][2] - s0["plate.pos"][2])
    palm = float(s1["gripper/palm.pos"][2] - s0["gripper/palm.pos"][2])
    return rise, palm, bool(sim.check()["ok"])


def test_default_close_servo_drops_a_thin_plate():
    """Without ``squeeze`` the stopped fingers press 2.2 N: 1.4 N of friction under the plate's 3.4 N."""
    rise, palm, ok = _lift(0.0)
    assert ok and palm > 0.045, (rise, palm, ok)
    assert rise < 0.005, (rise, palm, ok)


def test_squeeze_carries_a_thin_plate():
    """With 10 mm of ``squeeze`` the plate rises with the palm (slip under 2 mm)."""
    rise, palm, ok = _lift(0.01)
    assert ok and palm > 0.045, (rise, palm, ok)
    assert palm - rise < 0.002, (rise, palm, ok)


def test_squeeze_leaves_the_default_gripper_alone():
    """``squeeze=0`` is the shipped MJCF, whose close range ends at the meeting point; a negative squeeze raises."""
    assert gantry_gripper() == gantry_gripper(squeeze=0.0)
    assert gantry_gripper().count(f'range="0 {FINGER_CLOSED:g}"') == 2
    assert gantry_gripper(squeeze=0.01).count(f'range="0 {FINGER_CLOSED + 0.01:g}"') == 2
    with pytest.raises(ValueError):
        gantry_gripper(squeeze=-0.001)
