# SPDX-License-Identifier: Apache-2.0
"""TRAIN-FIX item 5 (TA-4 F3): the shipped LEAP hand carries MuJoCo Playground's joint and servo dynamics, and random
finger targets neither drop nor fling a cube resting on the palm (CPU).

menagerie ships the hand with joint damping 0.03, armature 0, frictionloss 0.001 and no servo force limit; with random
finger targets its fingers flung Playground's 7 cm cube off the palm in every world (WP13 probe, 32 CPU worlds: all
dropped, 56 % ended 0.5 m away, peak speed p50 2.2 m/s; TA-4: 66 % thrown past 0.5 m on 1024 GPU worlds).
"""

from __future__ import annotations

import hashlib
import json

import mujoco
import numpy as np
import pytest

import alloy
from alloy.cards.library import default_root
from alloy.envs.mjcf import without_bodies
from alloy.tasks.emb_robots import PLAYGROUND_LEAP, PLAYGROUND_LEAP_SOURCE

PACK = default_root() / "robots" / "emb_e9_leap"
NAMES = [f"{f}_{j}" for f in ("if", "mf", "rf") for j in ("mcp", "rot", "pip", "dip")] + \
        [f"th_{j}" for j in ("cmc", "axl", "mcp", "ipl")]
# targets as TA-4's rotz3 (Playground's recipe): its home pose +- 0.6 rad, clipped to each joint's range
RANGES = {"mcp": (-0.314, 2.23), "rot": (-1.047, 1.047), "pip": (-0.506, 1.885), "dip": (-0.366, 2.042),
          "th_cmc": (-0.349, 2.094), "th_axl": (-0.349, 2.094), "th_mcp": (-0.47, 2.443), "th_ipl": (-1.34, 1.88)}
HOME = {"mcp": 0.8, "rot": 0.0, "pip": 0.8, "dip": 0.8, "th_cmc": 0.8, "th_axl": 0.8, "th_mcp": 0.8, "th_ipl": 0.0}
CUBE, REST = 0.07, (0.012889, 0.020074, 0.1287 + 0.5 * 0.07 + 0.001)   # Playground's cube: 7 cm, 0.108 kg
DROP_DZ, FAR_M, FLING_MPS = 0.03, 0.5, 0.25    # 0.13-0.20 m/s on 16-1024 worlds; a 3x cap gives 0.37 (AUDIT-BENCH)
# Playground v0.2.0, leap_rh_mjx.xml line 50 (the leap_rh joint default), as literals so a changed constant fails here:
# <joint axis="0 0 -1" damping="0.2" armature="0.00149376" actuatorfrcrange="-0.2196 0.2196" frictionloss="0.02"/>
SOURCE = {"damping": 0.2, "armature": 0.00149376, "frictionloss": 0.02, "torque_nm": 0.2196}


def _key(n: str) -> str:
    return n if n.startswith("th_") else n.split("_")[1]


def test_the_leap_pack_carries_playgrounds_dynamics_and_cites_them():
    m = mujoco.MjModel.from_xml_path(str(PACK / "scene.xml"))
    dofs = [m.jnt_dofadr[j] for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    assert len(dofs) == 16
    assert PLAYGROUND_LEAP == SOURCE
    np.testing.assert_allclose(m.dof_damping[dofs], SOURCE["damping"])
    np.testing.assert_allclose(m.dof_armature[dofs], SOURCE["armature"])
    np.testing.assert_allclose(m.dof_frictionloss[dofs], SOURCE["frictionloss"])
    assert m.nu == 16 and m.actuator_forcelimited.all()
    np.testing.assert_allclose(np.abs(m.actuator_forcerange), SOURCE["torque_nm"])
    src = json.loads((PACK / "source.json").read_text())["dynamics_source"]
    assert {k: src[k] for k in PLAYGROUND_LEAP_SOURCE} == PLAYGROUND_LEAP_SOURCE
    assert hashlib.sha256((PACK / "LICENSE.1").read_bytes()).hexdigest() == src["licence_sha256"]
    assert "Apache License" in (PACK / "LICENSE.1").read_text()[:200]


@pytest.mark.timeout(600)
def test_random_finger_targets_neither_drop_nor_fling_a_cube_on_the_palm():
    """16 worlds, 5 s of random targets at 20 Hz, every world starting alike (hand open, cube at rest): no cube drops
    3 cm or ends 0.5 m away, none moves faster than 0.25 m/s, and no world goes INVALID. Under the rotation env's own
    randomized reset (cube yaw +-180 deg) 15-19 % still drop: the yawed cube can start inside the fingers (TRAIN-FIX
    item 3's reset, not these values)."""
    W, rng = 16, np.random.default_rng(11)
    scene = alloy.Scene(worlds=W, dt=0.05)
    scene.add_robot("leap", without_bodies(PACK / "scene.xml", ["cube"]))
    scene.add_box("cube", size=(CUBE,) * 3, pos=REST, mass=0.108, friction=1.0)
    sim = alloy.build(scene, device="cpu", seed=11)
    lo = np.array([max(RANGES[_key(n)][0], HOME[_key(n)] - 0.6) for n in NAMES])
    hi = np.array([min(RANGES[_key(n)][1], HOME[_key(n)] + 0.6) for n in NAMES])
    start = np.asarray(sim.state(["cube"])["cube.pos"], np.float64)
    low, far, peak = np.inf, 0.0, 0.0
    for _ in range(100):
        a = np.clip(rng.standard_normal((W, 16)) * 0.6, -1.0, 1.0)
        tgt = lo + 0.5 * (a + 1.0) * (hi - lo)
        sim.set_targets({f"leap/{n}": tgt[:, i] for i, n in enumerate(NAMES)})
        sim.step(1)
        s = sim.state(["cube"])
        p, v = np.asarray(s["cube.pos"], np.float64), np.asarray(s["cube.vel"], np.float64)[:, :3]
        low, far = min(low, float(p[:, 2].min())), max(far, float(np.linalg.norm(p - start, axis=1).max()))
        peak = max(peak, float(np.linalg.norm(v, axis=1).max()))
    print(f"lowest cube z {low:.4f} m (rest {REST[2]:.4f}), farthest {far:.4f} m, peak speed {peak:.3f} m/s")
    assert low > REST[2] - DROP_DZ and far < FAR_M and peak < FLING_MPS
    assert sim.health()["invalid_worlds"] == [] and sim.check()["ok"]
