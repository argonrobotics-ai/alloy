# SPDX-License-Identifier: Apache-2.0
"""TRAIN-FIX 1b (TA-1 F1; WP3's root cause): the shipped arm packs (FR3, Panda, UR10e + 2F-85) have firm contacts, and
an arm pressing on its table or moving at random stays valid with a shallow press (CPU).

The sources keep MuJoCo's default contact (solref 0.02 s, solimp 0.9 0.95): on the 15 g Franka finger that rests 5.5 mm
deep at 91 N, and TA-1's scripted press (the shoulder target +0.05 rad per 20 ms step) froze 9 of 256 worlds INVALID,
18.9 mm deep. The arrival itself is not bounded: the hand lands at 1.3 m/s, 1.6 mm of travel per 1.25 ms step, below
what any solref at MuJoCo's 2-step floor can stop (impact peaks 1.2-5.2 mm on 256 worlds, reported in the commit);
the law bounds the steady press after landing (<= 2 mm) and asks for no INVALID world.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

import alloy
from alloy.cards.library import default_root
from alloy.envs import ActionSpec, Task, VectorEnv, jitter_body, jitter_joint
from alloy.envs.mjcf import without_bodies
from alloy.tasks.emb_firm import FIRM_SOLIMP, FIRM_SOLMIX, FIRM_SOLREF

PACKS = {"fr3": "emb_e1_fr3", "panda": "emb_e2_panda", "ur10e": "emb_e3_ur10e_2f85"}
WORLD = {"world", "table", "box", "cylinder", "mug"}
ARM = {"panda": [f"actuator{k}" for k in range(1, 8)] + ["hand_finger"],
       "fr3": [f"fr3_joint{k}" for k in range(1, 8)] + ["hand_finger"],
       "ur10e": ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3", "gripper_fingers_actuator"]}
GRIP_STEP = {"panda": 0.01, "fr3": 0.01, "ur10e": 0.2}
STEADY_MM = 1.0       # FEAT-RL's law says 2 mm; 1 mm holds the values (solref x2 or the default solimp fail it)
TA1_HOME = {"joint1": 0.0, "joint2": 0.393, "joint3": 0.0, "joint4": -1.963, "joint5": 0.0, "joint6": 2.356,
            "joint7": -0.7853}                       # TA-1's (ManiSkill's) home: the hand points down


@pytest.mark.parametrize("arm", sorted(PACKS))
def test_the_arm_packs_carry_firm_contacts(arm):
    pack = default_root() / "robots" / PACKS[arm]
    m = mujoco.MjModel.from_xml_path(str(pack / "scene.xml"))
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "world"
        np.testing.assert_allclose(m.geom_solref[g], FIRM_SOLREF, err_msg=body)
        assert m.geom_solimp[g][1] == pytest.approx(FIRM_SOLIMP[1]), body
        assert m.geom_solmix[g] == (1.0 if body in WORLD else FIRM_SOLMIX), body
    changes = json.loads((pack / "source.json").read_text())["changes"]
    assert any("alloy.tasks.emb_firm" in c for c in changes)


def _home(arm: str) -> dict[str, float]:
    """The Franka arms: TA-1's home (the hand low over the table, pointing down); the UR10e: its pack's start pose."""
    if arm in ("panda", "fr3"):
        return {(f"fr3_{j}" if arm == "fr3" else j): q for j, q in TA1_HOME.items()}
    root = ET.parse(default_root() / "robots" / PACKS[arm] / "scene.xml").getroot()
    qpos = [float(x) for x in root.find("keyframe/key[@name='init']").get("qpos").split()]
    joints = [j.get("name") for j in root.iter("joint") if j.get("type") in ("hinge", "slide")]
    return {j: q for j, q in zip(joints, qpos) if not j.startswith(("finger", "gripper"))}


def _env(arm: str, worlds: int) -> VectorEnv:
    """TA-1's recipe on any of the arms: a 4 cm 50 g cube on the table, joint-delta actions (0.05 rad per step), the
    start pose (``_home``) jittered 0.02 rad."""
    scene = alloy.Scene(worlds=worlds, dt=0.02)
    scene.add_robot(arm, without_bodies(default_root() / "robots" / PACKS[arm] / "scene.xml",
                                        ["box", "cylinder", "mug"]))
    scene.add_box("cube", size=(0.04,) * 3, pos=(0.5, 0.0, 0.02), mass=0.05, friction=1.0)
    acts = ARM[arm]
    starts = [jitter_joint(f"{arm}/{j}", by=0.02, at=q) for j, q in _home(arm).items()]
    task = Task(name=f"press-{arm}", obs=[arm, "cube"], reward={}, max_steps=10_000,
                actions=ActionSpec([f"{arm}/{a}" for a in acts], mode="delta",
                                   delta=[0.05] * (len(acts) - 1) + [GRIP_STEP[arm]]),
                randomize=starts + [jitter_body("cube", pos=(0.1, 0.1, 0.0), yaw_deg=45.0)])
    return VectorEnv(scene, task, worlds=worlds, device="cpu", seed=0, backend="numpy", autoreset="disabled")


def _run(env: VectorEnv, action, steps: int) -> tuple[float, float]:
    """(steady, peak) penetration in mm over all worlds: steady = the deepest contact at checks from step 40 on, once
    every arm has landed (the Panda by step 10; the FR3 and UR10e, which start higher, by about 25)."""
    steady = 0.0
    for k in range(1, steps + 1):
        env.step(action(k))
        if k >= 40 and k % 10 == 0:
            h = env.sim.health()
            steady = max(steady, max(float(w.get("max_penetration_m") or 0.0) for w in h["worlds"]))
    h = env.sim.health()
    assert h["invalid_worlds"] == [], h["invalid_worlds"]
    return 1e3 * steady, 1e3 * max(float(w.get("peak_penetration_m") or 0.0) for w in h["worlds"])


PRESS = {"panda": [(1, 1.0)], "fr3": [(1, 1.0)], "ur10e": [(1, 0.5), (2, 1.0)]}   # drives the hand onto the table


@pytest.mark.timeout(900)
@pytest.mark.parametrize("arm", sorted(PACKS))
def test_an_arm_pressing_on_its_table_stays_valid_and_shallow(arm):
    """TA-1's press (the UR10e: half the shoulder and the elbow; its shoulder alone swings past the table's edge)."""
    env = _env(arm, 8)
    env.reset(seed=0)

    def press(_k):
        a = np.zeros((8, env.writer.size), np.float32)
        for i, s in PRESS[arm]:
            a[:, i] = s
        return a

    steady, peak = _run(env, press, 100)
    print(f"{arm} press: steady {steady:.3f} mm, arrival peak {peak:.3f} mm")
    assert steady <= STEADY_MM


@pytest.mark.timeout(900)
@pytest.mark.parametrize("arm", sorted(PACKS))
def test_random_actions_keep_every_arm_valid_and_shallow(arm):
    """8 worlds, 3 s of uniform random joint deltas; some world touches the table or the cube (peak > 0.05 mm)."""
    env = _env(arm, 8)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    steady, peak = _run(env, lambda _k: rng.uniform(-1.0, 1.0, (8, env.writer.size)).astype(np.float32), 150)
    print(f"{arm} random: steady {steady:.3f} mm, peak {peak:.3f} mm")
    assert steady <= STEADY_MM and peak > 0.05


PART, TCP, PART_XY = 0.025, 0.1034, (0.6152, 0.0)    # the hand's xy at that home, measured


@pytest.mark.timeout(900)
def test_a_10_g_part_is_picked_and_lifted():
    """WP3: solmix 1000 lets the robot's contact govern a light part in the grip, not Alloy's own light-body sizing.
    The Panda picks a 10 g, 2.5 cm cube with end-effector actions (hand turn held) and lifts it. A no-regression case:
    it lifts on the soft packs too (peak contact 1.7 mm there, 0.3 mm here, 256 worlds)."""
    from alloy.envs.diffik import EEActionSpec

    W = 4
    scene = alloy.Scene(worlds=W, dt=0.02)
    scene.add_robot("panda", without_bodies(default_root() / "robots" / PACKS["panda"] / "scene.xml",
                                            ["box", "cylinder", "mug"]))
    scene.add_box("part", size=(PART,) * 3, pos=(*PART_XY, PART / 2 + 1e-4), mass=0.010, friction=1.0)
    spec = EEActionSpec(body="panda/hand", joints=[f"panda/actuator{k}" for k in range(1, 8)],
                        extra=["panda/hand_finger"], extra_delta=0.01, orientation=True)
    starts = [jitter_joint(f"panda/{j}", by=0.0, at=v) for j, v in TA1_HOME.items()]
    starts += [jitter_joint("panda/finger_joint1", at=0.04), jitter_joint("panda/finger_joint2", at=0.04)]
    task = Task(name="pick10g", obs=["panda"], actions=spec, reward={}, max_steps=10_000, randomize=starts)
    env = VectorEnv(scene, task, worlds=W, device="cpu", seed=0, backend="numpy", autoreset="disabled")
    env.reset(seed=0)
    hand_z = PART + TCP                       # fingertip centre at the part's top: the pads reach 9 mm below it
    hold_z = hand_z
    for phase, steps in (("down", 200), ("settle", 10), ("close", 20), ("hold", 10), ("up", 30), ("hold", 25)):
        if phase == "hold" and steps == 25:
            hold_z = np.asarray(env.state["panda/hand.pos"])[:, 2]             # hold the lifted height
        for _ in range(steps):
            a = np.zeros((W, env.writer.size), np.float32)
            z = np.asarray(env.state["panda/hand.pos"])[:, 2]
            if phase == "down" and (np.abs(z - hand_z) < 1e-3).all():
                break
            # targets start from the measured hand each step, so the hand's height is held in closed loop (an open-loop
            # 0 lets it sag onto the table); 1 cm per step at most
            a[:, 2] = 0.5 if phase == "up" else np.clip((hold_z - z) / 0.05, -0.5, 0.5)
            if phase not in ("down", "settle"):
                a[:, -1] = -1.0
            env.step(a)
    p, hand = np.asarray(env.state["part.pos"]), np.asarray(env.state["panda/hand.pos"])
    print(f"part z {p[:, 2].round(4).tolist()}, hand z {hand[:, 2].round(4).tolist()}")
    assert (p[:, 2] > 0.08).all() and (np.linalg.norm(p[:, :2] - hand[:, :2], axis=1) < 0.02).all()
    assert env.sim.health()["invalid_worlds"] == []


def test_a_box_rests_and_slides_on_the_pack_table():
    """WP3: the table's firm solref (solmix 1) mixes with every object resting on it. The pack's own 50 mm / 0.1 kg box
    rests 0.002 mm deep (0.034 at 0.02 s) without a bounce, then a 1.5 N push for 0.1 s slides it and friction
    (mu 1) stops it at 0.84 g (0.71 g at 0.02 s)."""
    scene = alloy.Scene(worlds=2, dt=0.01)
    scene.add_robot("panda", default_root() / "robots" / PACKS["panda"] / "scene.xml")
    sim = alloy.build(scene, device="cpu", seed=0)
    zs = []
    for _ in range(50):
        sim.step(1)
        zs.append(np.asarray(sim.state(["panda/box"])["panda/box.pos"])[:, 2])
    zs = np.array(zs)
    rest = zs[-10:].mean(0)
    assert np.all(1e3 * (0.025 - rest) < 0.01) and np.all(1e3 * (zs.max(0) - rest) < 0.01)
    sim.apply_wrench("panda/box", force=(1.5, 0.0, 0.0))
    sim.step(10)
    sim.apply_wrench("panda/box", force=(0.0, 0.0, 0.0))
    v = []
    for _ in range(5):
        sim.step(1)
        v.append(np.asarray(sim.state(["panda/box"])["panda/box.vel"])[:, 0])
    decel = (v[0] - v[-1]) / 0.04
    print(f"slide {v[0].round(3).tolist()} m/s, deceleration {decel.round(2).tolist()} m/s2 (mu g = 9.81)")
    assert np.all(v[0] > 0.1) and np.all((decel > 0.7 * 9.81) & (decel < 1.05 * 9.81))


USER_TABLE = ('<mujoco><worldbody><body name="slab" pos="0.6 0 0.03"><geom type="box" size="0.15 0.15 0.03"/>'
              '</body></worldbody></mujoco>')        # MuJoCo's default contact, solref 0.02 s, solmix 1


@pytest.mark.timeout(900)
def test_the_robot_governs_a_contact_with_a_users_default_table():
    """solmix 1000: the Panda pressing its hand on a user's own MJCF slab at MuJoCo's default 0.02 s stays shallow
    (0.22 mm steady); with the robot's solmix back at 1 the pair mixes to about 11 ms and rests 2.1 mm deep."""
    W = 8
    scene = alloy.Scene(worlds=W, dt=0.02)
    scene.add_robot("panda", without_bodies(default_root() / "robots" / PACKS["panda"] / "scene.xml",
                                            ["box", "cylinder", "mug"]))
    scene.add_robot("usertable", mjcf=USER_TABLE)
    acts = ARM["panda"]
    task = Task(name="press-slab", obs=["panda"], reward={}, max_steps=10_000,
                actions=ActionSpec([f"panda/{a}" for a in acts], mode="delta", delta=[0.05] * 7 + [0.01]),
                randomize=[jitter_joint(f"panda/{j}", by=0.02, at=q) for j, q in _home("panda").items()])
    env = VectorEnv(scene, task, worlds=W, device="cpu", seed=0, backend="numpy", autoreset="disabled")
    env.reset(seed=0)
    press = np.zeros((W, env.writer.size), np.float32)
    press[:, 1] = 1.0
    steady, peak = _run(env, lambda _k: press, 100)
    print(f"panda on a 0.02 s slab: steady {steady:.3f} mm, peak {peak:.3f} mm")
    assert steady <= STEADY_MM
