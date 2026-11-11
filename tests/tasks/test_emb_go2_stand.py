# SPDX-License-Identifier: Apache-2.0
"""The shipped Go2 starts standing where alloy.build starts a robot (qpos0): its joint zero is Menagerie's stand
(alloy.tasks.emb_robots.zero_at_start; release red 2026-11-09: Menagerie's qpos0 put the feet 160 mm into the floor,
and its calf range excludes 0). It stands on the floor, not in it: Menagerie's foot contact (solimp 0.015 1 0.022) let
the 15 kg robot rest 13.7 mm inside a rigid floor, which health reads as a failed world once contact depth is
flagged on the MuJoCo route (GAP-E1, 2026-11-10); the feet take MuJoCo's default solimp and solref 0.005 s
(alloy.tasks.emb_robots.firm_feet)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from alloy.tasks.emb_robots import StartPose, zero_at_start
from alloy.tasks.emb_scene import scene_path
from alloy.tasks.emb_specs import EMBODIMENTS

GO2 = ("e8_go2", "e8_go2_stiff", "e8_go2_route")
STAND = {"hip": 0.0, "thigh": 0.9, "calf": -1.8}                 # [rad] Menagerie's 'home': the shipped joint zero
MENAGERIE_RANGE = {"hip": (-1.0472, 1.0472), "F_thigh": (-1.5708, 3.4907), "R_thigh": (-0.5236, 4.5379),
                   "calf": (-2.7227, -0.83776)}                  # menagerie unitree_go2 98a70c5 go2.xml
MENAGERIE_CTRL = {"hip": (-0.9472, 0.9472), "F_thigh": (-1.4, 2.5), "R_thigh": (-0.4236, 2.5),
                  "calf": (-2.6227, -0.84776)}                   # go2_mjx.xml
FEET_XY = {"FL": (0.19215678, 0.142), "FR": (0.19215678, -0.142), "RL": (-0.19464322, 0.142),
           "RR": (-0.19464322, -0.142)}                          # [m] the stand, as rc7's 'init' keyframe placed it
SETTLE_M, SLIDE_M = 0.042, 0.033      # [m] the source stand's own sag and slide (37.3, 30.0 mm in 1 s), +10 %
REST_DEPTH_M = 1e-3                   # [m] health's PEN_WARN: a world deeper than this is DEGRADED
DEFAULT_SOLIMP = (0.9, 0.95, 0.001)   # MuJoCo's default impedance (dmin, dmax, width)


def _model(key: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(scene_path(EMBODIMENTS[key])))


def _part(joint: str) -> str:
    leg, part = joint.split("_")[:2]
    return f"{leg[0]}_thigh" if part == "thigh" else part


@pytest.mark.parametrize("key", GO2)
def test_go2_qpos0_is_menageries_stand_on_the_floor(key: str) -> None:
    m = _model(key)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    assert np.array_equal(m.qpos0, m.key_qpos[m.key("init").id])           # where alloy.build starts it: 'init'
    assert np.array_equal(m.key_ctrl[m.key("init").id], np.zeros(m.nu))     # ctrl 0 holds the stand
    for leg, xy in FEET_XY.items():
        g = next(g for g in range(m.ngeom) if m.geom(g).name == leg)
        assert d.geom_xpos[g][:2] == pytest.approx(xy, abs=1e-8)
        assert d.geom_xpos[g][2] - m.geom_size[g][0] >= -0.5e-3              # no foot deeper than 0.5 mm
    for j in range(1, m.njnt):
        name, part = m.joint(j).name, _part(m.joint(j).name)
        s = STAND[part.split("_")[-1]]
        assert m.jnt_range[j] == pytest.approx(np.subtract(MENAGERIE_RANGE[part], s), abs=1e-12), name
        a = next(a for a in range(m.nu) if m.actuator_trnid[a, 0] == j)
        assert m.actuator_ctrlrange[a] == pytest.approx(np.subtract(MENAGERIE_CTRL[part], s), abs=1e-12), name


ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="a" pos="0 0 1">
  <joint name="j1" type="hinge" axis="0 1 0" range="-1 2"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
  <body name="b" pos="0.3 0 0" quat="0.9 0.1 0.3 0.2"><joint name="j2" type="hinge" axis="0.6 0 0.8" range="-2.5 -0.5"/>
  <geom type="box" pos="0.1 0.02 0" size="0.1 0.02 0.01"/><body name="c" pos="0.2 0 0">
  <joint name="j3" type="hinge" axis="0 0 1" range="-1 1"/><geom type="sphere" size="0.03" pos="0.05 0 0"/>
  </body></body></body></worldbody><actuator><position name="p1" joint="j1" kp="10" ctrlrange="-1 2"/>
  <position name="p2" joint="j2" kp="10" ctrlrange="-2.4 -0.6"/></actuator></mujoco>"""


def test_zero_at_start_keeps_every_motion() -> None:
    """Angle q after the shift is the source's q + s: every body, geom and inertia frame equals the source's."""
    src = mujoco.MjSpec.from_string(ARM)
    m0 = src.compile()
    spec = mujoco.MjSpec.from_string(ARM)
    start = zero_at_start(spec, StartPose({"j1": 0.7, "j2": -1.1, "j3": 0.0}, {"p1": 0.7, "p2": -1.1}))
    m1 = spec.compile()
    assert start == StartPose({"j1": 0.0, "j2": 0.0, "j3": 0.0}, {"p1": 0.0, "p2": 0.0})
    s = np.array([0.7, -1.1, 0.0])
    assert m1.jnt_range == pytest.approx(m0.jnt_range - s[:, None], abs=1e-12)
    assert m1.actuator_ctrlrange == pytest.approx(m0.actuator_ctrlrange - s[:2, None], abs=1e-12)
    d0, d1 = mujoco.MjData(m0), mujoco.MjData(m1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        d0.qpos[:] = rng.uniform(m0.jnt_range[:, 0], m0.jnt_range[:, 1])
        d1.qpos[:] = d0.qpos - s
        mujoco.mj_kinematics(m0, d0)
        mujoco.mj_kinematics(m1, d1)
        for field in ("xpos", "geom_xpos", "geom_xmat", "xipos", "ximat"):
            assert np.abs(getattr(d1, field) - getattr(d0, field)).max() < 1e-12, field


@pytest.mark.timeout(900)
@pytest.mark.parametrize("key", GO2)
def test_go2_stands_where_alloy_starts_it(key: str) -> None:
    """Added as a user adds it, default targets, 1 s on the CPU: the base moves by the source stand's own sag and
    slide, +10 % (coordinator 2026-11-09). Menagerie's stand at its own targets (MuJoCo C) settles 37.3 mm and slides
    30.0 mm (kp 50 under 15 kg); here 24.3 / 29.6 mm (24.2 / 29.6 route; 37.9 / 30.0 on Menagerie's soft feet, before
    firm_feet); rc7 shoved the base 518 mm up and slid it 507 mm."""
    import alloy

    scene = alloy.Scene(dt=0.01)
    scene.add_robot("go2", scene_path(EMBODIMENTS[key]))
    sim = alloy.build(scene, device="cpu")
    p = sim.run(1.0, record=["go2/base"]).world(0)["go2/base.pos"]
    dz, slide = p[:, 2] - p[0, 2], np.linalg.norm(p[:, :2] - p[0, :2], axis=1)
    assert dz.max() <= 5e-3 and -dz.min() <= SETTLE_M and slide.max() <= SLIDE_M, (dz.max(), dz.min(), slide.max())
    assert sim.check()["ok"]


@pytest.mark.parametrize("key", GO2)
def test_go2_feet_are_firm(key: str) -> None:
    """No geom is softer than MuJoCo's default solimp; the four foot balls keep their shape, priority and friction."""
    m = _model(key)
    assert (m.geom_solimp[:, 0] >= DEFAULT_SOLIMP[0]).all(), m.geom_solimp[m.geom_solimp[:, 0] < DEFAULT_SOLIMP[0]]
    for name in FEET_XY:
        g = m.geom(name)
        assert tuple(g.solimp[:3]) == DEFAULT_SOLIMP and g.solref[0] <= 0.005 and g.priority[0] == 1, name
        assert g.size[0] == 0.022 and g.condim[0] == 6 and tuple(g.friction) == (0.8, 0.02, 0.01), name


@pytest.mark.timeout(900)
@pytest.mark.parametrize("route", ["warp", "mjc"])
@pytest.mark.parametrize("key", GO2)
def test_go2_rests_on_the_floor_not_in_it(key: str, route: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Health's contact depth, read every 2 ms over 1.5 s from the start with default targets, never passes 1 mm (a
    world deeper is DEGRADED; past 1 cm held 0.1 s it is INVALID), on MuJoCo Warp and on the CPU's MuJoCo C route.
    On Menagerie's feet it peaked 15.3 mm and rested 13.7 mm; here 0.17 mm (the route variant 0.07 mm)."""
    import alloy

    monkeypatch.setenv("ALLOY_CPU_ROUTE", route)
    scene = alloy.Scene(dt=0.002)
    scene.add_robot("go2", scene_path(EMBODIMENTS[key]))
    sim = alloy.build(scene, device="cpu")
    depth = []
    for _ in range(750):
        sim.step(1)
        depth.append(max((w.get("max_penetration_m") or 0.0) for w in sim.health()["worlds"]))
    print(f"{key} {route}: peak {1e3 * max(depth):.3f} mm, last 0.5 s {1e3 * max(depth[-250:]):.3f} mm")
    assert 0.0 < max(depth) <= REST_DEPTH_M, (max(depth), int(np.argmax(depth)))
    assert sim.check()["ok"] and not sim.health()["invalid_worlds"]
