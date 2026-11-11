# SPDX-License-Identifier: Apache-2.0
"""Lightly pressed sliding contacts (known gap docs/known_gaps/contact-light-sliding-tip.md, narrowed 2026-11-07).

DC6's magnet eraser (48 g, 20 x 120 x 50 mm, mu 0.5) slid down its vertical board tipped over at the 1.25 ms rigid step
in the plug-in harness. The same contact, a box sliding down a 44 deg slope at mu 0.5 (the eraser's press / weight
ratio, 0.489 / 0.471 N), stays flat on the default route (public API, CPU: MuJoCo C at 1.25 ms), alone and on a light
dynamic plate. The control is a plain-MuJoCo replica of the harness scene: what tips it is the frictional rows'
reference at MuJoCo's floor (Newton's raw gains read kd 2e3 as tau = 2 / kd = 1 ms, which refsafe raises to 2 dt;
impratio 1); the route's gains (the box's material gains mixed with the static face's, impratio 300) hold it, and so
does a 25 ms friction time constant (solreffriction) at the raw normal gains."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

import alloy

ANGLE, MU, T = math.radians(44.0), 0.5, 0.3


def _slide(material: str | None = None, mass: float = 0.048, plate: float | None = None) -> dict[str, float]:
    """A 50 x 120 x 20 mm box sliding down a static 44 deg slope (on a dynamic plate of mass ``plate`` that friction
    holds on the slope, when given): its largest rotation [deg], lift off its face [m] and slide [m] over T."""
    n = np.array([math.sin(ANGLE), 0.0, math.cos(ANGLE)])
    rpy = (0.0, ANGLE, 0.0)
    kw = {"material": material, "rigid": True} if material else {}
    scene = alloy.Scene(dt=0.02)
    scene.add_box("slope", size=(1.0, 0.6, 0.02), pos=(0.0, 0.0, 0.5), rpy=rpy, static=True, friction=MU)
    base = np.array([0.0, 0.0, 0.5]) + 0.01 * n
    if plate is not None:
        base = base + 0.005 * n
        scene.add_box("plate", size=(0.6, 0.3, 0.005), pos=tuple(base - 0.0025 * n), rpy=rpy, mass=plate, **kw)
        scene.set_friction("plate", "slope", 2.0)
    scene.add_box("box", size=(0.05, 0.12, 0.02), pos=tuple(base + 0.01 * n), rpy=rpy, mass=mass,
                  **({"friction": MU} if plate is None else kw))
    if plate is not None:
        scene.set_friction("box", "plate", MU)
    sim = alloy.build(scene, device="cpu")
    tr = sim.run(T, record=["box"]).world(0)
    p, q = tr["box.pos"], tr["box.quat_wxyz"]
    rot = max(2.0 * math.degrees(math.acos(min(1.0, abs(float(np.dot(qk, q[0])))))) for qk in q)
    d = p[-1] - p[0]
    assert sim.check()["ok"]
    return {"rot": rot, "lift": float(((p - p[0]) @ n).max()), "slide": float(np.linalg.norm(d - (d @ n) * n))}


SLIDE = 0.5 * 9.81 * (math.sin(ANGLE) - MU * math.cos(ANGLE)) * T * T         # 147.9 mm


@pytest.mark.law
@pytest.mark.parametrize(("material", "mass", "plate"), [(None, 0.048, None), ("cardboard_corrugated", 0.02, 0.05)],
                         ids=["eraser_on_a_static_slope", "light_box_on_a_light_dynamic_plate"])
def test_a_lightly_pressed_box_slides_flat_on_the_default_route(material, mass, plate):
    """Rows (CPU): 0.11 deg, 0.08 mm, 148 mm; cardboard 20 g on 50 g: 0.07 deg, 0.04 mm, 148 mm."""
    r = _slide(material, mass, plate)
    assert r["rot"] < 0.5 and r["lift"] < 0.2e-3, r
    assert r["slide"] == pytest.approx(SLIDE, rel=0.02), r


def _replica(tc: float, dr: float, impratio: float, srf: str = "0 0") -> float:
    """Plain MuJoCo: the eraser pressed into a vertical board by tilted gravity, elliptic cone, 1.25 ms; the largest
    tilt [deg] over 0.3 s."""
    g_in = 0.489 / (400.0 * 8 * 0.01 * 0.06 * 0.025)
    xml = f"""<mujoco><option timestep="0.00125" gravity="{-g_in} 0 -9.81" cone="elliptic" impratio="{impratio}"
      iterations="100" integrator="implicitfast"/><worldbody>
      <geom name="board" type="box" pos="-0.01 0 0" size="0.01 0.3 1.0" friction="{MU} 0 0"/>
      <body pos="0.01 0 0.5"><freejoint/><geom name="box" type="box" size="0.01 0.06 0.025" density="400"
        friction="{MU} 0 0"/></body></worldbody>
      <contact><pair geom1="board" geom2="box" friction="{MU} {MU} 0 0 0" solref="{tc} {dr}"
        solreffriction="{srf}"/></contact></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    peak = 0.0
    for _ in range(int(round(T / m.opt.timestep))):
        mujoco.mj_step(m, d)
        peak = max(peak, abs(2.0 * math.degrees(math.asin(max(-1.0, min(1.0, d.qpos[5]))))))
    return peak


@pytest.mark.law
def test_the_tip_is_the_friction_reference_at_mujocos_floor():
    """Rows: raw gains 23.2 deg; the route's pair gains (0.0152 s, 0.61) at impratio 300 1.8 deg; raw normal gains with
    a 25 ms friction time constant 0.4 deg."""
    assert _replica(0.001, 1.0, 1.0) > 10.0
    assert _replica(0.0152, 0.608, 300.0) < 3.0
    assert _replica(0.001, 1.0, 1.0, srf="0.025 1") < 1.0
