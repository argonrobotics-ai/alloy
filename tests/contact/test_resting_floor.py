# SPDX-License-Identifier: Apache-2.0
"""Boxes and capsules come to rest after landing (2026-11-07, found checking WP2's per-world friction draws).

A box face or a capsule on its side rests on 2-4 contact points. With the card's restitution (e 0.5: damping ratio
0.215) MuJoCo's soft contacts at the route's 1.25 ms step never settle after a landing: a 0.1 kg box dropped 0.1-1 mm
keeps a 10-19 um normal oscillation for good (plain MuJoCo C too, frictionless too; a sphere settles), and on a slope
it walks down at a pair mu well above tan(theta): 1.7 mm/s at 10 deg, 6 mm/s at 40 deg (mu 1.09); placed in contact it
holds at 0.01 mm/s. Multi-point primitives now take the damping-ratio floor `native_gains.ZETA_MULTIPOINT` (0.7; 0.5
still left 2 of 16 landings walking), spheres keep the card's e."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

import alloy
from alloy.scene import native_gains as NG

TC, ZETA_E05 = 0.01042538, 0.21545376      # the route's solref for a 0.1 kg box at e 0.5 before the floor
W = 8
GAPS = np.linspace(0.05e-3, 2e-3, W)       # landing heights, one per world


def _plain_drop(shape: str, zeta: float, dt: float = 1.25e-3) -> float:
    """Plain MuJoCo C: a 0.1 kg body dropped 1 mm onto a plane at solref (TC, zeta); normal p2p [m] over 0.3-0.6 s."""
    geom, h = {"box": ('<geom type="box" size=".025 .025 .01" mass=".1"/>', 0.01),
               "capsule": ('<geom type="capsule" size=".01 .03" euler="0 90 0" mass=".1"/>', 0.01),
               "sphere": ('<geom type="sphere" size=".025" mass=".1"/>', 0.025)}[shape]
    xml = f"""<mujoco><option timestep="{dt}" integrator="implicitfast" cone="elliptic" impratio="300"/>
      <default><geom solref="{TC} {zeta}" solimp="0.9 0.95 0.001 0.5 2"/></default>
      <worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 {h + 1e-3}"><freejoint/>{geom}</body></worldbody>
      </mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    z = []
    for _ in range(int(round(0.6 / dt))):
        mujoco.mj_step(m, d)
        z.append(d.qpos[2])
    return float(np.ptp(z[int(round(0.3 / dt)):]))


@pytest.mark.parametrize("shape", ["box", "capsule"])
def test_multipoint_contact_rests_only_with_the_floor(shape: str) -> None:
    """The mechanism in plain MuJoCo: at e 0.5's damping ratio a box or a lying capsule never settles; at the floor it
    does, and a sphere (one point) settles either way."""
    assert _plain_drop(shape, ZETA_E05) > 5e-6
    assert _plain_drop(shape, NG.ZETA_MULTIPOINT) < 1e-7
    assert _plain_drop("sphere", ZETA_E05) < 1e-7


def _landed_box(deg: float) -> tuple[np.ndarray, dict]:
    """A 0.1 kg box (e 0.5, mu 1.2; pair mu with the slope's 1.0: 1.09) landed on a static slope from GAPS: per world,
    its slide [m] over 0.5-1.5 s, and sim.check()."""
    a = math.radians(deg)
    n = np.array([math.sin(a), 0.0, math.cos(a)])
    scene = alloy.Scene(worlds=W, dt=0.01)
    scene.add_ground()
    scene.add_box("slope", size=(0.6, 0.3, 0.02), pos=(0, 0, 0.3), rpy=(0, a, 0), static=True, friction=1.0)
    scene.add_material("blk_mat", density=1000.0, friction=1.2, restitution=0.5)
    pos = np.array([0, 0, 0.3]) + (0.02 + GAPS)[:, None] * n
    scene.add_box("blk", size=(0.05, 0.05, 0.02), pos=pos, rpy=(0, a, 0), mass=0.1, material="blk_mat")
    sim = alloy.build(scene, seed=0, device="cpu")
    sim.run(0.5)
    p0 = sim.state(["blk"])["blk.pos"].copy()
    sim.run(1.0)
    return np.linalg.norm(sim.state(["blk"])["blk.pos"] - p0, axis=-1), sim.check()


def test_landed_box_holds_on_a_slope() -> None:
    """40 deg (tan 0.84 < mu 1.09): every landing holds after 0.5 s (was up to 6 mm/s)."""
    slide, check = _landed_box(40.0)
    assert check["ok"], check["problems"][:3]
    assert slide.max() < 1e-4, f"landed boxes still slide: {np.round(slide * 1e3, 2).tolist()} mm/s"


def test_landed_box_slides_past_the_friction_angle() -> None:
    """Control: at 55 deg (tan 1.43 > mu 1.09) every box slides on."""
    slide, _ = _landed_box(55.0)
    assert slide.min() > 0.05


def test_floor_reaches_boxes_not_spheres() -> None:
    """The route's rows: a box takes the floor (e <= restitution_cap), a sphere keeps its card's e."""
    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_material("bouncy", density=1000.0, friction=0.5, restitution=0.5)
    scene.add_box("b", size=(0.05, 0.05, 0.02), pos=(0, 0, 0.0101), mass=0.1, material="bouncy")
    scene.add_sphere("s", 0.025, pos=(0.3, 0, 0.0251), mass=0.1, material="bouncy")
    sim = alloy.build(scene, device="cpu")
    zeta = {name: z for name, (_, z) in _geom_solref(sim).items()}
    assert zeta["b"] == pytest.approx(NG.ZETA_MULTIPOINT, rel=1e-6)
    assert zeta["s"] == pytest.approx(ZETA_E05, rel=1e-3)


def _geom_solref(sim) -> dict[str, tuple[float, float]]:
    """{object: its first geom's MuJoCo solref} from the CPU route's template model."""
    solver = next(s for s in sim.engine.stepper.solvers() if hasattr(s, "mj_model"))
    m = solver.mj_model
    out = {}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        obj = name.rsplit("_", 1)[0]
        if obj in ("b", "s") and obj not in out:
            out[obj] = (float(m.geom_solref[g, 0]), float(m.geom_solref[g, 1]))
    return out
