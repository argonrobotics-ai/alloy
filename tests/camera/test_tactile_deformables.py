# SPDX-License-Identifier: Apache-2.0
"""Deformables press the gel (audit S1-003): particles as spheres, cloth / soft surfaces as thickened triangles, and
the same through an ``alloy.Sim`` with a soft sponge, cloth, MPM sand and SPH liquid (they read 0 mm before)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.camera.tactile import Augment, TactileSensor

GEL_Z = 0.002  # gel surface on the pad body's +Z face
QUIET = Augment(noise_std=0.0)


def pad_builder() -> newton.ModelBuilder:
    b = newton.ModelBuilder()
    pad = b.add_body(label="pad")
    b.add_shape_box(pad, hx=0.012, hy=0.01, hz=0.002)
    return b


def press(model, state=None, deformables=True) -> np.ndarray:
    s = TactileSensor("gel", attach_to="pad", pose=((0, 0, GEL_Z), (0, 0, 0, 1)), width=80, markers=False,
                      augment=QUIET, deformables=deformables)
    return s.render(model, state or model.state(), output="all").press.numpy()[0].copy()


def test_particle_press_is_its_spherical_cap():
    b = pad_builder()
    r, depth = 0.002, 0.0005  # a 2 mm particle 0.5 mm into the gel, off centre
    b.add_particle(pos=(0.003, -0.002, GEL_Z + r - depth), vel=(0, 0, 0), mass=0.001, radius=r)
    p = press(b.finalize(device="cpu"))
    mm = 18.88 / 80
    yy, xx = np.mgrid[0:60, 0:80].astype(np.float64)
    x, y = (xx + 0.5 - 40) * mm, (yy + 0.5 - 30) * mm
    want = np.clip(np.sqrt(np.clip(4.0 - (x - 3.0) ** 2 - (y + 2.0) ** 2, 0, None)) - 1.5, 0, None)
    assert np.abs(p - want).max() < 2e-4 and p.max() > 0.4


def test_cloth_press_is_its_depth_plus_half_thickness():
    b = pad_builder()
    t = 0.0005  # particle radius = half the cloth's thickness
    b.add_cloth_grid(pos=(-0.006, -0.004, GEL_Z - 0.0003), rot=wp.quat_identity(), vel=(0, 0, 0), dim_x=6, dim_y=4,
                     cell_x=0.002, cell_y=0.002, mass=0.01, particle_radius=t)
    p = press(b.finalize(device="cpu"))
    inside = p[20:40, 20:60]
    assert np.allclose(inside, 0.3 + 0.5, atol=1e-4)  # 0.3 mm into the gel + 0.5 mm half thickness
    assert p[0, 0] == 0.0 and p[59, 79] == 0.0


def test_deformables_off_and_nothing_near_the_gel():
    b = pad_builder()
    b.add_particle(pos=(0.0, 0.0, GEL_Z + 0.0015), vel=(0, 0, 0), mass=0.001, radius=0.002)
    b.add_particle(pos=(0.0, 0.0, 0.05), vel=(0, 0, 0), mass=0.001, radius=0.002)  # far above: no press
    model = b.finalize(device="cpu")
    assert press(model).max() == pytest.approx(0.5, abs=0.01)  # pixel centres sit 0.17 mm off the axis
    assert press(model, deformables=False).max() == 0.0


ARM = """<mujoco model="gripper_unit"><worldbody><body name="finger" pos="0 0 0.020">
  <joint name="pz" type="slide" axis="0 0 -1" range="0 0.05"/>
  <geom name="pad" type="box" size="0.02 0.02 0.005" mass="0.4"/></body></worldbody>
  <actuator><position joint="pz" kp="400" kv="20"/></actuator></mujoco>"""


@pytest.mark.parametrize("kind", ["rigid", "soft_sponge", "cloth", "sand", "liquid"])
def test_audit_scene_every_material_presses(kind):
    """The audit's scene: a finger pad 2 mm into a box of each material at the start (S1-003 read 0.0 mm)."""
    import alloy
    sc = alloy.Scene(dt=0.01)
    sc.add_ground()
    sc.add_robot("arm_left", mjcf=ARM, pos=(0.0, 0.0, 0.0))
    box = dict(size=(0.04, 0.04, 0.017), pos=(0.0, 0.0, 0.0085))
    if kind == "rigid":
        sc.add_box("item", static=True, **box)
    elif kind == "soft_sponge":
        sc.add_box("item", material="pu_foam_sponge", rigid=False, mass=0.05, **box)
    elif kind == "cloth":
        sc.add_cloth("item", size=(0.06, 0.06), pos=(0.0, 0.0, 0.017), resolution=8, mass=0.03, self_contact=False)
    elif kind == "sand":
        sc.add_box("item", material="sand_silica_dry", **box)
    else:
        sc.add_liquid("item", **box)
    sim = alloy.build(sc, device="cpu")
    sensor = TactileSensor("gel", attach_to="arm_left/finger", pose=((0, 0, -0.005), (1, 0, 0, 0)), width=80)
    peak = float(sensor.render(sim, output="all").press.numpy().max())
    want = {"rigid": (2.0, 2.0), "soft_sponge": (2.0, 3.0), "cloth": (2.0, 3.5), "sand": (1.0, 2.5),
            "liquid": (0.5, 2.5)}[kind]
    assert want[0] - 1e-3 <= peak <= want[1] + 1e-3, (kind, peak)
    assert math.isfinite(peak)
