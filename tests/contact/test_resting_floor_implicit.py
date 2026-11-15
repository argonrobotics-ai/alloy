# SPDX-License-Identifier: Apache-2.0
"""Resting stacks stay still beside a robot or a moving fixture (2026-11-14, found on the classic Jenga sample).

Entries on implicitfast's own gains (a robot, a kinematic body, a chain) took the multi-point damping floor rk4
needs (0.7), and there a stack of light boxes never came to rest: an 18-level tower of 12.6 g crossed alder blocks
beside a kinematic body bobbed at 3.9 mm/s RMS for good, every level's contacts at 66-100 % of their friction limit,
so a push on one block sheared the whole tower (32 mm at the top). At 0.85 it still bobbed at 2.7 mm/s, at 0.95 and
1.0 at 0.001 mm/s; under rk4 it rests at 0.7. (On CUDA too: 3.3 mm/s at 0.7, 0.03 mm/s at 1.0.) Those entries now
take `native_gains.ZETA_MULTIPOINT_IMPLICIT` (1.0), at the price of a little box rebound; rk4 and free-body clutter
keep `ZETA_MULTIPOINT` (0.7), bit for bit (test_resting_floor.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import alloy
from alloy.scene import native_gains as NG

L, W, H, GAP = 0.075, 0.025, 0.015, 0.0005      # the sample's alder blocks (12.6 g)
LEVELS = 10


def _device(name: str) -> str:
    if name == "cuda":
        import warp as wp

        if not wp.is_cuda_available():
            pytest.skip("no CUDA device")
        return "cuda:0"
    return "cpu"


def _with_far_kinematic(scene: alloy.Scene) -> None:
    """A kinematic box 1.4 m away: the entry runs implicitfast on its own gains, as beside a robot."""
    scene.add_box("far", size=(0.01, 0.01, 0.01), pos=(1.0, 1.0, 0.005), kinematic=True)


def _tower(levels: int, device: str) -> tuple[alloy.Simulation, list[str]]:
    scene = alloy.Scene(dt=1.0 / 300)
    scene.add_ground()
    names = []
    for lv in range(levels):
        z = H / 2 + lv * H
        for k in (-1, 0, 1):
            pos, size = (((k * (W + GAP), 0.0, z), (W, L, H)) if lv % 2 == 0
                         else ((0.0, k * (W + GAP), z), (L, W, H)))
            names.append(scene.add_box(f"b{lv:02d}_{k + 1}", size=size, pos=pos, material="wood_alder"))
    _with_far_kinematic(scene)
    return alloy.build(scene, device=device, contact_capacity=2048), names


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_crossed_stack_beside_a_kinematic_body_rests(device: str) -> None:
    """After a 0.5 s settle a 10-level tower is still: on the CPU vertical speed under 0.01 mm/s RMS and no block
    drifting 0.01 mm in 0.5 s (at the old floor: 1.65 mm/s RMS and 0.36 mm over 1.7 s; 6 levels rested even then).
    On CUDA contact does not repeat from run to run, so its limits are 1.0 mm/s and 0.1 mm: on an L4 at the 1.0 floor
    the same stack read 0.0004-0.164 mm/s in seven runs and 0.565 mm/s once, at 0.7 1.69-1.80 mm/s and 0.13-0.14 mm."""
    sim, names = _tower(LEVELS, _device(device))
    sim.settle(0.5)
    p0 = sim.state(names, world=0)
    vz2 = []
    for _ in range(round(0.5 / sim.dt)):
        sim.step(1)
        st = sim.state(names, world=0)
        vz2 += [float(np.asarray(st[n + ".vel"]).reshape(3)[2]) ** 2 for n in names]
    drift = max(float(np.linalg.norm(np.asarray(st[n + ".pos"]) - np.asarray(p0[n + ".pos"]))) for n in names)
    check = sim.check()
    assert check["ok"], check["problems"][:3]
    vz_max, drift_max = (1e-5, 1e-5) if device == "cpu" else (1e-3, 1e-4)
    assert math.sqrt(np.mean(vz2)) < vz_max, f"the stack jitters: {math.sqrt(np.mean(vz2)) * 1e3:.3f} mm/s RMS"
    assert drift < drift_max, f"a block drifted {drift * 1e3:.3f} mm"


def test_the_floor_follows_the_gains_mode() -> None:
    """The rows written: beside a kinematic body a box takes 1.0, a sphere keeps its card's e (rk4 entries: 0.7,
    test_resting_floor.py)."""
    import mujoco

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_material("bouncy", density=1000.0, friction=0.5, restitution=0.5)
    scene.add_box("b", size=(0.05, 0.05, 0.02), pos=(0, 0, 0.0101), mass=0.1, material="bouncy")
    scene.add_sphere("s", 0.025, pos=(0.3, 0, 0.0251), mass=0.1, material="bouncy")
    _with_far_kinematic(scene)
    sim = alloy.build(scene, device="cpu")
    m = sim.engine.stepper.solver.mj_model
    zeta = {}
    for g in range(m.ngeom):
        obj = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").rsplit("_", 1)[0]
        zeta.setdefault(obj, float(m.geom_solref[g, 1]))
    assert NG.ZETA_MULTIPOINT_IMPLICIT == 1.0 and NG.ZETA_MULTIPOINT == 0.7
    assert zeta["b"] == pytest.approx(NG.ZETA_MULTIPOINT_IMPLICIT, rel=1e-6)
    assert zeta["s"] == pytest.approx(0.21545376, rel=1e-3)


def test_box_rebound_beside_a_kinematic_body_is_capped() -> None:
    """A 5 cm aluminium box (card e 0.6) dropped flat 10 cm onto a steel slab rebounds at e 0.22 (0.26 at the old
    floor): under 0.24, so the floor holds; above restitution_cap(1.0) (0.135), as the stability cap sets the
    contact's time constant."""
    scene = alloy.Scene(dt=1.0 / 300)
    scene.add_ground()
    scene.add_box("slab", size=(0.4, 0.4, 0.05), pos=(0, 0, 0.025), static=True, material="steel_carbon")
    scene.add_box("box", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.175), material="aluminium_6061_t6")
    _with_far_kinematic(scene)
    sim = alloy.build(scene, device="cpu")
    z = []
    for _ in range(round(0.6 / sim.dt)):
        sim.step(1)
        z.append(float(np.asarray(sim.state(["box"], world=0)["box.pos"]).reshape(3)[2]))
    z = np.asarray(z) - 0.075
    low = int(np.argmin(z[: round(0.25 / sim.dt)]))
    e = math.sqrt(max(float(z[low:].max()), 0.0) / 0.10)
    assert sim.check()["ok"]
    assert 0.15 < e < 0.24, f"rebound e {e:.3f}"


ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="0"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/></body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="0"/></actuator></mujoco>"""


def test_the_rigid_step_planner_reads_the_floor_the_build_writes(caplog) -> None:
    """Beside a robot with the rigid step pinned at 2.5 ms, the planner's broken sag cap states the sag of the gains
    the build writes for the box (it read the rk4 floor once: rest sag 0.0146 mm, where the written gains give
    0.0246 mm, and a band of masses took a step 20 % softer than the fixed one)."""
    import logging
    import re

    from alloy.scene import rigid_step as RS

    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    s.add_box("block", size=(0.1, 0.1, 0.1), pos=(0.6, 0, 0.05), mass=1.0)
    s.rigid_step = 0.0025
    with caplog.at_level(logging.WARNING):
        sim = alloy.build(s, device="cpu")
    text = " ".join(r.getMessage() for r in caplog.records)
    got = re.search(r"block: contacts [0-9.]+ as stiff as at the fixed step \(rest sag ([0-9.e-]+) mm against "
                    r"[0-9.e-]+ mm under load x([0-9.]+)", text)
    assert got, text[:500]
    model = sim.engine.model
    labels = [str(x) for x in (getattr(model, "shape_label", None) or model.shape_key)]
    i = next(k for k, n in enumerate(labels) if "block" in n)
    ke, kd = float(model.shape_material_ke.numpy()[i]), float(model.shape_material_kd.numpy()[i])
    written = RS.sag(ke, kd, float(got.group(2))) * 1e3
    assert written == pytest.approx(float(got.group(1)), rel=0.01), (got.group(0), written)
