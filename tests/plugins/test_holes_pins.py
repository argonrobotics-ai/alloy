# SPDX-License-Identifier: Apache-2.0
"""Listed pins against drilled holes (drilling v2, phase B): a pin slid off the plate and back, one moved sideways
under the face into a hole, and the rigid-wall shift of a tilted shank (CPU, light)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

from alloy.plugins.drilling import Drill
from alloy.plugins.holes.pin_geom import Holes, PinSpec, contact_of
from alloy.plugins.holes.pin_kernels import legal_shifts


def _slide_scene():
    """One bolt on plate_a (pushed at 20 N, free sideways) over a 7.9 mm hole at x = 10 mm and a 9 mm one at -12 mm."""
    from alloy.plugins.harness.check import fresh
    from alloy.plugins.harness.scene import build_simulation
    from alloy.plugins.holes import pin_laws as pl
    from alloy.plugins.holes.pins import PushHead

    law = dataclasses.replace(pl.STRONG, cases=(pl.Case(((0.0079, pl.THROUGH),), (), 0.0, "a"),), force=20.0)
    d = Drill()
    scene = dataclasses.replace(pl.pin_scene(d.config, law), control_dt=pl.STEP * 16,
                                companions=lambda: [PushHead(body="bolt", force=20.0, d_push=2500.0, d_side=0.0)])
    sim = build_simulation(scene, [fresh(d, {"layers": pl.LAYERS, "pins": law.pins()})], "cpu")
    drill = sim.plugins["drill"]
    S = int(drill.config.max_stages)
    tables = {k: getattr(drill, k).numpy().copy() for k in ("hole_n", "hole_entry", "hole_axis", "stage_n",
                                                             "stage_d", "stage_b")}
    tables["hole_n"][0, 0] = 2
    tables["hole_entry"][0, 0, 0], tables["hole_entry"][0, 1, 0] = 0.010, -0.012
    tables["hole_axis"][0, 1] = (0.0, 0.0, -1.0)
    tables["stage_n"][0, 1], tables["stage_d"][0, S], tables["stage_b"][0, S] = 1, 0.009, pl.THROUGH
    for k, v in tables.items():
        getattr(drill, k).assign(v)
    _place(sim, x=0.027, dz=-(pl.GAP - 0.0001))
    return sim, drill, pl


def _place(sim, x=None, dz=0.0, vx=0.0):
    q, qd = sim.state.joint_q.numpy(), sim.state.joint_qd.numpy()
    if x is not None:
        q[0] = x
    q[2] += dz
    qd[0] = vx
    sim.state.joint_q.assign(q)
    sim.state.joint_qd.assign(qd)
    newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)


@pytest.mark.compile_heavy
def test_a_pin_slid_off_the_plate_and_back_lands_again():
    """A bolt resting on plate_a's face slides at 0.3 m/s past its 30 mm edge: held until its whole 8 mm end is off
    (34 mm), then let go. It sinks 0.3 mm below the face and comes back: it lands on the face again and rises to it
    (no kick), rests across the 7.9 mm mouth it does not fit, and drops into the 9 mm hole."""
    sim, drill, pl = _slide_scene()
    face = 2.0 * pl.PLATE
    for _ in range(10):
        sim.step_period()
    _place(sim, vx=0.3)
    x = lambda: float(sim.state.body_q.numpy()[0, 0])                             # noqa: E731
    tip = lambda: float(sim.state.body_q.numpy()[0, 2]) - pl.SHANK                # noqa: E731
    while x() < 0.0335:
        sim.step_period()
        assert int(drill.read()["pin_land"][0, 0]) == 2 or x() > 0.034           # still on the plate's edge
    while x() < 0.036 or tip() > face - 0.0003:
        sim.step_period()
    assert int(drill.read()["pin_land"][0, 0]) == 0                               # let go: it sank past the face
    _place(sim, vx=-0.3)
    lowest, vmax = tip(), 0.0
    while x() > -0.0095:
        sim.step_period()
        r = drill.read()
        vmax = max(vmax, abs(float(sim.state.body_qd.numpy()[0, 0])))
        if x() < 0.029:
            assert int(r["pin_land"][0, 0]) != 0 and int(r["pin_solid"][0]) == 0     # on the face, or the mouth
            assert tip() >= lowest - 1e-6 and tip() > face - 0.0004                   # rises to the face, never in
    assert vmax < 0.3 * 1.01                                                     # no kick
    for _ in range(60):                                                          # 8 mm/s at 20 N
        sim.step_period()
    r = drill.read()
    assert tip() < face - 0.001 and int(r["pin_hole"][0, 0]) == 2 and int(r["pin_land"][0, 0]) == 0   # in the 9 mm


@pytest.mark.compile_heavy
def test_a_pin_moved_sideways_under_the_face_joins_a_hole_without_a_kick():
    """A bolt started 0.3 mm into plate_a's solid face (embedded) and moved sideways at 0.3 m/s toward a 9 mm hole:
    it joins the hole only where its end fits, so no wall spring switches on deep (no kick), and drops in."""
    sim, drill, pl = _slide_scene()
    _place(sim, x=-0.003, dz=-0.0004, vx=-0.3)                                   # from 0.1 mm above to 0.3 mm in
    face = 2.0 * pl.PLATE
    vmax = 0.0
    for _ in range(40):
        sim.step_period()
        vmax = max(vmax, abs(float(sim.state.body_qd.numpy()[0, 0])))
    assert vmax < 0.3 * 1.01, vmax                                               # no kick
    for _ in range(60):
        sim.step_period()
    tip = float(sim.state.body_q.numpy()[0, 2]) - pl.SHANK
    assert tip < face - 0.001 and int(drill.read()["pin_hole"][0, 0]) == 2       # in the 9 mm hole


@wp.kernel
def _shifts(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], pin_body: wp.array2d[wp.int32],
            face_body: wp.array2d[wp.int32], face_local: wp.array2d[wp.transform], pin: PinSpec,
            thickness: wp.array[wp.float32], half: wp.array[wp.vec2], bit: Holes, hole_entry: wp.array3d[wp.float32],
            hole_axis: wp.array3d[wp.float32], stage_n: wp.array2d[wp.int32], stage_d: wp.array2d[wp.float32],
            stage_b: wp.array2d[wp.float64], pin_hole: wp.array2d[wp.int32], out: wp.array[wp.vec3]):
    ct = contact_of(body_q, body_com, pin_body, face_body, face_local, pin, thickness, half, 0, 1, 2)
    sh = legal_shifts(body_q, body_com, pin_body, face_body, face_local, pin, thickness, half, 0, 1, 2, bit,
                      hole_entry, hole_axis, stage_n, stage_d, stage_b, pin_hole, ct)
    out[0] = sh.tip
    out[1] = sh.face


def test_a_tilted_shank_is_moved_onto_the_rigid_walls_line():
    """An 8 mm shank tilted 6 deg through plate_a's 8.4 mm hole (faces at z = 12 and 6 mm) presses its wall at both
    crossings, on opposite sides, 0.115 mm deep. With rigid walls it would lie on the line through the two crossings
    moved back to the wall: at its tip, 3 mm into plate_b, that line is d0 + 1.5 (d1 - d0) = +0.231 mm off where the
    tip is (a translation would give 0.115), and at plate_b's face crossing d1."""
    th, r, rho = math.radians(6.0), 0.004, 0.0042
    tip = np.array([-0.006 * math.tan(th), 0.0, 0.003])
    top = tip + 0.03 * np.array([math.sin(th), 0.0, math.cos(th)])
    q = np.array([[*top, 0.0, math.sin(th / 2.0), 0.0, math.cos(th / 2.0)]])
    pin = PinSpec()
    pin.top, pin.tip, pin.radius = wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, -0.03), r
    bit = Holes()
    bit.holes, bit.stages, bit.tan_half = 1, 1, math.tan(math.radians(59.0))
    dev = "cpu"
    arr = lambda v, t: wp.array(np.asarray(v), dtype=t, device=dev)                  # noqa: E731
    faces = [[(0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.006, 0.0, 0.0, 0.0, 1.0)]]
    out = wp.zeros(2, dtype=wp.vec3, device=dev)
    wp.launch(_shifts, dim=1, inputs=[
        arr(q, wp.transform), arr(np.zeros((1, 3)), wp.vec3), arr([[0]], wp.int32), arr([[-1, -1]], wp.int32),
        arr(faces, wp.transform), pin, arr([0.006, 0.006], wp.float32), arr([(0.03, 0.03), (0.03, 0.03)], wp.vec2),
        bit, arr([[(0.0, 0.0), (0.0, 0.0)]], wp.float32), arr([[(0.0, 0.0, -1.0)] * 2], wp.float32),
        arr([[1, 1]], wp.int32), arr([[2.0 * rho, 0.009]], wp.float32), arr([[0.016, 0.016]], wp.float64),
        arr([[1, 1]], wp.int32)], outputs=[out], device=dev)
    pen = 0.003 * math.tan(th) + r - rho
    sh_tip, sh_face = out.numpy().astype(np.float64)
    assert pen == pytest.approx(1.153e-4, rel=1e-3)
    assert sh_tip == pytest.approx((-pen + 1.5 * 2.0 * pen, 0.0, 0.0), abs=1e-7)
    assert sh_face == pytest.approx((pen, 0.0, 0.0), abs=1e-7)
