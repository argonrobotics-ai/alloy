# SPDX-License-Identifier: Apache-2.0
"""The thin-part PEN_FAIL rule reads body solids along the contact normal (coordinator rulings 2026-11-05 02:00Z and
refinement, SB-7 S1s).

A rigid contact deeper than half the thinner part is PEN_FAIL. The colliding shapes of one rigid body are one solid,
and its thickness is its width along the contact normal. Static shapes are separate parts, grouped only by their
decomposition parent. T12 declares one 75 x 25 x 15 mm block as three 8.33 mm boxes, and a fingertip as convex
pieces; read piece by piece, 3.9-4.7 mm contacts into 12.7 mm plywood were S1. Cases, on Newton contacts and on
MuJoCo Warp's native contacts:
- the three-box block 4.4 mm into plywood does not fire; one 8.33 mm box on its own body, thin side down, does;
- a fingertip in four 4 mm slabs on one body (T12's piece names) 3.9 mm into plywood does not fire;
- a 1 mm plate on its own body 0.8 mm into plywood fires; a static 1 mm plate on a static table, pierced 0.8 mm
  by a fingertip, fires;
- along the normal: the block flat (15 mm up) fires 8 mm deep, on its side (25 mm up) it does not;
- a static sheet authored as two 6.35 mm layers ``sheet_p0`` / ``sheet_p1`` reads 12.7 mm;
- the widths of every primitive along a tilted normal match their closed forms.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton
from alloy.precision.health import HealthMonitor
from alloy.precision.health_source_shapes import SolidTable, build_solids, solid_ids, solid_width
from alloy.precision.types import CheckPolicy, HealthFlags

POLICY = CheckPolicy(level="cheap", every=1, full_every=1, native_pen_flags=True)
PEN_FAIL = int(HealthFlags.PEN_FAIL)
PLY = 0.00635                       # half of T12's 12.7 mm plywood
BOX = (0.0375, 0.0041667, 0.0075)   # T12's block_k_box<i>: three side by side along y make 75 x 25 x 15 mm
SLAB = (0.015, 0.010, 0.002)        # one 4 mm piece of a 30 x 20 x 16 mm fingertip
I = wp.quat_identity()


def _slab_mesh(z0: float) -> newton.Mesh:
    hx, hy, hz = SLAB
    v = np.array([[x, y, z0 + z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], dtype=np.float32)
    tri = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                    [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], dtype=np.int32)
    return newton.Mesh(v, tri.reshape(-1))


def _floor(b: newton.ModelBuilder, case: str) -> float:
    """Static floor(s); returns the top surface height."""
    if case == "static_plate":  # a 1 mm plate lying on a 40 mm table: two static parts
        b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, -0.020), I), hx=0.2, hy=0.2, hz=0.020, label="table")
        b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.0005), I), hx=0.1, hy=0.1, hz=0.0005, label="plate")
        return 0.001
    if case == "layered":  # one 12.7 mm sheet authored as two static layers
        for i in range(2):
            b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, (i - 0.5) * PLY), I), hx=0.2, hy=0.2, hz=0.5 * PLY,
                            label=f"task/worldbody/sheet/sheet_p{i}")
        return PLY
    thick = 0.020 if case.startswith("block_") else PLY  # the orientation cases sit on a 40 mm slab
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, 0.0), I), hx=0.2, hy=0.2, hz=thick,
                    label="task/worldbody/plywood_sheet/plywood_sheet")
    return thick


def _body(b: newton.ModelBuilder, case: str, top: float, depth: float) -> None:
    if case in ("block", "layered", "block_flat", "block_side"):
        side = case == "block_side"  # 90 deg about x: the 25 mm side points up
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * math.pi) if side else I
        up = 3.0 * BOX[1] if side else BOX[2]
        body = b.add_body(xform=wp.transform((0.0, 0.0, top + up - depth), q), label="block_1")
        for i, y in enumerate((-2.0 * BOX[1], 0.0, 2.0 * BOX[1])):
            b.add_shape_box(body, xform=wp.transform((0.0, y, 0.0), I), hx=BOX[0], hy=BOX[1], hz=BOX[2],
                            label=f"task/worldbody/block_1/block_1_box{i}")
    elif case == "one_box":  # one 8.33 mm box on its own body, its thin side along the normal
        body = b.add_body(xform=wp.transform((0.0, 0.0, top + BOX[1] - depth), I), label="block_1")
        b.add_shape_box(body, hx=BOX[0], hy=BOX[2], hz=BOX[1], label="task/worldbody/block_1/block_1_box0")
    elif case == "thin_body":  # a 1 mm plate on its own body
        body = b.add_body(xform=wp.transform((0.0, 0.0, top + 0.0005 - depth), I), label="plate")
        b.add_shape_box(body, hx=0.03, hy=0.03, hz=0.0005, label="plate/plate")
    elif case == "hull":
        body = b.add_body(xform=wp.transform((0.0, 0.0, top + 4.0 * SLAB[2] - depth), I), label="finger")
        for i in range(4):  # 4 mm slabs stacked along z, the lowest touching; named as T12 names its pieces
            label = f"finger/left_hull_tip_{i}_p0" if i < 2 else f"finger/left_hull_tip_2_p{i - 2}"
            b.add_shape_convex_hull(body, mesh=_slab_mesh((2 * i - 3) * SLAB[2]), label=label)
    else:  # "static_plate": a fingertip sphere through the 1 mm plate
        body = b.add_body(xform=wp.transform((0.0, 0.0, top + 0.008 - depth), I), label="finger")
        b.add_shape_sphere(body, radius=0.008, label="finger/fingertip")


def _scene(case: str, depth: float) -> tuple[newton.Model, newton.State]:
    b = newton.ModelBuilder()
    _body(b, case, _floor(b, case), depth)
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(b, 1)  # world 0 (the health checks skip global shapes)
    model = scene.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return model, state


def _check(case: str, depth: float, native: bool = False) -> tuple[int, float]:
    """(PEN_FAIL bit, max_pen_m) of one cheap check of the scene's Newton or MuJoCo Warp contacts."""
    model, state = _scene(case, depth)
    if native:
        solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=True)
        s1 = model.state()
        solver.step(state, s1, model.control(), None, 1.0e-6)
        monitor = HealthMonitor(model, POLICY, solvers=[solver], energy="off")
        monitor.update(0, state, s1, None)
    else:
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        monitor = HealthMonitor(model, POLICY, energy="off")
        monitor.update(0, state, state, contacts)
    report = monitor.poll()
    return int(report.flags[0]) & PEN_FAIL, float(report.max_pen_m[0])


CASES = [  # (case, depth [m], fires)
    ("block", 0.0044, False),
    ("one_box", 0.0044, True),
    ("hull", 0.0039, False),
    ("thin_body", 0.0008, True),
    ("static_plate", 0.0008, True),
    ("block_flat", 0.0080, True),
    ("block_side", 0.0080, False),
    ("layered", 0.0044, False),
]


@pytest.mark.parametrize("native", [False, True], ids=["newton", "native"])
@pytest.mark.parametrize("case,depth,fires", CASES, ids=[c[0] for c in CASES])
def test_thin_part_rule_reads_body_solids(case: str, depth: float, fires: bool, native: bool) -> None:
    flag, pen = _check(case, depth, native)
    assert abs(pen - depth) < 2.5e-4, pen
    assert flag == (PEN_FAIL if fires else 0), f"{case} {depth * 1e3:.1f} mm: PEN_FAIL {bool(flag)}"


def test_solids_on_t12_style_bodies() -> None:
    model, _ = _scene("layered", 0.0)
    ids = solid_ids(model)
    assert ids[0] == ids[1] == 0 and len(set(ids[2:].tolist())) == 1  # two layers = one parent; block = its body
    _, summary = build_solids(model)
    assert (summary.solids, summary.multi) == (2, 2)
    model, _ = _scene("static_plate", 0.0)
    assert len(set(solid_ids(model)[:2].tolist())) == 2  # two unrelated static parts stay apart


@wp.kernel
def _widths(tbl: SolidTable, shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3],
            shape_transform: wp.array[wp.transform], n: wp.vec3, out: wp.array[float]):
    s = wp.tid()
    out[s] = solid_width(tbl, shape_type, shape_scale, shape_transform, s, n)


def test_primitive_widths_match_closed_forms() -> None:
    r, h, a = 0.01, 0.02, (0.01, 0.02, 0.03)
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.3)
    b = newton.ModelBuilder()
    adds = [lambda body: b.add_shape_sphere(body, radius=r),
            lambda body: b.add_shape_capsule(body, radius=r, half_height=h),
            lambda body: b.add_shape_cylinder(body, radius=r, half_height=h),
            lambda body: b.add_shape_cone(body, radius=r, half_height=h),
            lambda body: b.add_shape_ellipsoid(body, rx=a[0], ry=a[1], rz=a[2]),
            lambda body: b.add_shape_box(body, xform=wp.transform((0.0, 0.0, 0.0), tilt), hx=a[0], hy=a[1], hz=a[2])]
    for i, add in enumerate(adds):
        add(b.add_body(xform=wp.transform((i * 0.2, 0.0, 0.0), I)))
    model = b.finalize(device="cpu")
    tbl, _ = build_solids(model)
    n = np.array([0.36, 0.48, 0.8])  # unit
    out = wp.zeros(model.shape_count, dtype=float, device="cpu")
    wp.launch(_widths, dim=model.shape_count, inputs=[tbl, model.shape_type, model.shape_scale, model.shape_transform,
                                                       wp.vec3(*n), out], device="cpu")
    radial = math.hypot(n[0], n[1])
    nb = np.array(wp.quat_rotate_inv(tilt, wp.vec3(*n)))
    cone_lo = min(h * n[2], -h * n[2] - r * radial)  # apex at +h or the base rim at -h, whichever is lower
    want = [2 * r, 2 * (h * n[2] + r), 2 * (h * n[2] + r * radial), max(h * n[2], -h * n[2] + r * radial) - cone_lo,
            2 * math.sqrt(sum((ai * ni) ** 2 for ai, ni in zip(a, n))), 2 * sum(ai * abs(ni) for ai, ni in zip(a, nb))]
    np.testing.assert_allclose(out.numpy(), want, rtol=1e-5)
