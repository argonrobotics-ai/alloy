# SPDX-License-Identifier: Apache-2.0
"""Box against mesh triangles in Newton's CollisionPipeline: the contact depth is the true dip (CPU).

Newton 1.8's #4974 (536a3dd7, box-separating directions limited to a one-sided triangle's front) dropped the shallow
exit when it leaned a little toward the triangle's front, and the next front-side exit can cross the whole box: a 2 cm
cube mesh tilted 0.02 rad, 0.5 mm into the top face of a wide box, got one contact 65 mm deep pointing sideways
(XB4-2's bisect, [internal note] xb4/steel/repro; found as SolverSteel's 54x glitch rate, and the path of every mesh / sdf
shape on Newton's contact pipeline). Our copy carries XB4-2's fix (vendored patch 0011, VENDORED.md): such a direction
becomes its projection onto the triangle's plane, at the travel that clears the same axis.

Law:
- a cube mesh 0.5 or 0.9 mm into a box's top face, tilted 0.02 rad about three axes: no contact deeper than the dip
  + 0.1 mm, the deepest within 0.1 mm of the dip, every normal within twice the tilt of the face normal;
- what #4974 fixed stays: a thin box pitched 0.1 rad with its low edge 11.5 mm under a two-triangle ground mesh, across
  the seam, is pushed up by every contact (never down through the ground), none deeper than that edge.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

HALF = np.array([0.075, 0.125, 0.04])     # the wide static box
SIDE, TILT = 0.01, -0.02                  # the cube's half edge (m) and tilt (rad)
TOL = 1.0e-4                              # m


def _collide(builder: newton.ModelBuilder) -> tuple[newton.Model, newton.Contacts, int]:
    model = builder.finalize("cpu")
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    found = pipeline.contacts()
    pipeline.collide(model.state(), found)
    return model, found, int(found.rigid_contact_count.numpy()[0])


def _depths(found: newton.Contacts, count: int, first: int, pose_first: wp.transform, pose_other: wp.transform):
    """(depth, normal from shape ``first`` to the other) of every contact, in the world frame."""
    shape0, normal = found.rigid_contact_shape0.numpy(), found.rigid_contact_normal.numpy()
    point0, point1 = found.rigid_contact_point0.numpy(), found.rigid_contact_point1.numpy()
    out = []
    for i in range(count):
        is_first = shape0[i] == first
        on_first = wp.transform_point(pose_first, wp.vec3(*(point0[i] if is_first else point1[i])))
        on_other = wp.transform_point(pose_other, wp.vec3(*(point1[i] if is_first else point0[i])))
        direction = normal[i] if is_first else -normal[i]
        out.append((float((np.asarray(on_first) - np.asarray(on_other)) @ direction), direction))
    return out


@pytest.mark.parametrize("dip", [0.5e-3, 0.9e-3])
@pytest.mark.parametrize("axis", [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.6, 0.8, 0.0)])
def test_a_cube_mesh_dipping_into_a_box_face_reports_its_true_depth(axis, dip):
    cube = newton.Mesh.create_box(SIDE, SIDE, SIDE, duplicate_vertices=False, compute_normals=False,
                                  compute_uvs=False, compute_inertia=False)
    rotation = wp.quat_from_axis_angle(wp.vec3(*axis), TILT)
    lowest = min(float(wp.quat_rotate(rotation, wp.vec3(*v))[2]) for v in np.asarray(cube.vertices))
    pose = wp.transform(wp.vec3(-0.02, 0.03, float(HALF[2] - dip - lowest)), rotation)
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.0, gap=1.0e-3, density=0.0)
    body = builder.add_body(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
    box = builder.add_shape_box(-1, hx=HALF[0], hy=HALF[1], hz=HALF[2], cfg=cfg)
    builder.add_shape_mesh(body, mesh=cube, cfg=cfg)
    _, found, count = _collide(builder)
    rows = _depths(found, count, box, wp.transform_identity(), pose)
    assert rows
    deepest = max(d for d, _ in rows)
    print(f"tilt about {axis}, dip {1e3 * dip:.1f} mm: {count} contacts, deepest {1e3 * deepest:.4f} mm, "
          f"lowest normal z {min(n[2] for _, n in rows):.5f}")
    assert deepest == pytest.approx(dip, abs=TOL), rows
    assert min(n[2] for _, n in rows) >= math.cos(2.0 * TILT), rows


@pytest.mark.parametrize("yaw", [0.0, 0.4])
def test_a_thin_box_sunk_into_a_ground_mesh_is_pushed_up_across_the_seam(yaw):
    """#4974's case (Newton's test_refinement_triangle_box_seam): a 10 mm thick box pitched 0.1 rad whose low edge is
    11.5 mm under a two-triangle ground. The shallow exit is down through the ground; a one-sided mesh must push it up.
    Before #4974 (Newton ac7f7830) one of its contacts pushed it down."""
    half, height, pitch = (0.085, 0.03, 0.005), 0.002, 0.1
    depth = half[0] * math.sin(pitch) + half[2] * math.cos(pitch) - height
    ground = newton.Mesh(np.array([[-0.2, -0.2, 0], [0.2, -0.2, 0], [0.2, 0.2, 0], [-0.2, 0.2, 0]], dtype=np.float32),
                         np.array([0, 1, 2, 0, 2, 3], dtype=np.int32), compute_inertia=False)
    turn = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    pose = wp.transform(wp.vec3(0.0, 0.0, height), turn * wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), pitch))
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.0, gap=1.0e-3, density=0.0)
    body = builder.add_body(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
    mesh = builder.add_shape_mesh(-1, mesh=ground, cfg=cfg)
    builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
    _, found, count = _collide(builder)
    rows = _depths(found, count, mesh, wp.transform_identity(), pose)
    assert rows
    print(f"yaw {yaw}: {count} contacts, depths {[round(1e3 * d, 3) for d, _ in rows]} mm, "
          f"normal z {[round(float(n[2]), 3) for _, n in rows]}")
    assert min(n[2] for _, n in rows) > 0.0, rows                    # every contact lifts the box: the front side
    assert max(d for d, _ in rows) <= depth + TOL, (depth, rows)
