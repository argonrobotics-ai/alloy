# SPDX-License-Identifier: Apache-2.0
"""SB-7 scorer (CPU): mesh geoms collide as convex hulls, as MuJoCo collides them (``bench/xbench/ours/hull_meshes``).

``data/pair381.npz`` is WP3's T12 pair from episode (100000, 381), frame 2118 ([internal note] notes/wp3-item5): the
fingertip hull ``left_hull_linear_tip_left_4`` and the link-4 hull ``right_hull_model2__10_2``, vertices, triangles,
local shape transforms and body poses. MuJoCo's ``mj_geomDistance`` puts them +1.078 mm apart; Newton's mesh-mesh path
reported a -14.07 mm contact on that pose, a false S1 (pen_s1_ratio 1.41).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from alloy._vendor import newton

OURS = Path(__file__).resolve().parents[2] / "bench" / "xbench" / "ours"
sys.path.insert(0, str(OURS))

import hull_meshes  # noqa: E402
import score_s1  # noqa: E402
from packs_io import target_model  # noqa: E402

PAIR = Path(__file__).resolve().parent / "data" / "pair381.npz"
MUJOCO_GAP = 1.078e-3   # [m] mj_geomDistance on the posed hulls (WP3)


def _qrot(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    t = 2.0 * np.cross(q[:3], p)
    return p + q[3] * t + np.cross(q[:3], t)


def _pair_gaps(gap: float) -> tuple[list[int], list[float]]:
    """The pair as the scorer's model holds it (bodies at their poses, shapes at their local transforms, margin 0),
    hulled by ``hull_builder_meshes``: shape types and the signed gap of each contact Newton's pipeline reports."""
    z = np.load(PAIR)
    b = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=gap)
    poses = []
    for k in "ab":
        bq, sx = z[f"{k}_body_q"].astype(float), z[f"{k}_xform"].astype(float)
        body = b.add_body(xform=wp.transform(bq[:3], bq[3:]))
        mesh = newton.Mesh(z[f"{k}_verts"].astype(np.float32), z[f"{k}_tris"].reshape(-1).astype(np.int32))
        mesh.maxhullvert = -1   # T12's hull meshes (alloy.tasks.mjcf_tree.write_assets)
        b.add_shape_mesh(body, xform=wp.transform(sx[:3], sx[3:]), mesh=mesh, cfg=cfg)
        poses.append(bq)
    assert hull_meshes.hull_builder_meshes(b).shapes == 2
    m = b.finalize("cpu")
    st = m.state()
    st.body_q.assign(np.asarray(poses, dtype=np.float32))
    pipe = newton.CollisionPipeline(m, rigid_contact_max=64)
    c = pipe.contacts()
    pipe.collide(st, c)
    n, bq, sb = int(c.rigid_contact_count.numpy()[0]), st.body_q.numpy(), m.shape_body.numpy()
    gaps = []
    for i in range(n):
        w = [bq[sb[s[i]]][:3] + _qrot(bq[sb[s[i]]][3:], p[i]) for p, s in (
            (c.rigid_contact_point0.numpy(), c.rigid_contact_shape0.numpy()),
            (c.rigid_contact_point1.numpy(), c.rigid_contact_shape1.numpy()))]
        gaps.append(float(np.dot(w[1] - w[0], c.rigid_contact_normal.numpy()[i])))
    return m.shape_type.numpy().tolist(), gaps


def test_wp3_pair_as_hulls_has_no_false_contact():
    types, gaps = _pair_gaps(0.0)
    assert types == [int(newton.GeoType.CONVEX_MESH)] * 2
    assert gaps == []                                   # +1.08 mm apart: no contact at gap 0 (MESH: -14.07 mm)
    _, near = _pair_gaps(2e-3)                          # inside a 2 mm gap the hulls report their true distance
    assert len(near) == 1 and near[0] == pytest.approx(MUJOCO_GAP, abs=2e-5)


def test_hulls_are_shared_across_worlds_and_skip_sdf_shapes():
    tmpl = newton.ModelBuilder()
    mesh = newton.Mesh.create_box(0.1, 0.1, 0.1, compute_inertia=False)
    body = tmpl.add_body()
    tmpl.add_shape_mesh(body, mesh=mesh)
    tmpl.add_shape_mesh(body, mesh=mesh, xform=wp.transform((0.3, 0.0, 0.0), wp.quat_identity()))
    tmpl.add_shape_mesh(body, mesh=mesh, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False))
    tmpl.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    b = newton.ModelBuilder()
    b.replicate(tmpl, copy_count=3)
    b.shape_sdf_max_resolution[1] = 32                  # a precision shape: keeps MESH
    n = hull_meshes.hull_builder_meshes(b)
    assert (n.shapes, n.meshes, n.kept_sdf) == (5, 1, 1)
    hulls = {id(b.shape_source[i]) for i in range(len(b.shape_type)) if b.shape_type[i] == newton.GeoType.CONVEX_MESH}
    assert len(hulls) == 1                              # one hull for the shared mesh in every world
    assert [b.shape_type[i] for i in (1, 2, 3)] == [newton.GeoType.MESH] * 2 + [newton.GeoType.BOX]


@pytest.fixture(scope="module")
def t12_models():
    """T12's scorer model (hulls) and the live sim's model (its MJCF mesh geoms as MESH), one world each."""
    return target_model("task_t12", 1, "cpu"), target_model("task_t12", 1, "cpu", hulls=False)


def test_scorer_model_has_no_triangle_mesh_shapes_left(t12_models):
    hulls, live = t12_models
    assert hull_meshes.colliding_mesh_shapes(hulls).size == 0
    meshes = hull_meshes.colliding_mesh_shapes(live)
    assert meshes.size > 0
    assert (hulls.shape_type.numpy()[meshes] == int(newton.GeoType.CONVEX_MESH)).all()
    assert hulls.shape_count == live.shape_count
    assert np.array_equal(hulls.shape_body.numpy(), live.shape_body.numpy())


def test_live_model_scorer_collides_the_hull_twin(t12_models, caplog):
    hulls, live = t12_models
    with caplog.at_level(logging.WARNING, logger="xbench.ours"):
        assert score_s1.PipelineMonitors(live, collision_model=hulls).collision_model is hulls
        assert "not convex hulls" not in caplog.text
        score_s1.PipelineMonitors(live)
    assert "not convex hulls" in caplog.text
