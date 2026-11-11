# SPDX-License-Identifier: Apache-2.0
"""RenderRecorder end-to-end on tiny CPU Newton models (no solver: states are moved by hand)."""
from __future__ import annotations

import json

from alloy._vendor import newton
import numpy as np
import pytest
import warp as wp

from alloy.render import BundleError, RenderRecorder, load_bundle, validate_bundle
from tests.render.helpers import IDENT, move_bodies, rigid_world, shift_particles

CPU = "cpu"


def record(model, path, n=3, **kw):
    """Capture ``n`` frames, moving bodies +0.1 m in z and particles +0.1 m in x each frame."""
    rec = kw.pop("rec", None) or RenderRecorder(model, path, fps=30.0, **kw)
    st = model.state()
    for k in range(n):
        rec.capture(st, k / 30.0)
        if model.body_count:
            move_bodies(st, 0.1)
        if model.particle_count:
            shift_particles(st, 0.1)
    rec.close()
    b = load_bundle(path)
    assert validate_bundle(b) == []
    return b


def test_rigid_shapes_and_hidden(tmp_path):
    model = rigid_world().finalize(device=CPU)
    b = record(model, tmp_path / "b", meta={"clip": "T"})
    sc = b.scene
    assert sc["arb"] == 1 and sc["meta"]["clip"] == "T" and sc["meta"]["device"] == CPU
    assert b.frame_count == 3 and b.fps == 30.0 and b.duration_s == pytest.approx(2 / 30)
    labels = b.labels()
    assert labels["shapes"] == ["ground", "box_shape", "sphere_shape", "capsule_shape", "mesh_shape"]
    assert labels["bodies"] == ["box_body", "sphere_body", "capsule_body", "mesh_body"]
    by = {s["label"]: s for s in sc["shapes"]}
    assert by["ground"]["body"] == -1 and by["ground"]["type"] == "box"
    assert by["box_shape"]["size"] == pytest.approx([0.1, 0.2, 0.3])
    assert by["sphere_shape"]["size"] == pytest.approx([0.15])
    assert by["capsule_shape"]["size"] == pytest.approx([0.05, 0.2])
    m = by["mesh_shape"]
    assert m["type"] == "mesh" and m["scale"] == pytest.approx([2, 2, 2]) and m["mesh"] in sc["meshes"]
    assert sc["meshes"][m["mesh"]] == {"file": f"meshes/{m['mesh']}.npz", "verts": 4, "faces": 4, "uv": False}
    q = b.array("body_q")
    assert q.shape == (3, 4, 7)
    assert q[2, 0, 2] == pytest.approx(1.2) and q[0, 1, 0] == pytest.approx(1.0)
    assert b.array("t").dtype == np.float64 and b.nbytes() > 0
    assert sc["hero_world"] == 0 and sc["worlds"] == [0]


def test_identical_meshes_are_deduped(tmp_path):
    bld = rigid_world()
    other = bld.add_body(xform=wp.transform((4, 0, 1), IDENT), label="mesh2_body")
    from tests.render.helpers import tetra_mesh
    bld.add_shape_mesh(other, mesh=tetra_mesh(), label="mesh2_shape")
    b = record(bld.finalize(device=CPU), tmp_path / "b", n=1)
    assert len(b.scene["meshes"]) == 1


def two_world_model():
    sub = rigid_world()
    top = newton.ModelBuilder()
    top.add_shape_box(-1, hx=5, hy=5, hz=0.05, label="global_ground")
    top.replicate(sub, 2)
    return top.finalize(device=CPU)


def test_two_worlds_grid_and_replicas(tmp_path):
    model = two_world_model()
    b = record(model, tmp_path / "b", n=2, worlds="all", grid_m=10.0)
    sc = b.scene
    assert sc["worlds"] == [0, 1] and sc["world_offsets"] == [[0, 0, 0], [10, 0, 0]]
    ground = [s for s in sc["shapes"] if s["label"] == "global_ground"]
    assert len(ground) == 1 and ground[0]["world"] == -1 and ground[0]["xform"][0] == 0
    hero = {s["label"]: s for s in sc["shapes"] if s["world"] == 0}
    w1 = [s for s in sc["shapes"] if s["world"] == 1]
    assert len(w1) == len(hero)
    for s in w1:
        assert s["replica_of"] == hero[s["label"]]["id"]
    assert all(s["replica_of"] is None for s in sc["shapes"] if s["world"] != 1)
    static1 = next(s for s in w1 if s["body"] == -1)
    assert static1["xform"][0] == pytest.approx(hero[static1["label"]]["xform"][0] + 10.0)
    q = b.array("body_q")
    worlds = [e["world"] for e in sc["bodies"]]
    i0, i1 = worlds.index(0), worlds.index(1)
    assert q[0, i1, 0] - q[0, i0, 0] == pytest.approx(10.0)


def test_single_world_selection_and_hero(tmp_path):
    model = two_world_model()
    b = record(model, tmp_path / "b", n=1, worlds=[1])
    sc = b.scene
    assert sc["worlds"] == [1] and sc["hero_world"] == 1 and sc["world_offsets"] == [[0, 0, 0]]
    assert {e["world"] for e in sc["bodies"]} == {1} and len(sc["bodies"]) == 4
    assert all(s["replica_of"] is None for s in sc["shapes"])


def test_cloth_surface(tmp_path):
    bld = newton.ModelBuilder()
    bld.add_cloth_grid(pos=wp.vec3(0, 0, 1), rot=IDENT, vel=wp.vec3(0), dim_x=3, dim_y=2, cell_x=0.1, cell_y=0.1,
                       mass=0.1)
    b = record(bld.finalize(device=CPU), tmp_path / "b")
    (s,) = b.scene["surfaces"]
    assert s["name"] == "cloth_0" and s["kind"] == "cloth" and s["verts"] == 12 and s["faces"] == 12
    assert b.scene["points"] == []
    assert b.array("surf_cloth_0").shape == (3, 12, 3)
    assert b.array("surf_cloth_0")[2, 0, 0] == pytest.approx(0.2)
    with np.load(b.root / s["file"]) as z:
        assert z["face_tag"].sum() == 0 and z["twins"].shape == (0, 2)


def test_soft_grid_boundary_and_leftover_particles(tmp_path):
    bld = newton.ModelBuilder()
    bld.add_soft_grid(pos=wp.vec3(0, 0, 0), rot=IDENT, vel=wp.vec3(0), dim_x=2, dim_y=2, dim_z=2, cell_x=0.1,
                      cell_y=0.1, cell_z=0.1, density=100.0, k_mu=1e3, k_lambda=1e3, k_damp=0.0)
    for i in range(3):
        bld.add_particle((0, 0, 2 + i), (0, 0, 1.0), 1.0, radius=0.02)
    b = record(bld.finalize(device=CPU), tmp_path / "b")
    (s,) = b.scene["surfaces"]
    assert s["name"] == "soft_0" and s["kind"] == "soft"
    assert s["faces"] == 6 * 4 * 2 and s["verts"] == 26    # 2x2x2 cubes: 24 boundary quads = 48 tris, 27-1 verts
    (p,) = b.scene["points"]
    assert p["name"] == "particles_0" and p["count"] == 3 and p["radius"] == pytest.approx(0.02)
    assert b.array("pts_particles_0").shape == (3, 3, 3)
    assert b.array("vel_particles_0")[0, 0, 2] == pytest.approx(1.0)
    with np.load(b.root / s["file"]) as z:
        v = b.array("surf_soft_0")[0][z["faces"]]
        centre = b.array("surf_soft_0")[0].mean(axis=0)
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        assert ((n * (v[:, 0] - centre)).sum(axis=1) > 0).all()    # outward winding


def test_explicit_points_and_surface(tmp_path):
    bld = newton.ModelBuilder()
    for i in range(6):
        bld.add_particle((i * 0.1, 0, 1), (0, 0, 0), 1.0)
    rec = RenderRecorder(bld.finalize(device=CPU), tmp_path / "b", fps=10.0)
    rec.add_points("honey", [0, 1, 2], radius=0.01)
    rec.add_surface("sheet", np.array([[3, 4, 5]]))
    b = record(rec.model, tmp_path / "b", rec=rec)
    assert [p["name"] for p in b.scene["points"]] == ["honey"]
    assert b.scene["surfaces"][0]["name"] == "sheet"
    assert b.array("pts_honey").shape == (3, 3, 3)


def test_curve_from_particles(tmp_path):
    bld = newton.ModelBuilder()
    for i in range(5):
        bld.add_particle((i * 0.1, 0, 1), (0, 0, 0), 1.0, radius=0.004)
    rec = RenderRecorder(bld.finalize(device=CPU), tmp_path / "b", fps=10.0)
    rec.add_curve_from_particles("thread", np.arange(5))
    b = record(rec.model, tmp_path / "b", rec=rec)
    (c,) = b.scene["curves"]
    assert c["from"] == "particles" and c["count"] == 5 and c["radius"] == pytest.approx(0.004)
    assert b.scene["points"] == []


def chain_model(n=4, shuffle=False):
    bld = newton.ModelBuilder()
    order = list(range(n))
    if shuffle:
        order = order[::-1]
    for i in order:
        body = bld.add_body(xform=wp.transform((0, 0, 0.2 * i + 0.1), IDENT), label=f"harness/seg_{i}")
        bld.add_shape_capsule(body, radius=0.03, half_height=0.1, label=f"harness/cap_{i}")
    return bld.finalize(device=CPU)


@pytest.mark.parametrize("shuffle", [False, True])
def test_curve_from_bodies_is_ordered(tmp_path, shuffle):
    rec = RenderRecorder(chain_model(4, shuffle), tmp_path / "b", fps=10.0)
    rec.add_curve_from_bodies("harness", r"harness/seg_\d+")
    b = record(rec.model, tmp_path / "b", rec=rec)
    (c,) = b.scene["curves"]
    assert c["from"] == "bodies" and c["count"] == 6 and c["radius"] == pytest.approx(0.03)
    pts = b.array("curve_harness")
    assert pts.shape == (3, 6, 3)
    z0 = pts[0, :, 2]
    assert np.all(np.diff(z0) > 0) or np.all(np.diff(z0) < 0)       # monotone along the chain
    assert sorted(z0)[0] == pytest.approx(0.0, abs=1e-5) and sorted(z0)[-1] == pytest.approx(0.8, abs=1e-5)
    assert pts[2, 0, 2] - pts[0, 0, 2] == pytest.approx(0.2, abs=1e-5)
    assert all(s.get("curve") == "harness" for s in b.scene["shapes"])  # the tube replaces the capsules


def test_instancer(tmp_path):
    bld = newton.ModelBuilder()
    for i in range(5):
        body = bld.add_body(xform=wp.transform((i * 0.1, 0, 1), IDENT), label=f"rice_{i}")
        bld.add_shape_capsule(body, radius=0.002, half_height=0.004, label=f"rice_shape_{i}")
    other = bld.add_body(xform=wp.transform((1, 0, 1), IDENT), label="bowl")
    bld.add_shape_sphere(other, radius=0.1, label="bowl_shape")
    rec = RenderRecorder(bld.finalize(device=CPU), tmp_path / "b", fps=10.0)
    rec.add_instancer("rice", r"rice_\d+")
    b = record(rec.model, tmp_path / "b", rec=rec)
    sc = b.scene
    (inst,) = sc["instancers"]
    assert inst["count"] == 5 and inst["shape_type"] == "capsule" and inst["size"] == pytest.approx([0.002, 0.004])
    assert [s["label"] for s in sc["shapes"]] == ["bowl_shape"] and [e["label"] for e in sc["bodies"]] == ["bowl"]
    ip = b.array("inst_rice")
    assert ip.shape == (3, 5, 7) and ip[2, 3, 2] == pytest.approx(1.2) and ip[0, 3, 0] == pytest.approx(0.3)


def test_instancer_rejects_mixed_geometry(tmp_path):
    bld = newton.ModelBuilder()
    for i, r in enumerate((0.01, 0.02)):
        body = bld.add_body(label=f"g_{i}")
        bld.add_shape_sphere(body, radius=r)
    rec = RenderRecorder(bld.finalize(device=CPU), tmp_path / "b", fps=10.0)
    rec.add_instancer("g", r"g_\d")
    with pytest.raises(BundleError, match="different geometry"):
        rec.capture(rec.model.state(), 0.0)


def test_cut_twins_and_face_tags(tmp_path):
    from alloy.render.surfaces import twin_data
    src = np.array([2, 3, 4, 5, 6, 7], np.int32)
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    twins, tags = twin_data(np.array([[2, 5], [3, 4], [4, 7], [9, 10]]), src, faces)
    assert twins.tolist() == [[0, 3], [1, 2], [2, 5]]
    assert tags.tolist() == [1, 0]     # face 0 vertices 0,1,2 all twinned; face 1 has vertex 4 (id 6) untwinned


def test_twin_bonds_from_topology_tag_faces(tmp_path):
    from alloy.topology.bonds import add_spring_bond
    from alloy.topology.types import BondKind
    bld = newton.ModelBuilder()
    bld.add_cloth_grid(pos=wp.vec3(0, 0, 1), rot=IDENT, vel=wp.vec3(0), dim_x=2, dim_y=1, cell_x=0.1, cell_y=0.1,
                       mass=0.1)
    for a, c in ((0, 1), (3, 4)):
        add_spring_bond(bld, a, c, 1.0, kind=BondKind.CLOTH_EDGE)
    b = record(bld.finalize(device=CPU), tmp_path / "b", n=1)
    s = b.scene["surfaces"][0]
    with np.load(b.root / s["file"]) as z:
        assert z["twins"].tolist() == [[0, 1], [3, 4]]
        assert 0 < z["face_tag"].sum() < s["faces"]


def test_error_cases(tmp_path):
    model = rigid_world().finalize(device=CPU)
    st = model.state()
    rec = RenderRecorder(model, tmp_path / "b", fps=30.0)
    rec.capture(st, 0.0)
    with pytest.raises(BundleError, match="greater than previous"):
        rec.capture(st, 0.0)
    q = st.body_q.numpy().copy()
    q[1, 0] = np.nan
    st.body_q.assign(q)
    with pytest.raises(BundleError, match=r"frame 1: state\.body_q has NaN"):
        rec.capture(st, 0.1)
    with pytest.raises(BundleError, match="cannot declare"):
        rec.add_points("late", [0])
    rec.close()
    with pytest.raises(BundleError, match="after close"):
        rec.capture(st, 1.0)
    with pytest.raises(BundleError, match="twice"):
        rec.close()
    with pytest.raises(BundleError, match="no captured frames"):
        RenderRecorder(model, tmp_path / "c", fps=30.0).close()
    with pytest.raises(BundleError, match="fps"):
        RenderRecorder(model, tmp_path / "d", fps=0)
    with pytest.raises(BundleError, match="worlds must be"):
        RenderRecorder(model, tmp_path / "e", fps=1.0, worlds="some").capture(st, 0.0)
    rec2 = RenderRecorder(model, tmp_path / "f", fps=1.0)
    rec2.add_points("bad name!", [0])
    with pytest.raises(BundleError, match="must match"):
        rec2.capture(st, 0.0)
    assert json.loads((tmp_path / "b" / "scene.json").read_text())["frames"]["count"] == 1


def test_collision_only_objects_render(tmp_path):
    """A collision-only object (no visual geom on that MJCF body) is exported, flagged collision_fallback."""
    bld = newton.ModelBuilder()
    hidden = newton.ModelBuilder.ShapeConfig(is_visible=False)
    bld.add_shape_box(-1, hx=1.0, hy=1.0, hz=0.02, cfg=hidden, label="scene/worldbody/table/table_top")
    blk = bld.add_body(xform=wp.transform((0, 0, 0.1), IDENT), label="scene/worldbody/block")
    bld.add_shape_box(blk, hx=0.02, hy=0.02, hz=0.02, label="scene/worldbody/block/visual")
    bld.add_shape_box(blk, hx=0.02, hy=0.02, hz=0.02, cfg=hidden, label="scene/worldbody/block/collision")
    b = record(bld.finalize(device=CPU), tmp_path / "b")
    by_label = {s["label"]: s for s in b.scene["shapes"]}
    assert by_label["scene/worldbody/table/table_top"]["collision_fallback"] is True
    assert "scene/worldbody/block/visual" in by_label and "scene/worldbody/block/collision" not in by_label


def _two_piece_soft(bonded: bool):
    from alloy.topology.bonds import add_spring_bond, register_custom_attributes
    from alloy.topology.types import BondKind
    bld = newton.ModelBuilder()
    register_custom_attributes(bld)
    for x in (0.0, 0.1):
        bld.add_soft_grid(pos=wp.vec3(x, 0, 1), rot=IDENT, vel=wp.vec3(0), dim_x=1, dim_y=1, dim_z=1, cell_x=0.1,
                          cell_y=0.1, cell_z=0.1, density=100.0, k_mu=1e3, k_lambda=1e3, k_damp=0.0)
    if bonded:
        pos = np.array(bld.particle_q)
        left = [i for i in range(8) if abs(pos[i][0] - 0.1) < 1e-6]
        for a in left:
            b = next(j for j in range(8, 16) if np.allclose(pos[j], pos[a]))
            add_spring_bond(bld, a, b, 1.0, kind=BondKind.TET_CUT)
    return bld.finalize(device=CPU)


def test_pre_split_pieces_form_one_surface(tmp_path):
    """Pieces joined only by twin bonds (a pre-split loaf) export as one surface with cut-tagged inner faces."""
    b = record(_two_piece_soft(bonded=True), tmp_path / "one", n=1)
    (s,) = b.scene["surfaces"]
    with np.load(b.root / s["file"]) as z:
        assert len(z["twins"]) == 4 and z["face_tag"].sum() == 4     # the two coincident 2-triangle faces
    plain = record(_two_piece_soft(bonded=False), tmp_path / "two", n=1)
    assert len(plain.scene["surfaces"]) == 2


def _particle_model(n=20):
    bld = newton.ModelBuilder()
    for i in range(n):
        bld.add_particle(pos=wp.vec3(0.01 * i, 0, 1), vel=wp.vec3(0), mass=0.01, radius=0.002)
    return bld.finalize(device=CPU)


def test_leftover_points_off_and_skip(tmp_path):
    """SC5 C10: only the declared grain particles are exported and captured; the rest are dropped."""
    model = _particle_model()
    rec = RenderRecorder(model, tmp_path / "a", fps=10.0, leftover_points=False)
    rec.add_points("rice", np.arange(0, 20, 8))
    b = record(model, tmp_path / "a", rec=rec)
    assert [p["name"] for p in b.scene["points"]] == ["rice"] and b.array("pts_rice").shape[1] == 3
    assert rec._layout.cap_particles.tolist() == [0, 8, 16]          # nothing else is copied per frame
    rec2 = RenderRecorder(model, tmp_path / "b", fps=10.0)
    rec2.add_points("rice", [0, 8])
    rec2.skip_particles(np.arange(1, 8))
    b2 = record(model, tmp_path / "b", rec=rec2)
    leftover = [p for p in b2.scene["points"] if p["name"].startswith("particles_")]
    assert len(leftover) == 1 and leftover[0]["count"] == 20 - 2 - 7
