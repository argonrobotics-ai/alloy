# SPDX-License-Identifier: Apache-2.0
"""Deformables in the camera (CPU): an alloy.Sim's cloth is drawn from its state every render (per-world meshes
refitted in place), exactly as a scene built fresh at the same vertices; packs for Cycles carry the deformables."""
from __future__ import annotations

import json

import numpy as np
import pytest

import alloy
from alloy.camera import Camera
from alloy.camera.camera import camera_scene
from alloy.camera.deform import ICO_F, ICO_V, deform_objects, grain_mesh, has_deformables
from alloy.camera.deform_gpu import vertex_faces
from alloy.camera.model_pack import pack_from_model
from alloy.camera.scene import MeshGeom, SceneData
from alloy.camera.tracer.gpu_scene import GpuScene

LOOK_AT = ((0.6, -0.6, 0.6), (0.0, 0.0, 0.15))


@pytest.fixture(scope="module")
def cloth_sim():
    s = alloy.Scene(dt=0.01, worlds=2)
    s.add_ground()
    s.add_box("table", size=(0.4, 0.4, 0.2), pos=(0, 0, 0.1), static=True)
    s.add_cloth("towel", size=(0.3, 0.3), pos=(0, 0, 0.3), resolution=8, mass=0.05)
    return alloy.build(s, device="cpu")


def test_grain_mesh_is_an_icosahedron_of_each_radius():
    verts, tris = grain_mesh(np.array([[0, 0, 0], [1, 0, 0]], np.float32), np.array([0.1, 0.2], np.float32))
    assert verts.shape == (24, 3) and tris.shape == (40, 3) and tris.max() == 23
    assert np.allclose(np.linalg.norm(verts[12:] - [1, 0, 0], axis=1), 0.2, atol=1e-6)
    p = ICO_V[ICO_F]
    outward = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p.mean(1))
    assert (outward > 0).all()  # counter-clockwise seen from outside


def test_vertex_faces_lists_every_face_around_each_vertex_in_order():
    off, vf = vertex_faces(np.array([[0, 1, 2], [0, 2, 3]]), 5)
    assert off.tolist() == [0, 2, 3, 5, 6, 6]
    assert vf.tolist() == [0, 1, 0, 0, 1, 1]


def test_cloth_layout_follows_the_name_index(cloth_sim):
    lay = deform_objects(cloth_sim)
    (o,) = lay.objects
    r = cloth_sim.index.particles["towel"]
    assert has_deformables(cloth_sim) and lay.worlds == 2 and lay.stride == cloth_sim.index.stride["particle"]
    assert (o.name, o.kind, o.start, o.count) == ("towel", "surface", r.start, len(r))
    assert o.tris.shape == (2 * 8 * 8, 3) and o.tris.max() == len(r) - 1


def test_a_sim_without_deformables_draws_none():
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("cube", size=(0.05, 0.05, 0.05), pos=(0, 0, 0.1))
    sim = alloy.build(s, device="cpu")
    assert not has_deformables(sim) and deform_objects(sim) is None


def test_cloth_drawn_live_matches_a_fresh_scene_at_the_same_vertices(cloth_sim):
    cam = Camera("c", look_at=LOOK_AT, width=40, height=30, spp=2)
    before = cam.render(cloth_sim, seed=1).numpy().copy()
    cloth_sim.step(20)
    live = cam.render(cloth_sim, seed=1).numpy().copy()
    assert np.abs(live.astype(int) - before.astype(int)).mean() > 0.5  # the towel moved on screen
    sc = camera_scene(cloth_sim, device="cpu")
    ds, sd = sc.deform, sc.sd
    meshes = list(sd.meshes)
    for w in range(2):
        meshes[ds.first[0] + w] = MeshGeom(f"now{w}", cloth_sim.state(world=w)["towel.points"], ds.layout.objects[0].tris)
    ref = GpuScene(SceneData(meshes, sd.inst_mesh, sd.inst_group, sd.inst_body, sd.inst_local, sd.inst_color,
                             sd.num_worlds, sd.inst_name, sd.lighting), device="cpu")
    fresh = Camera("c", look_at=LOOK_AT, width=40, height=30, spp=2).render(ref, cloth_sim.engine.state, seed=1).numpy()
    assert np.array_equal(live, fresh)


def test_full_material_pack_draws_each_worlds_copy_in_its_world_only(cloth_sim):
    look = {"materials": {"cloth": {"preset": "plastic_matte", "hard": True}}, "assign": [{"match": "towel", "material": "cloth"}]}
    pk, frame, (body, local, present) = pack_from_model(cloth_sim.engine.model, look, deform=deform_objects(cloth_sim))
    names = [m.meta["name"] for m in frame.meshes]
    assert names[-2:] == ["towel@0", "towel@1"] and frame.meshes[-1].meta["dynamic"]
    assert present[-2:].tolist() == [[True, False], [False, True]] and present[:-2].all()
    assert (body[-2:] == -1).all()
    assert "cloth~soft" in pk.static["materials"]  # a deformable never gets the hard-surface Bevel
    bevel = [n for n in pk.static["materials"]["cloth~soft"]["tree"]["nodes"] if n["type"] == "ShaderNodeBevel"]
    assert not bevel


def test_write_sim_pack_reads_back_with_the_towel_in_world_space(cloth_sim, tmp_path):
    from alloy.camera.pack_write import write_sim_pack
    from alloy.camera.pbr.pack_io import load_pack
    look = {"lights": [{"type": "area", "eye": [0.5, -0.5, 1.0], "target": [0, 0, 0.2], "watts": 50.0}]}
    root = write_sim_pack(cloth_sim, tmp_path / "pack", LOOK_AT, look=look)
    pk = load_pack(root)
    fr = pk.frame(0)
    towel = next(m for m in fr.meshes if m.meta["name"] == "towel@0")
    assert np.allclose(towel.v, cloth_sim.state(world=0)["towel.points"], atol=1e-6)
    static = json.loads((root / "pack.json").read_text())
    assert static["world"]["tree"]["nodes"][1]["image"] == "world_sky" and (root / "textures/world_sky.hdr").exists()
    assert all("props" in m and "cycles" in m for m in static["materials"].values())
    assert fr.lights and all(not lt["visibility"]["visible_camera"] for lt in fr.lights)


@pytest.mark.gpu
def test_optix_refits_the_cloth_gas_in_place():
    from alloy.camera.optix import available
    ok, why = available()
    if not ok:
        pytest.skip(f"OptiX backend not available: {why}")
    s = alloy.Scene(dt=0.01, worlds=2)
    s.add_ground()
    s.add_box("table", size=(0.4, 0.4, 0.2), pos=(0, 0, 0.1), static=True)
    s.add_cloth("towel", size=(0.3, 0.3), pos=(0, 0, 0.3), resolution=8, mass=0.05)
    sim = alloy.build(s, device="cuda:0")
    cams = {b: Camera("c", look_at=LOOK_AT, width=40, height=30, spp=4, backend=b) for b in ("warp", "optix")}
    for cam in cams.values():
        cam.render(sim, seed=1)  # GAS built at the rest pose
    sim.step(20)
    warp_img, optix_img = (cams[b].render(sim, seed=1).numpy().astype(int) for b in ("warp", "optix"))
    assert np.abs(warp_img - optix_img).mean() < 0.5  # same refitted towel on RT cores


def test_a_topology_change_rebuilds_the_scene(cloth_sim):
    cam = Camera("c", look_at=LOOK_AT, width=16, height=12, spp=1)
    cam.render(cloth_sim)
    sc = camera_scene(cloth_sim, device="cpu")
    sc.deform.topology = (-1, -1, -1)  # as if a cutting plug-in had replaced the triangles
    cam.render(cloth_sim)
    fresh = camera_scene(cloth_sim, device="cpu")
    assert fresh is not sc and not fresh.deform.stale()
