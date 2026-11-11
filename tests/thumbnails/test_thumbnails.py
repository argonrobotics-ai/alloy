# SPDX-License-Identifier: Apache-2.0
"""tools/thumbnails on CPU: library layouts, framing, MJCF and GLB loaders, resumable batches (fake renderer)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "thumbnails"
sys.path.insert(0, str(TOOL))

from batch import read_state, run_jobs  # noqa: E402
from framing import frame, ground_plane  # noqa: E402
from libs import library_jobs  # noqa: E402
from visuals import Part, bounds, load_visual  # noqa: E402

MJCF = """<mujoco><asset>
  <texture name="t" type="2d" builtin="checker" width="8" height="8" rgb1="1 0 0" rgb2="0 0 1"/>
  <material name="m" texture="t" shininess="0.3" reflectance="0.1"/>
  <mesh name="q" vertex="0 0 0  1 0 0  1 1 0  0 1 0  0.5 0.5 0.5" face="0 1 4  1 2 4  2 3 4  3 0 4  0 2 1  0 3 2"
        texcoord="0 0  1 0  1 1  0 1  0.5 0.5"/></asset>
  <worldbody><body pos="0 0 0.2"><freejoint/>
    <geom type="mesh" mesh="q" material="m" contype="0" conaffinity="0"/>
    <geom type="box" size="0.5 0.5 0.25" pos="0.5 0.5 0.25" rgba="0 1 0 0"/></body></worldbody></mujoco>"""


class FakeRenderer:
    def __init__(self) -> None:
        self.calls = 0

    def render(self, parts, view):
        self.calls += 1
        return np.full((8, 8, 3), 128, np.uint8)


def _shared_lib(root: Path, names: list[str]) -> Path:
    objs = {}
    for n in names:
        d = root / "meshes" / n
        d.mkdir(parents=True)
        (d / "physics.obj").write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n")
        objs[n] = {"files": {"physics.obj": {}}, "kind": "rigid"}
    (root / "manifest.json").write_text(json.dumps({"lib": "toy", "objects": objs}))
    return root


def test_shared_layout_jobs_point_at_the_mesh_and_thumb(tmp_path):
    jobs = library_jobs(_shared_lib(tmp_path / "toy", ["b", "a"]))
    assert [j.name for j in jobs] == ["a", "b"]
    assert jobs[0].source.name == "physics.obj" and jobs[0].thumb == tmp_path / "toy/meshes/a/thumb.png"


def test_robocasa_layout_uses_the_model_and_listed_thumb(tmp_path):
    files = {"mjcf/x/model.xml": {}, "thumbs/cat/x.png": {}}
    (tmp_path / "manifest.json").write_text(json.dumps({"assets": {"x": {"model": "mjcf/x/model.xml", "files": files}}}))
    (j,) = library_jobs(tmp_path, "robocasa")
    assert j.source == tmp_path / "mjcf/x/model.xml" and j.thumb == tmp_path / "thumbs/cat/x.png"


def _in_view(view, pts):
    eye, c = np.array(view.eye), np.array(view.target)
    f = (c - eye) / np.linalg.norm(c - eye)
    right = np.cross(f, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    rel = pts - eye
    t = np.tan(np.radians(view.fov_deg / 2))
    z = rel @ f
    return np.maximum(np.abs(rel @ right), np.abs(rel @ up)) / (z * t)


def test_frame_fits_the_box_from_front_right_above():
    lo, hi = np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.2])
    v = frame(lo, hi)
    eye, c = np.array(v.eye), np.array(v.target)
    assert c == pytest.approx([0, 0, 0.1]) and eye[0] > 0 > eye[1] and eye[2] > c[2]
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    span = _in_view(v, corners)
    assert span.max() <= 1 / 1.15 + 1e-6 and span.max() > 0.8          # tight: the box nearly fills the frame
    gp, gt = ground_plane(lo, hi)
    assert gp[:, 2].max() <= lo[2] + 1e-9 and gt.shape[1] == 3


def test_flat_objects_are_seen_from_higher_up():
    low = frame(np.array([-0.3, -0.2, 0.0]), np.array([0.3, 0.2, 0.01]))
    tall = frame(np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.2]))
    elev = [np.degrees(np.arcsin((v.eye[2] - v.target[2]) / np.linalg.norm(np.subtract(v.eye, v.target))))
            for v in (low, tall)]
    assert elev[0] == pytest.approx(55.0) and elev[1] == pytest.approx(25.0)


def test_mjcf_loader_keeps_visual_geoms_texture_and_uv(tmp_path):
    pytest.importorskip("mujoco")
    (tmp_path / "model.xml").write_text(MJCF)
    (part,) = load_visual(tmp_path / "model.xml")         # the invisible collision box is not a visual geom
    assert part.points.shape == (18, 3) and part.tris.shape == (6, 3)
    assert part.texture is not None and part.texture.shape == (8, 8, 3)
    assert part.uv.shape == (18, 2) and part.uv.min() >= 0 and part.uv.max() <= 1
    assert part.roughness == pytest.approx(0.7) and part.metallic == pytest.approx(0.1)
    lo, hi = bounds([part])
    assert lo == pytest.approx([0, 0, 0.2], abs=1e-6) and hi == pytest.approx([1, 1, 0.7], abs=1e-6)


def test_glb_loader_turns_gltf_y_up_into_z_up(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box(extents=(0.2, 0.4, 0.1))      # glTF frame: y is up
    box.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
        baseColorFactor=[1.0, 0.0, 0.0, 1.0], roughnessFactor=0.3))
    box.export(tmp_path / "visual.glb")
    (part,) = load_visual(tmp_path / "visual.glb")
    lo, hi = bounds([part])
    assert hi - lo == pytest.approx([0.2, 0.1, 0.4], abs=1e-6)
    assert part.base_color == pytest.approx((1.0, 0.0, 0.0), abs=0.01) and part.roughness == pytest.approx(0.3)


def test_batch_resumes_and_records_failures(tmp_path):
    pytest.importorskip("PIL")
    root = _shared_lib(tmp_path / "toy", ["a", "b", "c"])
    (root / "meshes/c/physics.obj").unlink()
    state, r = tmp_path / "state" / "toy.jsonl", FakeRenderer()
    s1 = run_jobs(library_jobs(root), r, state)
    assert s1["ok"] == 2 and list(s1["failed"]) == ["c"] and "fetch" in s1["failed"]["c"]
    assert (root / "meshes/a/thumb.png").is_file() and r.calls == 2
    s2 = run_jobs(library_jobs(root), r, state)
    assert s2["rendered_now"] == 1 and r.calls == 2                     # only the failed one is retried
    assert read_state(state)["a"]["sha256"] and run_jobs(library_jobs(root), r, state, redo=True)["rendered_now"] == 3



def test_batch_survives_a_crash_truncated_state(tmp_path):
    """A preempted VM left a NUL-filled tail on the state file and can leave a thumbnail half-written."""
    pytest.importorskip("PIL")
    root = _shared_lib(tmp_path / "toy", ["a", "b", "c"])
    state, r = tmp_path / "state" / "toy.jsonl", FakeRenderer()
    run_jobs(library_jobs(root), r, state)
    with open(state, "ab") as fh:
        fh.write(b"\x00" * 300)
    (root / "meshes/b/thumb.png").write_bytes(b"\x00" * 10)
    s = run_jobs(library_jobs(root), r, state)
    assert s["rendered_now"] == 1 and s["ok"] == 3 and r.calls == 4     # only the damaged thumbnail is rendered again
    assert set(read_state(state)) == {"a", "b", "c"}

def test_part_dataclass_defaults_are_opaque_grey():
    p = Part("x", np.zeros((3, 3), np.float32), np.arange(3).reshape(1, 3))
    assert not p.glass and p.texture is None and p.base_color == pytest.approx((0.62, 0.62, 0.62))


def test_textured_mesh_keeps_uv_only_with_a_texture():
    meshes = pytest.importorskip("alloy.camera.meshes")
    from backend import textured_mesh

    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    tris = np.arange(3, dtype=np.int32).reshape(1, 3)
    uv = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    tex = textured_mesh(meshes.TexturedMesh, Part("a", pts, tris, uv, np.zeros((2, 2, 3), np.uint8), (0.2, 0.4, 0.6)))
    flat = textured_mesh(meshes.TexturedMesh, Part("b", pts, tris, uv, None, (0.2, 0.4, 0.6), roughness=0.9))
    assert meshes._checked_arrays(tex)[2].shape == (1, 3, 2) and tex.color == (0.2, 0.4, 0.6)
    assert flat.uv is None and flat.texture is None and flat.roughness == 0.9


def test_manifest_lists_hashes_and_showcase_flags(tmp_path):
    pytest.importorskip("PIL")
    from manifest import build, showcase

    rules = {"names": {"b": "brand on the box"}, "prefixes": {"c": "packaging category"}}
    assert showcase("a", rules) == (True, "") and showcase("b", rules)[0] is False and showcase("cx", rules)[0] is False
    root = _shared_lib(tmp_path / "toy", ["a", "b", "cx"])
    state = tmp_path / "state" / "toy.jsonl"
    run_jobs(library_jobs(root), FakeRenderer(), state)
    m = build(root, "toy", read_state(state), rules)
    assert m["count"] == 3 and m["showcase_ok"] == 1 and m["objects"]["a"]["thumb"] == "meshes/a/thumb.png"
    assert m["objects"]["b"]["showcase_note"] == "brand on the box" and m["objects"]["a"]["bytes"] > 0


def test_library_manifest_gets_the_new_thumbnail_hashes(tmp_path):
    pytest.importorskip("PIL")
    from manifest import build, update_library_manifest

    root = _shared_lib(tmp_path / "toy", ["a", "b"])
    state = tmp_path / "state" / "toy.jsonl"
    run_jobs(library_jobs(root), FakeRenderer(), state)
    m = build(root, "toy", read_state(state), {"names": {}, "prefixes": {}})
    man = json.loads((root / "manifest.json").read_text())
    man["objects"]["a"]["files"]["thumb.png"] = {"sha256": "old", "bytes": 1}       # an importer's earlier entry
    (root / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    assert update_library_manifest(root, m) == 2 and update_library_manifest(root, m) == 0
    assert list(json.loads((root / "manifest.json").read_text())["objects"]["a"]["files"]["thumb.png"]) == [
        "sha256", "bytes"]                                                          # the entry's key order is kept
    man = json.loads((root / "manifest.json").read_text())
    assert man["objects"]["a"]["files"]["thumb.png"]["sha256"] == m["objects"]["a"]["sha256"]
    assert man["objects"]["a"]["files"]["physics.obj"] == {} and man["lib"] == "toy"
