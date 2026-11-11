# SPDX-License-Identifier: Apache-2.0
"""CPU checks: the synthetic bundle follows ARB v1, and the Blender side's pure-numpy helpers behave."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.render import synth, synth_models

BLENDER_DIR = Path(__file__).resolve().parents[2] / "alloy" / "render" / "blender"
sys.path.insert(0, str(BLENDER_DIR))

import io_bundle  # noqa: E402
import look as blook  # noqa: E402
import motion  # noqa: E402
import presets  # noqa: E402
import primitives  # noqa: E402
import skinning  # noqa: E402
import weld  # noqa: E402


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("synth") / "bundle"
    synth.write_synth(root, frames=24)
    return root


def test_scene_json_fields(bundle_dir: Path) -> None:
    scene = json.loads((bundle_dir / "scene.json").read_text())
    assert scene["arb"] == 1
    for key in ("meta", "frames", "worlds", "world_offsets", "bodies", "shapes", "meshes", "surfaces", "points",
                "curves", "instancers"):
        assert key in scene, key
    assert scene["frames"] == {"count": 24, "fps": synth.FPS}
    types = {s["type"] for s in scene["shapes"]}
    assert {"plane", "box", "sphere", "mesh", "capsule"} <= types
    for s in scene["shapes"]:
        assert len(s["xform"]) == 7 and len(s["color"]) == 3
        assert s["body"] == -1 or 0 <= s["body"] < len(scene["bodies"])


def test_frame_arrays(bundle_dir: Path) -> None:
    fr = bundle_dir / "frames"
    t = np.load(fr / "t.npy")
    assert t.dtype == np.float64 and t.shape == (24,)
    bq = np.load(fr / "body_q.npy")
    assert bq.dtype == np.float32 and bq.shape == (24, 11 + synth.STRIP_N, 7)
    assert np.allclose(np.linalg.norm(bq[..., 3:], axis=-1), 1.0, atol=1e-5)
    for name in ("honey", "rice_mpm"):
        p, v = np.load(fr / f"pts_{name}.npy"), np.load(fr / f"vel_{name}.npy")
        assert p.shape == v.shape and p.dtype == np.float32 and p.shape[2] == 3
    assert np.load(fr / "inst_rice.npy").shape == (24, 2000, 7)
    assert np.load(fr / "curve_cable.npy").shape == (24, 24, 3)


def test_surfaces_twins_and_cut_tags(bundle_dir: Path) -> None:
    for name, expect_cut in (("towel", False), ("loaf", True)):
        z = np.load(bundle_dir / "surfaces" / f"{name}.npz")
        pos = np.load(bundle_dir / "frames" / f"surf_{name}.npy")
        faces, twins, tags = z["faces"], z["twins"], z["face_tag"]
        assert faces.dtype == np.int32 and faces.max() < pos.shape[1] and len(z["src"]) == pos.shape[1]
        gap = np.linalg.norm(pos[0, twins[:, 0]] - pos[0, twins[:, 1]], axis=1)
        assert gap.max() < 2e-5, "twins start welded (intact seam)"
        twinned = np.zeros(pos.shape[1], bool)
        twinned[twins.ravel()] = True
        assert np.array_equal(tags == 1, twinned[faces].all(axis=1)), "tag 1 iff every vertex is a twin"
        assert bool((tags == 1).any()) == expect_cut


def test_replicas_follow_offsets(bundle_dir: Path) -> None:
    scene = json.loads((bundle_dir / "scene.json").read_text())
    bq = np.load(bundle_dir / "frames" / "body_q.npy")
    by_id = {s["id"]: s for s in scene["shapes"]}
    reps = [s for s in scene["shapes"] if s["replica_of"] is not None]
    assert len(reps) == 8
    for s in reps:
        hero = by_id[s["replica_of"]]
        assert hero["world"] == 0 and hero["type"] == s["type"]
        off = np.asarray(scene["world_offsets"][s["world"]])
        if s["body"] >= 0:
            assert np.allclose(bq[:, s["body"], :3], bq[:, hero["body"], :3] + off, atol=1e-6)


def test_renderer_reader_accepts_synth(bundle_dir: Path) -> None:
    b = io_bundle.load_bundle(bundle_dir)
    assert b.count == 24 and b.body_q.shape[1] == 11 + synth.STRIP_N
    assert np.allclose(b.offset(1), [synth.GRID_M, 0, 0])
    assert abs(b.time_at(2.5) - 2.5 / synth.FPS) < 1e-12


def test_reader_rejects_bad_bundle(tmp_path: Path) -> None:
    root = tmp_path / "b"
    synth.write_synth(root, frames=4)
    scene = json.loads((root / "scene.json").read_text())
    scene["shapes"][2]["type"] = "torus"
    (root / "scene.json").write_text(json.dumps(scene))
    with pytest.raises(io_bundle.BundleError, match="torus"):
        io_bundle.load_bundle(root)


def test_look_parses_and_frames() -> None:
    lk = blook.parse_look(synth.make_look())
    assert lk.assigned(["nut", "nut_body"]) == "nut"
    assert lk.assigned(["", "box_fast_body"]) == "red"
    spec = lk.material_spec("red")
    assert spec["base_color"] == [0.7, 0.06, 0.04]
    assert spec["roughness"] == presets.PRESETS["plastic_gloss"]["roughness"]
    assert blook.frame_list({"start": 0, "end": -1, "step": 4}, 10) == [0, 4, 8]
    assert blook.frame_list({"step": 1}, 10, (3, 6)) == [3, 4, 5]
    bad = synth.make_look()
    bad["materials"]["x"] = {"preset": "honey", "colour": [1, 0, 0]}
    with pytest.raises(blook.LookError, match="unknown keys"):
        blook.parse_look(bad)


def test_presets_complete() -> None:
    required = ("plastic_matte plastic_gloss nylon_natural abs_black rubber_black steel_brushed steel_polished "
                "zinc_plated brass aluminum_bare aluminum_anodized_black copper glass_clear water honey tea "
                "ceramic_white wood_oak marble_white linen cotton_terry cardboard_kraft bread_crust bread_crumb dough "
                "rice_grain paper pvc_cable_black pvc_cable_grey robot_painted_white robot_painted_dark pcb_green")
    assert set(required.split()) <= set(presets.PRESETS)
    allowed = set(presets.DEFAULTS)
    for name, p in presets.PRESETS.items():
        assert set(p) <= allowed, name


def test_motion_math() -> None:
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = synth.quat_axis_angle([0, 0, 1], np.array(np.pi / 2))
    mid = motion.slerp(q0, q1, 0.5)
    assert np.allclose(mid, synth.quat_axis_angle([0, 0, 1], np.array(np.pi / 4)), atol=1e-9)
    keys_t, keys_v = np.array([0.0, 1.0, 3.0]), np.array([[0.0, 0, 0], [1.0, 2, 0], [2.0, 0, 1]])
    for tk, vk in zip(keys_t, keys_v):
        assert np.allclose(motion.catmull_rom(keys_t, keys_v, tk), vk)
    assert np.allclose(motion.catmull_rom(keys_t[:1], keys_v[:1], 5.0), keys_v[0])
    m = motion.look_at([1.0, 0, 0], [0.0, 0, 0])
    assert np.allclose(-m[:3, 2], [-1, 0, 0]) and np.allclose(m[:3, 1], [0, 0, 1])
    warm, white = motion.kelvin_to_rgb(3000), motion.kelvin_to_rgb(6500)
    assert warm[0] > warm[2] * 2 and white.min() > 0.85
    color, dens = motion.absorption_density([0.9, 0.5, 0.06], 0.01)
    trans = np.exp(-dens * (1 - color) * 0.01)
    assert np.allclose(trans, motion.srgb_to_linear([0.9, 0.5, 0.06]), rtol=1e-3, atol=1e-4)


def _signed_volume(v: np.ndarray, corners: np.ndarray, sizes: np.ndarray) -> float:
    vol, start = 0.0, 0
    for n in sizes:
        poly = v[corners[start:start + n]].astype(np.float64)
        for k in range(1, n - 1):
            vol += np.dot(poly[0], np.cross(poly[k], poly[k + 1])) / 6.0
        start += n
    return vol


@pytest.mark.parametrize("typ,size,volume", [
    ("box", [0.1, 0.2, 0.3], 8 * 0.1 * 0.2 * 0.3), ("sphere", [0.1], 4 / 3 * np.pi * 1e-3),
    ("capsule", [0.1, 0.2], np.pi * 0.01 * 0.4 + 4 / 3 * np.pi * 1e-3), ("cylinder", [0.1, 0.2], np.pi * 0.01 * 0.4),
    ("cone", [0.1, 0.2], np.pi * 0.01 * 0.4 / 3), ("ellipsoid", [0.1, 0.2, 0.3], 4 / 3 * np.pi * 0.006),
])
def test_primitives_closed_and_outward(typ: str, size: list, volume: float) -> None:
    v, c, s = primitives.shape_mesh(typ, size)
    assert _signed_volume(v, c, s) == pytest.approx(volume, rel=0.02)


def test_grain_and_weld() -> None:
    v, c, s = primitives.grain_mesh("rice_grain")
    assert np.ptp(v[:, 0]) == pytest.approx(0.0064, rel=0.05) and _signed_volume(v, c, s) > 0
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float32)
    wv, wf = primitives.weld_exact(verts, np.array([[0, 1, 2], [3, 5, 4]]))
    assert len(wv) == 4 and wf[0, 1] == wf[1, 0]


def _hinge_thickness(bundle_dir: Path, blend, frame: int) -> float:
    """Shell thickness at the 90 degree hinge after skinning a two-layer copy of the strip."""
    b = io_bundle.load_bundle(bundle_dir)
    verts, _, _ = skinning.read_obj(bundle_dir / "extra" / "strip_rest.obj")
    labels, idx, w = skinning.read_weights(bundle_dir / "extra" / "strip_weights.json", len(verts))
    ids = skinning.body_ids(b, labels, "skins.strip")
    half = np.array([0.0, 0.0, synth.STRIP_T / 2])
    v0 = np.concatenate([verts + half, verts - half]).astype(np.float64)
    t, q = skinning.relative_poses(b, ids, 0, float(frame))
    out = blend(v0, np.concatenate([idx, idx]), np.concatenate([w, w]), t, q)
    hinge = np.isclose(verts[:, 0] - synth.STRIP_BASE[0], 5 * synth.STRIP_L, atol=1e-6) & np.isclose(
        verts[:, 1], synth.STRIP_BASE[1], atol=1e-6)
    k = int(np.flatnonzero(hinge)[0])
    return float(np.linalg.norm(out[k] - out[k + len(verts)]))


def test_skin_rest_frame_is_identity(bundle_dir: Path) -> None:
    b = io_bundle.load_bundle(bundle_dir)
    verts, corners, sizes = skinning.read_obj(bundle_dir / "extra" / "strip_rest.obj")
    labels, idx, w = skinning.read_weights(bundle_dir / "extra" / "strip_weights.json", len(verts))
    assert len(sizes) == 800 and np.allclose(w.sum(axis=1), 1.0)
    t, q = skinning.relative_poses(b, skinning.body_ids(b, labels, "x"), 0, 0.0)
    out = skinning.blend_dual_quat(verts.astype(np.float64), idx, w, t, q)
    assert np.abs(out - verts).max() < 1e-6
    with pytest.raises(blook.LookError, match="strip_99"):
        skinning.body_ids(b, ["strip_99"], "skins.strip")


def test_dual_quat_keeps_hinge_thickness(bundle_dir: Path) -> None:
    dq = _hinge_thickness(bundle_dir, skinning.blend_dual_quat, 20)
    lbs = _hinge_thickness(bundle_dir, skinning.blend_linear, 20)
    assert dq == pytest.approx(synth.STRIP_T, rel=0.02)
    assert lbs < 0.8 * synth.STRIP_T, "linear blending collapses a 90 degree hinge (why dual quaternions are default)"


def test_weld_hysteresis(bundle_dir: Path) -> None:
    """Seam 0|1 never opens; seam 1|2 opens then touches again but stays open; seam 2|3 opens and stays open."""
    pos = np.load(bundle_dir / "frames" / "surf_loaf.npy", mmap_mode="r")
    twins = np.load(bundle_dir / "surfaces" / "loaf.npz")["twins"]
    broken = weld.first_break(pos, twins)
    rest = np.asarray(pos[0])
    seam = np.round((rest[twins[:, 0], 0] - rest[:, 0].min()) / 0.0175).astype(int)  # 1, 2, 3 = cut plane
    assert np.isinf(broken[seam == 1]).all()
    assert np.isfinite(broken[seam == 2]).all() and np.isfinite(broken[seam == 3]).all()
    last = pos.shape[0] - 1
    end = np.asarray(pos[last])
    gap_12 = np.linalg.norm(end[twins[seam == 2, 0]] - end[twins[seam == 2, 1]], axis=1)
    assert gap_12.max() < weld.WELD_DISTANCE_M, "seam 1|2 has closed again by the last frame"
    flags = weld.sealed_flags(end, twins, broken, float(last))
    assert flags[twins[seam == 1].ravel()].all() and not flags[twins[seam == 2].ravel()].any()
    assert weld.sealed_flags(rest, twins, broken, 0.0)[twins.ravel()].all()


def test_white_balance_helpers() -> None:
    flat = np.tile([0.5, 0.4, 0.3, 1.0], 16 * 32).astype(np.float32)
    assert np.allclose(motion.equirect_mean(flat, 16, 32, stride=1), [0.5, 0.4, 0.3])
    near = motion.light_irradiance({"type": "area", "eye": [0, 0, 1], "watts": 10, "kelvin": 6500}, np.zeros(3))
    far = motion.light_irradiance({"type": "area", "eye": [0, 0, 2], "watts": 10, "kelvin": 6500}, np.zeros(3))
    assert np.allclose(near, 4 * far)
    lamp = {"type": "area", "eye": [0, 0, 1], "watts": 10, "kelvin": 3200}
    warm = motion.scene_illuminant([lamp], None, 0, [0, 0, 0])
    assert warm[0] > 1.5 * warm[2]


def test_look_props_wb_and_texture_override() -> None:
    lk = blook.parse_look(synth.make_look())
    assert lk.render["white_balance"] == "auto" and len(lk.raw["props"]) == 3
    bad = synth.make_look()
    bad["props"] = [{"model": "http://example.com/x.glb"}]
    with pytest.raises(blook.LookError, match="props\\[0\\].model"):
        blook.parse_look(bad)
    bad = synth.make_look()
    bad["render"]["white_balance"] = "daylight"
    with pytest.raises(blook.LookError, match="white_balance"):
        blook.parse_look(bad)
    raw = synth.make_look()
    raw["render"]["white_balance"] = {"kelvin": 5200, "tint": 10}
    raw["materials"]["wall"] = {"preset": "plaster_paint", "texture": "polyhaven:painted_plaster_wall",
                                "roughness": 0.7}
    spec = blook.parse_look(raw).material_spec("wall")
    assert spec["texture"] == "polyhaven:painted_plaster_wall" and spec["roughness"] == 0.7
    assert spec["micro_bump_m"] == presets.PRESETS["plaster_paint"]["micro_bump_m"]
    assert "roughness" in spec["_overrides"]
    assert presets.PRESETS["stainless_steel_brushed"] == presets.PRESETS["steel_brushed"]
    assert {"wood_walnut_oiled", "wood_plywood", "plaster_paint", "linen"} <= set(presets.PRESETS)


def test_rj45_models(bundle_dir: Path) -> None:
    glb = (bundle_dir / "extra" / "rj45_plug.glb").read_bytes()
    magic, version, length = np.frombuffer(glb[:12], np.uint32)
    assert magic == 0x46546C67 and version == 2 and length == len(glb)
    js_len = int(np.frombuffer(glb[12:16], np.uint32)[0])
    doc = json.loads(glb[20:20 + js_len])
    assert [m["name"] for m in doc["materials"]] == ["pc_clear", "gold_blade", "wire"]
    assert len(doc["meshes"][0]["primitives"]) == 3
    obj = skinning.ObjMesh(bundle_dir / "extra" / "switch_port.obj")
    assert obj.groups == ["housing", "shield", "led"] and obj.corner_uv is not None
    assert len(obj.face_group) == len(obj.sizes) and set(obj.face_group.tolist()) == {0, 1, 2}
    assert (bundle_dir / "extra" / "jacket_print.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_rj45_look_and_time_mapping(bundle_dir: Path) -> None:
    lk = blook.parse_look(synth.make_look())
    assert lk.raw["skins"]["plug"]["local"] and lk.raw["render"]["glare"]["type"] == "fog_glow"
    b = io_bundle.load_bundle(bundle_dir)
    assert b.frame_at_time(0.1) == pytest.approx(0.1 * synth.FPS)
    assert b.frame_at_time(b.time_at(7.25)) == pytest.approx(7.25)
    bad = synth.make_look()
    bad["skins"]["port"]["local"] = True
    del bad["skins"]["port"]["body"]
    with pytest.raises(blook.LookError, match="local needs 'body'"):
        blook.parse_look(bad)
    s = motion.rest_arc_length(np.load(bundle_dir / "frames" / "curve_rj45_cable.npy")[0])
    assert s[0] == 0.0 and np.all(np.diff(s) > 0)


def test_glass_of_water_bundle(tmp_path: Path) -> None:
    synth_models.write_glass(tmp_path / "glass", frames=8)
    b = io_bundle.load_bundle(tmp_path / "glass")
    m = b.mesh("glass")
    sizes = np.full(len(m["faces"]), 3)
    ro, ri, h, base = synth_models.GLASS_RO, synth_models.GLASS_RI, synth_models.GLASS_H, synth_models.GLASS_BASE
    shell = np.pi * (ro ** 2 * h - ri ** 2 * (h - base))
    assert _signed_volume(m["vertices"], m["faces"].ravel(), sizes) == pytest.approx(shell, rel=0.03)
    pts, vel = b.array("pts", "water"), b.array("vel", "water")
    assert pts.shape[1] > 20000 and vel.shape == pts.shape
    local = pts - b.body_q[:, :, :3]
    r = np.hypot(local[..., 0], local[..., 1])
    gap = synth_models.GLASS_GAP
    assert r.max() <= ri - gap + 1e-6 and local[..., 2].min() >= base + gap - 1e-6
    assert b.body_q[-1, 0, 0] > 0.005 and vel[-1, :, 0].min() > 0.05


def test_liquid_contact_look_keys() -> None:
    on, off = synth_models.glass_look(contact=True), synth_models.glass_look(contact=False)
    assert blook.parse_look(on).section("liquids")["water"]["contact"] == [{"match": "^glass$"}]
    assert "contact" not in blook.parse_look(off).section("liquids")["water"]
    for rule, msg in (({"match": "("}, "contact\\[0\\].match"), ({"match": "x", "distance_m": 0}, "distance_m"),
                      ({"match": "x", "overlap_m": -1e-4}, "overlap_m"), ("glass", "expected")):
        bad = synth_models.glass_look()
        bad["liquids"]["water"]["contact"] = [rule]
        with pytest.raises(blook.LookError, match=msg):
            blook.parse_look(bad)
    bad = synth_models.glass_look()
    bad["liquids"]["water"]["contact"] = {"match": "glass"}
    with pytest.raises(blook.LookError, match="expected a list"):
        blook.parse_look(bad)
