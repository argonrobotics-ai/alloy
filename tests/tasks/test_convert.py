# SPDX-License-Identifier: Apache-2.0
"""mjb -> MJCF conversion: the reloaded MJCF equals the compiled model and replays the recording (CPU MuJoCo)."""

from __future__ import annotations

import copy
from pathlib import Path

import mujoco
import numpy as np
import pytest

from alloy.core.errors import TraceError
from alloy.tasks.mjb_convert import load_recorded, write_scene
from alloy.tasks.mjb_verify import compare_models, replay_cpu
from alloy.tasks.mjcf_tree import fmt
from alloy.tasks.msh import MANIFEST_NAME, MeshData, check_manifest, read_msh, write_msh


def test_fmt_round_trips_float64() -> None:
    vals = np.array([0.0008333333333, 1e-10, -0.7777, 1 / 3, 9.81])
    assert np.array_equal(np.array(fmt(vals).split(), dtype=np.float64), vals)


def test_floor_friction_raises_only_the_terms_in_use_under_mjminmu_in_float32() -> None:
    """MuJoCo Warp warns about a term its condim uses whose float32 is under mjMINMU (1e-5 is: 9.99999975e-6); the
    floor is the next float32 up, so MuJoCo's own clamp to 1e-5 moves by 6.6e-8 of itself."""
    import xml.etree.ElementTree as ET

    from alloy.tasks.mjb_convert import MINMU_F32, floor_friction

    root = ET.fromstring(
        '<mujoco><worldbody>'
        '<geom name="a" condim="6" friction="0.5 0.005 1e-05"/>'              # rolling in use: floored
        '<geom name="b" condim="3" friction="0.5 1e-06 1e-06"/>'              # torsional, rolling unused: kept
        '<geom name="c" condim="4" friction="0.5 1.5e-05 0.0"/>'              # clear of it; rolling unused: kept
        '<geom name="d" condim="1" friction="0.0 0.0 0.0"/>'                  # frictionless: kept
        '</worldbody><contact>'
        '<pair geom1="a" geom2="b" condim="4" friction="1.0 1.0 4e-06 9e-07 9e-07"/>'   # [1, 2] in use
        '</contact></mujoco>')
    assert floor_friction(root) == 2
    f = {el.get("name", "pair"): el.get("friction") for el in (*root.iter("geom"), *root.iter("pair"))}
    assert float(np.float32(MINMU_F32)) > mujoco.mjMINMU > float(np.float32(1e-5))
    assert repr(MINMU_F32) == "1.0000001e-05"
    assert f == {"a": "0.5 0.005 1.0000001e-05", "b": "0.5 1e-06 1e-06", "c": "0.5 1.5e-05 0.0", "d": "0.0 0.0 0.0",
                 "pair": "1.0 1.0 1.0000001e-05 9e-07 9e-07"}
    assert floor_friction(root) == 0                                          # idempotent


def test_msh_round_trip(tmp_path: Path) -> None:
    mesh = MeshData(np.random.default_rng(0).random((7, 3), dtype=np.float32), np.array([[0, 1, 2], [2, 3, 4]]))
    write_msh(tmp_path / "a.msh", mesh)
    back = read_msh(tmp_path / "a.msh")
    assert np.array_equal(back.vertices, mesh.vertices) and np.array_equal(back.faces, mesh.faces)


def test_converted_model_equals_original(synth_model: mujoco.MjModel, tmp_path: Path) -> None:
    xml = write_scene(synth_model, tmp_path / "robots", "synth")
    new = mujoco.MjModel.from_xml_path(str(xml))
    worst = {d.field: d.max_abs for d in compare_models(synth_model, new) if d.max_abs > 1e-9 * max(1, d.ref_scale)}
    assert worst == {}


def test_manifest_detects_changes(synth_model: mujoco.MjModel, tmp_path: Path) -> None:
    out = tmp_path / "robots"
    xml = write_scene(synth_model, out, "synth")
    assert check_manifest(out) == []
    xml.write_text(xml.read_text() + " ")
    (out / "extra.txt").write_text("x")
    problems = check_manifest(out)
    assert any("mismatch: synth/scene.xml" in p for p in problems) and any("unlisted: extra.txt" in p for p in problems)
    assert (out / MANIFEST_NAME).is_file() and any((out / "meshes").glob("wedge-*.msh"))


POLYGON_FIELDS = ("mesh_polynum", "mesh_polyadr", "mesh_polynormal", "mesh_polyvertadr", "mesh_polyvertnum",
                  "mesh_polyvert", "mesh_polymapadr", "mesh_polymapnum", "mesh_polymap")


def _polygons(m: mujoco.MjModel) -> list[tuple[int, ...]]:
    """The mesh's hull polygons as vertex sets, in no order."""
    out = []
    for p in range(int(m.mesh_polynum[0])):
        a, k = int(m.mesh_polyvertadr[p]), int(m.mesh_polyvertnum[p])
        out.append(tuple(sorted(m.mesh_polyvert[a:a + k].tolist())))
    return sorted(out)


def test_converted_scene_replays_recording(synth_recording: Path, tmp_path: Path) -> None:
    """The converted wedge mesh has the recorded hull (the same polygons), and the converted scene replays the
    recording to 1e-6 when the hull's polygons are in the recording's order. MuJoCo builds that order itself from
    the vertices, and the float32 source-frame round trip (~1e-9 m) reorders it. MuJoCo 3.17's multi-CCD reads the
    order (WP0's Newton bump, 2026-11-09): as converted, the replay drifts 9.7 mm at step 100, then 1.5 mm. With
    the recording's order it drifts 2e-9; with multi-CCD off, 3e-8 (MuJoCo 3.16: ~1e-8 as converted)."""
    rec = load_recorded(synth_recording)
    xml = write_scene(rec.model, tmp_path / "robots", rec.name)
    new = mujoco.MjModel.from_xml_path(str(xml))
    q_orig = replay_cpu(copy.copy(rec.model), rec, rec.steps - 1)
    assert np.abs(q_orig - rec.ref_qpos).max() < 1e-12
    assert _polygons(new) == _polygons(rec.model)
    for field in POLYGON_FIELDS:
        getattr(new, field)[:] = getattr(rec.model, field)
    q_new = replay_cpu(new, rec, rec.steps - 1)
    assert np.abs(q_new - rec.ref_qpos).max() < 1e-6


def test_unsupported_parts_raise(tmp_path: Path) -> None:
    m = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><site name="s1"/><body><joint/><geom size="0.1"/><site name="s2"/></body></worldbody>'
        '<tendon><spatial name="t"><site site="s1"/><site site="s2"/></spatial></tendon></mujoco>'
    )
    with pytest.raises(TraceError, match="ntendon"):
        write_scene(m, tmp_path / "robots", "tendon")


def test_rotation_angle_is_robust_at_small_angles() -> None:
    from alloy.tasks.attribution import rotation_angle

    theta = 1e-6
    q = np.array([np.sin(theta / 2), 0.0, 0.0, np.cos(theta / 2)])
    q32 = q.astype(np.float32)
    assert abs(rotation_angle(q32, np.array([0.0, 0.0, 0.0, 1.0]), True) - theta) < 2e-7
    wxyz = np.array([np.cos(0.3), 0.0, np.sin(0.3), 0.0])
    assert abs(rotation_angle(wxyz, np.array([1.0, 0, 0, 0]), False) - 0.6) < 1e-12
