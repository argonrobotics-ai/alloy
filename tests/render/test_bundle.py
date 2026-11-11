# SPDX-License-Identifier: Apache-2.0
"""Bundle reader/validator: version, missing files, and each class of inconsistency."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from alloy.render import ARB_VERSION, BundleError, RenderRecorder, load_bundle, validate_bundle
from tests.render.helpers import move_bodies, rigid_world


@pytest.fixture(scope="module")
def good(tmp_path_factory):
    path = tmp_path_factory.mktemp("bundle") / "good"
    model = rigid_world().finalize(device="cpu")
    rec = RenderRecorder(model, path, fps=24.0)
    st = model.state()
    for k in range(3):
        rec.capture(st, k / 24.0)
        move_bodies(st, 0.05)
    rec.close()
    return path


@pytest.fixture
def copy(good, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(good, dst)
    return dst


def edit_scene(root, fn):
    f = root / "scene.json"
    sc = json.loads(f.read_text())
    fn(sc)
    f.write_text(json.dumps(sc))


def test_roundtrip_basics(good):
    b = load_bundle(good)
    assert ARB_VERSION == 1 and validate_bundle(b) == []
    assert b.array("body_q").shape == (3, 4, 7)
    assert not b.array("body_q", mmap=True).flags.writeable
    assert isinstance(b.array("body_q", mmap=False), np.ndarray)
    assert set(b.labels()) == {"shapes", "bodies", "surfaces", "points", "curves", "instancers"}
    with pytest.raises(BundleError, match="missing frames/nope.npy"):
        b.array("nope")


def test_load_errors(tmp_path, copy):
    with pytest.raises(BundleError, match="no scene.json"):
        load_bundle(tmp_path / "empty")
    edit_scene(copy, lambda sc: sc.update(arb=2))
    with pytest.raises(BundleError, match="arb=2"):
        load_bundle(copy)
    edit_scene(copy, lambda sc: sc.update(arb=1))
    (copy / "frames" / "body_q.npy").unlink()
    with pytest.raises(BundleError, match="missing files.*body_q"):
        load_bundle(copy)
    (copy / "scene.json").write_text("{nope")
    with pytest.raises(BundleError, match="invalid JSON"):
        load_bundle(copy)


def test_validate_bad_refs(copy):
    def bad(sc):
        sc["shapes"][1]["body"] = 99
        sc["shapes"][4]["mesh"] = "missing"
        sc["shapes"][2]["replica_of"] = 1234
    edit_scene(copy, bad)
    problems = validate_bundle(load_bundle(copy))
    assert any("body 99 out of range" in p for p in problems)
    assert any("unknown mesh 'missing'" in p for p in problems)
    assert any("replica_of 1234" in p for p in problems)


def test_validate_array_shape_and_nan(copy):
    arr = np.load(copy / "frames" / "body_q.npy")
    arr[1, 2, 3] = np.nan
    np.save(copy / "frames" / "body_q.npy", arr)
    problems = validate_bundle(load_bundle(copy))
    assert problems == ["frames/body_q.npy has NaN/Inf at frame 1"]
    np.save(copy / "frames" / "body_q.npy", arr[:, :2])
    assert "expected (3, 4, 7)" in validate_bundle(load_bundle(copy))[0]


def test_validate_time_and_mesh(copy):
    np.save(copy / "frames" / "t.npy", np.array([0.0, 0.0, 1.0]))
    assert any("not strictly increasing" in p for p in validate_bundle(load_bundle(copy)))
    sc = json.loads((copy / "scene.json").read_text())
    mid = next(iter(sc["meshes"]))
    path = copy / sc["meshes"][mid]["file"]
    np.savez(path, vertices=np.zeros((4, 3), np.float32), faces=np.full((4, 3), 7, np.int32))
    assert any("face index out of range" in p for p in validate_bundle(load_bundle(copy)))
