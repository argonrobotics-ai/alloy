# SPDX-License-Identifier: Apache-2.0
"""OSS-19 MJCF objects (collision shapes joined as convex hulls, a concave L keeps its notch) and on-demand fetch of an
imported object's files from an asset store (both manifest layouts, sha256 checked, errors that say what to set)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from alloy.compose import geom
from alloy.compose.body import Body, mesh_com
from alloy.compose.fetch import STORE_ENV, fetch_object, files_of, missing
from alloy.compose.mjcf import mjcf_mesh
from alloy.compose.spec import ComposeError

L_SHAPE = """<mujoco model="ell">
  <worldbody>
    <body name="ell" pos="1 2 3">
      <freejoint/>
      <geom type="box" size="0.05 0.01 0.01" pos="0 0 0.01"/>
      <geom type="box" size="0.01 0.01 0.04" pos="-0.04 0 0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_mjcf_shapes_join_without_filling_the_notch(tmp_path):
    path = tmp_path / "ell.xml"
    path.write_text(L_SHAPE, encoding="utf-8")
    v, f = mjcf_mesh(path, "ell")
    assert np.allclose(v.min(axis=0), [-0.05, -0.01, 0.0], atol=1e-6)
    assert np.allclose(v.max(axis=0), [0.05, 0.01, 0.09], atol=1e-6)        # the body frame, not the world
    tris = v[f]
    assert geom.ray_down(tris, np.array([[0.03, 0.0]]))[0] == pytest.approx(0.02)   # the foot, not the post top
    body = Body.from_mesh("ell", v, f, mesh_com(v, f))
    assert not body.reoriented


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _root(tmp_path: Path, layout: str) -> tuple[Path, Path, str]:
    root = tmp_path / "imports" / "demo"
    root.mkdir(parents=True)
    store = tmp_path / "store"
    data = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    rel = "meshes/demo_1/physics.obj" if layout == "objects" else "mjcf/demo_1/hulls/hull_00.msh"
    (store / "demo" / rel).parent.mkdir(parents=True)
    (store / "demo" / rel).write_bytes(data)
    files = {("physics.obj" if layout == "objects" else rel): {"sha256": _digest(data), "bytes": len(data)}}
    (root / "manifest.json").write_text(json.dumps({layout: {"demo_1": {"files": files}}}), encoding="utf-8")
    return root, store, rel


@pytest.mark.parametrize("layout", ["objects", "assets"])
def test_fetch_copies_missing_files_and_checks_them(tmp_path, layout):
    root, store, rel = _root(tmp_path, layout)
    assert list(files_of(root, "demo_1")) == [rel] and missing(root, "demo_1") == [rel]
    assert fetch_object(root, "demo_1", str(store)) == 1
    assert (root / rel).is_file() and missing(root, "demo_1") == []
    assert fetch_object(root, "demo_1", str(store)) == 0                     # already there


def test_fetch_errors_say_what_to_set(tmp_path, monkeypatch):
    root, store, rel = _root(tmp_path, "objects")
    monkeypatch.delenv(STORE_ENV, raising=False)
    with pytest.raises(ComposeError) as e:
        fetch_object(root, "demo_1")
    assert STORE_ENV in e.value.fix
    (store / "demo" / rel).write_bytes(b"stale")
    with pytest.raises(ComposeError) as e:
        fetch_object(root, "demo_1", str(store))
    assert "wrong sha256" in e.value.message and not (root / rel).exists()


def _assets_root(tmp_path: Path) -> tuple[Path, Path]:
    """An 'assets'-layout import root with model.xml in it (as in git) and its hull only in the store."""
    root = tmp_path / "pkg" / "imports" / "demo"
    store = tmp_path / "store"
    xml, hull = b"<mujoco/>", b"hull bytes"
    files = {"mjcf/demo_1/model.xml": xml, "mjcf/demo_1/hulls/hull_00.msh": hull}
    (root / "mjcf" / "demo_1").mkdir(parents=True)
    (root / "mjcf/demo_1/model.xml").write_bytes(xml)
    (store / "demo/mjcf/demo_1/hulls").mkdir(parents=True)
    (store / "demo/mjcf/demo_1/hulls/hull_00.msh").write_bytes(hull)
    manifest = {"assets": {"demo_1": {"files": {k: {"sha256": _digest(v)} for k, v in files.items()}}}}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, store


def test_a_read_only_root_fetches_into_the_cache_complete(tmp_path, monkeypatch):
    from alloy.compose.fetch import CACHE_ENV, cache_root, cache_roots, default_dest

    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    root, store = _assets_root(tmp_path)
    assert default_dest(root) == root.resolve()                              # a writable checkout: in place
    monkeypatch.setattr("alloy.compose.fetch.os.access", lambda p, mode: False)   # read-only (system / container)
    cache = cache_root("demo")
    assert default_dest(root) == cache and cache == tmp_path / "cache" / "imports" / "demo"
    assert sorted(missing(root, "demo_1")) == ["mjcf/demo_1/hulls/hull_00.msh", "mjcf/demo_1/model.xml"]
    assert fetch_object(root, "demo_1", str(store)) == 2                    # model.xml copied, hull fetched
    assert (cache / "mjcf/demo_1/model.xml").read_bytes() == b"<mujoco/>"
    assert not (root / "mjcf/demo_1/hulls").exists() and missing(root, "demo_1") == []
    assert cache_roots() == [cache]


def test_a_root_inside_the_installed_package_is_never_written(tmp_path, monkeypatch):
    import alloy
    from alloy.compose.fetch import CACHE_ENV, default_dest

    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "cache"))
    packaged = Path(alloy.__file__).resolve().parent / "_data" / "imports" / "ycb_gso"
    assert default_dest(packaged) == tmp_path / "cache" / "imports" / "ycb_gso"
