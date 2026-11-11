# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import pytest

from bench.render_gates import refs

from .conftest import make_frame


def test_pack_select_and_fetch_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "refs"
    make_frame(src, "c3", "10", {"cycles_s1": 1.0, "alloy": 1.0})  # the engine folder must stay out of the tar
    (src / "c3" / "10" / "cycles" / "view.json").write_text("{}")
    tars = tmp_path / "tars"
    entry = refs.pack_set(src, tars, "c3", "32x32", "png")
    assert entry and entry["images"] == 11 and entry["file"] == "c3_32x32_png.tar"
    assert refs.pack_set(src, tars, "c3", "64x64", "png") is None
    man = {"schema": "alloy.render_gates.refs/1", "sets": [entry]}
    got = refs.fetch(refs.select(man, ["c3"], ["32x32"]), tmp_path / "dest", uri=str(tars))
    assert got == ["c3_32x32_png.tar"]
    assert (tmp_path / "dest" / "c3" / "10" / "cycles" / "32x32_4096spp_on.png").exists()
    assert (tmp_path / "dest" / "c3" / "10" / "cycles" / "view.json").exists()


def test_fetch_rejects_a_corrupt_tar(tmp_path: Path) -> None:
    (tmp_path / "x.tar").write_bytes(b"not a tar")
    entry = {"scene": "c3", "res": "32x32", "kind": "png", "file": "x.tar", "sha256": "0" * 64}
    with pytest.raises(refs.RefsError, match="sha256"):
        refs.fetch([entry], tmp_path / "dest", uri=str(tmp_path))


def test_fetch_without_location_says_what_to_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(refs.URI_ENV, raising=False)
    with pytest.raises(refs.RefsError, match=refs.URI_ENV):
        refs.fetch([], tmp_path)


def test_shipped_manifest_loads() -> None:
    m = refs.load_manifest()
    assert m["schema"] == "alloy.render_gates.refs/1"
    for e in m["sets"]:
        assert {"scene", "res", "kind", "file", "sha256", "bytes"} <= set(e)


def test_scene_pack_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "packs" / "c3"
    (src / "frames" / "420").mkdir(parents=True)
    (src / "pack.json").write_text("{}")
    (src / "frames" / "420" / "frame.json").write_text("{}")
    entry = refs.pack_scene(src, tmp_path / "tars", "c3")
    assert entry["kind"] == "pack" and entry["res"] == "*" and entry["files"] == 2
    man = {"sets": [entry]}
    assert refs.select(man, ["c3"], ["640x480"], kind="pack") == [entry]
    refs.fetch([entry], tmp_path / "got", uri=str(tmp_path / "tars"))
    assert (tmp_path / "got" / "c3" / "frames" / "420" / "frame.json").exists()


def test_passes_are_their_own_kind(tmp_path: Path) -> None:
    eng = tmp_path / "refs" / "c3" / "10" / "cycles_s1"
    eng.mkdir(parents=True)
    for name in ("32x32_4_spp_off.exr", "32x32_4spp_off.exr", "32x32_4spp_off_passes.exr", "64x64_4spp_off_passes.exr"):
        (eng / name).write_bytes(b"x")
    exr = refs.pack_set(tmp_path / "refs", tmp_path / "t", "c3", "32x32", "exr")
    passes = refs.pack_set(tmp_path / "refs", tmp_path / "t", "c3", "32x32", "passes")
    assert exr["images"] == 1 and passes["images"] == 1 and passes["file"] == "c3_32x32_passes.tar"
