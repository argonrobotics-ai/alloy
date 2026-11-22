# SPDX-License-Identifier: Apache-2.0
"""Law: a look file path works like the same look dict on scanned objects, and a still render never drops a look.

``scan_objects`` loads a look path before matching scans (it used to pass the Path on and fail); ``render`` with a
``liquids={...}`` dict beside cloth, soft bodies or MPM raises instead of silently dropping ``look=`` and ``device=``,
and a plain still of a cloth scene still draws the cloth.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

import alloy
from alloy.camera import scan_pack, still
from alloy.camera.errors import CameraError
from alloy.camera.scan_textures import ScanSource

RED = {"red": {"color": [0.8, 0.1, 0.1]}}


class _Kept(Exception):
    """Raised by the stand-in texture step with the scan sources that reached it."""


def _stop_at_textures(monkeypatch) -> None:
    from alloy.camera import from_newton

    def spy(_sd, _sim, sources, _info):
        raise _Kept(list(sources))
    monkeypatch.setattr(scan_pack, "with_scan_textures", spy)
    monkeypatch.setattr(from_newton, "scene_from_model", lambda _model: None)


def _mug(tmp_path, monkeypatch):
    src = ScanSource(label="mug/visual", glb=tmp_path / "mug.glb", scale=(1.0, 1.0, 1.0))
    sim = SimpleNamespace(scene=SimpleNamespace(items={"mug": {"card_name": "ycb_025_mug"}}))
    monkeypatch.setattr(scan_pack, "scan_sources", lambda _sim: [src])
    return src, sim


def test_scan_objects_resolves_look_file_path_against_scan_sources(tmp_path, monkeypatch):
    """scan_objects must load a Path look file before passing it to look_over_scan."""
    look_file = tmp_path / "look.json"
    look_file.write_text(
        json.dumps({"materials": {"red": {"color": [0.8, 0.1, 0.1]}}, "assign": [{"match": "mug", "material": "red"}]}),
        encoding="utf-8",
    )
    _src, sim = _mug(tmp_path, monkeypatch)
    objs, rep = scan_pack.scan_objects(object(), sim, look=look_file)
    assert objs == [] and rep == set()


def test_scan_objects_keeps_a_scan_that_the_look_does_not_name(tmp_path, monkeypatch):
    """A look path that names no scanned object keeps the scan, and the path and the same dict keep the same list."""
    look = {"materials": RED, "assign": [{"match": "plate", "material": "red"}]}
    look_file = tmp_path / "look.json"
    look_file.write_text(json.dumps(look), encoding="utf-8")
    src, sim = _mug(tmp_path, monkeypatch)
    _stop_at_textures(monkeypatch)
    kept = []
    for arg in (look_file, look):
        with pytest.raises(_Kept) as got:
            scan_pack.scan_objects(object(), sim, look=arg)
        kept.append(got.value.args[0])
    assert kept[0] == kept[1] == [src]


def test_still_render_refuses_liquids_dict_with_deformables(monkeypatch):
    """render(..., liquids={...}) on a sim with deformables must raise CameraError instead of dropping look/device."""
    from alloy.camera import liquid, meshes

    monkeypatch.setattr(still, "_has_liquid", lambda _sim: True)
    monkeypatch.setattr(still, "_has_deform", lambda _sim: True)
    monkeypatch.setattr(liquid, "liquid_meshes", lambda sim, world, material: [])
    monkeypatch.setattr(meshes, "model_scene", lambda model, water, **kw: object())
    monkeypatch.setattr(still, "snapshot", lambda target, state, **kw: np.zeros((4, 4, 3), dtype=np.uint8))
    sim = SimpleNamespace(engine=SimpleNamespace(model=object(), state=None))
    with pytest.raises(CameraError, match="liquids=True"):
        still.render(sim, liquids={"ior": 1.4}, look="my.json", device="cpu", items=None)


def test_still_render_draws_cloth_in_snapshot():
    """A plain still render of a cloth scene must draw the cloth sheet (not drop live deformables)."""
    empty = alloy.Scene(dt=0.01)
    empty.add_ground()
    sim_empty = alloy.build(empty, device="cpu")
    img_empty = sim_empty.render(width=64, height=64, spp=2, look_at=((0.0, -1.0, 0.8), (0.0, 0.0, 0.2)), device="cpu")
    sim_empty.close()

    with_cloth = alloy.Scene(dt=0.01)
    with_cloth.add_ground()
    with_cloth.add_cloth("sheet", size=(0.5, 0.5), pos=(0.0, 0.0, 0.3), resolution=6)
    sim_cloth = alloy.build(with_cloth, device="cpu")
    img_cloth = sim_cloth.render(width=64, height=64, spp=2, look_at=((0.0, -1.0, 0.8), (0.0, 0.0, 0.2)), device="cpu")
    sim_cloth.close()

    assert np.abs(img_cloth.astype(np.int16) - img_empty.astype(np.int16)).mean() > 1.0
