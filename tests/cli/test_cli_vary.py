# SPDX-License-Identifier: Apache-2.0
"""The ``vary`` operation (scene variants, WP2's ``alloy.scene.variants.vary``): a command and an MCP tool whose
arguments map onto ``vary``, whose reply names absolute files, and whose contact sheet comes back as an MCP image.
One real CPU run: two variants of a table scene, each a Scene JSON file the other operations build from."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from alloy.cli import main
from alloy.cli.ops_vary import TEXTURE_MODES
from alloy.cli.registry import OPS, respond
from alloy.cli.session import Session
from alloy.cli.spec import check_request
from alloy.cli.testing import write_png
from alloy.mcp.tools import call_tool, tool_list

TABLE = {"dt": 0.01, "objects": [
    {"add": "ground"},
    {"add": "box", "name": "table", "size": [0.3, 0.3, 0.4], "pos": [0, 0, 0.2], "static": True,
     "material": "wood_pine"},
    {"add": "box", "name": "a", "size": [0.04, 0.04, 0.04], "pos": [-0.06, 0, 0.42], "material": "wood_beech"},
    {"add": "box", "name": "b", "size": [0.04, 0.04, 0.04], "pos": [0.06, 0, 0.42], "material": "abs"}]}


@pytest.fixture
def fake_vary(monkeypatch):
    """``vary`` replaced by a recorder that writes the files the real one writes (manifest, JSONs, sheet)."""
    calls = []

    def fake(scene, n, *, out, contact_sheet, sheet, **kw):
        calls.append({"n": n, "out": out, "contact_sheet": contact_sheet, "sheet": sheet, **kw})
        out.mkdir(parents=True, exist_ok=True)
        rows = [{"file": f"variant_{k:03d}.json", "draws": 1, "textures": {}, "moves": {}} for k in range(n)]
        for row in rows:
            (out / row["file"]).write_text(scene.to_json())
        manifest = {"n": n, "seed": kw["seed"], "groups": [["cube"]], "families": {"wood": ["w1", "w2"]},
                    "variants": rows}
        if contact_sheet:
            write_png(out / "contact_sheet.png", np.full((8, 8, 3), 200, np.uint8))
            manifest["contact_sheet"] = {"file": "contact_sheet.png", "left_out": ["tex_tint"]}
        (out / "variants.json").write_text(json.dumps(manifest))

    monkeypatch.setattr("alloy.scene.variants.vary", fake)
    return calls


def test_vary_is_a_command_and_a_tool_with_a_valid_example():
    from alloy.scene.variants.looks import MODES

    assert TEXTURE_MODES == MODES
    tool = {t["name"]: t for t in tool_list()}["vary"]
    props = tool["inputSchema"]["properties"]
    assert {"n", "pos", "yaw", "textures", "seed", "out", "sheet"} <= set(props)
    assert set(props["sheet_options"]["properties"]) == {"width", "height", "spp", "quality"}
    check_request(OPS["vary"], OPS["vary"].example)
    with pytest.raises(SystemExit) as info:
        main(["vary", "--help"], stdout=io.StringIO())
    assert info.value.code == 0


def test_flags_map_onto_vary_and_the_reply_names_absolute_files(fake_vary, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = io.StringIO()
    code = main(["vary", json.dumps(TABLE), "--n", "3", "--pos", "0.01", "--yaw", "8", "--textures", "keep",
                 "--seed", "5", "--out", "v", "--sheet", "--sheet-options", '{"quality": "fast", "width": 64}'],
                stdout=out)
    reply = json.loads(out.getvalue())
    assert code == 0 and reply["ok"], reply
    call = fake_vary[0]
    assert (call["n"], call["pos"], call["yaw_deg"], call["textures"], call["seed"]) == (3, 0.01, 8.0, "keep", 5)
    assert call["sheet"] == {"width": 64, "height": 180, "spp": 4, "quality": "fast"} and call["settle"] is None
    res = reply["result"]
    assert res["files"] == [str(tmp_path / "v" / f"variant_{k:03d}.json") for k in range(3)]
    assert res["sheet"] == str(tmp_path / "v" / "contact_sheet.png") and res["families"] == {"wood": 2}
    built = respond(Session(), {"op": "scene", "scene": res["files"][1]})     # a variant file is a scene
    assert built["ok"] and {"a", "b"} <= set(json.dumps(built["result"]["names"]).replace('"', " ").split()), built


def test_no_out_writes_a_temporary_folder_and_mcp_returns_the_sheet(fake_vary):
    result = call_tool(Session(), "vary", {"scene": TABLE, "n": 2, "sheet": True})
    assert not result["isError"], result
    body = json.loads(result["content"][0]["text"])
    assert Path(body["out"]).name.startswith("alloy-vary-") and len(body["files"]) == 2
    assert [c["type"] for c in result["content"]] == ["text", "image"]
    assert result["content"][1]["mimeType"] == "image/png"
    plain = call_tool(Session(), "vary", {"scene": TABLE, "n": 1})
    assert [c["type"] for c in plain["content"]] == ["text"] and fake_vary[-1]["sheet"] is None


@pytest.mark.parametrize(("args", "field"), [
    ({"sheet": True, "sheet_options": {"colour": 1}}, "sheet_options"),
    ({"sheet": True, "sheet_options": {"quality": "best"}}, "sheet_options"),
    ({"sheet": True, "sheet_options": {"spp": 0}}, "sheet_options"),
    ({"textures": "random"}, "textures"),
    ({"n": 0}, "n"),
])
def test_bad_arguments_name_the_field_and_a_fix(fake_vary, args, field):
    reply = respond(Session(), {"op": "vary", "scene": TABLE, **args})
    assert not reply["ok"] and reply["error"]["field"] == field and reply["error"]["fix"], reply
    assert fake_vary == []


def test_two_variants_on_cpu_move_the_free_boxes_only(tmp_path):
    session = Session()
    reply = respond(session, {"op": "vary", "scene": TABLE, "n": 2, "pos": 0.01, "yaw": 5, "seed": 1,
                              "out": str(tmp_path / "v"), "sheet": True,
                              "sheet_options": {"quality": "fast", "width": 32, "height": 24, "spp": 1}})
    assert reply["ok"], reply
    res = reply["result"]
    assert len(res["files"]) == 2 and Path(res["sheet"]).read_bytes()[:4] == b"\x89PNG"
    moved = {name for row in res["variants"] for name in row["moves"]}
    assert moved <= {"a", "b"} and moved, res["variants"]
    first = json.loads(Path(res["files"][0]).read_text())
    assert first["schema"] == "alloy.api.scene/1"
    again = respond(session, {"op": "vary", "scene": TABLE, "n": 2, "pos": 0.01, "yaw": 5, "seed": 1,
                              "out": str(tmp_path / "w")})["result"]
    assert [row["moves"] for row in again["variants"]] == [row["moves"] for row in res["variants"]]
