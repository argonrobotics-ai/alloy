# SPDX-License-Identifier: Apache-2.0
"""create and audit as `alloy` commands (and so MCP tools): a mesh in millimetres becomes a card through
`alloy call {"op": "create", ...}`, and a scene with it is audited through `{"op": "audit", ...}`, JSON in and out."""

from __future__ import annotations

import io
import json

import pytest

pytest.importorskip("trimesh")
pytest.importorskip("coacd")
pytest.importorskip("fast_simplification")

from alloy.cli import main  # noqa: E402

SCENE = """schema: alloy.scene/1
robots: []
objects:
- object: table_top_plywood
  pose: [0.0, 0.0, -0.009, 0.0, 0.0, 0.0, 1.0]
- object: cli_block
  pose: [0.0, 0.0, 0.0105, 0.0, 0.0, 0.0, 1.0]
"""


def _call(request: dict) -> tuple[int, dict]:
    out = io.StringIO()
    code = main(["call", json.dumps(request)], stdout=out)
    return code, json.loads(out.getvalue())


@pytest.mark.timeout(240)
def test_create_then_audit_through_the_alloy_command(tmp_path):
    import trimesh

    mesh = tmp_path / "cli_block.stl"
    trimesh.creation.box((40.0, 30.0, 20.0)).export(mesh)                     # millimetres, like most CAD exports
    lib = tmp_path / "lib"
    code, reply = _call({"op": "create", "mesh": str(mesh), "material": "wood", "out": str(lib), "check": False})
    made = reply["result"]
    assert code == 0 and made["schema"] == "alloy.create/1" and made["name"] == "cli_block"
    assert made["values"][0]["value"] == pytest.approx(0.04) and (lib / "objects/cli_block.yaml").is_file()
    (tmp_path / "scene.yaml").write_text(SCENE)
    code, reply = _call({"op": "audit", "scene": str(tmp_path / "scene.yaml"), "library": [str(lib)], "rounds": 1,
                         "episodes": 1, "device": "cpu", "worlds": 1, "seconds": 0.2})
    report = reply["result"]
    assert code == 0 and report["schema"] == "alloy.audit/1" and report["ok"], report
    assert report["runs"]["rounds"] == 1
