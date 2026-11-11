# SPDX-License-Identifier: Apache-2.0
"""import and export through the command line: a library object to MJCF, back into cards, and a check of the
imported scene's description."""

from __future__ import annotations

import io
import json

from alloy.cli import main


def run(argv):
    out = io.StringIO()
    code = main(argv, stdout=out)
    return code, json.loads(out.getvalue())


def test_export_object_then_import_it(tmp_path):
    code, exp = run(["export", "--object", "cube_beech_30", "--out", str(tmp_path / "exp"), "--formats", "mjcf"])
    assert code == 0 and exp["ok"], exp
    xml = exp["result"]["files"]["mjcf"]
    assert xml.endswith(".xml") and exp["result"]["counts"]["mjcf"]["bodies"] == 1
    code, imp = run(["import", xml, "--out", str(tmp_path / "lib")])
    assert code == 0 and imp["result"]["objects"] == ["cube_beech_30"], imp
    code, scene = run(["scene", imp["result"]["scene"]])
    assert code == 0 and scene["result"]["names"] == ["cube_beech_30"]


def test_export_needs_one_source_and_import_needs_a_file(tmp_path):
    code, reply = run(["export", "--out", str(tmp_path)])
    assert code == 2 and "exactly one of scene / object" in reply["error"]["message"]
    code, reply = run(["import", str(tmp_path / "missing.urdf"), "--out", str(tmp_path / "lib")])
    assert code == 2 and reply["error"]["fix"]
    code, reply = run(["import", "--out", str(tmp_path / "lib")])
    assert code == 2 and reply["error"]["field"] == "file"
