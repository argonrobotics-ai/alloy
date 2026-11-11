# SPDX-License-Identifier: Apache-2.0
"""Operations other packages provide (SCR-OSS19-3, SCR OSS-11-5): create, audit, compose and fetch are commands and
MCP tools, their specs are valid, a missing provider is skipped and a clashing name raises. fetch runs against a fake
Hub (no network): a listing comes back as JSON, a private dataset as a HubError whose fix says how to get access."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from alloy.cli import main
from alloy.cli.registry import OPS, PROVIDERS, add_providers
from alloy.cli.spec import Op, check_request, json_schema
from alloy.mcp.tools import tool_list

PROVIDED = {"create", "audit", "compose", "fetch"}
INDEX = [{"name": "ycb_025_mug", "library": "ycb_gso", "category": "ycb", "size": 1000},
         {"name": "gso_shoe_1", "library": "ycb_gso", "category": "gso", "size": 2000},
         {"name": "objv_mug_1a2b3c4d", "library": "objaverse", "category": "mug", "size": 500}]


def test_provided_ops_are_tools_and_commands():
    assert PROVIDED <= set(OPS)
    tools = {t["name"]: t for t in tool_list()}
    assert PROVIDED <= set(tools)
    for name in PROVIDED:
        assert tools[name]["inputSchema"] == json_schema(OPS[name]) and tools[name]["description"]
        check_request(OPS[name], OPS[name].example)              # the example is a valid request
    out = io.StringIO()
    assert main(["list", "ops"], stdout=out) == 0
    assert PROVIDED <= {r["name"] for r in json.loads(out.getvalue())["result"]["items"]}
    for name in PROVIDED:
        with pytest.raises(SystemExit) as info:                   # `alloy <name> --help` exists
            main([name, "--help"], stdout=io.StringIO())
        assert info.value.code == 0


def test_missing_provider_is_skipped_and_clash_raises(caplog):
    ops: dict[str, Op] = {}
    assert add_providers(ops, ("alloy.no_such_package.ops",)) == [] and ops == {}
    assert "not available" in caplog.text
    assert add_providers(ops, PROVIDERS[:1]) == ["compose"]
    with pytest.raises(RuntimeError, match="already defined"):
        add_providers(ops, PROVIDERS[:1])


def test_bad_arguments_get_a_fix():
    out = io.StringIO()
    assert main(["call", json.dumps({"op": "create", "colour": "red"})], stdout=out) == 2
    err = json.loads(out.getvalue())["error"]
    assert err["type"] == "RequestError" and err["field"] == "colour" and "mesh" in err["fix"]


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake Hub: ``download`` serves ``INDEX`` as index.jsonl, or raises what ``hub.fail`` holds."""
    from alloy.assets.hub import source

    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path / "cache"))

    class Hub:
        fail: Exception | None = None

    def fake_download(path, local_dir, repo=None):
        if Hub.fail is not None:
            raise Hub.fail
        dst = Path(local_dir) / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("".join(json.dumps(row) + "\n" for row in INDEX))
        return dst

    monkeypatch.setattr(source, "download", fake_download)
    return Hub


def test_fetch_list_returns_json(hub):
    out = io.StringIO()
    assert main(["fetch", "--list", "--library", "ycb_gso"], stdout=out) == 0
    res = json.loads(out.getvalue())["result"]
    assert res["objects"] == 2 and res["categories"] == {"ycb_gso/gso": 1, "ycb_gso/ycb": 1}


def test_fetch_from_a_private_dataset_says_request_access_then_log_in(hub):
    from alloy.assets.hub import source

    hub.fail = source.HubError("the object library owner/alloy-library is private or does not exist (index.jsonl)",
                               source.ACCESS_FIX)
    out = io.StringIO()
    assert main(["fetch", "--list", "--library", "ycb_gso"], stdout=out) == 2
    err = json.loads(out.getvalue())["error"]
    assert err["type"] == "HubError" and "request access" in err["fix"] and "hf auth login" in err["fix"]


def test_fetch_takes_several_names_as_words(monkeypatch):
    seen = []
    monkeypatch.setattr("alloy.assets.hub.fetch.fetch", lambda names, *rest: seen.append(names) or {"objects": 0})
    for argv in (["fetch", "ycb_025_mug", "objv_mug_1a2b3c4d"], ["fetch", "ycb_025_mug,objv_mug_1a2b3c4d"]):
        assert main(argv, stdout=io.StringIO()) == 0
    assert seen == [["ycb_025_mug", "objv_mug_1a2b3c4d"]] * 2
