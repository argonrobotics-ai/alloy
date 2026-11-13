# SPDX-License-Identifier: Apache-2.0
"""``alloy compose`` and the MCP ``compose`` tool when a scene leaves something out: each miss keeps its own reason
and fix (an amount that does not fit names what the container holds), only a word no object matches gets "use
another word", and with ``out`` the scenes that were built are saved before the call fails. Before, every miss read
"no object on this install matches it" and the folder was not written."""

from __future__ import annotations

from alloy.cli.registry import respond
from alloy.cli.session import Session
from alloy.compose.ops import missing_fix


def _compose(**params) -> dict:
    return respond(Session(), {"op": "compose", "check": "none", **params})


def test_a_capacity_miss_says_what_fits_and_the_scene_is_kept(tmp_path):
    out = tmp_path / "out"
    r = _compose(sentence="a bin with 2000 ml of water", out=str(out))
    assert not r["ok"]
    msg, fix = r["error"]["message"], r["error"]["fix"]
    assert "2000 ml of water does not fit in" in msg and "ask for at most" in msg, msg
    assert "no object on this install matches" not in msg and "another word" not in fix
    assert f"saved in {out}" in msg
    assert (out / "scenes" / "scene.yaml").is_file() and (out / "compose.json").is_file()


def test_a_word_no_object_matches_still_gets_the_other_word_fix():
    r = _compose(sentence="a mug and a zorblax on a table")
    assert "1 x zorblax (no object on this install matches it)" in r["error"]["message"]
    assert r["error"]["fix"] == missing_fix()
