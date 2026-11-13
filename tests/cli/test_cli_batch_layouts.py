# SPDX-License-Identifier: Apache-2.0
"""``alloy batch`` reads a JSON array or one JSON object in any layout (a "{" at the start of a line included), and
JSON lines; a broken document says where it breaks."""

from __future__ import annotations

import io
import json

import pytest

from alloy.cli import main
from alloy.cli.command import read_requests
from alloy.cli.errors import RequestError

LIST_OPS, LIST_ADDS = {"op": "list", "kind": "ops"}, {"op": "list", "kind": "adds"}


@pytest.mark.parametrize(("text", "n"), [
    ('[\n{"op": "list", "kind": "ops"},\n{"op": "list", "kind": "adds"}\n]\n', 2),        # "{" at column 0
    ('{\n"op": "scene",\n"scene":\n{\n"objects": [{"add": "ground"}]\n}\n}\n', 1),       # nested "{" at column 0
    (json.dumps([LIST_OPS, LIST_ADDS], indent=2), 2),
    (json.dumps([LIST_OPS, LIST_ADDS]), 2),
    (json.dumps(LIST_OPS) + "\n\n" + json.dumps(LIST_ADDS) + "\n", 2),                      # JSON lines
])
def test_every_layout_is_read(text, n):
    requests = read_requests(text)
    assert len(requests) == n and all(isinstance(r, dict) and "op" in r for r in requests), requests


def test_a_column_zero_array_runs_from_a_file(tmp_path):
    path = tmp_path / "requests.json"
    path.write_text('[\n{"op": "list", "kind": "ops"},\n{"op": "list", "kind": "adds"}\n]\n', encoding="utf-8")
    out = io.StringIO()
    code = main(["batch", str(path)], stdout=out)
    reply = json.loads(out.getvalue())
    assert code == 0 and reply["ok"] and reply["result"]["ran"] == 2, reply


@pytest.mark.parametrize(("text", "where"), [
    ('[\n{"op": "list"},\n{"op": \n]', "batch is not valid JSON"),           # a broken document: its own line
    ('{"op": "list"}\n{"op": nope}\n', "batch line 2"),                       # JSON lines: the broken line
])
def test_a_broken_batch_says_where(text, where):
    with pytest.raises(RequestError, match=where):
        read_requests(text)
