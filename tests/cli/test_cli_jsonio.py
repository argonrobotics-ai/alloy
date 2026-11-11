# SPDX-License-Identifier: Apache-2.0
"""JSON conversion (NumPy, float32, NaN) and the stdout guard."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from alloy.cli.errors import RequestError, error_payload
from alloy.cli.jsonio import dumps, loads_request, plain


def test_plain_numpy_and_non_finite():
    data = {"pos": np.array([0.02, 0.0, 1.0], np.float32), "nan": float("nan"), "grid": np.array([[1.0, np.inf]]),
            "f32": np.float32(0.1), "i": np.int64(3), "b": np.bool_(True), "empty": np.zeros((0, 3), np.float32),
            "nested": [(1, 2), {"k": np.arange(2)}], "path": __import__("pathlib").Path("/x")}
    out = plain(data)
    assert out == {"pos": [0.02, 0.0, 1.0], "nan": None, "grid": [[1.0, None]], "f32": 0.1, "i": 3, "b": True,
                   "empty": [], "nested": [[1, 2], {"k": [0, 1]}], "path": "/x"}
    assert json.loads(dumps(data)) == out
    assert "\n" in dumps({"a": 1}, pretty=True) and "\n" not in dumps({"a": [1, 2]})


def test_request_json_errors_say_where():
    with pytest.raises(RequestError, match="line 1 column 2"):
        loads_request("{op: 1}", "request")
    assert loads_request('{"op": "list"}') == {"op": "list"}


def test_error_payloads_carry_fixes():
    assert error_payload(FileNotFoundError("x.xml"))["fix"].startswith("pass the path")
    assert error_payload(ValueError("bad"))["type"] == "ValueError"
    payload = error_payload(ZeroDivisionError("oops"))
    assert "bug in Alloy" in payload["fix"] and payload["message"] == "oops"


def test_json_stdout_keeps_prints_off_stdout():
    code = ("import os, sys\nfrom alloy.cli.jsonio import json_stdout\n"
            "with json_stdout() as out:\n    print('noise')\n    os.write(1, b'c-noise\\n')\n"
            "    out.write('{\"ok\": true}\\n')\nprint('after')\n")
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert res.returncode == 0, res.stderr
    assert res.stdout.splitlines() == ['{"ok": true}', "after"]
    assert "noise" in res.stderr and "c-noise" in res.stderr
