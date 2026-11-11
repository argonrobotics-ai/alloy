# SPDX-License-Identifier: Apache-2.0
"""JSON in and out: make results JSON-safe (NumPy arrays, paths, NaN), and keep stdout for JSON only.

Example::

    import numpy as np
    from alloy.cli.jsonio import plain, dumps
    plain({"pos": np.array([0.02, 0.0, 1.0], np.float32), "bad": float("nan")})
    # {"pos": [0.02, 0.0, 1.0], "bad": None}       (float32 values keep their short form, NaN becomes null)
    dumps({"ok": True})                             # '{"ok": true}'
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import IO, Any


def plain(value: Any) -> Any:
    """``value`` as JSON-ready data: dicts, lists, str, int, float, bool, None. NumPy arrays become (nested) lists,
    float32 numbers keep their shortest decimal form (no float64 noise), non-finite floats become None, paths and
    unknown objects become strings. Example: ``plain(np.float32(0.1)) == 0.1``."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            return _array(np, value)
        if isinstance(value, np.generic):
            return _array(np, np.asarray(value))
    if hasattr(value, "to_dict"):
        return plain(value.to_dict())
    return str(value)


def _array(np: Any, a: Any) -> Any:
    if a.dtype.kind == "f":
        if a.dtype.itemsize < 8:
            a = a.astype(str).astype(np.float64)    # the shortest decimal of each float32, not its float64 noise
        out = a.astype(object)
        out[~np.isfinite(a)] = None
        return out.tolist()
    if a.dtype.kind in "biu":
        return a.tolist()
    return plain(a.tolist())


def dumps(value: Any, *, pretty: bool = False) -> str:
    """Strict JSON text of ``plain(value)`` (never NaN): one line, or indented with ``pretty``.

    Example: ``dumps({"a": np.zeros(2)}) == '{"a": [0.0, 0.0]}'``."""
    return json.dumps(plain(value), indent=2 if pretty else None, allow_nan=False,
                      separators=None if pretty else (", ", ": "))


def loads_request(text: str, source: str = "input") -> Any:
    """Parse request JSON; a syntax error becomes a ``RequestError`` saying where. Example: ``loads_request('{}')``."""
    from alloy.cli.errors import RequestError  # noqa: PLC0415

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RequestError(f"{source} is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}",
                           'pass one JSON object such as {"op": "list", "kind": "scenes"}', field=source) from None


@contextlib.contextmanager
def json_stdout() -> Iterator[IO[str]]:
    """Yield a stream on the real stdout, while everything else that writes to stdout (``print``, Warp's start-up
    lines, C libraries writing to fd 1) goes to stderr instead. JSON tools and the MCP stdio transport need a
    stdout that carries nothing but their messages.

    Example::

        with json_stdout() as out:
            result = run_engine()          # its prints land on stderr
            out.write(dumps(result) + "\\n")
    """
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    old = sys.stdout
    sys.stdout = sys.stderr
    stream = os.fdopen(os.dup(saved), "w", encoding="utf-8", buffering=1)
    try:
        yield stream
    finally:
        stream.flush()
        stream.close()
        sys.stderr.flush()
        os.dup2(saved, 1)
        os.close(saved)
        sys.stdout = old
