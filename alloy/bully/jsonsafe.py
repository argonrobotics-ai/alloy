# SPDX-License-Identifier: Apache-2.0
"""JSON the bully writes (stdout records, ``finding.json``, ``replays.jsonl``): strict RFC 8259.

A world that blows up gives monitor values of ``inf`` or ``nan``; ``json.dumps`` writes those as the bare tokens
``Infinity`` / ``NaN``, which strict JSON parsers reject. ``plain`` turns them into the strings ``"inf"``, ``"-inf"``
and ``"nan"`` (``float()`` reads them back), and numpy values and dataclasses into plain Python ones; ``dumps`` then
writes with ``allow_nan=False``, so a non-finite number can never reach the output unconverted.

Example::

    >>> dumps({"value": float("inf"), "ok": True})
    '{"ok": true, "value": "inf"}'
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

import numpy as np

NON_FINITE = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def plain(x: Any) -> Any:
    """``x`` with dicts, lists and tuples walked, numpy arrays and scalars and dataclasses as Python values, and every
    non-finite float as its string (module doc)."""
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return plain(dataclasses.asdict(x))
    if isinstance(x, float) and not math.isfinite(x):
        return "nan" if math.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


def dumps(obj: Any, **kwargs: Any) -> str:
    """``json.dumps`` of ``plain(obj)``, sorted keys, strict (``allow_nan=False``); other types as ``str``."""
    return json.dumps(plain(obj), allow_nan=False, sort_keys=True, default=str, **kwargs)


def number(v: Any) -> Any:
    """A number field read back from JSON: the strings of ``plain`` as floats, anything else unchanged."""
    return NON_FINITE[v] if isinstance(v, str) and v in NON_FINITE else v
