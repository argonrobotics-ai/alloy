# SPDX-License-Identifier: Apache-2.0
"""Canonical hashing (SPEC §3.1): stable digests of cards, configs and arrays across processes and machines.

``canonical_json`` writes compact JSON with sorted keys and every float as ``format(x, ".17g")`` (17 significant
digits round-trip a float64 exactly). Dataclasses encode as their fields (``asdict`` semantics, no type name), enums
as their value, tuples/lists as lists, sets as sorted lists, mappings with non-string keys use the canonical text of
the key, and numpy/Warp arrays as ``{"__ndarray__": [dtype, shape, sha256 of the C-order bytes]}``. Non-finite
floats encode as the strings "NaN", "Infinity", "-Infinity".
A dataclass field declared with ``metadata=OMIT_IF_NONE`` is left out while its value is None, so adding an optional
block to a card keeps every existing digest (WP1/WP4 2026-11-06: a new ``None`` field would move every card_hash).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import math
from collections.abc import Mapping, Set
from pathlib import PurePath
from typing import Any

import numpy as np

from alloy.core.errors import DigestError

#: ``field(default=None, metadata=OMIT_IF_NONE)``: the field is not digested while it is None.
OMIT_IF_NONE: Mapping[str, bool] = {"digest_omit_none": True}


def sha256_hex(data: bytes) -> str:
    """Hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj: Any) -> bytes:
    """Canonical UTF-8 JSON bytes of ``obj`` (see module docstring). Raises ``DigestError`` on unsupported types."""
    return _encode(obj).encode("utf-8")


def digest_of(obj: Any) -> str:
    """``sha256_hex(canonical_json(obj))``."""
    return sha256_hex(canonical_json(obj))


def _encode(obj: Any) -> str:
    scalar = _encode_scalar(obj)
    if scalar is not None:
        return scalar
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _encode_mapping({f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)
                                if not (f.metadata.get("digest_omit_none") and getattr(obj, f.name) is None)})
    if isinstance(obj, Mapping):
        return _encode_mapping(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_encode(v) for v in obj) + "]"
    if isinstance(obj, (Set, frozenset)):
        return "[" + ",".join(sorted(_encode(v) for v in obj)) + "]"
    array = _as_ndarray(obj)
    if array is not None:
        return _encode_array(array)
    raise DigestError(f"canonical_json cannot encode {type(obj).__module__}.{type(obj).__qualname__}")


def _encode_scalar(obj: Any) -> str | None:
    if obj is None:
        return "null"
    if isinstance(obj, bool | np.bool_):
        return "true" if obj else "false"
    if isinstance(obj, enum.Enum):
        return _encode(obj.value)
    if isinstance(obj, int | np.integer):
        return str(int(obj))
    if isinstance(obj, float | np.floating):
        return _encode_float(float(obj))
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=True)
    if isinstance(obj, PurePath):
        return json.dumps(obj.as_posix(), ensure_ascii=True)
    if isinstance(obj, bytes):
        return json.dumps({"__bytes__": sha256_hex(obj)}, ensure_ascii=True)
    return None


def _encode_float(x: float) -> str:
    if math.isnan(x):
        return '"NaN"'
    if math.isinf(x):
        return '"Infinity"' if x > 0 else '"-Infinity"'
    return format(x, ".17g")


def _encode_mapping(mapping: Mapping[Any, Any]) -> str:
    items = []
    for key, value in mapping.items():
        text_key = key if isinstance(key, str) else _encode(key)
        items.append((text_key, value))
    items.sort(key=lambda kv: kv[0])
    keys = [k for k, _ in items]
    if len(set(keys)) != len(keys):
        raise DigestError(f"mapping keys collide after canonicalisation: {keys}")
    return "{" + ",".join(json.dumps(k, ensure_ascii=True) + ":" + _encode(v) for k, v in items) + "}"


def _as_ndarray(obj: Any) -> np.ndarray | None:
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "numpy") and hasattr(obj, "dtype") and hasattr(obj, "shape") and hasattr(obj, "device"):
        return np.asarray(obj.numpy())  # warp array: host copy
    return None


def _encode_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    if contiguous.dtype.byteorder == ">":
        contiguous = contiguous.astype(contiguous.dtype.newbyteorder("<"))
    meta = [contiguous.dtype.str, list(contiguous.shape), sha256_hex(contiguous.tobytes())]
    return '{"__ndarray__":' + _encode(meta) + "}"
