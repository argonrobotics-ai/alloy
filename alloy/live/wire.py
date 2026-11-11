# SPDX-License-Identifier: Apache-2.0
"""Wire format between a training worker and the live hub: one framed message = a JSON header plus a binary payload.

Frame: ``u32 total`` (big-endian, bytes after this field), ``u8 kind``, ``u32 header_len``, the JSON header, then the
payload. Numeric arrays travel packed in the payload; the header lists them as ``{"arrays": [[name, dtype, shape,
offset], ...]}`` (:func:`pack_arrays`). Stdlib + NumPy only, so the worker side needs nothing new installed.

Example::

    meta, blob = pack_arrays({"reward": np.zeros(64, np.float32)})
    frame = encode(Kind.NUMBERS, {"t": 1.0, **meta}, blob)
    kind, header, payload = decode_body(frame[4:])
    arrays = unpack_arrays(header, payload)
"""
from __future__ import annotations

import json
import socket
import struct
from enum import IntEnum
from typing import Any

import numpy as np

MAX_FRAME = 64 << 20          # a sanity bound: a frame larger than this is a protocol error, not data
_HEAD = struct.Struct(">BI")
_LEN = struct.Struct(">I")


class Kind(IntEnum):
    """Message kinds. Worker -> hub: HELLO, NUMBERS, STATE, MESH, VIDEO, RESULT, LOG. Hub -> worker: PICK, COMMAND."""
    HELLO = 1
    NUMBERS = 2
    STATE = 3
    MESH = 4
    VIDEO = 5
    RESULT = 6
    LOG = 7
    PICK = 20
    COMMAND = 21


class WireError(ValueError):
    """A malformed or oversized frame."""


def encode(kind: Kind, header: dict[str, Any], payload: bytes = b"") -> bytes:
    """One frame, ready for ``sock.sendall``."""
    head = json.dumps(header, separators=(",", ":")).encode()
    body_len = _HEAD.size + len(head) + len(payload)
    if body_len > MAX_FRAME:
        raise WireError(f"frame of {body_len} bytes is over the {MAX_FRAME} byte bound")
    return _LEN.pack(body_len) + _HEAD.pack(int(kind), len(head)) + head + payload


def decode_body(body: bytes | memoryview) -> tuple[Kind, dict[str, Any], memoryview]:
    """Split a frame body (everything after the length field) into kind, header, payload."""
    view = memoryview(body)
    if len(view) < _HEAD.size:
        raise WireError("frame shorter than its header")
    kind, head_len = _HEAD.unpack_from(view)
    end = _HEAD.size + head_len
    if end > len(view):
        raise WireError("header runs past the frame")
    header = json.loads(bytes(view[_HEAD.size:end]))
    return Kind(kind), header, view[end:]


def read_frame(sock: socket.socket) -> tuple[Kind, dict[str, Any], memoryview] | None:
    """Blocking read of one frame from a socket; None when the peer closed."""
    raw = _read_exact(sock, _LEN.size)
    if raw is None:
        return None
    (n,) = _LEN.unpack(raw)
    if n > MAX_FRAME:
        raise WireError(f"peer announced a {n} byte frame")
    body = _read_exact(sock, n)
    return None if body is None else decode_body(body)


def _read_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def pack_arrays(arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], bytes]:
    """Pack named arrays into one payload; returns ``({"arrays": [...]}, payload)``."""
    parts, table, offset = [], [], 0
    for name, arr in arrays.items():
        a = np.ascontiguousarray(arr)
        raw = a.tobytes()
        table.append([name, a.dtype.str, list(a.shape), offset])
        parts.append(raw)
        offset += len(raw)
    return {"arrays": table}, b"".join(parts)


def unpack_arrays(header: dict[str, Any], payload: bytes | memoryview) -> dict[str, np.ndarray]:
    """Inverse of :func:`pack_arrays` (copies, so the arrays outlive the payload buffer)."""
    out = {}
    for name, dtype, shape, offset in header.get("arrays", []):
        dt = np.dtype(dtype)
        count = int(np.prod(shape)) if shape else 1
        out[name] = np.frombuffer(payload, dtype=dt, count=count, offset=offset).reshape(shape).copy()
    return out
