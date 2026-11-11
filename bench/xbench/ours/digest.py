# SPDX-License-Identifier: Apache-2.0
"""SB-11 per-world digests of an engine's raw state (numpy only; imported by engine runners and the scorer).

A world's digest is SHA-256 over every ``raw_*`` array in name order: the name, dtype and per-world shape, then the
bytes of that world's row (``arr[w]``, C order). It changes when any bit of the world's state changes and does not
depend on the other worlds, which is what Alloy's ``alloy.precision.digest.world_digests`` guarantees for Newton
state (bitwise equality per world, batch invariance R4); Alloy's own rows use that function, rival rows this one.
Arrays must be the engine's own state exactly as its API returns it (no conversion, no reordering), world-major.
"""

from __future__ import annotations

import hashlib

import numpy as np


def world_digests(raw: dict[str, np.ndarray]) -> list[str]:
    """Hex digest per world of ``raw`` ({name: [W, ...] array}); every array must share the leading W."""
    names = sorted(raw)
    if not names:
        raise ValueError("no raw arrays to digest")
    w = {np.asarray(raw[n]).shape[0] for n in names}
    if len(w) != 1:
        raise ValueError(f"raw arrays disagree on the world count: {sorted(w)}")
    out = []
    for k in range(w.pop()):
        h = hashlib.sha256()
        for n in names:
            row = np.ascontiguousarray(np.asarray(raw[n])[k])
            h.update(f"{n}|{row.dtype.str}|{row.shape}".encode())
            h.update(row.tobytes())
        out.append(h.hexdigest()[:32])
    return out


def equal_share(a: list[str], b: list[str]) -> float:
    """Share of worlds whose digests are equal (lists of the same length)."""
    if len(a) != len(b) or not a:
        raise ValueError(f"digest lists differ in length: {len(a)} vs {len(b)}")
    return float(np.mean([x == y for x, y in zip(a, b, strict=True)]))


__all__ = ["equal_share", "world_digests"]
