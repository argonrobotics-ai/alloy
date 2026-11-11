# SPDX-License-Identifier: Apache-2.0
"""Twin-seam bookkeeping for pre-cut surfaces (pure numpy, no bpy).

A twin pair is sealed (welded for display) while its gap is under WELD_DISTANCE_M and it has never opened before.
The first opening frame is found from the whole recording, so the answer for a frame does not depend on which
frames a farm VM rendered before it: once a cut opens, the pieces stay apart even if they touch again.
"""

from __future__ import annotations

import numpy as np

WELD_DISTANCE_M = 2e-5  # twins closer than 0.02 mm are an intact seam (measured recorder gaps: ~2.6 um)
SCAN_CHUNK = 64


def first_break(frames: np.ndarray, twins: np.ndarray, tol: float = WELD_DISTANCE_M) -> np.ndarray:
    """Per twin pair, the first recorded frame whose gap exceeds tol (inf if the seam never opens)."""
    out = np.full(len(twins), np.inf)
    if len(twins) == 0:
        return out
    for k0 in range(0, frames.shape[0], SCAN_CHUNK):
        block = np.asarray(frames[k0:k0 + SCAN_CHUNK], np.float32)
        opened = np.linalg.norm(block[:, twins[:, 0]] - block[:, twins[:, 1]], axis=2) > tol
        hit = opened.any(axis=0) & np.isinf(out)
        out[hit] = k0 + np.argmax(opened[:, hit], axis=0)
        if not np.isinf(out).any():
            break
    return out


def sealed_flags(pos: np.ndarray, twins: np.ndarray, broken_at: np.ndarray, fr: float,
                 tol: float = WELD_DISTANCE_M) -> np.ndarray:
    """1.0 on both vertices of every pair that is still sealed at fractional frame fr, else 0.0."""
    flag = np.zeros(len(pos), np.float32)
    if len(twins):
        gap = np.linalg.norm(pos[twins[:, 0]] - pos[twins[:, 1]], axis=1)
        sealed = (gap < tol) & (fr < broken_at)
        flag[twins[sealed].ravel()] = 1.0
    return flag
