# SPDX-License-Identifier: Apache-2.0
"""Small helpers shared by the Panda task checkers (E20-E22): time windows and episode ends. Numpy only."""
from __future__ import annotations

import numpy as np

from bench.newenv.checkers import common as c

COAST_S = 0.5


def coast_rows(t: np.ndarray) -> int:
    """Number of earlier rows still counted as "just touched" (COAST_S of the phase's clock)."""
    return int(round(COAST_S / float(np.median(np.diff(t))))) if len(t) > 1 else 0


def recent(flag: np.ndarray, rows: int) -> np.ndarray:
    """(T, N) bool: ``flag`` was True at this row or within the previous ``rows`` rows."""
    total = np.cumsum(np.concatenate([np.zeros((1,) + flag.shape[1:]), flag.astype(float)]), axis=0)
    first = np.maximum(np.arange(1, len(flag) + 1) - rows - 1, 0)
    return (total[1:] - total[first]) > 0


def episode_end(p: c.Phase) -> np.ndarray:
    """(N,) index of the last row of each world's first episode (a done row already shows the next start)."""
    ends = np.full(p.num_envs, len(p.t) - 1)
    for w in range(p.num_envs):
        hits = np.flatnonzero(p.done[:, w])
        if len(hits):
            ends[w] = max(hits[0] - 1, 0)
    return ends
