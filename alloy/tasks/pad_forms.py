# SPDX-License-Identifier: Apache-2.0
"""``forms_undisturbed`` by proxy for the carriage tasks (WP15's ruling 2026-11-09 ~05:40Z, pad-trowel v4).

Their criterion is the peak horizontal force on the forms (T:707-733). Here the forms are cells, not bodies (WP6's
known gap; ``concrete_pad`` and ``pad_run.NOT_MEASURED``), so no force on them can be measured. The proxy is every
tool's lowest sole clearance over the form tops along its whole commanded path, in every world. The path covers the
set-downs, ramps, runs, lifts and the flights between, from the carriage's keys as they stand at the end of the run
(``ToolCarriage._table``, the lifts' z rewrites included). Position, yaw and pitch are linear between keys, as
``concrete_tools._carriage_kernel`` interpolates them; its servo, at 400 rad/s, lags by ~2.5 ms. The clearance is the
sole's lowest corner (pitch included) minus ``FORM_H`` at every form cell under its footprint. A minimum >= 0 is no touch
and passes; below 0 is a touch and fails. It is a geometric proxy, not a force measurement.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.tasks.concrete_pad_start import FORM_H

DT = 0.01                    # s between samples along a key segment (every key itself is sampled too)


@dataclass(frozen=True)
class Clearance:
    """The lowest sole clearance [m] over a form-top cell for one world: which tool, when and where (inf: none of its
    tools is ever over a form)."""

    min_m: float
    tool: int = -1
    t: float = math.nan
    at: tuple[float, float] = (math.nan, math.nan)

    @property
    def passed(self) -> bool:
        return self.min_m >= 0.0


def path(row: np.ndarray, n: int, dt: float = DT) -> dict[str, np.ndarray]:
    """One program row's commanded path: every key plus samples ``dt`` apart, with pos, yaw and pitch linear between
    keys."""
    k = row[:n]
    t = k["t"].astype(np.float64)
    ts = np.union1d(t, np.arange(t[0], t[-1], dt)) if n > 1 else t
    out = {"t": ts}
    for name, col in (("x", k["pos"][:, 0]), ("y", k["pos"][:, 1]), ("z", k["pos"][:, 2]), ("yaw", k["yaw"]),
                      ("pitch", k["pitch"])):
        out[name] = np.interp(ts, t, col.astype(np.float64))
    return out


def tool_clearance(p: dict[str, np.ndarray], half: Sequence[float], f: dict[str, Any]) -> tuple[float, int]:
    """The lowest clearance [m] of one tool's sole over the form cells along path ``p`` and the sample index (inf, -1
    when it is never over a form). ``half``: the tool's half length, width and thickness (body frame: x along yaw)."""
    ha, hc, hz = (float(v) for v in half)
    x, y = np.asarray(f["x"]), np.asarray(f["y"])
    form = np.asarray(f["form_mask"]) > 0.5
    cell = float(f["cell"])
    reach = math.hypot(ha, hc) + cell
    low = p["z"] - hz * np.cos(p["pitch"]) - ha * np.abs(np.sin(p["pitch"]))   # the sole's lowest corner
    for i in np.argsort(low, kind="stable"):            # lowest first: the first sample over a form is the minimum
        cx, cy = p["x"][i], p["y"][i]
        ix = slice(int(np.searchsorted(x, cx - reach)), int(np.searchsorted(x, cx + reach)) + 1)
        iy = slice(int(np.searchsorted(y, cy - reach)), int(np.searchsorted(y, cy + reach)) + 1)
        sub = form[ix, iy]
        if not sub.any():
            continue
        X, Y = np.meshgrid(x[ix] - cx, y[iy] - cy, indexing="ij")
        c, s = math.cos(p["yaw"][i]), math.sin(p["yaw"][i])
        if (sub & (np.abs(X * c + Y * s) <= ha + 1e-9) & (np.abs(Y * c - X * s) <= hc + 1e-9)).any():
            return float(low[i] - FORM_H), int(i)
    return math.inf, -1


def clearance(table: np.ndarray, n: np.ndarray, tools: int, halves: Sequence[Sequence[float]], f: dict[str, Any],
              dt: float = DT) -> list[Clearance]:
    """Per world, the lowest sole clearance over the form tops of every tool along its commanded path: ``table`` and
    ``n`` are the carriage's key table (row w * tools + k) and key counts; ``halves`` each tool's half extents."""
    worlds = table.shape[0] // tools
    out = []
    for w in range(worlds):
        best = Clearance(math.inf)
        for k in range(tools):
            p = path(table[w * tools + k], int(n[w * tools + k]), dt)
            m, i = tool_clearance(p, halves[k], f)
            if i >= 0 and m < best.min_m:
                best = Clearance(m, k, float(p["t"][i]), (float(p["x"][i]), float(p["y"][i])))
        out.append(best)
    return out


__all__ = ["DT", "Clearance", "clearance", "path", "tool_clearance"]
