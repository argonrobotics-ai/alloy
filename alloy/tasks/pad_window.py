# SPDX-License-Identifier: Apache-2.0
"""A corner window of a pad task scored as part of their whole pad (host-side numpy).

A CPU run works a window from the south-west inner corner (its south and west forms real, its north and east edges
open). Beyond the reach of the window's runs toward the open edges, its lanes leave paste a whole pad's neighbouring
lanes would work again (end spill, a reversal's roll): those cells are reset to the start, every other cell is scored
as the run left it, set into their whole pad's start. Scores: smooth_hard over their core (> 30 mm from the forms)
within the window, the radius lines along its two forms, the whole pad's flatness, and the other criteria of their
whole-pad verdict.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.tasks import concrete_metrics as M
from alloy.tasks import pad_run as PR

FIELDS = ("h", "t", "D", "a", "E", "S", "w", "q", "bl", "late", "worked", "wet_worked", "dwc")


@dataclass
class WindowScore:
    """A corner window's scores: smooth_hard over their core in the window (reset cells left out), the radius lines
    along its two forms, the whole pad's flatness with the window set in, and the whole-pad verdict's other criteria
    (all but its smooth_hard) once the layer publishes dwc and bl (None before)."""

    smooth_hard: float
    radius_profile: float
    flat_straightedge_mm: float
    flat_plane_mm: float
    criteria: dict | None = None


def window_slices(f: dict[str, Any], whole_start: dict[str, Any]) -> tuple[slice, slice]:
    """The window's cells in their whole pad's grid."""
    i0 = int(np.flatnonzero(np.isclose(whole_start["x"], f["x"][0]))[0])
    j0 = int(np.flatnonzero(np.isclose(whole_start["y"], f["y"][0]))[0])
    return slice(i0, i0 + len(f["x"])), slice(j0, j0 + len(f["y"]))


def has_water_states(g: M.PasteGrid) -> bool:
    """The layer publishes dwc and bl (WP6's paste): the trowel's and the finish's verdicts read them."""
    return g.dwc is not None and g.bl is not None


def embed(g: M.PasteGrid, f: dict[str, Any], whole_start: dict[str, Any]) -> M.PasteGrid:
    """A window's end grid set into their whole pad's start: the window worked, the rest as it started. A field the
    window lacks is dropped from the whole pad too (never filled from the start)."""
    full = PR.start_grid(whole_start)
    sl = window_slices(f, whole_start)
    for name in (*FIELDS, "bleed_left"):
        v = getattr(g, name)
        if v is None:
            setattr(full, name, None)
            continue
        base = getattr(full, name)
        arr = np.zeros(full.h.shape, np.asarray(v).dtype) if base is None else np.array(base, np.asarray(v).dtype)
        arr[sl] = v
        setattr(full, name, arr)
    return full


def window_radius(full: M.PasteGrid, f: dict[str, Any], whole_start: dict[str, Any]) -> float:
    """Their radius lines (T:281-300, ``_arc_ok`` per line) along the window's south and west forms only."""
    d = M.edge_distance(full)
    i_in = np.flatnonzero(np.isin(whole_start["x"], f["x"]) & full.paste_mask.any(axis=1))
    j_in = np.flatnonzero(np.isin(whole_start["y"], f["y"]) & full.paste_mask.any(axis=0))
    j0, i0 = j_in.min(), i_in.min()
    ok = n = 0
    for i in i_in:                                                     # south side: a line along y per column
        sel = full.paste_mask[i] & ((np.arange(full.h.shape[1]) - j0 + 0.5) * full.cell < M.EDGE_BAND_M)
        n, ok = n + 1, ok + M._arc_ok(full.h[i][sel], d[i][sel], full.cell)
    for j in j_in:                                                     # west side: a line along x per row
        sel = full.paste_mask[:, j] & ((np.arange(full.h.shape[0]) - i0 + 0.5) * full.cell < M.EDGE_BAND_M)
        n, ok = n + 1, ok + M._arc_ok(full.h[:, j][sel], d[:, j][sel], full.cell)
    return ok / max(n, 1)


def window_score(g: M.PasteGrid, f: dict[str, Any], whole_start: dict[str, Any], reset: np.ndarray,
                 pad: M.PadScalars, verdict: Callable[..., Any] = M.pad_trowel) -> WindowScore:
    """The window's end grid with the ``reset`` cells back at the start, set into their whole pad: smooth_hard over
    their core within the window, the radius lines along its forms, flatness, and the verdict's other criteria."""
    g, start = dataclasses.replace(g), PR.start_grid(f)
    for name in FIELDS:
        v, v0 = getattr(g, name), getattr(start, name)
        if v is not None and v0 is not None:
            setattr(g, name, np.where(reset, np.asarray(v0, np.asarray(v).dtype), v))
    full = embed(g, f, whole_start)
    core = M.pad_core(full)[window_slices(f, whole_start)] & ~reset
    smooth = float(((g.D >= M.DENS_MIN) & (g.a <= M.TROWEL_TEX))[core].mean())
    _, _, flat = M._flat_criteria(full)
    crit = None
    if has_water_states(full):
        crit = PR.verdict_dict(verdict(full, pad))["criteria"]
        crit.pop("smooth_hard")            # their share over the whole pad, which a window leaves mostly unworked
    return WindowScore(smooth, window_radius(full, f, whole_start), flat[0].value, flat[1].value, crit)


def window_passed(w: WindowScore, tools_off: int) -> bool:
    """A window passes on its own smooth_hard, radius lines and flatness (their pad-trowel thresholds), the tools
    back, and every other criterion of their whole-pad verdict once the layer publishes dwc and bl."""
    own = (w.smooth_hard >= M.TEX_SHARE and w.radius_profile >= M.EDGE_SHARE_TROWEL
           and w.flat_straightedge_mm <= M.GAP_MM and w.flat_plane_mm <= M.PLANE_MM and tools_off == 0)
    return own and all(c["passed"] is not False for c in (w.criteria or {}).values())     # None: not judged
