# SPDX-License-Identifier: Apache-2.0
"""cut-in-trim's verdict: Gym-Anything's painting-tasks-v1 checks read from the surface-flow film (``read()``).

Checks (``PT/task.py:17-18, 300-370`` with the painting-tasks-v1 contract, ``PT/scoring.py``; limits in
:mod:`alloy.tasks.paint_metrics` unless named here):
- ``hide``: the first coat (:meth:`TrimLayout.first_coat`, Gym-Anything's patchy 43 µm) plus the new dry film within
  ΔE*ab 1 of the paint's own colour on >= 95 % of the bands (bands: 95 %, not the strip's 98 %; ``PT/task.py:303``);
- ``no_runs``: the v1 rule (``PT/task.py:355-361``): dry paint the film's flow carried into a cell, more than
  ``RUN_FLOW`` 50 µm, on at most ``RUN_AREA`` 1 cm² of the bands, read after the settle. The flowed paint per cell is
  its dry film less the solids laid on it net of those taken off it (a fresh film: nothing there at the start);
- ``no_laps``: fresh paint over film past its wet edge on at most 1 cm² of the bands (the process stage's ``laps``);
- ``protected``: paint (> 1 µm dry) on the trim's or the plate's footprint, plus the brush's drips near the wall as
  Gym-Anything's sinks count them (their volume spread 100 µm thick), at most 0.5 cm²;
- ``untouched``: not judged (None): no robot here, and no marks on the wet film are tracked;
- ``books``: the film's, the brushes' and the pail's ledgers close to 1e-12 of the paint picked up;
  ``solids_ledger``: the dry film holds the solids laid, to 1e-12 (the port's own checks, not Gym-Anything criteria).

Of Gym-Anything's five criteria four are judged (audit AB-gym-6; the verdict's notes and ``SCENE`` say the same):
``hide`` and ``no_laps`` as theirs, ``no_runs`` by the port's own, stricter measure of flowed paint, and
``protected`` on a smaller region (the trim's and the plate's footprints inside the two cropped charts, plus drips
near the wall).
"""

from __future__ import annotations

import numpy as np

from alloy.tasks import paint_metrics as pm
from alloy.tasks import trim_sash as ts
from alloy.tasks.paint_materials import COLOURS, optics, paint
from alloy.tasks.paint_optics import srgb8_to_linear
from alloy.tasks.trim_layout import PAIL, PATCHY, Crop, TrimLayout

HIDE_BANDS = 0.95               # Gym-Anything's hide share for bands (PT/task.py:303, scoring.py:83)
RUN_FLOW = 50e-6                # m dry carried in by flow (PT/task.py:86-87, run_flow)
RUN_AREA = 1e-4                 # m² (run_area)
OWN_FLOW = ("judged by the port's own measure of flowed paint, stricter than Gym-Anything's: a cell's dry film less "
            "the solids the brushes laid on it, net of those they took, so paint that crept in and was brushed back "
            "out still counts in full (theirs lowers it by the tool's take); equal when no brush takes film back")
IN_CROPS = ("judged on a smaller region: paint on the trim's and the plate's footprints inside the two cropped "
            "charts, plus the brushes' drips near the wall; Gym-Anything reads every trim face, sink and drop in "
            "the room")
SCENE = ("smaller and easier than Gym-Anything's: two cropped wall charts (the casing band and the outlet ring, each "
         "with 40 mm round it) instead of the whole wall; one free-moving sash brush per chart on a script (theirs: "
         "one brush held by a mobile FR3); a settle of 60 s by default (theirs: until the film sets, up to 1800 s); "
         "their seed-0 wall only, the seed changes nothing")
LEDGER_REL = 1e-12


def crop_centres(crop: Crop) -> np.ndarray:
    """The crop's cell centres in its box frame [nx, ny, 2] (as the plug-in's ``cell_centres``, without a build)."""
    nx, ny = crop.cells
    x = (np.arange(nx) + 0.5) * crop.cell - 0.5 * nx * crop.cell
    y = (np.arange(ny) + 0.5) * crop.cell - 0.5 * ny * crop.cell
    return np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1)


def chart_slices(layout: TrimLayout) -> list[tuple[slice, tuple[int, int]]]:
    """Each crop's cells in the film's per-cell fields [W, C, ...] (charts in order, cells i-major) and its shape."""
    out, start = [], 0
    for c in layout.crops():
        nx, ny = c.cells
        out.append((slice(start, start + nx * ny), (nx, ny)))
        start += nx * ny
    return out


def field(read: dict, key: str, layout: TrimLayout, chart: int, world: int = 0, k: int | None = None) -> np.ndarray:
    """Chart ``chart``'s slice of a per-cell field as [nx, ny] (species fields: species ``k``)."""
    sl, shape = chart_slices(layout)[chart]
    a = np.asarray(read[key])[world, sl]
    return (a if k is None else a[:, k]).reshape(shape)


def total_to_hide(layout: TrimLayout) -> float:
    """The least dry film [m] (first coat + new) that hides the primer, by bisection on the optics."""
    opt, rg = optics(layout.paint, layout.colour), srgb8_to_linear(COLOURS[layout.substrate])
    lo, hi = 0.0, 500.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if bool(opt.hidden(np.array([mid]), rg[None])[0]) else (mid, hi)
    return hi * 1e-6


def wet_to_hide(layout: TrimLayout) -> float:
    """The wet coat [m] that hides where Gym-Anything's first coat is thinnest (its -15 %)."""
    return (total_to_hide(layout) - (1.0 - PATCHY) * layout.first_coat_dft) / paint(layout.paint).volume_solids


def flowed(read: dict, layout: TrimLayout, chart: int, world: int = 0) -> np.ndarray:
    """Dry paint [m] the film's flow carried into each cell: its solids less those laid on it net of those taken."""
    area = layout.cell * layout.cell
    net = field(read, "species_supplied", layout, chart, world, 0) - field(read, "species_out", layout, chart, world, 0)
    return np.maximum(field(read, "dft", layout, chart, world) - net / area, 0.0)


def evaluate(layout: TrimLayout, read: dict, near_wall_drips: float = 0.0, world: int = 0) -> dict:
    """The cut-in-trim verdict of one world from the film's ``read()`` after the settle; ``near_wall_drips`` [m³]: what
    the brushes dripped within 5 cm of the wall (it lands on the trim below)."""
    opt, area = optics(layout.paint, layout.colour), layout.cell * layout.cell
    r_g = srgb8_to_linear(COLOURS[layout.substrate])
    hid, tgt, prot, new, laps, flow_in, wet = [], [], [], [], [], [], []
    for k, crop in enumerate(layout.crops()):
        xz = crop.world_xz(crop_centres(crop))
        m = layout.masks(crop, xz)
        t = m["target"]
        d = field(read, "dft", layout, k, world)
        rg = np.broadcast_to(r_g, (int(t.sum()), 3))
        hid.append(opt.hidden(1e6 * (layout.first_coat(xz)[t] + d[t]), rg))
        tgt.append(t.ravel())
        prot.append(m["protected"].ravel())
        new.append(d.ravel())
        laps.append(field(read, "laps", layout, k, world).ravel())
        flow_in.append(flowed(read, layout, k, world)[t])
        wet.append(field(read, "h", layout, k, world)[t])
    frac = float(np.concatenate(hid).mean())
    target, dry = np.concatenate(tgt), np.concatenate(new)
    run_a = float((np.concatenate(flow_in) > RUN_FLOW).sum()) * area
    checks = [pm.Check("hide", frac, HIDE_BANDS, frac >= HIDE_BANDS),
              pm.Check("no_runs", run_a, RUN_AREA, run_a <= RUN_AREA, OWN_FLOW),
              pm.laps(np.concatenate(laps), target, area),
              pm.protected(dry, np.concatenate(prot), area, lost_volume=near_wall_drips).differs(IN_CROPS),
              pm.untouched(None, target, area),
              ts.brush_books(read, PAIL["fill"], world, rel=LEDGER_REL),
              pm.solids_ledger(read, world=world, rel=LEDGER_REL)]  # fmt: skip
    return {**pm.verdict(checks, scene=SCENE),
            "dry_bands": pm.dry_summary(dry[target], np.ones(int(target.sum()), bool)),
            "wet_max_um": round(float(np.concatenate(wet).max()) * 1e6, 2),
            "flowed_max_um": round(float(np.concatenate(flow_in).max()) * 1e6, 2)}


__all__ = ["HIDE_BANDS", "LEDGER_REL", "RUN_AREA", "RUN_FLOW", "chart_slices", "crop_centres", "evaluate", "field",
           "flowed", "total_to_hide", "wet_to_hide"]
