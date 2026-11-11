# SPDX-License-Identifier: Apache-2.0
"""Success checks of the painting proof tasks, ported from Gym-Anything's ``painting-tasks-v1`` contract.

Every check reads per-world numpy fields from the surface-flow plug-in (``depth()``, ``read()``; until the B2 process
states land, the dry film thickness is ``h·φ0`` and ``lap`` / ``age`` are zero) and returns a :class:`Check`.

What a verdict claims (audit AB-gym-6, 2026-11-09; the rule of :mod:`alloy.tasks.rinse_score`): a check is True or
False only when it is read from the simulation's state. A Gym-Anything criterion that a port cannot fail (no marks
are tracked, the tool is never let go, the scene has nothing to protect) is None, not judged, and its note says why;
a check that reads another quantity or a smaller region than Gym-Anything's says so in its note. A task succeeds when
every judged check passes; the verdict says how many of Gym-Anything's criteria that is ("k of n"), and how the
port's scene differs from theirs (Gym-Anything: score 1.0 on success, else 0.9 × progress).

Checks and limits:
- roll-strip: hide >= 0.98 of the strip (ΔE*ab <= 1, :mod:`alloy.tasks.paint_optics`); no runs (wet <= 250 µm, the
  ASTM D4400 multinotch sag range, and sag travel <= 1 mm); laps <= 1 cm²; uniform (ΔE <= 1 over 50 mm patches);
  protected <= 0.5 cm²; the roller back in the tray (z < 0.12 m);
- the roller's load (roll-strip's first half): >= 50 g on the nap; >= 0.8 of the store bins above 0.2 of capacity;
  held above the tray (0.10 < z < 0.6 m); no drip for 10 s;
- brush-stroke: >= 0.95 of the S template's core at >= 5 µm dry; <= 0.5 cm² of paint (> 3 µm) beyond 1 mm outside
  the outline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from alloy.tasks.paint_optics import Optics, delta_e
from alloy.tasks.rinse_score import partly

# Gym-Anything painting-tasks-v1 limits (their params.EVAL and task.py; sources in docs/notes/soft-3-painting.md)
HIDE_FRAC = 0.98
RUN_H = 250e-6              # m wet
SAG_MOVE = 1e-3             # m
LAP_AREA = 1e-4             # m²
UNIFORM_DE = 1.0
PATCH = 0.05                # m
PATCH_COVER = 0.8
PROTECTED_AREA = 0.5e-4     # m²
PAINTED_DFT = 1e-6          # m: a protected cell counts as painted above this
LOAD_G = 50.0
SPREAD = 0.8
SPREAD_OF_CAP = 0.2
HOLD_S = 10.0
TRAY_Z = (0.10, 0.6)        # m above the tray: held
TRAY_REST_Z = 0.12          # m: back in the tray
STROKE_DFT = 5e-6           # m
STROKE_COVER = 0.95
OVERRUN = 1e-3              # m beyond the outline
OVERRUN_DFT = 3e-6          # m
OVERRUN_AREA = 0.5e-4       # m²
SPRAY_DFT = (25e-6, 50e-6)  # m: the can's 1-2 mil dry coat (Gym-Anything's spray_dft)
STENCIL_FRAC = 0.90
GAUGE = 5e-3                # m: a DFT gauge's probe (SSPC-PA 2 spot readings)
BLEED = 0.5e-3              # m beyond the stencil's aperture (at least one cell)
BLEED_DFT = 3e-6            # m
BLEED_FRAC = 0.02           # of the target's cells
OVERSPRAY = 0.05            # of the emitted solids
SINK_FILM = 100e-6          # m: paint lost into a sink counts as its volume spread this thick (Gym-Anything)
PIGMENT_SHARE = 0.95        # the wanted paint's share of a covered cell's dry film
LEDGER_REL = 1e-6           # ledger drift relative to the paint put in (set + supplied)
NO_MARKS = ("not judged: the port tracks no marks on wet paint (the tool's geoms do not touch the film and no robot "
            "link is checked against it); Gym-Anything: marred cells <= 0.5 cm2")
NO_RELEASE = ("the tool's rest position is judged; its release is not (the port's tool is fixed in the hand or "
              "moved by the script, and is never let go); Gym-Anything: in the tray and released")
ONE_PAINT = ("the film's thickness is judged; the wanted paint's share of it is not (the port has one paint, so the "
             "share is 1 by construction); Gym-Anything: two paints on the palette, share >= 0.95")
# the ports' own checks (their paint books; Gym-Anything calls an open ledger a backend fault, not a criterion)
BOOKS = ("ledger", "solids_ledger", "tool_books", "books", "pigment_books", "supply", "solids", "dust")


@dataclass(frozen=True)
class Check:
    """One check of a verdict. ``ok`` is None when the port does not judge Gym-Anything's criterion, or judges only
    a part of it and that part passed; ``note`` says which part and why, or how a judged check differs from theirs."""

    name: str
    value: float
    limit: float
    ok: bool | None
    note: str = ""

    def row(self) -> dict:
        row = {"value": round(float(self.value), 6), "limit": self.limit,
               "ok": None if self.ok is None else bool(self.ok)}
        return {**row, "note": self.note} if self.note else row

    def differs(self, note: str) -> Check:
        """The same check, judged by another measure than Gym-Anything's: ``note`` says how."""
        return replace(self, note=note)

    def in_part(self, note: str) -> Check:
        """The same check with a part of Gym-Anything's criterion not judged (``note`` says which): a failure of the
        judged part still fails it, a pass of the judged part alone is not a pass (None)."""
        return replace(self, ok=partly(self.ok, None), note=note)


def verdict(checks: Sequence[Check], extra: Sequence[str] = BOOKS, scene: str = "") -> dict:
    """The verdict of ``checks``: ``succeeded`` when every judged check passes and at least one of Gym-Anything's
    criteria is judged. ``extra`` names the port's own checks (books and ledgers, which Gym-Anything does not ask):
    they count for success, not in ``judged`` ("k of n" of Gym-Anything's criteria). ``not_judged`` gives the reason
    for each criterion left out, and ``scene`` how the port's scene differs from Gym-Anything's.
    Example: ``verdict([Check("hide", 1.0, 0.98, True), untouched(None, m, a)])["judged"]`` is ``"1 of 2"``."""
    ga = [c for c in checks if c.name not in extra]
    judged = [c.name for c in ga if c.ok is not None]
    missing = {c.name: c.note for c in ga if c.ok is None}
    passed = all(bool(c.ok) for c in checks if c.ok is not None)
    summary = (f"{len(judged)} of Gym-Anything's {len(ga)} criteria judged"
               + (f"; not judged: {', '.join(missing)}" if missing else "") + (f". Scene: {scene}" if scene else ""))
    return {"checks": {c.name: c.row() for c in checks}, "succeeded": bool(judged) and passed,
            "judged": f"{len(judged)} of {len(ga)}", "not_judged": missing, "scene": scene, "summary": summary}


# --- wall (roll-strip) ----------------------------------------------------------------------------------------------


def hide(dft: np.ndarray, mask: np.ndarray, optics: Optics, r_g: np.ndarray) -> Check:
    """Share of the strip's cells (``mask`` [nx, ny]) whose film (``dft`` [nx, ny] m) hides the substrate ``r_g``
    ([nx, ny, 3] or [3])."""
    rg = np.broadcast_to(np.asarray(r_g, dtype=np.float64), dft.shape + (3,))
    frac = float(optics.hidden(1e6 * dft[mask], rg[mask]).mean()) if mask.any() else 0.0
    return Check("hide", frac, HIDE_FRAC, frac >= HIDE_FRAC)


def no_runs(h_max: float, travel: float, settled: bool = True) -> Check:
    """Gym-Anything's run check: the wet film at most ``RUN_H`` unless the film has settled, and once settled no part
    of it travelled more than ``SAG_MOVE`` during the settle. The value is the travel when settled, else the depth."""
    ok = (h_max <= RUN_H or settled) and (travel <= SAG_MOVE if settled else True)
    return Check("no_runs", travel if settled else h_max, SAG_MOVE if settled else RUN_H, ok)


def settle_travel(travel_start: np.ndarray, travel_end: np.ndarray, mask: np.ndarray) -> float:
    """The furthest any cell's film travelled [m] between two readings of the plug-in's per-cell ``travel`` (its
    accumulated |film velocity|·dt while wet), over ``mask``: Gym-Anything's per-cell run tracker."""
    d = (np.asarray(travel_end, dtype=np.float64) - np.asarray(travel_start, dtype=np.float64))[mask]
    return float(d.max()) if d.size else 0.0


def laps(lap: np.ndarray, mask: np.ndarray, area: float) -> Check:
    a = float(((lap > 0) & mask).sum() * area)
    return Check("no_laps", a, LAP_AREA, a <= LAP_AREA)


def _patch_starts(lo: int, hi: int, n: int) -> list[int]:
    """First indices of n-cell patches tiling lo..hi from lo, plus one flush with hi when the tiling falls short."""
    starts = list(range(lo, hi + 1, n))
    if hi - lo + 1 >= n and starts[-1] + n - 1 != hi:
        starts = [k for k in starts if k + n - 1 <= hi] + [hi - n + 1]
    return starts


def uniform(dft: np.ndarray, mask: np.ndarray, optics: Optics, r_g: np.ndarray, cell: float) -> Check:
    """Largest ΔE*ab of a 50 mm patch's mean colour from the median patch, over patches at least 80 % in the strip.
    The patches tile from the strip's first corner, plus a row and a column flush with its far edges, so a band along
    any edge is measured."""
    n = max(1, int(round(PATCH / cell)))
    rg = np.broadcast_to(np.asarray(r_g, dtype=np.float64), dft.shape + (3,))
    refl = optics.reflectance(1e6 * dft, rg)
    ii, jj = np.nonzero(mask)
    if ii.size == 0:
        return Check("uniform", np.inf, UNIFORM_DE, False)
    cols = []
    for i in _patch_starts(int(ii.min()), int(ii.max()), n):
        for j in _patch_starts(int(jj.min()), int(jj.max()), n):
            m = mask[i:i + n, j:j + n]
            if m.size == n * n and m.mean() >= PATCH_COVER:
                cols.append(refl[i:i + n, j:j + n][m].mean(axis=0))
    if not cols:
        return Check("uniform", np.inf, UNIFORM_DE, False)
    cols = np.asarray(cols)
    worst = float(delta_e(cols, np.median(cols, axis=0)).max())
    return Check("uniform", worst, UNIFORM_DE, worst <= UNIFORM_DE)


def protected(dft: np.ndarray, protected_mask: np.ndarray, area: float, lost_volume: float = 0.0) -> Check:
    """Painted area on the protected bands, plus paint lost into sinks (spilled off the chart, dripped onto other
    objects) as its volume spread ``SINK_FILM`` thick."""
    a = float(((dft > PAINTED_DFT) & protected_mask).sum() * area) + max(lost_volume, 0.0) / SINK_FILM
    return Check("protected", a, PROTECTED_AREA, a <= PROTECTED_AREA)


def untouched(marred: np.ndarray | None, mask: np.ndarray, area: float, name: str = "untouched") -> Check:
    """Wet paint marred by a robot link (touched by anything but the tool), over ``mask``. ``marred`` None: the run
    tracked no marks, so the criterion is not judged (``NO_MARKS``), never passed on an empty array."""
    if marred is None:
        return Check(name, 0.0, PROTECTED_AREA, None, NO_MARKS)
    a = float((np.asarray(marred, dtype=bool) & mask).sum() * area)
    return Check(name, a, PROTECTED_AREA, a <= PROTECTED_AREA)


def in_tray(pos_in_tray: np.ndarray, half_size, released: bool | None) -> Check:
    """The roller's cover centre, in the tray's frame, inside its footprint, below ``TRAY_REST_Z``, and let go.
    ``released`` None: the run cannot let the tool go, so only the position is judged (``NO_RELEASE``)."""
    x, y, z = (float(v) for v in pos_in_tray)
    inside = abs(x) < half_size[0] and abs(y) < half_size[1] and z < TRAY_REST_Z
    if released is None:
        return Check("roller_in_tray", z, TRAY_REST_Z, inside).in_part(NO_RELEASE)
    return Check("roller_in_tray", z, TRAY_REST_Z, inside and released)


# --- the roller's load (store bins) --------------------------------------------------------------------------------


def roller_load(load_m3: np.ndarray, capacity_m3: np.ndarray, density: float, saturation: float = 1.0) -> list[Check]:
    """``load`` and ``spread`` of one roller's store bins (``tool_load`` / ``tool_capacity`` of one world and tool).
    A bin counts as loaded above 0.2 of its saturation load, Gym-Anything's ``0.2·SAT·C0``: pass ``saturation`` = the
    saturated share of the capacity when ``capacity`` is a pore volume (0.154 for their nap), 1 when it is already
    the saturation load."""
    grams = 1e3 * density * float(np.sum(load_m3))
    spread = float(np.mean(np.asarray(load_m3) > SPREAD_OF_CAP * saturation * np.asarray(capacity_m3)))
    return [Check("load", grams, LOAD_G, grams >= LOAD_G), Check("spread", spread, SPREAD, spread >= SPREAD)]


def held_above_tray(pos_in_tray: np.ndarray, half_size) -> Check:
    x, y, z = (float(v) for v in pos_in_tray)
    ok = abs(x) < half_size[0] and abs(y) < half_size[1] and TRAY_Z[0] < z < TRAY_Z[1]
    return Check("held_above_tray", z, TRAY_Z[0], ok)


def no_drip(dripped_m3: np.ndarray, times: np.ndarray, drop: float = 0.0) -> Check:
    """No drop falls over the last ``HOLD_S`` seconds of the series (``times`` ascending): ``tool_dripped`` rises by
    no more than ``drop`` [m³], one drop's volume (Gym-Anything counts drops; a store's drip decays without end).
    With ``drop`` > 0 the check is looser than Gym-Anything's, which restarts the 10 s on any drop: its note says so."""
    t = np.asarray(times, dtype=np.float64)
    d = np.asarray(dripped_m3, dtype=np.float64)
    first = max(int(np.searchsorted(t, t[-1] - HOLD_S, side="right")) - 1, 0)   # the sample that opens the window
    quiet = float(t[-1] - t[first])
    ok = quiet >= HOLD_S - 1e-9 and float(d[first:].max() - d[first:].min()) <= drop
    note = ("judged, looser than Gym-Anything: a rise of up to one drop's volume in the 10 s passes; theirs restarts "
            "the 10 s on any drop that lands or is in flight") if drop > 0.0 else ""
    return Check("no_drip_10s", quiet, HOLD_S, ok, note)


# --- paper (brush-stroke) ------------------------------------------------------------------------------------------


def s_curve(n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Gym-Anything's pencilled S template in the paper's frame: points [n, 2] (m) and half widths [n] (m):
    centre (-0.10 + 0.20 t, 0.035 sin 2πt), width 2 mm -> 6 mm -> 2 mm (2 + 4 sin πt mm)."""
    t = np.linspace(0.0, 1.0, n)
    pts = np.stack([-0.10 + 0.20 * t, 0.035 * np.sin(2.0 * np.pi * t)], axis=1)
    return pts, 0.5 * (2e-3 + 4e-3 * np.sin(np.pi * t))


def template_masks(centres_xy: np.ndarray, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """(core, outside) masks over cell centres [nx, ny, 2] (paper frame): the core must be painted (within half width
    - 0.5 mm, at least 0.5 mm), outside is beyond half width + 1 mm. One template point at a time (memory O(cells))."""
    pts, half = s_curve(n)
    c = centres_xy.reshape(-1, 2).astype(np.float64)
    best = np.full(len(c), np.inf)
    k = np.zeros(len(c), dtype=np.int64)
    for i, p in enumerate(pts):
        d2 = (c[:, 0] - p[0]) ** 2 + (c[:, 1] - p[1]) ** 2
        closer = d2 < best
        best[closer], k[closer] = d2[closer], i
    d = np.sqrt(best)
    core = d <= np.maximum(half[k] - 0.5e-3, 0.5e-3)
    outside = d > half[k] + OVERRUN
    shape = centres_xy.shape[:2]
    return core.reshape(shape), outside.reshape(shape)


def stroke(dft: np.ndarray, core: np.ndarray, outside: np.ndarray, area: float,
           share: np.ndarray | None = None, dft_min: float = STROKE_DFT) -> list[Check]:
    """Coverage of the core at ``dft_min`` (default ``STROKE_DFT``; a swatch judges hide at 15 µm) with the wanted
    paint's ``share`` of the dry film at least ``PIGMENT_SHARE``, and the paint beyond the outline. ``share`` None:
    one paint on the paper, so only the thickness part of coverage is judged (``ONE_PAINT``)."""
    good = (dft >= dft_min) & (True if share is None else np.asarray(share) >= PIGMENT_SHARE)
    cover = float(good[core].mean()) if core.any() else 0.0
    over = float(((dft > OVERRUN_DFT) & outside).sum() * area)
    coverage = Check("coverage", cover, STROKE_COVER, cover >= STROKE_COVER)
    return [coverage.in_part(ONE_PAINT) if share is None else coverage,
            Check("within_line", over, OVERRUN_AREA, over <= OVERRUN_AREA)]


# --- spray-stencil ----------------------------------------------------------------------------------------------------


def gauge(dft: np.ndarray, mask: np.ndarray, cell: float) -> np.ndarray:
    """The dry film as a DFT gauge reads it: the mean over its ``GAUGE`` probe inside the coated area ``mask``
    (Gym-Anything's ``_gauge``: one cell carries the spray's discreteness, not the coat)."""
    from scipy.ndimage import uniform_filter  # noqa: PLC0415

    k = max(1, int(round(GAUGE / cell)))
    m = np.asarray(mask, dtype=np.float64)
    num = uniform_filter(np.asarray(dft, dtype=np.float64) * m, k, mode="constant")
    den = uniform_filter(m, k, mode="constant")
    return np.where(den > 1e-9, num / np.maximum(den, 1e-9), 0.0)


def spray_coat(dft: np.ndarray, target: np.ndarray, cell: float) -> list[Check]:
    """``coat``: the share of the target's cells whose gauge reading lies in ``SPRAY_DFT``; ``sharp_edges``: cells
    above ``BLEED_DFT`` outside the target grown by ``BLEED`` (at least one cell), per target cell."""
    from scipy.ndimage import binary_dilation  # noqa: PLC0415

    n = int(target.sum())
    g = gauge(dft, target, cell)
    lo, hi = SPRAY_DFT
    inband = float(((g[target] >= lo) & (g[target] <= hi)).mean()) if n else 0.0
    grown = binary_dilation(target, iterations=max(1, int(round(BLEED / cell))))
    bleed = float(((np.asarray(dft) > BLEED_DFT) & ~grown).sum()) / max(1, n)
    return [Check("coat", inband, STENCIL_FRAC, inband >= STENCIL_FRAC),
            Check("sharp_edges", bleed, BLEED_FRAC, bleed <= BLEED_FRAC)]


def overspray(outside_solids: float, emitted_solids: float) -> Check:
    """Solids that left the spray box (airborne, landed outside it, on robot links) per emitted solids."""
    frac = outside_solids / emitted_solids if emitted_solids > 0 else (0.0 if outside_solids <= 0 else 1.0)
    return Check("overspray", frac, OVERSPRAY, frac <= OVERSPRAY)


def chart_field(read: dict, key: str, shape: tuple[int, int], world: int = 0) -> np.ndarray | None:
    """Chart 0's slice of a per-cell ``read()`` field [W, C] as [nx, ny] (cells run i-major), or None if absent."""
    if key not in read:
        return None
    return np.asarray(read[key])[world, : shape[0] * shape[1]].reshape(shape)


# --- books -----------------------------------------------------------------------------------------------------------


def solids_ledger(read: dict, set_solids: float = 0.0, world: int = 0, rel: float = LEDGER_REL) -> Check:
    """The film's solids ledger closes for ``world``: held + carried out - supplied equals the solids the task set
    directly (``set_species``) [m³], within ``rel`` of the solids put in. It fails when the film carries no solids
    (the paint's ``species``, the film's first)."""
    if "species_ledger" not in read:
        return Check("solids_ledger", float("inf"), rel, False)
    led = float(np.asarray(read["species_ledger"])[world, 0])
    put_in = set_solids + float(np.asarray(read["species_supplied"])[world, :, 0].sum())
    drift = abs(led - set_solids) / max(put_in, 1e-30)
    return Check("solids_ledger", drift, rel, drift <= rel)


def dry_summary(dry: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """The dry film over ``mask`` [µm]: 5th, 50th and 95th percentiles (for reports and value tests)."""
    v = 1e6 * np.asarray(dry, dtype=np.float64)[mask]
    if v.size == 0:
        return {"p5_um": float("nan"), "p50_um": float("nan"), "p95_um": float("nan")}
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    return {"p5_um": round(float(p5), 4), "p50_um": round(float(p50), 4), "p95_um": round(float(p95), 4)}


def ledger(read: dict, set_volume: float = 0.0, world: int = 0, rel: float = LEDGER_REL) -> Check:
    """The plug-in's volume ledger closes for ``world``: film + spilled + evaporated + drained + clipped - supplied
    (+ what tools took) equals the film the task set directly (``set_depth``), within ``rel`` of the paint put in:
    set, supplied, and laid by tools. The value is the relative drift."""
    put_in = set_volume + _world(read, "supplied_volume", world) + max(-_world(read, "to_tools_volume", world), 0.0)
    drift = abs(_world(read, "ledger", world) - set_volume) / max(put_in, 1e-15)
    return Check("ledger", drift, rel, drift <= rel)


def _world(read: dict, key: str, world: int) -> float:
    """``read[key]`` [W] of ``world`` (0 when the film has no such column)."""
    return float(np.asarray(read[key]).reshape(-1)[world]) if key in read else 0.0


__all__ = ["BOOKS", "Check", "NO_MARKS", "NO_RELEASE", "ONE_PAINT", "chart_field", "dry_summary", "gauge",
           "held_above_tray", "hide", "in_tray", "laps", "ledger", "no_drip", "no_runs", "overspray", "protected",
           "roller_load", "s_curve", "settle_travel", "solids_ledger", "spray_coat", "stroke", "template_masks",
           "uniform", "untouched", "verdict"]
