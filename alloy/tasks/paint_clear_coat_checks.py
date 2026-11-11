# SPDX-License-Identifier: Apache-2.0
"""clear-coat's books and verdict (:mod:`alloy.tasks.paint_clear_coat`), ported from Gym-Anything's painting task set
(``gym-anything-robotics/benchmarks/roboworld/environments/painting``, read-only in [internal note]; file:line):

- the task and its text: ``layout.py:121-123``; the lacquer: ``params.py:70-71``; the aerosol: ``params.py:196-204``;
- checks: ``task.py:422-433`` (coat 25-50 µm on >= 95 % of the print's 5 mm gauge readings, ratio <= 2, wet <= 250
  µm at declaration), ``task.py:515-548`` (overspray: solids outside the box per solids emitted <= 5 %),
  ``task.py:448-457`` (untouched); the gauge ``task.py:105-117``;
- the score: 1 on success, else 0.9 x progress, ``scoring.py:20-26, 91, 110, 117-177``: cover from each gauge
  reading's band credit min(g/25 µm, (100 µm - g)/50 µm) with 5 % forgiven, times (0.8 + 0.2 contain)(0.8 + 0.2
  finish), less pooling (wet above 250 µm) and marks per the area to paint at the start;
- the mist: dried in the air and dropped on the first surface below, drifting σ = 0.25 x its fall
  (``params.py:196-204``, ``aerosol.py:276-321``; SOFT-3's rule, :func:`alloy.tasks.paint_spray_stencil.period_dust`).
  Gym-Anything drops each parcel from where it slowed or aged out (3 s); here, as SOFT-3's, it falls from the nozzle.
  The dust's books (:func:`dust_books`): DC5's mist x φ0 = dust on the chart + dust on the box + mist that left the
  box, with the chart's share an analytic integral, independent of the dust laid.

Escape: Gym-Anything drops slow or old parcels as dust on the first surface below (``aerosol.py:280-282, 288-318``).
Only parcels still in flight at the check (``task.py:543``) and dust with no surface below (``aerosol.py:310-311``)
count as outside. Their own clear-coat replay books 20.17 % outside (``report_painting.txt:439-441``); what makes that
up is not known, and it is open with SOFT-3. Here, with SOFT-3's rule, mist released below the rim and over the box
lands in the box as dust (:func:`escape_share`, :func:`mist_escape`).

What the verdict judges (audit AB-gym-6; ``SCENE`` and each check's note say the same in the result). Of
Gym-Anything's five clear-coat criteria four are judged: ``coat``, ``even``, ``no_runs`` and ``overspray`` (the
port's own count). ``untouched`` is not judged: no marks on the wet film are tracked. The print and box are theirs;
the can is a free-moving body, the settle is 10 s (theirs until the film sets, up to 1800 s) and worlds start alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.special import erf

from alloy.tasks import paint_metrics as pm
from alloy.tasks.paint_spray_stencil import DRIFT
from alloy.tasks.paint_supply import SUPPLY_REL, supply_drift

CELL = 1e-3                     # m: Gym-Anything's spray charts
PHI0 = 0.20                     # lacquer's volume solids at landing (Gym-Anything's LACQUER)
DENSITY = 1000.0                # kg/m³, lacquer at landing (Gym-Anything's LACQUER)
FLASH = 0.6                     # share of the can's liquid mass that flashes in flight (Gym-Anything's AEROSOL)
COAT_FRAC = 0.95                # gauge in band on at least 95 % of the print
EVEN = 2.0                      # gauge max/min at most 2
SOLIDS_REL = 1e-12              # the film's solids against the solids that landed and the dust laid
DUST_REL = 1e-12                # the dust's books, relative to the mist's solids
MAX_DEPTH = 3e-4                # m: the film's ceiling (SOFT-3's spray film), over the 250 µm run height
FAILED = 0.9                    # the score of a failed episode: 0.9 x progress
CAP_RUNS = "judged as Gym-Anything does, and stricter: a film clipped at the port's 300 µm depth cap also fails"
OWN_COUNT = ("judged from the port's own count: the lacquer that left the box (rays and mist outside it at the rim) "
             "per lacquer sent; Gym-Anything sums its sinks and charts outside the box, runoff, robot skins and "
             "airborne paint")
SCENE = ("the print, the box and the chart cells are Gym-Anything's; easier than theirs in that the can is a "
         "free-moving body on a script with its valve as an input (theirs: a mobile FR3 holds it and a fingertip "
         "presses the button), the verdict follows a settle of 10 s by default (theirs: until the film sets, up to "
         "1800 s), and every world starts the same (the seed changes nothing)")


def interval_share(a: np.ndarray, s: float, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """The share of a 1-D Gaussian (centre ``a``, σ ``s``) between ``lo`` and ``hi`` (broadcast)."""
    r = 1.0 / (math.sqrt(2.0) * s)
    return 0.5 * (erf((hi - a) * r) - erf((lo - a) * r))


def rect_share(x: float, y: float, s: float, hx: float, hy: float) -> float:
    """The share of a 2-D Gaussian (centre (x, y), σ ``s``) inside the rectangle |x| <= hx, |y| <= hy."""
    return float(interval_share(np.float64(x), s, -hx, hx) * interval_share(np.float64(y), s, -hy, hy))


def mist_escape(lay: Any, pos: tuple[float, float, float]) -> float:
    """The share of mist released at ``pos`` (box frame) that leaves the box: from above the rim, what has drifted
    past the opening (σ = ``DRIFT`` x the fall to the rim) when it falls to the rim; from below it, none over the box,
    all of it beside the box."""
    x, y, z = pos
    hx, hy, rim = 0.5 * lay.box[0], 0.5 * lay.box[1], lay.rim_z
    if z > rim:
        return 1.0 - rect_share(x, y, DRIFT * (z - rim), hx, hy)
    return 0.0 if abs(x) <= hx and abs(y) <= hy else 1.0


def escape_share(lay: Any, efficiency: float, rays: tuple[np.ndarray, np.ndarray],
                 pos: tuple[float, float, float]) -> float:
    """Share of what the can sends from nozzle position ``pos`` (box frame; the can upright) that leaves the box:
    its rays (``rays``: DC5's table, directions [R, 3] and fractions [R]) that cross the rim's plane outside the
    opening or leave through it, and its mist (``1 - efficiency``, :func:`mist_escape`). The rays' share is of the
    table's own total: its fractions are normalised, but their sum rounds off 1 by an ulp where the CPU's exp and
    arccos round otherwise (AVX-512), and a can whose every ray and its mist leave sends exactly all of it."""
    x, y, z = pos
    hx, hy, rim = 0.5 * lay.box[0], 0.5 * lay.box[1], lay.rim_z
    over = abs(x) <= hx and abs(y) <= hy
    dirs, wt = rays
    dz = dirs[:, 2]
    mist = (1.0 - efficiency) * mist_escape(lay, pos)
    if over and z < rim and (dz <= 0.0).all():
        return mist                                         # in the box, spraying down: walls and floor catch all
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dz != 0.0, (rim - z) / dz, np.inf)
    in_open = (np.abs(x + t * dirs[:, 0]) <= hx) & (np.abs(y + t * dirs[:, 1]) <= hy)
    if z >= rim:
        leaves = (dz >= 0.0) | ~in_open                     # up, or down past the opening
    elif over:
        leaves = (dz > 0.0) & in_open                       # up through the opening (a wall catches the rest)
    else:
        leaves = np.ones_like(dz, dtype=bool)
    return efficiency * float(wt[leaves].sum() / wt.sum()) + mist


def mist_dust(xe: np.ndarray, ye: np.ndarray, areas: np.ndarray, aims: np.ndarray, fall: np.ndarray,
              solids: np.ndarray) -> np.ndarray:
    """The mist's dry dust [W, nx, ny] (m of solids) on a chart with cell edges ``xe`` [nx + 1], ``ye`` [ny + 1] and
    cell areas ``areas`` [nx, ny] (the film's): ``solids`` [W, P] (m³) released at the nozzle's positions ``aims``
    [P, 2], falling ``fall`` [W] m and drifting σ = ``DRIFT`` x the fall, integrated over each cell. What lands past
    the chart is not on it."""
    out = np.zeros((len(fall), len(xe) - 1, len(ye) - 1))
    for w, f in enumerate(fall):
        s = DRIFT * f
        cx = np.diff(interval_share(aims[:, :1], s, -np.inf, xe[None, :]), axis=1)      # [P, nx] shares per column
        cy = np.diff(interval_share(aims[:, 1:], s, -np.inf, ye[None, :]), axis=1)
        out[w] = ((solids[w][:, None] * cx).T @ cy) / areas
    return out


@dataclass
class DustBooks:
    """The mist's solids [m³] per world: ``mist`` = ``on_chart`` + ``in_box`` + ``outside``. ``on_chart`` is what was
    laid (the cells' dust x their areas); ``in_box`` and ``outside`` come from the release points: what stays in the
    box less the chart's analytic share of it, and what leaves (:func:`mist_escape`)."""

    mist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    on_chart: np.ndarray = field(default_factory=lambda: np.zeros(0))
    in_box: np.ndarray = field(default_factory=lambda: np.zeros(0))
    outside: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def add(self, mist: np.ndarray, on_chart: np.ndarray, in_box: np.ndarray, outside: np.ndarray) -> None:
        for name, v in (("mist", mist), ("on_chart", on_chart), ("in_box", in_box), ("outside", outside)):
            old = getattr(self, name)
            setattr(self, name, (old if old.size else np.zeros_like(v)) + v)

    def closure(self, world: int, mist_solids: float) -> float:
        """Relative imbalance: the books against ``mist_solids`` (DC5's mist x φ0, m³), and their own sum."""
        m = float(self.mist[world]) if self.mist.size else 0.0
        rest = sum(float(a[world]) for a in (self.on_chart, self.in_box, self.outside) if a.size)
        return max(abs(m - mist_solids), abs(m - rest)) / max(mist_solids, 1e-30)


def dust_books(lay: Any, aims: np.ndarray, z: np.ndarray, fall: np.ndarray, solids: np.ndarray,
               laid: np.ndarray, books: DustBooks) -> None:
    """Book one batch of mist: ``solids`` [W, P] m³ released at ``aims`` [P, 2] from heights ``z`` [W] (box frame),
    falling ``fall`` [W] m; ``laid`` [W] m³ the dust put on the chart."""
    hx, hy = 0.5 * lay.chart_size[0], 0.5 * lay.chart_size[1]
    out = np.array([[mist_escape(lay, (a[0], a[1], zw)) for a in aims] for zw in z]).reshape(solids.shape)
    stays = solids * (1.0 - out)
    chart = np.array([[rect_share(a[0], a[1], DRIFT * f, hx, hy) for a in aims] for f in fall]).reshape(solids.shape)
    books.add(solids.sum(axis=1), laid, (stays * (1.0 - chart)).sum(axis=1), (solids * out).sum(axis=1))


def can_books(read: dict, world: int = 0) -> dict:
    """The can's contents [kg] by the supply's ledger: what it sent as lacquer, the flashed mass beside it, and the
    lacquer still in flight in the valve's edge (``edge_ledger`` counts it)."""
    sent = float(read["supply_sent"][world])
    in_valve = -float(read["supply_edge_ledger"][world, 0]) - sent
    landed_kg = sent * DENSITY
    return {"sent_m3": sent, "landed_kg": landed_kg, "flashed_kg": landed_kg * FLASH / (1.0 - FLASH),
            "used_kg": landed_kg / (1.0 - FLASH), "in_valve_m3": in_valve}


def coat_checks(dry: np.ndarray, mask: np.ndarray) -> list[pm.Check]:
    """``coat``: the share of the print's cells (``mask``) whose gauge reading lies in ``SPRAY_DFT`` (at least
    ``COAT_FRAC``); ``even``: the gauge's max/min over the print (at most ``EVEN``; 99 when either is 0, as
    Gym-Anything). The gauge's probe reads the print only."""
    g = pm.gauge(dry, mask, CELL)[mask]
    lo, hi = pm.SPRAY_DFT
    inband = float(((g >= lo) & (g <= hi)).mean())
    ratio = float(g.max() / g.min()) if g.min() > 0.0 else 99.0
    return [pm.Check("coat", inband, COAT_FRAC, inband >= COAT_FRAC), pm.Check("even", ratio, EVEN, ratio <= EVEN)]


def spray_progress(dry: np.ndarray, mask: np.ndarray, wet: np.ndarray, out_frac: float, marred_area: float,
                   area: float) -> dict:
    """Gym-Anything's spray progress on the print (``scoring.py:117-177``): ``wet`` the film at declaration, ``area``
    one cell's [m²]."""
    g = pm.gauge(dry, mask, CELL)[mask]
    lo, hi = pm.SPRAY_DFT
    credit = np.clip(np.minimum(g / lo, (2.0 * hi - g) / hi), 0.0, 1.0)
    r0 = max(COAT_FRAC * g.size * area, area)                 # what was left to paint at the start (nothing laid)
    remaining = max(0.0, float((1.0 - credit).sum()) * area - (1.0 - COAT_FRAC) * g.size * area)
    cover = float(np.clip(1.0 - remaining / r0, 0.0, 1.0))
    contain = float(np.clip((0.25 - out_frac) / 0.20, 0.0, 1.0))
    seen = g[g >= 0.5 * lo]
    finish = float(np.clip((4.0 - (seen.max() / seen.min() if seen.size else 99.0)) / 2.0, 0.0, 1.0))
    quality = (0.8 + 0.2 * contain) * (0.8 + 0.2 * finish)
    pooling = float(np.clip((wet[mask] - pm.RUN_H) / pm.RUN_H, 0.0, 1.0).sum()) * area
    marks = max(0.0, marred_area - pm.PROTECTED_AREA)
    progress = float(np.clip(cover * quality - (pooling + marks) / r0, 0.0, 1.0))
    return {"progress": progress, "cover": cover, "quality": quality, "pooling_cm2": pooling * 1e4,
            "marks_cm2": marks * 1e4}


def evaluate(h_declared: np.ndarray, read: dict, outside: float, dust: DustBooks, mask: np.ndarray, area: float,
             world: int = 0, marred: np.ndarray | None = None) -> dict:
    """One world's verdict and score. Fields are the chart's [nx, ny], ``mask`` the print; ``outside`` m³ of lacquer
    left the box, ``dust`` the mist's books (its ``on_chart`` laid as dry solids). The dry film is the process stage's
    ``dft``; the wet film at declaration judges runs, on the print, and a film clipped at ``MAX_DEPTH`` on the print
    counts as a run. ``marred`` None: no marks were tracked, ``untouched`` is not judged and takes nothing off the
    progress. Of Gym-Anything's five clear-coat criteria four are judged (``SCENE`` says how the scene differs)."""
    dry = pm.chart_field(read, "dft", mask.shape, world)
    if dry is None:
        raise KeyError("clear-coat reads the dry film from the process stage's dft: configure process drying")
    sent = float(read["supply_sent"][world])
    laid = float(dust.on_chart[world]) if dust.on_chart.size else 0.0
    dust_rel = dust.closure(world, float(read["supply_mist"][world]) * PHI0)
    landed = (float(read["supplied_volume"][world]) + float(read["pending_volume"][world])) * PHI0 + laid
    solids = abs(float(np.asarray(read["species_mass"])[world, 0]) - landed) / max(landed, 1e-30)
    marked = 0.0 if marred is None else float((marred & mask).sum()) * area
    drift = supply_drift(read, world)
    off_print = sum(float(read["supply_" + k][world]) for k in ("overspray", "mist", "caught", "lumped"))
    clipped = pm.chart_field(read, "clipped", mask.shape, world)
    wet = max(float(h_declared[mask].max()), MAX_DEPTH if clipped is not None and clipped[mask].any() else 0.0)
    led = pm.ledger(read, world=world)
    out = pm.verdict([*coat_checks(dry, mask), pm.no_runs(wet, 0.0, settled=False).differs(CAP_RUNS),
                      pm.overspray(outside * PHI0, sent * PHI0).differs(OWN_COUNT),
                      pm.untouched(marred, mask, area), led,
                      pm.Check("supply", drift, SUPPLY_REL, drift <= SUPPLY_REL and outside <= off_print),
                      pm.Check("solids", solids, SOLIDS_REL, solids <= SOLIDS_REL),
                      pm.Check("dust", dust_rel, DUST_REL, dust_rel <= DUST_REL)], scene=SCENE)
    prog = spray_progress(dry, mask, h_declared, outside / max(sent, 1e-30), marked, area)
    out["score"] = 1.0 if out["succeeded"] else FAILED * prog["progress"]
    out["progress"] = prog
    out["raw"] = {"ledger": led.value, "supply": drift, "solids": solids,        # unrounded, for exact checks
                  "dust": dust_rel}
    g = pm.gauge(dry, mask, CELL)[mask]
    out["film"] = {"gauge_um": [round(1e6 * float(g.min()), 3), round(1e6 * float(g.mean()), 3),
                                round(1e6 * float(g.max()), 3)],
                   "wet_max_um": round(1e6 * float(h_declared[mask].max()), 3),
                   "evaporated_ul": round(1e9 * float(read["evaporated_volume"][world]), 3)}
    sup = np.asarray(read["supplied"])[world, :mask.size].reshape(mask.shape)
    out["books_m3"] = {"sent": sent, "on_print": float(sup[mask].sum()), "on_bed": float(sup[~mask].sum()),
                       "off_chart_and_mist": off_print, "outside": outside, "dust_laid": laid,
                       "mist_solids": float(dust.mist[world]) if dust.mist.size else 0.0, **can_books(read, world)}
    return out


__all__ = ["CELL", "COAT_FRAC", "DENSITY", "DUST_REL", "DustBooks", "EVEN", "FAILED", "FLASH", "MAX_DEPTH", "PHI0",
           "SOLIDS_REL", "can_books", "coat_checks", "dust_books", "escape_share", "evaluate", "interval_share",
           "mist_dust", "mist_escape", "rect_share", "spray_progress"]
