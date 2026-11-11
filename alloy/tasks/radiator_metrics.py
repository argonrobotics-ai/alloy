# SPDX-License-Identifier: Apache-2.0
"""behind-radiator's verdict and score: Gym-Anything's progress-v1 contract, term for term (``PT/task.py:21-41``
(criteria), ``:57-87`` (``BEHIND_RADIATOR``), ``:141-149`` (start), ``:299-372`` (measure), ``:476-511``
(``_remaining``, ``_reward``); ``PT/params.py:238-252`` (``EVAL``)). All read from the settled film:

- **hide**: the band less the pocket hidden (the dry stack within ΔE*ab 1.0 of the paint's own colour) on >= 98 %, and
  no 8-connected unhidden patch over 20 cm²;
- **no_runs**: band cells into which the film's flow carried more than 50 µm of dry paint, net of what the brush took
  back up (``flowed``), on <= 1 cm², the band's bottom 4 mm (``run_edge``) not read; before the settle a wet film over
  250 µm also fails;
- **no_marks**: wet band film touched by a link or a tool's non-applicator part and not re-laid, <= 0.5 cm² (no
  robot here: none);
- **protected**: paint on the radiator and the trim, <= 0.5 cm²: cells of their charts over 1 µm dry, plus film lost
  into sinks (off the radiator's top or the ledge, drips caught by the radiator's or the trim's shapes, and drips
  the supply books lost: onto the floor, protected in Gym-Anything's room; worst case) spread 100 µm thick
  (``PT/task.py:224-250``); the wall's edges are closed, as Gym-Anything's;
- laps are reported, not judged; the score is 1 on success, else 0.9 x progress.
The check names are Gym-Anything's (``hide``, ``no_runs``, ``no_marks``, ``protected``). The film's books (the
plug-in's volume ledger, ``paint_metrics.ledger``, and the solids ledger) are checks here; in Gym-Anything a ledger
fault is a backend fault, not a task result (``PT/task.py:41``).

``flowed`` in Alloy: per cell, the dry film now less the dry film set at the start and the solids the tools laid
(``species_supplied``) net of what they took (``species_out``), its positive part (WP15's reading of Gym-Anything's
``flowed``). Drips that leave the brush land where they fall (the supply's stage drips): on a protected chart
they count as paint there.

Where Alloy's reading differs from Gym-Anything's:
- ``flowed`` (stricter): Gym-Anything sums the film flow's net solids into each cell (``PT/film.py:490``) and, when a
  tool takes a share of a cell's wet film, takes the same share of its positive ``flowed`` (``PT/film.py:364-367``).
  Here the dry film less the tools' net lay is the flow's net solids exactly (the books close), and a tool's take
  never lowers it: paint that crept into a cell and was brushed back out still counts in full. Equal when no tool
  takes film back from a cell paint flowed into.
- ``protected``'s sinks (looser for a thin smear): Gym-Anything counts each sink as the larger of its 4 mm voxels
  holding >= 1 µm and its volume over 100 µm (``PT/task.py:243-249``); the film lost here has no footprint, so it
  counts as its volume over 100 µm only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks import paint_metrics as pm

# PT/task.py:86 and PT/params.py:238 (EVAL)
HIDE_FRAC, HOLIDAY = 0.98, 20e-4            # -, m²
RUN_AREA, RUN_EDGE, RUN_FLOW = 1e-4, 0.004, 50e-6     # m², m, m dry
RUN_H = pm.RUN_H                            # 250 µm wet
MARK_AREA = PROTECTED_AREA = pm.PROTECTED_AREA     # 0.5 cm²
PAINTED_DFT, SINK_FILM = pm.PAINTED_DFT, pm.SINK_FILM
DAMAGE_WEIGHT, FAIL_SCALE = 10.0, 0.9


@dataclass(frozen=True)
class Masks:
    """The wall chart's masks [nx, ny]: the band, its scored part (less the pocket), its run-edge rows."""

    band: np.ndarray
    scored: np.ndarray
    edge: np.ndarray


def flowed_in(dft: np.ndarray, dft0: np.ndarray, supplied: np.ndarray, taken: np.ndarray, area: float) -> np.ndarray:
    """Dry paint the film's flow carried into each cell [m]: the dry film less what was set there and the tools'
    net lay (``supplied`` and ``taken`` are the cell's solids volumes [m³])."""
    return np.maximum(np.asarray(dft, np.float64) - dft0 - (supplied - taken) / area, 0.0)


def remaining(hidden: np.ndarray, scored: np.ndarray, area: float) -> tuple[float, float]:
    """(area still to paint before hide can pass, largest unhidden patch) over ``scored`` [m²]: the larger of the
    unhidden area beyond the 2 % allowance and the 8-connected unhidden patches' excess over 20 cm²
    (``PT/task.py:476-487``)."""
    from scipy.ndimage import label  # noqa: PLC0415

    un = scored & ~hidden
    lab, n = label(un, structure=np.ones((3, 3), int))
    sizes = np.bincount(lab.ravel())[1:] * area if n else np.zeros(1)
    excess = float(np.maximum(sizes - HOLIDAY, 0.0).sum())
    beyond = float(un.sum()) * area - (1.0 - HIDE_FRAC) * float(scored.sum()) * area
    return max(beyond, excess, 0.0), float(sizes.max())


def hide(hidden: np.ndarray, m: Masks, area: float) -> tuple[pm.Check, dict]:
    frac = float(hidden[m.scored].mean()) if m.scored.any() else 0.0
    rem, big = remaining(hidden, m.scored, area)
    ok = frac >= HIDE_FRAC and big <= HOLIDAY
    return pm.Check("hide", frac, HIDE_FRAC, ok), {"largest_unhidden_cm2": round(big * 1e4, 2),
                                                    "whole_band": round(float(hidden[m.band].mean()), 4),
                                                    "still_to_paint_cm2": round(rem * 1e4, 2)}


def runs(flowed: np.ndarray, wet: np.ndarray, m: Masks, area: float, settled: bool = True) -> tuple[pm.Check, dict,
                                                                                                     np.ndarray]:
    """The run cells (flowed over ``RUN_FLOW``, in the band, not its edge rows), the check and its detail."""
    run = (flowed > RUN_FLOW) & m.band & ~m.edge
    run_a = float(run.sum()) * area
    h_max = float(wet[m.band].max()) if m.band.any() else 0.0
    me = m.band & ~m.edge
    ok = (h_max <= RUN_H or settled) and run_a <= RUN_AREA
    return (pm.Check("no_runs", run_a, RUN_AREA, ok),
            {"wet_max_um": round(h_max * 1e6, 1), "flowed_in_max_um": round(float(flowed[me].max()) * 1e6, 2)
             if me.any() else 0.0}, run)  # fmt: skip


def protected(dft_protected: list[tuple[np.ndarray, np.ndarray]], lost_volume: float) -> pm.Check:
    """Painted cells of the protected charts (each its dry film and cell areas [n]) plus film lost into sinks."""
    a = sum(float(ca[d > PAINTED_DFT].sum()) for d, ca in dft_protected) + max(lost_volume, 0.0) / SINK_FILM
    return pm.Check("protected", a, PROTECTED_AREA, a <= PROTECTED_AREA)


def marks(marred: np.ndarray | None, m: Masks, area: float) -> pm.Check:
    """Wet paint marred on the band. ``marred`` None: no marks were tracked, so the criterion is not judged
    (``paint_metrics.NO_MARKS``), never passed on an empty array."""
    if marred is None:
        return pm.Check("no_marks", 0.0, MARK_AREA, None, pm.NO_MARKS)
    a = float((marred & m.band).sum()) * area
    return pm.Check("no_marks", a, MARK_AREA, a <= MARK_AREA)


def progress(hidden: np.ndarray, m: Masks, area: float, start: float, marred: np.ndarray, run: np.ndarray,
             protected_area: float) -> dict:
    """Gym-Anything's progress-v1 (``PT/task.py:489-511``): coverage less the defects beyond their tolerances,
    damage x 10, over the area to paint at the start."""
    rem, _ = remaining(hidden, m.scored, area)
    cov = min(max(1.0 - rem / start, 0.0), 1.0)
    mk = max(0.0, float((marred & hidden).sum()) * area - MARK_AREA)
    rn = max(0.0, float((run & hidden & ~marred).sum()) * area - RUN_AREA)
    dmg = DAMAGE_WEIGHT * max(0.0, protected_area - PROTECTED_AREA)
    prog = min(max(cov - (mk + rn + dmg) / start, 0.0), 1.0)
    return {"progress": round(prog, 6), "coverage": round(cov, 6), "start_to_paint_cm2": round(start * 1e4, 1),
            "defects_cm2": {"marks": round(mk * 1e4, 3), "runs": round(rn * 1e4, 3), "damage": round(dmg * 1e4, 3)}}


def start_area(hidden0: np.ndarray, m: Masks, area: float) -> float:
    """The area to paint at the start, the score's unit: at least one cell (``PT/task.py:149``)."""
    return max(remaining(hidden0, m.scored, area)[0], area)


__all__ = ["DAMAGE_WEIGHT", "FAIL_SCALE", "HIDE_FRAC", "HOLIDAY", "Masks", "RUN_AREA", "RUN_EDGE", "RUN_FLOW",
           "flowed_in", "hide", "marks", "progress", "protected", "remaining", "runs", "start_area"]
