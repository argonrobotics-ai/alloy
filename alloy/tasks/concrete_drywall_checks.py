# SPDX-License-Identifier: Apache-2.0
"""patch-drywall's verdict (:mod:`alloy.tasks.concrete_drywall`), ported from Gym-Anything's concrete_finishing task set
(``gym-anything-robotics/benchmarks/roboworld/environments/concrete_finishing``, read-only in [internal note]; file:line).

"Fill the anchor holes and knife the patch flush" (``task.py:31``). Five criteria, all must pass (``task.py:683-705``;
thresholds ``params.py:176-184``), on the surface h (base + compound, against the wall plane) and the compound t:
- ``holes_filled``: the lowest h over the anchor holes at least -0.3 mm;
- ``flush``: compound cells (t > 0.05 mm) off the mesh within 0.3 mm of the wall plane;
- ``over_mesh``: compound cells on the mesh within 0.3 mm of the mesh's own plane (0.5 mm);
- ``mesh_covered``: under 2 % of the mesh below the fibre-top envelope by more than 0.01 mm (``paste.py:128-135``);
- ``no_stray_compound``: no compound cell beyond 150 mm of the holes and the patch (``task.py:226-231``), and no more
  than 2e-8 m³ off the modelled board (``task.py:159-167``).

Score, progress-v1 (``task_set_progress.py:197-223``, ``:251-281``): 1 on success, else 0.95 x progress. Quantities
that pass at the start become gates. ``hole_i`` (group holes): each hole's lowest h, target -0.3 mm. ``flush`` and
``over_mesh``: half the criterion's falloff to 0.6 mm, half the share of cells' error over 0.3 mm graded to 3 mm.
``mesh_covered``: half the criterion's share, half the mean deficit graded by the mesh thickness. ``no_stray_compound``:
the compound above 0.05 mm beyond 150 mm as an area at 0.05 mm, kept to 10 cm².

The verdict is Gym-Anything's on their 1 mm grid. A coarser grid is a smoke, lenient in every criterion: a hole cell
there reads h over its share of hole (:func:`surface`), what a 1 mm cell in the hole reads only with the board around
it bare (a skim beside a short hole averages it away, and a skim on a cell with a small share of hole reads larger in
``flush``); the other criteria read the averaged cells (narrow ridges, smears and bare mesh openings average away;
``mesh_covered`` at the start reads 0.048 at 1 mm, 0.020 at 5 mm on seed 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks.concrete_drywall_start import HOLES, MESH_T, Wallboard
from alloy.tasks.concrete_metrics import Criterion, Verdict
from alloy.tasks.concrete_progress import FAIL_FACTOR, Quantity, clip, falloff, progress_v1

FLUSH_MM = 0.3                  # params.py:184 (flush_mm)
COMP_T = 5.0e-5                 # m: compound visible (task.py:686)
HOLE_MM = -0.3                  # task.py:696-697
BARE_SHARE = 0.02               # task.py:703
ENVELOPE = 1.0e-5               # m: under the fibre tops by more than this is bare (task.py:693-695)
OFF_TOL = 2.0e-8                # m³ off the modelled board (task.py:54)
STRAY_ZERO_CM2 = 10.0           # task_set_progress.py:221-223
AREA_ZERO_MM = 3.0              # task_set_progress.py:212-214


@dataclass
class DrywallGrid:
    """One world's end (or start) state on our grid [nu, nv]: surface ``h`` and compound ``t`` [m], the board's masks
    and fields (:class:`~alloy.tasks.concrete_drywall_start.Wallboard`) and the compound off the board [m³]."""

    h: np.ndarray
    t: np.ndarray
    board: Wallboard
    off_m3: float = 0.0


def _c(name: str, value: float, ok: bool, threshold: float, unit: str = "") -> Criterion:
    return Criterion(name, float(value), float(threshold), bool(ok), unit)


def surface(g: DrywallGrid) -> np.ndarray:
    """The surface as their 1 mm grid reads it: a hole cell's h over its share of hole (exact at 1 mm)."""
    frac = g.board.frac
    return np.where(frac > 0.0, g.h / np.where(frac > 0.0, frac, 1.0), g.h)


def hole_masks(board: Wallboard) -> list[np.ndarray]:
    """Each anchor hole's cells: the hole cells nearest its centre."""
    d = np.stack([np.hypot(board.x - hx, board.z - hz) for hx, hz in HOLES])
    nearest = d.argmin(axis=0)
    return [board.holes & (nearest == i) for i in range(len(HOLES))]


def criteria(g: DrywallGrid) -> list[Criterion]:
    """The five criteria (``task.py:683-705``)."""
    b, h, comp = g.board, surface(g), g.t > COMP_T
    filled = float(h[b.holes].min()) * 1e3
    off = comp & ~b.mesh
    flush = float(np.abs(h[off]).max()) * 1e3 if off.any() else 0.0
    on = comp & b.mesh
    over = float(np.abs(h[on] - MESH_T).max()) * 1e3 if on.any() else 0.0
    bare = float((h[b.mesh] < b.support()[b.mesh] - ENVELOPE).mean()) if b.mesh.any() else 0.0
    far = int((comp & ~b.near()).sum())
    stray = far * b.cell ** 2 * 1e4
    return [_c("holes_filled", filled, filled >= HOLE_MM, HOLE_MM, "mm"),
            _c("flush", flush, flush <= FLUSH_MM, FLUSH_MM, "mm"),
            _c("over_mesh", over, over <= FLUSH_MM, FLUSH_MM, "mm"),
            _c("mesh_covered", bare, bare < BARE_SHARE, BARE_SHARE),
            _c("no_stray_compound", stray, far == 0 and g.off_m3 <= OFF_TOL, 0.0, "cm2")]


def quantities(g: DrywallGrid, crits: list[Criterion]) -> dict[str, Quantity]:
    """progress-v1's graded quantities (``task_set_progress.py:197-223``)."""
    c = {k.name: k for k in crits}
    b, h, comp = g.board, surface(g), g.t > COMP_T
    q: dict[str, Quantity] = {}
    for i, mask in enumerate(hole_masks(b)):
        q[f"hole_{i}"] = Quantity(float(h[mask].min()) * 1e3, HOLE_MM, c["holes_filled"].passed, 1.0, "holes")
    for name, mask, plane in (("flush", comp & ~b.mesh, 0.0), ("over_mesh", comp & b.mesh, MESH_T)):
        k = c[name]
        q[name] = Quantity(-k.value, -k.threshold, k.passed, falloff(k.value, k.threshold, 2.0 * FLUSH_MM), name, 0.5)
        err = np.abs(h[mask] - plane) * 1e3
        graded = float(np.clip(1.0 - np.maximum(err - FLUSH_MM, 0.0) / AREA_ZERO_MM, 0.0, 1.0).mean()) if err.size \
            else 1.0
        q[name + "_area"] = Quantity(graded, 1.0, k.passed, 1.0, name, 0.5)
    deficit = np.maximum(b.support()[b.mesh] - ENVELOPE - h[b.mesh], 0.0)
    graded = float(np.clip(1.0 - deficit / MESH_T, 0.0, 1.0).mean())
    mc = c["mesh_covered"]
    q["mesh_covered"] = Quantity(0.5 * (1.0 - mc.value) + 0.5 * graded, 1.0 - BARE_SHARE, mc.passed,
                                 clip(1.0 - float((deficit > 0.0).mean())), "mesh_covered")
    excess = float(np.maximum(g.t[~b.near()] - COMP_T, 0.0).sum()) * b.cell ** 2 + max(g.off_m3 - OFF_TOL, 0.0)
    area = excess / COMP_T * 1e4
    q["no_stray_compound"] = Quantity(-area, 0.0, c["no_stray_compound"].passed, falloff(area, 0.0, STRAY_ZERO_CM2),
                                      "no_stray_compound")
    return q


def patch_drywall(grid: DrywallGrid, start: DrywallGrid | None = None) -> Verdict:
    """The verdict: success when all five pass; with ``start``, progress-v1 and its score."""
    crits = criteria(grid)
    passed = [k.passed for k in crits]
    success = all(passed)
    prog = None
    if start is not None:
        prog = progress_v1(quantities(grid, crits), {}, quantities(start, criteria(start)))
    score = 1.0 if success else (FAIL_FACTOR * prog.value if prog is not None else None)
    return Verdict("patch-drywall", crits, success, sum(passed) / len(passed), prog, score)


__all__ = ["BARE_SHARE", "COMP_T", "DrywallGrid", "FLUSH_MM", "HOLE_MM", "OFF_TOL", "criteria", "hole_masks",
           "patch_drywall", "quantities", "surface"]
