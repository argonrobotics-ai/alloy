# SPDX-License-Identifier: Apache-2.0
"""The reference benchmark's ``bag_not_torn`` from a tearable bag's seams (numpy only; docs/notes/paper-pack.md).

The reference (Gym-Anything, "GA", gift_bags/task.py:73 and 622-637) fails ``bag_not_torn`` when the bag's new crack
length since the start passes 5 mm. A film on the film route tears only along the tear lines its card declares: the
topology pre-split duplicates the vertices on them, and each twin pair is a seam bond that breaks (alloy.topology,
alloy/solvers/film/seams.py). So a bag tears here only along its card's tear lines, and its crack length is the
summed length of its broken seams.

A cloth seam's break force is the sheet's fracture stress x thickness x the seam's share of its tear line
(alloy/topology/emit.py), so ``seam_lengths`` reads each seam's length from the break forces as built (read them
before the first step: the film publishes other break forces at crack tips and free edges as it runs).
"""

from __future__ import annotations

import numpy as np

from alloy.topology.types import BondKind

TEAR_TOL_MM = 5.0                       # GA: new crack length allowed in the bag (task.py:73)
CLOTH_EDGE = int(BondKind.CLOTH_EDGE)   # a cloth seam's bond kind


def seam_lengths(break_force: np.ndarray, kind: np.ndarray, fracture_stress: float, thickness: float) -> np.ndarray:
    """[B] the length of tear line each cloth seam stands for [m] (0 for other bonds and unbreakable ones)."""
    f = np.asarray(break_force, np.float64)
    seam = (np.asarray(kind) == CLOTH_EDGE) & np.isfinite(f)
    return np.where(seam, f / (float(fracture_stress) * float(thickness)), 0.0)


def torn_mm(intact: np.ndarray, world: np.ndarray, lengths: np.ndarray, worlds: int) -> np.ndarray:
    """[worlds] the summed length of the broken seams in each world [mm]."""
    broken = (np.asarray(intact) == 0) & (np.asarray(lengths) > 0.0)
    out = np.zeros(int(worlds))
    np.add.at(out, np.asarray(world)[broken], np.asarray(lengths, np.float64)[broken] * 1e3)
    return out


def bag_seams(sim, fracture_stress: float, thickness: float) -> dict | None:
    """The scene's cloth seams as built (call it before the first step): each one's length and world. None when the
    scene has no tear lines (``sim.engine.topology`` holds them, alloy.topology)."""
    topo = getattr(sim.engine, "topology", None)
    if topo is None or topo.bonds.count == 0:
        return None
    b = topo.bonds
    lengths = seam_lengths(b.break_force.numpy().copy(), b.kind.numpy().copy(), fracture_stress, thickness)
    return {"length": lengths, "world": b.world.numpy().copy(), "worlds": int(sim.worlds)}


def torn_now(sim, seams: dict | None) -> np.ndarray | None:
    """[worlds] each world's crack length so far [mm] (``bag_seams``' seams), None without tear lines."""
    if seams is None:
        return None
    intact = sim.engine.topology.bonds.intact.numpy().copy()
    return torn_mm(intact, seams["world"], seams["length"], seams["worlds"])


def bag_not_torn(torn: float | None, tol_mm: float = TEAR_TOL_MM) -> dict:
    """GA's ``bag_not_torn`` for one world from its crack length [mm]; None when the bag has no tear lines."""
    if torn is None:
        return {"bag_not_torn": None}
    return {"bag_not_torn": bool(torn <= tol_mm), "bag_torn_mm": round(float(torn), 1)}
