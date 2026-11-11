# SPDX-License-Identifier: Apache-2.0
"""Pre-split cloth and soft meshes for objects that can tear or be cut (WP10, D19, SPEC §3.8).

An object whose card has a `TopologyBudget` and whose material has `failure` values is emitted through WP10's
`add_presplit_cloth` / `add_presplit_soft` instead of `add_cloth_mesh` / `add_soft_mesh`. The seam plan comes from
`presplit.plan_for_budget`: the budget's `split_spacing`, else the finest regular pattern whose duplicated vertices
fit `spare_vertices`, cutting across the sheet's two in-plane axes (cloth) or all three (tets). The seam bonds (twin
springs, CLOTH_EDGE / TET_CUT) carry thresholds from the failure values; the BondSet is read back after finalize
(`alloy.topology.bonds_from_cards`). Mass comes from area/volume at build time, so splitting conserves it.
A soft object whose budget says `cut_model: copies` is emitted by `alloy.topology.copies_model.add_soft_copies`
instead: the whole mesh with dormant virtual-node copies, cuttable along any blade path, no seam plan or spares.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.cards.types import ResolvedObject

log = logging.getLogger("alloy.scene")


def wants_presplit(obj: ResolvedObject) -> bool:
    """Emit through WP10: pre-split seams need the material's failure values (thresholds); virtual-node copies
    (``cut_model: copies``) need only the budget (a blade cuts them; the strain gate is 0 without failure values)."""
    budget = obj.card.topology
    return budget is not None and (obj.material.failure is not None or budget.cut_model == "copies")


def _plan(vertices: np.ndarray, elements: np.ndarray, obj: ResolvedObject, axes: tuple[int, ...]):
    from alloy.topology.presplit import plan_for_budget

    budget = obj.card.topology
    return plan_for_budget(vertices, elements, split_spacing=budget.split_spacing,
                           spare_vertices=budget.spare_vertices, axes=axes)


def sheet_plane_axes(vertices: np.ndarray) -> tuple[int, int]:
    """The two mesh-frame axes with the largest extent (a grid sheet lies in its x-y plane)."""
    ext = np.ptp(np.asarray(vertices, np.float64).reshape(-1, 3), axis=0)
    a, b = np.argsort(-ext)[:2]
    return tuple(sorted((int(a), int(b))))  # type: ignore[return-value]


def add_cloth(b: newton.ModelBuilder, obj: ResolvedObject, vertices: np.ndarray, tris: np.ndarray,
              cloth_kwargs: Mapping[str, Any], thickness: float) -> None:
    from alloy.core.errors import AssemblyError
    from alloy.topology.emit import add_presplit_cloth

    if obj.card.topology.cut_model != "seams":
        raise AssemblyError(f"{obj.card.name}: topology.cut_model {obj.card.topology.cut_model!r} is for soft (tet) "
                            "objects; cloth is pre-split (cut_model: seams)")
    verts = np.asarray(vertices, np.float64).reshape(-1, 3)
    plan = _plan(verts, np.asarray(tris).reshape(-1, 3), obj, sheet_plane_axes(verts))
    rec = add_presplit_cloth(b, verts, np.asarray(tris).reshape(-1, 3), plan, cloth_kwargs, budget=obj.card.topology,
                             failure=obj.material.failure, elastic=obj.material.elastic, thickness=thickness)
    log.info("%s: pre-split cloth, +%d vertices, %d seam bonds", obj.card.name, rec.vertices_added, len(rec.bonds))


def add_soft(b: newton.ModelBuilder, obj: ResolvedObject, vertices: np.ndarray, tets: np.ndarray,
             soft_kwargs: Mapping[str, Any]) -> None:
    from alloy.topology.emit import add_presplit_soft

    verts = np.asarray(vertices, np.float64).reshape(-1, 3)
    if obj.card.topology.cut_model == "copies":          # cuttable along any blade path (WP10, GO 2026-11-06T19:36Z)
        from alloy.topology.copies_model import add_soft_copies

        failure = obj.material.failure
        gate = failure.cut_min_strain if failure is not None else None
        add_soft_copies(b, verts, np.asarray(tets).reshape(-1, 4), soft_kwargs,
                        cut_strain=float(gate.value) if gate is not None else 0.0)
        return
    plan = _plan(verts, np.asarray(tets).reshape(-1, 4), obj, (0, 1, 2))
    rec = add_presplit_soft(b, verts, np.asarray(tets).reshape(-1, 4), plan, soft_kwargs, budget=obj.card.topology,
                            failure=obj.material.failure)
    log.info("%s: pre-split tets, +%d vertices, %d seam bonds", obj.card.name, rec.vertices_added, len(rec.bonds))


__all__ = ["wants_presplit", "add_cloth", "add_soft", "sheet_plane_axes"]
