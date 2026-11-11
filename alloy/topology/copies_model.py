# SPDX-License-Identifier: Apache-2.0
"""Virtual-node copies through cards and the Simulation (WP10 item 9; coordinator GO 2026-11-06T19:36Z).

A soft object opts in on its card with ``topology: {cut_model: copies}`` (``TopologyBudget.cut_model``; default
"seams", so existing cards are unchanged).
- Emit time (``add_soft_copies``, called by the soft emitter): the soft mesh, its dormant copies
  (``tet_copies.add_tet_copies``) and the copy rows as ``topology:*`` custom attributes.
  - Per particle: a copy's hub, side and weld spring.
  - Per tet: a fragment's whole tet and side, and a cuttable tet's mass and strain gate.
  - The index columns declare ``references`` (particle, tetrahedron, spring), so ``replicate`` offsets them per
    world.
- After finalize: ``copies_from_model`` rebuilds the ``TetCopies`` record over every world. ``prepare_copies``
  recolours the copies (a copy takes its hub's colour + 1, ``tet_copies.recolor_model``) before any solver is built.
- ``vnode_topology.VNodeTopology`` runs the cutter at the Simulation's topology hooks.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.topology.bonds import NS
from alloy.topology.tet_copies import WELD_KE_RATIO, TetCopies, add_tet_copies, edge_table, recolor_model

log = logging.getLogger("alloy.topology")

# (name, dtype, frequency, default, references)
_COLUMNS = (
    ("copy_hub", wp.int32, "PARTICLE", -1, "particle"),        # a copy's hub particle (-1: not a copy)
    ("copy_side", wp.int32, "PARTICLE", -1, None),             # 0 = v+, 1 = v-
    ("copy_weld", wp.int32, "PARTICLE", -1, "spring"),         # the weld spring hub -> this copy
    ("copy_whole", wp.int32, "TETRAHEDRON", -1, "tetrahedron"),  # a fragment's whole tet (-1: not a fragment)
    ("copy_tet_side", wp.int32, "TETRAHEDRON", -1, None),      # 0 = T+, 1 = T-
    ("cut_mass", wp.float32, "TETRAHEDRON", 0.0, None),        # a cuttable tet's mass [kg] (density x volume)
    ("cut_strain", wp.float32, "TETRAHEDRON", 0.0, None),      # its strain gate (Failure.cut_min_strain; 0 = open)
)


def register_copy_attributes(builder: newton.ModelBuilder) -> None:
    """Declare the copy columns (idempotent)."""
    if builder.has_custom_attribute(f"{NS}:copy_hub"):
        return
    ca, freq = newton.ModelBuilder.CustomAttribute, newton.Model.AttributeFrequency
    for name, dtype, frequency, default, refs in _COLUMNS:
        builder.add_custom_attribute(ca(name=name, dtype=dtype, frequency=getattr(freq, frequency), namespace=NS,
                                        default=default, references=refs))


def _set(builder: newton.ModelBuilder, name: str, index: np.ndarray, value: np.ndarray) -> None:
    attr = builder.custom_attributes[f"{NS}:{name}"]
    attr.values = attr.values or {}
    for i, v in zip(np.asarray(index).reshape(-1), np.asarray(value).reshape(-1), strict=True):
        attr.values[int(i)] = v.item() if hasattr(v, "item") else v


def add_soft_copies(builder: newton.ModelBuilder, vertices: np.ndarray, tets: np.ndarray,
                    soft_kwargs: Mapping[str, Any], cut_strain: float = 0.0) -> TetCopies:
    """``add_soft_mesh(vertices, tets, **soft_kwargs)`` with every tet cuttable: dormant copies and their rows."""
    register_copy_attributes(builder)
    t0 = len(builder.tet_indices)
    builder.add_soft_mesh(vertices=[wp.vec3(*v) for v in np.asarray(vertices, np.float64).reshape(-1, 3)],
                          indices=np.asarray(tets, np.int64).reshape(-1).tolist(), **soft_kwargs)
    rec = add_tet_copies(builder, range(t0, len(builder.tet_indices)), float(soft_kwargs["density"]))
    hub = np.repeat(rec.node_hub[:, None], 2, axis=1)
    _set(builder, "copy_hub", rec.node, hub)
    _set(builder, "copy_side", rec.node, np.tile([0, 1], (len(rec.node_hub), 1)))
    _set(builder, "copy_weld", rec.node, rec.weld)
    _set(builder, "copy_whole", rec.copy_tet, np.repeat(rec.tet[:, None], 2, axis=1))
    _set(builder, "copy_tet_side", rec.copy_tet, np.tile([0, 1], (rec.count, 1)))
    _set(builder, "cut_mass", rec.tet, rec.mass.astype(np.float32))
    _set(builder, "cut_strain", rec.tet, np.full(rec.count, float(cut_strain), np.float32))
    log.info("%s: soft mesh with virtual-node copies, %d cuttable tets, %d hubs", soft_kwargs.get("label") or "",
             rec.count, len(rec.node_hub))
    return rec


def has_copies(model: newton.Model) -> bool:
    ns = getattr(model, NS, None)
    hub = getattr(ns, "copy_hub", None) if ns is not None else None
    return hub is not None and hub.shape[0] > 0 and bool((hub.numpy() >= 0).any())


def _column(model: newton.Model, name: str) -> np.ndarray:
    return getattr(getattr(model, NS), name).numpy()


def copies_from_model(model: newton.Model) -> TetCopies:
    """The ``TetCopies`` record over every world from the copy columns (model indices; hub rows in particle order,
    cuttable tets in tet order)."""
    chub, cside, cweld = _column(model, "copy_hub"), _column(model, "copy_side"), _column(model, "copy_weld")
    copies = np.flatnonzero(chub >= 0)
    node_hub = np.unique(chub[copies])
    row = np.searchsorted(node_hub, chub[copies])
    node, weld = np.full((len(node_hub), 2), -1, np.int64), np.full((len(node_hub), 2), -1, np.int64)
    node[row, cside[copies]], weld[row, cside[copies]] = copies, cweld[copies]
    whole, tside = _column(model, "copy_whole"), _column(model, "copy_tet_side")
    frags = np.flatnonzero(whole >= 0)
    tet = np.unique(whole[frags])
    copy_tet = np.full((len(tet), 2), -1, np.int64)
    copy_tet[np.searchsorted(tet, whole[frags]), tside[frags]] = frags
    if (node < 0).any() or (weld < 0).any() or (copy_tet < 0).any():
        raise ValueError("copies_from_model: incomplete copy rows (a hub without both copies or welds, or a tet "
                         "without both fragments)")
    hub = model.tet_indices.numpy()[tet].astype(np.int64)
    hub_row = np.searchsorted(node_hub, hub)
    mat = model.tet_materials.numpy()[tet].astype(np.float64)
    x = model.particle_q.numpy().astype(np.float64)[hub]
    vol = np.abs(np.linalg.det(x[:, 1:] - x[:, :1])) / 6.0
    weld_ke = np.zeros(len(node_hub))
    np.maximum.at(weld_ke, hub_row.reshape(-1), np.repeat(WELD_KE_RATIO * (mat[:, 0] + mat[:, 1]) * np.cbrt(vol), 4))
    edge, tet_edge, flip = edge_table(hub_row, len(node_hub))
    return TetCopies(tet, hub, hub_row, copy_tet, mat, _column(model, "cut_mass")[tet].astype(np.float64), node_hub,
                     node, weld, weld_ke, edge, tet_edge, flip)


def cut_strain_gate(model: newton.Model, copies: TetCopies) -> float:
    """The scene's strain gate for the work rule: the smallest over the cuttable tets (one value per cutter)."""
    gate = _column(model, "cut_strain")[copies.tet]
    return float(gate.min()) if gate.size else 0.0


def prepare_copies(model: newton.Model) -> TetCopies | None:
    """After finalize, before any solver: the record, with the copies recoloured; None without copies."""
    if not has_copies(model):
        return None
    rec = copies_from_model(model)
    recolor_model(model, rec)
    log.info("copies: %d cuttable tets, %d hubs over %d worlds; copies recoloured", rec.count, len(rec.node_hub),
             max(int(model.world_count), 1))
    return rec


def whole_tets(model: newton.Model, tets: np.ndarray) -> np.ndarray:
    """``tets`` [W, n] (an object's tets per world) without the fragment tets of virtual-node copies: a fragment
    carries no material until a split moves part of its whole tet's onto it (card readback reads the whole tets)."""
    ns = getattr(model, NS, None)
    col = getattr(ns, "copy_whole", None) if ns is not None else None
    tets = np.asarray(tets)
    if col is None or tets.size == 0:
        return tets
    keep = col.numpy()[tets.reshape(tets.shape[0], -1)[0]] < 0         # the same columns in every world
    return tets[..., keep]
