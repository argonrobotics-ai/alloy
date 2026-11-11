# SPDX-License-Identifier: Apache-2.0
"""Snapshot arrays of a ``VNodeCutter`` (WP10 item 9 step 4; WP4 review of step 3, item 1).

A split rewrites model arrays: tet materials (the whole tet and its fragments), particle mass, inverse mass and flags
(hubs and copies), and weld stiffness. The cutter also keeps split state across steps. A rollback, an onset re-run or
a health latch that restores ``State`` alone would keep a discarded attempt's splits and mass hand-overs. The same
class of bug as the MPM history and the health arrays. ``state_arrays`` names every array the next step reads that
the cutter changes, and ``state_worlds`` maps each one's items to their worlds (``WorldSnapshot`` ``extra`` /
``extra_world``):
- tets (model ``tet_materials``): their first corner's world;
- particles (mass, inverse mass, flags, ``particle_side``, ``released``): ``particle_world``;
- springs (weld stiffness): their first particle's world;
- cuttable tet rows (``split``, ``side_mask``, ``frac``, ``front``): ``TetTable.world``;
- hub rows (side, conflicts, on-plane, added mass, the work rule's arrays): their hub's world;
- mesh edges (``edge_t``): their first hub's world.
Keys of the model arrays are those of ``solver_hooks`` (``topology.model.*``), so a seam controller in the same
model names the same arrays alike. ``split_now`` is cleared by every pass and is left out.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from alloy.topology.vnode_cut import VNodeCutter


def _world_ids(model, particles: np.ndarray) -> np.ndarray:
    """World of each particle; a model built without ``replicate`` (world -1) is world 0."""
    return np.maximum(model.particle_world.numpy()[particles], 0).astype(np.int32)


def cutter_state_arrays(cut: VNodeCutter) -> dict[str, wp.array]:
    m, wr = cut.model, cut.work_rule
    out = {"topology.model.tet_materials": m.tet_materials, "topology.model.particle_mass": m.particle_mass,
           "topology.model.particle_inv_mass": m.particle_inv_mass, "topology.model.particle_flags": m.particle_flags}
    if m.spring_count:
        out["topology.model.spring_stiffness"] = m.spring_stiffness
    names = {"split": cut.split, "side_mask": cut.side_mask, "frac": cut.frac, "front": cut.front,
             "edge_t": cut.edge_t, "hub_side": cut.hub_side, "conflicts": cut.conflicts, "on_plane": cut.on_plane,
             "particle_side": cut.particle_side, "added": cut.added, "work": wr.work, "depth": wr.depth,
             "edge0": wr.edge0, "edge1": wr.edge1, "tracked": wr.tracked, "cut": wr.cut, "bond": wr.bond,
             "released": wr.released}
    out.update({f"topology.vnode.{k}": v for k, v in names.items()})
    return out


def cutter_state_worlds(cut: VNodeCutter) -> Mapping[str, wp.array]:
    m, c, dev = cut.model, cut.copies, cut.model.device
    i32 = lambda a: wp.array(np.ascontiguousarray(a, np.int32), dtype=wp.int32, device=dev)  # noqa: E731
    particle = i32(_world_ids(m, np.arange(m.particle_count)))
    tet = i32(_world_ids(m, m.tet_indices.numpy()[:, 0]))
    hub = i32(_world_ids(m, c.node_hub))
    row = cut.tets.world
    out = {"topology.model.tet_materials": tet, "topology.model.particle_mass": particle,
           "topology.model.particle_inv_mass": particle, "topology.model.particle_flags": particle,
           "topology.vnode.particle_side": particle, "topology.vnode.released": particle,
           "topology.vnode.edge_t": i32(_world_ids(m, c.node_hub[c.edge[:, 0]]))}
    if m.spring_count:
        out["topology.model.spring_stiffness"] = i32(_world_ids(m, m.spring_indices.numpy().reshape(-1, 2)[:, 0]))
    out.update({f"topology.vnode.{k}": row for k in ("split", "side_mask", "frac", "front")})
    out.update({f"topology.vnode.{k}": hub for k in ("hub_side", "conflicts", "on_plane", "added", "work", "depth",
                                                      "edge0", "edge1", "tracked", "cut", "bond")})
    return out
