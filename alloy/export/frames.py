# SPDX-License-Identifier: Apache-2.0
"""What a frame holds (SCR-WP13-EXPORT; docs/guide/frame-export.md), assembled on the host from one staging slot's
copied arrays. Every array is the frame's own copy (the pinned buffers are reused)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.export.plan import ExportPlan

ACTIVE = 1                                              # newton ParticleFlags.ACTIVE


@dataclass
class Topology:
    """The part of a frame that changes only with the topology: sent on a world's first frame and on every change."""

    body_names: list[str]
    body_active: np.ndarray                             # [B] bool
    body_shapes: list[list[dict[str, Any]]]             # per body: its shapes in the body frame
    statics: list[dict[str, Any]]                       # ground, tables, bins: shapes with a world pose
    mesh_faces: dict[str, np.ndarray]                   # name -> [F, 3] int32, into the mesh's vertices
    mesh_thickness: dict[str, float]                    # name -> m; vertices are the mid-surface (offset h/2 to draw)
    mesh_vertex_active: dict[str, np.ndarray]           # name -> [V] bool; False: a dormant virtual-node copy
    mesh_tets: dict[str, np.ndarray]                    # soft body name -> [T, 4] int32, into the mesh's vertices
    mesh_tet_active: dict[str, np.ndarray]              # name -> [T] bool; False: a dormant fragment or a split tet
    particles: dict[str, dict[str, Any]]                # name -> route, material, material_id, count, indices, radius
    rods: dict[str, dict[str, Any]]                     # name -> segments, route, bodies, drawn_by_bodies


@dataclass
class Frame:
    """One picked world at one moment. Poses are raw body frames, quaternions in ``quat_order``."""

    world: int
    period: int
    t: float
    topology_version: int
    body_pos: np.ndarray                                # [B, 3] float32
    body_quat_xyzw: np.ndarray                          # [B, 4] float32
    meshes: dict[str, np.ndarray]                       # name -> [V, 3] float32
    particles: dict[str, dict[str, np.ndarray]]         # name -> {"pos": [k, 3] float32, "active": [k] bool}
    rods: dict[str, np.ndarray]                         # name -> [N + 1, 3] float32
    topology: Topology | None
    quat_order: str = "xyzw"


def topology(plan: ExportPlan, p: int, host: dict[str, np.ndarray], radius: dict[str, Any]) -> Topology:
    """The topology part of picked world ``p`` (index into ``plan.worlds``)."""
    active = host["active"][p] != 0
    verts = {n: (host["mesh_flags"][p, sl] & ACTIVE) != 0 for n, sl in plan.mesh.items()}
    tets = {n: host["tet_live"][p, sl] != 0 for n, sl in plan.tet.items()}
    parts = {s.name: {"route": s.route, "material": s.material, "material_id": s.material_id, "count": s.count,
                      "indices": s.local.astype(np.int32), "radius": radius[s.name]} for s in plan.sets}
    thick = {name: 2.0 * float(radius.get(f"mesh:{name}", 0.0)) for name in plan.mesh}
    rods = {name: dict(info) for name, info in plan.rod_info.items()}
    return Topology(list(plan.body_names), active, plan.shapes, plan.static[plan.worlds[p]],
                    {k: v.copy() for k, v in plan.faces.items()}, thick, verts,
                    {k: v.copy() for k, v in plan.tets.items()}, tets, parts, rods)


def assemble(plan: ExportPlan, host: dict[str, np.ndarray], meta: tuple[int, float], with_topology: set[int],
             radius: dict[str, Any], versions: list[int]) -> list[Frame]:
    """The frames of every picked world from one slot's host arrays; ``with_topology``: the picked indices that get
    the topology part."""
    period, t = meta
    xf = host["body_q"].reshape(len(plan.worlds), -1, 7)
    out = []
    for p, w in enumerate(plan.worlds):
        meshes = {n: host["mesh"][p, sl].copy() for n, sl in plan.mesh.items()}
        parts = {n: {"pos": host["part"][p, sl].copy(), "active": (host["flags"][p, sl] & ACTIVE) != 0}
                 for n, sl in plan.part.items()}
        rods = {n: host["rod"][p, sl].copy() for n, sl in plan.rod.items()}
        topo = topology(plan, p, host, radius) if p in with_topology else None
        out.append(Frame(w, period, t, versions[p], xf[p, :, :3].copy(), xf[p, :, 3:7].copy(), meshes,
                         parts, rods, topo))
    return out


__all__ = ["Frame", "Topology", "assemble", "topology"]
