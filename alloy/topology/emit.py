# SPDX-License-Identifier: Apache-2.0
"""Emit pre-split cloth and tet meshes into a ModelBuilder with twin bonds (D19). Called by WP2 emitters.

Twin springs have rest length 0 and stiffness ``seam_ke_ratio`` x the element stiffness scale. For cloth the scale is
``tri_ke``. Spike wp10-spikes/spike1_vbd_seam.py: a ratio of 100 keeps a pulled strip within 0.4 % of the uncut strip
at 10 VBD iterations. 1000 converges slowly, because VBD colouring ignores springs and the twins share a colour.
For tets the scale is ``(k_mu + k_lambda)·sqrt(A)``, with A the bond's tributary seam area, and the ratio is
``SOFT_SEAM_KE_RATIO`` = 1 (coordinator GO 2026-11-06, [internal note] scr/WP10.md). Stiffer twins converge slowly at 10
VBD iterations: a welded pair moves only ~k_material / k_twin per sweep. The sweep ([internal note]
notes/wp10-runs/step3/seam_sag.py, seam_ratio_sweep.py; 10 iterations), top sag of the TOP-2 block (one seam) at 1 g,
vs the solid block:
- ratio 1: +7 %; 2: +15 %; 3: +24 %; 5: +40 %; 10: +83 %; 100: +827 %;
- cantilevered 20 mm block at 10 g, end displacement in shear / tension across the seam: ratio 1: +19 % / +40 %,
  2: +20 % / +33 %, 100: +744 % / +962 %;
- with a seam on every 5 mm cell face (``cut_block_free``): ratio 1: +180 % / +237 %, 100: 90x / 133x. Known gap:
  lattice-split blocks stay 2-3x too compliant under load at 10 iterations.

Thresholds come from ``Failure`` (§3.2):
- ``break_strain``: ``fracture_strain``; with ``tear_energy`` it is capped by the Griffith strain
  ``sqrt(2·G_c/(E·h·ℓ))`` at element size ℓ (cloth only, UNVERIFIED mesh-scale approximation).
- ``break_force``: ``fracture_stress`` x h x tributary length for cloth, ``fracture_stress`` x area for tets.
- ``cut_strain`` / ``cut_force``: ``cut_min_strain`` / ``cut_min_force``, the blade gates.

Mass is computed from area or volume by the builder (newton@ac7f7830:newton/_src/sim/builder.py:11765-11769,
12286-12289), so a split conserves it exactly.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.core.errors import CapacityError
from alloy.topology.bonds import add_spring_bond, register_custom_attributes, tag_cross_seam_edge
from alloy.topology.presplit import PresplitMesh, SplitPlan, presplit_mesh
from alloy.topology.types import BondKind, TopologyBudget

if TYPE_CHECKING:
    from alloy.cards.types import Elastic, Failure

log = logging.getLogger("alloy.topology")

SEAM_KE_RATIO = 100.0          # cloth twins (spike1: a pulled strip within 0.4 % of the uncut one)
SOFT_SEAM_KE_RATIO = 1.0       # tet twins (sweep above: the only ratio in 1-10 within 10 % of the solid block)


@dataclass(frozen=True)
class PresplitRecord:
    """Template-local index ranges added by one presplit emit."""

    particles: range
    bonds: range
    cross_seam_edges: int
    vertices_added: int


def _q(x: Any) -> float | None:
    return None if x is None else float(getattr(x, "value", x))


def seam_thresholds(failure: Failure | None, elastic: Elastic | None, *, thickness: float | None,
                    measure: np.ndarray, element_size: float, cloth: bool) -> dict[str, np.ndarray]:
    """Per-bond break/cut thresholds (inf where the card gives none)."""
    n = len(measure)
    inf = np.full(n, math.inf)
    out = {"break_strain": inf.copy(), "break_force": inf.copy(), "cut_strain": inf.copy(), "cut_force": inf.copy()}
    if failure is None:
        return out
    eps, sig, g_c = _q(failure.fracture_strain), _q(failure.fracture_stress), _q(failure.tear_energy)
    youngs = _q(elastic.youngs) if elastic is not None else None
    if eps is not None:
        out["break_strain"][:] = eps
    if g_c is not None and cloth and youngs and thickness and element_size > 0.0:
        out["break_strain"] = np.minimum(out["break_strain"], math.sqrt(2.0 * g_c / (youngs * thickness * element_size)))
    if sig is not None and cloth and not thickness:
        log.warning("seam thresholds: fracture_stress given for cloth without a thickness; no force criterion")
    elif sig is not None:
        out["break_force"] = sig * measure * (thickness if cloth else 1.0)      # N: Pa·m·m (cloth) or Pa·m²
    if _q(failure.cut_min_strain) is not None:
        out["cut_strain"][:] = _q(failure.cut_min_strain)
    if _q(failure.cut_min_force) is not None:
        out["cut_force"][:] = _q(failure.cut_min_force)
    return out


def _check_budget(mesh: PresplitMesh, budget: TopologyBudget | None, label: str | None) -> None:
    spare = budget.spare_vertices if budget is not None else 0
    if mesh.n_added > spare:
        raise CapacityError(f"presplit {label or ''}: {mesh.n_added} duplicated vertices exceed "
                            f"TopologyBudget.spare_vertices = {spare}")


def _mean_edge(vertices: np.ndarray, elements: np.ndarray) -> float:
    p = vertices[elements]
    return float(np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2).mean())


def _rotate(rot: Any, v: np.ndarray) -> np.ndarray:
    q = np.asarray(rot, dtype=np.float64)                  # (x, y, z, w)
    u, w = q[:3], q[3]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def _add_twin_bonds(builder: newton.ModelBuilder, mesh: PresplitMesh, p0: int, ke: np.ndarray, kind: BondKind,
                    thr: Mapping[str, np.ndarray], rot: Any) -> range:
    first = None
    normals = _rotate(rot, mesh.normal) if len(mesh.normal) else mesh.normal
    for k, (a, b) in enumerate(mesh.twins):
        idx = add_spring_bond(builder, p0 + int(a), p0 + int(b), float(ke[k]), 0.0, kind=kind,
                              break_force=thr["break_force"][k], break_strain=thr["break_strain"][k],
                              cut_strain=thr["cut_strain"][k], cut_force=thr["cut_force"][k],
                              normal=tuple(normals[k]), measure=float(mesh.measure[k]))
        first = idx if first is None else first
    return range(first, first + len(mesh.twins)) if first is not None else range(0)


def _cross_seam_edges(builder: newton.ModelBuilder, mesh: PresplitMesh, p0: int, bonds: range, edge_ke: float,
                      edge_kd: float) -> int:
    """One bending element per seam edge, spanning the two sides; tagged with the bonds at its two ends."""
    if edge_ke <= 0.0 or len(mesh.seam_pairs) == 0:
        return 0
    bond_of = {(int(a), int(b)): bonds[k] for k, (a, b) in enumerate(mesh.twins)}
    src = mesh.source
    quads, ends = [], []
    for e1, e2 in mesh.seam_pairs:
        t1, t2 = mesh.elements[e1], mesh.elements[e2]
        shared = sorted(set(src[t1].tolist()) & set(src[t2].tolist()))
        if len(shared) != 2:
            continue
        side1 = {int(src[v]): int(v) for v in t1}
        side2 = {int(src[v]): int(v) for v in t2}
        o1 = next(int(v) for v in t1 if int(src[v]) not in shared)
        o2 = next(int(v) for v in t2 if int(src[v]) not in shared)
        quads.append((o1, o2, side1[shared[0]], side1[shared[1]]))
        ends.append([bond_of.get(tuple(sorted((side1[s], side2[s]))), -1) for s in shared])
    if not quads:
        return 0
    q = np.asarray(quads, dtype=np.int64) + p0
    e0 = builder.edge_count
    builder.add_edges(q[:, 0].tolist(), q[:, 1].tolist(), q[:, 2].tolist(), q[:, 3].tolist(),
                      edge_ke=[edge_ke] * len(q), edge_kd=[edge_kd] * len(q))
    for k, (b0, b1) in enumerate(ends):
        tag_cross_seam_edge(builder, e0 + k, b0, b1)
    return len(q)


def add_presplit_cloth(builder: newton.ModelBuilder, vertices: np.ndarray, indices: np.ndarray, plan: SplitPlan,
                       cloth_kwargs: Mapping[str, Any], *, budget: TopologyBudget | None = None,
                       failure: Failure | None = None, elastic: Elastic | None = None, thickness: float | None = None,
                       seam_ke_ratio: float = SEAM_KE_RATIO) -> PresplitRecord:
    """Pre-split a triangle mesh, add it with ``add_cloth_mesh(**cloth_kwargs)``, twin CLOTH_EDGE bonds and
    cross-seam bending. ``cloth_kwargs`` are add_cloth_mesh keywords (pos, rot, scale, vel, density, tri_ke, ...)."""
    register_custom_attributes(builder)
    mesh = presplit_mesh(vertices, indices, plan)
    _check_budget(mesh, budget, cloth_kwargs.get("label"))
    scale = float(cloth_kwargs.get("scale", 1.0))
    kw = dict(cloth_kwargs)
    p0 = builder.particle_count
    builder.add_cloth_mesh(vertices=[wp.vec3(*v) for v in mesh.vertices], indices=mesh.elements.ravel().tolist(),
                           **kw)
    tri_ke = float(kw.get("tri_ke") if kw.get("tri_ke") is not None else builder.default_tri_ke)
    measure = mesh.measure * scale
    thr = seam_thresholds(failure, elastic, thickness=thickness, measure=measure,
                          element_size=_mean_edge(mesh.vertices, mesh.elements) * scale, cloth=True)
    ke = np.full(len(mesh.twins), seam_ke_ratio * tri_ke)
    rot = kw.get("rot", wp.quat_identity())
    bonds = _add_twin_bonds(builder, mesh, p0, ke, BondKind.CLOTH_EDGE, thr, rot)
    edge_ke = float(kw.get("edge_ke") if kw.get("edge_ke") is not None else builder.default_edge_ke)
    n_cross = _cross_seam_edges(builder, mesh, p0, bonds, edge_ke, float(kw.get("edge_kd") or 0.0))
    log.info("presplit cloth %s: +%d vertices, %d CLOTH_EDGE bonds, %d cross-seam bending elements",
             kw.get("label") or "", mesh.n_added, len(bonds), n_cross)
    return PresplitRecord(range(p0, builder.particle_count), bonds, n_cross, mesh.n_added)


def add_presplit_soft(builder: newton.ModelBuilder, vertices: np.ndarray, tets: np.ndarray, plan: SplitPlan,
                      soft_kwargs: Mapping[str, Any], *, budget: TopologyBudget | None = None,
                      failure: Failure | None = None, seam_ke_ratio: float = SOFT_SEAM_KE_RATIO) -> PresplitRecord:
    """Pre-split a tet mesh, add it with ``add_soft_mesh(**soft_kwargs)`` and twin TET_CUT bonds."""
    register_custom_attributes(builder)
    mesh = presplit_mesh(vertices, tets, plan)
    _check_budget(mesh, budget, soft_kwargs.get("label"))
    kw = dict(soft_kwargs)
    scale = float(kw.get("scale", 1.0))
    p0 = builder.particle_count
    builder.add_soft_mesh(vertices=[wp.vec3(*v) for v in mesh.vertices], indices=mesh.elements.ravel().tolist(),
                          **kw)
    measure = mesh.measure * scale * scale
    thr = seam_thresholds(failure, None, thickness=None, measure=measure, element_size=0.0, cloth=False)
    k_el = float(kw.get("k_mu") or 0.0) + float(kw.get("k_lambda") or 0.0)
    ke = seam_ke_ratio * k_el * np.sqrt(np.maximum(measure, 0.0))
    bonds = _add_twin_bonds(builder, mesh, p0, ke, BondKind.TET_CUT, thr, kw.get("rot", wp.quat_identity()))
    log.info("presplit soft %s: +%d vertices, %d TET_CUT bonds", kw.get("label") or "", mesh.n_added, len(bonds))
    return PresplitRecord(range(p0, builder.particle_count), bonds, 0, mesh.n_added)
