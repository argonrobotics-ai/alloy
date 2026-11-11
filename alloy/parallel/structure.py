# SPDX-License-Identifier: Apache-2.0
"""Structure dispatch report: islands per world and the solve class each would need (SPEC §3.9, parallel §4.4).

``classify_structure`` runs union-find over the bodies of the template world (world 0; every world of a pack shares
topology) with three kinds of edges:

- joints: ``joint_parent[j] -> joint_child[j]`` (newton@ac7f7830:newton/_src/sim/model.py:1223-1225);
- MuJoCo equalities: ``model.mujoco.equality_constraint_body1/2`` (solvers/mujoco/equality.py:76-87), if present;
- contacts: with ``contacts=None`` the *potential* contact graph (dynamic bodies whose shapes have COLLIDE_SHAPES and
  collision groups that may touch, geometry/broad_phase_common.py:235), which is the conservative island a solver
  must be ready for; with ``contacts`` the active rigid contacts of world 0 (persistent contacts, settled scenes).

Each island is labelled by ordered rules: an unbranched joint chain of >= 16 links holding at least half the island's
DOF -> CHAIN_PCR; nv <= 32 -> DENSE_SMALL; a single joint tree -> TREE_SPARSE; free bodies holding at least half the
DOF and nv > 128 -> MATRIX_FREE_COLORED; nv <= 128 -> DENSE_BLOCKED; otherwise MATRIX_FREE_COLORED. v1 picks among
existing Newton solvers only; classes without a shipped solve are listed in ``unavailable`` (upstream work, §8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.parallel.types import IslandPlan, StructureClass, StructurePlan

log = logging.getLogger("alloy.parallel")

DENSE_SMALL_MAX_NV = 32
DENSE_BLOCKED_MAX_NV = 128
CHAIN_MIN_LINKS = 16
WORLDS_PER_SM = 4  # resident worlds per SM before a per-world-block kernel fills the GPU (parallel §1.1)
_FREE = int(newton.JointType.FREE)
_COLLIDE_SHAPES = int(newton.ShapeFlags.COLLIDE_SHAPES)

# Classes with a shipped solve in Newton/MJW at the pin: MJW one-tile and blocked tile Cholesky
# (mjw@a9daa05f:mujoco_warp/_src/solver.py:40, 3082-3113). Tree LDLᵀ past 64 DOF, chain PCR and coloured
# matrix-free piles are upstream / wave-3 work (SPEC §8.3).
AVAILABLE = frozenset({StructureClass.DENSE_SMALL, StructureClass.DENSE_BLOCKED})


@dataclass
class _Topology:
    """Host copy of world 0's rigid topology, local body indices 0..n-1."""

    nv: np.ndarray  # [n] DOF per body (its parent joint's dof)
    free: np.ndarray  # [n] bool: parent joint is FREE
    children: list[list[int]]  # joint children per body
    has_parent: np.ndarray  # [n] bool: joint parent is a body (not the world)
    collide_group: dict[int, set[int]] = field(default_factory=dict)  # collision group -> bodies with a shape in it


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def classify_structure(model: newton.Model, *, contacts: newton.Contacts | None = None,
                       sm_count: int | None = None) -> StructurePlan:
    """Islands of the template world, their structure class and the parallel unit for ``model.world_count``."""
    bodies = _world0_bodies(model)
    if bodies.size == 0:
        return StructurePlan(islands=(), parallel_unit="world", recommended_solver="none", unavailable=())
    topo = _topology(model, bodies)
    uf = _UnionFind(bodies.size)
    _union_joints(topo, uf)
    _union_equalities(model, bodies, uf)
    if contacts is None:
        _union_potential_contacts(topo, uf)
    else:
        _union_active_contacts(model, contacts, bodies, uf)
    islands = _islands(topo, uf)
    plans = tuple(_plan_island(k, topo, members) for k, members in enumerate(islands))
    unavailable = tuple(sorted({p.cls.value for p in plans if not p.available}))
    unit = _parallel_unit(model.world_count, len(plans), _sm_count(model, sm_count))
    log.info("structure: %d islands, nv max %d, classes %s, unit %s", len(plans), max((p.nv for p in plans), default=0),
             sorted({p.cls.value for p in plans}), unit)
    return StructurePlan(islands=plans, parallel_unit=unit, recommended_solver="mujoco", unavailable=unavailable)


def _world0_bodies(model: newton.Model) -> np.ndarray:
    if model.body_count == 0:
        return np.zeros(0, dtype=np.int64)
    body_world = model.body_world.numpy()
    first = 0 if np.any(body_world == 0) else int(body_world.max())
    if np.unique(body_world[body_world >= 0]).size > 1:
        counts = np.bincount(body_world[body_world >= 0])
        if np.any(counts != counts[0]):
            log.warning("worlds differ in body count %s; classifying world %d only", np.unique(counts), first)
    return np.flatnonzero(body_world == first)


def _topology(model: newton.Model, bodies: np.ndarray) -> _Topology:
    local = {int(b): i for i, b in enumerate(bodies)}
    n = bodies.size
    topo = _Topology(nv=np.zeros(n, dtype=np.int64), free=np.zeros(n, dtype=bool), children=[[] for _ in range(n)],
                     has_parent=np.zeros(n, dtype=bool))
    if model.joint_count:
        _fill_joints(model, local, topo)
    if model.shape_count:
        _fill_shapes(model, local, topo)
    return topo


def _fill_joints(model: newton.Model, local: dict[int, int], topo: _Topology) -> None:
    parent = model.joint_parent.numpy()
    child = model.joint_child.numpy()
    jtype = model.joint_type.numpy()
    dof = model.joint_dof_dim.numpy().sum(axis=1)
    for j in range(parent.size):
        c = local.get(int(child[j]))
        if c is None:
            continue
        topo.nv[c] += int(dof[j])
        topo.free[c] |= int(jtype[j]) == _FREE
        p = local.get(int(parent[j]))
        if p is not None:
            topo.children[p].append(c)
            topo.has_parent[c] = True


def _fill_shapes(model: newton.Model, local: dict[int, int], topo: _Topology) -> None:
    shape_body = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    groups = model.shape_collision_group.numpy()
    for s in range(shape_body.size):
        b = local.get(int(shape_body[s]))
        if b is None or not flags[s] & _COLLIDE_SHAPES or groups[s] == 0:
            continue
        topo.collide_group.setdefault(int(groups[s]), set()).add(b)


def _union_joints(topo: _Topology, uf: _UnionFind) -> None:
    for p, kids in enumerate(topo.children):
        for c in kids:
            uf.union(p, c)


def _union_equalities(model: newton.Model, bodies: np.ndarray, uf: _UnionFind) -> None:
    ns = getattr(model, "mujoco", None)
    b1 = getattr(ns, "equality_constraint_body1", None)
    b2 = getattr(ns, "equality_constraint_body2", None)
    if b1 is None or b2 is None or b1.shape[0] == 0:
        return
    local = {int(b): i for i, b in enumerate(bodies)}
    for a, b in zip(b1.numpy(), b2.numpy(), strict=True):
        la, lb = local.get(int(a)), local.get(int(b))
        if la is not None and lb is not None:
            uf.union(la, lb)


def _groups_touch(ga: int, gb: int) -> bool:
    """Newton's collision-group rule (newton@ac7f7830:newton/_src/geometry/broad_phase_common.py:235-253)."""
    if ga == 0 or gb == 0:
        return False
    if ga > 0:
        return ga == gb or gb < 0
    return ga != gb


def _union_potential_contacts(topo: _Topology, uf: _UnionFind) -> None:
    groups = sorted(topo.collide_group)
    for i, ga in enumerate(groups):
        for gb in groups[i:]:
            if not _groups_touch(ga, gb):
                continue
            members = sorted(topo.collide_group[ga] | topo.collide_group[gb])
            for b in members[1:]:
                uf.union(members[0], b)


def _union_active_contacts(model: newton.Model, contacts: newton.Contacts, bodies: np.ndarray,
                           uf: _UnionFind) -> None:
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count == 0:
        return
    shape_body = model.shape_body.numpy()
    s0 = contacts.rigid_contact_shape0.numpy()[:count]
    s1 = contacts.rigid_contact_shape1.numpy()[:count]
    local = {int(b): i for i, b in enumerate(bodies)}
    for a, b in zip(s0, s1, strict=True):
        la, lb = local.get(int(shape_body[a])), local.get(int(shape_body[b]))
        if la is not None and lb is not None:
            uf.union(la, lb)


def _islands(topo: _Topology, uf: _UnionFind) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for b in range(topo.nv.size):
        groups.setdefault(uf.find(b), []).append(b)
    return [m for m in groups.values() if topo.nv[m].sum() > 0]


def _plan_island(k: int, topo: _Topology, members: list[int]) -> IslandPlan:
    nv = int(topo.nv[members].sum())
    roots = [b for b in members if not topo.has_parent[b]]
    depth = max(_depth(topo, r) for r in roots)
    chain_len, chain_nv = _longest_chain(topo, roots)
    is_tree = len(roots) == 1  # one root: every member hangs off it by joints (no contact-merged parts)
    free_nv = int(topo.nv[[b for b in members if topo.free[b] and not topo.children[b]]].sum())
    cls = _classify(nv, chain_len, chain_nv, is_tree, free_nv)
    return IslandPlan(island=k, nv=nv, depth=depth, chain_len=chain_len, cls=cls, available=cls in AVAILABLE)


def _classify(nv: int, chain_len: int, chain_nv: int, is_tree: bool, free_nv: int) -> StructureClass:
    if chain_len >= CHAIN_MIN_LINKS and 2 * chain_nv >= nv:
        return StructureClass.CHAIN_PCR
    if nv <= DENSE_SMALL_MAX_NV:
        return StructureClass.DENSE_SMALL
    if is_tree:
        return StructureClass.TREE_SPARSE
    if 2 * free_nv >= nv and nv > DENSE_BLOCKED_MAX_NV:
        return StructureClass.MATRIX_FREE_COLORED
    if nv <= DENSE_BLOCKED_MAX_NV:
        return StructureClass.DENSE_BLOCKED
    return StructureClass.MATRIX_FREE_COLORED


def _depth(topo: _Topology, root: int) -> int:
    best, stack = 0, [(root, 1)]
    while stack:
        b, d = stack.pop()
        best = max(best, d)
        stack.extend((c, d + 1) for c in topo.children[b])
    return best


def _longest_chain(topo: _Topology, roots: list[int]) -> tuple[int, int]:
    """Longest unbranched run of joint-connected links (each link but the last has one child): (links, DOF)."""
    best = (0, 0)
    stack = [(r, 0, 0) for r in roots]  # (body, links in the current run before it, dof in the run)
    while stack:
        b, run, run_nv = stack.pop()
        run, run_nv = run + 1, run_nv + int(topo.nv[b])
        best = max(best, (run, run_nv))
        kids = topo.children[b]
        for c in kids:
            stack.append((c, run, run_nv) if len(kids) == 1 else (c, 0, 0))
    return best


def _sm_count(model: newton.Model, sm_count: int | None) -> int:
    if sm_count is not None:
        return sm_count
    dev = wp.get_device(model.device)
    return int(dev.sm_count) if dev.is_cuda else 0


def _parallel_unit(n_worlds: int, islands_per_world: int, sm_count: int) -> str:
    """World-parallel when N >= 4 n_SM (or on CPU), island blocks when islands x N fill it, else colours."""
    need = WORLDS_PER_SM * sm_count
    if sm_count == 0 or n_worlds >= need:
        return "world"
    if islands_per_world * n_worlds >= need:
        return "island"
    return "color"
