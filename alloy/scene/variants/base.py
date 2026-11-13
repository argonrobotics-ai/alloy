# SPDX-License-Identifier: Apache-2.0
"""The scene as the variants see it: one CPU assembly (no solvers) with its start state, owner tables and a contact
probe (``alloy.check``), which free objects move, and which move together.

- Free objects: dynamic placed objects that are not robots, fixtures (static or kinematic parts) or fills.
- Fills (liquids, grains, pastes and pours) never move alone: one in or on a free object at the start (water in a
  free mug; for a pour, the free object under its nozzle) moves with it, any other stays.
- Groups: free objects that touch at the start (a block stacked on a block, a spoon in a cup) or are attached move
  as one rigid group, so what rests on or in a free object keeps resting there. Particle items (fills, cloth, soft
  bodies) touch by geometry (``particle_links``): in an object's box, or their bottom layer just above its top.
"""

from __future__ import annotations

import copy
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alloy.scene.variants.xform import to_world

log = logging.getLogger("alloy.scene")

FIXED_KINDS = frozenset({"robot", "ground"})
FILL_ROUTES = frozenset({"liquid_sph", "liquid_mpm", "grain_mpm", "paste_mpm"})
TOUCH_M = 5.0e-4    # [m] two objects closer than this at the start touch (resting contact)
ON_GAP = 5.0e-3     # [m] a particle item's bottom this close to an object's top (or inside its box) rests on it
IN_SHARE = 0.5      # a particle item with this share of its points in an object's box is in it
LAYER = 5.0e-3      # [m] a particle item's bottom layer: its points this close to its lowest one


@dataclass
class Base:
    """One assembly of the scene: ``built`` (BuiltScene), ``lib`` (its CardLibrary), ``index`` (ObjectIndex),
    ``probe`` (ContactProbe), the start ``body_q`` / ``particle_q`` and the start poses (x, y, z, qx, qy, qz, qw)."""

    built: Any
    lib: Any
    index: Any
    probe: Any
    body_q: np.ndarray
    particle_q: np.ndarray
    poses: dict[str, np.ndarray]
    workdir: Any = field(default=None, repr=False)     # keeps the materialized cards alive

    def state(self) -> Any:
        """A fresh start state of the model (the probe and the camera write into it)."""
        from alloy.check.overlap import initial_state  # noqa: PLC0415

        return initial_state(self.built.model)


def assemble(scene: Any) -> Base:
    """Materialize ``scene`` (an ``alloy.Scene``) with one world and assemble it on the CPU (no solvers)."""
    from alloy.api import catalog  # noqa: PLC0415
    from alloy.api._materialize import materialize  # noqa: PLC0415
    from alloy.cards.library import CardLibrary  # noqa: PLC0415
    from alloy.cards.roots import assets_root  # noqa: PLC0415
    from alloy.check.contacts import ContactProbe  # noqa: PLC0415
    from alloy.check.objects import ObjectIndex  # noqa: PLC0415
    from alloy.scene.assembler import SceneAssembler  # noqa: PLC0415

    one = copy.copy(scene)
    one.worlds = 1
    work = tempfile.TemporaryDirectory(prefix="alloy-variants-")
    mat = materialize(one, Path(work.name))
    lib = CardLibrary(roots=catalog.card_roots([work.name, *scene.roots]), assets=assets_root())
    card = lib.scene(mat.scene_file.stem)
    built = SceneAssembler(lib).assemble(card, "cpu", world_count=1)
    index = ObjectIndex.build(built, card, lib)
    base = Base(built, lib, index, ContactProbe(built), np.zeros((0, 7)), np.zeros((0, 3)),
                {n: np.asarray(p, np.float64) for n, p in index.poses.items()}, work)
    state = base.state()
    m = built.model
    base.body_q = state.body_q.numpy().astype(np.float64) if m.body_count else np.zeros((0, 7))
    base.particle_q = state.particle_q.numpy().astype(np.float64) if m.particle_count else np.zeros((0, 3))
    log.info("variants: base scene assembled (%d bodies, %d shapes, %d particles)", m.body_count, m.shape_count,
             m.particle_count)
    return base


def free_objects(scene: Any, base: Base) -> list[str]:
    """Names of the scene's free objects (module docstring), in scene order."""
    out = []
    for name, item in scene.items.items():
        if item.get("kind") in FIXED_KINDS or item.get("static") is True or is_fill(name, item, base):
            continue
        if (item.get("card_extra") or {}).get("motion") == "kinematic":
            continue
        if name in base.index.names and base.index.movable(name):
            out.append(name)
    return out


def is_fill(name: str, item: dict[str, Any], base: Base) -> bool:
    """A liquid, grain or paste item (``Scene.add_liquid`` / ``add_pour``, or a card on a liquid, grain or paste
    route): it moves only with what it is in or on."""
    if item.get("kind") == "liquid" or item.get("pour"):
        return True
    d = base.built.report.decisions.get(name)
    return d is not None and str(d.route) in FILL_ROUTES


@dataclass
class ParticleLinks:
    """What the scene's particle items rest in or on at the start (``particle_links``)."""

    carrier: dict[str, str]                         # particle item -> the object it is in or on (a pour: under)
    points: dict[str, np.ndarray]                   # particle item -> its start points (n, 3)
    boxes: dict[str, tuple[np.ndarray, np.ndarray]]  # object with collision shapes -> its start world box
    riders: set[frozenset] = field(default_factory=set)  # {rigid object, particle item}: the object rests on it


def particle_links(scene: Any, base: Base) -> ParticleLinks:
    """The carrier of every particle item (fills, cloth, soft bodies; ``rest_of``) and the rigid objects resting on
    one (a cup on a towel), from the start state's geometry."""
    boxes = owner_boxes(base)
    links = ParticleLinks({}, {}, boxes)
    for name, item in scene.items.items():
        k = base.index.names.index(name) if name in base.index.names else -1
        has = k >= 0 and len(base.particle_q) and bool(np.any(base.index.particle_obj == k))
        if not (has or is_fill(name, item, base)):
            continue
        pts = base.particle_q[base.index.particle_obj == k] if has else \
            np.asarray(base.poses.get(name, item.get("pose") or [0.0, 0.0, 0.0]), np.float64)[None, :3]
        links.points[name] = pts
        nozzle = (item.get("pour") or {}).get("pos")
        carrier = under(np.asarray(nozzle, np.float64), boxes) if nozzle is not None else \
            rest_of(pts, {o: b for o, b in boxes.items() if o != name})
        if carrier is not None:
            links.carrier[name] = carrier
        for other, (lo, hi) in boxes.items():
            below = pts[np.all((pts[:, :2] >= lo[:2]) & (pts[:, :2] <= hi[:2]), axis=1)]
            if other != name and len(below) and abs(lo[2] - below[:, 2].max()) <= ON_GAP:
                links.riders.add(frozenset((other, name)))
    return links


def owner_boxes(base: Base) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World box of every object with collision shapes at the start (planes have none)."""
    from alloy.check.bounds import shape_aabb  # noqa: PLC0415

    m, out = base.built.model, {}
    for s in range(int(m.shape_count)):
        name = base.index.name_of("shape", s)
        box = shape_aabb(m, base.body_q, s) if name is not None else None
        if box is not None:
            lo, hi = out.get(name, box)
            out[name] = (np.minimum(lo, box[0]), np.maximum(hi, box[1]))
    return out


def rest_of(pts: np.ndarray, boxes: dict[str, tuple[np.ndarray, np.ndarray]]) -> str | None:
    """The object a particle item at ``pts`` is in (IN_SHARE of its points in the object's box widened by ON_GAP;
    the smallest such box) or else on (its bottom layer's centre over the box, within ON_GAP of its top; the
    highest), or None."""
    bottom = pts[pts[:, 2] <= pts[:, 2].min() + LAYER]
    zmin, cxy = float(pts[:, 2].min()), bottom[:, :2].mean(axis=0)
    inside, on = [], []
    for name, (lo, hi) in boxes.items():
        share = float(np.mean(np.all((pts >= lo - ON_GAP) & (pts <= hi + ON_GAP), axis=1)))
        if share >= IN_SHARE:
            inside.append((float(np.prod(hi - lo)), name))
        elif abs(zmin - hi[2]) <= ON_GAP and np.all(cxy >= lo[:2]) and np.all(cxy <= hi[:2]):
            on.append((-float(hi[2]), name))
    return min(inside)[1] if inside else min(on)[1] if on else None


def under(point: np.ndarray, boxes: dict[str, tuple[np.ndarray, np.ndarray]]) -> str | None:
    """The highest object whose box holds ``point``'s x, y below it (a pour's target), or None."""
    hits = [(-float(hi[2]), n) for n, (lo, hi) in boxes.items()
            if np.all(point[:2] >= lo[:2]) and np.all(point[:2] <= hi[:2]) and hi[2] <= point[2]]
    return min(hits)[1] if hits else None


def groups(names: list[str], base: Base, extra: set[frozenset] = frozenset()) -> list[list[str]]:
    """``names`` split into rigid groups: objects that touch at the start, are attached, or form a pair in
    ``extra`` (particle links) share a group."""
    parent = {n: n for n in names}

    def root(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    linked = {frozenset((a, b)) for a, b, _ in touching(base, base.state())} | set(extra)
    for a in names:
        for b in names:
            if a < b and (frozenset((a, b)) in linked or base.index.is_attached(a, b)):
                parent[root(a)] = root(b)
    out: dict[str, list[str]] = {}
    for n in names:
        out.setdefault(root(n), []).append(n)
    return list(out.values())


def touching(base: Base, state: Any, contacts: Any = None) -> list[tuple[str, str, np.ndarray]]:
    """(a, b, unit normal from a to b) for every rigid contact between two owners closer than ``TOUCH_M`` at
    ``state`` (overlaps included): resting contacts as well as penetrations. ``contacts``: the probe's buffer already
    collided at ``state`` (else it is collided here)."""
    from alloy.check.contacts import GROUND  # noqa: PLC0415

    c = contacts if contacts is not None else base.probe.collide(state)
    n = int(c.rigid_contact_count.numpy()[0]) if c.rigid_contact_max else 0
    if n == 0:
        return []
    m = base.built.model
    s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
    body_q = state.body_q.numpy() if m.body_count else np.zeros((0, 7))
    x0 = to_world(body_q, m.shape_body.numpy()[s0], c.rigid_contact_point0.numpy()[:n])
    x1 = to_world(body_q, m.shape_body.numpy()[s1], c.rigid_contact_point1.numpy()[:n])
    nrm = c.rigid_contact_normal.numpy()[:n].astype(np.float64)
    margins = c.rigid_contact_margin0.numpy()[:n] + c.rigid_contact_margin1.numpy()[:n]
    gap = np.einsum("ij,ij->i", nrm, x1 - x0) - margins
    out = []
    for k in np.flatnonzero(gap < TOUCH_M):
        a = base.index.name_of("shape", int(s0[k])) or GROUND
        b = base.index.name_of("shape", int(s1[k])) or GROUND
        if a != b:
            out.append((a, b, nrm[k]))
    return out


__all__ = ["Base", "ParticleLinks", "TOUCH_M", "assemble", "free_objects", "groups", "is_fill", "particle_links",
           "rest_of", "touching", "under"]
