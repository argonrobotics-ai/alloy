# SPDX-License-Identifier: Apache-2.0
"""The scene as the variants see it: one CPU assembly (no solvers) with its start state, owner tables and a contact
probe (``alloy.check``), which free objects move, and which move together.

- Free objects: dynamic placed objects that are not robots, fixtures (static or kinematic parts), liquids or pours
  (a liquid fills its container and stays with it).
- Groups: free objects that touch at the start (a block stacked on a block, a spoon in a cup) or are attached move
  as one rigid group, so what rests on or in a free object keeps resting there.
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

FIXED_KINDS = frozenset({"robot", "liquid", "ground"})
TOUCH_M = 5.0e-4    # [m] two objects closer than this at the start touch (resting contact)


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
        if item.get("kind") in FIXED_KINDS or item.get("pour") or item.get("static") is True:
            continue
        if (item.get("card_extra") or {}).get("motion") == "kinematic":
            continue
        if name in base.index.names and base.index.movable(name):
            out.append(name)
    return out


def groups(names: list[str], base: Base) -> list[list[str]]:
    """``names`` split into rigid groups: objects that touch at the start or are attached share a group."""
    parent = {n: n for n in names}

    def root(n: str) -> str:
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    linked = {frozenset((a, b)) for a, b, _ in touching(base, base.state())}
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


__all__ = ["Base", "TOUCH_M", "assemble", "free_objects", "groups", "touching"]
