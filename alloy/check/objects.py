# SPDX-License-Identifier: Apache-2.0
"""Which placed object (or robot) owns each shape, body and particle of an assembled scene, so every finding can
name objects instead of indices.

Names follow the assembler: the first placement of an object is its name, the k-th repeat ``name#k``; robots by
their ``name``. ``placement`` maps a name to its index in ``SceneCard.objects`` (robots: absent).

Example::

    >>> idx = ObjectIndex.build(built, scene_card, lib)        # doctest: +SKIP
    >>> idx.names[idx.shape_obj[0]]
    'table_top_plywood'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("alloy.check")

KINDS = ("shape", "body", "particle")


def placement_names(scene: Any) -> list[str]:
    """Assembler names of ``scene.objects`` in order: ``name``, then ``name#1``, ``name#2`` for repeats.

    Example::

        >>> from types import SimpleNamespace as NS
        >>> placement_names(NS(objects=[NS(object="cube"), NS(object="cube"), NS(object="ball")]))
        ['cube', 'cube#1', 'ball']
    """
    counts: dict[str, int] = {}
    out = []
    for pl in scene.objects:
        k = counts.get(pl.object, 0)
        counts[pl.object] = k + 1
        out.append(pl.object if k == 0 else f"{pl.object}#{k}")
    return out


@dataclass
class ObjectIndex:
    """Owner tables for one ``BuiltScene``: ``shape_obj[s]``, ``body_obj[b]``, ``particle_obj[p]`` index ``names``
    (-1: no placed object, e.g. a ground plane). ``motion[name]`` is the card's motion ("dynamic", "static",
    "kinematic") or "robot".

    Example::

        >>> index = ObjectIndex.build(built, scene_card, lib)  # doctest: +SKIP
        >>> index.movable("mug"), index.world_count
        (True, 4)
    """

    names: list[str]
    motion: dict[str, str]
    placement: dict[str, int]
    shape_obj: np.ndarray
    body_obj: np.ndarray
    particle_obj: np.ndarray
    world_count: int
    poses: dict[str, tuple[float, ...]] = field(default_factory=dict)
    attached: set[frozenset[str]] = field(default_factory=set)

    @classmethod
    def build(cls, built: Any, scene: Any = None, lib: Any = None) -> ObjectIndex:
        """Owner tables from ``built.handles`` (template ranges replicated with each handle's world stride). Without
        ``scene``/``lib`` the assembler's own plan (``built._plan``: scene card and resolved object cards) is used.

        Example::

            >>> index = ObjectIndex.build(built)  # doctest: +SKIP
            >>> index.placement["mug"]
            1
        """
        plan = getattr(built, "_plan", None)
        scene = scene if scene is not None else getattr(plan, "scene", None)
        if scene is None:
            raise ValueError("ObjectIndex.build: pass the SceneCard (this BuiltScene carries no assembly plan)")
        m, w = built.model, int(built.world_count)
        names = list(built.handles)
        counts = {"shape": int(m.shape_count), "body": int(m.body_count), "particle": int(m.particle_count)}
        tables = {k: np.full(n, -1, np.int32) for k, n in counts.items()}
        for i, name in enumerate(names):
            h = built.handles[name]
            for kind in KINDS:
                ids = _global_ids(getattr(h.template, kind), len(getattr(h.world_stride, kind)), w)
                ids = ids[ids < counts[kind]]
                tables[kind][ids] = i
        placed = placement_names(scene)
        placement = {n: i for i, n in enumerate(placed)}
        motion = {n: _motion(lib, plan, scene, n, placement.get(n)) for n in names}
        poses = {n: tuple(float(v) for v in scene.objects[placement[n]].pose) for n in names if n in placement}
        poses.update({r.name: tuple(float(v) for v in r.base_pose) for r in scene.robots})
        log.debug("check objects: %d named, %d shapes unowned", len(names), int(np.sum(tables["shape"] < 0)))
        attached = {frozenset((_end(a.a), _end(a.b))) for a in scene.attachments}
        return cls(names, motion, placement, tables["shape"], tables["body"], tables["particle"], w, poses, attached)

    def name_of(self, kind: str, index: int) -> str | None:
        """The owner name of shape/body/particle ``index``, or None.

        Example::

            >>> index.name_of("shape", 1), index.name_of("shape", 99)  # doctest: +SKIP
            ('mug', None)
        """
        table = {"shape": self.shape_obj, "body": self.body_obj, "particle": self.particle_obj}[kind]
        if index < 0 or index >= table.size or table[index] < 0:
            return None
        return self.names[int(table[index])]

    def is_attached(self, a: str, b: str) -> bool:
        """True when a scene attachment joins ``a`` and ``b``.

        Example::

            >>> index.is_attached("mug", "table")  # doctest: +SKIP
            False
        """
        return frozenset((a, b)) in self.attached

    def movable(self, name: str | None) -> bool:
        """True for a dynamic placed object (one a pose edit should move).

        Example::

            >>> index.movable("mug"), index.movable("table")  # doctest: +SKIP
            (True, False)
        """
        return name is not None and self.motion.get(name) == "dynamic"

    def bodies_of(self, name: str) -> np.ndarray:
        """Global body ids of ``name`` in every world.

        Example::

            >>> index.bodies_of("mug")  # doctest: +SKIP
            array([1])
        """
        return np.flatnonzero(self.body_obj == self.names.index(name))

    def dynamic_names(self) -> list[str]:
        """Dynamic placed objects that own bodies or particles (what rest-drift checks watch).

        Example::

            >>> index.dynamic_names()  # doctest: +SKIP
            ['mug']
        """
        owned = set(self.body_obj[self.body_obj >= 0].tolist())
        owned |= set(self.particle_obj[self.particle_obj >= 0].tolist())
        return [n for i, n in enumerate(self.names) if i in owned and self.motion.get(n) == "dynamic"]


def _global_ids(rng: range, stride: int, worlds: int) -> np.ndarray:
    if len(rng) == 0:
        return np.zeros(0, np.int64)
    base = np.arange(rng.start, rng.stop, dtype=np.int64)
    return (base[None, :] + stride * np.arange(worlds, dtype=np.int64)[:, None]).ravel()


def _end(name: str) -> str:
    """An attachment end ("object" or "object:link") as a placement name."""
    return name.split(":", 1)[0]


def _motion(lib: Any, plan: Any, scene: Any, name: str, placement: int | None) -> str:
    """The card motion of a placement ("robot" for robots); from the library, else the assembly plan."""
    if placement is None:
        return "robot"
    try:
        if lib is not None:
            return str(lib.peek_object(scene.objects[placement].object).motion)
        resolved = getattr(plan, "resolved", {})
        res = resolved.get(name) or resolved.get(scene.objects[placement].object)
        return str(res.card.motion) if res is not None else "dynamic"
    except Exception as exc:  # noqa: BLE001 - a card error is reported by the cards check
        log.debug("check objects: motion of placement %d unknown (%s)", placement, exc)
        return "dynamic"


__all__ = ["ObjectIndex", "placement_names"]
