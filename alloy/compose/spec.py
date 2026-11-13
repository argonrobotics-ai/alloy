# SPDX-License-Identifier: Apache-2.0
"""Data of the scene builder: what to build (``SceneSpec``, ``ItemSpec``), plain JSON in and out.

A ``SceneSpec`` is what ``alloy.compose.parse(sentence)`` returns and what ``alloy.compose.compose`` builds; an agent
can also write one directly (``SceneSpec.from_dict``) when it wants exact control.

Example::

    >>> spec = SceneSpec.from_dict({"items": [{"query": "mug"}, {"query": "cube", "count": 3}], "clutter": "tidy"})
    >>> spec.items[1].count, spec.support
    (3, 'table')
    >>> SceneSpec.from_dict(spec.to_dict()) == spec
    True
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

from alloy.compose.errors import ComposeError
from alloy.compose.soft.spec import SoftSpec

CLUTTER_LEVELS = ("tidy", "normal", "cluttered")
ROBOT_KINDS = ("mjcf", "urdf", "usd")
ARRANGEMENTS = ("scatter", "row", "grid")
RELATIONS = ("on", "in", "over")
SPEC_SCHEMA = "alloy.compose_spec/1"



def _whole(value: Any) -> bool:
    """An int that is not a bool (True is an int in Python). Example: ``_whole(True) is False``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(values: Any) -> bool:
    """Every value a real, finite number (no bool, NaN or Infinity). Example: ``_finite((0, 1e400)) is False``."""
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)

@dataclass(frozen=True)
class ItemSpec:
    """One kind of object to place.

    ``query`` is the noun ("mug") or an exact object card name; ``words`` are hints that rank matches (material
    "wooden", size "small"); ``count`` how many; ``relation``/``target``: placed ``on`` or ``in`` the item whose
    query is ``target`` (None: on the support); ``over``: a cloth draped over it. ``object`` pins an exact card (skips
    the search; for a liquid or grain, the material card). ``soft``: a liquid, grains, cloth or a cable the builder
    makes (``alloy.compose.soft.SoftSpec``).

    Example: ``ItemSpec("nut", count=3, relation="in", target="tray")``,
    ``ItemSpec("water", relation="in", target="cup", soft=SoftSpec("liquid", volume_ml=200))``."""

    query: str
    count: int = 1
    words: tuple[str, ...] = ()
    relation: Literal["on", "in"] | None = None
    target: str | None = None
    object: str | None = None
    soft: SoftSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ComposeError(f"item query {self.query!r} is empty", "give each item a noun, e.g. {'query': 'mug'}",
                               field="items.query")
        if not _whole(self.count) or self.count < 1:
            raise ComposeError(f"item {self.query!r}: count {self.count!r} is not a positive integer",
                               "pass count=1 or more", field="items.count")
        if self.relation not in (None, *RELATIONS):
            raise ComposeError(f"item {self.query!r}: relation {self.relation!r} is unknown",
                               f"use one of {RELATIONS} or leave it out", field="items.relation")
        if (self.relation is None) != (self.target is None):
            raise ComposeError(f"item {self.query!r}: relation and target go together",
                               "pass both (relation='in', target='tray') or neither", field="items.target")

        if self.relation == "over" and (self.soft is None or self.soft.kind != "cloth"):
            raise ComposeError(f"item {self.query!r}: only a cloth goes 'over' another item",
                               "use relation 'on' or 'in', or make the item a cloth (soft={'kind': 'cloth'})",
                               field="items.relation")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["words"] = list(self.words)
        d["soft"] = self.soft.to_dict() if self.soft is not None else None
        return {k: v for k, v in d.items() if v not in (None, [], ())}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ItemSpec:
        _known(data, cls, "items")
        soft = SoftSpec.from_dict(data["soft"]) if data.get("soft") is not None else None
        return cls(**{**data, "words": tuple(data.get("words", ())), "soft": soft})


@dataclass(frozen=True)
class RobotSpec:
    """A robot to add: its file, kind, scene name, optional base pose (x, y, z, yaw), the radius kept free around
    its base and the reach objects are kept within (m). Example: ``RobotSpec("arm.xml", reach=0.7)``."""

    path: str
    kind: str = "mjcf"
    name: str = "robot"
    pose: tuple[float, float, float, float] | None = None
    clear: float = 0.15
    reach: float = 0.8

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ComposeError("a robot needs a file path", "pass {'path': 'arm.xml'}", field="robots.path")
        if self.kind not in ROBOT_KINDS:
            raise ComposeError(f"robot kind {self.kind!r} is unknown", f"use one of {ROBOT_KINDS}", field="robots.kind")
        if self.pose is not None and (len(self.pose) != 4 or not _finite(self.pose)):
            raise ComposeError(f"robot pose {self.pose!r} needs 4 finite numbers", "pass pose=(x, y, z, yaw)",
                               field="robots.pose")
        if not _finite((self.clear, self.reach)) or not 0 < self.clear < self.reach:
            raise ComposeError(f"robot clear {self.clear} / reach {self.reach}", "use 0 < clear < reach (m)",
                               field="robots.reach")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["pose"] = list(self.pose) if self.pose is not None else None
        return {k: v for k, v in d.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotSpec:
        if isinstance(data, str):
            data = {"path": data}
        extra = sorted(set(data) - {f.name for f in fields(cls)})
        if extra:
            raise ComposeError(f"unknown robot key(s) {extra}", f"use only {[f.name for f in fields(cls)]}",
                               field=f"robots.{extra[0]}")
        pose = data.get("pose")
        return cls(**{**data, "pose": tuple(float(v) for v in pose) if pose is not None else None})


@dataclass(frozen=True)
class SceneSpec:
    """A whole request: items, the support they stand on, how cluttered, how many parallel worlds.

    ``support``: "table" (the library plywood table top, top face at z = 0), "floor", or an object card name.
    ``clutter``: "tidy" (wide gaps), "normal" or "cluttered" (tight gaps, smaller area). ``arrangement``: "scatter",
    "row" or "grid". ``jitter``: each of the ``worlds`` moves free objects by up to half the clutter gap in x and y
    (no new overlaps). ``robots``/``area``: see ``RobotSpec`` and ``alloy.compose.robots``. ``notes``: what the
    parser understood or ignored. Example: see the module docstring."""

    items: tuple[ItemSpec, ...]
    support: str = "table"
    clutter: str = "normal"
    arrangement: str = "scatter"
    worlds: int = 1
    robots: tuple[RobotSpec, ...] = ()
    area: tuple[float, float, float, float] | None = None
    jitter: bool = False
    sentence: str = ""
    notes: tuple[str, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if not self.items:
            raise ComposeError("the request names no objects", "name at least one object, e.g. 'a mug on a table'",
                               field="items")
        if self.clutter not in CLUTTER_LEVELS:
            raise ComposeError(f"clutter {self.clutter!r} is unknown", f"use one of {CLUTTER_LEVELS}", field="clutter")
        if self.arrangement not in ARRANGEMENTS:
            raise ComposeError(f"arrangement {self.arrangement!r} is unknown", f"use one of {ARRANGEMENTS}",
                               field="arrangement")
        if not _whole(self.worlds) or self.worlds < 1:
            raise ComposeError(f"worlds {self.worlds!r} is not a positive integer", "pass worlds=1 or more",
                               field="worlds")
        if self.area is not None and (len(self.area) != 4 or not _finite(self.area) or self.area[2] <= self.area[0]
                                      or self.area[3] <= self.area[1]):
            raise ComposeError(f"area {self.area!r} is not (x0, y0, x1, y1), finite, with x1 > x0 and y1 > y0",
                               "pass area=(-0.3, -0.2, 0.3, 0.2) (m, on the table, top face at z = 0)", field="area")
        names = [r.name for r in self.robots]
        if len(set(names)) != len(names):
            raise ComposeError(f"robot names {names} repeat", "give each robot its own name", field="robots.name")
        queries = {i.query for i in self.items}
        for item in self.items:
            if item.target is not None and item.target not in queries:
                raise ComposeError(f"item {item.query!r} is {item.relation} {item.target!r}, which is not an item",
                                   f"add an item with query {item.target!r} or use one of {sorted(queries)}",
                                   field="items.target")
            refs = (*item.soft.attach, *((item.soft.source,) if item.soft.source else ())) if item.soft else ()
            for ref in refs:
                if ref not in queries:
                    raise ComposeError(f"item {item.query!r} refers to {ref!r}, which is not an item",
                                       f"add an item with query {ref!r} or use one of {sorted(queries)}",
                                       field="items.soft")

    def total(self) -> int:
        """Number of objects asked for (support excluded)."""
        return sum(i.count for i in self.items)

    def to_dict(self) -> dict[str, Any]:
        out = {"schema": SPEC_SCHEMA, "items": [i.to_dict() for i in self.items], "support": self.support,
               "clutter": self.clutter, "arrangement": self.arrangement, "worlds": self.worlds,
               "sentence": self.sentence, "notes": list(self.notes)}
        if self.robots:
            out["robots"] = [r.to_dict() for r in self.robots]
        if self.area is not None:
            out["area"] = list(self.area)
        if self.jitter:
            out["jitter"] = True
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneSpec:
        """Build from JSON data (``to_dict`` output or hand-written; ``schema`` optional)."""
        if not isinstance(data, dict):
            raise ComposeError(f"a scene spec must be a JSON object, got {type(data).__name__}",
                               "pass {'items': [{'query': 'mug'}], 'support': 'table'}")
        data = {k: v for k, v in data.items() if k != "schema"}
        _known(data, cls, "")
        items = data.get("items")
        if not isinstance(items, list):
            raise ComposeError("spec 'items' must be a list", "pass items=[{'query': 'mug', 'count': 1}]",
                               field="items")
        area = data.get("area")
        return cls(**{**data, "items": tuple(ItemSpec.from_dict(_item(i)) for i in items),
                      "robots": tuple(RobotSpec.from_dict(r) for r in data.get("robots") or ()),
                      "area": tuple(float(v) for v in area) if area is not None else None,
                      "notes": tuple(data.get("notes", ()))})


def _item(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        return {"query": raw}
    if not isinstance(raw, dict):
        raise ComposeError(f"an item must be an object or a string, got {raw!r}", "pass {'query': 'mug'}",
                           field="items")
    return raw


def _known(data: dict[str, Any], cls: type, where: str) -> None:
    allowed = {f.name for f in fields(cls)}
    extra = sorted(set(data) - allowed)
    if extra:
        raise ComposeError(f"unknown spec key(s) {extra}{' in ' + where if where else ''}",
                           f"use only {sorted(allowed)}", field=f"{where}.{extra[0]}" if where else extra[0])


__all__ = ["ARRANGEMENTS", "CLUTTER_LEVELS", "ComposeError", "ItemSpec", "RELATIONS", "ROBOT_KINDS", "RobotSpec",
           "SPEC_SCHEMA", "SceneSpec"]
