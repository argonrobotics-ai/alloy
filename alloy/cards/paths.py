# SPDX-License-Identifier: Apache-2.0
"""Dotted card paths: one grammar for randomization fields, default-leaf lists and validation messages.

A path is relative to a placed object: ``loads.grip_force``, ``features.thickness``, ``shape.size.0`` (ObjectCard
fields; tuple items by index; the shape-source tag is not part of the path), ``material.surface.mu_static`` (the
object's material), ``part_materials.<part>.density`` and ``pose.<i>`` (the placement pose). A path ends at a
numeric leaf: a ``Q`` (its value is meant) or a float.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
import types
import typing
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from functools import cache
from typing import Any, Union, get_args, get_origin

from alloy.cards import types as ct
from alloy.cards.units_yaml import LeafKind, kind_of
from alloy.core.quantity import Q

_EXTRA_NS_MODULES = ("alloy.core.quantity", "alloy.topology.types", "alloy.cards.types")


@cache
def _extra_ns() -> dict[str, Any]:
    ns: dict[str, Any] = {}
    for name in _EXTRA_NS_MODULES:
        try:
            ns.update(vars(importlib.import_module(name)))
        except ImportError:  # pragma: no cover - only if a WP0 module is missing
            continue
    return ns


@cache
def type_hints(cls: type) -> dict[str, Any]:
    """Resolved annotations of a dataclass (string annotations and forward references included)."""
    module_ns = vars(sys.modules[cls.__module__])
    return typing.get_type_hints(cls, globalns={**_extra_ns(), **module_ns})


def union_args(tp: Any) -> tuple[Any, ...] | None:
    """Members of a ``Union``/``X | Y`` annotation, or None if ``tp`` is not a union."""
    if get_origin(tp) in (Union, types.UnionType):
        return get_args(tp)
    return None


def non_none(tp: Any) -> Any:
    """``X`` for ``X | None``; ``tp`` unchanged otherwise."""
    args = union_args(tp)
    if args is None:
        return tp
    rest = tuple(a for a in args if a is not type(None))
    return rest[0] if len(rest) == 1 else Union[rest]  # noqa: UP007 - runtime construction


@dataclass(frozen=True)
class CardView:
    """What a path addresses: an object card, its resolved materials and (optionally) its placement pose."""

    card: ct.ObjectCard
    material: ct.MaterialCard
    parts: Mapping[str, ct.MaterialCard] = dataclasses.field(default_factory=dict)
    pose: tuple[float, ...] | None = None


class PathError(KeyError):
    """A dotted path that does not name a numeric leaf; ``.path`` is the offending path."""

    def __init__(self, path: str, message: str) -> None:
        super().__init__(f"{path}: {message}")
        self.path, self.message = path, message


def _is_index(seg: str) -> bool:
    return seg.isascii() and seg.isdigit()


def split(path: str) -> list[str]:
    parts = path.split(".")
    if not path or any(p == "" for p in parts):
        raise PathError(path, "empty path segment")
    return parts


def _root(view: CardView, parts: list[str], path: str) -> tuple[Any, list[str]]:
    head = parts[0]
    if head == "material":
        return view.material, parts[1:]
    if head == "part_materials":
        if len(parts) < 3 or parts[1] not in view.parts:
            raise PathError(path, f"unknown part (parts: {sorted(view.parts)})")
        return view.parts[parts[1]], parts[2:]
    if head == "pose":
        if view.pose is None:
            raise PathError(path, "no placement pose in this context")
        return view.pose, parts[1:]
    return view.card, parts


def _child(node: Any, seg: str, path: str) -> Any:
    if isinstance(node, Q):
        raise PathError(path, f"{seg!r} below a quantity leaf (the path names the leaf itself)")
    if is_dataclass(node):
        if seg not in {f.name for f in fields(node)}:
            raise PathError(path, f"{type(node).__name__} has no field {seg!r}")
        return getattr(node, seg)
    if isinstance(node, tuple):
        if not _is_index(seg) or int(seg) >= len(node):
            raise PathError(path, f"index {seg!r} out of range for {len(node)} items")
        return node[int(seg)]
    if isinstance(node, Mapping):
        if seg not in node:
            raise PathError(path, f"no key {seg!r}")
        return node[seg]
    raise PathError(path, f"cannot descend into {type(node).__name__} at {seg!r}")


def get_value(view: CardView, path: str) -> float:
    """The numeric value at ``path`` (a ``Q`` leaf's value or a float)."""
    node, rest = _root(view, split(path), path)
    for seg in rest:
        node = _child(node, seg, path)
    if isinstance(node, Q):
        return node.value
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return float(node)
    raise PathError(path, f"not a numeric leaf ({type(node).__name__})")


def _replace(node: Any, segs: list[str], value: float, path: str) -> Any:
    if not segs:
        if isinstance(node, Q):
            return dataclasses.replace(node, value=float(value))
        if isinstance(node, (int, float)) and not isinstance(node, bool):
            return float(value)
        raise PathError(path, f"not a numeric leaf ({type(node).__name__})")
    seg, rest = segs[0], segs[1:]
    child = _child(node, seg, path)
    new = _replace(child, rest, value, path)
    if is_dataclass(node):
        return dataclasses.replace(node, **{seg: new})
    if isinstance(node, tuple):
        i = int(seg)
        return node[:i] + (new,) + node[i + 1:]
    return {**node, seg: new}


def with_value(view: CardView, path: str, value: float) -> CardView:
    """A copy of ``view`` with the leaf at ``path`` set to ``value`` (source and note kept)."""
    parts = split(path)
    head = parts[0]
    if head == "material":
        return dataclasses.replace(view, material=_replace(view.material, parts[1:], value, path))
    if head == "part_materials":
        _root(view, parts, path)
        new_part = _replace(view.parts[parts[1]], parts[2:], value, path)
        return dataclasses.replace(view, parts={**view.parts, parts[1]: new_part})
    if head == "pose":
        _root(view, parts, path)
        return dataclasses.replace(view, pose=_replace(view.pose, parts[1:], value, path))
    return dataclasses.replace(view, card=_replace(view.card, parts, value, path))


def _type_root(parts: list[str]) -> tuple[Any, list[str], str | None]:
    head = parts[0]
    if head == "material":
        return ct.MaterialCard, parts[1:], None
    if head == "part_materials":
        return ct.MaterialCard, parts[2:], None
    if head == "pose":
        return tuple[float, ...], parts[1:], "Placement.pose"
    return ct.ObjectCard, parts, None


def _type_child(tp: Any, seg: str, path: str) -> list[tuple[Any, str | None]]:
    """Possible (type, kind key) after descending ``seg`` from type ``tp`` (unions give several)."""
    tp = non_none(tp)
    members = union_args(tp) or (tp,)
    out: list[tuple[Any, str | None]] = []
    for m in members:
        if is_dataclass(m) and seg in type_hints(m):
            out.append((type_hints(m)[seg], f"{m.__name__}.{seg}"))
        elif get_origin(m) is tuple and _is_index(seg):
            args = get_args(m)
            item = args[0] if len(args) == 2 and args[1] is Ellipsis else (args[int(seg)] if int(seg) < len(args) else None)
            if item is not None:
                out.append((item, None))
        elif get_origin(m) in (Mapping, dict) or m in (Mapping, dict):
            out.append((get_args(m)[1] if get_args(m) else Any, None))
    if not out:
        raise PathError(path, f"no field {seg!r} here")
    return out


def kind_for_path(path: str) -> LeafKind | None:
    """Unit kind of the leaf a path names (type-level walk, so it works before a card exists)."""
    parts = split(path)
    tp, rest, key = _type_root(parts)
    frontier: list[tuple[Any, str | None]] = [(tp, key)]
    for seg in rest:
        nxt: list[tuple[Any, str | None]] = []
        for t, k in frontier:
            try:
                nxt.extend((ct2, k2 or k) for ct2, k2 in _type_child(t, seg, path))
            except PathError:
                continue
        if not nxt:
            raise PathError(path, f"no field {seg!r} here")
        frontier = nxt
    for t, k in frontier:
        leaf = non_none(t)
        if leaf is Q or leaf is float or get_origin(leaf) is tuple:
            if k is None:
                return None
            owner, field = k.split(".")
            return kind_of(owner, field)
    raise PathError(path, "not a numeric leaf")


def iter_q_leaves(node: Any, prefix: str = "") -> Iterator[tuple[str, Q]]:
    """Every ``Q`` below ``node`` with its dotted path (dataclass fields, tuple items, mapping values)."""
    if isinstance(node, Q):
        yield prefix, node
        return
    if is_dataclass(node):
        for f in fields(node):
            yield from iter_q_leaves(getattr(node, f.name), f"{prefix}.{f.name}" if prefix else f.name)
    elif isinstance(node, tuple):
        for i, item in enumerate(node):
            yield from iter_q_leaves(item, f"{prefix}.{i}" if prefix else str(i))
    elif isinstance(node, Mapping):
        for key in sorted(node):
            yield from iter_q_leaves(node[key], f"{prefix}.{key}" if prefix else str(key))


def view_leaves(view: CardView) -> Iterator[tuple[str, Q]]:
    """All ``Q`` leaves of a placed object, with object-relative paths."""
    yield from iter_q_leaves(view.card)
    yield from iter_q_leaves(view.material, "material")
    for part in sorted(view.parts):
        yield from iter_q_leaves(view.parts[part], f"part_materials.{part}")
