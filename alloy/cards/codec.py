# SPDX-License-Identifier: Apache-2.0
"""Type-hint-driven conversion between card dataclasses and plain YAML data.

Decoding is strict: unknown keys, missing required fields, wrong types and values outside a ``Literal`` raise
``CardError`` with the dotted field path. A ``Q`` is written either as a mapping ``{value, source, unc, note}`` or
as a bare number/unit string, which means ``source: default``. The shape source is a one-key mapping whose key
names the variant (``primitive``, ``mesh``, ``sdf``, ``tet``, ``curve``, ``particles``, ``usd``, ``mjcf``); field
paths skip that key (``shape.size``, not ``shape.primitive.size``). Encoding omits fields equal to their default.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import MISSING, Field, dataclass, fields, is_dataclass
from typing import Any, Literal, get_args, get_origin

from alloy.cards import types as ct
from alloy.cards.paths import PathError, kind_for_path, non_none, type_hints, union_args
from alloy.cards.units_yaml import FlowList, LeafKind, kind_of, parse_number
from alloy.core.errors import CardError
from alloy.core.quantity import Q

SHAPE_TAGS: dict[str, type] = {
    "primitive": ct.Primitive, "mesh": ct.MeshSource, "sdf": ct.SDFSource, "tet": ct.TetSource,
    "curve": ct.CurveSource, "particles": ct.ParticleSource, "usd": ct.USDSource, "mjcf": ct.MJCFSource,
}
TAG_OF: dict[type, str] = {cls: tag for tag, cls in SHAPE_TAGS.items()}
Q_KEYS = ("value", "source", "unc", "note")
SOURCES: tuple[str, ...] = ("measured", "catalog", "fit", "model", "default")
_PCT = re.compile(r"^\s*(?:\+-|±)?\s*([0-9]*\.?[0-9]+)\s*%\s*$")


@dataclass(frozen=True)
class Ctx:
    """Decoding context: the card name used in every ``CardError``."""

    card: str

    def err(self, path: str, message: str) -> CardError:
        return CardError(self.card, path or "<card>", message)


def join(path: str, name: str | int) -> str:
    return f"{path}.{name}" if path else str(name)


# ----------------------------------------------------------------------------------------------- decoding


def decode(tp: Any, raw: Any, path: str, ctx: Ctx, kind: LeafKind | None = None) -> Any:
    """Convert YAML data ``raw`` into an instance of annotation ``tp``."""
    if tp is Any:
        return raw
    if tp is Q:
        return _decode_q(raw, path, ctx, kind)
    if is_dataclass(tp):
        return _decode_dataclass(tp, raw, path, ctx)
    if union_args(tp) is not None:
        return _decode_union(tp, raw, path, ctx, kind)
    origin = get_origin(tp)
    if origin is Literal:
        if raw not in get_args(tp) or isinstance(raw, bool):
            raise ctx.err(path, f"must be one of {list(get_args(tp))}, got {raw!r}")
        return raw
    if origin is tuple:
        return _decode_tuple(tp, raw, path, ctx, kind)
    if origin in (Mapping, dict):
        return _decode_mapping(tp, raw, path, ctx)
    return _decode_scalar(tp, raw, path, ctx, kind)


def _decode_scalar(tp: Any, raw: Any, path: str, ctx: Ctx, kind: LeafKind | None) -> Any:
    if tp is float:
        return parse_number(raw, kind, ctx.card, path)
    if tp is int:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ctx.err(path, f"expected an integer, got {raw!r}")
        return raw
    if tp is bool:
        if not isinstance(raw, bool):
            raise ctx.err(path, f"expected true/false, got {raw!r}")
        return raw
    if tp is str:
        if not isinstance(raw, str):
            raise ctx.err(path, f"expected a string, got {type(raw).__name__} {raw!r}")
        return raw
    raise TypeError(f"alloy.cards.codec: unsupported annotation {tp!r} at {path}")


def _field_kind(cls: type, f: Field, raw: Mapping[str, Any], path: str, ctx: Ctx) -> LeafKind | None:
    if cls is ct.Randomization and f.name in ("lo", "hi"):
        target = raw.get("field")
        if not isinstance(target, str):
            return None
        try:
            return kind_for_path(target)
        except PathError as exc:
            raise ctx.err(join(path, "field"), f"unknown card path: {exc.message}") from exc
    return kind_of(cls.__name__, f.name)


def _decode_dataclass(cls: type, raw: Any, path: str, ctx: Ctx) -> Any:
    if not isinstance(raw, Mapping):
        raise ctx.err(path, f"expected a mapping for {cls.__name__}, got {type(raw).__name__} {raw!r}")
    hints = type_hints(cls)
    names = [f.name for f in fields(cls)]
    for key in raw:
        if key not in names:
            raise ctx.err(join(path, key), f"unknown key for {cls.__name__} (allowed: {', '.join(names)})")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        fpath = join(path, f.name)
        if f.name not in raw:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ctx.err(fpath, "missing required field")
            continue
        kwargs[f.name] = decode(hints[f.name], raw[f.name], fpath, ctx, _field_kind(cls, f, raw, path, ctx))
    try:
        return cls(**kwargs)
    except (ValueError, TypeError) as exc:
        raise ctx.err(path, f"{cls.__name__}: {exc}") from exc


def _decode_unc(raw: Any, value: float, path: str, ctx: Ctx, kind: LeafKind | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    if isinstance(raw, str) and (m := _PCT.match(raw)):
        frac = float(m.group(1)) / 100.0
        lo, hi = value * (1.0 - frac), value * (1.0 + frac)
        return (min(lo, hi), max(lo, hi))
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ctx.err(path, f"unc must be [lo, hi] (absolute, SI) or 'N%', got {raw!r}")
    return (parse_number(raw[0], kind, ctx.card, join(path, 0)), parse_number(raw[1], kind, ctx.card, join(path, 1)))


def _decode_q(raw: Any, path: str, ctx: Ctx, kind: LeafKind | None) -> Q:
    if not isinstance(raw, Mapping):
        return _make_q(parse_number(raw, kind, ctx.card, path), "default", None, "", path, ctx)
    for key in raw:
        if key not in Q_KEYS:
            raise ctx.err(join(path, key), f"unknown key for a quantity (allowed: {', '.join(Q_KEYS)})")
    if "value" not in raw:
        raise ctx.err(join(path, "value"), "missing required field")
    if "source" not in raw:
        raise ctx.err(join(path, "source"), f"missing required field (one of {list(SOURCES)})")
    value = parse_number(raw["value"], kind, ctx.card, join(path, "value"))
    source = raw["source"]
    if source not in SOURCES:
        raise ctx.err(join(path, "source"), f"must be one of {list(SOURCES)}, got {source!r}")
    note = raw.get("note", "")
    if not isinstance(note, str):
        raise ctx.err(join(path, "note"), f"expected a string, got {note!r}")
    unc = _decode_unc(raw.get("unc"), value, join(path, "unc"), ctx, kind)
    return _make_q(value, source, unc, note, path, ctx)


def _make_q(value: float, source: str, unc: tuple[float, float] | None, note: str, path: str, ctx: Ctx) -> Q:
    try:
        return Q(value, source, unc, note)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise ctx.err(path, str(exc)) from exc


def _decode_union(tp: Any, raw: Any, path: str, ctx: Ctx, kind: LeafKind | None) -> Any:
    args = union_args(tp) or ()
    members = tuple(a for a in args if a is not type(None))
    if raw is None:
        if len(members) < len(args):
            return None
        raise ctx.err(path, "null is not allowed here")
    if len(members) == 1:
        return decode(members[0], raw, path, ctx, kind)
    if all(m in TAG_OF for m in members):
        return _decode_tagged(members, raw, path, ctx)
    raise TypeError(f"alloy.cards.codec: untagged union {tp!r} at {path}")


def _decode_tagged(members: tuple[type, ...], raw: Any, path: str, ctx: Ctx) -> Any:
    tags = [TAG_OF[m] for m in members]
    if not isinstance(raw, Mapping) or len(raw) != 1:
        raise ctx.err(path, f"expected a one-key mapping naming the variant, one of {tags}")
    (tag, body), = raw.items()
    if tag not in tags:
        raise ctx.err(join(path, tag), f"unknown variant (allowed: {', '.join(tags)})")
    return decode(SHAPE_TAGS[tag], body, path, ctx)


def _decode_tuple(tp: Any, raw: Any, path: str, ctx: Ctx, kind: LeafKind | None) -> tuple[Any, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ctx.err(path, f"expected a list, got {type(raw).__name__} {raw!r}")
    args = get_args(tp)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(decode(args[0], x, join(path, i), ctx, kind) for i, x in enumerate(raw))
    if len(raw) != len(args):
        raise ctx.err(path, f"expected {len(args)} items, got {len(raw)}")
    return tuple(decode(a, x, join(path, i), ctx, kind) for i, (a, x) in enumerate(zip(args, raw)))


def _decode_mapping(tp: Any, raw: Any, path: str, ctx: Ctx) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ctx.err(path, f"expected a mapping, got {type(raw).__name__} {raw!r}")
    args = get_args(tp)
    vtype = args[1] if len(args) == 2 else Any
    return {str(k): decode(vtype, v, join(path, k), ctx) for k, v in raw.items()}


# ----------------------------------------------------------------------------------------------- encoding


def same(a: Any, b: Any) -> bool:
    """Equal and with the same repr (so -0.0 and 0.0, or 1 and 1.0, are different defaults)."""
    return type(a) is type(b) and a == b and repr(a) == repr(b)


def _default_of(f: Field) -> Any:
    if f.default is not MISSING:
        return f.default
    if f.default_factory is not MISSING:
        return f.default_factory()
    return MISSING


def _item_type(tp: Any, i: int) -> Any:
    args = get_args(non_none(tp)) if tp is not None else ()
    if len(args) == 2 and args[1] is Ellipsis:
        return args[0]
    return args[i] if i < len(args) else None


def encode(obj: Any, tp: Any = None) -> Any:
    """Plain YAML data for a card value; inverse of ``decode``. ``tp`` (the declared annotation) turns ints on
    float fields into floats, so a programmatic card and its YAML twin encode, and hash, the same."""
    if isinstance(obj, Q):
        return _encode_q(obj)
    if is_dataclass(obj):
        return _encode_dataclass(obj)
    if isinstance(obj, tuple):
        items = [encode(x, _item_type(tp, i)) for i, x in enumerate(obj)]
        flat = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in items)
        return FlowList(items) if flat else items
    if isinstance(obj, Mapping):
        vt = get_args(non_none(tp))[1] if tp is not None and len(get_args(non_none(tp))) == 2 else None
        return {str(k): encode(obj[k], vt) for k in sorted(obj)}
    if isinstance(obj, int) and not isinstance(obj, bool) and tp is not None and non_none(tp) is float:
        return float(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"alloy.cards.codec: cannot encode {type(obj).__name__}")


def normalize(card: Any) -> Any:
    """The card as its YAML round trip would give it (ints on float fields become floats); hashing input."""
    return decode(type(card), encode(card), "", Ctx(getattr(card, "name", type(card).__name__)))


def _encode_q(q: Q) -> Any:
    if q.source == "default" and q.unc is None and not q.note and math.isfinite(q.value):
        return float(q.value)
    out: dict[str, Any] = {"value": float(q.value), "source": q.source}
    if q.unc is not None:
        out["unc"] = FlowList([float(q.unc[0]), float(q.unc[1])])
    if q.note:
        out["note"] = q.note
    return out


def _encode_dataclass(obj: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in fields(obj):
        value = getattr(obj, f.name)
        default = _default_of(f)
        if default is not MISSING and same(value, default):
            continue
        enc = encode(value, type_hints(type(obj))[f.name])
        out[f.name] = {TAG_OF[type(value)]: enc} if type(value) in TAG_OF else enc
    return out


def field_type(cls: type, name: str) -> Any:
    """Resolved annotation of ``cls.name`` without ``| None``."""
    return non_none(type_hints(cls)[name])
