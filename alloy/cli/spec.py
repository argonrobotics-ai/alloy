# SPDX-License-Identifier: Apache-2.0
"""Operation specs: one table of parameters per operation gives the JSON Schema (tool server), the command-line
flags and the request checking, so the three never disagree. Standard library only.

Example::

    from alloy.cli.spec import Op, Param, check_request, json_schema
    op = Op("echo", "Return the text.", (Param("text", "string", "what to return", required=True),), handler=None)
    json_schema(op)                       # {"type": "object", "properties": {"text": {...}}, "required": ["text"]}
    check_request(op, {"text": "hi"})     # {"text": "hi"}
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from alloy.cli.errors import RequestError

#: parameter type -> JSON Schema fragment
_SCHEMAS: dict[str, dict[str, Any]] = {
    "string": {"type": "string"},
    "integer": {"type": "integer"},
    "number": {"type": "number"},
    "boolean": {"type": "boolean"},
    "object": {"type": "object"},
    "strings": {"type": "array", "items": {"type": "string"}},
    "numbers": {"type": "array", "items": {"type": "number"}},
    "scene": {"type": ["string", "object"]},
    "world": {"anyOf": [{"type": "integer", "minimum": 0}, {"const": "all"}]},
    "any": {},
}


@dataclass(frozen=True)
class Param:
    """One argument of an operation. ``kind`` is a key of ``_SCHEMAS``; ``positional`` makes it the first
    command-line argument; ``schema`` overrides the generated JSON Schema fragment.

    Example: ``Param("worlds", "integer", "parallel copies", minimum=1)``."""

    name: str
    kind: str
    help: str
    default: Any = None
    required: bool = False
    positional: bool = False
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    schema: dict[str, Any] | None = None

    def req(self) -> Param:
        """The same parameter, required. Example: ``SCENE.req()``."""
        return replace(self, required=True)

    def json_schema(self) -> dict[str, Any]:
        """The JSON Schema of this argument. Example: ``Param("n", "integer", "count").json_schema()``."""
        out = dict(self.schema) if self.schema is not None else dict(_SCHEMAS[self.kind])
        out["description"] = self.help
        if self.choices:
            out["enum"] = list(self.choices)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.default is not None:
            out["default"] = self.default
        return out


@dataclass(frozen=True)
class Op:
    """One operation: name, one-line summary, longer description, parameters and the handler
    ``handler(session, args) -> dict``. ``read_only``: changes nothing outside the process; ``engine``: the call
    imports the engine; ``image``: the result key holding a PNG path that the tool server also returns as an image.

    Example: ``Op("list", "List library cards.", (Param("kind", "string", "what"),), handler=list_cards)``."""

    name: str
    summary: str
    params: tuple[Param, ...]
    handler: Callable[..., dict[str, Any]] | None
    description: str = ""
    example: dict[str, Any] = field(default_factory=dict)
    read_only: bool = False
    engine: bool = True             # imports the engine (Warp): its start-up logs are quieted first
    image: str = ""                 # result key of a PNG path sent inline by the MCP server ("" none)

    def param(self, name: str) -> Param | None:
        """The parameter called ``name``, or None. Example: ``op.param("worlds")``."""
        return next((p for p in self.params if p.name == name), None)


def json_schema(op: Op) -> dict[str, Any]:
    """The input JSON Schema of ``op`` (an object, no extra keys). Example: ``json_schema(OPS["step"])``."""
    required = [p.name for p in op.params if p.required]
    out: dict[str, Any] = {"type": "object", "properties": {p.name: p.json_schema() for p in op.params},
                           "additionalProperties": False}
    if required:
        out["required"] = required
    return out


# --- request checking ------------------------------------------------------------------------------------------

def check_request(op: Op, args: dict[str, Any] | None) -> dict[str, Any]:
    """``args`` checked against ``op``: unknown keys and wrong types raise ``RequestError`` (with a fix); missing
    optional keys get their defaults. Example: ``check_request(OPS["step"], {"steps": 10})``."""
    args = {} if args is None else args
    if not isinstance(args, dict):
        raise RequestError(f"{op.name}: arguments must be a JSON object, got {type(args).__name__}",
                           f'pass {{"{op.params[0].name}": ...}}' if op.params else "pass {}")
    names = [p.name for p in op.params]
    unknown = [k for k in args if k not in names]
    if unknown:
        raise RequestError(f"{op.name}: unknown argument(s) {unknown}", f"use only {names} (alloy ops {op.name} "
                           "shows each)", field=unknown[0])
    out: dict[str, Any] = {}
    for p in op.params:
        if p.name not in args or args[p.name] is None:
            if p.required:
                raise RequestError(f"{op.name}: missing argument {p.name!r} ({p.help})",
                                   f"pass {p.name}=<{p.kind}>", field=p.name)
            out[p.name] = p.default
            continue
        out[p.name] = _check_value(op, p, args[p.name])
    return out


def _check_value(op: Op, p: Param, value: Any) -> Any:
    ok = _TYPE_OK.get(p.kind, lambda v: True)(value)
    if not ok:
        raise RequestError(f"{op.name}: {p.name} must be {p.kind}, got {value!r}"[:300], _fix_for(p), field=p.name)
    if p.kind == "number":
        value = float(value)
        if not math.isfinite(value):
            raise RequestError(f"{op.name}: {p.name} must be finite, got {value!r}", _fix_for(p), field=p.name)
    if p.choices and value not in p.choices:
        raise RequestError(f"{op.name}: {p.name} must be one of {list(p.choices)}, got {value!r}",
                           f"pass {p.name}={p.choices[0]!r}", field=p.name)
    if p.minimum is not None and isinstance(value, (int, float)) and value < p.minimum:
        raise RequestError(f"{op.name}: {p.name} must be >= {p.minimum}, got {value!r}",
                           f"pass {p.name}={p.default if p.default is not None else p.minimum}", field=p.name)
    return value


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_TYPE_OK: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": _is_int,
    "number": _is_number,
    "boolean": lambda v: isinstance(v, bool),
    "object": lambda v: isinstance(v, dict),
    "strings": lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v),
    "numbers": lambda v: isinstance(v, list) and all(_is_number(x) for x in v),
    "scene": lambda v: isinstance(v, (str, dict)),
    "world": lambda v: v == "all" or (_is_int(v) and v >= 0),
}


def _fix_for(p: Param) -> str:
    examples = {"string": '"text"', "integer": "10", "number": "0.5", "boolean": "true", "object": '{"key": 1}',
                "strings": '["a", "b"]', "numbers": "[0.1, 0.2]", "scene": '"demo_peg_d12" or {"objects": [...]}',
                "world": '0 (a world index) or "all"'}
    return f"pass {p.name}={examples.get(p.kind, 'a JSON value')} ({p.help})"


# --- command line ----------------------------------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser, op: Op) -> None:
    """Flags for ``op`` on ``parser``: ``--worlds 4``, ``--record cube,peg``, ``--targets '{"j": 0.1}'``,
    booleans as ``--quick / --no-quick``; a positional list takes several words (``alloy fetch a b``); values are
    parsed by ``parse_flag``. Example: see ``alloy.cli.command``."""
    for p in op.params:
        flag = "--" + p.name.replace("_", "-")
        if p.kind == "boolean":
            parser.add_argument(flag, dest=p.name, action=argparse.BooleanOptionalAction, default=None, help=p.help)
            continue
        if p.positional:
            parser.add_argument(p.name, nargs="*" if p.kind == "strings" else "?", default=None, help=p.help)
            continue
        metavar = {"strings": "A,B", "numbers": "X,Y", "object": "JSON", "scene": "SCENE"}.get(p.kind, p.kind.upper())
        parser.add_argument(flag, dest=p.name, default=None, metavar=metavar, help=_flag_help(p))


def _flag_help(p: Param) -> str:
    extra = f" (one of {', '.join(map(str, p.choices))})" if p.choices else ""
    return p.help + extra + (f" [default {p.default}]" if p.default is not None else "")


def parse_flag(p: Param, text: Any) -> Any:
    """A command-line string as the JSON value ``p`` takes: numbers, comma lists or JSON, scene names or JSON.

    Example: ``parse_flag(Param("record", "strings", ""), "cube,peg") == ["cube", "peg"]``."""
    if not isinstance(text, str):
        return text
    if p.kind == "integer" or (p.kind == "world" and text != "all"):
        return _parse(text, int, p)
    if p.kind == "number":
        return _parse(text, float, p)
    if p.kind in ("strings", "numbers"):
        if text.lstrip().startswith("["):
            return _json(text, p)
        conv = str if p.kind == "strings" else (lambda s: _parse(s, float, p))
        return [conv(s.strip()) for s in text.split(",") if s.strip()]
    if p.kind == "object":
        return _json(text, p)
    if p.kind in ("scene", "any") and text.lstrip()[:1] in ("{", "["):
        return _json(text, p)
    if p.kind == "any":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _parse(text: str, conv: Callable[[str], Any], p: Param) -> Any:
    try:
        return conv(text)
    except ValueError:
        raise RequestError(f"--{p.name} must be {p.kind}, got {text!r}", _fix_for(p), field=p.name) from None


def _json(text: str, p: Param) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RequestError(f"--{p.name} is not valid JSON ({exc.msg} at column {exc.colno})", _fix_for(p),
                           field=p.name) from None


def flags_to_args(op: Op, ns: argparse.Namespace) -> dict[str, Any]:
    """The JSON arguments given as flags (unset flags are left out). Example: see ``alloy.cli.command``."""
    out = {}
    for p in op.params:
        value = getattr(ns, p.name, None)
        if isinstance(value, list):                     # positional list words: "a b" and "a,b c" alike
            value = ",".join(value) or None
        if value is not None:
            out[p.name] = parse_flag(p, value)
    return out


def describe(ops: Sequence[Op]) -> list[dict[str, Any]]:
    """Every operation as JSON: name, summary, description, input schema and an example request.

    Example: ``describe(OPS.values())[0]["name"]``."""
    return [{"name": op.name, "summary": op.summary, "description": op.description, "read_only": op.read_only,
             "input_schema": json_schema(op), "example": {"op": op.name, **op.example}} for op in ops]
