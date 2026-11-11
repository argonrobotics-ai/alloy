# SPDX-License-Identifier: Apache-2.0
"""YAML I/O for cards: a strict loader, a deterministic dumper and the unit kind of every numeric card field.

The loader follows the YAML 1.2 core schema for implicit scalars, so ``1e-4`` is a float and ``on``/``yes`` stay
strings (PyYAML's 1.1 default reads ``1e-4`` as a string and ``on`` as True, [internal reference]). Duplicate
keys are errors (PyYAML keeps the last one silently). Dates are not resolved. Floats are dumped with ``repr`` so a
dump followed by a load returns the same bits.
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

import yaml

from alloy.core.errors import CardError, UnitError
from alloy.core.units import UnitKind, parse_si

log = logging.getLogger("alloy.cards")

_BOOL = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")
_INT = re.compile(r"^[-+]?[0-9]+$")
_FLOAT = re.compile(r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$")
_NULL = re.compile(r"^(?:~|null|Null|NULL|)$")
_DROP = {"tag:yaml.org,2002:bool", "tag:yaml.org,2002:int", "tag:yaml.org,2002:float",
         "tag:yaml.org,2002:null", "tag:yaml.org,2002:timestamp"}


class YamlKeyError(ValueError):
    """A duplicate or non-string mapping key; ``key`` and ``line`` (1-based) locate it."""

    def __init__(self, key: object, line: int, message: str) -> None:
        super().__init__(f"{message} {key!r} at line {line}")
        self.key, self.line = key, line


class MarkedDict(dict):
    """A mapping that remembers the source line of each key (for error messages)."""

    lines: dict[str, int]


def _core_resolvers() -> dict[str, list[tuple[str, re.Pattern[str]]]]:
    table = {k: [r for r in v if r[0] not in _DROP] for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()}
    adds = (("tag:yaml.org,2002:bool", _BOOL, "tTfF"), ("tag:yaml.org,2002:int", _INT, "-+0123456789"),
            ("tag:yaml.org,2002:float", _FLOAT, "-+0123456789."), ("tag:yaml.org,2002:null", _NULL, ["~", "n", "N", ""]))
    for tag, regexp, first in adds:
        for ch in first:
            table.setdefault(ch, []).append((tag, regexp))
    return table


class CardLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 core implicit scalars, duplicate-key errors and per-key line numbers."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> MarkedDict:  # type: ignore[override]
        self.flatten_mapping(node)
        out = MarkedDict()
        out.lines = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            line = key_node.start_mark.line + 1
            if not isinstance(key, str):
                raise YamlKeyError(key, line, "mapping keys must be strings, got")
            if key in out:
                raise YamlKeyError(key, line, "duplicate key")
            out[key] = self.construct_object(value_node, deep=deep)
            out.lines[key] = line
        return out


def _construct_int(loader: CardLoader, node: yaml.ScalarNode) -> int:
    """YAML 1.2 decimal ints (``010`` is ten, not PyYAML's 1.1 octal eight)."""
    return int(loader.construct_scalar(node), 10)


CardLoader.yaml_implicit_resolvers = _core_resolvers()
CardLoader.add_constructor("tag:yaml.org,2002:map", CardLoader.construct_mapping)
CardLoader.add_constructor("tag:yaml.org,2002:int", _construct_int)


class _Resolver12(yaml.resolver.Resolver):
    """The loader's implicit-scalar rules, usable on their own."""


_Resolver12.yaml_implicit_resolvers = CardLoader.yaml_implicit_resolvers
_RESOLVERS = (yaml.resolver.Resolver(), _Resolver12())


def _plain_is_ambiguous(text: str) -> bool:
    """True if a plain (unquoted) scalar would not read back as this string under YAML 1.1 or 1.2 rules."""
    return any(r.resolve(yaml.ScalarNode, text, (True, False)) != "tag:yaml.org,2002:str" for r in _RESOLVERS)


class FlowList(list):
    """A list dumped in flow style (``[a, b, c]``): numeric tuples such as poses and sizes."""


class CardDumper(yaml.SafeDumper):
    """Block-style dumper; quotes any string a YAML 1.1 or 1.2 reader would not read back as that string."""


def _needs_escapes(text: str) -> bool:
    """Control characters and Unicode line/paragraph separators (NEL, U+2028/9) only survive double quotes."""
    return any(c != "\n" and unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp", "Cs") for c in text)


def _represent_str(dumper: CardDumper, data: str) -> yaml.ScalarNode:
    style = "'" if _plain_is_ambiguous(data) else None
    if _needs_escapes(data):
        style = '"'
    elif "\n" in data:
        style = "|"
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


def _represent_flow(dumper: CardDumper, data: FlowList) -> yaml.SequenceNode:
    return dumper.represent_sequence("tag:yaml.org,2002:seq", list(data), flow_style=True)


def _represent_float(dumper: CardDumper, data: float) -> yaml.ScalarNode:
    if math.isnan(data):
        text = ".nan"
    elif math.isinf(data):
        text = ".inf" if data > 0 else "-.inf"
    else:
        text = repr(data).lower()
        if "." not in text and "e" in text:
            text = text.replace("e", ".0e", 1)
    return dumper.represent_scalar("tag:yaml.org,2002:float", text)


CardDumper.add_representer(str, _represent_str)
CardDumper.add_representer(FlowList, _represent_flow)
CardDumper.add_representer(float, _represent_float)
CardDumper.add_representer(MarkedDict, yaml.SafeDumper.represent_dict)


def _key_path(text: str, key: object, line: int) -> str:
    """Dotted path of the mapping key ``key`` that starts on ``line`` (1-based), from the node tree."""
    try:
        root = yaml.compose(text, Loader=CardLoader)  # noqa: S506 - composing only, no construction
    except yaml.YAMLError:
        return str(key)
    stack: list[tuple[yaml.Node, str]] = [(root, "")] if root is not None else []
    while stack:
        node, path = stack.pop()
        if isinstance(node, yaml.MappingNode):
            for k, v in node.value:
                here = f"{path}.{k.value}" if path else str(k.value)
                if k.start_mark.line + 1 == line and k.value == str(key):
                    return here
                stack.append((v, here))
        elif isinstance(node, yaml.SequenceNode):
            stack.extend((v, f"{path}.{i}" if path else str(i)) for i, v in enumerate(node.value))
    return str(key)


def yaml_load(text: str, card: str) -> Any:
    """Parse card YAML; syntax, duplicate keys, non-string keys and bad tagged scalars raise ``CardError``."""
    try:
        return yaml.load(text, Loader=CardLoader)  # noqa: S506 - CardLoader is a SafeLoader subclass
    except YamlKeyError as exc:
        raise CardError(card, _key_path(text, exc.key, exc.line), str(exc)) from exc
    except yaml.YAMLError as exc:
        raise CardError(card, "<yaml>", f"YAML syntax error: {exc}") from exc
    except (ValueError, TypeError, OverflowError) as exc:
        raise CardError(card, "<yaml>", f"bad YAML scalar: {exc}") from exc


def yaml_dump(data: Mapping[str, Any], header: str = "") -> str:
    """Dump a plain mapping (from ``codec.encode``) as block YAML with keys in insertion order."""
    body = yaml.dump(data, Dumper=CardDumper, sort_keys=False, allow_unicode=True, width=120,
                     default_flow_style=False)
    if not header:
        return body
    lines = "".join(f"# {h}\n" for h in header.splitlines())
    return lines + body


# Unit kind of every numeric card field, keyed "Class.field". Tuple-valued fields apply the kind per element.
# A tuple of kinds means "try each in order" (e.g. a break limit that is a force for welds and a torque for joints).
LeafKind = UnitKind | tuple[UnitKind, ...]
LEAF_KINDS: dict[str, LeafKind] = {
    "Surface.mu_static": "dimensionless", "Surface.mu_dynamic": "dimensionless",
    "Surface.restitution": "dimensionless", "Surface.torsional_lever": "length", "Surface.rolling_lever": "length",
    "Surface.contact_stiffness": "stiffness", "Surface.hydro_modulus": "pressure",
    "Surface.contact_damping_ratio": "dimensionless",
    "ClothAniso.stretch_warp": "stiffness", "ClothAniso.stretch_weft": "stiffness", "ClothAniso.shear": "stiffness",
    "ClothAniso.bend_warp": "torque", "ClothAniso.bend_weft": "torque",
    "Elastic.youngs": "pressure", "Elastic.poisson": "dimensionless", "Elastic.rayleigh_beta": "time",
    "Elastic.bending": "torque", "Elastic.bending_hysteresis": "force",     # 2HB: N·m per m of width
    "Plastic.yield_stress": "pressure", "Plastic.yield_pressure": "pressure", "Plastic.friction_angle": "angle",
    "Plastic.cohesion": "pressure", "Plastic.hardening": "dimensionless", "Plastic.dilatancy": "dimensionless",
    "Plastic.hardening_modulus": "pressure",
    "Fluid.viscosity": "viscosity", "Fluid.surface_tension": "surface_tension", "Fluid.adhesion": "dimensionless",
    "Fluid.sound_speed": "speed", "Fluid.flow_index": "dimensionless",
    "Fluid.consistency": "viscosity",          # Pa·s^n: the prefix on Pa converts (mPa·s^n); the time unit must be s
    "Failure.fracture_strain": "dimensionless", "Failure.fracture_stress": "pressure",
    "Failure.tear_energy": "surface_tension", "Failure.cut_min_force": "force",
    "Failure.cut_min_strain": "dimensionless",
    "Film.dead_fold_yield": "angle", "Film.contact_radius": "length", "Film.air_scale": "dimensionless",
    "Film.crease_ratio": "dimensionless", "Film.crease_yield": "angle",
    "MaterialCard.density": "density",
    "PairCard.mu_static": "dimensionless", "PairCard.mu_dynamic": "dimensionless",
    "PairCard.restitution": "dimensionless", "PairCard.torsional_lever": "length",
    "Primitive.size": "length", "MeshSource.scale": "dimensionless", "SDFSource.voxel": "length",
    "TetSource.edge_length": "length", "CurveSource.points": "length", "CurveSource.radius": "length",
    "ParticleSource.region": "length", "ParticleSource.spacing": "length", "ParticleSource.grain_diameter": "length",
    "LoadEnvelope.grip_force": "force", "LoadEnvelope.bend_force": "force", "LoadEnvelope.contact_speed": "speed",
    "LoadEnvelope.tolerance": "length", "LoadEnvelope.design_force": "force",
    "Crease.a": "length", "Crease.b": "length", "Crease.stiffness": "torque",
    "Features.thickness": "length", "Features.passage_width": "length", "Features.precision_anchor": "length",
    "Features.min_wall": "length",                                  # bend/twist_stiffness (N·m²): SI numbers only
    "BladeSpec.edge_a": "length", "BladeSpec.edge_b": "length", "BladeSpec.edge_radius": "length",
    "BladeSpec.softness": "dimensionless", "BladeSpec.cut_spring_ke": "stiffness",
    "PrecisionFeature.crop": "length", "PrecisionFeature.voxel": "length", "PrecisionFeature.clearance": "length",
    "Placement.pose": "length", "RobotRef.base_pose": "length", "RobotRef.contact_gap": "length",
    "Attachment.break_force": ("force", "torque"), "Attachment.break_strain": "dimensionless",
    "Attachment.anchor": "length", "Attachment.timeconst": "time", "Attachment.dampratio": "dimensionless",
    "EmitterSpec.pose": "length", "EmitterSpec.half_extents": "length", "EmitterSpec.velocity": "speed",
    "EmitterSpec.rate": "volume_rate",
    "EmitterSpec.start_s": "time", "EmitterSpec.stop_s": "time",
    "SceneCard.control_dt": "time",
}


def kind_of(owner: str, field: str) -> LeafKind | None:
    """Unit kind of ``owner.field`` or None (plain number, no unit strings accepted)."""
    return LEAF_KINDS.get(f"{owner}.{field}")


def parse_number(raw: object, kind: LeafKind | None, card: str, field: str) -> float:
    """A float from a YAML number or a unit string such as ``"0.2 mm"``; errors name the dotted field."""
    if isinstance(raw, bool):
        raise CardError(card, field, f"expected a number, got boolean {raw!r}")
    if isinstance(raw, (int, float)):
        try:
            value = float(raw)
        except OverflowError as exc:
            raise CardError(card, field, f"number too large for float64: {str(raw)[:20]}...") from exc
        if not math.isfinite(value):
            raise CardError(card, field, f"non-finite value {value}")
        return value
    if not isinstance(raw, str):
        raise CardError(card, field, f"expected a number or unit string, got {type(raw).__name__}")
    if kind is None:
        raise CardError(card, field, f"unit strings are not accepted here (SI number expected), got {raw!r}")
    kinds = kind if isinstance(kind, tuple) else (kind,)
    last: Exception | None = None
    for k in kinds:
        try:
            value = float(parse_si(raw, k))
        except UnitError as exc:
            last = exc
            continue
        if not math.isfinite(value):
            raise CardError(card, field, f"non-finite value {raw!r}")
        return value
    raise CardError(card, field, f"bad value {raw!r} for {'/'.join(kinds)}: {last}") from last
