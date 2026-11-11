# SPDX-License-Identifier: Apache-2.0
"""The values the sourcing agent fills, their SI units, plausible bounds and card paths.

A field name is the key used everywhere in ``alloy.sourcing`` (records, the category cache, the queue). ``card_path``
is where the value lands on a resolved object (``alloy.cards.paths`` grammar); ``mass`` has no card path because a card
carries density, so a sourced mass becomes a density through the mesh volume (``alloy.sourcing.store``).

Example:
    >>> from alloy.sourcing.fields import parse_value
    >>> round(parse_value("12 oz", "mass"), 4)
    0.3402
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Final

import yaml

from alloy.core.errors import QuantityError
from alloy.core.units import parse_si


class SourcingError(ValueError):
    """A sourcing record, cache entry or request is malformed; the message says what to change."""


@dataclass(frozen=True)
class FieldSpec:
    """One fillable value: SI unit, the unit kind ``parse_si`` understands, plausible bounds and card path."""

    unit: str
    kind: str
    lo: float
    hi: float
    card_path: str | None


FIELDS: Final[dict[str, FieldSpec]] = {
    "mass": FieldSpec("kg", "mass", 1e-6, 1e4, None),
    "density": FieldSpec("kg/m^3", "density", 10.0, 25_000.0, "material.density"),
    "effective_density": FieldSpec("kg/m^3", "density", 1.0, 25_000.0, "material.density"),
    "mu_static": FieldSpec("-", "dimensionless", 0.0, 2.0, "material.surface.mu_static"),
    "mu_dynamic": FieldSpec("-", "dimensionless", 0.0, 2.0, "material.surface.mu_dynamic"),
    "restitution": FieldSpec("-", "dimensionless", 0.0, 1.0, "material.surface.restitution"),
    "youngs": FieldSpec("Pa", "pressure", 1e3, 1e12, "material.elastic.youngs"),
    "poisson": FieldSpec("-", "dimensionless", -0.99, 0.5, "material.elastic.poisson"),
}
"""``effective_density`` is mass over the enclosed (mesh) volume of a whole object, hollow parts included."""

PATH_TO_FIELD: Final[dict[str, str]] = {
    "density": "density", "surface.mu_static": "mu_static", "surface.mu_dynamic": "mu_dynamic",
    "surface.restitution": "restitution", "elastic.youngs": "youngs", "elastic.poisson": "poisson",
}
"""Material-card leaf path (relative to the material) -> field name, for the queue scan."""

_IMPERIAL_MASS: Final[dict[str, float]] = {"oz": 0.028349523125, "lb": 0.45359237, "lbs": 0.45359237}
_NUM_UNIT = re.compile(r"^\s*([-+0-9.eE]+)\s*([A-Za-z]+)\s*$")


_MATERIAL_ROOTS: Final[tuple[str, ...]] = ("surface", "elastic", "plastic", "fluid", "failure")
_OBJECT_ROOTS: Final[tuple[str, ...]] = ("features", "precision_features", "loads")


def canonical(field: str) -> str:
    """The short field name for a material card path (``surface.mu_static`` -> ``mu_static``), else ``field``.

    Example:
        >>> canonical("material.surface.mu_static"), canonical("features.min_wall")
        ('mu_static', 'features.min_wall')
    """
    rel = field.removeprefix("material.")
    return PATH_TO_FIELD.get(rel, field)


def spec(field: str) -> FieldSpec:
    """The ``FieldSpec`` for ``field``: a named field, or any material / object card path (``fluid.viscosity``,
    ``features.min_wall``, ``precision_features.0.clearance``) in SI with no bounds; raises ``SourcingError`` else.

    Example:
        >>> spec("density").unit, spec("features.min_wall").card_path
        ('kg/m^3', 'features.min_wall')
    """
    name = canonical(field)
    if name in FIELDS:
        return FIELDS[name]
    root = name.removeprefix("material.").split(".")[0]
    if root in _MATERIAL_ROOTS:
        return FieldSpec("SI", "si", -math.inf, math.inf, "material." + name.removeprefix("material."))
    if root in _OBJECT_ROOTS:
        return FieldSpec("SI", "si", -math.inf, math.inf, name)
    raise SourcingError(f"unknown field {field!r}; use one of {sorted(FIELDS)} or a card path "
                        f"under {list(_MATERIAL_ROOTS + _OBJECT_ROOTS)}")


def parse_value(raw: str | float | int, field: str) -> float:
    """A number or unit string (``"350 g"``, ``"12 oz"``, ``"1.2 g/cm^3"``, ``"2.1 GPa"``) in the field's SI unit.

    Example:
        >>> parse_value("1.2 g/cm^3", "density")
        1200.0
    """
    fs = spec(field)
    if isinstance(raw, str):
        m = _NUM_UNIT.match(raw)
        if m and fs.kind == "mass" and m.group(2).lower() in _IMPERIAL_MASS:
            return float(m.group(1)) * _IMPERIAL_MASS[m.group(2).lower()]
    if fs.kind in ("dimensionless", "si"):
        value = float(raw)
    else:
        try:
            value = parse_si(raw, fs.kind)  # type: ignore[arg-type]
        except QuantityError as e:
            raise SourcingError(f"{field}: cannot read {raw!r} ({e}); write a number in {fs.unit} or a unit string") from e
    if not math.isfinite(value):
        raise SourcingError(f"{field}: value {raw!r} is not finite")
    return value


def check_bounds(field: str, value: float, where: str) -> None:
    """Raise ``SourcingError`` if ``value`` is outside the field's plausible bounds (a unit slip, usually).

    Example:
        >>> check_bounds("mu_static", 0.4, "mug")
    """
    fs = spec(field)
    if not (fs.lo <= value <= fs.hi):
        raise SourcingError(
            f"{where}: {field} = {value:g} {fs.unit} is outside [{fs.lo:g}, {fs.hi:g}]; check the unit conversion")


_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)


def yaml_load(text: str) -> object:
    """Safe YAML load (libyaml when available: thousands of records load in seconds).

    Example:
        >>> yaml_load("a: 1")
        {'a': 1}
    """
    return yaml.load(text, Loader=_LOADER)  # noqa: S506 (safe loader)


def yaml_dump(data: object) -> str:
    """Safe YAML dump in the given key order, 120 columns, unicode kept.

    Example:
        >>> yaml_dump({"b": 1, "a": 2}).splitlines()
        ['b: 1', 'a: 2']
    """
    return yaml.dump(data, Dumper=_DUMPER, sort_keys=False, allow_unicode=True, width=120)
