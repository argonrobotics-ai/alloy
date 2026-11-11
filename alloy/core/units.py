# SPDX-License-Identifier: Apache-2.0
"""SI parsing for card values (SPEC §3.1, §4.1).

``parse_si("0.2 mm", "length") -> 2e-4``. The unit whitelist is closed: an unknown unit, a unit of another kind, a
malformed or non-finite number raises ``UnitError``. A bare number (or a number string without a unit) is taken as
already SI. Unit spellings are normalised first: ``μ`` -> ``µ``, ``²``/``³`` -> ``^2``/``^3``, and ``*``, ``⋅``
or inner spaces between unit factors -> ``·``.
"""

from __future__ import annotations

import math
import re
from types import MappingProxyType
from typing import Final, Literal, Mapping, get_args

from alloy.core.errors import UnitError

UnitKind = Literal["length", "mass", "time", "force", "pressure", "angle", "stiffness", "torque", "viscosity",
                   "density", "speed", "energy", "surface_tension", "areal_density", "volume_rate",
                   "dimensionless"]
UNIT_KINDS: Final[frozenset[str]] = frozenset(get_args(UnitKind))

_DEG = math.pi / 180.0

_TABLE: dict[str, dict[str, float]] = {
    "length": {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "µm": 1e-6, "um": 1e-6, "nm": 1e-9, "km": 1e3, "in": 0.0254},
    "mass": {"kg": 1.0, "g": 1e-3, "mg": 1e-6, "t": 1e3},
    "time": {"s": 1.0, "ms": 1e-3, "µs": 1e-6, "us": 1e-6, "ns": 1e-9, "min": 60.0, "h": 3600.0},
    "force": {"N": 1.0, "kN": 1e3, "mN": 1e-3, "µN": 1e-6, "uN": 1e-6, "kgf": 9.80665},
    "pressure": {"Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "bar": 1e5, "mbar": 1e2, "psi": 6894.757293168,
                 "N/m^2": 1.0, "N/mm^2": 1e6},
    "angle": {"rad": 1.0, "mrad": 1e-3, "deg": _DEG, "°": _DEG},
    "stiffness": {"N/m": 1.0, "N/mm": 1e3, "kN/m": 1e3, "kN/mm": 1e6, "MN/m": 1e6, "N/µm": 1e6, "N/um": 1e6},
    "torque": {"N·m": 1.0, "Nm": 1.0, "N·mm": 1e-3, "Nmm": 1e-3, "mN·m": 1e-3, "kN·m": 1e3},
    "viscosity": {"Pa·s": 1.0, "mPa·s": 1e-3, "cP": 1e-3, "P": 0.1},
    "density": {"kg/m^3": 1.0, "g/cm^3": 1e3, "g/mL": 1e3, "g/ml": 1e3, "kg/L": 1e3, "g/L": 1.0},
    "speed": {"m/s": 1.0, "mm/s": 1e-3, "cm/s": 1e-2, "µm/s": 1e-6, "um/s": 1e-6, "km/h": 1.0 / 3.6},
    "energy": {"J": 1.0, "mJ": 1e-3, "kJ": 1e3, "µJ": 1e-6, "uJ": 1e-6},
    "surface_tension": {"N/m": 1.0, "mN/m": 1e-3, "dyn/cm": 1e-3,          # also tear energy (J/m^2 = N/m)
                        "J/m^2": 1.0, "kJ/m^2": 1e3, "mJ/m^2": 1e-3},
    "volume_rate": {"m^3/s": 1.0, "L/s": 1e-3, "mL/s": 1e-6, "L/min": 1e-3 / 60.0, "mL/min": 1e-6 / 60.0},
    "areal_density": {"kg/m^2": 1.0, "g/m^2": 1e-3, "gsm": 1e-3, "g/cm^2": 10.0},
    "dimensionless": {"-": 1.0, "%": 1e-2, "ppm": 1e-6},
}
UNITS: Final[Mapping[str, Mapping[str, float]]] = MappingProxyType(
    {k: MappingProxyType(v) for k, v in _TABLE.items()})

_NUMBER = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*?)\s*$")
_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("μ", "µ"), ("³", "^3"), ("²", "^2"), ("**", "^"), ("⋅", "·"), ("*", "·"),
)


def _normalise_unit(unit: str) -> str:
    """Canonical spelling of a unit string: ``kg/m3`` -> ``kg/m^3``, ``Pa s`` -> ``Pa·s``."""
    for old, new in _REPLACEMENTS:
        unit = unit.replace(old, new)
    unit = re.sub(r"\s+", "·", unit.strip())
    return re.sub(r"(?<=[a-zA-Z])([23])$", r"^\1", unit)


def kinds_of(unit: str) -> tuple[str, ...]:
    """Every kind whose whitelist contains ``unit`` (after normalisation); () if unknown."""
    norm = _normalise_unit(unit)
    return tuple(k for k, table in _TABLE.items() if norm in table)


def parse_si(text: str | float | int, kind: UnitKind) -> float:
    """Parse ``text`` as a value of ``kind`` and return it in canonical SI (float64).

    '0.2 mm' -> 2e-4, '2.1 GPa' -> 2.1e9, '35 deg' -> 0.6109; bare numbers are SI. Raises ``UnitError`` for an
    unknown kind or unit, a unit of another kind, a malformed number, a bool, or a non-finite value.
    """
    if kind not in UNIT_KINDS:
        raise UnitError(f"unknown unit kind {kind!r}; expected one of {sorted(UNIT_KINDS)}")
    if isinstance(text, bool):
        raise UnitError(f"{kind}: a bool is not a quantity ({text!r})")
    if isinstance(text, (int, float)):
        return _finite(float(text), text, kind)
    if not isinstance(text, str):
        raise UnitError(f"{kind}: expected str or number, got {type(text).__name__}")
    match = _NUMBER.match(text)
    if match is None:
        raise UnitError(f"{kind}: cannot parse {text!r} as '<number> <unit>'")
    number, unit = float(match.group(1)), match.group(2)
    if not unit:
        return _finite(number, text, kind)
    return _finite(number * _factor(unit, kind, text), text, kind)


def _factor(unit: str, kind: str, text: str) -> float:
    norm = _normalise_unit(unit)
    table = _TABLE[kind]
    if norm in table:
        return table[norm]
    other = kinds_of(unit)
    if other:
        raise UnitError(f"{kind}: unit {unit!r} in {text!r} is a {'/'.join(other)} unit, not {kind}")
    raise UnitError(f"{kind}: unknown unit {unit!r} in {text!r}; allowed: {sorted(table)}")


def _finite(value: float, text: object, kind: str) -> float:
    if not math.isfinite(value):
        raise UnitError(f"{kind}: non-finite value {text!r}")
    return value
