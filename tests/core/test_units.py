# SPDX-License-Identifier: Apache-2.0
"""parse_si: the closed unit whitelist (SPEC §3.1, WP0 acceptance: >= 40 cases incl. rejects)."""

from __future__ import annotations

import math

import pytest

from alloy.core.errors import UnitError
from alloy.core.units import UNIT_KINDS, UNITS, kinds_of, parse_si

DEG = math.pi / 180.0

GOOD = [
    ("0.2 mm", "length", 2e-4), ("2 cm", "length", 0.02), ("1.5 m", "length", 1.5), ("50 um", "length", 5e-5),
    ("50 µm", "length", 5e-5), ("50 μm", "length", 5e-5), ("1 in", "length", 0.0254), ("3e-3", "length", 3e-3),
    ("35 g", "mass", 0.035), ("2 kg", "mass", 2.0), ("1 t", "mass", 1000.0),
    ("0.3125 ms", "time", 3.125e-4), ("20 us", "time", 2e-5), ("2 min", "time", 120.0),
    ("50 N", "force", 50.0), ("2 kN", "force", 2000.0), ("1 kgf", "force", 9.80665),
    ("2.1 GPa", "pressure", 2.1e9), ("200 kPa", "pressure", 2e5), ("1 bar", "pressure", 1e5),
    ("3 N/mm^2", "pressure", 3e6), ("35 deg", "angle", 35 * DEG), ("35°", "angle", 35 * DEG),
    ("0.5 rad", "angle", 0.5), ("10 mrad", "angle", 0.01), ("2e6 N/m", "stiffness", 2e6),
    ("2 N/mm", "stiffness", 2000.0), ("1 kN/m", "stiffness", 1000.0), ("1.5 N·m", "torque", 1.5),
    ("1.5 N*m", "torque", 1.5), ("1.5 Nm", "torque", 1.5), ("20 N·mm", "torque", 0.02),
    ("1 mPa·s", "viscosity", 1e-3), ("1 mPa s", "viscosity", 1e-3), ("1 cP", "viscosity", 1e-3),
    ("1 Pa*s", "viscosity", 1.0), ("1000 kg/m^3", "density", 1000.0), ("1000 kg/m3", "density", 1000.0),
    ("1000 kg/m³", "density", 1000.0), ("7.8 g/cm^3", "density", 7800.0), ("5 mm/s", "speed", 5e-3),
    ("0.5 m/s", "speed", 0.5), ("36 km/h", "speed", 10.0), ("3 mJ", "energy", 3e-3), ("72 mN/m", "surface_tension", 0.072),
    ("72 dyn/cm", "surface_tension", 0.072), ("200 g/m^2", "areal_density", 0.2), ("200 gsm", "areal_density", 0.2),
    ("5 %", "dimensionless", 0.05), ("0.3", "dimensionless", 0.3), (0.45, "dimensionless", 0.45), (7, "mass", 7.0),
    ("300 J/m^2", "surface_tension", 300.0), ("1.2 kJ/m²", "surface_tension", 1200.0),
    ("0.5 L/s", "volume_rate", 5e-4), ("6 L/min", "volume_rate", 1e-4), ("2 mL/s", "volume_rate", 2e-6),
    ("  .5 mm ", "length", 5e-4), ("-2 mm", "length", -2e-3), ("1E3 N", "force", 1000.0),
]

BAD = [
    ("0.2 mm", "mass"), ("2 GPa", "length"), ("35 degrees", "angle"), ("1 furlong", "length"), ("mm", "length"),
    ("", "length"), ("1.2.3 mm", "length"), ("nan", "length"), ("inf mm", "length"), (float("nan"), "length"),
    (float("inf"), "force"), (True, "dimensionless"), ("1 N/m", "torque"), ("1 J", "torque"), ("1 N·m", "energy"),
    ("5 m/s", "length"), ("1 kg", "density"), ("1 mm", "lengths"), ("1 L/s", "speed"), (None, "length"), ("1 Mm", "length"),
]


@pytest.mark.parametrize(("text", "kind", "expected"), GOOD)
def test_parse_si_good(text: object, kind: str, expected: float) -> None:
    assert parse_si(text, kind) == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize(("text", "kind"), BAD)
def test_parse_si_rejects(text: object, kind: str) -> None:
    with pytest.raises(UnitError):
        parse_si(text, kind)  # type: ignore[arg-type]


def test_case_count_meets_acceptance() -> None:
    assert len(GOOD) + len(BAD) >= 40


def test_unit_error_is_value_error() -> None:
    with pytest.raises(ValueError):
        parse_si("1 parsec", "length")


def test_every_kind_has_a_table_and_si_unit_factor_one() -> None:
    assert set(UNITS) == set(UNIT_KINDS)
    si = {"length": "m", "mass": "kg", "time": "s", "force": "N", "pressure": "Pa", "angle": "rad",
          "stiffness": "N/m", "torque": "N·m", "viscosity": "Pa·s", "density": "kg/m^3", "speed": "m/s",
          "energy": "J", "surface_tension": "N/m", "areal_density": "kg/m^2", "volume_rate": "m^3/s",
          "dimensionless": "-"}
    for kind, unit in si.items():
        assert UNITS[kind][unit] == 1.0


def test_wrong_kind_message_names_the_right_kind() -> None:
    with pytest.raises(UnitError, match="length"):
        parse_si("3 mm", "mass")
    assert kinds_of("N/m") == ("stiffness", "surface_tension")
