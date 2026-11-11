# SPDX-License-Identifier: Apache-2.0
"""The exporter -> importer hand-off: one ``gar.spec/1`` JSON file per Gym-Anything object.

Standard library only, so the exporters (which run in Gym-Anything's own Python 3.12 env, not Alloy's) can load
this file by path to check what they write. Every number carries the label Gym-Anything gave it and the file:line
it came from, so the Alloy card keeps the citation.

Example::

    >>> s = parse_spec({"schema": "gar.spec/1", "key": "mug", "sim": "wiping", "name": "Mug",
    ...                 "category": "container", "mesh": "mug.obj",
    ...                 "mass": {"value": 0.3, "basis": "datasheet", "cite": "wiping/params.py:12"},
    ...                 "material": {"name": "stoneware"}}, "mug.json")
    >>> s.card_name, s.mass.alloy_source
    ('gar_wiping_mug', 'catalog')
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "gar.spec/1"
CARD_PREFIX = "gar_"
KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")

# Gym-Anything's parameter-ledger labels (their params.py docstrings) -> Alloy ``Q.source`` (library/SOURCES.md).
BASIS_TO_SOURCE: dict[str, str] = {
    "measured": "measured",
    "datasheet": "catalog", "standard": "catalog", "literature-read": "catalog", "literature": "catalog",
    "search": "catalog",
    "calibrated": "fit",
    "derived": "model", "estimate": "model", "authored": "model",
}
CATEGORIES = ("container", "tool", "fixture", "food", "furniture", "articulated", "cloth", "cable", "other")
KINDS = ("rigid", "cloth", "cable", "soft")
MOTIONS = ("dynamic", "static", "kinematic")
MEASURES = ("x", "y", "z", "max", "min", "mid", "diameter_xy")


class SpecError(ValueError):
    """A spec is malformed; the message names the file, the field and what to change."""

    def __init__(self, where: str, fld: str, message: str) -> None:
        super().__init__(f"{where}: {fld}: {message}")
        self.where, self.field, self.message = where, fld, message


@dataclass(frozen=True)
class Cited:
    """One sourced number: ``value`` (SI), Gym-Anything's ``basis`` label, ``cite`` (file:line in their repo, or
    ``REPORT.pdf p.N``), their own primary-source text in ``quote``, optional absolute ``unc`` (lo, hi)."""

    value: float
    basis: str
    cite: str
    quote: str = ""
    unc: tuple[float, float] | None = None

    @property
    def alloy_source(self) -> str:
        """The Alloy ``Q.source`` this basis maps to (e.g. ``datasheet`` -> ``catalog``)."""
        return BASIS_TO_SOURCE[self.basis]


@dataclass(frozen=True)
class Dim:
    """A cited dimension for the size check: ``measure`` is an axis, ``max/mid/min`` OBB extent or ``diameter_xy``."""

    name: str
    value_m: float
    measure: str
    cite: str


@dataclass(frozen=True)
class MaterialSpec:
    """The bulk material: ``name`` in words (``"soda-lime glass"``), optional cited density and elastic moduli."""

    name: str
    density: Cited | None = None
    youngs: Cited | None = None
    poisson: Cited | None = None


@dataclass(frozen=True)
class ClothSpec:
    """Thin-sheet constants: membrane stretch/shear in N/m, bending in N·m, sheet thickness in m."""

    thickness: Cited
    stretch_warp: Cited
    stretch_weft: Cited
    shear: Cited
    bend_warp: Cited
    bend_weft: Cited


@dataclass(frozen=True)
class CableSpec:
    """A rod/cable: centre-line ``points`` (body frame, m), ``radius`` (m), optional measured EI (N·m^2)."""

    points: tuple[tuple[float, float, float], ...]
    radius: Cited
    bend_stiffness: Cited | None = None


@dataclass(frozen=True)
class VisualPart:
    """One render part: its mesh (body frame, may be open; OBJ ``vt`` UVs kept), base colour (sRGB 0..1) and the PBR
    scalars their render code sets, an optional base-colour ``texture`` image mapped by the UVs, and ``cite``."""

    mesh: Path
    base_color: tuple[float, float, float]
    cite: str
    roughness: float | None = None
    metallic: float | None = None
    texture: Path | None = None
    name: str = ""


@dataclass(frozen=True)
class GarSpec:
    """One Gym-Anything object, as exported from their code (see the module docstring for the file format)."""

    key: str
    sim: str
    name: str
    category: str
    mesh: Path | None
    material: MaterialSpec
    kind: str = "rigid"
    motion: str = "dynamic"
    mass: Cited | None = None
    friction: Cited | None = None
    friction_partner: str = ""
    restitution: Cited | None = None
    thickness: Cited | None = None
    dims: tuple[Dim, ...] = ()
    source_ref: str = ""
    licence: str = "MIT"
    notes: str = ""
    color_rgb: tuple[float, float, float] | None = None
    cloth: ClothSpec | None = None
    cable: CableSpec | None = None
    visual: tuple[VisualPart, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def card_name(self) -> str:
        """Alloy card name: ``gar_<sim>_<key>`` (shared by the object and its material card)."""
        return f"{CARD_PREFIX}{self.sim}_{self.key}"


_TOP_KEYS = {"schema", "key", "sim", "name", "category", "kind", "motion", "mesh", "mass", "material", "friction",
             "friction_partner", "restitution", "thickness", "dims", "source_ref", "licence", "notes", "color_rgb",
             "cloth", "cable", "visual", "extra"}


def _num(raw: Any, where: str, fld: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
        raise SpecError(where, fld, f"expected a finite number, got {raw!r}")
    return float(raw)


def _str(raw: Any, where: str, fld: str, required: bool = True) -> str:
    if raw is None and not required:
        return ""
    if not isinstance(raw, str) or (required and not raw.strip()):
        raise SpecError(where, fld, f"expected a non-empty string, got {raw!r}")
    return raw


def _choice(raw: Any, options: tuple[str, ...], where: str, fld: str) -> str:
    if raw not in options:
        raise SpecError(where, fld, f"{raw!r} is not one of {list(options)}")
    return raw


def parse_cited(raw: Any, where: str, fld: str) -> Cited | None:
    """A ``{value, basis, cite, quote?, unc?}`` mapping (or null) -> ``Cited``; positive values only."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise SpecError(where, fld, "expected {value, basis, cite[, quote, unc]} or null")
    unknown = set(raw) - {"value", "basis", "cite", "quote", "unc"}
    if unknown:
        raise SpecError(where, fld, f"unknown keys {sorted(unknown)}; allowed: value, basis, cite, quote, unc")
    value = _num(raw.get("value"), where, f"{fld}.value")
    basis = _choice(raw.get("basis"), tuple(BASIS_TO_SOURCE), where, f"{fld}.basis")
    cite = _str(raw.get("cite"), where, f"{fld}.cite")
    unc = raw.get("unc")
    if unc is not None:
        if not isinstance(unc, (list, tuple)) or len(unc) != 2:
            raise SpecError(where, f"{fld}.unc", "expected [lo, hi]")
        unc = (_num(unc[0], where, f"{fld}.unc"), _num(unc[1], where, f"{fld}.unc"))
        if not unc[0] <= value <= unc[1]:
            raise SpecError(where, f"{fld}.unc", f"{list(unc)} does not contain value {value}")
    return Cited(value, basis, cite, _str(raw.get("quote"), where, f"{fld}.quote", required=False), unc)


def _positive(c: Cited | None, where: str, fld: str) -> Cited | None:
    if c is not None and c.value <= 0:
        raise SpecError(where, fld, f"must be > 0, got {c.value}")
    return c


def _material(raw: Any, where: str) -> MaterialSpec:
    if not isinstance(raw, dict) or not raw.get("name"):
        raise SpecError(where, "material", "expected {name[, density, youngs, poisson]} with a material name")
    unknown = set(raw) - {"name", "density", "youngs", "poisson"}
    if unknown:
        raise SpecError(where, "material", f"unknown keys {sorted(unknown)}")
    return MaterialSpec(_str(raw["name"], where, "material.name"),
                        _positive(parse_cited(raw.get("density"), where, "material.density"), where, "material.density"),
                        _positive(parse_cited(raw.get("youngs"), where, "material.youngs"), where, "material.youngs"),
                        parse_cited(raw.get("poisson"), where, "material.poisson"))


def _dims(raw: Any, where: str) -> tuple[Dim, ...]:
    out = []
    for i, d in enumerate(raw or ()):
        f = f"dims.{i}"
        if not isinstance(d, dict):
            raise SpecError(where, f, "expected {name, value_m, measure, cite}")
        out.append(Dim(_str(d.get("name"), where, f"{f}.name"), _num(d.get("value_m"), where, f"{f}.value_m"),
                       _choice(d.get("measure"), MEASURES, where, f"{f}.measure"), _str(d.get("cite"), where, f"{f}.cite")))
    return tuple(out)


def _cloth(raw: Any, where: str) -> ClothSpec | None:
    if raw is None:
        return None
    names = ("thickness", "stretch_warp", "stretch_weft", "shear", "bend_warp", "bend_weft")
    vals = {n: _positive(parse_cited((raw or {}).get(n), where, f"cloth.{n}"), where, f"cloth.{n}") for n in names}
    missing = [n for n, v in vals.items() if v is None]
    if missing:
        raise SpecError(where, "cloth", f"missing {missing} (every cloth constant needs a value and a cite)")
    return ClothSpec(**vals)  # type: ignore[arg-type]


def _cable(raw: Any, where: str) -> CableSpec | None:
    if raw is None:
        return None
    pts = raw.get("points") if isinstance(raw, dict) else None
    if not isinstance(pts, list) or len(pts) < 2:
        raise SpecError(where, "cable.points", "expected >= 2 [x, y, z] points (body frame, m)")
    points = tuple(tuple(_num(c, where, "cable.points") for c in p) for p in pts)
    if any(len(p) != 3 for p in points):
        raise SpecError(where, "cable.points", "every point needs 3 coordinates")
    radius = _positive(parse_cited(raw.get("radius"), where, "cable.radius"), where, "cable.radius")
    if radius is None:
        raise SpecError(where, "cable.radius", "a cable needs a cited radius")
    ei = _positive(parse_cited(raw.get("bend_stiffness"), where, "cable.bend_stiffness"), where, "cable.bend_stiffness")
    return CableSpec(points, radius, ei)  # type: ignore[arg-type]


def _unit(raw: Any, where: str, fld: str) -> float | None:
    if raw is None:
        return None
    v = _num(raw, where, fld)
    if not 0.0 <= v <= 1.0:
        raise SpecError(where, fld, f"{v} outside [0, 1]")
    return v


def _visual(raw: Any, where: str, base: Path) -> tuple[VisualPart, ...]:
    out = []
    for i, d in enumerate(raw or ()):
        f = f"visual.{i}"
        if not isinstance(d, dict) or "mesh" not in d or "base_color" not in d:
            raise SpecError(where, f, "expected {mesh, base_color, cite[, roughness, metallic, texture, name]}")
        col = d["base_color"]
        if not isinstance(col, (list, tuple)) or len(col) != 3:
            raise SpecError(where, f"{f}.base_color", "expected [r, g, b] in 0..1 (sRGB)")
        tex = d.get("texture")
        out.append(VisualPart(base / _str(d["mesh"], where, f"{f}.mesh"),
                              tuple(_unit(c, where, f"{f}.base_color") for c in col),  # type: ignore[arg-type]
                              _str(d.get("cite"), where, f"{f}.cite"), _unit(d.get("roughness"), where, f"{f}.roughness"),
                              _unit(d.get("metallic"), where, f"{f}.metallic"),
                              base / _str(tex, where, f"{f}.texture") if tex else None, str(d.get("name") or "")))
    return tuple(out)


def _ident(raw: dict[str, Any], where: str) -> dict[str, Any]:
    if raw.get("schema") != SCHEMA:
        raise SpecError(where, "schema", f"expected {SCHEMA!r}, got {raw.get('schema')!r}")
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise SpecError(where, "<top>", f"unknown keys {sorted(unknown)}; put free-form data under 'extra'")
    key, sim = _str(raw.get("key"), where, "key"), _str(raw.get("sim"), where, "sim")
    for fld, v in (("key", key), ("sim", sim)):
        if not KEY_RE.match(v):
            raise SpecError(where, fld, f"{v!r}: use lower-case letters, digits and '_' only")
    return {"key": key, "sim": sim, "name": _str(raw.get("name"), where, "name"),
            "category": _choice(raw.get("category"), CATEGORIES, where, "category"),
            "kind": _choice(raw.get("kind", "rigid"), KINDS, where, "kind"),
            "motion": _choice(raw.get("motion", "dynamic"), MOTIONS, where, "motion")}


def parse_spec(raw: Any, where: str, base: Path | None = None) -> GarSpec:
    """Validate one decoded spec; ``mesh`` resolves against ``base`` (the spec file's folder). Raises SpecError."""
    if not isinstance(raw, dict):
        raise SpecError(where, "<top>", "a spec is a JSON object")
    ident = _ident(raw, where)
    mesh = raw.get("mesh")
    if mesh is None and ident["kind"] != "cable":
        raise SpecError(where, "mesh", "every non-cable object needs a mesh (OBJ/STL, body frame, metres)")
    color = raw.get("color_rgb")
    spec = GarSpec(
        **ident, mesh=(base or Path(".")) / _str(mesh, where, "mesh") if mesh is not None else None,
        material=_material(raw.get("material"), where),
        mass=_positive(parse_cited(raw.get("mass"), where, "mass"), where, "mass"),
        friction=parse_cited(raw.get("friction"), where, "friction"),
        friction_partner=_str(raw.get("friction_partner"), where, "friction_partner", required=False),
        restitution=parse_cited(raw.get("restitution"), where, "restitution"),
        thickness=_positive(parse_cited(raw.get("thickness"), where, "thickness"), where, "thickness"),
        dims=_dims(raw.get("dims"), where), source_ref=_str(raw.get("source_ref"), where, "source_ref", required=False),
        licence=_str(raw.get("licence", "MIT"), where, "licence"), notes=_str(raw.get("notes"), where, "notes", False),
        color_rgb=tuple(_num(c, where, "color_rgb") for c in color) if color is not None else None,  # type: ignore[arg-type]
        cloth=_cloth(raw.get("cloth"), where), cable=_cable(raw.get("cable"), where),
        visual=_visual(raw.get("visual"), where, base or Path(".")), extra=dict(raw.get("extra") or {}))
    _check_kind(spec, where)
    return spec


def _check_kind(s: GarSpec, where: str) -> None:
    if s.friction is not None and not 0.0 <= s.friction.value <= 2.0:
        raise SpecError(where, "friction.value", f"{s.friction.value} outside [0, 2]")
    if s.restitution is not None and not 0.0 <= s.restitution.value <= 1.0:
        raise SpecError(where, "restitution.value", f"{s.restitution.value} outside [0, 1]")
    if s.kind == "cloth" and s.cloth is None:
        raise SpecError(where, "cloth", "kind 'cloth' needs the cloth block")
    if s.kind == "cable" and s.cable is None:
        raise SpecError(where, "cable", "kind 'cable' needs the cable block")
    if s.kind == "soft" and (s.material.youngs is None or s.material.poisson is None):
        raise SpecError(where, "material", "kind 'soft' needs material.youngs and material.poisson")
    if s.color_rgb is not None and (len(s.color_rgb) != 3 or not all(0 <= c <= 1 for c in s.color_rgb)):
        raise SpecError(where, "color_rgb", "expected [r, g, b] in 0..1")


def load_spec(path: str | Path) -> GarSpec:
    """Read and validate one ``*.json`` spec file.

    Example: ``load_spec("specs/wine_pouring/wine_glass.json").card_name == "gar_wine_pouring_wine_glass"``.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(str(path), "<file>", f"cannot read JSON: {exc}") from exc
    return parse_spec(raw, str(path), path.parent)


def find_specs(root: str | Path) -> list[Path]:
    """Every ``*.json`` spec under ``root`` (recursive, sorted); files starting with ``_`` are skipped."""
    return sorted(p for p in Path(root).rglob("*.json") if not p.name.startswith("_"))
