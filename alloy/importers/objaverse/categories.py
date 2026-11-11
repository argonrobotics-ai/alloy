# SPDX-License-Identifier: Apache-2.0
"""Robot-relevant Objaverse LVIS categories: group, base material, real-world size and mass priors.

Objaverse meshes carry no real-world scale or mass. Every imported object is rescaled so its longest side matches its
category's size prior, and its mass is checked against (or, for hollow scans, set from) the category's mass prior.
Both priors are ESTIMATES (typical retail items), labelled as such in every card, and queued for the sourcing agent.

The table lives in ``categories.yaml`` next to this module (schema ``alloy.objaverse.categories/1``).

Example:
    >>> from alloy.importers.objaverse.categories import load_categories
    >>> cats = load_categories()
    >>> mug = cats["mug"]
    >>> mug.group, mug.material, 0.05 < mug.size.value < 0.2
    ('kitchen', 'stoneware_glazed', True)
"""

from __future__ import annotations

import functools
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from alloy.importers.objaverse.errors import ImporterConfigError

TABLE = Path(__file__).with_name("categories.yaml")
SCHEMA = "alloy.objaverse.categories/1"
PREFIX = "objv"
GROUPS = ("kitchen", "lab", "tools", "containers", "food", "office", "packaging")


@dataclass(frozen=True)
class Prior:
    """A typical value with a plausible range [lo, hi] and the note that says where it comes from."""

    value: float
    lo: float
    hi: float
    note: str

    def contains(self, x: float, slack: float = 1.0) -> bool:
        """True if ``x`` lies in [lo / slack, hi * slack].

        Example:
            >>> Prior(1.0, 0.5, 2.0, "n").contains(3.0, slack=2.0)
            True
        """
        return self.lo / slack <= x <= self.hi * slack


@dataclass(frozen=True)
class Category:
    """One robot-relevant LVIS category.

    ``size`` is the longest side [m]; ``mass`` [kg]; ``hollow`` marks items whose scans are usually closed outer
    shells (bottles, cans, boxes), so density x enclosed volume overstates the mass; ``flat`` marks items that are
    thin by nature (plates, knives); ``cap`` limits how many objects of this category the library keeps.
    """

    name: str
    group: str
    material: str
    size: Prior
    mass: Prior
    hollow: bool = False
    flat: bool = False
    cap: int = 30
    lvis: tuple[str, ...] = ()
    spec: str = ""


def _triple(raw: Any, where: str) -> tuple[float, float, float]:
    try:
        lo, value, hi = (float(v) for v in raw)
    except (TypeError, ValueError):
        raise ImporterConfigError(f"{where}: expected [lo, typical, hi], got {raw!r}") from None
    if not 0.0 < lo <= value <= hi:
        raise ImporterConfigError(f"{where}: need 0 < lo <= typical <= hi, got [{lo}, {value}, {hi}]")
    return lo, value, hi


def _category(name: str, row: Any, defaults: Mapping[str, Any]) -> Category:
    """One table row: [group, material, size[lo, typ, hi], mass[lo, typ, hi], flags, spec, (lvis), (cap)]."""
    where = f"categories.{name}"
    if not isinstance(row, (list, tuple)) or not 6 <= len(row) <= 8:
        raise ImporterConfigError(f"{where}: expected [group, material, size, mass, flags, spec, (lvis), (cap)]")
    group, material, size, mass, flags, spec = row[:6]
    if group not in GROUPS:
        raise ImporterConfigError(f"{where}.group: {group!r} is not one of {GROUPS}")
    if not material:
        raise ImporterConfigError(f"{where}.material: name a material card (e.g. pp_homopolymer)")
    flag_set = {f.strip() for f in str(flags or "").split(",") if f.strip()}
    if flag_set - {"hollow", "flat"}:
        raise ImporterConfigError(f"{where}.flags: unknown {sorted(flag_set - {'hollow', 'flat'})} (hollow, flat)")
    slo, sv, shi = _triple(size, f"{where}.size")
    mlo, mv, mhi = _triple(mass, f"{where}.mass")
    lvis = tuple(row[6]) if len(row) > 6 and row[6] else (name,)
    cap = int(row[7]) if len(row) > 7 else int(defaults.get("cap", 30))
    return Category(name=name, group=str(group), material=str(material),
                    size=Prior(sv, slo, shi, f"ESTIMATE longest side of a typical {spec} (no single source)"),
                    mass=Prior(mv, mlo, mhi, f"ESTIMATE typical mass of a {spec} (no single source)"),
                    hollow="hollow" in flag_set, flat="flat" in flag_set, cap=cap, lvis=lvis, spec=str(spec))


def parse_categories(doc: Mapping[str, Any]) -> dict[str, Category]:
    """Parse a category table document (the YAML already loaded). Raises :class:`ImporterConfigError`.

    Example:
        >>> doc = {"schema": SCHEMA, "categories": {"cup": ["kitchen", "pp_homopolymer", [0.08, 0.1, 0.12],
        ...        [0.005, 0.01, 0.02], "hollow", "disposable cup"]}}
        >>> c = parse_categories(doc)["cup"]
        >>> c.cap, c.hollow, c.size.note
        (30, True, 'ESTIMATE longest side of a typical disposable cup (no single source)')
    """
    if doc.get("schema") != SCHEMA:
        raise ImporterConfigError(f"schema: expected {SCHEMA!r}, got {doc.get('schema')!r}")
    defaults = doc.get("defaults", {}) or {}
    cats = {name: _category(name, raw, defaults) for name, raw in (doc.get("categories") or {}).items()}
    seen: dict[str, str] = {}
    for c in cats.values():
        for label in c.lvis:
            if label in seen:
                raise ImporterConfigError(f"categories.{c.name}.lvis: {label!r} is already mapped by {seen[label]}")
            seen[label] = c.name
    return cats


@functools.cache
def load_categories(path: Path = TABLE) -> dict[str, Category]:
    """The category table (cached). ``path`` defaults to the packaged ``categories.yaml``."""
    return parse_categories(yaml.safe_load(Path(path).read_text()))


def lvis_index(cats: Mapping[str, Category]) -> dict[str, Category]:
    """LVIS label -> Category, for every label any category claims.

    Example:
        >>> idx = lvis_index(load_categories())
        >>> idx["mug"].name
        'mug'
    """
    return {label: c for c in cats.values() for label in c.lvis}


def card_name(category: str, uid: str) -> str:
    """Stable card name: ``objv_<category slug>_<first 8 hex of the uid>``.

    Example:
        >>> card_name("orange_(fruit)", "0f3a9c51d2e84b7a9e1c2d3f4a5b6c7d")
        'objv_orange_fruit_0f3a9c51'
    """
    slug = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
    return f"{PREFIX}_{slug}_{uid[:8]}"
