# SPDX-License-Identifier: Apache-2.0
"""The mesh's real size: from your ``size`` (longest side, m) or ``units``, else inferred, and absurd sizes rejected.

Inference for a mesh file: the unit (m, mm, cm, in) that puts the longest side inside the category's typical range
(within a factor of 2); without a category, metres when the longest side is 5 mm to 3 m, millimetres when it is
5 to 3000. A generated mesh (photo, text) has no units: the category's typical size is used (ESTIMATE). Anything that
ends up shorter than ``MIN_SIZE_M`` or longer than ``MAX_SIZE_M`` is refused with the argument that fixes it.

Example::

    >>> s = resolve_scale(longest=120.0, size=None, units=None, generated=False, cat=None)
    >>> s.factor, s.value.tag
    (0.001, 'ESTIMATE')
"""

from __future__ import annotations

from dataclasses import dataclass

from alloy.create.physics import Category, Value
from alloy.create.request import UNITS, CreateError

MIN_SIZE_M = 0.003
MAX_SIZE_M = 3.0
PRIOR_SLACK = 2.0                  # a unit fits a category when the size is within [lo / 2, hi * 2]
FAR_SLACK = 3.0                    # beyond [lo / 3, hi * 3] the result asks you to check the size


@dataclass(frozen=True)
class Scale:
    """Source units -> metres (``factor``) and the resulting longest side as a provenance row."""

    factor: float
    value: Value


def resolve_scale(longest: float, size: float | None, units: str | None, generated: bool,
                  cat: Category | None) -> Scale:
    """See the module docstring. ``longest``: the mesh's longest side in its own units."""
    if not longest > 0:
        raise CreateError("the mesh has no extent (all vertices at one point)", "check the file", field="mesh")
    if size is not None:
        out = Scale(size / longest, Value("size", size, "m", "USER", "longest side, given by you"))
    elif units is not None:
        out = Scale(UNITS[units], Value("size", longest * UNITS[units], "m", "USER", f"mesh units {units}, given"))
    elif generated:
        out = _from_category(longest, cat)
    else:
        out = _infer_units(longest, cat)
    _check(out.value.value, cat)
    out.value.verify = out.value.verify or _far(out.value.value, cat)
    return out


def _from_category(longest: float, cat: Category | None) -> Scale:
    if cat is None or cat.size is None:
        raise CreateError("a generated mesh has no real size and the object's category is unknown",
                          "pass size= (its longest side in metres, e.g. size=0.12) or category= (e.g. 'mug')",
                          field="size")
    lo, typ, hi = cat.size
    v = Value("size", typ, "m", "ESTIMATE", f"typical {cat.name} longest side ({cat.size_note})", lo, hi,
              verify=f"measure its longest side (typical {cat.name} {lo * 100:.0f}-{hi * 100:.0f} cm); pass size=")
    return Scale(typ / longest, v)


def _infer_units(longest: float, cat: Category | None) -> Scale:
    order = ("m", "mm", "cm", "in")
    if cat is not None and cat.size is not None:
        lo, _, hi = cat.size
        for u in order:
            if lo / PRIOR_SLACK <= longest * UNITS[u] <= hi * PRIOR_SLACK:
                return Scale(UNITS[u], Value("size", longest * UNITS[u], "m", "ESTIMATE",
                                             f"mesh units read as {u} (fits a typical {cat.name})",
                                             verify=f"check the units; pass units= if not {u}"))
    for u, (a, b) in (("m", (0.005, 3.0)), ("mm", (5.0, 3000.0))):
        if a <= longest <= b:
            return Scale(UNITS[u], Value("size", longest * UNITS[u], "m", "ESTIMATE", f"mesh units read as {u} "
                                         f"({longest:.4g} units long)", verify="check the units; pass units= or size="))
    raise CreateError(f"the mesh is {longest:.4g} units long and its units are unclear",
                      "pass size= (longest side in metres) or units= ('m', 'mm', 'cm', 'in')", field="size")


def _check(meters: float, cat: Category | None) -> None:
    if not MIN_SIZE_M <= meters <= MAX_SIZE_M:
        raise CreateError(f"the object would be {meters:.4g} m long, outside {MIN_SIZE_M}-{MAX_SIZE_M} m",
                          "pass the real longest side as size= (metres) or the mesh's units=", field="size")


def _far(meters: float, cat: Category | None) -> str:
    if cat is None or cat.size is None:
        return ""
    lo, _, hi = cat.size
    if lo / FAR_SLACK <= meters <= hi * FAR_SLACK:
        return ""
    return f"{meters:.3g} m is far from a typical {cat.name} ({lo:.3g}-{hi:.3g} m): check size"


__all__ = ["MAX_SIZE_M", "MIN_SIZE_M", "Scale", "resolve_scale"]
