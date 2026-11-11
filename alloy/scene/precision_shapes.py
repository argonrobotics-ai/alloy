# SPDX-License-Identifier: Apache-2.0
"""Touched shapes of a precision feature (SCR-DC3-5 part 1, coordinator ruling at the end of [internal note]).

A card's precision partner relation names two objects and one PrecisionFeature on each. An object with one shape
pairs that shape, exactly as before this rule (the crop is in that shape's frame). On an object with several shapes
(an MJCF plug: pad faces, leads, a latch arm), the feature covers each collidable shape that matches it:

- a label match: the feature name occurs in the shape's label (the MJCF geom name), e.g. feature "pad" → pad_0..pad_5.
  The whole shape is the feature (no crop);
- a crop match: the shape's centre (its local bounds' centre, in its body frame) lies inside the feature crop. On a
  multi-shape object the crop is given in the body frame. Each mesh shape gets the crop moved into its own frame,
  as the axis-aligned bounds of the box's eight corners, because the contact layer crops in the shape frame. A
  primitive shape is the feature as a whole (no crop).

A feature that matches no shape is a PrecisionSpecError. The pairs are every touched shape of one side against every
touched shape of the other, at most MAX_SHAPE_PAIRS per relation. Each pair is filtered from native collision, as
before.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.cards.types import PrecisionFeature
from alloy.core.errors import PrecisionSpecError

log = logging.getLogger("alloy.scene")
MAX_SHAPE_PAIRS = 64
_COLLIDE = int(newton.ShapeFlags.COLLIDE_SHAPES)
_MESHES = (int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH))


@dataclass(frozen=True)
class ShapePair:
    """One precision shape pair of a relation: template shape ids and each side's feature (crop in its shape frame)."""

    a: str
    b: str
    shape_a: int
    shape_b: int
    feature_a: PrecisionFeature
    feature_b: PrecisionFeature


def collidable(b: newton.ModelBuilder, shapes: range) -> list[int]:
    return [s for s in shapes if int(b.shape_flags[s]) & _COLLIDE]


def _local_center(b: newton.ModelBuilder, s: int) -> np.ndarray:
    """Centre of a shape's own bounds in its shape frame (the origin for primitives)."""
    src = b.shape_source[s]
    if int(b.shape_type[s]) in _MESHES and src is not None and getattr(src, "vertices", None) is not None:
        v = np.asarray(src.vertices, np.float64) * np.asarray(b.shape_scale[s], np.float64)
        return 0.5 * (v.min(axis=0) + v.max(axis=0))
    return np.zeros(3)


def body_center(b: newton.ModelBuilder, s: int) -> np.ndarray:
    """The shape's centre in its body frame."""
    x = wp.transform(*b.shape_transform[s])
    return np.asarray(wp.transform_point(x, wp.vec3(*_local_center(b, s))), np.float64)


def _inside(p: np.ndarray, crop) -> bool:
    lo, hi = np.asarray(crop[0], np.float64), np.asarray(crop[1], np.float64)
    return bool(np.all(p >= lo) and np.all(p <= hi))


def _label_match(label: str, feature: str) -> bool:
    return feature in str(label).rsplit("/", 1)[-1]


def shape_crop(b: newton.ModelBuilder, s: int, crop) -> tuple | None:
    """A body-frame crop box in shape s's own frame (bounds of the moved corners); None for a primitive."""
    if crop is None or int(b.shape_type[s]) not in _MESHES:
        return None
    inv = wp.transform_inverse(wp.transform(*b.shape_transform[s]))
    lo, hi = np.asarray(crop[0], np.float64), np.asarray(crop[1], np.float64)
    corners = [np.asarray(wp.transform_point(inv, wp.vec3(*np.where(m, hi, lo))), np.float64)
               for m in itertools.product((False, True), repeat=3)]
    pts = np.asarray(corners)
    return tuple(float(x) for x in pts.min(axis=0)), tuple(float(x) for x in pts.max(axis=0))


def touched(b: newton.ModelBuilder, name: str, shapes: range, feature: PrecisionFeature) -> list[tuple[int, PrecisionFeature]]:
    """(shape, feature as that shape sees it) for each shape the feature covers (module docstring)."""
    if len(shapes) <= 1:
        return [(shapes.start, feature)]                  # one shape: unchanged
    cand = collidable(b, shapes)
    label = [s for s in cand if _label_match(b.shape_label[s], feature.name)]
    inside = [s for s in cand if s not in label and feature.crop is not None
              and _inside(body_center(b, s), feature.crop)]
    if not label and not inside:
        raise PrecisionSpecError(f"{name}: precision feature {feature.name!r} matches no shape label and its crop "
                                 f"{feature.crop} (body frame) holds none of the object's {len(cand)} collidable "
                                 "shapes")
    out = [(s, replace(feature, crop=None)) for s in label]
    out += [(s, replace(feature, crop=shape_crop(b, s, feature.crop))) for s in inside]
    return sorted(out, key=lambda x: x[0])


def shape_pairs(b: newton.ModelBuilder, rels, ranges: dict[str, range]) -> list[ShapePair]:
    """Every touched shape of a against every touched shape of c, per relation (a, c, fa, fc)."""
    out = []
    for a, c, fa, fc in rels:
        sa, sc = touched(b, a, ranges[a], fa), touched(b, c, ranges[c], fc)
        if len(sa) * len(sc) > MAX_SHAPE_PAIRS:
            raise PrecisionSpecError(f"{a}/{c}: {len(sa)} x {len(sc)} touched shapes > {MAX_SHAPE_PAIRS} precision "
                                     "pairs; narrow the features' crops or labels")
        if len(sa) * len(sc) > 1:
            log.info("precision %s/%s: %d x %d touched shapes (%s / %s)", a, c, len(sa), len(sc), fa.name, fc.name)
        out.extend(ShapePair(a, c, x, y, gx, gy) for (x, gx), (y, gy) in itertools.product(sa, sc))
    return out


def shape_mass(b: newton.ModelBuilder, bodies: range, s: int, whole: float, moving: Sequence[int] | None) -> float:
    """Mass behind one shape: the object's (one body), else the shape's body when it moves (0 when it does not)."""
    if len(bodies) <= 1 or whole == 0.0:
        return whole
    body = int(b.shape_body[s])
    if moving is not None and body not in moving:
        return 0.0
    return float(b.body_mass[body])


__all__ = ["MAX_SHAPE_PAIRS", "ShapePair", "body_center", "collidable", "shape_crop", "shape_mass", "shape_pairs",
           "touched"]
