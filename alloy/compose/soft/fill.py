# SPDX-License-Identifier: Apache-2.0
"""Liquids and grains in containers: a fill starts at rest inside its container (a column standing on the inner
floor), a pour is an emitter stream above the container, from a held vessel when the sentence names one.

Never a block dropped into a container: a flat block striking a floor makes pressure jets that throw liquid out
(physics ruling on SCR-OSS3-2), so a fill stands on the floor as wide as the cavity allows, and a pour is a stream.

Amounts: ``volume_ml``, else ``mass_g`` / the material density, else ``fill`` x the container depth, else 60 % of
the depth. The settled level must stay below 90 % of the depth; the result lists the capacity either way.

Example::

    >>> col = column(interior, 200e-6, margin=0.005, box=False)        # doctest: +SKIP
    >>> round(col.radius, 3), round(col.height, 3)                     # doctest: +SKIP
    (0.033, 0.058)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.compose import geom
from alloy.compose.errors import ComposeError
from alloy.compose.soft import vocab as V
from alloy.compose.soft.interior import Interior
from alloy.compose.soft.spec import MAX_FILL, SoftSpec

SPH_RADIUS_MAX = 2e-3        # m: the default liquid_accuracy's particle radius (route.types.LIQUID_SPACING / 2)
SPH_PASSAGE = 16.0           # the router's old r = min(2 mm, passage / 16), kept for readers (no longer applied)
GRAIN_MARGIN = 3e-3          # m from walls and floor for a grain region
POUR_RATE = 5e-5             # m^3/s: 50 ml/s, a steady pour from a bottle (estimate)
POUR_SPEED = 0.5             # m/s, the stream's start speed (down)
POUR_DROP = 0.08             # m from the stream's start to the container's rim
POUR_MOUTH = 5e-3            # m, half width of the stream's start
POUR_TAIL_S = 0.6            # s after the stream stops for the liquid to settle
DEFAULT_POUR_ML = 150.0
COLUMN_ITERS = 8
RIM_GAP = 2e-3               # m: a starting column's top stays this far below the rim


@dataclass(frozen=True)
class Column:
    """A starting column in the container's rest frame: a cylinder (``radius``) or a box (half sides ``hx``,
    ``hy``), centre (x, y), bottom z and height (m)."""

    x: float
    y: float
    z0: float
    height: float
    radius: float = 0.0
    hx: float = 0.0
    hy: float = 0.0

    @property
    def box(self) -> bool:
        return self.hx > 0.0

    @property
    def area(self) -> float:
        return 4.0 * self.hx * self.hy if self.box else math.pi * self.radius ** 2

    def volume(self) -> float:
        return self.area * self.height

    def to_dict(self) -> dict:
        size = {"half_mm": [round(self.hx * 1e3, 1), round(self.hy * 1e3, 1)]} if self.box else \
            {"radius_mm": round(self.radius * 1e3, 1)}
        return {"shape": "box" if self.box else "cylinder", **size, "height_mm": round(self.height * 1e3, 1)}


def sph_margin(passage: float) -> float:
    """Gap kept from walls and floor for SPH particles: one particle radius plus 0.2 mm (the lattice's outer particle
    centres lie inside the column, so no particle touches a wall and none starts more than a radius above the
    floor)."""
    del passage                     # the router no longer refines below a narrow passage (SCR-WP9-LIQRES): the
    return SPH_RADIUS_MAX + 2e-4    # default radius, a safe gap for super's 1 mm particles too


def bulk_density(material: Any, kind: str) -> float:
    """kg/m^3 of ``material`` as it fills a space: a granular card's density is its grain (solid) density, and grains
    fill at the router's packing (alloy.route.checks.GRAIN_PACKING, 0.6), as the engine builds them; liquids and
    pastes fill at their density. Example: ``bulk_density(lib.material("rice_grain"), "grain")`` -> 876.0."""
    from alloy.route.checks import GRAIN_PACKING  # noqa: PLC0415

    rho = float(material.density.value)
    return GRAIN_PACKING * rho if kind == "grain" else rho


def amount_m3(soft: SoftSpec, interior: Interior | None, density: float) -> tuple[float, str]:
    """The volume asked for (m^3) and how it was read ("200 ml", "300 g at 1600 kg/m^3", "50 % of the depth")."""
    if soft.volume_ml is not None:
        return soft.volume_ml * 1e-6, f"{soft.volume_ml:g} ml"
    if soft.mass_g is not None:
        return soft.mass_g * 1e-3 / density, f"{soft.mass_g:g} g at {density:g} kg/m^3"
    if interior is None:
        return (V.DEFAULT_PILE_ML if soft.form == "pile" else DEFAULT_POUR_ML) * 1e-6, "default amount"
    share = soft.fill if soft.fill is not None else V.DEFAULT_FILL
    return interior.volume_below(interior.floor + share * interior.depth), f"{share:.0%} of the depth"


def check_capacity(interior: Interior, volume: float, what: str, where: str) -> float:
    """The settled level (rest frame z) of ``volume``; ``ComposeError`` when it passes 90 % of the depth."""
    level = interior.level_for(volume)
    limit = interior.floor + MAX_FILL * interior.depth
    if level > limit + 1e-9:
        cap = interior.capacity_ml(MAX_FILL)
        raise ComposeError(f"{volume * 1e6:.0f} ml of {what} does not fit in {where} (it holds {cap:.0f} ml to "
                           f"{MAX_FILL:.0%} of its depth)", f"ask for at most {cap:.0f} ml, or a larger container",
                           field="items.soft.volume_ml")
    return level


def column(interior: Interior, volume: float, *, margin: float, box: bool) -> Column | None:
    """The widest column of ``volume`` standing on the inner floor that fits in the cavity (``margin`` from walls,
    floor and any shoulder above it): a cylinder or a box, whichever holds more (``box=True``: a box only, as a
    particle region needs); None when none fits below the rim."""
    z0 = interior.floor + margin
    top = interior.rim - RIM_GAP
    h, shape = margin, None
    for _ in range(COLUMN_ITERS):
        shape = _widest(interior, z0, z0 + h + margin, margin, box)
        if shape is None:
            return None
        h = volume / shape.area
        if z0 + h > top:
            return None
    final = _widest(interior, z0, z0 + h + margin, margin, box)
    if final is None or final.area < shape.area - 1e-12:
        return None
    return Column(shape.x, shape.y, z0, h, shape.radius, shape.hx, shape.hy)


def _widest(it: Interior, z0: float, z1: float, margin: float, box: bool) -> Column | None:
    cx, cy = (float(v) for v in it.centre)
    hx, hy = (max(v - margin, 0.0) for v in it.rect_at(z0, z1))
    r = it.radius_at(z0, z1) - margin
    rect = Column(cx, cy, z0, 0.0, hx=hx, hy=hy) if min(hx, hy) > margin else None
    disc = Column(cx, cy, z0, 0.0, radius=r) if (r > margin and not box) else None
    options = [c for c in (rect, disc) if c is not None]
    return max(options, key=lambda c: c.area) if options else None


def column_pose(container, col: Column, *, bottom: bool = False) -> tuple[float, ...]:
    """World pose of a column in ``container`` (a layout ``Placed``): its centre, or its bottom centre with
    ``bottom=True`` (a particle region's origin), turned with the container's yaw."""
    rot = geom.yaw_matrix(container.yaw)
    z = col.z0 if bottom else col.z0 + 0.5 * col.height
    p = rot @ np.array([col.x, col.y, z]) + container.pos
    return tuple(float(v) for v in p) + geom.quat_from_matrix(rot)


@dataclass(frozen=True)
class Pour:
    """A pour: the stream's start (world), rate (m^3/s), duration (s) and, with a vessel, its held pose."""

    start: tuple[float, float, float]
    rate: float
    seconds: float
    vessel_pose: tuple[float, ...] | None = None


def pour_over(container, interior: Interior | None, volume: float) -> Pour:
    """A stream from ``POUR_DROP`` above the container's opening at ``POUR_RATE`` until ``volume`` has left."""
    rot = geom.yaw_matrix(container.yaw)
    centre = np.zeros(2) if interior is None else interior.centre
    rim = float(container.body.vertices[:, 2].max())
    p = rot @ np.array([centre[0], centre[1], rim + POUR_DROP]) + container.pos
    return Pour((float(p[0]), float(p[1]), float(p[2])), POUR_RATE, volume / POUR_RATE)


TILT = math.radians(125.0)          # a held vessel's tilt from upright (its mouth points down and out)


def hold_vessel(body, mouth_rest: np.ndarray, stream: np.ndarray, toward: float) -> tuple[float, ...]:
    """Pose of a vessel tilted ``TILT`` about the horizontal axis normal to ``toward`` (yaw, rad), its mouth just
    above the stream's start. ``mouth_rest``: the mouth centre in the vessel's rest frame."""
    yaw = geom.yaw_matrix(toward)
    axis = yaw @ np.array([0.0, 1.0, 0.0])
    tilt = _axis_angle(axis, TILT)
    rot = tilt @ yaw
    mouth_world = np.asarray(stream) + np.array([0.0, 0.0, 0.01])
    pos = mouth_world - rot @ mouth_rest
    return tuple(float(v) for v in pos) + geom.quat_from_matrix(rot)


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


__all__ = ["Column", "Pour", "amount_m3", "check_capacity", "column", "column_pose", "hold_vessel", "pour_over",
           "sph_margin"]
