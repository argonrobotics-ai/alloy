# SPDX-License-Identifier: Apache-2.0
"""Where each object goes: support, clearance and clutter rules, in pure geometry (no engine).

An object is dropped straight down at a sampled (x, y, yaw): its start height puts its underside ``GAP`` above the
highest surface beneath it (the table, or the floor of the tray it goes in), found by casting vertical rays at a
grid of points plus the corners of both shapes. It is accepted when

- its outline stays ``EDGE`` inside the table, or inside the outline of the object it goes on or in;
- its outline keeps the clutter level's clearance from every other object (except the one it rests on);
- the surface under it is flat, or its centre of mass sits inside the patch it touches (``STABLE_MARGIN``).

Example::

    >>> lay = Layout(table, clutter="normal", rng=np.random.default_rng(0))      # doctest: +SKIP
    >>> placed = lay.place(body, "cube_beech_30", "cube_beech_30")              # doctest: +SKIP
    >>> placed.pose                                                              # doctest: +SKIP
    (0.031, -0.12, 0.0155, 0.0, 0.0, 0.38, 0.92)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from alloy.compose import geom
from alloy.compose.body import Body

CLEARANCE = {"tidy": 0.04, "normal": 0.015, "cluttered": 0.004}       # m between outlines
FILL = {"tidy": 0.08, "normal": 0.18, "cluttered": 0.4}              # share of the work area the outlines cover
INSIDE_CLEARANCE = 0.003            # m between objects that share a container
EDGE = 0.02                         # m from the edge of the table
INSIDE_EDGE = 0.001                 # m inside the outline of the object an item goes in or on
GAP = 5e-4                          # m start gap above the surface (as alloy.check.check_object)
FLAT = 1.5e-3                       # m: a surface varying less than this under an object is flat
CONTACT = 1e-3                      # m: points this close to the lowest gap touch
STABLE_MARGIN = 2e-3                # m: centre of mass at least this far inside the contact patch
GRID = 9                            # sample points per axis under an object
TRIES = 80                          # candidate poses per object
MIN_AREA = (0.3, 0.2)               # m, smallest work area on a support
ALIGN_JITTER = math.radians(4.0)    # rad, spread of the yaw of objects set square to the table


@dataclass
class Placed:
    """One object put down: scene name ``key``, card, rest-frame body, yaw (rad) and origin position (m)."""

    key: str
    card: str
    body: Body
    yaw: float
    pos: np.ndarray
    target: str | None = None          # key of the object it is on or in (None: the support)
    item: int = -1                     # index of the spec item (-1: the support)
    _rot: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rot = geom.yaw_matrix(self.yaw)

    @property
    def pose(self) -> tuple[float, ...]:
        """Placement pose (x, y, z, qx, qy, qz, qw): yaw applied after the body's rest rotation."""
        rot = self._rot @ geom.matrix_from_quat(self.body.rest)
        return tuple(round(float(v), 7) for v in self.pos) + geom.quat_from_matrix(rot)     # quaternion unrounded

    @property
    def outline(self) -> np.ndarray:
        """Footprint polygon in world xy."""
        return self.body.footprint @ self._rot[:2, :2].T + self.pos[None, :2]

    def triangles(self) -> np.ndarray:
        return self.body.triangles(self._rot, self.pos)

    def com(self) -> np.ndarray:
        return self._rot @ self.body.com + self.pos


class Layout:
    """Objects placed so far on one support. ``clutter`` sets clearances and the work area; ``rng`` the sampling.

    Example: see the module docstring."""

    def __init__(self, support: Placed, clutter: str = "normal", rng: np.random.Generator | None = None) -> None:
        self.support = support
        self.clutter = clutter
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.placed: dict[str, Placed] = {}
        self.obstacles: list[np.ndarray] = []           # keep-out polygons (a robot's base), world xy
        self._within: tuple[np.ndarray, np.ndarray] | None = None
        self.area = self.table_box()
        self._slots: list[tuple[float, float, float]] = []

    # --- work area -----------------------------------------------------------------------------------------------

    def table_box(self) -> tuple[np.ndarray, np.ndarray]:
        out = self.support.outline
        return out.min(axis=0) + EDGE, out.max(axis=0) - EDGE

    def plan_area(self, bodies: list[Body], arrangement: str = "scatter",
                  within: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        """Size the work area for these objects (fewer, tidier objects spread wider) inside ``within`` (default the
        table) and, for "row"/"grid", the slots they take in order."""
        tlo, thi = self.table_box()
        lo, hi = (tlo, thi) if within is None else (np.maximum(within[0], tlo), np.minimum(within[1], thi))
        full = np.maximum(hi - lo, 1e-3)
        need = sum(b.area for b in bodies) / FILL[self.clutter]
        aspect = full[0] / max(full[1], 1e-9)
        w = math.sqrt(max(need, 1e-6) * aspect)
        size = np.clip([w, w / aspect], np.minimum(MIN_AREA, full), full)
        centre = (lo + hi) / 2
        self.area = (centre - size / 2, centre + size / 2)
        self._within = within
        self._slots = _slots(bodies, arrangement, centre, size, CLEARANCE[self.clutter]) if arrangement != \
            "scatter" else []

    # --- placing -------------------------------------------------------------------------------------------------

    def place(self, body: Body, card: str, key: str, target: str | None = None, item: int = -1,
              tries: int = TRIES) -> Placed | None:
        """Put one object down (on the support, or on/in ``target``); None when no try fits."""
        slot = self._slots.pop(0) if self._slots and target is None else None
        for k in range(tries):
            if target is not None and k >= tries - 2:             # last tries: centred, square to the target
                x, y, yaw = self._centred(body, target, quarter=tries - 1 - k)
            else:
                x, y, yaw = slot if (slot is not None and k == 0) else self._sample(target, wide=k >= tries // 2)
            p = self._fit(body, card, key, target, item, x, y, yaw)
            if p is not None:
                self.placed[key] = p
                return p
        return None

    def remove(self, key: str) -> list[Placed]:
        """Take an object and whatever stands on or in it off again; returns them, the object first."""
        if key not in self.placed:
            return []
        out = [self.placed.pop(key)]
        for k in [k for k, p in self.placed.items() if p.target == key]:
            out.extend(self.remove(k))
        return out

    def _sample(self, target: str | None, wide: bool = False) -> tuple[float, float, float]:
        """A position in the work area (``wide``: the whole table) or over the target, and a yaw: random half the
        time, else square to the table edges (within ``ALIGN_JITTER``), as people tend to set things down."""
        if target is None:
            lo, hi = self._wide() if wide else self.area
        else:
            out = self.placed[target].outline
            lo, hi = out.min(axis=0), out.max(axis=0)
        x, y = self.rng.uniform(lo, hi)
        if self.rng.random() < 0.5:
            yaw = self.rng.uniform(0.0, 2 * math.pi)
        else:
            yaw = int(self.rng.integers(4)) * math.pi / 2 + self.rng.normal(0.0, ALIGN_JITTER)
        return float(x), float(y), float(yaw)

    def _centred(self, body: Body, target: str, quarter: int) -> tuple[float, float, float]:
        """Origin x, y and yaw that centre ``body`` on ``target``'s footprint, square to it (turned ``quarter`` x 90
        deg): the pose a tight fit needs (a 200 x 100 mm board in a 203 x 114 mm bag)."""
        t = self.placed[target]
        rot = geom.yaw_matrix(-t.yaw)[:2, :2]                    # world -> the target's frame

        def mid(points: np.ndarray) -> np.ndarray:
            local = points @ rot.T
            return (local.min(axis=0) + local.max(axis=0)) / 2 @ rot

        yaw = t.yaw + quarter * math.pi / 2
        x, y = mid(t.outline) - mid(Placed("", "", body, yaw, np.zeros(3)).outline)
        return float(x), float(y), float(yaw)

    def _wide(self) -> tuple[np.ndarray, np.ndarray]:
        """Where a hard-to-fit object may also go: the whole table, or the work area grown by half when the area
        was given (a robot's reach)."""
        if self._within is None:
            return self.table_box()
        lo, hi = self.area
        tlo, thi = self.table_box()
        grow = (hi - lo) / 4
        return np.maximum(lo - grow, tlo), np.minimum(hi + grow, thi)

    def _fit(self, body: Body, card: str, key: str, target: str | None, item: int, x: float, y: float,
             yaw: float) -> Placed | None:
        p = Placed(key, card, body, yaw, np.array([x, y, 0.0]), target, item)
        outline = p.outline
        region, edge = (self.support.outline, EDGE) if target is None else (self.placed[target].outline,
                                                                            INSIDE_EDGE)
        if geom.depths(region, outline).min() < edge:
            return None
        if not self._clear(outline, target):
            return None
        z = self._rest_height(p, target)
        if z is None:
            return None
        p.pos = np.array([x, y, z])
        return p

    def _ancestors(self, target: str | None) -> list[Placed]:
        chain = [self.support]
        while target is not None:
            chain.append(self.placed[target])
            target = self.placed[target].target
        return chain

    def _clear(self, outline: np.ndarray, target: str | None) -> bool:
        if any(not geom.separated(outline, ob, CLEARANCE[self.clutter]) for ob in self.obstacles):
            return False
        skip = {p.key for p in self._ancestors(target)}
        for q in self.placed.values():
            if q.key in skip:
                continue
            gap = INSIDE_CLEARANCE if (target is not None and q.target == target) else CLEARANCE[self.clutter]
            if not geom.separated(outline, q.outline, gap):
                return False
        return True

    def _rest_height(self, p: Placed, target: str | None) -> float | None:
        """Origin z that leaves ``GAP`` under the object, or None if it would not stand there."""
        under = np.concatenate([a.triangles() for a in self._ancestors(target)])
        lo, hi = p.outline.min(axis=0), p.outline.max(axis=0)
        under = geom.cull_xy(under, lo, hi)
        xy = _samples(p, under)
        rot2 = geom.yaw_matrix(-p.yaw)[:2, :2]
        bottom = p.body.bottom.at((xy - p.pos[None, :2]) @ rot2.T)
        surface = geom.ray_down(under, xy)
        ok = np.isfinite(bottom) & (geom.depths(p.outline, xy) >= -1e-9)    # the map's cells reach past the edge
        if not ok.any() or not np.isfinite(surface[ok]).all():
            return None                                         # part of it would hang over nothing
        gaps = surface[ok] - bottom[ok]
        z = float(gaps.max())
        if np.ptp(surface[ok]) > FLAT and not _stands(p, xy[ok][gaps >= z - CONTACT], z):
            return None
        return z + GAP


def _samples(p: Placed, under: np.ndarray) -> np.ndarray:
    """Grid points over the object's outline box, its outline corners, and surface corners beneath it."""
    out = p.outline
    lo, hi = out.min(axis=0), out.max(axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], GRID), np.linspace(lo[1], hi[1], GRID), indexing="ij")
    pts = [np.stack([gx.ravel(), gy.ravel()], axis=1), out]
    if len(under):
        corners = np.unique(np.round(under.reshape(-1, 3)[:, :2], 5), axis=0)
        corners = corners[geom.depths(out, corners) >= 0]
        pts.append(corners[:: max(1, len(corners) // 400)])
    return np.concatenate(pts)


def _stands(p: Placed, contact_xy: np.ndarray, z: float) -> bool:
    """Centre of mass inside the touching patch, with margin, at origin height ``z``."""
    if len(contact_xy) < 3:
        return False
    com = p.com()
    return geom.inside_depth(geom.hull2d(contact_xy), com[:2]) >= STABLE_MARGIN


def _slots(bodies: list[Body], arrangement: str, centre: np.ndarray, size: np.ndarray, gap: float
           ) -> list[tuple[float, float, float]]:
    """(x, y, yaw) per object for a row (long sides side by side along x) or a square grid, centred."""
    n = len(bodies)
    cols = n if arrangement == "row" else max(1, math.ceil(math.sqrt(n)))
    widths = [b.size[1] + gap for b in bodies]        # yaw 90 deg: the short side runs along the row
    depth = max((b.size[0] for b in bodies), default=0.0) + gap
    rows = [list(range(i, min(i + cols, n))) for i in range(0, n, cols)]
    out: list[tuple[float, float, float]] = []
    for r, idx in enumerate(rows):
        total = sum(widths[i] for i in idx)
        x = centre[0] - total / 2
        y = centre[1] + (r - (len(rows) - 1) / 2) * depth
        for i in idx:
            out.append((x + widths[i] / 2, float(y), math.pi / 2))
            x += widths[i]
    return out


__all__ = ["CLEARANCE", "EDGE", "FILL", "GAP", "Layout", "Placed"]
