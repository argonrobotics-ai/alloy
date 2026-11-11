# SPDX-License-Identifier: Apache-2.0
"""Vessels of the rinse_* tasks: bodies of revolution with a cavity (a pan, a glass, a can), as Gym-Anything builds them
(``surface_rinsing/ware.py``): an inner wall r(z) from the inner floor to the rim, a wall and a floor of given
thicknesses, one water chart wrapping the solid, and a collider of boxes.

Frames: a vessel's body origin is the centre of its inner floor (the floor's solid lies below, z in (-bottom, 0)); its
axis is z. The chart's profile wraps the solid, material on its right: inner floor (from ``r_min``), inner wall, rim,
outer wall, outer bottom (to ``r_min``). A revolution chart has no pole cell: the discs inside ``r_min`` (the narrowest
sector half a cell wide) carry no film. The inner floor's edge is a wall (the vessel holds water); the outer
bottom's is open (it drips). The chart has no ``shape``: the whole body is exempt from occluding jets on it (the flat
collider boxes stand proud of a curved outer face and would block jets aimed at it).

Colliders: the floor as radial boxes (a cylinder on a box gets one MuJoCo contact: a pan's floor sank 2.3 mm into a
bowl's floor) and a ring of boxes chord-filling the wall, as their slab colliders (``ware.py:180-192, 323-341``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.surface_flow.charts import RevolutionChart


def xform(pos, quat=(0.0, 0.0, 0.0, 1.0)) -> wp.transform:
    return wp.transform(wp.vec3(*(float(v) for v in pos)), wp.quat(*(float(v) for v in quat)))


def quat_zy(yaw: float, tilt: float) -> tuple[float, float, float, float]:
    """(x, y, z, w) of a turn ``yaw`` about z after a tilt ``tilt`` about y (a wall box leaning out at azimuth 0)."""
    x1, y1, z1, w1 = 0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)
    x2, y2, z2, w2 = 0.0, math.sin(0.5 * tilt), 0.0, math.cos(0.5 * tilt)
    return (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)


@dataclass(frozen=True)
class Vessel:
    """A body of revolution with a cavity. Subclasses give the inner wall ``radius(z)``; sizes [m], mass [kg]."""

    depth: float = 0.05               # inner floor to rim
    wall: float = 0.003
    bottom: float = 0.003             # floor thickness
    mass: float = 1.0
    contact_angle: float = 75.0
    ds: float = 3.0e-3
    n_theta: int = 64
    ring: int = 24                    # wall boxes of the collider (per band)
    mu: float = 0.30
    bands: int = 1                    # collider bands up the wall: chords between ``bands + 1`` profile heights
    shell: float | None = None        # the collider's wall thickness (None: ``wall``), from the inner face outward

    def radius(self, z: np.ndarray | float) -> np.ndarray | float:
        raise NotImplementedError

    @property
    def r_min(self) -> float:
        """The chart's inner radius: the narrowest sector is half a cell wide."""
        return 1.02 * self.ds * self.n_theta / (4.0 * math.pi)

    def cavity(self, n: int = 11) -> tuple[tuple[float, float], ...]:
        """The inside as (r, z) nodes from the floor's edge up to the rim (a cavity pool's profile)."""
        return tuple((float(self.radius(h)), float(h)) for h in np.linspace(0.0, self.depth, n))

    def outer(self, n: int = 11) -> tuple[tuple[float, float], ...]:
        """The outside as (r, z) nodes, z rising from the underside to the rim (the hull envelope for DC5's
        ``Float(hull=...)``, origin at the body origin; the flat underside is implied)."""
        z = np.linspace(0.0, self.depth, n)
        return ((float(self.radius(0.0)) + self.wall, -self.bottom),
                *((float(self.radius(h)) + self.wall, float(h)) for h in z))

    def profile(self) -> tuple[tuple[float, float], ...]:
        """The chart: inner floor, inner wall, rim, outer wall, outer bottom; material on the right."""
        t, inner = self.wall, self.cavity()
        outer = [(r + t, z) for r, z in reversed(inner)]
        r0 = float(self.radius(0.0))
        return ((self.r_min, 0.0), *inner, *outer, (r0 + t, -self.bottom), (self.r_min, -self.bottom))

    def regions(self) -> dict[str, tuple[float, float]]:
        """Arc-length spans of the regions along the profile [m]: inner, rim, outer, foot."""
        p = np.asarray(self.profile())
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
        n = len(self.cavity())
        return {"inner": (0.0, s[n]), "rim": (s[n], s[n + 1]), "outer": (s[n + 1], s[2 * n]),
                "foot": (s[2 * n], s[-1])}

    def chart(self, body: str) -> RevolutionChart:
        return RevolutionChart(profile=self.profile(), n_theta=self.n_theta, ds=self.ds, body=body,
                               ends=("wall", "open"), contact_angle=self.contact_angle)

    def cavity_volume(self) -> float:
        z = np.linspace(0.0, self.depth, 2001)
        r = np.asarray(self.radius(z))
        return float(np.trapezoid(math.pi * r * r, z))

    def extra_volume(self) -> float:
        """Solid volume of parts other than the floor and wall (a handle), at ``extra_density``."""
        return 0.0

    def build_extra(self, b: newton.ModelBuilder, body: int, label: str, density: float) -> None:
        """Parts other than the floor and wall (a handle)."""

    def extra_mass(self) -> float:
        return 0.0

    def wall_bands(self) -> list[tuple[float, float, float, float, float]]:
        """The collider's wall bands: (centre radius, centre z, slant, tilt, box width) per band; boxes ``shell`` thick
        with their inner face on the chord between the band's ends."""
        t, sh = self.wall, self.shell or self.wall
        z = np.linspace(0.0, self.depth, self.bands + 1)
        r = [float(self.radius(h)) for h in z]
        out = []
        for za, zb, ra, rb in zip(z[:-1], z[1:], r[:-1], r[1:], strict=True):
            width = 2.0 * math.pi * 0.5 * (ra + rb + 2.0 * t) / self.ring * 1.05
            out.append((0.5 * (ra + rb) + 0.5 * sh, 0.5 * (za + zb), math.hypot(rb - ra, zb - za),
                        math.atan2(rb - ra, zb - za), width))
        return out

    def build(self, b: newton.ModelBuilder, label: str, pose: wp.transform) -> int:
        """A free body: ``ring // 2`` radial floor boxes, ``ring`` wall boxes per band, and the extras; total
        ``mass``."""
        body = b.add_body(xform=pose, label=label)
        n, nf, sh = self.ring, self.ring // 2, self.shell or self.wall
        r0 = float(self.radius(0.0))
        bands = self.wall_bands()
        foot = (r0 + self.wall, 2.0 * math.pi * (r0 + self.wall) / nf * 1.05)
        solid = nf * foot[0] * foot[1] * self.bottom + sum(n * slant * w * sh for _, _, slant, _, w in bands)
        cfg = newton.ModelBuilder.ShapeConfig(density=(self.mass - self.extra_mass()) / solid, mu=self.mu)
        for k in range(nf):
            a = 2.0 * math.pi * (k + 0.5) / nf
            c = (0.5 * foot[0] * math.cos(a), 0.5 * foot[0] * math.sin(a), -0.5 * self.bottom)
            b.add_shape_box(body, xform=xform(c, (0.0, 0.0, math.sin(0.5 * a), math.cos(0.5 * a))), hx=0.5 * foot[0],
                            hy=0.5 * foot[1], hz=0.5 * self.bottom, cfg=cfg, label=f"{label}_floor{k}")
        for j, (rc, zc, slant, tilt, width) in enumerate(bands):
            for k in range(n):
                a = 2.0 * math.pi * k / n
                b.add_shape_box(body, xform=xform((rc * math.cos(a), rc * math.sin(a), zc), quat_zy(a, tilt)),
                                hx=0.5 * sh, hy=0.5 * width, hz=0.5 * slant, cfg=cfg,
                                label=f"{label}_wall{k}" if self.bands == 1 else f"{label}_wall{j}_{k}")
        if self.extra_volume() > 0.0:
            self.build_extra(b, body, label, self.extra_mass() / self.extra_volume())
        return body


@dataclass(frozen=True)
class Pan(Vessel):
    """Wear-Ever 10 in aluminium fry pan (``params.py:193``, ``ware.py:416-437``). Their profile is the outer wall: from
    ``r_bot`` at the foot to ``R`` at the rim over ``height``, as s^1.5; the inner wall is it offset inward by ``wall``
    along its normal, over a ``bottom`` 4 mm thick (``_cup_polys``, ``ware.py:173-177``). The origin is the inner
    floor's centre, ``depth`` under the rim: inner radius 97.5 mm at the floor, 123.2 mm at the rim (1.74 L). The
    riveted 200 mm handle (along body +x) carries ``handle_share`` of the mass."""

    R: float = 0.127
    r_bot: float = 0.100
    height: float = 0.051
    bottom: float = 0.004
    depth: float = 0.047
    mass: float = 1.05
    handle_share: float = 0.20
    handle: tuple[float, float, float] = (0.200, 0.024, 0.010)     # length, width, thickness

    def radius(self, z: np.ndarray | float) -> np.ndarray | float:
        s = (np.clip(np.asarray(z, np.float64), 0.0, self.depth) + self.bottom) / self.height
        slope = 1.5 * (self.R - self.r_bot) / self.height * np.sqrt(s)            # dr/dz of the outer wall
        return self.r_bot + (self.R - self.r_bot) * s ** 1.5 - self.wall * np.sqrt(1.0 + slope * slope)

    @property
    def rim(self) -> float:
        """The inner radius at the rim [m]."""
        return float(self.radius(self.depth))

    def extra_mass(self) -> float:
        return self.handle_share * self.mass

    def extra_volume(self) -> float:
        return float(np.prod(self.handle))

    def build_extra(self, b: newton.ModelBuilder, body: int, label: str, density: float) -> None:
        hl, hw, ht = self.handle
        b.add_shape_box(body, xform=xform((self.rim + self.wall + 0.5 * hl, 0.0, self.depth - 0.5 * ht)),
                        hx=0.5 * hl, hy=0.5 * hw, hz=0.5 * ht,
                        cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=self.mu), label=f"{label}_handle")


@dataclass(frozen=True)
class Glass(Vessel):
    """A bar glass (``params.py:185-186``, ``ware.py:180-192``): the outer wall tapers linearly from ``r_bot`` at the
    foot to ``r_top`` at the rim over ``height``; inner = outer - ``wall``; glass, contact angle 20°, 2 mm cells."""

    height: float = 0.095
    r_top: float = 0.0445
    r_bot: float = 0.035
    wall: float = 0.003
    bottom: float = 0.012
    depth: float = 0.083
    mass: float = 0.32
    contact_angle: float = 20.0
    ds: float = 2.0e-3
    n_theta: int = 32
    ring: int = 20
    mu: float = 0.35

    def radius(self, z: np.ndarray | float) -> np.ndarray | float:
        zz = np.clip(np.asarray(z, np.float64), 0.0, self.depth) + self.bottom
        return self.r_bot + (self.r_top - self.r_bot) * zz / self.height - self.wall


def rocks_glass() -> Glass:
    return Glass()


def mixing_glass() -> Glass:
    return Glass(height=0.150, r_bot=0.032, wall=0.0035, depth=0.150 - 0.012, mass=0.42)


__all__ = ["Glass", "Pan", "Vessel", "mixing_glass", "quat_zy", "rocks_glass", "xform"]
