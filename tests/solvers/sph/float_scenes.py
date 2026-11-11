# SPDX-License-Identifier: Apache-2.0
"""Scenes for WP9 (a1) floating containers, through the public API (alloy.Scene -> MuJoCo + SPH, the proxy path):
a thin-walled open box (``add_mesh``, collision "sdf", flagged a container) released just above still water in a
static tank. Lengths in m, masses in kg; ``scale`` multiplies every length (Froude: masses by scale³)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import alloy

RHO = 998.2                                   # kg/m³, the library's water
FLOOR = 0.008                                 # m: the tank's floor (wall) thickness


def open_box_mesh(L: float, H: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """A watertight thin-walled open box: outer L x L x H, walls and floor t, centred on x-y, bottom at z = 0 (faces
    counter-clockwise seen from outside)."""
    a, b = L / 2, L / 2 - t
    o = [(-a, -a, 0), (a, -a, 0), (a, a, 0), (-a, a, 0), (-a, -a, H), (a, -a, H), (a, a, H), (-a, a, H)]
    i = [(-b, -b, t), (b, -b, t), (b, b, t), (-b, b, t), (-b, -b, H), (b, -b, H), (b, b, H), (-b, b, H)]
    f = [(0, 2, 1), (0, 3, 2), (8, 9, 10), (8, 10, 11)]              # outer bottom, cavity floor
    for k in range(4):
        k1 = (k + 1) % 4
        f += [(k, k1, 4 + k1), (k, 4 + k1, 4 + k),                   # outer side
              (4 + k, 4 + k1, 12 + k1), (4 + k, 12 + k1, 12 + k),    # rim
              (8 + k, 12 + k1, 8 + k1), (8 + k, 12 + k, 12 + k1)]    # inner side
    return np.array(o + i, float), np.array(f, np.int64)


@dataclass(frozen=True)
class Float:
    """A floating open box in a square tank: ``ratio`` = D5's m_a/m_s = ρ·L²·H/m."""
    ratio: float
    wall: float = 0.004
    L: float = 0.06
    H: float = 0.05
    tank: float = 0.14
    depth: float = 0.04
    drop: float = 0.001
    scale: float = 1.0

    @property
    def mass(self) -> float:
        return RHO * self.L * self.L * self.H * self.scale ** 3 / self.ratio

    @property
    def surface(self) -> float:
        return (FLOOR + self.depth) * self.scale

    def bottom_expected(self, mass: float | np.ndarray | None = None) -> float | np.ndarray:
        """The floating cup's bottom relative to the still surface: the tank's rise minus the draft."""
        m = self.mass if mass is None else mass
        L, T = self.L * self.scale, self.tank * self.scale
        return m / RHO / (T * T) - m / (RHO * L * L)

    @property
    def solid_volume(self) -> float:
        L, H, t = (x * self.scale for x in (self.L, self.H, self.wall))
        return L * L * t + (L * L - (L - 2 * t) ** 2) * (H - t)

    def scene(self, worlds: int = 1, inner: float = 0.0, rpy=(0.0, 0.0, 0.0), container: bool = True,
              ratios: tuple[float, float] | None = None) -> alloy.Scene:
        """The scene; ``inner`` [m] of water at rest in the cup (slosh), ``rpy`` the cup's start attitude (raised so
        its lowest edge starts ``drop`` over the water),
        ``ratios``: D5 ratios drawn per world (the cup's density randomized; the build checks the card's)."""
        s = self.scale
        L, H, t, T, D = self.L * s, self.H * s, self.wall * s, self.tank * s, self.depth * s
        sc = alloy.Scene(dt=0.01, worlds=worlds)
        sc.add_ground()
        sc.add_open_box("tank", inner=(T, T, 0.09 * s), wall=FLOOR * s)
        sc.add_liquid("water", size=(T, T, D), pos=(0.0, 0.0, FLOOR * s + D / 2))
        v, f = open_box_mesh(L, H, t)
        wall_q = {"value": t, "source": "model", "note": "the open box's wall and floor"}
        extra = {"features": {"passage_width": {"value": L - 2 * t, "source": "model", "note": "cavity width"},
                              "min_wall": wall_q}}
        z0 = FLOOR * s + D + self.drop * s + 0.5 * L * abs(np.sin(rpy[0])) + 0.5 * L * abs(np.sin(rpy[1]))
        sc.add_mesh("cup", vertices=v, faces=f, collision="sdf", density=self.mass / self.solid_volume,
                    pos=(0.0, 0.0, z0), rpy=rpy, material="abs",
                    card_extra=extra if container else {"features": {"min_wall": wall_q}})
        if ratios is not None:                        # density per world: m = ρ·L²·H·s³/ratio
            dens = [RHO * L * L * H / r / self.solid_volume for r in ratios]
            sc.randomize("cup", "density", min(dens), max(dens))
        if inner > 0.0:                               # a level block clear of a cup tilted up to ~0.1 rad
            cav = L - 2 * t - 12.0e-3 * s
            sc.add_liquid("inner", size=(cav, cav, inner), pos=(0.0, 0.0, z0 + t + 3.0e-3 * s + inner / 2))
        return sc
