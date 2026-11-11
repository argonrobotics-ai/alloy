# SPDX-License-Identifier: Apache-2.0
"""patch-drywall start state: Gym-Anything's wallboard around the patch (``concrete_finishing/layout.py:529-562``), cell
for cell on their 1 mm grid with their seeded noise (SOFT-2's port, its parity checked against theirs on seeds 0 and
1), and block-averaged onto ours.

An 840 x 600 mm piece of 5/8 in gypsum partition (``params.py:55-62``):
- four 6 mm anchor holes through the board (base -15.9 mm);
- a 100 x 100 mm self-adhesive fibreglass mesh patch, 0.5 mm thick with its fibres 30 % of the envelope, so the base
  there is 0.15 mm while a blade rests on the fibre tops at 0.5 mm (``paste.py:128-135``);
- a mound of joint compound 1-3 mm proud with knife ridges (35 mm wavelength) and 0.3 mm noise, feathered out to 90 mm
  from the patch centre, its texture 0.4 mm, 4 min old at 21 °C.

The grid's u runs along -x of the wall, v up (+z); heights are out of the face (into the room), the face at 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy.tasks.concrete_materials import masco_factor
from alloy.tasks.concrete_pad_start import smooth_noise

PATCH = (1.80, 1.25)                     # m: the mesh patch's centre x, z (params.py:56)
HOLES = ((1.52, 1.42), (1.60, 1.42), (2.02, 1.08), (2.10, 1.08))   # m (params.py:57)
HOLE_D = 0.006                           # m (params.py:59)
BOARD_T = 0.0159                         # m: 5/8 in gypsum board (params.py:60)
MESH_T, MESH_SOLID = 0.0005, 0.30        # m, share (params.py:61-62)
MESH_HALF = 0.05                         # m: the patch's half side (layout.py:547)
BOX = (1.40, 2.24, 0.96, 1.56)           # m: the region x0, x1, z0, z1 (layout.py:533)
CELL = 0.001                             # m: their grid (layout.py:534)
AGE_MIN = 4.0                            # min at 21 °C (layout.py:211)
TEXTURE = 4.0e-4                         # m (layout.py:560)
NEAR = 0.15                              # m: "near" the holes or the patch (task.py:226-231)


@dataclass(frozen=True)
class Wallboard:
    """The board on a grid of ``cell`` [m]: masks ``holes`` and ``mesh``, the ``base`` [m] (-15.9 mm in the holes,
    0.15 mm on the mesh, 0 elsewhere), the compound ``film`` [m] over the base, texture ``a`` [m], start ``age`` [s],
    cell centres ``x``, ``z`` [m] (wall frame) and each cell's share of hole ``hole_frac`` (1 or 0 on their 1 mm
    grid). Arrays are [nu, nv]."""

    holes: np.ndarray
    mesh: np.ndarray
    base: np.ndarray
    film: np.ndarray
    a: np.ndarray
    age: float
    x: np.ndarray
    z: np.ndarray
    cell: float = CELL
    hole_frac: np.ndarray | None = None

    @property
    def frac(self) -> np.ndarray:
        """Each cell's share of hole (their grid: the mask)."""
        return self.holes.astype(float) if self.hole_frac is None else self.hole_frac

    @property
    def top(self) -> np.ndarray:
        """The surface h = base + film [m]."""
        return self.base + self.film

    def support(self) -> np.ndarray:
        """Where a blade rests: the fibre tops on the mesh, the base elsewhere (``paste.py:128-135``)."""
        return np.where(self.mesh, MESH_T, self.base)

    def near(self) -> np.ndarray:
        """Cells within ``NEAR`` of the patch centre or any hole (``task.py:226-231``)."""
        out = np.hypot(self.x - PATCH[0], self.z - PATCH[1]) < NEAR
        for hx, hz in HOLES:
            out |= np.hypot(self.x - hx, self.z - hz) < NEAR
        return out


def wallboard(seed: int = 0, temperature: float = 21.1) -> Wallboard:
    """Their region with their random stream (seed 6000 + seed) in their order, on their 1 mm grid."""
    rng = np.random.default_rng(6000 + seed)
    (cx, cz), (xa, xb, za, zb) = PATCH, BOX
    nu, nv = int(round((xb - xa) / CELL)), int(round((zb - za) / CELL))
    u = (np.arange(nu) + 0.5) * CELL
    v = (np.arange(nv) + 0.5) * CELL
    x = xb - u[:, None] + np.zeros((1, nv))
    z = za + np.zeros((nu, 1)) + v[None, :]
    holes = np.zeros(x.shape, bool)
    for hx, hz in HOLES:
        holes |= (x - hx) ** 2 + (z - hz) ** 2 < (HOLE_D / 2) ** 2
    base = np.where(holes, -BOARD_T, 0.0)
    mesh = (np.abs(x - cx) < MESH_HALF) & (np.abs(z - cz) < MESH_HALF)
    base = np.where(mesh, MESH_T * MESH_SOLID, base)
    r = np.hypot((x - cx) / 0.09, (z - cz) / 0.09)
    mound = np.clip(1 - r, 0, 1) ** 0.7 * 0.0022 + 0.0006 * np.sin(2 * math.pi * (x - cx) / 0.035) * (r < 0.9)
    mound += 0.0003 * smooth_noise(rng, x.shape, 2)
    film = np.where(r < 1.0, np.maximum(mound - np.maximum(base, 0.0), 0.0), 0.0)
    return Wallboard(holes=holes, mesh=mesh, base=base, film=film, a=np.where(film > 0, TEXTURE, 0.0),
                     age=AGE_MIN * 60.0 * masco_factor(temperature), x=x, z=z)


def _block(f: np.ndarray, k: int) -> np.ndarray:
    nu, nv = f.shape[0] // k, f.shape[1] // k
    return f[: nu * k, : nv * k].reshape(nu, k, nv, k).mean(axis=(1, 3))


def on_grid(wb: Wallboard, cell: float) -> Wallboard:
    """``wb`` (their 1 mm grid) block-averaged onto ``cell``: base and film keep their volume;
    a cell is on the mesh when at least half of it was (its edges are cell edges at 1, 2, 5 and 10 mm), and a hole
    when any of it was (a 6 mm hole spreads over up to four 5 mm cells, none half a hole)."""
    k = int(round(cell / CELL))
    if abs(k * CELL - cell) > 1e-12 or k not in (1, 2, 5, 10):
        raise ValueError(f"cell must be 1, 2, 5 or 10 mm (the mesh's edges are then cell edges), got {cell}")
    if k == 1:
        return wb
    frac = _block(wb.holes.astype(float), k)
    return Wallboard(holes=frac > 0.0, mesh=_block(wb.mesh.astype(float), k) >= 0.5, base=_block(wb.base, k),
                     film=_block(wb.film, k), a=_block(wb.a, k), age=wb.age, x=_block(wb.x, k), z=_block(wb.z, k),
                     cell=k * CELL, hole_frac=frac)


__all__ = ["BOARD_T", "BOX", "CELL", "HOLES", "HOLE_D", "MESH_HALF", "MESH_SOLID", "MESH_T", "NEAR", "PATCH",
           "Wallboard", "on_grid", "wallboard"]
