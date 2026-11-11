# SPDX-License-Identifier: Apache-2.0
"""tool-joints start state: Gym-Anything's block infill (``concrete_finishing/layout.py:98-122, 420-484``), cell for
cell on their 1.5 mm grid with their seeded noise, and the wall cropped to a window for light runs.

The infill is three 8x8x16 in CMU a course, four courses in running bond (half blocks at the ends of courses 2 and 4),
3/8 in joints recessed 20 mm behind the block faces and struck flush (+-0.4 mm correlated noise). One head joint in
course 1 is 3 mm short over 60 mm (a void); crumbs of squeeze-out (1.5 mm thick, 3x3 cells) sit on ~0.4 % of the face
cells next to the joints; the joint texture is 0.5 mm. Courses were laid 45, 37, 28 and 20 min before the start (at
21 C, scaled by the MASCO factor). The grid's u runs along +x of the wall and v up (+z); heights are out of the face
(into the room) with the block faces at 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks.concrete_materials import masco_factor

BLOCK = (0.397, 0.194, 0.194)            # m: length, thickness, height (params.py:44)
JOINT_W = 0.0095                         # m: 3/8 in (params.py:45)
INFILL_X = (3.89, 5.11)                  # m (params.py:46)
INFILL_Z0 = 0.813                        # m: sill (params.py:47)
COURSES = 4                              # (params.py:48)
BED_DEPTH = 0.020                        # m: joint depth behind the face (params.py:49)
AGES_MIN = (45.0, 37.0, 28.0, 20.0)      # course ages at the start, min at 21 C (layout.py:479)
CELL = 0.0015                            # m: their grid (layout.py:428)
MARGIN = 0.03                            # m: their region beyond the infill (layout.py:429)
FLUSH_NOISE = 0.0004                     # m (layout.py:461)
VOID_SHORT = 0.003                       # m: the short head joint (layout.py:468)
CRUMB_T, CRUMB_P = 0.0015, 0.004         # m, share (layout.py:471-477)
TEXTURE = 5.0e-4                         # m (layout.py:478)


def _noise(rng: np.random.Generator, shape: tuple[int, int], cells: int) -> np.ndarray:
    """Their smoothed unit noise (layout.py:214-221): (cells)^2 passes of the 3x3 binomial kernel."""
    k = rng.normal(0, 1, shape)
    for _ in range(max(1, int(cells)) ** 2):
        k = 0.5 * k + 0.125 * (np.roll(k, 1, 0) + np.roll(k, -1, 0) + np.roll(k, 1, 1) + np.roll(k, -1, 1))
    return k / max(float(k.std()), 1e-9)


@dataclass(frozen=True)
class Infill:
    """The wall on their grid: ``joint`` and ``course`` (-1 off the joints) masks, the joint ``base`` [m] (-20 mm in
    the joints, 0 on the faces), the paste ``top`` [m] (struck flush in the joints, crumbs on the faces, 0 elsewhere),
    the texture ``a`` [m] and the start ``age`` [s] per cell; ``origin`` (x, z) of cell (0, 0)'s corner [m]."""

    joint: np.ndarray
    course: np.ndarray
    base: np.ndarray
    top: np.ndarray
    a: np.ndarray
    age: np.ndarray
    void: np.ndarray
    crumbs: np.ndarray
    origin: tuple[float, float]
    cell: float = CELL


def _joints(X: np.ndarray, Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Joint and course masks (layout.py:438-460): bed joints, head joints per course (running bond), the top bed."""
    x0, x1 = INFILL_X
    j, (bl, _, bh), z0 = JOINT_W, BLOCK, INFILL_Z0
    z1 = z0 + COURSES * (bh + j) + j
    joint = np.zeros(X.shape, bool)
    course = np.full(X.shape, -1, int)
    inside = (X > x0) & (X < x1) & (Z > z0) & (Z < z1)
    for c in range(COURSES):
        zb = z0 + c * (bh + j)
        bed = inside & (Z >= zb) & (Z < zb + j)
        joint |= bed
        course = np.where(bed, c, course)
        row = inside & (Z >= zb + j) & (Z < zb + j + bh)
        heads = [x0, x1 - j]
        if c % 2 == 0:
            heads += [x0 + j + bl + k * (bl + j) for k in range(2)]
        else:
            half = (bl - j) / 2
            heads += [x0 + j + half, x0 + j + half + j + bl, x0 + j + half + j + 2 * bl + j]
        for hx in heads:
            hj = row & (X >= hx) & (X < hx + j)
            joint |= hj
            course = np.where(hj, c, course)
    top_bed = inside & (Z >= z1 - j)
    joint |= top_bed
    course = np.where(top_bed, COURSES - 1, course)
    return joint, course


def infill(seed: int = 0, temperature: float = 21.1) -> Infill:
    """Their infill region (layout.py:420-484) with their random stream (seed 4000 + seed) in their order."""
    rng = np.random.default_rng(4000 + seed)
    x0, x1 = INFILL_X
    j, (bl, _, bh), z0 = JOINT_W, BLOCK, INFILL_Z0
    z1 = z0 + COURSES * (bh + j) + j
    ox, oz = x0 - MARGIN, z0 - MARGIN
    nu, nv = int(round((x1 - x0 + 2 * MARGIN) / CELL)), int(round((z1 - z0 + 2 * MARGIN) / CELL))
    X = ox + ((np.arange(nu) + 0.5) * CELL)[:, None] + np.zeros((1, nv))
    Z = oz + np.zeros((nu, 1)) + ((np.arange(nv) + 0.5) * CELL)[None, :]
    joint, course = _joints(X, Z)
    base = np.where(joint, -BED_DEPTH, 0.0)
    tb = np.where(joint, BED_DEPTH + FLUSH_NOISE * _noise(rng, joint.shape, 2), 0.0)
    xv = x0 + j + (bl - j) / 2 + j + bl + j / 2                     # the short head joint's centre (layout.py:465)
    zv = z0 + (bh + j) + j
    void = joint & (course == 1) & (np.abs(X - xv) < j) & (Z > zv + 0.06) & (Z < zv + 0.12)
    tb = np.where(void, BED_DEPTH - VOID_SHORT, tb)
    inside = (X > x0) & (X < x1) & (Z > z0) & (Z < z1)
    crumbs = (~joint) & inside & (rng.random(joint.shape) < CRUMB_P)
    crumbs = np.maximum.reduce([np.roll(np.roll(crumbs, a, 0), b, 1) for a in (-1, 0, 1) for b in (-1, 0, 1)])
    near = np.maximum.reduce([np.roll(np.roll(joint, a, 0), b, 1) for a in range(-6, 7, 3) for b in range(-6, 7, 3)])
    crumbs &= near & ~joint
    tb = np.where(crumbs, CRUMB_T, tb)
    ages = np.array(AGES_MIN) * 60.0 * masco_factor(temperature)
    age = np.where(course >= 0, ages[np.clip(course, 0, COURSES - 1)], ages[0])
    return Infill(joint=joint, course=course, base=base, top=base + tb, a=np.where(joint, TEXTURE, 0.0), age=age,
                  void=void, crumbs=crumbs, origin=(ox, oz))


def crop(f: Infill, x: tuple[float, float], z: tuple[float, float]) -> Infill:
    """The cells whose centres lie in the window x0 <= x < x1, z0 <= z < z1 [m] (a light run's piece of the wall)."""
    i0, i1 = (int(round((v - f.origin[0]) / f.cell)) for v in x)
    k0, k1 = (int(round((v - f.origin[1]) / f.cell)) for v in z)
    i0, k0 = max(i0, 0), max(k0, 0)
    s = (slice(i0, i1), slice(k0, k1))
    return Infill(joint=f.joint[s], course=f.course[s], base=f.base[s], top=f.top[s], a=f.a[s], age=f.age[s],
                  void=f.void[s], crumbs=f.crumbs[s], origin=(f.origin[0] + i0 * f.cell, f.origin[1] + k0 * f.cell),
                  cell=f.cell)


def blocks(old_wall: float = MARGIN) -> list[tuple[float, float, float, float]]:
    """The rigid faces as wall rectangles (x0, x1, z0, z1) [m]: every block (layout.py:98-118) and the old wall
    around the opening (a frame ``old_wall`` wide beyond the infill: their region's margin by default). One change from theirs: three blocks
    and four joints are 1.229 m in their 1.22 m opening, so the third block of the first and third courses covers 9 of
    the last head joint's 9.5 mm (their rod could not reach mortar their verdict scores); here it ends at the joint."""
    (x0, x1), j, (bl, _, bh), z0 = INFILL_X, JOINT_W, BLOCK, INFILL_Z0
    z1 = z0 + COURSES * (bh + j) + j
    out = []
    for c in range(COURSES):
        zc = z0 + j + c * (bh + j)
        if c % 2 == 0:
            xs = [(x0 + j + k * (bl + j), min(bl, x1 - j - (x0 + j + k * (bl + j)))) for k in range(3)]
        else:
            half = (bl - j) / 2
            xa = x0 + j + half + j + 2 * (bl + j)
            xs = [(x0 + j, half)] + [(x0 + j + half + j + k * (bl + j), bl) for k in range(2)] + [(xa, x1 - j - xa)]
        out += [(xa, xa + w, zc, zc + bh) for xa, w in xs]
    m = old_wall
    out += [(x0 - m, x0, z0 - m, z1 + m), (x1, x1 + m, z0 - m, z1 + m), (x0, x1, z0 - m, z0), (x0, x1, z1, z1 + m)]
    return out
