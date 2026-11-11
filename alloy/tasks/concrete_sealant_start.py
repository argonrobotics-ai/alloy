# SPDX-License-Identifier: Apache-2.0
"""tool-sealant start state: Gym-Anything's control joint (``concrete_finishing/layout.py:486-527``), cell for cell on
their 1 mm grid with their seeded noise.

A 13 mm control joint in a wall, its backer rod face 8.5 mm behind the wall faces, filled with a convex sealant bead
1.5-2.7 mm proud, two gun stops (3 mm bumps 12 mm wide), 0.3 mm noise and a 40 mm thin stretch on the joint's north
side where the bead sagged off that wall (4 mm below the face). Wall contact at the start: the cells within 1.5 mm
of each wall touch it except a random quarter (and the thin stretch on the north wall). The task segment runs
z 0.45-2.00 m between two strips of tape; the region is 60 mm across and 0.10 m longer. The grid's u runs along -y of
the wall (across the joint), v up (+z); heights are out of the face (into the room) with the faces at 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.tasks.concrete_joints_start import _noise
from alloy.tasks.concrete_materials import masco_factor

CJ_Y = 2.00                              # m: the joint's centre (params.py:50)
CJ_W = 0.013                             # m: 1/2 in (params.py:51)
CJ_BACKER = 0.0085                       # m: backer rod face behind the wall face (params.py:52)
SEGMENT = (0.45, 2.00)                   # m: between the tape marks (params.py:53)
CELL = 0.001                             # m (layout.py:491)
HALF_U = 0.030                           # m: the region's half width across the joint (layout.py:492)
AGE_MIN = 10.0                           # min at 21 C at the start (layout.py:211)
TEXTURE = 3.0e-4                         # m (layout.py:523)
SKIN_H, SKIN_FAST_H = 2.0, 20.0 / 60.0   # h at 21 C: Tremco Dymonic 100, the fast-skinning grade (params.py:119;
                                         # variants.py:33)


@dataclass(frozen=True)
class Joint:
    """The control joint on their grid: masks ``joint``, ``edge_l`` / ``edge_r`` (within 1.5 mm of each wall),
    ``thin``, the base [m] (-8.5 mm in the joint), the sealant ``top`` [m], wall ``contact_l`` / ``contact_r`` at the
    start, texture ``a`` [m], start ``age`` [s], cell centres ``y`` (across, world) and ``z`` [m]."""

    joint: np.ndarray
    edge_l: np.ndarray
    edge_r: np.ndarray
    thin: np.ndarray
    base: np.ndarray
    top: np.ndarray
    contact_l: np.ndarray
    contact_r: np.ndarray
    a: np.ndarray
    age: float
    y: np.ndarray
    z: np.ndarray
    cell: float = CELL


def joint_region(seed: int = 0, temperature: float = 21.1) -> Joint:
    """Their region with their random stream (seed 5000 + seed) in their order."""
    rng = np.random.default_rng(5000 + seed)
    za, zb = SEGMENT
    w = CJ_W
    nu, nv = int(round(2 * HALF_U / CELL)), int(round((zb - za + 0.10) / CELL))
    u = (np.arange(nu) + 0.5) * CELL
    v = (np.arange(nv) + 0.5) * CELL
    yw = CJ_Y + HALF_U - u[:, None] + np.zeros((1, nv))
    z = (za - 0.05) + np.zeros((nu, 1)) + v[None, :]
    joint = np.abs(yw - CJ_Y) < w / 2
    base = np.where(joint, -CJ_BACKER, 0.0)
    across = (yw - CJ_Y) / (w / 2)
    bead = 0.0015 + 0.0012 * (1 - across**2)
    stops = np.zeros(joint.shape)
    for zc in (za + 0.35 + 0.1 * rng.random(), za + 1.05 + 0.1 * rng.random()):
        stops += 0.003 * np.exp(-(((z - zc) / 0.012) ** 2))
    top = bead + stops + 0.0003 * _noise(rng, joint.shape, 2)
    thin = joint & (z > za + 0.70) & (z < za + 0.74) & (yw > CJ_Y)
    top = np.where(thin, -0.004, top)
    tb = np.where(joint, np.maximum(top - base, 0.0), 0.0)
    edge_l = joint & (np.abs(yw - (CJ_Y - w / 2)) < 0.0015)
    edge_r = joint & (np.abs(yw - (CJ_Y + w / 2)) < 0.0015)
    gaps = rng.random(joint.shape) < 0.25
    return Joint(joint=joint, edge_l=edge_l, edge_r=edge_r, thin=thin, base=base, top=base + tb,
                 contact_l=np.where(edge_l & ~gaps, 1.0, 0.0), contact_r=np.where(edge_r & ~gaps & ~thin, 1.0, 0.0),
                 a=np.where(joint, TEXTURE, 0.0), age=AGE_MIN * 60.0 * masco_factor(temperature), y=yw, z=z)


def skin_s(temperature: float = 21.1, fast: bool = False) -> float:
    """The sealant's skin age [s]: 2 h (20 min for the fast grade) at 21 C, scaled by the MASCO factor."""
    return (SKIN_FAST_H if fast else SKIN_H) * 3600.0 * masco_factor(temperature)
