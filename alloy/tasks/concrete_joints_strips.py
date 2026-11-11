# SPDX-License-Identifier: Apache-2.0
"""tool-joints: where the margin trowel takes the tags, beside the joints and never over them.

The jointer's roll runs out beside its leading end over the arrises onto the block faces (Gym-Anything's tags,
``coupling.py:733-738``). A margin trowel cuts them off, and a loaded blade fills whatever lies below it from its carry
(``coupling.py:556-596``), so a blade over a tooled joint refills the groove. The verdict scores only a joint's centre
cells (``concrete_metrics._measure_joints``: distance from the joint's edge >= 0.75 x 4.75 mm), so the blade may
overlap a joint's outer cells but never touch a scored one. Each lane gets a strip on each side, the blade's inner edge
``TROWEL_IN`` over the joint, and one along its centre line where it runs past the joint's ends (the old wall beyond a
bed; the block face beyond the bed a head joint ends at); every strip stops ``CLEAR`` short of where the blade, soft
edges included, would touch a scored cell anywhere on the wall (the joints crossing its band, a T's inner corners),
and runs on ``overrun`` past its lane's ends otherwise. The centre-line strip past a crossing joint is the one that
takes a head joint's roll: the rod's leading end lays it up to 95 mm past the lane, over the bed and 85 mm onto the
next course's face, where the bed's own side strips reach 54 mm (the whole wall on rc3 kept 8.4-8.9 cm2/m of smear
there, limit 2.0; windows of the wall cropped those faces). Wall coordinates (x along the wall, z up) [m]; lanes are
axis-aligned.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import distance_transform_edt

from alloy.tasks import concrete_joints_start as JS

TROWEL_IN = 0.0015                       # m: the blade's inner edge this far over the joint (its two outer cells)
CENTRE_DIST = 0.75 * JS.JOINT_W / 2      # m: the verdict's centre cells (concrete_metrics, T:631)
CLEAR = 2.5e-4                           # m: kept between the blade's soft edge and a scored cell (the carriage's error)


class LaneLike(Protocol):
    p0: tuple[float, float]
    p1: tuple[float, float]


@dataclass(frozen=True)
class Strip:
    """One margin-trowel strip: its centre's working start ``a`` and end ``b`` (wall x, z) [m] and its heading ``yaw``
    (the blade's long side along the lane)."""

    a: tuple[float, float]
    b: tuple[float, float]
    yaw: float


@dataclass(frozen=True)
class Scored:
    """The scored joint cells' centres (wall x, z) [m] and the grid's cell [m]."""

    x: np.ndarray
    z: np.ndarray
    cell: float


def scored(f: JS.Infill) -> Scored:
    """The cells the verdict scores on ``f``'s grid: joint cells at least CENTRE_DIST from the joint's edge."""
    m = f.joint & (distance_transform_edt(f.joint) * f.cell >= CENTRE_DIST)
    i, k = np.nonzero(m)
    return Scored(x=f.origin[0] + (i + 0.5) * f.cell, z=f.origin[1] + (k + 0.5) * f.cell, cell=f.cell)


def _free(rng: tuple[float, float], blocked: np.ndarray, reach: float) -> list[tuple[float, float]]:
    """``rng`` minus the open ranges (q - reach, q + reach) for each q in ``blocked``: the closed ranges left."""
    out, lo = [], rng[0]
    for q in np.sort(blocked):
        a, b = q - reach, q + reach
        if b <= lo or a >= rng[1]:
            continue
        if a > lo:
            out.append((lo, a))
        lo = max(lo, b)
    if lo < rng[1]:
        out.append((lo, rng[1]))
    return out


def strips(ln: LaneLike, sc: Scored, half: tuple[float, float], overrun: float,
           window: tuple[float, float, float, float] | None = None, inset: float = TROWEL_IN) -> list[Strip]:
    """The strips along lane ``ln`` for a blade of half sizes ``half`` (along, across) [m], kept off every scored
    cell in ``sc`` (the whole wall's), within ``window`` (x0, x1, z0, z1) if given."""
    along = 1 if ln.p0[0] == ln.p1[0] else 0
    across, (hl, hw), h = 1 - along, half, sc.cell / 2
    q_al, q_ac = (sc.z, sc.x) if along == 1 else (sc.x, sc.z)
    lo, hi = sorted((ln.p0[along], ln.p1[along]))
    yaw = 0.5 * np.pi if along == 1 else 0.0
    off = JS.JOINT_W / 2 - inset + hw
    out = []
    for c in (ln.p0[across] - off, ln.p0[across] + off, ln.p0[across]):
        if window is not None and (c + hw <= window[2 * across] or c - hw >= window[2 * across + 1]):
            continue
        near = np.abs(q_ac - c) < hw + h + CLEAR            # scored cells the band touches across
        for s0, s1 in _free((lo - overrun, hi + overrun), q_al[near], hl + h + CLEAR):
            if c != ln.p0[across] and (s1 + hl <= lo or s0 - hl >= hi):
                continue                                    # beside another joint, not this one. The centre line's
                                                            # stays: past a crossing joint it takes this lane's roll
            if window is not None:                          # centres within the window (a strip wholly outside
                w0, w1 = window[2 * along], window[2 * along + 1]   # it whose blade still reaches in: its nearest end)
                if s1 + hl <= w0 or s0 - hl >= w1:
                    continue
                s0, s1 = max(s0, min(s1, w0)), min(s1, max(s0, w1))
            pa, pb = [0.0, 0.0], [0.0, 0.0]
            pa[along], pb[along], pa[across], pb[across] = s0, s1, c, c
            out.append(Strip(tuple(pa), tuple(pb), float(yaw)))
    return out


def plan(lanes: Sequence[LaneLike], sc: Scored, half: tuple[float, float], overrun: float,
         window: tuple[float, float, float, float] | None = None) -> list[Strip]:
    """Every lane's strips, in lane order."""
    return [st for ln in lanes for st in strips(ln, sc, half, overrun, window)]
