# SPDX-License-Identifier: Apache-2.0
"""Body-chain centrelines and pose math for ARB curves and instancers (numpy only)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` [..., 3] by quaternions ``q`` [..., 4] in (x, y, z, w) order."""
    qv, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def apply_pose(pose: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Transform ``local`` points [..., 3] by poses [..., 7] (px py pz qx qy qz qw)."""
    return quat_rotate(pose[..., 3:7], local) + pose[..., :3]


def invert_apply(pose: np.ndarray, world: np.ndarray) -> np.ndarray:
    """Map world points into the frame of ``pose`` (a single 7-vector)."""
    q = np.asarray(pose[3:7], dtype=np.float64).copy()
    q[:3] = -q[:3]
    return quat_rotate(q, np.asarray(world, dtype=np.float64) - pose[:3])


@dataclass(frozen=True)
class SegmentSpec:
    """Per-body centreline sample points, expressed in the body frame, in chain order."""

    centre: np.ndarray  # [n, 3]
    entry: np.ndarray   # [n, 3] end nearer the previous segment
    exit: np.ndarray    # [n, 3] end nearer the next segment


def _neighbours(ends: np.ndarray, tol: float) -> list[set[int]]:
    n = len(ends)
    flat = ends.reshape(2 * n, 3)
    close = np.linalg.norm(flat[:, None] - flat[None], axis=2) < tol
    out: list[set[int]] = [set() for _ in range(n)]
    for i, j in zip(*np.nonzero(close)):
        if i // 2 != j // 2:
            out[i // 2].add(int(j // 2))
    return out


def order_segments(ends: np.ndarray, radius: float) -> list[int]:
    """Order segments (``ends`` [n, 2, 3] world endpoints) along the chain they form."""
    n = len(ends)
    if n < 3:
        return list(range(n))
    seg_len = np.linalg.norm(ends[:, 0] - ends[:, 1], axis=1)
    nbrs = _neighbours(ends, 0.25 * float(seg_len.mean()) + 2.0 * radius)
    centres = ends.mean(axis=1)
    leaves = [i for i in range(n) if len(nbrs[i]) <= 1]
    order = [leaves[0] if leaves else 0]
    left = set(range(n)) - {order[0]}
    while left:
        cur = order[-1]
        pool = [j for j in nbrs[cur] if j in left] or list(left)
        nxt = min(pool, key=lambda j: np.linalg.norm(centres[j] - centres[cur]))
        order.append(nxt)
        left.remove(nxt)
    return order


def build_chain(ends_world: np.ndarray, to_local: list, order: list[int]) -> SegmentSpec:
    """Orient the ordered segments entry->exit and return their body-frame sample points.

    ``ends_world`` is [n, 2, 3] initial world endpoints per segment; ``to_local[i]`` maps a world
    point into segment ``i``'s body frame.
    """
    n = len(order)
    centre, entry, exit_ = (np.zeros((n, 3)) for _ in range(3))
    prev_exit = None
    for pos, i in enumerate(order):
        a, b = ends_world[i]
        ref = prev_exit if prev_exit is not None else (ends_world[order[1]].mean(axis=0) if n > 1 else None)
        if ref is None:
            flip = False
        elif prev_exit is not None:
            flip = np.linalg.norm(b - ref) < np.linalg.norm(a - ref)   # entry is the end nearer the previous exit
        else:
            flip = np.linalg.norm(a - ref) < np.linalg.norm(b - ref)   # first: exit is the end nearer the next
        e, x = (b, a) if flip else (a, b)
        entry[pos], exit_[pos], centre[pos] = to_local[i](e), to_local[i](x), to_local[i]((a + b) / 2.0)
        prev_exit = x
    return SegmentSpec(centre=centre, entry=entry, exit=exit_)


def centreline(spec: SegmentSpec, poses: np.ndarray) -> np.ndarray:
    """Centreline [T, n+2, 3]: first entry, every segment centre, last exit. ``poses`` is [T, n, 7] in chain order."""
    centres = apply_pose(poses, spec.centre[None])
    first = apply_pose(poses[:, :1], spec.entry[None, :1])
    last = apply_pose(poses[:, -1:], spec.exit[None, -1:])
    return np.concatenate([first, centres, last], axis=1)
