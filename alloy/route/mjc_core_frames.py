# SPDX-License-Identifier: Apache-2.0
"""The CPU route's per-world models keep MuJoCo's compile-time frame data true to their own values.

A world's ``MjModel`` is a copy of the template with its per-world rows written in (``mjc_core_models``). MuJoCo
derives two things from those rows when it compiles, and a copy keeps the template's:

- the frame shortcuts. A geom or site the template compiled onto its body's frame (``geom_sameframe``) is placed at
  the body's pose, its own pos / quat never read, and a body whose inertial frame is its own (``body_sameframe``) has
  its centre of mass at its origin;
- each body's bounding-volume tree (``bvh_aabb``): one box per geom in the body's inertial frame, and boxes around
  them. The collision midphase drops every geom pair whose boxes do not overlap. A plane's box is the half-space
  below it, so a floor moved up kept its old half-space, and nothing between the two heights touched it.

So a robot file's floor raised at a world's per-world start stood at world 0's height, and once that was placed
right it still held no contacts there (a Go2's feet sank 5 cm into it with none). A centre of mass that differs from
world 0's has the same two dependencies (the shortcut only where world 0's sits at the body's origin). ``refresh``
clears the shortcuts (``mjSAMEFRAME_NONE``: placed from its own values, right for any value) and refits the trees of
the bodies a world changes; a world that matches world 0 keeps the template's data and bits.
"""

from __future__ import annotations

import mujoco
import numpy as np

_NONE = int(mujoco.mjtSameFrame.mjSAMEFRAME_NONE)
_INERTIA = (int(mujoco.mjtSameFrame.mjSAMEFRAME_INERTIA), int(mujoco.mjtSameFrame.mjSAMEFRAME_INERTIAROT))


def _moved(diff: dict[str, np.ndarray], fields: tuple[str, ...], w: int, n: int) -> np.ndarray:
    """``[n]`` rows of ``fields`` where world ``w`` differs from world 0."""
    out = np.zeros(n, dtype=bool)
    for f in fields:
        if f in diff:
            out |= np.any((diff[f][w] != diff[f][0]).reshape(n, -1), axis=1)
    return out


def refresh(m: mujoco.MjModel, diff: dict[str, np.ndarray], w: int) -> None:
    """World ``w``'s model ``m`` (its rows of ``diff`` written in): frame shortcuts cleared and bounding trees refit
    where it differs from world 0 (module docstring). Example: ``refresh(m, batched_fields(solver, 4), 2)``."""
    com = _moved(diff, ("body_ipos", "body_iquat"), w, m.nbody)
    geoms = _moved(diff, ("geom_pos", "geom_quat", "geom_aabb"), w, m.ngeom)
    m.body_sameframe[com] = _NONE
    m.geom_sameframe[geoms | (com[m.geom_bodyid] & np.isin(m.geom_sameframe, _INERTIA))] = _NONE
    sites = _moved(diff, ("site_pos", "site_quat"), w, m.nsite)
    m.site_sameframe[sites | (com[m.site_bodyid] & np.isin(m.site_sameframe, _INERTIA))] = _NONE
    bodies = np.flatnonzero(com | np.isin(np.arange(m.nbody), m.geom_bodyid[geoms]))
    for b in bodies.tolist():
        refit_tree(m, b)


def _mat(q: np.ndarray) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, np.float64))
    return out.reshape(3, 3)


def refit_tree(m: mujoco.MjModel, b: int) -> None:
    """Body ``b``'s bounding-volume tree from its geoms' boxes now: each leaf the box (``geom_aabb``) of its geom in the
    body's inertial frame, each inner node the box around its children, as MuJoCo compiles it.

    Example: ``refit_tree(m, 0)`` after moving a world-body geom."""
    adr, num = int(m.body_bvhadr[b]), int(m.body_bvhnum[b])
    if adr < 0 or num == 0:
        return
    r_i = _mat(m.body_iquat[b]) if b else np.eye(3)
    p_i = np.asarray(m.body_ipos[b], np.float64) if b else np.zeros(3)
    for node in range(adr + num - 1, adr - 1, -1):          # children follow their parent: leaves first, root last
        g = int(m.bvh_nodeid[node])
        if g >= 0:
            rot = r_i.T @ _mat(m.geom_quat[g])
            centre = r_i.T @ (m.geom_pos[g] - p_i) + rot @ m.geom_aabb[g, :3]
            m.bvh_aabb[node] = np.concatenate([centre, np.abs(rot) @ m.geom_aabb[g, 3:]])
            continue
        kids = [adr + int(c) for c in m.bvh_child[node] if c >= 0]
        lo = np.min([m.bvh_aabb[k, :3] - m.bvh_aabb[k, 3:] for k in kids], axis=0)
        hi = np.max([m.bvh_aabb[k, :3] + m.bvh_aabb[k, 3:] for k in kids], axis=0)
        m.bvh_aabb[node] = np.concatenate([(lo + hi) / 2, (hi - lo) / 2])


__all__ = ["refit_tree", "refresh"]
