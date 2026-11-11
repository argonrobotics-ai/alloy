# SPDX-License-Identifier: Apache-2.0
"""Gap-ladder L1 on appended MuJoCo rows (D10 spike, SCR-WP3-1).

A separated precision candidate (``gap = dist - includemargin > 0``) becomes a frictionless row that MuJoCo admits
(``dist`` set just inside the margin) with Newton's speculative solref/solimp; after ``make_constraint`` the
``callback.passive`` hook restores the real clearance and sets the target ``v_next ≥ -gap/h``:
``aref = -(v + gap/h)/h``. Same maths and constants as newton@ac7f7830:newton/_src/solvers/mujoco/speculative.py:29-84,
which does this between ``step1`` and ``step2`` on the ``native="newton"`` path. ``passive`` runs in
``fwd_velocity`` after ``make_constraint`` and before ``solve`` (mujoco-warp 3.16.0 ``_src/forward.py:1464, 1548``,
``_src/passive.py:3695-3696``); it is skipped when SPRING and DAMPER are both disabled (passive.py:3395-3401).
"""

from __future__ import annotations

import warp as wp

vec5 = wp.types.vector(length=5, dtype=wp.float32)

SPEC_SOLIMP = vec5(0.95, 0.95, 0.001, 0.5, 2.0)      # speculative.py:58


@wp.func
def prepare_speculative_row(cid: int, h: float, dist: wp.array[float], margin: wp.array[float],
                            dim: wp.array[int], solref: wp.array[wp.vec2], solimp: wp.array[vec5],
                            clearance: wp.array[float]):
    """Turn row ``cid`` into a speculative row if separated; record its clearance (0 = ordinary row)."""
    gap = dist[cid] - margin[cid]
    clearance[cid] = 0.0
    if gap > 0.0:
        clearance[cid] = gap
        dist[cid] = margin[cid] - wp.max(1.0e-8, wp.abs(margin[cid]) * 1.0e-6)
        dim[cid] = 1
        solref[cid] = wp.vec2(2.0 * h, 1.0)
        solimp[cid] = vec5(0.95, 0.95, 0.001, 0.5, 2.0)


@wp.kernel
def set_speculative_targets(start: wp.array[wp.int32], end: wp.array[wp.int32], timestep: wp.array[float],
                            world: wp.array[int], clearance: wp.array[float], address: wp.array2d[int],
                            dist: wp.array[float], margin: wp.array[float], efc_vel: wp.array2d[float],
                            efc_pos: wp.array2d[float], efc_aref: wp.array2d[float]):
    """Over the appended range: restore ``dist`` and set ``efc.pos``/``efc.aref`` of speculative rows."""
    i = wp.tid()
    if i < start[0] or i >= end[0]:
        return
    gap = clearance[i]
    if gap <= 0.0:
        return
    dist[i] = gap + margin[i]
    row = address[i, 0]
    if row < 0:
        return
    w = world[i]
    h = timestep[w % timestep.shape[0]]
    efc_pos[w, row] = gap
    efc_aref[w, row] = -(efc_vel[w, row] + gap / h) / h


def l1_callback_available(disableflags: int) -> bool:
    """False when MJWarp would skip ``passive()`` entirely (SPRING and DAMPER both disabled)."""
    from alloy._vendor.mujoco_warp import DisableBit  # noqa: PLC0415

    both = int(DisableBit.SPRING) | int(DisableBit.DAMPER)
    return (int(disableflags) & both) != both
