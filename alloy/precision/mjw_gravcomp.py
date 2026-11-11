# SPDX-License-Identifier: Apache-2.0
"""Ordered gravity compensation for Alloy's ordered MuJoCo route (coordinator ruling 2026-11-06 06:23Z, GO).

MuJoCo Warp's ``passive._gravity_force`` (mujoco_warp@86e2fae0, passive.py:347) runs one thread per (world, body,
dof). Each adds its body's share to its dof with a float ``atomic_add`` into ``qfrc_gravcomp``. With two or more
gravcomp bodies below a dof, the order of those adds changes run to run, and the dof's force rounds differently.
That was C16's only race (docs/notes/wp5-mjw-determinism.md, round 6).

The replacement runs one thread per (world, dof). It sums the bodies' shares in body order, with the same
``support.jac_dof`` arithmetic, and stores the total. The stage zeroes ``qfrc_gravcomp`` just before the launch
(passive.py:3689), so storing the sum is what the atomic adds would build, in a fixed order. Each thread loops over
nbody, which is small for robot arms. With jnt_actgravcomp set, ``qfrc_actuator`` reads ``qfrc_gravcomp`` per dof
(forward.py:1993), so it repeats as well.
"""

from __future__ import annotations

import warp as wp
from alloy._vendor.mujoco_warp._src import support

from alloy.precision.mjw_launch import Launch


@wp.kernel(enable_backward=False)
def _gravity_force_ordered(
    opt_gravity: wp.array[wp.vec3],
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_mass: wp.array2d[float],
    body_gravcomp: wp.array2d[float],
    dof_bodyid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    xipos_in: wp.array2d[wp.vec3],
    subtree_com_in: wp.array2d[wp.vec3],
    cdof_in: wp.array2d[wp.spatial_vector],
    qfrc_gravcomp_out: wp.array2d[float],
):
    worldid, dofid = wp.tid()
    gravity = opt_gravity[worldid % opt_gravity.shape[0]]
    total = float(0.0)
    for bodyid in range(1, body_parentid.shape[0]):  # body order; skip the world body
        gravcomp = body_gravcomp[worldid % body_gravcomp.shape[0], bodyid]
        if gravcomp != 0.0 and body_isdofancestor[bodyid, dofid] != 0:
            force = -gravity * body_mass[worldid % body_mass.shape[0], bodyid] * gravcomp
            jac, _ = support.jac_dof(body_parentid, body_rootid, dof_bodyid, body_isdofancestor, subtree_com_in,
                                     cdof_in, xipos_in[worldid, bodyid], bodyid, dofid, worldid)
            total += wp.dot(jac, force)
    qfrc_gravcomp_out[worldid, dofid] = total


def gravcomp_handler(args: tuple, launch: Launch) -> None:
    """Replaces one ``_gravity_force`` launch (dim nworld x (nbody - 1) x nv) with the ordered per-dof sum."""
    out = launch.arg("qfrc_gravcomp_out")
    names = ("opt_gravity", "body_parentid", "body_rootid", "body_mass", "body_gravcomp", "dof_bodyid",
             "body_isdofancestor", "xipos_in", "subtree_com_in", "cdof_in")
    wp.launch(_gravity_force_ordered, dim=(int(out.shape[0]), int(out.shape[1])),
              inputs=[launch.arg(n) for n in names], outputs=[out], device=out.device)


__all__ = ["gravcomp_handler"]
