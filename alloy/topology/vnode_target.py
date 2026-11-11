# SPDX-License-Identifier: Apache-2.0
"""SB-7 target ``cut_block_free_copies`` (WP10 item 9 step 3 prototype; step 4 compares it with ``cut_block_free``).

``cut_block_free``'s class on the virtual-node rig: the same block (no seams, every tet cuttable by copies), the same
per-world blade yaw and lean (``bully_target.free_angles``; replay world k alone with ``first_world=k``), the same
translation-only actions in each world's blade frame with the +-1 mm face-first band (coordinator ruling
2026-11-06T14:17Z; WP7's ``bully_action_frame``) and the same rows (``VNodeSim.bully_monitors``). Only the cut model
differs. Registered by ``bully_target`` (WP7's provider list reaches it there).
"""

from __future__ import annotations

import numpy as np

from alloy._vendor import newton
from alloy.bully.types import BullyTarget
from alloy.topology import bully_target as bt
from alloy.topology.vnode_rig import VNodeSim, vnode_block


def cut_block_free_copies(first_world: int = 0) -> BullyTarget:
    """``cut_block_free`` on virtual-node copies; ``first_world`` numbers the batch's first world."""
    from alloy.bully.actions import frame_from_poses

    free = bt.cut_block_free(first_world)
    start = bt.free_start()

    def make_sim(worlds: int, device: str) -> VNodeSim:
        sc = vnode_block(device, worlds=worlds, cells=bt.CELLS, k_mu=5.0e4, k_lambda=5.0e4, gravity=0.0)
        m = sc.model
        m.soft_contact_ke = bt.STROKE_KE
        q, starts = m.joint_q.numpy(), m.joint_q_start.numpy()
        poses = np.stack([bt.free_pose(start, yaw, pitch) for yaw, pitch in bt.free_angles(worlds, first_world)])
        for w in range(worlds):
            q[starts[w]:starts[w] + 7] = poses[w]
        m.joint_q.assign(q)
        sim = VNodeSim(sc, newton.solvers.SolverVBD(m, iterations=4), control_dt=bt.CONTROL_DT, substeps=bt.SUBSTEPS)
        sim.action_frame = frame_from_poses(poses)  # blade frame: face normal, edge, cutting side
        return sim

    return BullyTarget(name="cut_block_free_copies", scene=None, make_sim=make_sim, actions=free.actions,
                       action_kind="body_pose", action_index=(0, 1, 2), robot_bodies=(0,))

