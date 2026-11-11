# SPDX-License-Identifier: Apache-2.0
"""Mass accounting for fixed-base articulated MJCF objects (SCR-DC4-2; routing in alloy/route/articulated.py).

Such an object routes STATIC with the MuJoCo entry. Its fixed root stays static, but the bodies below a non-fixed joint
move and count as the entry's mass. Builder lists and Model arrays both work: values go through `np.asarray`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from alloy._vendor import newton

from alloy.scene.template import TemplateObject


def is_fixed_base_articulated(t: TemplateObject) -> bool:
    """A static object card that the router gave an entry (a static MJCF file with nested joints)."""
    return t.obj is not None and t.obj.card.motion == "static" and bool(t.decision.entry)


def moving_bodies(bodies: range, joint_parent: Sequence[int], joint_child: Sequence[int],
                  joint_type: Sequence[int]) -> list[int]:
    """Bodies in `bodies` with a non-fixed joint on their path to the world."""
    parent, child, kind = (np.asarray(x) for x in (joint_parent, joint_child, joint_type))
    into = {int(c): j for j, c in enumerate(child) if int(c) in bodies}
    fixed = int(newton.JointType.FIXED)
    out = []
    for b in bodies:
        node, moves = b, False
        while node in into and not moves:
            j = into[node]
            moves = int(kind[j]) != fixed
            node = int(parent[j])
        if moves:
            out.append(b)
    return out


def moving_mass(bodies: range, body_mass: Sequence[float], joint_parent: Sequence[int], joint_child: Sequence[int],
                joint_type: Sequence[int]) -> float:
    """Mass of the moving bodies [kg]."""
    mass = np.asarray(body_mass, np.float64)
    return float(sum(mass[b] for b in moving_bodies(bodies, joint_parent, joint_child, joint_type)))


__all__ = ["is_fixed_base_articulated", "moving_bodies", "moving_mass"]
