# SPDX-License-Identifier: Apache-2.0
"""Commit and apply bond breaks: per-world event cap, native toggles, body-bond tests (SPEC §3.8).

Commit runs one thread per world over that world's bonds in index order. At most ``max_events_per_step`` breaks are
applied per world and step; the rest are counted as overflow, left intact and re-tested next step. The pool
overflow is counted and never fatal (§3.8 contract item 3). External requests (``request_bonds``, SCR-DC5-1) enter
the same commit, so they are capped and ledgered like any other break.
Toggles: a spring's stiffness and damping go to 0, read live by VBD and XPBD
(newton@ac7f7830:newton/_src/solvers/vbd/solver_vbd.py:4142-4245, xpbd/solver_xpbd.py:740-753). A MuJoCo equality
row is disabled, then SolverMuJoCo picks it up through ``notify_model_changed(CONSTRAINT_PROPERTIES)``
(solver_mujoco.py:10639-10703). A cross-seam bending element is zeroed (VBD skips ke <= 0,
particle_vbd_kernels.py:2626).
Body bonds (weld = RIGID_WELD, connect = JOINT): the force is the norm of the first three MJWarp equality rows of
that equality (rows come first in ``efc``; spike wp10-spikes/spike3_mujoco_weld.py). The strain is the
translation of b, in a's frame, away from its pose at construction, divided by the rest distance between the bodies.
"""

from __future__ import annotations

import warp as wp

from alloy.topology.tables import BODY_KINDS

EFC_EQUALITY = wp.constant(0)          # mujoco.mjtConstraint.mjCNSTR_EQUALITY
K_WELD = wp.constant(BODY_KINDS[0])
K_JOINT = wp.constant(BODY_KINDS[1])


@wp.func
def body_pose(q: wp.array[wp.transform], i: int) -> wp.transform:
    if i < 0:
        return wp.transform_identity()
    return q[i]


@wp.func
def efc_equality_force(w: int, eq_id: int, nefc: wp.array[wp.int32], efc_type: wp.array2d[wp.int32],
                       efc_id: wp.array2d[wp.int32], efc_force: wp.array2d[wp.float32]) -> float:
    """Norm of the translational rows of one equality in world w (0 if inactive or not found)."""
    n = nefc[w]
    r = int(0)
    while r < n:
        if efc_type[w, r] != EFC_EQUALITY:
            return 0.0
        if efc_id[w, r] == eq_id:
            f = wp.vec3(efc_force[w, r], efc_force[w, wp.min(r + 1, n - 1)], efc_force[w, wp.min(r + 2, n - 1)])
            return wp.length(f)
        r += 1
    return 0.0


@wp.struct
class Efc:
    use: int
    nefc: wp.array[wp.int32]
    type: wp.array2d[wp.int32]
    id: wp.array2d[wp.int32]
    force: wp.array2d[wp.float32]


@wp.kernel
def eval_body_bonds(
    kind: wp.array[wp.int8], a: wp.array[wp.int32], b: wp.array[wp.int32], world: wp.array[wp.int32],
    intact: wp.array[wp.int8], break_force: wp.array[wp.float32], break_strain: wp.array[wp.float32],
    rest: wp.array[wp.float32], rel0: wp.array[wp.transform], mjc_eq: wp.array[wp.int32],
    body_q: wp.array[wp.transform], efc: Efc, want: wp.array[wp.int8], strain_out: wp.array[wp.float32],
    force_out: wp.array[wp.float32],
):
    i = wp.tid()
    k = wp.int32(kind[i])
    if intact[i] == 0 or (k != K_WELD and k != K_JOINT):
        return
    rel = wp.transform_multiply(wp.transform_inverse(body_pose(body_q, a[i])), body_pose(body_q, b[i]))
    dp = wp.length(wp.transform_get_translation(rel) - wp.transform_get_translation(rel0[i]))
    strain = float(0.0)
    if rest[i] > 1.0e-9:
        strain = dp / rest[i]
    force = float(0.0)
    if efc.use != 0 and mjc_eq[i] >= 0:
        force = efc_equality_force(world[i], mjc_eq[i], efc.nefc, efc.type, efc.id, efc.force)
    strain_out[i] = strain
    force_out[i] = force
    if strain >= break_strain[i] or force >= break_force[i]:
        want[i] = wp.int8(1)


@wp.kernel
def request_bonds(local: wp.array[wp.int32], world_mask: wp.array[wp.int8], world_start: wp.array[wp.int32],
                  world_order: wp.array[wp.int32], request: wp.array[wp.int8]):
    """External break request (SCR-DC5-1): world-local bond index ``local[j]`` in every world with a nonzero mask.
    Out-of-range indices are ignored (the host path validates them first)."""
    w, j = wp.tid()
    k = local[j]
    if world_mask[w] == 0 or k < 0 or k >= world_start[w + 1] - world_start[w]:
        return
    request[world_order[world_start[w] + k]] = wp.int8(1)


@wp.kernel
def commit_bonds(
    world_start: wp.array[wp.int32], world_order: wp.array[wp.int32], want: wp.array[wp.int8],
    request: wp.array[wp.int8], intact: wp.array[wp.int8], damage: wp.array[wp.float32], cap: int,
    commit: wp.array[wp.int8], phase: wp.array[wp.int8], step_events: wp.array[wp.int32],
    overflow: wp.array[wp.int32], cut_done: wp.array[wp.int8],
):
    """A bond breaks when it is intact and either its criterion fired (``want``) or it was requested. Requests share
    the cap in index order. An overflowed request stays pending; a request on a broken bond is dropped."""
    w = wp.tid()
    n = int(0)
    over = int(0)
    for k in range(world_start[w], world_start[w + 1]):
        i = world_order[k]
        commit[i] = wp.int8(0)
        if intact[i] == 0:
            request[i] = wp.int8(0)
        elif want[i] == 1 or request[i] != 0:
            if n < cap:
                commit[i] = wp.int8(1)
                intact[i] = wp.int8(0)
                damage[i] = 1.0
                if phase[i] == wp.int8(1):
                    cut_done[i] = wp.int8(1)          # the bond was in the blade-cutting phase (TET_CUT only)
                phase[i] = wp.int8(2)
                request[i] = wp.int8(0)
                n += 1
            else:
                over += 1
        want[i] = wp.int8(0)
    step_events[w] = n
    overflow[w] = overflow[w] + over


@wp.kernel
def toggle_bond_natives(commit: wp.array[wp.int8], spring: wp.array[wp.int32], eq: wp.array[wp.int32],
                        spring_stiffness: wp.array[wp.float32], spring_damping: wp.array[wp.float32],
                        eq_enabled: wp.array[wp.bool]):
    i = wp.tid()
    if commit[i] == 0:
        return
    s = spring[i]
    if s >= 0:
        spring_stiffness[s] = 0.0
        spring_damping[s] = 0.0
    e = eq[i]
    if e >= 0:
        eq_enabled[e] = False


@wp.kernel
def toggle_cross_seam_edges(edge_ids: wp.array[wp.int32], edge_bonds: wp.array2d[wp.int32],
                            commit: wp.array[wp.int8], bending: wp.array2d[wp.float32]):
    j = wp.tid()
    b0 = edge_bonds[j, 0]
    b1 = edge_bonds[j, 1]
    hit = False
    if b0 >= 0:
        hit = commit[b0] == 1
    if b1 >= 0:
        hit = hit or commit[b1] == 1
    if hit:
        e = edge_ids[j]
        bending[e, 0] = 0.0
        bending[e, 1] = 0.0
