# SPDX-License-Identifier: Apache-2.0
"""A tree weld's parent keeps the contact weight it has on the equality path (SCR-DC3-14 part 2, coordinator ruling
2026-11-06 03:07Z in [internal note]). One physics for both weld paths, with the joint-1 armature rule and the
welded-pair filter of `alloy.scene.tree_weld`.

MuJoCo sizes a contact row in acceleration space. The row's weight (efc diagApprox) is the sum of the two bodies'
``body_invweight0``, and set_const computes that at qpos0 from the kinematic tree's mass matrix. A tree weld puts the
welded chain into the parent's tree, so the weights of the parent's bodies count the chain. On DC3's LC plug, tree vs
equality path:
- the plug: translation 270.75 vs 332.67 /kg, rotation 5.4e5 vs 1.8e6;
- the latch member: 346.0 vs 467.2 /kg.

Their contact rows were stiffer in force space, the latch ears held longer, and the plug left the nose.

The rule. Right after a MuJoCo entry is built, and after every ``notify_model_changed`` that recomputes the
constants, every non-chain body in a tree weld's tree takes the ``body_invweight0`` it has without the chain. That
value comes from MuJoCo C's ``mj_setConst`` per world, on a copy of the model in which:
- the chain bodies (link 0 and its subtree) are massless;
- their dofs are decoupled (unit armature);
- each world's body poses, inertias, armature and qpos0 come from MuJoCo Warp.

The parent's tree mass matrix is then the equality path's.

Not changed:
- the chain's own bodies;
- ``dof_invweight0`` (Newton pre-scales the joint-limit solref by it, so limit rows are force-space either way);
- every other tree, and every scene without a tree weld under a body.

The write needs no cache invalidation in Newton. At construction no contact has been converted yet. On a notify,
Newton's constant recompute has already invalidated its contact fast path, so the next step converts contacts in full
from the written weights. `alloy.contact.pair_fit` reads them next (its notify wrapper is installed outside this one).
"""

from __future__ import annotations

import copy
import hashlib
import logging
from dataclasses import dataclass

import numpy as np

from alloy._vendor import newton
from alloy.scene.tree_weld import TREE_LABEL

log = logging.getLogger("alloy.scene")

# The inputs of body_invweight0 that MuJoCo Warp may hold per world; each is copied into the C model before mj_setConst
WORLD_FIELDS = ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "dof_armature",
                "qpos0")


@dataclass(frozen=True)
class TreeSets:
    """MuJoCo indices of the tree welds' chains (link 0 and its subtree, with their dofs) and of the bodies whose
    weight is replaced: the rest of those chains' trees."""

    chain: np.ndarray
    chain_dofs: np.ndarray
    target: np.ndarray


def tree_sets(solver) -> TreeSets | None:
    """The tree welds under a body of a SolverMuJoCo entry, or None (no tree weld, or no Newton -> MuJoCo body map)."""
    model, mj = solver.model, solver.mj_model
    labels = [str(x) for x in (getattr(model, "joint_label", None) or ())]
    roots = [k for k, lab in enumerate(labels) if lab.endswith("/" + TREE_LABEL)]
    to_newton = getattr(solver, "mjc_body_to_newton", None)
    if not roots or to_newton is None:
        return None
    child = model.joint_child.numpy()
    links = {int(child[k]) for k in roots}
    m2n = to_newton.numpy()
    m2n = m2n[0] if m2n.ndim == 2 else m2n
    link0 = {b for b in range(int(mj.nbody)) if int(m2n[b]) in links}
    if not link0:
        return None
    parent, rootid = np.asarray(mj.body_parentid), np.asarray(mj.body_rootid)
    chain: set[int] = set()
    for b in range(int(mj.nbody)):            # MuJoCo orders every parent before its children
        if b in link0 or int(parent[b]) in chain:
            chain.add(b)
    trees = {int(rootid[b]) for b in link0}
    target = [b for b in range(1, int(mj.nbody)) if b not in chain and int(rootid[b]) in trees]
    dofs = [d for d in range(int(mj.nv)) if int(mj.dof_bodyid[d]) in chain]
    return TreeSets(np.array(sorted(chain), np.int64), np.array(dofs, np.int64), np.array(target, np.int64))


def world_inputs(solver) -> dict[str, np.ndarray]:
    """Each WORLD_FIELDS array as [n, *C shape], with n = 1 or the world count: MuJoCo Warp's batched copy, or on the
    CPU path each world's C model values (``_cpu_rows``)."""
    mj, mjw = solver.mj_model, getattr(solver, "mjw_model", None)
    cpu = bool(getattr(solver, "use_mujoco_cpu", False))
    out = {}
    for name in WORLD_FIELDS:
        ref = np.asarray(getattr(mj, name))
        arr = None if mjw is None else getattr(mjw, name, None)
        if arr is None:
            out[name] = ref[None]
            continue
        rows = np.asarray(arr.numpy(), np.float64).reshape(-1, *ref.shape)
        out[name] = _cpu_rows(ref, rows) if cpu else rows
    return out


def _cpu_rows(ref: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The CPU route's per-world C models (``alloy.route.mjc_core_models``): the template's float64 values, with a
    world's Warp value only where its row differs from world 0's. World 0, and any world with world 0's values, keep
    the template bit for bit; a drawn world gets its own chainless weight (WP8's review, 2026-11-07)."""
    if rows.shape[0] <= 1:
        return ref[None]
    out = np.broadcast_to(np.asarray(ref, np.float64), rows.shape).copy()
    differs = rows != rows[:1]
    out[differs] = rows[differs]
    return out


def _key(inputs: dict[str, np.ndarray]) -> str:
    h = hashlib.sha1()
    for name in WORLD_FIELDS:
        h.update(np.ascontiguousarray(inputs[name]).tobytes())
    return h.hexdigest()


def chainless_invweight(mj, sets: TreeSets, inputs: dict[str, np.ndarray], worlds: int) -> np.ndarray:
    """[worlds, targets, 2] body_invweight0 of the target bodies with the chains massless and decoupled (float64)."""
    import mujoco  # noqa: PLC0415

    m = copy.deepcopy(mj)
    d = mujoco.MjData(m)
    out = np.empty((worlds, sets.target.size, 2), np.float64)
    for w in range(worlds):
        for name, val in inputs.items():
            getattr(m, name)[...] = val[w % val.shape[0]]
        m.body_mass[sets.chain] = 0.0
        m.body_inertia[sets.chain] = 0.0
        m.dof_armature[sets.chain_dofs] = 1.0
        mujoco.mj_setConst(m, d)
        out[w] = m.body_invweight0[sets.target]
    return out


def apply_tree_invweight(solver) -> int:
    """Write the chainless weights into the entry's body_invweight0 (MuJoCo Warp, all worlds; the C model on the CPU
    path) when they differ from what is there. Returns the number of (world, body) rows written."""
    sets = getattr(solver, "_alloy_tree_sets", None)
    mjw = getattr(solver, "mjw_model", None)
    if sets is None or mjw is None:
        return 0
    iw, cpu = mjw.body_invweight0, bool(getattr(solver, "use_mujoco_cpu", False))
    arr = iw.numpy()
    rows = arr.reshape(-1, int(solver.mj_model.nbody), 2)
    cache = getattr(solver, "_alloy_tree_invw", None)
    if cache is not None:                     # Newton recomputes the C model's constants on the CPU path
        now = np.asarray(solver.mj_model.body_invweight0)[None, sets.target] if cpu else rows[:, sets.target]
        if np.array_equal(now, cache[1][:now.shape[0]].astype(now.dtype)):
            return 0
    inputs = world_inputs(solver)
    key = _key(inputs)
    own = cache[1] if cache is not None and cache[0] == key else chainless_invweight(solver.mj_model, sets, inputs,
                                                                                      rows.shape[0])
    solver._alloy_tree_invw = (key, own)
    rows[:, sets.target] = own.astype(rows.dtype)
    iw.assign(rows.reshape(arr.shape))
    if cpu:
        solver.mj_model.body_invweight0[sets.target] = own[0]
    log.info("tree weld: body_invweight0 of %d parent-tree bodies set without the chain (%d worlds, SCR-DC3-14); "
             "world 0: %s", sets.target.size, rows.shape[0],
             ", ".join(f"{int(b)} {own[0, i, 0]:.6g}/{own[0, i, 1]:.6g}" for i, b in enumerate(sets.target[:4])))
    return int(rows.shape[0] * sets.target.size)


def recomputes_constants(flags) -> bool:
    """Whether a ``notify_model_changed(flags)`` can recompute MuJoCo's constants (``body_invweight0``) and so needs the
    rule again. CONSTRAINT_PROPERTIES alone cannot: Newton's SolverMuJoCo then updates the equality rows (and connect
    anchors) only; set_const runs for body, joint, tendon and actuator changes (solver_mujoco.py
    ``_notify_model_changed``, main and the Newton 1.8 bump alike). That notification comes every exchange when a
    scene has equality rows (``alloy.topology.solver_hooks``), inside a captured period, where the rule's host read
    (``body_invweight0.numpy()``) aborted the CUDA graph capture (DC5, 1,024 worlds of the showcase, 2026-11-09)."""
    return bool(int(flags) & ~int(newton.ModelFlags.CONSTRAINT_PROPERTIES))


def install_tree_invweight(solver):
    """Apply once and wrap ``notify_model_changed`` so that every constant recompute is followed by the rule. Returns
    the solver. Without a tree weld under a body, nothing is installed."""
    if getattr(solver, "_alloy_tree_sets", None) is not None or getattr(solver, "mj_model", None) is None:
        return solver
    sets = tree_sets(solver)
    if sets is None or sets.target.size == 0:
        return solver
    solver._alloy_tree_sets = sets
    apply_tree_invweight(solver)
    inner = solver.notify_model_changed

    def notify(flags) -> None:
        inner(flags)
        if recomputes_constants(flags):
            apply_tree_invweight(solver)

    solver.notify_model_changed = notify
    return solver


__all__ = ["TreeSets", "WORLD_FIELDS", "apply_tree_invweight", "chainless_invweight", "install_tree_invweight",
           "recomputes_constants", "tree_sets", "world_inputs"]
