# SPDX-License-Identifier: Apache-2.0
"""Per-world indices of a rig's arms inside any assembled Newton model.

An arm added by ``mjcf.add_arm(builder, spec, base, prefix=p)`` has labels ``sc6_<p><key>/worldbody/.../<p><name>``;
it binds by that root and the prefixed name tail, so rigs sit beside any objects (or a YAM bench) in any number of
worlds. Arrays are ``[W, A, ...]`` in the rig's arm order; ``act`` lists the arm actuators then the gripper's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from alloy.arms.poses import compose


class BindingError(LookupError):
    """An arm joint, body or actuator is missing from the model."""


def _worlds(world: np.ndarray) -> tuple[np.ndarray, list[int]]:
    world = np.asarray(world)
    if (world < 0).all():
        world = np.zeros_like(world)
    return world, sorted(set(int(w) for w in world if w >= 0))


def find_named(labels, world_of: np.ndarray, root: str, names: list[str], what: str) -> np.ndarray:
    """``[W, len(names)]`` index of the item labelled ``<root>/.../<name>`` in every world (exactly one each)."""
    world, worlds = _worlds(world_of)
    pos: dict[tuple[int, str], list[int]] = {}
    for i, lbl in enumerate(labels):
        lbl = str(lbl)
        if lbl.startswith(root + "/"):
            pos.setdefault((int(world[i]), lbl.rsplit("/", 1)[-1]), []).append(i)
    out = np.full((len(worlds), len(names)), -1, dtype=np.int64)
    for wi, w in enumerate(worlds):
        for k, n in enumerate(names):
            hits = pos.get((w, n), [])
            if len(hits) != 1:
                raise BindingError(f"{what} {root}/{n}: {len(hits)} matches in world {w}, want 1")
            out[wi, k] = hits[0]
    return out


@dataclass(frozen=True, eq=False)
class ArmBinding:
    worlds: int
    arm_coord: np.ndarray        # [W, A, J] joint_q index
    arm_dof: np.ndarray          # [W, A, J] joint_qd index
    finger_coord: np.ndarray     # [W, A] gripper driver joint_q index
    finger_dof: np.ndarray       # [W, A]
    grip_coord: tuple            # per arm [W, G_a] every gripper joint (driver first)
    flange: np.ndarray           # [W, A] body the TCP is fixed to
    tcp_offset: np.ndarray       # [A, 7] TCP in the flange frame
    act: np.ndarray | None       # [W, A, J + 1] actuator index; None without MJCF actuators
    lower: np.ndarray            # [A, J] joint limits (rad), world 0
    upper: np.ndarray
    grippers: tuple              # per arm gripper model (alloy.arms.gripper)

    @staticmethod
    def bind(model, rig) -> ArmBinding:
        coords, dofs, gcs, gds, flanges, acts = [], [], [], [], [], []
        q0, qd0 = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
        for spec, prefix, grip in zip(rig.specs, rig.prefixes, rig.grippers, strict=True):
            root = rig.model_root(spec, prefix)
            names = [prefix + j for j in (*spec.joints, *grip.joints)]
            joints = find_named(model.joint_label, model.joint_world.numpy(), root, names, "joint")
            nj = len(spec.joints)
            coords.append(q0[joints[:, :nj]])
            dofs.append(qd0[joints[:, :nj]])
            gcs.append(q0[joints[:, nj:]])
            gds.append(qd0[joints[:, nj]])
            flanges.append(find_named(model.body_label, model.body_world.numpy(), root, [prefix + spec.flange],
                                      "body")[:, 0])  # fmt: skip
            acts.append(_actuators(model, np.concatenate([qd0[joints[:, :nj]], qd0[joints[:, nj : nj + 1]]], 1)))
        coord, dof = np.stack(coords, 1), np.stack(dofs, 1)
        lo, hi = model.joint_limit_lower.numpy(), model.joint_limit_upper.numpy()
        has_act = all(a is not None for a in acts)
        return ArmBinding(worlds=coord.shape[0], arm_coord=coord, arm_dof=dof,
                          finger_coord=np.stack([g[:, 0] for g in gcs], 1),
                          finger_dof=np.stack(gds, 1), grip_coord=tuple(gcs),
                          flange=np.stack(flanges, 1), tcp_offset=rig.tcp_offsets(),
                          act=np.stack(acts, 1) if has_act else None, lower=lo[dof[0]].astype(np.float64),
                          upper=hi[dof[0]].astype(np.float64), grippers=tuple(rig.grippers))  # fmt: skip

    def arm_q(self, joint_q: np.ndarray) -> np.ndarray:
        """``[W, A, J]`` arm joint angles from a flat ``joint_q`` (numpy)."""
        return np.asarray(joint_q, dtype=np.float64)[self.arm_coord]

    def width(self, joint_q: np.ndarray) -> np.ndarray:
        """``[W, A]`` pad opening (m)."""
        jq = np.asarray(joint_q, dtype=np.float64)
        return np.stack([g.width(jq[self.finger_coord[:, a]]) for a, g in enumerate(self.grippers)], 1)

    def tcp(self, body_q: np.ndarray) -> np.ndarray:
        """``[W, A, 7]`` TCP poses from ``body_q [B, 7]`` (numpy)."""
        return compose(np.asarray(body_q, dtype=np.float64)[self.flange], self.tcp_offset[None])

    def ctrl_index(self) -> np.ndarray:
        """``[W, A, J + 1]`` index into ``control.mujoco.ctrl`` (arm joints, then the gripper)."""
        if self.act is None:
            raise BindingError("model has no MJCF actuators (add the arm with ctrl_direct=True)")
        return self.act


def _actuators(model, dof: np.ndarray) -> np.ndarray | None:
    """Actuator per (world, column of ``dof``) from ``model.mujoco.actuator_trnid`` (Newton dof index)."""
    mj = getattr(model, "mujoco", None)
    if mj is None or not hasattr(mj, "actuator_trnid") or mj.actuator_trnid.shape[0] == 0:
        return None
    by_dof = {int(d): a for a, d in enumerate(mj.actuator_trnid.numpy()[:, 0])}
    out = np.full(dof.shape, -1, dtype=np.int64)
    for idx, d in np.ndenumerate(dof):
        if int(d) not in by_dof:
            raise BindingError(f"no MJCF actuator on dof {int(d)}; arm added without its actuators?")
        out[idx] = by_dof[int(d)]
    return out


__all__ = ["ArmBinding", "BindingError", "find_named"]
