# SPDX-License-Identifier: Apache-2.0
"""``Rig``: one or two arms at their bases, driven by a motion controller (``robot=rig``) or directly by a script.

With a controller that takes a ``robot`` (its ``enable_gravcomp``, ``set_ready`` and ``Controller``)::

    rig = Rig.single(FR3, base=(-0.45, 0.0, 0.0))          # script key "arm"; Rig.pair(...) -> "left" / "right"
    b = newton.ModelBuilder(); newton.solvers.SolverMuJoCo.register_custom_attributes(b)
    rig.add_to(b)                                           # MJCF with render meshes, MJCF actuators (ctrl_direct)
    ...scene objects..., model = b.finalize(device)  (or replicate the world W times)
    enable_gravcomp(model, robot=rig); ik = rig.make_ik(W, device)
    set_ready(model, rig.ready_tcp(), ik=ik, width=0.04, robot=rig)
    ctl = Controller(model, {"arm": [...]}, ik=ik, robot=rig)

Direct use (no kit Controller): ``rig.bind(model)`` gives per-world arm / gripper coordinates and the ctrl indices
(``ctrl_index() [W, A, J + 1]``), ``rig.grippers[a].ctrl(width, force, q_driver)`` the force-capped gripper command,
``.tcp_z(width)`` the pad centre vs opening, ``rig.make_ik`` batched IK, ``rig.gravity_offsets(q)`` the
``tau_gravity / kp`` lead a position servo needs without MuJoCo's gravcomp.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alloy.arms import mjcf
from alloy.arms.binding import ArmBinding
from alloy.arms.gripper import gripper_for, tcp_in_flange
from alloy.arms.specs import ArmSpec


def _pose7(base) -> tuple[float, ...]:
    base = tuple(float(x) for x in base)
    return base + ((0.0, 0.0, 0.0, 1.0) if len(base) == 3 else ())


@dataclass(frozen=True, eq=False)
class Rig:
    specs: tuple[ArmSpec, ...]
    bases: tuple[tuple[float, ...], ...]          # per arm x y z qx qy qz qw (world frame)
    arms: tuple[str, ...] = ("arm",)              # script keys
    prefixes: tuple[str, ...] = ("",)             # MJCF name prefix per arm
    visuals: bool = True
    grippers: tuple = field(init=False)

    def __post_init__(self) -> None:
        n = len(self.specs)
        if not (len(self.bases) == len(self.arms) == len(self.prefixes) == n):
            raise ValueError("Rig: specs, bases, arms and prefixes must have one entry per arm")
        if len({len(s.joints) for s in self.specs}) != 1:
            raise ValueError("Rig: every arm needs the same joint count (the kit's [W, A, J] arrays)")
        if len(set(self.model_root(s, p) for s, p in zip(self.specs, self.prefixes, strict=True))) != n:
            raise ValueError("Rig: two arms of the same model need distinct prefixes")
        object.__setattr__(self, "bases", tuple(_pose7(b) for b in self.bases))
        object.__setattr__(self, "grippers", tuple(gripper_for(s) for s in self.specs))

    # ------------------------------------------------------------------ layouts

    @staticmethod
    def single(spec: ArmSpec, base=(0.0, 0.0, 0.0), *, name: str = "arm", visuals: bool = True) -> Rig:
        return Rig((spec,), (base,), (name,), ("",), visuals)

    @staticmethod
    def pair(spec: ArmSpec, left_base, right_base, *, visuals: bool = True) -> Rig:
        """Two arms of ``spec``, script keys ``left`` / ``right``, MJCF names prefixed ``left_`` / ``right_``."""
        return Rig((spec, spec), (left_base, right_base), ("left", "right"), ("left_", "right_"), visuals)

    @staticmethod
    def model_root(spec: ArmSpec, prefix: str) -> str:
        return f"sc6_{prefix}{spec.key}"

    # ------------------------------------------------------------------ building

    def add_to(self, builder, **mjcf_kw) -> None:
        """Add every arm to a Newton builder (call ``SolverMuJoCo.register_custom_attributes(builder)`` first)."""
        for spec, base, prefix in zip(self.specs, self.bases, self.prefixes, strict=True):
            mjcf.add_arm(builder, spec, base, prefix=prefix, visuals=self.visuals, **mjcf_kw)

    def robot_refs(self, name: str = "sc6") -> tuple:
        """One ``RobotRef`` per arm (``SceneCard(robots=rig.robot_refs(), ...)`` for SceneAssembler / ``ClipSpec``)."""
        from alloy.cards.types import RobotRef

        return tuple(RobotRef(path=str(mjcf.robot_file(s, p, self.visuals)), kind="mjcf", base_pose=b,
                              name=f"{name}_{p}{s.key}")
                     for s, b, p in zip(self.specs, self.bases, self.prefixes, strict=True))  # fmt: skip

    def builder(self, visuals: bool | None = None):
        """A robot-only Newton builder of this rig."""
        from alloy._vendor import newton

        b = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(b)
        vis = self.visuals if visuals is None else visuals
        for spec, base, prefix in zip(self.specs, self.bases, self.prefixes, strict=True):
            mjcf.add_arm(b, spec, base, prefix=prefix, visuals=vis)
        return b

    def mj_model(self):
        """The rig alone as a MuJoCo C model (bases placed, no render meshes): gravity offsets, offline checks."""
        import mujoco

        root = mujoco.MjSpec()
        for k, (spec, base, prefix) in enumerate(zip(self.specs, self.bases, self.prefixes, strict=True)):
            arm = mujoco.MjSpec.from_string(mjcf.robot_xml(spec, prefix, False))
            if k == 0:
                for f in mjcf._OPTION_FIELDS:
                    setattr(root.option, f, getattr(arm.option, f))
            frame = root.worldbody.add_frame(pos=list(base[:3]), quat=[base[6], *base[3:6]])
            root.attach(arm, frame=frame, prefix="")
        return root.compile()

    def tcp_offsets(self) -> np.ndarray:
        return np.stack([tcp_in_flange(s) for s in self.specs])

    # ------------------------------------------------------------------ kit robot protocol

    @property
    def max_width(self) -> float:
        return float(min(g.max_width for g in self.grippers))

    @property
    def grip_force_max_n(self) -> float:
        return float(min(g.force_max_n for g in self.grippers))

    def bind(self, model) -> ArmBinding:
        return ArmBinding.bind(model, self)

    def make_ik(self, n: int, device: str = "cpu"):
        from alloy.arms.ik import ChainIK

        return ChainIK(self, n, device=device)

    def grip_ctrl(self, bind: ArmBinding, width: np.ndarray, force: np.ndarray, joint_q: np.ndarray) -> np.ndarray:
        jq = np.asarray(joint_q, dtype=np.float64)
        return np.stack([g.ctrl(width[:, a], force[:, a], jq[bind.finger_coord[:, a]])
                         for a, g in enumerate(self.grippers)], axis=1)  # fmt: skip

    def ready_seed(self, worlds: int) -> np.ndarray:
        return np.broadcast_to(np.array([s.home for s in self.specs], dtype=np.float64),
                               (worlds, len(self.specs), len(self.specs[0].joints))).copy()  # fmt: skip

    def write_ready(self, model, bind: ArmBinding, q: np.ndarray, width) -> None:
        """Arm angles ``q [W, A, J]`` and every gripper joint at opening ``width`` (m, scalar or per arm)."""
        jq = model.joint_q.numpy()
        jq[bind.arm_coord] = q
        widths = np.broadcast_to(np.asarray(width, dtype=np.float64), (len(self.specs),))
        for a, g in enumerate(self.grippers):
            jq[bind.grip_coord[a]] = g.joint_values(float(widths[a]))[None, :]
        model.joint_q.assign(jq.astype(np.float32))

    def gravcomp(self, model, bind: ArmBinding) -> None:
        """MuJoCo gravity compensation of every arm body (``gravcomp`` = 1), routed through the arm joint and gripper
        driver actuators (``actuatorgravcomp``, so it counts against their force limits) as real arm controllers
        compensate gravity. Call before the solver is built."""
        mj = model.mujoco
        gc, agc = mj.gravcomp.numpy(), mj.jnt_actgravcomp.numpy()
        roots = tuple(self.model_root(s, p) + "/" for s, p in zip(self.specs, self.prefixes, strict=True))
        for i, label in enumerate(model.body_label):
            if str(label).startswith(roots):
                gc[i] = 1.0
        agc[np.concatenate([bind.arm_dof.ravel(), bind.finger_dof.ravel()])] = True
        mj.gravcomp.assign(gc)
        mj.jnt_actgravcomp.assign(agc)

    # ------------------------------------------------------------------ direct-drive helpers

    def ready_tcp(self) -> np.ndarray:
        """``[A, 7]`` TCP poses at every arm's home angles."""
        return self.make_ik(1).fk(self.ready_seed(1))[0]

    def gravity_offsets(self, q: np.ndarray, width=0.04) -> np.ndarray:
        """``[..., A, J]`` position-servo lead ``tau_gravity(q) / kp`` per arm joint (MuJoCo C, rig alone, nothing
        held): ``ctrl = q + offset`` holds ``q`` without MuJoCo's gravcomp."""
        import mujoco

        if not hasattr(self, "_mj"):
            object.__setattr__(self, "_mj", self.mj_model())
        m = self._mj
        d = mujoco.MjData(m)
        names = [[p + j for j in s.joints] for s, p in zip(self.specs, self.prefixes, strict=True)]
        qadr = np.array([[m.joint(n).qposadr[0] for n in row] for row in names])
        vadr = np.array([[m.joint(n).dofadr[0] for n in row] for row in names])
        kp = np.array([[m.actuator_gainprm[_act_of(m, n), 0] for n in row] for row in names])
        for a, (g, p) in enumerate(zip(self.grippers, self.prefixes, strict=True)):
            d.qpos[[m.joint(p + j).qposadr[0] for j in g.joints]] = g.joint_values(width)
        q = np.asarray(q, dtype=np.float64)
        out = np.empty_like(q)
        for idx in np.ndindex(q.shape[:-2]):
            d.qpos[qadr] = q[idx]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            out[idx] = d.qfrc_bias[vadr] / kp
        return out


def _act_of(m, joint: str) -> int:
    jid = m.joint(joint).id
    for a in range(m.nu):
        if m.actuator_trnid[a, 0] == jid:
            return a
    raise LookupError(f"no actuator on {joint}")


__all__ = ["Rig"]
