# SPDX-License-Identifier: Apache-2.0
"""Gripper models for the kit's arms: opening <-> joint values, the force-capped servo command, pad centre vs opening.

Kit units: ``Grip(width, force_n)`` caps the *motor* force; the coupled fingers
squeeze the object with ``force_n / 2`` per pad. Here ``force_n`` means the same for every gripper: twice the per-pad
squeeze it may produce.

- **Franka Hand** (``finger_joint1`` actuated, ``finger_joint2`` = ``finger_joint1`` by equality): width = 2 q, both
  fingers slide along the hand y axis; pads do not move along the approach. Servo ``specs.FRANKA_HAND_SERVO``.
- **Robotiq 2F-85** (four-bar fingers; ``right_driver`` actuated, ``left_driver`` equal, couplers / followers passive
  through ``connect`` equalities): opening and pad height are measured once per process from the pack's own linkage
  in MuJoCo C (gravity off, driver swept to rest, ``robotiq_table``): 85.2 mm open -> pads touching at
  0.78 rad, the pad centre moves 13 mm along the approach (130.8 -> 144.0 mm from ``gripper_base``), 2 mm of it
  over 15-40 mm. The driver servo is the pack's (kp 100 N m/rad, +-5 N m) with kv 2 (``specs.ROBOTIQ_SERVO``):
  per-pad squeeze = torque / |d width / d driver|, at most 43 N closing (85 N in kit units).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from alloy.arms.specs import FRANKA_HAND_SERVO, ROBOTIQ_SERVO, ROBOTIQ_TCP_REF_WIDTH, ArmSpec

ROBOTIQ_JOINTS = tuple(f"gripper_{s}_{j}_joint" for s in ("right", "left")
                       for j in ("driver", "coupler", "spring_link", "follower"))  # fmt: skip
ROBOTIQ_KP, ROBOTIQ_TORQUE = ROBOTIQ_SERVO["kp"], ROBOTIQ_SERVO["force"][1]
ROBOTIQ_CTRL = ROBOTIQ_SERVO["ctrl"]
CLOSE_M = 1e-4                                    # a Grip to this opening or less is a full close


@dataclass(frozen=True)
class FrankaHand:
    """Pads: two columns of small boxes per finger at +-``pad_column`` across the jaw (along TCP y), the finger body
    1.5 mm behind them in between: anything thinner than ~3 mm held at the TCP centre is not pinched. Hold a thread
    or wire on a column (TCP shifted by ``pad_column`` along its y)."""

    kind = "franka_hand"
    pad_column = 5.5e-3
    joints = ("finger_joint1", "finger_joint2")   # every gripper joint, the driver first
    max_width = 0.08
    force_max_n = 70.0                             # motor cap = the servo force range (2 x 35 N per pad)
    kp = FRANKA_HAND_SERVO["kp"]
    ctrl_range = FRANKA_HAND_SERVO["ctrl"]

    def width(self, q_driver: np.ndarray) -> np.ndarray:
        return 2.0 * np.asarray(q_driver, dtype=np.float64)

    def joint_values(self, width: float) -> np.ndarray:
        """Every gripper joint (``joints`` order) at opening ``width``."""
        q = float(np.clip(width, 0.0, self.max_width)) / 2.0
        return np.array([q, q])

    def ctrl(self, width: np.ndarray, force_n: np.ndarray, q_driver: np.ndarray) -> np.ndarray:
        """Servo target: the planned half-width, kept within force/kp of the measured finger (quasi-static
        force = kp (target - q) <= force_n), inside the servo's ctrl range. A full close (width <= CLOSE_M) aims
        at the end of the range, so any object is squeezed with the full cap."""
        q = np.asarray(q_driver, dtype=np.float64)
        lead = np.asarray(force_n, dtype=np.float64) / self.kp
        w = np.asarray(width, dtype=np.float64)
        plan = np.where(w <= CLOSE_M, self.ctrl_range[0], np.clip(w / 2.0, 0.0, self.max_width / 2.0))
        return np.clip(np.clip(plan, q - lead, q + lead), *self.ctrl_range)

    def tcp_z(self, width) -> np.ndarray:
        """Pad centre along the approach, relative to the spec's TCP (Franka pads slide sideways only)."""
        return np.zeros_like(np.asarray(width, dtype=np.float64))


@dataclass(frozen=True)
class RobotiqTable:
    driver: np.ndarray       # [K] driver angle (rad), increasing = closing
    width: np.ndarray        # [K] pad face gap (m), decreasing
    pad_z: np.ndarray        # [K] pad centre along gripper_base z (m)
    joints: np.ndarray       # [K, 8] every gripper joint (ROBOTIQ_JOINTS order) at rest


@functools.lru_cache(maxsize=1)
def robotiq_table(samples: int = 41) -> RobotiqTable:
    """The 2F-85 linkage swept in MuJoCo C (UR10e pack's gripper; the UR5e pack carries the same one)."""
    import mujoco

    from alloy.arms.mjcf import robot_xml
    from alloy.arms.specs import UR10E

    m = mujoco.MjModel.from_xml_string(robot_xml(UR10E, "", False))
    m.opt.gravity[:] = 0.0
    d = mujoco.MjData(m)
    base = m.body("gripper_base").id
    pads = {s: [m.geom(f"gripper_{s}_pad{k}").id for k in (1, 2)] for s in ("right", "left")}
    half_y = float(m.geom_size[pads["right"][0]][1])
    adr = [m.joint(j).qposadr[0] for j in ROBOTIQ_JOINTS]
    d.qpos[:6] = UR10E.home
    rows = []
    for c in np.linspace(*ROBOTIQ_CTRL, samples):
        d.ctrl[:6], d.ctrl[6] = UR10E.home, c
        for _ in range(400):
            mujoco.mj_step(m, d)
        R, p0 = d.xmat[base].reshape(3, 3), d.xpos[base]
        cen = {s: np.array([R.T @ (d.geom_xpos[g] - p0) for g in pads[s]]) for s in pads}
        width = abs(cen["right"][:, 1].mean() - cen["left"][:, 1].mean()) - 2.0 * half_y
        rows.append((d.qpos[adr[0]], width, np.concatenate([cen["right"][:, 2], cen["left"][:, 2]]).mean(),
                     d.qpos[adr].copy()))  # fmt: skip
    drv, w, z, q = (np.array(x) for x in zip(*rows, strict=True))
    return RobotiqTable(driver=drv, width=w, pad_z=z, joints=q)


@dataclass(frozen=True)
class Robotiq2F85:
    kind = "robotiq_2f85"
    joints = ROBOTIQ_JOINTS
    kp = ROBOTIQ_KP

    @property
    def table(self) -> RobotiqTable:
        return robotiq_table()

    @property
    def max_width(self) -> float:
        return float(self.table.width[0])

    @property
    def force_max_n(self) -> float:
        """Kit units at the weakest point of the stroke: 2 x torque / max |d width / d driver|."""
        t = self.table
        return float(2.0 * ROBOTIQ_TORQUE / np.abs(np.gradient(t.width, t.driver)).max())

    def width(self, q_driver: np.ndarray) -> np.ndarray:
        t = self.table
        return np.interp(np.asarray(q_driver, dtype=np.float64), t.driver, t.width)

    def driver_for(self, width: np.ndarray) -> np.ndarray:
        t = self.table
        return np.interp(np.asarray(width, dtype=np.float64), t.width[::-1], t.driver[::-1])

    def joint_values(self, width: float) -> np.ndarray:
        t = self.table
        drv = float(self.driver_for(width))
        return np.array([np.interp(drv, t.driver, t.joints[:, k]) for k in range(t.joints.shape[1])])

    def ctrl(self, width: np.ndarray, force_n: np.ndarray, q_driver: np.ndarray) -> np.ndarray:
        """Driver target for the planned opening, within torque/kp of the measured driver, the torque being the one
        that squeezes ``force_n / 2`` per pad at the current linkage ratio (a full close aims at the range end)."""
        t = self.table
        q = np.asarray(q_driver, dtype=np.float64)
        ratio = np.abs(np.interp(q, t.driver, np.gradient(t.width, t.driver)))
        lead = np.minimum(np.asarray(force_n) / 2.0 * ratio, ROBOTIQ_TORQUE) / self.kp
        w = np.asarray(width, dtype=np.float64)
        plan = np.where(w <= CLOSE_M, ROBOTIQ_CTRL[1], self.driver_for(np.clip(w, 0.0, self.max_width)))
        return np.clip(np.clip(plan, q - lead, q + lead), *ROBOTIQ_CTRL)

    def tcp_z(self, width) -> np.ndarray:
        """Pad centre along the approach relative to the TCP (pad centre at ``ROBOTIQ_TCP_REF_WIDTH``): negative =
        the pads sit short of the TCP (opening wider than the reference)."""
        t = self.table
        ref = np.interp(ROBOTIQ_TCP_REF_WIDTH, t.width[::-1], t.pad_z[::-1])
        return np.interp(np.asarray(width, dtype=np.float64), t.width[::-1], t.pad_z[::-1]) - ref

    def tcp_ref_z(self) -> float:
        t = self.table
        return float(np.interp(ROBOTIQ_TCP_REF_WIDTH, t.width[::-1], t.pad_z[::-1]))


def gripper_for(spec: ArmSpec):
    return {"franka_hand": FrankaHand, "robotiq_2f85": Robotiq2F85}[spec.gripper.kind]()


def tcp_in_flange(spec: ArmSpec) -> np.ndarray:
    """The spec's TCP in its flange frame; the 2F-85's z is its pad centre at ``ROBOTIQ_TCP_REF_WIDTH``."""
    tcp = np.array(spec.tcp, dtype=np.float64)
    if spec.gripper.kind == "robotiq_2f85":
        tcp[2] = Robotiq2F85().tcp_ref_z()
    return tcp


__all__ = ["FrankaHand", "Robotiq2F85", "RobotiqTable", "gripper_for", "robotiq_table", "tcp_in_flange"]
