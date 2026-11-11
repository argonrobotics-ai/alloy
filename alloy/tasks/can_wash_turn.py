# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can, robot stage: the UR10e turns the can about its axis on the turntable (WP15's rig (b), the
"lathe"; the table: ``can_wash_table``).

The pinch: the 2F-85 closes on the rim on the side away from the drain, ``GRIP_DROP`` under its top, the jaw axis
radial (one finger either side of the wall), at ``PINCH_N``. The turn: the tool point orbits the can's axis with its
frame turning with the can, at ``omega`` with cosine ramps of ``RAMP_S`` at both ends (SOFT-3's ``Tracker`` IK). The
UR's servos and gripper are written directly each control period. Each turn is booked as a ``TurnLog``: the rim force
from the torque balance about the axis, the tip ratio, the can's slip on the table, the axis's drift and the can's rise
(WP15's condition 4), and the rim's slip against the pads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.tasks.can_wash_run import can_z
from alloy.tasks.can_wash_scene import CAN, CanWash
from alloy.tasks.can_wash_table import TABLE, TABLE_JOINT, TurnLog, rim_force, tip_ratio
from alloy.tasks.can_wash_tip import rotate
from alloy.tasks.rinse_flush_policy import CONTROL_DT, bodies

PINCH_N = 120.0          # N, the 2F-85's squeeze on the rim
RELEASE_N = 40.0         # N, the motor cap that opens the jaws (the kit's ``Grip`` default; at 0 the servo's target
                         # stays on the measured driver: the jaws hang limp and on MuJoCo Warp stayed shut, 15 mm)
GRIP_DROP = 0.015        # m, the pinch under the rim's top
OPEN = 0.085             # m, the jaws open (the 2F-85's stroke)
RAMP_S = 0.3             # s, the turn's cosine ramps
APPROACH = 0.10          # m, the pinch approached from above
G = 9.81


def quat_of(R: np.ndarray) -> np.ndarray:
    """xyzw of a rotation matrix (Shepperd's branch for turns near 180°)."""
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1], 4 * w * w]) / (4 * w)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = np.zeros(4)
    q[i], q[j], q[k], q[3] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s, (R[k, j] - R[j, k]) / s
    return q


def frame(z: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The rotation whose z is ``z`` and whose x is ``x`` made normal to it."""
    z = z / np.linalg.norm(z)
    x = x - z * (x @ z)
    x = x / np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def eased(total: float, omega: float, dt: float, ramp: float = RAMP_S) -> np.ndarray:
    """Angles [rad], one per control period, of a turn by ``total`` at ``omega`` with cosine speed ramps of ``ramp``
    at both ends."""
    T = abs(total) / omega + ramp
    t = np.arange(1, int(math.ceil(T / dt)) + 1) * dt
    v = 0.5 * (1.0 - np.cos(math.pi * np.clip(np.minimum(t, T - t) / ramp, 0.0, 1.0))) * omega
    a = np.cumsum(v) * dt
    return np.sign(total) * a * abs(total) / a[-1]


def put_ctrl(ctrl: np.ndarray, idx: np.ndarray, values: Any) -> None:
    """Write ``values`` (one command for every world, or one per world) at ``idx`` ([W, ...] indices into the flat
    ``ctrl``)."""
    ctrl[idx] = np.broadcast_to(np.asarray(values, ctrl.dtype), idx.shape)


@dataclass(frozen=True)
class Lathe:
    """The pinch and the turn's tool poses for ``task``'s can (bay frame), about the table's axis moved by ``shift``
    (x, y) [m]: the can's measured axis when the strokes are re-centred on it."""

    task: CanWash
    shift: tuple[float, float] = (0.0, 0.0)

    def centre(self) -> np.ndarray:
        return np.array([self.task.at[0] + self.shift[0], self.task.at[1] + self.shift[1], can_z(self.task)])

    def to_drain(self) -> np.ndarray:
        d = np.array([self.task.bay.drain[0] - self.task.at[0], self.task.bay.drain[1] - self.task.at[1], 0.0])
        return d / np.linalg.norm(d)

    def grip_pose(self, phi: float, lift: float = 0.0) -> np.ndarray:
        """``[7]`` the tool pose pinching the rim, the can turned ``phi`` about its axis, ``lift`` above the pinch."""
        can, c, d = self.task.can, self.centre(), self.to_drain()
        g0 = c + np.array([*(-(can.r_top - 0.5 * can.wall) * d[:2]), can.height - can.bottom - GRIP_DROP])
        q = np.array([0.0, 0.0, math.sin(phi / 2), math.cos(phi / 2)])
        Rq = np.stack([rotate(q, e) for e in np.eye(3)], 1)
        return np.array([*(c + Rq @ (g0 - c) + [0.0, 0.0, lift]), *quat_of(Rq @ frame(np.array([0, 0, -1.0]), d))])

    def approach(self, n: int = 50) -> list[np.ndarray]:
        """Down onto the pinch from ``APPROACH`` above, over ``n`` periods."""
        return [self.grip_pose(0.0, APPROACH)] + [self.grip_pose(0.0, APPROACH * (1 - k / n)) for k in range(1, n + 1)]

    def turn(self, total: float, omega: float, start: float = 0.0) -> list[np.ndarray]:
        return [self.grip_pose(start + p) for p in eased(total, omega, CONTROL_DT)]


class LipProbe:
    """The table lip's normal force on the can at the end of each control period [N], world 0: on MuJoCo Warp from its
    contacts (summed over the pyramid's edges as ``dc_show_cutter`` does); on Alloy's MuJoCo C route from world 0's
    ``MjData`` (``mj_contactForce``: the route exports contacts to ``mjw_data`` without their constraint rows). NaN
    without a MuJoCo solver."""

    def __init__(self, sim: Any) -> None:
        from alloy.tasks.rinse_hand import mjw_solver  # noqa: PLC0415

        self.solver = mjw_solver(sim)
        names = [str(lb).split("/")[-1] for lb in sim.model.shape_label]
        self.lip = np.array([n.startswith(f"{TABLE}_lip") for n in names])
        self.can = np.array([n.startswith(f"{CAN}_") for n in names])
        self.pyramidal = self.solver is not None and int(self.solver.mjw_model.opt.cone) == 0
        self.vertical = float("nan")      # the force-weighted share of the lip's normals along world z (C route)

    def read(self) -> float:
        if self.solver is None:
            return float("nan")
        if getattr(self.solver, "datas", None):
            return self._read_c()
        d = self.solver.mjw_data
        n = int(d.nacon.numpy()[0])
        geom, wid = d.contact.geom.numpy()[:n], d.contact.worldid.numpy()[:n]
        addr, dim = d.contact.efc_address.numpy()[:n], d.contact.dim.numpy()[:n]
        force, g2s = d.efc.force.numpy(), self.solver.mjc_geom_to_newton_shape.numpy()
        total = 0.0
        for c in range(n):
            s0, s1 = int(g2s[0, geom[c][0]]), int(g2s[0, geom[c][1]])
            if wid[c] != 0 or s0 < 0 or s1 < 0 or addr[c][0] < 0:
                continue
            if (self.lip[s0] and self.can[s1]) or (self.lip[s1] and self.can[s0]):
                k = range(2 * (int(dim[c]) - 1)) if self.pyramidal else range(1)
                total += float(sum(force[0, addr[c][j]] for j in k))
        return total

    def _read_c(self) -> float:
        import mujoco  # noqa: PLC0415

        m, d = self.solver.worlds[0], self.solver.datas[0]
        g2s = self.solver.mjc_geom_to_newton_shape.numpy().reshape(len(self.solver.datas), -1)[0]
        f6, total, up = np.zeros(6), 0.0, 0.0
        for c in range(d.ncon):
            s0, s1 = int(g2s[d.contact[c].geom1]), int(g2s[d.contact[c].geom2])
            if s0 >= 0 and s1 >= 0 and ((self.lip[s0] and self.can[s1]) or (self.lip[s1] and self.can[s0])):
                mujoco.mj_contactForce(m, d, c, f6)
                total += float(f6[0])
                up += float(f6[0]) * abs(float(d.contact[c].frame[2]))       # the normal's world z
        self.vertical = up / total if total > 0.0 else float("nan")
        return total


class LatheRun:
    """The bay built with the table (``can_wash_arms.ArmsBay``), the UR's servos and gripper written each period."""

    def __init__(self, task: CanWash, rig: Any, sim: Any, brake: Any = None) -> None:
        self.task, self.rig, self.sim, self.brake = task, rig, sim, brake
        self.bind = rig.bind(sim.model)
        self.can = int(bodies(sim.model, CAN)[0])
        self.pads = [i for i, lb in enumerate(sim.model.body_label) if "ur10e" in str(lb) and "pad" in str(lb)]
        joint = [i for i, lb in enumerate(sim.model.joint_label) if str(lb).endswith(TABLE_JOINT)][0]
        self.table_q = int(sim.model.joint_q_start.numpy()[joint])
        inertia = sim.model.body_inertia.numpy()
        self.i_can = float(inertia[self.can][2, 2])
        self.i_table = float(inertia[int(bodies(sim.model, TABLE)[0])][2, 2])
        self.mass = float(sim.model.body_mass.numpy()[self.can])
        self._yaw = (None, 0.0)
        self.lip = LipProbe(sim) if task.table is not None and task.table.lip > 0.0 else None
        self.stepper: Any = None
        self.last_yaw: float | None = None      # the can's unwrapped yaw where the last turn ended
        self.aboard: Any = None                 # () -> the water aboard the can now [kg] (G2's pool), or None: dry
        self.last_q: np.ndarray | None = None   # the UR's last joint command

    def hold_table(self, on: bool) -> None:
        """Engage (at each world's table's present angle) or release its brake."""
        if self.brake is None:
            raise ValueError("this run has no table brake")
        tb = np.asarray(bodies(self.sim.model, TABLE), int)
        q = self.sim.state.body_q.numpy()[tb][:, 3:].astype(np.float32)
        self.brake.set_input("target_quat", q[:, None, :].copy())
        self.brake.set_input("grip", np.full((len(tb), 1), 1.0 if on else 0.0, np.float32))

    def write(self, q: np.ndarray, width: float, force: float) -> None:
        """The UR's servo targets ``q`` and its gripper's opening and squeeze, into every world's controls (each
        world runs world 0's commands)."""
        act, sim = self.bind.act, self.sim
        ctrl = sim.control.mujoco.ctrl.numpy().copy()
        put_ctrl(ctrl, act[:, 0, :6], q)
        W, jq = act.shape[0], sim.state.joint_q.numpy()
        grip = self.rig.grip_ctrl(self.bind, np.full((W, 1), width), np.full((W, 1), force), jq)[:, 0]
        put_ctrl(ctrl, act[:, 0, 6], grip)
        sim.control.mujoco.ctrl.assign(ctrl.astype(np.float32))

    def step(self, q: np.ndarray, width: float, force: float) -> None:
        """One control period with the UR at ``q`` (``stepper(q, width, force)`` steps it when set: a driver that
        also commands the other arm and the plug-ins)."""
        self.last_q = np.asarray(q, np.float64)
        if self.stepper is not None:
            self.stepper(q, width, force)
            return
        self.write(q, width, force)
        self.sim.step_period()

    def offset(self) -> np.ndarray:
        """``[2]`` the can's axis (its origin, the inner floor's centre) off the table's axis now [m], world 0."""
        return self.pose()[:2] - np.asarray(self.task.at, np.float64)

    def pose(self) -> np.ndarray:
        return self.sim.state.body_q.numpy()[self.can].astype(np.float64)

    def in_can(self, p: np.ndarray) -> np.ndarray:
        X = self.pose()
        qc = X[3:].copy()
        qc[:3] *= -1
        return rotate(qc, p - X[:3])

    def pads_mid(self) -> np.ndarray:
        return self.sim.state.body_q.numpy()[self.pads][:, :3].astype(np.float64).mean(axis=0)

    def read(self) -> tuple[np.ndarray, float, float]:
        """The can's pose, its yaw unwrapped since the first read [rad] and the table's angle [rad]."""
        X = self.pose()
        x, y, z, w = X[3:]
        raw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        last, acc = self._yaw
        acc = raw if last is None else acc + (raw - last + math.pi) % (2 * math.pi) - math.pi
        self._yaw = (raw, acc)
        return X, acc, float(self.sim.state.joint_q.numpy()[self.table_q])

    def grip(self, qs: list[np.ndarray], close: int = 30, hold: int = 15) -> None:
        """Follow ``qs`` down onto the rim with the jaws open, close at ``PINCH_N`` and hold."""
        for q in qs:
            self.step(q, OPEN, RELEASE_N)
        for _ in range(close + hold):
            self.step(qs[-1], 0.0, PINCH_N)

    def turn(self, qs: list[np.ndarray], settle: int = 20) -> TurnLog:
        """Follow the turn's joint path ``qs`` (pinched), then hold ``settle`` periods; the turn's books. With a
        brake, the plate is free only for the turn: the brake holds it again after the settle."""
        can, table, dt = self.task.can, self.task.table, CONTROL_DT
        ref = self.in_can(self.pads_mid())
        X0, yaw0, ang0 = self.read()
        idle = 0.0 if self.last_yaw is None else yaw0 - self.last_yaw
        water = 0.0 if self.aboard is None else float(self.aboard())    # its weight on the foot resists the tip
        braked = 0.0                            # the brake's grip while the UR turns (0: released, no torque)
        if self.brake is not None:
            self.hold_table(False)
        prev = [(yaw0, ang0)] * 2
        worst = dict(force=0.0, slip=0.0, drift=0.0, rise=0.0, pad=0.0, offset=0.0)
        lip = []
        for q in [*qs, *[qs[-1]] * settle]:
            self.step(q, 0.0, PINCH_N)
            if self.brake is not None:
                braked = max(braked, float(self.brake.grip.numpy().max()))
            if self.lip is not None:
                lip.append(self.lip.read())
            X, yaw, ang = self.read()
            (y1, a1), (y2, a2) = prev[-1], prev[-2]
            prev.append((yaw, ang))
            f = rim_force((yaw - 2 * y1 + y2) / dt ** 2, (ang - 2 * a1 + a2) / dt ** 2, (ang - a1) / dt, self.i_can,
                          self.i_table, table, can.r_top)
            worst["force"] = max(worst["force"], f)
            worst["slip"] = max(worst["slip"], abs((yaw - yaw0) - (ang - ang0)) * can.r_bot)
            worst["drift"] = max(worst["drift"], float(np.linalg.norm(X[:2] - X0[:2])))
            worst["offset"] = max(worst["offset"], float(np.linalg.norm(X[:2] - np.asarray(self.task.at))))
            worst["rise"] = max(worst["rise"], float(X[2] - X0[2]))
            worst["pad"] = max(worst["pad"], float(np.linalg.norm(self.in_can(self.pads_mid()) - ref)))
        if self.brake is not None:
            self.hold_table(True)
        self.last_yaw = prev[-1][0]
        m = self.mass + water
        return TurnLog(force=worst["force"], tip_ratio=tip_ratio(worst["force"], can.height, m, G, can.r_bot),
                       slip=worst["slip"], drift=worst["drift"], rise=worst["rise"], pad_slip=worst["pad"],
                       turned=prev[-1][0] - yaw0, idle=idle, aboard=water, braked=braked, offset=worst["offset"],
                       **lip_books(lip))


def lip_books(forces: list[float], touch_n: float = 0.01) -> dict[str, float]:
    """The lip's peak force [N] and the share of periods over ``touch_n`` (NaN when it was not read)."""
    f = np.asarray(forces, np.float64)
    if f.size == 0 or np.isnan(f).all():
        return {}
    return {"lip_force": float(np.nanmax(f)), "lip_touch": float(np.mean(f > touch_n))}


__all__ = ["APPROACH", "GRIP_DROP", "OPEN", "PINCH_N", "RAMP_S", "RELEASE_N", "Lathe", "LatheRun", "LipProbe", "eased",
           "frame", "lip_books", "put_ctrl", "quat_of"]
