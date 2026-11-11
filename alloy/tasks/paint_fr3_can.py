# SPDX-License-Identifier: Apache-2.0
"""spray-stencil with a fixed FR3 holding the aerosol can (:mod:`alloy.tasks.paint_spray_supply`).

The FR3 stands on a 0.95 m pedestal 0.45 m from the box's centre, facing it (the base y, height scanned for the
widest joint margin over the rows: 0.26 rad, the arm 64 mm from the box). It holds the can from above by its top,
the fingers across it along the box's long side, the can upright with its nozzle down (the supply's nozzle points
along the can's -z): the hand stays over the box's rim while the can sprays from inside it. The can's geoms do not
collide.

The plan (:func:`plan_can`) moves the nozzle through the stand-in's rows at its speed with the valve open, one aim
point a sub-period at each sub-period's middle as the kinematic can sprays them. A kinematic can starts and reverses
a row at speed and steps 40 mm to the next row in one sub-period; the arm instead runs up to speed over ``RUN``
before each row and runs out after it, and steps between rows on a minimum-jerk path, the valve shut. The tool point
tracks the nozzle's path by IK from the start whose track keeps the widest joint margin; the servo targets are led
by the servos' lag (:func:`alloy.tasks.paint_arm.led_targets`).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from alloy.tasks.paint_arm import (ArmScene, Plan, Tracker, arm_scene, down_tcp, led_targets, servo_lead, smooth,
                                   static_clearance)
from alloy.tasks.paint_spray_stencil import CONTROL_DT, Can, SprayLayout

CAN_BASE = (0.0, -0.45, 0.95)   # m: the arm's base on its pedestal, facing the box (+y)
CAN_YAW = 90.0                  # deg
GRIP = 0.155                    # m: the tool point above the nozzle (the fingers on the can's top; it is 0.17 tall)
CAN_R, CAN_HALF, CAN_MID = 0.033, 0.08, 0.09    # m: the can's cylinder (the kinematic scene's); its centre's height
CAN_MASS = 0.34                 # kg (a full 340 g can)
RUN = 0.06                      # m: the run-up to a row's speed before it and the run-out after it, the valve shut
STEP_S = 0.5                    # s: the step to the next row
MOVE_RATE = 0.2                 # m/s: the move to the next coat's first row, on average (in the coats' pause)
CAN = "sc6_fr3/worldbody/base/fr3_link0/fr3_link1/fr3_link2/fr3_link3/fr3_link4/fr3_link5/fr3_link6/fr3_link7/" \
      "hand/can"                # the can's body label in the arm's MJCF


def _tcp():
    from alloy.arms.specs import FRANKA_TCP  # noqa: PLC0415

    return FRANKA_TCP


def can_xml() -> str:
    """The FR3's MJCF with the can in its hand: body ``can`` at the nozzle, ``GRIP`` past the tool point along the
    approach, turned so its z points back up the approach (upright when the hand points down)."""
    from alloy.arms import FR3  # noqa: PLC0415
    from alloy.arms.mjcf import robot_xml  # noqa: PLC0415

    root = ET.fromstring(robot_xml(FR3, "", False))
    hand = next(b for b in root.iter("body") if b.get("name") == "hand")
    can = ET.SubElement(hand, "body", name="can", pos=f"0 0 {_tcp()[2] + GRIP}", quat="0 1 0 0")
    side, axial = CAN_MASS * (3 * CAN_R ** 2 + (2 * CAN_HALF) ** 2) / 12, 0.5 * CAN_MASS * CAN_R ** 2
    ET.SubElement(can, "inertial", pos=f"0 0 {CAN_MID}", mass=str(CAN_MASS), diaginertia=f"{side} {side} {axial}")
    ET.SubElement(can, "geom", type="cylinder", size=f"{CAN_R} {CAN_HALF}", pos=f"0 0 {CAN_MID}", contype="0",
                  conaffinity="0")  # fmt: skip
    return ET.tostring(root, encoding="unicode")


def base_quat(yaw: float = CAN_YAW) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(math.radians(0.5 * yaw)), math.cos(math.radians(0.5 * yaw)))


def rig(base: tuple[float, float, float] = CAN_BASE):  # noqa: ANN201 - alloy.arms.Rig
    from alloy.arms import FR3, Rig  # noqa: PLC0415

    return Rig.single(FR3, (*base, *base_quat()), name="arm", visuals=False)


def tcp_for(nozzle) -> np.ndarray:  # noqa: ANN001
    """The tool point's pose that puts the nozzle at ``nozzle``, pointing down (:func:`~alloy.tasks.paint_arm.down_tcp`:
    the fingers closing along world x, the box's long side)."""
    return down_tcp(nozzle, GRIP)


def nozzle_z(layout: SprayLayout, can: Can) -> float:
    return layout.board_z + can.standoff


def sub_dt(can: Can) -> float:
    """s: one sub-period (an aim point)."""
    return CONTROL_DT / can.substeps


def _ramp(speed: float, dt: float, up: bool) -> tuple[np.ndarray, float]:
    """Distances [m] at each sub-period's end of a smooth ramp from rest to ``speed`` (``up``) or back: the velocity
    ``speed·(3τ² - 2τ³)``, no jump in acceleration. Returns them and the ramp's length."""
    n = max(1, int(round(2.0 * RUN / (speed * dt))))
    t = n * dt
    tau = np.arange(1, n + 1) / n
    if up:
        return speed * t * (tau ** 3 - 0.5 * tau ** 4), 0.5 * speed * t
    back = 1.0 - tau
    return 0.5 * speed * t - speed * t * (back ** 3 - 0.5 * back ** 4), 0.5 * speed * t


@dataclass
class Path:
    """The nozzle's path: its place at each sub-period's start, and whether the valve is open over that sub-period."""

    points: list
    valve: list

    def go(self, p, valve: bool = False) -> None:  # noqa: ANN001
        self.points.append(np.asarray(p, np.float64))
        self.valve.append(valve)


def nozzle_path(layout: SprayLayout, can: Can) -> Path:
    """The arm's nozzle path over the stand-in's rows (:func:`~alloy.tasks.paint_spray_stencil.schedule`): each row
    at ``can.speed`` with the valve open, its sub-periods' middles at the kinematic can's aim points; a run-up before
    and a run-out after it, then a minimum-jerk step to the next row's run-up (in the coats' pause, the move to the
    next coat's first row, then a wait to the pause's end), all with the valve shut."""
    dt, z = sub_dt(can), nozzle_z(layout, can)
    up, run = _ramp(can.speed, dt, True)
    down, _ = _ramp(can.speed, dt, False)
    m = int(math.ceil(2.0 * can.span / (can.speed * CONTROL_DT) - 1e-9)) * can.substeps
    path = Path([], [])
    for coat in range(can.coats):
        for k, y in enumerate(can.rows):
            sgn = 1.0 if (k + coat) % 2 == 0 else -1.0                  # back and forth, as the stand-in
            start = np.array([sgn * (-can.span - run), y, z])
            if not path.points:
                path.go(start)
            else:
                moved = smooth(path.points[-1], start, dt, rate=MOVE_RATE, least=STEP_S)
                wait = int(round(can.pause / dt)) - len(moved) if coat and k == 0 else 0
                for p in moved + [start] * max(0, wait):
                    path.go(p)
            for d in up:                                                # the last is the row's start, at speed
                path.go(start + (sgn * d, 0.0, 0.0))
            x0 = path.points[-1][0]
            path.valve[-1] = True
            for j in range(1, m + 1):
                path.go((x0 + sgn * j * can.speed * dt, y, z), j < m)
            end = path.points[-1]
            for d in down:
                path.go(end + (sgn * d, 0.0, 0.0))
    return path


@dataclass
class CanStep:
    """One sub-period of the arm's run: the joint angles at its start, the nozzle's planned place, the valve."""

    q: np.ndarray
    nozzle: np.ndarray
    valve: bool


def box_clearance(layout: SprayLayout, base: tuple[float, float, float], qs: list[np.ndarray]) -> float:
    """m: the arm's least distance to the box's floor and walls over the joint vectors ``qs`` (MuJoCo C)."""
    from alloy.tasks.paint_spray_supply import box_shapes  # noqa: PLC0415

    return static_clearance(can_xml(), (*base, *base_quat()), box_shapes(layout), qs)


def plan_can(layout: SprayLayout, can: Can, base: tuple[float, float, float] = CAN_BASE) -> Plan:
    """The arm's steps over :func:`nozzle_path`: the tool point tracks it by IK, from the start whose track keeps the
    widest joint margin; the clearance is the arm's least distance to the box (every 10th step)."""
    path = nozzle_path(layout, can)
    plan = Plan()
    tr = Tracker(rig(base), plan)
    targets = [tcp_for(p) for p in path.points]
    qs = tr.track(targets, tr.start(targets))
    plan.steps = [CanStep(q, p, v) for q, p, v in zip(qs, path.points, path.valve, strict=True)]
    plan.clearance = box_clearance(layout, base, qs[::10])
    return plan.finish(sub_dt(can))


def can_scene(layout: SprayLayout, can: Can, init, q0: np.ndarray,  # noqa: ANN001, ANN201 - a harness TestScene
              base: tuple[float, float, float] = CAN_BASE):
    """The FR3 holding the can on its pedestal, the crate, the stencil and the box (as the kinematic scene's): MuJoCo,
    eight steps a sub-period, the arm at ``q0``. The can does not collide; the arm does, with the box and the
    pedestal."""
    from alloy.tasks.paint_spray_supply import box_shapes, stencil_mesh  # noqa: PLC0415

    def statics(t, cfg) -> None:  # noqa: ANN001 - newton.ModelBuilder, ShapeConfig
        from alloy._vendor import newton  # noqa: PLC0415

        from alloy.plugins.harness.scene import body_xform  # noqa: PLC0415

        hx, hy = 0.5 * layout.crate[0], 0.5 * layout.crate[1]
        t.add_shape_box(-1, xform=body_xform(0.0, 0.0, layout.board_z - 0.006), hx=hx, hy=hy, hz=0.006, cfg=cfg,
                        label="crate")  # fmt: skip
        # the stencil blocks the spray (a colliding shape) but MuJoCo takes no flat mesh as a collider: group 0
        sheet = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, collision_group=0)
        t.add_shape_mesh(-1, xform=body_xform(0.0, 0.0, layout.board_z), mesh=stencil_mesh(layout), cfg=sheet,
                         label="stencil")  # fmt: skip
        for label, pos, half in box_shapes(layout):
            t.add_shape_box(-1, xform=body_xform(*pos), hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label=label)
        t.add_shape_box(-1, xform=body_xform(base[0], base[1], 0.5 * base[2]), hx=0.12, hy=0.12, hz=0.5 * base[2],
                        cfg=cfg, label="pedestal")  # fmt: skip

    dt = sub_dt(can)
    return arm_scene(ArmScene(can_xml(), (*base, *base_quat()), rig(base), statics, q0, dt, dt / 8), init)


class Fr3Can:
    """:func:`~alloy.tasks.paint_spray_supply.run_supply`'s driver for the FR3: the plan's joint angles, led by the
    servos' lag, written to the arm's position servos each sub-period; the valve open over the rows; the mist laid
    under the nozzle's place. Its report: the plan, and how far the nozzle strayed from its path while spraying."""

    can = CAN

    def __init__(self, layout: SprayLayout, can: Can, base: tuple[float, float, float] = CAN_BASE) -> None:
        self.layout, self.spec, self.base = layout, can, base
        self.plan = plan_can(layout, can, base)
        if not self.plan.ok:
            raise RuntimeError(f"the FR3 cannot follow the can's rows: {self.plan.report()}")
        self.steps = self.plan.steps
        self._index = {id(st): k for k, st in enumerate(self.steps)}
        self.targets = np.array([st.q for st in self.steps])
        self.act, self.body, self.spraying, self.stray = None, -1, False, 0.0

    def scene(self, init, dt: float):  # noqa: ANN001, ANN201
        if abs(dt - sub_dt(self.spec)) > 1e-12:
            raise ValueError(f"the plan's sub-period is {sub_dt(self.spec)} s, the run's {dt} s")
        return can_scene(self.layout, self.spec, init, self.steps[0].q, self.base)

    def bind(self, sim) -> None:  # noqa: ANN001
        m = sim.model
        self.act = rig(self.base).bind(m).act
        lead = servo_lead(m.mujoco.actuator_biasprm.numpy()[self.act[..., :7]]).reshape(-1, 7)[0]
        self.targets = led_targets(np.array([st.q for st in self.steps]), lead, sub_dt(self.spec))
        self.body = [str(lbl) for lbl in m.body_label].index(CAN)

    @staticmethod
    def valve(step: CanStep) -> bool:
        return step.valve

    def _nozzle(self, sim) -> np.ndarray:  # noqa: ANN001
        return sim.state.body_q.numpy()[self.body][:3].astype(np.float64)

    def apply(self, sim, step: CanStep | None) -> None:  # noqa: ANN001
        """The servo targets for ``step``'s sub-period (``None``: hold the last). Books the nozzle's stray from its
        path at the end of each sub-period sprayed."""
        if self.spraying and step is not None:
            self.stray = max(self.stray, float(np.linalg.norm(self._nozzle(sim) - step.nozzle)))
        self.spraying = step is not None and step.valve
        if step is None:
            return
        ctrl = sim.control.mujoco.ctrl.numpy().copy()
        ctrl[self.act[..., :7]] = self.targets[self._index[id(step)]]
        ctrl[self.act[..., 7]] = CAN_R                   # the fingers on the can
        sim.control.mujoco.ctrl.assign(ctrl.astype(np.float32))

    def nozzle_xy(self, sim) -> tuple[float, float]:  # noqa: ANN001
        """Where the mist lands: under the nozzle's place at the sub-period's end. The kinematic can lays it at its
        aim, the sub-period's middle: 1.2 mm behind at 0.19 m/s, against the mist's 75 mm drift."""
        x, y, _ = self._nozzle(sim)
        return (float(x), float(y))

    def report(self) -> dict:
        return {"robot": "fr3", "plan": self.plan.report(), "spray_stray_mm": round(1e3 * self.stray, 2)}


__all__ = ["CAN", "CAN_BASE", "GRIP", "CanStep", "Fr3Can", "Path", "base_quat", "box_clearance", "can_scene", "can_xml",
           "nozzle_path", "nozzle_z", "plan_can", "rig", "sub_dt", "tcp_for"]
