# SPDX-License-Identifier: Apache-2.0
"""brush-stroke with a fixed FR3 holding the round brush (:mod:`alloy.tasks.paint_brush`).

The FR3 stands on a pedestal beside the drawing board, facing it. It holds the brush straight down by its handle,
the tip ``GRIP`` below the tool point, the fingers across the handle. The brush is level, so the plan only moves its
tip: through the script of :func:`alloy.tasks.paint_brush_stroke.brush_script` (the dip, the hold over the well, the
landing, the S at the template's press, the lift), one tip point a control period, the tool point tracking it by IK
from the start whose track keeps the widest joint margin. The servo targets are led by the servos' lag
(:func:`alloy.tasks.paint_arm.led_targets`). The brush's geoms do not collide; the store's pad carries the tip's
contact with the paper.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from alloy.tasks import paint_brush as pb
from alloy.tasks.paint_arm import (ArmScene, Plan, Tracker, arm_scene, down_tcp, led_targets, ramped_path,
                                   servo_lead, static_clearance)
from alloy.tasks.paint_brush_stroke import BOARD_Z, PAPER, BrushStep, brush_script

BRUSH_BASE = (0.0, -0.45, 0.65)     # m: the arm's base on its pedestal, facing the board (+y)
BRUSH_YAW = 90.0                    # deg
GRIP = 0.10                         # m: the tip below the tool point (hair 22.5 mm, ferrule 25 mm, then handle)
HANDLE_R = 0.0045                   # m (Gym-Anything's round #8)
BRUSH_MASS = 0.010                  # kg (Gym-Anything's round #8)
HANDLE_ABOVE = 0.10                 # m of handle above the tool point
SUBSTEPS = 16                       # MuJoCo steps a control period
EASE_ACCEL = 1.0                    # m/s²: the tip's peak acceleration easing into and out of each move and the S
TIP = "sc6_fr3/worldbody/base/fr3_link0/fr3_link1/fr3_link2/fr3_link3/fr3_link4/fr3_link5/fr3_link6/fr3_link7/" \
      "hand/brush"                  # the brush's body label in the arm's MJCF


def _tcp_z() -> float:
    from alloy.arms.specs import FRANKA_TCP  # noqa: PLC0415

    return FRANKA_TCP[2]


def brush_xml() -> str:
    """The FR3's MJCF with the brush in its hand: body ``brush`` at the tip's point, ``GRIP`` past the tool point
    along the approach, turned so its z points back up the approach (the applicator's normal, its -z, down)."""
    from alloy.arms import FR3  # noqa: PLC0415
    from alloy.arms.mjcf import robot_xml  # noqa: PLC0415

    root = ET.fromstring(robot_xml(FR3, "", False))
    hand = next(b for b in root.iter("body") if b.get("name") == "hand")
    brush = ET.SubElement(hand, "body", name="brush", pos=f"0 0 {_tcp_z() + GRIP}", quat="0 1 0 0")
    length = GRIP + HANDLE_ABOVE
    side, axial = BRUSH_MASS * length ** 2 / 12, 0.5 * BRUSH_MASS * HANDLE_R ** 2
    ET.SubElement(brush, "inertial", pos=f"0 0 {0.5 * length}", mass=str(BRUSH_MASS),
                  diaginertia=f"{side} {side} {axial}")  # fmt: skip
    ET.SubElement(brush, "geom", type="capsule", fromto=f"0 0 {HANDLE_R} 0 0 {length}", size=str(HANDLE_R),
                  contype="0", conaffinity="0")  # fmt: skip
    return ET.tostring(root, encoding="unicode")


def base_quat(yaw: float = BRUSH_YAW) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(math.radians(0.5 * yaw)), math.cos(math.radians(0.5 * yaw)))


def rig(base: tuple[float, float, float] = BRUSH_BASE):  # noqa: ANN201 - alloy.arms.Rig
    from alloy.arms import FR3, Rig  # noqa: PLC0415

    return Rig.single(FR3, (*base, *base_quat()), name="arm", visuals=False)


def board_boxes() -> list:
    """The drawing board under the paper (label, centre, half extents), as the kinematic scene's."""
    return [("paper", (0.0, 0.0, BOARD_Z - 0.005), (0.5 * PAPER[0], 0.5 * PAPER[1], 0.005))]


@dataclass
class BrushArmStep:
    """One control period of the arm's run: the joint angles at its start, the tip's planned pose, painting or not."""

    q: np.ndarray
    pose: pb.BrushPose
    stroke: bool


def _brush_legs(script: list[BrushStep]) -> list[list[int]]:
    """Runs of step indices: the S (painting), and outside it each straight move or hold (one displacement a period)."""
    pos = [st.pose.pos for st in script]

    def step(k: int) -> np.ndarray:
        return pos[k] - pos[k - 1] if k else np.zeros(3)

    out: list[list[int]] = []
    for k, st in enumerate(script):
        if out:
            j = out[-1][-1]
            if script[j].stroke == st.stroke and (st.stroke or np.allclose(step(k), step(j), rtol=0.0, atol=1e-12)):
                out[-1].append(k)
                continue
        out.append([k])
    return out


def ease_script(script: list[BrushStep]) -> list[BrushStep]:
    """The script with every move eased: the tip starts each straight move and the S from rest and stops at their
    ends, on the same path, its speed up and down on minimum-jerk ramps at ``EASE_ACCEL``
    (:func:`~alloy.tasks.paint_arm.ramped_path`). Holds keep their steps. A kinematic tip changes velocity in one
    period: at the dip, the landing and the lift the joints took up to 21.3 rad/s² (the FR3's limit: 10). The S's
    first and last 2.3 mm slide slower, and meter less (the law goes as the square root of the speed)."""
    out: list[BrushStep] = []
    for leg in _brush_legs(script):
        first = script[leg[0]]
        before = script[leg[0] - 1].pose.pos if leg[0] else first.pose.pos
        path = np.array([before] + [script[k].pose.pos for k in leg])
        step = np.linalg.norm(np.diff(path, axis=0), axis=1)
        if not first.stroke and step.max() == 0.0:
            out += [script[k] for k in leg]
            continue
        speed = pb.SPEED if first.stroke else float(step[-1]) / pb.DT
        pts = ramped_path(path, speed, pb.DT, EASE_ACCEL)[1:] + [path[-1]]
        prev = before
        for q in pts:
            out.append(BrushStep(pb.BrushPose(q, pb.LEVEL, (q - prev) / pb.DT), first.stroke))
            prev = q
    return out


def plan_brush(script: list[BrushStep], base: tuple[float, float, float] = BRUSH_BASE) -> Plan:
    """The arm's steps over ``script``: the tool point tracks the tip's points by IK; the clearance is the arm's least
    distance to the board (every 10th step)."""
    plan = Plan()
    tr = Tracker(rig(base), plan)
    targets = [down_tcp(st.pose.pos, GRIP) for st in script]
    qs = tr.track(targets, tr.start(targets))
    plan.steps = [BrushArmStep(q, st.pose, st.stroke) for q, st in zip(qs, script, strict=True)]
    plan.clearance = static_clearance(brush_xml(), (*base, *base_quat()), board_boxes(), qs[::10])
    return plan.finish(pb.DT)


def brush_arm_scene(init, q0: np.ndarray, base: tuple[float, float, float] = BRUSH_BASE):  # noqa: ANN001, ANN201
    """The FR3 holding the brush on its pedestal, and the paper's board: MuJoCo, ``SUBSTEPS`` steps a control period,
    the arm at ``q0``. The brush does not collide; the arm does, with the board and the pedestal."""

    def statics(t, cfg) -> None:  # noqa: ANN001 - newton.ModelBuilder, ShapeConfig
        from alloy.plugins.harness.scene import body_xform  # noqa: PLC0415

        for label, pos, half in board_boxes():
            t.add_shape_box(-1, xform=body_xform(*pos), hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label=label)
        t.add_shape_box(-1, xform=body_xform(base[0], base[1], 0.5 * base[2]), hx=0.12, hy=0.12, hz=0.5 * base[2],
                        cfg=cfg, label="pedestal")  # fmt: skip

    return arm_scene(ArmScene(brush_xml(), (*base, *base_quat()), rig(base), statics, q0, pb.DT, pb.DT / SUBSTEPS),
                     init)


class Fr3Brush:
    """:func:`~alloy.tasks.paint_brush_stroke.run_brush`'s driver for the FR3: the plan's joint angles, led by the
    servos' lag, written to the arm's position servos each control period. Its report: the plan, and how far the tip
    strayed from its path while painting (across, and in height: the press)."""

    tool = TIP

    def __init__(self, loaded: bool = False, base: tuple[float, float, float] = BRUSH_BASE) -> None:
        self.base = base
        self.plan = plan_brush(ease_script(brush_script(loaded)), base)
        if not self.plan.ok:
            raise RuntimeError(f"the FR3 cannot follow the brush's script: {self.plan.report()}")
        self.steps = self.plan.steps
        self._index = {id(st): k for k, st in enumerate(self.steps)}
        self.targets = np.array([st.q for st in self.steps])
        self.act, self.body, self.painting = None, -1, False
        self.stray, self.press = 0.0, 0.0

    def scene(self, init):  # noqa: ANN001, ANN201
        return brush_arm_scene(init, self.steps[0].q, self.base)

    def bind(self, sim) -> None:  # noqa: ANN001
        m = sim.model
        self.act = rig(self.base).bind(m).act
        lead = servo_lead(m.mujoco.actuator_biasprm.numpy()[self.act[..., :7]]).reshape(-1, 7)[0]
        self.targets = led_targets(np.array([st.q for st in self.steps]), lead, pb.DT)
        self.body = [str(lbl) for lbl in m.body_label].index(TIP)

    def apply(self, sim, step: BrushArmStep | None) -> None:  # noqa: ANN001
        """The servo targets for ``step``'s period (``None``: hold the last). Books the tip's stray from its path at
        the end of each period painted."""
        if self.painting and step is not None:
            off = sim.state.body_q.numpy()[self.body][:3].astype(np.float64) - step.pose.pos
            self.stray = max(self.stray, float(np.linalg.norm(off[:2])))
            self.press = max(self.press, abs(float(off[2])))
        self.painting = step is not None and step.stroke
        if step is None:
            return
        ctrl = sim.control.mujoco.ctrl.numpy().copy()
        ctrl[self.act[..., :7]] = self.targets[self._index[id(step)]]
        ctrl[self.act[..., 7]] = HANDLE_R                   # the fingers on the handle
        sim.control.mujoco.ctrl.assign(ctrl.astype(np.float32))

    def report(self) -> dict:
        return {"robot": "fr3", "plan": self.plan.report(), "tip_stray_mm": round(1e3 * self.stray, 3),
                "press_error_mm": round(1e3 * self.press, 3)}


__all__ = ["BRUSH_BASE", "EASE_ACCEL", "GRIP", "TIP", "BrushArmStep", "Fr3Brush", "base_quat", "board_boxes",
           "brush_arm_scene", "brush_xml", "ease_script", "plan_brush", "rig"]
