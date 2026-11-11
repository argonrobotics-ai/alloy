# SPDX-License-Identifier: Apache-2.0
"""A fixed FR3 holding the real roller: the tool in the arm's MJCF, the handle's pitch, and a joint plan of the
roller's script (:mod:`alloy.tasks.paint_roller`).

The FR3 stands on a 0.4 m pedestal 0.90 m from the wall, facing it. Every contact of the strip is reachable from
0.65-0.75 m too, but along the middle lane the arm folds there (the hand 0.37 m from its base) and runs its joints
into their limits; from 0.90 m all three lanes track inside them. The roller is Gym-Anything's held tool: its frame
is fixed in the hand at the tool point, a 0.20 m handle along the approach to the axle, and the cover turns on the
axle (``axle``, a hinge). The tool's geoms do not collide: the film's pad carries the cover's contact with the
wall. A velocity servo on the axle turns the cover at the plan's rolling rate: the tray is no chart, so nothing
would turn a free cover in the paint (it loaded half its bins: 28 g, spread 0.5), and on the wall its unturned bins
left a third of the strip unhidden (hide 0.58).

The plan (:func:`plan_roller`) follows the roller's script one sub-period at a time:
- near the tray and the wall (loading, the lanes and their approaches) the tool point tracks the cover's pose by IK,
  each solve seeded by the last; in the tray the handle points straight down, on the wall it pitches from -30° at the
  strip's bottom (coming down to it) to +30° at its top (coming up to it);
- every move of the script is eased (:func:`ease_corners`): the cover starts it from rest and stops at its end, on
  the same line, its speed on minimum-jerk ramps (1.5 m/s² near the wall, 2.5 over the tray). A kinematic cover
  changes velocity in one sub-period: at the wall's reversals the arm followed it 6.8 mm late and up to 1.8 mm off
  the wall, past the 1.5 mm press, and at the landings, lifts, dips and transfers' joins the joints took up to
  108 rad/s² (the FR3's limit: 10). Eased, the whole plan stays inside the FR3's speed, acceleration and jerk limits;
- a transfer between the tray and the wall takes its own path: back off the wall by 0.20 m along the handle (IK,
  at the lane's pitch, from rest to rest), then the joints move on a minimum-jerk path (at 0.85 of each joint's
  speed limit, at most 8 rad/s²) straight to the other end's solution (in to the wall the same way), or through a
  solution over ``VIA`` when the straight move would bring the cover within 50 mm of the wall or the tray.
  Interpolating the joints from the wall itself swung the cover 114 mm into it; tracking the cover's orientation
  across the room ran the arm into its joint limits. The arm's transfers set the time between lanes, and a lane's
  thin edge that dries past the wet edge before the next lane overlaps it leaves a lap: eased at 1.5 m/s² everywhere
  with 1 rad/s transfers, 45.8 s between lanes left 1.12 cm² of laps; now 43.2 s, 0.64 cm²;
- each contact segment starts from the IK solution (of up to 8 distinct ones) whose track keeps the widest joint
  margin: the arm's spare degree of freedom let a seed good at a lane's bottom drift joint 6 into its limit at the top.

In the arm's scene the film exchanges at every MuJoCo step, 8 a sub-period (the kinematic cover's scene: once).
The stores' nip splits a cell once per passage. Taken across each exchange's step, at 0.39 mm a step, the arm's 0.1°
changes of heading skipped or re-split cells toward the cover's ends: one-row stripes at 0-17 µm dry in the full
wall's outer lanes (hide 0.975, laps 2.6 cm²). The stores now take a roller's crossing across its axis
(SCR-SOFT-3-4), and the full wall passes as the kinematic cover does.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

import numpy as np

from alloy.tasks import paint_roller as pr
from alloy.tasks.paint_arm import (FR3_SPEED, SPEED_SHARE, ArmScene, Plan, Tracker, arm_scene, ik_starts,
                                     led_targets, ramped_path, servo_lead, smooth)
from alloy.tasks.paint_scene import TRAY_SIZE, WallLayout

BASE = (0.0, -0.90, 0.40)       # m: the arm's base on its pedestal, facing the wall (+y)
YAW = 90.0                      # deg
HANDLE = 0.20                   # m: the tool point to the cover's axle, along the approach
FRAME_MASS, COVER_MASS = 0.34, 0.06     # kg (Gym-Anything: Sherlock R017-9 frame, the cover an estimate)
PITCH_MID, PITCH_SPAN = 0.756, math.radians(30.0)   # m (the strip's middle), and the pitch at its lines
AXLE_KV = 0.005                 # N m s/rad: the axle's servo (the cover's spin follows in ~6 ms)


def _tcp():
    from alloy.arms.specs import FRANKA_TCP  # noqa: PLC0415

    return FRANKA_TCP


def roller_xml() -> str:
    """The FR3's MJCF with the roller in its hand: ``roller_frame`` at the tool point, ``cover`` on the hinge
    ``axle`` along the hand's x, ``HANDLE`` out along the approach (the hand's z)."""
    from alloy.arms import FR3  # noqa: PLC0415
    from alloy.arms.mjcf import robot_xml  # noqa: PLC0415

    root = ET.fromstring(robot_xml(FR3, "", False))
    hand = next(b for b in root.iter("body") if b.get("name") == "hand")
    frame = ET.SubElement(hand, "body", name="roller_frame", pos=f"0 0 {_tcp()[2]}")
    ET.SubElement(frame, "inertial", pos=f"0 0 {0.5 * HANDLE}", mass=str(FRAME_MASS),
                  diaginertia=f"{FRAME_MASS * HANDLE ** 2 / 12} {FRAME_MASS * HANDLE ** 2 / 12} 5e-5")  # fmt: skip
    ET.SubElement(frame, "geom", type="capsule", fromto=f"0 0 0 0 0 {HANDLE}", size="0.008", contype="0",
                  conaffinity="0")  # fmt: skip
    cover = ET.SubElement(frame, "body", name="cover", pos=f"0 0 {HANDLE}")
    ET.SubElement(cover, "joint", name="axle", type="hinge", axis="1 0 0", damping="1e-4")
    spin, tip = 0.5 * COVER_MASS * pr.R ** 2, COVER_MASS * (3 * pr.R ** 2 + (2 * pr.HW) ** 2) / 12
    ET.SubElement(cover, "inertial", pos="0 0 0", mass=str(COVER_MASS), diaginertia=f"{spin} {tip} {tip}")
    ET.SubElement(cover, "geom", type="cylinder", size=f"{pr.R} {pr.HW}", quat="0.7071068 0 0.7071068 0",
                  contype="0", conaffinity="0")  # fmt: skip
    acts = root.find("actuator")
    if acts is None:
        acts = ET.SubElement(root, "actuator")
    ET.SubElement(acts, "velocity", name="axle_servo", joint="axle", kv=str(AXLE_KV), ctrllimited="false")
    return ET.tostring(root, encoding="unicode")


def base_quat() -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(math.radians(0.5 * YAW)), math.cos(math.radians(0.5 * YAW)))


def wall_pitch(z: float) -> float:
    """The handle's pitch [rad] about the cover's axis at a wall contact ``z`` m high: from above at the strip's
    bottom, from below at its top (the reachable bands: -80..+20° at the bottom, -20..+80° at the top)."""
    return float(np.clip((z - PITCH_MID) / 0.5 * PITCH_SPAN, -math.radians(60.0), math.radians(60.0)))


def pitch_of(pose: pr.Pose) -> float:
    """Down in and over the tray; the wall's pitch near the wall (within 0.10 m of it)."""
    return wall_pitch(pose.pos[2]) if pose.pos[1] > -0.10 else -0.5 * math.pi


def tcp_for(centre, pitch: float) -> np.ndarray:  # noqa: ANN001
    """The tool point's pose (x, y, z, qx, qy, qz, qw) that puts the cover's axis centre at ``centre``, its axis along
    world x, the approach (hand z, the tool point toward the axle) pitched ``pitch`` from +y toward +z."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    tcp = _tcp()
    d = np.array([0.0, math.cos(pitch), math.sin(pitch)])
    x = np.array([1.0, 0.0, 0.0])
    hand = Rotation.from_matrix(np.stack([x, np.cross(d, x), d], axis=1))
    origin = np.asarray(centre, np.float64) - (tcp[2] + HANDLE) * d
    return np.concatenate([origin + hand.apply(tcp[:3]), (hand * Rotation.from_quat(tcp[3:])).as_quat()])


def rig():  # noqa: ANN201 - alloy.arms.Rig
    from alloy.arms import FR3, Rig  # noqa: PLC0415

    return Rig.single(FR3, (*BASE, *base_quat()), name="arm", visuals=False)


PULL = 0.20                     # m: a transfer first pulls the cover back this far along the handle
VIA = (0.15, -0.40, 0.75)       # m: transfers pass the cover over this point, in front of the arm
VIA_PITCH = -math.radians(45.0)
TRANSFER_SPEED = 0.3            # m/s of the cover in the air
DIRECT_CLEAR = 0.05             # m: a transfer moves the joints straight across when the cover stays this clear
RAMP_ACCEL = 1.5                # m/s²: the cover's peak acceleration easing into and out of each move near the wall
TRAY_ACCEL = 2.5                # m/s²: and over the tray (the joints there: under 8 rad/s²)
TRANSFER_ACCEL = 2.0            # m/s²: its peak acceleration on a transfer's pull-back (in the air)


@dataclass
class Step:
    """One sub-period of the robot's run: the joint angles, the cover's planned pose (None in a transfer, in the air)
    and the handle's pitch."""

    q: np.ndarray
    pose: pr.Pose | None
    pitch: float


def _segments(poses: list[pr.Pose]) -> list[tuple[bool, int, int]]:
    """Runs of poses (transfer?, first, last + 1)."""
    out, start = [], 0
    for i in range(1, len(poses) + 1):
        if i == len(poses) or poses[i].transfer != poses[start].transfer:
            out.append((poses[start].transfer, start, i))
            start = i
    return out


def _line(a: np.ndarray, b: np.ndarray, speed: float = TRANSFER_SPEED) -> list[np.ndarray]:
    """Points from ``a`` to ``b`` a sub-period apart (``a`` not included, the last is ``b``): from rest to rest, at
    ``speed`` between minimum-jerk ramps at ``TRANSFER_ACCEL``. A transfer's pull-back at a steady 0.3 m/s met the
    joints' move from rest in one sub-period (75 rad/s²)."""
    return ramped_path(np.array([a, b]), speed, pr.SUB, TRANSFER_ACCEL)[1:] + [b]


def _pull(tr: Tracker, start: np.ndarray, off: np.ndarray, pitch: float, seed: np.ndarray) -> list[np.ndarray]:
    """Joint vectors backing the cover from ``start`` toward ``off`` (along the handle, at ``pitch``) from rest to rest,
    by IK from ``seed``: as far as the arm follows the line inside its limits. A line cut short is ramped again to its
    reach, so the leg still ends at rest (cut mid-ramp, the joints stopped in one sub-period: 72 rad/s²)."""
    end = off
    legs: list[np.ndarray] = []
    for _ in range(3):
        pts = _line(start, end)
        legs = tr.track([tcp_for(p, pitch) for p in pts], seed[None, None], True)
        if len(legs) == len(pts) or not legs:
            return legs
        end = pts[len(legs) - 1]
    return legs


def _cover_fk(rigged):  # noqa: ANN001, ANN202
    """``q -> the cover's axis centre`` (the tool point's forward kinematics, out along the approach)."""
    from scipy.spatial.transform import Rotation  # noqa: PLC0415

    ik, tcp = rigged.make_ik(1, device="cpu"), _tcp()
    jaw = Rotation.from_quat(tcp[3:]).inv()

    def cover(q: np.ndarray) -> np.ndarray:
        pose = ik.fk(q[None, None])[0, 0]
        hand = Rotation.from_quat(pose[3:]) * jaw
        return pose[:3] - hand.apply(tcp[:3]) + (tcp[2] + HANDLE) * hand.apply((0.0, 0.0, 1.0))

    return cover


def _clearance(layout: WallLayout, cover: np.ndarray) -> float:
    """How far the cover's axis centre is from touching the wall or the tray's rim (negative: it would)."""
    wall = layout.face_y - cover[1] - pr.R
    x, y, z = layout.tray_pos
    over = abs(cover[0] - x) < 0.5 * TRAY_SIZE[1] + pr.HW and abs(cover[1] - y) < 0.5 * TRAY_SIZE[0] + pr.R
    return min(wall, (cover[2] - pr.R - (z + TRAY_SIZE[2])) if over else math.inf)


def _legs(poses: list[pr.Pose]) -> list[list[pr.Pose]]:
    """Runs of poses with one velocity, roll and transfer flag: the script's straight moves, holds and transfers."""
    out: list[list[pr.Pose]] = []
    for p in poses:
        last = out[-1][-1] if out else None
        if last is not None and np.array_equal(last.vel, p.vel) and last.roll == p.roll and \
                last.transfer == p.transfer:
            out[-1].append(p)
        else:
            out.append([p])
    return out


def ease_corners(poses: list[pr.Pose], dt: float = pr.SUB) -> list[pr.Pose]:
    """The script's poses with every move eased: the cover starts each straight move from rest and
    stops at its end, on the same line, its speed up and down on minimum-jerk ramps at ``RAMP_ACCEL``
    (:func:`~alloy.tasks.paint_arm.ramped_path`). Holds and
    transfers (the robot's own path) keep theirs. A kinematic cover changes velocity in one sub-period: at the wall's
    reversals the arm followed it 6.8 mm late and up to 1.8 mm off the wall (past the 1.5 mm press), and the joints
    took up to 108 rad/s² at the landings, lifts, dips and the transfers' joins (the FR3's limit: 10). Each move takes
    ``1.875·V/(2·RAMP_ACCEL)`` longer (0.16 s at the wall's 0.25 m/s). The spins follow the velocities and the angles
    are integrated again."""
    out: list[pr.Pose] = []
    for leg in _legs(poses):
        first, last = leg[0], leg[-1]
        speed = float(np.linalg.norm(first.vel))
        if first.transfer or speed == 0.0:
            out += leg
            continue
        b = last.pos + last.vel * dt
        accel = RAMP_ACCEL if first.pos[1] > -0.10 else TRAY_ACCEL            # near the wall: its pitch (pitch_of)
        pts = ramped_path(np.array([first.pos, b]), speed, dt, accel) + [b]
        for p, q in zip(pts, pts[1:]):
            vel = (q - p) / dt
            spin = 0.0 if first.roll is None else -float(vel @ np.cross((1.0, 0.0, 0.0), pr.CONTACT[first.roll])) / pr.R
            out.append(pr.Pose(p, vel, 0.0, spin, first.roll, first.transfer))
    angle = poses[0].angle if poses else 0.0
    for k, p in enumerate(out):
        out[k] = pr.Pose(p.pos, p.vel, angle, p.spin, p.roll, p.transfer)
        angle += p.spin * dt
    return out


def plan_roller(layout: WallLayout, poses: list[pr.Pose]) -> Plan:
    """The robot's steps over ``poses``. Outside transfers the tool point tracks the cover's pose by IK (each contact
    segment seeded from 64 random starts, then by the last solve). A transfer replaces the script's straight move: a
    wall end first pulls back up to ``PULL`` along the handle (IK, as far as the arm follows it
    inside its limits), then the joints move smoothly between the two ends'
    solutions through a solution at ``VIA`` (:func:`~alloy.tasks.paint_arm.smooth`), the one nearest both ends."""
    poses = ease_corners(poses)
    plan = Plan()
    tr = Tracker(rig(), plan)
    vias = [q[0, 0] for q in ik_starts(tr.rigged, tcp_for(VIA, VIA_PITCH)[None])]
    cover = _cover_fk(tr.rigged)
    segs = _segments(poses)
    solved: dict[int, list[np.ndarray]] = {}
    for k, (transfer, a, b) in enumerate(segs):           # the contact segments, each from a fresh seed
        if not transfer:
            pts = [tcp_for(p.pos, pitch_of(p)) for p in poses[a:b]]
            solved[k] = tr.track(pts, tr.start(pts))
    for k, (transfer, a, b) in enumerate(segs):
        if not transfer:
            plan.steps += [Step(q, p, pitch_of(p)) for q, p in zip(solved[k], poses[a:b])]
            continue
        before, after = solved.get(k - 1), solved.get(k + 1)
        ends = []
        for side, pose, qs in (("from", poses[a - 1] if a else None, before), ("to", poses[b] if b < len(poses)
                                                                                  else None, after)):  # fmt: skip
            if pose is None or qs is None:
                ends.append(([], None))
                continue
            ph, at_wall = pitch_of(pose), pose.pos[1] > layout.face_y - 0.15
            back_along = np.array([0.0, math.cos(ph), math.sin(ph)])         # back along the handle
            off = pose.pos - PULL * back_along if at_wall else None
            if side == "from":
                legs = _pull(tr, pose.pos, off, ph, qs[-1]) if at_wall else []
                ends.append((legs, legs[-1] if legs else qs[-1]))
            else:
                back = _pull(tr, pose.pos, off, ph, qs[0]) if at_wall else []
                ends.append((back[::-1], back[-1] if back else qs[0]))
        (leave, qa), (arrive, qb) = ends
        qa = qa if qa is not None else qb
        qb = qb if qb is not None else qa
        across = smooth(qa, qb, pr.SUB, math.inf, peak=SPEED_SHARE * FR3_SPEED)
        if min(_clearance(layout, cover(q)) for q in across[:: max(1, len(across) // 40)]) < DIRECT_CLEAR:
            via = min(vias, key=lambda v: float(np.abs(v - qa).max() + np.abs(v - qb).max()))
            across = (smooth(qa, via, pr.SUB, math.inf, peak=SPEED_SHARE * FR3_SPEED) +
                      smooth(via, qb, pr.SUB, math.inf, peak=SPEED_SHARE * FR3_SPEED))
        for q in leave + across + arrive:
            plan.steps.append(Step(q, None, float("nan")))
        for q in across[:: max(1, len(across) // 20)]:
            plan.clearance = min(plan.clearance, _clearance(layout, cover(q)))
    return plan.finish(pr.SUB)


COVER = "sc6_fr3/worldbody/base/fr3_link0/fr3_link1/fr3_link2/fr3_link3/fr3_link4/fr3_link5/fr3_link6/fr3_link7/" \
        "hand/roller_frame/cover"     # the cover's body label in the arm's MJCF
DT = pr.SUB / 8                       # s: MuJoCo's step (eight a sub-period)


def fr3_scene(layout: WallLayout, init, q0: np.ndarray):  # noqa: ANN001, ANN201 - a harness TestScene
    """The FR3 with the roller on its pedestal, the wall, the tray: MuJoCo, gravity compensated (the tool too), the
    arm at ``q0``. The tool's geoms do not collide; the arm's do, with the wall, the tray and the pedestal."""

    def statics(t, cfg) -> None:  # noqa: ANN001 - newton.ModelBuilder, ShapeConfig
        import warp as wp  # noqa: PLC0415

        from alloy.tasks.paint_scene import wall_box  # noqa: PLC0415

        label, pos, quat, half = wall_box(layout)
        t.add_shape_box(-1, xform=wp.transform(wp.vec3(*pos), wp.quat(*quat)), hx=half[0], hy=half[1], hz=half[2],
                        cfg=cfg, label=label)  # fmt: skip
        t.add_shape_box(-1, xform=wp.transform((BASE[0], BASE[1], 0.5 * BASE[2]), wp.quat_identity()), hx=0.12,
                        hy=0.12, hz=0.5 * BASE[2], cfg=cfg, label="pedestal")  # fmt: skip
        x, y, z = layout.tray_pos
        t.add_shape_box(-1, xform=wp.transform((x, y, z + 0.5 * TRAY_SIZE[2]), wp.quat_identity()),
                        hx=0.5 * TRAY_SIZE[1], hy=0.5 * TRAY_SIZE[0], hz=0.5 * TRAY_SIZE[2], cfg=cfg, label="tray")

    return arm_scene(ArmScene(roller_xml(), (*BASE, *base_quat()), rig(), statics, q0, pr.SUB, DT), init)


class Fr3Roller:
    """``run_roller``'s driver for the FR3: the plan's joint angles written to the arm's position servos each
    sub-period, led by the servo's lag; the cover's planned pose for the film's bookkeeping.

    Each sub-period's target is the plan's angle at its middle plus the servo's lag at the plan's rate
    (:func:`alloy.tasks.paint_arm.led_targets`): the servo then holds the joint on the plan wherever its rate is
    steady. Writing the plan's own angles left the cover 26.7 mm behind its pose on the wall, short of every line's
    end (hide 0.944 on the small wall); led, 4.4 mm at most (hide 0.998). The joints' damping (0.21 N m s) and
    friction (up to 1.1 N m) are left out: at most 0.1 and 0.4 mrad against the servos' 2000-4500 N m/rad."""

    cover = COVER

    def __init__(self, layout: WallLayout, passes: int = 3) -> None:
        self.layout = layout
        start, moves = pr.script(layout, passes)
        self.plan = plan_roller(layout, list(pr.poses(start, moves)))
        if not self.plan.ok:
            raise RuntimeError(f"the FR3 cannot follow the roller's script: {self.plan_report()}")
        self.steps = self.plan.steps
        self._index = {id(st): k for k, st in enumerate(self.steps)}
        self.act = None
        self.targets = np.array([st.q for st in self.steps])
        self.body = -1

    def plan_report(self) -> dict:
        return self.plan.report()

    def scene(self, init):  # noqa: ANN001, ANN201
        return fr3_scene(self.layout, init, self.steps[0].q)

    def bind(self, sim) -> None:  # noqa: ANN001
        m = sim.model
        self.act = rig().bind(m).act
        lead = servo_lead(m.mujoco.actuator_biasprm.numpy()[self.act[..., :7]]).reshape(-1, 7)[0]
        self.targets = led_targets(np.array([st.q for st in self.steps]), lead, pr.SUB)
        self.body = [str(lbl) for lbl in m.body_label].index(COVER)
        axles = [j for j, lbl in enumerate(m.joint_label) if str(lbl).endswith("/axle")]
        by_dof = {int(d): a for a, d in enumerate(m.mujoco.actuator_trnid.numpy()[:, 0])}
        self.axle = np.array([by_dof[int(m.joint_qd_start.numpy()[j])] for j in axles])

    def apply(self, sim, step: Step) -> None:  # noqa: ANN001
        ctrl = sim.control.mujoco.ctrl.numpy().copy()
        ctrl[self.act[..., :7]] = self.targets[self._index[id(step)]]
        ctrl[self.act[..., 7]] = 0.0                    # the fingers closed on the handle
        ctrl[self.axle] = 0.0 if step.pose is None else step.pose.spin      # rolling without slip (the servo)
        sim.control.mujoco.ctrl.assign(ctrl.astype(np.float32))

    @staticmethod
    def pose(step: Step) -> pr.Pose | None:
        return step.pose

    def cover_pos(self, sim) -> np.ndarray:  # noqa: ANN001
        return sim.state.body_q.numpy()[self.body][:3].astype(np.float64)

    def rolling(self, sim, step: Step) -> pr.Pose | None:  # noqa: ANN001
        """The pose the cover rolls with this sub-period, at its place (the arm's lag moves it up to 2.3 mm from the
        plan): on the wall only while it presses it. For the film's age (where it was last rolled)."""
        if step.pose is None:
            return None
        pos = self.cover_pos(sim)
        on = step.pose.roll != "wall" or self.layout.face_y - pos[1] < pr.R
        return replace(step.pose, pos=pos, roll=step.pose.roll if on else None)

    def observe(self, sim, step: Step) -> None:  # noqa: ANN001
        """After a sub-period on the wall: how far the cover is from where the plan put it (the arm's lag), and how
        much further from the wall (the press it lost: the plan presses 1.5 mm)."""
        if step.pose is not None and step.pose.roll == "wall":
            planned = step.pose.pos + step.pose.vel * pr.SUB
            off = self.cover_pos(sim) - planned
            self.lag = max(getattr(self, "lag", 0.0), float(np.linalg.norm(off)))
            self.off_wall = max(getattr(self, "off_wall", 0.0), float(-off @ np.asarray(pr.CONTACT["wall"])))

    def report(self) -> dict:
        return {"robot": "fr3", "plan": self.plan_report(), "wall_lag_mm": round(1e3 * getattr(self, "lag", 0.0), 2),
                "off_wall_mm": round(1e3 * getattr(self, "off_wall", 0.0), 2)}


__all__ = ["BASE", "COVER", "Fr3Roller", "HANDLE", "PULL", "Step", "VIA", "base_quat", "fr3_scene",
           "pitch_of", "plan_roller", "rig", "roller_xml", "tcp_for", "wall_pitch"]
