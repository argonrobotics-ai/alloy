# SPDX-License-Identifier: Apache-2.0
"""P3-2b robot loaders: each returns the robot's ``MjSpec`` from its pinned source file and its start pose.

Changes to the source robots:
- Visual-only geoms (contype = conaffinity = 0) and their meshes are dropped: Alloy and the packs never use them.
- Franka Hand: its fixed tendon ``split`` and tendon actuator become a joint position actuator on ``finger_joint1``,
  with the same force law. The tendon was 0.5 (j1 + j2) with j1 = j2 by the hand's equality, so it equals j1. The
  actuator's kp is 100 N/m, kv 10 N s/m, force ±100 N, ctrl 0-0.04 m (the source remaps 0-0.04 m to 0-255).
  The equality is kept. Alloy's converter and the rival loaders need no tendons this way.
- Arms (FR3, Panda, UR10e + 2F-85): every robot geom's contact is firm, and so are the table, floor and props of the
  pack's world (:mod:`alloy.tasks.emb_firm`); the sources keep MuJoCo's default 0.02 s, and a 15 g Franka finger
  pressed by the arm rested 5.5-10.9 mm deep (TRAIN-FIX 1b, WP3's root cause).
- LEAP hand: MuJoCo Playground's joint and servo dynamics (``PLAYGROUND_LEAP``, ``playground_dynamics``); menagerie
  ships damping 0.03, armature 0, frictionloss 0.001 and no force limit, and its fingers fling objects (TA-4 F3).
- Options: the sparse Jacobian (WP5's ordered, repeatable MuJoCo Warp route needs it; it does not change the
  physics). Multi-CCD stays on, MuJoCo's default; :mod:`alloy.tasks.emb_scene` writes it out explicitly.
- Source keyframes are dropped. One keyframe ``init`` is added: the robot at its start pose with ctrl = the
  start joint targets, and the objects at rest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from alloy.core.errors import TraceError
from alloy.tasks.emb_specs import Embodiment

log = logging.getLogger("alloy.tasks")


@dataclass(frozen=True)
class StartPose:
    qpos: dict[str, float]          # joint name -> start value (hinge / slide joints of the robot)
    ctrl: dict[str, float]          # actuator name -> start command


def _visual_only(g: mujoco.MjsGeom) -> bool:
    return g.contype == 0 and g.conaffinity == 0


def strip_visuals(spec: mujoco.MjSpec) -> None:
    """Drop visual-only geoms, the meshes, materials and textures nothing uses any more, and the source keyframes."""
    for g in [g for g in spec.geoms if _visual_only(g)]:
        spec.delete(g)
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for mesh in [m for m in spec.meshes if m.name not in used]:
        spec.delete(mesh)
    materials = {g.material for g in spec.geoms} | {s.material for s in spec.sites}
    for mat in [m for m in spec.materials if m.name not in materials]:
        spec.delete(mat)
    textures = {t for m in spec.materials for t in m.textures if t}
    for tex in [t for t in spec.textures if t.name not in textures]:
        spec.delete(tex)
    for key in list(spec.keys):
        spec.delete(key)


def _position_actuator(spec: mujoco.MjSpec, name: str, joint: str, kp: float, kv: float,
                       ctrl: tuple[float, float], force: tuple[float, float]) -> None:  # fmt: skip
    act = spec.add_actuator(name=name, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.gaintype, act.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
    act.gainprm[0], act.biasprm[1], act.biasprm[2] = kp, -kp, -kv
    act.ctrlrange, act.forcerange = list(ctrl), list(force)
    act.ctrllimited = act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE


@dataclass(frozen=True)
class GripperDrive:
    """A gripper whose fixed tendon ``split`` = 0.5 (j1 + j2) and an equality j1 = j2: the tendon equals j1, so the
    tendon actuator becomes this joint position actuator on j1 with the same force law (module doc)."""
    name: str
    joint: str
    kp: float
    kv: float
    ctrl: tuple[float, float]
    force: tuple[float, float]


FRANKA_HAND = GripperDrive("hand_finger", "finger_joint1", 100.0, 10.0, (0.0, 0.04), (-100.0, 100.0))
ROBOTIQ_2F85 = GripperDrive("fingers_actuator", "right_driver_joint", 100.0, 10.0, (0.0, 0.8), (-5.0, 5.0))


def tendon_to_joint(spec: mujoco.MjSpec, drive: GripperDrive) -> None:
    for act in [a for a in spec.actuators if a.trntype == mujoco.mjtTrn.mjTRN_TENDON]:
        spec.delete(act)
    for tendon in list(spec.tendons):
        spec.delete(tendon)
    _position_actuator(spec, drive.name, drive.joint, drive.kp, drive.kv, drive.ctrl, drive.force)


def franka_tendon_to_joint(spec: mujoco.MjSpec) -> None:
    tendon_to_joint(spec, FRANKA_HAND)


def franka_hand(path: Path) -> mujoco.MjSpec:
    """menagerie's Franka Hand, tendon actuator as a joint actuator."""
    hand = mujoco.MjSpec.from_file(str(path))
    franka_tendon_to_joint(hand)
    strip_visuals(hand)
    return hand


def _fr3_hand_spec(e: Embodiment) -> tuple[mujoco.MjSpec, dict[str, float]]:
    root = e.source_root()
    arm = mujoco.MjSpec.from_file(str(root / e.files[0]))
    home = dict(zip((j.name for j in arm.joints), arm.keys[0].qpos, strict=True))  # fr3.xml "home"
    strip_visuals(arm)
    arm.attach(franka_hand(root / e.files[1]), site=arm.site("attachment_site"), prefix="")
    return arm, home | {"finger_joint1": 0.04, "finger_joint2": 0.04}


def _start(spec: mujoco.MjSpec, qpos: dict[str, float]) -> StartPose:
    """Start commands = the start pose of each actuator's joint."""
    return StartPose(qpos, {a.name: qpos[a.target] for a in spec.actuators})


def fr3_with_hand(e: Embodiment) -> tuple[mujoco.MjSpec, StartPose]:
    arm, qpos = _fr3_hand_spec(e)
    return arm, _start(arm, qpos)


def panda(e: Embodiment) -> tuple[mujoco.MjSpec, StartPose]:
    """menagerie's panda.xml (arm + hand), the hand's tendon actuator as a joint actuator; start = "home"."""
    spec = mujoco.MjSpec.from_file(str(e.source_root() / e.files[0]))
    home = dict(zip((j.name for j in spec.joints), spec.keys[0].qpos, strict=True))
    franka_tendon_to_joint(spec)
    strip_visuals(spec)
    return spec, _start(spec, home)


_OPTION_FIELDS = ("timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance",
                  "gravity", "integrator", "cone", "jacobian", "solver", "iterations", "ls_iterations",
                  "noslip_iterations", "ccd_iterations", "disableflags", "enableflags")  # fmt: skip
BIMANUAL_Y = 0.3                    # [m] arm bases at y = ±0.3 on the table, both facing +x


def fr3_bimanual(e: Embodiment) -> tuple[mujoco.MjSpec, StartPose]:
    """Two FR3 + Franka Hand, prefixed ``left_`` / ``right_``, bases at y = ±0.3 m; options from fr3.xml."""
    root = mujoco.MjSpec()
    qpos: dict[str, float] = {}
    for side, y in (("left_", BIMANUAL_Y), ("right_", -BIMANUAL_Y)):
        arm, home = _fr3_hand_spec(e)
        if side == "left_":
            for name in _OPTION_FIELDS:
                setattr(root.option, name, getattr(arm.option, name))
        root.attach(arm, frame=root.worldbody.add_frame(pos=[0.0, y, 0.0]), prefix=side)
        qpos |= {side + k: v for k, v in home.items()}
    return root, _start(root, qpos)


def g1_with_hands(e: Embodiment) -> tuple[mujoco.MjSpec, StartPose]:
    spec = mujoco.MjSpec.from_file(str(e.source_root() / e.files[0]))
    key = next(k for k in spec.keys if k.name == e.keyframe)
    hinge = [j for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    qpos = dict(zip((j.name for j in hinge), np.asarray(key.qpos)[7:], strict=True))
    ctrl = dict(zip((a.name for a in spec.actuators), np.asarray(key.ctrl), strict=True))
    base = np.asarray(key.qpos)[:7]
    free = next(j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE)
    body = next(b for b in spec.bodies if free.name in {j.name for j in b.joints})
    body.pos, body.quat = base[:3], base[3:]        # stand at the keyframe's pelvis pose (qpos0 of the free joint)
    return spec, StartPose(qpos, ctrl)


def _free_base(spec: mujoco.MjSpec) -> mujoco.MjsBody:
    free = next(j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE)
    return next(b for b in spec.bodies if free.name in {j.name for j in b.joints})


def _keyframe_start(spec: mujoco.MjSpec, key_name: str) -> StartPose:
    """Start pose from a source keyframe of a floating-base robot; the base stands at the key's pose (qpos0)."""
    key = next(k for k in spec.keys if k.name == key_name)
    hinge = [j for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE]
    qpos = dict(zip((j.name for j in hinge), np.asarray(key.qpos)[7:], strict=True))
    base = np.asarray(key.qpos)[:7]
    body = _free_base(spec)
    body.pos, body.quat = base[:3], base[3:]
    return StartPose(qpos, {a.name: qpos[a.target] for a in spec.actuators})


def go2_position(e: Embodiment) -> tuple[mujoco.MjSpec, StartPose]:
    """menagerie's go2.xml (full collision, elliptic cone) with its torque motors replaced by the position actuators
    menagerie ships in go2_mjx.xml: kp 50, kv 0.5 and the MJX ctrl ranges; torque limits are go2.xml's motor limits
    (±23.7 / ±45.43 N m). The MJX file itself is not used whole: its feet are tuned for its pyramidal cone, and on
    Alloy's route (elliptic) they rest 19 mm deep (pyramidal 5.4 mm). go2.xml's own feet rest 13.7 mm deep on a
    rigid floor, so they are made firm (``firm_feet``). Start = "home"."""
    root = e.source_root()
    spec = mujoco.MjSpec.from_file(str(root / e.files[0]))
    mjx = {a.name: a for a in mujoco.MjSpec.from_file(str(root / e.files[1])).actuators}
    motors = [(a.name, a.target, tuple(a.ctrlrange)) for a in spec.actuators]
    for act in list(spec.actuators):
        spec.delete(act)
    for name, joint, torque in motors:
        src = mjx[name]
        _position_actuator(spec, name, joint, float(src.gainprm[0]), float(-src.biasprm[2]), tuple(src.ctrlrange),
                           torque)  # fmt: skip
    firm_feet(spec)
    start = zero_at_start(spec, _keyframe_start(spec, e.keyframe))
    strip_visuals(spec)
    return spec, start


DUO_WHEELS = ("caster_front_left_link", "caster_rear_right_link", "argo_drive_front_link", "argo_drive_rear_link")
HARD_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)     # the packs' hard small parts (2F-85 pads, T14's mug)


def firm_wheels(spec: mujoco.MjSpec) -> list[str]:
    """The Duo's two casters and two drive wheels carry 3.2 kN on a rigid floor without sinking into it. The URDF
    gives no contact parameters, so MuJoCo's defaults applied (solref 0.02 s): the casters, 0.2 kg links under
    0.5-1.1 kN, rested 10.0 mm deep (peak 13.5 mm), the base 3.8 mm low and rolled 1.5 deg, which Alloy's health
    reads as a failed world. The four wheel geoms take ``STIFF_SOLREF`` and ``HARD_SOLIMP`` (hard wheels): they rest
    0.43 mm deep (peak 0.51) on both of Alloy's routes; at ``STIFF_SOLREF`` alone 0.70 mm. Shapes, friction and
    every other geom are unchanged. Returns the geoms changed."""
    from alloy.tasks.emb_specs import STIFF_SOLREF

    wheels = [g for g in spec.geoms if g.parent.name in DUO_WHEELS]
    if len(wheels) != len(DUO_WHEELS):
        raise TraceError(f"expected one collision geom on each of {DUO_WHEELS}, found {[g.name for g in wheels]}")
    for g in wheels:
        g.solref, g.solimp = list(STIFF_SOLREF), list(HARD_SOLIMP)
    return [g.name for g in wheels]


def firm_feet(spec: mujoco.MjSpec) -> list[str]:
    """Go2's four foot balls rest on a rigid floor like rubber, not 13.7 mm inside it. Menagerie's go2.xml gives
    them priority 1 with solimp 0.015 1 0.022: 1.5 % of the contact force at the surface, all of it only 22 mm deep,
    so the 15 kg robot stands 13.7 mm in the floor (peak 15.3 mm), which Alloy's health reads as a failed world.
    Every geom with a solimp softer than MuJoCo's default (those four) takes the default solimp and ``STIFF_SOLREF``,
    as the ``_stiff`` variant's feet already do: they then rest 0.17 mm deep on both of Alloy's routes (default
    solimp at the file's solref 0.02 s still rests 1.28 mm). Shapes, friction, condim, priority and every other
    geom are unchanged. Returns the geoms changed."""
    from alloy.tasks.emb_specs import DEFAULT_SOLIMP, STIFF_SOLREF

    soft = [g for g in spec.geoms if g.solimp[0] < DEFAULT_SOLIMP[0]]
    for g in soft:
        g.solimp, g.solref = list(DEFAULT_SOLIMP), list(STIFF_SOLREF)
    return [g.name for g in soft]


def _turn_body(body: mujoco.MjsBody, axis: np.ndarray, angle: float) -> None:
    turn, quat = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, axis, angle)
    mujoco.mju_mulQuat(quat, np.asarray(body.quat, float), turn)
    body.quat = quat


def _shift(value: float, s: float) -> float:
    """value - s, rounded to 1e-12 rad so a limit reads as written (-2.7227 + 1.8 = -0.9227, not ...0001)."""
    return round(float(value) - s, 12)


def _joints_of(spec: mujoco.MjSpec, body: mujoco.MjsBody) -> int:
    return sum(1 for k in spec.joints if k.parent.name == body.name)


def zero_at_start(spec: mujoco.MjSpec, start: StartPose) -> StartPose:
    """Measure each hinge from its start angle s: its body is turned by s (the subtree as at angle s), and its range,
    springref, its actuators' ctrl ranges and the start are shifted by -s. The robot and every motion are unchanged
    (angle q here is the source's q + s); only qpos0, where ``alloy.build`` starts a robot, becomes the start pose.
    Go2 (2026-11-09 release red): Menagerie's stand is thigh 0.9, calf -1.8 rad, and its calf range [-2.72, -0.84]
    excludes 0, so its qpos0 (legs straight, 160 mm into the floor) was no legal pose. Returns the shifted start."""
    shift = {}
    for j in spec.joints:
        s = float(start.qpos.get(j.name, 0.0))
        if j.type != mujoco.mjtJoint.mjJNT_HINGE or s == 0.0:
            continue
        if np.any(np.asarray(j.pos) != 0.0) or j.ref != 0.0 or _joints_of(spec, j.parent) != 1:
            raise TraceError(f"zero_at_start: joint {j.name} needs its body's only joint, at its origin, ref 0")
        _turn_body(j.parent, np.asarray(j.axis, float), s)
        j.range = [_shift(j.range[0], s), _shift(j.range[1], s)]
        j.springref = _shift(j.springref, s)
        shift[j.name] = s
    ctrl = dict(start.ctrl)
    for a in spec.actuators:
        if a.target in shift:
            s = shift[a.target]
            a.ctrlrange = [_shift(a.ctrlrange[0], s), _shift(a.ctrlrange[1], s)]
            ctrl[a.name] = _shift(ctrl[a.name], s)
    return StartPose({k: _shift(v, shift.get(k, 0.0)) for k, v in start.qpos.items()}, ctrl)


# The LEAP hand's joint and servo dynamics from MuJoCo Playground (Apache-2.0): google-deepmind/mujoco_playground
# v0.2.0 (124a73fa3303f75a62f8fe04d329b829ed0ebdfb), mujoco_playground/_src/manipulation/leap_hand/xmls/leap_rh_mjx.xml,
# whose joint default reads damping 0.2, armature 0.00149376, frictionloss 0.02, actuatorfrcrange +-0.2196. Its
# comments: armature = rotor inertia 1.7e-8 kg m^2 x gear 288^2; torque limit 0.2196 N m = 600 / 1000 x 0.366;
# frictionloss "rough guess, small ish value". Kp 3 is menagerie's and Playground's; menagerie's kv 0.01 is kept.
PLAYGROUND_LEAP = {"damping": 0.2, "armature": 0.00149376, "frictionloss": 0.02, "torque_nm": 0.2196}
PLAYGROUND_LEAP_SOURCE = {"url": "https://github.com/google-deepmind/mujoco_playground", "tag": "v0.2.0",
                          "commit": "124a73fa3303f75a62f8fe04d329b829ed0ebdfb",
                          "file": "mujoco_playground/_src/manipulation/leap_hand/xmls/leap_rh_mjx.xml",
                          "sha256": "335cb3e2906067f02b49d62753e38a55dc4059f090c5adb9c8a62826749782a1",
                          "licence": "Apache-2.0"}


def playground_dynamics(spec: mujoco.MjSpec) -> None:
    """Every hand joint gets ``PLAYGROUND_LEAP``'s damping, armature and frictionloss, and every servo its torque
    limit (as the actuator's force range: one position servo per joint, so the same cap as Playground's joint
    ``actuatorfrcrange``)."""
    if len(spec.joints) != 16 or len(spec.actuators) != 16:
        raise TraceError(f"playground_dynamics: expected 16 joints and 16 servos, got {len(spec.joints)} / "
                         f"{len(spec.actuators)}")
    for j in spec.joints:
        damping = np.array(j.damping, dtype=float)
        damping[0] = PLAYGROUND_LEAP["damping"]
        j.damping, j.armature, j.frictionloss = damping, PLAYGROUND_LEAP["armature"], PLAYGROUND_LEAP["frictionloss"]
    cap = PLAYGROUND_LEAP["torque_nm"]
    for a in spec.actuators:
        a.forcelimited, a.forcerange = mujoco.mjtLimited.mjLIMITED_TRUE, np.array([-cap, cap])


def leap_hand(e: Embodiment) -> tuple[mujoco.MjSpec, StartPose]:
    """menagerie's LEAP right hand (palm up, position servos kp 3) with Playground's joint and servo dynamics
    (``playground_dynamics``); start = open (every joint 0, inside its range)."""
    spec = mujoco.MjSpec.from_file(str(e.source_root() / e.files[0]))
    strip_visuals(spec)
    playground_dynamics(spec)
    qpos = {j.name: 0.0 for j in spec.joints}
    return spec, StartPose(qpos, {a.name: 0.0 for a in spec.actuators})


def ur10e_2f85(e: Embodiment) -> tuple[mujoco.MjSpec, StartPose]:
    """menagerie's UR10e with the Robotiq 2F-85 on its attachment site; the gripper's tendon actuator as a joint
    actuator on right_driver_joint (module doc), its names prefixed ``gripper_``. Start = the UR10e "home", gripper
    open (every gripper joint 0)."""
    root = e.source_root()
    arm = mujoco.MjSpec.from_file(str(root / e.files[0]))
    home = dict(zip((j.name for j in arm.joints), arm.keys[0].qpos, strict=True))
    strip_visuals(arm)
    gripper = mujoco.MjSpec.from_file(str(root / e.files[1]))
    tendon_to_joint(gripper, ROBOTIQ_2F85)
    strip_visuals(gripper)
    home |= {f"gripper_{j.name}": 0.0 for j in gripper.joints}
    arm.attach(gripper, site=arm.site("attachment_site"), prefix="gripper_")      # both models have a body "base"
    return arm, StartPose(home, {a.name: home[a.target] for a in arm.actuators})


# H1-2 stands. The "forklift" profile's gains hold a seated operator: two ankle-pitch servos at kp 60 give 120 N m/rad
# against the 590 N m/rad (67.4 kg x 9.81 m/s2 x 0.89 m) that keeps a standing H1-2 from tipping about its ankles, so the
# shipped robot leaned from 1 s, lay flat at 2 s and landed 20-33 mm deep in the floor (a failed world in health).
# The leg and torso servos' kp and kd are multiplied by STAND_GAIN (ankle pitch 960 N m/rad each); force limits and
# every arm, hand and head servo are the profile's. In plain MuJoCo x6 falls after 5 s and x8 stands leaning 2.6 deg.
STAND_GAIN = 16.0
STAND_JOINTS = ("hip", "knee", "ankle", "torso")


def _h1_2_gains(e: Embodiment) -> dict[str, dict]:
    import json

    from alloy.tasks.emb_specs import gym_anything_root

    robot = json.loads((gym_anything_root() / e.profile).read_text())
    return robot["actuation"]["profiles"]["forklift"]["joints"]


def h1_2(e: Embodiment) -> tuple[mujoco.MjSpec, StartPose]:
    """Unitree H1-2 from unitree_ros's URDF (gym-anything's pinned asset install), made floating (free joint on
    pelvis: MuJoCo's URDF reader fixes the root), with a position actuator on every non-mimic joint. Gains, force
    limits: gym-anything's h1_2 "forklift" profile (kp, kd, effort_limit; its only profile), with the 13 leg and
    torso servos stiffened so that it stands (``STAND_GAIN``). The 12 Inspire-hand mimic joints keep the URDF's
    equality couplings and no actuator. Integrator implicitfast (the URDF carries no MuJoCo options; Euler with
    these PD gains on the finger inertias diverges). Start = every joint 0 (standing straight)."""
    urdf = e.source_root() / e.files[0]
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.meshdir = str(urdf.parent)
    spec.body("pelvis").add_freejoint(name="floating_base")
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST   # a URDF has no options; as menagerie's
    gains = _h1_2_gains(e)                                              # humanoids (G1, H1): Euler is unstable
    mimic = {q.name1 for q in spec.equalities if q.type == mujoco.mjtEq.mjEQ_JOINT}
    for j in [j for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE and j.name not in mimic]:
        g, k = gains[j.name], STAND_GAIN if any(part in j.name for part in STAND_JOINTS) else 1.0
        _position_actuator(spec, j.name, j.name, k * float(g["kp"]), k * float(g["kd"]), tuple(j.range),
                           (-float(g["effort_limit"]), float(g["effort_limit"])))  # fmt: skip
    strip_visuals(spec)
    qpos = {j.name: 0.0 for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE}
    return spec, StartPose(qpos, {a.name: 0.0 for a in spec.actuators})


WHEEL_CTRL = (-np.pi, np.pi)        # [rad] B0 box of a continuous (drive-wheel) joint: B0 needs a finite box


def _profile(e: Embodiment, name: str) -> dict:
    import json

    from alloy.tasks.emb_specs import gym_anything_root

    return json.loads((gym_anything_root() / e.profile).read_text())["actuation"]["profiles"][name]


def _urdf_spec(urdf: Path, package_root: Path | None = None) -> mujoco.MjSpec:
    """A URDF as an MjSpec without its visuals (MuJoCo cannot read the .dae files) and with ``package://<pkg>/``
    resolved to ``package_root``."""
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf).getroot()
    for link in root.iter("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
    text = ET.tostring(root, encoding="unicode")
    if package_root is not None:
        text = text.replace("package://franka_description/", f"{package_root}/")
    return mujoco.MjSpec.from_string(text)


def _servo_joints(spec: mujoco.MjSpec, gains: dict[str, dict]) -> None:
    """A position actuator on every joint with a profile kp > 0 that is not a mimic follower; continuous joints get
    the ``WHEEL_CTRL`` box."""
    mimic = {q.name1 for q in spec.equalities if q.type == mujoco.mjtEq.mjEQ_JOINT}
    for j in [j for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE and j.name not in mimic]:
        g = gains.get(j.name, {})
        if float(g.get("kp", 0.0)) <= 0.0:
            continue
        ctrl = tuple(j.range) if j.limited != mujoco.mjtLimited.mjLIMITED_FALSE and j.range[1] > j.range[0] \
            else WHEEL_CTRL  # fmt: skip
        _position_actuator(spec, j.name, j.name, float(g["kp"]), float(g["kd"]), ctrl,
                           (-float(g["effort_limit"]), float(g["effort_limit"])))  # fmt: skip


def fr3_duo(e: Embodiment) -> tuple[mujoco.MjSpec, StartPose]:
    """Franka Mobile FR3 Duo from franka_description's xacro (expanded at the pin; [internal note]
    mobile-fr3-duo/GENERATED.json), floating base (free joint on base_link), integrator implicitfast. Servos and
    gains: gym-anything's mobile-fr3-duo "classroom" profile (arms kp 3000 kd 100, fingers 3500 / 40, steering
    400 / 12, drive wheels 300 / 25, spine 1e6 / 15000); casters and rocker passive; finger 2 follows finger 1 by
    the URDF mimic. Start = the profile's "stowed" pose."""
    from alloy.tasks.emb_specs import franka_description_root

    spec = _urdf_spec(e.source_root() / e.files[0], franka_description_root())
    spec.body("base_link").add_freejoint(name="floating_base")
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    _servo_joints(spec, _profile(e, "classroom")["joints"])
    firm_wheels(spec)
    import json

    from alloy.tasks.emb_specs import gym_anything_root

    pose = json.loads((gym_anything_root() / e.profile).read_text())["poses"]["stowed"]
    qpos = {j.name: float(pose.get(j.name, 0.0)) for j in spec.joints if j.type != mujoco.mjtJoint.mjJNT_FREE}
    return spec, StartPose(qpos, {a.name: qpos[a.target] for a in spec.actuators})


ROBOTS = {"e1_fr3": fr3_with_hand, "e2_panda": panda, "e3_ur10e_2f85": ur10e_2f85, "e4_fr3_bimanual": fr3_bimanual,
          "e5_fr3_duo": fr3_duo, "e6_g1": g1_with_hands, "e7_h1_2": h1_2, "e8_go2": go2_position, "e9_leap": leap_hand}  # fmt: skip
