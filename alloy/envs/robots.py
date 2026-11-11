# SPDX-License-Identifier: Apache-2.0
"""Robots for the ready-made tasks, as MJCF strings with position servos (``sim.set_targets`` and joint targets).

``gantry_gripper``: a Cartesian two-finger gripper, the usual "floating gripper" of manipulation benchmarks. Three
stacked slides (x, y, z), a wrist yaw and two fingers that close together (one actuator; the right finger mirrors
the left through a stiff equality constraint, as the fingers of a real parallel gripper are linked). Its joints
are "<robot>/x", "/y", "/z", "/yaw", "/close"; its fingers "<robot>/finger_left", "<robot>/finger_right"; the palm
"<robot>/palm". Example::

    scene.add_robot("gripper", mjcf=gantry_gripper(), pos=(0, 0, 0.2))     # palm 20 cm above the table
    task = Task(actions=gantry_actions("gripper"), ...)
"""

from __future__ import annotations

from alloy.envs.actions import ActionSpec

FINGER_OPEN = 0.0
FINGER_CLOSED = 0.04
COUPLING_TIMECONST = 0.0025     # s, finger-link solref[0]: 2 physics steps at the rigid tasks' 1.25 ms step


def gantry_gripper(reach_xy: float = 0.35, reach_z: float = 0.25, finger_gap: float = 0.08,
                   finger_length: float = 0.06, friction: float = 1.5,
                   coupling_timeconst: float = COUPLING_TIMECONST, squeeze: float = 0.0) -> str:
    """MJCF of the Cartesian gripper: slides within +-``reach_xy`` (x, y) and -``reach_z``..+0.1 m (z) of where it is
    placed, fingers ``finger_gap`` m apart when open and ``finger_length`` m long.

    The finger link's time constant ``coupling_timeconst`` (s) is written as its solref. MuJoCo raises any value under
    two physics steps to that, so the default is that floor at 1.25 ms. MuJoCo's own default, 0.02 s, let the finger
    on a load lag about 1 mm per N.

    ``squeeze`` (m) lets the close target run that far past ``FINGER_CLOSED``, where the fingers meet. The close servo
    (kp 1500 N/m) aimed at the meeting point presses a thin part only with kp x the part's half thickness: 2.6 N on
    a 3.5 mm cloth ridge. With ``squeeze`` the fingers close exactly as before and, once they stop on a part (or on
    each other), squeeze it with about kp x ``squeeze`` more: 15 N for 10 mm. ``close.q`` can then read past
    ``FINGER_CLOSED`` by the fingers' contact depth when they close on nothing. Example: ``gantry_gripper()``."""
    if squeeze < 0.0:
        raise ValueError(f"squeeze must be >= 0 m, got {squeeze!r}")
    half_gap, flen = finger_gap / 2.0 + 0.005, finger_length / 2.0
    close_hi = FINGER_CLOSED + squeeze
    carriage = '<geom type="sphere" size="0.005" mass="0.2" contype="0" conaffinity="0"/>'
    finger = (f'<geom type="box" size="0.012 0.005 {flen:.4f}" mass="0.03" friction="{friction} 0.02 0.002" '
              f'condim="4"/>')
    return f"""<mujoco>
  <worldbody>
    <body name="carriage_x">
      <joint name="x" type="slide" axis="1 0 0" range="{-reach_xy} {reach_xy}" damping="60"/>
      {carriage}
      <body name="carriage_y">
        <joint name="y" type="slide" axis="0 1 0" range="{-reach_xy} {reach_xy}" damping="60"/>
        {carriage}
        <body name="carriage_z">
          <joint name="z" type="slide" axis="0 0 1" range="{-reach_z} 0.1" damping="60"/>
          {carriage}
          <body name="palm">
            <joint name="yaw" type="hinge" axis="0 0 1" range="-1.6 1.6" damping="2"/>
            <geom type="box" size="0.03 {half_gap + 0.01:.4f} 0.01" mass="0.2" friction="{friction}"/>
            <body name="finger_left" pos="0 {half_gap:.4f} {-(0.01 + flen):.4f}">
              <joint name="close" type="slide" axis="0 -1 0" range="0 {close_hi:g}" damping="5"/>
              {finger}
            </body>
            <body name="finger_right" pos="0 {-half_gap:.4f} {-(0.01 + flen):.4f}">
              <joint name="close_mirror" type="slide" axis="0 1 0" range="0 {close_hi:g}" damping="5"/>
              {finger}
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <equality><joint joint1="close_mirror" joint2="close" polycoef="0 1 0 0 0"
    solref="{coupling_timeconst:g} 1"/></equality>
  <actuator>
    <position joint="x" kp="3000" kv="200"/>
    <position joint="y" kp="3000" kv="200"/>
    <position joint="z" kp="3000" kv="200"/>
    <position joint="yaw" kp="20" kv="2"/>
    <position joint="close" kp="1500" kv="30"/>
  </actuator>
</mujoco>"""


def gantry_actions(robot: str, step_m: float = 0.01, step_yaw: float = 0.1, step_close: float = 0.004,
                   yaw: bool = True) -> ActionSpec:
    """Delta actions for the Cartesian gripper: x, y, z (up to ``step_m`` per step), yaw, close.

    Example: ``Task(actions=gantry_actions("gripper"))``."""
    joints = [f"{robot}/x", f"{robot}/y", f"{robot}/z"] + ([f"{robot}/yaw"] if yaw else []) + [f"{robot}/close"]
    delta = [step_m] * 3 + ([step_yaw] if yaw else []) + [step_close]
    return ActionSpec(joints, mode="delta", delta=delta)


def fingers(robot: str) -> tuple[str, str]:
    """The two finger bodies of a Cartesian gripper. Example: ``C.between("cube", *fingers("gripper"))``."""
    return f"{robot}/finger_left", f"{robot}/finger_right"


__all__ = ["COUPLING_TIMECONST", "FINGER_CLOSED", "FINGER_OPEN", "fingers", "gantry_actions", "gantry_gripper"]
