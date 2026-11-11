# SPDX-License-Identifier: Apache-2.0
"""What a legged environment keeps between control steps: foot contacts and air time, an accelerometer reading and
joint accelerations. Updated once per control step, after the physics step.

Foot contact (``contact=`` of ``LocoEnv``):
- "height" (default): a foot touches when its sole is within ``contact_height`` (5 mm) of z = 0. Flat ground only;
  ``feet.force`` is then not measured (zeros).
- a reader object with ``read() -> (W, F)`` normal-direction contact force in newtons (``alloy.envs.loco.contact``):
  a foot touches when its force exceeds ``contact_force`` (1 N).

State keys this gives: ``feet.contact`` (W, F) 1 / 0, ``feet.force`` (W, F) N, ``feet.air_time`` (W, F) s the foot
has been off the ground (on the step it lands: the length of the flight that just ended), ``feet.touchdown`` (W, F)
1 on the step a foot lands, ``feet.vel`` (W, F, 3) m/s the foot centres' mean velocity over the last control step,
``body.contacts`` (W,) how many of the robot's other collision shapes (base, hips, thighs, shanks) carry a
contact force over ``contact_force`` (always 0 with "height" contact), ``base.accel_b`` (W, 3) m/s^2 the accelerometer
reading in the base frame (the base's
mean acceleration over the last control step minus gravity: (0, 0, +9.81) at rest), ``joints.acc`` (W, J) rad/s^2.
"""

from __future__ import annotations

from typing import Any

from alloy.envs.loco.state import LocoState, rotate_inverse

GRAVITY = 9.81


class LocoSensors:
    """Per-world buffers of the keys above. Example: ``sensors.advance(state); sensors.extra()["feet.contact"]``."""

    def __init__(self, worlds: int, feet: int, joints: int, dt: float, xp: Any, *, reader: Any = None,
                 contact_height: float = 0.005, contact_force: float = 1.0) -> None:
        self.xp, self.dt, self.reader = xp, float(dt), reader
        self.contact_height, self.contact_force = float(contact_height), float(contact_force)
        self.contact = xp.zeros((worlds, feet))
        self.force = xp.zeros((worlds, feet))
        self.air_time = xp.zeros((worlds, feet))
        self.reported_air = xp.zeros((worlds, feet))
        self.touchdown = xp.zeros((worlds, feet))
        self.accel = xp.zeros((worlds, 3)) + xp.asarray([0.0, 0.0, GRAVITY])
        self.joint_acc = xp.zeros((worlds, joints))
        self.body_contacts = xp.zeros(worlds)
        self.feet_vel = xp.zeros((worlds, feet, 3))
        self._feet_pos = xp.zeros((worlds, feet, 3))
        self._vel = xp.zeros((worlds, 3))
        self._joint_vel = xp.zeros((worlds, joints))

    def _touching(self, s: LocoState) -> Any:
        if self.reader is None:
            return s["feet.height"] < self.contact_height
        self.force = self.xp.copy(self.reader.read())
        if hasattr(self.reader, "read_others"):
            self.body_contacts = self.xp.float(self.reader.read_others() > self.contact_force).sum(-1)
        return self.force > self.contact_force

    def advance(self, s: LocoState) -> None:
        """Take in the state after one control step. Example: ``sensors.advance(env_state)``."""
        xp, dt = self.xp, self.dt
        touching = self._touching(s)
        was_touching = self.contact > 0.5
        self.contact = xp.float(touching)
        landed = touching & ~was_touching & (self.air_time > 0)
        self.touchdown = xp.float(landed)
        flight = self.air_time + dt
        self.reported_air = xp.where(touching, xp.where(landed, self.air_time, 0.0), flight)
        self.air_time = xp.where(touching, 0.0, flight)
        feet_pos = xp.copy(s["feet.pos"])
        self.feet_vel = (feet_pos - self._feet_pos) / dt
        self._feet_pos = feet_pos
        vel = xp.copy(s["base.lin_vel_w"])
        gravity = xp.zeros((s.worlds, 3)) + xp.asarray([0.0, 0.0, -GRAVITY])
        self.accel = rotate_inverse(xp, s["base.quat_xyzw"], (vel - self._vel) / dt - gravity)
        self._vel = vel
        joint_vel = xp.copy(s["joints.vel"])
        self.joint_acc = (joint_vel - self._joint_vel) / dt
        self._joint_vel = joint_vel

    def reset(self, s: LocoState, mask: Any) -> None:
        """Start the buffers of the worlds in the bool (W,) ``mask`` from their (just reset) state."""
        xp = self.xp
        m1 = mask[:, None]
        # by height here: a reader's force is still that of the step before the reset
        touching = xp.float(s["feet.height"] < self.contact_height)
        self.contact = xp.where(m1, touching, self.contact)
        for name in ("force", "air_time", "reported_air", "touchdown", "joint_acc"):
            setattr(self, name, xp.where(m1, 0.0, getattr(self, name)))
        self.body_contacts = xp.where(mask, 0.0, self.body_contacts)
        m2 = mask[:, None, None]
        self.feet_vel = xp.where(m2, 0.0, self.feet_vel)
        self._feet_pos = xp.where(m2, s["feet.pos"], self._feet_pos)
        rest = xp.zeros((s.worlds, 3)) + xp.asarray([0.0, 0.0, GRAVITY])
        self.accel = xp.where(m1, rotate_inverse(xp, s["base.quat_xyzw"], rest), self.accel)
        self._vel = xp.where(m1, s["base.lin_vel_w"], self._vel)
        self._joint_vel = xp.where(m1, s["joints.vel"], self._joint_vel)

    def extra(self) -> dict[str, Any]:
        """The buffers as state keys. Example: ``sensors.extra()["base.accel_b"]``."""
        return {"feet.contact": self.contact, "feet.force": self.force, "feet.air_time": self.reported_air,
                "feet.touchdown": self.touchdown, "feet.vel": self.feet_vel, "base.accel_b": self.accel,
                "body.contacts": self.body_contacts,
                "joints.acc": self.joint_acc}


__all__ = ["GRAVITY", "LocoSensors"]
