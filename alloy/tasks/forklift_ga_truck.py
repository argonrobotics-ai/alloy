# SPDX-License-Identifier: Apache-2.0
"""GA's scripted forklift operator on Alloy's truck (SCR-WP8-12 F5): what the operator reads from the simulation, what
its controls do to the vehicle plug-in and the mast's valves, and the map from a rig's frame to GA's facility frame.

- ``Frame``: a rig is built in its own frame (F4's dock with its face at x = 0, say); one turn about z and one shift
  take its poses to GA's facility frame (``forklift_site``), where the plans and the criteria live.
- ``TruckIO``: the F2 truck (``forklift_truck``) in a harness simulation with its ``Vehicle`` and ``FluidNetwork``
  plug-ins. ``observe`` gives the operator's ``Observation``; ``apply`` turns its ``Controls`` into the vehicle's
  throttle, brake and steer and the valves' openings; ``sample`` reads one control period's record for the episode.

The operator's key, engine and seat are signals only: the truck has an electric traction motor in place of GA's LPG
engine and torque converter (out of scope, SCR-WP8-12 ruling), so the throttle works whenever the engine signal is on
and a direction is selected. ``fork_height`` is the carriage's lift above its lowest (free lift plus twice the stage
travel), which is GA's blade-bottom height less the 9 mm the blades clear the floor at the bottom.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from alloy.tasks import forklift_truck as ft
from alloy.tasks.forklift_control import Controls, Observation

CHASSIS, SIDESHIFT = "forklift", "sideshift"
JOINTS = ("mast_tilt", "mast_stage", "mast_free", "sideshift_slide")
TIP = (ft.FORK_FACE + ft.FORK_LENGTH - ft.PIVOT[0], ft.FORK_BOTTOM - ft.PIVOT[2])   # blade tip bottom, side-shift body


@dataclass(frozen=True)
class Frame:
    """Rig frame -> GA's facility frame: turn by ``yaw`` about z, then shift by ``shift``."""

    yaw: float = 0.0
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def turn(self) -> Rotation:
        return Rotation.from_euler("z", self.yaw)

    def point(self, p: np.ndarray) -> np.ndarray:
        """Rig points [..., 3] in GA's frame."""
        return self.turn.apply(np.reshape(p, (-1, 3))).reshape(np.shape(p)) + np.asarray(self.shift)

    def quat(self, q_xyzw: np.ndarray) -> np.ndarray:
        """Rig orientations [..., 4] (x, y, z, w) in GA's frame."""
        q = np.reshape(q_xyzw, (-1, 4))
        return (self.turn * Rotation.from_quat(q)).as_quat().reshape(np.shape(q_xyzw))

    def to_rig(self, p: np.ndarray) -> np.ndarray:
        """GA points [..., 3] in the rig's frame."""
        return self.turn.inv().apply(np.reshape(p, (-1, 3)) - np.asarray(self.shift)).reshape(np.shape(p))


@dataclass(frozen=True)
class Extras:
    """Task-specific readings for the observation (each called with the ``TruckIO``)."""

    on_ramp: Callable[[TruckIO], bool] = lambda io: False
    load_on_forks: Callable[[TruckIO], bool] = lambda io: False
    restraint_green: Callable[[TruckIO], bool | None] = lambda io: None
    floor_z: Callable[[TruckIO], float] = lambda io: 0.0          # the floor under the fork tips (GA's frame)


class TruckIO:
    """The truck's readings and inputs in one world (``world``) of a harness simulation."""

    def __init__(self, sim: Any, car: Any, hydraulics: Any, frame: Frame = Frame(), extras: Extras = Extras(),
                 world: int = 0) -> None:
        self.sim, self.car, self.hyd, self.frame, self.extras, self.world = sim, car, hydraulics, frame, extras, world
        m = sim.model
        per_world = m.body_count // max(m.world_count, 1)
        labels = [str(x).rsplit("/", 1)[-1] for x in m.body_label]
        self.chassis = labels.index(CHASSIS) + world * per_world
        self.sideshift = labels.index(SIDESHIFT) + world * per_world
        jlabels = [str(x).rsplit("/", 1)[-1] for x in m.joint_label]
        per_joint = m.joint_count // max(m.world_count, 1)
        q_start = m.joint_q_start.numpy()
        self.q_index = {j: int(q_start[jlabels.index(j) + world * per_joint]) for j in JOINTS}
        self.signals = {"park": 1.0, "direction": 0.0, "engine_running": 0.0, "key_on": 0.0,
                        "seat_switch_closed": 1.0}
        self._q = self._qd = self._jq = None

    # -- reading ------------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        """Read the state once per control period (every reading below uses it)."""
        self._q = self.sim.state.body_q.numpy()
        self._qd = self.sim.state.body_qd.numpy()
        self._jq = self.sim.state.joint_q.numpy()

    def pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Chassis position and quaternion (x, y, z, w) in GA's frame."""
        q = self._q[self.chassis].astype(np.float64)
        return self.frame.point(q[:3]), self.frame.quat(q[3:])

    def speed(self) -> float:
        """Signed forward speed of the chassis (its centre of mass's velocity along its x)."""
        q = self._q[self.chassis].astype(np.float64)
        return float(Rotation.from_quat(q[3:]).apply([1.0, 0.0, 0.0]) @ self._qd[self.chassis, :3])

    def joint(self, name: str) -> float:
        return float(self._jq[self.q_index[name]])

    def fork_height(self) -> float:
        return self.joint("mast_free") + 2.0 * self.joint("mast_stage")

    def tips(self) -> np.ndarray:
        """The two blade tips' bottoms (left, right) in GA's frame [2, 3]."""
        q = self._q[self.sideshift].astype(np.float64)
        local = np.array([[TIP[0], ft.FORK_Y, TIP[1]], [TIP[0], -ft.FORK_Y, TIP[1]]])
        return self.frame.point(Rotation.from_quat(q[3:]).apply(local) + q[:3])

    def blade_pitch(self) -> float:
        """The blades' pitch (rad, + = tips up)."""
        axis = Rotation.from_quat(self._q[self.sideshift, 3:].astype(np.float64)).apply([1.0, 0.0, 0.0])
        return float(math.asin(np.clip(axis[2], -1.0, 1.0)))

    def grade(self) -> float:
        """The chassis's pitch along its heading (rad, + = nose up)."""
        axis = Rotation.from_quat(self._q[self.chassis, 3:].astype(np.float64)).apply([1.0, 0.0, 0.0])
        return float(math.asin(np.clip(axis[2], -1.0, 1.0)))

    def wheel_fz(self) -> np.ndarray:
        return self.car.read()["force"][self.world, :, 2].astype(np.float64)

    def observe(self, t: float) -> Observation:
        p, quat = self.pose()
        yaw = float(Rotation.from_quat(quat).as_euler("ZYX")[0])
        return Observation(t=t, x=float(p[0]), y=float(p[1]), yaw=yaw, speed=self.speed(),
                           fork_height=self.fork_height(), tilt=self.joint("mast_tilt"),
                           sideshift=self.joint("sideshift_slide"), on_ramp=bool(self.extras.on_ramp(self)),
                           grade=self.grade(), load_on_forks=bool(self.extras.load_on_forks(self)),
                           restraint_green=self.extras.restraint_green(self), blade_pitch=self.blade_pitch(),
                           tip_height=float(self.tips()[:, 2].min()) - self.extras.floor_z(self))

    def sample(self) -> dict[str, Any]:
        """One control period's record for the episode (GA's frame)."""
        p, quat = self.pose()
        return {"truck_pos": p, "truck_quat_xyzw": quat, "speed": self.speed(), "wheel_fz": self.wheel_fz(),
                "fork_tips": self.tips(), "blade_pitch": self.blade_pitch(), **dict(self.signals)}

    # -- commanding ---------------------------------------------------------------------------------------------
    def apply(self, c: Controls) -> None:
        """The operator's controls to the plug-ins' inputs (this world only)."""
        live = c.engine_on and c.key_on and c.direction != 0
        self._input(self.car, "set_input", "throttle", c.throttle if live else 0.0)
        self._input(self.car, "set_input", "brake", 1.0 if c.park else c.brake)
        self._input(self.car, "set_input", "steer", c.steer)
        pump = c.engine_on and c.key_on                    # the pump runs with the engine; lowering needs no pump
        self._valves({"lift_up": max(c.lift, 0.0) if pump else 0.0, "lift_down": max(-c.lift, 0.0)})
        fwd, back = (max(c.tilt, 0.0), max(-c.tilt, 0.0)) if pump else (0.0, 0.0)
        self._valves({"tilt_fwd": fwd, "tilt_b_drain": fwd, "tilt_back": back, "tilt_a_drain": back})
        left, right = (max(c.sideshift, 0.0), max(-c.sideshift, 0.0)) if pump else (0.0, 0.0)
        self._valves({"shift_left": left, "shift_b_drain": left, "shift_right": right, "shift_a_drain": right})
        self.signals.update(park=float(c.park), direction=float(c.direction), engine_running=float(c.engine_on),
                            key_on=float(c.key_on))

    def _valves(self, openings: dict[str, float]) -> None:
        for edge, value in openings.items():
            self._input(self.hyd, "set_opening", edge, value)

    def _input(self, plugin: Any, method: str, name: str, value: float) -> None:
        """Set one world's value: a per-world array when the simulation has several worlds."""
        worlds = self.sim.model.world_count
        if worlds <= 1:
            getattr(plugin, method)(name, float(value))
            return
        cache = plugin.__dict__.setdefault("_ga_inputs", {})
        row = cache.setdefault((method, name), np.zeros(worlds))
        row[self.world] = float(value)
        getattr(plugin, method)(name, row.copy())


__all__ = ["CHASSIS", "Extras", "Frame", "JOINTS", "SIDESHIFT", "TIP", "TruckIO"]
