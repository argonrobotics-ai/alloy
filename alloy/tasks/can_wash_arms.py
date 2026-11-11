# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can, robot stage: two fixed arms in the can-wash bay (WP15's rig (b), Gym-Anything's two hands).

- A UR10e with a Robiq 2F-85 (``library/robots/emb_e3_ur10e_2f85``) holds the can by its rim on the side away from
  the drain: it turns the can about its axis under the wand (the "lathe") and tips it out to 105°. Its payload,
  12.5 kg, covers the can's 4.5 kg and up to 6.4 kg of pool; an FR3 (3 kg) cannot hold even the empty can at 105°.
  The fingers pinch the rim; the can is a free body (real contact, no weld).
- An FR3 holds GA's hose valve fixed in its hand (SOFT-3's pattern, ``paint_fr3_can``): the valve body is a child of
  the hand with geoms that do not collide, its nozzle ``WAND`` past the tool point along the approach; the supply's
  hose nozzle rides on it. The FR3 sprays from one sector and docks the valve at GA's holster.
- Bases from reach scans ([internal note] ``notes/job3/can_arm_reach.md``): the UR10e on a 0.30 m pedestal east of the
  can reaches every tip pose; from any FR3 base the wand reaches at most 60° of the outer wall at all heights (the
  far side of the 0.30 m standoff circle is 1.0 m away, against the FR3's 0.855 m), hence the lathe. The FR3 stands
  on a 0.70 m pedestal between the holster and the can.

The arms' links and fingers are not film tools: the film is one-way with them (the fingers on a wet rim wipe nothing;
WP15's known gap). Only the wand's jets and the can's own charts touch the film.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np

from alloy.tasks.can_wash_scene import CAPACITY, CanWash
from alloy.tasks.rinse_flush_floor import NOZZLE, RIGID_DT, RIGID_GAP, VALVE
from alloy.tasks.rinse_flush_policy import CONTROL_DT

UR_BASE = (1.70, 1.10, 0.30)     # m: the UR10e's base on its pedestal
UR_YAW = 180.0                   # deg: facing the can (-x)
FR3_BASE = (0.70, 0.60, 0.70)    # m: the FR3's base, between the holster and the can, off the tipped can's footprint
FR3_YAW = 45.0                   # deg: facing the can
WAND = 0.15                      # m: the tool point to the nozzle, along the approach
VALVE_MASS = 0.45                # kg: the hose valve in the hand (an estimate: a pistol nozzle without its hose)
VALVE_R, VALVE_HALF = 0.027, 0.10   # m: SC9's valve cylinder


def yaw_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    """xyzw of a turn about z."""
    h = math.radians(0.5 * yaw_deg)
    return (0.0, 0.0, math.sin(h), math.cos(h))


def wand_xml() -> str:
    """The FR3's MJCF with GA's hose valve in its hand: body ``VALVE`` with its nozzle ``WAND`` past the tool point
    along the approach, its z pointing back up the approach (the nozzle is ``NOZZLE`` down the valve's -z, SC9's)."""
    from alloy.arms import FR3  # noqa: PLC0415
    from alloy.arms.mjcf import robot_xml  # noqa: PLC0415
    from alloy.arms.specs import FRANKA_TCP  # noqa: PLC0415

    root = ET.fromstring(robot_xml(FR3, "", False))
    hand = next(b for b in root.iter("body") if b.get("name") == "hand")
    valve = ET.SubElement(hand, "body", name=VALVE, pos=f"0 0 {FRANKA_TCP[2] + WAND - NOZZLE}", quat="0 1 0 0")
    side = VALVE_MASS * (3 * VALVE_R ** 2 + (2 * VALVE_HALF) ** 2) / 12.0
    ET.SubElement(valve, "inertial", pos="0 0 0", mass=str(VALVE_MASS),
                  diaginertia=f"{side} {side} {0.5 * VALVE_MASS * VALVE_R ** 2}")
    ET.SubElement(valve, "geom", type="cylinder", size=f"{VALVE_R} {VALVE_HALF}", contype="0", conaffinity="0")
    return ET.tostring(root, encoding="unicode")


def ur_xml() -> str:
    from alloy.arms import UR10E  # noqa: PLC0415
    from alloy.arms.mjcf import robot_xml  # noqa: PLC0415

    return robot_xml(UR10E, "", False)


@dataclass(frozen=True)
class Rigs:
    """The two arms' rigs (``alloy.arms.Rig``: IK, binding, gravity compensation, servo targets) and bases."""

    ur: Any
    fr3: Any

    @staticmethod
    def make(ur_base: tuple = (*UR_BASE, *yaw_quat(UR_YAW)), fr3_base: tuple = (*FR3_BASE, *yaw_quat(FR3_YAW))) -> Rigs:
        from alloy.arms import FR3, UR10E, Rig  # noqa: PLC0415

        return Rigs(Rig.single(UR10E, ur_base, name="ur", visuals=False),
                    Rig.single(FR3, fr3_base, name="fr3", visuals=False))


@dataclass(frozen=True)
class ArmsBay:
    """The arms' scene: the bay and the can of ``task`` (without the kinematic valve: the FR3 holds it), both arms at
    their bases and starts ``q_ur``, ``q_fr3`` [J], MuJoCo at SC9's ``RIGID_DT``, gravity compensated (the valve
    too)."""

    task: CanWash
    rigs: Rigs
    q_ur: np.ndarray
    q_fr3: np.ndarray

    def scene(self, init: Any, worlds: int = 1) -> Any:
        from alloy.plugins.harness.scene import TestScene  # noqa: PLC0415

        return TestScene(build=lambda b: None, init=init, worlds=worlds, control_dt=CONTROL_DT, dt=RIGID_DT,
                         make=self._make)

    def _make(self, plugins: Any, device: str, worlds: int, graph: bool) -> Any:  # noqa: ARG002
        import warp as wp  # noqa: PLC0415

        from alloy._vendor import newton  # noqa: PLC0415
        from alloy._vendor.newton.solvers import SolverMuJoCo  # noqa: PLC0415

        from alloy.plugins.host import simulation  # noqa: PLC0415
        from alloy.scene.builtin_plugins import register_builtin_solvers  # noqa: PLC0415
        from alloy.testing.integration import built_from_model, entry_spec  # noqa: PLC0415

        register_builtin_solvers()
        t = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(t)
        t.rigid_gap = RIGID_GAP
        for xml, base in ((ur_xml(), self.rigs.ur.bases[0]), (wand_xml(), self.rigs.fr3.bases[0])):
            t.add_mjcf(xml, xform=wp.transform(base[:3], wp.quat(*base[3:])), ctrl_direct=True, parse_visuals=False)
        self.task.build_static(t)
        root = newton.ModelBuilder()
        SolverMuJoCo.register_custom_attributes(root)
        root.replicate(t, worlds)
        model = root.finalize(device=device)
        for rig, q in ((self.rigs.ur, self.q_ur), (self.rigs.fr3, self.q_fr3)):
            bind = rig.bind(model)
            rig.gravcomp(model, bind)
            rig.write_ready(model, bind, np.broadcast_to(q, (worlds, 1, q.size)).copy(), 0.085 if rig is self.rigs.ur
                            else 0.0)
        entry = entry_spec(model, "mujoco", RIGID_DT, **CAPACITY)
        return simulation(built_from_model(model, [entry], control_dt=CONTROL_DT), list(plugins))


__all__ = ["FR3_BASE", "FR3_YAW", "UR_BASE", "UR_YAW", "WAND", "ArmsBay", "Rigs", "wand_xml",
           "ur_xml", "yaw_quat"]
