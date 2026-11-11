# SPDX-License-Identifier: Apache-2.0
"""Panda MJCF for E23: the arm at a start posture and a cup welded into the hand, held at its rim by the fingers.

Alloy has no call that sets a robot's start joint angles at build (an MJCF ``ref`` is ignored), and a liquid must start
inside the cup in world coordinates, so the start posture is baked into the file: each joint's body frame is turned by
the start angle and the joint's range shifted by it. Every joint then reads the DISPLACEMENT from the start posture;
``start_q`` adds the start back to give the absolute angle.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alloy.cards.library import default_root
from alloy.envs.mjcf import cache_dir, without_bodies

GRASP_Z = 0.1034              # hand frame: the finger pads' centre, along the hand's z
PAD_FACE = 0.0015             # a finger pad's inner face sits this far past the finger joint's travel


@dataclass(frozen=True)
class Cup:
    """A cup held by the hand with its axis along the hand's z (open end toward the palm), rim on the pads."""

    inner_radius: float = 0.030
    inner_depth: float = 0.090
    wall: float = 0.004
    rim_z: float = 0.0934     # hand frame z of the open end
    mass: float = 0.15
    segments: int = 16

    @property
    def outer_radius(self) -> float:
        return self.inner_radius + self.wall

    @property
    def floor_top_z(self) -> float:
        return self.rim_z + self.inner_depth

    @property
    def centre_z(self) -> float:
        return 0.5 * (self.rim_z + self.floor_top_z + self.wall)

    @property
    def finger_travel(self) -> float:
        """Each finger's joint travel that puts its pad face on the cup's outer wall (less 0.5 mm)."""
        return self.outer_radius - PAD_FACE - 0.0005


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u = np.array([0.0, *v])
    qc = q * np.array([1.0, -1.0, -1.0, -1.0])
    return _quat_mul(_quat_mul(q, u), qc)[1:]


def _floats(text: str | None, default: str) -> np.ndarray:
    return np.array([float(x) for x in (text or default).split()])


def _fmt(v: np.ndarray) -> str:
    return " ".join(f"{x:.9g}" for x in v)


def _bake_joint(body: ET.Element, joint: ET.Element, value: float) -> None:
    """Move ``body`` as its ``joint`` would at ``value`` and shift the joint's range, so the joint reads 0 there."""
    axis = _floats(joint.get("axis"), "0 0 1")
    quat = _floats(body.get("quat"), "1 0 0 0")
    if joint.get("type", "hinge") == "slide":
        body.set("pos", _fmt(_floats(body.get("pos"), "0 0 0") + _rotate(quat, axis / np.linalg.norm(axis)) * value))
    else:
        body.set("quat", _fmt(_quat_mul(quat, _quat_axis(axis, value))))
    lo, hi = _floats(joint.get("range"), "0 0")
    joint.set("range", _fmt(np.array([lo - value, hi - value])))


def _shift_actuators(root: ET.Element, values: dict[str, float]) -> None:
    """The servos' ``ctrlrange`` is in joint coordinates: shift it with the joint so a target 0 holds the baked start."""
    for act in root.iter("general"):
        value = values.get(act.get("joint", ""))
        if value is not None and act.get("ctrlrange"):
            lo, hi = _floats(act.get("ctrlrange"), "0 0")
            act.set("ctrlrange", _fmt(np.array([lo - value, hi - value])))


def bake_posture(root: ET.Element, values: dict[str, float]) -> None:
    """Bake joint values (name -> rad or m) into the body frames of ``root``, and into its servos' control ranges."""
    for body in root.iter("body"):
        for joint in body.findall("joint"):
            if joint.get("name") in values:
                _bake_joint(body, joint, values[joint.get("name")])
    _shift_actuators(root, values)


def _wall_geoms(cup: Cup) -> list[ET.Element]:
    r_mid = cup.inner_radius + cup.wall / 2
    half_tangent = 1.25 * r_mid * np.tan(np.pi / cup.segments)
    z_mid = 0.5 * (cup.rim_z + cup.floor_top_z + cup.wall)
    half_z = 0.5 * (cup.floor_top_z + cup.wall - cup.rim_z)
    mass_each = 0.6 * cup.mass / cup.segments
    geoms = []
    for k in range(cup.segments):
        phi = 2 * np.pi * k / cup.segments
        geoms.append(ET.Element("geom", {
            "name": f"cup_wall{k}", "type": "box", "size": _fmt(np.array([cup.wall / 2, half_tangent, half_z])),
            "pos": _fmt(np.array([r_mid * np.cos(phi), r_mid * np.sin(phi), z_mid])), "mass": f"{mass_each:.6g}",
            "quat": _fmt(_quat_axis(np.array([0.0, 0.0, 1.0]), phi)), "friction": "0.6 0.005 0.0001"}))
    geoms.append(ET.Element("geom", {
        "name": "cup_floor", "type": "cylinder", "size": f"{cup.outer_radius:.6g} {cup.wall / 2:.6g}",
        "pos": _fmt(np.array([0.0, 0.0, cup.floor_top_z + cup.wall / 2])), "mass": f"{0.4 * cup.mass:.6g}"}))
    return geoms


def add_cup(root: ET.Element, cup: Cup) -> None:
    """A body ``cup`` (no joint) under ``hand``: a ring of boxes and a floor."""
    hand = next(b for b in root.iter("body") if b.get("name") == "hand")
    body = ET.SubElement(hand, "body", {"name": "cup", "pos": "0 0 0"})
    body.extend(_wall_geoms(cup))


def panda_with_cup(start_q: np.ndarray, cup: Cup) -> Path:
    """The library Panda scene without its props, at ``start_q`` with ``cup`` welded to the hand; a cached file."""
    base = without_bodies(default_root() / "robots" / "emb_e2_panda" / "scene.xml", ["box", "cylinder", "mug"])
    key = hashlib.sha256((base.read_text() + repr(cup) + "v2" + repr(np.round(start_q, 9).tolist())).encode()).hexdigest()[:16]
    out = cache_dir() / f"panda_cup-{key}.xml"
    if out.is_file():
        return out
    root = ET.fromstring(base.read_text(encoding="utf-8"))
    values = {f"joint{k + 1}": float(v) for k, v in enumerate(start_q)}
    values.update({"finger_joint1": cup.finger_travel, "finger_joint2": cup.finger_travel})
    bake_posture(root, values)
    add_cup(root, cup)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="unicode")
    return out
