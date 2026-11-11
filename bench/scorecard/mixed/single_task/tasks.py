# SPDX-License-Identifier: Apache-2.0
"""hold-task tasks for the Alloy rows: the robot + scene standing for an Isaac Lab task, and its hold pose.

Each task starts from an embodiment scene that ships with Alloy (``library/robots/emb_*``: a MuJoCo Menagerie
robot with a table and props, or a floor). ``model`` writes the task's MJCF into the run folder:
- the pack's props the Isaac Lab task does not have are dropped (``drop``), and a lift cube is added (``cube``);
- the file is re-zeroed at the pack's ``init`` keyframe (``bake.py``), because ``alloy.build`` starts a robot at its
  file pose; the hold targets are the baked keyframe's ``ctrl``.
The Alloy rows load that file with ``scene.add_robot("r", path)``.

These are NOT the USD assets the Isaac Lab tasks load: same robot type, another model file (Menagerie MJCF), with
the differences named in each task's ``note``. ``score_spec`` gives ``pxs_score.score`` the task's tracked bodies and
floor in this scene.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIBRARY = Path(__file__).resolve().parents[4] / "library"     # the Alloy tree this harness sits in
CUBE = {"edge": 0.05, "mass": 0.2, "friction": 0.5}     # Isaac Lab's lift cube


@dataclass(frozen=True)
class HoldTask:
    name: str
    isaac_task: str                              # the Isaac Lab task this stands for
    pack: str                                    # library/robots/<pack>/scene.xml
    kind: str                                    # pxs_score kind: rest | free | stand | inhand
    track: tuple[tuple[str, str], ...]           # (pxs_score label, body in the file)
    floor_z: float                               # lowest valid body origin height (m)
    drop: tuple[str, ...] = ()                   # top-level bodies of the pack to leave out
    cube: tuple[float, float, float] | None = None    # centre of an added lift cube
    cube_top: float | None = None                # top of the cube's support (m)
    note: str = ""


TASKS = {t.name: t for t in (
    HoldTask("franka_lift", "Isaac-Lift-Franka", "emb_e2_panda", "rest", (("hand", "hand"), ("object", "cube")),
             -0.75, ("box", "cylinder", "mug"), (0.5, 0.0, 0.025), 0.0,
             "Menagerie Panda with its own servo gains on a table, the lift cube (CUBE) at rest on it"),
    HoldTask("franka_reach", "Isaac-Reach-Franka", "emb_e2_panda", "free", (("ee", "hand"),),
             -0.75, ("box", "cylinder", "mug"), note="Menagerie Panda on a table, no props"),
    HoldTask("ur10_reach", "Isaac-Reach-UR10", "emb_e3_ur10e_2f85", "free", (("ee", "wrist_3_link"),), -0.75,
             ("box", "cylinder", "mug"),
             note="Menagerie UR10e WITH its Robotiq 2F-85 gripper (15 more bodies than Isaac Lab's bare UR10), no props"),
    HoldTask("go2_flat", "Isaac-Velocity-Flat-UnitreeGo2", "emb_e8_go2", "stand", (("base", "base"),), 0.0,
             ("obstacle",), note="Menagerie Go2 standing on a floor"),
    HoldTask("g1_flat", "Isaac-Velocity-Flat-G1", "emb_e6_g1", "stand", (("base", "pelvis"),), 0.0, ("obstacle",),
             note="Menagerie G1 with hands (43 actuators) standing on a floor"),
    HoldTask("hand_cube", "Isaac-Reorient-Cube-Allegro", "emb_e9_leap", "inhand", (("palm", "palm"), ("object", "cube")),
             0.0, note="Menagerie LEAP hand (16 joints, as the Allegro) holding a cube; NOT an Allegro hand"),
)}


def _edit(root: ET.Element, task: HoldTask) -> None:
    """Drop the task's unused props (and what refers to them) and add its cube."""
    world = root.find("worldbody")
    for body in list(world):
        if body.tag == "body" and body.get("name") in task.drop:
            world.remove(body)
    kept = {b.get("name") for b in root.iter("body")}
    for contact in root.findall("contact"):
        for item in list(contact):
            if any(item.get(k) and item.get(k) not in kept for k in ("body1", "body2")):
                contact.remove(item)
    used = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    for asset in root.findall("asset"):
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") not in used:
                asset.remove(mesh)
    if task.cube is not None:
        h = CUBE["edge"] / 2
        body = ET.SubElement(world, "body", name="cube", pos=" ".join(map(str, task.cube)), quat="1 0 0 0")
        inertia = CUBE["mass"] * CUBE["edge"] ** 2 / 6.0
        ET.SubElement(body, "inertial", pos="0 0 0", mass=str(CUBE["mass"]), diaginertia=f"{inertia} {inertia} {inertia}")
        ET.SubElement(body, "joint", name="cube_free", type="free")
        ET.SubElement(body, "geom", name="cube", type="box", size=f"{h} {h} {h}",
                      friction=f"{CUBE['friction']} 0.005 0.0001")


def model(task: HoldTask, workdir: Path) -> tuple[Path, dict[str, float], dict[str, Any]]:
    """Write the task's baked MJCF into ``workdir``; returns (path, {"r/<actuator>": hold ctrl}, source record)."""
    import bake  # noqa: PLC0415 - needs mujoco
    import mujoco  # noqa: PLC0415
    src = LIBRARY / "robots" / task.pack / "scene.xml"
    pose, ctrl = bake.key_pose(src)
    root = ET.parse(src).getroot()
    _edit(root, task)
    out = workdir / f"{task.name}.xml"
    workdir.mkdir(parents=True, exist_ok=True)
    out.write_text(bake.bake_tree(root, src.parent, pose, ctrl))
    m = mujoco.MjModel.from_xml_path(str(out))
    hold = {f"r/{m.actuator(a).name}": float(m.key_ctrl[0, a]) for a in range(m.nu)}
    return out, hold, {"pack": task.pack, "file": f"library/robots/{task.pack}/scene.xml", "dropped": list(task.drop),
                       "cube": CUBE | {"pos": list(task.cube)} if task.cube else None, "baked": out.name,
                       "nbody": int(m.nbody - 1), "nv": int(m.nv), "nu": int(m.nu), "ngeom": int(m.ngeom)}


def score_spec(task: HoldTask) -> dict[str, Any]:
    """The task as ``pxs_score.score`` wants it (its ``TASKS`` entry for this scene)."""
    return {"kind": task.kind, "track": tuple((label, "r", body) for label, body in task.track),
            "floor_z": task.floor_z, "cube": (CUBE["edge"], task.cube_top) if task.cube else None}
