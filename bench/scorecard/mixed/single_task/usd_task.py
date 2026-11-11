# SPDX-License-Identifier: Apache-2.0
"""Alloy scenes from Isaac Lab's own task assets: the like-for-like Alloy rows of the hold-task bench.

A task spec (JSON, written on the Isaac Lab side; ``load`` documents the keys it reads) holds what an Isaac Lab task loads (the robot's USD file, joint names, default pose, the
gains its actuators write, the rigid objects). ``load`` turns that dump into what ``alloy_hold.py`` needs:

- the robot: a flattened copy of the task's USD file (``prep_robot``) with, per joint, the task's drive gains and
  default position authored the way Newton's USD importer reads them (``UsdPhysics.DriveAPI`` stiffness / damping,
  ``newton:angular:position`` / ``newton:linear:position``, ``newton:armature``), and the task's self-collision switch
  on the articulation root (``newton:selfCollisionEnabled``; the importer's default is ON, ``_self_collision``). ``scene.add_robot`` then starts the robot at the task's
  default pose with the task's gains, through Alloy's public API and its stock USD import;
- props: cuboids as ``scene.add_box`` (a table: static); a USD or multi-asset object becomes ONE box of its
  bounding size in every world (said in ``notes``);
- a ground plane for floating-base robots. Rough terrain is NOT rebuilt: those tasks get flat ground (``notes``).

Isaac Lab caches remote assets under /tmp/https/<host>/<path>; ``local`` maps a URL to that file, so this runs on a VM
where the Isaac Lab side has already run the task. Needs ``usd-core`` and ``newton-usd-schemas`` in Alloy's venv.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEG = 180.0 / math.pi
PREP_VERSION = 2          # 2: self-collision switch and multi-axis joint armature carried (a prepared file of 1 is stale)
DEFAULT_OBJECT = {"size": [0.06, 0.06, 0.06], "mass": 0.2}      # stand-in for a multi-asset object


@dataclass
class UsdTask:
    isaac_task: str
    robot_usd: Path
    root_pos: tuple[float, float, float]
    root_quat_wxyz: tuple[float, float, float, float]
    hold: dict[str, float]                     # "r/<joint>" -> default position
    root_body: str                             # "r/<first body>"
    boxes: list[dict[str, Any]] = field(default_factory=list)
    ground: bool = False
    notes: list[str] = field(default_factory=list)
    source: dict[str, Any] = field(default_factory=dict)


def local(url: str) -> Path:
    """The Isaac Lab cache file of a remote asset URL (a local path is returned as it is)."""
    if "://" not in url:
        return Path(url)
    scheme, rest = url.split("://", 1)
    return Path("/tmp") / scheme / rest


def prep_robot(src: Path, art: dict[str, Any], out: Path) -> dict[str, Any]:
    """Write ``out``: ``src`` flattened, with the task's gains, default joint positions, armature and self-collision
    flag (module docstring). Returns what was authored."""
    from pxr import Sdf, Usd, UsdPhysics  # noqa: PLC0415

    Usd.Stage.Open(str(src)).Flatten().Export(str(out))
    stage = Usd.Stage.Open(str(out))
    idx = {n: i for i, n in enumerate(art["joint_names"])}
    ke, kd, q0 = art["joint_stiffness"], art["joint_damping"], art["default_joint_pos"]
    arm, done = art.get("joint_armature") or [0.0] * len(idx), []
    selfcol = _self_collision(stage, art.get("self_collisions"))
    for prim in stage.Traverse():
        angular = prim.IsA(UsdPhysics.RevoluteJoint)
        if not (angular or prim.IsA(UsdPhysics.PrismaticJoint)) or prim.GetName() not in idx:
            continue
        i, kind, f = idx[prim.GetName()], "angular" if angular else "linear", DEG if angular else 1.0
        drive = UsdPhysics.DriveAPI.Apply(prim, kind)
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(ke[i]) / f)            # USD angular gains are per degree
        drive.CreateDampingAttr(float(kd[i]) / f)
        drive.CreateTargetPositionAttr(float(q0[i]) * f)
        prim.CreateAttribute(f"newton:{kind}:position", Sdf.ValueTypeNames.Float).Set(float(q0[i]) * f)
        if arm[i]:
            prim.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(float(arm[i]))
        done.append(prim.GetName())
    multi = _multi_axis(stage, art)
    stage.GetRootLayer().Save()
    return {"joints_authored": len(done), "joints_missing": sorted(set(idx) - set(done)), **selfcol,
            "multi_axis_armature": multi, "prep_version": PREP_VERSION}


def _multi_axis(stage: Any, art: dict[str, Any]) -> list[str]:
    """Armature for multi-axis (D6) joints. The spec names their coordinates "<joint>:<k>"; the file has one
    PhysicsJoint prim "<joint>" whose drives already carry the gains, and its armature in a PhysX attribute the
    importer does not read. Without armature Alloy's GPU route went non-finite on Isaac Lab's Humanoid."""
    from pxr import Sdf, UsdPhysics  # noqa: PLC0415

    per: dict[str, float] = {}
    for n, a in zip(art["joint_names"], art.get("joint_armature") or [], strict=False):
        if ":" in n:
            joint = n.rsplit(":", 1)[0]
            per[joint] = max(per.get(joint, 0.0), float(a))
    done = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and per.get(prim.GetName()):
            prim.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(per[prim.GetName()])
            done.append(prim.GetName())
    return done


def _self_collision(stage: Any, task_flag: bool | None) -> dict[str, Any]:
    """Author the robot's self-collision switch where Newton's importer reads it (``newton:selfCollisionEnabled`` on
    the articulation root; the importer's default is ON). The task's value wins; without one the file's own PhysX
    switch (``physxArticulation:enabledSelfCollisions``, which the importer does not read) is carried over."""
    from pxr import Sdf, UsdPhysics  # noqa: PLC0415

    flag, source = task_flag, "task" if task_flag is not None else None
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            continue
        own = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
        if flag is None and own and own.HasAuthoredValue():
            flag, source = bool(own.Get()), "file"
        if flag is not None:
            prim.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(bool(flag))
    return {"self_collisions": flag, "self_collisions_from": source}


def _box(name: str, obj: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    """An ``add_box`` call for one Isaac Lab rigid object."""
    pose = obj["default_root_pose_xyz_quat_xyzw"]
    size, mass = obj.get("size"), obj.get("mass")
    if size is None:
        size, mass = DEFAULT_OBJECT["size"], mass or DEFAULT_OBJECT["mass"]
        notes.append(f"{name}: Isaac Lab spawns {obj['spawn_kind']} ({obj.get('usd_path') or 'several assets'}); "
                     f"Alloy uses one {size[0] * 100:g} cm box of {mass} kg in every world")
    kw: dict[str, Any] = {"name": name, "size": [float(x) for x in size], "pos": [float(x) for x in pose[:3]],
                          "quat_xyzw": [float(x) for x in pose[3:7]]}
    if obj.get("kinematic") or name == "table":
        kw["static"] = True
    elif mass:
        kw["mass"] = float(mass)
    if obj.get("static_friction"):
        kw["friction"] = float(obj["static_friction"])
    return kw


def load(spec_json: str | Path, workdir: Path) -> UsdTask:
    """The Alloy task for one task spec (the JSON described in the module docstring); the prepared robot file goes into ``workdir``."""
    d = json.loads(Path(spec_json).read_text())
    spec, task = d["spec"], d["task"]
    art, notes = spec["articulations"]["robot"], []
    if len(spec["articulations"]) > 1:
        notes.append(f"only 'robot' of the articulations {sorted(spec['articulations'])} is loaded")
    src = local(art["usd_path"])
    workdir.mkdir(parents=True, exist_ok=True)
    out, side = workdir / f"{task}.usdc", workdir / f"{task}.prep.json"
    old = json.loads(side.read_text()) if side.exists() else {}
    fresh = old.get("prep_version") == PREP_VERSION
    if out.exists() and fresh:      # the flattened copy needs no asset cache (/tmp is wiped on a VM restart)
        authored = {**old, "reused": True}
    else:
        if out.exists() and not src.exists():      # a stale prepared copy and no asset cache: prepare from the copy
            src = out.with_suffix(".stale.usdc")
            out.replace(src)
        authored = prep_robot(src, art, out)
        side.write_text(json.dumps(authored))
    pose = art["default_root_pose_xyz_quat_xyzw"]
    x, y, z, w = pose[3:7]
    floating = not art.get("fix_root_link") and "Velocity" in task or task in ("Isaac-Ant", "Isaac-Humanoid",
                                                                              "Isaac-Humanoid-Direct")
    if "Rough" in task:
        notes.append("rough terrain is not rebuilt: flat ground in Alloy")
    if art.get("disable_gravity"):
        notes.append("Isaac Lab switches gravity off for the robot's bodies; in Alloy they keep gravity")
    explicit = sorted({a for a in art["actuators"].values() if "Implicit" not in a})
    if explicit:
        notes.append(f"Isaac Lab actuators {explicit} (torque computed outside the solver); Alloy uses a PD drive with "
                     "the same gains")
    return UsdTask(task, out, tuple(float(v) for v in pose[:3]), (float(w), float(x), float(y), float(z)),
                   {f"r/{n}": float(q) for n, q in zip(art["joint_names"], art["default_joint_pos"], strict=True)},
                   f"r/{art['body_names'][0]}", [_box(n, o, notes) for n, o in spec["rigid_objects"].items()],
                   floating, notes, {"spec": Path(spec_json).name, "usd": art["usd_path"], **authored})


def add_to(scene: Any, t: UsdTask) -> None:
    """The task's robot (name "r") and props in an ``alloy.Scene``."""
    if t.ground:
        scene.add_ground()
    scene.add_robot("r", t.robot_usd, pos=t.root_pos, quat_wxyz=t.root_quat_wxyz)
    for b in t.boxes:
        scene.add_box(**b)


def score_spec(t: UsdTask, isaac_spec: dict[str, Any]) -> dict[str, Any]:
    """``pxs_score``'s task entry with its tracked entities as Alloy body names: ("robot", None) is the root body,
    ("robot", body) is "r/<body>", an object keeps its own name. A lift without a grasp is scored at rest."""
    track = tuple((label, "r", (t.root_body if body is None else f"r/{body}") if ent == "robot" else ent)
                  for label, ent, body in isaac_spec["track"])
    return {**isaac_spec, "track": track, "kind": "rest" if isaac_spec["kind"] == "grasp" else isaac_spec["kind"]}
