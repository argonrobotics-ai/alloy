# SPDX-License-Identifier: Apache-2.0
"""The robot on a test stand: the scene a recording is replayed in.

``--robot`` is a shipped robot's name (``panda``, ``go2``, ``emb_e2_panda``, ...) or a robot file. The replay scene
holds that robot alone, as the recording must have been made: nothing in reach, and a robot with a free base (a
quadruped, a mobile base) hung so that nothing touches the ground. For an MJCF file ``bench_mjcf`` does that:

- a loose object of the file (a top-level body with a free joint and no actuated joint below it) is removed;
- a free-base robot (a top-level body with a free joint and actuated joints below it) is welded where it is,
  ``LIFT`` metres higher.

URDF and USD files are used as they are (their base is welded unless the file gives it a joint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from alloy.sysid.errors import SysidError

LIFT = 0.5          # m a welded free base is raised, so hanging legs clear the file's floor
NAME = "robot"      # the robot's name in the replay scene


def resolve(robot: str | Path) -> Path:
    """The robot file for a shipped robot's name or a path. Example: ``resolve("panda")`` -> ``.../scene.xml``."""
    from alloy.cards.library import default_root  # noqa: PLC0415

    path = Path(str(robot)).expanduser()
    if path.is_file():
        return path.resolve()
    root = default_root() / "robots"
    folders = sorted(p.name for p in root.iterdir() if (p / "scene.xml").is_file()) if root.is_dir() else []
    key = str(robot).lower()
    plain = [f for f in folders if not f.endswith(("_route", "_stiff"))]
    hits = [f for f in folders if f == key] or [f for f in plain if f.split("_", 2)[-1] == key] \
        or [f for f in plain if key in f.split("_")]
    if len(hits) == 1:
        return root / hits[0] / "scene.xml"
    names = sorted({f.split("_", 2)[-1] for f in plain})
    raise SysidError(f"robot {str(robot)!r} is neither a file nor one shipped robot" +
                     (f" (it matches {hits})" if hits else ""),
                     f"pass a robot file (MJCF, URDF or USD) or one of {names}", field="robot")


def bench_mjcf(path: Path) -> tuple[str, list[str]]:
    """``(MJCF text, notes)``: the file's robot alone on a stand (module docstring). File references are made
    absolute, so the text loads from anywhere. Example: ``bench_mjcf(resolve("go2"))[1]``."""
    tree = ET.parse(path)
    root = tree.getroot()
    _absolute_paths(root, path.parent)
    actuated = {a.get("joint") for a in root.iter() if a.get("joint") and _in(root, "actuator", a)}
    notes, gone = [], set()
    for world in root.findall("worldbody"):
        for body in list(world.findall("body")):
            free = [j for j in body if j.tag == "freejoint" or (j.tag == "joint" and j.get("type") == "free")]
            if not free:
                continue
            below = {j.get("name") for j in body.iter("joint")}
            if below & actuated:
                for j in free:
                    gone.add(j.get("name"))
                    body.remove(j)
                pos = [float(v) for v in (body.get("pos") or "0 0 0").split()]
                body.set("pos", f"{pos[0]} {pos[1]} {pos[2] + LIFT}")
                notes.append(f"free base {body.get('name')!r} welded {LIFT} m above its place in the file (record "
                             "with the robot hung so that nothing touches the ground)")
            else:
                gone |= {e.get("name") for e in body.iter() if e.get("name")}
                world.remove(body)
                notes.append(f"loose object {body.get('name')!r} left out")
    _drop_references(root, gone)
    return ET.tostring(root, encoding="unicode"), notes


def _in(root: ET.Element, section: str, element: ET.Element) -> bool:
    return any(element in list(s.iter()) for s in root.findall(section))


def _absolute_paths(root: ET.Element, folder: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    for key in ("meshdir", "texturedir", "assetdir"):
        given = compiler.get(key)
        if given is not None or key == "meshdir":
            compiler.set(key, str((folder / (given or "")).resolve()))
    for inc in root.iter("include"):
        if inc.get("file"):
            inc.set("file", str((folder / inc.get("file")).resolve()))


def _drop_references(root: ET.Element, gone: set[str | None]) -> None:
    gone.discard(None)
    for key in root.findall("keyframe"):
        root.remove(key)                      # key frames list every coordinate; the coordinates changed
    if not gone:
        return
    refs = ("joint", "joint1", "joint2", "body", "body1", "body2", "geom", "geom1", "geom2", "site", "objname")
    for section in ("equality", "contact", "sensor", "tendon", "actuator"):
        for parent in root.findall(section):
            for el in list(parent):
                if any(el.get(r) in gone for r in refs):
                    parent.remove(el)


def bench_scene(robot: str | Path, worlds: int, dt: float) -> tuple[Any, list[str]]:
    """``(scene, notes)``: an ``alloy.Scene`` of ``worlds`` copies of the robot on its stand, named ``"robot"``.

    Example: ``scene, notes = bench_scene("panda", worlds=64, dt=0.01)``."""
    import alloy  # noqa: PLC0415

    path = resolve(robot)
    scene = alloy.Scene(worlds=int(worlds), dt=float(dt))
    if path.suffix.lower() in (".xml", ".mjcf"):
        text, notes = bench_mjcf(path)
        scene.add_robot(NAME, mjcf=text)
    else:
        notes = []
        scene.add_robot(NAME, path)
    return scene, notes


__all__ = ["LIFT", "NAME", "bench_mjcf", "bench_scene", "resolve"]
