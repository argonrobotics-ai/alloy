# SPDX-License-Identifier: Apache-2.0
"""``alloy.load(source)`` -> a ready ``alloy.Sim``; ``alloy.Scene.load(source)`` -> an editable ``alloy.Scene``.

``source`` is a library scene name (``alloy.scenes()``), a scene card (.yaml), a Scene JSON file (``Scene.to_json``)
or a robot file (MJCF .xml, .urdf, .usd/.usdz): the robot on a ground plane at z = 0, or on its own floor when the MJCF
has a plane geom (MuJoCo Menagerie ``scene.xml`` files). The robot is named after the file stem. Another file (a mesh,
a binary) raises a SceneError naming what to pass instead. Example::

    sim = alloy.load("panda.xml")                      # build at once: GPU if present
    traj = sim.run(2.0)
    scene = alloy.Scene.load("demo_peg_d12")           # or edit first, then build
    scene.add_box("weight", size=(0.02, 0.02, 0.02), pos=(0, 0, 0.1), mass=0.05)
    sim = alloy.build(scene, worlds=16)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alloy.api import catalog
from alloy.api import _say
from alloy.api._items import ROBOT_KINDS
from alloy.api.errors import SceneError
from alloy.api.scene import Scene

ROBOT_SUFFIXES = tuple(ROBOT_KINDS)
SCENE_SUFFIXES = (".yaml", ".yml", ".json")


def load(source: str | Path | Scene | dict, *, worlds: int | None = None, device: str | None = None, seed: int = 0,
         profile: str = "standard") -> Any:
    """Load and build in one call: returns an ``alloy.Sim`` (arguments as in ``alloy.build``).

    Example: ``sim = alloy.load("arm.xml"); sim.run(1.0, record=["arm/hand"])``."""
    from alloy.api._build import build  # noqa: PLC0415

    scene = source if isinstance(source, (Scene, dict)) else load_scene(source)
    return build(scene, worlds=worlds, device=device, seed=seed, profile=profile)


def load_scene(source: str | Path, *, worlds: int | None = None) -> Scene:
    """An editable Scene from a library scene name, a scene card file, a Scene JSON file or a robot file
    (``alloy.Scene.load`` is the same). ``worlds`` overrides the card's world count.

    Example: ``scene = alloy.Scene.load("demo_pour", worlds=1)``."""
    path = Path(source).expanduser()
    suffix = path.suffix.lower()
    if suffix == ".json" and path.is_file():
        data = _json(path)
        if isinstance(data, dict) and data.get("format") == "urdf" and "groups" in data:
            raise SceneError(f"{path.name} is the URDF index alloy.io.export_scene writes (one .urdf per object and "
                             "its base pose), not a Scene file",
                             "turn it into a scene first: res = alloy.io.import_file(path, folder), then "
                             "alloy.load(res.scene); or load one of its .urdf files", field="source")
        scene = Scene.from_dict(data)
    elif suffix in ROBOT_SUFFIXES:
        scene = _robot_scene(path)
    elif suffix in (".yaml", ".yml") or (path.is_file() and not suffix):
        scene = _card_scene(path)
    elif path.is_file():
        raise _not_a_scene(path)
    else:
        scene = _library_scene(str(source))
    if worlds is not None:
        scene = Scene.from_dict({**scene.to_dict(), "worlds": worlds})
    return scene


def _robot_scene(path: Path) -> Scene:
    """The robot on a ground plane at z = 0, unless its MJCF (or a file it includes) has its own plane geom, as MuJoCo
    Menagerie ``scene.xml`` files do (their floor and table are part of the robot then)."""
    if not path.is_file():
        raise SceneError(f"robot file {str(path)!r} not found", "pass the path of an existing robot file "
                         f"({'/'.join(ROBOT_SUFFIXES)})", field="source")
    scene = Scene()
    if not has_floor(path):
        scene.add_ground()
    scene.add_robot(path.stem.replace(".", "_") if path.stem[:1].isalpha() else "robot", path)
    return scene


def has_floor(path: Path, depth: int = 0) -> bool:
    """True if an MJCF file (or a file it ``<include>``s) has a ``<geom type="plane">``. Example: a Menagerie
    ``scene.xml`` -> True, a bare ``panda.xml`` -> False. Non-MJCF files: False."""
    if path.suffix.lower() in (".usd", ".usda", ".usdc", ".usdz"):
        from alloy.io.usd_units import has_plane  # noqa: PLC0415

        return has_plane(path)          # a stage with its own colliding Plane prim; False without the USD packages
    if path.suffix.lower() not in (".xml", ".mjcf") or depth > 4:
        return False
    import xml.etree.ElementTree as ET  # noqa: PLC0415

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return False
    if any(g.get("type") == "plane" for g in root.iter("geom")):
        return True
    return any(has_floor(path.parent / inc.get("file", ""), depth + 1) for inc in root.iter("include")
               if inc.get("file"))


def _json(path: Path) -> dict[str, Any]:
    """A Scene JSON file's data; a SceneError for a file that is not JSON text."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SceneError(f"{path} is not a Scene JSON file ({exc})", "pass a file written by Scene.to_json() or "
                         "scene.save()", field="source") from exc


def _not_a_scene(path: Path, why: str = "") -> SceneError:
    """The SceneError for a file ``alloy.load`` cannot read: a mesh names the calls that take it, any other file the
    kinds ``alloy.load`` reads. Example: ``_not_a_scene(Path("part.stl")).fix`` names ``scene.add_mesh``."""
    from alloy.create.request import MESH_SUFFIXES  # noqa: PLC0415

    suffix = path.suffix.lower()
    if suffix in MESH_SUFFIXES:
        return SceneError(f"{path} is a mesh, not a scene or a robot", "place it in a scene with "
                          "scene.add_mesh(name, path) (.obj or .stl), or make it a library object with "
                          "alloy.create.create(mesh=path)", field="source")
    return SceneError(f"alloy.load cannot read {path}: {why or f'unknown suffix {suffix!r}'}",
                      "pass a library scene name (alloy.scenes()), a scene file or a robot file "
                      f"({'/'.join(SCENE_SUFFIXES + ROBOT_SUFFIXES)})", field="source")


def _library_scene(name: str) -> Scene:
    path = catalog.card_path("scenes", name)
    if path is None:
        raise SceneError(f"no scene file or library scene named {name!r}",
                         f"pass a .yaml/.xml path, or a library scene: {catalog.near('scenes', name)}",
                         field="source")
    return _card_scene(path)


def _card_scene(path: Path) -> Scene:
    """A Scene that places the card file's objects and robots (their cards stay in the file's library root)."""
    if not path.is_file():
        raise SceneError(f"scene file {str(path)!r} not found", "pass an existing scene card (.yaml)", field="source")
    from alloy.cards.units_yaml import yaml_load  # noqa: PLC0415
    from alloy.core.errors import CardError  # noqa: PLC0415

    try:
        data = yaml_load(path.read_text(encoding="utf-8"), path.stem)
    except UnicodeDecodeError as exc:
        raise _not_a_scene(path, "it is not a text file") from exc
    except CardError as exc:
        raise SceneError(f"{path} is not a readable scene card: {exc.message}", "fix the YAML at the place the "
                         "message names, or pass another scene card", field="source") from exc
    if not isinstance(data, dict) or data.get("schema") != "alloy.scene/1":
        raise SceneError(f"{path} is not a scene card (schema alloy.scene/1)", "pass a scene card; object cards "
                         "are placed with Scene.add_object", field="source")
    root = (path.parent.parent if path.parent.name == "scenes" else path.parent).absolute()  # cwd may change later
    scene = Scene(worlds=int(data.get("worlds", 1)), dt=float(data.get("control_dt", 0.02)))
    scene.roots = [str(root)] if root.resolve() != catalog.library_root().resolve() else []
    seen: dict[str, int] = {}
    for placement in data.get("objects") or []:
        card = str(placement["object"])
        k = seen.get(card, 0)
        seen[card] = k + 1
        name = card if k == 0 else f"{card}#{k}"          # the engine's own name for a repeated card
        scene.items[name] = {"kind": "card", "card_name": card, "static": None, "per_world_pos": None,
                             "pose": None, "randomize": [], "raw_placement": dict(placement)}
    for robot in data.get("robots") or []:
        scene.items[str(robot["name"])] = {"kind": "robot", "robot_kind": robot["kind"], "path": robot["path"],
                                           "mjcf": None, "pose": None, "raw": dict(robot)}
    scene.extra = {k: data[k] for k in ("pairs", "attachments", "combine", "inertia_floor") if k in data}
    if data.get("emitters"):
        _card_pours(scene, path)
    return scene


def _card_pours(scene: Scene, path: Path) -> None:
    """A scene card's ``emitters`` as ``Scene.add_pour`` streams (the engine does not run card emitters itself, so
    a card pour would build with no liquid). Sinks are dropped: the escape box retires liquid that
    leaves the scene. Raises for an emitter that is not a downward stream."""
    import math  # noqa: PLC0415

    from alloy.cards.loader import load_card  # noqa: PLC0415
    from alloy.cards.types import SceneCard  # noqa: PLC0415

    count: dict[str, int] = {}
    for e in load_card(path, SceneCard).emitters:
        if e.kind != "emitter":
            _say.log.info("%s: a card sink is not run; liquid that leaves the scene is retired by the escape box", path)
            continue
        vx, vy, vz = (float(v) for v in e.velocity)
        if math.hypot(vx, vy) > 1e-9 or vz >= 0.0 or e.stop_s is None:
            raise SceneError(f"{path}: emitter of {e.material} is not a finite downward stream (velocity "
                             f"{e.velocity}, stop_s {e.stop_s})", "give it velocity [0, 0, -v] and stop_s, or pour "
                             "with Scene.add_pour(name, volume, pos)", field="emitters")
        k = count.get(e.material, 0)
        count[e.material] = k + 1
        volume = e.rate * (float(e.stop_s) - float(e.start_s))
        scene.add_pour(f"{e.material}_pour_{k}", volume, pos=tuple(e.pose[:3]), rate=e.rate, speed=-vz,
                       material=e.material)
        if e.start_s:
            _say.warn("%s_pour_%d starts at t 0, not at start_s %.3g s (Scene.add_pour has no delay)", e.material,
                        k, e.start_s)
