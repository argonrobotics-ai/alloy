# SPDX-License-Identifier: Apache-2.0
"""The ``scene`` argument of every tool: a library scene name, a file, Scene JSON, or a recipe (JSON calls of
``alloy.Scene.add_*``), turned into an ``alloy.Scene``. No engine import.

A recipe::

    {"worlds": 4, "dt": 0.01, "liquid_accuracy": "super",       # optional: liquid particles 2 mm apart (default 4)
     "base": "demo_peg_d12",                                   # optional: start from a library scene or a file
     "materials": [{"name": "rubber_pad", "density": 1100, "friction": 1.2}],
     "objects": [{"add": "ground"},
                 {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1], "pos": [0, 0, 0.5], "mass": 1.0},
                 {"add": "robot", "name": "arm", "path": "arm.xml"}],
     "friction": [{"a": "cube", "b": "ground", "mu": 0.9}],
     "randomize": [{"name": "cube", "field": "x", "lo": -0.01, "hi": 0.01}]}

Each object's keys are the keyword arguments of ``Scene.add_<add>`` (``alloy ops scene`` lists the kinds).
Example: ``scene = to_scene({"objects": [{"add": "ground"}, {"add": "sphere", "name": "ball", "radius": 0.05,
"pos": [0, 0, 0.3]}]})``.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alloy.api import catalog
from alloy.api.errors import SceneError
from alloy.api.scene import Scene

RECIPE_KEYS = ("schema", "base", "worlds", "dt", "liquid_accuracy", "materials", "objects", "friction", "randomize")
#: recipe "add" kinds -> Scene method
ADDERS: dict[str, str] = {k: f"add_{k}" for k in (
    "ground", "box", "sphere", "cylinder", "capsule", "mesh", "open_box", "open_cylinder", "block_with_hole",
    "cloth", "liquid", "rod", "robot", "object")}
SCENE_SCHEMA = "alloy.api.scene/1"


def to_scene(value: Any, *, worlds: int | None = None, library: Sequence[str] = (),
             liquid_accuracy: str | None = None) -> Scene:
    """An ``alloy.Scene`` from a name, a path (a .json file may hold Scene JSON or a recipe), Scene JSON
    (``Scene.to_dict()``) or a recipe dict; ``worlds`` overrides the scene's world count, ``library`` adds card
    folders searched first and ``liquid_accuracy`` ("default" / "super") sets the scene's liquid particle size.

    Example: ``to_scene("demo_peg_d12", worlds=2).worlds == 2``."""
    if isinstance(value, Scene):
        scene = value
    elif isinstance(value, str):
        scene = _from_text(value, library)
    elif isinstance(value, dict) and value.get("schema") == SCENE_SCHEMA:
        scene = Scene.from_dict(value)
    elif isinstance(value, dict):
        scene = from_recipe(value, library)
    else:
        raise SceneError(f"scene must be a name, a path or a JSON object, got {type(value).__name__}",
                         'pass scene="demo_peg_d12" (alloy list scenes), a .yaml/.xml/.json path, or a recipe '
                         '{"objects": [{"add": "ground"}, ...]}', field="scene")
    if library:
        scene.roots = list(dict.fromkeys([*map(str, library), *scene.roots]))
    if worlds is not None and worlds != scene.worlds:
        scene = Scene.from_dict({**scene.to_dict(), "worlds": int(worlds)})
    if liquid_accuracy is not None and liquid_accuracy != scene.liquid_accuracy:
        scene = Scene.from_dict({**scene.to_dict(), "liquid_accuracy": liquid_accuracy})
    return scene


def _from_text(text: str, library: Sequence[str]) -> Scene:
    path = Path(text).expanduser()
    if path.suffix.lower() == ".json" and path.is_file():
        data = _read_json(path)
        return Scene.from_dict(data) if data.get("schema") == SCENE_SCHEMA else from_recipe(data, library)
    if not path.exists() and library:
        hit = catalog.card_path("scenes", text, library)
        if hit is not None:
            path = hit
    return Scene.load(path if path.exists() else text)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SceneError(f"{path} is not valid JSON ({exc.msg} at line {exc.lineno})", "fix the JSON in the file",
                         field="scene") from None
    if not isinstance(data, dict):
        raise SceneError(f"{path} holds a {type(data).__name__}, not a scene", 'write a recipe {"objects": [...]} '
                         "or Scene JSON (Scene.to_json()) to the file", field="scene")
    return data


def from_recipe(recipe: dict[str, Any], library: Sequence[str] = ()) -> Scene:
    """Run a recipe (see the module docstring) and return the Scene. Errors name the entry and what to change.

    Example: ``from_recipe({"worlds": 2, "objects": [{"add": "ground"}]}).names() == ["ground"]``."""
    unknown = [k for k in recipe if k not in RECIPE_KEYS]
    if unknown:
        raise SceneError(f"recipe has unknown key(s) {unknown}", f"use only {list(RECIPE_KEYS)}", field=unknown[0])
    scene = _base(recipe, library)
    for i, row in enumerate(_rows(recipe, "materials")):
        _call(scene, "add_material", row, f"materials[{i}]")
    for i, row in enumerate(_rows(recipe, "objects")):
        add_object(scene, row, f"objects[{i}]")
    for i, row in enumerate(_rows(recipe, "friction")):
        _call(scene, "set_friction", row, f"friction[{i}]")
    for i, row in enumerate(_rows(recipe, "randomize")):
        _call(scene, "randomize", row, f"randomize[{i}]")
    return scene


def _base(recipe: dict[str, Any], library: Sequence[str]) -> Scene:
    worlds, dt, liquid = recipe.get("worlds"), recipe.get("dt"), recipe.get("liquid_accuracy")
    if recipe.get("base") is not None:
        scene = to_scene(recipe["base"], worlds=worlds, library=library, liquid_accuracy=liquid)
        if dt is not None:
            scene.dt = float(dt)
        return scene
    kwargs: dict[str, Any] = {k: v for k, v in (("worlds", worlds), ("dt", dt), ("liquid_accuracy", liquid))
                              if v is not None}
    try:
        return Scene(**kwargs, roots=[str(r) for r in library])   # recipe objects find --library cards
    except TypeError as exc:
        raise SceneError(f"recipe worlds/dt: {exc}", "pass worlds as an integer and dt in seconds", field="worlds") \
            from None


def _rows(recipe: dict[str, Any], key: str) -> list[dict[str, Any]]:
    rows = recipe.get(key) or []
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise SceneError(f"recipe {key!r} must be a list of objects", f'pass "{key}": [{{...}}, ...]', field=key)
    return rows


def add_object(scene: Scene, row: dict[str, Any], where: str = "object") -> Any:
    """Apply one recipe row ``{"add": kind, ...keyword arguments}`` to ``scene``.

    Example: ``add_object(scene, {"add": "box", "name": "cube", "size": [0.1, 0.1, 0.1], "mass": 1.0})``."""
    kind = row.get("add")
    if kind not in ADDERS:
        raise SceneError(f"{where}: unknown add {kind!r}", f"use one of {sorted(ADDERS)}", field=f"{where}.add")
    args = {k: v for k, v in row.items() if k != "add"}
    return _call(scene, ADDERS[kind], args, where)


def _call(scene: Scene, method: str, args: dict[str, Any], where: str) -> Any:
    fn = getattr(scene, method)
    params = inspect.signature(fn).parameters
    allowed = [n for n, p in params.items() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        allowed += [n for n in inspect.signature(Scene.add_sphere).parameters if n not in ("self", "name", "radius")]
    unknown = [k for k in args if k not in allowed]
    if unknown:
        raise SceneError(f"{where}: {method} has no argument(s) {unknown}", f"use only {allowed}",
                         field=f"{where}.{unknown[0]}")
    missing = [n for n, p in params.items() if p.default is p.empty and n not in args
               and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
    if missing:
        raise SceneError(f"{where}: {method} needs {missing}", f"add {', '.join(repr(m) for m in missing)} to the "
                         "entry", field=f"{where}.{missing[0]}")
    try:
        return fn(**args)
    except SceneError as exc:
        raise SceneError(f"{where}: {exc.message}", exc.fix, field=f"{where}.{exc.field}" if exc.field else where) \
            from None


def describe_adders() -> dict[str, list[str]]:
    """Recipe "add" kinds and the arguments each takes (for help texts and the tool schema).

    Example: ``describe_adders()["box"][:3] == ["name", "size", "half_extents"]``."""
    out = {}
    for kind, method in ADDERS.items():
        params = inspect.signature(getattr(Scene, method)).parameters
        names = [n for n, p in params.items() if n != "self" and p.kind is not p.VAR_KEYWORD]
        if any(p.kind is p.VAR_KEYWORD for p in params.values()):
            names += [n for n in inspect.signature(Scene.add_sphere).parameters
                      if n not in ("self", "name", "radius", *names)]
        out[kind] = names
    return out
