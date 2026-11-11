# SPDX-License-Identifier: Apache-2.0
"""The ``alloy.plugins`` registry: built-in plug-ins plus any installed package's ``alloy.plugins`` entry points.

A package publishes plug-ins in its ``pyproject.toml``::

    [project.entry-points."alloy.plugins"]
    my_drag = "my_package.drag:Drag"           # a Plugin subclass

``available()`` lists them, ``get(name)`` returns the class and ``create(name, **params)`` an instance. A target
string may also be ``"package.module:Class"`` or a path to a ``.py`` file (``load_target``).
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import logging
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from alloy.plugins.base import KINDS, Plugin
from alloy.plugins.errors import PluginError

log = logging.getLogger("alloy.plugins")

GROUP = "alloy.plugins"
BUILTIN_MODULES = ("alloy.plugins.library.magnet", "alloy.plugins.library.ink", "alloy.plugins.library.hydraulic",
                   "alloy.plugins.tactile.sensor", "alloy.plugins.library.suction", "alloy.plugins.tactile.gel",
                   "alloy.plugins.thermal.sensor", "alloy.plugins.sound.sensor", "alloy.plugins.library.fluid_network",
                   "alloy.plugins.library.adhesion", "alloy.plugins.magnetism.plugin", "alloy.plugins.sports.ball",
                   "alloy.plugins.sports.aero", "alloy.plugins.sports.net", "alloy.plugins.bonds.plugin",
                   "alloy.plugins.thermal.particle",      # heat v2's particle_thermal (SCR-WP7-8)
                   "alloy.plugins.thermal.grid",          # heat v3's grid_thermal: weld heat (SCR-WP8-14)
                   "alloy.plugins.library.syringe")       # syringes, vials, needles (SCR-WP8-11 A)


@dataclass(frozen=True)
class PluginInfo:
    """One registry row (JSON via ``dataclasses.asdict``)."""

    name: str
    kind: str
    summary: str
    source: str            # "builtin", "entry_point:<distribution>" or "registered"
    target: str            # "module:Class"


_REGISTRY: dict[str, tuple[type[Plugin], str]] = {}
_LOADED = {"builtin": False, "entry_points": False}


def _target(cls: type) -> str:
    return f"{cls.__module__}:{cls.__qualname__}"


def register(cls: type[Plugin], *, source: str = "registered", replace: bool = False) -> type[Plugin]:
    """Add a plug-in class under ``cls.name`` (usable as a decorator). Re-registering the same class is a no-op;
    another class with the same name raises unless ``replace``.

    Example::

        @register
        class Drag(ForcePlugin):
            name = "drag"
    """
    if not (inspect.isclass(cls) and issubclass(cls, Plugin)):
        raise PluginError(f"register() takes a Plugin subclass, got {cls!r}")
    if not cls.name or cls.kind not in KINDS:
        raise PluginError(f"{cls.__qualname__}: set name = '...' and subclass one of ForcePlugin, SensorPlugin, "
                          f"ObjectPlugin, SolverPlugin (kind {cls.kind!r})")
    old = _REGISTRY.get(cls.name)
    if old is not None and _target(old[0]) != _target(cls) and not replace:
        raise PluginError(f"plug-in name {cls.name!r} is taken by {_target(old[0])}; rename yours")
    _REGISTRY[cls.name] = (cls, source if old is None or replace else old[1])
    return cls


def _load_builtins() -> None:
    if _LOADED["builtin"]:
        return
    _LOADED["builtin"] = True
    for mod in BUILTIN_MODULES:
        module = importlib.import_module(mod)
        for cls in _plugin_classes(module):
            register(cls, source="builtin")


def load_entry_points(*, force: bool = False) -> list[str]:
    """Import every installed ``alloy.plugins`` entry point; returns the names registered. A broken entry point is
    logged and skipped (one bad package must not hide the others)."""
    if _LOADED["entry_points"] and not force:
        return []
    _LOADED["entry_points"] = True
    names = []
    for ep in metadata.entry_points(group=GROUP):
        dist = getattr(getattr(ep, "dist", None), "name", "?")
        try:
            obj = ep.load()
        except Exception as err:  # noqa: BLE001 - report and continue
            log.warning("alloy.plugins entry point %s (%s) failed to load: %s", ep.name, dist, err)
            continue
        for cls in _as_classes(obj):
            register(cls, source=f"entry_point:{dist}")
            names.append(cls.name)
    return names


def _ensure_loaded() -> None:
    _load_builtins()
    load_entry_points()


def available(kind: str | None = None) -> list[PluginInfo]:
    """Every registered plug-in (built-ins and entry points), sorted by name; ``kind`` filters.
    Example: ``[p.name for p in available("force")]``."""
    _ensure_loaded()
    if kind is not None and kind not in KINDS:
        raise PluginError(f"kind {kind!r} is not one of {KINDS}")
    rows = [PluginInfo(c.name, c.kind, c.summary, src, _target(c)) for c, src in _REGISTRY.values()]
    return sorted((r for r in rows if kind is None or r.kind == kind), key=lambda r: r.name)


def get(name: str) -> type[Plugin]:
    """The plug-in class registered as ``name``. Example: ``get("magnet")``."""
    _ensure_loaded()
    try:
        return _REGISTRY[name][0]
    except KeyError:
        raise PluginError(f"no plug-in {name!r}; available: {sorted(_REGISTRY)}. Install its package or pass "
                          f"'module:Class' / a .py path") from None


def create(name: str, config: Any = None, **params: Any) -> Plugin:
    """An instance of plug-in ``name``. Example: ``create("magnet", moments=[0.1, 0.1])``."""
    return get(name)(config, **params)


def _field_comments(config_cls: type) -> dict[str, str]:
    """``{field: comment}`` from ``name: type = default  # comment`` lines of the Config source."""
    try:
        lines = inspect.getsource(config_cls).splitlines()
    except (OSError, TypeError):
        return {}
    out = {}
    for line in lines:
        m = re.match(r"\s+(\w+)\s*:[^#]*#\s*(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def describe(cls: type[Plugin]) -> dict[str, Any]:
    """JSON description of a plug-in class: name, kind, summary, target and every Config field (type, default, the
    comment after it). Example: ``describe(get("magnet"))["config"][0]["name"] == "bodies"``."""
    notes = _field_comments(cls.Config)
    fields = []
    for f in dataclasses.fields(cls.Config):
        default = f.default if f.default is not dataclasses.MISSING else (
            f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
        fields.append({"name": f.name, "type": str(f.type), "default": _jsonable(default),
                       "doc": f.metadata.get("doc", notes.get(f.name, ""))})
    return {"name": cls.name, "kind": cls.kind, "summary": cls.summary, "target": _target(cls),
            "doc": inspect.getdoc(cls) or "", "config": fields}


def _jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return repr(x)


def _plugin_classes(module: Any) -> list[type[Plugin]]:
    """Concrete plug-in classes defined in ``module`` or its submodules (not imported from elsewhere)."""
    out = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        mine = obj.__module__ == module.__name__ or obj.__module__.startswith(f"{module.__name__}.")
        if issubclass(obj, Plugin) and mine and obj.name and obj.kind in KINDS:
            out.append(obj)
    return out


def _as_classes(obj: Any) -> list[type[Plugin]]:
    if inspect.isclass(obj) and issubclass(obj, Plugin):
        return [obj]
    if inspect.ismodule(obj):
        return _plugin_classes(obj)
    raise PluginError(f"entry point object {obj!r} is not a Plugin subclass or a module of them")


TARGET_FORMS = ("a registered name (python -m alloy.plugins list), 'module:Class', a plugin.py file, a package "
                "folder, or the project folder `python -m alloy.plugins new` made")


def load_target(target: str) -> list[type[Plugin]]:
    """Plug-in classes named by ``target``: a registry name, ``"module:Class"``, ``"module"``, a ``.py`` path, a
    package folder, or the project folder ``new`` made (``my_force`` holding ``my_force/my_force/plugin.py``).
    Example: ``load_target("my_force")`` right after ``python -m alloy.plugins new my_force``."""
    path = Path(target)
    if target.endswith(".py"):
        return _from_path(path)
    if path.is_dir():
        try:
            return _from_package(package_dir(path))
        except PluginError:
            _ensure_loaded()
            if target in _REGISTRY:                 # a folder that only shares a registered plug-in's name
                return [_REGISTRY[target][0]]
            raise
    if ":" in target:
        mod, _, attr = target.partition(":")
        return _as_classes(getattr(importlib.import_module(mod), attr))
    _ensure_loaded()
    if target in _REGISTRY:
        return [_REGISTRY[target][0]]
    try:
        return _plugin_classes(importlib.import_module(target))
    except ImportError:
        raise PluginError(f"{target!r} is not a registered plug-in, an importable module, a .py file or a folder "
                          f"in {Path.cwd()}; available: {sorted(_REGISTRY)}",
                          fix=f"pass {TARGET_FORMS}") from None


def package_dir(path: Path) -> Path:
    """The plug-in package folder a directory target means: the folder itself when it is a package (it has an
    ``__init__.py``), else the package inside a project folder: ``<dir>/<dir name>/`` as ``new`` lays it out, or its
    only sub-package. Example: ``package_dir(Path("my_force")) == Path("my_force/my_force")``."""
    if (path / "__init__.py").is_file():
        return path
    inner = path / path.resolve().name
    if (inner / "__init__.py").is_file():
        return inner
    subs = sorted(d for d in path.iterdir() if (d / "__init__.py").is_file() and d.name not in ("tests", "test"))
    if len(subs) == 1:
        return subs[0]
    found = f"it holds the packages {[d.name for d in subs]}" if subs else "it holds no package"
    raise PluginError(f"{path} is not a plug-in folder: {found} (expected {inner / 'plugin.py'}, as `python -m "
                      f"alloy.plugins new {path.resolve().name}` makes it)",
                      fix="pass the plugin.py file or the package folder, or run `new` first")


_PATH_PACKAGES: dict[str, Path] = {}       # packages _from_package imported by folder: name -> its __init__.py


def _from_package(pkg: Path) -> list[type[Plugin]]:
    """The plug-in classes of package folder ``pkg``, imported under its own name so that its imports of itself
    (``from my_force.plugin import MyForce``) resolve without installing it."""
    name, init = pkg.resolve().name, (pkg / "__init__.py").resolve()
    module = sys.modules.get(name)
    if module is not None and Path(getattr(module, "__file__", None) or "/").resolve() != init:
        if _PATH_PACKAGES.get(name) is None:
            raise PluginError(f"cannot load the plug-in package {pkg}: another module named {name!r} is already "
                              f"imported ({getattr(module, '__file__', 'built in')})",
                              fix=f"rename the package, or pass its file: {pkg / 'plugin.py'}")
        for key in [k for k in sys.modules if k == name or k.startswith(f"{name}.")]:
            del sys.modules[key]                    # a same-named package this function loaded from another folder
        module = None
    if module is None:
        _PATH_PACKAGES[name] = init
        spec = importlib.util.spec_from_file_location(name, init, submodule_search_locations=[str(init.parent)])
        if spec is None or spec.loader is None:
            raise PluginError(f"cannot import {init}", fix="check that the file is readable Python")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            _PATH_PACKAGES.pop(name, None)
            raise
    classes = _plugin_classes(module)
    if not classes and (pkg / "plugin.py").is_file():          # a package whose __init__ does not import its plug-in
        classes = _plugin_classes(importlib.import_module(f"{name}.plugin"))
    if not classes:
        raise PluginError(f"{pkg} defines no Plugin subclass with a name",
                          fix="define class X(ForcePlugin) with name = 'x' in its plugin.py")
    return classes


def _from_path(path: Path) -> list[type[Plugin]]:
    if not path.is_file():
        raise PluginError(f"{path} does not exist (looked in {Path.cwd()})", fix=f"pass {TARGET_FORMS}")
    name = f"alloy_plugin_{abs(hash(str(path.resolve()))):x}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PluginError(f"cannot import {path}", fix="check that the file is readable Python")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    classes = _plugin_classes(module)
    if not classes:
        raise PluginError(f"{path} defines no Plugin subclass with a name",
                          fix="define class X(ForcePlugin) with name = 'x' in it")
    return classes


__all__ = ["GROUP", "PluginInfo", "available", "create", "describe", "get", "load_entry_points", "load_target",
           "package_dir", "register"]
