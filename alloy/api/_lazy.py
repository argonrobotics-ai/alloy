# SPDX-License-Identifier: Apache-2.0
"""Lazy public names (PEP 562): ``alloy.<name>`` and ``alloy.api.<name>`` import their module on first use.

Keeps ``import alloy`` fast: nothing heavy (NumPy, Warp, Newton) loads until a name that needs it is used. Standard
library only. A resolved name is cached in the module's globals; an unknown name raises ``AttributeError`` (so
``hasattr`` is False and ``from alloy import nope`` is an ``ImportError``).

Use in a package ``__init__``::

    from alloy.api._lazy import PUBLIC as __all__, module_hooks
    __getattr__, __dir__ = module_hooks(globals(), "alloy")
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, MutableMapping
from typing import Any

#: public name -> module that defines it: under ``alloy.api``, or a full ``alloy.`` path (no name equals an
#: ``alloy`` subpackage)
EXPORTS: dict[str, str] = {
    "Camera": "alloy.camera.camera",
    "Scene": "scene",
    "load": "_load",
    "build": "_build",
    "Sim": "sim",
    "Trajectory": "trajectory",
    "estimate": "_estimate",
    "vary": "_vary",
    "materials": "catalog",
    "scenes": "catalog",
    "AlloyAPIError": "errors",
    "SceneError": "errors",
    "BuildError": "errors",
    "SimulationError": "errors",
    "UnknownNameError": "errors",
    "FeatureUnavailableError": "errors",
}

PUBLIC: list[str] = sorted(EXPORTS)


def resolve(name: str, owner: str = "alloy") -> Any:
    """The object a public name stands for. Example: ``resolve("Scene")`` is ``alloy.api.scene.Scene``."""
    mod = EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module {owner!r} has no attribute {name!r} (public names: {', '.join(PUBLIC)})")
    return getattr(importlib.import_module(mod if mod.startswith("alloy.") else f"alloy.api.{mod}"), name)


def module_hooks(namespace: MutableMapping[str, Any], owner: str) -> tuple[Callable[[str], Any], Callable[[], list]]:
    """``(__getattr__, __dir__)`` for a module whose globals are ``namespace``. Example: see the module docstring."""

    def __getattr__(name: str) -> Any:
        value = resolve(name, owner)
        namespace[name] = value
        return value

    def __dir__() -> list[str]:
        return sorted(set(namespace) | set(PUBLIC))

    return __getattr__, __dir__
