# SPDX-License-Identifier: Apache-2.0
"""Lazy public names and the alloy.api namespace."""

from __future__ import annotations

import pkgutil

import pytest

import alloy
import alloy.api as api
from alloy.api import _lazy


def test_unknown_name_is_attribute_error():
    assert not hasattr(api, "no_such_name")
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        _lazy.resolve("nope")
    with pytest.raises(ImportError):
        exec("from alloy.api import nope", {})


def test_public_names_resolve_and_are_cached():
    ns: dict = {}
    getattr_, dir_ = _lazy.module_hooks(ns, "alloy")
    scene_cls = getattr_("Scene")
    assert ns["Scene"] is scene_cls
    assert set(_lazy.PUBLIC) <= set(dir_())
    for name in _lazy.PUBLIC:
        assert getattr(api, name) is not None


def test_no_public_name_is_a_subpackage():
    subpackages = {m.name for m in pkgutil.iter_modules(alloy.__path__)}
    assert not set(_lazy.PUBLIC) & subpackages


def test_api_init_is_standard_library_only():
    import ast
    from pathlib import Path

    for name in ("__init__.py", "_lazy.py", "_hooks.py"):
        tree = ast.parse((Path(api.__file__).parent / name).read_text())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        third_party = {m for m in mods if m and m.split(".")[0] not in ("__future__", "importlib", "sys",
                                                                          "threading", "collections", "typing",
                                                                          "alloy")}
        assert not third_party, (name, third_party)
        assert all(m in (None, "__future__") or not m.startswith("alloy") or m.startswith("alloy.api._")
                   for m in mods), (name, mods)


def test_public_names_survive_submodule_imports():
    """Importing every alloy.api module must not shadow a public function with a module of the same name."""
    import importlib
    import inspect

    for m in pkgutil.iter_modules(api.__path__):
        importlib.import_module(f"alloy.api.{m.name}")
    for name in _lazy.PUBLIC:
        assert not inspect.ismodule(getattr(api, name)), name
