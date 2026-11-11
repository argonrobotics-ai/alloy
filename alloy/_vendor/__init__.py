# SPDX-License-Identifier: Apache-2.0
"""Alloy's bundled engines: Newton (alloy._vendor.newton) and MuJoCo Warp (alloy._vendor.mujoco_warp), Apache-2.0,
at Alloy's pinned commits with Alloy's fixes. Upstream repositories, base commits, every patch and every change the
vendoring made are listed in VENDORED.md beside this file; each package keeps its own LICENSE and file headers.

Alloy imports these copies only (``from alloy._vendor import newton``), so installing Alloy never touches a newton or
mujoco_warp package of your own."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings
from types import ModuleType

ENGINES = ("newton", "mujoco_warp")
# What the vendored packages report as __version__ (pip's versions for the pins; tests check they agree).
VERSIONS = {"newton": "1.8.0rc1", "mujoco_warp": "3.17.0"}

# Newton's SolverMuJoCo compares the installed `newton` distribution's declared MuJoCo pins with the loaded ones. The
# vendored copy has no distribution, so the only `newton` it can find is a user's own: never Alloy's business.
warnings.filterwarnings("ignore", message="MuJoCo dependency version mismatch with Newton's declared requirements")


class _AliasLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolves ``newton`` / ``mujoco_warp`` (and submodules) to the vendored modules: the import returns the very
    module object Alloy uses, so classes are shared, never duplicated."""

    def find_spec(self, fullname: str, path: object = None, target: object = None):
        if fullname.split(".", 1)[0] not in ENGINES:
            return None
        return importlib.machinery.ModuleSpec(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        # the import system re-reads sys.modules after exec_module, so the import yields the vendored module
        sys.modules[module.__name__] = importlib.import_module(f"{__name__}.{module.__name__}")


def alias_top_level_names() -> None:
    """Opt-in, for running unchanged third-party scripts on Alloy's engines (alloy.xbench's engine shim): after this,
    ``import newton`` / ``import mujoco_warp`` in this process give the vendored copies. Alloy itself never needs it.
    Refuses when a top-level newton or mujoco_warp is already imported here (two copies would mix)."""
    loaded = sorted(m for m in sys.modules if m.split(".", 1)[0] in ENGINES)
    if loaded:
        raise ImportError(f"alias_top_level_names: {loaded[:3]} already imported from another copy in this process")
    if not any(isinstance(f, _AliasLoader) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasLoader())
