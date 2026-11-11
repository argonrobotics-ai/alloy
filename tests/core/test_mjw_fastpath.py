# SPDX-License-Identifier: Apache-2.0
"""SCR-WP8-5: MuJoCo Warp's two collision modules call ``upper_trid_index`` through a plain-Python twin
(alloy.core.mjw_fastpath). The twin equals the ``@wp.func`` everywhere it is used. It is installed in either import
order. In each rebound module the name is used in Python code only, never in a kernel or func; re-check this on every
MuJoCo Warp pin."""

from __future__ import annotations

import ast
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

import pytest

from alloy.core import mjw_fastpath

ROOT = Path(__file__).resolve().parents[2]
WARP_DECORATORS = {"kernel", "func", "func_native", "func_grad", "func_replay"}


def test_twin_matches_the_wp_func() -> None:
    from alloy._vendor.mujoco_warp._src import math as mjw_math
    from alloy._vendor.mujoco_warp._src.types import GeomType

    n = len(GeomType)
    for i in range(n):
        for j in range(n):
            assert mjw_fastpath.upper_trid_index(n, i, j) == mjw_math.upper_trid_index(n, i, j), (i, j)


@pytest.mark.parametrize("first", ["import alloy, alloy._vendor.mujoco_warp", "import alloy._vendor.mujoco_warp, alloy"])
def test_installed_in_either_import_order(first: str) -> None:
    code = (f"{first}; import sys; from alloy.core import mjw_fastpath as f; "
            "print(all(sys.modules[m].upper_trid_index is f.upper_trid_index for m in f.TARGETS))")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "True"


def _warp_decorated(node: ast.AST) -> bool:
    for dec in getattr(node, "decorator_list", ()):
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name in WARP_DECORATORS:
            return True
    return False


@pytest.mark.parametrize("module_name", mjw_fastpath.TARGETS)
def test_name_is_python_only_in_the_rebound_modules(module_name: str) -> None:
    """Every use of ``upper_trid_index`` in the module is a call from a plain Python function. A MuJoCo Warp bump that
    uses it inside a kernel or func would break: kernels read the same module globals."""
    tree = ast.parse(inspect.getsource(importlib.import_module(module_name)))
    calls = []

    def visit(node: ast.AST, in_warp: bool) -> None:
        in_warp = in_warp or _warp_decorated(node)
        if isinstance(node, ast.Name) and node.id == "upper_trid_index":
            calls.append(in_warp)
        for child in ast.iter_child_nodes(node):
            visit(child, in_warp)

    visit(tree, False)
    assert calls and not any(calls), f"{module_name}: upper_trid_index used inside a Warp kernel/func ({calls})"
