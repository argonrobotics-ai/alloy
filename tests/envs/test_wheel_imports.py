# SPDX-License-Identifier: Apache-2.0
"""The envs run from an installed wheel: nothing they import is left out of it (CPU, no simulation).

The release wheel holds the ``alloy`` package only, without the showcase clips and the paused task sets. Four painting
envs once built their FR3 from ``alloy.clips.arms`` and so raised ``ModuleNotFoundError`` from the wheel: the arm kit is
``alloy.arms`` now. Two checks keep it so: no module an env can reach (function-level imports included) imports a
package the wheel leaves out, and every such module imports, and the painting robots build, with ``alloy.clips``
blocked.
"""

from __future__ import annotations

import ast
import fnmatch
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
NOT_SHIPPED = ("alloy.clips*", "alloy.dcsolve*", "alloy.episode*", "alloy.tasks.dc_*", "alloy.tasks.job_*")
NOT_IN_WHEEL = ("bench", "tests", "tools", "scripts", "examples")      # repository folders beside the package
PAINT_ROBOTS = ("paint_fr3:roller_xml", "paint_fr3_brush:brush_xml", "paint_fr3_can:can_xml")

BLOCKED_RUN = """
import importlib, importlib.abc, sys

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "alloy.clips" or name.startswith("alloy.clips."):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked: not in the wheel)", name=name)

sys.meta_path.insert(0, Block())
modules, robots = sys.argv[1].split(","), sys.argv[2].split(",")
for m in modules:
    importlib.import_module(m)
for item in robots:
    module, fn = item.split(":")
    mod = importlib.import_module(f"alloy.tasks.{module}")
    assert "<mujoco" in getattr(mod, fn)() and mod.rig() is not None, item
from alloy.envs import REGISTRY
from alloy.envs.ports import PORTS
movers = [spec.mover() for spec in PORTS.values()]          # each port's action side imports its task modules
assert REGISTRY and len(movers) == len(PORTS) >= 28
assert not [m for m in sys.modules if m.startswith("alloy.clips")], "alloy.clips was imported"
print("ok", len(modules))
"""


def _file(module: str) -> Path | None:
    base = ROOT / module.replace(".", "/")
    return next((p for p in (base.with_suffix(".py"), base / "__init__.py") if p.is_file()), None)


def _named_tasks(tree: ast.AST) -> set[str]:
    """Port modules a family names as text and imports by name at run time (``_runner("pad_edge", ...)``)."""
    texts = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return {f"alloy.tasks.{t}" for t in texts if t.isidentifier() and _file(f"alloy.tasks.{t}")}


def imports_of(path: Path) -> set[str]:
    """Every module ``path`` imports, at module level or inside a function. Example: ``imports_of(Path(__file__))``."""
    tree, out = ast.parse(path.read_text()), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names if _file(f"{node.module}.{a.name}"))
    if "alloy/envs/ports" in path.as_posix():
        out |= _named_tasks(tree)
    return out


def reachable(start: str = "alloy.envs") -> dict[str, set[str]]:
    """Each first-party module reachable from ``start`` with what it imports. Example: ``reachable()["alloy.envs"]``."""
    seen: dict[str, set[str]] = {}
    todo = [start]
    while todo:
        module = todo.pop()
        path = _file(module)
        if module in seen or path is None or "/_vendor/" in path.as_posix():
            continue
        seen[module] = imports_of(path)
        todo.extend(m for m in seen[module] if m.startswith("alloy."))
    return seen


def _left_out(module: str) -> bool:
    return module.split(".")[0] in NOT_IN_WHEEL or any(fnmatch.fnmatch(module, p) for p in NOT_SHIPPED)


def test_no_env_reaches_a_module_the_wheel_leaves_out() -> None:
    graph = reachable()
    assert {"alloy.tasks.paint_fr3", "alloy.tasks.forklift_ga_park", "alloy.tasks.pad_edge"} <= set(graph)
    bad = {m: sorted(i for i in imps if _left_out(i)) for m, imps in graph.items()}
    bad = {m: i for m, i in bad.items() if i}
    assert not bad, f"modules the envs reach import packages that are not in the wheel: {bad}"


@pytest.mark.slow
def test_every_env_module_imports_and_the_paint_robots_build_without_the_clips() -> None:
    modules = sorted(m for m in reachable() if not _left_out(m))
    res = subprocess.run([sys.executable, "-c", BLOCKED_RUN, ",".join(modules), ",".join(PAINT_ROBOTS)], cwd=ROOT,
                         capture_output=True, text=True, timeout=900, check=False)
    assert res.returncode == 0 and f"ok {len(modules)}" in res.stdout, res.stderr[-3000:]
