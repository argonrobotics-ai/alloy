# SPDX-License-Identifier: Apache-2.0
"""The laundry tasks run from an installed wheel (CPU, no simulation).

The release wheel leaves ``alloy.clips`` out. ``laundry_crossings`` once took its crossing count and its table depth
from ``alloy.clips.fold_checks`` inside two functions, so ``laundry_fold``, ``laundry_pinch`` and ``laundry_tee``
imported from the wheel and then raised ``ModuleNotFoundError`` at the first count. Those checks are
``alloy.tasks.fold_checks`` now, and the clips' name is the same module. Three laws keep it so: no module a laundry
task reaches (function-level imports included) imports a package the wheel leaves out; with ``alloy.clips`` blocked
every laundry task imports and counts crossings; the old name is the shipped module itself.
"""

from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
LAUNDRY = sorted(f"alloy.tasks.{p.stem}" for p in (ROOT / "alloy" / "tasks").glob("laundry_*.py"))

BLOCKED_RUN = """
import importlib, importlib.abc, sys
import numpy as np

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "alloy.clips" or name.startswith("alloy.clips."):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked: not in the wheel)", name=name)

sys.meta_path.insert(0, Block())
for m in sys.argv[1].split(","):
    importlib.import_module(m)
from alloy.tasks.fold_checks import table_penetration
from alloy.tasks.laundry_crossings import _count
q = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0.2, 0.2, -1], [0.2, 0.2, 1], [3, 3, 1]], dtype=float)
linked, apart = np.array([[0, 1, 2], [3, 4, 5]]), q + np.repeat([[0.0, 0, 0], [0, 0, 5]], 3, axis=0)
assert _count(q, linked) == 2 and _count(apart, linked) == 0      # linked: an edge of each through the other
assert abs(table_penetration(q[:3], 0.004) - 0.004) < 1e-12
assert not [m for m in sys.modules if m.startswith("alloy.clips")], "alloy.clips was imported"
print("ok", len(sys.argv[1].split(",")))
"""


def _wheel_walk() -> ModuleType:
    """OSS-16's import walk (``reachable``, ``_left_out``) from tests/envs/test_wheel_imports.py, loaded by path."""
    spec = importlib.util.spec_from_file_location("wheel_walk", ROOT / "tests" / "envs" / "test_wheel_imports.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_laundry_task_reaches_a_module_the_wheel_leaves_out() -> None:
    walk, graph = _wheel_walk(), {}
    for task in LAUNDRY:
        graph.update(walk.reachable(task))
    assert {"alloy.tasks.laundry_fold", "alloy.tasks.laundry_pinch", "alloy.tasks.laundry_tee",
            "alloy.tasks.laundry_crossings", "alloy.tasks.fold_checks"} <= set(graph)
    bad = {m: sorted(i for i in imps if walk._left_out(i)) for m, imps in graph.items()}
    bad = {m: i for m, i in bad.items() if i}
    assert not bad, f"modules the laundry tasks reach import packages that are not in the wheel: {bad}"


def test_laundry_tasks_import_and_count_crossings_without_the_clips() -> None:
    res = subprocess.run([sys.executable, "-c", BLOCKED_RUN, ",".join(LAUNDRY)], cwd=ROOT, capture_output=True,
                         text=True, timeout=600, check=False)
    assert res.returncode == 0 and f"ok {len(LAUNDRY)}" in res.stdout, res.stderr[-3000:]


def test_the_clips_name_is_the_shipped_module() -> None:
    old, new = importlib.import_module("alloy.clips.fold_checks"), importlib.import_module("alloy.tasks.fold_checks")
    assert old is new and old.self_intersections is new.self_intersections
