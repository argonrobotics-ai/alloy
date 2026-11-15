# SPDX-License-Identifier: Apache-2.0
"""The first ``alloy.build()`` of a process is freed like every later one.

Laws (CPU, each in a fresh interpreter, so this process is the first to import Newton): after ``del sim`` and
``gc.collect()`` a weakref to the first build's sim is dead, and to a second build's too; a frame that imports
Newton first keeps none of its locals once it returns. Before, the vendored Newton kept the ``ImportError`` of its
optional ``newton_usd_schemas`` package with the traceback; that traceback's frames led back through the import to
whichever frame imported Newton first (``alloy.build``, which keeps ``sim``), so the first build stayed in memory
for the life of the process, with its device arrays.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import alloy

ROOT = Path(alloy.__file__).resolve().parents[1]

BUILDS = """
import gc, weakref
import alloy

def scene():
    s = alloy.Scene(dt=0.01)
    s.add_ground()
    s.add_box("block", size=(0.1, 0.1, 0.1), pos=(0, 0, 0.3), mass=0.2)
    return s

refs = []
for _ in range(2):
    sim = alloy.build(scene(), device="cpu")
    sim.step(2)
    refs.append(weakref.ref(sim))
    del sim
    gc.collect()
print(["kept" if r() is not None else "freed" for r in refs])
"""

FIRST_IMPORT = """
import gc, weakref

class Held:
    pass

def first_import():
    held = Held()
    import alloy._vendor.newton  # noqa: F401
    return weakref.ref(held)

ref = first_import()
gc.collect()
print("kept" if ref() is not None else "freed")
"""


def _fresh(code: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}      # keep HOME / WARP_CACHE_PATH: the worker's own kernel cache
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip().splitlines()[-1]


@pytest.mark.compile_heavy
def test_the_first_build_is_freed_after_del():
    assert _fresh(BUILDS) == "['freed', 'freed']"      # was ['kept', 'freed']


def test_the_frame_that_imports_newton_first_keeps_no_locals():
    assert _fresh(FIRST_IMPORT) == "freed"
