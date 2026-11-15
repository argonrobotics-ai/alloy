# SPDX-License-Identifier: Apache-2.0
"""The precision layer's kernel launches name their device (WP3's stick-anchor flake, found in these hooks by WP13).

MuJoCo Warp's own launches name no device: Newton runs its step inside ``wp.ScopedDevice(model.device)``. A hook
called outside that scope (a unit law calling ``sort_constraints`` on a ``device="cpu"`` model, as
``test_mjw_order.py`` does) launched on Warp's default device, which on a machine with a GPU is ``cuda:0``, against
arrays on the CPU. Warp's default (RELAXED) array access does not check that and the launch is asynchronous, so a law
could read its CPU array before the GPU wrote it. Each launch now names the device of the data it writes, which
inside the solver's scope is the same device, so runs do not change.

Law: every ``wp.launch`` / ``wp.launch_tiled`` in ``alloy/precision`` passes ``device=`` (new hooks included).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import alloy.precision

DIR = Path(alloy.precision.__file__).parent
HOOKS = ("mjw_order.py", "mjw_rows.py", "mjw_sums.py", "mjw_jmove.py", "mjw_lazy_factor.py", "mjw_fixed.py")


def _launches(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) in
            ("launch", "launch_tiled") and getattr(getattr(n.func, "value", None), "id", None) == "wp"]


@pytest.mark.parametrize("name", sorted(p.name for p in DIR.glob("*.py")))
def test_every_launch_names_its_device(name):
    bare = [n.lineno for n in _launches(DIR / name) if not any(k.arg == "device" for k in n.keywords)]
    assert not bare, f"{name}: wp.launch without device= at lines {bare}"


def test_the_hooks_that_had_bare_launches_are_scanned():
    assert all(_launches(DIR / name) for name in HOOKS)
