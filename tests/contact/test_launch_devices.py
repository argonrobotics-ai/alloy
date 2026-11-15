# SPDX-License-Identifier: Apache-2.0
"""The contact layer's MuJoCo Warp hooks name the device of every kernel launch (the stick-anchor xdist flake).

MuJoCo Warp's own launches name no device: Newton runs its step inside ``wp.ScopedDevice(model.device)``. A hook
called outside that scope (a unit law calling ``launch_anchor`` directly) launched on Warp's default device, which on
a machine with a GPU is ``cuda:0``, against arrays on the CPU. Warp's default (RELAXED) array access does not check
that, the launch is asynchronous, and the law read its CPU array before the GPU wrote it: the anchor's pull read 0.0
(DC5's L4, 7 xdist workers sharing the GPU; every CPU-only host passes). Each launch now names the device of the data
it writes, which inside the solver's scope is the same device.

Law: every ``wp.launch`` / ``wp.launch_tiled`` in ``alloy/contact`` passes ``device=`` (new hooks included).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import alloy.contact

DIR = Path(alloy.contact.__file__).parent
HOOKS = ("mjw_free_gyro.py", "mjw_friction_cap.py", "mjw_stick_anchor.py", "noslip.py")


def _launches(path: Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) in
            ("launch", "launch_tiled") and getattr(getattr(n.func, "value", None), "id", None) == "wp"]


@pytest.mark.parametrize("name", sorted(p.name for p in DIR.glob("*.py")))
def test_every_launch_names_its_device(name):
    bare = [n.lineno for n in _launches(DIR / name) if not any(k.arg == "device" for k in n.keywords)]
    assert not bare, f"{name}: wp.launch without device= at lines {bare}"


def test_the_flake_modules_are_scanned():
    assert all(_launches(DIR / name) for name in HOOKS)
