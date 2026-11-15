# SPDX-License-Identifier: Apache-2.0
"""The release inventory agrees with the vendored engines (physics 2026-11-14: after the Newton pin moved to 89bf98d1 in
bd390e3dc, tools/deps/pins.json still named 25b4603c, so tools/deps/wheels.py check_vendored refused every staged
tree: "vendored engine incomplete: alloy/_vendor/newton (lacks: pin_in_vendored_md)"). The gates run tests/ only, so
these are the release tools' own checks where every vendored change runs them (scripts/gate_select.py VENDOR_ALWAYS
and PATH_TESTS tools/deps/*): each vendored pin in pins.json is named on its VENDORED.md row with its path, the copy
has its package and LICENSE, the pins equal the code's pin constants, and Newton's patch list ends at its pin."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("alloy_deps_pins", ROOT / "tools" / "deps" / "pins.py")
pins = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pins)
INVENTORY = pins.load_pins(ROOT / "tools" / "deps" / "pins.json")
VENDORED = [d for d in INVENTORY["deps"] if d["kind"] == "vendored"]


def _const(rel: str, name: str) -> str:
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        target = (node.target if isinstance(node, ast.AnnAssign)
                  else node.targets[0] if isinstance(node, ast.Assign) else None)
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not in {rel}")


def test_each_vendored_pin_is_named_in_vendored_md_as_the_wheel_check_reads_it() -> None:
    assert {d["name"] for d in VENDORED} == {"newton", "mujoco-warp"}
    bad = {d["name"]: row for d in VENDORED if not (row := pins.vendored(d, ROOT / "alloy"))["match"]}
    assert bad == {}, (f"tools/deps/pins.json and alloy/_vendor/VENDORED.md disagree (tools/deps/wheels.py "
                       f"check_vendored refuses the release tree): {bad}")


def test_the_inventory_pins_are_the_code_pins_and_newtons_patches_end_at_its_pin() -> None:
    newton, mjw = (next(d for d in VENDORED if d["name"] == n) for n in ("newton", "mujoco-warp"))
    assert newton["pin"] == _const("alloy/core/newton_compat.py", "NEWTON_PIN")
    assert mjw["pin"] == _const("alloy/core/newton_compat.py", "MJWARP_PIN")
    assert newton["patches"][-1]["commit"] == newton["pin"]
    if (ROOT / "vendor/newton/PIN").is_file():
        pin_file = (ROOT / "vendor/newton/PIN").read_text()
        assert f"head {newton['pin']}" in pin_file and f"base {newton['base']}" in pin_file
