# SPDX-License-Identifier: Apache-2.0
"""Every Warp kernel launch in Alloy's own code names its device (physics 2026-11-14, after 9f68f9be2).

A ``wp.launch`` without ``device=`` runs on Warp's current device. One called outside Newton's ScopedDevice ran on
cuda:0 against CPU arrays (Warp's RELAXED access does not catch that), and its async read saw 0.0, only on L4 hosts
under xdist. So every ``wp.launch`` / ``wp.launch_tiled`` under alloy/ names ``device=`` (or passes it as the 7th
positional argument). alloy/_vendor is upstream's code and is not checked. A launch that must take the current device
says why on one of its own lines: ``# launch-device: <reason>``."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("launch", "launch_tiled")
DEVICE_POSITION = 6                      # wp.launch(kernel, dim, inputs, outputs, adj_inputs, adj_outputs, device, ...)
WAIVER = re.compile(r"#\s*launch-device:(.*)$")


def _warp_names(tree: ast.AST) -> tuple[set[str], set[str]]:
    """The names this module binds to the warp module (``import warp as wp``) and to its launch functions."""
    mods, funcs = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.asname or a.name for a in node.names if a.name == "warp")
        elif isinstance(node, ast.ImportFrom) and node.module == "warp":
            funcs.update(a.asname or a.name for a in node.names if a.name in NAMES)
    return mods, funcs


def _is_launch(call: ast.Call, mods: set[str], funcs: set[str]) -> bool:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr in NAMES and isinstance(f.value, ast.Name) and f.value.id in mods
    return isinstance(f, ast.Name) and f.id in funcs


def _names_device(call: ast.Call) -> bool:
    if any(k.arg == "device" for k in call.keywords):
        return True
    return len(call.args) > DEVICE_POSITION and not any(isinstance(a, ast.Starred) for a in call.args)


def bare_launches(source: str, rel: str) -> list[str]:
    """``rel:line`` of each launch in ``source`` that names no device and has no waiver with a reason."""
    tree, lines = ast.parse(source), source.splitlines()
    mods, funcs = _warp_names(tree)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_launch(node, mods, funcs) or _names_device(node):
            continue
        span = lines[node.lineno - 1:node.end_lineno]
        if any((m := WAIVER.search(line)) and m.group(1).strip() for line in span):
            continue
        out.append(f"{rel}:{node.lineno}")
    return out


def test_every_warp_launch_in_alloy_names_its_device() -> None:
    bad = []
    for path in sorted((ROOT / "alloy").rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if not rel.startswith("alloy/_vendor/"):
            bad += bare_launches(path.read_text(encoding="utf-8"), rel)
    assert bad == [], (f"{len(bad)} Warp launches name no device (pass device= the arrays' device, or say why on the "
                       f"call: '# launch-device: <reason>'):\n  " + "\n  ".join(bad))


CASES = '''
import warp as wp
import warp as w
from warp import launch as go

def f(k, d, kw, a):
    wp.launch(k, dim=1, inputs=[])                                    # 7 bare
    wp.launch(k, dim=1, inputs=[], device=d)
    w.launch_tiled(k, dim=1, inputs=[], block_dim=64)                 # 9 bare (another alias)
    go(k, 1)                                                          # 10 bare (from-import)
    wp.launch(k, 1, [], [], [], [], d)
    wp.launch(k, dim=1, **kw)                                         # 12 bare (a splat may not carry it)
    wp.launch(k, 1, *a)                                               # 13 bare
    wp.launch(k, dim=1,
              inputs=[])  # launch-device: the caller's ScopedDevice picks it (test)
    wp.launch(k, dim=1, inputs=[])  # launch-device:
    graph.launch(k)
    "wp.launch(k, dim=1) in a string is not a call"
'''


def test_the_scan_finds_bare_launches_and_takes_waivers_with_a_reason() -> None:
    """Each bare form is planted on a numbered line; line 16 is a waiver without its reason."""
    assert bare_launches(CASES, "x.py") == ["x.py:7", "x.py:9", "x.py:10", "x.py:12", "x.py:13", "x.py:16"]
