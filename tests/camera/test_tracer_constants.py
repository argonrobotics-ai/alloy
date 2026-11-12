# SPDX-License-Identifier: Apache-2.0
"""GEO P5 (2026-11-11): the first camera render printed a flood of clang -Wconstant-conversion warnings (hundreds on
macOS). Warp compiles an integer above INT32_MAX as `const wp::int32 v = 3545902487;` before any cast, unless it is a
typed constant: `wp.constant(wp.uint32(...))` compiles to `3545902487u`, the same bits. Static check over the camera
package: no such literal inside a kernel, and no untyped `wp.constant` above INT32_MAX."""

from __future__ import annotations

import ast
from pathlib import Path

CAMERA = Path(__file__).resolve().parents[2] / "alloy" / "camera"
INT32_MAX = 0x7FFFFFFF


def _is_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == name \
        and isinstance(node.func.value, ast.Name) and node.func.value.id == "wp"


def _int_value(node: ast.AST) -> int | None:
    """The value of a constant integer expression (literals and + - * << >> & | ^), else None."""
    try:
        code = compile(ast.Expression(node), "<const>", "eval")
        if any(isinstance(n, (ast.Name, ast.Call, ast.Attribute)) for n in ast.walk(node)):
            return None
        value = eval(code, {"__builtins__": {}})  # noqa: S307 - literals and operators only (checked above)
    except Exception:  # noqa: BLE001
        return None
    return value if isinstance(value, int) else None


def overflowing(path: Path) -> list[str]:
    """``file:line`` of every integer above INT32_MAX that Warp would compile as an int32 constant."""
    tree = ast.parse(path.read_text())
    typed = set()                                  # wp.uint32(...) directly inside wp.constant(...): fine
    out = []
    for node in ast.walk(tree):
        if _is_call(node, "constant") and node.args:
            arg = node.args[0]
            if _is_call(arg, "uint32") or _is_call(arg, "uint64"):
                typed.add(id(arg))
            else:
                v = _int_value(arg)
                if v is not None and v > INT32_MAX:
                    out.append(f"{path.name}:{node.lineno} untyped wp.constant {v}")
    for node in ast.walk(tree):
        if _is_call(node, "uint32") and id(node) not in typed and node.args:
            v = _int_value(node.args[0])
            if v is not None and v > INT32_MAX:
                out.append(f"{path.name}:{node.lineno} inline wp.uint32({v:#x})")
    return out


def test_no_int32_overflowing_constant_in_the_camera_kernels():
    found = [f for p in sorted(CAMERA.rglob("*.py")) for f in overflowing(p)]
    assert found == [], found


def test_the_check_sees_both_forms(tmp_path):
    bad = tmp_path / "k.py"
    bad.write_text("import warp as wp\nA = wp.constant(0xDEADBEEF + 1)\nB = wp.constant(wp.uint32(0xDEADBEEF))\n"
                   "@wp.func\ndef f(x: wp.uint32) -> wp.uint32:\n    return x * wp.uint32(0xD35A2D97) + wp.uint32(7)\n")
    assert overflowing(bad) == ["k.py:2 untyped wp.constant 3735928560", "k.py:6 inline wp.uint32(0xd35a2d97)"]
