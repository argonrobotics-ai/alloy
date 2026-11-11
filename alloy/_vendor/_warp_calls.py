# SPDX-License-Identifier: Apache-2.0
"""The SCR-WP0-COEXIST rewrite of vendored engine files (tooling: pure ``ast``, no Warp; ``scripts/vendor_engines.py``
loads it by path). Why: ``alloy._vendor._warp``.

Three kinds of site, each edited in place (line numbers never move):
- ``wp.kernel(..., module=...)`` (decorator or call form), and ``wp.kernel(**options)`` (the wrapper then sees
  whatever module the options hold), becomes ``_alloy_warp.kernel(...)``;
- ``wp.func(..., module=X)`` keeps its callee and reads ``module=_alloy_warp.module_name(X)``;
- ``wp.get_module(X)`` reads ``wp.get_module(_alloy_warp.module_name(X))``.
The file's top-level ``import warp as wp`` statement gains ``; from alloy._vendor import _warp as _alloy_warp`` at its
end (before any comment on the line). Rerunning changes nothing. Forms the rewrite cannot see into raise
``CallsError`` (so ``--check`` fails): ``*args`` on these calls, ``**kwargs`` on ``func`` / ``get_module``,
``from warp import kernel`` (or func, get_module), a ``module=`` / ``**kwargs`` call through ``warp.`` or another
alias than ``wp``, and ``wp.kernel`` (or func, get_module) bound to a name or passed to ``partial`` (WP8's reviews).
``wp.Module(name)`` is left alone: ``_alloy_warp.kernel`` raises on a Module object whose name is not Alloy's own.
"""

from __future__ import annotations

import ast
from pathlib import Path

IMPORT_TAIL = "; from alloy._vendor import _warp as _alloy_warp"
WRAP = "_alloy_warp.module_name("
NAMES = ("kernel", "func", "get_module")


class CallsError(RuntimeError):
    """A vendored file the rewrite cannot handle safely (it names the file and line)."""


def _wrapped(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and ast.unparse(node.func) == "_alloy_warp.module_name"


def _kwargs(n: ast.Call) -> bool:
    return any(k.arg is None for k in n.keywords)


def _foreign(tree: ast.AST) -> set[str]:
    """Names Warp is bound to other than ``wp``: ``warp`` itself and any ``import warp as <alias>``."""
    return {"warp"} | {a.asname for n in ast.walk(tree) if isinstance(n, ast.Import)
                       for a in n.names if a.name == "warp" and a.asname not in (None, "wp")}


def _warp_attr(node: ast.AST, receivers: set[str]) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr in NAMES and isinstance(node.value, ast.Name)
            and node.value.id in receivers)


def _refuse(n: ast.AST, foreign: set[str]) -> None:
    """Forms that would hide a module name from the rewrite (WP8's reviews)."""
    if isinstance(n, ast.ImportFrom) and n.module == "warp" and any(a.name in NAMES for a in n.names):
        raise CallsError(f"line {n.lineno}: from warp import {', '.join(a.name for a in n.names)}")
    if isinstance(n, ast.Assign) and _warp_attr(n.value, foreign | {"wp"}):
        raise CallsError(f"line {n.lineno}: {ast.unparse(n.value)} bound to another name")
    if not isinstance(n, ast.Call):
        return
    if ast.unparse(n.func).split(".")[-1] == "partial" and n.args and _warp_attr(n.args[0], foreign | {"wp"}):
        raise CallsError(f"line {n.lineno}: partial({ast.unparse(n.args[0])}, ...)")
    f = n.func
    if not _warp_attr(f, foreign | {"wp"}):
        return
    starred = any(isinstance(a, ast.Starred) for a in n.args)
    if f.value.id == "wp" and (starred or (f.attr != "kernel" and _kwargs(n))):
        raise CallsError(f"line {n.lineno}: wp.{f.attr}(*args / **kwargs)")
    if f.value.id in foreign and (f.attr == "get_module" or starred or _kwargs(n)
                                  or any(k.arg == "module" for k in n.keywords)):
        raise CallsError(f"line {n.lineno}: {f.value.id}.{f.attr} with a module (only `wp.` is rewritten)")


def edits(tree: ast.AST) -> list[tuple[int, int, int, int, str]]:
    """(line, byte col, end line, end byte col, replacement) for every site still to rewrite (module docstring)."""
    out, foreign = [], _foreign(tree)
    for n in ast.walk(tree):
        _refuse(n, foreign)
        f = getattr(n, "func", None)
        if not (isinstance(n, ast.Call) and isinstance(f, ast.Attribute) and f.attr in NAMES):
            continue
        if not (isinstance(f.value, ast.Name) and f.value.id == "wp"):
            continue
        module = next((k.value for k in n.keywords if k.arg == "module"), None)
        if f.attr == "kernel" and (module is not None or _kwargs(n)):
            out.append((f.lineno, f.col_offset, f.end_lineno, f.end_col_offset, "_alloy_warp.kernel"))
            continue
        if f.attr == "func":
            target = module
        elif f.attr == "get_module":
            target = n.args[0] if n.args else next((k.value for k in n.keywords if k.arg == "name"), None)
        else:
            target = None
        if target is not None and not _wrapped(target):
            out.append((target.end_lineno, target.end_col_offset, target.end_lineno, target.end_col_offset, ")"))
            out.append((target.lineno, target.col_offset, target.lineno, target.col_offset, WRAP))
    return out


def _import_end(tree: ast.Module) -> tuple[int, int]:
    """End (line, byte col) of the module's top-level ``import warp as wp`` (one alias)."""
    for n in tree.body:
        if isinstance(n, ast.Import) and [(a.name, a.asname) for a in n.names] == [("warp", "wp")]:
            return n.end_lineno, n.end_col_offset
    raise CallsError("no top-level `import warp as wp` to carry the _alloy_warp import")


def _apply(lines: list[str], line: int, c0: int, end_line: int, c1: int, text: str) -> None:
    """Replace bytes [c0, c1) of ``line`` .. ``end_line`` (1-based, UTF-8 offsets as ast gives them) with ``text``."""
    first, last = lines[line - 1].encode(), lines[end_line - 1].encode()
    lines[line - 1:end_line] = [(first[:c0] + text.encode() + last[c1:]).decode()]


def rewrite(src: str, rel: str = "") -> tuple[str, int]:
    """(new text, sites rewritten) for one vendored file (module docstring)."""
    try:
        tree = ast.parse(src)
        found = edits(tree)
        if not found:
            return src, 0
        imp = _import_end(tree)
    except CallsError as exc:
        raise CallsError(f"{rel}: {exc}") from exc
    lines = src.splitlines(keepends=True)
    if not lines[imp[0] - 1].encode()[imp[1]:].decode().startswith(IMPORT_TAIL):
        found.append((imp[0], imp[1], imp[0], imp[1], IMPORT_TAIL))
    for e in sorted(found, key=lambda e: (e[0], e[1]), reverse=True):
        _apply(lines, *e)
    return "".join(lines), sum(1 for e in found if e[4] not in (")", IMPORT_TAIL))


def upstream(text: str) -> str:
    """The stock form of rewritten text (a file or one function's source), the inverse of ``rewrite``: source pins
    (``alloy.precision.mjw_pin``) hash this, so they keep the upstream values across the rewrite."""
    text = text.replace(IMPORT_TAIL, "").replace("_alloy_warp.kernel(", "wp.kernel(")
    while (i := text.find(WRAP)) >= 0:
        j, depth = i + len(WRAP), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(text[j], 0)
            j += 1
        text = text[:i] + text[i + len(WRAP):j - 1] + text[j:]
    return text


def unwrapped(src: str) -> list[int]:
    """Lines of sites a run would still rewrite (empty when the file is done); raises on refused forms."""
    return sorted({e[0] for e in edits(ast.parse(src)) if e[4] != ")"})


def vendored_files(vendor: Path) -> list[Path]:
    """The vendored engines' .py files (newton, mujoco_warp)."""
    return sorted(p for pkg in ("newton", "mujoco_warp") for p in (vendor / pkg).rglob("*.py"))


def rewrite_tree(vendor: Path, check: bool) -> dict[str, int | str]:
    """Rewrite (or, with ``check``, only report) every vendored file in place: {relative path: sites, or refusal}."""
    out: dict[str, int | str] = {}
    for path in vendored_files(vendor):
        rel = str(path.relative_to(vendor))
        src = path.read_text()
        try:
            new, n = rewrite(src, rel)
        except CallsError as exc:
            out[rel] = f"refused: {exc}"
            continue
        if new != src:
            out[rel] = n
            if not check:
                path.write_text(new)
    return out


__all__ = ["IMPORT_TAIL", "NAMES", "WRAP", "CallsError", "edits", "rewrite", "rewrite_tree", "unwrapped", "upstream",
           "vendored_files"]
