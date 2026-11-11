# SPDX-License-Identifier: Apache-2.0
"""Determinism lint for Warp kernels (SPEC §3.7, D9 rule D3: no float atomics in Alloy code).

Usage: ``python -m alloy.tools.lint_atomics alloy/ [more paths]``; exit 0 clean, 1 findings, 2 parse errors.

Rules (inside ``@wp.kernel`` / ``@wp.func`` bodies):

- DET001 float ``atomic_add``/``atomic_sub``/``tile_atomic_add``, or an array-slot ``+=``/``-=`` (Warp lowers those to
  atomics, warp 1.18 ``_src/codegen.py`` ``emit_AugAssign``) on a non-integer target.
- DET002 target dtype could not be resolved (annotate the parameter, or mark the line).
- DET003 order-dependent atomics on any dtype: ``atomic_exch`` (last writer wins), ``tile_scatter_add``.
- DET004 a ``# alloy: det-2term`` sum whose target is not zero-filled before every launch of the kernel, in the same
  host function (``.zero_()``, ``.fill_(0)``, ``wp.zeros*``, or a launch of a kernel that stores 0 into it). Two-term
  float sums are order-independent only into a zero (a + b == b + a; (0 + a) + b == (0 + b) + a).
- DET005 ``# alloy: det-ok`` without a reason.
- DET006 a ``wp.func_native`` snippet containing ``atomicAdd``.

Escape hatch: a line comment ``# alloy: det-ok <reason>`` on any line of the flagged statement. ``atomic_max``,
``atomic_min``, bitwise and integer atomics are deterministic and allowed.
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from alloy.tools.lint_index import (
    FLOAT_COMPOSITE,
    FLOAT_SCALARS,
    INT_SCALARS,
    LAUNCHERS,
    UNKNOWN,
    ZERO_ALLOC,
    LaunchSite,
    ModuleInfo,
    RepoIndex,
    TypeInfo,
    is_kernel,
    is_warp_function,
    last_name,
    load_modules,
)

log = logging.getLogger("alloy.tools")

SUM_ATOMICS = frozenset({"atomic_add", "atomic_sub", "tile_atomic_add"})
ORDER_ATOMICS = frozenset({"atomic_exch", "tile_scatter_add"})
DET_OK = re.compile(r"#\s*alloy:\s*det-ok\b(.*)$")
TWO_TERM = re.compile(r"#\s*alloy:\s*det-2term\b")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    message: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} {self.message}"


def _marker(mod: ModuleInfo, node: ast.AST, pattern: re.Pattern[str]) -> re.Match[str] | None:
    end = getattr(node, "end_lineno", None) or node.lineno
    for ln in range(node.lineno, end + 1):
        m = pattern.search(mod.lines[ln - 1]) if ln - 1 < len(mod.lines) else None
        if m:
            return m
    return None


def _is_zero(expr: ast.AST | None) -> bool:
    if isinstance(expr, ast.Constant):
        return expr.value in (0, 0.0, False) and not isinstance(expr.value, str)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, (ast.USub, ast.UAdd)):
        return _is_zero(expr.operand)
    if isinstance(expr, ast.Call):
        name = last_name(expr.func) or ""
        typed = name in FLOAT_SCALARS or name in INT_SCALARS or bool(FLOAT_COMPOSITE.match(name))
        return typed and all(_is_zero(a) for a in expr.args)
    return False


class KernelChecker:
    """Checks one ``@wp.kernel``/``@wp.func`` body."""

    def __init__(self, index: RepoIndex, mod: ModuleInfo, fn: ast.FunctionDef) -> None:
        self.index, self.mod, self.fn = index, mod, fn
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        self.params = [a.arg for a in args]
        self.env: dict[str, TypeInfo] = {a.arg: index.resolve_annotation(a.annotation, mod) for a in args}
        self._track_locals()
        self.tid = self._tid_names()
        self.findings: list[Finding] = []

    def _tid_names(self) -> tuple[str, ...]:
        """Names bound by ``x = wp.tid()`` or ``i, j = wp.tid()`` and never rebound (empty if none)."""
        tid = self._tid_binding()
        bound = [t for n in ast.walk(self.fn) for t in _bound_names(n)]
        return tid if tid and all(bound.count(name) == 1 for name in tid) else ()

    def _tid_binding(self) -> tuple[str, ...]:
        for node in ast.walk(self.fn):
            if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)):
                continue
            if last_name(node.value.func) != "tid" or len(node.targets) != 1:
                continue
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                return (tgt.id,)
            if isinstance(tgt, ast.Tuple) and all(isinstance(e, ast.Name) for e in tgt.elts):
                return tuple(e.id for e in tgt.elts)
        return ()

    def _single_writer(self, indices: list[ast.AST]) -> bool:
        """True if the leading indices are exactly the thread id(s): each thread owns its slots."""
        names = [i.id if isinstance(i, ast.Name) else None for i in indices[: len(self.tid)]]
        return bool(self.tid) and tuple(names) == self.tid

    def _track_locals(self) -> None:
        assigns = sorted((n for n in ast.walk(self.fn) if isinstance(n, ast.Assign)), key=lambda n: n.lineno)
        for node in assigns:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id not in self.params:
                    t = self.resolve(node.value)
                    self.env[tgt.id] = t if t.kind in ("array", "struct") else TypeInfo("value")

    def resolve(self, expr: ast.AST) -> TypeInfo:
        if isinstance(expr, ast.Name):
            return self.env.get(expr.id, UNKNOWN)
        if isinstance(expr, ast.Attribute):
            base = self.resolve(expr.value)
            if base.kind == "struct" and base.struct is not None:
                return self.index.struct_field(base.struct, expr.attr)
            return UNKNOWN
        if isinstance(expr, ast.Subscript):
            base = self.resolve(expr.value)
            return TypeInfo("value", base.elem) if base.kind == "array" else UNKNOWN
        return UNKNOWN

    def run(self) -> list[Finding]:
        for node in ast.walk(self.fn):
            if isinstance(node, ast.Call):
                self._check_call(node)
            elif isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
                self._check_augassign(node)
        return self.findings

    def _add(self, node: ast.AST, rule: str, msg: str) -> None:
        ok = _marker(self.mod, node, DET_OK)
        if ok is not None:
            if ok.group(1).strip():
                return
            rule, msg = "DET005", "det-ok needs a reason: '# alloy: det-ok <reason>'"
        self.findings.append(Finding(str(self.mod.path), node.lineno, rule, msg))

    def _check_call(self, call: ast.Call) -> None:
        name = last_name(call.func)
        if name in ORDER_ATOMICS:
            self._add(call, "DET003", f"{name} is order-dependent on every dtype")
            return
        if name not in SUM_ATOMICS or not call.args:
            return
        if name != "tile_atomic_add" and self._single_writer(list(call.args[1:-1])):
            return
        self._check_target(call, call.args[0], name)

    def _check_augassign(self, node: ast.AugAssign) -> None:
        array = self._array_of_target(node.target)
        if array is None:
            return  # a local value (vector component, scalar): no atomic
        op = "+=" if isinstance(node.op, ast.Add) else "-="
        self._check_target(node, array, f"array-slot '{op}' (lowered to an atomic)")

    def _array_of_target(self, target: ast.AST) -> ast.AST | None:
        """``a[i] += x``, ``a[i][0] += x``, ``s.arr[i] += x`` -> the array expression; locals and slots owned by
        the thread (``a[tid] += x``, ``a[i, j, k] += x`` with ``i, j = wp.tid()``) -> None."""
        node = target
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            if isinstance(node, ast.Subscript) and self._maybe_array(node.value):
                sl = node.slice
                indices = list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
                return None if self._single_writer(indices) else node.value
            node = node.value
        return None

    def _maybe_array(self, expr: ast.AST) -> bool:
        """An array, or an unresolved parameter (``Any``, a type from outside the scanned paths) that may be one."""
        t = self.resolve(expr)
        if t.kind == "array":
            return True
        root = expr
        while isinstance(root, ast.Attribute):
            root = root.value
        return t is UNKNOWN and isinstance(root, ast.Name) and root.id in self.params

    def _check_target(self, node: ast.AST, target: ast.AST, what: str) -> None:
        t = self.resolve(target)
        if t.kind == "array" and t.elem == "int":
            return
        text = ast.unparse(target)
        if t.kind == "array" and t.elem == "float":
            if _marker(self.mod, node, TWO_TERM):
                problem = self._verify_two_term(target)
                if problem:
                    self._add(node, "DET004", f"det-2term into '{text}': {problem}")
                return
            self._add(node, "DET001", f"{what} into float target '{text}'")
            return
        self._add(node, "DET002", f"{what} into '{text}' of unknown dtype (annotate it or mark the line)")

    def _verify_two_term(self, target: ast.AST) -> str | None:
        if not is_kernel(self.fn):
            return "two-term sums must sit in the @wp.kernel body"
        if isinstance(target, ast.Name):
            param, field_name = target.id, None
        elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
            param, field_name = target.value.id, target.attr
        else:
            return "target must be a kernel parameter (or a field of one)"
        if param not in self.params:
            return "target must be a kernel parameter (or a field of one)"
        sites = self.index.launches.get(self.fn.name, [])
        if not sites:
            return f"no wp.launch of {self.fn.name} found, so the zero-fill cannot be verified"
        idx = self.params.index(param)
        for site in sites:
            problem = _check_site(self.index, site, idx, field_name)
            if problem:
                return problem
        return None


def _check_site(index: RepoIndex, site: LaunchSite, idx: int, field_name: str | None) -> str | None:
    where = f"{site.module.path}:{site.call.lineno}"
    if site.bound is None or idx >= len(site.bound):
        return f"{where}: launch arguments are not a literal inputs/outputs list"
    text = ast.unparse(site.bound[idx]) + (f".{field_name}" if field_name else "")
    if _zero_filled_before(index, site, text):
        return None
    return f"{where}: '{text}' is not zero-filled earlier in the same host function"


def _bound_names(node: ast.AST) -> list[str]:
    """Names (re)bound by an assignment-like statement."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.For)):
        targets = [node.target]
    else:
        return []
    out: list[str] = []
    for t in targets:
        out.extend(e.id for e in ast.walk(t) if isinstance(e, ast.Name) and isinstance(e.ctx, ast.Store))
    return out


def _innermost_loop(host: ast.AST, call: ast.AST) -> ast.AST:
    """The innermost for/while loop of ``host`` that contains ``call`` (``host`` itself if none)."""
    best = host
    for node in ast.walk(host):
        if isinstance(node, (ast.For, ast.While)) and node is not call and call in set(ast.walk(node)):
            if best is host or node in set(ast.walk(best)):
                best = node
    return best


def _zero_filled_before(index: RepoIndex, site: LaunchSite, text: str) -> bool:
    """A zero-fill of ``text`` earlier in the same host function, inside the innermost loop around the launch (a
    fill before a loop of launches zeroes only the first)."""
    scope = _innermost_loop(site.host, site.call)
    for node in ast.walk(scope):
        if getattr(node, "lineno", 1 << 30) >= site.call.lineno:
            continue
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            alloc = last_name(node.value.func) in ZERO_ALLOC
            if alloc and any(ast.unparse(t) == text for t in node.targets):
                return True
        if isinstance(node, ast.Call) and _call_zeroes(index, node, text):
            return True
    return False


def _call_zeroes(index: RepoIndex, call: ast.Call, text: str) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and ast.unparse(func.value) == text:
        if func.attr == "zero_":
            return True
        if func.attr == "fill_":
            value = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "value"), None)
            return _is_zero(value)
    if last_name(func) not in LAUNCHERS:
        return False
    kname = last_name(call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "kernel"), None))
    site_bound = next((s.bound for s in index.launches.get(kname or "", []) if s.call is call), None)
    if not site_bound:
        return False
    for pos, arg in enumerate(site_bound):
        arg_text = ast.unparse(arg)
        if arg_text == text or text.startswith(arg_text + "."):
            field_path = text[len(arg_text) + 1:] if text != arg_text else None
            if _kernel_stores_zero(index, kname or "", pos, field_path):
                return True
    return False


def _kernel_stores_zero(index: RepoIndex, kname: str, pos: int, field_path: str | None) -> bool:
    for _, kernel in index.kernels.get(kname, []):
        params = [a.arg for a in kernel.args.posonlyargs + kernel.args.args]
        if pos >= len(params):
            continue
        want = params[pos] + (f".{field_path}" if field_path else "")
        for node in ast.walk(kernel):
            if not (isinstance(node, ast.Assign) and _is_zero(node.value)):
                continue
            if any(isinstance(t, ast.Subscript) and ast.unparse(t.value) == want for t in node.targets):
                return True
    return False


def _native_snippets(index: RepoIndex, mod: ModuleInfo) -> list[Finding]:
    out: list[Finding] = []
    for fn in (n for n in ast.walk(mod.tree) if isinstance(n, ast.FunctionDef)):
        for dec in fn.decorator_list:
            if not (isinstance(dec, ast.Call) and last_name(dec.func) == "func_native" and dec.args):
                continue
            snippet = dec.args[0]
            if isinstance(snippet, ast.Name) and snippet.id in mod.aliases:
                snippet = mod.aliases[snippet.id]
            text = snippet.value if isinstance(snippet, ast.Constant) and isinstance(snippet.value, str) else ""
            if "atomicAdd" not in text:
                continue
            ok = _marker(mod, dec, DET_OK)
            if ok is None or not ok.group(1).strip():
                out.append(Finding(str(mod.path), dec.lineno, "DET006", f"native snippet of {fn.name} uses atomicAdd"))
    return out


def lint(paths: list[Path]) -> tuple[list[Finding], list[tuple[Path, str]]]:
    """Lint every ``*.py`` under ``paths``; returns (findings sorted by path and line, parse errors)."""
    modules, errors = load_modules(paths)
    index = RepoIndex(modules)
    findings: list[Finding] = []
    for mod in modules:
        for fn in (n for n in ast.walk(mod.tree) if isinstance(n, ast.FunctionDef)):
            if is_warp_function(fn):
                findings.extend(KernelChecker(index, mod, fn).run())
        findings.extend(_native_snippets(index, mod))
    log.debug("lint_atomics: %d files, %d kernels, %d findings", len(modules), len(index.kernels), len(findings))
    return sorted(set(findings), key=lambda f: (f.path, f.line, f.rule)), errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alloy.tools.lint_atomics", description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=["alloy"], type=Path)
    args = parser.parse_args(argv)
    findings, errors = lint(args.paths)
    for path, err in errors:
        sys.stderr.write(f"{path}: parse error: {err}\n")
    for f in findings:
        sys.stdout.write(f.format() + "\n")
    sys.stdout.write(f"lint_atomics: {len(findings)} finding(s) in {', '.join(map(str, args.paths))}\n")
    if errors:
        return 2
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
