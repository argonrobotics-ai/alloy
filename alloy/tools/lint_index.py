# SPDX-License-Identifier: Apache-2.0
"""Static type and launch index for ``lint_atomics`` (SPEC §3.7 determinism rule D3).

Resolves the element dtype of Warp arrays named in kernel bodies from annotations, module-level aliases and
``@wp.struct`` fields across every scanned file, and indexes ``wp.launch`` call sites so the two-term rule can check
that a target was zero-filled earlier in the same host function.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

ARRAY_CTORS = frozenset(
    {"array", "array1d", "array2d", "array3d", "array4d", "indexedarray", "indexedarray1d", "indexedarray2d",
     "indexedarray3d", "indexedarray4d", "fabricarray", "indexedfabricarray", "fixedarray"}
)
INT_SCALARS = frozenset(
    {"int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"}
)
INT_VECTOR = re.compile(r"^vec[2-4](i|ui|l|ul|s|us|b|ub)$")
FLOAT_SCALARS = frozenset({"float", "float16", "float32", "float64", "half"})
FLOAT_COMPOSITE = re.compile(r"^(vec[2-4][hfd]?|mat[2-4][2-4]?[hfd]?|quat[hfd]?|transform[hfd]?|spatial_\w+)$")
VECTOR_CTORS = frozenset({"vector", "matrix", "quaternion", "transformation"})
ZERO_ALLOC = frozenset({"zeros", "zeros_like"})
LAUNCHERS = frozenset({"launch", "launch_tiled"})


@dataclass(frozen=True)
class TypeInfo:
    """What a name refers to inside a kernel: ``kind`` array/struct/value/unknown; ``elem`` int/float/unknown."""

    kind: str = "unknown"
    elem: str = "unknown"
    struct: str | None = None


UNKNOWN = TypeInfo()


def last_name(expr: ast.AST | None) -> str | None:
    """``wp.types.int32`` -> ``int32``; ``int32`` -> ``int32``; anything else -> None."""
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def _kw(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


@dataclass
class ModuleInfo:
    """One parsed file: tree, source lines, module-level aliases (``name = <expr>``), ``@wp.struct`` classes and
    ``from x import y [as z]`` names (z -> (x, y))."""

    path: Path
    tree: ast.Module
    lines: list[str]
    aliases: dict[str, ast.AST] = field(default_factory=dict)
    structs: dict[str, dict[str, ast.AST]] = field(default_factory=dict)
    imports: dict[str, tuple[str, str]] = field(default_factory=dict)

    @property
    def dotted(self) -> str:
        """Import name from the ``__init__.py`` package chain (``alloy.precision.health``); without a package, from
        the last ``alloy`` path part, else the file stem."""
        parts = [self.path.stem] if self.path.stem != "__init__" else []
        parent = self.path.parent
        while (parent / "__init__.py").exists():
            parts.insert(0, parent.name)
            parent = parent.parent
        if len(parts) > 1 or "alloy" not in self.path.parts:
            return ".".join(parts)
        all_parts = list(self.path.with_suffix("").parts)
        start = max(i for i, p in enumerate(all_parts) if p == "alloy")
        return ".".join(p for p in all_parts[start:] if p != "__init__")


@dataclass(frozen=True)
class LaunchSite:
    """A ``wp.launch(kernel, ..., inputs=[...], outputs=[...])`` call and the host function around it."""

    module: ModuleInfo
    host: ast.AST
    call: ast.Call
    bound: tuple[ast.AST, ...] | None  # inputs + outputs, None if not a literal list


class RepoIndex:
    """Structs, aliases and launch sites over a set of files; resolves dtypes by name across files."""

    def __init__(self, modules: list[ModuleInfo]) -> None:
        self.modules = modules
        self.by_dotted = {m.dotted: m for m in modules}
        self.by_path = {str(m.path): m for m in modules}
        self.structs: dict[str, list[tuple[ModuleInfo, dict[str, ast.AST]]]] = {}
        self.global_aliases: dict[str, tuple[ModuleInfo, ast.AST]] = {}
        self._alias_count: dict[str, int] = {}
        self.launches: dict[str, list[LaunchSite]] = {}
        self.kernels: dict[str, list[tuple[ModuleInfo, ast.FunctionDef]]] = {}
        for mod in modules:
            self._index_module(mod)

    # ----- indexing -----
    def _index_module(self, mod: ModuleInfo) -> None:
        for node in mod.tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                mod.aliases[node.targets[0].id] = node.value
                self.global_aliases.setdefault(node.targets[0].id, (mod, node.value))
                self._alias_count[node.targets[0].id] = self._alias_count.get(node.targets[0].id, 0) + 1
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                mod.aliases[node.target.id] = node.value
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    mod.imports[a.asname or a.name] = (node.module, a.name)
            if isinstance(node, ast.ClassDef) and any(_is_decorator(d, "struct") for d in node.decorator_list):
                fields = {s.target.id: s.annotation for s in node.body
                          if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
                mod.structs[node.name] = fields
                self.structs.setdefault(node.name, []).append((mod, fields))
            elif isinstance(node, ast.FunctionDef):
                if is_kernel(node):
                    self.kernels.setdefault(node.name, []).append((mod, node))
                self._index_launches(mod, node)

    def _index_launches(self, mod: ModuleInfo, host: ast.FunctionDef) -> None:
        for call in _calls_in(host):
            if last_name(call.func) not in LAUNCHERS or not _is_warp_call(call.func):
                continue
            kernel = call.args[0] if call.args else _kw(call, "kernel")
            name = last_name(kernel)
            if name is None:
                continue
            self.launches.setdefault(name, []).append(LaunchSite(mod, host, call, _bound_args(call)))

    # ----- type resolution -----
    def resolve_annotation(self, expr: ast.AST | None, mod: ModuleInfo, depth: int = 0) -> TypeInfo:
        """Type of an annotation expression (string annotations are parsed)."""
        if expr is None or depth > 8:
            return UNKNOWN
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            try:
                return self.resolve_annotation(ast.parse(expr.value, mode="eval").body, mod, depth + 1)
            except SyntaxError:
                return UNKNOWN
        if isinstance(expr, ast.Call) and last_name(expr.func) in ARRAY_CTORS:
            dtype = _kw(expr, "dtype") or (expr.args[0] if expr.args else None)
            return TypeInfo("array", self.classify_dtype(dtype, mod, depth + 1))
        if isinstance(expr, ast.Subscript) and last_name(expr.value) in ARRAY_CTORS:
            sl = expr.slice.elts[0] if isinstance(expr.slice, ast.Tuple) else expr.slice
            return TypeInfo("array", self.classify_dtype(sl, mod, depth + 1))
        name = last_name(expr)
        if name is None:
            return UNKNOWN
        owner = self._struct_owner(name, mod)
        if owner is not None:
            return TypeInfo("struct", "unknown", f"{owner[0].path}::{owner[1]}")
        alias = self._alias(name, mod)
        if alias is not None:
            amod, aexpr = alias
            return self.resolve_annotation(aexpr, amod, depth + 1)
        elem = self.classify_dtype(expr, mod, depth + 1)
        return TypeInfo("value", elem) if elem != "unknown" else UNKNOWN

    def classify_dtype(self, expr: ast.AST | None, mod: ModuleInfo, depth: int = 0) -> str:
        """``int`` for integer scalars and integer vectors, ``float`` for float types, else ``unknown``."""
        if expr is None or depth > 8:
            return "unknown"
        if isinstance(expr, ast.Call) and last_name(expr.func) in VECTOR_CTORS:
            return self.classify_dtype(_kw(expr, "dtype") or (expr.args[1] if len(expr.args) > 1 else None),
                                       mod, depth + 1)
        name = last_name(expr)
        if name is None:
            return "unknown"
        if name in INT_SCALARS or INT_VECTOR.match(name):
            return "int"
        if name in FLOAT_SCALARS or FLOAT_COMPOSITE.match(name):
            return "float"
        alias = self._alias(name, mod)
        if alias is not None:
            return self.classify_dtype(alias[1], alias[0], depth + 1)
        return "unknown"

    def _struct_owner(self, name: str, mod: ModuleInfo) -> tuple[ModuleInfo, str] | None:
        """(defining module, struct name) for ``name`` as seen from ``mod``: its own struct, an imported one (through
        ``from x import S as name``), or a unique global struct of that name."""
        if name in mod.structs:
            return mod, name
        if name in mod.imports:
            src, orig = mod.imports[name]
            owner = self.by_dotted.get(src)
            if owner is not None and orig in owner.structs:
                return owner, orig
            cands = self.structs.get(orig, [])
            return (cands[0][0], orig) if len(cands) == 1 else None
        cands = self.structs.get(name, [])
        return (cands[0][0], name) if len(cands) == 1 else None

    def struct_field(self, struct: str, field_name: str) -> TypeInfo:
        path, name = struct.rsplit("::", 1)
        mod = self.by_path[path]
        return self.resolve_annotation(mod.structs[name].get(field_name), mod)

    def _alias(self, name: str, mod: ModuleInfo) -> tuple[ModuleInfo, ast.AST] | None:
        """A module-level alias: local, else from the module it is imported from, else a unique global name."""
        if name in mod.aliases:
            return mod, mod.aliases[name]
        if name in mod.imports:
            src, orig = mod.imports[name]
            owner = self.by_dotted.get(src)
            if owner is not None and orig in owner.aliases:
                return owner, owner.aliases[orig]
            return None  # imported from outside the scanned paths: unknown
        return self.global_aliases.get(name) if self._alias_count.get(name, 0) == 1 else None


def _is_warp_call(func: ast.AST) -> bool:
    if isinstance(func, ast.Attribute):
        return last_name(func.value) in {"wp", "warp"}
    return isinstance(func, ast.Name)


def _is_decorator(dec: ast.AST, name: str) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    if isinstance(target, ast.Attribute):
        return target.attr == name and last_name(target.value) in {"wp", "warp"}
    return isinstance(target, ast.Name) and target.id == name


def is_kernel(fn: ast.FunctionDef) -> bool:
    return any(_is_decorator(d, "kernel") for d in fn.decorator_list)


def is_warp_function(fn: ast.FunctionDef) -> bool:
    """``@wp.kernel`` or ``@wp.func`` (native snippets are checked separately)."""
    return any(_is_decorator(d, n) for d in fn.decorator_list for n in ("kernel", "func"))


def _calls_in(node: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def _bound_args(call: ast.Call) -> tuple[ast.AST, ...] | None:
    inputs = _kw(call, "inputs") or (call.args[2] if len(call.args) > 2 else None)
    outputs = _kw(call, "outputs") or (call.args[3] if len(call.args) > 3 else None)
    parts: list[ast.AST] = []
    for seq in (inputs, outputs):
        if seq is None:
            continue
        if not isinstance(seq, (ast.List, ast.Tuple)):
            return None
        parts.extend(seq.elts)
    return tuple(parts)


def load_modules(paths: list[Path]) -> tuple[list[ModuleInfo], list[tuple[Path, str]]]:
    """Parse every ``*.py`` under ``paths``; returns (modules, [(path, syntax error)])."""
    files: list[Path] = []
    for p in paths:   # a directory walk skips the vendored engines (alloy/_vendor, upstream code); name it to lint it
        walk = (q for q in p.rglob("*.py") if "_vendor" not in q.relative_to(p).parts) if p.is_dir() else ()
        files.extend(sorted(walk) if p.is_dir() else [p])
    mods: list[ModuleInfo] = []
    errors: list[tuple[Path, str]] = []
    for f in files:
        if "__pycache__" in f.parts:
            continue
        text = f.read_text(encoding="utf-8")
        try:
            mods.append(ModuleInfo(f, ast.parse(text, filename=str(f)), text.splitlines()))
        except SyntaxError as exc:
            errors.append((f, str(exc)))
    return mods, errors
