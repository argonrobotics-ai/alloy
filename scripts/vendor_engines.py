# SPDX-License-Identifier: Apache-2.0
"""Vendor Newton and MuJoCo Warp inside Alloy (coordinator ruling 2026-11-07, scr/WP0.md): copy each pinned package
into alloy/_vendor/<name>, rewrite imports by script, and make every Alloy file import the vendored copies.

Rerunnable: the vendored trees are deleted and copied again from the pinned checkouts; Alloy's own files are rewritten
in place and a second run changes nothing. Only import statements and reviewed module-name strings change, one
statement stays on one line (Alloy's line numbers are kept), and each vendored .py file gets the Apache-2.0 4(b)
NOTICE line under its header (vendored line numbers are upstream's + 1). Data namespaces (USD / MJCF
``newton:`` attributes, recording type tags, cache names, logger names) are never touched.

    python scripts/vendor_engines.py --newton-src DIR --mjw-src DIR [--check]
    python scripts/vendor_engines.py --alloy-only     # a builder's branch after rebasing onto the vendoring

``--check`` rewrites nothing and exits 1 when a run would change a file (the gate's drift test).
"""

from __future__ import annotations

import argparse
import importlib.util
import ast
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("alloy.scripts.vendor_engines")

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "alloy" / "_vendor"
PREFIX = "alloy._vendor."
ROOTS = ("newton", "mujoco_warp")
NEWTON_SHA = "25b4603cfb37dfa250d8b9e1727b39514ab6e8a0"     # NEWTON_PIN: our 12 commits on release-1.8 db915e1c
NEWTON_BASE = "db915e1c79e9e4a09f61221f743e1e4fe814d75c"    # public upstream base of our patches (release-1.8)
MJW_SHA = "f34c6dd52813852d0f1a2ab6d1f4d21ef1f0829f"        # MJWARP_PIN
NEWTON_VERSION, MJW_VERSION = "1.8.0rc1", "3.17.0"        # what pip reported for the pinned installs

# Not shipped: each package's own tests and test data (Alloy never imports them).
NEWTON_EXCLUDE = (re.compile(r"^tests/"), re.compile(r"(^|/)\.gitignore$"))   # .gitignore: meaningless in a package
MJW_EXCLUDE = (re.compile(r"_test\.py$"), re.compile(r"^conftest\.py$"), re.compile(r"^test_data/"))
# Every Alloy file outside the vendored trees is rewritten (the release scan refuses any top-level engine import).
# This script is skipped (its tables hold the upstream names), and alloy/isaaclab/compat.py (Isaac Lab's mujoco_warp).
# Scripts that target a given engine keep their own `import newton` / `import mujoco_warp`: xbench scenes (whichever
# engine the benchmark provides: stock Newton, or Alloy's copy through the shim's alias), XB4 scripts (other engines'
# installs: Isaac Lab's Newton, MuJoCo 4.0's), xbench results (what ran) and tools/deps (a user's own newton, OSS-2).
ALLOY_SKIP = (re.compile(r"^alloy/_vendor/"), re.compile(r"^scripts/vendor_engines\.py$"),
              re.compile(r"^bench/xbench/scenes/(?!engine_shim\.py$)"), re.compile(r"^bench/xbench/xb4/"),
              re.compile(r"^bench/xbench/results/"), re.compile(r"^(tools/deps/|alloy/isaaclab/compat\.py$)"))
# Apache-2.0 4(b): every vendored .py file carries this notice under its licence header (the release adds none then).
NOTICE = "# Vendored in Alloy: alloy/_vendor/VENDORED.md lists the changes made to upstream, imports included."

MODULE_STR = re.compile(r"^(newton|mujoco_warp)(\.[A-Za-z_]\w*)*\.?$")
FILE_EXT = {"json", "jsonl", "py", "txt", "csv", "npz", "npy", "md", "yaml", "yml", "xml", "usd", "usda", "log",
            "png", "html"}
# Bare "newton" / "mujoco_warp" is mostly a solver, backend or engine name; only these calls name the module.
BARE_MODULE_CALLS = re.compile(r"(^|\.)(_after_import|import_module|__import__|importorskip|find_spec)$|"
                               r"__module__\.startswith$")
# Every module-name string in the vendored code, reviewed: True = rewrite, False = keep (a name, not a module).
VENDORED_STRINGS: dict[tuple[str, str], bool] = {
    ("newton/_src/actuators/usd_parser.py", "newton"): False,             # USD attribute namespace
    ("newton/_src/diagnostics.py", "newton"): False,                       # logger name
    ("newton/_src/geometry/_sdf_cache.py", "newton.texture_sdf"): False,  # on-disk cache kind
    ("newton/_src/solvers/kamino/_src/utils/logger.py", "newton._src.solvers.kamino"): True,   # = its __name__
    ("newton/_src/solvers/kamino/_src/utils/logger.py", "newton.tests.kamino"): False,   # logger of Newton's tests
    ("newton/_src/solvers/kamino/tests/test_utils_logger.py", "newton"): False,           # logger name
    ("newton/_src/solvers/mujoco/solver_mujoco.py", "mujoco_warp._src.smooth"): True,   # scoped MJW execution
    ("newton/_src/solvers/mujoco/solver_mujoco.py", "mujoco_warp._src."): True,         # scoped MJW execution
    ("newton/_src/solvers/mujoco/solver_mujoco.py", "newton"): False,      # MuJoCo solver type name / dist name
    ("newton/_src/usd/_resolution_policy.py", "newton"): False,            # USD namespace
    ("newton/_src/usd/schemas.py", "newton"): False,                       # USD schema namespace
    ("newton/_src/usd/utils.py", "newton"): False,                         # USD namespace
    ("newton/_src/utils/deprecation.py", "newton._src."): True,            # caller-frame module test
    ("newton/_src/viewer/picking.py", "newton._src.viewer.kernels"): True,       # wp.load_module
    ("newton/_src/viewer/picking.py", "newton._src.sim.articulation"): True,     # wp.load_module
    ("newton/_src/viewer/viewer_file.py", "newton.geometry.Mesh"): False,        # recording type tag
    ("newton/_src/viewer/viewer_file.py", "newton.geometry.Mesh_ref"): False,    # recording type tag
    ("newton/_src/viewer/viewer_rtx.py", "newton.ViewerRTX"): False,       # stage name
    ("newton/_version.py", "newton"): False,                               # replaced by a static version
    ("newton/config.py", "newton"): False,                                 # cache directory name (shared, as today)
    ("newton/examples/__init__.py", "newton.examples."): True,             # example module map
    ("mujoco_warp/__init__.py", "mujoco_warp"): False,                     # replaced by a static version
    ("mujoco_warp/_src/cli.py", "mujoco_warp"): True,                      # epath.resource_path(package)
    ("mujoco_warp/record.py", "mujoco_warp.record"): False,                # argv[0] text
    ("mujoco_warp/testspeed.py", "mujoco_warp.testspeed"): False,          # argv[0] text
    ("mujoco_warp/viewer.py", "mujoco_warp.viewer"): False,                # argv[0] text
}
# Tables in Alloy whose bare "newton" / "mujoco_warp" values are module names (file, enclosing node), reviewed:
# newton_compat._HOOK_OWNERS maps a hook owner to (module, attribute).
BARE_MODULE_SITES = {("alloy/core/newton_compat.py", "Dict:_HOOK_OWNERS")}
VENDORED_NAME_DIRS = ("newton/examples/",)   # SolverMuJoCo(..., solver="newton") in examples: a solver name


def _load(path: Path):
    """A module from ``path`` without importing its package (alloy/_vendor/_warp_calls.py: pure ast, no Warp)."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WARP_CALLS = _load(VENDOR / "_warp_calls.py")   # SCR-WP0-COEXIST: vendored wp.kernel/func(module=...) -> _alloy_warp


class VendorError(RuntimeError):
    """The pinned sources or the vendored tree are not what this script expects."""


class Report:
    """Counts of one run (a plain class: the merge gate loads this file by path, outside sys.modules)."""

    def __init__(self) -> None:
        self.imports, self.strings = 0, 0
        self.kept: dict[str, int] = {}
        self.files: set[str] = set()


# ---- import statements ------------------------------------------------------------------------------------------
def _root(name: str | None) -> str | None:
    head = (name or "").split(".", 1)[0]
    return head if head in ROOTS else None


def _import_text(node: ast.Import) -> str | None:
    """The vendored form of ``import ...`` (one physical line), or None when no alias names newton / mujoco_warp."""
    if not any(_root(a.name) for a in node.names):
        return None
    parts = []
    for a in node.names:
        r = _root(a.name)
        if r is None:
            parts.append(f"import {a.name}" + (f" as {a.asname}" if a.asname else ""))
        elif a.name == r:
            parts.append(f"from alloy._vendor import {r}" + (f" as {a.asname}" if a.asname else ""))
        elif a.asname:
            parts.append(f"import {PREFIX}{a.name} as {a.asname}")
        else:   # `import newton.x.y` binds `newton` and loads the submodule
            parts.append(f"import {PREFIX}{a.name}; from alloy._vendor import {r}")
    return "; ".join(parts)


def _rewrite_imports(src: str, rel: str, rep: Report) -> str:
    tree = ast.parse(src)
    lines = src.split("\n")
    edits: list[tuple[int, int, int, str]] = []    # (line index, col start, col end, text)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and _root(node.module):
            line = lines[node.lineno - 1]
            m = re.compile(r"from\s+" + re.escape(node.module) + r"\b").match(line, node.col_offset)
            if m is None:
                raise VendorError(f"{rel}:{node.lineno}: cannot place 'from {node.module}'")
            edits.append((node.lineno - 1, m.start(), m.end(), f"from {PREFIX}{node.module}"))
        elif isinstance(node, ast.Import):
            text = _import_text(node)
            if text is None:
                continue
            if node.lineno != node.end_lineno:
                raise VendorError(f"{rel}:{node.lineno}: multi-line import statement; split it by hand")
            edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, text))
    for i, a, b, text in sorted(edits, reverse=True):
        lines[i] = lines[i][:a] + text + lines[i][b:]
    rep.imports += len(edits)
    return "\n".join(lines)


# ---- module-name strings ------------------------------------------------------------------------------------------
def _context(parents: dict, node: ast.AST) -> str:
    p = parents.get(node)
    while isinstance(p, (ast.Tuple, ast.List, ast.Set, ast.keyword, ast.Starred, ast.JoinedStr,
                         ast.FormattedValue)):
        p = parents.get(p)
    if isinstance(p, ast.Call) and not isinstance(p.func, ast.Lambda):
        return ast.unparse(p.func)
    if isinstance(p, ast.Dict):          # a dict literal: name the table it is assigned to
        owner = parents.get(p)
        if isinstance(owner, (ast.Assign, ast.AnnAssign)):
            target = owner.targets[0] if isinstance(owner, ast.Assign) else owner.target
            return f"Dict:{ast.unparse(target)}"
    return type(p).__name__ if p is not None else ""


def _alloy_rule(value: str, ctx: str, rel: str) -> tuple[bool, str]:
    """(rewrite?, reason) for a module-name string in Alloy's own code."""
    if (rel, ctx) in BARE_MODULE_SITES:
        return True, f"bare, reviewed table in {rel}"
    if "." in value:
        last = value.rstrip(".").rsplit(".", 1)[-1]
        return (last not in FILE_EXT, "dotted module path" if last not in FILE_EXT else "file name")
    if BARE_MODULE_CALLS.search(ctx):
        return True, f"bare, {ctx}"
    return False, "bare name (solver / backend / engine label)"


def _vendored_rule(value: str, rel: str) -> tuple[bool, str]:
    key = (rel, value)
    if key in VENDORED_STRINGS:
        return VENDORED_STRINGS[key], "reviewed"
    if value in ROOTS and rel.startswith(VENDORED_NAME_DIRS):
        return False, "solver name in an example"
    raise VendorError(f"{rel}: unreviewed module-name string {value!r}: add it to VENDORED_STRINGS")


def _rewrite_strings(src: str, rel: str, vendored: bool, rep: Report) -> str:
    tree = ast.parse(src)
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    lines = src.split("\n")
    edits = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_STR.match(node.value)):
            continue
        rewrite, why = (_vendored_rule(node.value, rel) if vendored
                        else _alloy_rule(node.value, _context(parents, node), rel))
        if not rewrite:
            rep.kept[why] = rep.kept.get(why, 0) + 1
            continue
        if node.lineno != node.end_lineno:
            raise VendorError(f"{rel}:{node.lineno}: multi-line string {node.value!r}")
        line = lines[node.lineno - 1]
        seg = line[node.col_offset:node.end_col_offset]
        q = seg.lstrip("rbuRBUfF")[:1]
        if seg.lstrip("rbuRBUfF") not in (f"{q}{node.value}{q}",):
            # inside an f-string the Constant spans only its text, without quotes
            if seg != node.value:
                raise VendorError(f"{rel}:{node.lineno}: cannot place string {node.value!r} ({seg!r})")
            edits.append((node.lineno - 1, node.col_offset, node.end_col_offset, PREFIX + node.value))
        else:
            edits.append((node.lineno - 1, node.col_offset, node.end_col_offset,
                          seg.replace(node.value, PREFIX + node.value, 1)))
    for i, a, b, text in sorted(edits, reverse=True):
        lines[i] = lines[i][:a] + text + lines[i][b:]
    rep.strings += len(edits)
    return "\n".join(lines)


_DOC_IMPORT = re.compile(r"^(\s*(?:>>> |\.\.\. )?)((?:import|from)\s+(?:newton|mujoco_warp)\b.*)$")


def _rewrite_doc_examples(src: str, rel: str, rep: Report) -> str:
    """Example lines in Alloy's docstrings that are an engine import statement (``import newton, warp as wp``) get the
    same rewrite as code, so a copied example uses Alloy's engines; prose is never touched."""
    tree = ast.parse(src)
    lines = src.split("\n")
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            continue
        doc = body[0].value
        for i in range(doc.lineno - 1, doc.end_lineno):
            m = _DOC_IMPORT.match(lines[i])
            if not m:
                continue
            try:
                stmt = ast.parse(m.group(2)).body[0]
            except SyntaxError:
                continue
            if isinstance(stmt, ast.Import):
                text = _import_text(stmt)
            elif isinstance(stmt, ast.ImportFrom) and _root(stmt.module):
                text = m.group(2).replace(f"from {stmt.module}", f"from {PREFIX}{stmt.module}", 1)
            else:
                continue
            if text:
                lines[i] = m.group(1) + text
                rep.imports += 1
    return "\n".join(lines)


def rewrite_file(path: Path, rel: str, vendored: bool, rep: Report, check: bool) -> bool:
    """Rewrite one file; True when it changed (or would change, with ``check``)."""
    src = path.read_text()
    out = _rewrite_strings(_rewrite_imports(src, rel, rep), rel, vendored, rep)
    if not vendored:
        out = _rewrite_doc_examples(out, rel, rep)
    else:
        out = WARP_CALLS.rewrite(out, rel)[0]
    if out == src:
        return False
    rep.files.add(rel)
    if not check:
        path.write_text(out)
    return True


def rewrite_text(src: str, rel: str) -> str:
    """What a run makes of one Alloy file's text (no file IO)."""
    rep = Report()
    return _rewrite_doc_examples(_rewrite_strings(_rewrite_imports(src, rel, rep), rel, False, rep), rel, rep)


def mechanical(tree: Path, files: list[str], base: str = "main") -> set[str]:
    """Of ``files`` (changed over ``base``), those whose text is exactly this script's rewrite of the base text. The
    merge gate exempts them from ownership and the line cap, as it does alloy/_vendor/** (ruling 2026-11-07)."""
    out = set()
    for rel in files:
        path = tree / rel
        if not rel.endswith(".py") or rel.startswith("alloy/_vendor/") or not path.is_file():
            continue
        old = subprocess.run(["git", "show", f"{base}:{rel}"], cwd=tree, capture_output=True, text=True)
        new = path.read_text()
        if old.returncode == 0 and old.stdout != new and rewrite_text(old.stdout, rel) == new:
            out.add(rel)
    return out


# ---- vendored trees -----------------------------------------------------------------------------------------------
def _git(src: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(src), *args], capture_output=True, text=True, check=True).stdout.strip()


def _check_source(src: Path, sha: str, pkg: str) -> None:
    head = _git(src, "rev-parse", "HEAD")
    if head != sha:
        raise VendorError(f"{src}: HEAD {head} != pinned {sha}")
    if _git(src, "status", "--porcelain", "--", pkg):
        raise VendorError(f"{src}: {pkg}/ has local changes")


def _copy_tree(src: Path, pkg: str, exclude: tuple[re.Pattern, ...], extra: tuple[str, ...]) -> list[str]:
    dst = VENDOR / pkg
    if dst.exists():
        shutil.rmtree(dst)
    files = [f for f in _git(src, "ls-files", pkg).split("\n") if f]
    copied = []
    for f in files:
        rel = f[len(pkg) + 1:]
        if any(p.search(rel) for p in exclude):
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, dst / rel)
        copied.append(f)
    for name in extra:          # licence and credit files from the repository root
        shutil.copy2(src / name, dst / name)
    return copied


def _static_version(path: Path, pattern: str, replacement: str) -> None:
    text = path.read_text()
    new, n = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if n != 1:
        raise VendorError(f"{path}: version block not found")
    path.write_text(new)


def vendor_trees(newton_src: Path, mjw_src: Path, rep: Report) -> dict[str, list[str]]:
    _check_source(newton_src, NEWTON_SHA, "newton")
    _check_source(mjw_src, MJW_SHA, "mujoco_warp")
    VENDOR.mkdir(parents=True, exist_ok=True)
    copied = {"newton": _copy_tree(newton_src, "newton", NEWTON_EXCLUDE, ("LICENSE.md", "CITATION.cff")),
              "mujoco_warp": _copy_tree(mjw_src, "mujoco_warp", MJW_EXCLUDE, ("LICENSE", "AUTHORS"))}
    # Vendored, the packages have no distribution metadata: their version is the one pip reported for the pins.
    _static_version(VENDOR / "newton/_version.py",
                    r"from importlib\.metadata import PackageNotFoundError, version\n\ntry:\n.*?\"unknown\"\n",
                    f'__version__ = "{NEWTON_VERSION}"  # Alloy: vendored, no distribution metadata (VENDORED.md)\n')
    _static_version(VENDOR / "mujoco_warp/__init__.py",
                    r"from importlib import metadata\n\ntry:\n  __version__ = metadata\.version\(\"mujoco_warp\"\)\n"
                    r"except metadata\.PackageNotFoundError:\n  __version__ = \"unknown\"\n",
                    f'__version__ = "{MJW_VERSION}"  # Alloy: vendored, no distribution metadata (VENDORED.md)\n')
    for pkg in ROOTS:
        for path in sorted((VENDOR / pkg).rglob("*.py")):
            rel = str(path.relative_to(VENDOR))
            rewrite_file(path, rel, True, rep, check=False)
            _add_notice(path)
    return copied


def _add_notice(path: Path) -> None:
    """NOTICE as the line after the file's leading comment block (its licence header), or first without one."""
    lines = path.read_text().split("\n")
    if NOTICE in lines:
        return
    i = 0
    while i < len(lines) and lines[i].startswith("#") and not lines[i].startswith("#!"):
        i += 1
    if i == 0 and lines and lines[0].startswith("#!"):
        i = 1
    lines.insert(i, NOTICE)
    path.write_text("\n".join(lines))


PATCHES = VENDOR / "patches"   # Alloy's own edits to the vendored code (VENDORED.md), re-applied after every copy


def apply_patches(check: bool) -> list[str]:
    """Apply alloy/_vendor/patches/*.patch in name order (``git diff`` output, paths from the repo root) to the
    vendored tree. With ``check`` only verify each is applied. Returns the patches that are not applied."""
    missing = []
    for patch in sorted(PATCHES.glob("*.patch")):
        applied = subprocess.run(["git", "apply", "--check", "--reverse", str(patch)], cwd=ROOT,
                                 capture_output=True).returncode == 0
        if applied:
            continue
        if check:
            missing.append(patch.name)
            continue
        res = subprocess.run(["git", "apply", str(patch)], cwd=ROOT, capture_output=True, text=True)
        if res.returncode:
            raise VendorError(f"{patch.name} no longer applies to the vendored tree (pin moved?): {res.stderr[:300]}")
        log.info("applied %s", patch.name)
    return missing


EPA_FILE = ROOT / "alloy" / "contact" / "mjw_epa.py"
EPA_KEYS = {"GJK": "collision_gjk", '"alloy._vendor.mujoco_warp._src.collision_convex"': "collision_convex",
            '"alloy._vendor.mujoco_warp._src.collision_flex"': "collision_flex"}


def rekey_epa(check: bool) -> bool:
    """alloy.contact.mjw_epa pins the sha256 of the three stock MuJoCo Warp files its vendored PR copies were
    cherry-picked onto: re-key them on the vendored bytes (copied from MJW_SHA). True when a key changed."""
    import hashlib  # noqa: PLC0415

    text = EPA_FILE.read_text()
    base = re.search(r'^BASE_PIN = "([0-9a-f]{40})"$', text, re.M)
    if base is None or base.group(1) != MJW_SHA:
        raise VendorError(f"{EPA_FILE.name}: the PR #1845 copies were merged onto MuJoCo Warp "
                          f"{base.group(1)[:8] if base else '?'}, not {MJW_SHA[:8]}: re-merge the PR copies onto the new "
                          "pin first, then set BASE_PIN (WP3)")
    out = text
    for key, mod in EPA_KEYS.items():
        digest = hashlib.sha256((VENDOR / "mujoco_warp" / "_src" / f"{mod}.py").read_bytes()).hexdigest()
        out, n = re.subn(r"(    " + re.escape(key) + r': ")[0-9a-f]{64}(")', lambda m: m.group(1) + digest + m.group(2),
                         out)
        if n != 1:
            raise VendorError(f"{EPA_FILE}: BASE_SHA256 entry {key} not found")
    if out != text and not check:
        EPA_FILE.write_text(out)
    return out != text


_WALK_SKIP = {".git", ".venv", "venv", "build", "dist", "__pycache__", "node_modules"}


def _alloy_py_files() -> list[str]:
    """Alloy's tracked .py files; outside a git checkout (an sdist or tarball), every .py under the root."""
    try:
        res = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.py"], capture_output=True, text=True)
        if res.returncode == 0:
            return res.stdout.split()
    except FileNotFoundError:   # no git at all
        pass
    files = []
    for path in ROOT.rglob("*.py"):
        parts = path.relative_to(ROOT).parts
        if not _WALK_SKIP & set(parts) and not any(p.endswith(".egg-info") for p in parts):
            files.append("/".join(parts))
    return sorted(files)


def rewrite_alloy(rep: Report, check: bool) -> list[str]:
    tracked = _alloy_py_files()
    changed = []
    for rel in tracked:
        if any(p.search(rel) for p in ALLOY_SKIP):
            continue
        path = ROOT / rel
        src = path.read_text()
        if not any(r in src for r in ROOTS):
            continue
        if rewrite_file(path, rel, False, rep, check):
            changed.append(rel)
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--newton-src", type=Path, help="Newton checkout at the pinned commit")
    ap.add_argument("--mjw-src", type=Path, help="MuJoCo Warp checkout at the pinned commit")
    ap.add_argument("--check", action="store_true", help="only report Alloy files a run would change (exit 1)")
    ap.add_argument("--alloy-only", action="store_true",
                    help="rewrite Alloy's own files only (a branch made before the vendoring: run after rebasing)")
    ap.add_argument("--vendored-calls", action="store_true",
                    help="apply the _alloy_warp call rewrite to the vendored trees in place (SCR-WP0-COEXIST)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rep = Report()
    if args.vendored_calls:
        log.info("vendored calls rewritten: %d", sum(WARP_CALLS.rewrite_tree(VENDOR, check=False).values()))
    elif not (args.check or args.alloy_only):
        if not (args.newton_src and args.mjw_src):
            ap.error("--newton-src and --mjw-src are required (or --check)")
        copied = vendor_trees(args.newton_src, args.mjw_src, rep)
        apply_patches(check=False)
        log.info("vendored: newton %d files, mujoco_warp %d files", len(copied["newton"]), len(copied["mujoco_warp"]))
    changed = rewrite_alloy(rep, args.check)
    if rekey_epa(args.check):
        changed.append(str(EPA_FILE.relative_to(ROOT)))
    log.info("Alloy files %s: %d; import statements %d, module strings %d rewritten; kept %s",
             "to rewrite" if args.check else "rewritten", len(changed), rep.imports, rep.strings, rep.kept)
    unapplied = apply_patches(check=True) if args.check else []
    calls = WARP_CALLS.rewrite_tree(VENDOR, check=True) if args.check else {}
    if args.check and (changed or unapplied or calls):
        log.error("not vendored yet: %s; patches not applied: %s; unwrapped module= calls: %s", changed[:20],
                  unapplied, sorted(calls)[:20])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
