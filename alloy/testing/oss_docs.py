# SPDX-License-Identifier: Apache-2.0
"""Helpers for the agent-docs tests (``tests/oss_docs``): find the agent docs, their code blocks, links and names.

Standard library only, so the fast docs tests never import the engine.

Example::

    from alloy.testing.oss_docs import agent_docs, python_blocks
    for doc in agent_docs():
        for block in python_blocks(doc):
            compile(block.code, block.label, "exec")
"""

from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC_GLOBS = ("AGENTS.md", "llms.txt", "docs/guide/*.md", "examples/README.md")
_FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_TICKED = re.compile(r"`([^`\s]+)`")
_PATH_ROOTS = ("alloy/", "library/", "docs/", "examples/", "tests/", "scripts/", "bench/")


@dataclass(frozen=True)
class Block:
    """One fenced ```python block: the file it is in, its index there, and the code."""

    doc: Path
    index: int
    code: str

    @property
    def label(self) -> str:
        """``<doc path relative to the repo>#<index>``, e.g. ``AGENTS.md#0``."""
        return f"{self.doc.relative_to(REPO)}#{self.index}"


def agent_docs(root: Path = REPO) -> list[Path]:
    """Every agent-facing doc that exists, in a stable order. Example: ``[p.name for p in agent_docs()]``."""
    out: list[Path] = []
    for pattern in DOC_GLOBS:
        out.extend(sorted(root.glob(pattern)))
    return out


def python_blocks(doc: Path) -> list[Block]:
    """The ```python blocks of ``doc``. Example: ``python_blocks(REPO / "AGENTS.md")[0].code``."""
    return [Block(doc, i, m.group(1)) for i, m in enumerate(_FENCE.finditer(doc.read_text(encoding="utf-8")))]


def local_links(doc: Path) -> list[str]:
    """Relative link targets ``[text](target)`` in ``doc`` (no URLs, no anchors-only). Example: ``["AGENTS.md"]``."""
    text = doc.read_text(encoding="utf-8")
    targets = [t.split("#", 1)[0] for t in _MD_LINK.findall(text)]
    return [t for t in targets if t and "://" not in t and not t.startswith("mailto:")]


def ticked_paths(doc: Path) -> list[str]:
    """Back-ticked repo paths such as ``library/objects/peg_d12.yaml`` (globs and ``<placeholders>`` skipped)."""
    out = []
    for tok in _TICKED.findall(doc.read_text(encoding="utf-8")):
        tok = tok.rstrip(".,:;)")
        if tok.startswith(_PATH_ROOTS) and not any(c in tok for c in "*<>{}()=\"'"):
            out.append(tok)
    return out


def resolve(doc: Path, target: str) -> Path:
    """Where a link in ``doc`` points: relative to the doc's folder. Example: ``resolve(REPO/"llms.txt", "AGENTS.md")``."""
    return (doc.parent / target).resolve()


def class_methods(module: Path, cls: str) -> set[str]:
    """Public methods, properties and ``self.<attr>`` set in ``__init__`` of class ``cls`` and of its base classes
    defined in the same package folder (parsed, not imported).

    Example: ``"add_open_box" in class_methods(scene_py, "Scene")`` (inherited from FixtureMixin)."""
    node = _find_class(module.parent, cls, module)
    if node is None:
        raise LookupError(f"class {cls} not found in {module}")
    names = _own_members(node)
    for base in node.bases:
        if isinstance(base, ast.Name) and _find_class(module.parent, base.id) is not None:
            names |= class_methods(_class_file(module.parent, base.id), base.id)
    return names


def _find_class(folder: Path, cls: str, first: Path | None = None) -> ast.ClassDef | None:
    path = first if first is not None else _class_file(folder, cls)
    if path is None:
        return None
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return node
    return None


def _class_file(folder: Path, cls: str) -> Path | None:
    for path in sorted(folder.glob("*.py")):
        if f"class {cls}" in path.read_text(encoding="utf-8"):
            return path
    return None


def _own_members(node: ast.ClassDef) -> set[str]:
    funcs = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = {f.name for f in funcs if not f.name.startswith("_")}
    for init in (f for f in funcs if f.name == "__init__"):
        names |= {a.attr for a in ast.walk(init) if isinstance(a, ast.Attribute)
                  and isinstance(a.value, ast.Name) and a.value.id == "self" and not a.attr.startswith("_")}
    return names


def public_names(lazy_module: Path = REPO / "alloy/api/_lazy.py") -> set[str]:
    """The top-level public names (``alloy.<name>``) from the API's lazy export table (parsed, not imported)."""
    tree = ast.parse(lazy_module.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
            keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)]
            if "Scene" in keys:
                return set(keys)
    raise LookupError(f"no export table with 'Scene' in {lazy_module}")


def called_names(code: str, var: str) -> set[str]:
    """Attribute names used on variable ``var`` in ``code``: ``scene.add_box(...)`` -> ``{"add_box"}``."""
    names = set()
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == var:
            names.add(node.attr)
    return names


IMPORT_EXTRA = ("trimesh", "coacd", "skimage")  # what the ``import`` extra (mesh processing) brings
_ENV_NEEDS = ("ALLOY_GEN3D_URL",)                 # services an example cannot run without


def missing_requirements(path: Path) -> list[str]:
    """What ``path`` (an example script) needs that this environment lacks: third-party modules it imports
    (``required_imports``), the ``import`` extra when its docstring says so, command-line arguments it cannot run
    without, and what ``text_requirements`` finds in its text.

    Example: ``missing_requirements(REPO / "examples/envs/04_train_ppo.py")`` -> ``["module torch"]`` without PyTorch."""
    import importlib.util

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    mods = required_imports(tree)
    doc = ast.get_docstring(tree) or ""
    if "import`` extra" in doc or "[import]" in doc:
        mods |= set(IMPORT_EXTRA)
    local = {p.stem for p in path.parent.glob("*.py")}
    missing = [f"module {m}" for m in sorted(mods - local - set(sys.builtin_module_names))
               if importlib.util.find_spec(m) is None]
    args = required_arguments(tree)
    if args:
        missing.append(f"arguments {', '.join(args)} (a tool over files made elsewhere: its docstring shows the run)")
    return missing + text_requirements(text)


def required_imports(tree: ast.Module) -> set[str]:
    """Top-level module names a script imports anywhere (also inside ``main()``, e.g. ``import rerun as rr``), except
    imports under a ``try`` (an optional module the script can run without).

    Example: ``required_imports(ast.parse("def main():\n    import rerun as rr\n"))`` -> ``{"rerun"}``."""
    guarded = {id(n) for t in ast.walk(tree) if isinstance(t, ast.Try) for b in t.body for n in ast.walk(b)}
    mods = set()
    for n in ast.walk(tree):
        if id(n) in guarded:
            continue
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            mods.add(n.module.split(".")[0])
    return mods


def required_arguments(tree: ast.Module) -> list[str]:
    """The command-line arguments a script cannot run without: ``add_argument`` calls with ``required=True``, and
    positional ones (no leading ``-``) unless ``nargs`` is ``"?"`` or ``"*"`` or they have a default.

    Example: ``required_arguments(ast.parse('ap.add_argument("--scenes", nargs="+", required=True)'))`` ->
    ``["--scenes"]``."""
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args and isinstance(node.args[0], ast.Constant)):
            continue
        flag, kw = str(node.args[0].value), {k.arg: k.value for k in node.keywords if k.arg}
        if flag.startswith("-"):
            needed = isinstance(kw.get("required"), ast.Constant) and kw["required"].value is True
        else:
            nargs = kw.get("nargs")
            needed = "default" not in kw and not (isinstance(nargs, ast.Constant) and nargs.value in ("?", "*"))
        if needed:
            out.append(flag)
    return out


_NO_MESHES = re.compile(r"SceneError: '([^']+)' \(imported library [^)]*\) has no mesh files on this machine")


def missing_library_meshes(output: str) -> str | None:
    """The skip reason when a script stopped only because this machine lacks a library object's mesh files (the
    build's ``SceneError``, which names the object and its fix), else None.

    Example: ``missing_library_meshes("SceneError: 'ycb_077_rubiks_cube' (imported library ycb_gso) has no mesh files
    on this machine")`` -> ``"library meshes of ycb_077_rubiks_cube (alloy fetch ycb_077_rubiks_cube)"``."""
    names = list(dict.fromkeys(_NO_MESHES.findall(output)))
    if not names:
        return None
    return f"library meshes of {', '.join(names)} (alloy fetch {' '.join(names)})"


# pip names in a "fix: pip install ..." line whose import name differs, and what each Alloy extra brings.
_PIP_MODULES = {"rsl-rl-lib": ("rsl_rl",), "stable-baselines3": ("stable_baselines3",), "scikit-image": ("skimage",),
                "usd-core": ("pxr",), "rerun-sdk": ("rerun",), "huggingface-hub": ("huggingface_hub",)}
_EXTRA_MODULES = {"import": IMPORT_EXTRA, "usd": ("pxr",), "envs": ("torch", "huggingface_hub")}
_NO_MODULE = re.compile(r"(?:ModuleNotFoundError|ImportError): No module named '([A-Za-z0-9_]+)")
_PIP_INSTALL = re.compile(r"pip install ([^\n]+)")
_PIP_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([A-Za-z0-9_,-]+)\])?")


def _pip_modules(requirement: str) -> tuple[str, ...]:
    """Import names behind one pip requirement. Example: ``_pip_modules('"viser>=1.1.1",')`` -> ``("viser",)``."""
    m = _PIP_NAME.match(requirement.strip("\"'`,.;:()"))
    if not m:
        return ()
    name, extras = m.group(1).lower().replace("_", "-"), m.group(2)
    if name == "alloy-engine":
        return tuple(mod for e in (extras or "").split(",") for mod in _EXTRA_MODULES.get(e, ()))
    return _PIP_MODULES.get(name, (name.replace("-", "_"),))


def _absent(module: str) -> bool:
    """True when ``module`` cannot be imported here. Example: ``_absent("zz_absent")`` -> ``True``."""
    import importlib.util

    try:
        return importlib.util.find_spec(module) is None
    except (ImportError, ValueError):
        return True


def missing_optional_package(output: str) -> str | None:
    """The skip reason when a run stopped only because an optional package is not installed here, else None. Read
    from the end of its error output: a ``No module named 'x'``, or an error that says "not installed" with a
    ``pip install ...`` fix. It is a reason only when a named package really cannot be imported in this
    environment, so a wrong import in a snippet (``No module named 'alloy.nope'``) stays a failure.

    Example: ``missing_optional_package("ModuleNotFoundError: No module named 'zz_absent'")`` ->
    ``"module zz_absent (optional package not installed here)"``."""
    tail = "\n".join(output.strip().splitlines()[-12:])
    named = list(_NO_MODULE.findall(tail))
    if "not installed" in tail:
        for line in _PIP_INSTALL.findall(tail):
            for token in line.split():
                if token in ("or", "and", "then") or token.startswith("-"):
                    break
                named.extend(_pip_modules(token))
    absent = [m for m in dict.fromkeys(named) if m != "alloy" and _absent(m)]
    return f"module {', '.join(absent)} (optional package not installed here)" if absent else None


def missing_modules(code: str) -> list[str]:
    """Third-party modules a doc block imports that this environment lacks (imports under ``try`` excepted).

    Example: ``missing_modules("import zz_absent\n")`` -> ``["module zz_absent"]``."""
    try:
        mods = required_imports(ast.parse(code))
    except SyntaxError:
        return []
    return [f"module {m}" for m in sorted(mods) if m != "alloy" and _absent(m)]


def text_requirements(text: str) -> list[str]:
    """What a script or doc block needs beyond its imports: service URLs it reads from the environment, and
    huggingface_hub when it reads the object library on the Hub (``alloy.assets.hub``; the library is public, so no
    login).

    Example: ``text_requirements("from alloy.assets.hub.fetch import fetch")`` -> ``[]`` with huggingface_hub."""
    import importlib.util
    import os

    missing = [f"environment {v}" for v in _ENV_NEEDS if v in text and not os.environ.get(v)
               and f"--{v.lower()}" not in text and "default" not in _env_line(text, v)]
    if "alloy.assets.hub" in text and importlib.util.find_spec("huggingface_hub") is None:
        missing.append("module huggingface_hub")
    return missing


def _env_line(text: str, var: str) -> str:
    return next((ln.lower() for ln in text.splitlines() if var in ln and "environ" in ln), "")


def example_rows(root: Path = REPO) -> dict[str, list[tuple[str, str]]]:
    """``{area: [(path, first docstring sentence), ...]}`` for every ``examples/<area>/*.py``, sorted.

    Example: ``example_rows()["api"][0]`` -> ``("examples/api/01_drop_box.py", "Drop a box onto a table ...")``."""
    rows: dict[str, list[tuple[str, str]]] = {}
    paths = sorted(p for p in (root / "examples").glob("*/*") if p.suffix in (".py", ".sh") and p.name[0] != "_")
    for path in paths:
        rows.setdefault(path.parent.name, []).append((str(path.relative_to(root)), _first_sentence(path)))
    return rows


def _first_sentence(path: Path) -> str:
    """A script's description: its docstring's first paragraph (.py) or its first comment line (.sh)."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        doc = ast.get_docstring(ast.parse(text)) or ""
        return doc.split("\n\n", 1)[0].replace("\n", " ").strip()
    lines = [ln[1:].strip() for ln in text.splitlines()[1:] if ln.startswith("#")]
    return lines[0] if lines else ""


def examples_table(root: Path = REPO) -> str:
    """The examples index as Markdown tables, one per area, for ``examples/README.md``.

    Example: ``print(examples_table())`` (or ``python -m alloy.testing.oss_docs``)."""
    parts = []
    for area, rows in example_rows(root).items():
        parts.append(f"## {area}\n\n| Example | What it shows |\n|---|---|")
        parts.extend(f"| [`{Path(p).name}`]({Path(p).relative_to('examples')}) | {d} |" for p, d in rows)
        parts.append("")
    return "\n".join(parts)


if __name__ == "__main__":
    print(examples_table())
