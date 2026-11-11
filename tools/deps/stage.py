# SPDX-License-Identifier: Apache-2.0
"""Stage an Alloy source tree as a buildable release tree: the ``alloy`` package with the card library inside it.

    python tools/deps/stage.py <tree> <stage> [<assets>]   # one JSON document: stage, version, files, imports, ...

The library is copied to ``alloy/_data/library``, as ``setup.py`` does for a plain ``pip install .``. With an alloy-assets
checkout, its imported libraries' cards (``imports/<source>/``: cards, manifests, attributions, MJCF; never meshes,
which are fetched on first use) go to ``alloy/_data/imports/<source>/``; a card marked ``redistribute: no`` is left
out, and each source reports how many cards carry no redistribution flag at all. ``alloy.cards.library.default_root()``
falls back to ``alloy/_data/library`` when there is no ``<repo>/library``.

Example (Python):
    >>> "+g" in tree_version(Path("."))      # a git checkout: <version>+g<commit8>
    True
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("alloy.deps")
# Caches, binaries compiled on this host (a py3-none-any wheel must never carry a liballoy_mjc.so built here), and
# VCS / desktop dotfiles (the vendored Newton has two .gitignore files; a wheel's globs skip dotfiles anyway).
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "*.so", "*.dylib", "*.dll", "*.o",
                                ".gitignore", ".gitattributes", ".DS_Store")
# Copied to alloy/_data beside the library (TRAIN-AUDIT fix 1, 2026-11-11): examples/ whole, AGENTS.md (user
# 17:20Z), and of docs/ ONLY the public set (an allow-list: never docs/notes, oss, research, ENV.md or SPEC.md).
# Same lists in setup.py, tools/release/stage_release.py and MANIFEST.in (tests/core/test_packaging.py).
DOCS_PUBLIC = ("public", "known_gaps", "guide")   # public pages, the gap pages limits.md links, guide/ as reference
#                                                  (coordinator 17:5xZ; physics ruling (A): AGENTS.md and code point at it)
DATA_FILES = ("AGENTS.md",)
TOP_FILES = ("LICENSE", "NOTICE", "README.md")
IMPORT_SUFFIXES = frozenset({".yaml", ".json", ".jsonl", ".md", ".xml"})  # cards and their text; no meshes
NO_REDISTRIBUTE = re.compile(r"\bredistribute:\s*no\b")


@dataclass(frozen=True)
class StageResult:
    """What ``stage_tree`` staged: the release version and, per imported source, the copy counts."""

    version: str
    imports: dict[str, dict[str, int]]
    assets_commit: str | None


class StageError(RuntimeError):
    """Staging failed; the message says what to change."""


def tree_version(tree: Path) -> str:
    """``<pyproject version>+g<commit8>`` (``.dirty`` appended when tracked files are modified)."""
    text = (tree / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.M)
    if not m:
        raise StageError(f"{tree}/pyproject.toml has no static version")
    git = ["git", "-C", str(tree)]
    sha = subprocess.run([*git, "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True, check=False)
    if sha.returncode != 0:
        return m.group(1)
    dirty = subprocess.run([*git, "diff", "--quiet", "HEAD", "--", "alloy", "library"], check=False).returncode
    return f"{m.group(1)}+g{sha.stdout.strip()}" + (".dirty" if dirty else "")


def _copy_import_source(src: Path, dest: Path) -> dict[str, int]:
    """Copy one ``imports/<source>/`` tree's text files; skip ``redistribute: no`` cards and other files."""
    counts = {"copied": 0, "skipped_no_redistribute": 0, "cards_unflagged": 0, "skipped_other": 0}
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix not in IMPORT_SUFFIXES:
            counts["skipped_other"] += 1
            continue
        if f.suffix == ".yaml":
            text = f.read_text(errors="replace")
            if NO_REDISTRIBUTE.search(text):
                counts["skipped_no_redistribute"] += 1
                continue
            counts["cards_unflagged"] += "redistribute:" not in text
        target = dest / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        counts["copied"] += 1
    return counts


def stage_imports(assets: Path, stage: Path) -> dict[str, dict[str, int]]:
    """Copy every ``<assets>/imports/<source>/`` into ``<stage>/alloy/_data/imports/<source>/``; counts per source."""
    root = assets / "imports"
    if not root.is_dir():
        raise StageError(f"{assets} has no imports/ directory: pass an alloy-assets checkout, or none")
    sources = sorted(p for p in root.iterdir() if p.is_dir())
    return {src.name: _copy_import_source(src, stage / "alloy" / "_data" / "imports" / src.name) for src in sources}


def _git_commit(path: Path) -> str | None:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True,
                         check=False)
    return out.stdout.strip() if out.returncode == 0 else None


def stage_tree(tree: Path, stage: Path, assets: Path | None = None) -> StageResult:
    """Copy ``tree``'s package and library (and ``assets``' import cards) into a fresh ``stage``."""
    if not (tree / "alloy" / "__init__.py").is_file() or not (tree / "library").is_dir():
        raise StageError(f"{tree} is not an Alloy source tree (needs alloy/__init__.py and library/)")
    version = tree_version(tree)
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(tree / "alloy", stage / "alloy", ignore=IGNORE)
    shutil.copytree(tree / "library", stage / "alloy" / "_data" / "library", ignore=IGNORE)
    data = stage / "alloy" / "_data"   # docs and examples ride inside the package too (python -m alloy.docs)
    pairs = [(tree / n, data / n) for n in ("examples", *DATA_FILES)]
    for src, dst in pairs + [(tree / "docs" / n, data / "docs" / n) for n in DOCS_PUBLIC]:
        if src.is_dir():
            shutil.copytree(src, dst, ignore=IGNORE)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    for name in TOP_FILES:
        if (tree / name).is_file():
            shutil.copy2(tree / name, stage / name)
    imports = stage_imports(assets, stage) if assets is not None else {}
    log.info("staged %s at %s (version %s, imports %s)", tree, stage, version, sorted(imports))
    return StageResult(version, imports, _git_commit(assets) if assets is not None else None)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``stage.py <tree> <stage>``; one JSON document on stdout."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) not in (2, 3):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    tree, stage = Path(args[0]).resolve(), Path(args[1]).resolve()
    assets = Path(args[2]).resolve() if len(args) == 3 else None
    try:
        res = stage_tree(tree, stage, assets)
    except (StageError, OSError) as exc:
        json.dump({"error": str(exc)}, sys.stdout)
        return 1
    files = sum(1 for p in stage.rglob("*") if p.is_file())
    json.dump({"stage": str(stage), "version": res.version, "files": files, "imports": res.imports,
               "assets_commit": res.assets_commit}, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    sys.exit(main())
