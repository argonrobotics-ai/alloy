# SPDX-License-Identifier: Apache-2.0
"""Build Alloy's installable wheel: ``alloy-engine``, with Newton and MuJoCo Warp vendored inside it.

    python tools/deps/wheels.py --out wheelhouse [--tree <release tree>]

Then, in a fresh environment, one command installs it (CPU or GPU; mujoco, warp-lang, numpy etc. come from PyPI):

    pip install --find-links wheelhouse "alloy-engine==<version printed in the JSON>"

Prints one JSON document: the version, the wheel's path, size and sha256, the vendored engine copies and the install
command. Exit 1 on a build failure, with the failing command and what to change.

Alloy's pinned Newton and MuJoCo Warp are not separate wheels: they ship as ``alloy/_vendor/newton`` and
``alloy/_vendor/mujoco_warp`` (``kind: vendored`` in ``pins.json``; ``alloy/_vendor/VENDORED.md`` lists their base
commits and patches), so the wheel requires only PyPI packages and never replaces a user's own newton or
mujoco-warp. The build refuses a tree whose vendored copy is missing, lacks its LICENSE, or is not the pinned commit.

The alloy wheel ships the whole staged ``alloy`` package (code, the vendored engine, ``_data/`` library and import
cards, the C route sources, camera tables, tactile calibrations, licence notices), plus the tree's own
``[tool.setuptools.package-data]`` globs. After the build, ``check_wheel`` refuses a wheel that lacks a staged file or
in which a package-data glob matches nothing (a release wheel once shipped without ``alloy/route/_mjc/step.c``, so
every CPU robot scene failed).

Example (Python):
    >>> meta = TreeMeta(extras={"mesh": ["coacd"]})
    >>> "mujoco==3.17.0" in release_pyproject("0.0.1", ["mujoco==3.17.0"], meta)
    True
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from pins import load_pins, requirements, vendored  # noqa: E402
from stage import StageResult, stage_tree  # noqa: E402

log = logging.getLogger("alloy.deps")
DIST_NAME = "alloy-engine"  # pip install alloy-engine; import alloy
# Every file of the staged package ships (the staging step already dropped caches and compiled binaries).
RELEASE_PACKAGE_DATA = {"alloy": ["**/*"]}


class BuildError(RuntimeError):
    """A wheel build step failed; the message names the command and what to change."""


def run(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    """Run ``cmd``; raise ``BuildError`` with its stderr tail on failure."""
    log.info("run: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False, env=env)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout)[-1500:]
        raise BuildError(f"{' '.join(cmd)} failed ({proc.returncode}):\n{tail}")
    return proc.stdout


def commit_time(repo: Path, commit: str = "HEAD") -> int:
    """Committer time of ``commit`` (s since the epoch); 315532800 (1980-01-01, the zip minimum) when unknown."""
    out = subprocess.run(["git", "-C", str(repo), "show", "-s", "--format=%ct", commit], capture_output=True,
                         text=True, check=False)
    return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip().isdigit() else 315532800


def build_wheel(src: Path, out: Path, epoch: int) -> Path:
    """``uv build --wheel`` of the project in ``src`` into ``out``; return the new wheel.

    ``epoch`` becomes ``SOURCE_DATE_EPOCH``, so setuptools stamps every zip entry with the source commit's time and a
    rebuild of the same commit is byte-identical."""
    before = set(out.glob("*.whl"))
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(max(epoch, 315532800)))
    run(["uv", "build", "--wheel", "--no-sources", "--out-dir", str(out), str(src)], env=env)
    new = sorted(set(out.glob("*.whl")) - before)
    if len(new) != 1:
        raise BuildError(f"expected one new wheel from {src}, found {[p.name for p in new]}")
    return new[0]


def describe(wheel: Path) -> dict[str, object]:
    """Name, path, size and sha256 of a wheel."""
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    return {"file": wheel.name, "path": str(wheel), "bytes": wheel.stat().st_size, "sha256": digest}


# [project] fields the release pyproject copies from the tree as written: the release export sets the summary and the
# README there, so the wheel's METADATA (``pip show``, twine's long-description check) matches the sdist's.
PROJECT_FIELDS = ("description", "readme", "requires-python", "license", "license-files", "authors", "maintainers",
                  "keywords", "classifiers", "urls")
PROJECT_DEFAULTS = {"requires-python": ">=3.11,<3.13", "license": "Apache-2.0"}


@dataclass(frozen=True)
class TreeMeta:
    """What the release pyproject takes from the source tree's pyproject.toml."""

    extras: dict[str, list[str]] = field(default_factory=dict)       # optional dependencies, without ``dev``
    scripts: dict[str, str] = field(default_factory=dict)            # console scripts
    package_data: dict[str, list[str]] = field(default_factory=dict)  # [tool.setuptools.package-data]
    dependencies: list[str] = field(default_factory=list)            # [project] dependencies
    project: dict[str, Any] = field(default_factory=dict)            # PROJECT_FIELDS the tree sets (summary, readme)
    entry_points: dict[str, dict[str, str]] = field(default_factory=dict)  # plug-in groups (isaaclab.tasks)


def tree_meta(tree: Path) -> TreeMeta:
    """The source tree's optional dependencies (without ``dev``), console scripts, entry-point groups, package-data
    globs and ``PROJECT_FIELDS``.

    >>> meta = tree_meta(Path("."))
    >>> "mesh" in meta.extras and "dev" not in meta.extras and "**/*" in meta.package_data["alloy"]
    True
    >>> "description" in meta.project and "version" not in meta.project
    True
    >>> meta.entry_points["isaaclab.tasks"]
    {'alloy': 'alloy.isaaclab.register'}
    """
    doc = tomllib.loads((tree / "pyproject.toml").read_text())
    project = doc["project"]
    extras = {k: list(v) for k, v in project.get("optional-dependencies", {}).items() if k != "dev"}
    data = doc.get("tool", {}).get("setuptools", {}).get("package-data", {})
    return TreeMeta(extras, dict(project.get("scripts", {})), {k: list(v) for k, v in data.items()},
                    list(project.get("dependencies", [])), {k: project[k] for k in PROJECT_FIELDS if k in project},
                    {g: dict(v) for g, v in project.get("entry-points", {}).items()})


def dist_name(requirement: str) -> str:
    """The normalised distribution name of a requirement string.

    >>> dist_name("etils[epath]>=1"), dist_name("mujoco_warp @ git+https://x")
    ('etils', 'mujoco-warp')
    """
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    if not m:
        raise BuildError(f"cannot read a package name from the requirement {requirement!r} in pyproject.toml")
    return re.sub(r"[-_.]+", "-", m.group(1)).lower()


def release_requirements(pins: dict[str, Any], meta: TreeMeta) -> list[str]:
    """The wheel's dependencies: the PyPI pins (``pins.requirements``), then the tree's other ``[project]``
    dependencies (e.g. MuJoCo Warp's runtime deps). A tree that still requires a vendored package is refused.

    >>> release_requirements(load_pins(), TreeMeta(dependencies=["mujoco>=3", "absl-py"]))[-1]
    'absl-py'
    """
    reqs = requirements(pins)
    vendored_names = {d["name"] for d in pins["deps"] if d["kind"] == "vendored"}
    clash = [r for r in meta.dependencies if dist_name(r) in vendored_names]
    if clash:
        raise BuildError(f"pyproject.toml still requires {clash}, which ship vendored inside alloy/_vendor: drop "
                         "them from [project] dependencies")
    have = {dist_name(r) for r in reqs}
    return reqs + [r for r in meta.dependencies if dist_name(r) not in have]


def merged_package_data(tree_data: dict[str, list[str]]) -> dict[str, list[str]]:
    """The tree's package-data globs plus ``RELEASE_PACKAGE_DATA`` (each package's globs in order, no repeats).

    >>> merged_package_data({"alloy.route._mjc": ["*.c"], "alloy": ["x/*"]})
    {'alloy.route._mjc': ['*.c'], 'alloy': ['x/*', '**/*']}
    """
    out = {k: list(v) for k, v in tree_data.items()}
    for pkg, globs in RELEASE_PACKAGE_DATA.items():
        out.setdefault(pkg, [])
        out[pkg] += [g for g in globs if g not in out[pkg]]
    return out


def release_pyproject(version: str, requires: list[str], meta: TreeMeta | None = None, name: str = DIST_NAME) -> str:
    """The pyproject.toml of the staged release tree: ``requires`` as dependencies, ``meta``'s extras, scripts,
    entry-point groups (Isaac Lab finds ``physics=alloy`` through ``isaaclab.tasks``) and ``PROJECT_FIELDS``
    (summary, readme, licence; ``PROJECT_DEFAULTS`` where the tree sets none), and the tree's package-data merged
    with ``RELEASE_PACKAGE_DATA``.

    >>> doc = tomllib.loads(release_pyproject("0.0.1", [], TreeMeta(project={"description": "An engine"})))
    >>> doc["project"]["description"], doc["project"]["license"]
    ('An engine', 'Apache-2.0')
    >>> eps = {"isaaclab.tasks": {"alloy": "alloy.isaaclab.register"}}
    >>> tomllib.loads(release_pyproject("0.0.1", [], TreeMeta(entry_points=eps)))["project"]["entry-points"] == eps
    True
    """
    meta = meta or TreeMeta()
    deps = "".join(f'    "{r}",\n' for r in requires)
    extra_lines = "".join(f"{json.dumps(k)} = {json.dumps(v)}\n" for k, v in meta.extras.items())
    script_lines = "".join(f"{json.dumps(k)} = {json.dumps(v)}\n" for k, v in meta.scripts.items())
    entry_lines = "".join(f"[project.entry-points.{json.dumps(g)}]\n"
                          + "".join(f"{json.dumps(k)} = {json.dumps(v)}\n" for k, v in eps.items()) + "\n"
                          for g, eps in meta.entry_points.items())
    data_lines = "".join(f"{json.dumps(k)} = {json.dumps(v)}\n"
                         for k, v in merged_package_data(meta.package_data).items())
    fields = {**PROJECT_DEFAULTS, **meta.project}
    field_lines = "".join(f"{k} = {toml_value(v)}\n" for k, v in fields.items())
    return (f'[project]\nname = "{name}"\nversion = "{version}"\n{field_lines}'
            f"dependencies = [\n{deps}]\n\n[project.optional-dependencies]\n{extra_lines}\n"
            + (f"[project.scripts]\n{script_lines}\n" if script_lines else "")
            + entry_lines
            + '[build-system]\nrequires = ["setuptools>=77"]\nbuild-backend = "setuptools.build_meta"\n\n'
            '[tool.setuptools.packages.find]\ninclude = ["alloy*"]\n\n'
            f"[tool.setuptools.package-data]\n{data_lines}")


def toml_value(value: Any) -> str:
    """A TOML literal for a pyproject value read with ``tomllib``: a string, number, bool, array or inline table.

    >>> toml_value({"file": "README.md", "content-type": "text/markdown"})
    '{"file" = "README.md", "content-type" = "text/markdown"}'
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(k)} = {toml_value(v)}" for k, v in value.items()) + "}"
    raise BuildError(f"cannot write the pyproject value {value!r} ({type(value).__name__}) into the release "
                     "pyproject: use a string, number, list or table in pyproject.toml's [project]")


def stage_project_files(tree: Path, stage: Path, project: dict[str, Any]) -> list[str]:
    """Copy the files the tree's ``readme`` and ``license-files`` name into ``stage`` (where staging has not already
    put them), so the release pyproject's references resolve; the relative paths copied.

    Raises ``BuildError`` when a named readme is missing or a path leaves the tree."""
    readme = project.get("readme")
    readme = readme.get("file") if isinstance(readme, dict) else readme  # a table can hold ``text`` instead
    files = []
    if readme:
        if not (tree / readme).is_file():
            raise BuildError(f"pyproject.toml's readme names {readme!r}, which is not in {tree}: add the file or fix "
                             "[project] readme")
        files.append(tree / readme)
    files += [f for g in project.get("license-files", []) for f in sorted(tree.glob(g)) if f.is_file()]
    root, copied = tree.resolve(), []
    for f in files:
        if not f.resolve().is_relative_to(root):
            raise BuildError(f"pyproject.toml names {f}, outside the tree {tree}: name files inside the tree")
        rel = f.resolve().relative_to(root)
        if not (stage / rel).exists():
            (stage / rel).parent.mkdir(parents=True, exist_ok=True)
            (stage / rel).write_bytes(f.read_bytes())
            copied.append(rel.as_posix())
    return copied


def glob_regex(pattern: str) -> re.Pattern[str]:
    """A setuptools package-data glob as a regex over ``/`` paths: ``**/`` any directories, ``*`` within one.

    >>> bool(glob_regex("alloy/_data/**/*").fullmatch("alloy/_data/library/a.yaml"))
    True
    >>> bool(glob_regex("alloy/route/_mjc/*.c").fullmatch("alloy/route/_mjc/sub/x.c"))
    False
    """
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        else:
            ch = pattern[i]
            out.append({"*": "[^/]*", "?": "[^/]"}.get(ch, re.escape(ch)))
            i += 1
    return re.compile("".join(out))


def package_data_misses(names: Iterable[str], package_data: dict[str, list[str]]) -> list[str]:
    """``"<package>: <glob>"`` for each package-data glob that matches no file among ``names`` (wheel entries).

    >>> package_data_misses(["alloy/route/_mjc/step.c"], {"alloy.route._mjc": ["*.c", "*.h"]})
    ['alloy.route._mjc: *.h']
    """
    files = [n for n in names if not n.endswith("/")]
    misses = []
    for pkg, globs in package_data.items():
        prefix = "(?:[^/]+/)*" if pkg == "*" else re.escape(pkg.replace(".", "/") + "/")
        for g in globs:
            rx = re.compile(prefix + glob_regex(g).pattern)
            if not any(rx.fullmatch(n) for n in files):
                misses.append(f"{pkg}: {g}")
    return misses


def check_wheel(wheel: Path, stage: Path, package_data: dict[str, list[str]]) -> dict[str, int]:
    """Refuse a wheel that lacks a file of the staged ``alloy`` package or whose package-data globs match nothing.

    Returns the counts checked; raises ``BuildError`` naming the missing files and globs."""
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    staged = {p.relative_to(stage).as_posix() for p in (stage / "alloy").rglob("*") if p.is_file()}
    absent = sorted(staged - names)
    misses = package_data_misses(names, package_data)
    if absent or misses:
        raise BuildError(f"{wheel.name} is incomplete: {len(absent)} staged files absent (first: {absent[:8]}); "
                         f"package-data globs matching nothing: {misses}. Fix [tool.setuptools.package-data] in "
                         "pyproject.toml or RELEASE_PACKAGE_DATA in tools/deps/wheels.py")
    return {"staged_files": len(staged), "package_data_globs": sum(len(v) for v in package_data.values())}


def default_assets(tree: Path) -> Path | None:
    """The alloy-assets checkout whose import cards ship in the wheel: ``$ALLOY_ASSETS``, else a sibling checkout."""
    raw = os.environ.get("ALLOY_ASSETS", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    sibling = tree.parent / "alloy-assets"
    return sibling.resolve() if (sibling / "imports").is_dir() else None


def check_vendored(stage: Path, pins: dict[str, Any]) -> list[dict[str, str]]:
    """Refuse a staged tree whose vendored engine copy (``kind: vendored`` in ``pins.json``) is missing, lacks its
    LICENSE, or is not named at its pinned commit in ``alloy/_vendor/VENDORED.md``; return one row per copy."""
    rows, bad = [], []
    for dep in (d for d in pins["deps"] if d["kind"] == "vendored"):
        got = vendored(dep, stage / "alloy")
        rows.append({"name": dep["name"], "path": dep["path"], "pin": dep["pin"][:8]})
        if not got["match"]:
            lacks = [k for k in ("package", "licence", "pin_in_vendored_md") if not got[k]]
            bad.append(f"{dep['path']} (lacks: {', '.join(lacks)})")
    if bad:
        raise BuildError(f"vendored engine incomplete: {'; '.join(bad)}. The tree must carry the pinned copies with "
                         "their LICENSE files, and alloy/_vendor/VENDORED.md must name each pinned commit")
    return rows


def alloy_wheel(args: argparse.Namespace, work: Path, out: Path) -> tuple[Path, StageResult, list[dict[str, str]]]:
    """Stage the release tree (package, vendored engine, library, import cards), build the alloy wheel and check it
    is complete."""
    stage = work / "alloy-stage"
    tree = args.tree.resolve()
    assets = None if args.no_imports else (args.assets.resolve() if args.assets else default_assets(tree))
    res = stage_tree(tree, stage, assets)
    pins = load_pins()
    engine = check_vendored(stage, pins)
    meta = tree_meta(tree)
    stage_project_files(tree, stage, meta.project)
    (stage / "pyproject.toml").write_text(release_pyproject(res.version, release_requirements(pins, meta), meta,
                                                            args.name))
    wheel = build_wheel(stage, out, commit_time(tree))
    counts = check_wheel(wheel, stage, merged_package_data(meta.package_data))
    log.info("%s carries all %d staged files (vendored: %s) and %d package-data globs", wheel.name,
             counts["staged_files"], ", ".join(f"{e['name']} {e['pin']}" for e in engine), counts["package_data_globs"])
    return wheel, res, engine


def build_all(args: argparse.Namespace) -> dict[str, object]:
    """Build the alloy wheel; return the JSON report. Without ``--work`` the scratch directory (the staged tree) is
    a temp dir removed on exit, success or failure; ``--work`` keeps it."""
    if getattr(args, "skip_engine", False) or getattr(args, "newton_src", None) is not None:
        log.warning("--skip-engine and --newton-src are ignored: Newton and MuJoCo Warp ship inside the alloy wheel")
    if args.work is not None:
        return _build_in(args, args.work.resolve())
    with tempfile.TemporaryDirectory(prefix="alloy-wheels-") as tmp:
        return _build_in(args, Path(tmp).resolve())


def _build_in(args: argparse.Namespace, work: Path) -> dict[str, object]:
    """``build_all`` with its scratch directory chosen."""
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    wheel, res, engine = alloy_wheel(args, work, out)
    install = f'pip install --find-links {out} "{args.name}=={res.version}"'
    return {"version": res.version, "wheels": [describe(wheel)], "vendored": engine, "install": install,
            "imports": res.imports, "assets_commit": res.assets_commit}


def main(argv: list[str] | None = None) -> int:
    """CLI entry: one JSON document on stdout; exit 0 built, 1 failed."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True, help="wheelhouse directory")
    ap.add_argument("--tree", type=Path, default=Path(__file__).resolve().parents[2], help="Alloy source tree")
    ap.add_argument("--work", type=Path, default=None,
                    help="scratch directory, kept afterwards (default: a temp dir, removed on exit)")
    ap.add_argument("--name", default=DIST_NAME, help="distribution name (import name stays alloy)")
    # Accepted and ignored since the engine is vendored (callers written for the separate engine wheels still run).
    ap.add_argument("--skip-engine", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--newton-src", type=Path, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--assets", type=Path, default=None,
                    help="alloy-assets checkout whose imports/ cards ship in the wheel (default: $ALLOY_ASSETS, "
                         "else a sibling checkout)")
    ap.add_argument("--no-imports", action="store_true", help="ship no imported-library cards")
    args = ap.parse_args(argv)
    try:
        doc, code = build_all(args), 0
    except (BuildError, subprocess.CalledProcessError, ValueError) as exc:
        doc, code = {"error": str(exc)}, 1
    json.dump(doc, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return code


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    sys.exit(main())
