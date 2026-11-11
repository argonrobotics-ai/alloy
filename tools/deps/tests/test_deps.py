# SPDX-License-Identifier: Apache-2.0
"""Offline tests for tools/deps: the pin inventory stays in step with the code, and the release staging works.

    python -m pytest tools/deps/tests -q
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib
from pathlib import Path

import pytest

DEPS = Path(__file__).resolve().parents[1]
REPO = DEPS.parents[1]
sys.path.insert(0, str(DEPS))

import coexist_probe  # noqa: E402
import pins  # noqa: E402
import release_check  # noqa: E402
import stage  # noqa: E402
import usd_extra_check  # noqa: E402
import wheel_smoke  # noqa: E402
import wheels  # noqa: E402


def _const(path: str, name: str) -> str:
    """A ``NAME... = "<40 hex>"`` constant from a repo source file."""
    m = re.search(rf'^{name}(?::[^=]+)?\s*=\s*"([0-9a-f]{{40}})"', (REPO / path).read_text(), flags=re.M)
    assert m, f"{name} not found in {path}"
    return m.group(1)


def _dep(name: str) -> dict:
    return next(d for d in pins.load_pins()["deps"] if d["name"] == name)


def test_inventory_matches_the_code_pins():
    assert _dep("newton")["pin"] == _const("alloy/core/newton_compat.py", "NEWTON_PIN")
    assert _dep("mujoco-warp")["pin"] == _const("alloy/core/newton_compat.py", "MJWARP_PIN")
    assert _dep("mujoco-warp")["pin"] == _const("alloy/precision/mjw_pin.py", "PIN")
    if (REPO / "vendor/newton/PIN").is_file():  # the private bundle's pin file, until the vendor route drops it
        pin_file = (REPO / "vendor/newton/PIN").read_text()
        assert f"head {_dep('newton')['pin']}" in pin_file
        assert f"base {_dep('newton')['base']}" in pin_file
    if (REPO / "alloy/_vendor").is_dir():  # vendor route: VENDORED.md names both pinned commits
        doc, ok = pins.check(pins.load_pins())
        assert all(r["match"] for r in doc["deps"] if r["name"] in ("newton", "mujoco-warp")), doc


def test_newton_patches_end_at_the_pin_and_each_has_a_reason():
    newton = _dep("newton")
    assert newton["patches"][-1]["commit"] == newton["pin"]
    states = {"merged", "pr-open", "not-posted"}
    for p in newton["patches"]:
        assert p["why"] and p["upstream"] in states, p
        assert (p["pr"] is None) == (p["upstream"] == "not-posted"), p


def test_every_dep_says_why_and_how_it_ships():
    for dep in pins.load_pins()["deps"]:
        assert len(dep["why"]) > 20 and dep["routes"], dep["name"]
        if dep["kind"] == "vendored":
            assert dep["path"] == "alloy/_vendor/" + dep["module"].split(".")[-1], dep["name"]
            assert {"vendored", "release"} <= set(dep["routes"]), dep["name"]


def test_bad_schema_is_refused(tmp_path):
    bad = tmp_path / "pins.json"
    bad.write_text('{"schema": "x", "deps": []}')
    with pytest.raises(ValueError, match="schema"):
        pins.load_pins(bad)


def test_requirements_are_pypi_only_and_leave_the_engine_out():
    reqs = pins.requirements(pins.load_pins())
    assert reqs[:2] == ["mujoco==3.17.0", "warp-lang==1.18.0"]
    assert not any(r.startswith(("newton", "mujoco-warp")) or "+g" in r or "extra ==" in r for r in reqs)


def _vendor(alloy_dir: Path, licence: bool = True, pins_md: bool = True) -> None:
    """Fake vendored newton / mujoco_warp under ``alloy_dir/_vendor`` (optionally without licences or pins)."""
    lines = []
    for dep in (d for d in pins.load_pins()["deps"] if d["kind"] == "vendored"):
        pkg = alloy_dir.parent / dep["path"]
        pkg.mkdir(parents=True, exist_ok=True)
        (pkg / "__init__.py").write_text("")
        if licence:
            (pkg / "LICENSE").write_text("Apache License 2.0\n")
        lines.append(f"| `{dep['path']}` | {dep['pin'][:8] if pins_md else 'unknown'} |")
    (alloy_dir / "_vendor" / "__init__.py").write_text("")
    (alloy_dir / "_vendor" / "VENDORED.md").write_text("# Vendored\n" + "\n".join(lines) + "\n")


def test_check_finds_the_vendored_engine_and_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(pins, "alloy_root", lambda: tmp_path / "alloy")
    monkeypatch.setattr(pins, "installed", lambda dist: {"version": "1.8.0rc1", "commit": None})
    _vendor(tmp_path / "alloy")
    rows = {r["name"]: r for r in pins.check(pins.load_pins())[0]["deps"]}
    assert rows["newton"]["match"] and rows["mujoco-warp"]["match"]
    assert rows["newton"]["separate_install"] == "1.8.0rc1"  # a user's own newton: reported, not a mismatch
    md = tmp_path / "alloy/_vendor/VENDORED.md"
    pin = _dep("newton")["pin"]
    md.write_text(md.read_text().replace(pin[:8], "00000000") + f"an earlier patch: {pin}\n")
    assert not next(r for r in pins.check(pins.load_pins())[0]["deps"] if r["name"] == "newton")["pin_in_vendored_md"]
    (tmp_path / "alloy/_vendor/mujoco_warp/LICENSE").unlink()
    row = next(r for r in pins.check(pins.load_pins())[0]["deps"] if r["name"] == "mujoco-warp")
    assert not row["match"] and not row["licence"] and "LICENSE" in row["fix"]


def test_check_reports_a_missing_package_with_a_fix(monkeypatch, tmp_path):
    monkeypatch.setattr(pins, "installed", lambda dist: {"version": None, "commit": None})
    monkeypatch.setattr(pins, "alloy_root", lambda: tmp_path / "alloy")
    doc, ok = pins.check(pins.load_pins())
    assert not ok and all("fix" in row for row in doc["deps"])


def test_release_pyproject_is_valid_toml_with_package_data():
    reqs = pins.requirements(pins.load_pins())
    meta = wheels.tree_meta(REPO)
    doc = tomllib.loads(wheels.release_pyproject("0.0.1+g1", reqs, meta))
    assert doc["project"]["dependencies"] == reqs and "newton" not in str(doc["project"]["dependencies"])
    assert doc["project"]["optional-dependencies"] == meta.extras and "dev" not in meta.extras
    assert doc["project"].get("scripts", {}) == meta.scripts
    tree_project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    for key in wheels.PROJECT_FIELDS:  # summary, readme, licence: as the tree says (the export sets them there)
        assert doc["project"].get(key) == tree_project.get(key, wheels.PROJECT_DEFAULTS.get(key)), key
    data = doc["tool"]["setuptools"]["package-data"]
    assert data["alloy"][-1] == "**/*"
    tree_data = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]
    for pkg, globs in tree_data.items():  # F1 (11-08): the tree's globs (step.c, bsdf tables, ...) are carried over
        assert set(globs) <= set(data[pkg]), pkg


def _tiny_tree(root: Path, project: str = "") -> Path:
    """A minimal Alloy-shaped tree with a C source, a data table and a licence beside the code; ``project`` adds
    lines to its pyproject's [project]."""
    tree = root / "tree"
    for d in ("alloy/route/_mjc", "alloy/camera/tactile/data", "library/materials"):
        (tree / d).mkdir(parents=True)
    for f in ("alloy/__init__.py", "alloy/route/__init__.py", "alloy/route/_mjc/__init__.py",
              "alloy/camera/__init__.py", "alloy/camera/tactile/__init__.py"):
        (tree / f).write_text("")
    (tree / "alloy/route/_mjc/step.c").write_text("int x;\n")
    (tree / "alloy/route/_mjc/step.h").write_text("int x;\n")
    (tree / "alloy/route/_mjc/liballoy_mjc.so").write_bytes(b"\x7fELF host build")
    (tree / "alloy/route/_mjc/.gitignore").write_text("*.so\n")
    (tree / "alloy/camera/tactile/data/gel.npz").write_bytes(b"PK")
    (tree / "alloy/camera/LICENSING.md").write_text("notices\n")
    (tree / "library/materials/abs.yaml").write_text("name: abs\n")
    _vendor(tree / "alloy")
    (tree / "pyproject.toml").write_text(
        f'[project]\nname = "alloy"\nversion = "0.0.1"\n{project}\n[tool.setuptools.package-data]\n'
        '"alloy.route._mjc" = ["*.c", "*.h"]\n"alloy.camera.tactile" = ["data/*.npz"]\n')
    return tree


@pytest.mark.skipif(not shutil.which("uv"), reason="needs uv to build a wheel")
def test_built_wheel_takes_its_summary_and_readme_from_the_tree(tmp_path):
    """The release wheel once wrote its own summary and no readme: twine check --strict failed on the missing long
    description, and pip show disagreed with the sdist."""
    summary = "An engine \u2013 \"quoted\""
    tree = _tiny_tree(tmp_path, f'description = {json.dumps(summary, ensure_ascii=False)}\nreadme = "README.md"\n'
                                'license-files = ["LICENSE", "NOTICE"]\n')
    (tree / "README.md").write_text("# Alloy\n\nA long description.\n")
    for name in ("LICENSE", "NOTICE"):
        (tree / name).write_text(f"{name} text\n")
    args = wheels.argparse.Namespace(tree=tree, no_imports=True, assets=None, name="alloy-engine")
    wheel, _, _ = wheels.alloy_wheel(args, tmp_path / "work", tmp_path / "out")
    with wheels.zipfile.ZipFile(wheel) as zf:
        meta = zf.read(next(n for n in zf.namelist() if n.endswith(".dist-info/METADATA"))).decode()
    head, _, body = meta.partition("\n\n")
    assert f"Summary: {summary}" in head.splitlines() and "Description-Content-Type: text/markdown" in head
    assert body.startswith("# Alloy") and "A long description." in body
    assert {"License-File: LICENSE", "License-File: NOTICE"} <= set(head.splitlines())


def test_stage_project_files_refuses_a_missing_readme(tmp_path):
    with pytest.raises(wheels.BuildError, match="readme names 'README.md'"):
        wheels.stage_project_files(tmp_path, tmp_path / "stage", {"readme": {"file": "README.md"}})


@pytest.mark.skipif(not shutil.which("uv"), reason="needs uv to build a wheel")
def test_built_wheel_carries_every_package_data_glob_and_staged_file(tmp_path):
    """F1 (11-08): the release wheel dropped the tree's package data (step.c, bsdf tables, tactile data)."""
    tree = _tiny_tree(tmp_path)
    args = wheels.argparse.Namespace(tree=tree, no_imports=True, assets=None, name="alloy-engine")
    wheel, _, engine = wheels.alloy_wheel(args, tmp_path / "work", tmp_path / "out")
    names = set(wheels.zipfile.ZipFile(wheel).namelist())
    meta = wheels.tree_meta(tree)
    assert wheels.package_data_misses(names, meta.package_data) == []
    assert {"alloy/route/_mjc/step.c", "alloy/camera/tactile/data/gel.npz", "alloy/camera/LICENSING.md",
            "alloy/_data/library/materials/abs.yaml"} <= names
    assert "alloy/route/_mjc/liballoy_mjc.so" not in names  # a host build never ships
    assert "alloy/route/_mjc/.gitignore" not in names  # nor VCS dotfiles (staging drops them; the check then passes)
    vendored = {"alloy/_vendor/VENDORED.md", "alloy/_vendor/newton/LICENSE", "alloy/_vendor/mujoco_warp/__init__.py"}
    assert vendored <= names
    assert [e["name"] for e in engine] == ["newton", "mujoco-warp"]
    requires = wheels.zipfile.ZipFile(wheel).read(f"alloy_engine-{wheel.name.split('-')[1]}.dist-info/METADATA")
    assert b"Requires-Dist: mujoco==3.17.0" in requires and b"Requires-Dist: newton" not in requires


@pytest.mark.skipif(not shutil.which("uv"), reason="needs uv to build a wheel")
def test_a_plain_install_of_the_tree_carries_the_card_library(tmp_path):
    """Mac test (11-08): `pip install .` of a clone left library/ and 18 data files of alloy/ out, so the quick start
    knew no materials. The repo's own pyproject.toml, setup.py and MANIFEST.in build a tiny tree: the sdist carries
    library/, the wheel built from that sdist carries it at alloy/_data/library (where default_root() looks) and the
    package's data files, and no Python cache or host-built library rides along."""
    tree = tmp_path / "clone"
    files = {name: (REPO / name).read_text() for name in ("pyproject.toml", "setup.py", "MANIFEST.in")}
    files |= {"alloy/__init__.py": "", "alloy/camera/denoiser/weights.json": "{}", "alloy/route/liballoy_mjc.so": "",
              "alloy/__pycache__/__init__.cpython-312.pyc": "", "library/materials/abs.yaml": "name: abs\n",
              "library/materials/__pycache__/x.cpython-312.pyc": ""}
    for name, text in files.items():
        (tree / name).parent.mkdir(parents=True, exist_ok=True)
        (tree / name).write_text(text)
    release_check._run(["uv", "build", "-q", "--out-dir", str(tmp_path / "dist"), str(tree)])  # sdist, then its wheel
    with tarfile.open(next((tmp_path / "dist").glob("*.tar.gz"))) as tar:
        sdist = tar.getnames()
    wheel = wheels.zipfile.ZipFile(next((tmp_path / "dist").glob("*.whl"))).namelist()
    assert any(n.endswith("/library/materials/abs.yaml") for n in sdist)
    assert {"alloy/_data/library/materials/abs.yaml", "alloy/camera/denoiser/weights.json"} <= set(wheel)
    assert not [n for n in sdist + wheel if n.endswith((".pyc", ".so"))]


def test_release_check_installs_what_a_clone_has(tmp_path):
    """The source leg installs the tree's committed HEAD (a git archive), never uncommitted files or a non-clone."""
    tree = tmp_path / "tree"
    tree.mkdir()
    with pytest.raises(wheels.BuildError, match="not a git checkout"):
        release_check.git_archive(tree, tmp_path / "none")
    git = ["git", "-C", str(tree), "-c", "user.name=t", "-c", "user.email=t@localhost"]
    release_check._run([*git, "init", "-q"])
    (tree / "kept.txt").write_text("committed")
    release_check._run([*git, "add", "kept.txt"])
    release_check._run([*git, "commit", "-q", "-m", "one file"])
    (tree / "local.txt").write_text("not committed")
    out = release_check.git_archive(tree, tmp_path / "src")
    assert sorted(p.name for p in out.iterdir()) == ["kept.txt"]


def test_release_requirements_take_the_pins_then_the_trees_other_dependencies():
    meta = wheels.TreeMeta(dependencies=["warp-lang>=1.17", "mujoco==3.17.0", "absl-py", "etils[epath]", "numpy"])
    reqs = wheels.release_requirements(pins.load_pins(), meta)
    assert reqs == pins.requirements(pins.load_pins()) + ["absl-py", "etils[epath]"]
    stale = wheels.TreeMeta(dependencies=["newton[sim]==1.8.0.dev0", "mujoco_warp @ git+https://example/x"])
    with pytest.raises(wheels.BuildError, match=r"still requires .*newton\[sim\].*mujoco_warp"):
        wheels.release_requirements(pins.load_pins(), stale)


def test_a_tree_without_its_vendored_engine_is_refused(tmp_path):
    tree = _tiny_tree(tmp_path)
    (tree / "alloy/_vendor/newton/LICENSE").unlink()
    (tree / "alloy/_vendor/VENDORED.md").write_text("no pins here\n")
    args = wheels.argparse.Namespace(tree=tree, no_imports=True, assets=None, name="alloy-engine")
    with pytest.raises(wheels.BuildError, match=r"alloy/_vendor/newton \(lacks: licence, pin_in_vendored_md\)"):
        wheels.alloy_wheel(args, tmp_path / "work", tmp_path / "out")


def test_check_wheel_names_what_a_wheel_lacks(tmp_path):
    stage = tmp_path / "stage"
    (stage / "alloy/route/_mjc").mkdir(parents=True)
    (stage / "alloy/__init__.py").write_text("")
    (stage / "alloy/route/_mjc/step.c").write_text("int x;\n")
    wheel = tmp_path / "a-0-py3-none-any.whl"
    with wheels.zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("alloy/__init__.py", "")
    with pytest.raises(wheels.BuildError, match=r"1 staged files absent.*step\.c.*alloy\.route\._mjc: \*\.c"):
        wheels.check_wheel(wheel, stage, {"alloy.route._mjc": ["*.c"]})
    with wheels.zipfile.ZipFile(wheel, "a") as zf:
        zf.writestr("alloy/route/_mjc/step.c", "int x;\n")
    assert wheels.check_wheel(wheel, stage, {"alloy.route._mjc": ["*.c"]}) == {"staged_files": 2,
                                                                             "package_data_globs": 1}


def test_stage_copies_the_library_into_the_package(tmp_path):
    tree = tmp_path / "tree"
    (tree / "alloy").mkdir(parents=True)
    (tree / "library" / "materials").mkdir(parents=True)
    (tree / "alloy" / "__init__.py").write_text("")
    (tree / "library" / "materials" / "abs.yaml").write_text("name: abs\n")
    (tree / "pyproject.toml").write_text('version = "0.0.1"\n')
    res = stage.stage_tree(tree, tmp_path / "stage")
    assert res.version == "0.0.1" and res.imports == {}  # not a git checkout: no local label; no assets
    assert (tmp_path / "stage/alloy/_data/library/materials/abs.yaml").is_file()


def test_a_staged_package_finds_its_own_library(tmp_path, monkeypatch):
    """default_root() of a staged copy (no <repo>/library beside it) is the packaged alloy/_data/library, unless
    $ALLOY_LIBRARY names another folder."""
    stage_dir = tmp_path / "stage"
    (stage_dir / "alloy" / "cards").mkdir(parents=True)
    (stage_dir / "alloy" / "_data" / "library").mkdir(parents=True)
    src = (REPO / "alloy/cards/library.py").read_text()
    body = ("import os\nfrom pathlib import Path\n\n\nclass CardError(Exception):\n    pass\n\n\n"
            + src[src.index("LIBRARY_ENV = "):src.index("LIBRARY_FIX = ")])
    monkeypatch.delenv("ALLOY_LIBRARY", raising=False)
    path = stage_dir / "alloy" / "cards" / "library.py"
    path.write_text(body)
    ns: dict = {"__file__": str(path)}
    exec(compile(body, str(path), "exec"), ns)  # noqa: S102 (the function under test, from the repo source)
    assert ns["default_root"]() == (stage_dir / "alloy" / "_data" / "library").resolve()
    monkeypatch.setenv("ALLOY_LIBRARY", str(tmp_path))
    assert ns["default_root"]() == tmp_path.resolve()


def test_import_cards_ship_without_meshes_or_withheld_cards(tmp_path):
    src = tmp_path / "assets" / "imports" / "ycb_gso"
    (src / "objects").mkdir(parents=True)
    (src / "meshes" / "mug").mkdir(parents=True)
    (src / "objects" / "mug.yaml").write_text("# licence: CC-BY-4.0; redistribute: yes\nschema: alloy.object/1\n")
    (src / "objects" / "cad.yaml").write_text("# licence: maker; redistribute: no\nschema: alloy.object/1\n")
    (src / "objects" / "plain.yaml").write_text("schema: alloy.object/1\n")
    (src / "manifest.json").write_text("{}")
    (src / "meshes" / "mug" / "visual.glb").write_bytes(b"glTF")
    counts = stage.stage_imports(tmp_path / "assets", tmp_path / "stage")
    assert counts == {"ycb_gso": {"copied": 3, "skipped_no_redistribute": 1, "cards_unflagged": 1, "skipped_other": 1}}
    shipped = tmp_path / "stage/alloy/_data/imports/ycb_gso"
    assert (shipped / "objects/mug.yaml").is_file() and (shipped / "manifest.json").is_file()
    assert not (shipped / "objects/cad.yaml").exists() and not (shipped / "meshes").exists()


def test_imports_need_an_imports_directory(tmp_path):
    with pytest.raises(stage.StageError, match="no imports/ directory"):
        stage.stage_imports(tmp_path, tmp_path / "stage")


def test_closure_pins_and_verifies_a_wheelhouse(tmp_path):
    import closure

    assert closure.wheel_pin("PyYAML-6.0.3-cp312-cp312-manylinux_2_28_x86_64.whl") == "pyyaml==6.0.3"
    src = tmp_path / "src"
    src.mkdir()
    for name in ("absl_py-2.5.1-py3-none-any.whl", "newton-1.8.0.dev0+g4f3b41ef-py3-none-any.whl"):
        (src / name).write_bytes(name.encode())
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale-0.1-py3-none-any.whl").write_bytes(b"old")
    doc = closure.write_closure(sorted(src.glob("*.whl")), out)
    assert len(doc["wheels"]) == 2 and not (out / "stale-0.1-py3-none-any.whl").exists()
    assert (out / "requirements.txt").read_text().split() == ["absl-py==2.5.1", "newton==1.8.0.dev0+g4f3b41ef"]
    assert closure.verify(out)["ok"]
    (out / "absl_py-2.5.1-py3-none-any.whl").write_bytes(b"tampered")
    assert closure.verify(out)["sha_mismatch_or_missing"] == ["absl_py-2.5.1-py3-none-any.whl"]
    with pytest.raises(closure.ClosureError, match="not a wheel"):
        closure.wheel_pin("numpy-2.0.tar.gz")


def test_build_all_removes_its_temp_work_dir_but_keeps_an_explicit_one(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(wheels, "_build_in", lambda args, work: seen.append(work) or {"work": str(work)})
    args = wheels.argparse.Namespace(work=None, out=tmp_path / "out")
    wheels.build_all(args)
    assert seen[-1].name.startswith("alloy-wheels-") and not seen[-1].exists()
    args.work = tmp_path / "keep"
    args.work.mkdir()
    wheels.build_all(args)
    assert seen[-1] == args.work.resolve() and args.work.is_dir()


def test_release_check_user_engine_is_on_the_vendored_copies_release_line():
    """The coexistence venv installs a user's newton / mujoco-warp from PyPI at exact versions on the release line
    of Alloy's vendored copies (same major.minor): there the two copies share the most kernel code, so a Warp module
    or kernel left unrenamed shows. The copies are told apart by their directories, not their versions."""
    user = dict(r.split("==") for r in release_check.USER_ENGINE)
    assert set(user) == {"newton", "mujoco-warp"}
    for dep in (d for d in pins.load_pins()["deps"] if d["kind"] == "vendored"):
        assert user[dep["name"]].split(".")[:2] == dep["version"].split(".")[:2], dep["name"]


def test_coexist_probe_flags_warp_modules_a_user_copy_can_share():
    """A vendored kernel in a Warp module not named alloy._vendor.* / alloy_vendor__* (main before SCR-WP0-COEXIST:
    soft_face_contacts_False and 26 more beside PyPI newton 1.8.0rc1), or a kernel key whose copy changed, fails."""
    origins = {"alloy._vendor.newton._src.geometry.kernels": ["alloy._vendor.newton._src.geometry.kernels"],
               "alloy_vendor__narrow_phase_mesh_overlaps_0": ["alloy._vendor.newton._src.geometry.narrow_phase"],
               "soft_face_contacts_False": ["alloy._vendor.newton._src.geometry.soft_contacts_sdf"],
               "soft_edge_contacts_True": [f"{coexist_probe.ENGINES[0]}._src.geometry.soft_contacts_sdf"]}
    assert coexist_probe.unshared(origins) == ["soft_face_contacts_False"]
    before = {"soft_face_contacts_False": {"k": "alloy"}, "alloy_vendor__x_1a": {"k": "alloy"}}
    after = {"soft_face_contacts_False": {"k": "user"}, "alloy_vendor__x_1a": {"k": "alloy", "k2": "alloy"}}
    assert coexist_probe.taken_over(before, after) == ["soft_face_contacts_False"]



def test_the_vendor_rewrite_leaves_the_user_copy_alone():
    """``scripts/vendor_engines.py --alloy-only`` (every branch's pre-gate step) points bare engine imports and module
    strings at Alloy's copies; tools/deps reaches the user's own copy on purpose, so it must find nothing to change
    here (WP8, 2026-11-08: it had rewritten coexist_probe's ``import newton``)."""
    spec = importlib.util.spec_from_file_location("vendor_engines_for_deps", REPO / "scripts" / "vendor_engines.py")
    rewriter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rewriter)
    files = [p for p in sorted(DEPS.rglob("*.py")) if "__pycache__" not in p.parts]
    rels = [p.relative_to(REPO).as_posix() for p in files]
    hit = [rel for p, rel in zip(files, rels, strict=True)
           if rewriter.rewrite_text(p.read_text(), rel) != p.read_text()]
    assert hit == [], (f"{hit}: reach the user's copy by name (importlib.import_module(coexist_probe.ENGINES[0])), "
                       "not by a bare import or module-path string")

# SCR-WP0-COEXIST: until alloy/_vendor/_warp.py renames the vendored engines' Warp modules, they share Warp modules and
# kernels with a user's own copy, so the one-process test below is an expected failure (strict: it must start passing
# with the renaming).
WP0_RENAMES = (REPO / "alloy" / "_vendor" / "_warp.py").is_file()
USER_ENGINES = all(importlib.util.find_spec(name) for name in coexist_probe.ENGINES)


@pytest.mark.skipif(not USER_ENGINES, reason="needs a top-level newton and mujoco_warp (the user's own copy): "
                    "pip install newton mujoco-warp")
@pytest.mark.xfail(not WP0_RENAMES, raises=AssertionError, strict=True,
                   reason="SCR-WP0-COEXIST: the vendored engines share Warp modules with a user's copy until "
                          "alloy/_vendor/_warp.py renames them")
@pytest.mark.timeout(900)
@pytest.mark.parametrize("first", ["alloy", "user"])
def test_alloy_and_a_users_own_engines_step_in_one_process(tmp_path, first):
    """This checkout's Alloy beside the venv's top-level newton / mujoco_warp (a user's pip install), ``first``
    imported first: both copies step on the CPU and share no Warp module or kernel (``coexist_probe.py``, which
    ``release_check.py`` runs on the built wheel)."""
    out = tmp_path / "row.json"
    proc = subprocess.run([sys.executable, str(DEPS / "coexist_probe.py"), "--first", first, "--out", str(out)],
                          cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(REPO)}, capture_output=True, text=True,
                          timeout=840, check=False)
    if not out.is_file():
        pytest.fail(f"coexist_probe.py exited {proc.returncode} without a row:\n{proc.stderr[-2000:]}")
    row = json.loads(out.read_text())
    assert row["ok"], {k: row.get(k) for k in ("alloy", "user", *release_check.PROBE_FIELDS, "fix")}

def test_wheel_smoke_finds_real_engine_imports_only(tmp_path):
    pkg = tmp_path / "alloy"
    pkg.mkdir()
    (pkg / "a.py").write_text('"""Built on Newton:\n\n    import newton\n"""\nfrom . import newton\n'
                              "from alloy._vendor.newton import x\nimport newtonian\n")
    (pkg / "b.py").write_text("def f():\n    from mujoco_warp._src import io\n    import newton.solvers as s\n")
    (pkg / "c.py").write_text("this is not python (\n")
    assert wheel_smoke.engine_imports(pkg) == ["alloy/b.py:2", "alloy/b.py:3"]


def test_release_check_runs_with_no_checkout_on_the_path(tmp_path):
    env = release_check.clean_env({"PYTHONPATH": str(REPO), "ALLOY_CPU_ROUTE": "warp", "PATH": "/bin"}, tmp_path)
    assert env == {"PATH": "/bin", "ALLOY_CACHE_DIR": str(tmp_path)}


def test_wheel_smoke_lists_files_the_tree_has():
    """The installed check's run-time files exist in the source package (so a miss means the wheel dropped them)."""
    for pkg, names in wheel_smoke.PACKAGE_FILES.items():
        for name in names:
            assert (REPO / pkg.replace(".", "/") / name).is_file(), f"{pkg}/{name}"


def test_the_usd_check_names_the_trees_usd_extra_and_stops_at_what_is_missing(monkeypatch):
    """The check's fix names an extra pyproject has, that extra installs the packages the check looks for, and a
    missing package stops the check at its first step with that fix (once, no extra installed them)."""
    extras = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert "alloy-engine[usd]" in usd_extra_check.FIX and "usd" in extras
    installs = {re.split(r"[<>=!~; ]", req, maxsplit=1)[0] for req in extras["usd"]}
    assert installs == {dist for dists in usd_extra_check.PACKAGES.values() for dist in dists}

    def no_pxr(work):
        raise ModuleNotFoundError("No module named 'pxr'")

    monkeypatch.setattr(usd_extra_check, "STEPS", (("packages", no_pxr), *usd_extra_check.STEPS[1:]))
    doc = usd_extra_check.run()
    assert not doc["ok"] and doc["failed"] == "packages" and "import_file" not in doc
    assert "pxr" in doc["error"] and usd_extra_check.FIX in doc["fix"]
