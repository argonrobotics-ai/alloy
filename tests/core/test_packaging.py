# SPDX-License-Identifier: Apache-2.0
"""Every directory of python modules under alloy/ is a regular package, so a wheel built with setuptools'
``packages.find`` (pyproject) ships it: find_packages skips a directory without ``__init__.py`` and everything below
it (SCR-OSS-10 item 3; alloy/scene/emitters was missing one)."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# Run by Blender's own python (``blender -P alloy/render/blender/build.py``), never imported as alloy.render.blender;
# how a release ships them (package data or not at all) is OSS-2's call.
# alloy/_vendor/*: Newton and MuJoCo Warp keep their upstream layout (namespace subpackages such as newton/_src);
# packages.find ships them (namespaces on by default) and the release wheel smoke imports them (OSS-2).
NOT_PACKAGES = ("alloy/render/blender", "alloy/_vendor")


def test_every_module_directory_is_a_package() -> None:
    tracked = subprocess.run(["git", "ls-files", "--", "alloy/*.py"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout.split()
    dirs = {Path(f).parent for f in tracked}
    needed = {a for d in dirs for a in (d, *d.parents) if a.parts[:1] == ("alloy",)}
    needed = {d for d in needed if not any(str(d) == n or str(d).startswith(n + "/") for n in NOT_PACKAGES)}
    missing = sorted(str(d / "__init__.py") for d in needed if not (ROOT / d / "__init__.py").exists())
    assert not missing, f"add these (setuptools find_packages skips the directory otherwise): {missing}"


def test_importer_extra() -> None:
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert set(extras["import"]) == {"trimesh", "manifold3d", "coacd", "pillow", "huggingface_hub",
                                     "scikit-image==0.25.2"}  # SCR-OSS19-4


def test_usd_extra_carries_pxr_and_the_bundled_newtons_schema_pin() -> None:
    """Reading USD needs pxr and newton_usd_schemas, which no other extra installs (a wheel install could not load
    the .usda its own export_scene wrote). The pins are the bundled Newton 1.8's own: usd-core off aarch64,
    usd-exchange on it, newton-usd-schemas 0.6.x (pre-1.0: 0.5 -> 0.6 broke the reader, so the next minor is not
    taken unseen)."""
    usd = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]["usd"]
    names = {r.split(";")[0].split(">")[0].strip(): r for r in usd}
    assert set(names) == {"usd-core", "usd-exchange", "newton-usd-schemas"}
    assert "platform_machine != 'aarch64'" in names["usd-core"]
    assert "platform_machine == 'aarch64'" in names["usd-exchange"]
    assert names["newton-usd-schemas"] == "newton-usd-schemas>=0.6.0,<0.7"


def test_il_and_viewer_extras_name_their_lazy_imports() -> None:
    """alloy.il writes HDF5 through h5py and alloy.viewer opens viser or a pyglet / imgui window, all imported lazily;
    the extras are what their error messages tell the user to install."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["il"] == ["h5py"]
    assert extras["viewer"] == ["viser>=1.1.1", "pyglet>=2.1.6,<3", "imgui_bundle>=1.92.0"]


def test_envs_extra_carries_gymnasium_and_lerobot_has_its_own() -> None:
    """TRAIN-AUDIT (2026-11-11): the envs extra brings Gymnasium (its vector API from 1.0); LeRobot export is an extra
    of its own (Python 3.12, tested 0.7.0), so the light `il` extra keeps h5py alone."""
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["envs"] == ["torch", "huggingface_hub", "gymnasium>=1.0,<2"]
    assert extras["lerobot"] == ["lerobot[dataset]>=0.7.0,<0.8; python_version >= '3.12'"]
    assert extras["il"] == ["h5py"]


def _constants(path: Path) -> dict[str, tuple[str, ...]]:
    """The module-level tuple constants DOCS_PUBLIC and DATA_FILES of a source file, read without importing it."""
    tree = ast.parse(path.read_text())
    return {t.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
            for t in node.targets if isinstance(t, ast.Name) and t.id in ("DOCS_PUBLIC", "DATA_FILES")}


def test_sdist_manifest_ships_only_the_public_docs_the_examples_and_quick_start() -> None:
    """The sdist grafts docs/ by the allow-list only (2026-11-11 leak defence) and no install.sh (pip only)."""
    manifest = {line for line in (ROOT / "MANIFEST.in").read_text().splitlines() if not line.startswith("#")}
    docs = _constants(ROOT / "setup.py")["DOCS_PUBLIC"]
    want = {f"graft docs/{n}" if (ROOT / "docs" / n).is_dir() else f"include docs/{n}" for n in docs}
    assert {"graft library", "graft examples", "include AGENTS.md llms.txt"} | want <= manifest
    assert "graft docs" not in manifest and not any("install.sh" in line for line in manifest)
    assert {line for line in manifest if line.split()[1:2] and line.split()[1].startswith("docs")} == want


def test_barcode_extra_matches_an_installed_decoder() -> None:
    """SCR-SC9-1: the barcode extra pins zxing-cpp exactly; where the decoder is installed (the gate venvs), it is
    that version."""
    from importlib.metadata import PackageNotFoundError, version

    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["barcode"] == ["zxing-cpp==3.1.1"]
    try:
        installed = version("zxing-cpp")
    except PackageNotFoundError:
        return
    assert installed == "3.1.1"


def test_engines_are_vendored_not_dependencies() -> None:
    """Newton and MuJoCo Warp ship inside Alloy (alloy/_vendor, coordinator ruling 2026-11-07): pyproject lists
    neither; Warp and MuJoCo stay exact PyPI pins, and the installed ones are those versions."""
    from importlib.metadata import version

    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    names = {d.split("==", 1)[0].split("[", 1)[0].split(" ", 1)[0] for d in deps}
    assert not names & {"newton", "mujoco-warp", "mujoco_warp"}, names
    pins = dict(d.split("==", 1) for d in deps if "==" in d)
    assert version("warp-lang") == pins["warp-lang"]
    assert version("mujoco") == pins["mujoco"]


def test_console_scripts_resolve() -> None:
    """SCR-OSS15-2: every ``[project.scripts]`` entry point imports and names a callable."""
    import importlib

    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["alloy"] == "alloy.cli:main"
    for name, target in scripts.items():
        module, _, attr = target.partition(":")
        assert callable(getattr(importlib.import_module(module), attr)), name


def test_alloy_docs_finds_the_checkouts_folders_here_and_the_packaged_copy_in_an_install(tmp_path, capsys) -> None:
    """TRAIN-AUDIT fix 1 (2026-11-11): python -m alloy.docs prints where docs/, examples/ and AGENTS.md are, and
    ``agents`` prints AGENTS.md (user 17:20Z); a wheel carries them at alloy/_data/ (setup.py and tools/deps/stage.py
    stage them beside the card library)."""
    import alloy.docs as docs

    assert docs.docs_root() == ROOT / "docs" and docs.examples_root() == ROOT / "examples"
    assert docs.agents_md() == ROOT / "AGENTS.md"
    assert docs.main([]) == 0 and capsys.readouterr().out.split() == [
        "docs", str(ROOT / "docs"), "examples", str(ROOT / "examples"), "agents", str(ROOT / "AGENTS.md")]
    assert docs.main(["agents"]) == 0 and capsys.readouterr().out == (ROOT / "AGENTS.md").read_text()
    assert docs.main(["nope"]) == 2 and "usage" in capsys.readouterr().err
    site = tmp_path / "site-packages"
    (site / "alloy" / "_data" / "docs" / "guide").mkdir(parents=True)
    (site / "alloy" / "_data" / "AGENTS.md").write_text("# AGENTS\n")
    (site / "docs").mkdir()                     # another package's stray top-level docs/ is never taken
    assert docs._root("docs", site / "alloy") == site / "alloy" / "_data" / "docs"
    assert docs._find("AGENTS.md", site / "alloy") == site / "alloy" / "_data" / "AGENTS.md"
    with pytest.raises(FileNotFoundError, match="pip install"):
        docs._root("examples", site / "alloy")


def test_every_build_and_check_uses_the_same_docs_allow_list() -> None:
    """setup.py (pip install ., the sdist), tools/deps/stage.py (the release wheel) and the release content check
    name the same public docs and the same alloy/_data files: AGENTS.md and the allow-listed docs, never docs/ whole."""
    lists = {p: _constants(ROOT / p) for p in ("setup.py", "tools/deps/stage.py", "tools/release/stage_release.py")}
    assert all(v == lists["setup.py"] for v in lists.values()), lists
    assert lists["setup.py"] == {"DOCS_PUBLIC": ("public", "known_gaps", "guide"),
                                 "DATA_FILES": ("AGENTS.md", "llms.txt")}
    assert not {"notes", "oss", "research", "ENV.md", "SPEC.md"} & set(lists["setup.py"]["DOCS_PUBLIC"])
    assert all((ROOT / "docs" / n).exists() for n in lists["setup.py"]["DOCS_PUBLIC"])


def _stage_public_docs():
    """setup.py's stage_public_docs with its constants, without running setup()."""
    tree = ast.parse((ROOT / "setup.py").read_text())
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign))
            and not (isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("setuptools"))]
    keep += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "stage_public_docs"]
    ns: dict = {"__file__": str(ROOT / "setup.py")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "setup.py", "exec"), ns)  # noqa: S102 - our own file
    return ns["stage_public_docs"]


def test_pip_install_stages_only_the_public_docs_and_agents_md(tmp_path) -> None:
    root, data = tmp_path / "src", tmp_path / "build" / "alloy" / "_data"
    for rel in ("docs/public/index.md", "docs/public/img/a.png", "docs/known_gaps/b.md", "docs/guide/g.md",
                "docs/RENDER.md", "docs/notes/n.md", "docs/oss/o.md", "docs/research/r.md", "docs/SPEC.md", "docs/ENV.md",
                "docs/plan.md", "examples/x/y.py", "AGENTS.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("x\n")
    _stage_public_docs()(root, data)
    shipped = sorted(p.relative_to(data).as_posix() for p in data.rglob("*") if p.is_file())
    assert shipped == ["AGENTS.md", "docs/guide/g.md", "docs/known_gaps/b.md", "docs/public/img/a.png",
                       "docs/public/index.md", "examples/x/y.py"]   # RENDER.md, benchmarks/ stay in the repo
    shutil.rmtree(root / "docs" / "known_gaps")
    _stage_public_docs()(root, data)            # a rebuild replaces the staged copy


def test_version_reads_the_installed_dist_info_and_never_raises(tmp_path) -> None:
    """GEO P3: 0.1.0.post1 reported 0.1.0. ``alloy.__version__`` is the one ``alloy_engine-<v>.dist-info`` beside the
    package, else the source line (a checkout, an editable install, none or two found, an unreadable folder)."""
    import alloy

    site = tmp_path / "site"
    (site / "alloy").mkdir(parents=True)
    assert alloy._installed_version("9.9", str(site / "alloy")) == "9.9"                 # nothing installed here
    (site / "alloy_engine-0.1.0.post1.dist-info").mkdir()
    (site / "numpy-2.0.0.dist-info").mkdir()
    assert alloy._installed_version("9.9", str(site / "alloy")) == "0.1.0.post1"
    (site / "alloy_engine-0.1.1.dist-info").mkdir()
    assert alloy._installed_version("9.9", str(site / "alloy")) == "9.9"                 # two: do not guess
    assert alloy._installed_version("9.9", str(tmp_path / "gone" / "alloy")) == "9.9"   # never raises


def test_a_checkout_reports_its_pyproject_version_and_an_install_its_dist_info(tmp_path) -> None:
    import alloy

    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    assert alloy.__version__ == version
    site = tmp_path / "site"
    site.mkdir()
    (site / "alloy").symlink_to(ROOT / "alloy", target_is_directory=True)               # an installed layout
    (site / "alloy_engine-0.1.0.post1.dist-info").mkdir()
    out = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(site)!r}); import alloy; "
                          "print(alloy.__version__, alloy.__file__)"], cwd=tmp_path, capture_output=True, text=True,
                         check=True).stdout.split()
    assert out == ["0.1.0.post1", str(site / "alloy" / "__init__.py")]


def test_python_dash_m_alloy_prints_the_command_help(tmp_path) -> None:
    """GEO P3: ``python -m alloy`` failed with "No module named alloy.__main__"."""
    p = subprocess.run([sys.executable, "-m", "alloy"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert p.returncode == 0 and p.stdout.startswith("usage: alloy") and "COMMAND" in p.stdout
    v = subprocess.run([sys.executable, "-m", "alloy", "--version"], cwd=ROOT, capture_output=True, text=True,
                       check=False)
    assert v.returncode == 0 and tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"] in v.stdout


def test_fetching_textures_without_huggingface_hub_names_the_import_extra(monkeypatch, tmp_path, capsys) -> None:
    """WEB-AUDIT (2): on a base install ``python -m alloy.looks --fetch-textures`` ended in a traceback whose fix
    led with ``pip install huggingface_hub``. It now prints one error and leads with the import extra (exit 2)."""
    from alloy.looks.__main__ import main

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)        # as on a base install: the import fails
    monkeypatch.setenv("ALLOY_CACHE", str(tmp_path))
    assert main(["--fetch-textures", "--family", "wood"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: downloading from the Hugging Face Hub (library objects, texture sets) needs")
    assert 'fix: pip install "alloy-engine[import]"' in err and "Traceback" not in err
