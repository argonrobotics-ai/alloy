# SPDX-License-Identifier: Apache-2.0
"""Fast checks of the agent docs (AGENTS.md, llms.txt, docs/guide, examples/README.md); no engine import.

Every link and repo path resolves, every ```python block compiles, every API name and method the docs use exists
in ``alloy.api``, llms.txt keeps its format, and the docs carry no internal names.
"""

from __future__ import annotations

import re

import pytest

from alloy.testing.oss_docs import (REPO, agent_docs, called_names, class_methods, local_links, public_names,
                                    python_blocks, resolve, ticked_paths)

DOCS = agent_docs()
BLOCKS = [b for d in DOCS for b in python_blocks(d)]
API = REPO / "alloy" / "api"
# Generic internal markers that must not reach public agent docs: home paths, work-package and change-request codes,
# cloud commands. Names (people, projects, repositories, sites) are the release names scanner's job (tools/release),
# and the release scrub rewrites them, so they are never spelled out here.
INTERNAL = re.compile(r"/home/|\bWP\d+\b|\bSCR-|\bXB4|\bOSS-\d|\bDC\d+\b|gcloud", re.I)


def _id(path) -> str:
    return str(path.relative_to(REPO))


def test_docs_exist() -> None:
    names = {_id(d) for d in DOCS}
    assert {"AGENTS.md", "llms.txt", "docs/guide/README.md"} <= names, f"missing agent docs; found {sorted(names)}"


@pytest.mark.parametrize("doc", DOCS, ids=_id)
def test_links_resolve(doc) -> None:
    bad = [t for t in local_links(doc) if not resolve(doc, t).exists()]
    bad += [p for p in ticked_paths(doc) if not (REPO / p).exists()]
    assert not bad, f"{_id(doc)}: these links or paths do not exist (fix or drop them): {bad}"


@pytest.mark.parametrize("block", BLOCKS, ids=lambda b: b.label)
def test_python_blocks_compile(block) -> None:
    compile(block.code, block.label, "exec")


@pytest.mark.parametrize("block", BLOCKS, ids=lambda b: b.label)
def test_api_names_exist(block) -> None:
    if "import alloy" not in block.code:
        pytest.skip("block does not use alloy")
    public = public_names() | {"api", "__version__", "cards", "scene", "integration", "camera", "check", "io",
                               "plugins", "compose", "envs", "cli", "mcp", "create", "audit", "viewer", "il", "sysid"}  # subpackages the docs import
    unknown = {n for n in called_names(block.code, "alloy") if n not in public}
    scene_api = class_methods(API / "scene.py", "Scene")
    sim_api = class_methods(API / "sim.py", "Sim") | {"engine", "built", "device", "profile", "scene"}
    unknown |= {f"scene.{n}" for n in called_names(block.code, "scene") if n not in scene_api}
    unknown |= {f"sim.{n}" for n in called_names(block.code, "sim") if n not in sim_api}
    assert not unknown, f"{block.label} uses names alloy.api does not have: {sorted(unknown)} (renamed?)"


@pytest.mark.parametrize("doc", DOCS, ids=_id)
def test_no_internal_names(doc) -> None:
    hits = sorted({m.group(0) for m in INTERNAL.finditer(doc.read_text(encoding="utf-8"))})
    assert not hits, f"{_id(doc)} contains internal names {hits}: agent docs ship publicly; reword them"


def test_llms_txt_format() -> None:
    lines = (REPO / "llms.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# "), "llms.txt must start with '# <project name>'"
    assert any(line.startswith("> ") for line in lines[:4]), "llms.txt needs a '> summary' line under the title"
    sections = [line for line in lines if line.startswith("## ")]
    assert sections, "llms.txt needs '## <section>' lists of links"
    items = [line for line in lines if line.startswith("- ")]
    assert items and all(re.match(r"- \[[^\]]+\]\([^)]+\)(: .+)?$", it) for it in items), \
        "llms.txt list items must be '- [name](link): notes'"


def test_hub_scripts_need_huggingface_hub_only(monkeypatch) -> None:
    """A block that reads the object library on the Hub needs huggingface_hub and no login (the library is public);
    other blocks need neither."""
    import importlib.util

    import alloy.testing.oss_docs as od

    hub = "from alloy.assets.hub.fetch import fetch\nfetch(['ycb_025_mug'])\n"
    monkeypatch.setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert od.text_requirements("import alloy\n") == []
    assert od.text_requirements(hub) in ([], ["module huggingface_hub"])
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda m, *a: None if m == "huggingface_hub" else real(m, *a))
    assert od.text_requirements(hub) == ["module huggingface_hub"]
    assert od.text_requirements("import alloy\n") == []


def test_tools_that_need_arguments_and_missing_meshes_skip() -> None:
    """An example that cannot run without arguments (soft_speed.py: --scenes) skips by name; a run that stopped only
    because the machine lacks a library object's meshes skips with the fetch; other failures still fail."""
    import ast

    import alloy.testing.oss_docs as od

    assert od.required_arguments(ast.parse('ap.add_argument("--scenes", nargs="+", required=True)')) == ["--scenes"]
    assert od.required_arguments(ast.parse('ap.add_argument("--out", default="x")\nap.add_argument("p", nargs="?")'
                                           '\nap.add_argument("q", default=1)')) == []
    assert od.required_arguments(ast.parse('ap.add_argument("card")')) == ["card"]
    assert od.required_imports(ast.parse("def main():\n    import rerun as rr\n")) == {"rerun"}
    assert od.required_imports(ast.parse("try:\n    import torch\nexcept ImportError:\n    torch = None\n")) == set()
    speed = REPO / "examples/compose/soft_speed.py"
    assert any(m.startswith("arguments --scenes") for m in od.missing_requirements(speed))
    err = ("alloy.api.errors.SceneError: 'ycb_077_rubiks_cube' (imported library ycb_gso) has no mesh files on this "
           "machine\n  fix: run `alloy fetch ycb_077_rubiks_cube`")
    assert od.missing_library_meshes(err) == "library meshes of ycb_077_rubiks_cube (alloy fetch ycb_077_rubiks_cube)"
    assert od.missing_library_meshes("alloy.api.errors.SceneError: 'x' is not a material") is None


def test_missing_optional_package_skips_only_what_is_absent() -> None:
    """A run that stopped for a package this machine lacks is a skip with the package's name; a run that stopped for
    anything else (a snippet's own bug, a wrong alloy import, a package that is installed) stays a failure."""
    import alloy.testing.oss_docs as od

    absent = "module zz_absent (optional package not installed here)"
    assert od.missing_optional_package("Traceback ...\nModuleNotFoundError: No module named 'zz_absent'") == absent
    assert od.missing_optional_package("ModuleNotFoundError: No module named 'zz_absent.sub'") == absent
    fix = ("alloy.envs.errors.EnvError: the wrapper needs zz-absent and numpy, and one of them is not installed\n"
           "  fix: pip install numpy zz-absent, or pass backend='numpy'")
    assert od.missing_optional_package(fix) == absent
    quoted = ('alloy.api.errors.FeatureUnavailableError: the viewer backend cannot start: zz_absent is not installed\n'
              '  fix: pip install "zz_absent>=1.1.1", or pick an installed backend')
    assert od.missing_optional_package(quoted) == absent
    assert od.missing_optional_package("examples/x.py needs zz, which is not installed: pip install zz-absent") == absent
    # not skips
    assert od.missing_optional_package("NameError: name 'alloy' is not defined") is None
    assert od.missing_optional_package("ModuleNotFoundError: No module named 'alloy.nope'") is None
    assert od.missing_optional_package("EnvError: numpy is not installed\n  fix: pip install numpy") is None
    assert od.missing_optional_package("SceneError: 'x' is not a material\n  fix: pip install nothing") is None
    early = "ModuleNotFoundError: No module named 'zz_absent'\n" + "line\n" * 20 + "ValueError: something else"
    assert od.missing_optional_package(early) is None          # only the end of the output counts
    assert od.missing_modules("import zz_absent\nimport numpy\nimport alloy\n") == ["module zz_absent"]
    assert od.missing_modules("try:\n    import zz_absent\nexcept ImportError:\n    pass\n") == []
