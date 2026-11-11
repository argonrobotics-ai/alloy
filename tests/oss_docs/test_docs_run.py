# SPDX-License-Identifier: Apache-2.0
"""Run every ```python block of the agent docs and every example script end to end (slow: kernel compiles).

Each runs in a fresh interpreter in an empty directory. Marked ``gpu`` and ``slow``: on a CPU the 40-odd scripts take
an hour or more (cold kernel compiles, cloth), so they run on GPU runners only and skip cleanly elsewhere.
Run: ``python -m pytest tests/oss_docs -m slow -q`` on a machine with an NVIDIA GPU.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from alloy.testing.oss_docs import (REPO, agent_docs, missing_library_meshes, missing_modules,
                                    missing_optional_package, missing_requirements, python_blocks, text_requirements)

BLOCKS = [b for d in agent_docs() for b in python_blocks(d)]
EXAMPLES = sorted(p for p in (REPO / "examples").rglob("*.py") if not p.name.startswith("_"))
TIMEOUT_S = 1500  # a cold CPU kernel compile of a cloth or liquid scene takes minutes
# Examples slower than that on one L4 even at their short default, with their budget in seconds. ClothFold: about
# 10 min cold compile plus about 10 min for 64 steps (30 g fingers on cloth: 512 force exchanges per step).
SLOW_S = {"examples/envs/09_cloth_fold.py": 2700}
# The viewer examples draw until the viewer closes; with these arguments (their own docstrings' line for a run with
# no window) they step 0.2 s and end, with no optional package.
VIEWER_ARGS = ["--backend", "null", "--seconds", "0.2"]


def _run(args: list[str], cwd, timeout: int = TIMEOUT_S) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(REPO), os.environ.get("PYTHONPATH", "")]))
    return subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout, check=False)


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.timeout(TIMEOUT_S + 60)
@pytest.mark.parametrize("block", BLOCKS, ids=lambda b: b.label)
def test_doc_block_runs(block, tmp_path) -> None:
    missing = text_requirements(block.code) + missing_modules(block.code)
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")
    script = tmp_path / "snippet.py"
    script.write_text(block.code, encoding="utf-8")
    res = _run([str(script)], tmp_path)
    _skip_unless_installed(res)
    assert res.returncode == 0, f"{block.label} failed (exit {res.returncode}):\n{res.stderr[-3000:]}"


def _skip_unless_installed(res: subprocess.CompletedProcess) -> None:
    """A run that stopped only for something this machine lacks skips, and the skip line names it: the mesh files of
    a library object (`alloy fetch <name>` before the run makes it a real run), or an optional package (torch, viser,
    an RL library: install it and the block runs). Any other failure stays a failure."""
    if res.returncode == 0:
        return
    reason = missing_library_meshes(res.stderr) or missing_optional_package(res.stderr)
    if reason:
        pytest.skip(f"needs {reason}")


@pytest.mark.slow
@pytest.mark.gpu
@pytest.mark.timeout(max([TIMEOUT_S, *SLOW_S.values()]) + 60)
@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: str(p.relative_to(REPO)))
def test_example_runs(example, tmp_path) -> None:
    missing = missing_requirements(example)
    if missing:
        pytest.skip(f"needs {', '.join(missing)}")
    rel = str(example.relative_to(REPO))
    args = VIEWER_ARGS if rel.startswith("examples/viewer/") else []   # the others: default arguments
    res = _run([str(example), *args], tmp_path, SLOW_S.get(rel, TIMEOUT_S))   # outputs in the empty cwd
    _skip_unless_installed(res)
    assert res.returncode == 0, f"{example.relative_to(REPO)} failed (exit {res.returncode}):\n{res.stderr[-3000:]}"
