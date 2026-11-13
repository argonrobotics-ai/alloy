# SPDX-License-Identifier: Apache-2.0
"""The public docs set (``docs/public``): short pages whose code runs as written and prints what the page says.

Each page's ``python`` blocks, joined in order, are one program; its ``text`` blocks, joined, are that program's
expected output. The program runs in an empty folder. ``ALLOY_DOCS_PYTHON`` names the interpreter of an environment
with the release wheel installed; the run then has no source tree on its path. Without it the test uses this tree.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from alloy.testing.oss_docs import REPO

PUBLIC = REPO / "docs" / "public"
PAGES = sorted(PUBLIC.rglob("*.md"))
MAX_LINES = 120                       # a page is one idea: longer means it should be two pages or lose text
TIMEOUT_S = 1500                      # a cold kernel compile takes minutes
NOT_PIP = re.compile(r"install\.sh|git clone|\bclone\b|pip install -e|conda |docker |github", re.I)
REPO_ONLY = re.compile(r"(?<![\w/])(?:bench|tests|tools|scripts)/[\w./-]+")   # paths a pip install does not have
LICENCE_URL = "https://github.com/argonrobotics-ai/alloy/blob/main/LICENSE"   # the one GitHub link (Home)
MCP_URL = "https://github.com/argonrobotics-ai/alloy/tree/main/alloy/mcp"        # and the MCP server's code (agents page)
COLAB_URL = ("https://colab.research.google.com/assets/colab-badge.svg)]"       # the quick start's notebook badge
             "(https://colab.research.google.com/github/argonrobotics-ai/alloy/blob/main/notebooks/quickstart.ipynb")
LONG_PAGES = {"keep-in-mind.md"}            # one line per known-limit page: as long as that list
_BLOCK = re.compile(r"^```(python|text)\n(.*?)^```", re.S | re.M)
_NUMBER = re.compile(r"-?\d+\.\d+(?:e-?\d+)?|-?\d+\.(?!\d)")
ABS_TOL = 0.006                       # a printed, rounded number may differ in its last digit between devices


def _id(page: Path) -> str:
    return str(page.relative_to(PUBLIC))


def page_program(page: Path) -> tuple[str, str]:
    """The page's program and its expected output. Example: ``code, out = page_program(PUBLIC / "index.md")``."""
    parts: dict[str, list[str]] = {"python": [], "text": []}
    for kind, body in _BLOCK.findall(page.read_text(encoding="utf-8")):
        parts[kind].append(body)
    return "\n".join(parts["python"]), "".join(parts["text"])


def same_output(got: str, want: str) -> bool:
    """True when the two texts match word for word, numbers with a decimal point within ``ABS_TOL``."""
    got_lines, want_lines = got.strip().splitlines(), want.strip().splitlines()
    if len(got_lines) != len(want_lines):
        return False
    return all(_same_line(g.strip(), w.strip()) for g, w in zip(got_lines, want_lines))


def _tokens(line: str) -> str:
    """A line with array brackets set apart and a negative zero (``-0.``) read as zero."""
    line = re.sub(r"-(0\.0*)(?![\d])", r"\1", line)
    return line.replace("[", " [ ").replace("]", " ] ")


def _same_line(got: str, want: str) -> bool:
    got, want = _tokens(got), _tokens(want)
    if _NUMBER.sub("#", got).split() != _NUMBER.sub("#", want).split():
        return False
    pairs = zip(_NUMBER.findall(got), _NUMBER.findall(want))
    return all(abs(float(g) - float(w)) <= ABS_TOL for g, w in pairs)


def _run(code: str, cwd: Path) -> subprocess.CompletedProcess:
    python = os.environ.get("ALLOY_DOCS_PYTHON")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if python is None:
        python, env["PYTHONPATH"] = sys.executable, str(REPO)
    script = cwd / "page.py"
    script.write_text(code, encoding="utf-8")
    return subprocess.run([python, script.name], cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT_S)


NAV = {"index.md", "quickstart.md", "what-ships.md", "agents.md", "errors.md", "keep-in-mind.md",
       "guides/robots.md", "guides/rl.md", "guides/imitation.md", "guides/camera.md", "guides/liquids.md",
       "guides/cloth.md", "guides/cables.md", "guides/health.md"}
COMPARE = {"compare/index.md", "compare/mujoco.md", "compare/newton.md", "compare/genesis.md",
           "compare/mujoco-playground.md", "compare/maniskill.md"}
HOW_TO = {f"how-to/{name}.md" for name in (
    "mac-without-gpu", "camera-many-worlds", "coding-agents", "cable-or-wire", "quadruped-rl", "cloth-folding",
    "pour-liquid-robot-arm", "make-a-dataset", "evaluate-checkpoint")}
HOW_TO |= {f"how-to/{name}.md" for name in (      # round 2
    "gymnasium-vector-api", "imitation-data-with-liquids", "lerobot-dataset", "mjcf-robot-with-cloth-or-liquid",
    "randomise-parallel-worlds", "squeeze-soft-object")}
QUESTIONS = {f"answers/{name}.md" for name in (
    "built-on-mujoco-warp-or-newton", "deformable-object-manipulation", "did-my-simulation-blow-up",
    "error-messages", "gpu-parallel-engines-for-rl", "mcp-server-physics-engine",
    "mujoco-models-with-liquids-and-sand", "new-robotics-simulators-2026", "open-source-multi-physics-engines",
    "pip-installable-simulators", "robots-cloth-liquids-one-scene", "sand-and-granular-materials",
    "soft-body-manipulation-python")}
NOT_RUN = {"how-to/evaluate-checkpoint.md": "needs rsl_rl and a trained checkpoint; run by hand on the 0.1.2 wheel",
           "how-to/lerobot-dataset.md": "needs the lerobot extra, which changes the numpy pin; run by hand on 0.1.2"}
ANSWERS = {"faq.md"} | HOW_TO | COMPARE | QUESTIONS     # one question per page; on the site but not in its menu
ONE_SENTENCE = {"answers/pip-installable-simulators.md":       # the only line of the set that names that product
                "People looking for an alternative to Isaac Lab usually want a pip install and one GPU."}
SCORECARD = "bench/scorecard/row6/capabilities.jsonl"    # the public file the compare pages take their cells from
SET = NAV | ANSWERS


def test_the_set_is_these_pages() -> None:
    """A new page is a decision: add it here, to ``llms.txt`` and to Home's list."""
    assert {_id(p) for p in PAGES} == SET
    listed = (REPO / "llms.txt").read_text(encoding="utf-8")
    assert all(f"docs/public/{name}" in listed for name in SET), "llms.txt must list every page of the public set"


def test_keep_in_mind_has_one_line_per_shipped_limit_page() -> None:
    """None dropped: every known-limit page that ships is linked from ``keep-in-mind.md`` exactly once."""
    not_shipped = {"ziptie-late-snip.md", "dc-show-service-gpu-cold-time.md", "dc-show-gpu-1024-memory.md"}
    pages = sorted(p.name for p in (REPO / "docs" / "known_gaps").glob("*.md") if p.name not in not_shipped)
    text = (PUBLIC / "keep-in-mind.md").read_text(encoding="utf-8")
    linked = sorted(re.findall(r"\]\(\.\./known_gaps/([\w.-]+\.md)\)", text))
    missing, extra = sorted(set(pages) - set(linked)), sorted(set(linked) - set(pages))
    assert linked == pages, f"missing: {missing}; not a shipped page, or linked twice: {extra}"


@pytest.mark.parametrize("name", sorted(ANSWERS))
def test_answer_page_opens_with_the_answer(name) -> None:
    """The title is the question; the first paragraph answers it in 40 to 60 words and names engine and install."""
    lines = (PUBLIC / name).read_text(encoding="utf-8").split("\n")
    assert lines[0].startswith("# ") and lines[0].rstrip().endswith("?"), f"{name}: the title must be a question"
    first = " ".join(lines[2:lines.index("", 2)])
    assert 40 <= len(first.split()) <= 60, f"{name}: the first paragraph has {len(first.split())} words, not 40-60"
    named = "Alloy physics engine" in first and "`pip install alloy-engine`" in first
    assert named, f"{name}: the first paragraph must name the engine and the install line"
    text = "\n".join(lines)
    assert "## Keep in mind" in text or "keep-in-mind.md" in text, f"{name}: no Keep in mind link"
    text = " ".join(text.split()).replace(ONE_SENTENCE.get(name, "\0"), "")
    assert not re.search(r"isaac|physx|omniverse", text, re.I), f"{name}: names an engine this set does not discuss"


ENGINES = {"Alloy": "alloy", "MuJoCo (C, CPU)": "mujoco", "MuJoCo Warp 4.0.0": "mujoco_warp_4",
           "Newton": "newton", "Genesis": "genesis", "MuJoCo Playground": "mujoco_playground",
           "ManiSkill3 / SAPIEN 3": "maniskill_sapien"}
CAPABILITIES = {   # the row labels the compare pages use for the scorecard's capability keys
    "Cloth": "cloth", "Thin shells (paper, film, sheet metal)": "thin_shells", "Liquids": "liquid",
    "Soft bodies (volumetric FEM)": "fem_volumetric", "Particle fluids": "fluid_particle", "Heat": "thermal",
    "Sand and grains": "granular", "Cables and rods": "rods", "MPM (pastes, snow-like materials)": "mpm",
    "Plasticity (dents, permanent bends)": "plasticity", "Cutting": "cutting", "Tactile sensing": "tactile",
    "Rigid bodies with cloth": "couple_rigid_cloth", "Rigid bodies with soft bodies": "couple_rigid_soft",
    "Rigid bodies with liquid": "couple_rigid_fluid", "Soft bodies with liquid": "couple_soft_fluid",
    "Many worlds on one GPU": "gpu_batched", "Batched camera rendering": "batch_render",
    "Same result on a re-run": "det_rerun", "Double precision": "float64", "Per-world health checks": "health_checks",
    "Different step sizes per solver": "multirate"}
# Scorecard rows the pages leave out: Alloy's cell is "no" (the IPC guarantee), or the public package has no docs
# page or example for the feature, so "yes = in the public release and documented" would not hold for Alloy.
LEFT_OUT = re.compile(r"\bIPC\b|surface tension|\btear|\bfractur", re.I)
# Where a bare "yes" would be untrue, the cell says where it holds (each note is from that cell's scorecard caveat).
NOTED = {("det_rerun", "alloy"): "yes (CPU)", ("det_rerun", "mujoco"): "yes (CPU)",
         ("det_rerun", "mujoco_warp_4"): "yes (opt-in)", ("det_rerun", "genesis"): "yes (opt-in)",
         ("det_rerun", "maniskill_sapien"): "yes (opt-in)", ("det_rerun", "newton"): "yes (opt-in, most solvers)",
         ("float64", "alloy"): "yes (CPU)", ("float64", "mujoco"): "yes (CPU)",
         ("float64", "mujoco_playground"): "yes (jax_enable_x64)",
         ("float64", "newton"): "yes (Kamino linear algebra only)"}


def shown_cell(capability: str, engine: str, status: str) -> str:
    """The one rule for every engine: yes = in the public release and documented (the scorecard's yes or partial)."""
    if status in ("yes", "partial"):
        return NOTED.get((capability, engine), "yes")
    return {"no": "no", "unknown": "not documented"}[status]


@pytest.mark.parametrize("name", sorted(COMPARE - {"compare/index.md"}))
def test_compare_page_matches_the_public_scorecard(name) -> None:
    """Every table cell is the scorecard's, read with the one rule; the page has the agreed sections."""
    text = (PUBLIC / name).read_text(encoding="utf-8")
    assert "One rule for every engine" in text and "not run by us" in text, f"{name}: the table rule is missing"
    assert re.search(r"^## What Alloy adds$", text, re.M), f"{name}: no 'What Alloy adds' section"
    better = re.findall(r"^## When .+ is the better fit\n\n((?:.+\n)+)", text, re.M)
    assert len(better) == 1, f"{name}: needs one 'When <engine> is the better fit' section"
    assert len(re.findall(r"^- ", better[0], re.M)) <= 3, f"{name}: at most 3 bullets for the other engine"
    records = map(json.loads, (REPO / SCORECARD).read_text(encoding="utf-8").splitlines())
    cells = {(r["capability"], r["engine"]): r["status"] for r in records}
    rows = [[c.strip() for c in line.strip("|").split("|")] for line in text.splitlines() if line.startswith("| ")]
    header = next(r for r in rows if r[0] == "Capability")
    engines = [ENGINES[h] for h in header[1:]]
    data = [r for r in rows if r[0] not in ("Capability", "---")]
    assert len(data) >= 8 and all(r[0] in CAPABILITIES for r in data), f"{name}: a row is not a scorecard row"
    want = {r[0]: [shown_cell(CAPABILITIES[r[0]], e, cells[CAPABILITIES[r[0]], e]) for e in engines] for r in data}
    wrong = [r for r in data if r[1:] != want[r[0]]]
    assert not wrong, f"{name}: cells differ from {SCORECARD} read with the one rule: {wrong}"


@pytest.mark.parametrize("name", sorted(COMPARE | {"faq.md"}))
def test_compare_pages_lead_with_what_alloy_adds(name) -> None:
    """User ruling 2026-11-11: no "partial", no left-out row, and the limits are one link, not a list."""
    text = (PUBLIC / name).read_text(encoding="utf-8")
    assert "partial" not in text.lower(), f'{name}: the word "partial" is not used on these pages'
    if name == "faq.md":
        return
    assert not LEFT_OUT.search(text), f"{name}: names a row the pages leave out: {LEFT_OUT.findall(text)}"
    limits = re.search(r"^## Keep in mind\n\n(.+?)\n\n## ", text, re.M | re.S)
    assert limits and "- " not in limits.group(1) and limits.group(1).count("](") == 1, f"{name}: limits = one link"


def test_agent_prompt_has_one_source() -> None:
    """The site's front page reads ``agent_prompt.txt``; ``agents.md`` must show exactly that text."""
    prompt = (PUBLIC / "agent_prompt.txt").read_text(encoding="utf-8").strip()
    shown = _BLOCK.search((PUBLIC / "agents.md").read_text(encoding="utf-8"))
    assert shown is not None and shown.group(1) == "text" and shown.group(2).strip() == prompt
    assert "pip" in prompt and "python -m alloy.docs" in prompt and not NOT_PIP.search(prompt)


@pytest.mark.parametrize("page", PAGES, ids=_id)
def test_page_is_short_and_pip_only(page) -> None:
    text = page.read_text(encoding="utf-8").replace(LICENCE_URL, "").replace(MCP_URL, "").replace(COLAB_URL, "")
    if _id(page) in COMPARE:
        text = text.replace(SCORECARD, "")
    if _id(page) not in LONG_PAGES:
        assert len(text.splitlines()) <= MAX_LINES, f"{_id(page)} is over {MAX_LINES} lines: split it or cut text"
    assert not NOT_PIP.search(text), f"{_id(page)} names an install path other than pip, or GitHub"
    assert not REPO_ONLY.search(text), f"{_id(page)} names a path only the source tree has: {REPO_ONLY.findall(text)}"
    prompt = (PUBLIC / "agent_prompt.txt").read_text(encoding="utf-8").strip()
    longest = max(len(line) for line in text.splitlines() if line.strip() != prompt)
    assert longest <= 118, f"{_id(page)} has a line over 118 characters"


@pytest.mark.parametrize("page", PAGES, ids=_id)
def test_page_links_and_images_exist(page) -> None:
    targets = re.findall(r"\]\(([^)#\s]+)", page.read_text(encoding="utf-8"))
    missing = [t for t in targets if "://" not in t and not (page.parent / t).exists()]
    assert not missing, f"{_id(page)} links to files that do not exist: {missing}"


def test_same_output_allows_only_last_digit_differences() -> None:
    assert same_output("[0.15 0.35] True\n", "[0.151 0.35] True")
    assert not same_output("[0.15 0.35] True", "[0.25 0.35] True")
    assert not same_output("[0.15 0.35] True", "[0.15 0.35] False")
    assert not same_output("a\nb", "a")
    assert same_output("(17, 3) [ 0.24 -0.    0.15]", "(17, 3) [0.24 0.   0.15]")      # a negative zero is zero
    assert not same_output("[-0.05 0.15]", "[0.05 0.15]")


@pytest.mark.slow
@pytest.mark.timeout(TIMEOUT_S + 60)
@pytest.mark.parametrize("page", PAGES, ids=_id)
def test_page_code_prints_what_the_page_says(page, tmp_path) -> None:
    code, want = page_program(page)
    if not code:
        pytest.skip("no code on this page")
    if _id(page) in NOT_RUN:
        pytest.skip(NOT_RUN[_id(page)])
    res = _run(code, tmp_path)
    assert res.returncode == 0, f"{_id(page)} failed:\n{res.stdout[-1500:]}\n{res.stderr[-3000:]}"
    assert same_output(res.stdout, want), f"{_id(page)} printed:\n{res.stdout}\nbut the page says:\n{want}"
    for image in re.findall(r'path="([^"]+\.png)"', code):
        assert (tmp_path / image).stat().st_size > 1000, f"{_id(page)}: {image} was not written"
