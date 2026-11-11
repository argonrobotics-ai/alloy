# SPDX-License-Identifier: Apache-2.0
"""The public docs set (``docs/public``): short pages whose code runs as written and prints what the page says.

Each page's ``python`` blocks, joined in order, are one program; its ``text`` blocks, joined, are that program's
expected output. The program runs in an empty folder. ``ALLOY_DOCS_PYTHON`` names the interpreter of an environment
with the release wheel installed; the run then has no source tree on its path. Without it the test uses this tree.
"""
from __future__ import annotations

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
LONG_PAGES = {"limits.md"}            # one line per known-limit page: as long as that list
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


SET = {"index.md", "quickstart.md", "agents.md", "errors.md", "limits.md", "guides/robots.md", "guides/rl.md",
       "guides/imitation.md", "guides/camera.md", "guides/liquids.md", "guides/cloth.md", "guides/cables.md",
       "guides/health.md"}


def test_the_set_is_these_pages() -> None:
    """A new page is a decision: add it here, to ``llms.txt`` and to Home's list."""
    assert {_id(p) for p in PAGES} == SET
    listed = (REPO / "llms.txt").read_text(encoding="utf-8")
    assert all(f"docs/public/{name}" in listed for name in SET), "llms.txt must list every page of the public set"


def test_known_limits_has_one_line_per_shipped_limit_page() -> None:
    """None dropped: every known-limit page that ships is linked from ``limits.md`` exactly once."""
    not_shipped = {"ziptie-late-snip.md", "dc-show-service-gpu-cold-time.md", "dc-show-gpu-1024-memory.md"}
    pages = sorted(p.name for p in (REPO / "docs" / "known_gaps").glob("*.md") if p.name not in not_shipped)
    text = (PUBLIC / "limits.md").read_text(encoding="utf-8")
    linked = sorted(re.findall(r"\]\(\.\./known_gaps/([\w.-]+\.md)\)", text))
    missing, extra = sorted(set(pages) - set(linked)), sorted(set(linked) - set(pages))
    assert linked == pages, f"missing: {missing}; not a shipped page, or linked twice: {extra}"


def test_agent_prompt_has_one_source() -> None:
    """The site's front page reads ``agent_prompt.txt``; ``agents.md`` must show exactly that text."""
    prompt = (PUBLIC / "agent_prompt.txt").read_text(encoding="utf-8").strip()
    shown = _BLOCK.search((PUBLIC / "agents.md").read_text(encoding="utf-8"))
    assert shown is not None and shown.group(1) == "text" and shown.group(2).strip() == prompt
    assert "pip" in prompt and "python -m alloy.docs" in prompt and not NOT_PIP.search(prompt)


@pytest.mark.parametrize("page", PAGES, ids=_id)
def test_page_is_short_and_pip_only(page) -> None:
    text = page.read_text(encoding="utf-8")
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
    res = _run(code, tmp_path)
    assert res.returncode == 0, f"{_id(page)} failed:\n{res.stdout[-1500:]}\n{res.stderr[-3000:]}"
    assert same_output(res.stdout, want), f"{_id(page)} printed:\n{res.stdout}\nbut the page says:\n{want}"
    for image in re.findall(r'path="([^"]+\.png)"', code):
        assert (tmp_path / image).stat().st_size > 1000, f"{_id(page)}: {image} was not written"
