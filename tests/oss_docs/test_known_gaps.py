# SPDX-License-Identifier: Apache-2.0
"""Every known-limit file (``docs/known_gaps/*.md``) names its owner, has a title and the five required sections.

A section is a ``## <name>`` heading or a paragraph that starts with a bold lead-in ``**<name>.**``; each name has a
few accepted spellings (``SECTIONS``). The template is in ``docs/guide/known_limits.md``; this test also checks that
the template passes. Check your own file before merging: ``python -m pytest tests/oss_docs/test_known_gaps.py -q``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from alloy.testing.oss_docs import REPO

GAPS = REPO / "docs" / "known_gaps"
GUIDE = REPO / "docs" / "guide" / "known_limits.md"
SECTIONS = {  # required section -> accepted names (any case, trailing "." or ":" ignored)
    "What is wrong": ("what is wrong", "what"),
    "Number": ("number", "numbers", "the number", "the numbers"),
    "Why": ("why",),
    "Workaround": ("workaround", "workarounds"),
    "Planned fix": ("planned fix", "the fix", "the fix and when", "fix"),
}
_FENCE = re.compile(r"^```.*?^```", re.S | re.M)
_HEADING = re.compile(r"## +(.+?)\s*$")
_LEAD = re.compile(r"\*\*([^*]+?)\*\*[.:]?\s*(.*)$")


def gap_files() -> list[Path]:
    """The limit files: every ``*.md`` in ``docs/known_gaps`` except a folder README."""
    return sorted(p for p in GAPS.glob("*.md") if p.name.lower() != "readme.md") if GAPS.is_dir() else []


def front_matter(text: str) -> tuple[dict[str, str], str]:
    """``({key: value}, rest)`` for a leading ``---`` block of ``key: value`` lines; ``({}, text)`` without one."""
    if not text.startswith("---\n") or "\n---" not in text[3:]:
        return {}, text
    head, rest = text[4:].split("\n---", 1)
    pairs = (ln.split(":", 1) for ln in head.splitlines() if ":" in ln)
    return {k.strip(): v.strip() for k, v in pairs}, rest.split("\n", 1)[1] if "\n" in rest else ""


def sections(text: str) -> list[tuple[str, str]]:
    """``[(name, body), ...]`` in file order: ``## `` headings and bold lead-ins outside code fences, names folded."""
    out: list[tuple[str, list[str]]] = []
    for line in _FENCE.sub("", text).splitlines():
        head = _HEADING.match(line)
        lead = None if head else _LEAD.match(line)
        if head or lead:
            name = (head or lead).group(1).strip().rstrip(".:").strip().lower()
            out.append((name, [] if head else [lead.group(2)]))
        elif out:
            out[-1][1].append(line)
    return [(name, "\n".join(body).strip()) for name, body in out]


def problems(text: str) -> list[str]:
    """What a limit file lacks: an ``owner:`` in front matter, a ``# `` title, each of SECTIONS with text.

    Example: ``problems("---\\nowner: me\\n---\\n# t\\n## What\\nx\\n")`` -> ``["missing section 'Number' ...", ...]``."""
    meta, body = front_matter(text)
    out = [] if meta.get("owner") else ["no owner: start the file with front matter '---' / 'owner: <owner>' / '---'"]
    first = next((ln for ln in body.splitlines() if ln.strip()), "")
    if not (first.startswith("# ") and first[2:].strip()):
        out.append("the first line after the front matter must be the title: '# <the limit in one line>'")
    found = sections(body)
    for want, names in SECTIONS.items():
        bodies = [b for name, b in found if name in names]
        if not bodies:
            out.append(f"missing section '{want}' (a '## {want}' heading or a '**{want}.**' lead-in)")
        elif not any(bodies):
            out.append(f"section '{want}' is empty")
    return out


@pytest.mark.parametrize("path", gap_files(), ids=lambda p: p.name)
def test_known_gap_has_required_sections(path: Path) -> None:
    bad = problems(path.read_text(encoding="utf-8"))
    assert not bad, f"{path.relative_to(REPO)}: " + "; ".join(bad) + f" (template: {GUIDE.relative_to(REPO)})"


def test_guide_template_passes() -> None:
    blocks = re.findall(r"^```markdown\n(.*?)^```", GUIDE.read_text(encoding="utf-8"), re.S | re.M)
    assert blocks, f"{GUIDE.relative_to(REPO)} lost its ```markdown template"
    assert not problems(blocks[0]), problems(blocks[0])
    assert [name for name, _ in sections(front_matter(blocks[0])[1])] == [k.lower() for k in SECTIONS]


GOOD_HEADINGS = "---\nowner: me\n---\n# A limit\n## What\na\n## Number\nb\n## Why\nc\n## Workaround\nd\n## Planned fix\ne\n"
GOOD_LEADS = ("---\nowner: me\n---\n\n# A limit\n\n**What is wrong.** a\n\n**The number.** b\n\n**Why.** c\n\n"
              "**Workaround.** d\n\n**The fix and when.** e\n")


@pytest.mark.parametrize("text, expect", [
    (GOOD_HEADINGS, []),
    (GOOD_LEADS, []),
    (GOOD_HEADINGS.replace("## Number\nb\n", "## Number\n\n"), ["section 'Number' is empty"]),
    (GOOD_LEADS.replace("**Why.** c", "c"), ["missing section 'Why' (a '## Why' heading or a '**Why.**' lead-in)"]),
    (GOOD_HEADINGS.replace("---\nowner: me\n---\n", ""),
     ["no owner: start the file with front matter '---' / 'owner: <owner>' / '---'"]),
    (GOOD_HEADINGS.replace("# A limit\n", "Intro\n"),
     ["the first line after the front matter must be the title: '# <the limit in one line>'"]),
    (GOOD_HEADINGS.replace("## Why\nc\n", "```\n## Why\nc\n```\n"),
     ["missing section 'Why' (a '## Why' heading or a '**Why.**' lead-in)"]),
])
def test_problems_finds_each_defect(text: str, expect: list[str]) -> None:
    assert problems(text) == expect
