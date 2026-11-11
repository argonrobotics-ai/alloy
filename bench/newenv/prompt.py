#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Assemble one run's TASK.md: preamble (engine fields) + the engine-neutral contract + the request's block.

Standalone on purpose (standard library only): the overlay harness (harness.py) installs this file as
``sandbox/build_prompt.py`` on the eval VM, where run_one.sh calls ``build_prompt.py ENGINE REQUEST_ID --root H``.
Layout under ROOT: sandbox/engines.json, tasks/preamble.md, tasks/preamble_contract.md, tasks/blocks/<id>.md (the
request-specific block, written by make_task.py when the overlay is built).

    python3 prompt.py ENGINE E01 [--root bench/newenv-layout-dir]  > TASK.md

Example:
    >>> fill("Hi {engine_name}, {request} {{x}}", {"engine_name": "A", "request": "r"})
    'Hi A, r {x}'
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FIELD = re.compile(r"(?<!\{)\{([a-z_]+)\}(?!\})")
AGENT_PROMPT = "Read ~/task/TASK.md and carry out the task it describes."


def fill(template: str, values: dict[str, str]) -> str:
    """Replace {name} fields; an unknown {name} raises (a typo must not reach an agent). `{{` and `}}` become literal braces (as in str.format); other braces stay."""
    unknown = sorted({m for m in FIELD.findall(template) if m not in values})
    if unknown:
        raise KeyError(f"preamble has unknown fields: {', '.join(unknown)}")
    return FIELD.sub(lambda m: values[m.group(1)], template).replace("{{", "{").replace("}}", "}")


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:g}"


def build(engine: str, request_id: str, root: Path) -> str:
    engines = json.loads((root / "sandbox" / "engines.json").read_text())
    block = root / "tasks" / "blocks" / f"{request_id}.md"
    if engine not in engines or engine.startswith("_"):
        raise KeyError(f"unknown engine {engine!r}; known: {', '.join(k for k in engines if not k.startswith('_'))}")
    if not block.is_file():
        known = sorted(p.stem for p in block.parent.glob("*.md"))
        raise KeyError(f"unknown request {request_id!r}; known: {', '.join(known)}")
    spec, defaults = engines[engine], engines.get("_defaults", {})
    values = {"engine_name": spec["name"], "engine_install": spec["install"], "engine_docs": spec["docs"],
              "agent_minutes": _minutes(spec.get("agent_s", defaults.get("agent_s", 2700))),
              "rerun_minutes": _minutes(spec.get("rerun_s", defaults.get("rerun_s", 900)))}
    head = fill((root / "tasks" / "preamble.md").read_text(), values)
    return head.rstrip("\n") + "\n" + (root / "tasks" / "preamble_contract.md").read_text() + block.read_text()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("request")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    a = ap.parse_args()
    print(build(a.engine, a.request, a.root), end="")


if __name__ == "__main__":
    main()
