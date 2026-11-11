# SPDX-License-Identifier: Apache-2.0
"""Physics checks on the written Objaverse import root, and pruning of the objects that fail them.

Every object is loaded through ``CardLibrary`` (Alloy's card validators: any error fails it) and dropped onto a plane
with its hulls, card mass, friction and restitution (``alloy.importers.common.droptest``: no tunnelling, sinking,
launch or non-finite state; it must come to rest). Failures are removed from the root: both cards, local meshes, the
manifest entry and the sourcing row; ``checks.json`` keeps every result.

Example:
    >>> from alloy.importers.objaverse.check import failures
    >>> failures({"a": {"cards": {"errors": []}, "drop": {"ok": True}},
    ...           "b": {"cards": {"errors": ["min_wall: missing"]}, "drop": {"ok": True}}})
    ['b']
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from alloy.importers.common import cards as cw
from alloy.importers.common import checks as ch
from alloy.importers.common import manifest as mf
from alloy.importers.common import queue as qu
from alloy.importers.objaverse.cards import LIBRARY


def failures(checks: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Names with a card error or a failed drop."""
    return sorted(n for n, c in checks.items() if c["cards"]["errors"] or not c["drop"].get("ok"))


def run_checks(root: Path, workers: int = 3) -> dict[str, dict]:
    """Validate and drop-test every object in the root's manifest (``common.checks``); returns ``checks.json``."""
    ch.run_checks(root, LIBRARY, None, workers)
    return json.loads((root / ch.CHECKS).read_text())


def prune(root: Path, names: list[str]) -> int:
    """Remove ``names`` from the root (cards, local meshes, manifest entries, queue rows); returns how many."""
    man = mf.load(root / "manifest.json", LIBRARY)
    for n in names:
        (root / "objects" / LIBRARY / f"{n}.yaml").unlink(missing_ok=True)
        (root / "materials" / LIBRARY / f"{n}{cw.EFF_SUFFIX}.yaml").unlink(missing_ok=True)
        shutil.rmtree(root / mf.MESH_DIR / n, ignore_errors=True)
        man["objects"].pop(n, None)
    mf.save(root / "manifest.json", man)
    qu.drop(root / "queue.jsonl", [(n, "material.density") for n in names])
    from alloy.importers.objaverse.assemble import write_attribution  # noqa: PLC0415

    write_attribution(root, man)
    return len(names)
