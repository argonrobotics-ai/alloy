# SPDX-License-Identifier: Apache-2.0
"""Exact control: write the scene request as JSON (a SceneSpec) instead of a sentence.

Pin exact cards with ``object``, put items ``in`` or ``on`` another item, choose clutter and arrangement, and ask
for several parallel worlds. ``python -m alloy.compose parse "<sentence>"`` prints the spec a sentence gives, a
good starting point to edit.

Run:

    python examples/compose/spec_json.py                       # builds, checks and prints the summary
    python -m alloy.compose build --spec spec.json --out out    # the same from a JSON file
"""

from __future__ import annotations

import json

import alloy.compose

SPEC = {
    "items": [
        {"query": "tray", "object": "tray_aluminium"},                  # an exact card
        {"query": "nut", "count": 3, "relation": "in", "target": "tray"},
        {"query": "mug"},
        {"query": "spoon", "count": 2},                  # not pencils: a round one lying down can roll
    ],
    "support": "table",            # "table", "floor" or a static object card
    "clutter": "normal",           # "tidy", "normal" or "cluttered"
    "arrangement": "scatter",      # "scatter", "row" or "grid"
    "worlds": 16,                  # parallel worlds in the saved scene card
}


def main() -> int:
    spec = alloy.compose.SceneSpec.from_dict(SPEC)
    result = alloy.compose.compose(spec, variations=1, seed=7)
    print(result.summary())
    print(json.dumps(result.scenes[0].objects, indent=1))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
