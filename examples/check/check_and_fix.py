# SPDX-License-Identifier: Apache-2.0
"""Check a scene, apply the suggested fixes, check again.

A 30 mm beech cube is placed 3 mm into a plywood table top. The load check finds the overlap and says how far to
lift the cube; ``apply_fixes`` writes the new pose; the full check (a short simulation) then passes.

Run (CPU is fine; a GPU makes the full check faster):

    python examples/check/check_and_fix.py            # prints both reports as text
    python examples/check/check_and_fix.py --json     # prints the final report as JSON

The same check from the command line: ``python -m alloy.check library/scenes/demo_peg_d12.yaml --json``.
"""

from __future__ import annotations

import sys

import alloy.check
from alloy.cards import types as ct


def scene(cube_z: float) -> ct.SceneCard:
    """A table top (top face at z = 0) and a 30 mm cube whose centre is at ``cube_z``."""
    table = ct.Placement("table_top_plywood", (0.0, 0.0, -0.009, 0.0, 0.0, 0.0, 1.0))
    cube = ct.Placement("cube_beech_30", (0.0, 0.0, cube_z, 0.0, 0.0, 0.0, 1.0))
    return ct.SceneCard(robots=(), objects=(table, cube))


def main() -> int:
    card = scene(0.012)                                   # 3 mm too low: overlaps the table
    first = alloy.check.check_load(card, worlds=2)        # cards + overlap, no simulation
    print(first.summary())
    card, applied = alloy.check.apply_fixes(card, first)
    print("applied:", applied)
    final = alloy.check(card, worlds=2, seconds=0.25)     # every check, with a short run
    print(final.to_json() if "--json" in sys.argv else final.summary())
    return 0 if final.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
