# SPDX-License-Identifier: Apache-2.0
"""A sentence in, a physics-checked scene and two variations out, saved as scene cards, then simulated.

The builder picks a mug, a spoon and three wooden cubes from the card library, places them on the plywood table with
clearance, checks the scene after every placement and once more with a short simulation, and fixes what the checks
flag. Variations move the objects and may swap in other equally good matches.

Run (CPU is fine; a GPU makes the checks faster):

    python examples/compose/sentence_to_scene.py                # summary, saves to ./composed_scenes
    python examples/compose/sentence_to_scene.py --json         # the full result as JSON

The same from the command line:
``python -m alloy.compose "a mug, a spoon and three wooden cubes on a table" --variations 2 --out composed_scenes``.
"""

from __future__ import annotations

import sys

import alloy.api
import alloy.compose


def main() -> int:
    result = alloy.compose.compose("a mug, a spoon and three wooden cubes on a table", variations=2)
    print(result.to_json() if "--json" in sys.argv else result.summary())
    paths = result.save("composed_scenes")                 # scenes/scene.yaml, scenes/scene_v01.yaml, ...
    print("saved:", [str(p) for p in paths])
    sim = alloy.api.build(result.scene(0), worlds=4)        # an editable alloy.Scene -> a running Sim
    sim.run(0.5)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
