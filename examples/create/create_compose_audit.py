# SPDX-License-Identifier: Apache-2.0
"""Your own object, in a scene, audited: ``alloy create`` -> ``alloy.compose`` -> ``alloy audit``.

1. create: a 3D-printed cup (a mesh in millimetres) becomes ``my_objects/objects/printed_cup.yaml``; with
   ``--text`` (and a generator, see ``alloy/create/README.md``) a second object is made from a description;
2. compose: the new objects and a wooden cube go on a table, placed and checked (``my_scenes/scenes/*.yaml``);
3. audit: ``alloy.check`` over several seeds plus the bully agent's random legal episodes; the findings come back
   by severity, each with a replay command and a fix.

Run (needs the ``import`` extra; CPU is fine, about 2 minutes):

    python examples/create/create_compose_audit.py
    python examples/create/create_compose_audit.py --text "a red ceramic coffee mug with a handle" --json

The same from the command line::

    python -m alloy.create --mesh cup.stl --material plastic --out my_objects
    echo '{"items": [{"query": "printed_cup", "object": "printed_cup"}, {"query": "cube"}]}' > scene.json
    python -m alloy.compose build --spec scene.json --root my_objects --out my_scenes
    python -m alloy.audit my_scenes/scenes/<scene>.yaml --library my_objects --library my_scenes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import alloy.audit
import alloy.compose
import alloy.create

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_to_card import write_cup  # noqa: E402 - the cup mesh from the mesh example


def make_objects(out: str, text: str | None) -> list[str]:
    """Create the cup (and the object described by ``text``); returns the new card names."""
    Path(out).mkdir(parents=True, exist_ok=True)
    cup = alloy.create.create(mesh=str(write_cup(Path(out) / "printed_cup.stl")), material="plastic", out=out,
                              device="cpu")
    print(cup.summary())
    names = [cup.name]
    if text:
        made = alloy.create.create(text=text, out=out, device="cpu")
        print(made.summary())
        names.append(made.name)
    return names


def make_scene(names: list[str], out: str, scenes: str) -> Path:
    """A table with the new objects (pinned by name) and a wooden cube; returns the saved scene card (cards compose
    makes for the scene, if any, go to ``scenes/objects``)."""
    items = [{"query": n, "object": n} for n in names] + [{"query": "cube", "words": ["wooden"]}]
    result = alloy.compose.compose({"items": items, "support": "table", "clutter": "tidy"}, roots=[out],
                                   device="cpu")
    if not result.ok:
        raise SystemExit(f"compose failed: {result.to_dict().get('notes')}")
    return next(p for p in result.save(scenes) if p.parent.name == "scenes")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--text", default=None, help="also create an object from this description (needs a generator)")
    p.add_argument("--out", default="my_objects")
    p.add_argument("--scenes", default="my_scenes")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    scene = make_scene(make_objects(a.out, a.text), a.out, a.scenes)
    report = alloy.audit.audit(str(scene), rounds=a.rounds, episodes=a.episodes, device="cpu",
                               library=[a.out, a.scenes])
    print(report.to_json() if a.json else report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
