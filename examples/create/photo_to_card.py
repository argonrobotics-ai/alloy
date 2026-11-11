# SPDX-License-Identifier: Apache-2.0
"""Turn one photo of an object into a checked object card.

The photo goes to the 3D generator (background removed, image-to-3D with TripoSR, MIT licence), the mesh is stood
on its base, sized from the object's category (or your ``--size``), and then follows the mesh path: physics mesh,
hulls, material, mass, drop test. Everything that came from a guess is listed under ``estimates`` and ``verify``.

Needs a generator: a GPU here with the generator installed, or ``ALLOY_GEN3D_URL`` pointing at a machine running
``python -m alloy.create.serve`` (see ``alloy/create/README.md``). Best photos: the whole object, plain background,
seen from the side and a little above.

Run:

    python examples/create/photo_to_card.py bowl.jpg --name yellow_bowl
    python examples/create/photo_to_card.py bowl.jpg --name yellow_bowl --size 0.15 --material ceramic --json

The same from the command line: ``python -m alloy.create --photo bowl.jpg --name yellow_bowl --out my_objects``.
"""

from __future__ import annotations

import argparse
import json

import alloy.create
from alloy.create.request import CreateError


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("photo", help="a JPEG or PNG of one object")
    p.add_argument("--name", required=True, help="the card's name (snake_case), e.g. yellow_bowl")
    p.add_argument("--size", type=float, default=None, help="its longest side in metres, if you measured it")
    p.add_argument("--material", default=None, help="a word (ceramic, wood, steel) or a library material name")
    p.add_argument("--out", default="my_objects")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    try:
        result = alloy.create.create(photo=a.photo, name=a.name, size=a.size, material=a.material, out=a.out,
                                     device="cpu")
    except CreateError as exc:
        print(json.dumps(exc.to_dict(), indent=2))
        return 2
    print(result.to_json() if a.json else result.summary())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
