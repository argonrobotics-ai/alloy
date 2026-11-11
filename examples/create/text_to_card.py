# SPDX-License-Identifier: Apache-2.0
"""Turn a text description into a checked object card.

The text becomes a product-style image (Kandinsky 2.2, Apache-2.0), then a mesh (TripoSR, MIT), then a card exactly
as for a photo. The category in the text ("mug") gives the typical size and mass, marked ESTIMATE with a range.

Needs a generator: a GPU here with the generator installed, or ``ALLOY_GEN3D_URL`` pointing at a machine running
``python -m alloy.create.serve`` (see ``alloy/create/README.md``).

Run:

    python examples/create/text_to_card.py "a red ceramic coffee mug with a handle"
    python examples/create/text_to_card.py "a small wooden toy truck" --size 0.12 --json

The same from the command line: ``python -m alloy.create --text "a red ceramic mug" --out my_objects``.
"""

from __future__ import annotations

import argparse
import json

import alloy.create
from alloy.create.request import CreateError


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("text", nargs="?", default="a red ceramic coffee mug with a handle")
    p.add_argument("--name", default=None, help="the card's name (default: from the text)")
    p.add_argument("--size", type=float, default=None, help="its longest side in metres")
    p.add_argument("--out", default="my_objects")
    p.add_argument("--json", action="store_true")
    a = p.parse_args()
    try:
        result = alloy.create.create(text=a.text, name=a.name, size=a.size, out=a.out, device="cpu")
    except CreateError as exc:
        print(json.dumps(exc.to_dict(), indent=2))
        return 2
    print(result.to_json() if a.json else result.summary())
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
