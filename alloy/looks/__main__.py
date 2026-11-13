# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.looks``: how every hand-authored object card resolves to a look (JSON); exit 1 if any is grey.

An unknown object (``--show``) or a fetch that cannot run prints one ``error:`` line and its ``fix:`` on stderr and
exits 2; an unknown ``--family`` is a usage error (exit 2, the families listed).

Example::

    python -m alloy.looks                 # library + alloy-assets checkout (if present)
    python -m alloy.looks --show towel_cotton
    python -m alloy.looks --pool                      # texture pool: textures per family
    python -m alloy.looks --fetch-textures --family wood --source origin   # into the render asset cache
"""

from __future__ import annotations

import argparse
import json
import sys

from alloy.cards import CardLibrary
from alloy.core.errors import CardError
from alloy.looks.resolve import coverage, resolve
from alloy.looks.schema import FAMILIES


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.looks", description=__doc__.splitlines()[0])
    ap.add_argument("--show", metavar="OBJECT", help="print the look one object resolves to")
    ap.add_argument("--pool", action="store_true", help="print the texture pool (alloy.looks.pool) per family")
    ap.add_argument("--fetch-textures", action="store_true", help="put the pool's texture sets in the render cache")
    ap.add_argument("--family", choices=FAMILIES, help="only this texture family (with --pool / --fetch-textures)")
    ap.add_argument("--source", choices=("hub", "origin"), default="hub",
                    help="--fetch-textures from the library dataset (hub) or Poly Haven / ambientCG (origin)")
    args = ap.parse_args(argv)
    lib = CardLibrary()
    if args.pool or args.fetch_textures:
        return _pool_command(args, lib)
    if args.show:
        try:
            look = resolve(args.show, lib)
        except CardError as e:             # an unknown or broken card: one line and its fix, not a traceback
            return _fail(str(e), "`alloy list objects` names the objects; check the spelling" if e.field == "name"
                         else f"correct '{e.field}' in the card {e.card}")
        print(json.dumps({"object": look.name, "source": look.source, "path": look.path,
                          "materials": look.materials, "assign": list(look.assign), "default": look.default}, indent=1))
        return 0
    report = coverage(lib)
    print(json.dumps(report, indent=1))
    return 1 if report["grey"] else 0


def _pool_command(args, lib: CardLibrary) -> int:
    from alloy.looks import pool
    if args.fetch_textures:
        from alloy.assets.hub.source import HubError
        try:
            report = pool.fetch(args.family, args.source, lib)
        except HubError as e:              # no huggingface_hub, offline, ...: one line and its fix, not a traceback
            return _fail(e.message, e.fix)
        print(json.dumps(report, indent=1))
        return 1 if report["missing"] else 0
    rows = {f: [{"look": e.name, "texture": e.texture, "author": e.author, "source": e.source}
                for e in pool.entries(lib, f)] for f in ([args.family] if args.family else pool.FAMILIES)}
    print(json.dumps({"families": {f: len(r) for f, r in rows.items()}, "textures": rows}, indent=1))
    return 0


def _fail(message: str, fix: str) -> int:
    """One ``error:`` line and its ``fix:`` on stderr; exit code 2."""
    print(f"error: {message}\nfix: {fix}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
