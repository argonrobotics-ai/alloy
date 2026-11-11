# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.assets.hub``: fetch or list library objects without the ``alloy`` command (JSON out).

Example::

    python -m alloy.assets.hub fetch ycb_025_mug
    python -m alloy.assets.hub list --library robocasa
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    from alloy.assets.hub.fetch import fetch, list_objects  # noqa: PLC0415
    from alloy.assets.hub.source import HubError  # noqa: PLC0415

    ap = argparse.ArgumentParser(prog="python -m alloy.assets.hub", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["fetch", "list"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--library")
    ap.add_argument("--category")
    ap.add_argument("--repo")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        out = (list_objects(a.library, a.category, a.repo) if a.cmd == "list"
               else fetch(a.names or None, a.library, a.category, a.repo, a.workers))
    except HubError as exc:
        json.dump(exc.to_dict(), sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 1
    json.dump(out, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
