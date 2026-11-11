# SPDX-License-Identifier: Apache-2.0
"""Command line of the OptiX helper: ``python -m alloy.camera.optix build|status``."""
from __future__ import annotations

import argparse
import sys

from ..errors import CameraError
from . import available
from .build import LICENCE_ENV, build


def main(argv: list[str] | None = None) -> int:
    """Run the CLI; returns the process exit code (0 ok, 2 refused or failed)."""
    ap = argparse.ArgumentParser(prog="python -m alloy.camera.optix", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="compile libowf.so and the device programs into the cache")
    b.add_argument("--accept-nvidia-optix-license", action="store_true", dest="accept",
                   help=f"accept NVIDIA's OptiX SDK licence (or set {LICENCE_ENV}=1)")
    b.add_argument("--force", action="store_true", help="rebuild even if the cache already has a build")
    sub.add_parser("status", help="say whether the OptiX backend can run here")
    args = ap.parse_args(argv)
    if args.cmd == "status":
        ok, why = available()
        sys.stdout.write("OptiX backend available\n" if ok else f"OptiX backend not available: {why}\n")
        return 0 if ok else 2
    try:
        out = build(accept_licence=args.accept, force=args.force)
    except CameraError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    sys.stdout.write(f"built {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
