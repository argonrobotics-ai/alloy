# SPDX-License-Identifier: Apache-2.0
"""``python -m alloy.viewer <scene>``: build a scene and show it in a viewer.

``<scene>`` is anything ``alloy.load`` takes: a library scene name, a scene card (.yaml), a Scene JSON file, or a
robot file (MJCF / URDF / USD). Example::

    python -m alloy.viewer demo_peg_d12 --backend viser --worlds 4 --show 0,3 --port 8080
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

BACKEND_NAMES = ("viser", "gl", "rerun", "usd", "null")


def _world_list(text: str) -> list[int] | None:
    """``"0,3"`` -> ``[0, 3]``; ``"all"`` -> None."""
    if text.strip().lower() in ("", "all"):
        return None
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--show takes world indices such as 0,3 (got {text!r})") from exc


def parser() -> argparse.ArgumentParser:
    """The command-line parser."""
    p = argparse.ArgumentParser(prog="python -m alloy.viewer", description="Show an Alloy scene in a live viewer.")
    p.add_argument("scene", help="library scene name, scene card (.yaml), Scene JSON, or robot file (.xml/.urdf/.usd)")
    p.add_argument("--backend", default="viser", choices=BACKEND_NAMES, help="viewer backend (default: viser)")
    p.add_argument("--worlds", type=int, default=None, help="number of worlds to build (default: the scene's)")
    p.add_argument("--show", type=_world_list, default=None, help="worlds to show, e.g. 0,3 (default: all)")
    p.add_argument("--seconds", type=float, default=None, help="simulated seconds to run (default: until closed)")
    p.add_argument("--port", type=int, default=None, help="viser port, or rerun's web port")
    p.add_argument("--host", default=None, help="viser address to bind (default: every interface)")
    p.add_argument("--path", default=None, help="usd output file, or a rerun .rrd recording")
    p.add_argument("--device", default=None, help="cpu or cuda:0 (default: a GPU when there is one)")
    p.add_argument("--paused", action="store_true", help="start paused (step from the viewer)")
    p.add_argument("--no-contacts", action="store_true", help="do not draw the contact overlay")
    p.add_argument("--joints", action="store_true", help="add a slider per position-driven joint (sets targets)")
    p.add_argument("--drag", action="store_true", help="let the mouse grab and pull bodies (applies a spring force)")
    return p


def _backend_options(args: argparse.Namespace) -> dict[str, Any]:
    picked = {"port": args.port, "host": args.host, "path": args.path}
    return {k: v for k, v in picked.items() if v is not None}


def _show(args: argparse.Namespace) -> None:
    import alloy  # noqa: PLC0415 - the parser and --help stay fast
    from alloy.viewer import launch  # noqa: PLC0415

    print(f"building {args.scene} (the first build of a scene compiles kernels: up to a few minutes, cached "
          "afterwards) ...", flush=True)
    sim = alloy.load(args.scene, worlds=args.worlds, device=args.device)
    with launch(sim, backend=args.backend, worlds=args.show, contacts=not args.no_contacts, paused=args.paused,
                joints=args.joints, drag=args.drag, **_backend_options(args)) as view:
        if view.url:
            print(f"viewer: {view.url}", flush=True)
        view.run(seconds=args.seconds)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command; returns the exit code (0 done, 2 an Alloy error, 130 Ctrl-C while loading)."""
    args = parser().parse_args(argv)
    from alloy.api.errors import AlloyAPIError  # noqa: PLC0415

    try:
        _show(args)
    except AlloyAPIError as exc:
        print(f"error: {exc.message}\n  fix: {exc.fix}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
