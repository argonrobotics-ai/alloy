# SPDX-License-Identifier: Apache-2.0
"""Fast-camera speed with scanned objects' photo textures: images per second at 128 and 640 px, warm, 3 runs each.

It uses only the public API (``alloy.Scene``, ``alloy.Camera``), so one copy of this file times any tree: run it
with ``PYTHONPATH=<tree>`` on the commit before the textures and on the one after. Two scenes, the same table and
camera: ``ycb`` (four fetched YCB scans on a table: sugar box, mug, mustard bottle, tomato soup can; ``alloy fetch``
them first) and ``plain`` (boxes and cylinders of the same sizes, no textures: the cost for untextured scenes).

    PYTHONPATH=. python tests/camera/bench_scan_textures.py --device cuda:0 --out rows.json
    PYTHONPATH=. python tests/camera/bench_scan_textures.py --device cpu --worlds128 2 --worlds640 1 --renders 2

Each row: scene, size, worlds, spp, the three runs' images per second and their median, the device and GPU, the
tree's commit and the tracer backend. Rendering only (no physics step between renders), after two warm-up renders.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

log = logging.getLogger("bench_scan_textures")

OBJECTS = [  # (card, plain stand-in, position on the table top, yaw rad)
    ("ycb_004_sugar_box", ("box", (0.09, 0.045, 0.175)), (-0.12, 0.06), 0.3),
    ("ycb_025_mug", ("cylinder", (0.042, 0.08)), (0.05, 0.10), -0.6),
    ("ycb_006_mustard_bottle", ("box", (0.095, 0.06, 0.19)), (0.12, -0.06), 0.9),
    ("ycb_005_tomato_soup_can", ("cylinder", (0.034, 0.10)), (-0.04, -0.10), 0.0),
]
TABLE = (0.8, 0.6, 0.4)  # full edges (m); top at z = 0.4
EYE, TARGET = (0.55, -0.55, 0.75), (0.0, 0.0, 0.45)


@dataclass
class Row:
    scene: str
    size: int
    worlds: int
    spp: int
    runs: list[float] = field(default_factory=list)
    median: float = 0.0
    device: str = ""
    gpu: str = ""
    commit: str = ""
    backend: str = ""


def build(kind: str, worlds: int, device: str):
    """The bench scene: a static table and four free objects standing on it (scans or plain shapes), in every world."""
    import alloy
    scene = alloy.Scene(worlds=worlds, dt=0.01)
    scene.add_ground()
    scene.add_box("table", size=TABLE, pos=(0.0, 0.0, 0.5 * TABLE[2]), static=True)
    for card, (shape, dims), (x, y), yaw in OBJECTS:
        name = card.split("_", 2)[-1]
        if kind == "ycb":
            scene.add_object(name, card, pos=(x, y, TABLE[2]), rpy=(0.0, 0.0, yaw))
        elif shape == "box":
            scene.add_box(name, size=dims, pos=(x, y, TABLE[2] + 0.5 * dims[2]), rpy=(0.0, 0.0, yaw))
        else:
            scene.add_cylinder(name, dims[0], dims[1], pos=(x, y, TABLE[2] + 0.5 * dims[1]))
    return alloy.build(scene, device=device)


def time_renders(sim, size: int, spp: int, renders: int, runs: int = 3) -> tuple[list[float], str]:
    """Images per second of ``runs`` timed loops of ``renders`` renders of every world, after two warm-up renders."""
    import warp as wp
    import alloy
    cam = alloy.Camera("bench", look_at=(EYE, TARGET), width=size, height=size, spp=spp)
    for _ in range(2):
        cam.render(sim)
    wp.synchronize_device(sim.device)
    out = []
    for _ in range(runs):
        t0 = time.perf_counter()
        for _ in range(renders):
            cam.render(sim)
        wp.synchronize_device(sim.device)
        out.append(sim.worlds * renders / (time.perf_counter() - t0))
    bound = getattr(cam, "_bound", None)
    backend = getattr(getattr(bound[1], "st", None), "backend", "") if bound else ""
    return out, str(backend or "")


def sig4(x: float) -> float:
    """``x`` to 4 significant figures (CPU rates are fractions of an image per second at 640 px)."""
    return float(f"{x:.4g}")


def commit_of(module_file: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(module_file).parent), "rev-parse", "--short=9", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def gpu_name(device: str) -> str:
    if not device.startswith("cuda"):
        return ""
    import warp as wp
    return str(wp.get_device(device).name)


def main(argv=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--scenes", default="ycb,plain")
    ap.add_argument("--worlds128", type=int, default=256)
    ap.add_argument("--worlds640", type=int, default=16)
    ap.add_argument("--spp", type=int, default=16)
    ap.add_argument("--renders", type=int, default=10, help="renders per timed run")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import alloy
    rows = []
    for kind in args.scenes.split(","):
        for size, worlds in ((128, args.worlds128), (640, args.worlds640)):
            sim = build(kind, worlds, args.device)
            runs, backend = time_renders(sim, size, args.spp, args.renders)
            row = Row(kind, size, worlds, args.spp, [sig4(r) for r in runs], sig4(statistics.median(runs)),
                      args.device, gpu_name(args.device), commit_of(alloy.__file__), backend)
            log.info("%s", json.dumps(asdict(row)))
            rows.append(asdict(row))
            sim.close()
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
    return rows


if __name__ == "__main__":
    main()
