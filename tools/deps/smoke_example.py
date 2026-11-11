# SPDX-License-Identifier: Apache-2.0
"""Install smoke example: drop a 3 cm ABS block on a plywood slab for 1 s and check that it comes to rest.

    python tools/deps/smoke_example.py [--device cpu|cuda:0] [--seconds 1.0]

Uses only an installed Alloy (cards from the shipped library, no repository checkout). Prints one JSON line:
device, versions, the block's start and final height, its last-period motion, wall time and ``ok``. Exit 0 when the
block fell and rests, 1 otherwise.

Example (Python):
    >>> "worlds: 1" in SCENE
    True
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

OBJECTS = {
    "slab": "schema: alloy.object/1\nname: slab\nshape: {primitive: {kind: box, size: [0.3, 0.3, 0.02]}}\n"
            "material: plywood_marine\nmotion: static\n",
    "block": "schema: alloy.object/1\nname: block\nshape: {primitive: {kind: box, size: [0.03, 0.03, 0.03]}}\n"
             "material: abs\n",
}
SCENE = """schema: alloy.scene/1
robots: []
objects:
  - {object: slab,  pose: [0, 0, -0.02, 0, 0, 0, 1]}
  - {object: block, pose: [0, 0, 0.08, 0, 0, 0, 1]}
worlds: 1
control_dt: 0.01
"""
START_Z = 0.08
REST_MOTION = 1e-4  # m per control period


def write_cards(root: Path) -> None:
    """The two object cards and the scene card, under ``root``."""
    (root / "objects").mkdir(parents=True)
    (root / "scenes").mkdir()
    for name, text in OBJECTS.items():
        (root / "objects" / f"{name}.yaml").write_text(text)
    (root / "scenes" / "drop.yaml").write_text(SCENE)


def versions() -> dict[str, str]:
    """Installed versions of alloy and the engine packages it takes from PyPI."""
    import importlib.metadata as md

    out = {}
    for dist in ("alloy-engine", "mujoco", "warp-lang"):  # newton and mujoco-warp are vendored inside alloy
        try:
            out[dist] = md.version(dist)
        except md.PackageNotFoundError:
            out[dist] = "missing"
    return out


def run(device: str | None, seconds: float) -> dict[str, object]:
    """Build the scene from cards, step it, and report the block's motion."""
    import warp as wp

    from alloy.cards.library import CardLibrary
    from alloy.scene.assembler import SceneAssembler
    from alloy.sim import Simulation

    wp.config.log_level = wp.LOG_WARNING
    wp.config.quiet = True
    dev = device or ("cuda:0" if wp.get_cuda_device_count() else "cpu")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp, wp.ScopedDevice(dev):
        write_cards(Path(tmp))
        lib = CardLibrary(roots=[Path(tmp), CardLibrary().roots[0]])  # shipped library supplies the materials
        built = SceneAssembler(lib).assemble(lib.scene("drop"), dev)
        sim = Simulation(built)
        block = next(k for k, label in enumerate(built.model.body_label) if "block" in str(label))
        t_build = time.time() - t0
        zs = []
        for _ in range(int(round(seconds / built.control_dt))):
            sim.step_period()
            zs.append(float(sim.state.body_q.numpy()[block, 2]))
    motion = abs(zs[-1] - zs[-2]) if len(zs) > 1 else float("inf")
    ok = zs[-1] < START_Z - 0.03 and motion < REST_MOTION
    return {"device": dev, "versions": versions(), "start_z": START_Z, "final_z": round(zs[-1], 6),
            "last_period_motion_m": motion, "build_s": round(t_build, 1), "wall_s": round(time.time() - t0, 1),
            "ok": ok}


def main(argv: list[str] | None = None) -> int:
    """CLI entry: one JSON document on stdout."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--device", default=None, help="cpu or cuda:N (default: cuda:0 when present, else cpu)")
    ap.add_argument("--seconds", type=float, default=1.0)
    args = ap.parse_args(argv)
    doc = run(args.device, args.seconds)
    print(json.dumps(doc))  # one line: Warp's own start-up lines may precede it
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
