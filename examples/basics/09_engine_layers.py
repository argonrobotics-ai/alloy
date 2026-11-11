# SPDX-License-Identifier: Apache-2.0
"""The engine layers under alloy.build: hand-written YAML cards, CardLibrary, SceneAssembler, build_simulation.

Use these only when the API cannot express something; alloy.build does the same three steps.
Run: python examples/basics/09_engine_layers.py
"""
import json
import tempfile
from pathlib import Path

import warp as wp

wp.config.log_level = wp.LOG_WARNING      # before anything initializes Warp: no banner, no per-module load lines

from alloy.cards.library import CardLibrary, default_root  # noqa: E402
from alloy.integration.wiring import build_simulation  # noqa: E402
from alloy.scene.assembler import SceneAssembler  # noqa: E402

CARDS = {
    "objects/slab.yaml": "schema: alloy.object/1\nname: slab\nshape: {primitive: {kind: box, size: [0.3, 0.3, 0.02]}}\n"
                         "material: plywood_marine\nmotion: static\n",                    # sizes in cards: half extents
    "objects/block.yaml": "schema: alloy.object/1\nname: block\nshape: {primitive: {kind: box, size: [0.02, 0.02, "
                          "0.02]}}\nmaterial: abs\n",
    "scenes/drop.yaml": "schema: alloy.scene/1\nrobots: []\nobjects:\n  - {object: slab, pose: [0, 0, -0.02, 0, 0, 0, 1]}\n"
                        "  - {object: block, pose: [0, 0, 0.1, 0, 0, 0, 1]}\nworlds: 4\ncontrol_dt: 0.01\n",
}
device = "cuda:0" if wp.get_cuda_device_count() else "cpu"
with tempfile.TemporaryDirectory() as tmp, wp.ScopedDevice(device):
    for rel, text in CARDS.items():
        (Path(tmp) / rel).parent.mkdir(parents=True, exist_ok=True)
        (Path(tmp) / rel).write_text(text)
    lib = CardLibrary(roots=[Path(tmp), default_root()])          # first root wins; the library gives the materials
    built = SceneAssembler(lib).assemble(lib.scene("drop"), device)
    sim = build_simulation(built, "standard", seed=0)
    for _ in range(50):
        sim.step_period()                                         # one control period (many physics substeps)
    q = sim.state.body_q.numpy().reshape(built.world_count, -1, 7)  # (worlds, bodies, xyz + quat xyzw)
    routes = {name: str(d.route) for name, d in built.report.decisions.items()}
    print(json.dumps({"device": device, "routes": routes, "block_z_per_world": q[:, 0, 2].round(4).tolist(),
                      "health_ok": not sim.report()["health"].invalid_worlds}))
