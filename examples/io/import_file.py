# SPDX-License-Identifier: Apache-2.0
"""Bring your own asset: import an MJCF, URDF or USD file into Alloy cards, then assemble and step it in Alloy.

Run:  python examples/io/import_file.py [file] [library_dir]
(default: the Panda scene shipped in the card library; any MJCF/URDF/USD works)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from alloy.cards.library import CardLibrary, default_root
from alloy.io import import_file


def main(path: str | None = None, out: str = "out/io_import") -> dict:
    src = Path(path) if path else default_root() / "robots" / "emb_e2_panda" / "scene.xml"
    res = import_file(src, out)                        # objects/, materials/, assets/, scenes/<name>.yaml
    print(json.dumps({k: res.to_json()[k] for k in ("objects", "robots", "scene")}, indent=2))
    for row in res.unmapped:                           # what the cards could not hold
        print(f"not imported: {row.item}: {row.reason}")
    for issue in res.issues:                           # card validation (errors would stop the assembler)
        print(f"{issue['level']}: {issue['card']} {issue['field']}: {issue['message']}")
    lib = CardLibrary([Path(out), default_root()])
    from alloy.scene.assembler import SceneAssembler
    from alloy.sim import Simulation

    sim = Simulation(SceneAssembler(lib).assemble(lib.scene(res.name), "cpu", world_count=1))
    for _ in range(5):
        sim.step_period()
    print(f"stepped {sim.report()['t']:.2f} s in Alloy")
    return res.to_json()


if __name__ == "__main__":
    main(*sys.argv[1:3])
