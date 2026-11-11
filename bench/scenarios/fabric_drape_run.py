# SPDX-License-Identifier: Apache-2.0
"""D03M runner (SCR-WP2-14): build the fabric rigs, drape them, score every rig, append one JSON row per rig.

    python -m bench.scenarios.fabric_drape_run [--pack bench/scenarios/packs/fabric_drape_petrak2021.json]
        --out DIR [--rigs L,T,S,L:geo] [--fabrics F1,F4,F9] [--worlds 8] [--device cuda:0] [--accuracy super]
        [--iterations N] [--edge 0.008] [--mesh r6|grid] [--seconds 4.5] [--seed 0] [--label NAME]

The build, the gravity ramp and the scoring are Alloy's (``alloy.scene.twins.fabric_run``, shared with the R6 law's
production-route twin); ``--iterations`` is its convergence probe (iterations merged into the VBD entry's kwargs).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from alloy.scene.twins.fabric_run import RunOpts, run_rigs
from bench.scenarios import fabric_drape as FD

log = logging.getLogger("bench.d03m")
PACK = FD.PACKS / "fabric_drape_petrak2021.json"


def run(pack: dict, a: argparse.Namespace) -> list[dict]:
    """One build of the rigs the arguments name, on the mesh they name: one row per rig (``fabric_run.run_rigs``)."""
    rigs = FD.rigs_for(pack, a.rigs, a.fabrics)
    radius = 0.5 * pack["rig"]["specimen_diameter_m"]
    mesh = FD.specimen_mesh(radius, a.edge, a.seed, a.mesh)
    opts = RunOpts(out=a.out, worlds=a.worlds, device=a.device, accuracy=a.accuracy, iterations=a.iterations,
                   edge=a.edge, seconds=a.seconds, seed=a.seed, mesh_kind=a.mesh, clamp=not a.no_clamp)
    return run_rigs(pack, rigs, opts, mesh)[0]


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", type=Path, default=PACK)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rigs", default=FD.HEADLINE)
    ap.add_argument("--fabrics", default="")
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--accuracy", default="super", choices=("default", "super"))
    ap.add_argument("--iterations", type=int, default=0, help="VBD iterations above the setting's (0: as set)")
    ap.add_argument("--edge", type=float, default=0.008)
    ap.add_argument("--mesh", default="r6", choices=FD.MESHES)
    ap.add_argument("--seconds", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-clamp", action="store_true", help="rest the specimen on the support (a diagnostic)")
    ap.add_argument("--label", default="")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    a = parse(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = run(json.loads(a.pack.read_text()), a)
    with (a.out / "rows.jsonl").open("a") as f:
        for row in rows:
            line = json.dumps({"label": a.label, **row}, default=str)
            f.write(line + "\n")
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
