# SPDX-License-Identifier: Apache-2.0
"""Soft scene from a sentence: water filled into a mug and a towel draped over a jar, checked, saved and simulated;
``--cable`` adds a USB cable plugged into the jar.

The builder measures the mug's cavity and stands 200 ml of water on its inner floor (never dropped in) and lays the
towel flat a few millimetres above the jar so it falls over it. With ``--cable`` it also lays a 1 m cable in a
loose curve from the jar's side and welds its end there. The engine's router picks SPH for the water, VBD for the
towel and a chain for the cable. The result lists every soft item (amount, container, route) and what the builder
estimated.

Run (a GPU makes the checks and the run faster):

    python examples/compose/soft_scene.py                 # load checks, then a 0.02 s run on a GPU
    python examples/compose/soft_scene.py --cable         # with the cable: read the cost below first
    python examples/compose/soft_scene.py --seconds 0.2   # long enough for the towel to settle over the jar
    python examples/compose/soft_scene.py --check full    # plus a short run and the soft checks (GPU)
    python examples/compose/soft_scene.py --all-libraries # search every installed object library too

Cost, measured on one cloud GPU machine (an L4). The first run compiles kernels and took 8.5 min; the next took
21 s. The towel beside the water makes the engine take 256 physics steps per 20 ms control period, so the default
run is one period; on the CPU a period takes minutes on one core, so the run is skipped there unless ``--seconds``
is given.

The cable costs more, which is why it is an option. The first time a machine routes a cable of a given kind and
length, the engine runs its cable self-test on one CPU core (about 11 min on that machine; the result is cached
under ``~/.cache/alloy``, so later runs skip it), and beside the towel and the water the cable raises the step
count from 256 to 1,024 per control period. The first ``--cable`` run took 14 min with the kernels compiled, the
next 4 min.

It searches the objects shipped with Alloy (``Catalog(imports=False)``), so it builds the same scene on every
machine; ``--all-libraries`` also searches imported and fetched libraries, which can pick another mug or jar.

The same from the command line:
``python -m alloy.compose "a mug with 200 ml of water and a towel draped over a jar"``.
"""

from __future__ import annotations

import argparse
import json

import alloy.api
import alloy.compose
from alloy.compose.catalog import Catalog

SENTENCE = "a mug with 200 ml of water and a towel draped over a jar"
WITH_CABLE = "a mug with 200 ml of water, a towel draped over a jar and a usb cable plugged into the jar"
CABLE_FIRST_USE = ("the first cable of a kind and length on a machine waits for the engine's cable self-test: about "
                   "11 min on one CPU core of a cloud GPU machine, then cached")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", default="load", choices=("full", "load", "none"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--all-libraries", action="store_true", help="search imported and fetched libraries too")
    ap.add_argument("--seconds", type=float, default=None, help="seconds to run (default: 0.02 on a GPU, 0 on CPU)")
    ap.add_argument("--cable", action="store_true", help="add a USB cable plugged into the jar (slow the first time)")
    args = ap.parse_args()
    catalog = Catalog() if args.all_libraries else Catalog(imports=False)
    if args.cable:
        print(f"with the cable: {CABLE_FIRST_USE}", flush=True)
    result = alloy.compose.compose(WITH_CABLE if args.cable else SENTENCE, check=args.check, device=args.device,
                                   catalog=catalog)
    print(result.summary())
    scene = result.scenes[0]
    for row in scene.soft.get("items", []):
        print(json.dumps(row))
    print("estimated:", scene.soft.get("estimated"), "settles in", scene.soft.get("settle_s"), "s")
    result.save("soft_scene")                                  # scenes/scene.yaml + objects/<generated cards>
    sim = alloy.api.build(result.scene(0), worlds=1, device=args.device)
    seconds = args.seconds if args.seconds is not None else (0.0 if sim.device == "cpu" else 0.02)
    if seconds > 0:
        steps0 = sim.physics_steps
        sim.run(seconds)                                   # the towel starts to fall over the jar, the water settles
        print("health:", sim.health(), f"after {seconds} s ({sim.physics_steps - steps0} physics steps)")
    else:
        print("run skipped on the CPU (minutes per 20 ms simulated); pass --seconds 0.02 to run it anyway")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
