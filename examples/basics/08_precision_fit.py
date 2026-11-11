# SPDX-License-Identifier: Apache-2.0
"""A sub-millimetre fit from the library (a 12 mm peg over a 12.2 mm bore) on the precision profile.

Shows: alloy.load(<library scene>, profile="precision"), precision pairs declared on object cards, sim.report()["gap"]
(measured pair penetration), health per world. Write your own precision cards as in docs/guide/cards.md.
Run: python examples/basics/08_precision_fit.py
"""
import json

import alloy

print("library scenes:", alloy.scenes())
sim = alloy.load("demo_peg_d12", worlds=2, profile="precision")    # raises SimulationError if a world goes INVALID
sim.run(0.5, record=[])
gap = sim.report()["gap"]
print(json.dumps({"names": sim.names()["bodies"], "pair_penetration_max_m": float(gap.pair_max_m.max()),
                  "severity": [w["severity"] for w in sim.health()["worlds"]], "ok": sim.check()["ok"]}))
