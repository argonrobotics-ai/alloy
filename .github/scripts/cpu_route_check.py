# SPDX-License-Identifier: Apache-2.0
"""CI check: which step the CPU route runs on this machine, and that a robot still moves right on it.

Builds AGENTS.md section 5's arm on the CPU, ramps the shoulder to 0.8 rad and reads the route's backend:

- ``--expect c``: the C step (``alloy/route/_mjc``, compiled on first use with this machine's C compiler) runs, and
  no fallback warning is given;
- ``--expect numpy``: the route falls back to its numpy step (``ALLOY_NO_EXT=1``, or no working C compiler, as on a
  Mac without Apple clang) with exactly one user-facing warning saying so, and the arm still follows its ramp.

Run it with ``python -I`` from outside the checkout, so the installed Alloy is the one imported. Prints one JSON line
(with ``log``: what Alloy logged, the C step's build included); exit 0 when the check passes, 1 otherwise.

Example::

    cd /tmp && ALLOY_NO_EXT=1 <venv>/bin/python -I <checkout>/.github/scripts/cpu_route_check.py --expect numpy
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

ARM = """<mujoco><compiler angle="radian"/><worldbody><body name="link1" pos="0 0 0.1">
  <joint name="shoulder" type="hinge" axis="0 1 0" range="-2 2" damping="2"/>
  <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02" mass="1"/>
  </body></worldbody>
  <actuator><position joint="shoulder" kp="200" kv="10"/></actuator></mujoco>"""
TARGET_RAD, TARGET_TOL = 0.8, 0.05
FALLBACK_WORDS = ("fast C step", "numpy step")   # the route's fallback warning (alloy.route.mjc_core)


class Kept(logging.Handler):
    """Keeps what Alloy logs: warnings (the fallback warning goes to the ``alloy.api`` logger) and the route's info
    lines (the C step's build)."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def lines(self, limit: int = 20, width: int = 400) -> list[str]:
        return [f"{r.levelname} {r.name}: {r.getMessage()}"[:width] for r in self.records[-limit:]]


def run_arm() -> dict[str, Any]:
    """The arm's shoulder after its ramp and hold, the run's health and each CPU-route solver's backend."""
    import alloy  # noqa: PLC0415 - imported after the warning handler is in place

    scene = alloy.Scene(dt=0.01)
    scene.add_ground()
    scene.add_robot("arm", mjcf=ARM, pos=(0, 0, 0.5))
    sim = alloy.build(scene, device="cpu")
    sim.run(1.0, targets={"shoulder": (0.0, TARGET_RAD)})
    sim.set_targets({"shoulder": TARGET_RAD})
    sim.step(50)
    reports = [s.route_report() for s in sim.engine.solvers if hasattr(s, "route_report")]
    return {"q_rad": round(float(sim.state(world=0)["arm/shoulder.q"]), 4), "health_ok": bool(sim.check()["ok"]),
            "backends": [r["backend"] for r in reports], "backend_reasons": [r["backend_reason"] for r in reports]}


def check(expect: str) -> dict[str, Any]:
    """Run the arm and judge the result against the expected backend (``c`` or ``numpy``)."""
    kept = Kept()
    logging.getLogger("alloy").addHandler(kept)
    logging.getLogger("alloy.route").setLevel(logging.INFO)
    row = {"expect": expect, **run_arm()}
    warned = [r.getMessage() for r in kept.records
              if r.levelno >= logging.WARNING and all(w in r.getMessage() for w in FALLBACK_WORDS)]
    row["fallback_warnings"] = warned
    follows = abs(row["q_rad"] - TARGET_RAD) < TARGET_TOL and row["health_ok"]
    on_expected = bool(row["backends"]) and all(b == expect for b in row["backends"])
    row["ok"] = follows and on_expected and len(warned) == (0 if expect == "c" else 1)
    if not row["ok"]:
        row["fix"] = ("the arm did not follow its ramp: read q_rad and health_ok" if not follows else
                      "no CPU-route solver ran: the arm went to another route" if not row["backends"] else
                      f"the route ran {row['backends']} ({row['backend_reasons']}) with "
                      f"{len(warned)} fallback warning(s); expected {expect}")
    row["log"] = kept.lines()
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--expect", choices=("c", "numpy"), required=True, help="the step the route must run")
    row = check(ap.parse_args(argv).expect)
    print(json.dumps(row))
    return 0 if row["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
