# SPDX-License-Identifier: Apache-2.0
"""Per-scene opt-out from the SCR-DC1-2 rule: Newton for a chain attachment that fails the rule, with evidence.

The coordinator's ruling ([internal note]) allows the opt-out only with evidence, recorded in the route
report: a finite-step probe over at least 1,024 worlds × the scene's period count, with zero INVALID. A record holds
the rule's probe key (`newton_rule.probe_key`: the template model digest, the Newton candidate's MuJoCo options and
the rule version). It applies only while the scene still has that key, so any change to the model or to its options
drops it.

To make a record, run on a GPU:

    python -m alloy.scene.newton_evidence SCENE --periods N [--worlds 1024] [--device cuda:0]

This steps the scene's Newton candidate through the standard integration profile (health monitor on) and prints a
`Record(...)` literal to add to RECORDS, with the INVALID count it saw. N is the scene's period count: the episode
the scene is run for.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass

MIN_WORLDS = 1024


@dataclass(frozen=True)
class Record:
    scene: str              # library scene name (a label; the key is what matches)
    key: str                # newton_rule probe key
    worlds: int
    periods: int            # the scene's period count
    invalid: int            # worlds ever INVALID (sticky health)
    device: str
    commit: str             # Alloy commit the probe ran at
    rows: str               # where the run's rows and log are kept

    def valid(self) -> bool:
        return self.worlds >= MIN_WORLDS and self.periods >= 1 and self.invalid == 0

    def describe(self) -> str:
        return (f"{self.scene}: {self.worlds} worlds x {self.periods} periods on {self.device}, {self.invalid} "
                f"INVALID, Alloy {self.commit}; rows {self.rows}")


RECORDS: tuple[Record, ...] = ()


def find(key: str) -> Record | None:
    """The valid record for probe key `key`, if any."""
    return next((r for r in RECORDS if r.key == key and r.valid()), None)


def _key(built) -> str:
    m = next((re.search(r"probe key (\w+)", x) for x in built.report.entry_options if "probe key" in x), None)
    if m is None:
        raise SystemExit("this scene has no chain attachment on CG: the SCR-DC1-2 rule does not apply")
    return m.group(1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scene")
    ap.add_argument("--periods", type=int, required=True)
    ap.add_argument("--worlds", type=int, default=MIN_WORLDS)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)

    from alloy.cards.library import CardLibrary
    from alloy.integration.wiring import build_simulation
    from alloy.scene.assembler import SceneAssembler

    lib = CardLibrary()
    built = SceneAssembler(lib, newton_candidate=True).assemble(lib.scene(a.scene), a.device, world_count=a.worlds)
    sim = build_simulation(built, "standard")
    for _ in range(a.periods):
        sim.step_period()
    invalid = len(sim.report()["health"].invalid_worlds)
    commit = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(Record(a.scene, _key(built), a.worlds, a.periods, invalid, a.device, commit or "unknown", "<rows path>"))


if __name__ == "__main__":
    main()
