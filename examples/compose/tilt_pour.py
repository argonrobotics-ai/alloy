# SPDX-License-Identifier: Apache-2.0
"""Compose a tilting pour from a sentence and run it: water in a jar held over a bin, tipped about the jar's lip.

``alloy.compose.compose("pour 120 ml of water from a jar tilted at 25 deg/s to 110 degrees into a bin")`` places
the bin, stands 120 ml of water at rest in a kinematic copy of the jar with the jar's lip over the bin's opening, and
returns the tilt as a row of ``scene.soft["items"]``: the vessel, the pivot on its lip, the angular velocity, when
to start and for how long, and the bin's opening. ``alloy.compose.tilt(sim, row)`` starts it (``sim.drive``). This
script records the water and reports the share of it that ends below the bin's rim inside its opening.

Run (a GPU takes a few minutes; the CPU works but is slow):

    python examples/compose/tilt_pour.py out                  # out/scene/ (composed folder), out/pour.npz, out/pour.json
    python examples/compose/tilt_pour.py out --seconds 0.5    # a short look: 0.5 s from the start of the tilt
    python examples/compose/tilt_pour.py out --sentence "pour 80 ml of water from a jar tilted at 20 deg/s into a mug"

The same drive on hand-built boxes, without compose: ``examples/api/06_tilt_pour.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import alloy
import alloy.compose

SENTENCE = "pour 120 ml of water from a jar tilted at 25 deg/s to 110 degrees into a bin"


def compose_pour(sentence: str, out: Path, seed: int) -> tuple[alloy.Scene, dict]:
    """The composed scene (saved under ``out/scene``) and its tilt row."""
    result = alloy.compose.compose(sentence, seed=seed, check="load")
    if not result.ok:
        raise SystemExit(result.summary())
    result.save(out / "scene")
    plan = next(row for row in result.scenes[0].soft["items"] if "pivot" in row)
    return result.scene(0), plan


def share_inside(points: np.ndarray, plan: dict) -> float:
    """Share of the particles below the target's rim and within its opening (the inscribed circle, so a box-shaped
    bin's corners count as outside: a lower bound)."""
    centre = np.asarray(plan["opening_centre"])
    off = np.linalg.norm(points[:, :2] - centre[:2], axis=1)
    return float(((off <= plan["opening_radius_m"]) & (points[:, 2] <= centre[2])).mean())


def run(sentence: str, out: Path, seconds: float | None, every: int, device: str | None, seed: int) -> dict:
    """Compose, build, rest until the tilt's start, tilt, record; writes pour.npz and pour.json under ``out``."""
    scene, plan = compose_pour(sentence, out, seed)
    sim = alloy.build(scene, device=device)
    sim.run(plan["start_s"])
    alloy.compose.tilt(sim, plan)
    span = plan["duration_s"] + 1.0 if seconds is None else seconds
    traj = sim.run(span, record=[plan["liquid"], plan["vessel"]], every=every).world(0)
    traj.save(out / "pour.npz")
    points = traj[f"{plan['liquid']}.points"]
    summary = {"sentence": sentence, "vessel": plan["vessel"], "into": plan["into"], "water_ml": plan["volume_ml"],
               "tilt_deg_s": plan["rate_deg_s"], "simulated_s": round(sim.t, 3),
               "share_inside_start": round(share_inside(points[0], plan), 4),
               "share_inside_end": round(share_inside(points[-1], plan), 4), "check_ok": sim.check()["ok"]}
    (out / "pour.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="?", default="out_tilt_pour")
    ap.add_argument("--sentence", default=SENTENCE)
    ap.add_argument("--seconds", type=float, default=None, help="run time from the start of the tilt "
                                                                  "(default: the tilt plus 1 s)")
    ap.add_argument("--every", type=int, default=10, help="record every n control periods")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(run(args.sentence, out, args.seconds, args.every, args.device, args.seed)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
