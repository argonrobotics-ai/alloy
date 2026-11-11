# SPDX-License-Identifier: Apache-2.0
"""task_tape_peel: a gantry gripper holds the 5 mm tab of a 25 mm wide tape bonded to a steel plate and peels the
whole strip, at 90° or 180°. This is Gym-Anything's open-carton seam tape (REPORT.md §2), with the carton as a rigid
plate in v1.

- **Tape.** The 3M 7818 card: 0.08 mm PET face stock on 310 acrylic. It is the only sourced tape data with both a
  90° and a 180° row (``alloy.plugins.bonds.adhesives``). The tape starts bonded where it lies.
- **Peel.** ``alloy.plugins.bonds``: a resolved front, Kendall's balance, G(theta) calibrated from both rows.
- **Robot.** The rig's MuJoCo gantry (``bonds.rig.PeelRig``), driven inside the step by ``bonds.driver``:
  - at 90° the grip stays above the front (a sliding fixture);
  - at 180° the grip folds the arm back 3 mm above the tape.
  The pull speed is 0.1 m/s, not ASTM D3330's 300 mm/min. The model has no rate dependence (a stated limit), so the
  speed only shortens the run.

**Success, per world:**
- every bond is released (the tape is off);
- the mean pull per width while the front is between 20 % and 80 % of the strip is within 10 % of the card's row at
  that angle.

``python -m alloy.tasks.tape_peel --angle 90 --worlds 4`` prints one JSON row.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np

from alloy.plugins.bonds import ROWS_7818, Bonds
from alloy.plugins.bonds.driver import GantryScript
from alloy.plugins.bonds.laws import PEEL
from alloy.plugins.bonds.rig import PeelRig, simulation

SPEED = 0.1                         # m/s grip speed
MODES = {90: "follow", 180: "pull180"}
TOL = 0.10                          # relative, mean F/b vs the card's row
STEADY = (0.2, 0.8)                 # fraction of the strip: the front's steady part


def rig(length: float = 0.100) -> PeelRig:
    return PeelRig(length=length, plate_half=(0.5 * length + 0.025, 0.03, 0.005))


def periods(angle: int, r: PeelRig) -> int:
    """Control periods to peel the whole strip: the front moves at v (90°) or v/2 (180°), plus a 30 % margin."""
    travel = r.length - r.tab
    seconds = travel / SPEED if angle == 90 else 0.1 + 2.0 * travel / SPEED
    return int(np.ceil(1.3 * seconds / r.control_dt))


def run(angle: int, worlds: int, device: str, graph: bool = False, length: float = 0.100,
        substrate: str = "stainless_steel") -> dict:
    """Peel ``worlds`` copies; per-world success and the mean F/b, wall ms per exchange after the first period."""
    r = rig(length)
    cfg = {**PEEL, "adhesives": (f"3m_7818/{substrate}",)}
    bonds, drv = Bonds(**cfg), GantryScript(mode=MODES[angle], speed=SPEED)
    sim = simulation(r, [drv, bonds], worlds, device, graph)
    drv.bind(sim, bonds)
    rows = {k: [] for k in ("peel_force", "width", "front", "bonded")}
    t_wall, steps = 0.0, 0
    for k in range(periods(angle, r)):
        t0, s0 = time.perf_counter(), int(sim.steps)
        sim.step_period()
        read = bonds.read()                              # syncs the device
        if k >= 1:                                       # period 0 compiles / captures
            t_wall += time.perf_counter() - t0
            steps += int(sim.steps) - s0
        for name, series in rows.items():
            series.append(read[name])
    row = _score({k: np.asarray(v) for k, v in rows.items()}, r, ROWS_7818[substrate][0 if angle == 90 else 1])
    return {"task": "task_tape_peel", "angle": angle, "substrate": substrate, "worlds": worlds, "device": device,
            "graph": graph, **row, "ms_per_exchange": 1e3 * t_wall / max(steps, 1)}


def _score(s: dict[str, np.ndarray], r: PeelRig, row: float) -> dict:
    """Per world: tape off, and the mean F/b over the steady part against ``row`` [N/m]."""
    off = s["bonded"][-1] <= 0.0
    lo, hi = STEADY[0] * r.length, STEADY[1] * r.length
    fb = np.full(s["front"].shape[1], np.nan)
    for w in range(fb.size):
        win = (s["front"][:, w] >= lo) & (s["front"][:, w] <= hi)
        if win.any():
            fb[w] = float(np.mean(s["peel_force"][win, w] / s["width"][win, w]))
    err = np.abs(fb - row) / row
    ok = off & (err <= TOL)
    return {"success": int(np.sum(ok)), "row_n_per_m": row, "f_per_b_mean": float(np.nanmean(fb)),
            "f_per_b_err_max": float(np.nanmax(err)) if np.isfinite(err).any() else float("nan"),
            "tape_off": int(np.sum(off))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--angle", type=int, choices=sorted(MODES), default=90)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--length", type=float, default=0.100)
    ap.add_argument("--substrate", default="stainless_steel")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    row = run(a.angle, a.worlds, a.device, a.graph, a.length, a.substrate)
    print(json.dumps(row))
    if a.out is not None:
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0 if row["success"] == a.worlds else 1


if __name__ == "__main__":
    raise SystemExit(main())
