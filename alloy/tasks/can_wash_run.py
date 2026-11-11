# SPDX-License-Identifier: Apache-2.0
"""rinse-garbage-can, run: the scripted rinse in the can-wash bay (:mod:`alloy.tasks.can_wash_scene`), the measured
state (``can_wash_score.CanReadout``, from :mod:`alloy.tasks.can_wash_measure`) and each world's verdict.

Stage 1 (WP15's order: the scene, the gates and the score first). The hose valve is a kinematic body that the script
moves, as in SC9's flush-floor:
1. Out of the holster, the valve stands over the mouth and sprays the inner floor (``Rinse.floor`` rings, the
   innermost first) while the pool is shallow, then the inner wall from the rim down (``Rinse.inner`` heights). The
   jets reach the floor directly until it floods; then only their submerged shear does. The water pools in the cavity.
2. It circles the can and sprays its outer wall at ``Rinse.outer`` heights; that water runs off the foot.
3. It shuts and docks. The score's detergent, debris and shelf are read then (GA's evaluator reads them live), and the
   bay settles hands-off as GA's (``task.py:817-912``): 2 s, then 1 s at a time until quiet, at most ``Rinse.settle``
   s. A world is quiet when the hose is off, the can is still, the film's water and what left it change by at most
   1 g/s (SC9's ``quiet``) and the can's free water by at most 0.2 mL or 2 % per second. The gates are read then.
   The readout's ``held`` is the script's own hold (a hand's grip, ``Handles.tool_fixed``), not a state read: it can
   fail hands-free, never pass it, and without it the hands part is not judged (``can_wash_score``, AB-gym-3).
This run never tips the can, so the cavity's detergent stays and ``emptied_upright`` fails by design. Stage 2
(:mod:`alloy.tasks.can_wash_tip`) rinses, tips the pool out with WP8's ``PoseHand`` (the pool on the can as mass:
DC5's G2) and sprays the outside last.

Every period ORs MuJoCo Warp's capacity overflow bits of every substep into the fault gate (WP8's ``OverflowWatch``,
read by ``rinse_hand.period_overflow``; Alloy's CPU route writes the same bits per call, SCR-WP8-8).

Example::

    from alloy.tasks.can_wash_run import Rinse, run
    from alloy.tasks.can_wash_scene import CROP
    print(run(CROP, Rinse(speed=1.0, settle=3.0))["worlds"][0]["failed"])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from alloy.tasks.can_wash_books import books, ledgers, with_pending
from alloy.tasks.can_wash_measure import CanProbe, free_ml, soap_cells, soap_kg, tilt_deg, world_normal_z
from alloy.tasks.can_wash_score import CanReadout, row, verdict
from alloy.tasks.can_wash_scene import CAN, CAVITY, POUR, CanWash
from alloy.tasks.rinse_flush_checks import (SETTLE_MIN, SETTLE_STEP, Columns, quiet, quiet_books, shelf_grams,
                                            shelf_slots)
from alloy.tasks.rinse_flush_floor import EDGE, GPM, VALVE
from alloy.tasks.rinse_flush_policy import CONTROL_DT, NOZZLE, bodies, parked, place, q_starts, quat_down_to
from alloy.tasks.rinse_hand import OverflowWatch, mjw_solver, overflow_watch, period_overflow
from alloy.tasks.rinse_plates import adhered, gauges

FREE_QUIET_ML = 0.2                   # mL per second: GA's quiet item (task.py:843-844)
FREE_QUIET_REL = 0.02                 # or 2 % of its free water per second


@dataclass(frozen=True)
class Rinse:
    """The script: ring radii on the inner floor and aim heights on the inner and outer walls (body frame, m), the
    aim point's speed along them, the nozzle's height over the rim inside, and its standoff from the outer wall and
    lean below level there."""

    floor: tuple[float, ...] = (0.09, 0.17)       # the innermost first: sprayed through the pool it kept debris
    inner: tuple[float, ...] = (0.60, 0.45, 0.30, 0.12)
    outer: tuple[float, ...] = (0.62, 0.40, 0.18)
    speed: float = 0.25                   # m/s along the wall
    over_rim: float = 0.30                # m. Coverage: from 0.30 m the 6° cone reaches the floor's inner rings and
                                          # the band under the rim; from 0.10 m both kept debris (CROP: 3.8 mg). The
                                          # inside aims stop at 0.60 m: aimed at the 0.67 m rim band the cone's top
                                          # rays cleared the rim and reached the shelf (0.46 g in 1.6 s). From 0.30 m
                                          # the farthest rim-plane crossing is 240 mm against the 260 mm rim
    standoff: float = 0.30                # m
    lean: float = math.radians(30.0)
    settle: float = 30.0                  # s, the settle's cap (GA's)


def pose(tip: np.ndarray, aim: np.ndarray) -> tuple[float, ...]:
    """The valve's pose (x, y, z, qx, qy, qz, qw) with its nozzle at ``tip`` and its jets toward ``aim``."""
    j = (aim - tip) / np.linalg.norm(aim - tip)
    return (*(tip - NOZZLE * j), *quat_down_to(j))


def _circle(r: float, speed: float) -> np.ndarray:
    n = max(4, int(math.ceil(2.0 * math.pi * r / (speed * CONTROL_DT))))
    return np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)


def can_z(task: CanWash) -> float:
    """The can's body origin (its inner floor's centre) above the tile at the start (on the turntable when there is
    one)."""
    return task.stand + task.can.bottom + 1.0e-4


def plan(task: CanWash, rs: Rinse) -> list[tuple[float, ...]]:
    """The valve's poses, one per control period, open all along (module doc, steps 1-2)."""
    c, can = np.array([*task.at, can_z(task)]), task.can
    rim = float(can.radius(can.depth))
    inside = [(r, 0.0) for r in rs.floor] + [(float(can.radius(z)), z) for z in rs.inner]
    out = []
    for r, z in inside:
        for a in _circle(r, rs.speed):
            u = np.array([math.cos(a), math.sin(a), 0.0])
            tip = c - 0.5 * rim * u + [0.0, 0.0, can.depth + rs.over_rim]
            out.append(pose(tip, c + r * u + [0.0, 0.0, z]))
    for z in rs.outer:
        r = float(can.radius(z)) + can.wall
        for a in _circle(r, rs.speed):
            u = np.array([math.cos(a), math.sin(a), 0.0])
            aim = c + r * u + [0.0, 0.0, z]
            out.append(pose(aim + rs.standoff * (math.cos(rs.lean) * u + [0.0, 0.0, math.sin(rs.lean)]), aim))
    return out


@dataclass
class Handles:
    """What the readout needs from the built simulation."""

    flow: Any
    sim: Any
    probe: CanProbe
    can: np.ndarray                       # body index per world
    valve: np.ndarray
    cols: Columns                         # the valve's edge, the shelf's occluder slots (SC9's)
    pour: int                             # the cavity's pour edge
    overflow: np.ndarray                  # [W] bool, ORed every period
    solver: Any = None                    # the MuJoCo solver with the overflow word (None: no bits)
    grip: np.ndarray | None = None        # [W] a hand's grip on the can (stage 2); None: no hand
    tool_fixed: bool = False              # a tool fixed in a robot's hand, never released (the robot stage's valve)
    watch: Any = None                     # WP8's OverflowWatch (every substep); None: the period's last substep


def valve_distance(task: CanWash, valve_q: np.ndarray) -> np.ndarray:
    """``[W]`` the valve body's origin to the holster's docked centre (SC9's ``parked``) [m]."""
    return np.linalg.norm(np.asarray(valve_q)[:, :3] - np.asarray(task.bay.holster), axis=1)


def shelf_g(out: dict, n_floor: int, cols: Columns) -> np.ndarray:
    """``[W]`` water on the shelf [g], SC9's ``shelf_grams`` per world (``n_floor``: the cells before the shelf's)."""
    return np.array([shelf_grams(out, n_floor, cols, w) for w in range(out["h"].shape[0])])


def finite(r: CanReadout, overflow: np.ndarray) -> np.ndarray:
    """``[W]`` every measured value finite and no contact dropped (GA's ``finite`` covers every measure)."""
    ok = ~np.asarray(overflow, bool)
    for f in fields(r):
        v = getattr(r, f.name)
        if isinstance(v, np.ndarray) and f.name != "finite":
            ok &= np.isfinite(v.astype(float).reshape(len(ok), -1)).all(axis=1)
    return ok


def region_soap(task: CanWash, out: dict, area: np.ndarray, probe: CanProbe) -> dict[str, list[float]]:
    """Detergent on each region's cells (film + walls) [mg] per world."""
    cells = soap_cells(out, probe)[:, probe.cells] * area[probe.cells]
    return {k: (cells[:, m].sum(axis=1) * 1e6).tolist() for k, m in task.regions().items()}


def readout(task: CanWash, hd: Handles, ref: dict | None = None) -> tuple[CanReadout, dict, dict]:
    """The measured state of every world now (``CanReadout``), the books (``can_wash_books.books``, with
    ``start_kg``) and the flow's read. ``ref``: the start's books, which the ledgers close against (None: now)."""
    flow, t = hd.flow, hd.flow.tables
    out, area = with_pending(flow.read(), t.area_np), t.area_np
    q = hd.sim.state.body_q.numpy()
    can_q = q[hd.can]
    net_sp = flow.supply.fluid.species.numpy()
    soap = soap_kg(out, area, net_sp, hd.probe)
    now = {**books(out, net_sp, hd.probe.tank, hd.pour), "start_kg": np.stack([soap, out["soil_start_mass"]], 1)}
    ref = ref or now
    water, rel, kg = ledgers(out, net_sp, ref, hd.probe.tank, hd.pour)
    k, W = task.can_chart, len(hd.can)
    nz = world_normal_z(t.cnrm.numpy()[hd.probe.inner], can_q[:, 3:])
    pool = out["supply_amount"][:, hd.probe.tank]
    held = np.full(W, hd.tool_fixed) if hd.grip is None else (np.asarray(hd.grip) > 0.0) | hd.tool_fixed
    r = CanReadout(soap_kg=soap, soap_gauge=gauges(flow, soap_cells(out, hd.probe), k),
                   soil_gauge=gauges(flow, adhered(flow), k), free_ml=free_ml(out["h"], nz, area, pool, hd.probe),
                   tilt_deg=tilt_deg(can_q[:, 3:]), valve_m=valve_distance(task, q[hd.valve]),
                   shelf_g=shelf_g(out, t.cells_of(k).stop, hd.cols),
                   hose_gpm=np.abs(out["supply_discharge"][:, hd.cols.outlet]) / GPM, held=held,
                   water_rel=water, species_rel=rel, species_kg=kg, finite=np.ones(W, bool),
                   area_m2=float(area[t.cells_of(k)].sum()))
    r.finite = finite(r, hd.overflow)
    return r, now, out


def make_probe(task: CanWash, flow: Any) -> CanProbe:
    cells = np.arange(flow.tables.cell_count)[flow.tables.cells_of(task.can_chart)]
    net = flow.supply.net
    return CanProbe(cells=cells, inner=cells[task.regions()["inner"]], tank=net.node_index(CAVITY), soap=0,
                    soap_net=0)


def step_watched(hd: Handles) -> None:
    hd.sim.step_period()
    bits = period_overflow(hd.watch, hd.solver)
    if bits is not None:
        hd.overflow |= bits


def settle_steps(cap: float) -> list[float]:
    """GA's settle schedule [s]: ``SETTLE_MIN`` hands-off, then ``SETTLE_STEP`` at a time, each a quiet check, the
    last one cut at ``cap``."""
    out, t = [], 0.0
    while not out or t < cap - 1e-9:
        out.append(max(min(SETTLE_MIN if not out else SETTLE_STEP, cap - t), 0.0))
        t += out[-1]
    return out


def quiet_now(prev: tuple[np.ndarray, np.ndarray], now: tuple[np.ndarray, np.ndarray], hose_gpm: np.ndarray,
              speed: np.ndarray, step: float) -> np.ndarray:
    """``[W]`` quiet over the last ``step`` s: SC9's ``quiet`` (hose off, the can still within ``QUIET_V``, the film's
    water and what left it within 1 g/s) and the can's free water within 0.2 mL or 2 % per second (GA's
    ``task.py:843-844``). ``prev``, ``now``: (``quiet_books``, free water [mL])."""
    free = np.abs(now[1] - prev[1]) <= np.maximum(FREE_QUIET_ML, FREE_QUIET_REL * now[1]) * step   # GA: of now
    return quiet(prev[0], now[0], hose_gpm, speed) & free


def settle(task: CanWash, hd: Handles, ref: dict, cap: float) -> tuple[list[CanReadout], list[float]]:
    """GA's settle (module doc, step 3): each world's readout when it is quiet (``quiet_now``) after a step of
    ``settle_steps``, or at ``cap`` s, and when."""
    W = len(hd.can)
    rows: list[CanReadout | None] = [None] * W
    at = [0.0] * W
    t, prev = 0.0, None
    for step in settle_steps(cap):
        for _ in range(max(1, int(round(step / CONTROL_DT)))):
            step_watched(hd)
        t += step
        r, _, out = readout(task, hd, ref)
        now = (quiet_books(out), r.free_ml.copy())
        done = np.full(W, t >= cap - 1e-9)
        if prev is not None:
            speed = np.linalg.norm(hd.sim.state.body_qd.numpy()[hd.can][:, :3], axis=1)
            done |= quiet_now(prev, now, r.hose_gpm, speed, step)
        for w in np.flatnonzero(done):
            if rows[w] is None:
                rows[w], at[w] = r, round(t, 3)
        prev = now
        if all(x is not None for x in rows):
            break
    return rows, at


def run(task: CanWash | None = None, rs: Rinse | None = None, worlds: int = 1, seed: int = 0,
        device: str = "cpu") -> dict:
    """The scripted rinse (module doc) in ``worlds`` worlds; each world's verdict and its start, live and settled
    readouts."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415 - simulation stack only when run

    task, rs = task or CanWash(), rs or Rinse()
    flow = task.make_flow()
    shut, made = np.zeros(worlds, np.float32), {}

    def init(sim: Any) -> None:
        flow.set_input(EDGE, shut)
        made.update(task.start(flow, worlds, seed))

    sim = build_simulation(task.scene(init, worlds), [flow, OverflowWatch()], device)
    net = flow.supply.net
    cols = Columns(outlet=net.edge_index(EDGE), shelf=shelf_slots(sim.model, flow.supply.occluder_shapes))
    hd = Handles(flow, sim, make_probe(task, flow), bodies(sim.model, CAN), bodies(sim.model, VALVE), cols,
                 net.edge_index(POUR), np.zeros(worlds, bool), mjw_solver(sim), watch=overflow_watch(sim))
    r0, ref, out0 = readout(task, hd)
    soap0 = region_soap(task, out0, flow.tables.area_np, hd.probe)
    starts = q_starts(sim.model, hd.valve)
    poses = plan(task, rs)
    flow.set_input(EDGE, np.ones(worlds, np.float32))
    for p in poses:
        place(sim, starts, p)
        step_watched(hd)
    flow.set_input(EDGE, shut)
    place(sim, starts, parked(task.bay))
    step_watched(hd)
    live, _, _ = readout(task, hd, ref)
    ends, at = settle(task, hd, ref, rs.settle)
    out = flow.read()
    rows = []
    for w in range(worlds):
        v = verdict(r0, ends[w], w, live=live)
        rows.append(row(v, at[w]))
    return {"worlds": rows, "start": as_row(r0), "live": as_row(live), "end": [as_row(e) for e in ends],
            "spray_s": round(len(poses) * CONTROL_DT, 2), "seed": seed,
            "strength": np.asarray(made["strength"]).tolist(),
            "pool_L": (out["supply_amount"][:, hd.probe.tank] * 1e3).tolist(),
            "soil_mg": [(out["soil_start_mass"] * 1e6).tolist(), (out["soil_mass"] * 1e6).tolist()],
            "soap_mg_regions": {"start": soap0, "end": region_soap(task, out, flow.tables.area_np, hd.probe)},
            "overflow": hd.overflow.tolist(), "solver": type(hd.solver).__name__ if hd.solver else None,
            "watch": hd.watch is not None and hd.watch.bits is not None}


def as_row(r: CanReadout) -> dict:
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(r).items()}


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m alloy.tasks.can_wash_run [--crop] [--worlds W] [--device D] [--speed V] [--settle S]
    [--seed N] --out f.json``."""
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415
    import time  # noqa: PLC0415

    from alloy.tasks.can_wash_scene import CROP  # noqa: PLC0415

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--crop", action="store_true")
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--speed", type=float, default=Rinse.speed)
    ap.add_argument("--settle", type=float, default=Rinse.settle)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    res = run(CROP if a.crop else None, Rinse(speed=a.speed, settle=a.settle), a.worlds, a.seed, a.device)
    res.update(wall_s=round(time.perf_counter() - t0, 1), device=a.device, crop=a.crop, args=vars(a))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"rinse-garbage-can: {sum(r['success'] for r in res['worlds'])}/{a.worlds} worlds pass, wall "
          f"{res['wall_s']} s -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FREE_QUIET_ML", "FREE_QUIET_REL", "Handles", "Rinse", "as_row", "can_z", "finite", "main", "make_probe",
           "plan", "pose", "quiet_now", "readout", "region_soap", "run", "settle", "settle_steps", "shelf_g",
           "step_watched", "valve_distance"]
