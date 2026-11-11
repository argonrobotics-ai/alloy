# SPDX-License-Identifier: Apache-2.0
"""task rinse_drain: Gym-Anything's drain-wash-compartment, the first surface-flow proof for rinsing.

"Drain the wash compartment (spent wash water), close its waste, leave rinse and sanitizer as they are"
(``task_set/catalogue.py:51-54``). The scene is the three-bowl pot sink (``rinse_sink.PotSink``): each bowl an open
tank filled to 10 in (about 70 L), a basket drain per bowl opened by a lever waste, an overflow at each rim onto the
floor, and a swing faucet (off). The plumbing (``rinse_levers.SinkPlumbing``) opens each drain by its lever's angle;
a privileged scripted hand (``rinse_levers.LeverHand``) works the levers. No robot: like ``sand_plate``, the proof
is the physics and the verdict, not the arm.

The policy, per world: twist the wash lever open (a fraction ``open_frac`` of its stroke, drawn per world), let go
(the lever holds by its joint friction), wait until the wash bowl holds ``close_at_l``, twist the lever closed against
its stop, let go. Then the verdict settles at least 2 s, and on until no water moves (at most 30 s), as Gym-Anything's
does (``task.py:849-912``).

Gates (``task_set/evaluator.py:97-108`` and the common gates, ``task.py:677-692``): ``drained`` (wash <= 0.5 L),
``waste_closed`` (every lever within 5° of closed), ``others_kept`` (rinse and sanitizer each change and drain at most
1 L), ``floor_dry`` (<= 50 mL on the floor), ``water_off_and_hands_free``, ``intact_and_not_dropped`` (no ware here),
``no_simulation_fault`` (finite, water and surfactant ledgers closed to 1e-9). Score: ``task_set/scoring.py:55-66``.
Honest verdict (audit AB-gym-3): ``floor_dry`` judges the overflow ledger and not the rest of Gym-Anything's floor
water; ``water_off_and_hands_free`` judges the outlet and can only fail on the script's grip flag; the empty
``intact_and_not_dropped`` is not judged. ``Verdict.summary()`` says how many criteria are judged.

Run: ``python -m alloy.tasks.rinse_drain [--worlds W] [--device cuda:0] [--fill M] [--json out.json]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.tasks.rinse_levers import LeverHand, SinkPlumbing, lever_angles
from alloy.tasks.rinse_score import (FLOOR_WATER_ML, OUTLET_OFF_GPM, Verdict, clip, excess, finish, finite,
                                     hands_gate, ledger_closed, partly)
from alloy.tasks.rinse_sink import BOWLS, GPM, LEVER_RANGE, PotSink, drain_edge, lever_joint, overflow_edge

log = logging.getLogger("alloy.tasks.rinse_drain")

OPEN, WAIT, CLOSE, DONE = 0, 1, 2, 3
OVERDRIVE = 0.05                      # rad: aim this far past a stop (kp·0.05 = 1 N·m beats the 0.30 N·m friction)
CLOSED_TOL = math.radians(0.5)        # rad: the hand lets go once the lever is this close to its stop
HOLD_S = 0.2                          # s at the stop before letting go
STILL = math.radians(0.05)            # rad per period: the hand lets go of a lever that has stopped turning
WASH_MAX_L = 0.5
OTHERS_MAX_L = 1.0
CLOSED_DEG = 5.0


@dataclass(frozen=True)
class DrainWash:
    """Task parameters. ``open_frac`` is the range per-world lever openings are drawn from (seeded), unless
    ``openings`` gives one per world. Below a full opening the lever rests on its joint friction, not on its stop."""

    sink: PotSink = field(default_factory=PotSink)
    open_frac: tuple[float, float] = (0.6, 1.0)
    openings: tuple[float, ...] | None = None     # per-world lever openings (overrides the draws from open_frac)
    close_at_l: float = 0.25          # L: the policy closes the waste at this wash volume
    seed: int = 0
    hand: dict[str, Any] = field(default_factory=dict)
    control_dt: float = 0.05          # s
    dt: float = 3.2e-3                # s, target exchange step
    settle_min_s: float = 2.0
    settle_max_s: float = 30.0
    max_s: float = 400.0              # s: stop acting after this and judge what is there

    def plugins(self) -> tuple[SinkPlumbing, LeverHand]:
        s = self.sink
        plumb = SinkPlumbing(nodes=s.nodes(), edges=s.edges(), openings=s.openings(),
                             levers=tuple((drain_edge(b), lever_joint(b)) for b in BOWLS), q_open=LEVER_RANGE[1])
        return plumb, LeverHand(levers=tuple(lever_joint(b) for b in BOWLS), **self.hand)

    def scene(self, worlds: int) -> TestScene:
        return TestScene(build=self.sink.build_levers, worlds=worlds, ground=False, control_dt=self.control_dt,
                         dt=self.dt)


@dataclass
class Readout:
    """Host snapshot of every world (``[W]`` or ``[W, 3]`` in BOWLS order)."""

    t: float
    volume_l: np.ndarray
    drained_l: np.ndarray
    floor_ml: np.ndarray
    waste_rad: np.ndarray
    faucet_gpm: np.ndarray
    flow_gpm: np.ndarray              # the largest |discharge| of any edge (the settle rule's "water moving")
    held: np.ndarray
    water_rel: np.ndarray
    species_kg: np.ndarray
    species_rel: np.ndarray
    finite: np.ndarray


def read(sim: Any, plumb: SinkPlumbing, hand: LeverHand, t: float) -> Readout:
    f, net = plumb.fluid, plumb.fluid.net
    bowls = [net.node_index(b) for b in BOWLS]
    vol = f.amount(f.pressure.numpy(), f.volume.numpy())
    ledger, flow = f.ledger.numpy(), f.discharge.numpy()
    v0 = np.asarray([nd.volume for nd in net.nodes])
    water = vol.sum(1) - v0.sum() - ledger[:, :f.m].sum(1)
    k = net.species
    s0 = np.asarray([nd.volume * (nd.species[0] if nd.species else 0.0) for nd in net.nodes]).sum()
    s_now = f.species.numpy()[:, :len(net.nodes), :k].sum((1, 2)) if k else np.zeros(len(vol))
    s_err = s_now - s0 - (f.species_ledger.numpy()[:, :f.m, :k].sum((1, 2)) if k else 0.0)
    angles = lever_angles(sim.model, sim.state.body_q.numpy(), hand.levers.numpy())
    return Readout(t=t, volume_l=vol[:, bowls] * 1e3,
                   drained_l=-ledger[:, [net.edge_index(drain_edge(b)) for b in BOWLS]] * 1e3,
                   floor_ml=-ledger[:, [net.edge_index(overflow_edge(b)) for b in BOWLS]].sum(1) * 1e6,
                   waste_rad=angles, faucet_gpm=np.abs(flow[:, net.edge_index("faucet")]) / GPM,
                   flow_gpm=np.abs(flow[:, :f.m]).max(1) / GPM, held=hand.grip.numpy().max(1) > 0.0,
                   water_rel=np.abs(water) / v0.sum(), species_kg=np.abs(s_err),
                   species_rel=np.abs(s_err) / max(s0, 1e-30),
                   finite=np.isfinite(vol).all(1) & np.isfinite(angles).all(1))


class LeverPolicy:
    """Per-world phases of the scripted hand on the wash lever (index 0); the other levers are never touched."""

    def __init__(self, task: DrainWash, worlds: int) -> None:
        rng = np.random.default_rng(task.seed)
        self.task = task
        self.frac = (rng.uniform(*task.open_frac, size=worlds) if task.openings is None
                     else np.broadcast_to(np.asarray(task.openings, np.float64), (worlds,)).copy())
        self.phase = np.full(worlds, OPEN)
        self.at_stop = np.zeros(worlds)
        self.q_prev = np.zeros(worlds)
        nan = np.full(worlds, math.nan)
        self.wait = {k: nan.copy() for k in ("t0", "t1", "v0", "v1", "q0", "q1")}   # the hands-off drain window

    def _mark(self, which: str, mask: np.ndarray, r: Readout) -> None:
        self.wait[f"t{which}"][mask] = r.t
        self.wait[f"v{which}"][mask] = r.volume_l[mask, 0]
        self.wait[f"q{which}"][mask] = r.waste_rad[mask, 0]

    def done(self) -> bool:
        return bool(np.all(self.phase == DONE))

    def act(self, r: Readout, hand: LeverHand, dt: float) -> None:
        q, full = r.waste_rad[:, 0], LEVER_RANGE[1]
        aim = np.where(self.frac >= 1.0, full + OVERDRIVE, self.frac * full)
        still = np.abs(q - self.q_prev) < STILL
        self.q_prev = q.copy()
        opened = (self.phase == OPEN) & (q >= np.minimum(aim, full) - math.radians(2.0)) & still
        self._mark("0", opened, r)
        self.phase[opened] = WAIT
        drained = (self.phase == WAIT) & (r.volume_l[:, 0] <= self.task.close_at_l)
        self._mark("1", drained, r)
        self.phase[drained] = CLOSE
        closing = self.phase == CLOSE
        self.at_stop = np.where(closing & (q <= CLOSED_TOL), self.at_stop + dt, 0.0)
        self.phase[closing & (self.at_stop >= HOLD_S)] = DONE
        target = np.zeros((len(q), len(BOWLS)), np.float32)
        grip = np.zeros_like(target)
        target[:, 0] = np.where(self.phase == OPEN, aim, -OVERDRIVE)
        grip[:, 0] = np.isin(self.phase, (OPEN, CLOSE)).astype(np.float32)
        hand.set_input("target", target)
        hand.set_input("grip", grip)


def gates(start: Readout, r: Readout, w: int) -> dict[str, tuple[bool | None, Any]]:
    """World ``w``'s gates (``task_set/evaluator.py:97-108``, ``task.py:677-692``)."""
    changes = {b: (float(r.volume_l[w, k] - start.volume_l[w, k]), float(r.drained_l[w, k] - start.drained_l[w, k]))
               for k, b in enumerate(BOWLS) if b != "wash"}
    closed = bool(r.finite[w]) and ledger_closed(float(r.water_rel[w]), float(r.species_rel[w]),
                                                 float(r.species_kg[w]))
    outlet_off = bool(r.faucet_gpm[w] <= OUTLET_OFF_GPM)
    return {
        "drained": (bool(r.volume_l[w, 0] <= WASH_MAX_L), float(r.volume_l[w, 0])),
        "waste_closed": (bool(np.all(np.abs(r.waste_rad[w]) <= math.radians(CLOSED_DEG))),
                         [round(math.degrees(a), 3) for a in r.waste_rad[w]]),
        "others_kept": (all(abs(dv) <= OTHERS_MAX_L and dd <= OTHERS_MAX_L for dv, dd in changes.values()), changes),
        "floor_dry": (partly(bool(r.floor_ml[w] <= FLOOR_WATER_ML), None),
                      {"overflow_ledger_mL": float(r.floor_ml[w]),
                       "floor_sinks_and_film_charts": "not judged: the port reads the bowls' overflow ledger only; "
                                                      "Gym-Anything sums floor sinks and floor-chart films"}),
        "water_off_and_hands_free": hands_gate(outlet_off, bool(r.held[w]), faucet_gpm=float(r.faucet_gpm[w])),
        "intact_and_not_dropped": (None, "vacuous in Gym-Anything too: no ware in this task, nothing to break or "
                                         "drop"),
        "no_simulation_fault": (closed, {"water_rel": float(r.water_rel[w]), "species_kg": float(r.species_kg[w])}),
    }


def verdict(start: Readout, live: Readout, r: Readout, w: int) -> Verdict:
    """World ``w``'s gates on the settled state ``r`` and progress on the ``live`` state before the settle
    (``task_set/evaluator.py:110-118, 130-133``: the settle may drain passively), score per
    ``task_set/scoring.py:55-66, 126-139``."""
    v = Verdict(criteria=gates(start, r, w))
    v0, r = float(start.volume_l[w, 0]), live
    d = clip((v0 - float(r.volume_l[w, 0])) / (v0 - WASH_MAX_L))
    released = bool(r.faucet_gpm[w] <= OUTLET_OFF_GPM and not r.held[w])        # progress: the outlet and the grip
    k = float(v.criteria["waste_closed"][0]) * float(released)
    lost = sum(max(0.0, float(start.volume_l[w, i] - r.volume_l[w, i]),
                   float(r.drained_l[w, i] - start.drained_l[w, i])) for i in (1, 2))
    units = lost / v0 + excess(float(r.floor_ml[w]), FLOOR_WATER_ML)
    v.terms = {"drained": d, "closed_released": k, "other_water_lost_L": lost}
    ok = bool(r.finite[w]) and v.criteria["no_simulation_fault"][0] and finite(v.terms)
    return finish(v, (d + d * k) / 2.0 - units, ok)


@dataclass
class DrainResult:
    worlds: int
    success: list[bool]
    score: list[float | None]
    failed: list[list[str]]
    wash_l: list[float]
    others_change_l: list[float]          # rinse or sanitizer: the change of largest magnitude [L]
    waste_deg: list[list[float]]
    floor_ml: list[float]
    open_frac: list[float]                # the policy's aim, fraction of the lever's stroke
    open_measured: list[float]            # the drain opening while the hand was off (mean of let-go and close)
    wait_s: list[float]                   # the hands-off drain window: from let-go to the close
    wait_l: list[list[float]]             # wash volume at its start and end [L]
    closed_at_s: list[float]
    seconds: float
    water_rel_max: float
    wall_s: float
    judged: list[list[int]] = field(default_factory=list)      # per world: [judged, all] Gym-Anything criteria
    not_judged: list[list[str]] = field(default_factory=list)  # per world: the criteria this port does not judge
    summary: list[str] = field(default_factory=list)           # per world: "k of n Gym-Anything criteria judged ..."


def _settle(sim: Any, plumb: SinkPlumbing, hand: LeverHand, task: DrainWash, t: float) -> tuple[Readout, float]:
    """Run the verdict's settle: at least ``settle_min_s``, then 1 s at a time until no edge moves water."""
    waited = 0.0
    while True:
        n = max(int(round((task.settle_min_s if waited == 0.0 else 1.0) / task.control_dt)), 1)
        for _ in range(n):
            sim.step_period()
        t += n * task.control_dt
        waited += n * task.control_dt
        r = read(sim, plumb, hand, t)
        if waited >= task.settle_max_s or (waited >= task.settle_min_s and np.all(r.flow_gpm <= OUTLET_OFF_GPM)):
            return r, t


def run(task: DrainWash | None = None, *, worlds: int = 1, device: str = "cpu",
        graph: bool | None = None) -> DrainResult:
    """Run the task in ``worlds`` worlds; returns every world's verdict. ``graph`` (default: on CUDA only) captures
    each control period; the CPU default route (MuJoCo C) reads host arrays and is not captured."""
    graph = device.startswith("cuda") if graph is None else graph
    task = task or DrainWash()
    plumb, hand = task.plugins()
    wall = time.perf_counter()
    sim = build_simulation(task.scene(worlds), [plumb, hand], device, graph=graph)
    t = 0.0
    start = r = read(sim, plumb, hand, t)
    policy = LeverPolicy(task, worlds)
    closed_at = np.full(worlds, math.nan)
    while not policy.done() and t < task.max_s:
        policy.act(r, hand, task.control_dt)
        closed_at = np.where(np.isnan(closed_at) & (policy.phase == DONE), t, closed_at)
        sim.step_period()
        t += task.control_dt
        r = read(sim, plumb, hand, t)
    policy.act(r, hand, task.control_dt)
    live = r
    r, t = _settle(sim, plumb, hand, task, t)
    vs = [verdict(start, live, r, w) for w in range(worlds)]
    change = r.volume_l[:, 1:] - start.volume_l[:, 1:]
    wt = policy.wait
    res = DrainResult(worlds=worlds, success=[v.success for v in vs], score=[v.score for v in vs],
                      failed=[v.failed() for v in vs], judged=[list(v.judged) for v in vs],
                      summary=[v.summary() for v in vs], not_judged=[v.not_judged() for v in vs],
                      wash_l=r.volume_l[:, 0].tolist(),
                      others_change_l=np.take_along_axis(change, np.abs(change).argmax(1)[:, None], 1)[:, 0].tolist(),
                      waste_deg=np.degrees(r.waste_rad).round(3).tolist(), floor_ml=r.floor_ml.tolist(),
                      open_frac=policy.frac.tolist(),
                      open_measured=((wt["q0"] + wt["q1"]) / (2.0 * LEVER_RANGE[1])).tolist(),
                      wait_s=(wt["t1"] - wt["t0"]).tolist(), wait_l=np.stack([wt["v0"], wt["v1"]], 1).tolist(),
                      closed_at_s=closed_at.tolist(), seconds=t, water_rel_max=float(r.water_rel.max()),
                      wall_s=time.perf_counter() - wall)
    log.info("rinse_drain: %d/%d worlds pass, %.1f s simulated in %.1f s wall", sum(res.success), worlds, t,
             res.wall_s)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--fill", type=float, default=None, help="start depth of the wash bowl [m] (default 10 in)")
    ap.add_argument("--open-frac", type=float, nargs=2, default=DrainWash().open_frac)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sink = PotSink() if args.fill is None else PotSink(fill=(args.fill, *PotSink().fill[1:]))
    res = run(DrainWash(sink=sink, open_frac=tuple(args.open_frac)), worlds=args.worlds, device=args.device)
    text = json.dumps(asdict(res), indent=1)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text)
    print(text)
    return 0 if all(res.success) else 1


if __name__ == "__main__":
    raise SystemExit(main())
