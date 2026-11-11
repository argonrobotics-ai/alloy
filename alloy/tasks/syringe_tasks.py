# SPDX-License-Identifier: Apache-2.0
"""Gym-Anything's syringe tasks as finger-rig analogues, scored on GA's own fluid criteria (SCR-WP8-11 Part A).

GA's syringe_handling family failed only on air in a liquid chamber (RESCORE_1108 gas_air), which the ``syringe``
plug-in now has. Each task here is a scripted rig (``syringe_rig``): the barrel tilts, a hand's grip pushes and pulls
the plunger (the joint's drive), and needles are ports the script opens (to the air, a vial, the tissue, a vein). Each
row reports GA's fluid criteria per world. Uncapping, Luer screwing, puncture and disposal are rigid manipulation,
outside this analogue, and so is GA's ordering of saline ahead of medication in the lock's line (volumes only, no
species).

Sources: barrels BD 1/3/10 mL at 4.70/8.59/14.43 mm bore (GA's report, makers' bore tables); needle bores ISO 9626
regular wall (18 G 0.838, 25 G 0.260 mm); the 20 G cannula's 0.80 mm bore and 32 mm length (a typical 20 G IV
catheter; an estimate); the lock's line of 1.1 mL (GA) as 2.0 mm bore (an estimate, so 0.35 m); contrast at 11.6 mPa·s
(GA); the 10R vial (ISO 8362-1) as 13.5 mL brimful, 11 mm inner radius; a vein at 1.1 kPa over the air (8 mmHg, an
estimate); hub dead space 0.07 mL (an estimate). Steps: the largest the plug-in's ``check_dt`` allows at control_dt /
2^k. Run: ``python -m alloy.tasks.syringe_tasks --task set-dose --worlds 2`` (one JSON row).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.plugins.fluid import P_ATM, Edge
from alloy.plugins.fluid.mixed import MU_AIR
from alloy.plugins.harness.scene import TestScene, build_simulation
from alloy.plugins.library.syringe import Syringe, Syringes, Vial
from alloy.tasks.syringe_rig import Drive, Rig, build

BORE = {1: 4.70e-3, 3: 8.59e-3, 10: 14.43e-3}
STROKE = {1: 0.062, 3: 0.057, 10: 0.066}           # m: the capacity's travel + 5 mm
DEAD = 7.0e-8
WATER, CONTRAST = 1.0e-3, 11.6e-3                   # Pa·s
N18, N25 = ((0.838e-3, 0.038),), ((0.260e-3, 0.016),)
LOCK = ((2.0e-3, 0.35), (0.80e-3, 0.032))           # the lock's line, then the 20 G cannula
TIP = ((2.0e-3, 0.008),)                            # a bare Luer tip
VEIN = P_ATM + 1.1e3
CONTROL_DT = 0.02
REACH = 1.0e-3                                      # m: how far past the plunger's stops a finger's target may go


def port(name: str, a: str, segments: tuple, b: str | None = None, viscosity: float = WATER, **kw) -> Edge:
    """A port through tubes in series (``segments``: (bore, length)), Poiseuille for liquid and air."""
    r = sum(128.0 * length / (math.pi * d ** 4) for d, length in segments)        # m⁻³: R / μ
    return Edge(name, "port", a, b, conductance=1.0 / (r * viscosity), gas_conductance=1.0 / (r * MU_AIR), **kw)


def column(bore: float, segments: tuple, density: float = 1000.0) -> float:
    """The plunger's added mass [kg] from the liquid in the tubes in series: Σ ρ·L·A²/a."""
    big = math.pi * bore ** 2 / 4.0
    return sum(density * length * big ** 2 / (math.pi * d ** 2 / 4.0) for d, length in segments)


def syringe(label: str, size: int, liquid: float, air: float = 0.0) -> Syringe:
    """A barrel of ``size`` mL holding ``liquid`` (dead space included) and ``air`` m³, the plunger where they fit."""
    area = math.pi * BORE[size] ** 2 / 4.0
    x0 = max(liquid + air - DEAD, 0.0) / area
    return Syringe(label, f"{label}_plunger", bore=BORE[size], stroke=STROKE[size], x0=x0, dead=DEAD, liquid=liquid)


@dataclass
class Phase:
    """One scripted step: the finger's speed [m/s along the barrel, + pulls the plunger out], the tilt target, port
    openings and vial attitudes set at its start; it ends when ``until(task)`` holds (world 0) or after ``max_s``."""

    name: str
    rig: str = ""
    speed: float = 0.0
    tilt: float | None = None
    open: dict[str, float] = field(default_factory=dict)
    up: dict[str, float] = field(default_factory=dict)
    until: Callable[[Any], bool] = lambda t: False
    max_s: float = 60.0


class Task:
    """A built task world: the simulation, the plug-in, the drive, and readings in mL / Pa per world."""

    def __init__(self, rigs: tuple[Rig, ...], plug: Syringes, worlds: int, dt: float) -> None:
        scene = TestScene(build=build(rigs), worlds=worlds, ground=False, control_dt=CONTROL_DT, dt=dt)
        self.plug, self.rigs = plug, rigs
        self.sim = build_simulation(scene, [plug], "cpu")
        self.step = dt
        self.drive = Drive(self.sim, tuple(r.label for r in rigs))
        self.fluid = plug.fluid
        self.t = 0.0
        self.log: list[dict[str, np.ndarray]] = []
        self.finger = {r.label: 0.0 for r in rigs}
        self.rig = {r.label: r for r in rigs}

    def node(self, name: str) -> int:
        return self.fluid.net.node_index(name)

    def liquid_ml(self, name: str) -> np.ndarray:
        return self.fluid.mixed.liquid.numpy()[:, self.node(name)] * 1e6

    def travel(self, label: str) -> np.ndarray:
        """The stopper's travel from the head [W] m."""
        return self.plug.travel.numpy()[:, [s.name for s in self.plug.config.syringes].index(label)]

    def air_ml(self, name: str) -> np.ndarray:
        return self.plug.air_ml(name)

    def out_ml(self, edge: str) -> np.ndarray:
        """Liquid a port carried out of its node a [mL] (into a reservoir or node b)."""
        return -self.fluid.ledger.numpy()[:, self.fluid.net.edge_index(edge)] * 1e6

    def gauge(self, name: str) -> np.ndarray:
        return self.fluid.pressure.numpy()[:, self.node(name)] - P_ATM

    def run(self, phases: list[Phase]) -> None:
        for ph in phases:
            for e, v in ph.open.items():
                self.plug.set_opening(e, v)
            self._check_capped(ph)
            if ph.up:
                up = self.fluid.mixed.up.numpy()
                for n, v in ph.up.items():
                    up[:, self.node(n)] = v
                self.fluid.mixed.up.assign(up)
            t0 = self.t
            while self.t - t0 < ph.max_s - 1e-9 and not ph.until(self):
                speed = self._advance(ph)
                for lab in self.finger:                  # aim at the period's midpoint: no lag, no lead on a ramp
                    v = speed if lab == ph.rig else 0.0
                    self.drive.set(lab, finger=self.finger[lab] - 0.5 * v * CONTROL_DT,
                                   tilt=ph.tilt if lab == ph.rig else None, speed=v)
                self.sim.step_period()
                self.t += CONTROL_DT
                self.log.append(self._frame())

    def _check_capped(self, ph: Phase) -> None:
        """A syringe with every port closed is a spring A²/C (C = L/K + V_g/(n·P), its contents' compliance) on the
        plunger's armature: refuse a phase that caps one stiffer than the step allows (2·sqrt(I/k), the plug-in's
        check_dt; a plunger on its stop at the head is held by the stop instead)."""
        opening, f = self.fluid.opening.numpy(), self.fluid
        for s in self.plug.config.syringes:
            ports = [k for k, e in enumerate(f.net.edges) if s.name in (e.a, e.b)]
            i = self.node(s.name)
            if np.any(opening[:, ports] > 0.0) or np.all(self.travel(s.name) <= 0.0):
                continue
            c = (f.mixed.liquid.numpy()[:, i] / s.bulk_modulus
                 + f.mixed.gas_volume()[:, i] / (s.polytropic * f.pressure.numpy()[:, i]))
            bound = 2.0 * math.sqrt(self.rig[s.name].armature * float(np.min(c)) / s.area ** 2)
            if self.step > bound:
                raise ValueError(f"phase {ph.name!r} caps syringe {s.name!r} (all its ports closed): a spring that "
                                 f"needs a step under {bound:.3g} s, the task's is {self.step:.3g} s; keep a port open "
                                 f"(a needle in the air) or cap it with more liquid or air")

    def _advance(self, ph: Phase) -> float:
        """Moves the phase's finger target one period, at most REACH past the plunger's stops; its speed (0 there)."""
        if not ph.rig:
            return 0.0
        r = self.rig[ph.rig]
        lo, hi = -r.x0 - REACH, r.stroke - r.x0 + REACH
        want = self.finger[ph.rig] + ph.speed * CONTROL_DT
        self.finger[ph.rig] = min(max(want, lo), hi)
        return ph.speed if lo < want < hi else 0.0

    def _frame(self) -> dict[str, np.ndarray]:
        f = self.fluid
        return {"t": self.t, "out": -f.ledger.numpy().copy() * 1e6, "p": f.pressure.numpy().copy() - P_ATM,
                "air_out": np.stack([self.plug.air_out_ml(e.name) for e in f.net.edges], axis=1)}

    def series(self, edge: str) -> tuple[np.ndarray, np.ndarray]:
        """(t [K], liquid out through ``edge`` [K, W] mL) over the run."""
        e = self.fluid.net.edge_index(edge)
        return np.array([fr["t"] for fr in self.log]), np.stack([fr["out"][:, e] for fr in self.log])

    def peak_gauge(self, name: str) -> np.ndarray:
        i = self.node(name)
        return np.max(np.stack([fr["p"][:, i] for fr in self.log]), axis=0)


def window_max(t: np.ndarray, v: np.ndarray, span: float) -> np.ndarray:
    """The most that ``v`` (cumulative, [K, W]) grew within any ``span`` seconds, per world."""
    j = np.searchsorted(t, t + span, side="right") - 1
    return np.max(v[j] - v, axis=0)


def spread_time(t: np.ndarray, v: np.ndarray, total: Any, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    """Seconds between ``lo`` and ``hi`` of ``total`` (one, or one per world) having passed, per world (inf if
    never)."""
    out = []
    for w, tot in enumerate(np.broadcast_to(np.asarray(total, np.float64), (v.shape[1],))):
        a, b = np.argmax(v[:, w] >= lo * tot), np.argmax(v[:, w] >= hi * tot)
        out.append(t[b] - t[a] if v[b, w] >= hi * tot and v[a, w] >= lo * tot else math.inf)
    return np.asarray(out)


def _rig(s: Syringe, segments: tuple, at=(0.0, 0.0, 0.3), density: float = 1000.0) -> Rig:
    return Rig(s.name, s.stroke, s.x0, column(s.bore, segments, density), at=at)


def _step(plug: Syringes, rigs: tuple[Rig, ...]) -> float:
    """The largest control_dt / 2^k the plug-in's check_dt allows (mass of each plunger: its armature)."""
    for k in range(3, 9):
        dt = CONTROL_DT / 2 ** k
        ok = True
        for s, r in zip(plug.config.syringes, rigs, strict=True):
            g = [e.conductance for e in plug.config.ports if s.name in (e.a, e.b)]
            c = s.area ** 2 / min(g)
            k_s = s.area ** 2 * s.bulk_modulus / (s.dead + (s.liquid or s.area * s.stroke))
            ok &= dt <= min(2.0 * r.armature / c, 2.0 * math.sqrt(r.armature / k_s))
        if ok:
            return dt
    return CONTROL_DT / 2 ** 9


def _world(syringes: tuple[Syringe, ...], vials: tuple[Vial, ...], ports: tuple[Edge, ...],
           columns: dict[str, tuple], worlds: int, density: float = 1000.0) -> Task:
    """Rigs side by side (0.1 m apart), the plug-in, closed ports (the script opens them)."""
    plug = Syringes(syringes=syringes, vials=vials, ports=ports, openings=tuple((e.name, 0.0) for e in ports))
    rigs = tuple(_rig(s, columns[s.name], at=(0.1 * k, 0.0, 0.3), density=density) for k, s in enumerate(syringes))
    return Task(rigs, plug, worlds, _step(plug, rigs))


def _done(cond: Callable[[Task], np.ndarray]) -> Callable[[Task], bool]:
    return lambda t: bool(np.all(cond(t)))


HEAD_UP, HEAD_DOWN = 0.0, math.pi
VIAL_LIQUID = 8.0e-6
VIAL = Vial("vial", volume=13.5e-6, liquid=VIAL_LIQUID, radius=11.0e-3)


def set_dose(worlds: int) -> tuple[Task, list[Phase], Callable[[Task], dict]]:
    """2.8 mL of water and 0.3 mL of air, needle up, into the basin until 2.5 mL remains with no air."""
    s = syringe("syr", 3, 2.8e-6, 0.3e-6)
    task = _world((s,), (), (port("needle", "syr", N25),), {"syr": N25}, worlds)
    phases = [Phase("settle", "syr", tilt=HEAD_UP, max_s=0.2),
              Phase("expel", "syr", -2e-3, open={"needle": 1.0}, until=_done(lambda t: t.liquid_ml("syr") <= 2.56)),
              Phase("set", "syr", -3e-4, until=_done(lambda t: t.liquid_ml("syr") <= 2.5)),
              Phase("hold", "syr", max_s=0.5)]

    def score(t: Task) -> dict:
        dose, air = t.liquid_ml("syr"), t.air_ml("syr")
        return {"dose_ml": dose, "air_ml": air, "success": (np.abs(dose - 2.5) <= 0.125) & (air <= 0.05)}
    return task, phases, score


def _draw(t_name: str, dose: float) -> list[Phase]:
    """Air into the upright vial first, invert, draw past the dose, the bubble back, down to the dose, needle out."""
    s = "syr"
    return [Phase("air in", s, 4e-3, tilt=HEAD_UP, open={"air18": 1.0},
                  until=_done(lambda t: t.air_ml(s) >= dose + 0.07)),
            Phase("over the vial", s, tilt=HEAD_DOWN, open={"air18": 0.0}, max_s=1.5),
            Phase("give the air", s, -4e-3, open={"vial": 1.0}, until=_done(lambda t: t.travel(s) <= 1e-4)),
            Phase("invert", s, tilt=HEAD_UP, up={"vial": -1.0}, max_s=1.5),
            Phase("draw", s, 2e-3, until=_done(lambda t: t.liquid_ml(s) >= dose + 0.15)),
            Phase("bubble back", s, -5e-4, until=_done(lambda t: t.air_ml(s) <= 2e-3)),
            Phase("to the dose", s, -3e-4, until=_done(lambda t: t.liquid_ml(s) <= dose + DEAD * 1e6)),
            Phase("needle out", s, open={"vial": 0.0, "air18": 1.0}, up={"vial": 1.0}, max_s=0.5)]


def draw_dose(worlds: int) -> tuple[Task, list[Phase], Callable[[Task], dict]]:
    """An empty 3 mL syringe with an 18 G fill needle draws 2.5 mL from the 10R vial, giving 2.5 mL of air first."""
    s = syringe("syr", 3, 0.0, DEAD)
    ports = (port("air18", "syr", N18), port("vial", "syr", N18, "vial", depth_b=0.005))
    task = _world((s,), (VIAL,), ports, {"syr": N18}, worlds)

    def score(t: Task) -> dict:
        dose = t.liquid_ml("syr") - DEAD * 1e6
        air, head = t.air_ml("syr"), t.gauge("vial") / 1e3
        loss = VIAL_LIQUID * 1e6 - t.liquid_ml("vial")
        ok = (np.abs(dose - 2.5) <= 0.125) & (air <= 0.05) & (np.abs(head) <= 20.0) & (loss <= dose + 0.1)
        return {"dose_ml": dose, "air_ml": air, "headspace_kpa": head, "vial_loss_ml": loss,
                "vial_upright": t.fluid.mixed.up.numpy()[:, t.node("vial")] > 0.99, "success": ok}
    return task, _draw("draw-dose", 2.5), score


def _inject(s: str, dose_ml: float, rate: float, area: float, air: str = "") -> list[Phase]:
    """Needle down, then into the tissue (the needle leaves the air: port ``air`` closes as the tissue's opens)."""
    shut = {air: 0.0} if air else {}
    return [Phase("needle down", s, tilt=HEAD_DOWN, max_s=1.5),
            Phase("inject", s, -rate * 1e-6 / area, open={"tissue": 1.0, **shut},
                  until=_done(lambda t: t.out_ml("tissue") >= dose_ml - 1e-3), max_s=dose_ml * 12.0),
            Phase("wait", s, max_s=0.5)]


def _inject_score(t: Task, dose_ml: float) -> dict:
    tt, v = t.series("tissue")
    into = t.out_ml("tissue")
    air = t.plug.air_out_ml("tissue")
    burst = window_max(tt, v, 2.0)
    pace = spread_time(tt, v, into, 0.0001, 0.9999) / np.maximum(into, 1e-9)     # over what went in
    ok = (into >= 0.9 * dose_ml) & (air <= 0.05) & (burst <= 0.3 * dose_ml / 0.5) & (pace <= 10.0)
    return {"into_tissue_ml": into, "air_into_tissue_ml": air, "max_ml_in_2s": burst, "s_per_ml": pace, "success": ok}


def inject(worlds: int) -> tuple[Task, list[Phase], Callable[[Task], dict]]:
    """0.5 mL (no air) through a 25 G needle into the muscle, square to the pad, at 0.11 mL/s."""
    s = syringe("syr", 3, 0.5e-6 + DEAD)
    task = _world((s,), (), (port("tissue", "syr", N25, reservoir_medium="liquid"),), {"syr": N25}, worlds)
    return task, _inject("syr", 0.5, 0.11, s.area), lambda t: _inject_score(t, 0.5)


def vial_to_pad(worlds: int) -> tuple[Task, list[Phase], Callable[[Task], dict]]:
    """1 mL drawn with the fill needle (air first), the needle changed to 25 G, the bubble purged, 1 mL into muscle."""
    s = syringe("syr", 3, 0.0, DEAD)
    ports = (port("air18", "syr", N18), port("vial", "syr", N18, "vial", depth_b=0.005), port("air25", "syr", N25),
             port("tissue", "syr", N25, reservoir_medium="liquid"))
    task = _world((s,), (VIAL,), ports, {"syr": N25}, worlds)
    phases = _draw("vial-to-pad", 1.0)[:-1] + [
        Phase("change needle", "syr", tilt=HEAD_UP, open={"vial": 0.0, "air25": 1.0}, up={"vial": 1.0}, max_s=1.0),
        Phase("purge", "syr", -5e-4, until=_done(lambda t: t.air_ml("syr") <= 2e-3), max_s=20.0)] + \
        _inject("syr", 1.0, 0.12, s.area, air="air25")

    def score(t: Task) -> dict:
        return {**_inject_score(t, 1.0), "headspace_kpa": t.gauge("vial") / 1e3}
    return task, phases, score


def _flush(s: str, ml: float, rate: float, area: float) -> list[Phase]:
    """De-air through the bare tip, then push the rest (``ml``) into the vein through the lock at ``rate`` mL/s, to
    the plunger's stop (a syringe capped near empty is a spring the step cannot hold: ``Task._check_capped``)."""
    return [Phase("de-air", s, -1e-3, tilt=HEAD_UP, open={f"{s}_tip": 1.0},
                  until=_done(lambda t: t.air_ml(s) <= 2e-3), max_s=20.0),
            Phase("connect", s, open={f"{s}_tip": 0.0, f"{s}_lock": 1.0}, max_s=0.2),
            Phase("push", s, -rate * 1e-6 / area, until=_done(lambda t: t.travel(s) <= 0.0),
                  max_s=ml / rate * 1.5 + 5.0),
            Phase("clamp", s, open={f"{s}_lock": 0.0}, max_s=0.2)]


def _flush_ports(s: str, viscosity: float = WATER) -> tuple[Edge, Edge]:
    return (port(f"{s}_tip", s, TIP, viscosity=viscosity),
            port(f"{s}_lock", s, LOCK, reservoir=VEIN, reservoir_medium="liquid", viscosity=viscosity))


def flush_lock(worlds: int) -> tuple[Task, list[Phase], Callable[[Task], dict]]:
    """A 10 mL saline flush (5 mL and a 0.2 mL bubble): de-air, connect, 5 mL into the vein at 0.5 mL/s, clamp."""
    s = syringe("flush", 10, 5.0e-6 + DEAD, 0.2e-6)
    task = _world((s,), (), _flush_ports("flush"), {"flush": LOCK}, worlds)

    def score(t: Task) -> dict:
        into, air, peak = t.out_ml("flush_lock"), t.plug.air_out_ml("flush_lock"), t.peak_gauge("flush") / 1e3
        return {"into_vein_ml": into, "air_into_vein_ml": air, "line_peak_kpa": peak,
                "success": (into >= 0.95 * 5.0) & (air <= 0.05) & (peak <= 300.0)}
    return task, _flush("flush", 5.0, 0.5, s.area), score


def _med_flush(med_ml: float, rate: float, flush_rate: float, viscosity: float, worlds: int):
    med, fl = syringe("med", 10, med_ml * 1e-6 + DEAD), syringe("flush", 10, 5.0e-6 + DEAD, 0.2e-6)
    ports = (port("med_lock", "med", LOCK, reservoir=VEIN, reservoir_medium="liquid", viscosity=viscosity),
             *_flush_ports("flush"))
    density = 1350.0 if viscosity == CONTRAST else 1000.0          # iodinated contrast (an estimate)
    task = _world((med, fl), (), ports, {"med": LOCK, "flush": LOCK}, worlds, density)
    phases = [Phase("med", "med", -rate * 1e-6 / med.area, tilt=HEAD_UP, open={"med_lock": 1.0},
                    until=_done(lambda t: t.travel("med") <= 0.0), max_s=med_ml / rate * 1.5 + 5),
              Phase("med off", "med", open={"med_lock": 0.0}, max_s=0.2)] + _flush("flush", 5.0, flush_rate, fl.area)
    return task, phases


def slow_push(worlds: int) -> tuple[Task, list[Phase], Callable[[Task], dict]]:
    """4 mL of medication over 120 s through the lock (0.033 mL/s), then the 5 mL flush at the same rate."""
    task, phases = _med_flush(4.0, 4.0 / 120.0, 4.0 / 120.0, WATER, worlds)

    def score(t: Task) -> dict:
        tt, v = t.series("med_lock")
        into, fl = t.out_ml("med_lock"), t.out_ml("flush_lock")
        burst, spread = window_max(tt, v, 15.0), spread_time(tt, v, 4.0)
        ok = (into >= 0.95 * 4.0) & (burst <= 0.75) & (spread >= 86.0) & (fl >= 0.95 * 5.0)
        return {"into_vein_ml": into, "max_ml_in_15s": burst, "s_5_95": spread, "flush_ml": fl, "success": ok}
    return task, phases, score


def fast_bolus(worlds: int) -> tuple[Task, list[Phase], Callable[[Task], dict]]:
    """10 mL of contrast (11.6 mPa·s) through the 20 G cannula at 1.6 mL/s, then the 5 mL flush at 1 mL/s."""
    task, phases = _med_flush(10.0, 1.6, 1.0, CONTRAST, worlds)

    def score(t: Task) -> dict:
        tt, v = t.series("med_lock")
        into, fl = t.out_ml("med_lock"), t.out_ml("flush_lock")
        spread, peak = spread_time(tt, v, 10.0), np.maximum(t.peak_gauge("med"), t.peak_gauge("flush")) / 1e3
        ok = (into >= 0.95 * 10.0) & (spread <= 7.2) & (peak <= 300.0) & (fl >= 0.95 * 5.0)
        return {"into_vein_ml": into, "s_5_95": spread, "line_peak_kpa": peak, "flush_ml": fl, "success": ok}
    return task, phases, score


def push_check(capped: bool, worlds: int = 2, speed: float = 1e-3, seconds: float = 2.0) -> dict[str, np.ndarray]:
    """WP13's condition (b): at the tasks' own step, the finger pushes a full 3 mL syringe (2.9 mL, no air), capped or
    through a 25 G needle into tissue, needle down. Returns per-period plunger travel, finger target and pressure."""
    s = syringe("syr", 3, 2.9e-6 + DEAD)
    task = _world((s,), (), (port("tissue", "syr", N25, reservoir_medium="liquid"),), {"syr": N25}, worlds)
    out = {"t": [], "x": [], "p": [], "target": [], "air": []}
    task.run([Phase("needle down", "syr", tilt=HEAD_DOWN, max_s=1.0),
              Phase("open", "syr", open={"tissue": 0.0 if capped else 1.0}, max_s=0.1)])
    for _ in range(int(seconds / CONTROL_DT)):
        task.run([Phase("push", "syr", -speed, max_s=CONTROL_DT)])
        for k, v in (("t", task.t), ("x", task.plug.travel.numpy()[:, 0].copy()), ("p", task.gauge("syr")),
                     ("target", task.finger["syr"]), ("air", task.air_ml("syr"))):
            out[k].append(v)
    res = {k: np.asarray(v) for k, v in out.items()}
    res["area"], res["step"] = s.area, task.step
    return res


TASKS = {"set-dose": set_dose, "draw-dose": draw_dose, "inject": inject, "vial-to-pad": vial_to_pad,
         "flush-lock": flush_lock, "slow-push": slow_push, "fast-bolus": fast_bolus}


def run(name: str, worlds: int = 2) -> dict[str, Any]:
    """Build, script and score one task; a JSON-ready row."""
    t0 = time.perf_counter()
    task, phases, score = TASKS[name](worlds)
    task.run(phases)
    res = score(task)
    row = {"task": name, "worlds": worlds, "sim_s": round(task.t, 3), "wall_s": round(time.perf_counter() - t0, 1),
           "step_s": task.step}
    row.update({k: (np.asarray(v) if np.asarray(v).dtype == bool else np.asarray(v, np.float64).round(5)).tolist()
                for k, v in res.items()})
    row["success_rate"] = float(np.mean(res["success"]))
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--worlds", type=int, default=2)
    a = ap.parse_args(argv)
    row = run(a.task, a.worlds)
    print(json.dumps(row))
    return 0 if row["success_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["TASKS", "Phase", "Task", "column", "main", "port", "push_check", "run", "spread_time", "syringe",
           "window_max"]
