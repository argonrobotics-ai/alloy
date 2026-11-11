# SPDX-License-Identifier: Apache-2.0
"""pad-finish's rest watch and waits (:mod:`alloy.tasks.pad_finish_stages`): the film's rest checks after a program
and after every wait command, their wait until a stage is ready, and the readiness rules.

A check (``Watch.after``) steps the film 0.2 s with the tools clear (their RESEAT_S, ``pad_finish_stage.rest_after``)
and reads the rest rule (``film_rest``: the bound under 1, or every face held by the film's own step). A film still
moving is stepped on, at most ``FILM_SETTLE_S``, until it rests: their relax would have moved it during their wait,
so the flow comes after the command instead (disclosed); the check keeps its first refusal (``stepped_on``, with the
film the step-on moved) and the summary counts them and sums that volume (DC5 S2). Their wait (``wait_for``) runs in
``WAIT_CHUNK`` commands of ``pad_run.material_wait`` (the layer's clock alone) until the stage is ready, within the
budget left of their 5 h; a stage not ready at the budget, or a wait their rule refuses, is returned as a refusal for
the stage's hand-off, not raised (DC5 S1, N8).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alloy.plugins.errors import PluginError
from alloy.tasks import concrete_materials as CM
from alloy.tasks import pad_finish_stage as ST
from alloy.tasks import pad_run as PR
from alloy.tasks.concrete_paste_work import MIN_ADVANCE
from alloy.tasks.pad_finish_handoff import START_STATES

PERIOD = PR.PERIOD
WAIT_CHUNK = 60.0                          # s per wait command
SETTLE_S = 5.0                             # s: at most this long held after a program for the tools to come still
FILM_SETTLE_S = 60.0                       # s: at most this long a rest watch steps on a film still moving
STILL = 0.1 * MIN_ADVANCE                  # m/s: a tool's low-passed advance counts as still under this
Ready = Callable[[dict[str, np.ndarray]], bool]   # the next stage's readiness from the layer's w, age [W, pad cells]


def snapshot(flow: Any, layer: Any, crop: Callable[[np.ndarray], np.ndarray]) -> dict[str, np.ndarray]:
    """The surface and the layer's start states and records on the pad's cells (before a wait: DC5's wait check)."""
    r = layer.read()
    out = {k: crop(np.asarray(r[k])) for k in ("age", *START_STATES, "late", "wet_worked") if k in r}
    out["surface"] = crop(flow.base.numpy().astype(np.float64) + flow.h.numpy())
    return out


def settle(sim: Any, layer: Any, carriage: Any) -> int:
    """Hold after a program (the tools on their carts) until every tool's low-passed advance is under ``STILL`` and
    none is loaded, at most ``SETTLE_S``: their wait asks the tools to be still (env.py:88-126). Returns the periods."""
    for i in range(int(SETTLE_S / PERIOD)):
        adv = np.linalg.norm(layer.advance.numpy().reshape(-1, 3), axis=1)
        if float(adv.max()) < STILL and float(np.abs(carriage.load.numpy()).max()) <= 0.0:
            return i
        sim.step_period()
    return int(SETTLE_S / PERIOD)


@dataclass
class Watch:
    """A stage's film rest checks in order (``rest_after``: 0.2 s stepped with the tools clear after the settle and
    after every wait command, their RESEAT_S), and the periods they stepped. Where a check finds a face moving, the
    stage steps on, at most ``FILM_SETTLE_S``, until it rests (DC5: step it before the hand-off; their relax would move
    it during their wait); the check keeps that first refusal (``stepped_on``). Each check first writes the film's
    yields as the paste has them now (``seed_yields``, with ``layer``): the layer writes them on the paste only, so the
    forms' smear kept the yield of its stage's start (the darby's 5 min, 1641 Pa, at its 10 min hand-off: rest ratio
    1.003 there, 0.96 at its own age)."""

    checks: list[dict[str, Any]] = field(default_factory=list)
    periods: int = 0

    def after(self, sim: Any, flow: Any, carriage: Any, layer: Any = None) -> dict[str, Any]:
        if layer is not None:
            ST.seed_yields(flow, layer)
        first = r = ST.rest_after(sim, flow, carriage)
        h0 = None if first["rest"] else _film(flow)
        n = r["periods"]
        while not r["rest"] and n < FILM_SETTLE_S / PERIOD:
            r = ST.rest_after(sim, flow, carriage)
            n += r["periods"]
        check = {**r, "periods": n}
        if h0 is not None:
            check["stepped_on"] = {**{k: first[k] for k in ("bound", "face_max", "over")},
                                   "moved_m3": moved(flow, h0, _film(flow))}
        self.checks.append(check)
        self.periods += n
        return check

    @property
    def last(self) -> dict[str, Any]:
        return self.checks[-1]

    def summary(self) -> dict[str, Any]:
        """For the row (WP15, DC5 S2): the hand-off's bound, faces over 1, face speed and the rule that passed; over
        all checks how many passed by the film's own step, how many were refused (still moving at the cap), how many
        were stepped on before they rested, the largest bound and count over 1 (first refusals included), the
        largest face speed a check found before stepping on, and the film the step-ons moved (``moved``)."""
        c = self.checks
        firsts = [x["stepped_on"] for x in c if "stepped_on" in x]
        return {**{f"film.{k}": v for k, v in self.last.items()}, "film.checks": len(c),
                "film.held_checks": sum(x["rule"] == "held" for x in c),
                "film.refused_checks": sum(not x["rest"] for x in c), "film.stepped_on_checks": len(firsts),
                "film.max_bound": max(x["bound"] for x in [*c, *firsts]),
                "film.max_over": max(x["over"] for x in [*c, *firsts]),
                "film.max_face_before": max((x["face_max"] for x in firsts), default=0.0),
                "film.stepped_on_moved_m3": sum(x["moved_m3"] for x in firsts)}


def _film(flow: Any) -> np.ndarray:
    return flow.h.numpy().astype(np.float64).copy()


def moved(flow: Any, h0: np.ndarray, h1: np.ndarray) -> float:
    """The film a step-on moved [m³]: half the depth changed over the chart's cells, the most of any world (what
    flowed out of cells flowed into others; their relax would have moved it during the wait)."""
    area = np.asarray(flow.tables.area_np, np.float64)
    return float((0.5 * np.abs(h1 - h0).reshape(h0.shape[0], -1) @ area).max())


@dataclass
class Waited:
    """Their wait's outcome: the seconds waited, and a refusal for the stage's hand-off (not ready within the budget,
    or a command their rule refused), or None."""

    seconds: float = 0.0
    refusal: str | None = None


def wait_for(sim: Any, flow: Any, layer: Any, carriage: Any, ready: Ready | None, limit: float,
             crop: Callable[[np.ndarray], np.ndarray], watch: Watch) -> Waited:
    """Their wait (``pad_run.material_wait`` with the rest rule ``film_rest``, as the watch last found it), in
    ``WAIT_CHUNK`` commands each followed by ``watch.after`` (RESEAT_S of full ticks), until ``ready`` (on the pad's
    cells, ``crop``) or ``limit`` seconds (the budget left of their 5 h)."""
    def view() -> dict[str, np.ndarray]:
        r = layer.read()
        return {k: crop(np.asarray(r[k])).reshape(len(r[k]), -1) for k in ("w", "age")}
    out = Waited()
    while ready is not None and not ready(view()):
        if out.seconds >= limit:
            out.refusal = f"pad-finish wait: not ready within the budget ({limit:.0f} s left of their 5 h)"
            break
        rest = bool(watch.last["rest"])                  # the watch's last check: the bound or the film's own hold
        try:
            out.seconds += PR.material_wait(sim, flow, layer, carriage, min(WAIT_CHUNK, limit - out.seconds),
                                            rest=lambda _flow, ok=rest: ok)
        except PluginError as e:
            out.refusal = f"pad-finish wait refused: {e}"
            break
        watch.after(sim, flow, carriage, layer)
    return out


def sheen_gone(mask: np.ndarray) -> Ready:
    """Ready when no cell of ``mask`` (flat, the stage's chart) holds a bleed film over ``SHEEN_W`` in any world."""
    m = np.asarray(mask, bool).reshape(-1)
    return lambda r: float(r["w"][:, m].max()) <= CM.SHEEN_W


def age_at_least(mask: np.ndarray, age: float) -> Ready:
    """Ready when every cell of ``mask`` in every world is at least ``age`` [s] old (a stage's authored start: their
    pad-edge at 60 min, pad-trowel at 165 min, x MASCO; ``concrete_pad_start.AGES_21C``)."""
    m = np.asarray(mask, bool).reshape(-1)
    return lambda r: float(r["age"][:, m].min()) >= age


def all_of(*rules: Ready) -> Ready:
    return lambda r: all(rule(r) for rule in rules)


def bleed_over(mask: np.ndarray, temperature: float) -> Ready:
    """Ready when the bleed has ended (no water left to rise: the trowel's sealing traps none) and no sheen stands."""
    end, sheen = CM.Concrete().bleed_end_time(temperature), sheen_gone(mask)
    m = np.asarray(mask, bool).reshape(-1)
    return lambda r: float(r["age"][:, m].min()) >= end and sheen(r)


__all__ = ["FILM_SETTLE_S", "PERIOD", "Ready", "SETTLE_S", "STILL", "WAIT_CHUNK", "Waited", "Watch", "age_at_least",
           "all_of", "bleed_over", "moved", "settle", "sheen_gone", "snapshot", "wait_for"]
