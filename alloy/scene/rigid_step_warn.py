# SPDX-License-Identifier: Apache-2.0
"""The build's warning for a start drop that breaks the travel-per-step limit at the chosen rigid step (audit ERR-2;
SCR-WP0-RIGIDDT).

The rule is ``rigid_step.cap_gap``: a body must not travel more than ``RIGID_GAP_M`` (5 mm) in one rigid step. (MuJoCo
applies no contact force until the surfaces overlap, so a body meets a part about v dt deep:
docs/known_gaps/rigid-tunnelling.md.) Under ``mode="auto"`` the step scan never goes below the fixed step, so a body
that breaks the cap even there was silent; scenes that ``why_not`` refuses (robots, kinematic objects, ...) never ran the
caps at all. This module checks every free dynamic body the facts can describe, at whatever step was chosen, against its
START-DROP impact speed ``sqrt(2 g drop)`` (``BodyFacts.drop``; the card's nominal contact speed is not the claim here),
and words the result: the travel per step, what it can miss (the tunnel cap's rule, ``2 v dt``) and the fixes with
numbers. A fall is warned only where a pass is possible: onto the ground plane, or onto a part (static or another body)
at least ``2 v dt`` thick along the fall, nothing is lost. Nothing here changes a step; a body it cannot describe is
skipped.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from alloy.core.defaults import GRAVITY, RIGID_GAP_M
from alloy.core.timebase import TICK_EXP
from alloy.scene import rigid_step_bodies as RB

PREFIX = "start drop breaks the travel-per-step limit: "   # every warning here starts with it (estimate filters on it)
MAX_NAMED = 5                                       # bodies named one by one; the rest in one closing line
SAMPLE_WORLDS = 64                                  # worlds checked when worlds x bodies^2 exceeds COUNT_CELLS
COUNT_CELLS = 50_000_000
TOL = 1.0 + 1e-9


@dataclass(frozen=True)
class Starts:
    """What ``RB.drops_and_loads`` read: positions (W, n, 3), randomized half ranges, each body's box offset and half
    extent, masses and the static shapes."""

    P: np.ndarray
    U: np.ndarray
    offset: np.ndarray
    half: np.ndarray
    mass: np.ndarray
    fixed: tuple[np.ndarray, np.ndarray, float]


@dataclass(frozen=True)
class Landing:
    """What body ``i`` lands on in one world: the fall [m], and the part's thickness along it [m] (inf for the ground
    plane or nothing below) and name (None for a static part)."""

    drop: float
    thick: float
    name: str | None = None


@dataclass(frozen=True)
class Breaker:
    """A body whose start drop can pass through what it lands on at the step: the worst such world, and how many."""

    name: str
    landing: Landing
    speed: float            # m/s: sqrt(2 g drop) in the worst world
    worlds_broken: int      # -1 when only a sample of worlds was checked
    worlds: int


def drop_speed(drop: float) -> float:
    """Impact speed [m/s] of a fall of ``drop`` metres."""
    return math.sqrt(2.0 * GRAVITY * max(drop, 0.0))


def _breaks(speed: float, dt: float) -> bool:
    return speed * dt > RIGID_GAP_M * TOL


def passes(land: Landing, dt: float) -> bool:
    """A pass is possible: the impact travels over the limit per step and the part is thinner than ``2 v dt``."""
    v = drop_speed(land.drop)
    return _breaks(v, dt) and land.thick < 2.0 * v * dt * TOL


def _candidates(starts: Starts, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tops (n, m), thicknesses (m,) of everything a body of world ``w`` can land on: other bodies, static boxes, the
    plane (-inf where its footprint does not meet body i's, or the part is not below it)."""
    sc, sh, plane = starts.fixed
    Pw, half, U = starts.P[w] + starts.offset, starts.half, starts.U
    bottom = Pw[:, 2] - half[:, 2]
    centres = np.vstack([Pw, sc]) if len(sc) else Pw
    halves = np.vstack([half, sh]) if len(sc) else half
    ranges = np.vstack([U, np.zeros_like(sh)]) if len(sc) else U
    gap = np.abs(Pw[:, None, :2] - centres[None, :, :2])
    meets = np.all(gap < half[:, None, :2] + halves[None, :, :2] - U[:, None, :2] - ranges[None, :, :2], axis=-1)
    tops = centres[:, 2] + halves[:, 2]
    ok = meets & (tops[None, :] <= bottom[:, None] + RB.RESTING_TOL)
    ok[np.arange(len(Pw)), np.arange(len(Pw))] = False
    cand = np.where(ok, tops[None, :], -np.inf)
    plane_col = np.full((len(Pw), 1), plane)
    return np.hstack([cand, plane_col]), np.append(2.0 * halves[:, 2], math.inf), bottom


def landings(starts: Starts, w: int, names: Sequence[str]) -> list[Landing]:
    """Each body's landing in world ``w``: the highest part under its footprint, else no fall (drop 0)."""
    tops, thick, bottom = _candidates(starts, w)
    out = []
    for i in range(len(bottom)):
        j = int(np.argmax(tops[i]))
        if not np.isfinite(tops[i, j]):
            out.append(Landing(0.0, math.inf))
            continue
        out.append(Landing(max(float(bottom[i] - tops[i, j]), 0.0), float(thick[j]), names[j] if j < len(names) else None))
    return out


def _worlds(starts: Starts) -> tuple[list[int], bool]:
    W, n = starts.P.shape[:2]
    if W * n * n <= COUNT_CELLS:
        return list(range(W)), False
    return sorted({int(x) for x in np.linspace(0, W - 1, min(W, SAMPLE_WORLDS))}), True


def breakers(facts: Sequence[RB.BodyFacts], starts: Starts, dt: float) -> list[Breaker]:
    """The bodies that can pass through what they land on at step ``dt`` (worst world each), worst first."""
    names = [f.name for f in facts]
    checked, sampled = _worlds(starts)
    per_world = [landings(starts, w, names) for w in checked]
    found = []
    for i, f in enumerate(facts):
        bad = [lw[i] for lw in per_world if passes(lw[i], dt)]
        if bad:
            worst = max(bad, key=lambda x: x.drop)
            found.append(Breaker(f.name, worst, drop_speed(worst.drop), -1 if sampled else len(bad), len(starts.P)))
    return sorted(found, key=lambda x: -x.speed)


def grid_step_for(speed: float, control_dt: float) -> float | None:
    """The largest grid step ``control_dt / 2**j`` (j <= TICK_EXP) with ``speed`` x step <= the gap, else None."""
    for j in range(TICK_EXP + 1):
        if not _breaks(speed, control_dt / 2.0**j):
            return control_dt / 2.0**j
    return None


def control_dt_for(speed: float) -> float:
    """A control period [s] whose fixed step (= the period when below 1/600 s) meets the cap: the gap over the speed,
    rounded down to one significant figure."""
    limit = RIGID_GAP_M / speed
    mag = 10.0 ** math.floor(math.log10(limit))
    return math.floor(limit / mag + 1e-9) * mag


def _mm(x: float) -> str:
    return f"{x * 1e3:.3g} mm"


def _m(x: float) -> str:
    return f"{x:.3g} m"


def _ms(x: float) -> str:
    return f"{x * 1e3:.4g} ms"


def max_drop(dt: float) -> float:
    """The highest start drop [m] whose impact keeps ``speed x dt`` within the gap at step ``dt``."""
    return (RIGID_GAP_M / dt) ** 2 / (2.0 * GRAVITY)


def step_fix(speed: float, control_dt: float, can_pin: bool) -> str:
    """The rigid-step fix: a pin on the control grid, or (a pin is ignored, or no grid step fits) a smaller Scene dt."""
    step = grid_step_for(speed, control_dt)
    if can_pin and step is not None:
        return f"scene.rigid_step = {step!r} ({_ms(step)}; set before alloy.build)"
    dt = control_dt_for(speed)
    why = "this scene keeps its fixed step, so scene.rigid_step is ignored" if not can_pin else "no grid step is fine enough"
    return f"alloy.Scene(dt={dt!r}) ({_ms(dt)} control period, which is then the rigid step; {why})"


def _target(land: Landing) -> str:
    return f"{land.name!r} ({_mm(land.thick)} thick)" if land.name else f"a static part {_mm(land.thick)} thick"


def message(b: Breaker, dt: float, control_dt: float, can_pin: bool) -> str:
    """One body's warning: the fall and what it lands on, the travel per step against the limit, what it can miss, the
    fixes."""
    where = f" in {b.worlds_broken} of {b.worlds} worlds (worst shown)" if b.worlds_broken > 0 and b.worlds > 1 else ""
    if b.worlds_broken < 0:
        where = f" in some of {b.worlds} worlds (worst shown)"
    bound = _mm(2.0 * b.speed * dt)
    return (f"{PREFIX}{b.name!r} falls {_m(b.landing.drop)} onto {_target(b.landing)}{where} and lands at "
            f"{b.speed:.3g} m/s; it travels {_mm(b.speed * dt)} per rigid step ({_ms(dt)}), over the "
            f"{_mm(RIGID_GAP_M)} travel-per-step limit, so it can pass through parts thinner than about {bound} "
            f"(MuJoCo applies no force until the surfaces overlap). Fix: {step_fix(b.speed, control_dt, can_pin)}; or "
            f"drop it from at most {_m(max_drop(dt))} at this step; or land it on a part at least {bound} thick")


def gap_warnings(facts: Sequence[RB.BodyFacts], starts: Starts | None, dt: float, control_dt: float,
                 can_pin: bool) -> list[str]:
    """The warnings for the bodies of ``facts`` whose start drop breaks the gap cap at ``dt``: at most MAX_NAMED named,
    then one line counting the rest. ``can_pin``: a number given to ``scene.rigid_step`` is honoured (an eligible
    scene)."""
    if starts is None or not facts:
        return []
    found = breakers(facts, starts, dt)
    out = [message(b, dt, control_dt, can_pin) for b in found[:MAX_NAMED]]
    if len(found) > MAX_NAMED:
        out.append(f"{PREFIX}{len(found) - MAX_NAMED} more bodies break it at the {_ms(dt)} rigid step "
                   f"(the {MAX_NAMED} worst are named above)")
    return out
