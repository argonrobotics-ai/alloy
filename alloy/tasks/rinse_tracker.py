# SPDX-License-Identifier: Apache-2.0
"""Host-side rules of the dunk proofs (scrub-rinse-pan, rinse-glassware): the sanitizer session clock and the pan's
progress-v1 score. Their pan tracker (``task.py``) and the v1 glass tracker (``task_set/sanitizer.py:25-141``) share
the clock's rules; the glass's limits (10 µg, c_rinse 5 mg/L) come in through ``soap_limit`` and the caller's
``carried``.

Ported from Gym-Anything's RoboWorld ``surface_rinsing`` environment (paths under
``benchmarks/roboworld/environments/surface_rinsing/``; file:line). Pure Python: the caller reads the simulator
and fills one ``PanSample`` every 10 ms; nothing here steps or imports a simulator.

- ``PanTracker`` is ``PanTracker.sample`` of ``task.py:247-333`` (class doc ``task.py:131-163``).
- ``pan_progress`` / ``clean_term`` / ``rinsed_term`` are ``PanTracker.measure`` / ``progress``
  (``task.py:337-382``); ``pan_score`` is ``Evaluator._score`` (``task.py:782-802``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from alloy.tasks.rinse_score import FAIL_SCALE, OUTLET_OFF_GPM, Verdict, clip, finish

log = logging.getLogger(__name__)

SAMPLE_DT_S = 0.01             # one sample = 20 ticks; clock step "20*P.DT" (task.py:324; "Sampled every 20 ticks
                               # (10 ms)", 133)
SANITIZE_S = 30.0              # params.py:251 CRITERIA sanitize_s
VRL_KG_M2 = 1.0e-5             # visible-residue limit 1 ug/cm2: params.py:158
SURF_PER_ITEM_KG = 10.0e-9     # params.py:251 CRITERIA surf_per_item
SOAP_REF_AREA_M2 = 0.085       # soap limit = surf_per_item / 0.085 * area: task.py:195 (and gate task.py:723)
DRAINED_BOWL_M3 = 5.0e-6       # params.py:252 CRITERIA drained_bowl; c_rinse = soap limit / this (task.py:196)
PAN_INSIDE_WET = 0.85          # task.py:52-53
PAN_STREAM_RINSE_ML = 5.0      # task.py:54-56
PAN_SURF_REF_KG = 13.5e-6      # task.py:57-60 (scaled by the seed's c_wash / default c_wash: task.py:198)
PAN_FLOOR_Z_M = 0.25           # task.py:61
METAL_UNDER_FRAC = 0.999       # metal fully under the pool: task.py:322
STAGE_GATES = ("placed", "drained", "soap_free")   # T: task.py:380
_MIN_SOLUTE = 1e-12            # m3: a cavity this small holds no standing water (task.py:223)


def soap_limit_kg(area_m2: float) -> float:
    """Detergent a pan may carry into the sanitizer or store: ``task.py:195`` (16.5 ug on their pan)."""
    return SURF_PER_ITEM_KG / SOAP_REF_AREA_M2 * area_m2


def rinse_strength_kg_m3(soap_limit: float) -> float:
    """c_rinse, the soap limit in the drained bowl (3.3 mg/L): ``task.py:196``."""
    return soap_limit / DRAINED_BOWL_M3


def carried_detergent_kg(faces_kg: float, cavity_kg: float, cavity_m3: float, c_rinse: float) -> float:
    """Detergent the pan carries: face solution + foam + adsorbed, plus the cavity's solute when it is stronger than
    rinse strength (``task.py:218-225``). ``faces_kg`` = sum((s + gamma) * area); ``cavity_kg`` = sp[0] + sp[5] of the
    cavity pool."""
    if cavity_m3 <= _MIN_SOLUTE or cavity_kg / cavity_m3 > c_rinse:
        return faces_kg + cavity_kg
    return faces_kg


@dataclass
class PanSample:
    """One 10 ms reading of the pan (``task.py:247-266``)."""

    t: float                          # s, simulated time
    gauge: float                      # kg/m2, worst 1 cm adhered-soil gauge over the pan (task.py:262)
    carried: float                    # kg, carried_detergent_kg(...) (task.py:218-225)
    z: float                          # m, pan origin height (floor test, task.py:267)
    inside_wet: float                 # area share of the inner faces under any liquid, 0..1 (task.py:265)
    pool: str | None = None           # bowl whose liquid immerses the pan's body ('wash'/'rinse'/'sanitize'), else None
    pool_frac: float = 0.0            # share of the metal under that pool (task.py:322 needs > 0.999)
    touch_sanitizer: bool = False     # pool == 'sanitize' or any outer cell under the sanitizer (task.py:259)
    other_wet: str | None = None      # 'wash'/'rinse' when that bowl's liquid wets the pan (pool or outer cell,
                                      # task.py:260)
    in_bowl: bool = False             # in a pool, or an outer cell under any bowl's liquid (task.py:261)
    centred_over_sanitizer: bool = False   # origin x, y inside the sanitizer bowl's footprint (task.py:329)
    outlet_running: bool = False      # any outlet above OUTLET_OFF_GPM (1e-4 gpm; task.py:305)
    water_ml: float = 0.0             # mL, all water on the pan (film everywhere + cavity), absolute (task.py:299)
    inside_water_ml: float = 0.0      # mL, inner film + cavity, absolute (task.py:298)


@dataclass
class RinseOwed:
    """Why a rinse is owed and since when (``task.py:271-272``)."""

    t: float
    why: str


@dataclass
class EntryRecord:
    """One entry into the sanitizer (``task.py:277-279``)."""

    t: float
    carried_kg: float
    gauge: float
    clean: bool
    within_soap_limit: bool
    rinse_owed: bool
    qualifies: bool


@dataclass
class PanTracker:
    """The sanitizer session clock and the rinse-owed rule (``task.py:131-163``, ``247-333``).

    ``sanitized_s`` is the current clock (theirs keeps no separate best: reset sets it to 0, ``task.py:244``).
    """

    soap_limit: float                       # kg, soap_limit_kg(area)
    dt: float = SAMPLE_DT_S                 # s per sample (theirs 10 ms; a task samples once per control period)
    sanitized_s: float = 0.0
    rinse_owed: RinseOwed | None = None
    session_ok: bool = False                # the current entry qualified (their st["ok"])
    live: bool = False                      # the clock ran at the last sample
    into_sanitizer_kg: float = 0.0          # carry-in of the entry the clock depends on (task.py:287-292)
    entries: list[EntryRecord] = field(default_factory=list)
    resets: list[tuple[float, str]] = field(default_factory=list)
    rinses_after_owed: list[tuple[float, str]] = field(default_factory=list)
    _in_san: bool = False
    _prev: tuple[float, float] | None = None       # (carried, gauge) at the last sample with no cell under the
                                                   # sanitizer
    _window: tuple[float, float] | None = None     # max of those while centred over the sanitizer bowl
    _outside_prev: bool = False
    _water_prev: float | None = None
    _water_in_prev: float | None = None
    _stream_ml: float = 0.0
    _stream_in_ml: float = 0.0

    def sample(self, s: PanSample) -> None:
        """Advance the clock by one sample (``dt``; theirs 10 ms), in their order (task.py:268-333)."""
        rinsed = self._wetted_by_other(s)
        self._floor(s)
        if s.touch_sanitizer and not self._in_san:
            self._enter(s)
        self._in_san = s.touch_sanitizer
        rinsed = self._outlet_stream(s) or rinsed
        if rinsed and self.rinse_owed is not None and s.z >= PAN_FLOOR_Z_M:
            self.rinses_after_owed.append((s.t, rinsed))
            log.info("pan rinse owed since t=%.2f cleared at t=%.2f by %s", self.rinse_owed.t, s.t, rinsed)
            self.rinse_owed = None
        self.live = bool(self.session_ok and s.pool == "sanitize" and s.pool_frac > METAL_UNDER_FRAC
                         and s.inside_wet >= PAN_INSIDE_WET)
        if self.live:
            self.sanitized_s += self.dt
        self._track_window(s)

    def _reset(self, t: float, why: str) -> None:
        self.resets.append((t, why))
        log.info("pan sanitizer clock reset at t=%.2f (%.2f s): %s", t, self.sanitized_s, why)
        self.sanitized_s = 0.0
        self.session_ok = False

    def _in_session(self) -> bool:
        return self.session_ok or self.sanitized_s > 0

    def _wetted_by_other(self, s: PanSample) -> str | None:
        """Wash/rinse wetting restarts the session; a full rinse-bowl immersion with the inside flooded is a rinse."""
        if s.other_wet is None:
            return None
        if self._in_session():
            self._reset(s.t, f"wetted by the {s.other_wet} compartment's water")
        full = s.pool == "rinse" and s.pool_frac > METAL_UNDER_FRAC and s.inside_wet >= PAN_INSIDE_WET
        return "the rinse compartment's water, the inside flooded" if s.other_wet == "rinse" and full else None

    def _floor(self, s: PanSample) -> None:
        if s.z >= PAN_FLOOR_Z_M:
            return
        if self._in_session():
            self._reset(s.t, "touched the floor")
        if self.rinse_owed is None:
            self.rinse_owed = RinseOwed(s.t, "touched the floor")

    def _enter(self, s: PanSample) -> None:
        """First sample under the sanitizer: qualify on the largest carry and gauge of now, before, and the window."""
        cands = [(s.carried, s.gauge)] + [c for c in (self._prev, self._window) if c is not None]
        into, g_in = max(c[0] for c in cands), max(c[1] for c in cands)
        clean, within = g_in <= VRL_KG_M2, into <= self.soap_limit
        qualifies = clean and within and self.rinse_owed is None
        self.entries.append(EntryRecord(s.t, into, g_in, clean, within, self.rinse_owed is not None, qualifies))
        if qualifies:
            self.into_sanitizer_kg = max(self.into_sanitizer_kg if self.sanitized_s > 0 else 0.0, into)
            self.session_ok = s.other_wet is None
        else:
            why = ("not visibly clean" if not clean else "carrying detergent" if not within else
                   f"before a rinse owed since t={self.rinse_owed.t} ({self.rinse_owed.why})")
            self.into_sanitizer_kg = into
            self._reset(s.t, "entered the sanitizer " + why)
            if self.rinse_owed is None:
                self.rinse_owed = RinseOwed(s.t, "entered the sanitizer " + why)
        self._window, self._stream_ml, self._stream_in_ml = None, 0.0, 0.0

    def _outlet_stream(self, s: PanSample) -> str | None:
        """> 5 mL from a running outlet outside every bowl resets the clock; 5 mL gained inside is a rinse."""
        outside = not s.touch_sanitizer and not s.in_bowl
        rinsed = None
        if outside and self._outside_prev and self._water_prev is not None and s.outlet_running:
            self._stream_ml += max(0.0, s.water_ml - self._water_prev)
            prev = s.inside_water_ml if self._water_in_prev is None else self._water_in_prev
            self._stream_in_ml += max(0.0, s.inside_water_ml - prev)
            if self._stream_ml > PAN_STREAM_RINSE_ML:
                if self._in_session():
                    self._reset(s.t, f"rinsed by a running outlet after sanitizing ({self._stream_ml:.1f} mL)")
                self._stream_ml = 0.0
            if self._stream_in_ml > PAN_STREAM_RINSE_ML:
                rinsed, self._stream_in_ml = "a running outlet's water on the inside", 0.0
        self._outside_prev, self._water_prev, self._water_in_prev = outside, s.water_ml, s.inside_water_ml
        return rinsed

    def _track_window(self, s: PanSample) -> None:
        if s.touch_sanitizer:
            return
        now = (s.carried, s.gauge)
        self._prev = now
        if not s.centred_over_sanitizer:
            self._window = None
        elif self._window is None:
            self._window = now
        else:
            self._window = (max(self._window[0], now[0]), max(self._window[1], now[1]))


def clean_term(gauges0: list[float], gauges: list[float], areas: list[float]) -> float:
    """C: area-weighted share of the log way from the start load to VRL over gauges soiled at the start; 1 if none
    (``task.py:341-347``)."""
    soiled = [(g0, max(g, 1e-30), a) for g0, g, a in zip(gauges0, gauges, areas) if g0 > VRL_KG_M2]
    if not soiled:
        return 1.0
    parts = [(clip(math.log(g0 / g) / math.log(g0 / VRL_KG_M2)), a) for g0, g, a in soiled]
    return sum(p * a for p, a in parts) / sum(a for _, a in parts)


def rinsed_term(detergent_kg: float, surf_ref: float = PAN_SURF_REF_KG, soap_limit: float = 16.5e-9) -> float:
    """R: log share of the way from ``surf_ref`` (what a poured pan carries from the wash) to the soap limit, with
    S = faces' detergent + the detergent of the water the pan would take away (``task.py:358-361``)."""
    return clip(math.log(surf_ref / max(detergent_kg, 1e-15)) / math.log(surf_ref / soap_limit))


def pan_progress(clean: float, rinsed: float, sanitized_s: float, stored: bool) -> tuple[float, dict[str, float]]:
    """progress-v1 = round((C + C R + C R Z + C R Z T)/4, 4), Z = min(sanitized_s / 30, 1) (``task.py:377-382``).
    Stages are credited only on a pan that keeps the earlier ones. Their C and R are rounded to 4 places first
    (``task.py:364``); rounded here too."""
    c, r = round(clean, 4), round(rinsed, 4)
    z, t = min(sanitized_s / SANITIZE_S, 1.0), 1.0 if stored else 0.0
    value = round((c + c * r + c * r * z + c * r * z * t) / 4, 4)
    return value, dict(clean=c, rinsed=r, sanitized=round(z, 4), stored=t)


def pan_score(v: Verdict, clean: float, rinsed: float, sanitized_s: float, verifiable: bool) -> Verdict:
    """Fill ``v`` for the pan: 1 if every gate passes else 0.9 x progress; None when unverifiable
    (``task.py:782-802``). Stored T needs the placed, drained and soap_free gates in ``v.criteria``."""
    stored = all(v.criteria.get(k, (False, None))[0] is not False for k in STAGE_GATES)   # None: no judged part failed
    value, v.terms = pan_progress(clean, rinsed, sanitized_s, stored)
    finish(v, value, verifiable)
    if v.score is not None:
        v.score = 1.0 if v.success else round(FAIL_SCALE * v.progress, 4)
    return v


__all__ = ["EntryRecord", "OUTLET_OFF_GPM", "PanSample", "PanTracker", "RinseOwed", "carried_detergent_kg",
           "clean_term", "pan_progress", "pan_score", "rinse_strength_kg_m3", "rinsed_term", "soap_limit_kg"]
