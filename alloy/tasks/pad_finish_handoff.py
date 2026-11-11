# SPDX-License-Identifier: Apache-2.0
"""pad-finish's hand-off between stages (:mod:`alloy.tasks.pad_finish`): one stage's end, at a rest point, as the
next stage's start fields (``concrete_pad_start.pad_fields`` format), and the paste's records across stages.

Gym-Anything runs pad-finish as one world (``task.py:26``); here each stage (WP6's darby, WP4's edger and trowel) runs
its own tools and chart split on its own simulation, and stages meet where their wait would be allowed: every tool
back on the cart, the film at rest (``env.py:113-126``). What carries over:
- the surface: h = b + tb + ts everywhere, as theirs (``layout.py:256-269``): on the paste with the cream ``ts`` kept
  where the surface still holds it (the next stage splits base and film from b, tb, ts by its own rule); on the forms
  ``tb`` is what stands on their top, the smear (``smear`` repeats it there, as ``pad_fields``). The paste's volume is
  kept exactly;
- the layer's start states (``START_STATES``, ``ConcretePaste.set_start``), age as one site clock;
- the records the layer cannot be started with (``RECORDS``): a late or wet working stays recorded (OR), the distance
  worked adds up;
- fields other plug-ins keep (the edger's lip compaction ``compact``), passed through ``extra``.

WP15's conditions for chaining (review of the design, 11-07): the same chart at every hand-off (``same_chart``); no
tool loaded and nothing in a store or a roll (the stages check before handing off); states written only through the
host API between periods; one continuous clock; and the chart's float32 base picked first, the film then the target
surface less that base in float64 (``exact_split``), so the surface carries over exactly (a float64 base rounded to
float32 at 0.1 m moves ~7e-9 m per cell).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

START_STATES = ("D", "a", "E", "S", "C", "w", "q", "dwc", "bl")    # ConcretePaste.set_start's fields besides age
RECORDS = {"late": "or", "wet_worked": "or", "worked": "sum"}      # read by the verdict; never started
AGE_TOL = 1e-6                                                     # s: one clock over the pad


@dataclass
class Records:
    """The paste's records over the stages so far, per world [W, nx, ny]."""

    fields: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, read: Mapping[str, np.ndarray]) -> None:
        """Fold one stage's records in (``RECORDS``' rule each)."""
        for name, rule in RECORDS.items():
            now = np.asarray(read[name])
            old = self.fields.get(name)
            if old is None:
                self.fields[name] = now.copy()
            elif rule == "or":
                self.fields[name] = np.maximum(old, now)
            else:
                self.fields[name] = old + now


def age_of(age: np.ndarray, paste: np.ndarray) -> float:
    """The site clock [s]: the paste cells' age, one value (their region ages as one, ``paste.py`` t0)."""
    a = np.asarray(age, np.float64)[np.asarray(paste, bool)]
    if a.size and float(a.max() - a.min()) > AGE_TOL:
        raise ValueError(f"the paste's age is not one clock: {a.min():.6f} to {a.max():.6f} s")
    return float(a.mean()) if a.size else 0.0


def after_stage(f: Mapping[str, Any], surface: np.ndarray, states: Mapping[str, np.ndarray],
                extra: Mapping[str, np.ndarray] | None = None) -> dict[str, Any]:
    """The next stage's start fields: ``f`` (the stage's start, one world) with the surface ``surface`` [nx, ny]
    (base + film at its end), the layer's start ``states`` [nx, ny] (``START_STATES`` and ``age``) and ``extra``
    fields replaced; the cream ``ts`` kept where the surface holds it. ``h`` is the surface itself: a form cell the
    stage left a float32 step under ``b`` (0.089 m is not a float32; a form's film emptied to its float32 base) stays
    there, not raised back to ``b`` (that raised 15,681 bare form cells 5.9 nm each, 3.7e-10 m3, at the darby's
    hand-off on the L4 run: paste the darby's books counted gone, back on the pad)."""
    paste = np.asarray(f["paste_mask"], bool)
    b = np.asarray(f["b"], np.float64)
    s = np.asarray(surface, np.float64)
    above = np.maximum(s - b, 0.0)
    out = dict(f)
    out["ts"] = np.where(paste, np.minimum(np.asarray(f["ts"], np.float64), above), 0.0)
    out["tb"] = above - out["ts"]                                  # on the forms: the smear (their tb there)
    out["smear"] = np.where(paste, 0.0, above)
    out["h"] = s
    for name in START_STATES:
        if name in states:
            out[name] = np.where(paste, np.asarray(states[name]), np.asarray(f.get(name, 0.0)))
    out["age"] = age_of(states["age"], paste) if "age" in states else f["age"]
    out.update({k: np.asarray(v) for k, v in (extra or {}).items()})
    return out


def exact_split(chart: Mapping[str, np.ndarray], surface: np.ndarray) -> dict[str, np.ndarray]:
    """A stage's chart split (``base``, ``film``, ``paste``: its owner's rule) made exact for ``surface``: the base
    as the chart holds it (float32), the film the surface less that base in float64 (WP15)."""
    s = np.asarray(surface, np.float64)
    if float((np.asarray(chart["base"], np.float64) - s).max()) > 1e-6:
        raise ValueError(f"the split's base stands {float((chart['base'] - s).max()):.3g} m above the surface")
    base = np.minimum(np.asarray(chart["base"], np.float64), s).astype(np.float32)
    high = base.astype(np.float64) > s                       # rounded up past the surface (a base at it): one ulp down
    base = np.where(high, np.nextafter(base, np.float32(-np.inf)), base)
    return {"base": base, "film": s - base.astype(np.float64), "paste": np.asarray(chart["paste"])}


def same_chart(a: Mapping[str, Any], b: Mapping[str, Any]) -> None:
    """Two stages' starts on the same cells (their fields map 1:1): shape and cell centres equal."""
    if np.shape(a["h"]) != np.shape(b["h"]) or not (np.array_equal(a["x"], b["x"]) and np.array_equal(a["y"], b["y"])):
        raise ValueError(f"the stages' charts differ: {np.shape(a['h'])} vs {np.shape(b['h'])} or their cell centres")


def paste_volume(f: Mapping[str, Any]) -> float:
    """m³ of paste over the subgrade on the pad and over the forms' tops (b excluded; their tb + ts everywhere)."""
    cell = float(f["cell"])
    return float((np.asarray(f["tb"]) + np.asarray(f["ts"])).sum()) * cell * cell


__all__ = ["AGE_TOL", "RECORDS", "START_STATES", "Records", "after_stage", "age_of", "exact_split", "paste_volume",
           "same_chart"]
