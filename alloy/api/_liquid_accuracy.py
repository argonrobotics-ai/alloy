# SPDX-License-Identifier: Apache-2.0
"""Liquid accuracy: ``liquid_accuracy="default"`` (4 mm particle spacing), ``"super"`` (2 mm) or ``"ultra"``
(1.5 mm), one build option (SCR-WP9-LIQRES, user decision 2026-11-08; ultra for 15 mm openings, 2026-11-16).
Super has 8x the particles and takes about 2x the steps: about 16x the work (11-20x wall time measured on CPU pours).
Ultra has 19x the default's particles and 2.7x its steps: about 50x the work; one world on one L4 ran P1 (a 50 mm
beaker poured into a 15 mm vial) at 21 s per simulated s, 3.0x super (7.1 s).

A build warns once when the setting is too coarse for the scene, listing each item and naming the setting that fixes
it (``coarse`` / ``warn``). Counted as particles across, width / spacing:
- an opening a liquid sits in or must pass (a card's ``passage_width``, an open cylinder's inner diameter, an open
  box's narrower inner side, a block's hole): fewer than ``OPENING_ACROSS`` = 8 across (LIQ-6) at any setting. The
  24 mm floor at any setting went on 2026-11-17: water at rest in cups of 8-24 mm stays healthy at 1.5 and 2 mm and
  pours into 12-32 mm cups keep 98-100 % (docs/notes/wp9-results.md, liquid fix 2); before, cups 3-6 across
  started INVALID at 4 mm and 8 and 16 mm cups at 2 mm (the cup's 24 wall slabs counted twice at the seams, a crowded
  floor slot, an overfilled round lattice).
- a pour's stream (``add_pour``'s nozzle): fewer than ``STREAM_ACROSS`` = 2 across. One particle across lands 36 % of
  it in a 70 mm cup at either spacing; 2-5 across land 99-100 % (CPU probes, notes/wp9-runs/liqres/).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from alloy.api import _say
from alloy.api.errors import SceneError

SPACING = {"default": 4.0e-3, "super": 2.0e-3, "ultra": 1.5e-3}   # m: SPH particle spacing (radius = half)
#: measured accuracy per setting (CPU quick laws and pours, [internal note] notes/wp9-runs/liqres/README.md)
ACCURACY = {
    "default": "hydrostatic wall force 1.1 %, floating draft 4.9 % (5 particles deep), dam front within "
               "Ritter's bound; pours keep 99.7 % in cups from 32 mm",
    "super": "hydrostatic wall force 0.2 %, floating draft 4.2 %, dam front within Ritter's bound; pours keep "
             "98.5-99.9 % in cups from 12 mm",
    "ultra": "hydrostatic wall force 1.2 %, floating draft 2.1 %, dam front within Ritter's bound; pours keep "
             "98.1-100 % in cups from 12 mm",
}
#: cost against the default (4 mm) setting: the work (particles x steps); measured rows in docs/notes/wp9-results.md
COST = {"super": "about 16x the cost", "ultra": "about 50x the cost"}
OPENING_ACROSS = 8.0
STREAM_ACROSS = 2.0


@dataclass(frozen=True)
class Coarse:
    """One item the setting resolves too coarsely: ``what`` is "opening" or "stream"; ``width`` m."""

    item: str
    what: str
    width: float
    across: float


def checked(accuracy: Any) -> str:
    """``accuracy`` if it is a setting, else a SceneError naming both. Example: ``checked("super")`` -> "super"."""
    if not isinstance(accuracy, str) or accuracy not in SPACING:
        raise SceneError(f"liquid_accuracy must be one of {sorted(SPACING)}, got {accuracy!r}",
                         f"pass liquid_accuracy='default' (4 mm particles), 'super' (2 mm, {COST['super']}) or "
                         f"'ultra' (1.5 mm, {COST['ultra']})",
                         field="liquid_accuracy")
    return accuracy


def saved(accuracy: str) -> dict[str, str]:
    """The Scene JSON entry: nothing at the default, so scene JSON and hashes written before the field are unchanged
    (coordinator 02:58Z). Example: ``saved("super")`` -> ``{"liquid_accuracy": "super"}``."""
    return {} if accuracy == "default" else {"liquid_accuracy": checked(accuracy)}


def spacing(accuracy: str | None) -> float:
    """The particle spacing [m] of a setting (None = "default"). Example: ``spacing("super")`` -> 0.002."""
    return SPACING[checked("default" if accuracy is None else accuracy)]


def _openings(items: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, float]]:
    """(fixture, width m) of every fixture's opening (``fixtures.place`` records it on each part)."""
    seen: set[str] = set()
    for name, it in items.items():
        fixture, width = it.get("fixture") or name, it.get("opening")
        if width is not None and fixture not in seen:
            seen.add(fixture)
            yield fixture, float(width)


def card_openings(built: Any) -> list[tuple[str, float]]:
    """(liquid, width m) of each liquid whose containers' cards give a ``passage_width`` (the router records the
    narrowest as ``passage_width`` in the liquid's route numbers)."""
    out = []
    for name, d in built.report.decisions.items():
        width = (d.numbers or {}).get("passage_width")
        if width:
            out.append((f"{name}'s container", float(width)))
    return out


def _streams(items: Mapping[str, Mapping[str, Any]]) -> Iterable[tuple[str, float]]:
    """(name, width m) of every pour's nozzle: its area is rate / exit speed. Example: ``list(_streams(items))``."""
    for name, it in items.items():
        pour = it.get("pour")
        if pour:
            speed = math.sqrt(sum(v * v for v in pour["velocity"]))
            yield name, math.sqrt(pour["rate"] / speed) if speed > 0 else 0.0


def has_liquid(items: Mapping[str, Mapping[str, Any]]) -> bool:
    """True when the scene holds a liquid. Example: ``has_liquid(scene.items)``."""
    return any(it.get("kind") == "liquid" for it in items.values())


def coarse(items: Mapping[str, Mapping[str, Any]], spacing_m: float,
           cards: Iterable[tuple[str, float]] = ()) -> list[Coarse]:
    """The scene's items too coarse at particle spacing ``spacing_m`` (empty without a liquid); ``cards``: openings
    from cards (``card_openings``). Example: ``coarse(scene.items, 4e-3)``."""
    if not has_liquid(items):
        return []
    s = float(spacing_m)
    out = [Coarse(n, "opening", w, w / s) for n, w in [*_openings(items), *cards]
           if w / s < OPENING_ACROSS - 1e-9]
    out += [Coarse(n, "stream", w, w / s) for n, w in _streams(items) if w / s < STREAM_ACROSS - 1e-9]
    return out


def built_spacing(built: Any, accuracy: str) -> float:
    """The finest liquid spacing the router gave (2 x particle_radius), else the setting's.
    Example: ``built_spacing(built, "default")`` -> 0.004."""
    radii = [float(d.numbers["particle_radius"]) for d in built.report.decisions.values()
             if str(d.route) == "liquid_sph" and "particle_radius" in (d.numbers or {})]
    return 2.0 * min(radii) if radii else spacing(accuracy)


def _fix(c: Coarse, accuracy: str) -> str:
    """What to change for one coarse item: the cheapest finer setting that resolves it, else the geometry or the
    pour."""
    other = "widen it" if c.what == "opening" else "raise the pour's rate or lower its speed"
    limit = OPENING_ACROSS if c.what == "opening" else STREAM_ACROSS
    finer = [a for a in ("super", "ultra") if SPACING[a] < SPACING[accuracy]]
    for acc in finer:                                  # the cheapest finer setting that resolves it
        if c.width / SPACING[acc] >= limit - 1e-9:
            return f"build with liquid_accuracy={acc!r} ({SPACING[acc] * 1e3:g} mm particles, {COST[acc]})"
    return (f"even liquid_accuracy={finer[-1]!r} is too coarse: " if finer else "") + other


def _needs(c: Coarse) -> str:
    """The threshold the item missed, in particles across (OSS-17's review)."""
    return f"{OPENING_ACROSS if c.what == 'opening' else STREAM_ACROSS:g} across"


def message(found: list[Coarse], accuracy: str | None, spacing_m: float | None = None) -> str:
    """One warning line naming each coarse item and its fix. Example: ``message(coarse(items, 4e-3), "default")``."""
    acc = "default" if accuracy is None else accuracy
    s = (SPACING[acc] if spacing_m is None else spacing_m) * 1e3
    rows = "; ".join(f"{c.item}: {c.what} {c.width * 1e3:.1f} mm = {c.across:.1f} particles across "
                     f"(needs {_needs(c)}): {_fix(c, acc)}" for c in found)
    return f"liquid at {s:g} mm particles (liquid_accuracy={acc!r}) is too coarse here: {rows}"


def warn(scene: Any, built: Any = None) -> list[Coarse]:
    """Warn once (``alloy: ...`` when no log handler shows it) if ``scene.liquid_accuracy`` is too coarse for its
    openings and pours (and, given the assembled scene ``built``, its cards' openings); returns the items."""
    s = built_spacing(built, scene.liquid_accuracy) if built is not None else spacing(scene.liquid_accuracy)
    found = coarse(scene.items, s, card_openings(built) if built is not None else ())
    if found:
        _say.warn("%s", message(found, scene.liquid_accuracy, s))
    return found


def router_for(scene: Any) -> Any:
    """The router at ``scene.liquid_accuracy``'s spacing and ``scene.cloth_accuracy`` (SCR-WP2-13: "super" tags
    the scene's CLOTH_VBD decisions). Example: ``SceneAssembler(lib, router_for(scene))``."""
    from alloy.route.router import Router  # noqa: PLC0415
    from alloy.route.types import RouterConfig  # noqa: PLC0415
    from alloy.scene import cloth_accuracy  # noqa: PLC0415

    cloth = cloth_accuracy.checked(getattr(scene, "cloth_accuracy", "default"), SceneError)
    return Router(RouterConfig(liquid_accuracy=checked(scene.liquid_accuracy), cloth_accuracy=cloth,
                               liquid_tension=bool(getattr(scene, "liquid_tension", False))))


__all__ = ["ACCURACY", "COST", "Coarse", "OPENING_ACROSS", "SPACING", "STREAM_ACROSS", "built_spacing",
           "card_openings", "checked", "coarse", "message", "router_for", "saved", "spacing", "warn"]
