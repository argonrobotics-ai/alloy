# SPDX-License-Identifier: Apache-2.0
"""Liquid accuracy: ``liquid_accuracy="default"`` (4 mm particle spacing) or ``"super"`` (2 mm), one build option
(SCR-WP9-LIQRES; user decision 2026-11-08). Super has 8x the particles and takes about 2x the steps: about 16x the
cost (11-20x wall time measured on pours).

A build warns once when the setting is too coarse for the scene, listing each item and naming the setting that fixes
it (``coarse`` / ``warn``). Counted as particles across, width / spacing (CPU probes,
[internal note] ``notes/wp9-runs/liqres/``):
- an opening a liquid sits in or must pass (a card's ``passage_width``, an open cylinder's inner diameter, an open
  box's narrower inner side, a block's hole): fewer than ``OPENING_ACROSS`` = 8 across (LIQ-6), or under
  ``OPENING_MIN`` = 24 mm at any setting. At 4 mm, a pour into a 24 mm cup (6 across) kept 95.6 % against 99.7 % from
  8 across, and water at rest in cups 3-6 across started INVALID; at 2 mm a 16 mm cup started INVALID too.
- a pour's stream (``add_pour``'s nozzle): fewer than ``STREAM_ACROSS`` = 2 across. One particle across lands 36 % of
  it in a 70 mm cup at either spacing; 2-5 across land 99-100 %.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from alloy.api import _say
from alloy.api.errors import SceneError

SPACING = {"default": 4.0e-3, "super": 2.0e-3}       # m: SPH particle spacing per setting (radius = half)
#: measured accuracy per setting (CPU quick laws and pours, [internal note] notes/wp9-runs/liqres/README.md)
ACCURACY = {
    "default": "hydrostatic wall force 1.1 %, floating draft 4.9 % (5 particles deep), dam front within "
               "Ritter's bound; pours keep 99.7 % in cups from 32 mm",
    "super": "hydrostatic wall force 0.2 %, floating draft 4.2 %, dam front within Ritter's bound; pours keep "
             "97.6-99.8 % in cups from 24 mm",
}
OPENING_ACROSS = 8.0
OPENING_MIN = 0.024                                   # m
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
                         "pass liquid_accuracy='default' (4 mm particles) or 'super' (2 mm, about 16x the cost)",
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
           if w / s < OPENING_ACROSS - 1e-9 or w < OPENING_MIN - 1e-12]
    out += [Coarse(n, "stream", w, w / s) for n, w in _streams(items) if w / s < STREAM_ACROSS - 1e-9]
    return out


def built_spacing(built: Any, accuracy: str) -> float:
    """The finest liquid spacing the router gave (2 x particle_radius), else the setting's.
    Example: ``built_spacing(built, "default")`` -> 0.004."""
    radii = [float(d.numbers["particle_radius"]) for d in built.report.decisions.values()
             if str(d.route) == "liquid_sph" and "particle_radius" in (d.numbers or {})]
    return 2.0 * min(radii) if radii else spacing(accuracy)


def _fix(c: Coarse, accuracy: str) -> str:
    """What to change for one coarse item: super if it resolves it, else the geometry or the pour."""
    other = "widen it" if c.what == "opening" else "raise the pour's rate or lower its speed"
    if c.what == "opening" and c.width < OPENING_MIN - 1e-12:
        return f"widen it to {OPENING_MIN * 1e3:.0f} mm or more (narrower containers are unreliable at any setting)"
    limit = OPENING_ACROSS if c.what == "opening" else STREAM_ACROSS
    if accuracy != "super" and c.width / SPACING["super"] >= limit - 1e-9:
        return "build with liquid_accuracy='super' (2 mm particles, about 16x the cost)"
    return ("even liquid_accuracy='super' is too coarse: " if accuracy != "super" else "") + other


def _needs(c: Coarse) -> str:
    """The threshold the item missed: the 24 mm floor, else particles across (OSS-17's review)."""
    if c.what == "opening" and c.width < OPENING_MIN - 1e-12 and c.across >= OPENING_ACROSS - 1e-9:
        return f"{OPENING_MIN * 1e3:.0f} mm"
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
    return Router(RouterConfig(liquid_accuracy=checked(scene.liquid_accuracy), cloth_accuracy=cloth))


__all__ = ["ACCURACY", "Coarse", "OPENING_ACROSS", "OPENING_MIN", "SPACING", "STREAM_ACROSS", "built_spacing",
           "card_openings", "checked", "coarse", "message", "router_for", "saved", "spacing", "warn"]
