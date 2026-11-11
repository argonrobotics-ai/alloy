# SPDX-License-Identifier: Apache-2.0
"""The route ``Scene.add_cloth`` asks for (SCR-WP7-15 P3): the sheet of a material that yields is a film sheet, and a
sheet of a paste is refused."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alloy.api import catalog
from alloy.api.errors import SceneError
from alloy.core.errors import UnitError
from alloy.core.units import parse_si

_REASON = "set through alloy.Scene.add_cloth: "
#: Pa. A material that yields is a film sheet only this stiff: the library's papers and polymer films are 190 MPa to
#: 4.6 GPa, dough 52 kPa (on the film it would stretch 3.8 % and never flow).
FILM_MIN_YOUNGS = 1.0e6


def add_cloth_route(material: str, scene_materials: Mapping[str, Any], roots: Sequence[str | Path]) -> dict:
    """``card_extra`` for ``Scene.add_cloth``: the sheet routes cloth_film when its material card is made for the
    film, since the scene asked for a sheet; {} otherwise (the router decides, e.g. cotton -> cloth_vbd). A scene
    material made ``like=`` a library card is followed to that card (add_material sets only density, friction and
    restitution). See ``film_reason`` for which cards; ``refuse_paste`` raises for a paste. Without the film plug-in
    the router keeps its own route and warns for a j2 sheet (router._no_film), and raises naming register() for the
    others.
    Example: ``add_cloth_route("paper_80gsm", {}, [])["route"]["route"]`` -> ``"cloth_film"``."""
    import yaml  # noqa: PLC0415

    name, seen = material, set()
    while name in scene_materials and name not in seen:
        seen.add(name)
        name = scene_materials[name].get("like") or ""
    path = catalog.card_path("materials", name, roots) if name else None
    card = yaml.safe_load(path.read_text(encoding="utf-8")) if path is not None else None
    if not isinstance(card, dict):
        return {}
    refuse_paste(material, card)
    why = film_reason(card)
    return {} if why is None else {"route": {"route": "cloth_film", "reason": _REASON + why}}


def film_reason(card: Mapping[str, Any]) -> str | None:
    """Why a material card's sheet belongs on the film, or None:
    - ``plastic.model: j2`` (a metal): the J2 hinges keep a dent. Without this a small or thick metal sheet routed
      rigid by the bending test, silently (a 6 x 6 cm, 0.5 mm sheet);
    - a ``film:`` block (the papers): the card carries the film's own values;
    - a plastic ``yield_stress`` with an elastic ``youngs`` of at least FILM_MIN_YOUNGS (the polymer films): the
      film caps the stretch at yield / E. Without this the router's paste rule took the sheet and the build failed
      (CapacityError: a 0.2 m paper or LDPE sheet asked for 5,120,000 MPM particles against a budget of 81).
    Example: ``film_reason({"film": {}})`` -> ``"a film: block, the film's own values"``."""
    plastic = card.get("plastic") or {}
    if plastic.get("model") == "j2":
        return "a plastic.model j2 sheet dents"
    if card.get("film") is not None:
        return "a film: block, the film's own values"
    if plastic.get("yield_stress") is not None and _youngs(card) >= FILM_MIN_YOUNGS:
        return "a sheet that yields (plastic.yield_stress, elastic.youngs): the film caps its stretch at yield / E"
    return None


def refuse_paste(material: str, card: Mapping[str, Any]) -> None:
    """Raise SceneError for a sheet of a paste: a card with a plastic ``yield_stress`` that is not a film sheet
    (``film_reason`` None: no modulus, or one under FILM_MIN_YOUNGS, e.g. mayonnaise, dough). Its sheet went to the
    paste route at the sheet's thickness and the build failed with CapacityError (5,120,000 MPM particles against a
    budget of 81); on the film it would stretch a few percent and never flow.
    Example: ``refuse_paste("dough_wheat", card)`` raises; ``refuse_paste("cotton_woven", card)`` returns None."""
    if (card.get("plastic") or {}).get("yield_stress") is None or film_reason(card) is not None:
        return
    raise SceneError(f"add_cloth: {material!r} is a paste (it yields and flows), and add_cloth makes sheets that "
                     f"bend and stretch",
                     f"give the paste a solid shape instead, e.g. add_box(name, size=(x, y, depth), "
                     f"material={material!r}); or pick a sheet material (alloy.materials())", field="material")


def _youngs(card: Mapping[str, Any]) -> float:
    """The card's elastic youngs in Pa (a number, a unit string or a {value: ...} quantity), 0 when it has none."""
    E = (card.get("elastic") or {}).get("youngs")
    if isinstance(E, Mapping):
        E = E.get("value")
    try:
        return parse_si(E, "pressure") if E is not None else 0.0
    except UnitError:
        return 0.0
