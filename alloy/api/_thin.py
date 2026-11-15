# SPDX-License-Identifier: Apache-2.0
"""Parts thinner than the card check's thin-member rule, from ``alloy.Scene``: the form that keeps one that thin.

The card check refuses a solid part whose thinnest member is under 2 x contact speed x rigid step (1.25 mm for an
``alloy.Scene`` part, whatever ``Scene(dt=)``) unless its card declares ``features.min_wall``. Declaring it resizes
nothing (no step, margin or contact reads it); it says the part is meant to be that thin. ``add_box``,
``add_cylinder``, ``add_capsule`` and ``add_mesh`` pass ``card_extra=`` on to the part's card, as ``add_cloth``
declares its sheet's thickness; ``add_sphere``, ``add_rod`` and the fixtures (``add_open_box``, ...) take none.

The form is ``rigid=True, card_extra={...}``: with the wall declared and no ``rigid=True`` the router makes a part
this thin a cloth sheet (a box, a mesh), a chain (a capsule) or refuses it (a cylinder). Measured (CPU, 100
periods): a 0.2 mm tab, a 0.61 mm wire and a 0.2 mm tab on a 0.5 mm plate rest on the ground or a table with
``sim.check()`` ok; the tab dropped 2 mm (0.2 m/s at impact) went 1 mm deep, past half its thickness, and the check
failed that world. A part landing or hit faster than its thickness / (2 x rigid step) (0.08 m/s for 0.2 mm) can do
that, so the fix says the speed.
"""

from __future__ import annotations

import re
from typing import Any

#: the ``add_*`` calls that pass ``card_extra=`` on to the part's card
CARD_EXTRA_CALLS = ("add_box", "add_cylinder", "add_capsule", "add_mesh")
#: what the fix asks to add to that call, followed by the ``card_extra`` dict
FORM = "rigid=True, card_extra="
#: the card check's error for an undeclared thin member (``alloy.cards.validate_object``): the part's thinnest
#: member and the limit, both in m
_CARD_ERROR = re.compile(r"thinnest member ([0-9.eE+-]+) m < [^=]*= ([0-9.eE+-]+) m")


def min_wall_extra(thin: float) -> dict[str, Any]:
    """The ``card_extra`` that declares a part's thinnest member ``thin`` (m).

    Example::

        >>> min_wall_extra(0.0002)["features"]["min_wall"]["value"]
        0.0002
    """
    return {"features": {"min_wall": {"value": float(f"{thin:.4g}"), "source": "model",
                                      "note": "thinnest member, as the build measures it"}}}


def keep_thin(thin: float, call: str | None = None) -> str:
    """The fix clause that keeps a part ``thin`` (m) thick and rigid: ``rigid=True, card_extra=...`` on ``call`` (one
    of ``CARD_EXTRA_CALLS``), on whichever of them made the part when ``call`` is None, or that ``call`` has none.

    Example: ``keep_thin(0.0002, "add_box")`` -> ``'or, to keep it this thin and rigid, add rigid=True,
    card_extra={...} to its add_box call ...'``."""
    from alloy.cards.library import RIGID_DT  # noqa: PLC0415

    if call is not None and call not in CARD_EXTRA_CALLS:
        return (f"{call} takes no card_extra, so a part this thin cannot be kept there "
                f"({' / '.join(CARD_EXTRA_CALLS)} take one)")
    where = call or " / ".join(CARD_EXTRA_CALLS)
    return (f"or, to keep it this thin and rigid, add {FORM}{min_wall_extra(thin)!r} to its {where} call "
            "(without rigid=True the router makes a part this thin a cloth sheet or a chain, or refuses it): it "
            f"builds and rests, but landing or hit faster than {thin / (2.0 * RIGID_DT):.2g} m/s it can go more "
            "than half its thickness deep, and sim.check() then fails that world")


def card_error_fix(exc: BaseException) -> str | None:
    """The fix for the card check's thin-member ``CardError`` (``features.min_wall`` undeclared), naming the
    ``alloy.Scene`` form that keeps the part; None for any other error.

    Example: ``card_error_fix(CardError("film", "features.min_wall",
    "thinnest member 0.0003 m < 2·contact_speed·rigid_dt = 0.00125 m ..."))``."""
    if not str(getattr(exc, "field", "")).endswith("features.min_wall"):
        return None
    found = _CARD_ERROR.search(str(getattr(exc, "message", exc)))
    if found is None:
        return None
    thin, limit = float(found.group(1)), float(found.group(2))
    return (f"make it at least {limit * 1e3:.3g} mm thick (add_rod: a radius of at least {limit * 5e2:.3g} mm); "
            f"{keep_thin(thin)}. add_sphere, add_rod and the fixtures take no card_extra")


__all__ = ["CARD_EXTRA_CALLS", "FORM", "card_error_fix", "keep_thin", "min_wall_extra"]
