# SPDX-License-Identifier: Apache-2.0
"""What a soft item is: a liquid, grains, a cloth or a cable, in what form and how much (``SoftSpec``, plain JSON).

An ``ItemSpec`` with a ``soft`` field is not searched as a rigid object: the builder makes a card for it (a liquid
column in a container, a pile, a sheet, a cable) from a library material or template card. ``kind`` and ``form``:

| kind | forms | needs |
|---|---|---|
| liquid | ``fill`` (in a container), ``pour`` (a stream, or from ``source`` tilted by ``tilt_deg_s`` up to
``tilt_to_deg``) | relation "in" + target |
| grain | ``fill`` (in a container), ``pile`` (on the support or on a target, at the angle of repose) | |
| cloth | ``flat``, ``draped`` (over a target), ``hanging`` (over the table edge), ``folded`` (in half) | |
| cable | ``loose`` (a gentle curve), ``straight`` | ``attach``: items its ends are fixed to |

Amounts: ``volume_ml`` or ``mass_g`` (liquids and grains), or ``fill`` (share of the container's depth); with none,
a liquid fills 60 % of its container and a pile holds 100 ml. Sizes: ``length_m`` (cable length, cloth long side),
``width_m`` (cloth short side), ``diameter_mm`` (cable gauge). Unset values come from the library card.

Example::

    >>> s = SoftSpec.from_dict({"kind": "liquid", "volume_ml": 200})
    >>> s.form, s.volume_ml
    ('fill', 200.0)
    >>> SoftSpec.from_dict(s.to_dict()) == s
    True
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from alloy.compose.errors import ComposeError

KINDS = ("liquid", "grain", "cloth", "cable")
FORMS: dict[str, tuple[str, ...]] = {
    "liquid": ("fill", "pour"),
    "grain": ("fill", "pile"),
    "cloth": ("flat", "draped", "hanging", "folded"),
    "cable": ("loose", "straight"),
}
MAX_FILL = 0.9                       # share of a container's depth a fill may take (room for the surface to move)


@dataclass(frozen=True)
class SoftSpec:
    """A soft item's kind, form and size; see the module docstring.

    Example: ``SoftSpec("cable", length_m=0.5, attach=("laptop",))``."""

    kind: str
    form: str = ""
    volume_ml: float | None = None
    mass_g: float | None = None
    fill: float | None = None
    length_m: float | None = None
    width_m: float | None = None
    diameter_mm: float | None = None
    source: str | None = None
    attach: tuple[str, ...] = ()
    tilt_deg_s: float | None = None
    tilt_to_deg: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ComposeError(f"soft kind {self.kind!r} is unknown", f"use one of {KINDS}", field="items.soft.kind")
        if not self.form:
            object.__setattr__(self, "form", FORMS[self.kind][0])
        if self.form not in FORMS[self.kind]:
            raise ComposeError(f"a {self.kind} cannot be {self.form!r}", f"use one of {FORMS[self.kind]}",
                               field="items.soft.form")
        for name in ("volume_ml", "mass_g", "fill", "length_m", "width_m", "diameter_mm", "tilt_deg_s", "tilt_to_deg"):
            _positive(self, name)
        if (self.tilt_deg_s or self.tilt_to_deg) and (self.form != "pour" or self.source is None):
            raise ComposeError(f"a tilt on a {self.kind} {self.form}", "only a pour from a named vessel tilts: "
                               "'pour 200 ml of water from a jar tilted at 40 deg/s into a mug'",
                               field="items.soft.tilt_deg_s")
        if sum(v is not None for v in (self.volume_ml, self.mass_g, self.fill)) > 1:
            raise ComposeError(f"{self.kind}: give one amount (volume_ml, mass_g or fill), not several",
                               "keep the one you mean, e.g. {'volume_ml': 200}", field="items.soft.volume_ml")
        if self.fill is not None and self.fill > MAX_FILL:
            raise ComposeError(f"fill {self.fill} leaves no room in the container", f"pass fill <= {MAX_FILL}",
                               field="items.soft.fill")
        amounts = self.volume_ml is not None or self.mass_g is not None or self.fill is not None
        if self.kind in ("cloth", "cable") and amounts:
            raise ComposeError(f"a {self.kind} has no volume or mass to ask for", "give length_m / width_m "
                               "(cloth) or length_m / diameter_mm (cable)", field="items.soft.volume_ml")
        if self.kind in ("liquid", "grain") and (self.length_m or self.width_m or self.diameter_mm):
            raise ComposeError(f"a {self.kind} has no length, width or gauge", "give volume_ml, mass_g or fill",
                               field="items.soft.length_m")
        if self.attach and self.kind != "cable" or len(self.attach) > 2:
            raise ComposeError(f"attach={list(self.attach)} on a {self.kind}", "only a cable's two ends attach: "
                               "attach=['laptop'] or ['laptop', 'monitor']", field="items.soft.attach")
        if self.source is not None and self.form != "pour":
            raise ComposeError(f"source={self.source!r} on a {self.kind} {self.form}", "only a pouring liquid has a "
                               "source (the item it pours from)", field="items.soft.source")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"kind": self.kind, "form": self.form}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name not in out and v not in (None, ()):
                out[f.name] = list(v) if isinstance(v, tuple) else v
        return out

    @classmethod
    def from_dict(cls, data: Any) -> SoftSpec:
        if isinstance(data, str):
            data = {"kind": data}
        if not isinstance(data, dict):
            raise ComposeError(f"soft must be an object, got {data!r}", "pass {'kind': 'liquid', 'volume_ml': 200}",
                               field="items.soft")
        allowed = {f.name for f in fields(cls)}
        extra = sorted(set(data) - allowed)
        if extra:
            raise ComposeError(f"unknown soft key(s) {extra}", f"use only {sorted(allowed)}",
                               field=f"items.soft.{extra[0]}")
        out = dict(data)
        for k in ("volume_ml", "mass_g", "fill", "length_m", "width_m", "diameter_mm", "tilt_deg_s", "tilt_to_deg"):
            if out.get(k) is not None:
                out[k] = _number(out[k], k)
        out["attach"] = tuple(out.get("attach") or ())
        return cls(**out)


def _number(v: Any, name: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ComposeError(f"soft {name}={v!r} is not a number", f"pass {name} as a number",
                           field=f"items.soft.{name}") from None


def _positive(s: SoftSpec, name: str) -> None:
    v = getattr(s, name)
    if v is not None and not (isinstance(v, (int, float)) and v > 0):
        raise ComposeError(f"{s.kind}: {name}={v!r} must be a positive number", f"pass {name} > 0 or leave it out",
                           field=f"items.soft.{name}")


__all__ = ["FORMS", "KINDS", "MAX_FILL", "SoftSpec"]
