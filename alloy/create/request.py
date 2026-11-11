# SPDX-License-Identifier: Apache-2.0
"""What to create: the input (a mesh file, a photo or a text description) and what you know about the object.

Everything is optional except the input. What you leave out is estimated and labelled ESTIMATE in the result, never
refused: the real size from the object's category, the mass from cited category values or the material's density,
friction and restitution from a library material.

Example::

    >>> r = CreateRequest.from_dict({"mesh": "mug.obj", "size": 0.11, "category": "mug"})
    >>> r.kind, r.card_name
    ('mesh', 'mug')
    >>> CreateRequest.from_dict(r.to_dict()) == r
    True
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from alloy.api.errors import AlloyAPIError

MESH_SUFFIXES = (".obj", ".stl", ".glb", ".gltf", ".ply", ".off")
MODEL_SUFFIXES = (".xml", ".mjcf", ".urdf", ".usd", ".usda", ".usdc", ".usdz")
PHOTO_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}
UP_AXES = ("z", "y", "-y", "x")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class CreateError(AlloyAPIError, ValueError):
    """The object cannot be created as asked; ``message`` says why, ``fix`` what to change.

    Example: ``CreateError("the mesh is 1200 units long", "pass size=0.12 (m) or units='mm'")``."""


@dataclass(frozen=True)
class CreateRequest:
    """One object to make.

    Input (exactly one): ``mesh`` (OBJ, STL, GLB/GLTF, PLY; or MJCF/URDF/USD through ``alloy.io``), ``photo`` (an
    image of the object) or ``text`` (a description). ``name``: the card name (default from the file or text).
    ``size``: the real longest side in metres; ``units``: the mesh's units ("mm", "cm", "m", "in"); ``up``: the
    mesh's up axis. ``mass`` (kg) with ``mass_source`` (where it comes from: a datasheet, a scale); ``material``: a
    library material name or a word ("wood", "steel"); ``category``: what it is ("mug"), used for estimates.
    ``out``: the folder (a card root) the cards and meshes are written to. Example: see the module docstring."""

    mesh: str | None = None
    photo: str | None = None
    text: str | None = None
    name: str | None = None
    size: float | None = None
    units: str | None = None
    up: str = "z"
    mass: float | None = None
    mass_source: str | None = None
    material: str | None = None
    category: str | None = None
    out: str = "my_objects"

    def __post_init__(self) -> None:
        given = [k for k in ("mesh", "photo", "text") if getattr(self, k)]
        if len(given) != 1:
            raise CreateError(f"give exactly one input, got {given or 'none'}",
                              "pass mesh='part.obj', photo='mug.jpg' or text='a ceramic mug'", field="input")
        if self.units is not None and self.units not in UNITS:
            raise CreateError(f"units {self.units!r} unknown", f"use one of {sorted(UNITS)}", field="units")
        if self.up not in UP_AXES:
            raise CreateError(f"up {self.up!r} unknown", f"use one of {UP_AXES}", field="up")
        for key in ("size", "mass"):
            v = getattr(self, key)
            if v is not None and not (isinstance(v, (int, float)) and v > 0):
                raise CreateError(f"{key} {v!r} is not a positive number", f"pass {key} in SI units "
                                  "(metres, kilograms) or leave it out to estimate it", field=key)
        if self.name is not None and not _NAME.match(self.name):
            raise CreateError(f"name {self.name!r} is not snake_case", "use lower case, digits and _, e.g. 'red_mug'",
                              field="name")

    @property
    def kind(self) -> str:
        return "mesh" if self.mesh else ("photo" if self.photo else "text")

    @property
    def card_name(self) -> str:
        """The name to use: given, else from the input file name or the first words of the text."""
        if self.name:
            return self.name
        raw = Path(self.mesh or self.photo).stem if (self.mesh or self.photo) else " ".join(self.text.split()[:4])
        slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_") or "object"
        return slug if slug[0].isalpha() else f"obj_{slug}"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreateRequest:
        if not isinstance(data, dict):
            raise CreateError(f"a create request must be a JSON object, got {type(data).__name__}",
                              "pass {'mesh': 'part.obj', 'size': 0.1}")
        allowed = {f.name for f in fields(cls)}
        extra = sorted(set(data) - allowed)
        if extra:
            raise CreateError(f"unknown key(s) {extra}", f"use only {sorted(allowed)}", field=extra[0])
        return cls(**data)


__all__ = ["CreateError", "CreateRequest", "MESH_SUFFIXES", "MODEL_SUFFIXES", "PHOTO_SUFFIXES", "UNITS", "UP_AXES"]
