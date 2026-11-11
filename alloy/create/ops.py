# SPDX-License-Identifier: Apache-2.0
"""``create`` for the ``alloy`` command line and the MCP tool server (an ``alloy.cli.spec.Op``; JSON in and out).

Registered in ``alloy.cli.registry.OPS`` by its provider list. Example::

    alloy create --mesh part.stl --units mm --material steel --out my_objects
    alloy create --text "a red ceramic coffee mug" --size 0.11
"""

from __future__ import annotations

from typing import Any

from alloy.cli.spec import Op, Param
from alloy.create.request import UNITS, UP_AXES

FIELDS = ("mesh", "photo", "text", "name", "size", "units", "up", "mass", "mass_source", "material", "category",
          "out")


def op_create(session: Any, a: dict[str, Any]) -> dict[str, Any]:
    """Make one object card; returns ``CreateResult.to_dict()`` (values with their sources, what to verify)."""
    from alloy.create.build import create  # noqa: PLC0415

    fields = {k: a[k] for k in FIELDS if a.get(k) is not None}
    return create(check=a.get("check", True) is not False, device=a.get("device"), **fields).to_dict()


OPS: tuple[Op, ...] = (Op(
    "create", "Make a validated object card from your mesh, a photo or a text description.",
    (Param("mesh", "string", "mesh file: OBJ, STL, GLB/GLTF, PLY, or MJCF/URDF/USD"),
     Param("photo", "string", "a photo of the object (needs a 3D generator: $ALLOY_GEN3D_URL or a GPU)"),
     Param("text", "string", "a description, e.g. 'a red ceramic coffee mug' (needs a 3D generator)"),
     Param("name", "string", "card name, snake_case (default: from the file or text)"),
     Param("size", "number", "real longest side in metres (default: inferred, or the category's typical size)"),
     Param("units", "string", "the mesh's units", choices=tuple(sorted(UNITS))),
     Param("up", "string", "the mesh's up axis", default="z", choices=UP_AXES),
     Param("mass", "number", "mass in kg (default: cited category values or an ESTIMATE)"),
     Param("mass_source", "string", "where the mass comes from (datasheet, scale reading)"),
     Param("material", "string", "library material name or a word: wood, steel, glass, plastic, ceramic..."),
     Param("category", "string", "what it is (mug, bowl, wrench...): improves the estimates"),
     Param("out", "string", "card folder to write to", default="my_objects"),
     Param("check", "boolean", "run the alloy.check drop test", default=True),
     Param("device", "string", "'cpu' or 'cuda:0' for the drop test (default: the GPU if there is one)")),
    op_create,
    "Turn a mesh file, a photo or a text into an object card with a repaired watertight mesh, convex hulls, a "
    "visual GLB, and mass, friction and restitution with their sources (USER, CITED, LIBRARY or ESTIMATE with a "
    "range). Nothing is refused for a missing citation: estimates are labelled and listed under 'verify'. The card is "
    "drop-tested with alloy.check and written to `out` (use it with alloy.compose or Scene.add_object).",
    {"mesh": "part.stl", "units": "mm", "material": "steel", "out": "my_objects"},
    read_only=False, engine=True),)

__all__ = ["OPS", "op_create"]
