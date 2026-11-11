# SPDX-License-Identifier: Apache-2.0
"""Import realistic, licence-clean objects from Objaverse 1.0 into Alloy object cards.

Pipeline (each stage is a JSON-in/JSON-out command of ``python -m alloy.importers.objaverse``):

1. ``select``: Objaverse annotations + LVIS labels -> candidates. Keeps CC0 / CC BY 4.0 uploads in 270 robot-relevant
   categories (kitchen, lab, tools, containers, food, office, packaging), textured, not animated, no brand/game terms;
   ranked by likes, views, staff picks and scan/PBR tags, capped per category.
2. ``survey`` (batch, ``worker.py``): download each GLB, reject untextured meshes, scenes, display bases and degenerate
   proportions; render a preview.
3. review: numbered contact sheets of the previews; a reviewer rejects wrong labels, readable brands, props on boards
   and fantasy items (:mod:`alloy.importers.objaverse.review`).
4. mesh stage (shared pipeline, ``alloy.importers.common``): watertight ``physics.obj``, CoACD ``physics.hulls.obj``,
   textured ``visual.glb``, ``thumb.png``, at the category's real-world size.
5. ``cards``: ``alloy.object/1`` cards with ``collision: decompose``, base or effective materials (mass rule in
   :mod:`alloy.importers.objaverse.physical`), CC attribution in each card header, ESTIMATE rows for the sourcing agent.

Example:
    >>> from alloy.importers.objaverse import check_licence, load_categories, screen
    >>> check_licence("by").spdx, len(load_categories()) >= 200
    ('CC-BY-4.0', True)
    >>> screen({"uid": "u", "license": "cc0", "lvis": ["banana"], "name": "Banana scan", "faceCount": 9000,
    ...         "glb_textures": 1, "glb_size": 1e6, "user_display": "Ann"}).category
    'banana'
"""

from __future__ import annotations

from alloy.importers.objaverse.categories import Category, load_categories
from alloy.importers.objaverse.errors import ImporterConfigError, ImportRejected
from alloy.importers.objaverse.licence import attribution, check_licence
from alloy.importers.objaverse.physical import choose_mass, scale_to_category
from alloy.importers.objaverse.select import Candidate, Limits, screen

__all__ = [
    "Candidate", "Category", "ImportRejected", "ImporterConfigError", "Limits", "attribution", "check_licence",
    "choose_mass", "load_categories", "scale_to_category", "screen",
]
