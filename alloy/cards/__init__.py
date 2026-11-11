# SPDX-License-Identifier: Apache-2.0
"""Object cards (WP1): describe any object once, in SI, with sourced values; types live in ``alloy.cards.types`` (WP0).

- ``loader``: ``load_material/pair/object/scene``, ``loads_card``, ``dump_card`` (canonical YAML, ``load(dump(c)) == c``).
- ``validate``: ``validate_material/pair/object/scene`` -> ``CardIssue`` lists (SPEC §3.2 rules, fields named).
- ``library``: ``CardLibrary`` over ``library/`` (materials, pairs, objects, scenes, meshes), ``resolve``, ``put``.
- ``roots``: the alloy-assets checkout (``$ALLOY_ASSETS``, else ``<repo>/../alloy-assets``), read by ``CardLibrary()``
  as one more root (cards found recursively, meshes relative to it); a name in both it and ``library/`` raises.
- ``identity``: ``ResolvedObject`` (extents, worst-case randomization, default leaves, ``card_hash``).
- ``paths``: the dotted card-path grammar (``material.elastic.youngs``, ``loads.grip_force``, ``pose.2``).
- ``mass``: mass implied by a card (Newton's inertia routine); ``geometry``/``meshgen``: mesh facts and library meshes.

YAML: a quantity is a bare number/unit string (source ``default``) or ``{value, source, unc, note}``; a shape is a
one-key mapping naming its variant (``primitive``, ``mesh``, ``sdf``, ``tet``, ``curve``, ``particles``, ``usd``,
``mjcf``). Unknown or duplicate keys are errors.
"""

from __future__ import annotations

from alloy.cards.library import CardLibrary, default_root
from alloy.cards.loader import dump_card, load_material, load_object, load_pair, load_scene, loads_card
from alloy.cards.roots import assets_root
from alloy.cards.validate import validate_material, validate_object, validate_pair, validate_scene

__all__ = [
    "CardLibrary", "assets_root", "default_root", "dump_card", "load_material", "load_object", "load_pair", "load_scene",
    "loads_card", "validate_material", "validate_object", "validate_pair", "validate_scene",
]
