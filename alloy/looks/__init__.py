# SPDX-License-Identifier: Apache-2.0
"""Looks for cards: how an object renders, kept apart from its physics card.

Every card resolves to a look (``resolve``): the object's own look file if it has one, else its material's default
look, and the renderer's grey fallback only for a material that has no look file (logged, and ``coverage`` lists it).
Look files live beside the cards in each card root: ``looks/materials/<material>.json`` and
``looks/objects/[<area>/]<object>.json`` (``library/looks``, ``alloy-assets/looks``, an import root's ``looks``).
A look material is the WP20 spec that both renderers read: ``{"preset": <WP20 preset>, <override keys>}``.
Renderers take ``fallback_look(labels)``: assign rules in the WP20 look format that give each card's shapes its look,
consulted after a scene's own look so a shape no rule matched never falls back to grey while its material has a look.
``alloy.looks.pool`` is the texture pool: CC0 texture looks grouped in families (``looks/textures``), for scene
variants that swap a look for another texture of the same family.

Example::

    >>> from alloy.looks import resolve
    >>> look = resolve("mug_stoneware")          # doctest: +SKIP
    >>> look.source, look.spec()["preset"]        # doctest: +SKIP
    ('material', 'ceramic_white')
    >>> resolve("screwdriver_pz2").spec("grip")["base_color"]   # doctest: +SKIP
    [0.72, 0.06, 0.05]
"""

from alloy.looks.fallback import fallback_look
from alloy.looks.resolve import Look, coverage, material_look, object_look, resolve
from alloy.looks.schema import LookFileError, validate_look

__all__ = ["Look", "LookFileError", "coverage", "fallback_look", "material_look", "object_look", "resolve", "validate_look"]
