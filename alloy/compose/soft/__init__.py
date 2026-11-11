# SPDX-License-Identifier: Apache-2.0
"""Soft items in composed scenes: liquids filled into containers or poured, grain fills and piles, cloth laid flat,
draped, hung over the table edge or folded, and cables with a length, a gauge and attached ends.

The parser (``alloy.compose.soft.parse``) turns "a cup with 200 ml of water" into an item with a ``SoftSpec``; the
builder makes a card for it from a library material or template card (``alloy.compose.soft.cards``), puts it in or
on what it belongs to, and the soft probe (``alloy.compose.soft.probe``) checks it stays where it should, keeps its
volume, mass or length, and settles. The engine's router picks the solver from the card.

Example::

    >>> from alloy.compose.soft.spec import SoftSpec
    >>> SoftSpec("grain", "pile", mass_g=300).to_dict()
    {'kind': 'grain', 'form': 'pile', 'mass_g': 300}
"""
