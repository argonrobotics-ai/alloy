# SPDX-License-Identifier: Apache-2.0
"""Import RoboCasa's kitchen objects and articulated fixtures as Alloy cards.

Only packs with a clear, allowed licence are imported (``packs``): the Lightwheel objects and fixtures (CC BY 4.0).
For each asset the pipeline (``pipeline.import_asset``) extracts it from the pack zip (``source``), parses the MJCF
(``mjcf_in``), computes CoACD collision hulls with at most 64 vertices each (``hulls``), the mass and inertia from
per-part densities (``materials``, ``massprops``; tagged ESTIMATE), writes an Alloy MJCF with ``.msh`` meshes
(``mjcf_out``, ``meshio``) and a card with its licence header (``cards_out``), checks it in MuJoCo (``checks``: a
drop test for objects, a joint sweep for fixtures) and renders a thumbnail (``thumbs``). Fixtures keep their doors,
drawers and knobs as hinge/slide joints and become ``motion: static`` cards.

Example (command line, JSON out)::

    python -m alloy.importers.robocasa list --pack objects_lightwheel --category glass_cup
    python -m alloy.importers.robocasa batch --pack objects_lightwheel --category glass_cup --out imports/robocasa
    python -m alloy.importers.robocasa finalize --out imports/robocasa

Example (Python)::

    >>> from pathlib import Path
    >>> from alloy.cards import CardLibrary, default_root
    >>> from alloy.importers.robocasa import import_asset
    >>> rec = import_asset("objects_lightwheel", "glass_cup/GlassCup003", Path("imports/robocasa"))  # doctest: +SKIP
    >>> lib = CardLibrary([default_root(), Path("imports/robocasa")])                              # doctest: +SKIP
    >>> lib.object("robocasa_glass_cup_003").material                                              # doctest: +SKIP
    'glass_soda_lime'
"""

from __future__ import annotations

from alloy.importers.robocasa.packs import PACKS, kept_packs
from alloy.importers.robocasa.pipeline import import_asset

__all__ = ["PACKS", "import_asset", "kept_packs"]
