# SPDX-License-Identifier: Apache-2.0
"""Card names and output paths for RoboCasa assets.

Every card is ``robocasa_<snake name>``: ``glass_cup/GlassCup003`` -> ``robocasa_glass_cup_003``,
``saucepan/Saucepan002/Saucepan002_Lid`` -> ``robocasa_saucepan_002_lid``. Files go under the import root:
``objects/<category>/<name>.yaml`` (card), ``mjcf/<category>/<name>/model.xml`` (model, meshes, textures, hulls).

Example::

    >>> card_name("glass_cup/GlassCup003")
    'robocasa_glass_cup_003'
    >>> card_name("fridges/Refrigerator031")
    'robocasa_refrigerator_031'
    >>> model_rel("glass_cup/GlassCup003")
    'mjcf/glass_cup/robocasa_glass_cup_003/model.xml'
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath

PREFIX = "robocasa_"


def snake(text: str) -> str:
    """CamelCase/mixed text to lower snake case, digits split from letters."""
    s = re.sub(r"([a-z])([A-Z])", r"\1_\2", text)
    s = re.sub(r"([A-Za-z])(\d)", r"\1_\2", s)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def card_name(asset: str) -> str:
    """The Alloy card name of an asset id (last folder, snake-cased, prefixed)."""
    leaf = PurePosixPath(asset).name
    return PREFIX + snake(leaf)


def category(asset: str) -> str:
    """The first folder of an asset id."""
    return PurePosixPath(asset).parts[0]


def model_dir(asset: str) -> str:
    """Folder of the model and its meshes, relative to the import root."""
    return f"mjcf/{category(asset)}/{card_name(asset)}"


def model_rel(asset: str) -> str:
    """The MJCF path a card points at, relative to the import root."""
    return f"{model_dir(asset)}/model.xml"


def card_rel(asset: str) -> str:
    """The card file, relative to the import root."""
    return f"objects/{category(asset)}/{card_name(asset)}.yaml"


__all__ = ["PREFIX", "card_name", "card_rel", "category", "model_dir", "model_rel", "snake"]
