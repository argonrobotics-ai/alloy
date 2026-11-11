# SPDX-License-Identifier: Apache-2.0
"""The labelled things of the barcode tasks (Gym-Anything gap "camera / imager in the loop"): a shelf location label,
a retail item box and the 75 cl wine bottle of the held-pour rig with its EAN-13, and the scenes that hold them
next to the held-pour FR3 (``held_pour_grip``: the wine proof's bar and arm mount).

Label sizes follow the standards and Gym-Anything's artwork: retail items carry an EAN-13 at 100 % magnification
(X = 0.330 mm, bars 22.85 mm); the shelf location label is 76.2 x 25.4 mm with a Code 128 at X = 0.381 mm (15 mil)
and 11 mm bars. GTINs are made up (fictional brands, valid check digits).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from alloy.tasks.barcode_scene import Decal
from alloy.tasks.barcode_symbols import Symbol, label_image

LOCATION = Symbol("A-01-03", "Code128", 0.381e-3, 11.0e-3)
LOCATION_LABEL = (0.0762, 0.0254)            # m
ITEM = Symbol("400638133393")                 # EAN-13 (check digit 1 added)
WINE = Symbol("376012345001")                 # EAN-13 (check digit added)
POST = (0.10, 0.01, 0.30)                     # m, full edges: the static post the location label is on
POST_XY = (-0.10, 0.565)                      # front faces of post and box at y = 0.56
BOX = (0.07, 0.05, 0.28)                      # m, full edges: the item (a cereal-box-like carton), 0.40 kg
BOX_KG = 0.40
BOX_XY = (0.08, 0.585)
LABEL_Z = 0.22                                # m above the bar: centre height of both shelf labels
TOWARD_ROBOT = (0.0, -1.0, 0.0)               # the labels face the arm (it stands on -y, facing +y)


@dataclass(frozen=True)
class Labelled:
    """A label to read: its decal and the text a correct read gives (``barcode_decode`` normalisation)."""

    decal: Decal
    symbol: Symbol

    def expected(self) -> str:
        from alloy.tasks.barcode_decode import normalise

        return normalise(self.symbol.kind, self.symbol.text())


def _label(symbol: Symbol, size: tuple[float, float], body: str, center, normal, radius: float = 0.0) -> Labelled:
    return Labelled(Decal(body, label_image(symbol, size, 6), size, tuple(center), tuple(normal), radius=radius),
                    symbol)


def shelf_labels() -> tuple[Labelled, Labelled]:
    """(location label on the post, fixed in the world; EAN-13 on the item box), both facing the arm, centres at
    ``LABEL_Z``."""
    loc = _label(LOCATION, LOCATION_LABEL, "", (POST_XY[0], POST_XY[1] - 0.5 * POST[1], LABEL_Z), TOWARD_ROBOT)
    w, h = ITEM.size()
    item = _label(ITEM, (w + 0.004, h + 0.008), "item", (0.0, -0.5 * BOX[1], LABEL_Z - 0.5 * BOX[2]), TOWARD_ROBOT)
    return loc, item


def add_shelf(b) -> int:
    """The post (fixed, on the bar; its label is world-fixed) and the item box on the bar: the item's body."""
    import warp as wp

    from alloy._vendor import newton

    b.add_shape_box(-1, xform=wp.transform((*POST_XY, 0.5 * POST[2]), wp.quat_identity()), hx=0.5 * POST[0],
                    hy=0.5 * POST[1], hz=0.5 * POST[2], label="post",
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
    item = b.add_body(xform=wp.transform((*BOX_XY, 0.5 * BOX[2] + 2.0e-4), wp.quat_identity()), label="item")
    vol = float(np.prod(BOX))
    b.add_shape_box(item, hx=0.5 * BOX[0], hy=0.5 * BOX[1], hz=0.5 * BOX[2], label="item_box",
                    cfg=newton.ModelBuilder.ShapeConfig(density=BOX_KG / vol, mu=0.5))
    return item


def bottle_label(grasp_in_bottle, radius: float = 0.0385, height: float = 0.065) -> Labelled:
    """The wine's EAN-13 wrapped round the bottle at ``height``, on the side away from the hand (the grasp's
    approach direction, made horizontal), bars along the bottle (picket fence)."""
    approach = Rotation.from_quat(np.asarray(grasp_in_bottle, np.float64)[3:]).apply((0.0, 0.0, 1.0))
    n = np.array([approach[0], approach[1], 0.0])
    n /= np.linalg.norm(n)
    w, h = WINE.size()
    return _label(WINE, (w + 0.006, h + 0.008), "bottle", n * radius + np.array([0.0, 0.0, height]), n, radius)


__all__ = ["BOX", "ITEM", "LOCATION", "Labelled", "WINE", "add_shelf", "bottle_label", "shelf_labels"]
