# SPDX-License-Identifier: Apache-2.0
"""The router's feature size (alloy.route.phase.feature_size): which rigid partners set a container inner size. A moving
solid primitive is a tool, not a cavity (SOFT-2, SC10: a kinematic 5 mm rod in a tub of mayonnaise asked for 830,208
MPM particles, a 1.5 mm whisk wire 31 M); static boxes (add_open_box's walls) and meshes (cups) still count."""

from __future__ import annotations

from dataclasses import replace

import pytest

from alloy.cards.types import MeshSource
from alloy.route.phase import feature_size
from alloy.testing import route as T

MAT = T.material("paste_like", 1000.0)
PASTE = T.box("paste", MAT, 0.05, 0.05, 0.015)          # 100 x 100 x 30 mm: its own size 30 mm


def _size(partner) -> float:
    ctx = T.context(frozenset(), touching={"paste": (partner.card.name,)}, partners={partner.card.name: partner})
    return feature_size(PASTE, ctx)[0]


@pytest.mark.parametrize("motion", ["kinematic", "dynamic"])
@pytest.mark.parametrize("radius", [0.005, 0.0025, 0.00075])
def test_a_moving_rod_tool_sets_no_inner_size(motion, radius):
    assert _size(T.rod("rod", MAT, radius, 0.1, motion=motion)) == pytest.approx(0.03)


def test_a_moving_box_tool_sets_no_inner_size():
    assert _size(T.box("spatula", MAT, 0.015, 0.05, 0.001, motion="kinematic")) == pytest.approx(0.03)


def test_static_boxes_still_bound_the_size():
    """add_open_box's walls are static boxes: their middle extent is the cavity's width (here 20 mm < 30 mm)."""
    assert _size(T.box("wall", MAT, 0.05, 0.01, 0.002, motion="static")) == pytest.approx(0.02)


def test_a_moving_mesh_still_bounds_the_size():
    """A cup is a mesh that may hold the paste: its middle extent stays a container inner size."""
    box = T.box("cup", MAT, 0.04, 0.01, 0.01)
    cup = replace(box, card=replace(box.card, shape=MeshSource(path="cup.obj", sha256="")), extents=(0.08, 0.02, 0.02))
    assert _size(cup) == pytest.approx(0.02)
