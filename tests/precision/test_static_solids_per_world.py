# SPDX-License-Identifier: Apache-2.0
"""Regression test: solid_ids must keep static decomposition pieces separate across worlds."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from alloy.precision.health_source_shapes import COLLIDE, solid_ids


def test_solid_ids_keeps_decomposed_static_shapes_distinct_across_worlds():
    """Static decomposition pieces matching SOURCE_PART_RULES in different worlds must not merge across worlds."""
    model = SimpleNamespace(
        shape_count=2,
        shape_body=SimpleNamespace(numpy=lambda: np.asarray([-1, -1])),
        shape_world=SimpleNamespace(numpy=lambda: np.asarray([0, 1])),
        shape_flags=SimpleNamespace(numpy=lambda: np.asarray([COLLIDE, COLLIDE])),
        shape_label=["plate_box0", "plate_box1"],
    )
    ids = solid_ids(model)
    np.testing.assert_array_equal(ids, [0, 1])
