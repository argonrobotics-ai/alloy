# SPDX-License-Identifier: Apache-2.0
"""GridDescriptor's constructor (alloy.solvers.sph.grid_descriptor), CPU: the grid alone, or with 05baba777's
(grid, dim, device) arguments, which plug-ins already pass (the species plug-in); a dim that is not the grid's
raises. On the CPU the guard is off and a build is the plain grid build."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

from alloy.solvers.sph.grid_descriptor import GridDescriptor


def test_both_signatures_build_and_a_wrong_dim_raises():
    grid = wp.HashGrid(16, 16, 16, device="cpu")
    for desc in (GridDescriptor(grid), GridDescriptor(grid, 16, "cpu")):
        assert desc.dims == [16, 16, 16] and not desc.on
        desc.build(wp.array(np.zeros((4, 3), np.float32), dtype=wp.vec3, device="cpu"), 0.1)
    with pytest.raises(ValueError, match="dim 32"):
        GridDescriptor(grid, 32, "cpu")
