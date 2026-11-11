# SPDX-License-Identifier: Apache-2.0
"""Soft tet strain beyond the model range (coordinator ruling 2026-11-06T14:17Z, end of [internal note]).

Cut models here (seams, lattices, virtual-node copies) hold soft material together until a blade edge parts it, so a
flat face pushed into it squeezes it without limit: under WP7's random traces on ``cut_block_free`` (5e4 Pa block),
whole tets reached p90 85 % and max 208 % principal strain before a split. That is outside the model's range (known
gap: large compression of soft FEM under a flat tool; remeshing is post-launch), and the engine must not pass it
silently: the health monitor flags a world whose soft tets exceed ``SOFT_STRAIN_RANGE``.

``SoftStrain(model)`` measures, per world, the largest principal engineering strain over the live soft tets (nonzero
``tet_materials`` mu or lambda): max(stretch_max - 1, 1 - stretch_min) from the SVD of F, tension or compression;
an inverted tet counts as ``geometry.INVERTED_STRAIN`` (1e6). ``update(state)`` is two launches (graph-capturable) and returns
the device array (float32 [world_count]). WP5's health monitor owns the flag and its threshold; this module owns the
measure, as ``cut_front`` does for the PEN rule.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.topology.geometry import stretch_to_strain, tet_stretch_range

SOFT_STRAIN_RANGE = 0.6        # principal strain past which soft FEM leaves the model range (start value, ruling)


@wp.kernel
def tet_strain_max(tets: wp.array2d(dtype=wp.int32), poses: wp.array(dtype=wp.mat33),
                   materials: wp.array2d(dtype=wp.float32), world: wp.array(dtype=wp.int32),
                   x: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    """One thread per tet: its principal strain into its world's max (live tets only)."""
    t = wp.tid()
    if materials[t, 0] == 0.0 and materials[t, 1] == 0.0:
        return
    st = stretch_to_strain(tet_stretch_range(x[tets[t, 0]], x[tets[t, 1]], x[tets[t, 2]], x[tets[t, 3]], poses[t]))
    wp.atomic_max(out, world[t], wp.max(st[0], st[1]))


class SoftStrain:
    """Per-world max principal strain of a model's live soft tets (see the module docstring)."""

    def __init__(self, model: newton.Model) -> None:
        self.model = model
        n = int(model.tet_count)
        corner = model.tet_indices.numpy()[:, 0] if n else np.zeros(0, np.int64)
        world = np.maximum(model.particle_world.numpy()[corner], 0).astype(np.int32)   # -1 (global) is world 0
        self._world = wp.array(world, dtype=wp.int32, device=model.device)
        self.strain = wp.zeros(max(int(model.world_count), 1), dtype=wp.float32, device=model.device)

    def update(self, state: newton.State) -> wp.array:
        """[world_count] float32 on device: the strain now (0 for a world without live soft tets)."""
        m = self.model
        self.strain.zero_()
        if m.tet_count:
            wp.launch(tet_strain_max, dim=m.tet_count, device=m.device,
                      inputs=[m.tet_indices, m.tet_poses, m.tet_materials, self._world, state.particle_q, self.strain])
        return self.strain
