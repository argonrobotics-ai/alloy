# SPDX-License-Identifier: Apache-2.0
"""Wave-3 stubs (SPEC §7 WP3): ``ColliderField`` (rigid colliders as a distance/velocity field for particle solvers)
and the CROSS_PARTICLE contact stream. Both raise ``NotImplementedError`` until wave 3; their tests are xfail."""

from __future__ import annotations

import warp as wp

from alloy.core.log import get_logger

log = get_logger("contact.field")

WAVE3 = "wave 3 (SPEC §7 WP3): ColliderField / CROSS_PARTICLE are stubs"


class ColliderField:
    """Signed distance, outward normal, surface velocity and shape id of the nearest rigid collider per query point."""

    def __init__(self, layer) -> None:
        self.layer = layer

    def query(self, points: wp.array, world: wp.array, out_dist: wp.array, out_normal: wp.array,
              out_vel: wp.array, out_shape: wp.array) -> None:
        raise NotImplementedError(WAVE3)


def append_cross_particle(layer, particle_q: wp.array, particle_radius: wp.array, contacts) -> None:
    """Append particle-vs-precision-shape rows as ``ContactClass.CROSS_PARTICLE`` (wave 3)."""
    raise NotImplementedError(WAVE3)
