# SPDX-License-Identifier: Apache-2.0
"""Host-side (numpy, float64) geometry for law checks: Newton transforms are (x, y, z, qx, qy, qz, qw).

Example::

    from alloy.plugins.npgeom import transform_point
    tip = transform_point(run.trace.body_q[-1, pen], (0.0, 0.0, -0.01))     # world position of the pen tip
"""

from __future__ import annotations

from typing import Any

import numpy as np


def quat_rotate(q: Any, v: Any) -> np.ndarray:
    """Rotate ``v [..., 3]`` by unit quaternions ``q [..., 4]`` (x, y, z, w); broadcasts."""
    q, v = np.asarray(q, np.float64), np.asarray(v, np.float64)
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def transform_point(xf: Any, p: Any) -> np.ndarray:
    """World position of body-frame point ``p`` for transforms ``xf [..., 7]``."""
    xf = np.asarray(xf, np.float64)
    return xf[..., :3] + quat_rotate(xf[..., 3:7], p)


__all__ = ["quat_rotate", "transform_point"]
