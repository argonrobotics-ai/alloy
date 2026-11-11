# SPDX-License-Identifier: Apache-2.0
"""Camera poses and lens maths. Camera frame: +X right, +Y up, looking along -Z (Blender / OpenGL convention).

Example::

    import numpy as np
    from alloy.camera.pose import look_at, image_extent
    xf = look_at(eye=(2.0, -2.0, 1.5), target=(0.0, 0.0, 0.3))      # (px, py, pz, qx, qy, qz, qw)
    hx, hy = image_extent(fov_deg=60.0, width=640, height=480)      # tan(half angles)
    assert abs(hy - np.tan(np.radians(30))) < 1e-9
"""
from __future__ import annotations

import math

import numpy as np

from .errors import CameraError
from .scene import matrix_to_quat


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Rotation whose columns are the camera x (right), y (up), z (back) axes in world; the camera looks at target."""
    eye, target, up = (np.asarray(v, np.float64) for v in (eye, target, up))
    f = target - eye
    if np.linalg.norm(f) < 1e-12:
        raise CameraError("eye and target are the same point", "move the eye away from the target")
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-9:  # looking straight along up: pick any right vector
        r = np.cross(f, [1.0, 0.0, 0.0] if abs(f[0]) < 0.9 else [0.0, 1.0, 0.0])
    r /= np.linalg.norm(r)
    return np.stack([r, np.cross(r, f), -f], axis=1)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera pose (px, py, pz, qx, qy, qz, qw) at ``eye`` looking at ``target`` with ``up`` roughly up."""
    q = matrix_to_quat(look_at_matrix(eye, target, up))
    return np.concatenate([np.asarray(eye, np.float64), q]).astype(np.float32)


def image_extent(fov_deg: float, width: int, height: int, fit: str = "VERTICAL") -> tuple[float, float]:
    """Half extents of the image plane at distance 1. ``fov_deg`` spans the image height (VERTICAL), the width
    (HORIZONTAL) or the longer side (AUTO, Blender's sensor fit)."""
    if not 0.0 < fov_deg < 180.0:
        raise CameraError(f"fov {fov_deg} deg", "use a field of view between 0 and 180 degrees (60 is typical)")
    if width < 1 or height < 1:
        raise CameraError(f"image size {width}x{height}", "use a width and height of at least 1 pixel")
    t = math.tan(0.5 * math.radians(fov_deg))
    if fit == "AUTO":
        fit = "HORIZONTAL" if width >= height else "VERTICAL"
    if fit == "VERTICAL":
        return t * width / height, t
    if fit == "HORIZONTAL":
        return t, t * height / width
    raise CameraError(f"sensor fit {fit!r}", "use 'VERTICAL', 'HORIZONTAL' or 'AUTO'")


def as_poses(pose, n: int) -> np.ndarray:
    """One pose (7,) or per-world poses (n, 7) -> (n, 7) float32, quaternions normalised."""
    a = np.asarray(pose, np.float64)
    if a.shape == (7,):
        a = np.tile(a, (n, 1))
    if a.shape != (n, 7):
        raise CameraError(f"pose has shape {a.shape}", f"pass (px, py, pz, qx, qy, qz, qw) or an ({n}, 7) array")
    q = a[:, 3:]
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if not np.isfinite(a).all() or (norm < 1e-9).any():
        raise CameraError("pose has a NaN or zero quaternion", "pass a unit quaternion (qx, qy, qz, qw)")
    a[:, 3:] = q / norm
    return a.astype(np.float32)


def auto_frame(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """A three-quarter view framing the box [lo, hi] (used by ``snapshot`` when no pose is given)."""
    target = 0.5 * (lo + hi)
    span = max(float(np.linalg.norm(hi - lo)), 0.3)
    return look_at(target + span * np.array([0.9, -1.1, 0.8]), target)
