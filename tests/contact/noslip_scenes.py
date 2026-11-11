# SPDX-License-Identifier: Apache-2.0
"""Small scenes for the scoped noslip tests (CPU and CUDA): two boxes on a plank (geom-pair scope, skipped trees) and a
hinged box snake on a slope (one tree over 8 dofs: the warp-wide factor)."""

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.testing.scenes import RIGID_DT, RefScene


def two_boxes_on_plank(theta: float = 0.4, mu: float = 0.8, device: str = "cpu") -> RefScene:
    """Template shapes: 0 plank (static), 1 box A, 2 box B; both boxes stick (tan theta < mu)."""
    b = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=500.0)
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta)
    b.add_shape_box(-1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), tilt), hx=1.0, hy=0.5, hz=0.05, cfg=cfg)
    for y in (-0.2, 0.2):
        p = wp.vec3(0.1 * np.sin(theta), y, 0.5 + 0.1 * np.cos(theta))
        b.add_shape_box(b.add_body(xform=wp.transform(p, tilt)), hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    root = newton.ModelBuilder()
    root.replicate(b, 1)
    return RefScene("two_boxes", root.finalize(device=device), RIGID_DT, "mujoco", {}, {})


def snake_on_slope(links: int = 5, worlds: int = 1, device: str = "cpu") -> RefScene:
    """A free root box plus ``links - 1`` boxes on yaw hinges (6 + links - 1 dofs, one tree) lying on the ground,
    gravity tilted 0.2 rad (tan 0.2 < mu 0.8: it sticks)."""
    b = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, density=500.0)
    parent, joints = -1, []
    for k in range(links):
        body = b.add_link(xform=wp.transform(wp.vec3(0.11 * k, 0.0, 0.03), wp.quat_identity()))
        b.add_shape_box(body, hx=0.05, hy=0.03, hz=0.03, cfg=cfg)
        if parent < 0:
            joints.append(b.add_joint_free(child=body))
        else:
            joints.append(b.add_joint_revolute(parent=parent, child=body, axis=(0.0, 0.0, 1.0),
                                               parent_xform=wp.transform(wp.vec3(0.055, 0.0, 0.0), wp.quat_identity()),
                                               child_xform=wp.transform(wp.vec3(-0.055, 0.0, 0.0), wp.quat_identity())))
        parent = body
    b.add_articulation(joints)
    root = newton.ModelBuilder(gravity=(9.81 * np.sin(0.2), 0.0, -9.81 * np.cos(0.2)))
    root.replicate(b, worlds)
    root.add_ground_plane(cfg=cfg)
    return RefScene("snake", root.finalize(device=device), RIGID_DT, "mujoco", {}, {})
