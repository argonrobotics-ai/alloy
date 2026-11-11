# SPDX-License-Identifier: Apache-2.0
"""Scene pieces for the sports plug-ins: a football, launches, the goal frame.

``add_ball`` builds a body with the thin-shell inertia and a sphere that the solver does not collide with shapes
(``PressureBall`` owns those contacts) but particles still meet (a cloth net catches it).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.harness.scene import body_xform
from alloy.plugins.sports import model as M

Vec = tuple[float, float, float]

# IFAB Law 1: 7.32 m between the inside of the posts, 2.44 m to the lower edge of the bar, posts <= 12 cm wide.
GOAL_WIDTH, GOAL_HEIGHT, POST_R = 7.32, 2.44, 0.06
PENALTY_MARK = 11.0                    # m from the goal line (IFAB Law 1)


def ball_shape_config() -> newton.ModelBuilder.ShapeConfig:
    return newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=True)


def add_ball(b: newton.ModelBuilder, pos: Sequence[float], label: str = "ball", radius: float = M.R_BALL,
             mass: float = M.M_BALL) -> int:
    """A ball body at ``pos`` with I = ⅔·m·R² (thin shell) and a solver sphere that only particles meet."""
    i = M.I_FACTOR * mass * radius * radius
    body = b.add_body(xform=body_xform(*pos), mass=mass, inertia=wp.mat33(i, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, i),
                      lock_inertia=True, label=label)
    b.add_shape_sphere(body, radius=radius, cfg=ball_shape_config(), label=label)
    return body


def launch(per_world: Sequence[tuple[Vec, Vec]], label: str = "ball") -> Callable[[Any], None]:
    """``init(sim)``: world w's ``label`` body starts with (linear, angular) velocity ``per_world[w]`` (the last
    entry repeats). The body must hang on a free joint (``add_ball``)."""
    def init(sim: Any) -> None:
        model = sim.model
        starts, child = model.joint_qd_start.numpy(), model.joint_child.numpy()
        qd = sim.state.joint_qd.numpy()
        world = model.body_world.numpy()
        for b in np.flatnonzero(np.array(list(model.body_label)) == label):
            v, w = per_world[min(int(world[b]), len(per_world) - 1)]
            j = int(np.flatnonzero(child == b)[0])
            qd[starts[j]:starts[j] + 6] = (*v, *w)
        sim.state.joint_qd.assign(qd)
        newton.eval_fk(model, sim.state.joint_q, sim.state.joint_qd, sim.state)
    return init


def with_inputs(init: Callable[[Any], None] | None, plugin: str, **inputs: Any) -> Callable[[Any], None]:
    """``init`` followed by ``sim.plugins[plugin].set_input(name, value)`` for each input."""
    def run(sim: Any) -> None:
        if init is not None:
            init(sim)
        for name, value in inputs.items():
            sim.plugins[plugin].set_input(name, value)
    return run


def add_goal(b: newton.ModelBuilder, x: float = 0.0, cfg: newton.ModelBuilder.ShapeConfig | None = None) -> list[str]:
    """Static posts and bar (cylinders, IFAB size) with the goal line at ``x``, the goal opening towards -x;
    returns the shape labels (list them as ``PressureBall`` colliders)."""
    cfg = cfg or newton.ModelBuilder.ShapeConfig(has_particle_collision=False)
    half = 0.5 * GOAL_WIDTH + POST_R
    for side, y in (("post_l", half), ("post_r", -half)):
        b.add_shape_cylinder(-1, xform=body_xform(x + POST_R, y, 0.5 * (GOAL_HEIGHT + POST_R)), radius=POST_R,
                             half_height=0.5 * (GOAL_HEIGHT + POST_R), cfg=cfg, label=side)
    bar_q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.5 * np.pi)
    b.add_shape_cylinder(-1, xform=wp.transform(wp.vec3(x + POST_R, 0.0, GOAL_HEIGHT + POST_R), bar_q),
                         radius=POST_R, half_height=half + POST_R, cfg=cfg, label="bar")
    return ["post_l", "post_r", "bar"]


__all__ = ["GOAL_HEIGHT", "GOAL_WIDTH", "PENALTY_MARK", "POST_R", "add_ball", "add_goal", "ball_shape_config",
           "launch", "with_inputs"]
