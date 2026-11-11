# SPDX-License-Identifier: Apache-2.0
"""LIQ-8 leak monitor (SPEC §5.2): SPH particles passing through walls, as a step observer.

A particle leaks in a step when its path crosses solid: the straight segment from its start to its end position,
taken in the frame of each boundary shape near it (start in the shape's start-of-step pose, end in its new pose, so
a moving cup carries its liquid without false events), reaches deeper than ``depth_tol`` (in particle radii) into
the shape. Positions are post-projection at both ends, so ordinary wall contact never reaches depth: a flat wall is
convex and the segment between two outside points stays outside. A genuine tunnel through a wall of thickness T
reaches T/2 (>= 2r for SPH walls, which are at least h = 4r thick); a cut across a convex edge in one step stays
shallow. Each particle counts once per episode (``leaked``); the deepest reach per world is a continuous signal for
an adversarial search. Escapes are the solver's own world-bounds counter (``report().escaped``); the monitor also
records through which face of the escape box each one left (bottom / side / top), in the step it left.
A particle that moved more than ``integrate.SWEEP`` h in the step is tested against every shape of its world, not
only its slots (the same coverage as the projection's sweep). Box walls are tested exactly at any step length (slab
interval, then the deepest point by ternary search); other shapes with samples at most r apart (up to 64).
In solid (``in_solid``): a particle that ends a step inside any shape of its world by more than ``IN_SOLID_TOL``
projection skins, counted once per episode, with the deepest end-of-step reach per world. The path depth above is
measured to the nearest face of one box, and inside a wall tiled from boxes that face can be an internal seam: a
particle left 1 um inside a box tank's y wall at its corner slid out along the seams at 1e-3 r path depth
(integrate.project_kernel). The end-of-step test sees it.
Deterministic: integer atomics and ``atomic_max`` only.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .boundary import ShapeArrays, shape_sdf, shape_world_transform
from .integrate import GEO_BOX, SWEEP, box_span, near_path, path_samples, path_sdf
from .pressure import Particles, is_fluid

DEPTH_TOL = 1.0  # [r] deeper than this along a step path = through the wall
SAMPLES = 8  # segment samples per shape (non-box shapes)
_TERNARY = 24  # box depth search steps ((2/3)^24 ~ 6e-5 of the inside span)
IN_SOLID_TOL = 0.25  # [projection skins] deeper than this inside a shape at the end of a step = in solid


@wp.func
def box_depth(shapes: ShapeArrays, s: int, l0: wp.vec3, l1: wp.vec3) -> float:
    """Exact deepest reach [m] of the segment l0 -> l1 into box ``s`` (its frame) at any length: the part inside
    (``box_span``) is searched by ternary search, as the depth min_a (half_a - |x_a|) is concave along a line."""
    span = box_span(shapes.scale[s], l0, l1)
    a = wp.max(span[0], 0.0)
    b = wp.min(span[1], 1.0)
    if a > b:
        return 0.0
    for _k in range(_TERNARY):
        m1 = a + (b - a) / 3.0
        m2 = b - (b - a) / 3.0
        if path_sdf(shapes, s, l0, l1, m1) > path_sdf(shapes, s, l0, l1, m2):
            a = m1
        else:
            b = m2
    return wp.max(-path_sdf(shapes, s, l0, l1, 0.5 * (a + b)), 0.0)


@wp.func
def path_depth(shapes: ShapeArrays, s: int, x0: wp.vec3, x1: wp.vec3, body_q_start: wp.array[wp.transform],
               body_q_end: wp.array[wp.transform], samples: int) -> float:
    """Deepest reach [m] into shape ``s`` along the step path (start in its start pose, end in its end pose)."""
    l0 = wp.transform_point(wp.transform_inverse(shape_world_transform(s, shapes.body, shapes.xform, body_q_start)),
                            x0)
    l1 = wp.transform_point(wp.transform_inverse(shape_world_transform(s, shapes.body, shapes.xform, body_q_end)), x1)
    if shapes.geo[s] == GEO_BOX:
        return box_depth(shapes, s, l0, l1)
    deepest = float(0.0)
    for k in range(samples + 1):
        deepest = wp.max(deepest, -path_sdf(shapes, s, l0, l1, float(k) / float(samples)))
    return deepest


@wp.kernel(enable_backward=False)
def leak_kernel(p: Particles, q_start: wp.array[wp.vec3], slot_shape: wp.array2d[wp.int32], shapes: ShapeArrays,
                body_q_start: wp.array[wp.transform], body_q_end: wp.array[wp.transform], radius: float,
                depth_tol: float, samples: int, sweep_len: float, has_bounds: int, lo: wp.vec3, hi: wp.vec3,
                leaked: wp.array[wp.int32], events: wp.array[wp.int32], max_depth: wp.array[wp.float32],
                exits: wp.array2d[wp.int32]):
    i = wp.tid()
    if not is_fluid(p, i):
        return
    x0 = q_start[i]
    x1 = p.q[i]
    deepest = float(0.0)
    step = wp.length(x1 - x0)
    if step > sweep_len:  # fast: every shape of the world, samples <= r apart
        n = path_samples(step, radius)
        for c in range(shapes.start[p.world[i]], shapes.start[p.world[i] + 1]):
            s = shapes.ids[c]
            X0 = shape_world_transform(s, shapes.body, shapes.xform, body_q_start)
            if near_path(shapes, s, X0, x0, step + radius):
                deepest = wp.max(deepest, path_depth(shapes, s, x0, x1, body_q_start, body_q_end, n))
    else:
        for b in range(4):
            s = slot_shape[i, b]
            if s >= 0 and s < shapes.soft.base:  # soft faces: not monitored in P1 (soft_walls.py)
                deepest = wp.max(deepest, path_depth(shapes, s, x0, x1, body_q_start, body_q_end, samples))
    w = p.world[i]
    wp.atomic_max(max_depth, w, deepest / radius)
    if deepest > depth_tol * radius and (leaked[i] & 1) == 0:
        leaked[i] = leaked[i] | 1
        wp.atomic_add(events, w, 1)
    if has_bounds != 0 and (leaked[i] & 2) == 0:  # leaving the escape box this step (retired right after)
        side = int(-1)
        if x1[2] < lo[2]:
            side = 0
        elif x1[2] > hi[2]:
            side = 2
        elif x1[0] < lo[0] or x1[0] > hi[0] or x1[1] < lo[1] or x1[1] > hi[1]:
            side = 1
        if side >= 0:
            leaked[i] = leaked[i] | 2
            wp.atomic_add(exits, w, side, 1)


@wp.kernel(enable_backward=False)
def in_solid_kernel(p: Particles, shapes: ShapeArrays, body_q: wp.array[wp.transform], radius: float, tol: float,
                    leaked: wp.array[wp.int32], events: wp.array[wp.int32], max_depth: wp.array[wp.float32]):
    """End-of-step reach into every shape of the particle's world (bounding-sphere filtered); module doc."""
    i = wp.tid()
    if not is_fluid(p, i):
        return
    x = p.q[i]
    w = p.world[i]
    deepest = float(0.0)
    for c in range(shapes.start[w], shapes.start[w + 1]):
        s = shapes.ids[c]
        X = shape_world_transform(s, shapes.body, shapes.xform, body_q)
        if near_path(shapes, s, X, x, 0.0):
            _lo, phi, _g = shape_sdf(shapes, s, wp.transform_point(wp.transform_inverse(X), x))
            deepest = wp.max(deepest, -phi)
    if deepest > tol:
        wp.atomic_max(max_depth, w, deepest / radius)
        if (leaked[i] & 4) == 0:
            leaked[i] = leaked[i] | 4
            wp.atomic_add(events, w, 1)


class LeakMonitor:
    """Per-world leak counts of one SolverDFSPH. Enabled with ``SolverDFSPH.enable_leak_monitor()``; the solver
    calls ``observe`` at the end of each step with its own states, so body and shape indices are the solver's even
    when it is a coupled entry on a model view."""

    def __init__(self, solver, depth_tol: float = DEPTH_TOL, samples: int = SAMPLES):
        self.solver, self.depth_tol, self.samples = solver, float(depth_tol), int(samples)
        d = solver.device
        n, W = max(solver.layout.n, 1), max(solver.layout.world_count, 1)
        self.leaked = wp.zeros(n, dtype=wp.int32, device=d)
        self.events = wp.zeros(W, dtype=wp.int32, device=d)
        self.max_depth = wp.zeros(W, dtype=wp.float32, device=d)
        self.exits = wp.zeros((W, 3), dtype=wp.int32, device=d)  # escape-box exits: bottom, side, top
        self.in_solid = wp.zeros(W, dtype=wp.int32, device=d)
        self.in_solid_depth = wp.zeros(W, dtype=wp.float32, device=d)
        b = solver.config.world_bounds
        self._bounds = (1, wp.vec3(*b[0]), wp.vec3(*b[1])) if b is not None else (0, wp.vec3(), wp.vec3())

    def reset(self) -> None:
        self.leaked.zero_()
        self.events.zero_()
        self.max_depth.zero_()
        self.exits.zero_()
        self.in_solid.zero_()
        self.in_solid_depth.zero_()

    def observe(self, state_in, state_out, body_q_in, body_q_out) -> None:
        """One step of the owning solver: start positions from ``state_in``, end positions and bodies after it."""
        s = self.solver
        if s.layout.n == 0:
            return
        wp.launch(leak_kernel, dim=s.layout.n,
                  inputs=[s._particles(state_out), state_in.particle_q, s._slots.shape, s.shapes.arrays, body_q_in,
                          body_q_out, s.config.particle_radius, self.depth_tol, self.samples, SWEEP * s.config.h,
                          *self._bounds,
                          self.leaked, self.events, self.max_depth, self.exits], device=s.device)
        from .solver import PROJECT_EPS  # noqa: PLC0415 (solver imports this module lazily)

        r = s.config.particle_radius
        wp.launch(in_solid_kernel, dim=s.layout.n,
                  inputs=[s._particles(state_out), s.shapes.arrays, body_q_out, r, IN_SOLID_TOL * PROJECT_EPS * r,
                          self.leaked, self.in_solid, self.in_solid_depth], device=s.device)

    def counts(self) -> dict[str, np.ndarray]:
        """Per world: through-wall particles, deepest path reach [r], escaped particles (world bounds) and where
        they left the escape box (bottom / side / top), particles that ended a step inside a shape and the deepest
        such reach [r] (module doc)."""
        W = self.solver.layout.world_count
        ex = self.exits.numpy()[:W]
        return {"leaks": self.events.numpy()[:W].copy(), "max_depth_r": self.max_depth.numpy()[:W].copy(),
                "escaped": np.asarray(self.solver.report().escaped).copy(), "exit_bottom": ex[:, 0].copy(),
                "exit_side": ex[:, 1].copy(), "exit_top": ex[:, 2].copy(),
                "in_solid": self.in_solid.numpy()[:W].copy(), "in_solid_depth_r": self.in_solid_depth.numpy()[:W].copy()}


__all__ = ["DEPTH_TOL", "IN_SOLID_TOL", "LeakMonitor", "in_solid_kernel", "leak_kernel"]
