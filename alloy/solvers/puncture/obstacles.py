# SPDX-License-Identifier: Apache-2.0
"""Thread contact with rigid shapes (SC11, C19: the jaws, the needle, the fingers and the riser a thread drapes over).

A thread's particles meet the pad through VBD's self-contact; nothing else touches them. A clip gives the plug-in the
rigid shapes its threads must not pass through as analytic obstacles, each fixed to a body or to the world:
- BOX: an oriented box (half extents ``size``) at ``xform`` in its body's frame;
- ARC: the centreline of needle ``needle`` (the plug-in's needle geometry), a tube of its wire radius.

In the chain solve (chain.py) a thread particle of radius r at signed distance d < r from an obstacle gets a penalty
½ k (r − d)² along the obstacle's outward normal, and smoothed Coulomb friction μ f_n on its motion over the step
relative to the obstacle's surface point (IPC's lagged friction, ε_u = ε_v dt). The first ``skip`` particles of a
thread ignore an obstacle (the needle's own swage crimps them). Not booked by the energy ledger, like VBD's contacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

from . import needle as N
from .needle import Needles

BOX = 0
ARC = 1


@dataclass(frozen=True)
class Obstacle:
    kind: int                                       # BOX or ARC
    world: int                                      # the threads of this world collide with it
    body: int = -1                                  # BOX: -1 for the world
    xform: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)   # BOX pose in its body frame (p, q xyzw)
    size: tuple = (0.0, 0.0, 0.0)                   # BOX half extents [m]
    needle: int = -1                                # ARC: index into the solver's needles
    skip: int = 0                                   # the thread's first `skip` particles ignore it


@dataclass(frozen=True)
class ObstacleConfig:
    k: float = 1.0e5           # penalty stiffness [N/m]: 2.5 N of thread tension round an edge sinks it 0.035 mm
    mu: float = 0.2            # thread-on-steel friction (model value)
    items: tuple[Obstacle, ...] = field(default_factory=tuple)


@wp.struct
class Obstacles:
    start: wp.array[wp.int32]       # per thread (chain index): first obstacle
    count: wp.array[wp.int32]
    kind: wp.array[wp.int32]
    body: wp.array[wp.int32]
    xform: wp.array[wp.transform]
    size: wp.array[wp.vec3]
    needle: wp.array[wp.int32]
    skip: wp.array[wp.int32]
    k: float
    mu: float


def build(cfg: ObstacleConfig, thread_worlds: list[int], device) -> Obstacles:
    """Obstacles grouped per thread (a thread collides with every obstacle of its world)."""
    rows = [o for w in thread_worlds for o in cfg.items if o.world == w]
    counts = [sum(1 for o in cfg.items if o.world == w) for w in thread_worlds]
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) if counts else np.zeros(0)

    ob = Obstacles()
    ob.start = wp.array(np.asarray(starts if len(starts) else [0], np.int32), dtype=wp.int32, device=device)
    ob.count = wp.array(np.asarray(counts if counts else [0], np.int32), dtype=wp.int32, device=device)
    ob.kind = wp.array(np.asarray([o.kind for o in rows] or [0], np.int32), dtype=wp.int32, device=device)
    ob.body = wp.array(np.asarray([o.body for o in rows] or [-1], np.int32), dtype=wp.int32, device=device)
    ob.xform = wp.array(np.asarray([o.xform for o in rows] or [(0, 0, 0, 0, 0, 0, 1)], np.float32),
                        dtype=wp.transform, device=device)
    ob.size = wp.array(np.asarray([o.size for o in rows] or [(0, 0, 0)], np.float32), dtype=wp.vec3, device=device)
    ob.needle = wp.array(np.asarray([o.needle for o in rows] or [-1], np.int32), dtype=wp.int32, device=device)
    ob.skip = wp.array(np.asarray([o.skip for o in rows] or [0], np.int32), dtype=wp.int32, device=device)
    ob.k = float(cfg.k)
    ob.mu = float(cfg.mu)
    return ob


@wp.func
def _side(a: float) -> float:
    return wp.where(a < 0.0, -1.0, 1.0)


@wp.func
def box_distance(p: wp.vec3, h: wp.vec3) -> wp.vec4:
    """Signed distance (w) and outward unit normal (xyz) of a box of half extents h at point p (box frame)."""
    q = wp.vec3(wp.abs(p[0]) - h[0], wp.abs(p[1]) - h[1], wp.abs(p[2]) - h[2])
    m = wp.max(q[0], wp.max(q[1], q[2]))
    if m <= 0.0:                                                   # inside: out through the nearest face
        if q[0] >= q[1] and q[0] >= q[2]:
            return wp.vec4(_side(p[0]), 0.0, 0.0, m)
        if q[1] >= q[2]:
            return wp.vec4(0.0, _side(p[1]), 0.0, m)
        return wp.vec4(0.0, 0.0, _side(p[2]), m)
    o = wp.vec3(wp.max(q[0], 0.0) * _side(p[0]), wp.max(q[1], 0.0) * _side(p[1]), wp.max(q[2], 0.0) * _side(p[2]))
    d = wp.length(o)
    u = o / d
    return wp.vec4(u[0], u[1], u[2], d)


@wp.func
def frame(ob: Obstacles, nd: Needles, o: int, body_q: wp.array[wp.transform]) -> wp.transform:
    """World <- obstacle frame (BOX: body pose x xform; ARC: the needle's body pose)."""
    if ob.kind[o] == ARC:
        return N.pose(nd, ob.needle[o], body_q)
    b = ob.body[o]
    if b < 0:
        return ob.xform[o]
    return wp.transform_multiply(body_q[b], ob.xform[o])


@wp.func
def distance(ob: Obstacles, nd: Needles, o: int, pl: wp.vec3) -> wp.vec4:
    """Signed distance (w) and outward normal (xyz) of obstacle o at point pl in its frame."""
    if ob.kind[o] == ARC:
        i = ob.needle[o]
        s = wp.clamp(N.project_local(nd, i, pl), 0.0, nd.length[i])
        v = pl - N.point_local(nd, i, s)
        l = wp.length(v)
        u = v / wp.max(l, 1.0e-12)
        return wp.vec4(u[0], u[1], u[2], l - nd.wire[i])
    return box_distance(pl, ob.size[o])


@wp.func
def contact_terms(ob: Obstacles, nd: Needles, o: int, p: wp.vec3, p0: wp.vec3, radius: float,
                  body_q: wp.array[wp.transform], body_q0: wp.array[wp.transform], eps_u: float):
    """Penalty and friction of obstacle o on a particle at p (p0 at the step start): (force, Hessian)."""
    X = frame(ob, nd, o, body_q)
    pl = wp.transform_point(wp.transform_inverse(X), p)
    dn = distance(ob, nd, o, pl)
    gap = dn[3] - radius
    if gap >= 0.0:
        return wp.vec3(0.0), wp.mat33(0.0)
    n = wp.transform_vector(X, wp.vec3(dn[0], dn[1], dn[2]))
    fn = ob.k * (-gap)
    nn = wp.outer(n, n)
    f = fn * n
    h = ob.k * nn
    if ob.mu > 0.0:
        X0 = frame(ob, nd, o, body_q0)
        u = (p - p0) - (wp.transform_point(X, pl) - wp.transform_point(X0, pl))
        ut = u - wp.dot(u, n) * n
        lt = wp.length(ut)
        c = ob.mu * fn / wp.max(lt, eps_u)
        f = f - c * ut
        h = h + c * (wp.identity(n=3, dtype=float) - nn)
    return f, h
