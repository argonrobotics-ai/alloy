# SPDX-License-Identifier: Apache-2.0
"""Narrow-phase bias (CT-7, law A8's centred gap): Newton's reported separation vs the known geometric one.

Two shapes are stacked along +Z with a known surface separation ``s`` (negative = penetration); a private
``CollisionPipeline`` (explicit pair, deterministic) collides them and the reported separation is the minimum over the
pair's rows of ``n·(x1 - x0) - (margin0 + margin1)`` (the quantity every solver consumes, sim/contacts.py:72-94).
Round shapes have their axis on Z, so every stack touches along a known support direction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from alloy._vendor import newton

# (kind, scale) -> half extent along Z; plane is an infinite half-space below z = 0
_PRIMS = ("sphere", "box", "capsule", "cylinder", "plane")


@dataclass(frozen=True)
class Prim:
    kind: str                             # one of _PRIMS
    size: tuple[float, ...]               # sphere (r,), box (hx, hy, hz), capsule/cylinder (r, half_height)

    @property
    def half_z(self) -> float:
        if self.kind == "sphere":
            return self.size[0]
        if self.kind == "box":
            return self.size[2]
        if self.kind == "capsule":
            return self.size[0] + self.size[1]
        if self.kind == "cylinder":
            return self.size[1]
        return 0.0


def _add(b: newton.ModelBuilder, body: int, prim: Prim, z: float, cfg) -> int:
    xf = wp.transform((0.0, 0.0, z), wp.quat_identity())
    if prim.kind == "sphere":
        return b.add_shape_sphere(body, xform=xf, radius=prim.size[0], cfg=cfg)
    if prim.kind == "box":
        return b.add_shape_box(body, xform=xf, hx=prim.size[0], hy=prim.size[1], hz=prim.size[2], cfg=cfg)
    if prim.kind == "capsule":
        return b.add_shape_capsule(body, xform=xf, radius=prim.size[0], half_height=prim.size[1], cfg=cfg)
    if prim.kind == "cylinder":
        return b.add_shape_cylinder(body, xform=xf, radius=prim.size[0], half_height=prim.size[1], cfg=cfg)
    if prim.kind == "plane":
        return b.add_shape_plane(body=body, xform=xf, width=0.0, length=0.0, cfg=cfg)
    raise ValueError(f"unknown primitive {prim.kind!r}")


def stacked_separation(lower: Prim, upper: Prim, separations: list[float], device: str = "cpu",
                       tilt_rad: float = 0.0, lateral: float = 0.0) -> np.ndarray:
    """Reported separation [m] for each requested one; ``lower`` static, top face at z = 0; ``upper`` on a body
    shifted by ``lateral`` in x and y and tilted by ``tilt_rad`` about X (with 0 tilt the separation is exact).
    One world per separation."""
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, gap=2.5e-3, margin=0.0)
    b = newton.ModelBuilder()
    pairs = []
    for s in separations:
        b.begin_world()
        sa = _add(b, -1, lower, -lower.half_z, cfg)
        q = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), tilt_rad)
        body = b.add_body(xform=wp.transform((lateral, lateral, s + upper.half_z), q))
        sb = _add(b, body, upper, 0.0, cfg)
        b.end_world()
        pairs.append((min(sa, sb), max(sa, sb)))
    model = b.finalize(device=device)
    state = model.state()
    pipe = newton.CollisionPipeline(model, broad_phase="explicit", deterministic=True, publish_capacity=False,
                                    shape_pairs_filtered=wp.array(np.asarray(pairs, np.int32), dtype=wp.vec2i,
                                                                  device=device), rigid_contact_max=64 * len(pairs))
    contacts = pipe.contacts()
    pipe.collide(state, contacts)
    return _min_separation(model, state, contacts, len(separations))


def _min_separation(model, state, c, worlds: int) -> np.ndarray:
    """Per world, the minimum over rows of ``n·(x1 - x0) - (margin0 + margin1)`` (inf where a world has no row)."""
    n = int(c.rigid_contact_count.numpy()[0])
    s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
    nrm = c.rigid_contact_normal.numpy()[:n].astype(np.float64)
    m = (c.rigid_contact_margin0.numpy()[:n] + c.rigid_contact_margin1.numpy()[:n]).astype(np.float64)
    body, world = model.shape_body.numpy(), model.shape_world.numpy()
    bq = state.body_q.numpy().astype(np.float64)
    x0 = _world_points(body[s0], c.rigid_contact_point0.numpy()[:n], bq)
    x1 = _world_points(body[s1], c.rigid_contact_point1.numpy()[:n], bq)
    d = np.einsum("ij,ij->i", nrm, x1 - x0) - m
    w = np.maximum(world[s0], world[s1])
    out = np.full(worlds, np.inf)
    np.minimum.at(out, w, d)
    return out


def _world_points(body: np.ndarray, p: np.ndarray, bq: np.ndarray) -> np.ndarray:
    """Body-frame points to world (static rows, body < 0, are already in world)."""
    p = p.astype(np.float64)
    out = p.copy()
    dyn = body >= 0
    if dyn.any():
        t, q = bq[body[dyn], :3], bq[body[dyn], 3:7]
        v, w = q[:, :3], q[:, 3:4]
        u = 2.0 * np.cross(v, p[dyn])
        out[dyn] = t + p[dyn] + w * u + np.cross(v, u)
    return out
