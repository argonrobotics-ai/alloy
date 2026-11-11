# SPDX-License-Identifier: Apache-2.0
"""Bodies in pools (WP15's ask for WP8's pan, can and tomatoes): buoyancy, drag and a vessel's water weight, applied
in ``apply`` as wrenches on ``body_f`` (one thread per world, floats then vessels in order: deterministic, no atomics).

- **Buoyancy.** ``Float(body, hull=((r, z), ...) | sphere=R, origin=...)``: ρ·g·V_sub up at the centre of buoyancy.
  A hull is the body's outer revolution profile about its z axis through ``origin`` (a vessel's envelope: a pan floats
  like a boat; its own water rides on it as weight). V_sub is the part below the water plane: per slice exact (the
  chord integral of ``supply_cavity``), the centroid per slice from its mid-height chord (second order in the slice);
  a sphere's is the closed-form cap. The water is the first pool, cavities first, whose region holds the float's
  origin: a cavity's surface (normal to gravity), else a box pool's footprint and level (along its body's z).
- **Drag.** On the submerged share f = V_sub/V: -f·(c_lin + ½·ρ·cd·a_ref·|v|)·v at the centre of buoyancy, v the
  body's velocity there relative to the water (a vessel's own motion in a cavity). Defaults: a sphere of the hull's
  volume, Stokes c_lin = 6π·μ·R_eq, a_ref = π·R_eq², cd 1.0 (ESTIMATE).
- **Water weight.** A cavity pool on a body carries its water, ρ·V·g at the water's centroid.

The pool's level does not rise with what floats in it, and nothing turns a body back but the buoyancy's own torque;
heave is nearly undamped without wave radiation (damping ratio ~0.01 for WP8's pan: raise ``c_lin`` to settle it).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench, com_world, point_velocity, wrench_at
from alloy.plugins.surface_flow.supply_cavity import Cavity, disc_below, in_cavity, slab_below
from alloy.plugins.surface_flow.supply_kernels import PoolSpec, pool_frame

PI = wp.constant(3.14159265358979)
SPHERE = wp.constant(1)


@dataclass(frozen=True)
class Float:
    """A body that floats in the pools (module docstring): its outer ``hull`` (r, z) profile about its z axis through
    ``origin`` (body frame), or a ``sphere`` radius about ``origin``; drag ``cd`` (ESTIMATE), ``c_lin`` [N·s/m] and
    ``a_ref`` [m²] (None: from a sphere of its volume). ``slices``: the volume is exact per slice, the centre of
    buoyancy converges as 1/slices² (48: 0.6 % of a puck's righting moment at 5°)."""

    body: str
    hull: tuple[tuple[float, float], ...] | None = None
    sphere: float | None = None
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    slices: int = 48
    cd: float = 1.0
    c_lin: float | None = None
    a_ref: float | None = None

    def __post_init__(self) -> None:
        if (self.hull is None) == (self.sphere is None):
            raise PluginError(f"float {self.body!r}: give a hull profile or a sphere radius, not both")
        if self.sphere is not None and not self.sphere > 0:
            raise PluginError(f"float {self.body!r}: sphere radius > 0")
        if self.hull is not None:
            z = [float(n[1]) for n in self.hull]
            if len(z) < 2 or any(b <= a for a, b in zip(z[:-1], z[1:])) or min(n[0] for n in self.hull) < 0:
                raise PluginError(f"float {self.body!r}: a hull is >= 2 (r >= 0, z) nodes with z rising")
        if self.slices < 4 or self.cd < 0 or (self.c_lin is not None and self.c_lin < 0):
            raise PluginError(f"float {self.body!r}: slices >= 4, cd >= 0, c_lin >= 0")


@wp.struct
class Hull:
    kind: wp.int32            # 0: revolution profile, 1: sphere, 2: a vessel carried as mass (no weight force)
    origin: wp.vec3
    s0: wp.int32              # slices [s0, s0 + ns): (z_a, z_b, r) from the origin
    ns: wp.int32
    radius: wp.float32        # a sphere's
    vol: wp.float64
    cd: wp.float32
    c_lin: wp.float32
    a_ref: wp.float32
    pool: wp.int32            # its own cavity pool (a vessel does not float in its own water), or -1


@wp.func
def slice_moment(za: wp.float64, zb: wp.float64, r: wp.float64, up: wp.vec3d, h: wp.float64,
                 v: wp.float64) -> wp.vec3d:
    """First moment ∫x dV of a slab's part below the plane up·x = h (volume ``v``), from its mid-height chord: the
    circular segment's centroid -(2/3)(r² - t²)^(3/2)/A(t) along the plane's tilt (the lowest edge when the mid chord
    is dry), at the mid height of the wet part."""
    s = wp.sqrt(up[0] * up[0] + up[1] * up[1])
    zl = za                                           # the wet part's height range along the axis (approximate)
    zh = zb
    if wp.abs(up[2]) > wp.float64(1.0e-9):
        zc = h / up[2]
        if up[2] > wp.float64(0.0):
            zh = wp.min(zb, wp.max(za, zc + s * r / up[2]))
        else:
            zl = wp.max(za, wp.min(zb, zc - s * r / up[2]))
    zm = wp.float64(0.5) * (zl + zh)
    xe = wp.float64(0.0)
    e = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
    if s > wp.float64(1.0e-9):
        e = wp.vec3d(up[0] / s, up[1] / s, wp.float64(0.0))
        t = (h - up[2] * zm) / s
        a = disc_below(r, t)
        if t <= -r or a <= wp.float64(1.0e-30):
            xe = -r
        elif t < r:
            q = r * r - t * t
            xe = wp.float64(-2.0 / 3.0) * q * wp.sqrt(q) / a
    return v * (xe * e + wp.vec3d(wp.float64(0.0), wp.float64(0.0), zm))


@wp.func
def below(hull: Hull, slices: wp.array[wp.vec3d], up: wp.vec3d, h: wp.float64) -> wp.vec4d:
    """(V, centroid) of the hull's part below the plane up·x = h (x from its origin, its frame)."""
    v = wp.float64(0.0)
    m = wp.vec3d()
    if hull.kind == SPHERE:
        rr = wp.float64(hull.radius)
        hc = wp.clamp(rr + h, wp.float64(0.0), wp.float64(2.0) * rr)       # the wet cap's height
        if hc <= wp.float64(0.0):
            return wp.vec4d()
        v = wp.float64(PI) * hc * hc * (wp.float64(3.0) * rr - hc) / wp.float64(3.0)
        off = wp.float64(3.0) * (wp.float64(2.0) * rr - hc) * (wp.float64(2.0) * rr - hc) / (
            wp.float64(4.0) * (wp.float64(3.0) * rr - hc))
        c = -off * up
        return wp.vec4d(v, c[0], c[1], c[2])
    for k in range(hull.ns):
        sl = slices[hull.s0 + k]
        vk = slab_below(sl[0], sl[1], sl[2], up, h)
        if vk > wp.float64(0.0):
            v = v + vk
            m = m + slice_moment(sl[0], sl[1], sl[2], up, h, vk)
    if v <= wp.float64(0.0):
        return wp.vec4d()
    return wp.vec4d(v, m[0] / v, m[1] / v, m[2] / v)


@wp.func
def water_of(w: int, x: wp.vec3, skip: int, body_q: wp.array[wp.transform], pools: wp.array[PoolSpec],
             pbody: wp.array2d[wp.int32], cavs: wp.array[Cavity], cprof: wp.array[wp.vec2]) -> int:
    """The pool whose water region holds world point ``x``: the first cavity whose inside holds it, else the first box
    pool whose footprint does (pool ``skip`` left out), or -1."""
    found = int(-1)
    for pass_ in range(2):
        for p in range(pools.shape[0]):
            cav = cavs[p]
            if found >= 0 or p == skip or (pass_ == 0) != (cav.on != 0):
                continue
            pool = pools[p]
            q = wp.transform_point(wp.transform_inverse(pool_frame(body_q, pbody[w, p])), x) - pool.floor
            if cav.on != 0:
                if in_cavity(cav, cprof, q, wp.vec3(0.0, 0.0, 1.0)) == 1:
                    found = p
            elif wp.abs(q[0]) <= pool.half[0] and wp.abs(q[1]) <= pool.half[1]:
                found = p
    return found


@wp.func
def plane_of(w: int, p: int, body_q: wp.array[wp.transform], pools: wp.array[PoolSpec], pbody: wp.array2d[wp.int32],
             pressure: wp.array2d[wp.float64], cavs: wp.array[Cavity], surf: wp.array2d[wp.vec4]) -> wp.vec4:
    """Pool p's water plane in the world: (unit normal n, n·point): a cavity's surface (normal to gravity), a box
    pool's level along its body's z."""
    pool = pools[p]
    F = pool_frame(body_q, pbody[w, p])
    if cavs[p].on != 0:
        sv = surf[w, p]
        u = wp.vec3(sv[0], sv[1], sv[2])
        n = wp.transform_vector(F, u)
        return wp.vec4(n[0], n[1], n[2], wp.dot(n, wp.transform_point(F, pool.floor + u * sv[3])))
    level = float(pool.kappa * wp.max(pressure[w, pool.tank] - pool.p_ref, wp.float64(0.0)) / pool.area)
    n = wp.transform_vector(F, wp.vec3(0.0, 0.0, 1.0))
    return wp.vec4(n[0], n[1], n[2], wp.dot(n, wp.transform_point(F, pool.floor + wp.vec3(0.0, 0.0, level))))


@wp.kernel
def float_forces(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                 pools: wp.array[PoolSpec], pbody: wp.array2d[wp.int32], pressure: wp.array2d[wp.float64],
                 cavs: wp.array[Cavity], cprof: wp.array[wp.vec2], surf: wp.array2d[wp.vec4],
                 hulls: wp.array[Hull], hsl: wp.array[wp.vec3d], fbody: wp.array2d[wp.int32],
                 vessels: wp.array[Hull], gravity: wp.vec3, body_f: wp.array[wp.spatial_vector]):
    """Per world: each float's buoyancy and drag, then each vessel's water weight (module docstring)."""
    w = wp.tid()
    g = wp.length(gravity)
    if g <= 0.0:
        return
    for f in range(hulls.shape[0]):
        b = fbody[w, f]
        hull = hulls[f]
        if b < 0:
            continue
        X = body_q[b]
        o = wp.transform_point(X, hull.origin)
        p = water_of(w, o, hull.pool, body_q, pools, pbody, cavs, cprof)
        if p < 0:
            continue
        pl = plane_of(w, p, body_q, pools, pbody, pressure, cavs, surf)
        n = wp.vec3(pl[0], pl[1], pl[2])
        if wp.length(n) < 0.5:                                 # a cavity whose surface is not set yet
            continue
        u = wp.transform_vector(wp.transform_inverse(X), n)
        vc = below(hull, hsl, wp.vec3d(wp.float64(u[0]), wp.float64(u[1]), wp.float64(u[2])),
                   wp.float64(pl[3] - wp.dot(n, o)))
        if vc[0] <= wp.float64(0.0):
            continue
        rho = pools[p].rho
        cb = wp.transform_point(X, hull.origin + wp.vec3(float(vc[1]), float(vc[2]), float(vc[3])))
        com = com_world(body_q, body_com, b)
        v = point_velocity(body_qd, com, b, cb)
        if cavs[p].on != 0 and pbody[w, p] >= 0:               # relative to the vessel's water
            bp = pbody[w, p]
            v = v - point_velocity(body_qd, com_world(body_q, body_com, bp), bp, cb)
        c_lin = hull.c_lin
        if c_lin < 0.0:                                        # Stokes on the sphere of the hull's volume
            c_lin = 6.0 * PI * rho * pools[p].nu * wp.pow(3.0 * float(hull.vol) / (4.0 * PI), 1.0 / 3.0)
        frac = float(vc[0] / hull.vol)
        force = rho * g * float(vc[0]) * n - frac * (c_lin + 0.5 * rho * hull.cd * hull.a_ref * wp.length(v)) * v
        add_wrench(body_f, b, wrench_at(force, cb, com))
    for p in range(vessels.shape[0]):
        ves = vessels[p]
        b = pbody[w, p]
        if ves.ns == 0 or b < 0 or ves.kind == 2:          # carried as mass: gravity acts on it (supply_carry)
            continue
        pool = pools[p]
        vol = pool.kappa * wp.max(pressure[w, pool.tank] - pool.p_ref, wp.float64(0.0))
        if vol <= wp.float64(0.0):
            continue
        sv = surf[w, p]
        if sv[0] * sv[0] + sv[1] * sv[1] + sv[2] * sv[2] < 0.25:
            continue
        vc = below(ves, hsl, wp.vec3d(wp.float64(sv[0]), wp.float64(sv[1]), wp.float64(sv[2])), wp.float64(sv[3]))
        if vc[0] <= wp.float64(0.0):
            continue
        cw = wp.transform_point(pool_frame(body_q, b), pool.floor + wp.vec3(float(vc[1]), float(vc[2]), float(vc[3])))
        add_wrench(body_f, b, wrench_at(float(pool.rho) * float(vol) * gravity, cw, com_world(body_q, body_com, b)))


# ---- host side -----------------------------------------------------------------------------------------------------

def _slices(profile: Any, n: int) -> tuple[np.ndarray, float]:
    """Slices (z_a, z_b, r at the mid height) of a revolution profile and their volume."""
    rz = np.asarray(profile, np.float64)
    zs = np.linspace(rz[0, 1], rz[-1, 1], n + 1)
    rm = np.interp(0.5 * (zs[:-1] + zs[1:]), rz[:, 1], rz[:, 0])
    return np.stack([zs[:-1], zs[1:], rm], 1), float(np.sum(np.pi * rm ** 2 * np.diff(zs)))


def float_tables(floats: tuple[Float, ...], pools: tuple[Any, ...]) -> tuple[list[Hull], np.ndarray, list[Hull]]:
    """The floats' hulls and slices (from their origins), and per pool a vessel hull over its cavity's slices (the
    cavity's own slice range; off for box pools and cavities without a body)."""
    hulls, slices = [], []
    for f in floats:
        h = Hull()
        h.origin, h.cd, h.pool = wp.vec3(*f.origin), f.cd, -1
        for p, pool in enumerate(pools):
            if pool.cavity is not None and pool.body == f.body:
                h.pool = p                                     # a vessel does not float in its own water
        if f.sphere is not None:
            h.kind, h.radius, h.ns, h.s0 = 1, f.sphere, 0, 0
            h.vol = 4.0 / 3.0 * math.pi * f.sphere ** 3
        else:
            sl, vol = _slices(f.hull, f.slices)
            h.kind, h.s0, h.ns, h.vol = 0, len(slices), f.slices, vol
            slices.extend(sl.tolist())
        r_eq = (3.0 * h.vol / (4.0 * math.pi)) ** (1.0 / 3.0)
        h.c_lin = -1.0 if f.c_lin is None else f.c_lin
        h.a_ref = math.pi * r_eq * r_eq if f.a_ref is None else f.a_ref
        hulls.append(h)
    vessels = []
    for pool in pools:
        v = Hull()
        v.kind, v.ns = 0, 0
        if pool.cavity is not None and pool.body is not None:
            sl, vol = _slices(pool.cavity, pool.slices)
            v.s0, v.ns, v.vol = len(slices), pool.slices, vol
            v.kind = 2 if pool.carry == "mass" else 0
            slices.extend(sl.tolist())
        vessels.append(v)
    return hulls, np.asarray(slices or [(0.0, 0.0, 0.0)], np.float64), vessels


class Floating:
    """``Supply``'s bodies in pools (a mixin: it uses the supply's pools, cavities and body tables)."""

    def _setup_floats(self, ctx: Any) -> None:
        hulls, hsl, vessels = float_tables(self.floats, self.pools)
        self.floating = bool(self.floats) or any(v.ns > 0 for v in vessels)
        self.hulls = wp.array(hulls or [Hull()], dtype=Hull, device=ctx.device)
        self.vessels = wp.array(vessels or [Hull()], dtype=Hull, device=ctx.device)
        self.hull_slices = ctx.array(hsl, wp.vec3d)
        fb = [self._bodies(ctx, f.body) for f in self.floats] or [np.full(ctx.world_count, -1, np.int32)]
        self.fbody = ctx.array(np.stack(fb, 1), wp.int32)

    def _float_forces(self, state: Any, forces: Any) -> None:
        """In ``apply``: the cavities' surfaces at this state, then ``float_forces``."""
        if not self.floating:
            return
        if self.cavities:
            self._cavity_state(state.body_q)
        self._carry(state)                                # carried pools first: the forces' centres (supply_carry)
        wp.launch(float_forces, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, self.ctx.model.body_com, self.pool_spec, self.pbody,
                          self.fluid.pressure, self.cav, self.cav_prof, self.cav_surf, self.hulls, self.hull_slices,
                          self.fbody, self.vessels, wp.vec3(*self.gravity)], outputs=[forces.body_f],
                  device=self.ctx.device)


__all__ = ["Float", "Floating", "Hull", "below", "float_forces", "float_tables", "plane_of", "slice_moment", "water_of"]
