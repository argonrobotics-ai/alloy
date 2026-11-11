# SPDX-License-Identifier: Apache-2.0
"""Suction cups (engine item 10): a bellows cup object and a vacuum force on the lumped-fluid core.

``SuctionCup`` (object) builds a mount body (fixed to the world, or free) and a cup body under it on a prismatic
"bellows" joint along the cup axis (a spring to the relaxed length; ``stroke = 0``: rigid, a fixed joint). The cup is
a cylinder whose bottom face is the lip; labels ``<label>_mount`` and ``<label>_cup``.

``Suction`` (force) models the cup's air as one isothermal gas chamber on ``alloy.plugins.fluid``:

    lip:   N points on a circle of radius R around the lip centre, in the plane normal to the cup axis
    seal:  a point is sealed on a target body when its signed distance d to one of the body's shapes has
           -``seal_depth`` <= d <= ``seal_gap`` (a lip pressed into the surface stays sealed: soft contacts sink
           about a millimetre under the full pressure force) and the shape normal there is within
           ``seal_max_angle_deg`` of the axis back at the cup;
           s = the largest sealed share over the targets (geometry, not contacts: works with every solver)
    gas:   V·dP/dt = S·u_pump·(P_ult - P) + (C_0 + C_open·(1 - s) + C_vent·u_vent)·(P_atm - P)   [Pa·m³/s]
           V = dead_volume + bellows_area·(lip extension from the mount since the start); stepped exactly
    force: F = (P_atm - P)·π·R²·s along the axis at the lip centre: on the target toward the cup, and the reaction on
           the mount (the bellows' parent), so the vacuum compresses the bellows

``pump`` and ``vent`` (0..1 per world) are inputs: ``sim.plugins["suction"].set_input("pump", 1.0)``. States: ``seal``
(s), ``target`` (index into ``targets`` of the sealed body, -1: none), ``force`` [N] and the chamber's arrays.

v1 limits: s scales the enclosed area (a stand-in for losing it when the lip opens); targets are primitives (box,
sphere, capsule, cylinder, plane) or meshes (Newton's mesh SDF, scaled by the smallest scale factor); bellows dynamics
are the joint spring; the force is
explicit and switches with the seal across ``seal_gap`` (attach and release are a step change).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.geometry import sdf_box, sdf_capsule, sdf_cylinder, sdf_mesh, sdf_plane, sdf_sphere

from alloy.plugins.base import ForcePlugin, Forces, ObjectPlugin
from alloy.plugins.fluid import P_ATM, Edge, Network, Node
from alloy.plugins.funcs import add_wrench, body_point, body_vector, com_world, wrench_at
from alloy.plugins.harness.scene import Law, TestScene, body_xform

GEO_PLANE, GEO_SPHERE, GEO_CAPSULE = wp.constant(1), wp.constant(3), wp.constant(4)
GEO_CYLINDER, GEO_BOX, GEO_MESH = wp.constant(6), wp.constant(7), wp.constant(8)
FAR = wp.constant(1.0e6)
EDGES = ("pump", "seal", "leak", "vent")               # the chamber's edges, in this order


class SuctionCup(ObjectPlugin):
    """A cup body on a spring bellows (prismatic, along local z) under a mount body.

    Example::

        SuctionCup(radius=0.02, stroke=0.006).build(builder, body_xform(0, 0, 0.3), "suction")   # lip faces -z
    """

    name = "suction_cup"
    summary = "suction cup: a cylinder lip on a spring bellows under a mount (pairs with the suction force)"

    @dataclass
    class Config:
        radius: float = 0.02                 # m, cup (and lip) radius
        height: float = 0.02                 # m, cup body height; the lip is its bottom face
        stroke: float = 0.006                # m, bellows travel up toward the mount; 0 = rigid (fixed joint)
        bellows_stiffness: float = 400.0     # N/m to the relaxed length
        bellows_damping: float = 4.0         # N·s/m
        cup_mass: float = 0.02               # kg
        mount_mass: float = 0.2              # kg
        mu: float = 1.0                      # lip friction (MuJoCo mixes pairs by max)
        contact_kd: float | None = None      # lip contact damping (ShapeConfig.kd); None: Newton's default (100), which
                                             # with contact_ke 1e5 is MuJoCo dampratio ~0.15: use 2·sqrt(ke) (critical)
        contact_ke: float | None = None      # lip contact stiffness (Newton ShapeConfig.ke); None: Newton's default.
                                             # MuJoCo reads it mass-normalised: raise it where the pressure force
                                             # would sink the lip millimetres into the part
        mount: str = "free"                  # "fixed" to the world at xform, or "free"
        lip_shape: str = "cylinder"          # collision shape of the cup: "cylinder" or "box" (a square pad)

        def __post_init__(self) -> None:
            if self.mount not in ("fixed", "free"):
                raise ValueError(f"mount must be 'fixed' or 'free', got {self.mount!r}")
            if min(self.radius, self.height, self.cup_mass, self.mount_mass) <= 0 or self.stroke < 0:
                raise ValueError("radius, height, cup_mass, mount_mass must be > 0 and stroke >= 0")
            if min(self.bellows_stiffness, self.bellows_damping, self.mu) < 0:
                raise ValueError("bellows_stiffness, bellows_damping and mu must be >= 0")

    def build(self, builder: newton.ModelBuilder, xform: wp.transform, label: str) -> list[int]:
        c = self.config
        mh = 0.005
        mount = builder.add_link(xform=xform, label=f"{label}_mount")
        builder.add_shape_box(mount, hx=c.radius, hy=c.radius, hz=mh, cfg=newton.ModelBuilder.ShapeConfig(
            density=c.mount_mass / (8.0 * c.radius * c.radius * mh), mu=c.mu))
        rel = body_xform(0.0, 0.0, -(mh + 0.5 * c.height))
        cup = builder.add_link(xform=wp.transform_multiply(xform, rel), label=f"{label}_cup")
        lip = newton.ModelBuilder.ShapeConfig(density=c.cup_mass / (math.pi * c.radius ** 2 * c.height), mu=c.mu)
        if c.contact_ke is not None:
            lip.ke = c.contact_ke
        if c.contact_kd is not None:
            lip.kd = c.contact_kd
        if c.lip_shape == "box":                       # a square pad: MuJoCo gives a face patch of contacts
            builder.add_shape_box(cup, hx=c.radius, hy=c.radius, hz=0.5 * c.height, cfg=lip, label=f"{label}_lip")
        else:
            builder.add_shape_cylinder(cup, radius=c.radius, half_height=0.5 * c.height, cfg=lip, label=f"{label}_lip")
        if c.mount == "fixed":
            j0 = builder.add_joint_fixed(-1, mount, parent_xform=xform, label=f"{label}_base")
        else:
            j0 = builder.add_joint_free(mount, label=f"{label}_base")
        if c.stroke > 0:
            j1 = builder.add_joint_prismatic(mount, cup, parent_xform=rel, axis=(0.0, 0.0, 1.0), target_pos=0.0,
                                             target_ke=c.bellows_stiffness, target_kd=c.bellows_damping,
                                             limit_lower=0.0, limit_upper=c.stroke, limit_ke=1.0e4, limit_kd=50.0,
                                             label=f"{label}_bellows")
        else:
            j1 = builder.add_joint_fixed(mount, cup, parent_xform=rel, label=f"{label}_bellows")
        builder.add_articulation([j0, j1], label=label)
        return [mount, cup]


@wp.func
def _sdf(gtype: wp.int32, scale: wp.vec3, mesh: wp.uint64, p: wp.vec3) -> float:
    if gtype == GEO_MESH:                                # scaled mesh: query unscaled, distance × the smallest scale
        smin = wp.min(scale[0], wp.min(scale[1], scale[2]))
        q = wp.vec3(p[0] / scale[0], p[1] / scale[1], p[2] / scale[2])
        return sdf_mesh(mesh, q, 0.1 / smin) * smin
    if gtype == GEO_BOX:
        return sdf_box(p, scale[0], scale[1], scale[2])
    if gtype == GEO_SPHERE:
        return sdf_sphere(p, scale[0])
    if gtype == GEO_CAPSULE:
        return sdf_capsule(p, scale[0], scale[1], 2)
    if gtype == GEO_CYLINDER:
        return sdf_cylinder(p, scale[0], scale[1], 2)
    if gtype == GEO_PLANE:
        return sdf_plane(p, scale[0], scale[1])
    return FAR


@wp.func
def _normal(gtype: wp.int32, scale: wp.vec3, mesh: wp.uint64, p: wp.vec3) -> wp.vec3:
    """Outward normal by central differences of the signed distance (local frame)."""
    h = 1.0e-4
    gx = _sdf(gtype, scale, mesh, p + wp.vec3(h, 0.0, 0.0)) - _sdf(gtype, scale, mesh, p - wp.vec3(h, 0.0, 0.0))
    gy = _sdf(gtype, scale, mesh, p + wp.vec3(0.0, h, 0.0)) - _sdf(gtype, scale, mesh, p - wp.vec3(0.0, h, 0.0))
    gz = _sdf(gtype, scale, mesh, p + wp.vec3(0.0, 0.0, h)) - _sdf(gtype, scale, mesh, p - wp.vec3(0.0, 0.0, h))
    g = wp.vec3(gx, gy, gz)
    n = wp.length(g)
    if n > 0.0:
        return g / n
    return g


@wp.func
def point_sealed(p: wp.vec3, back: wp.vec3, w: wp.int32, t: wp.int32, body_q: wp.array[wp.transform],
            ids: wp.array2d[wp.int32], tgt: wp.array[wp.int32], shape_body: wp.array[wp.int32],
            shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3], shape_xf: wp.array[wp.transform],
            shape_mesh: wp.array[wp.uint64], gap: float, depth: float, cos_max: float) -> wp.int32:
    """1 if point p (world) lies on one of target t's shapes in world w: -depth <= d <= gap, the shape normal within
    acos(cos_max) of ``back`` (shared with ``adhesion``)."""
    for j in range(ids.shape[1]):
        if tgt[j] == t:
            s = ids[w, j]
            xf = shape_xf[s]
            b = shape_body[s]
            if b >= 0:
                xf = wp.transform_multiply(body_q[b], xf)
            pl = wp.transform_point(wp.transform_inverse(xf), p)
            d = _sdf(shape_type[s], shape_scale[s], shape_mesh[s], pl)
            if d <= gap and d >= -depth:
                nw = wp.transform_vector(xf, _normal(shape_type[s], shape_scale[s], shape_mesh[s], pl))
                if wp.dot(nw, back) >= cos_max:
                    return 1
    return 0


@wp.kernel
def _suction_apply(body_q: wp.array[wp.transform], body_com: wp.array[wp.vec3], cup: wp.array[wp.int32],
                   mount: wp.array[wp.int32], lip_c: wp.vec3, axis: wp.vec3, radius: float, n_lip: wp.int32,
                   gap: float, depth: float, cos_max: float, area: float, p_atm: float, n_tgt: wp.int32,
                   tgt_body: wp.array2d[wp.int32], ids: wp.array2d[wp.int32], tgt: wp.array[wp.int32],
                   shape_body: wp.array[wp.int32], shape_type: wp.array[wp.int32], shape_scale: wp.array[wp.vec3],
                   shape_xf: wp.array[wp.transform], shape_mesh: wp.array[wp.uint64], pressure: wp.array2d[wp.float64],
                   dead: float, bellows: float,
                   x0: wp.array[wp.float32], pump: wp.array[wp.float32], vent: wp.array[wp.float32],
                   opening: wp.array2d[wp.float32], volume: wp.array2d[wp.float64], seal: wp.array[wp.float32],
                   target: wp.array[wp.int32], force: wp.array[wp.float32], body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    c, mb = cup[w], mount[w]
    lc = body_point(body_q, c, lip_c)
    ax = wp.normalize(body_vector(body_q, c, axis))
    ref = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(ax[0]) > 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    u1 = wp.normalize(wp.cross(ax, ref))
    u2 = wp.cross(ax, u1)
    best, best_t = int(0), int(-1)
    for t in range(n_tgt):
        count = int(0)
        for k in range(n_lip):
            phi = 2.0 * wp.pi * float(k) / float(n_lip)
            p = lc + radius * (wp.cos(phi) * u1 + wp.sin(phi) * u2)
            count += point_sealed(p, -ax, w, t, body_q, ids, tgt, shape_body, shape_type, shape_scale, shape_xf,
                                  shape_mesh, gap, depth, cos_max)
        if count > best:
            best, best_t = count, t
    s = float(best) / float(n_lip)
    f = (p_atm - float(pressure[w, 0])) * area * s
    tb = int(-1)
    if best_t >= 0:
        tb = tgt_body[w, best_t]
        if tb >= 0:
            add_wrench(body_f, tb, wrench_at(-f * ax, lc, com_world(body_q, body_com, tb)))
        add_wrench(body_f, mb, wrench_at(f * ax, lc, com_world(body_q, body_com, mb)))
    seal[w] = s
    target[w] = best_t
    force[w] = f
    x = wp.dot(lc - wp.transform_get_translation(body_q[mb]), ax) - x0[w]
    volume[w, 0] = wp.float64(wp.max(dead + bellows * x, 0.05 * dead))
    opening[w, 0] = pump[w]
    opening[w, 1] = 1.0 - s
    opening[w, 2] = 1.0
    opening[w, 3] = vent[w]


class Suction(ForcePlugin):
    """A vacuum suction cup: geometric lip seal on target bodies, one isothermal gas chamber (pump, vent, leaks),
    the pressure force on the target and its reaction on the mount. Pairs with ``SuctionCup``.

    Example::

        suc = Suction(cup="suction_cup", mount="suction_mount", targets=("box",))
        sim = alloy.plugins.simulation(built, [suc]); suc.set_input("pump", 1.0)
    """

    name = "suction"
    summary = "vacuum suction cup: lip seal by geometry, gas chamber with pump / vent / leaks, pressure force"
    internal = True

    @dataclass
    class Config:
        cup: str = "suction_cup"                              # body label of the lip body
        mount: str = "suction_mount"                          # body label taking the reaction (the bellows' parent)
        lip_center: tuple[float, float, float] = (0.0, 0.0, -0.01)   # m, in the cup frame
        axis: tuple[float, float, float] = (0.0, 0.0, -1.0)   # out of the lip, in the cup frame
        radius: float = 0.02                                  # m, lip radius R (area π·R²)
        lip_points: int = 16                                  # N
        seal_gap: float = 1.0e-3                              # m above the surface
        seal_depth: float = 5.0e-3                            # m into the surface (a pressed lip stays sealed)
        seal_max_angle_deg: float = 30.0                      # surface normal vs the axis back at the cup
        targets: tuple[str, ...] = ("box",)                   # body labels the cup may seal on
        dead_volume: float = 5.0e-6                           # m³ at the start
        bellows_area: float = 0.0                             # m²: V = dead_volume + bellows_area·extension
        pump_speed: float = 1.0e-4                            # m³/s, S
        ultimate_pressure: float = 16.0e3                     # Pa abs, the pump's P_ult
        leak: float = 1.0e-6                                  # m³/s, C_0 (sealed lip)
        open_leak: float = 1.0e-2                             # m³/s, C_open (lip fully open)
        vent: float = 1.0e-3                                  # m³/s, C_vent (vent valve open)
        atmosphere: float = P_ATM                             # Pa
        initial_pressure: float | None = None                 # Pa abs; None: the atmosphere

        def __post_init__(self) -> None:
            if self.lip_points < 3 or min(self.radius, self.seal_gap, self.dead_volume, self.atmosphere) <= 0:
                raise ValueError("lip_points >= 3; radius, seal_gap, dead_volume, atmosphere > 0 are required")
            if self.seal_depth < 0:
                raise ValueError("seal_depth must be >= 0")
            if min(self.pump_speed, self.ultimate_pressure, self.leak, self.open_leak, self.vent, self.bellows_area) < 0:
                raise ValueError("speeds, conductances, ultimate_pressure and bellows_area must be >= 0")
            if not self.targets:
                raise ValueError("targets: at least one body label")

    @property
    def area(self) -> float:
        return math.pi * self.config.radius ** 2

    def steady_pressure(self, seal: float = 1.0, pump: float = 1.0, vent: float = 0.0) -> float:
        """The chamber's steady pressure for a seal fraction and inputs [Pa abs]."""
        c = self.config
        s, g = c.pump_speed * pump, c.leak + c.open_leak * (1.0 - seal) + c.vent * vent
        return (s * c.ultimate_pressure + g * c.atmosphere) / max(s + g, 1e-30)

    def network(self) -> Network:
        c = self.config
        p0 = c.atmosphere if c.initial_pressure is None else c.initial_pressure
        return Network([Node("cup", volume=c.dead_volume, pressure=p0)],
                       [Edge("pump", "pump_gas", "cup", reservoir=c.ultimate_pressure, conductance=c.pump_speed),
                        Edge("seal", "linear", "cup", reservoir=c.atmosphere, conductance=c.open_leak),
                        Edge("leak", "linear", "cup", reservoir=c.atmosphere, conductance=c.leak),
                        Edge("vent", "linear", "cup", reservoir=c.atmosphere, conductance=c.vent)])

    def setup(self, ctx: Any) -> None:
        c, model, W = self.config, ctx.model, ctx.world_count
        self.cup, self.mount = ctx.bodies(c.cup), ctx.bodies(c.mount)
        self.fluid = self.network().compile(ctx.device, W)
        for k, a in self.fluid.state_arrays().items():
            self.add_state(k, a)
        self.tgt_body, self.ids, self.tgt = target_shapes(ctx, c.targets, self.label)
        q = model.body_q.numpy().astype(np.float64)
        self.x0 = ctx.array(np.array([_extension(q[cb], q[mb], c) for cb, mb in
                                      zip(self.cup.numpy(), self.mount.numpy(), strict=True)], np.float32))
        self.seal = self.add_state("seal", ctx.zeros(W))
        self.target = self.add_state("target", ctx.full(W, -1, wp.int32))
        self.force = self.add_state("force", ctx.zeros(W))
        self.pump = self.add_input("pump", ctx.zeros(W))
        self.vent_in = self.add_input("vent", ctx.zeros(W))

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        c, m, f = self.config, self.ctx.model, self.fluid
        wp.launch(_suction_apply, dim=self.ctx.world_count,
                  inputs=[state.body_q, m.body_com, self.cup, self.mount, wp.vec3(*c.lip_center), wp.vec3(*c.axis),
                          float(c.radius), int(c.lip_points), float(c.seal_gap), float(c.seal_depth),
                          float(math.cos(math.radians(c.seal_max_angle_deg))), float(self.area), float(c.atmosphere),
                          len(c.targets), self.tgt_body, self.ids, self.tgt, m.shape_body, m.shape_type, m.shape_scale,
                          m.shape_transform, m.shape_source_ptr, f.pressure, float(c.dead_volume),
                          float(c.bellows_area), self.x0,
                          self.pump, self.vent_in],
                  outputs=[f.opening, f.volume, self.seal, self.target, self.force, forces.body_f],
                  device=self.ctx.device)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        self.fluid.step(dt)

    def test_scene(self) -> TestScene:
        from alloy.plugins.library.suction_laws import G, f_max, rest_scene  # noqa: PLC0415

        return rest_scene(self, mass=5.0 * f_max(self) / G)       # sealed on a heavy box, the pump evacuating

    def laws(self) -> list[Law]:
        from alloy.plugins.library.suction_laws import suction_laws  # noqa: PLC0415

        return suction_laws(self)


def target_shapes(ctx: Any, targets: tuple[str, ...], who: str) -> tuple[wp.array, wp.array, wp.array]:
    """Device tables of the target bodies' shapes: body ``[W, T]``, shape ids ``[W, S]`` and each id's target ``[S]``
    (the same shape count in every world). Shared with ``adhesion``. Display-only shapes, which collide with neither
    shapes nor particles (a decompose card's ``<name>/visual`` mesh, SCR OSS-9-5; robot visual layers), are no surface
    to seal on and stay out; a target that only skips shape contacts (adhesion's energy_cycle box) stays in."""
    sb = ctx.model.shape_body.numpy()
    any_collision = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
    collide = (ctx.model.shape_flags.numpy() & any_collision) != 0
    tb = np.stack([ctx.bodies(t).numpy() for t in targets], axis=1).astype(np.int32)        # [W, T]
    rows = [[(int(s), t) for t in range(len(targets)) for s in np.nonzero((sb == tb[w, t]) & collide)[0]]
            for w in range(ctx.world_count)]
    if len({len(r) for r in rows}) != 1 or not rows[0]:
        raise ValueError(f"{who}: target bodies {targets} need shapes, the same number in every world")
    return (ctx.array(tb, wp.int32), ctx.array(np.array([[s for s, _ in r] for r in rows], np.int32), wp.int32),
            ctx.array(np.array([t for _, t in rows[0]], np.int32), wp.int32))


def _extension(q_cup: np.ndarray, q_mount: np.ndarray, c: Any) -> float:
    """Host twin of the kernel's lip extension along the axis from the mount origin (the start reference)."""
    from alloy.plugins.npgeom import quat_rotate, transform_point  # noqa: PLC0415

    lc = transform_point(q_cup, np.asarray(c.lip_center, np.float64))
    ax = quat_rotate(q_cup[3:7], np.asarray(c.axis, np.float64))
    ax = ax / np.linalg.norm(ax)
    return float((lc - q_mount[:3]) @ ax)


__all__ = ["Suction", "SuctionCup", "point_sealed", "target_shapes"]
