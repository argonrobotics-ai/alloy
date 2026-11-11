# SPDX-License-Identifier: Apache-2.0
"""Primitive reference scenes for CPU tests (SPEC §6.1, WP0).

Each function returns a ``RefScene``: a finalized ``newton.Model`` with ``n`` identical worlds (built by
``replicate`` with zero spacing, F1), a step size on the D4 grid, the body roles of world 0 and the parameters used.
``scene.runner(kind)`` gives a minimal eager stepper (collide + solver.step). This is a test fixture, not the
``Simulation`` loop (WP2). Solver kinds: "xpbd" (cheapest on CPU), "mujoco", "featherstone", "semi_implicit", "vbd",
and "coupled" for ``box_on_cloth`` (Newton's stock SolverCoupledProxy: MuJoCo source -> VBD destination).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton import Contacts, State
from alloy._vendor.newton.solvers import SolverBase

from alloy.core.newton_compat import SolverCoupledProxy

log = logging.getLogger("alloy.testing")

RIGID_DT = 0.02 / 16           # 1.25e-3 s, rigid target on the D4 grid
CLOTH_DT = 0.02 / 32           # 6.25e-4 s

_STOCK: dict[str, Callable[..., SolverBase]] = {
    "xpbd": lambda m, **kw: newton.solvers.SolverXPBD(m, **kw),
    "mujoco": lambda m, **kw: newton.solvers.SolverMuJoCo(m, **kw),
    "featherstone": lambda m, **kw: newton.solvers.SolverFeatherstone(m, **kw),
    "semi_implicit": lambda m, **kw: newton.solvers.SolverSemiImplicit(m, **kw),
    "vbd": lambda m, **kw: newton.solvers.SolverVBD(m, **kw),
}


@dataclass(eq=False)
class RefScene:
    """A reference scene: model, grid dt, default solver and the roles of world 0."""

    name: str
    model: newton.Model
    dt: float
    default_solver: str
    bodies: dict[str, int]
    params: dict[str, float]
    particles: range = range(0)
    factories: dict[str, Callable[..., SolverBase]] = field(default_factory=dict)

    @property
    def world_count(self) -> int:
        return int(self.model.world_count)

    def make_solver(self, kind: str | None = None, **kwargs: Any) -> SolverBase:
        """Build solver ``kind`` (default ``default_solver``) on the scene model."""
        kind = kind or self.default_solver
        factory = self.factories.get(kind) or _STOCK.get(kind)
        if factory is None:
            raise ValueError(f"{self.name}: unknown solver kind {kind!r}")
        return factory(self.model, **kwargs)

    def runner(self, kind: str | None = None, **kwargs: Any) -> SceneRunner:
        """An eager stepper for this scene with solver ``kind``."""
        kind = kind or self.default_solver
        native = kind == "mujoco" and kwargs.get("use_mujoco_contacts", True)
        return SceneRunner(self, self.make_solver(kind, **kwargs), newton_contacts=not native)


class SceneRunner:
    """collide -> solver.step -> swap, eagerly on the scene device. ``state`` is the current state."""

    def __init__(self, scene: RefScene, solver: SolverBase, *, newton_contacts: bool = True) -> None:
        self.scene = scene
        self.solver = solver
        model = scene.model
        self.state_0: State = model.state()
        self.state_1: State = model.state()
        self.control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, self.state_0)
        self.pipeline = newton.CollisionPipeline(model) if newton_contacts else None
        self.contacts: Contacts | None = self.pipeline.contacts() if self.pipeline is not None else None
        self.steps = 0

    @property
    def state(self) -> State:
        return self.state_0

    def step(self, n: int = 1) -> State:
        """Advance ``n`` steps of ``scene.dt``; returns the current state."""
        for _ in range(n):
            self.state_0.clear_forces()
            if self.pipeline is not None:
                self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.scene.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.steps += 1
        return self.state_0


def _finalize(template: newton.ModelBuilder, n: int, device: str | None, *, ground: bool = True,
              color: bool = False) -> newton.Model:
    if n < 1:
        raise ValueError(f"world count must be >= 1, got {n}")
    root = newton.ModelBuilder()
    root.replicate(template, n)
    if ground:
        root.add_ground_plane()
    if color:
        root.color()
    return root.finalize(device=device)


def _cfg(density: float, *, mu: float = 0.5, collide: bool = True) -> newton.ModelBuilder.ShapeConfig:
    """Shape config whose density gives the body its mass and inertia (no inertia correction)."""
    return newton.ModelBuilder.ShapeConfig(density=density, mu=mu, has_shape_collision=collide,
                                           has_particle_collision=collide)


def _xform(x: float, y: float, z: float, q: wp.quat | None = None) -> wp.transform:
    return wp.transform(wp.vec3(x, y, z), q if q is not None else wp.quat_identity())


def box_on_plane(n: int, *, half: float = 0.1, mass: float = 1.0, mu: float = 0.5, gap: float = 0.0,
                 device: str | None = None) -> RefScene:
    """A box (half-extent ``half`` m, ``mass`` kg) resting on the ground plane, ``gap`` m above it, per world."""
    b = newton.ModelBuilder()
    box = b.add_body(xform=_xform(0.0, 0.0, half + gap), label="box")
    b.add_shape_box(box, hx=half, hy=half, hz=half, cfg=_cfg(mass / (8.0 * half**3), mu=mu))
    model = _finalize(b, n, device)
    return RefScene("box_on_plane", model, RIGID_DT, "xpbd", {"box": box},
                    {"half": half, "mass": mass, "mu": mu, "gap": gap})


def pendulum(n: int, amplitude: float, *, length: float = 0.5, mass: float = 1.0, radius: float = 0.05,
             device: str | None = None) -> RefScene:
    """A point-like pendulum: revolute joint about +y at (0, 0, 1) m, bob ``length`` m below, released from
    ``amplitude`` rad. Small-angle period 2π·sqrt(L/g) for the bob's COM (sphere inertia included)."""
    b = newton.ModelBuilder()
    bob = b.add_link(xform=_xform(0.0, 0.0, 1.0 - length), label="bob")
    density = mass / (4.0 / 3.0 * math.pi * radius**3)
    b.add_shape_sphere(bob, radius=radius, cfg=_cfg(density, collide=False))
    hinge = b.add_joint_revolute(parent=-1, child=bob, parent_xform=_xform(0.0, 0.0, 1.0),
                                 child_xform=_xform(0.0, 0.0, length), axis=(0.0, 1.0, 0.0), label="hinge")
    b.add_articulation([hinge], label="pendulum")
    b.joint_q[-1] = amplitude
    model = _finalize(b, n, device, ground=False)
    return RefScene("pendulum", model, RIGID_DT, "featherstone", {"bob": bob},
                    {"amplitude": amplitude, "length": length, "mass": mass, "radius": radius})


def incline(n: int, mu: float, theta: float, *, half: float = 0.05, mass: float = 0.5,
            device: str | None = None) -> RefScene:
    """A box on a static plank tilted by ``theta`` rad about +y; both surfaces have friction ``mu``. The box
    sticks if tan(theta) < mu (Coulomb)."""
    b = newton.ModelBuilder()
    cfg = _cfg(mass / (8.0 * half**3), mu=mu)
    tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), theta)
    plank_half = (1.0, 0.5, 0.05)
    b.add_shape_box(-1, xform=_xform(0.0, 0.0, 0.5, tilt), hx=plank_half[0], hy=plank_half[1], hz=plank_half[2],
                    cfg=cfg, label="plank")
    normal = (math.sin(theta), 0.0, math.cos(theta))
    off = plank_half[2] + half
    box = b.add_body(xform=_xform(normal[0] * off, 0.0, 0.5 + normal[2] * off, tilt), label="box")
    b.add_shape_box(box, hx=half, hy=half, hz=half, cfg=cfg)
    model = _finalize(b, n, device, ground=False)
    return RefScene("incline", model, RIGID_DT, "xpbd", {"box": box},
                    {"mu": mu, "theta": theta, "half": half, "mass": mass})


def peg_in_ring(n: int, clearance: float, *, radius: float = 0.01, segments: int = 16, depth: float = 0.02,
                device: str | None = None) -> RefScene:
    """A capsule peg (radius ``radius`` m) above a static ring of ``segments`` boxes whose inner radius leaves a
    diametral ``clearance`` m. Primitive stand-in for the meshed peg-hole (alloy.testing.contact, WP3)."""
    if clearance <= 0.0:
        raise ValueError(f"clearance must be > 0, got {clearance}")
    b = newton.ModelBuilder()
    cfg = _cfg(7800.0, mu=0.3)
    inner = radius + 0.5 * clearance
    wall = 0.005
    half_chord = (inner + wall) * math.tan(math.pi / segments) + 1e-4
    for k in range(segments):
        ang = 2.0 * math.pi * k / segments
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), ang)
        c = inner + 0.5 * wall
        b.add_shape_box(-1, xform=_xform(c * math.cos(ang), c * math.sin(ang), 0.5 * depth, q),
                        hx=0.5 * wall, hy=half_chord, hz=0.5 * depth, cfg=cfg, label=f"ring_{k}")
    peg = b.add_body(xform=_xform(0.0, 0.0, depth + 0.04), label="peg")
    b.add_shape_capsule(peg, radius=radius, half_height=0.02, cfg=cfg)
    model = _finalize(b, n, device)
    return RefScene("peg_in_ring", model, RIGID_DT / 4, "xpbd", {"peg": peg},
                    {"clearance": clearance, "radius": radius, "inner_radius": inner, "depth": depth})


def box_on_cloth(n: int, box_mass: float, *, cells: int = 8, size: float = 0.4, half: float = 0.05,
                 device: str | None = None) -> RefScene:
    """The coupling-time toy: a box (MuJoCo) dropped onto a cloth (VBD) pinned on two edges, per world.
    Default solver "coupled" = Newton's stock SolverCoupledProxy, lagged proxy MuJoCo -> VBD."""
    b = newton.ModelBuilder()
    b.add_cloth_grid(pos=wp.vec3(-0.5 * size, -0.5 * size, 0.5), rot=wp.quat_identity(), vel=wp.vec3(0.0),
                     dim_x=cells, dim_y=cells, cell_x=size / cells, cell_y=size / cells, mass=0.02,
                     fix_left=True, fix_right=True, tri_ke=1.0e4, tri_ka=1.0e4, edge_ke=1.0e-2,
                     particle_radius=0.01, label="cloth")
    box = b.add_body(xform=_xform(0.0, 0.0, 0.5 + half + 0.02), label="box")
    b.add_shape_box(box, hx=half, hy=half, hz=half, cfg=_cfg(box_mass / (8.0 * half**3), mu=0.5))
    n_particles = b.particle_count
    model = _finalize(b, n, device, ground=False, color=True)
    model.soft_contact_ke = 1.0e4
    model.soft_contact_mu = 0.5
    scene = RefScene("box_on_cloth", model, CLOTH_DT, "coupled", {"box": box},
                     {"box_mass": box_mass, "cells": cells, "size": size, "half": half}, range(n_particles))
    scene.factories["coupled"] = _coupled_factory
    return scene


def _coupled_factory(model: newton.Model, *, iterations: int = 2, mode: str = "lagged",
                     mujoco_kwargs: Mapping[str, Any] | None = None,
                     vbd_kwargs: Mapping[str, Any] | None = None) -> SolverBase:
    """MuJoCo owns every body and joint; VBD owns the particles; one lagged proxy MuJoCo -> VBD."""
    mj_kw = {"use_mujoco_contacts": False, "njmax": 64, "nconmax": 64, **dict(mujoco_kwargs or {})}
    vbd_kw = {"iterations": 10, **dict(vbd_kwargs or {})}
    bodies = list(range(model.body_count))
    entries = [
        SolverCoupledProxy.Entry(name="mjc", solver=lambda v: newton.solvers.SolverMuJoCo(model=v, **mj_kw),
                                 bodies=bodies, joints=list(range(model.joint_count))),
        SolverCoupledProxy.Entry(name="vbd", solver=lambda v: newton.solvers.SolverVBD(model=v, **vbd_kw),
                                 particles=list(range(model.particle_count))),
    ]
    proxy = SolverCoupledProxy.Proxy(source="mjc", destination="vbd", bodies=bodies, mode=mode,
                                     collision_pipeline=lambda v: newton.CollisionPipeline(v), collide_interval=1)
    config = SolverCoupledProxy.Config(proxies=[proxy], iterations=iterations)
    return SolverCoupledProxy(model=model, entries=entries, coupling=config)
