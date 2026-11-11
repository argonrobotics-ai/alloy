# SPDX-License-Identifier: Apache-2.0
"""PluginName: a minimal solver for free particles (gravity, linear drag, a bouncy ground plane at z = 0).

TEMPLATE: a complete, passing solver plug-in (SPEC §3.8 in its simplest form). A solver is a
``newton.solvers.SolverBase`` with ``step(state_in, state_out, control, contacts, dt)``; ``capability()`` registers
it (name, factory, what it owns). The harness runs ``test_scene()`` with ``solver=<name>`` and checks crash,
determinism, world independence and your laws. Check it with:  python -m alloy.plugins check <this file>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase

from alloy.core.protocols import DetCapability
from alloy.core.registry import SolverCapability
from alloy.plugins import Law, LawResult, SolverPlugin, TestScene


@wp.kernel
def _integrate(q: wp.array[wp.vec3], qd: wp.array[wp.vec3], f: wp.array[wp.vec3], inv_mass: wp.array[wp.float32],
               radius: wp.array[wp.float32], gravity: wp.vec3, drag: wp.float32, restitution: wp.float32,
               dt: wp.float32, q_out: wp.array[wp.vec3], qd_out: wp.array[wp.vec3]):
    i = wp.tid()                                     # one thread per particle: deterministic
    v = qd[i]
    if inv_mass[i] > 0.0:
        v = v + (gravity + (f[i] - drag * v) * inv_mass[i]) * dt     # semi-implicit Euler
    x = q[i] + v * dt
    r = radius[i]
    if x[2] < r and v[2] < 0.0:                      # ground plane at z = 0
        x = wp.vec3(x[0], x[1], r)
        v = wp.vec3(v[0], v[1], -restitution * v[2])
    q_out[i] = x
    qd_out[i] = v


class ParticleSolver(SolverBase):
    """Free particles with gravity, linear drag [N·s/m] and a ground plane with ``restitution``."""

    def __init__(self, model: newton.Model, drag: float = 0.0, restitution: float = 0.5) -> None:
        super().__init__(model)
        self.drag, self.restitution = float(drag), float(restitution)
        self.gravity = wp.vec3(*np.asarray(model.gravity.numpy()).reshape(-1, 3)[0])

    def step(self, state_in: newton.State, state_out: newton.State, control: Any, contacts: Any,
             dt: float) -> None:
        m = self.model
        f = state_in.particle_f if state_in.particle_f is not None else state_out.particle_qd
        wp.launch(_integrate, dim=m.particle_count,
                  inputs=[state_in.particle_q, state_in.particle_qd, f, m.particle_inv_mass, m.particle_radius,
                          self.gravity, self.drag, self.restitution, float(dt)],
                  outputs=[state_out.particle_q, state_out.particle_qd], device=m.device)


class PluginName(SolverPlugin):
    """A particle solver plug-in: gravity, drag and a ground plane with restitution (template).

    Example::

        register_solver(PluginName().capability())         # then entries can use capability "plugin_name"
    """

    name = "plugin_name"
    summary = "free-particle solver with drag and a bouncy ground (template)"

    @dataclass
    class Config:
        drag: float = 0.0              # N·s/m
        restitution: float = 0.5       # ground bounce, 0..1

    def capability(self) -> SolverCapability:
        kw = {"drag": self.config.drag, "restitution": self.config.restitution}
        return SolverCapability(name=self.name, factory=lambda model, extra: ParticleSolver(model, **{**kw, **extra}),
                                routes=frozenset(), owns=frozenset({"particle"}), coupling=False, uniform_worlds=False,
                                per_contact_friction=False, determinism=DetCapability.NATIVE, solver_cls=ParticleSolver)

    def test_scene(self) -> TestScene:
        """Ten particles dropped from 0.5-1.4 m onto the ground."""
        return TestScene(build=_particles, solver=self.name, ground=False, periods=60)

    def laws(self) -> list[Law]:
        fall = TestScene(build=_particles, solver=self.name, ground=False, periods=10, gravity=(0.0, 0.0, -9.81))
        return [Law("free_fall", _check_fall, fall, "z(t) = z0 - g t²/2 before the first bounce"),
                Law("stays_above_ground", _check_ground, None, "no particle below the plane")]


def _particles(b: newton.ModelBuilder) -> None:
    for k in range(10):
        b.add_particle((0.1 * k, 0.0, 0.5 + 0.1 * k), (0.0, 0.0, 0.0), mass=0.01, radius=0.01)


def _check_fall(run) -> LawResult:
    t = np.asarray(run.trace.t)
    z = run.trace.particle_q[:, 9, 2].astype(np.float64)     # the highest particle of world 0
    want = 1.4 - 0.5 * 9.81 * t**2
    return LawResult.within(float(np.abs(z - want).max()) / 1.4, 0.01, "max |z - (z0 - g t²/2)| / z0",
                            "integrate v += g dt then x += v dt")


def _check_ground(run) -> LawResult:
    low = float((run.trace.particle_q[..., 2] - 0.01).min())
    return LawResult.within(min(low, 0.0), 1e-6, f"lowest particle bottom {low:.3g} m", "clamp to the plane")
