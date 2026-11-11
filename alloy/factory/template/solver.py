# SPDX-License-Identifier: Apache-2.0
"""Template plug-in solver: particles under gravity on a penalty ground plane (SPEC §3.8 contract, smallest form).

Agents start from this package. It shows every contract item: SolverBase + CouplingInterface, per-particle
material data as custom attributes in the plug-in's namespace, fixed capacity (all arrays allocated in
``__init__``), per-world ``reset`` and reductions, no host readback in ``step``, no atomics (NATIVE determinism),
``alloy_state_arrays`` and ``alloy_capacity_report``, and the ``StabilityEstimator`` pair
(``alloy_stable_dt``, ``alloy_energy``).

Physics: symplectic Euler, ``v' = v + dt (g + f/m)``, ``x' = x + dt v'``, with a ground spring-damper
``f_z = ke·(d - dt·v'_z) - kd·v'_z`` taken implicitly in ``v'_z`` whenever the step would end with the particle
sphere overlapping the plane (``d = ground_z + r - z``, predictive), and only while it pushes. The bully found
both earlier versions wrong: an explicit spring at the B0 scene's omega·dt = 0.56 gained 3.6 % of E_scale per
step, and an implicit spring switched on one step late injected the overlap energy ke·(v·dt)²/2 (2.3 %).
``alloy_stable_dt`` reports ``sqrt(m/ke)``, the step that keeps omega·dt <= 1 for accuracy.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy._vendor.newton.solvers import SolverBase

from alloy.core.newton_compat import CouplingInterface
from alloy.core.protocols import DetCapability

if TYPE_CHECKING:
    from alloy.cards.types import ResolvedObject
    from alloy.route.types import RouteDecision

PLUGIN_NAME = "template"  # the factory renames this one constant when it copies the template
NAMESPACE = PLUGIN_NAME  # custom attributes live under the plug-in's namespace (§3.8 item 2)
log = logging.getLogger("alloy.factory")


@wp.kernel
def _step_kernel(
    q_in: wp.array[wp.vec3],
    qd_in: wp.array[wp.vec3],
    f_ext: wp.array[wp.vec3],
    inv_mass: wp.array[float],
    radius: wp.array[float],
    flags: wp.array[wp.int32],
    world: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    ground_ke: wp.array[float],
    ground_kd: wp.array[float],
    ground_z: float,
    dt: float,
    q_out: wp.array[wp.vec3],
    qd_out: wp.array[wp.vec3],
):
    p = wp.tid()
    x = q_in[p]
    v = qd_in[p]
    if (flags[p] & 1) == 0 or inv_mass[p] == 0.0:  # spare or pinned: unchanged
        q_out[p] = x
        qd_out[p] = v
        return
    w = wp.max(world[p], 0)
    v = v + dt * (gravity[w] + f_ext[p] * inv_mass[p])
    d = ground_z + radius[p] - x[2]  # overlap now (negative = gap)
    if d - dt * v[2] > 0.0:  # predictive: the step would end overlapping
        # v'_z (1 + dt kd/m + dt^2 ke/m) = v_z + dt ke d / m  (implicit spring-damper on the end overlap d - dt v'_z)
        k = ground_ke[p] * inv_mass[p]
        c = ground_kd[p] * inv_mass[p]
        vz = (v[2] + dt * k * d) / (1.0 + dt * c + dt * dt * k)
        if d - dt * vz > 0.0:  # the spring only pushes (no adhesion)
            v = wp.vec3(v[0], v[1], vz)
    q_out[p] = x + dt * v
    qd_out[p] = v


@wp.kernel
def _reset_kernel(
    q0: wp.array[wp.vec3],
    qd0: wp.array[wp.vec3],
    world: wp.array[wp.int32],
    mask: wp.array[wp.bool],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    p = wp.tid()
    w = world[p]
    if w < 0:
        w = mask.shape[0] - 1  # global entities use the last mask slot (SolverBase.reset)
    if mask[w]:
        q[p] = q0[p]
        qd[p] = qd0[p]


@wp.kernel
def _stable_dt_kernel(
    mass: wp.array[float],
    ground_ke: wp.array[float],
    start: wp.array[wp.int32],
    index: wp.array[wp.int32],
    out: wp.array[wp.float32],
):
    w = wp.tid()
    best = float(1.0e30)
    for i in range(start[w], start[w + 1]):
        p = index[i]
        if mass[p] > 0.0 and ground_ke[p] > 0.0:
            best = wp.min(best, wp.sqrt(mass[p] / ground_ke[p]))  # omega dt <= 1 (accuracy; implicit is stable)
    out[w] = wp.float32(best)


@wp.kernel
def _energy_kernel(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    mass: wp.array[float],
    radius: wp.array[float],
    flags: wp.array[wp.int32],
    gravity: wp.array[wp.vec3],
    ground_ke: wp.array[float],
    ground_z: float,
    z_ref: wp.array[wp.float64],
    start: wp.array[wp.int32],
    index: wp.array[wp.int32],
    out: wp.array[wp.float64],
):
    w = wp.tid()
    e = wp.float64(0.0)
    for i in range(start[w], start[w + 1]):
        p = index[i]
        if (flags[p] & 1) != 0 and mass[p] > 0.0:
            v = qd[p]
            x = q[p]
            d = wp.max(ground_z + radius[p] - x[2], 0.0)
            h = wp.float64(x[2]) - z_ref[w]
            e += wp.float64(0.5 * mass[p] * wp.dot(v, v) + 0.5 * ground_ke[p] * d * d)
            e += wp.float64(-mass[p] * gravity[w][2]) * h
    out[w] = e


@dataclass(frozen=True)
class TemplateConfig:
    """Solver-wide settings; per-particle material comes from ``model.template.*``."""

    ground_z: float = 0.0  # [m]
    default_ke: float = 1.0e4  # [N/m] used when the custom attribute is absent
    default_kd: float = 5.0  # [N·s/m]


class SolverTemplate(SolverBase, CouplingInterface):
    """Particle-on-plane plug-in; copy this package to ``alloy/solvers/contrib/<name>/`` and change the physics."""

    plugin_name: ClassVar[str] = PLUGIN_NAME
    routes: ClassVar[frozenset[str]] = frozenset({f"contrib:{PLUGIN_NAME}"})  # never a router Route until shipped
    determinism: ClassVar[DetCapability] = DetCapability.NATIVE

    def __init__(self, model: newton.Model, config: TemplateConfig | None = None) -> None:
        super().__init__(model)
        self.config = config or TemplateConfig()
        n, dev = model.particle_count, model.device
        ns = getattr(model, NAMESPACE, None)
        self.ground_ke = _attr_or(ns, "ground_ke", n, self.config.default_ke, dev)
        self.ground_kd = _attr_or(ns, "ground_kd", n, self.config.default_kd, dev)
        self._zero_f = wp.zeros(n, dtype=wp.vec3, device=dev)
        self._q0 = wp.clone(model.particle_q)
        self._qd0 = wp.clone(model.particle_qd)
        start, index = _particle_csr(model)
        self._start = wp.array(start, dtype=wp.int32, device=dev)
        self._index = wp.array(index, dtype=wp.int32, device=dev)
        self._steps = 0

    @staticmethod
    def register_custom_attributes(builder: newton.ModelBuilder) -> None:
        for name, default in (("ground_ke", TemplateConfig.default_ke), ("ground_kd", TemplateConfig.default_kd)):
            builder.add_custom_attribute(newton.ModelBuilder.CustomAttribute(
                name=name, dtype=wp.float32, default=default,
                frequency=newton.Model.AttributeFrequency.PARTICLE, namespace=NAMESPACE))

    @staticmethod
    def emit(builder: newton.ModelBuilder, obj: ResolvedObject, decision: RouteDecision) -> None:
        """A 2 x 2 x 2 particle lattice filling the object's OBB (bottom face on z = 0); mass from the card density."""
        l1, l2, l3 = obj.extents
        n = 2
        mass = obj.material.density.value * l1 * l2 * l3 / n**3
        radius = 0.25 * min(l1, l2, l3)
        for i, j, k in itertools.product(range(n), repeat=3):
            pos = (((i + 0.5) / n - 0.5) * l1, ((j + 0.5) / n - 0.5) * l2, (k + 0.5) / n * l3)
            builder.add_particle(pos, (0.0, 0.0, 0.0), mass, radius=radius)
        log.debug("%s.emit %s: %d particles of %.4g kg (route %s)", PLUGIN_NAME, obj.card.name, n**3, mass,
                  decision.route)

    def step(self, state_in: newton.State, state_out: newton.State, control: newton.Control | None,
             contacts: newton.Contacts | None, dt: float, *, observables: SolverBase.Observables | None = None
             ) -> None:
        m = self.model
        if m.particle_count == 0:
            return
        f_ext = state_in.particle_f if state_in.particle_f is not None else self._zero_f
        wp.launch(_step_kernel, dim=m.particle_count, device=m.device, inputs=[
            state_in.particle_q, state_in.particle_qd, f_ext, m.particle_inv_mass, m.particle_radius,
            m.particle_flags, m.particle_world, m.gravity, self.ground_ke, self.ground_kd,
            self.config.ground_z, dt], outputs=[state_out.particle_q, state_out.particle_qd])
        self._steps += 1

    def reset(self, state: newton.State, world_mask: wp.array | None = None, flags: int | None = None) -> None:
        m = self.model
        if m.particle_count == 0:
            return
        mask = world_mask
        if mask is None:
            mask = wp.array(np.ones(m.world_count + 1, dtype=bool), dtype=wp.bool, device=m.device)
        wp.launch(_reset_kernel, dim=m.particle_count, device=m.device,
                  inputs=[self._q0, self._qd0, m.particle_world, mask], outputs=[state.particle_q, state.particle_qd])

    # StabilityEstimator (§3.1)
    def alloy_stable_dt(self, state: newton.State, contacts: newton.Contacts | None, out: wp.array) -> None:
        wp.launch(_stable_dt_kernel, dim=self.model.world_count, device=self.model.device,
                  inputs=[self.model.particle_mass, self.ground_ke, self._start, self._index], outputs=[out])

    def alloy_energy(self, state: newton.State, z_ref: wp.array, out: wp.array) -> None:
        m = self.model
        wp.launch(_energy_kernel, dim=m.world_count, device=m.device, inputs=[
            state.particle_q, state.particle_qd, m.particle_mass, m.particle_radius, m.particle_flags, m.gravity,
            self.ground_ke, self.config.ground_z, z_ref, self._start, self._index], outputs=[out])

    # AlloySolverPlugin (§3.1)
    def alloy_state_arrays(self, state: newton.State) -> Mapping[str, wp.array]:
        return {"particle_q": state.particle_q, "particle_qd": state.particle_qd}

    def alloy_state_worlds(self) -> Mapping[str, wp.array]:
        return {"particle_q": self.model.particle_world, "particle_qd": self.model.particle_world}

    def alloy_capacity_report(self) -> Mapping[str, int]:
        return {"particles": int(self.model.particle_count), "overflow": 0}


def _attr_or(ns: object | None, name: str, n: int, default: float, device: str) -> wp.array:
    arr = getattr(ns, name, None) if ns is not None else None
    if arr is not None:
        return arr
    return wp.full(n, default, dtype=wp.float32, device=device)


def _particle_csr(model: newton.Model) -> tuple[np.ndarray, np.ndarray]:
    world = model.particle_world.numpy() if model.particle_count else np.zeros(0, np.int32)
    keep = np.nonzero(world >= 0)[0]
    order = keep[np.argsort(world[keep], kind="stable")]
    counts = np.bincount(world[keep], minlength=model.world_count)
    return np.concatenate([[0], np.cumsum(counts)]).astype(np.int32), order.astype(np.int32)
