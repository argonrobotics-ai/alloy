# SPDX-License-Identifier: Apache-2.0
"""TOP-2 SimTwin: knife force vs depth on a pre-split block (SPEC §3.8, §5.2 TOP-2; rig law, phase 2).

Each world holds a block (agar or cucumber flesh) pre-split by one plane through its middle, with the bottom face
clamped, and a kinematic knife. The knife is a 0.1 mm-radius capsule edge plus a thin back box (GOTCHAS:538: the
real edge round, never a zero-thickness box). It descends through the plane at constant speed; VBD soft contact
pushes the material and the topology controller cuts the TET_CUT bonds (cut.py).

The recorded knife force is the reaction of the penalty soft contacts on the knife shapes,
``Σ ke·max(0, r − d)·(−n_z)`` over the pipeline's soft contacts. It is a re-evaluation of VBD's penalty law and
ignores friction and damping (UNVERIFIED against a force sensor until rig data exists). The scalars are this
force at fixed depths. The fit parameters are the blade's cut parameters (``CUT_KEYS``) plus ``youngs``.
Real data: not_run until the cucumber/agar rig exists (SPEC §5).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import warp as wp

from alloy._vendor import newton
from alloy.bench.types import Condition, TwinLog
from alloy.cards.types import Failure
from alloy.core.quantity import Q
from alloy.topology.bonds import bonds_from_model
from alloy.topology.controller import TopologyController
from alloy.topology.emit import add_presplit_soft
from alloy.topology.meshes import grid_tet_mesh
from alloy.topology.presplit import SplitPlan
from alloy.topology.types import Blade, TopologyBudget

log = logging.getLogger("alloy.topology")

LAW_ID = "TOP-2.knife_force"
DEPTHS = (1.0e-3, 2.0e-3, 3.0e-3)                     # [m] force sample depths
CUT_KEYS = ("softness", "cut_spring_ke")              # Blade cut parameters a sample may set (else Blade defaults)
MATERIALS = {"agar": {"youngs": 5.0e4, "poisson": 0.45, "density": 1010.0},        # UNVERIFIED nominal values
             "cucumber": {"youngs": 1.5e6, "poisson": 0.4, "density": 950.0}}


def lame(youngs: float, poisson: float) -> tuple[float, float]:
    """(mu, lambda) [Pa] from Young's modulus and Poisson's ratio."""
    return youngs / (2.0 * (1.0 + poisson)), youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))


def _clamp(model: newton.Model, ids: np.ndarray) -> None:
    for name in ("particle_mass", "particle_inv_mass"):
        arr = getattr(model, name).numpy()
        arr[ids] = 0.0
        getattr(model, name).assign(arr)


@dataclass
class _World:
    knife: int
    z0: float
    sample: Mapping[str, float]


class KnifeStepper(newton.solvers.SolverBase):
    """One step = topology pre_step, VBD step, topology apply (the §2.3 order for a single VBD entry)."""

    def __init__(self, model: newton.Model, controller: TopologyController, iterations: int = 10) -> None:
        super().__init__(model)
        self.controller = controller
        self.vbd = newton.solvers.SolverVBD(model, iterations=iterations)

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None) -> None:
        self.controller.pre_step(state_in, contacts, dt)
        self.vbd.step(state_in, state_out, control, contacts, dt)
        self.controller.apply(state_out)


@wp.kernel
def _drive_knives(knives: wp.array[wp.int32], z0: wp.array[wp.float32], x: wp.vec3, t: float, speed: float,
                  body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector]):
    w = wp.tid()
    body_q[knives[w]] = wp.transform(wp.vec3(x[0], x[1], z0[w] - speed * t), wp.quat_identity())
    body_qd[knives[w]] = wp.spatial_vector(wp.vec3(0.0, 0.0, -speed), wp.vec3(0.0))


@wp.kernel
def _knife_force(knives: wp.array[wp.int32], count: wp.array[wp.int32], particle: wp.array[wp.int32],
                 shape: wp.array[wp.int32], body_pos: wp.array[wp.vec3], normal: wp.array[wp.vec3],
                 shape_body: wp.array[wp.int32], body_q: wp.array[wp.transform], x: wp.array[wp.vec3],
                 radius: wp.array[wp.float32], ke: float, col: int, out: wp.array2d[wp.float32]):
    w = wp.tid()
    f = float(0.0)
    for k in range(count[0]):
        s = shape[k]
        if s < 0 or shape_body[s] != knives[w]:
            continue
        p = particle[k]
        c = wp.dot(normal[k], x[p] - wp.transform_point(body_q[knives[w]], body_pos[k])) - radius[p]
        if c < 0.0:
            f += ke * (-c) * (-normal[k][2])
    out[w, col] = f


@dataclass
class KnifeForceTwin:
    """SimTwin for TOP-2 (alloy.bench.types.SimTwin). Sample keys: youngs and ``CUT_KEYS``."""

    law_id: str = LAW_ID
    cells: tuple[int, int, int] = (8, 4, 4)
    cell: float = 2.5e-3
    speed: float = 0.02                                 # [m/s] knife descent
    contact_ke: float = 2.0e3                           # [N/m] VBD soft contact stiffness
    units: Mapping[str, str] = field(default_factory=lambda: {f"force_d{int(d * 1e3)}mm": "N" for d in DEPTHS})
    _worlds: list[_World] = field(default_factory=list)
    _clamp: list[int] = field(default_factory=list)

    @property
    def top(self) -> float:
        return self.cells[2] * self.cell

    @property
    def plane_x(self) -> float:
        return 0.5 * self.cells[0] * self.cell

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return (Condition("agar", dict(MATERIALS["agar"])),) if quick else tuple(
            Condition(k, dict(v)) for k, v in MATERIALS.items())

    def build_world(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        p = {**cond.params, **sample}
        mu, lam = lame(float(p["youngs"]), float(p["poisson"]))
        v, t = grid_tet_mesh(*self.cells, self.cell)
        p0 = b.particle_count
        add_presplit_soft(b, v, t, SplitPlan(planes=(((1.0, 0.0, 0.0), self.plane_x),)),
                          dict(pos=wp.vec3(0.0, 0.0, 0.0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0.0, 0.0, 0.0),
                               density=float(p["density"]), k_mu=mu, k_lambda=lam, k_damp=0.0, particle_radius=1.0e-3),
                          budget=TopologyBudget(spare_vertices=10_000),
                          failure=Failure(cut_min_strain=Q(float(p.get("cut_min_strain", 0.05)), "model",
                                                           note="TOP-2 twin gate")))
        self._clamp += [p0 + i for i in range(b.particle_count - p0) if b.particle_q[p0 + i][2] <= 1e-9]
        knife = b.add_body(xform=wp.transform(wp.vec3(self.plane_x, 0.5 * self.cells[1] * self.cell, self.top + 1e-3),
                                              wp.quat_identity()), mass=0.0)
        cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
        b.add_shape_capsule(knife, radius=1.0e-4, half_height=0.02, cfg=cfg,
                            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), 0.5 * np.pi)))
        b.add_shape_box(knife, hx=1.0e-4, hy=0.02, hz=0.01, cfg=cfg,
                        xform=wp.transform(wp.vec3(0.0, 0.0, 0.01), wp.quat_identity()))
        self._worlds.append(_World(knife, self.top + 1e-3, dict(p)))

    def make_pipeline(self, model: newton.Model, solver: newton.solvers.SolverBase) -> newton.CollisionPipeline:
        return newton.CollisionPipeline(model, soft_contact_gap=1.0e-3)

    def make_solver(self, model: newton.Model, solver: str = "vbd") -> newton.solvers.SolverBase:
        self._model = model
        _clamp(model, np.asarray(self._clamp, dtype=np.int64))
        model.soft_contact_ke, model.soft_contact_kd, model.soft_contact_mu = self.contact_ke, 1e-2, 0.3
        blades = [Blade(w.knife, -1, (0.0, -0.02, 0.0), (0.0, 0.02, 0.0),
                        **{k: float(w.sample[k]) for k in CUT_KEYS if k in w.sample}) for w in self._worlds]
        ctl = TopologyController(model, bonds_from_model(model), TopologyBudget(), blades)
        dev = model.device
        self._knives = wp.array([w.knife for w in self._worlds], dtype=wp.int32, device=dev)
        self._z0 = wp.array([w.z0 for w in self._worlds], dtype=wp.float32, device=dev)
        self._ke = float(model.soft_contact_ke)
        return KnifeStepper(model, ctl)

    def drive(self, step: int, t: float, state: newton.State, control: newton.Control) -> None:
        """The knife's pose and its descent velocity: VBD's contact friction and damping read the body's velocity, so
        a knife posed without it would slide past the material with no friction (WP15, gap #9)."""
        xy = wp.vec3(self.plane_x, 0.5 * self.cells[1] * self.cell, 0.0)
        wp.launch(_drive_knives, dim=len(self._worlds), device=state.body_q.device,
                  inputs=[self._knives, self._z0, xy, float(t), self.speed, state.body_q, state.body_qd])

    def record(self, step: int, state: newton.State, contacts: newton.Contacts | None, obs, log: TwinLog) -> None:
        col = log.column(step)
        if col < 0 or contacts is None:
            return
        model = self._model
        wp.launch(_knife_force, dim=len(self._worlds), device=model.device, inputs=[
            self._knives, contacts.soft_contact_count, contacts.soft_contact_particle, contacts.soft_contact_shape,
            contacts.soft_contact_body_pos, contacts.soft_contact_normal, model.shape_body, state.body_q,
            state.particle_q, model.particle_radius, self._ke, col, log.channel("force")])

    def scalars(self, log: TwinLog) -> dict[str, np.ndarray]:
        force = log.series("force")
        depth = self.top - (np.asarray([w.z0 for w in self._worlds])[:, None] - self.speed * log.times()[None, :]
                            - 1.0e-4)
        out = {}
        for d in DEPTHS:
            out[f"force_d{int(d * 1e3)}mm"] = np.array([np.interp(d, depth[w], force[w]) for w in range(len(force))])
        return out
