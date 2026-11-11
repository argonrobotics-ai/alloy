# SPDX-License-Identifier: Apache-2.0
"""A6 restitution: a sphere dropped from h0 rebounds to h1 = e^2 h0 (bench §4.5, SPEC §5.3 amendment).

Tolerance (§5.3): |h1 - e^2 h0| <= 0.02 h1 + 5 dt sqrt(2 g h1) (2 % plus a 5-step timing allowance expressed as a
length). The metric is the per-world ratio error / bound, tolerance 1.0. h1 is the apex height above the resting
centre height (radius) after the first impact.

§5.3: restitution goes through Alloy's damping path wherever the solver has one (`SolverCapability.restitution_path`,
alloy/scene/builtin_plugins.py), exactly as the assembler writes it, with the default material stiffness
(`DEFAULT_CONTACT_STIFFNESS` per material, series pair value via `derive.pair_stiffness`):
- MuJoCo (native contacts): `native_gains(K_material, m, e, dt=dt)` on the shapes of dynamic bodies (with the
  assembler's stability cap at the law dt) and `STATIC_SOLMIX` on static shapes (alloy/scene/mujoco_pairs.py
  `static_solmix`), so the contact takes the ball's own solref, as the routed scene does;
- Featherstone / SemiImplicit (Newton pipeline): per-contact `rigid_contact_stiffness = K_pair` and
  `rigid_contact_damping = 2 zeta(e) sqrt(K_pair m_eff)`, the `PairTable` rule (alloy/scene/types.py `PairTable`);
- XPBD has no damping path: `ShapeConfig.restitution` with `enable_restitution=True`
  (newton@ac7f7830:newton/_src/sim/builder.py:1009-1017).
`e_correction` stays 1 (no correction is fitted here).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from alloy._vendor import newton
import numpy as np
import warp as wp

from alloy.bench import default_route as DR
from alloy.bench.analytic.common import G, BaseTwin
from alloy.bench.solvers import RIGID_SOLVERS, SolverSettings
from alloy.bench.twin import DeviceTwinLog
from alloy.bench.types import Condition, LawSpec, Tolerance
from alloy.cards.defaults import DEFAULT_CONTACT_STIFFNESS
from alloy.scene import derive
from alloy.scene.mujoco_pairs import STATIC_SOLMIX
from alloy.scene.native_gains import native_gains

H0 = 0.1  # [m] drop height of the sphere bottom above the plane
RADIUS = 0.02  # [m]
E_VALUES = (0.0, 0.5, 0.8)
MU = 0.01  # vertical drop, friction irrelevant; MuJoCo warns that mu < 1e-5 may give NaN (mujoco_warp io.py:416)
K_MATERIAL = DEFAULT_CONTACT_STIFFNESS  # [N/m] per material (alloy/cards/defaults.py)
K_PAIR = derive.pair_stiffness(K_MATERIAL, K_MATERIAL)  # [N/m] series pair value the PairTable writes
PAIR_TABLE_SOLVERS = ("featherstone", "semi_implicit")

LAW = LawSpec(
    id="A6.restitution", suite="analytic",
    metrics={"rebound_err_ratio": Tolerance(abs=1.0, source="SPEC §5.3 A6: |h1 - e^2 h0| <= 0.02 h1 + 5 dt sqrt(2 g h1)")},
    stat={"rebound_err_ratio": "mean"},
    dt=1.25e-3, substeps=1, duration_s=1.0, worlds_per_condition=1, solvers=RIGID_SOLVERS,
)


@wp.kernel
def _pair_table_gains(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),
                      shape1: wp.array(dtype=wp.int32), shape_body: wp.array(dtype=wp.int32),
                      inv_mass: wp.array(dtype=float), shape_two_zeta: wp.array(dtype=float), k_pair: float,
                      stiffness: wp.array(dtype=float), damping: wp.array(dtype=float)):
    """PairTable rule per contact: K = k_pair, kd = 2 zeta(e) sqrt(K m_eff), m_eff from the bodies' inverse masses."""
    i = wp.tid()
    if i >= count[0]:
        return
    a, b = shape0[i], shape1[i]
    inv = float(0.0)
    if shape_body[a] >= 0:
        inv += inv_mass[shape_body[a]]
    if shape_body[b] >= 0:
        inv += inv_mass[shape_body[b]]
    stiffness[i] = k_pair
    damping[i] = 0.0
    if inv > 0.0:
        damping[i] = wp.max(shape_two_zeta[a], shape_two_zeta[b]) * wp.sqrt(k_pair / inv)


def first_rebound(t: np.ndarray, z: np.ndarray) -> float:
    """Apex height of z after its first minimum (the first impact), relative to nothing (raw z). NaN if no impact."""
    vz = np.diff(z)
    down = np.flatnonzero(vz < 0)
    if down.size == 0:
        return math.nan
    after = np.flatnonzero((np.arange(vz.size) > down[0]) & (vz >= 0))
    if after.size == 0:
        return math.nan
    i_min = int(after[0])
    falling_again = np.flatnonzero((np.arange(vz.size) > i_min) & (vz < 0))
    end = int(falling_again[0]) + 1 if falling_again.size else z.size
    return float(np.max(z[i_min:end]))


class RestitutionTwin(BaseTwin):
    law_id = LAW.id
    units = {"rebound_err_ratio": "1"}
    settings = {"xpbd": SolverSettings("xpbd", {"enable_restitution": True})}

    def conditions(self, quick: bool) -> tuple[Condition, ...]:
        return tuple(Condition(f"e{e:g}", {"e": e}) for e in E_VALUES)

    def __init__(self) -> None:
        super().__init__()
        self._two_zeta: dict[int, float] = {}
        self._shape_two_zeta: wp.array | None = None
        self._builder_solver = ""
        self._balls: list[tuple[int, int, float]] = []

    def prepare_builder(self, b: newton.ModelBuilder, solver: str) -> None:
        self._builder_solver = solver

    def build(self, b: newton.ModelBuilder, cond: Condition, sample: Mapping[str, float]) -> None:
        e = cond.params["e"]
        cfg = newton.ModelBuilder.ShapeConfig(mu=MU, restitution=e)
        ground = b.add_shape_plane(plane=(0.0, 0.0, 1.0, 0.0), width=0.0, length=0.0, cfg=cfg)  # per-world ground
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + H0), wp.quat_identity()), label="ball")
        ball = b.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        if self._builder_solver == "mujoco":  # the assembler's static solmix; ball gains follow in on_model
            b.custom_attributes["mujoco:geom_solmix"].values[ground] = STATIC_SOLMIX
        two_zeta = 2.0 * derive.zeta_from_restitution(e)
        self._two_zeta.update({ground: two_zeta, ball: two_zeta})
        self._balls.append((ball, body, e))

    def on_model(self, model: newton.Model) -> None:
        z = np.zeros(model.shape_count, dtype=np.float32)
        for shape, value in self._two_zeta.items():
            z[shape] = value
        self._shape_two_zeta = wp.array(z, dtype=float, device=model.device)
        if self.solver_name == "mujoco":
            self._write_native_gains(model)

    def _write_native_gains(self, model: newton.Model) -> None:
        """Ball gains as the assembler writes them for the scene's gains mode (SCR-WP6-4: rk4 takes a quarter of the
        stability cap, and so do F6 R1 scenes, which run implicitfast on rk4's gains)."""
        integrator = DR.routed_gains(model)
        ke, kd, mass = model.shape_material_ke.numpy(), model.shape_material_kd.numpy(), model.body_mass.numpy()
        for shape, body, e in self._balls:
            ke[shape], kd[shape] = native_gains(K_MATERIAL, float(mass[body]), e, dt=LAW.dt, integrator=integrator)
        model.shape_material_ke.assign(ke)
        model.shape_material_kd.assign(kd)
        self.monitors["integrator"] = DR.routed_integrator(model)
        self.monitors["gains"] = integrator

    def make_contacts(self, pipeline: newton.CollisionPipeline) -> newton.Contacts:
        if self.solver_name not in PAIR_TABLE_SOLVERS:
            return pipeline.contacts()
        return newton.Contacts(pipeline.rigid_contact_max, 0, device=self.model.device,
                               per_contact_shape_properties=True)

    def after_collide(self, state: Any, contacts: newton.Contacts) -> None:
        if self.solver_name not in PAIR_TABLE_SOLVERS:
            return
        m = self.model
        wp.launch(_pair_table_gains, dim=contacts.rigid_contact_max,
                  inputs=[contacts.rigid_contact_count, contacts.rigid_contact_shape0, contacts.rigid_contact_shape1,
                          m.shape_body, m.body_inv_mass, self._shape_two_zeta, K_PAIR],
                  outputs=[contacts.rigid_contact_stiffness, contacts.rigid_contact_damping], device=m.device)

    def record(self, step: int, state: Any, contacts: Any, obs: Any, log: DeviceTwinLog) -> None:
        self.record_first_body(step, state, log)

    def scalars(self, log: DeviceTwinLog) -> dict[str, np.ndarray]:
        t = log.times()
        z = log.series("z").astype(np.float64)
        e = self.param("e")
        h1 = np.array([first_rebound(t, z[w]) for w in range(len(self.worlds))]) - RADIUS
        h1 = np.where(np.isfinite(h1), np.maximum(h1, 0.0), h1)
        law = e * e * H0
        bound = 0.02 * h1 + 5.0 * log.dt * np.sqrt(2.0 * G * np.maximum(h1, 0.0))
        err = np.abs(h1 - law)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(bound > 0, err / bound, np.where(err == 0, 0.0, math.inf))
        self.monitors.update({"h1_m": h1.tolist(), "h1_law_m": law.tolist(), "bound_m": bound.tolist()})
        return {"rebound_err_ratio": np.where(np.isfinite(h1), ratio, math.inf)}


__all__ = ["LAW", "RestitutionTwin", "first_rebound"]
