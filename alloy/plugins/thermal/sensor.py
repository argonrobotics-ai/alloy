# SPDX-License-Identifier: Apache-2.0
"""``Thermal``: lumped temperatures of bodies and motors (engine item 10, heat v1). No feedback into dynamics.

Each thermal body is one temperature node, with heat capacity m·c_p and convection h·A to ``ambient``. A is its
collision shapes' surface area. Every step, from the solver's own contacts (``needs_contacts``, as tactile):
- **conduction** G = ``contact_conductance``·``contact_area`` through every compressive contact between two sides
  that have temperatures. A side is a thermal body, or a ``plate`` (a fixed-temperature body: a stove, a cold
  plate).
- **friction heat** |f_t·v_slip|, half to each side. The slip velocity is the step's mean.

Motors are optional two-node models (winding and housing). They take I²R heating from the ``motor_torque`` input
(I = τ/k_t) and report a derated ``torque_limit``: the full ``torque_max`` below ``t_derate``, falling linearly to 0
at ``t_max``. The limit is a reading for the controller; v1 does not change the physics.

**External heat** (input ``external_heat`` [W, B], int64 µJ). Other plug-ins put heat into a body with
``wp.atomic_add`` during a step (heat v2's particle heaters). The next update adds it to the body's energy, inside
the semi-implicit numerator, and zeroes it. That is one exchange of lag; the energy is exact.

Readings (``read()["thermal"]``):
- ``temperature`` [W, B] K;
- ``conductance`` [W, B] W/K (last step);
- ``friction_heat`` [W, B] J, cumulative;
- ``winding`` and ``housing`` [W, M] K;
- ``torque_limit`` [W, M] N·m;
- ``external_heat_total`` [W, B] J, cumulative.

Example::

    th = Thermal(bodies=("pan", "egg"), initial=(453.15, 278.15), specific_heat=(900.0, 3500.0), convection=10.0)
    sim = alloy.plugins.simulation(built, [th]); sim.step_period(); sim.plugins.read()["thermal"]["temperature"]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import SensorPlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.thermal import kernels as K

SPHERE, CAPSULE, CYLINDER, BOX = 3, 4, 6, 7      # newton.GeoType


def shape_area(kind: int, scale: np.ndarray) -> float:
    """Surface area [m²] of a primitive shape (Newton conventions: half extents, radius / half height)."""
    if kind == BOX:
        hx, hy, hz = scale[:3]
        return 8.0 * (hx * hy + hy * hz + hz * hx)
    if kind == SPHERE:
        return 4.0 * math.pi * scale[0] ** 2
    if kind == CAPSULE:
        return 4.0 * math.pi * scale[0] * scale[1] + 4.0 * math.pi * scale[0] ** 2
    if kind == CYLINDER:
        return 4.0 * math.pi * scale[0] * scale[1] + 2.0 * math.pi * scale[0] ** 2
    return 0.0


def _per(values: tuple, n: int, name: str) -> list:
    if len(values) not in (1, n):
        raise ValueError(f"{name}: one value, or one per body ({n})")
    return [values[min(i, len(values) - 1)] for i in range(n)]


class Thermal(SensorPlugin):
    """Lumped body and motor temperatures: convection, contact conduction, friction heating, I²R motors with a
    derated torque limit (see the module docstring)."""

    name = "thermal"
    summary = "lumped temperatures: convection, contact conduction, friction heat, motor I²R with derating"
    needs_contacts: ClassVar[bool] = True

    @dataclass
    class Config:
        bodies: tuple[str, ...] = ("box",)            # thermal bodies (labels)
        initial: tuple[float, ...] = (293.15,)       # K, one or per body
        specific_heat: tuple[float, ...] = (900.0,)  # J/(kg·K), one or per body
        ambient: float = 293.15                      # K
        convection: float = 10.0                     # W/(m²·K)
        contact_conductance: float = 1000.0          # W/(m²·K)
        contact_area: float = 1.0e-5                 # m² apparent patch per contact
        friction_heat: bool = True
        plates: tuple[str, ...] = ()                 # fixed-temperature bodies
        plate_temperatures: tuple[float, ...] = ()   # K, one per plate
        motors: tuple[str, ...] = ()                 # motor names; torque from input "motor_torque" [W, M]
        torque_constant: float = 0.1                 # N·m/A
        winding_resistance: float = 1.0              # Ω
        winding_capacity: float = 50.0               # J/K
        housing_capacity: float = 500.0              # J/K
        r_winding_housing: float = 1.0               # K/W
        r_housing_ambient: float = 2.0               # K/W
        t_derate: float = 373.15                     # K: derating starts
        t_max: float = 403.15                        # K: limit 0
        torque_max: float = 10.0                     # N·m

        def __post_init__(self) -> None:
            n = len(self.bodies)
            _per(self.initial, n, "initial")
            _per(self.specific_heat, n, "specific_heat")
            if len(self.plate_temperatures) != len(self.plates):
                raise ValueError("plate_temperatures: one per plate")
            if min(self.convection, self.contact_conductance, self.contact_area) < 0.0 or self.ambient <= 0.0:
                raise ValueError("convection, contact_conductance, contact_area >= 0; ambient > 0 K")
            if self.motors and (min(self.torque_constant, self.winding_capacity, self.housing_capacity,
                                    self.r_winding_housing, self.r_housing_ambient) <= 0.0
                                or self.t_max <= self.t_derate):
                raise ValueError("motor constants must be > 0 and t_max > t_derate")

    def setup(self, ctx: Any) -> None:
        c, W = self.config, ctx.world_count
        self.B, self.M = len(c.bodies), len(c.motors)
        therm, cap, harea = self._bodies(ctx)
        self._therm_of_body = ctx.array(therm, wp.int32)
        self._capacity, self._h_area = ctx.array(cap, wp.float64), ctx.array(harea, wp.float64)
        self._plate_of_body, self._plate_temp = self._plates(ctx)
        init = np.tile(np.asarray(_per(c.initial, self.B, "initial"), np.float64), W).reshape(W, self.B)
        self.temperature = self.add_state("temperature", ctx.array(init, wp.float64))
        self.conductance = self.add_state("conductance", ctx.zeros((W, self.B), wp.float64))
        self.friction_heat = self.add_state("friction_heat", ctx.zeros((W, self.B), wp.float64))
        self._acc = [ctx.zeros(W * self.B, wp.int64) for _ in range(3)]
        self.external_heat = self.add_input("external_heat", ctx.zeros((W, self.B), wp.int64))      # µJ, consumed
        self.external_heat_total = self.add_state("external_heat_total", ctx.zeros((W, self.B), wp.float64))
        m = max(self.M, 1)
        self.winding = self.add_state("winding", ctx.array(np.full((W, m), c.ambient), wp.float64))
        self.housing = self.add_state("housing", ctx.array(np.full((W, m), c.ambient), wp.float64))
        self.torque_limit = self.add_state("torque_limit", ctx.array(np.full((W, m), c.torque_max, np.float32)))
        self.motor_torque = self.add_input("motor_torque", ctx.zeros((W, m)))
        self._motor_prm = ctx.array(np.asarray([c.torque_constant, c.winding_resistance, c.winding_capacity,
                                                c.housing_capacity, c.r_winding_housing, c.r_housing_ambient,
                                                c.t_derate, c.t_max, c.torque_max], np.float64), wp.float64)
        self._flat = {k: a.reshape((-1,)) for k, a in (("temp", self.temperature), ("g", self.conductance),
                      ("fh", self.friction_heat), ("w", self.winding), ("h", self.housing),
                      ("lim", self.torque_limit), ("tau", self.motor_torque), ("ext", self.external_heat),
                      ("ext_total", self.external_heat_total))}

    def _bodies(self, ctx: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c, model = self.config, ctx.model
        cp = _per(c.specific_heat, self.B, "specific_heat")
        mass, sb = model.body_mass.numpy(), model.shape_body.numpy()
        kinds, scales = model.shape_type.numpy(), model.shape_scale.numpy()
        therm = np.full(model.body_count, -1, np.int32)
        cap = np.zeros(ctx.world_count * self.B)
        harea = np.zeros(ctx.world_count * self.B)
        for b, label in enumerate(c.bodies):
            for w, body in enumerate(ctx.bodies(label).numpy()):
                t = w * self.B + b
                therm[body] = t
                cap[t] = float(mass[body]) * cp[b]
                harea[t] = c.convection * sum(shape_area(int(kinds[s]), scales[s]) for s in np.flatnonzero(sb == body))
        if (cap <= 0.0).any():
            raise PluginError("thermal: every thermal body needs mass > 0 and specific_heat > 0")
        return therm, cap, harea

    def _plates(self, ctx: Any) -> tuple[wp.array, wp.array]:
        c, P = self.config, len(self.config.plates)
        owner = np.full(ctx.model.body_count, -1, np.int32)
        temps = np.zeros(max(ctx.world_count * P, 1))
        for p, label in enumerate(c.plates):
            for w, body in enumerate(ctx.bodies(label).numpy()):
                owner[body] = w * P + p
                temps[w * P + p] = c.plate_temperatures[p]
        return ctx.array(owner, wp.int32), ctx.array(temps, wp.float64)

    def measure(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        ctx, c, m, f = self.ctx, self.config, self.ctx.model, self._flat
        con, frc = ctx.contacts, ctx.contact_force
        if con is None or frc is None:
            raise PluginError("thermal needs contact forces: run it on the MuJoCo route (MuJoCo Warp), where the "
                              "plug-in host exports them")
        dev, W = ctx.device, ctx.world_count
        wp.launch(K.contact_heat, dim=con.rigid_contact_max, device=dev,
                  inputs=[con.rigid_contact_count, con.rigid_contact_shape0, con.rigid_contact_shape1,
                          con.rigid_contact_point0, con.rigid_contact_normal, frc, m.shape_body, state_in.body_q,
                          state_in.body_qd, state_out.body_qd, m.body_com, self._therm_of_body, self._plate_of_body,
                          f["temp"], self._plate_temp, float(c.contact_conductance * c.contact_area),
                          int(c.friction_heat)], outputs=self._acc)
        wp.launch(K.update_bodies, dim=W * self.B, device=dev,
                  inputs=[self._capacity, self._h_area, wp.float64(c.ambient), float(dt), *self._acc, f["ext"]],
                  outputs=[f["temp"], f["g"], f["fh"], f["ext_total"]])
        if self.M:
            wp.launch(K.update_motors, dim=W * self.M, device=dev,
                      inputs=[f["tau"], self._motor_prm, wp.float64(c.ambient), float(dt)],
                      outputs=[f["w"], f["h"], f["lim"]])

    def test_scene(self):
        from alloy.plugins.thermal.laws import float_scene  # noqa: PLC0415 - laws import this module

        return float_scene()

    def laws(self):
        from alloy.plugins.thermal.laws import thermal_laws  # noqa: PLC0415

        return thermal_laws()
