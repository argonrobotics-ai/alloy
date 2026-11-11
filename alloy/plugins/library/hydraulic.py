# SPDX-License-Identifier: Apache-2.0
"""A single-acting hydraulic piston: a cylinder object and a lumped-fluid force plug-in.

``HydraulicCylinder`` (object) builds a barrel (fixed to the world, or free) and a rod on a prismatic joint along
the barrel's local +z, with a load plate on the rod's top.

``Hydraulic`` (force) models the chamber as one lumped fluid volume ``V`` (m³) with an effective bulk modulus
``K`` (Pa; oil plus hose and air compliance):

    x   = rod extension along the axis since the start            V_c = dead_volume + A·x   (A = π·bore²/4)
    P   = max(K·(V - V_c)/V_c, min_pressure)                      gauge pressure [Pa]
    F   = (P·A - damping·ẋ) on the rod along +axis, -F on the barrel, both at the rod anchor
    dV/dt = Q_pump - Q_out,   Q_out = relief_conductance·max(P - relief_pressure, 0) + leak_conductance·max(P, 0)

``pump_flow`` (m³/s, one per world) is an input: ``sim.plugins["hydraulic"].set_input("pump_flow", 1e-5)``.
Forces are explicit, so the hydraulic stiffness K·A²/V_c limits the step: ``check_dt`` says what to change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces, ObjectPlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench, com_world, point_velocity, wrench_at
from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform
from alloy.plugins.npgeom import quat_rotate, transform_point


class HydraulicCylinder(ObjectPlugin):
    """A barrel and a rod on a prismatic joint (local +z), with a load plate; labels ``<label>_barrel/_rod``.

    Example::

        HydraulicCylinder(bore=0.02, stroke=0.1).build(builder, wp.transform_identity(), "cylinder")
    """

    name = "hydraulic_cylinder"
    summary = "barrel + rod on a prismatic joint with a load plate (pairs with the hydraulic force plug-in)"
    drop_height = 0.0

    @dataclass
    class Config:
        bore: float = 0.02               # m, piston diameter
        stroke: float = 0.10             # m, rod travel out from the start (joint upper limit)
        retract: float = 0.02            # m, rod travel in from the start (joint lower limit)
        barrel_length: float = 0.12      # m
        plate_mass: float = 0.5          # kg on the rod's top
        limit_stiffness: float = 1.0e5   # N/m, end stops
        limit_damping: float = 500.0     # N·s/m, end stops (about critical for 1 kg)
        mount: str = "fixed"             # "fixed" to the world at xform, or "free"

        def __post_init__(self) -> None:
            if self.mount not in ("fixed", "free"):
                raise ValueError(f"mount must be 'fixed' or 'free', got {self.mount!r}")
            if min(self.bore, self.stroke, self.barrel_length, self.plate_mass) <= 0 or self.retract < 0:
                raise ValueError("bore, stroke, barrel_length, plate_mass must be > 0 and retract >= 0")

    def build(self, builder: newton.ModelBuilder, xform: wp.transform, label: str) -> list[int]:
        c = self.config
        r, h = 0.5 * c.bore, 0.5 * c.barrel_length
        steel = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.5)
        base = wp.transform_multiply(xform, body_xform(0.0, 0.0, h))
        barrel = builder.add_link(xform=base, label=f"{label}_barrel")
        builder.add_shape_box(barrel, hx=r + 0.004, hy=r + 0.004, hz=h, cfg=steel)
        rod = builder.add_link(xform=wp.transform_multiply(base, body_xform(0.0, 0.0, h)), label=f"{label}_rod")
        builder.add_shape_box(rod, xform=body_xform(0.0, 0.0, -0.5 * h), hx=0.25 * r, hy=0.25 * r, hz=0.5 * h,
                              cfg=steel)
        plate_hz = 0.005
        plate = newton.ModelBuilder.ShapeConfig(density=c.plate_mass / (0.04 * 0.04 * 2 * plate_hz), mu=0.5)
        builder.add_shape_box(rod, xform=body_xform(0.0, 0.0, plate_hz), hx=0.02, hy=0.02, hz=plate_hz, cfg=plate)
        if c.mount == "fixed":
            j0 = builder.add_joint_fixed(-1, barrel, parent_xform=base, label=f"{label}_mount")
        else:
            j0 = builder.add_joint_free(barrel, label=f"{label}_mount")
        j1 = builder.add_joint_prismatic(barrel, rod, parent_xform=body_xform(0.0, 0.0, h), axis=(0.0, 0.0, 1.0),
                                         limit_lower=-c.retract, limit_upper=c.stroke, limit_ke=c.limit_stiffness,
                                         limit_kd=c.limit_damping, label=f"{label}_slide")
        builder.add_articulation([j0, j1], label=label)
        return [barrel, rod]


@wp.kernel
def _piston_force(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector],
                  body_com: wp.array[wp.vec3], barrel: wp.array[wp.int32], rod: wp.array[wp.int32],
                  axis: wp.vec3, anchor: wp.vec3, x0: wp.array[wp.float32], area: wp.float32,
                  dead: wp.float32, bulk: wp.float32, damping: wp.float32, p_min: wp.float32,
                  volume: wp.array[wp.float64], pressure: wp.array[wp.float32], extension: wp.array[wp.float32],
                  body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    cb, rb = barrel[w], rod[w]
    a = wp.transform_vector(body_q[cb], axis)
    p = wp.transform_point(body_q[rb], anchor)
    x = wp.dot(p - wp.transform_get_translation(body_q[cb]), a) - x0[w]
    vc = wp.max(dead + area * x, 1.0e-3 * dead)
    pr = wp.max(bulk * (float(volume[w]) - vc) / vc, p_min)
    ccb, crb = com_world(body_q, body_com, cb), com_world(body_q, body_com, rb)
    xdot = wp.dot(point_velocity(body_qd, crb, rb, p) - point_velocity(body_qd, ccb, cb, p), a)
    f = (pr * area - damping * xdot) * a
    add_wrench(body_f, rb, wrench_at(f, p, crb))
    add_wrench(body_f, cb, wrench_at(-f, p, ccb))
    pressure[w] = pr
    extension[w] = x


@wp.kernel
def _fluid_update(pressure: wp.array[wp.float32], pump: wp.array[wp.float32], relief_p: wp.float32,
                  relief_g: wp.float32, leak_g: wp.float32, dt: wp.float64, volume: wp.array[wp.float64],
                  pumped: wp.array[wp.float64], vented: wp.array[wp.float64]):
    w = wp.tid()
    pr = pressure[w]
    q_out = relief_g * wp.max(pr - relief_p, 0.0) + leak_g * wp.max(pr, 0.0)
    q_in = pump[w]
    volume[w] = volume[w] + (wp.float64(q_in) - wp.float64(q_out)) * dt
    pumped[w] = pumped[w] + wp.float64(q_in) * dt
    vented[w] = vented[w] + wp.float64(q_out) * dt


class Hydraulic(ForcePlugin):
    """Lumped single-chamber hydraulics between a barrel and a rod body: pressure from fluid volume, pump input,
    relief valve, leak, viscous damping. Pairs with ``HydraulicCylinder``.

    Example::

        hyd = Hydraulic(barrel="cylinder_barrel", rod="cylinder_rod", bore=0.02)
        sim = alloy.plugins.simulation(built, [hyd]); hyd.set_input("pump_flow", 1e-5)   # m³/s, every world
    """

    name = "hydraulic"
    summary = "single-acting hydraulic piston: lumped fluid volume, pump input, relief valve, leak, damping"
    internal = True

    @dataclass
    class Config:
        barrel: str = "cylinder_barrel"                      # body label
        rod: str = "cylinder_rod"                            # body label
        axis: tuple[float, float, float] = (0.0, 0.0, 1.0)   # rod direction in the barrel frame
        anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)  # force point in the rod frame (on the axis)
        bore: float = 0.02                                   # m
        dead_volume: float = 3.0e-5                          # m³ at the start extension (chamber + hose)
        bulk_modulus: float = 2.0e7                          # Pa, effective (oil + hose + air)
        damping: float = 200.0                               # N·s/m along the axis (seals, orifice)
        relief_pressure: float = 5.0e6                       # Pa gauge
        relief_conductance: float = 1.0e-9                   # m³/(s·Pa) above the relief pressure
        leak_conductance: float = 0.0                        # m³/(s·Pa) to tank
        min_pressure: float = -9.0e4                         # Pa gauge (vapour / cavitation floor)

        def __post_init__(self) -> None:
            if min(self.bore, self.dead_volume, self.bulk_modulus) <= 0:
                raise ValueError("bore, dead_volume and bulk_modulus must be > 0")
            if min(self.damping, self.relief_conductance, self.leak_conductance) < 0:
                raise ValueError("damping and conductances must be >= 0")

    @property
    def area(self) -> float:
        return math.pi * 0.25 * self.config.bore ** 2

    def setup(self, ctx: Any) -> None:
        c, model, W = self.config, ctx.model, ctx.world_count
        self.barrel, self.rod = ctx.bodies(c.barrel), ctx.bodies(c.rod)
        q = model.body_q.numpy().astype(np.float64)
        axis = np.asarray(c.axis, np.float64) / np.linalg.norm(c.axis)
        x0 = [_extension(q[cb], q[rb], axis, np.asarray(c.anchor)) for cb, rb in
              zip(self.barrel.numpy(), self.rod.numpy(), strict=True)]
        self.axis = wp.vec3(*axis)
        self.x0 = ctx.array(np.asarray(x0, np.float32))
        self.volume = self.add_state("volume", ctx.array(np.full(W, c.dead_volume), wp.float64))
        self.pumped = self.add_state("pumped", ctx.zeros(W, wp.float64))
        self.vented = self.add_state("vented", ctx.zeros(W, wp.float64))
        self.pressure = self.add_state("pressure", ctx.zeros(W))
        self.extension = self.add_state("extension", ctx.zeros(W))
        self.pump_flow = self.add_input("pump_flow", ctx.zeros(W))
        self._rod_mass = float(model.body_mass.numpy()[self.rod.numpy()].min())

    def check_dt(self, dt: float) -> None:
        c = self.config
        k = c.bulk_modulus * self.area ** 2 / c.dead_volume
        omega = math.sqrt(k / self._rod_mass)
        relief = c.bulk_modulus * (c.relief_conductance + c.leak_conductance) * dt / c.dead_volume
        safe = min(1.0 / omega, self._rod_mass / max(c.damping, 1e-9))
        if omega * dt > 1.0 or c.damping * dt / self._rod_mass > 1.0 or relief > 1.0:
            raise PluginError(
                f"hydraulic: unstable at dt = {dt:.3g} s (stiffness K·A²/V = {k:.3g} N/m on a {self._rod_mass:.3g} kg "
                f"rod: ω·dt = {omega * dt:.2f}; damping·dt/m = {c.damping * dt / self._rod_mass:.2f}; relief "
                f"K·G·dt/V = {relief:.2f}; each must be <= 1). Lower bulk_modulus, raise dead_volume, lower "
                f"damping or the conductances, or step at dt <= {safe:.2g} s")

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        c = self.config
        wp.launch(_piston_force, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, self.ctx.model.body_com, self.barrel, self.rod, self.axis,
                          wp.vec3(*c.anchor), self.x0, self.area, c.dead_volume, c.bulk_modulus, c.damping,
                          c.min_pressure, self.volume],
                  outputs=[self.pressure, self.extension, forces.body_f], device=self.ctx.device)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        c = self.config
        wp.launch(_fluid_update, dim=self.ctx.world_count,
                  inputs=[self.pressure, self.pump_flow, c.relief_pressure, c.relief_conductance,
                          c.leak_conductance, wp.float64(dt)],
                  outputs=[self.volume, self.pumped, self.vented], device=self.ctx.device)

    def test_scene(self) -> TestScene:
        """A fixed vertical cylinder; the pump pushes 10 mL/s, so the rod rises at Q/A ≈ 3 cm/s."""
        return _cylinder_scene(self.label, pump=1.0e-5, periods=25)

    def laws(self) -> list[Law]:
        return [
            Law("volume_balance", _check_balance, None, "V - V0 = pumped - vented, every period"),
            Law("pressure_holds_load", _check_hold, _cylinder_scene(self.label, 0.0, 50), "P·A = m·g at rest"),
            Law("pump_speed", _check_speed, _cylinder_scene(self.label, 1.0e-5, 40), "rod speed = Q/A"),
            Law("relief_caps_pressure", _check_relief, _cylinder_scene(self.label, 2.0e-5, 60, stroke=0.01),
                "blocked rod: P settles at relief_pressure + Q/G", {"relief_pressure": 5.0e5}),
        ]


def _extension(qb: np.ndarray, qr: np.ndarray, axis: np.ndarray, anchor: np.ndarray) -> float:
    return float((transform_point(qr, anchor) - qb[:3]) @ quat_rotate(qb[3:7], axis))


def _cylinder_scene(label: str, pump: float, periods: int, stroke: float = 0.10) -> TestScene:
    def build(b: newton.ModelBuilder) -> None:
        HydraulicCylinder(stroke=stroke).build(b, wp.transform_identity(), "cylinder")

    def init(sim: Any) -> None:
        if label in sim.plugins.plugins:              # the speed baseline runs the scene without the plug-in
            sim.plugins[label].set_input("pump_flow", pump)

    return TestScene(build=build, init=init, periods=periods)


def _hyd(run: Any) -> Hydraulic:
    return run.plugin


def _series(run: Any, name: str) -> np.ndarray:
    return run.trace.plugin(_hyd(run).label, name).astype(np.float64)        # [K, W]


def _check_balance(run: Any) -> LawResult:
    v, pumped, vented = _series(run, "volume"), _series(run, "pumped"), _series(run, "vented")
    err = float(np.abs(v - v[0] - pumped + vented).max())
    moved = float(np.abs(pumped).max())
    return LawResult.within(err, 1e-12 + 1e-9 * float(v[0].max()), f"m³ of fluid unaccounted ({moved:.3g} m³ pumped)",
                            "integrate every flow into the volume and into its pumped/vented ledger")


def _check_hold(run: Any) -> LawResult:
    hyd = _hyd(run)
    rod = int(run.bodies(hyd.config.rod)[0])
    m = float(run.model.body_mass.numpy()[rod])
    want = m * 9.81 / hyd.area
    got = float(_series(run, "pressure")[-1, 0])
    return LawResult.within((got - want) / want, 0.02, f"pressure {got:.4g} Pa vs m·g/A {want:.4g} Pa at rest",
                            "force = pressure x bore area along the axis")


def _check_speed(run: Any) -> LawResult:
    hyd = _hyd(run)
    x, t = _series(run, "extension")[:, 0], np.asarray(run.trace.t)
    k = len(t) // 2
    speed = float((x[-1] - x[k]) / (t[-1] - t[k]))
    want = 1.0e-5 / hyd.area
    return LawResult.within((speed - want) / want, 0.03, f"rod speed {speed * 1e3:.3g} mm/s vs Q/A {want * 1e3:.3g}",
                            "chamber volume = dead_volume + area x extension; pressure from the volume difference")


def _check_relief(run: Any) -> LawResult:
    c = _hyd(run).config
    p = _series(run, "pressure")[:, 0]
    want = c.relief_pressure + 2.0e-5 / c.relief_conductance
    over = float(p.max() / want - 1.0)
    end = float(p[-1] / want - 1.0)
    detail = f"blocked rod: end {p[-1]:.4g} Pa, peak {p.max():.4g} Pa vs relief + Q/G {want:.4g} Pa"
    return LawResult.within(max(abs(end), over), 0.02, detail,
                            "vent relief_conductance x (P - relief_pressure) above relief")


__all__ = ["Hydraulic", "HydraulicCylinder"]
