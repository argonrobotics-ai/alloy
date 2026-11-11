# SPDX-License-Identifier: Apache-2.0
"""Wheeled vehicles: virtual wheels on a chassis body with slip-based tyre forces, rolling resistance, a motor, a
gearbox and an open differential (forklifts, carts, mobile bases).

The plug-in owns the wheels: their spin is its state, and their contact with a flat ground is computed from the pose of
the body each wheel sits on (``kernels.py``): the chassis, or another body named in ``bodies`` (a forklift's steer axle
on a roll hinge with stops, which makes the three-point stability triangle). ``apply`` adds every tyre's load, drive
and cornering force to its wheel's body at the contact point;
``update`` steps the wheel spins implicitly against the motor, the brakes and the rolling resistance. Inputs per world:
``throttle`` in [-1, 1] (negative drives backwards), ``brake`` in [0, 1] and ``steer`` [rad] (each wheel turns by
``steer`` times its own steering factor, or with ``ackermann`` square to the turning centre ``steer`` sets on the
unsteered axle's line). Keep the bodies' shapes above the ground: the solver never sees the wheels.

Defaults (sources and choices in docs/notes/vehicle-v1.md, vehicle-v2.md): an 800 kg cart-like test vehicle on 0.3 m
wheels, peak friction 0.9, magic-formula shape factors B, C = 10, 1.65 (longitudinal) and 8, 1.3 (lateral), rolling
resistance 0.015, tread relaxation lengths 0.3 m (Pacejka's transient tyre: it holds at standstill).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import Law, TestScene
from alloy.plugins.vehicle import laws
from alloy.plugins.vehicle.kernels import tyre_force_kernel, wheel_kernel
from alloy.plugins.vehicle.mobility import in_plane_max, wheel_mobility
from alloy.plugins.vehicle.surfaces import surface_checks, surface_index, surface_mobility, surfaces_struct
from alloy.plugins.vehicle.wheels import drive_of, tyre_of, wheel_array, wheel_specs

log = logging.getLogger("alloy.plugins.vehicle")

STABLE = 1.0
CORNERS = ((0.6, 0.45, 0.0), (0.6, -0.45, 0.0), (-0.6, 0.45, 0.0), (-0.6, -0.45, 0.0))


class Vehicle(ForcePlugin):
    """Virtual wheels on a chassis body: magic-formula tyres, rolling resistance, motor + gear + open differential.

    Example (a rear-drive, front-steer cart)::

        car = Vehicle(chassis="cart", positions=((0.6, 0.45, 0), (0.6, -0.45, 0), (-0.6, 0.45, 0), (-0.6, -0.45, 0)),
                      steer=(1, 1, 0, 0), driven=(False, False, True, True))
        sim = alloy.plugins.simulation(built, [car])
        car.set_input("throttle", 0.5); car.set_input("steer", 0.1)
    """

    name = "vehicle"
    summary = "wheeled vehicle: virtual wheels with magic-formula tyres, rolling resistance, motor, gear, differential"

    @dataclass
    class Config:
        chassis: str = "chassis"                                # body label of the chassis
        positions: tuple[Any, ...] = CORNERS                    # wheel centres, each in its body's frame [m]
        radius: tuple[float, ...] = (0.3,)                      # m (one, or per wheel)
        steer: tuple[float, ...] = (1.0, 1.0, 0.0, 0.0)         # steering factor per wheel
        driven: tuple[bool, ...] = (False, False, True, True)   # driven by the differential
        inertia: tuple[float, ...] = (1.0,)                     # wheel spin inertia [kg·m²]
        radial_stiffness: tuple[float, ...] = (2.0e5,)          # tyre radial stiffness [N/m]
        radial_damping: tuple[float, ...] = (4.0e3,)            # N·s/m
        brake_torque: tuple[float, ...] = (1500.0,)             # N·m at full brake
        mu: float = 0.9                                         # peak friction coefficient
        slip_b: float = 10.0                                    # magic formula B, C for longitudinal slip
        slip_c: float = 1.65                                    # (a locked wheel keeps sin(1.65·π/2) = 65 % of the peak)
        angle_b: float = 8.0                                    # B [1/rad], C for the slip angle
        angle_c: float = 1.3
        rolling_resistance: float = 0.015                       # c_rr
        relaxation_x: float = 0.3                               # m: tread relaxation length along the wheel
        relaxation_y: float = 0.3                               # m: and across (Pacejka's transient tyre)
        low_speed: float = 1.0                                  # m/s: the low-speed tread damper is gone above this
        low_speed_damping: float = 0.5                          # its fraction of critical
        ground: float = 0.0                                     # m: ground plane height
        ground_plane: bool = True                               # the flat ground plane is a drivable surface
        surfaces: tuple[str, ...] = ()                          # box shape labels whose top faces are drivable
        resting: tuple[tuple[str, str], ...] = ()               # (face, support): of the surfaces, faces whose
                                                                # bodies rest on another under load (a leveler, a
                                                                # ramp's lip plates), each with the drivable surface
                                                                # (or "world") it rests on. Left out of the step
                                                                # check: the support, checked with every wheel on
                                                                # it, carries the wheel's load (MuJoCo's contact)
        motor_torque: float = 150.0                             # N·m peak
        motor_free_speed: float = 600.0                         # rad/s where the torque fades to 0
        motor_inertia: float = 0.02                             # kg·m²
        gear: float = 10.0                                      # motor turns per wheel turn
        bodies: tuple[str, ...] = ("",)                         # body label per wheel ("": the chassis)
        ackermann: tuple[float, ...] = (0.0,)                   # per wheel: its distance to the unsteered axle [m];
                                                                # > 0 steers it by Ackermann about that axle's line
                                                                # (``steer`` the centreline's angle, factor ±1). The
                                                                # wheel's y in its body's frame is taken as its offset
                                                                # from the centreline: that body's origin (an axle's
                                                                # pin) must lie on the centreline
        arm_slope: tuple[float, ...] = (0.0,)                   # trailing arm: -slope along the body's x per metre of
                                                                # radial compression (a trailer's dock walk)

        def __post_init__(self) -> None:
            positive = {k: getattr(self, k) for k in ("mu", "slip_b", "slip_c", "angle_b", "angle_c", "relaxation_x",
                                                      "relaxation_y", "low_speed", "motor_free_speed", "gear")}
            nonneg = {k: getattr(self, k) for k in ("rolling_resistance", "motor_torque", "motor_inertia",
                                                    "low_speed_damping")}
            bad = [k for k, v in positive.items() if not (math.isfinite(v) and v > 0.0)]
            bad += [k for k, v in nonneg.items() if not (math.isfinite(v) and v >= 0.0)]
            bad += [k for k in ("slip_c", "angle_c") if not getattr(self, k) < 2.0]   # past the peak F keeps its sign
            _check_resting(self.resting, self.surfaces)
            if bad or not math.isfinite(self.ground):
                raise ValueError(f"{bad or ['ground']} must be finite and > 0 (or >= 0 for torque, inertia, c_rr; "
                                 f"the shape factors slip_c, angle_c < 2)")
            wheel_specs(self)

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        self.specs = wheel_specs(c)
        W, N = ctx.world_count, len(self.specs)
        self.chassis = ctx.bodies(c.chassis)
        self.body_index = _wheel_bodies(ctx, c.chassis, self.specs)
        self.wheel_body = ctx.array(self.body_index, wp.int32)
        self.wheels = wheel_array(self.specs, c, ctx.device)
        self.surface_index = surface_index(ctx, tuple(c.surfaces))
        self.surf = surfaces_struct(ctx, self.surface_index, c.ground, c.ground_plane)
        self.tyre, self.drive = tyre_of(c), drive_of(c, self.specs)
        self.mob = wheel_mobility(model, self.body_index, self.chassis.numpy(), self.specs)
        self.mobility = ctx.array(self.mob, wp.float32)
        self.throttle = self.add_input("throttle", ctx.zeros(W))
        self.brake = self.add_input("brake", ctx.zeros(W))
        self.steer = self.add_input("steer", ctx.zeros(W))
        self.omega = self.add_state("omega", ctx.zeros((W, N)))
        self.defl = self.add_state("deflection", ctx.zeros((W, N, 2)))   # tread u (along), q (across) [m]
        self.rate = self.add_state("deflection_rate", ctx.zeros((W, N, 2)))
        self.force = self.add_state("force", ctx.zeros((W, N, 3)))
        self.slip = self.add_state("slip", ctx.zeros((W, N, 2)))
        self.motor = self.add_state("motor", ctx.zeros((W, 2)))
        log.info("vehicle %s: %d wheels on %r (%d driven), gear %.3g, motor %.3g N·m", self.label, N,
                 sorted({s.body or c.chassis for s in self.specs}), self.drive.driven, c.gear, c.motor_torque)

    def check_dt(self, dt: float) -> None:
        """The tyres' radial springs and dampers, and the tread springs ``D·B·C/σ`` with their low-speed dampers, on
        each body that carries wheels: summed over its wheels, each on that body's point mobility at its contact
        (worst body, worst world). A light axle body carries its own wheels, so it can set the step. The tread load
        is bounded by 1.5x the weight of one world's wheel-carrying bodies on one wheel (a payload margin)."""
        k = np.asarray([s.radial_stiffness for s in self.specs])
        c = np.asarray([s.radial_damping for s in self.specs])
        wz = self.mob[:, :, 3]
        springs, dampers = _per_body(self.body_index, wz * k), _per_body(self.body_index, wz * c)
        checks = {"tyre springs": float(np.sqrt(springs)) * dt, "tyre damping": dampers * dt}
        checks.update(self._tread_checks(dt))
        checks.update(self._surface_checks(dt))
        bad = {name: round(v, 3) for name, v in checks.items() if v >= STABLE}
        if bad:
            raise PluginError(f"{self.label}: unstable at dt={dt:.3g} s: {bad} (each must be < {STABLE}); lower "
                              f"radial_stiffness / radial_damping or the exchange step, or give the wheels' bodies "
                              f"more mass")

    def _tread_load(self) -> float:
        """1.5 g x the wheel-carrying bodies of one world (the heaviest). ``body_index`` is [worlds, wheels]: summing
        its unique bodies over every world grew the load with the world count (100k red, rc-1109: task_forklift_pick
        refused at 512 and 1,024 worlds, the tread checks up to 10.5 against 1)."""
        mass = self.ctx.model.body_mass.numpy()
        heaviest = max(float(mass[np.unique(row)].sum()) for row in self.body_index)
        return 1.5 * float(np.linalg.norm(self.ctx.gravity)) * heaviest

    def _surface_checks(self, dt: float) -> dict[str, float]:
        c = self.config
        resting = {face for face, _ in c.resting}
        keep = [k for k, name in enumerate(c.surfaces) if name not in resting]
        mobility = surface_mobility(self.ctx.model, self.surface_index[:, keep])
        shape = max(c.slip_b * c.slip_c / c.relaxation_x, c.angle_b * c.angle_c / c.relaxation_y)
        k_t = c.mu * self._tread_load() * shape
        w_ip = in_plane_max(self.mob)
        c_t = 2.0 * c.low_speed_damping * np.sqrt(k_t / np.maximum(w_ip, 1.0e-12))
        k_r = np.array([s.radial_stiffness for s in self.specs])
        c_r = np.array([s.radial_damping for s in self.specs])
        return surface_checks(mobility, self.body_index, (k_r, c_r), (k_t, c_t), (self.mob[:, :, 3], w_ip), dt)

    def _tread_checks(self, dt: float) -> dict[str, float]:
        c = self.config
        load = self._tread_load()
        out = {}
        w = in_plane_max(self.mob)                              # any steer angle: the worst in-plane direction
        for axis, (b, cc, sigma) in enumerate(((c.slip_b, c.slip_c, c.relaxation_x),
                                               (c.angle_b, c.angle_c, c.relaxation_y))):
            k = c.mu * load * b * cc / sigma
            name = "along" if axis == 0 else "across"
            out[f"tread springs {name}"] = float(np.sqrt(_per_body(self.body_index, w * k))) * dt
            out[f"tread damping {name}"] = _per_body(self.body_index, 2.0 * c.low_speed_damping * np.sqrt(k * w)) * dt
        return out

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        wp.launch(tyre_force_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, self.ctx.model.body_com, self.wheel_body, self.wheels,
                          self.tyre, self.surf, self.steer, self.defl, self.rate, self.mobility],
                  outputs=[forces.body_f], device=self.ctx.device)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        # the wheels and treads step against the bodies' end-of-step velocity: an implicit spin step against a frozen
        # chassis speed adds dt·R·dF_x/dω of false inertia while the chassis accelerates (the wheels lag and drag)
        wp.launch(wheel_kernel, dim=self.ctx.world_count,
                  inputs=[state_out.body_q, state_out.body_qd, self.ctx.model.body_com, self.wheel_body, self.wheels,
                          self.tyre, self.surf, self.drive, self.throttle, self.brake, self.steer, self.mobility,
                          float(dt)],
                  outputs=[self.omega, self.defl, self.rate, self.force, self.slip, self.motor],
                  device=self.ctx.device)

    # --- host side (between periods) ---------------------------------------------------------------------------

    def set_spin(self, omega: Any) -> None:
        """Wheel spins [rad/s], ``[N]`` or ``[W, N]`` (e.g. a rolling start: v / R)."""
        src = np.broadcast_to(np.asarray(omega, np.float32), self.omega.shape)
        if not np.isfinite(src).all():
            raise PluginError(f"{self.label}: wheel spins must be finite")
        self.omega.assign(np.ascontiguousarray(src))

    def effective_mass(self, mass: float) -> float:
        """Chassis mass plus every wheel's spin inertia (with the motor's share) over R²: the mass a drive force sees."""
        from alloy.plugins.vehicle.wheels import effective_inertia  # noqa: PLC0415

        n = self.drive.driven
        return mass + sum(effective_inertia(s, self.config, n) / s.radius**2 for s in self.specs)

    def test_scene(self) -> TestScene:
        return laws.drive_scene(self.config)

    def laws(self) -> list[Law]:
        return laws.vehicle_laws(self.config)


def _check_resting(resting: tuple, surfaces: tuple) -> None:
    """Each resting face is a drivable surface and rests on one that is checked (not itself resting), or on the world:
    the support's bound, with every wheel summed on it, then covers the face's load."""
    faces = {pair[0] for pair in resting if len(pair) == 2}
    bad = [pair for pair in resting if len(pair) != 2 or pair[0] not in surfaces
           or not (pair[1] == "world" or (pair[1] in surfaces and pair[1] not in faces))]
    if bad:
        raise ValueError(f"resting {bad}: each entry is (face, support), the face one of the drivable surfaces "
                         f"{list(surfaces)} and the support another of them that is not resting, or 'world'")


def _wheel_bodies(ctx: Any, chassis: str, specs: list[Any]) -> np.ndarray:
    """``int32 [W, N]``: the global index of the body each wheel sits on, in every world."""
    labels = sorted({s.body or chassis for s in specs})
    missing = [name for name in labels if not ctx.has(name)]
    if missing:
        raise PluginError(f"vehicle: no body labelled {missing} for the wheels; bodies: {ctx.labels('body')[:20]}")
    per = {name: ctx.bodies(name).numpy() for name in labels}
    return np.stack([per[s.body or chassis] for s in specs], axis=1).astype(np.int32)


def _per_body(index: np.ndarray, values: np.ndarray) -> float:
    """The largest per-body sum of ``values [W, N]`` over the wheels on one body (worst body, worst world)."""
    worst = 0.0
    for w in range(index.shape[0]):
        for b in np.unique(index[w]):
            worst = max(worst, float(values[w][index[w] == b].sum()))
    return worst


__all__ = ["Vehicle"]
