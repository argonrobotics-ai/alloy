# SPDX-License-Identifier: Apache-2.0
"""Pins that pass through drilled holes (drilling v2, phase B): the ``Pin`` config of the drill plug-in, the set of
pins it drives (``PinSet``), and ``PushHead``, a companion that pushes a pin along an axis with a set force.

A pin is a bolt, a dowel or an anchor: a body with a shank from ``top`` (just under its head) to ``tip`` in its body
frame, radius ``radius``, and a head of radius ``head`` (0: none). A pin with ``anchor`` > 0 is an expansion anchor
set in the stack's last layer (``anchors.py``; nut torque per world: ``set_input("nut_torque", ...)``). Give the pin
(shank and head) no solver collision with the drilled parts (``ShapeConfig(has_shape_collision=False)``, or a
collision filter): the drill owns that contact and lets the pin through its holes only (``pin_kernels.py``). Every
other body still meets the undrilled shapes (``docs/known_gaps/drilling-holes-pins-only.md``).

Example::

    drill = Drill(layers=(Layer("angle"), Layer("bar")), pins=(Pin("bolt", tip=(0, 0, -0.04), radius=0.004,
                                                                     head=0.0065),))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.errors import PluginError
from alloy.plugins.funcs import add_wrench
from alloy.plugins.holes.anchors import anchor_force_kernel, anchor_update_kernel
from alloy.plugins.holes.pin_kernels import Holes, PinSpec, pin_force_kernel, pin_member_kernel
from alloy.plugins.removal.stability import body_of, smallest_inertia

STABLE = 1.0
SPRINGS = 4                     # contact springs on one pin per layer at once (2 faces, tip, head)


CHAMFER_DAMPING = 2.0e-4       # s: the least damping of a chamfered pin (the known gap: below it a seated end pumps)

@dataclass
class Pin:
    """One pin the holes guide (see the module docstring). ``stiffness`` is per contact spring."""

    body: str                                            # body label of the pin
    top: tuple[float, float, float] = (0.0, 0.0, 0.0)    # shank top (under the head), body frame [m]
    tip: tuple[float, float, float] = (0.0, 0.0, -0.04)  # shank tip, body frame [m]
    radius: float = 0.004                                # shank radius [m]
    head: float = 0.0                                    # head radius [m] (0: no head)
    stiffness: float = 2.0e5                             # N/m per contact spring
    damping: float = 2.0e-4                              # s (light: the pusher damps the approach)
    friction: float = 0.0                                # Coulomb coefficient against the hole walls and faces
    chamfer: float = 0.0                                 # m: the end's 45 deg chamfer width (guides rim offsets in)
    slip: float = 1.0e-3                                 # m/s: friction is viscous below this
    anchor: float = 0.0                                  # m: an expansion anchor's least embed (0: not an anchor)
    nut_factor: float = 0.2                              # K in T = K F d (NASA RP-1228 p. 21: "commonly assumed")
    anchor_hole: float = 0.0                             # m: the widest hole its expansion grips (anchors: required)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.radius) and self.radius > 0.0):
            raise ValueError(f"pin {self.body!r}: radius must be finite and > 0")
        if np.linalg.norm(np.subtract(self.tip, self.top)) <= 0.0:
            raise ValueError(f"pin {self.body!r}: top and tip must differ")
        bad = [k for k in ("head", "stiffness", "damping", "friction", "slip", "chamfer", "anchor", "nut_factor",
                           "anchor_hole")
               if not (math.isfinite(getattr(self, k)) and getattr(self, k) >= 0.0)]
        if bad or self.stiffness <= 0.0 or self.slip <= 0.0 or self.nut_factor <= 0.0:
            raise ValueError(f"pin {self.body!r}: {bad or ['stiffness', 'slip', 'nut_factor']} must be finite and > 0 "
                             f"(head, damping, friction, chamfer, anchor >= 0)")
        if self.chamfer > 0.0 and self.damping < CHAMFER_DAMPING:
            raise ValueError(f"pin {self.body!r}: a chamfered pin needs damping >= {CHAMFER_DAMPING:g} s: less lets "
                             f"its end gain energy resting on a mouth (docs/known_gaps/drilling-holes-pins-only.md)")
        if self.anchor > 0.0 and not self.anchor_hole >= 2.0 * self.radius:
            raise ValueError(f"pin {self.body!r}: an anchor needs anchor_hole, the widest hole its expansion grips "
                             f"(the maker's bit tolerance), at least its diameter")


class PinSet:
    """The drill's pins on one model: per-world bodies, device specs, the stability check and the force launch."""

    def __init__(self, ctx: Any, drill: Any, pins: tuple[Pin, ...], patches: list[list[Any]], works: list[Any],
                 bit: Any) -> None:
        model = ctx.model
        self.pins = pins
        self.body = ctx.array(np.stack([ctx.bodies(p.body).numpy() for p in pins], axis=1).astype(np.int32),
                              wp.int32)                                             # [W, P]
        layer_mob = max((_layer_mobility(model, row) for row in patches), default=0.0)
        self.mobility = [_pin_mobility(model, self.body.numpy()[:, j], p) + layer_mob for j, p in enumerate(pins)]
        self.specs = wp.array([_spec(p, m) for p, m in zip(pins, self.mobility, strict=True)], dtype=PinSpec,
                              device=ctx.device)
        self.thickness = ctx.array([float(w.thickness) for w in works], wp.float32)
        self.half = ctx.array([[float(w.half[0]), float(w.half[1])] for w in works], wp.vec2)
        self.holes = Holes()
        self.holes.holes, self.holes.stages, self.holes.tan_half = int(bit.holes), int(bit.stages), float(bit.tan_half)
        self.layers = len(works)
        W = ctx.world_count
        self.member = drill.add_state("pin_hole", ctx.zeros((W, len(pins) * self.layers), wp.int32))   # hole + 1
        self.land = drill.add_state("pin_land", ctx.zeros((W, len(pins) * self.layers), wp.int32))   # LAND_*
        self.land_u = drill.add_state("pin_land_u", ctx.zeros((W, len(pins) * self.layers), wp.float32))
        self.seat = drill.add_state("pin_seat", ctx.zeros((W, len(pins) * self.layers), wp.int32))   # SEAT_*
        self.seat_u = drill.add_state("pin_seat_u", ctx.zeros((W, len(pins) * self.layers), wp.float32))
        self.solid = drill.add_state("pin_solid", ctx.zeros(W, wp.int32))
        self.anchors = any(p.anchor > 0.0 for p in pins)
        if self.anchors:
            P = len(pins)
            self.nut_torque = drill.add_input("nut_torque", ctx.zeros((W, P), wp.float32))     # N m per anchor
            self.embed = drill.add_state("pin_embed", ctx.zeros((W, P), wp.float32))
            self.anchored = drill.add_state("pin_anchored", ctx.zeros((W, P), wp.int32))
            self.anchor_u = drill.add_state("pin_anchor_u", ctx.zeros((W, P), wp.float32))

    def check_dt(self, label: str, dt: float) -> None:
        bad = {}
        for p, mob in zip(self.pins, self.mobility, strict=True):
            k = (SPRINGS * self.layers + (1 if p.anchor > 0.0 else 0)) * p.stiffness     # + an anchor's clip
            for name, v in ((f"pin {p.body} spring", math.sqrt(k * mob) * dt),
                            (f"pin {p.body} damping", k * p.damping * mob * dt)):
                if v >= STABLE:
                    bad[name] = round(v, 3)
        if bad:
            raise PluginError(f"{label}: unstable at dt={dt:.3g} s: {bad} (each < {STABLE}); lower the pins' stiffness "
                              f"or damping, shrink the step, or give the pins more mass (a holder, a gripper)")

    def apply(self, state: newton.State, forces: Any, dt: float, drill: Any) -> None:
        m = drill.ctx.model
        wp.launch(pin_force_kernel, dim=drill.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, m.body_com, self.body, self.specs, len(self.pins),
                          drill.face_body, drill.face_local, self.holes, self.thickness, self.half, self.layers,
                          float(dt), drill.hole_n, drill.hole_entry, drill.hole_axis, drill.stage_n, drill.stage_d,
                          drill.stage_b, self.member, self.land, self.land_u, self.seat, self.seat_u],
                  outputs=[forces.body_f], device=drill.ctx.device)
        if self.anchors:
            wp.launch(anchor_force_kernel, dim=(drill.ctx.world_count, len(self.pins)),
                      inputs=[state.body_q, state.body_qd, m.body_com, self.body, self.specs, drill.face_body,
                              drill.face_local, self.holes, self.thickness, self.half, self.layers, float(dt),
                              drill.hole_entry, drill.hole_axis, self.member, self.anchored, self.anchor_u],
                      outputs=[forces.body_f], device=drill.ctx.device)

    def update(self, state_in: newton.State, state_out: newton.State, drill: Any) -> None:
        """After the step: membership, landings and seats from the step's start and end poses."""
        m = drill.ctx.model
        wp.launch(pin_member_kernel, dim=drill.ctx.world_count,
                  inputs=[state_in.body_q, state_out.body_q, m.body_com, self.body, self.specs, len(self.pins),
                          drill.face_body, drill.face_local, self.holes, self.thickness, self.half, self.layers,
                          drill.hole_n, drill.hole_entry, drill.hole_axis, drill.stage_n, drill.stage_d,
                          drill.stage_b],
                  outputs=[self.member, self.land, self.land_u, self.seat, self.seat_u, self.solid],
                  device=drill.ctx.device)
        if self.anchors:
            wp.launch(anchor_update_kernel, dim=(drill.ctx.world_count, len(self.pins)),
                      inputs=[state_out.body_q, m.body_com, self.body, self.specs, drill.face_body, drill.face_local,
                              self.holes, self.thickness, self.half, self.layers, drill.hole_entry, drill.hole_axis,
                              drill.stage_n, drill.stage_d, self.member, self.nut_torque],
                      outputs=[self.embed, self.anchored, self.anchor_u], device=drill.ctx.device)

    def clamp(self, read: dict) -> np.ndarray:
        """[W, P] each engaged anchor's clamp load F = T / (K d) [N] (0: not engaged, or not an anchor)."""
        k = np.asarray([p.nut_factor * 2.0 * p.radius for p in self.pins])
        return np.where(read["pin_anchored"] != 0, read["nut_torque"] / k, 0.0)


def _spec(p: Pin, mobility: float) -> PinSpec:
    s = PinSpec()
    s.top, s.tip = wp.vec3(*p.top), wp.vec3(*p.tip)
    s.radius, s.head, s.stiffness, s.damping = float(p.radius), float(p.head), float(p.stiffness), float(p.damping)
    s.friction, s.slip, s.mobility, s.chamfer = float(p.friction), float(p.slip), float(mobility), float(p.chamfer)
    s.anchor, s.nut, s.anchor_hole = float(p.anchor), float(p.nut_factor), float(p.anchor_hole)
    return s


def _pin_mobility(model: newton.Model, bodies: np.ndarray, p: Pin) -> float:
    """Worst over worlds: 1/m + r²/I_min at the farther shank end."""
    worst = 0.0
    for b in bodies:
        body = body_of(model, int(b), f"pin {p.body!r}")
        reach = max(float(np.linalg.norm(np.subtract(e, body.com))) for e in (p.top, p.tip))
        worst = max(worst, 1.0 / body.mass + reach**2 / smallest_inertia(body.inertia))
    return worst


def _layer_mobility(model: newton.Model, row: list[Any]) -> float:
    """The worst floating layer of one world (static layers add nothing)."""
    from alloy.plugins.drilling.stability import face_mobility  # noqa: PLC0415 (drilling imports this module)

    return max((face_mobility(body_of(model, p.body, "workpiece"), p) for p in row if p.body >= 0), default=0.0)


# --- a companion that pushes a pin --------------------------------------------------------------------------------

@wp.kernel
def _push_kernel(body_q: wp.array[wp.transform], body_qd: wp.array[wp.spatial_vector], body_com: wp.array[wp.vec3],
                 body: wp.array[wp.int32], force: wp.array[wp.float32], direction: wp.array[wp.vec3],
                 axis: wp.vec3, k_tilt: wp.float32, d_tilt: wp.float32, d_side: wp.float32, d_push: wp.float32,
                 body_f: wp.array[wp.spatial_vector]):
    w = wp.tid()
    b = body[w]
    d = direction[w]
    qd = body_qd[b]
    v = wp.vec3(qd[0], qd[1], qd[2])
    omega = wp.vec3(qd[3], qd[4], qd[5])
    a = wp.transform_vector(body_q[b], axis)
    along = wp.dot(v, d)
    f = (force[w] - d_push * along) * d - d_side * (v - along * d)
    torque = k_tilt * wp.cross(a, d) - d_tilt * (omega - wp.dot(omega, a) * a)
    add_wrench(body_f, b, wp.spatial_vector(f[0], f[1], f[2], torque[0], torque[1], torque[2]))


class PushHead(ForcePlugin):
    """Pushes a body with a set force along a direction and keeps its ``axis`` (body frame) along it: a hand pushing
    a bolt into its holes, at most force / d_push fast (20 N at 400 N s/m: 50 mm/s). Free across the direction (the
    holes centre the pin), damped there by ``d_side``. Force and direction are per-world inputs
    (``set_input("force", ...)``, ``set_direction(...)``)."""

    name = "push_head"
    summary = "pushes a pin along a direction with a set force, holding its axis (per-world inputs)"

    @dataclass
    class Config:
        body: str = "pin"
        force: float = 20.0                                   # N along the direction
        direction: tuple[float, float, float] = (0.0, 0.0, -1.0)
        axis: tuple[float, float, float] = (0.0, 0.0, -1.0)    # body frame axis held along the direction
        k_tilt: float = 20.0                                  # N m/rad
        d_tilt: float = 0.2                                   # N m s/rad
        d_side: float = 5.0                                   # N s/m
        d_push: float = 400.0                                 # N s/m along the direction (speed = force / d_push)

        def __post_init__(self) -> None:
            if not all(math.isfinite(v) and v >= 0.0 for v in (self.k_tilt, self.d_tilt, self.d_side, self.d_push)):
                raise ValueError("k_tilt, d_tilt, d_side and d_push must be finite and >= 0")
            if abs(float(np.linalg.norm(self.direction)) - 1.0) > 1e-6 or np.linalg.norm(self.axis) <= 0.0:
                raise ValueError("direction must be a unit vector and axis non-zero")

    def setup(self, ctx: Any) -> None:
        c = self.config
        self.body = ctx.bodies(c.body)
        b = int(self.body.numpy()[0])
        self.mass = float(ctx.model.body_mass.numpy()[b])
        self.inertia = smallest_inertia(np.asarray(ctx.model.body_inertia.numpy()[b], np.float64).reshape(3, 3))
        W = ctx.world_count
        self.force = self.add_input("force", ctx.full(W, float(c.force), wp.float32))
        self.direction = self.add_input("direction", ctx.array(np.tile(np.asarray(c.direction, np.float32), (W, 1)),
                                                               wp.vec3))
        self.axis = wp.vec3(*(np.asarray(c.axis, np.float64) / np.linalg.norm(c.axis)))

    def check_dt(self, dt: float) -> None:
        c = self.config
        checks = {"k_tilt": math.sqrt(c.k_tilt / self.inertia) * dt, "d_tilt": c.d_tilt * dt / self.inertia,
                  "d_side": c.d_side * dt / self.mass, "d_push": c.d_push * dt / self.mass}
        bad = {k: round(v, 3) for k, v in checks.items() if v >= STABLE}
        if bad:
            raise PluginError(f"{self.label}: gains unstable at dt={dt:.3g} s: {bad} (each < {STABLE}); lower them")

    def set_direction(self, direction: Any) -> None:
        d = np.broadcast_to(np.asarray(direction, np.float64), (self.ctx.world_count, 3))
        n = np.linalg.norm(d, axis=1, keepdims=True)
        if not (np.isfinite(d).all() and (n > 0.0).all()):
            raise PluginError(f"{self.label}: directions must be finite and non-zero")
        self.set_input("direction", d / n)

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        c = self.config
        wp.launch(_push_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, self.ctx.model.body_com, self.body, self.force,
                          self.direction, self.axis, float(c.k_tilt), float(c.d_tilt), float(c.d_side),
                          float(c.d_push)],
                  outputs=[forces.body_f], device=self.ctx.device)


__all__ = ["Pin", "PinSet", "PushHead"]
