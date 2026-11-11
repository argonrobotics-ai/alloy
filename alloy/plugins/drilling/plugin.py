# SPDX-License-Identifier: Apache-2.0
"""Drilling by a twist drill on a tool body into workpiece faces: Shaw-Oxford thrust and torque, a table of holes.

The plug-in owns the drill-workpiece contact (the solver never collides the bit: give the tool no collision shape
along the bit, and keep its housing above the face by more than the hole depth). The workpiece is one face, or a
stack of ``layers`` (a cover on a backer, two boards, a ledger on a wall), each with its own material and its own
hole table (``alloy.plugins.holes``): holes run along the drill's real axis, a hole holds one stage per drill
diameter (a pilot, then the drill), and ``holes()`` returns them as records other parts can use. Every step it:

- ``apply`` (before the solver): each layer's share of the cutting front's thrust along the hole axis and its
  cutting torque against the spin, plus hole wall springs that keep the bit on the hole axis; the reactions go to
  the layers that are bodies.
- ``update`` (after the solver): advances the stage bottoms by the feed actually cut and moves the removed mass to
  the chip sink (fraction ``capture``) or onto the part (``loose``), exactly: removed = density·V(stages).

Force model (``kernels.py``): Shaw and Oxford (1957), in the form of Machinery's Handbook ("Drilling: power, torque and
thrust"): thrust T = 2·Kd·f^0.8·d^0.8·B·W + Kd·d^2·J·W [lbf] and torque M = Kd·f^0.8·d^1.8·A·W [in·lbf] with f
[in/rev] and d [in], converted exactly to SI. Defaults: a standard 118° HSS drill (c/d = 0.18: A, B, J of the
Handbook's worked example), W = 1.30 (ordinary drilling) and Kd for aluminium alloys. Sources: docs/notes/drilling-v1.md.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin
from alloy.plugins.drilling import laws
from alloy.plugins.drilling.kernels import Bit, Shaw, Work, cut_kernel, drill_force_kernel
from alloy.plugins.drilling.stability import DrillResponse, drill_response
from alloy.plugins.errors import PluginError
from alloy.plugins.harness.scene import Law, TestScene
from alloy.plugins.holes.bars import Bar, bars_struct
from alloy.plugins.holes.pins import Pin, PinSet
from alloy.plugins.holes.table import Hole, holes_of
from alloy.plugins.holes.work import PercussiveMaterial, work_material
from alloy.plugins.npgeom import quat_rotate
from alloy.plugins.removal.surface import Patch, patch_of

log = logging.getLogger("alloy.plugins.drilling")

STABLE = 1.0          # explicit springs and dampers need omega·dt and c·dt/m below this (GUIDE section 5)
LBF = 4.4482216152605         # N per pound-force
IN_LBF = LBF * 0.0254         # N·m per inch-pound-force
INCH = 0.0254                 # m


@dataclass
class Layer:
    """One workpiece layer of a stack, top first: a box shape's top face, or a face of ``size`` at ``surface_offset``.

    ``work`` names a row of ``alloy.plugins.holes.work``: its Kd for thrust and torque, or, for a rotary hammer, its
    blow energy per removed volume (``PERCUSSIVE``); None uses the drill's ``work_factor``. ``bars`` are bars in the
    layer that stop the bit (rebar: ``holes.bars``; pressed hard it cuts through one at an estimated rate). Example:
    ``Layer("cover", work="steel_200bhn", density=7850.0)``.
    """

    workpiece: str                                          # shape label
    work: str | None = None
    density: float = 2700.0                                 # kg/m^3
    size: tuple[float, float] | None = None                 # face size in the shape's local x, y; None: box
    surface_offset: float | None = None                     # face height in the shape frame; None: box top
    thickness: float | None = None                          # material under the face [m]; None: box height
    bars: tuple[Bar, ...] = ()                              # bars that stop the bit, in its face frame

    def __post_init__(self) -> None:
        self.bars = tuple(b if isinstance(b, Bar) else Bar(**b) for b in self.bars)
        if not (math.isfinite(self.density) and self.density > 0.0):
            raise ValueError(f"layer {self.workpiece!r}: density must be finite and > 0")
        if self.thickness is not None and not (math.isfinite(self.thickness) and self.thickness > 0.0):
            raise ValueError(f"layer {self.workpiece!r}: thickness must be finite and > 0")
        if self.work is not None:
            work_material(self.work)                        # an unknown row raises with the known ones
        if self.size is not None:
            self.size = (float(self.size[0]), float(self.size[1]))


class Drill(ForcePlugin):
    """Drilling: a twist drill on a tool body cuts holes into a workpiece face or a stack of layers (Shaw-Oxford
    thrust and torque from the feed per revolution, holes along the drill's axis, wall springs, an exact chip ledger).

    Example::

        drill = Drill(tool="spindle", tip=(0, 0, -0.07), workpiece="plate", diameter=0.006)
        sim = alloy.plugins.simulation(built, [drill, FeedHead(head="spindle")])
        ...; r = sim.plugins["drill"].read(); r["depth"][0], r["thrust"][0], r["torque"][0]
        sim.plugins["drill"].holes(world=0)        # [Hole(layer, entry, axis, stages, ...)]
    """

    name = "drill"
    summary = "twist drilling: Shaw-Oxford thrust/torque from feed per rev, holes along the bit, stacks, exact ledger"

    @dataclass
    class Config:
        tool: str = "spindle"                                   # body label of the tool (spindle) carrying the drill
        tip: tuple[float, float, float] = (0.0, 0.0, -0.07)     # drill tip, tool frame [m]
        axis: tuple[float, float, float] = (0.0, 0.0, -1.0)     # drill axis, shank to tip, tool frame
        diameter: float = 0.006                                 # m (per world: ``set_diameter``)
        point_angle: float = 118.0                              # deg, full point angle
        chisel_ratio: float = 0.18                              # chisel edge length / d (split point: 0.03)
        chisel_a: float = 1.085                                 # Handbook Table 7 torque factor A (c/d = 0.18)
        chisel_b: float = 1.355                                 # thrust factor B (lips)
        chisel_j: float = 0.030                                 # thrust factor J (chisel edge)
        work_factor: float = 7000.0                             # Kd [lbf/in^1.6]: most aluminium alloys
        wear_factor: float = 1.30                               # W: ordinary drilling (dull: 1.50)
        workpiece: str = "workpiece"                            # shape label of the workpiece (one layer)
        size: tuple[float, float] | None = None                 # face size in the shape's local x, y; None: box
        surface_offset: float | None = None                     # face height in the shape frame; None: box top
        thickness: float | None = None                          # material under the face [m]; None: box height
        density: float = 2700.0                                 # kg/m^3 (aluminium)
        capture: float = 0.0                                    # fraction of chips collected by the tool
        stiffness: float = 1.0e6                                # N/m: cutting-front contact
        damping: float = 1.0e-3                                 # s: its Kelvin-Voigt time
        radial_stiffness: float = 1.0e5                         # N/m per hole wall spring (tip, face crossing)
        radial_damping: float = 1.0e-3                          # s
        feed_max: float = 1.0e-3                                # m/rev: largest chip load the flutes take
        spin_eps: float = 1.0                                   # rad/s: the torque is viscous below this spin
        layers: tuple[Layer, ...] = field(default_factory=tuple)   # a stack, top first; (): one layer from above
        max_holes: int = 8                                      # hole slots per layer and world
        max_stages: int = 3                                     # stages (drill diameters) per hole
        pins: tuple[Pin, ...] = field(default_factory=tuple)    # bolts and anchors the holes guide (holes.pins)
        blow_energy: float = 0.0                                # J: a rotary hammer's single blow (0: no hammer)
        blow_rate: float = 0.0                                  # blows/s (per world: ``set_hammer``)

        def __post_init__(self) -> None:
            positive = {"diameter": self.diameter, "work_factor": self.work_factor, "wear_factor": self.wear_factor,
                        "density": self.density, "stiffness": self.stiffness, "feed_max": self.feed_max,
                        "spin_eps": self.spin_eps}
            bad = [k for k, v in positive.items() if not (math.isfinite(v) and v > 0.0)]
            if bad:
                raise ValueError(f"{bad} must be finite and > 0")
            nonneg = {"damping": self.damping, "radial_stiffness": self.radial_stiffness,
                      "radial_damping": self.radial_damping, "chisel_a": self.chisel_a, "chisel_b": self.chisel_b,
                      "chisel_j": self.chisel_j, "blow_energy": self.blow_energy, "blow_rate": self.blow_rate}
            bad = [k for k, v in nonneg.items() if not (math.isfinite(v) and v >= 0.0)]
            if bad:
                raise ValueError(f"{bad} must be finite and >= 0")
            if not 0.0 < self.point_angle < 180.0:
                raise ValueError(f"point_angle must be in (0, 180) deg, got {self.point_angle}")
            if not 0.0 <= self.chisel_ratio < 1.0:
                raise ValueError(f"chisel_ratio must be in [0, 1), got {self.chisel_ratio}")
            if not 0.0 <= self.capture <= 1.0:
                raise ValueError(f"capture must be in [0, 1], got {self.capture}")
            if self.thickness is not None and not (math.isfinite(self.thickness) and self.thickness > 0.0):
                raise ValueError(f"thickness must be finite and > 0, got {self.thickness}")
            if np.linalg.norm(self.axis) <= 0.0:
                raise ValueError("axis must be a non-zero vector")
            if int(self.max_holes) < 1 or int(self.max_stages) < 1:
                raise ValueError("max_holes and max_stages must be >= 1")
            self.layers = tuple(lay if isinstance(lay, Layer) else Layer(**lay) for lay in self.layers)
            self.pins = tuple(p if isinstance(p, Pin) else Pin(**p) for p in self.pins)

    def stack(self) -> tuple[Layer, ...]:
        """The layers, top first (one layer built from the single-workpiece fields when ``layers`` is empty)."""
        c = self.config
        return c.layers or (Layer(c.workpiece, None, c.density, c.size, c.surface_offset, c.thickness),)

    def setup(self, ctx: Any) -> None:
        c, model = self.config, ctx.model
        self.tool = ctx.bodies(c.tool)
        stack = self.stack()
        W, L, H, S = ctx.world_count, len(stack), int(c.max_holes), int(c.max_stages)
        per_layer = []
        works = []
        for lay in stack:
            shapes = ctx.shapes(lay.workpiece).numpy()
            patches = [patch_of(model, int(s), lay.workpiece, lay.size, lay.surface_offset) for s in shapes]
            per_layer.append(patches)
            works.append(_work(lay, c.capture, model, int(shapes[0]), patches[0].size))
        self.patches = [list(row) for row in zip(*per_layer, strict=True)]          # [W][L]
        self.face_body = ctx.array([[p.body for p in row] for row in self.patches], wp.int32)
        self.face_local = ctx.array(np.asarray([[p.local for p in row] for row in self.patches], np.float32),
                                    wp.transform)
        self.response = _stack_response(model, self.tool.numpy(), per_layer, c)
        self.bit = _bit(c, self.response)
        self.layer_shaws = [shaw_coefficients(c, lay) for lay in stack]
        self.layer_works = works
        self.shaws = wp.array(self.layer_shaws, dtype=Shaw, device=ctx.device)
        self.works = wp.array(works, dtype=Work, device=ctx.device)
        self.work = works[0]
        self.diameter = self.add_input("diameter", ctx.full(W, float(c.diameter), wp.float32))
        self.blow_energy = self.add_input("blow_energy", ctx.full(W, float(c.blow_energy), wp.float32))
        self.blow_rate = self.add_input("blow_rate", ctx.full(W, float(c.blow_rate), wp.float32))
        self.bars = bars_struct([lay.bars for lay in stack], ctx.device,
                                lambda work: shaw_coefficients(c, Layer("bar", work=work)))
        self.hole_n = self.add_state("hole_n", ctx.zeros((W, L), wp.int32))
        self.hole_entry = self.add_state("hole_entry", ctx.zeros((W, L * H, 2), wp.float32))
        self.hole_axis = self.add_state("hole_axis", ctx.zeros((W, L * H, 3), wp.float32))
        self.stage_n = self.add_state("stage_n", ctx.zeros((W, L * H), wp.int32))
        self.stage_d = self.add_state("stage_d", ctx.zeros((W, L * H * S), wp.float32))
        self.stage_b = self.add_state("stage_b", ctx.zeros((W, L * H * S), wp.float64))
        self.cur = self.add_state("cur", ctx.zeros((W, L), wp.int32))           # 0: in no hole, else hole + 1
        self.depth = self.add_state("depth", ctx.zeros(W, wp.float64))           # readouts of layer 0's last hole
        self.entry = self.add_state("entry", ctx.zeros((W, 2), wp.float32))
        self.engaged = self.add_state("engaged", ctx.zeros(W, wp.int32))     # 0 no hole, 1 tip in a hole, 2 retracted
        self.holes_count = self.add_state("holes", ctx.zeros(W, wp.int32))
        self.full = self.add_state("full", ctx.zeros(W, wp.int32))
        self.removed = self.add_state("removed", ctx.zeros(W, wp.float64))
        self.sink = self.add_state("sink", ctx.zeros(W, wp.float64))
        self.loose = self.add_state("loose", ctx.zeros(W, wp.float64))
        self.layer_removed = self.add_state("layer_removed", ctx.zeros((W, L), wp.float64))
        self.thrust = self.add_state("thrust", ctx.zeros(W, wp.float32))
        self.torque = self.add_state("torque", ctx.zeros(W, wp.float32))
        self.feed = self.add_state("feed", ctx.zeros(W, wp.float32))
        self.jammed = self.add_state("jammed", ctx.zeros(W, wp.int32))          # 1: the bit met a bar it cannot cut
        self.bar_cut = self.add_state("bar_cut", ctx.zeros(W, wp.float64))      # deepest overlap into a bar [m]
        self.pin_set = PinSet(ctx, self, c.pins, self.patches, works, self.bit) if c.pins else None
        log.info("drill %s: d=%.3g mm on %r into %s, front k=%.3g N/m", self.label, c.diameter * 1e3, c.tool,
                 ", ".join(f"{lay.workpiece!r} (t={float(wk.thickness) * 1e3:.3g} mm, {lay.work or c.work_factor})"
                           for lay, wk in zip(stack, works, strict=True)), c.stiffness)

    def check_dt(self, dt: float) -> None:
        """The front and the wall springs and dampers (two per layer) on the worst world's point mobility at the tip
        (tool + floating layers, translation and rotation: ``stability.py``). The torque's viscous end is bounded
        in the kernel."""
        c, r = self.config, self.response
        wall = 2.0 * c.radial_stiffness * len(self.stack())
        checks = {
            "front spring": math.sqrt(c.stiffness * r.front_mobility) * dt,
            "front damping": c.stiffness * c.damping * r.front_mobility * dt,
            "wall springs": math.sqrt(wall * r.wall_mobility) * dt,
            "wall damping": wall * c.radial_damping * r.wall_mobility * dt,
        }
        bad = {k: round(v, 3) for k, v in checks.items() if v >= STABLE}
        if bad:
            raise PluginError(f"{self.label}: unstable at dt={dt:.3g} s: {bad} (each must be < {STABLE}); lower "
                              f"stiffness / radial_stiffness or their damping, shrink the exchange step, or give the "
                              f"tool (and a floating workpiece) more mass and inertia")
        if self.pin_set is not None:
            self.pin_set.check_dt(self.label, dt)

    def _tables(self) -> list:
        return [self.hole_n, self.hole_entry, self.hole_axis, self.stage_n, self.stage_d, self.stage_b, self.cur]

    def apply(self, state: newton.State, forces: Any, dt: float) -> None:
        m = self.ctx.model
        wp.launch(drill_force_kernel, dim=self.ctx.world_count,
                  inputs=[state.body_q, state.body_qd, m.body_com, self.tool, self.face_body, self.face_local,
                          self.bit, self.shaws, self.works, len(self.layer_works), self.diameter, float(dt),
                          *self._tables(), self.blow_energy, self.blow_rate, self.bars],
                  outputs=[forces.body_f], device=self.ctx.device)
        if self.pin_set is not None:
            self.pin_set.apply(state, forces, dt, self)

    def update(self, state_in: newton.State, state_out: newton.State, dt: float) -> None:
        m = self.ctx.model
        wp.launch(cut_kernel, dim=self.ctx.world_count,
                  inputs=[state_in.body_q, state_in.body_qd, m.body_com, self.tool, self.face_body, self.face_local,
                          self.bit, self.shaws, self.works, len(self.layer_works), self.diameter, float(dt),
                          self.blow_energy, self.blow_rate, self.bars],
                  outputs=[*self._tables(), self.depth, self.entry, self.engaged, self.holes_count, self.full,
                           self.removed, self.sink, self.loose, self.layer_removed, self.thrust, self.torque,
                           self.feed, self.jammed, self.bar_cut],
                  device=self.ctx.device)
        if self.pin_set is not None:
            self.pin_set.update(state_in, state_out, self)

    # --- host side ----------------------------------------------------------------------------------------------

    def set_diameter(self, diameter: Any) -> None:
        """Drill diameter [m] per world (or one for all), host side between periods: a tool change before a hole."""
        d = np.broadcast_to(np.asarray(diameter, np.float64), (self.ctx.world_count,))
        if not (np.isfinite(d).all() and (d > 0.0).all()):
            raise PluginError(f"{self.label}: diameters must be finite and > 0")
        self.set_input("diameter", d)

    def set_hammer(self, energy: Any, rate: Any) -> None:
        """A rotary hammer's blow energy [J] and blow rate [1/s] per world (or one for all); 0 turns it off."""
        W = self.ctx.world_count
        e = np.broadcast_to(np.asarray(energy, np.float64), (W,))
        f = np.broadcast_to(np.asarray(rate, np.float64), (W,))
        if not (np.isfinite(e).all() and np.isfinite(f).all() and (e >= 0.0).all() and (f >= 0.0).all()):
            raise PluginError(f"{self.label}: blow energies and rates must be finite and >= 0")
        self.set_input("blow_energy", e)
        self.set_input("blow_rate", f)

    def thrust_law(self, feed: Any, diameter: Any | None = None, layer: int = 0) -> np.ndarray:
        """Shaw-Oxford thrust [N] of a fully engaged drill at feed per revolution ``feed`` [m] (host side)."""
        d = self.config.diameter if diameter is None else diameter
        f, d = np.asarray(feed, np.float64), np.asarray(d, np.float64)
        return float(self.coefficients_of(layer).lips) * f**0.8 * d**0.8 + self.chisel_thrust(d, layer)

    def chisel_thrust(self, diameter: Any, layer: int = 0) -> np.ndarray:
        """The chisel edge's thrust [N] (independent of the feed), host side."""
        return float(self.coefficients_of(layer).chisel) * np.asarray(diameter, np.float64) ** 2

    def torque_law(self, feed: Any, diameter: Any | None = None, layer: int = 0) -> np.ndarray:
        """Shaw-Oxford torque [N·m] of a fully engaged drill (host side)."""
        d = self.config.diameter if diameter is None else diameter
        f, d = np.asarray(feed, np.float64), np.asarray(d, np.float64)
        return float(self.coefficients_of(layer).torque) * f**0.8 * d**1.8

    @property
    def coefficients(self) -> Shaw:
        """The SI Shaw-Oxford constants of the first layer (usable before setup)."""
        return self.coefficients_of(0)

    def coefficients_of(self, layer: int) -> Shaw:
        return shaw_coefficients(self.config, self.stack()[layer])

    def holes(self, world: int = 0, layer: int | None = None) -> list[Hole]:
        """The holes cut so far in ``world`` (every layer, or one), as records in each layer's face frame."""
        read = self.read()
        c = self.config
        layers = range(len(self.layer_works)) if layer is None else (layer,)
        f32 = lambda x: float(np.float32(x))                                    # noqa: E731 (as the kernels see it)
        return [h for i in layers for h in holes_of(read, world, i, int(c.max_holes), int(c.max_stages),
                                                     f32(self.layer_works[i].thickness), f32(self.bit.tan_half))]

    def face(self, world: int, layer: int, body_q: np.ndarray) -> np.ndarray:
        """A layer's face transform in the world (x, y, z, qx, qy, qz, qw) for host ``body_q`` (``state.body_q``)."""
        p: Patch = self.patches[world][layer]
        local = np.asarray(p.local, np.float64)
        if p.body < 0:
            return local
        q = np.asarray(body_q, np.float64)[p.body]
        return np.concatenate([q[:3] + quat_rotate(q[3:], local[:3]), _quat_mul(q[3:], local[3:])])

    def read(self) -> dict[str, np.ndarray]:
        out = super().read()
        density = np.asarray([float(w.density) for w in self.layer_works])
        out["volume"] = (out["layer_removed"] / density).sum(axis=1)
        out["diameter"] = self.diameter.numpy().copy()
        if self.pin_set is not None and self.pin_set.anchors:
            out["nut_torque"] = self.pin_set.nut_torque.numpy().copy()
            out["anchor_clamp"] = self.pin_set.clamp(out)
        if out["full"].any():
            log.warning("%s: a hole table filled up in worlds %s; raise max_holes / max_stages", self.label,
                        np.flatnonzero(out["full"]).tolist())
        return out

    def test_scene(self) -> TestScene:
        return laws.drilling_scene(self.config)

    def laws(self) -> list[Law]:
        return laws.drill_laws(self.config)


def shaw_coefficients(c: Drill.Config, layer: Layer | None = None) -> Shaw:
    """Machinery's Handbook's Shaw-Oxford constants in SI (f and d in m): exact unit conversion from lbf and inches.
    A layer naming a work material uses its Kd for thrust and for torque; otherwise the config's ``work_factor``."""
    kt = km = c.work_factor
    if layer is not None and layer.work is not None:
        row = work_material(layer.work)
        kt, km = (0.0, 0.0) if isinstance(row, PercussiveMaterial) else (row.kd_thrust, row.kd_torque)
    s = Shaw()
    w = c.wear_factor
    s.lips = float(LBF * 2.0 * kt * c.chisel_b * w / INCH**1.6)
    s.chisel = float(LBF * kt * c.chisel_j * w / INCH**2)
    s.torque = float(IN_LBF * km * c.chisel_a * w / INCH**2.6)
    return s


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def _stack_response(model: newton.Model, tools: np.ndarray, per_layer: list[list[Patch]], c: Drill.Config):
    """The worst layer's front and wall mobility (a floating layer adds its own) and the smallest spin inertia."""
    rs = [drill_response(model, tools, patches, c) for patches in per_layer]
    return DrillResponse(max(r.front_mobility for r in rs), max(r.wall_mobility for r in rs),
                         min(r.spin_inertia for r in rs))


def _bit(c: Drill.Config, r: DrillResponse) -> Bit:
    bit = Bit()
    bit.tip = wp.vec3(*c.tip)
    bit.axis = wp.vec3(*(np.asarray(c.axis, np.float64) / np.linalg.norm(c.axis)))
    bit.tan_half = float(math.tan(math.radians(0.5 * c.point_angle)))
    bit.chisel = float(c.chisel_ratio)
    bit.stiffness, bit.damping = float(c.stiffness), float(c.damping)
    bit.radial, bit.radial_damping = float(c.radial_stiffness), float(c.radial_damping)
    bit.feed_max, bit.spin_eps = float(c.feed_max), float(c.spin_eps)
    bit.spin_inertia = float(r.spin_inertia)
    bit.holes, bit.stages = int(c.max_holes), int(c.max_stages)
    return bit


def _work(lay: Layer, capture: float, model: newton.Model, shape: int, size: tuple[float, float]) -> Work:
    if lay.thickness is not None:
        thickness = lay.thickness
    elif int(model.shape_type.numpy()[shape]) == int(newton.GeoType.BOX):
        thickness = 2.0 * float(model.shape_scale.numpy()[shape][2])
    else:
        raise PluginError(f"workpiece {lay.workpiece!r} is not a box: set thickness=")
    work = Work()
    work.thickness, work.density, work.capture = float(thickness), float(lay.density), float(capture)
    work.half = wp.vec2(0.5 * size[0], 0.5 * size[1])
    row = work_material(lay.work) if lay.work is not None else None
    work.ev_n = 0
    if isinstance(row, PercussiveMaterial):                 # rotary hammer: e_v at up to four diameters
        if len(row.diameters) > 4:
            raise PluginError(f"layer {lay.workpiece!r}: percussive row {row.name!r} has more than 4 diameters")
        pad = lambda v: [*v, *([v[-1]] * (4 - len(v)))]    # noqa: E731
        work.ev_n = len(row.diameters)
        work.ev_d, work.ev = wp.vec4(*pad(row.diameters)), wp.vec4(*pad(row.energy))
    return work


__all__ = ["Bar", "Drill", "Layer", "Pin", "shaw_coefficients"]
