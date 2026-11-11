# SPDX-License-Identifier: Apache-2.0
"""Test scenes and laws of the removal plug-in (plan laws R-1..R-3, notes/wp13-engine10-plans.md section 2).

- R-1 ``archard``: a pin on a flat at constant load (its weight, no friction) slides a distance s; the removed volume
  is V = (K / H)·F_N·s within 5 %.
- R-2 ``preston``: a flat pad spins in place on a flat at uniform pressure p; Preston's law dh/dt = k_p·p·v with
  v = ω·r gives the radial profile h(r) = k_p·p·ω·r·t. The fitted slope is within 5 % and the profile is linear.
- R-3 ``mass_ledger``: a sanding block with dust capture and pick-up slides over loose dirt; per world,
  removed + loose(t=0) = loose + sink to float64 rounding.
- ``pad_load``: a resting tool's pad carries its weight (1 %); ``no_removal_when_lifted``: a pad 5 mm above the face
  removes nothing and pushes nothing.

Every scene puts a static plate (the workpiece) under a free tool block whose pad face sits 5 mm below the block, so
the solver never touches the plate: only the pad does.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import Law, LawResult, TestScene, body_xform

G = 9.81
PLATE_HALF = (0.1, 0.1, 0.005)       # m
GAP = 0.005                          # m between the pad face and the tool block's rigid bottom
TOOL_DENSITY = 2000.0                # kg/m^3

PIN = {"friction": 0.0, "pad_radius": 0.01, "pad_center": (0.0, 0.0, -0.015), "pad_damping": 1.0e-3}
SPIN = {"friction": 0.0, "pad_radius": 0.03, "pad_center": (0.0, 0.0, -0.015), "pad_stiffness": 1.0e7}
LEDGER = {"capture": 0.5, "wipe_length": 0.02}
PIN_SPEED, SPIN_RATE, PUSH = 0.2, 20.0, 0.5      # m/s, rad/s, m/s
DIRT = 1.0e-6                                    # kg of loose dirt per cell in the ledger scene


def merged(c: Any, overrides: dict[str, Any]) -> Any:
    return dataclasses.replace(c, **overrides)


def tool_half(c: Any) -> tuple[float, float, float]:
    """Tool block half extents: wider than the pad, its bottom GAP above the pad face."""
    side = c.pad_radius + 0.005
    return side, side, max(-float(c.pad_center[2]) - GAP, 0.005)


def resting_height(c: Any, lift: float = 0.0) -> float:
    """Tool body height that puts its pad face at the static indentation under its weight (plus ``lift``)."""
    hx, hy, hz = tool_half(c)
    weight = TOOL_DENSITY * 8.0 * hx * hy * hz * G
    sink = weight / (c.pad_stiffness * math.pi * c.pad_radius**2)
    return 2.0 * PLATE_HALF[2] - float(c.pad_center[2]) - sink + lift


def plate_and_tool(c: Any, x0: float = 0.0, lift: float = 0.0):
    def build(b: newton.ModelBuilder) -> None:
        b.add_shape_box(-1, xform=body_xform(0.0, 0.0, PLATE_HALF[2]), hx=PLATE_HALF[0], hy=PLATE_HALF[1],
                        hz=PLATE_HALF[2], label=c.workpiece)
        tool = b.add_body(xform=body_xform(x0, 0.0, resting_height(c, lift)), label=c.tool)
        hx, hy, hz = tool_half(c)
        b.add_shape_box(tool, hx=hx, hy=hy, hz=hz, cfg=newton.ModelBuilder.ShapeConfig(density=TOOL_DENSITY))
    return build


def start_velocity(linear: tuple[float, float, float] = (0.0, 0.0, 0.0),
                   angular: tuple[float, float, float] = (0.0, 0.0, 0.0), dirt: float = 0.0):
    def init(sim: Any) -> None:
        qd = sim.state.joint_qd.numpy().reshape(-1, 6)    # one free joint per world: (vx, vy, vz, wx, wy, wz)
        qd[:, :3], qd[:, 3:] = linear, angular
        sim.state.joint_qd.assign(qd.reshape(-1))
        newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)
        if dirt > 0.0:
            for p in sim.plugins.plugins.values():
                if hasattr(p, "set_loose"):
                    p.set_loose(np.full(p.loose.shape[1:], dirt))
    return init


def sanding_scene(c: Any) -> TestScene:
    """The block is pushed at 0.5 m/s across the plate and slows to rest under the pad's friction."""
    return TestScene(build=plate_and_tool(c, x0=-0.03), init=start_velocity((PUSH, 0.0, 0.0)), ground=False,
                     periods=15)


def removal_laws(c: Any) -> list[Law]:
    pin, spin, ledger = merged(c, PIN), merged(c, SPIN), merged(c, LEDGER)
    return [
        Law("archard", check_archard, TestScene(build=plate_and_tool(pin, x0=-0.05), ground=False, periods=25,
                                                 init=start_velocity((PIN_SPEED, 0.0, 0.0))),
            "R-1: V = (K/H) F_N s for a pin at constant load", config=PIN),
        Law("preston", check_preston, TestScene(build=plate_and_tool(spin), ground=False, periods=25,
                                                 init=start_velocity(angular=(0.0, 0.0, SPIN_RATE))),
            "R-2: h(r) = k_p p omega r t under a spinning flat pad", config=SPIN),
        Law("mass_ledger", check_ledger, TestScene(build=plate_and_tool(ledger, x0=-0.03), ground=False, periods=15,
                                                    init=start_velocity((PUSH, 0.0, 0.0), dirt=DIRT)),
            "R-3: removed + loose(0) = loose + sink per world", config=LEDGER),
        Law("pad_load", check_load, TestScene(build=plate_and_tool(c), ground=False, periods=15),
            "a resting tool's pad carries its weight"),
        Law("no_removal_when_lifted", check_lifted,
            TestScene(build=plate_and_tool(c, lift=0.005), ground=False, gravity=(0.0, 0.0, 0.0), periods=10,
                      init=start_velocity((PUSH, 0.0, 0.0))),
            "a pad 5 mm above the face removes nothing"),
    ]


# --- checks ------------------------------------------------------------------------------------------------------

def _series(run: Any, name: str) -> np.ndarray:
    return run.trace.plugin(run.plugin.label, name)


def _weight(run: Any) -> float:
    b = run.bodies(run.plugin.config.tool)[0]
    return float(run.model.body_mass.numpy()[b]) * G


def _rate(run: Any) -> float:
    c = run.plugin.config
    return c.wear_coefficient / c.hardness


def check_archard(run: Any) -> LawResult:
    b = run.bodies(run.plugin.config.tool)[0]
    x = run.trace.body_q[:, b, :2].astype(np.float64)
    slid = float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())
    got = float(_series(run, "volume")[-1, 0])
    want = _rate(run) * _weight(run) * slid
    if want <= 0.0:
        return LawResult(False, math.nan, 0.05, "the pin did not slide", "the pin must slide on its pad")
    return LawResult.within((got - want) / want, 0.05,
                            f"removed {got:.4g} m^3 vs (K/H) F s = {want:.4g} m^3 over {slid * 1e3:.1f} mm",
                            "remove dh = (K/H) p ds per cell, with ds the slip of the pad over that cell")


def check_preston(run: Any) -> LawResult:
    p, c = run.plugin, run.plugin.config
    b = run.bodies(c.tool)[0]
    depth = _series(run, "depth")[-1, 0]
    nx, ny = depth.shape
    hx, hy = float(p.grid.half[0]), float(p.grid.half[1])
    dx, dy = 2.0 * hx / nx, 2.0 * hy / ny
    xc = (np.arange(nx) + 0.5) * dx - hx
    yc = (np.arange(ny) + 0.5) * dy - hy
    centre = run.trace.body_q[:, b, :2].astype(np.float64).mean(axis=0)
    r = np.hypot(xc[:, None] - centre[0], yc[None, :] - centre[1])
    inner = r < c.pad_radius - 2.0 * max(dx, dy)
    slope = float((depth[inner] * r[inner]).sum() / (r[inner] ** 2).sum())
    spin = float(np.mean(run.trace.body_qd[1:, b, 5]))
    elapsed = float(run.trace.t[-1] - run.trace.t[0])
    want = _rate(run) * _weight(run) / (math.pi * c.pad_radius**2) * spin * elapsed
    residual = float(np.sqrt(np.mean((depth[inner] - slope * r[inner]) ** 2)) / (slope * r[inner].max()))
    err = (slope - want) / want
    worst = max(abs(err), residual)
    return LawResult.within(worst, 0.05, f"dh/dr {slope:.4g} vs k_p p omega t {want:.4g} ({err:+.2%}), "
                            f"profile residual {residual:.2%} of h(r_max)",
                            "removal rate per cell must be k_p p |v_slip| with the slip of the pad point over it")


def check_ledger(run: Any) -> LawResult:
    removed = _series(run, "removed")
    loose = _series(run, "loose_total")
    sink = _series(run, "sink")
    balance = removed + loose[0] - loose - sink
    scale = float(np.abs(removed + loose[0]).max())
    if float(removed[-1].min()) <= 0.0 or float(sink[-1].min()) <= 0.0:
        return LawResult(False, math.nan, 1e-9, "nothing removed or nothing reached the sink",
                         "the block must slide over the plate with its pad pressed")
    return LawResult.within(float(np.abs(balance).max()) / scale, 1.0e-9,
                            f"relative ledger error ({float(removed[-1, 0]):.3g} kg removed, "
                            f"{float(sink[-1, 0]):.3g} kg in the sink)",
                            "add to loose or sink exactly what leaves the surface, in float64")


def check_load(run: Any) -> LawResult:
    load = float(_series(run, "load")[-1, 0])
    weight = _weight(run)
    return LawResult.within((load - weight) / weight, 0.01, f"pad load {load:.4g} N vs weight {weight:.4g} N",
                            "pressure stiffness x indentation, summed with each cell's covered area")


def check_lifted(run: Any) -> LawResult:
    moved = float(np.abs(_series(run, "volume")).max()) + float(np.abs(_series(run, "load")).max())
    return LawResult.within(moved, 0.0, "removed volume + pad load of a pad hovering 5 mm above the face",
                            "press only cells whose surface point lies above the pad face")


__all__ = ["merged", "removal_laws", "sanding_scene"]
