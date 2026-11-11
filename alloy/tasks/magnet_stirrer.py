# SPDX-License-Identifier: Apache-2.0
"""task_magnet_stirrer (Gym-Anything titration "start-stirring"): spin a magnetic stirrer up to a steady stir.

Scene (one world):
- a hot-plate top;
- a glass dish (floor + 12-segment wall, 60 mm inside, static);
- a PTFE stir bar, 25 x 8 x 8 mm and 4 g, standing on its pivot ring: a 1.5 mm sphere under its centre, with the
  bar 0.5 mm clear of the floor. It meets the glass at μ 0.1 (ESTIMATE). The contact is stiff: ke 1e5 and kd
  2·sqrt(ke).
  - A plain box bar spins on its 4 corners. That contact chatters, and in a first run it threw a 600 rpm bar out of
    step while 800 and 1,000 rpm ran. Real stir bars carry the ring for this reason.
The drive magnet (``Magnetism``: 0.375 A·m², ESTIMATE N35-scale) sits 15 mm below the bar's centre and turns at the
commanded rpm. Water's drag on the bar is ``StirDrag``:
- a torque -k·|ω|·ω about the vertical, with k = ρ·C_d·w·L⁴ / 64 for a rod of length L and height w in cross-flow
  (C_d 1.2, ESTIMATE);
- it ignores the swirl the bar sets up in the water, so it over-estimates the drag at speed;
- no liquid particles: the task stays light, and the SPH water route can replace this term.

Script: the target rpm differs per world. The drive ramps 0 → target over 1 s (cosine), then holds for 1.5 s.

Success per world, over the last 1 s:
- the bar's mean spin is within 5 % of the drive's;
- its centre stays within 5 mm of the drive axis;
- it stays on the floor (z within 2 mm).

Above the decoupling speed (``decoupling_rpm``: the bar's pull-out torque m·B equals drag + floor friction) the bar
falls out of step and spins erratically, which is the stirrer "spinning out".

    python -m alloy.tasks.magnet_stirrer --worlds 8 --device cpu --out rows.jsonl
writes one row: the per-world targets and outcomes, the predicted decoupling speed and the cost per step with
and without the plug-ins.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from alloy._vendor import newton

from alloy.plugins.base import ForcePlugin, Forces
from alloy.plugins.funcs import add_wrench, torque_wrench
from alloy.plugins.harness.scene import TestScene, body_xform
from alloy.plugins.magnetism import Magnetism

BAR = (0.0125, 0.004, 0.004)                 # half extents (m)
BAR_MASS = 0.004                             # kg
M_BAR, M_DRIVE, H_DRIVE = 0.0675, 0.375, 0.015
MU_PTFE = 0.1
STIFF = {"ke": 1.0e5, "kd": 2.0 * math.sqrt(1.0e5)}
RHO_WATER, CD_ROD = 998.0, 1.2
K_DRAG = RHO_WATER * CD_ROD * (2 * BAR[2]) * (2 * BAR[0]) ** 4 / 64.0       # N·m·s², torque = k ω²
FLOOR_Z = 0.0                                # dish floor top
RING_R, RING_CLEAR = 0.0015, 0.0005          # pivot ring radius; bar clearance above the floor
REST_Z = FLOOR_Z + BAR[2] + RING_CLEAR       # bar centre at rest
RAMP_S, HOLD_S, CONTROL_DT = 1.0, 1.5, 0.02
PERIODS = int(round((RAMP_S + HOLD_S) / CONTROL_DT))


@wp.kernel
def _drag(body_qd: wp.array(dtype=wp.spatial_vector), bodies: wp.array(dtype=wp.int32), k: float,
          body_f: wp.array(dtype=wp.spatial_vector)):
    w = wp.tid()
    b = bodies[w]
    wz = wp.spatial_bottom(body_qd[b])[2]
    add_wrench(body_f, b, torque_wrench(wp.vec3(0.0, 0.0, -k * wp.abs(wz) * wz)))


class StirDrag(ForcePlugin):
    """Water drag on a spinning stir bar: torque -k·|ω_z|·ω_z about the vertical (ESTIMATE, see the module)."""

    name = "stir_drag"
    summary = "quadratic spin drag on a stir bar (task_magnet_stirrer)"

    @dataclass
    class Config:
        body: str = "stir_bar"
        k: float = K_DRAG                     # N·m·s²

    def setup(self, ctx: Any) -> None:
        self.bodies = ctx.bodies(self.config.body)

    def apply(self, state: newton.State, forces: Forces, dt: float) -> None:
        wp.launch(_drag, dim=self.ctx.world_count, device=self.ctx.device,
                  inputs=[state.body_qd, self.bodies, float(self.config.k)], outputs=[forces.body_f])

    def test_scene(self) -> TestScene:
        return scene(3, 25)

    def laws(self) -> list:
        return []


def build(b: newton.ModelBuilder) -> None:
    glass = newton.ModelBuilder.ShapeConfig(mu=MU_PTFE, **STIFF)
    b.add_shape_box(-1, xform=wp.transform((0.0, 0.0, FLOOR_Z - 0.002), wp.quat_identity()), hx=0.05, hy=0.05,
                    hz=0.002, cfg=glass, label="dish_floor")
    for k in range(12):
        a = 2 * math.pi * k / 12
        q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), a)
        b.add_shape_box(-1, xform=wp.transform((0.033 * math.cos(a), 0.033 * math.sin(a), FLOOR_Z + 0.015), q),
                        hx=0.003, hy=0.009, hz=0.015, cfg=glass, label=f"dish_wall_{k}")
    bar = b.add_body(xform=body_xform(0.0, 0.0, REST_Z, (0.0, 0.0, 1.0, 0.0)), label="stir_bar")
    vol = 8 * BAR[0] * BAR[1] * BAR[2]
    ptfe = newton.ModelBuilder.ShapeConfig(density=BAR_MASS / vol, mu=MU_PTFE, **STIFF)
    b.add_shape_box(bar, hx=BAR[0], hy=BAR[1], hz=BAR[2], cfg=ptfe)
    ring = newton.ModelBuilder.ShapeConfig(density=0.0, mu=MU_PTFE, **STIFF)
    b.add_shape_sphere(bar, xform=wp.transform((0.0, 0.0, -BAR[2] - RING_CLEAR + RING_R), wp.quat_identity()),
                       radius=RING_R, cfg=ring)


def scene(worlds: int = 3, periods: int = PERIODS) -> TestScene:
    return TestScene(build=build, worlds=worlds, periods=periods, ground=False)


def magnets() -> Magnetism:
    return Magnetism(bodies=("stir_bar",), moments=(M_BAR,), axes=((1.0, 0.0, 0.0),),
                     drives=((0.0, 0.0, FLOOR_Z + BAR[2] - H_DRIVE),), drive_moments=(M_DRIVE,), drive_rpm=(0.0,))


def decoupling_rpm() -> float:
    """The drive speed where the pull-out torque m·B equals the drag k ω². The pivot ring is a point contact, so
    floor friction adds no spin torque."""
    b = 1e-7 * M_DRIVE / H_DRIVE ** 3
    return math.sqrt(M_BAR * b / K_DRAG) * 60.0 / (2 * math.pi)


def rpm_at(t: float, target: np.ndarray) -> np.ndarray:
    s = min(t / RAMP_S, 1.0)
    return target * 0.5 * (1.0 - math.cos(math.pi * s))


def run(targets: np.ndarray, device: str = "cpu", with_plugins: bool = True, graph: bool = False) -> dict:
    """Run the script with one target rpm per world; per-world outcome and the wall ms per exchange step."""
    from alloy.plugins.harness.scene import build_simulation  # noqa: PLC0415

    targets = np.asarray(targets, np.float32)
    worlds = len(targets)
    plugs = [magnets(), StirDrag()] if with_plugins else []
    sim = build_simulation(scene(worlds), plugs, device, worlds, graph=graph)
    bar = np.flatnonzero(np.array(list(sim.model.body_label)) == "stir_bar")
    yaw, ts, xy, z = [], [], [], []
    t_wall, steps = 0.0, 0
    for k in range(PERIODS):
        t = k * CONTROL_DT
        if plugs:
            plugs[0].set_input("rpm", rpm_at(t + CONTROL_DT, targets)[:, None])
        t0 = time.perf_counter()
        sim.step_period()
        if k >= 1:
            wp.synchronize_device(sim.model.device)
            t_wall += time.perf_counter() - t0
            steps += 2 ** int(sim.clock.j)
        q = sim.state.body_q.numpy()[bar]
        yaw.append(2.0 * np.arctan2(q[:, 5], q[:, 6]))
        ts.append(t + CONTROL_DT)
        xy.append(np.hypot(q[:, 0], q[:, 1]))
        z.append(q[:, 2])
    return {"task": "task_magnet_stirrer", "plugins": with_plugins, "worlds": worlds, "device": device,
            **score(np.asarray(ts), np.asarray(yaw), np.asarray(xy), np.asarray(z), targets),
            "ms_per_step": 1e3 * t_wall / max(steps, 1)}


def score(t: np.ndarray, yaw: np.ndarray, off_axis: np.ndarray, z: np.ndarray, targets: np.ndarray) -> dict:
    """Per-world success from the bar's yaw, distance from the axis and height per period ([K, W] each)."""
    late = t > t[-1] - 1.0
    spin = np.unwrap(yaw, axis=0)
    i0 = int(np.argmax(late))
    speed = (spin[-1] - spin[i0]) / (t[-1] - t[i0])                     # rad/s, per world
    omega = targets * 2 * np.pi / 60
    ratio = np.where(omega > 0, speed / np.where(omega > 0, omega, 1.0), 0.0)
    ok = (np.abs(ratio - 1.0) <= 0.05) & (off_axis[late].max(axis=0) <= 5e-3) & \
        (np.abs(z[late] - REST_Z).max(axis=0) <= 2e-3)
    return {"targets_rpm": targets.tolist(), "success": ok.tolist(), "spin_ratio": np.round(ratio, 4).tolist(),
            "off_axis_mm_max": np.round(off_axis[late].max(axis=0) * 1e3, 3).tolist(),
            "decoupling_rpm": round(decoupling_rpm(), 1)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--worlds", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    targets = np.linspace(300.0, 1500.0, a.worlds)
    on = run(targets, a.device, True, a.graph)
    off = run(targets, a.device, False, a.graph)
    row = {**on, "ms_per_step_without": off["ms_per_step"], "cost_ratio": on["ms_per_step"] / off["ms_per_step"]}
    print(json.dumps(row))
    if a.out is not None:
        with a.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
