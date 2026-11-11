# SPDX-License-Identifier: Apache-2.0
"""GA's fillet welding on Alloy (SCR-WP8-14): the coupon, its thermal grid, a scripted torch pass and the run.

Gym-Anything's welding cell (report_welding §3-6, §10) lays a ¼ in horizontal fillet along a T-joint. This port holds
the coupon on ``GridThermal`` (heat v3: grid heat with melting, the moving arc, deposition with the cap model) and
drives the torch as a kinematic body along GA's passes; ``weld_ga_measures`` reads GA's sections and
``weld_ga_criteria`` judges them as GA does.

The weld frame is GA's: x along the joint, y out of the bracket's weld face, z up from the plate; the root is the x
axis.
- Coupon: A36 plate 200 x 150 x 10 mm, bracket 150 x 75 x 8 mm standing on it with its weld face at y = 0 over
  x ∈ [0, 150] mm, the two sharing their x-y centre (GA l.78-79, 311-317, 1919-1927; the axis split is read from
  the layout, not stated). The plate lies on a steel table: the grid's underside sink (1500 W/(m²·K), GA §4).
- Grid: GA's, 0.5 mm along x over the bracket and 0.25 mm across in the band y, z ∈ [−10, 16] mm, growing 1.2x to
  3 mm (GA §3); ``grid_config(fine=...)`` coarsens it for CPU checks.
- Tacks (the tacked start state): GA's fixture (env.py ``_tack_fixture``): a torch on the joint's own spray schedule
  at 19 mm runs two 15 mm tacks, 8-23 and 127-142 mm, at 10 mm/s, aimed at the root with a 5° push, the arc off 0.5 s
  between them, then 8 s to cool (``TACK_PASSES``).
- Process (GA §5, the Lincoln schedule): 0.9 mm ER70S-6 at 12.7 m/min, 29 V, 241.3 A in spray, η = 0.80, so 5.60 kW
  into the work, 98 % of the wire deposited; σ = 4 mm. The torch's work angle is 45°, its push 7.5°, the tip 19 mm
  from the aim point. The reference pass is GA's aim_tacks (l.2364-2367): from x = 1 mm at 5.0 mm/s to 147 mm, a
  1.2 s crater dwell, then the arc off and 5 s to cool. The torch aims at the root, and 1 mm out on the plate over
  each tack's start (5-13 and 124-132 mm), which washes in the tacks' cold laps (GA l.2366). A pass with another push
  turns the torch about its tip.

- Pool: the wire lands where the arc was 0.5 s before. With ``config={"arc_pressure": ARC_PRESSURE}`` the liquid also
  flows along the weld to one pressure under GA's arc force (0.0214 N over σ 1.79 mm at 241.3 A, ``grid_flow``):
  every task scores higher with it (heat-v3-design.md §Slice 9); the tasks switch to it on physics-coord's ruling.

Out of scope (heat v3's limits, docs/notes/heat-v3-design.md): the robots (a kinematic torch), the arc circuit (a
fixed operating point), shielding gas and porosity, the pool's gravity, undercut, spatter and fume (and, by default,
the pool's flow along the weld).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from alloy._vendor import newton

from alloy.plugins.harness.scene import TestScene, body_xform, build_simulation
from alloy.plugins.thermal.grid import GridThermal, graded

PLATE = (-0.025, -0.079, -0.010, 0.175, 0.071, 0.0)    # m, (x0, y0, z0, x1, y1, z1), weld frame
BRACKET = (0.0, -0.008, 0.0, 0.150, 0.0, 0.075)
TACKS = ((0.008, 0.023), (0.127, 0.142))               # m along x: GA env.TACKS
AIM_TACKS = ((0.005, 0.013, 1.0e-3, 0.0), (0.124, 0.132, 1.0e-3, 0.0))   # (x0, x1, y, z): 1 mm out, over tack starts
VOLTS, AMPS, ETA = 29.0, 241.3, 0.80
FEED = 12.7 / 60.0                                    # m/s
SIGMA = 4.0e-3
CTWD = 0.019
LAG = 0.5                                              # s: the wire lands where the arc was (2.5 mm at 5 mm/s)
ARC_PRESSURE = (-0.04017 + 2.5533e-4 * AMPS, 1.4875e-3 + 1.233e-6 * AMPS)   # N, m: GA's ARC_FORCE, ARC_PRESSURE_SIGMA
LABEL, TORCH = "grid_thermal", "torch"
WELD_CODE = 3                                          # cell code of weld metal: plate 1, bracket 2, weld 3


@dataclass(frozen=True)
class Pass:
    """A scripted pass: strike at ``start``, travel at ``speed`` to ``end``, dwell, arc off, cool."""

    start: float = 0.001
    end: float = 0.147
    speed: float = 0.005
    dwell: float = 1.2
    cool: float = 5.0
    work_deg: float = 45.0
    push_deg: float = 7.5
    aim: tuple[float, float] = (0.0, 0.0)           # (y, z) of the aim point: the root
    aim_zones: tuple[tuple[float, float, float, float], ...] = ()   # (x0, x1, y, z): aim elsewhere over [x0, x1)
    power: float = ETA * VOLTS * AMPS                # W into the work
    feed: float = FEED

    @property
    def travel_s(self) -> float:
        return (self.end - self.start) / self.speed

    @property
    def axis(self) -> np.ndarray:
        """Unit beam direction (torch to work), weld frame: down the 45° bisector toward the root, pushed forward."""
        work = math.radians(self.work_deg)
        base = np.array([0.0, -math.sin(work), -math.cos(work)])
        push = math.radians(self.push_deg)
        d = math.cos(push) * base + math.sin(push) * np.array([1.0, 0.0, 0.0])
        return d / np.linalg.norm(d)


def grid_config(fine: float = 2.5e-4, along: float = 5.0e-4, coarse: float = 3.0e-3) -> dict:
    """GridThermal's config for the coupon (GA's grid at the defaults)."""
    return dict(material="steel_a36",
                x=graded(PLATE[0], PLATE[3], along, band=(BRACKET[0], BRACKET[3]), coarse=coarse),
                y=graded(PLATE[1], PLATE[4], fine, band=(-0.010, 0.016), coarse=coarse),
                z=graded(PLATE[2], BRACKET[5], fine, band=(-0.010, 0.016), coarse=coarse),
                boxes=(PLATE, BRACKET), parts=(0, 1), weld_part=WELD_CODE - 1, sink=("z-",), torch=TORCH,
                arc_sigma=SIGMA, bead_model="cap", joint_root=(0.0, 0.0), joint_rays=((1.0, 0.0), (0.0, 1.0)),
                deposit_reach=0.012, max_fill=512, deposit_lag=LAG)


def _scene(proc: Pass) -> TestScene:
    tip = _tip(proc)

    def build(b: newton.ModelBuilder) -> None:
        torch = b.add_body(xform=body_xform(*tip), mass=1.0, label=TORCH)
        b.add_shape_sphere(torch, radius=2.0e-3, label=TORCH)
    return TestScene(build=build, worlds=1, periods=1, control_dt=0.01, dt=0.01, ground=False,
                     gravity=(0.0, 0.0, 0.0))


def _set_torch(sim, v: float, at: np.ndarray | None = None, quat: np.ndarray | None = None) -> None:
    """The torch's free joint: speed along x, and (``at``, ``quat``) its pose (a jump between passes, arc off)."""
    if at is not None:
        q = sim.state.joint_q.numpy()
        q[:3] = at
        if quat is not None:
            q[3:7] = quat
        sim.state.joint_q.assign(q)
    qd = sim.state.joint_qd.numpy()
    qd[:3] = (v, 0.0, 0.0)
    sim.state.joint_qd.assign(qd)
    newton.eval_fk(sim.model, sim.state.joint_q, sim.state.joint_qd, sim.state)


def _turn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The quaternion (x, y, z, w) turning unit vector a onto b."""
    c = np.cross(a, b)
    s = float(np.linalg.norm(c))
    if s < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ang = math.atan2(s, float(a @ b))
    return np.array([*(c / s * math.sin(0.5 * ang)), math.cos(0.5 * ang)])


def _tip(proc: Pass) -> np.ndarray:
    return _tip_at(proc, proc.start)


def _aim(proc: Pass, x: float) -> tuple[float, float]:
    for x0, x1, y, z in proc.aim_zones:
        if x0 <= x < x1:
            return y, z
    return proc.aim


def _tip_at(proc: Pass, x: float) -> np.ndarray:
    return np.array([x, *_aim(proc, x)]) - CTWD * proc.axis


def _segments(proc: Pass) -> list[tuple[float, float]]:
    """The travel split where the aim changes: [(x0, x1), ...] from start to end."""
    cuts = {proc.start, proc.end} | {v for z in proc.aim_zones for v in z[:2] if proc.start < v < proc.end}
    xs = sorted(cuts)
    return list(zip(xs[:-1], xs[1:], strict=True))


def _arc(grid: GridThermal, on: Pass | None) -> None:
    grid.set_input("arc_power", np.array([on.power if on else 0.0], np.float32))
    grid.set_input("wire_feed", np.array([on.feed if on else 0.0], np.float32))


def run(passes: Pass | tuple[Pass, ...] = Pass(), device: str = "cuda:0", config: dict | None = None,
        log=print, after_pass=None) -> GridThermal:
    """Run the passes in order on the bare coupon (each: strike, travel along +x, dwell, arc off, cool; the torch then
    jumps to the next start, turned to its push); ``after_pass(i, grid)`` sees the grid after each. Returns the
    plug-in (``fields(0)``, ``read()``; ``weld_ga_measures.measure`` reads GA's sections)."""
    passes = passes if isinstance(passes, tuple) else (passes,)
    if any(p.end <= p.start or p.speed <= 0.0 for p in passes):
        raise ValueError("passes travel along +x: end > start, speed > 0 (one torch axis for the run)")
    cfg = {**grid_config(), "torch_axis": tuple(float(v) for v in passes[0].axis), **(config or {})}
    sim = build_simulation(_scene(passes[0]), [GridThermal(**cfg)], device)
    GridThermal.exclusive(sim.plugins)
    grid = sim.plugins[LABEL]
    grid.set_input("arc_sigma", np.array([SIGMA], np.float32))
    period, t, k = 0.01, 0.0, 0                        # the scene's control period
    for i, proc in enumerate(passes):
        _arc(grid, proc)
        legs = [((b - a) / proc.speed, ("move", a)) for a, b in _segments(proc)]
        for seconds, (change, x) in (*legs, (proc.dwell, ("dwell", 0.0)), (proc.cool, ("off", 0.0))):
            if change == "move":                       # (re)aim at the leg's start: the torch jumps sideways
                _set_torch(sim, proc.speed, _tip_at(proc, x) if (i or x > proc.start) else None,
                           _turn(passes[0].axis, proc.axis))
            elif change == "dwell":
                _set_torch(sim, 0.0)
            elif change == "off":
                _arc(grid, None)
            for _ in range(round(seconds / period)):
                sim.step_period()
                k, t = k + 1, t + period
                if log and k % 500 == 0:
                    s = {key: float(v[0]) for key, v in grid.read().items() if np.ndim(v) == 1}
                    log(f"pass {i + 1}/{len(passes)}, t {t:.2f} s: T max {s['t_max']:.0f} K, deposited "
                        f"{s['deposited_m3'] * 1e9:.0f} mm³, residual {s['residual_J']:.3g} J")
        if after_pass is not None:
            after_pass(i, grid)
    return grid


TACK_PASSES = tuple(Pass(start=a, end=b, speed=0.010, dwell=0.0, cool=c, push_deg=5.0)
                    for (a, b), c in zip(TACKS, (0.5, 8.0), strict=True))      # GA's fixture: env._tack_fixture


__all__ = ["BRACKET", "PLATE", "Pass", "TACK_PASSES", "WELD_CODE", "grid_config", "run"]
