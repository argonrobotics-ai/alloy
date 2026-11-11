# SPDX-License-Identifier: Apache-2.0
"""Soil dig-and-pour with a kinematic dustpan scoop (WP12 soil task; docs/notes/soil-design.md).

A sand bed fills a tray; a scoop made of four kinematic steel plates (floor, back, two sides; open top and front)
digs a scoopful and pours it into a bucket beside the tray, through the public API (``alloy.Scene``, ``Sim.drive``).
The plates move as one: every leg drives all four with the same twist about a shared pivot (the back plate's outer
bottom edge). The scoop digs edge first, as a trowel does: it starts tilted front down by ``stab_deg`` with its tip
just above the bed, stabs along its floor to ``depth``, drags forward, curls back (front up) about the pivot, lifts,
carries over the bucket and tilts front down to pour. A plate face never pushes straight into the soil (a level
floor punched down squeezed grains out at 30-50 m/s and froze the world). Tray and bucket are five boxes each whose
walls run down through the floor: in ``add_open_box`` the walls stand on the floor, their bottom faces on its top
face, and for an MPM grid node inside a wall's foot (half a voxel from both) the nearest collider face is that
bottom face, pointing down. Bottom-layer grains beside a wall sank through the floor, fell below the ground and
froze the world 2 s later. (A single tray mesh avoids the seam, but health checks a static mesh by its convex
hull, so the scoop inside the tray read as a 1 cm penetration.)

    python -m alloy.tasks.soil_dig --worlds 1 --device cpu --out dig.json

Reports the soil mass lifted after the dig, the mass in the bucket at the end, the spill (outside tray and bucket)
and the health check. Masses use the built particle masses (grain_mpm bulk density).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("alloy.tasks")

__all__ = ["DigSpec", "build", "main", "ramp_profile", "run"]


@dataclass(frozen=True)
class DigSpec:
    """Geometry [m], angles [deg], speeds [m/s] and the soil bed card of the task."""

    bed: str = "soil_bed_dry"                 # dry sand pours; damp (c 4.85 kPa) stays in the scoop
    tray_inner: tuple[float, float, float] = (0.1, 0.08, 0.05)
    bucket_inner: tuple[float, float, float] = (0.06, 0.06, 0.04)
    bucket_x: float = 0.16
    scoop: tuple[float, float, float, float] = (0.04, 0.04, 0.025, 0.006)    # floor length, width, wall height, plate
    flare_deg: float = 0.0                     # side plates lean out by this much
    tray_wall: float = 0.01                    # walls >= 2 MPM voxels (5 mm): thinner walls leak soil
    bucket_wall: float = 0.01
    pivot_x: float = -0.043                    # the pivot (scoop back-bottom edge) at the start
    bed_top: float = 0.0378                    # settled bed top
    clearance: float = 0.005                   # tip above the bed top at the start
    stab_deg: float = 45.0                     # front-down tilt of the stab
    depth: float = 0.015                       # tip under the bed top after the stab
    drag: float = 0.013
    dig_speed: float = 0.05                    # stab, drag and curl (tip speed)
    speed: float = 0.1                         # lift and carry; tip speed of the pour
    tilt_deg: float = 30.0                     # front up after the curl
    pour_deg: float = 80.0                     # front down at the end of the pour
    pour_back: float = 0.03                    # pivot this far before the bucket centre while pouring
    lift: float = 0.06
    dt: float = 0.02
    ramp_s: float = 0.1                        # each leg speeds up and slows down over this long (no velocity steps)
    worlds: int = 1
    device: str = "cpu"
    roots: tuple[str, ...] = ()                # extra card roots (a custom bed card)
    parts: tuple[str, ...] = field(default=("scoop_floor", "scoop_back", "scoop_side_p", "scoop_side_n"))


Box = tuple[tuple[float, float, float], tuple[float, float, float], float]


def _local_boxes(s: DigSpec) -> dict[str, Box]:
    """Plate name -> (full size, centre, roll) of the level scoop in the pivot frame (pivot at the origin, tip at
    x = length + plate). The side plates hinge on the floor's long edges and lean out by ``flare_deg``."""
    length, width, height, t = s.scoop
    f = math.radians(s.flare_deg)
    out: dict[str, Box] = {"scoop_floor": ((length, width, t), (t + length / 2, 0.0, t / 2), 0.0),
                           "scoop_back": ((t, width, height), (t / 2, 0.0, height / 2), 0.0)}
    for name, sign in (("scoop_side_p", 1.0), ("scoop_side_n", -1.0)):
        y = sign * (width / 2 + t / 2 + 0.5 * height * math.sin(f))
        out[name] = ((length, t, height), (t + length / 2, y, 0.5 * height * math.cos(f)), -sign * f)
    return out


def _rot(axis: int, a: float) -> np.ndarray:
    """Rotation by ``a`` about x (0) or y (1); +a about y tilts the scoop front down."""
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) if axis == 0 else np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _pivot0(s: DigSpec) -> np.ndarray:
    """The pivot at the start: the tip ``clearance`` above the bed top at the stab tilt."""
    r = s.scoop[0] + s.scoop[3]
    return np.array([s.pivot_x, 0.0, s.bed_top + s.clearance + r * math.sin(math.radians(s.stab_deg))])


def _scoop_boxes(s: DigSpec) -> dict[str, tuple[tuple[float, ...], tuple[float, ...], tuple[float, float, float]]]:
    """Plate name -> (full size, centre, rpy) at the start (tilted front down by ``stab_deg`` about the pivot)."""
    a, p0 = math.radians(s.stab_deg), _pivot0(s)
    return {name: (size, tuple(float(v) for v in p0 + _rot(1, a) @ np.asarray(c)), (roll, a, 0.0))
            for name, (size, c, roll) in _local_boxes(s).items()}


def _container(scene: Any, name: str, inner: Sequence[float], wall: float, x: float) -> None:
    """A static open box from five boxes whose walls run the full outer length and down through the floor to z = 0
    (module docstring: no wall face lies on the floor's top or on another wall's inner face)."""
    ix, iy, depth = inner
    ox, oy, h = ix + 2 * wall, iy + 2 * wall, wall + depth
    parts = {"floor": ((ox, oy, wall), (0.0, 0.0, wall / 2)),
             "wall_xp": ((wall, oy, h), ((ix + wall) / 2, 0.0, h / 2)),
             "wall_xn": ((wall, oy, h), (-(ix + wall) / 2, 0.0, h / 2)),
             "wall_yp": ((ox, wall, h), (0.0, (iy + wall) / 2, h / 2)),
             "wall_yn": ((ox, wall, h), (0.0, -(iy + wall) / 2, h / 2))}
    for part, (size, c) in parts.items():
        scene.add_box(f"{name}_{part}", size=size, pos=(x + c[0], c[1], c[2]), static=True, material="abs")


def build(s: DigSpec) -> Any:
    """The scene (module docstring) built on ``s.device``."""
    import alloy  # noqa: PLC0415

    scene = alloy.Scene(dt=s.dt, worlds=s.worlds, **({"roots": list(s.roots)} if s.roots else {}))
    scene.add_ground()
    _container(scene, "tray", s.tray_inner, s.tray_wall, 0.0)
    _container(scene, "bucket", s.bucket_inner, s.bucket_wall, s.bucket_x)
    scene.add_object("soil", s.bed, pos=(0.0, 0.0, s.tray_wall - 0.004))         # the bed card sits on a 4 mm floor
    for name, (size, pos, rpy) in _scoop_boxes(s).items():
        scene.add_box(name, size=size, pos=pos, rpy=rpy, kinematic=True, material="steel_carbon")
    return alloy.build(scene, device=s.device)


class _Scoop:
    """Moves the four plates as one rigid scoop; tracks the shared pivot."""

    def __init__(self, sim: Any, s: DigSpec) -> None:
        self.sim, self.s = sim, s
        self.pivot = _pivot0(s)

    def move(self, velocity: Sequence[float], angular: Sequence[float], seconds: float) -> None:
        """One leg: the twist (``velocity`` of the pivot, ``angular``) scaled by a trapezoid profile per control
        period that covers the same distance and angle as ``seconds`` at the constant twist."""
        dt = float(self.sim.dt)
        for f in ramp_profile(seconds, dt, self.s.ramp_s):
            v, w = f * np.asarray(velocity, float), f * np.asarray(angular, float)
            for name in self.s.parts:
                self.sim.drive(name, velocity=tuple(v), angular_velocity=tuple(w), pivot=tuple(self.pivot), duration=dt)
            self.sim.step(1)
            self.pivot = self.pivot + v * dt


def ramp_profile(seconds: float, dt: float, ramp: float) -> np.ndarray:
    """Per-period speed factors: linear up over ``ramp``, flat, linear down; they sum to ``seconds / dt`` (the leg's
    distance at full speed)."""
    n = max(1, int(round(seconds / dt)))
    k = min(int(round(ramp / dt)), n // 2)
    f = np.ones(n)
    if k:
        up = (np.arange(k) + 0.5) / k
        f[:k], f[n - k:] = up, up[::-1]
    return f * (seconds / dt) / f.sum()


Leg = tuple[str, tuple[float, float, float], tuple[float, float, float], float]


def _legs(s: DigSpec) -> list[Leg]:
    """(name, pivot velocity, angular velocity, seconds) per leg; curl and pour turn about the resting pivot."""
    r = s.scoop[0] + s.scoop[3]
    a, back, down = (math.radians(d) for d in (s.stab_deg, s.tilt_deg, s.pour_deg))
    vd, v = s.dig_speed, s.speed
    stab = (s.clearance + s.depth) / math.sin(a)                        # along the floor
    carry = s.bucket_x - s.pour_back - (s.pivot_x + stab * math.cos(a) + s.drag)
    return [("stab", (vd * math.cos(a), 0.0, -vd * math.sin(a)), (0.0, 0.0, 0.0), stab / vd),
            ("drag", (vd, 0.0, 0.0), (0.0, 0.0, 0.0), s.drag / vd),
            ("curl", (0.0, 0.0, 0.0), (0.0, -vd / r, 0.0), (a + back) * r / vd),
            ("lift", (0.0, 0.0, v), (0.0, 0.0, 0.0), s.lift / v),
            ("carry", (v, 0.0, 0.0), (0.0, 0.0, 0.0), carry / v),
            ("pour", (0.0, 0.0, 0.0), (0.0, v / r, 0.0), (back + down) * r / v),
            ("rest", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 0.5)]


def scoop_path(s: DigSpec, per_leg: int = 16) -> list[tuple[str, np.ndarray]]:
    """(leg, points [n, 3]) along the motion: each plate's corners, edge midpoints and face centres (geometry only,
    for clearance checks against the tray and bucket)."""
    grid = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float) / 2
    local = np.concatenate([np.asarray(c) + (_rot(0, roll) @ (grid * size).T).T
                            for size, c, roll in _local_boxes(s).values()])
    pivot, theta, out = _pivot0(s), math.radians(s.stab_deg), []
    for name, v, w, sec in _legs(s):
        for u in np.linspace(0.0, 1.0, per_leg):
            out.append((name, pivot + u * sec * np.asarray(v) + (_rot(1, theta + u * sec * w[1]) @ local.T).T))
        pivot, theta = pivot + sec * np.asarray(v), theta + sec * w[1]
    return out


def _masses(sim: Any, s: DigSpec, top: float) -> dict[str, float]:
    """World-mean soil masses [kg]: lifted above the bed, in the bucket, spilled outside tray and bucket."""
    p = np.asarray(sim.state(["soil"])["soil.points"])                  # (worlds, n, 3)
    m = float(np.asarray(sim.engine.model.particle_mass.numpy())[0])
    tx, ty, _ = s.tray_inner
    bx, by, bz = s.bucket_inner
    in_tray = (np.abs(p[..., 0]) <= tx / 2 + s.tray_wall) & (np.abs(p[..., 1]) <= ty / 2 + s.tray_wall)
    in_bucket = ((np.abs(p[..., 0] - s.bucket_x) <= bx / 2) & (np.abs(p[..., 1]) <= by / 2)
                 & (p[..., 2] <= s.bucket_wall + bz + 0.005))
    return {"lifted_kg": float((p[..., 2] > top + 0.01).sum(-1).mean() * m),
            "bucket_kg": float(in_bucket.sum(-1).mean() * m),
            "spill_kg": float((~in_tray & ~in_bucket).sum(-1).mean() * m),
            "total_kg": float(p.shape[1] * m)}


def run(s: DigSpec = DigSpec(), points: Path | None = None) -> dict[str, Any]:
    """The task; ``points``: an .npz of the soil points [worlds, n, 3] after each leg (diagnostics)."""
    t0 = time.time()
    sim = build(s)
    sim.settle(0.2)
    top = float(np.percentile(np.asarray(sim.state(["soil"])["soil.points"])[0][:, 2], 99))
    scoop, rows, snaps = _Scoop(sim, s), [], {}
    for name, v, w, seconds in _legs(s):
        scoop.move(v, w, seconds)
        rows.append({"leg": name, "t": round(float(sim.t), 3), **_masses(sim, s, top)})
        snaps[name.replace(" ", "_")] = np.asarray(sim.state(["soil"])["soil.points"])
        log.info("soil_dig %s: %s", name, rows[-1])
    out = {"spec": asdict(s), "bed_top": top, "legs": rows, "check_ok": bool(sim.check()["ok"]),
           "wall_s": round(time.time() - t0, 1)}
    out.update({k: rows[-1][k] for k in ("bucket_kg", "spill_kg", "total_kg")})
    out["dug_kg"] = next(r["lifted_kg"] for r in rows if r["leg"] == "lift")
    if points is not None:
        np.savez_compressed(points, **snaps)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.tasks.soil_dig", description=__doc__.split("\n\n")[0])
    ap.add_argument("--worlds", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bed", default=DigSpec.bed)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--points", type=Path, default=None, help="npz of the soil points after each leg")
    ap.add_argument("--roots", nargs="*", default=(), help="extra card roots (a custom bed card)")
    ap.add_argument("--tray", type=float, nargs=3, default=None, help="tray inner x y depth [m]")
    ap.add_argument("--bed-top", type=float, default=None, help="settled bed top [m]")
    ap.add_argument("--set", nargs="*", default=(), metavar="FIELD=VALUE", help="other DigSpec fields (numbers)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    spec = DigSpec(bed=args.bed, worlds=args.worlds, device=args.device, roots=tuple(args.roots))
    if args.tray is not None:
        spec = replace(spec, tray_inner=tuple(args.tray))
    if args.bed_top is not None:
        spec = replace(spec, bed_top=args.bed_top)
    for item in args.set:
        key, value = item.split("=", 1)
        spec = replace(spec, **{key: float(value)})
    res = run(spec, args.points)
    text = json.dumps(res, indent=1, default=float)
    if args.out is not None:
        args.out.write_text(text)
    print(json.dumps({k: res[k] for k in ("dug_kg", "bucket_kg", "spill_kg", "total_kg", "check_ok", "wall_s")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
