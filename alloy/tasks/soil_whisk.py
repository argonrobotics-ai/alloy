# SPDX-License-Identifier: Apache-2.0
"""Dry whisk: whisk baking powder, salt and sugar evenly through flour (WP12 soil item 4; Gym-Anything's
ingredient-mixing dry-whisk; docs/notes/soil-design.md).

Flour (a cohesive powder card) fills a round bowl on the implicit-MPM granular route. Spoonfuls of sugar, salt and
baking powder lie on its top as species labels (``alloy.plugins.species``, labels only: they move with the grains
and share the flour's mechanics, so nothing segregates). A whisk stirs through the public API (``alloy.Scene``,
``Sim.drive``). It is a paddle stirrer: a star of four kinematic steel plates (18 mm arms, 5 mm thick, 40 mm tall) in
vertical planes through the handle axis, the solid envelope of a balloon whisk's loops. A wire whisk lets the flour
pass between its wires; the plates plough it, and the flour mounds up the bowl wall (the 80 mm bowl holds it). Rods
in the wires' place do not fit the MPM resolution: the router sizes the voxel to a quarter of a partner's middle
extent, so 6 mm rods give a 1.5 mm voxel and 888,040 flour particles (over the 200k budget), where the plates give
4.5 mm and 25,155. The star spins about its own axis while that axis circles the bowl (a planetary stroke, as a hand
circling the bowl); every leg ramps its speed (no velocity steps) and the star keeps two voxels from the bowl wall.

Mixing is Lacey's index per species over cubic sampling cells: M = (s0^2 - s^2) / (s0^2 - sr^2), with s^2 the
cell-weighted variance of the species fraction, s0^2 = p (1 - p) fully segregated and sr^2 = s0^2 / n for a random
mixture of n particles per cell. M is 1 for a random mixture at that cell size; spoonfuls smaller than a cell
start above 0. The flour's occupied volume (voxels holding particles; ``packed``: at least half full) at the start and
the end separates ploughing and MPM dilation from mixing. Readouts that need the global particle numbering (the
species plug-in's ids) use the engine state (``sim.engine.state.particle_q``) and the route's voxel
(``sim.built.report``).

    python -m alloy.tasks.soil_whisk --seconds 4 --device cpu --out whisk.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("alloy.tasks")

__all__ = ["WhiskSpec", "build", "lacey", "main", "occupied_volume", "reach", "run", "star_arms"]

SPECIES = ("sugar", "salt", "baking_powder")


@dataclass(frozen=True)
class WhiskSpec:
    """Geometry [m], rates [rev/s] and the flour card of the task."""

    material: str = "flour_wheat"
    bowl_radius: float = 0.06
    bowl_depth: float = 0.08
    bowl_wall: float = 0.01                    # walls and floor >= 2 MPM voxels
    flour_radius: float = 0.056
    flour_height: float = 0.03
    arm: tuple[float, float, float, float] = (0.018, 0.005, 0.04, 0.003)   # length, thickness, height, inner radius
    floor_gap: float = 0.005                   # star bottom above the bowl floor
    orbit_radius: float = 0.025                # the star's axis circles the bowl centre at this radius
    orbit_hz: float = 0.5
    spin_hz: float = 2.0                       # the star's own turn rate (world frame)
    ramp_s: float = 0.2
    seconds: float = 4.0
    settle_s: float = 0.1
    dt: float = 0.01
    cell: float = 0.015                        # Lacey sampling cell
    report_s: float = 0.25
    # (x, y, half-width, depth) of each spoonful on the flour's top, in SPECIES order
    spoons: tuple[tuple[float, float, float, float], ...] = (
        (0.025, 0.0, 0.015, 0.010), (-0.022, 0.022, 0.006, 0.006), (-0.02, -0.025, 0.008, 0.008))
    worlds: int = 1
    device: str = "cpu"
    roots: tuple[str, ...] = ()


def star_arms(s: WhiskSpec, theta: float, centre: tuple[float, float]) -> dict[str, tuple]:
    """The four arm plates (size, pos, rpy) at spin angle ``theta`` about the axis at ``centre`` (x, y)."""
    length, thick, height, r0 = s.arm
    zc = s.bowl_wall + s.floor_gap + 0.5 * height
    out = {}
    for k in range(4):
        a = theta + 0.5 * math.pi * k
        r = r0 + 0.5 * length
        out[f"whisk_arm{k}"] = ((length, thick, height), (centre[0] + r * math.cos(a), centre[1] + r * math.sin(a), zc),
                                (0.0, 0.0, a))
    return out


def reach(s: WhiskSpec) -> float:
    """Farthest point of the star from the bowl's axis [m]."""
    return s.orbit_radius + s.arm[3] + math.hypot(s.arm[0], 0.5 * s.arm[1])


def spoon_regions(s: WhiskSpec) -> tuple[tuple[float, ...], ...]:
    """Species plug-in regions: each spoonful a box on the flour's top with concentration 1 of its species."""
    top = s.bowl_wall + s.flour_height
    out = []
    for k, (x, y, h, d) in enumerate(s.spoons):
        conc = [0.0] * len(SPECIES)
        conc[k] = 1.0
        out.append((x - h, y - h, top - d, x + h, y + h, top + 0.01, *conc))
    return tuple(out)


def build(s: WhiskSpec) -> tuple[Any, Any]:
    """(sim, species plug-in) of the scene in the module docstring, built on ``s.device``."""
    import alloy  # noqa: PLC0415
    from alloy.plugins.species import Species  # noqa: PLC0415

    if s.bowl_radius - reach(s) < 0.01:
        raise ValueError(f"the whisk reaches {reach(s):.3f} m: keep 10 mm (two voxels) from the {s.bowl_radius} m wall")
    scene = alloy.Scene(dt=s.dt, worlds=s.worlds, **({"roots": list(s.roots)} if s.roots else {}))
    scene.add_ground()
    scene.add_open_cylinder("bowl", inner_radius=s.bowl_radius, depth=s.bowl_depth, wall=s.bowl_wall,
                            material="steel_carbon")
    scene.add_cylinder("flour", s.flour_radius, s.flour_height,
                       pos=(0.0, 0.0, s.bowl_wall + 0.5 * s.flour_height + 0.0005), material=s.material)
    for name, (size, pos, rpy) in star_arms(s, 0.0, (s.orbit_radius, 0.0)).items():
        scene.add_box(name, size=size, pos=pos, rpy=rpy, kinematic=True, material="steel_carbon")
    sp = Species(species=SPECIES, diffusivity=(0.0,) * len(SPECIES), regions=spoon_regions(s))
    return alloy.build(scene, device=s.device, plugins=[sp]), sp


def lacey(x: np.ndarray, lab: np.ndarray, cell: float, min_count: int = 20) -> dict[str, float]:
    """Lacey's index per species (module docstring) of particles ``x`` (N, 3) with labels ``lab`` (N, S) in 0/1,
    over ``cell`` cubes holding at least ``min_count`` particles; ``cells`` is how many were used."""
    _, inv, counts = np.unique(np.floor(x / cell).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    keep = counts[inv] >= min_count
    _, c = np.unique(inv[keep], return_inverse=True)
    n_c = np.bincount(c).astype(np.float64)
    out: dict[str, float] = {"cells": float(len(n_c))}
    for k, name in enumerate(SPECIES):
        m = lab[keep, k].astype(np.float64)
        p = float(m.mean())
        var = float(np.sum(n_c * (np.bincount(c, weights=m) / n_c - p) ** 2) / n_c.sum())
        s0, sr = p * (1.0 - p), p * (1.0 - p) / float(n_c.mean())
        out[name] = (s0 - var) / (s0 - sr) if s0 > sr else float("nan")
    return out


def occupied_volume(x: np.ndarray, voxel: float, min_count: int = 1) -> float:
    """Volume [m^3] of the ``voxel`` cubes holding at least ``min_count`` of the points ``x`` (N, 3). The MPM fill puts
    8 particles in a voxel (spacing half a voxel): ``min_count`` 4 counts the voxels at least half packed."""
    _, counts = np.unique(np.floor(x / voxel).astype(np.int64), axis=0, return_counts=True)
    return float(np.count_nonzero(counts >= min_count)) * voxel ** 3


def _voxel(sim: Any) -> float:
    """The flour's MPM voxel from its route decision (the router's min(d_feature / 4, 5 mm))."""
    return float(sim.built.report.decisions["flour"].numbers["voxel"])


def _mixing(sim: Any, sp: Any, s: WhiskSpec) -> list[dict[str, float]]:
    """Per world: Lacey from the species labels and the particles' current positions, the flour's top (99th
    percentile height), its occupied volume and the fraction out of the bowl (beyond its wall or below its floor)."""
    conc, ids = sp.read()["conc"], sp.particles()
    q, voxel = sim.engine.state.particle_q.numpy(), _voxel(sim)
    out = []
    for w in range(ids.shape[0]):
        x = q[ids[w]]
        out_of_bowl = (np.hypot(x[:, 0], x[:, 1]) > s.bowl_radius + s.bowl_wall) | (x[:, 2] < s.bowl_wall - 0.002)
        out.append({**lacey(x, (conc[w] > 0.5).astype(np.float64), s.cell), "top_mm": float(np.percentile(x[:, 2], 99))
                    * 1e3, "occupied_ml": occupied_volume(x, voxel) * 1e6,
                    "packed_ml": occupied_volume(x, voxel, 4) * 1e6, "spill": float(out_of_bowl.mean())})
    return out


def _whisk(sim: Any, s: WhiskSpec, rows: list, sp: Any) -> None:
    """The planetary stroke, one drive per control period (chord velocity: the axis lands on its circle)."""
    dt, phi = float(sim.dt), 0.0
    centre = np.array([s.orbit_radius, 0.0, 0.0])
    n, every = int(round(s.seconds / dt)), max(1, int(round(s.report_s / dt)))
    for k in range(n):
        f = min(1.0, (k + 0.5) * dt / s.ramp_s, (n - k - 0.5) * dt / s.ramp_s)
        phi += 2.0 * math.pi * s.orbit_hz * f * dt
        nxt = np.array([s.orbit_radius * math.cos(phi), s.orbit_radius * math.sin(phi), 0.0])
        v, w = tuple((nxt - centre) / dt), (0.0, 0.0, 2.0 * math.pi * s.spin_hz * f)
        for name in star_arms(s, 0.0, (0.0, 0.0)):
            sim.drive(name, velocity=v, angular_velocity=w, pivot=tuple(centre), duration=dt)
        sim.step(1)
        centre = nxt
        if (k + 1) % every == 0 or k + 1 == n:
            rows.append(_row(sim, sp, s, round((k + 1) * dt, 4)))
            log.info("soil_whisk %s", rows[-1])


def _row(sim: Any, sp: Any, s: WhiskSpec, t: float) -> dict[str, Any]:
    """Lacey per species (lowest over worlds), the highest flour top and spill fraction over worlds."""
    mix = _mixing(sim, sp, s)
    return {"t": t, **{name: float(np.min([m[name] for m in mix])) for name in SPECIES},
            "top_mm": max(m["top_mm"] for m in mix), "occupied_ml": max(m["occupied_ml"] for m in mix),
            "packed_ml": max(m["packed_ml"] for m in mix),
            "spill": max(m["spill"] for m in mix)}


def run(s: WhiskSpec) -> dict[str, Any]:
    """Build, settle, whisk; the Lacey index per species (the lowest over worlds) at the start and every
    ``report_s``, the flour's top, occupied volume (start, end) and spill, label conservation and the health check."""
    t0 = time.perf_counter()
    sim, sp = build(s)
    rows = [_row(sim, sp, s, 0.0)]
    sim.step(int(round(s.settle_s / s.dt)))
    _whisk(sim, s, rows, sp)
    chk = sim.check()
    err = np.abs(sp.read()["conservation_error"]).max()
    end = rows[-1]
    return {"spec": asdict(s), "voxel": _voxel(sim), "rows": rows, "lacey_start": {k: rows[0][k] for k in SPECIES},
            "lacey_end": {k: end[k] for k in SPECIES}, "spill": end["spill"],
            "occupied_ml": (rows[0]["occupied_ml"], end["occupied_ml"]), "packed_ml": (rows[0]["packed_ml"],
                                                                                    end["packed_ml"]),
            "labels_conserved": bool(err == 0),
            "check_ok": bool(chk["ok"]), "problems": chk["problems"][:5], "wall_s": round(time.perf_counter() - t0, 1)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seconds", type=float, default=WhiskSpec.seconds)
    p.add_argument("--material", default=WhiskSpec.material)
    p.add_argument("--worlds", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--roots", nargs="*", default=())
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    s = replace(WhiskSpec(), seconds=a.seconds, material=a.material, worlds=a.worlds, device=a.device,
                roots=tuple(a.roots))
    res = run(s)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("voxel", "lacey_start", "lacey_end", "spill", "occupied_ml", "packed_ml",
                                          "labels_conserved", "check_ok", "wall_s")}))


if __name__ == "__main__":
    main()
