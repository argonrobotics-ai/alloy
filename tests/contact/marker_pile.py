# SPDX-License-Identifier: Apache-2.0
"""WP13's clutter pile (multi-object sweep, 2026-11-08; [internal note]), rigid
only: ``n`` objects dropped from a spawn grid into a static open bin, the same draws as the kit for the same seed.

40 % YCB cards (soup can, tuna can, pudding box, gelatin box, meat can, wood block, mug, large marker, padlock), 20 %
boxes, 20 % cylinders, 20 % random convex meshes. Each start is jittered by up to 4 mm per world. Objects fall up to
0.55 m (3.3 m/s at the floor). The YCB meshes come from ``alloy fetch`` (~/.cache/alloy/imports)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

import alloy

YCB = ("ycb_005_tomato_soup_can", "ycb_007_tuna_fish_can", "ycb_008_pudding_box", "ycb_009_gelatin_box",
       "ycb_010_potted_meat_can", "ycb_036_wood_block", "ycb_025_mug", "ycb_040_large_marker", "ycb_038_padlock")
KINDS = ("ycb", "box", "cylinder", "convex", "ycb")
MESHES = Path.home() / ".cache/alloy/imports/ycb_gso/meshes"
SPACING, MARGIN, LAYERS, WALL, CLEAR = 0.13, 0.08, 6, 0.01, 0.005


@dataclass
class Spec:
    name: str
    kind: str
    centre: np.ndarray            # [W, 3]
    half: np.ndarray              # [3] AABB half extents
    args: dict = field(default_factory=dict)


def fetched() -> bool:
    return all((MESHES / c / "physics.obj").exists() for c in YCB)


def kind(name: str) -> str:
    i = int(name[1:])
    return YCB[i % len(YCB)] if KINDS[i % len(KINDS)] == "ycb" else KINDS[i % len(KINDS)]


def _verts(card: str) -> np.ndarray:
    return np.array([[float(x) for x in ln.split()[1:4]] for ln in open(MESHES / card / "physics.obj")
                     if ln.startswith("v ")])


def _convex(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    axes = rng.uniform(0.018, 0.03, 3)
    p = rng.normal(size=(20, 3))
    p = p / np.linalg.norm(p, axis=1, keepdims=True) * axes
    faces, c = ConvexHull(p).simplices.copy(), p.mean(axis=0)
    for f in faces:
        if np.dot(np.cross(p[f[1]] - p[f[0]], p[f[2]] - p[f[0]]), p[f[0]] - c) < 0:
            f[1], f[2] = f[2], f[1]
    used = np.unique(faces)
    return p[used], np.vectorize({int(v): i for i, v in enumerate(used)}.get)(faces)


def _spec(i: int, k: str, pos: np.ndarray, rng: np.random.Generator) -> Spec:
    name = f"o{i:03d}"
    if k == "ycb":
        card = YCB[i % len(YCB)]
        v = _verts(card)
        return Spec(name, k, pos, 0.5 * (v.max(0) - v.min(0)), {"card": card, "half_z": 0.5 * float(np.ptp(v[:, 2]))})
    if k == "box":
        size = tuple(rng.uniform(0.03, 0.06, 3))
        return Spec(name, k, pos, 0.5 * np.asarray(size), {"size": size, "mass": float(rng.uniform(0.05, 0.2))})
    if k == "cylinder":
        r, h = float(rng.uniform(0.015, 0.03)), float(rng.uniform(0.03, 0.08))
        return Spec(name, k, pos, np.array([r, r, 0.5 * h]), {"r": r, "h": h, "mass": float(rng.uniform(0.05, 0.2))})
    v, f = _convex(rng)
    mid = 0.5 * (v.max(axis=0) + v.min(axis=0))
    return Spec(name, k, pos + mid, 0.5 * (v.max(0) - v.min(0)), {"v": v, "f": f, "mid": mid,
                                                                   "mass": float(rng.uniform(0.05, 0.2))})


def _deoverlap(specs: list[Spec]) -> None:
    """Lift objects (bottom first) until every pair is CLEAR apart in every world and all sit CLEAR above the floor."""
    placed: list[Spec] = []
    for s in sorted(specs, key=lambda q: float(q.centre[0, 2] - q.half[2])):
        need = WALL + CLEAR
        for o in placed:
            near = np.abs(s.centre[:, :2] - o.centre[:, :2]) < (s.half[:2] + o.half[:2] + CLEAR)
            if np.any(near.all(axis=1)):
                need = max(need, float((o.centre[:, 2] + o.half[2]).max()) + CLEAR)
        lift = need - float((s.centre[:, 2] - s.half[2]).min())
        if lift > 0.0:
            s.centre = s.centre + np.array([0.0, 0.0, lift])
        placed.append(s)


def _add(scene: alloy.Scene, s: Spec) -> None:
    a = s.args
    if s.kind == "ycb":
        scene.add_object(s.name, a["card"], pos=s.centre - np.array([0.0, 0.0, a["half_z"]]), anchor="base")
    elif s.kind == "box":
        scene.add_box(s.name, size=a["size"], pos=s.centre, mass=a["mass"])
    elif s.kind == "cylinder":
        scene.add_cylinder(s.name, a["r"], a["h"], pos=s.centre, mass=a["mass"])
    else:
        scene.add_mesh(s.name, vertices=a["v"], faces=a["f"], collision="convex", pos=s.centre - a["mid"],
                       mass=a["mass"])


def _rotations(quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = (quat_xyzw[:, k] for k in range(4))
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


class PileWatch:
    """The pile law's "clean" (inertia_floor.CLEAN_SPEED / ENERGY_TOL; ruling 2026-11-08 13:36Z), read once per
    period: each world's fastest body and its mechanical energy (translational + rotational KE about the COM, the
    floor's armature included, + m g z of the COM) against its start; at the end, INVALID worlds and the run's fastest
    spin against SPIN's limit. Example: ``w = PileWatch(sim, names)``, then ``sim.step(1); w.update()`` per period,
    then ``w.verdict()["clean"]``."""

    def __init__(self, sim: Any, names: list[str]) -> None:
        self.sim, self.names = sim, names
        m = sim.engine.model
        self.mass, self.world = m.body_mass.numpy().astype(float), m.body_world.numpy()
        self.inertia, self.com = m.body_inertia.numpy().astype(float), m.body_com.numpy().astype(float)
        self.arm = np.zeros(len(self.mass))
        from alloy.api.names import NameIndex  # noqa: PLC0415

        ix = NameIndex.from_built(sim.engine.built)
        for name, (before, floor) in sim.built.report.inertia_floor.items():
            for w in range(sim.worlds):
                self.arm[w * ix.stride["body"] + ix.bodies[name]] = floor - before
        self.dynamic = self.mass > 0.0
        self.e0 = self.energy()
        self.fastest, self.gain = np.zeros(sim.worlds), np.full(sim.worlds, -np.inf)

    def energy(self) -> np.ndarray:
        """[W] each world's mechanical energy (J)."""
        st = self.sim.engine.state
        q, qd = st.body_q.numpy().astype(float), st.body_qd.numpy().astype(float)
        rot = _rotations(q[:, 3:7])
        z = q[:, 2] + np.einsum("bij,bj->bi", rot, self.com)[:, 2]
        wb = np.einsum("bji,bj->bi", rot, qd[:, 3:])
        e = (0.5 * self.mass * (qd[:, :3] ** 2).sum(1) + 0.5 * np.einsum("bi,bij,bj->b", wb, self.inertia, wb)
             + 0.5 * self.arm * (wb ** 2).sum(1) + self.mass * 9.81 * z)
        return np.array([e[self.dynamic & (self.world == w)].sum() for w in range(self.sim.worlds)])

    def update(self) -> None:
        """Read the period just stepped."""
        st = self.sim.state(self.names)
        speed = np.stack([np.linalg.norm(np.asarray(st[f"{n}.vel"])[..., :3], axis=-1) for n in self.names], -1)
        self.fastest = np.maximum(self.fastest, np.nan_to_num(speed, nan=np.inf).max(-1))
        self.gain = np.maximum(self.gain, (self.energy() - self.e0) / np.abs(self.e0))

    def verdict(self) -> dict:
        """The run's numbers and whether it is clean."""
        from alloy.api import health as H  # noqa: PLC0415
        from alloy.contact.inertia_floor import CLEAN_SPEED, ENERGY_TOL  # noqa: PLC0415

        rep = H.poll(self.sim.engine)
        invalid = [w["severity"] for w in self.sim.health()["worlds"]].count("INVALID")
        spin = float(np.max(rep.spin_rad)) if rep is not None and rep.spin_rad is not None else 0.0
        limit = float(rep.spin_limit) if rep is not None else 0.5
        out = {"invalid": invalid, "fastest_m_s": np.round(self.fastest, 2).tolist(),
               "energy_gain": np.round(self.gain, 4).tolist(), "spin_rad_per_step": round(spin, 3)}
        out["clean"] = bool(invalid == 0 and self.fastest.max() <= CLEAN_SPEED and self.gain.max() <= ENERGY_TOL
                            and spin <= limit)
        return out


def pile(n: int = 100, worlds: int = 8, seed: int = 0, dt: float = 0.01) -> tuple[alloy.Scene, list[str]]:
    """The pile's scene and its object names."""
    rng = np.random.default_rng(seed)
    scene = alloy.Scene(worlds=worlds, dt=dt)
    scene.add_ground()
    per = max(4, math.ceil(n / LAYERS))
    nx = max(2, math.ceil(math.sqrt(per * 4.0 / 3.0)))
    ny = max(2, math.ceil(per / nx))
    inner = ((nx - 1) * SPACING + 2 * MARGIN, (ny - 1) * SPACING + 2 * MARGIN, 0.20 if n <= 50 else 0.30)
    scene.add_open_box("bin", inner=inner, wall=WALL, pos=(0.0, 0.0, 0.0))
    gx = max(1, round((inner[0] - 2 * MARGIN) / SPACING) + 1)
    gy = max(1, round((inner[1] - 2 * MARGIN) / SPACING) + 1)
    xs, ys = (np.arange(gx) - 0.5 * (gx - 1)) * SPACING, (np.arange(gy) - 0.5 * (gy - 1)) * SPACING
    pts = np.asarray([(x, y, 0.11 + k * SPACING) for k in range(math.ceil(n / (gx * gy))) for x in xs for y in ys][:n])
    order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]) - 10.0 * pts[:, 2])
    jitter = lambda b: b[None, :] + rng.uniform(-0.004, 0.004, (worlds, 3)) * np.array([1.0, 1.0, 0.0])  # noqa: E731
    specs = [_spec(i, KINDS[i % len(KINDS)], jitter(pts[order[i]]), rng) for i in range(n)]
    _deoverlap(specs)
    for s in specs:
        _add(scene, s)
    return scene, [s.name for s in specs]
