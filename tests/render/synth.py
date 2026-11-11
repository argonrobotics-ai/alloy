# SPDX-License-Identifier: Apache-2.0
"""Synthetic ARB v1 bundle and look file covering every entity type (numpy only).

    python tests/render/synth.py OUT_DIR [--frames 24] [--fps 120] [--scene glass]

writes OUT_DIR/bundle (scene.json, frames/, meshes/, surfaces/) and OUT_DIR/look.json. The scene is a tabletop:
a static plane and block, a fast box (motion blur), a tumbling ball, a spinning hex nut mesh (focus target), a
draped towel with an intact twinned seam, a soft loaf that splits along a pre-cut (cut faces tagged 1), a honey
stream of particles with velocities, a capsule-chain cable, a 2,000-grain instancer, a 600-point grain set, and
two replica worlds on a grid. Positions include each world's offset, as the recorder writes them.
--scene glass writes a glass of water instead: OUT_DIR/bundle plus look_on.json / look_off.json (liquid contact).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from tests.render import synth_models
except ImportError:  # run as a script
    import synth_models

FPS, FRAMES = 120.0, 24
GRID_M = 0.35
IDENT_Q = [0.0, 0.0, 0.0, 1.0]
WORLD_OFFSETS = [[0.0, 0.0, 0.0], [GRID_M, 0.0, 0.0], [0.0, GRID_M, 0.0]]


def grid_offsets(n: int, spacing: float = GRID_M) -> list[list[float]]:
    """World offsets on a square grid (the recorder's layout for grid_m); n = 3 keeps the default three."""
    if n == 3 and spacing == GRID_M:
        return WORLD_OFFSETS
    side = max(1, math.ceil(math.sqrt(n)))
    return [[(i % side) * spacing, (i // side) * spacing, 0.0] for i in range(n)]


def quat_axis_angle(axis, angle: np.ndarray) -> np.ndarray:
    """xyzw quaternions for rotations by `angle` (array) about a fixed or per-row axis."""
    axis = np.asarray(axis, np.float64)
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = np.asarray(angle, np.float64)[..., None] / 2.0
    return np.concatenate([axis * np.sin(half), np.cos(half)], axis=-1)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def hex_nut(across_flats: float = 0.013, hole_r: float = 0.0042, height: float = 0.0065, n: int = 48):
    """Watertight hex nut (outer hexagon sampled at the hole's angles), triangles, origin at its centre."""
    ang = np.arange(n) * 2 * math.pi / n
    r_hex = (across_flats / 2) / np.cos((ang % (math.pi / 3)) - math.pi / 6)
    rings = []
    for z in (height / 2, -height / 2):
        rings.append(np.stack([r_hex * np.cos(ang), r_hex * np.sin(ang), np.full(n, z)], 1))
        rings.append(np.stack([hole_r * np.cos(ang), hole_r * np.sin(ang), np.full(n, z)], 1))
    verts = np.concatenate(rings).astype(np.float32)
    ot, it, ob, ib = (np.arange(n) + k * n for k in range(4))
    nx = lambda a: np.roll(a, -1)  # noqa: E731 - next index around a ring
    quads = [np.stack([ot, it, nx(it), nx(ot)], 1), np.stack([ob, nx(ob), nx(ib), ib], 1),
             np.stack([ot, nx(ot), nx(ob), ob], 1), np.stack([it, ib, nx(ib), nx(it)], 1)]
    q = np.concatenate(quads)
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]).astype(np.int32)
    return verts, faces


def grid_tris(nx: int, ny: int) -> np.ndarray:
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (i * ny + j).ravel()
    b, c, d = a + ny, a + ny + 1, a + 1
    return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int32)


def towel(t: np.ndarray, nx: int = 41, ny: int = 31, seam: int = 20):
    """Draped sheet with a travelling fold; column `seam` is duplicated for the right half (an intact seam)."""
    x = np.linspace(-0.07, 0.07, nx)
    y = np.linspace(-0.05, 0.05, ny)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    faces = grid_tris(nx, ny)
    dup = np.arange(seam * ny, (seam + 1) * ny)
    new_ids = np.arange(nx * ny, nx * ny + ny)
    remap = np.arange(nx * ny)
    right = (faces // ny > seam).any(axis=1)
    for old, new in zip(dup, new_ids):
        remap[old] = new
    faces[right] = np.where(np.isin(faces[right], dup), remap[faces[right]], faces[right])
    pos = np.empty((len(t), nx * ny + ny, 3), np.float32)
    for k, tk in enumerate(t):
        fold = 0.012 * np.exp(-((gx - (-0.05 + 0.4 * tk)) / 0.014) ** 2) * (0.6 + 0.4 * np.sin(gy * 40))
        z = 0.0012 + fold + 0.0015 * np.sin(gx * 90 + gy * 55)
        p = np.stack([gx - 0.04, gy + 0.13, z], -1).reshape(-1, 3)
        pos[k] = np.concatenate([p, p[dup]])
    return pos, faces, np.stack([dup, new_ids], 1).astype(np.int32)


def box_shell(lo, hi, step: float):
    """Closed triangulated box surface (outward faces) with vertices welded."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    n = np.maximum(np.ceil((hi - lo) / step).astype(int), 1) + 1
    verts, faces = [], []
    for ax in range(3):
        u, v = [a for a in range(3) if a != ax]
        for side, val in ((0, lo[ax]), (1, hi[ax])):
            gu, gv = np.meshgrid(np.linspace(lo[u], hi[u], n[u]), np.linspace(lo[v], hi[v], n[v]), indexing="ij")
            p = np.zeros(gu.shape + (3,))
            p[..., u], p[..., v], p[..., ax] = gu, gv, val
            f = grid_tris(n[u], n[v]) + sum(len(x) for x in verts)
            outward = (np.cross(np.eye(3)[u], np.eye(3)[v])[ax] > 0) == (side == 1)
            faces.append(f if outward else f[:, ::-1])
            verts.append(p.reshape(-1, 3))
    verts, faces = np.concatenate(verts), np.concatenate(faces)
    key = np.round(verts * 1e7).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return verts[first], inv.reshape(-1)[faces]


def _ramp(t: float, t0: float, t1: float) -> float:
    return min(max((t - t0) / (t1 - t0), 0.0), 1.0)


def loaf_offsets(t: float) -> list[float]:
    """X shift of each of the 4 loaf pieces: seam 0|1 stays intact; seam 1|2 opens 3 mm then closes again
    (it must not re-weld); seam 2|3 opens 8 mm and stays open."""
    d2 = 0.003 * (_ramp(t, 0.04, 0.08) - _ramp(t, 0.12, 0.16))
    return [0.0, 0.0, d2, d2 + 0.008 * _ramp(t, 0.07, 0.15)]


def _twin_pairs(a: np.ndarray, b: np.ndarray, base_b: int, x: float) -> list[list[int]]:
    """Index pairs (a-local, base_b + b-local) of coincident vertices on the plane X = x."""
    key = {tuple(np.round(b[j] * 1e7).astype(np.int64)): j for j in np.flatnonzero(np.isclose(b[:, 0], x))}
    on_plane = np.flatnonzero(np.isclose(a[:, 0], x))
    return [[i, base_b + key[tuple(np.round(a[i] * 1e7).astype(np.int64))]] for i in on_plane]


def loaf(t: np.ndarray, centre=(0.045, 0.14, 0.0), size=(0.07, 0.045, 0.04)):
    """A block pre-cut into 4 pieces along X (one surface); twins on the 3 cut planes; cut faces tag 1."""
    hx, hy, hz = np.asarray(size) / 2
    cuts = np.linspace(-hx, hx, 5)
    pieces = [box_shell([cuts[i], -hy, 0], [cuts[i + 1], hy, 2 * hz], 0.004) for i in range(4)]
    base = np.cumsum([0] + [len(v) for v, _ in pieces])
    verts = np.concatenate([v for v, _ in pieces])
    faces = np.concatenate([f + base[i] for i, (_, f) in enumerate(pieces)]).astype(np.int32)
    twins = []
    for i in range(3):
        pairs = _twin_pairs(pieces[i][0], pieces[i + 1][0], int(base[i + 1]), cuts[i + 1])
        twins += [[a + int(base[i]), b] for a, b in pairs]
    twins = np.asarray(twins, np.int32)
    piece_of = np.repeat(np.arange(4), np.diff(base))
    pos = np.empty((len(t), len(verts), 3), np.float32)
    for k, tk in enumerate(t):
        p = verts.copy()
        p[:, 2] *= 1.0 - 0.02 * math.sin(tk * 40.0)
        p[:, 0] += np.asarray(loaf_offsets(tk))[piece_of]
        pos[k] = p + np.asarray(centre)
    twinned = np.zeros(len(verts), bool)
    twinned[twins.ravel()] = True
    tags = twinned[faces].all(axis=1).astype(np.uint8)
    return pos, faces, twins, tags


def honey(t: np.ndarray, n_stream: int = 4000, n_pool: int = 100000, centre=(0.022, -0.012), h0: float = 0.16):
    """A falling stream (positions and velocities from a phase that advances with time) and a pool on the table."""
    rng = np.random.default_rng(3)
    phase, ang, rad = rng.random(n_stream), rng.random(n_stream) * 2 * math.pi, np.sqrt(rng.random(n_stream))
    pr, pa, pz = _pool_lattice(n_pool, 0.017, 0.0009, rng)
    rate = 2.4  # stream phase per second
    n_pool = len(pr)
    pos = np.empty((len(t), n_stream + n_pool, 3), np.float32)
    vel = np.zeros_like(pos)
    for k, tk in enumerate(t):
        u = (phase + tk * rate) % 1.0
        r = 0.0013 * (1.0 - 0.45 * u) * rad
        pos[k, :n_stream] = np.stack([centre[0] + r * np.cos(ang), centre[1] + r * np.sin(ang),
                                      h0 * (1 - u ** 2) + 0.001], 1)
        vel[k, :n_stream, 2] = -2 * h0 * u * rate
        swirl = pa + 0.3 * tk
        pos[k, n_stream:] = np.stack([centre[0] + pr * np.cos(swirl), centre[1] + pr * np.sin(swirl), pz], 1)
        vel[k, n_stream:, 0] = -0.3 * pr * np.sin(swirl)
        vel[k, n_stream:, 1] = 0.3 * pr * np.cos(swirl)
    return pos, vel


def _pool_lattice(n: int, radius: float, spacing: float, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SPH-like pool: jittered lattice layers in a disc, thinning toward the rim (polar r, angle, z)."""
    g = np.arange(-radius, radius + spacing, spacing)
    gx, gy = np.meshgrid(g, g)
    r = np.hypot(gx, gy).ravel()
    keep = r < radius
    x, y, r = gx.ravel()[keep], gy.ravel()[keep], r[keep]
    top = 0.0006 + 0.0022 * np.sqrt(np.clip(1 - (r / radius) ** 2, 0, 1))  # settled dome, smooth top
    pts = []
    for i in range(len(x)):
        layers = max(1, int(np.ceil(top[i] / spacing)))
        pts += [(x[i], y[i], top[i] * (k + 1) / layers) for k in range(layers)]
    p = np.asarray(pts[:n]) + rng.normal(0, spacing * 0.05, (min(n, len(pts)), 3))
    return np.hypot(p[:, 0], p[:, 1]), np.arctan2(p[:, 1], p[:, 0]), p[:, 2]


def cable(t: np.ndarray, k: int = 24):
    x = np.linspace(-0.1, 0.12, k)
    out = np.empty((len(t), k, 3), np.float32)
    for i, tk in enumerate(t):
        out[i] = np.stack([x, 0.21 + 0.018 * np.sin(2 * math.pi * x / 0.15 + 6 * tk), np.full(k, 0.0035)], 1)
    return out


def grains(t: np.ndarray, n: int, centre, radius: float, height: float, seed: int):
    """Mound of lying grains: positions settle over the first frames; returns [T, n, 7] poses (xyzw)."""
    rng = np.random.default_rng(seed)
    r, a = radius * np.sqrt(rng.random(n)), rng.random(n) * 2 * math.pi
    z = 0.0011 + height * (1 - (r / radius) ** 2) * rng.random(n)
    yaw = quat_axis_angle([0, 0, 1], rng.random(n) * 2 * math.pi)
    tilt_axis = np.stack([np.cos(a), np.sin(a), np.zeros(n)], 1)
    lie = quat_axis_angle(tilt_axis, math.pi / 2 + 0.3 * (rng.random(n) - 0.5))
    q = quat_mul(yaw, lie)
    out = np.empty((len(t), n, 7), np.float32)
    for k in range(len(t)):
        drop = 0.003 * max(0.0, 1.0 - k / 10.0)
        out[k, :, :3] = np.stack([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a), z + drop], 1)
        out[k, :, 3:] = q
    return out


def body_poses(t: np.ndarray) -> dict[str, np.ndarray]:
    """[T, 7] poses of the hero world's bodies."""
    nt = len(t)
    nut = np.zeros((nt, 7))
    nut[:, :3] = [0.0, 0.0, 0.00325]
    nut[:, 3:] = quat_axis_angle([0, 0, 1], 1.5 * t)
    box = np.zeros((nt, 7))
    box[:, :3] = np.stack([-0.15 + 1.5 * t, np.full(nt, -0.045), np.full(nt, 0.01)], 1)
    box[:, 3:] = quat_axis_angle([0, 0, 1], 0.4 + 0 * t)
    ball = np.zeros((nt, 7))
    ball[:, :3] = np.stack([np.full(nt, 0.036), np.full(nt, 0.05), 0.012 + 0.012 * np.abs(np.sin(15 * t))], 1)
    ball[:, 3:] = quat_axis_angle([1, 0.3, 0.2], 25.0 * t)
    return {"nut": nut, "box_fast": box, "ball": ball}


def rigid(t: np.ndarray, nut_mesh_id: str, offsets: list):
    """bodies, shapes and body_q for three worlds (world 0 hero, worlds 1-2 replicas)."""
    poses = body_poses(t)
    bodies, shapes, cols = [], [], []
    shapes.append(_shape(0, "table", -1, -1, "plane", [4.0, 4.0], [0, 0, 0] + IDENT_Q, [0.6, 0.5, 0.4]))
    shapes.append(_shape(1, "cable_seg_00", -1, -1, "capsule", [0.0035, 0.004], [0, 0.21, 0.0035] + IDENT_Q,
                         [0.1, 0.1, 0.1], curve="cable"))
    shapes.append(_shape(2, "grey_card", -1, 0, "box", [0.012, 0.0005, 0.009], [0.014, 0.034, 0.009] + IDENT_Q,
                         [0.46, 0.46, 0.46]))
    heroes: dict[str, int] = {}
    for w, off in enumerate(offsets):
        for name, geo in (("nut", ("mesh", None)), ("box_fast", ("box", [0.01] * 3)), ("ball", ("sphere", [0.012]))):
            bid = len(bodies)
            bodies.append({"id": bid, "label": f"{name}_body", "world": w})
            pose = poses[name].copy()
            pose[:, :3] += off
            cols.append(pose)
            sid = len(shapes)
            typ, size = geo
            s = _shape(sid, name, bid, w, typ, size or [1, 1, 1], [0, 0, 0] + IDENT_Q, [0.8, 0.2, 0.2])
            s["mesh"] = nut_mesh_id if typ == "mesh" else None
            s["replica_of"] = heroes.get(name) if w else None
            heroes.setdefault(name, sid)
            shapes.append(s)
        blk = [-0.042 + off[0], 0.045 + off[1], 0.012] + list(quat_axis_angle([0, 0, 1], np.array(0.35)))
        s = _shape(len(shapes), "block", -1, w, "box", [0.012] * 3, blk, [0.1, 0.1, 0.1])
        s["replica_of"] = heroes.get("block") if w else None
        heroes.setdefault("block", s["id"])
        shapes.append(s)
    return bodies, shapes, np.stack(cols, 1).astype(np.float32)


STRIP_N, STRIP_L, STRIP_W, STRIP_T = 10, 0.008, 0.02, 0.001
STRIP_BASE = np.array([0.08, 0.06, STRIP_T / 2])
STRIP_BENDS_DEG = [10, 10, 10, 10, 90, 10, 10, 10, 10]  # joint 4 is a sharp 90 degree hinge


def strip_poses(t: np.ndarray) -> np.ndarray:
    """[T, 10, 7] poses of a hinged strip of boxes curling up about -Y (forward kinematics, segment centres)."""
    out = np.zeros((len(t), STRIP_N, 7))
    half = np.array([STRIP_L / 2, 0.0, 0.0])
    for k, tk in enumerate(t):
        ramp = min(tk / 0.15, 1.0)
        q = np.array(IDENT_Q, float)
        p = STRIP_BASE + half
        for i in range(STRIP_N):
            if i:
                joint = p + _rot(q, half)
                q = quat_mul(q, quat_axis_angle([0, -1, 0], np.array(math.radians(STRIP_BENDS_DEG[i - 1]) * ramp)))
                p = joint + _rot(q, half)
            out[k, i, :3], out[k, i, 3:] = p, q
    return out


def _rot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[:3], q[3]
    tt = 2.0 * np.cross(u, v)
    return v + w * tt + np.cross(u, tt)


def write_strip_skin(root: Path) -> None:
    """Rest-pose OBJ (flat mid-plane sheet, 1 mm grid) and per-vertex weights blending neighbouring segments."""
    (root / "extra").mkdir(exist_ok=True)
    xs = np.linspace(0.0, STRIP_N * STRIP_L, 81)
    ys = np.linspace(-STRIP_W / 2, STRIP_W / 2, 11)
    lines = [f"v {STRIP_BASE[0] + x:.6f} {STRIP_BASE[1] + y:.6f} {STRIP_BASE[2]:.6f}" for x in xs for y in ys]
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            a = i * len(ys) + j + 1
            lines.append(f"f {a} {a + len(ys)} {a + len(ys) + 1} {a + 1}")
    (root / "extra" / "strip_rest.obj").write_text("\n".join(lines) + "\n")
    rows = []
    for x in xs:
        s = x / STRIP_L - 0.5
        i0 = int(np.clip(np.floor(s), 0, STRIP_N - 2))
        f = float(np.clip(s - i0, 0.0, 1.0))
        rows += [[[i0, 1.0 - f], [i0 + 1, f]]] * len(ys)
    labels = [f"strip_{i:02d}" for i in range(STRIP_N)]
    (root / "extra" / "strip_weights.json").write_text(json.dumps({"bodies": labels, "weights": rows}))


def _shape(sid, label, body, world, typ, size, xform, color, **extra) -> dict:
    return {"id": sid, "label": label, "body": body, "world": world, "type": typ, "size": [float(x) for x in size],
            "xform": [float(x) for x in xform], "scale": [1.0, 1.0, 1.0], "mesh": None, "color": color,
            "opacity": 1.0, "replica_of": None, **extra}


def write_synth(root: Path, frames: int = FRAMES, fps: float = FPS, worlds: int = 3) -> dict:
    """Write the bundle into root; returns scene.json as a dict."""
    for sub in ("frames", "meshes", "surfaces"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    t = np.arange(frames) / fps
    np.save(root / "frames" / "t.npy", t.astype(np.float64))
    nv, nf = hex_nut()
    np.savez(root / "meshes" / "nut_m8.npz", vertices=nv, faces=nf)
    offsets = grid_offsets(worlds)
    bodies, shapes, body_q = rigid(t, "nut_m8", offsets)
    strip = strip_poses(t)
    for i in range(STRIP_N):
        bodies.append({"id": len(bodies), "label": f"strip_{i:02d}", "world": 0})
        shapes.append(_shape(len(shapes), f"strip_seg_{i:02d}", len(bodies) - 1, 0, "box",
                             [STRIP_L / 2, STRIP_W / 2, STRIP_T / 2], [0, 0, 0] + IDENT_Q, [0.9, 0.9, 0.9]))
    body_q = np.concatenate([body_q, strip.astype(np.float32)], axis=1)
    body_q = synth_models.add_rj45(root, t, bodies, shapes, body_q, _shape)
    np.save(root / "frames" / "body_q.npy", body_q)
    write_strip_skin(root)
    surfaces = [_surface(root, "towel", "cloth", *towel(t), None), _surface(root, "loaf", "soft", *loaf(t))]
    hp, hv = honey(t)
    rp = grains(t, 600, (0.062, 0.02), 0.012, 0.006, seed=5)[..., :3]
    points = []
    for name, p, v, r in (("honey", hp, hv, 0.0006), ("rice_mpm", rp, np.zeros_like(rp), 0.0012)):
        np.save(root / "frames" / f"pts_{name}.npy", p.astype(np.float32))
        np.save(root / "frames" / f"vel_{name}.npy", v.astype(np.float32))
        points.append({"name": name, "world": 0, "count": int(p.shape[1]), "radius": r})
    np.save(root / "frames" / "curve_cable.npy", cable(t))
    rice = grains(t, 2000, (-0.04, 0.0), 0.024, 0.012, seed=4)
    np.save(root / "frames" / "inst_rice.npy", rice)
    scene = {
        "arb": 1, "meta": {"clip": "synth", "alloy_commit": "synthetic", "created": "synthetic", "device": "cpu",
                           "world_count": worlds},
        "frames": {"count": frames, "fps": fps}, "worlds": list(range(worlds)), "world_offsets": offsets,
        "hero_world": 0, "bodies": bodies, "shapes": shapes,
        "meshes": {"nut_m8": {"file": "meshes/nut_m8.npz", "verts": len(nv), "faces": len(nf), "uv": False}},
        "surfaces": surfaces, "points": points,
        "curves": [{"name": "cable", "world": 0, "count": 24, "radius": 0.0035, "from": "bodies"},
                   {"name": "rj45_cable", "world": 0, "count": 20, "radius": 0.003, "from": "bodies"}],
        "instancers": [{"name": "rice", "world": 0, "count": 2000, "shape_type": "capsule", "size": [0.0011, 0.0024],
                        "xform": [0, 0, 0] + IDENT_Q, "scale": [1.0, 1.0, 1.0]}],
    }
    (root / "scene.json").write_text(json.dumps(scene, indent=1))
    return scene


def _surface(root: Path, name: str, kind: str, pos, faces, twins, tags) -> dict:
    tags = np.zeros(len(faces), np.uint8) if tags is None else tags
    np.save(root / "frames" / f"surf_{name}.npy", pos.astype(np.float32))
    np.savez(root / "surfaces" / f"{name}.npz", faces=faces.astype(np.int32),
             src=np.arange(pos.shape[1], dtype=np.int32),
             face_tag=tags.astype(np.uint8), twins=twins.astype(np.int32))
    return {"name": name, "kind": kind, "world": 0, "file": f"surfaces/{name}.npz", "verts": int(pos.shape[1]),
            "faces": int(len(faces)), "tags": ["outer", "cut"]}


def _cameras() -> dict:
    """Vertical hero (85 mm f/2.8, focus on the nut), check cameras, and a wide horizontal."""
    return {
        "vertical": {"lens_mm": 85, "sensor_mm": 36, "fstop": 2.8, "focus": {"label": "^nut$"},
                     "keys": [{"t": 0.0, "eye": [0.06, -0.36, 0.15], "target": [0.0, 0.01, 0.018]},
                              {"t": 0.2, "eye": [0.057, -0.357, 0.149], "target": [0.0, 0.01, 0.018]}]},
        "detail": {"resolution": [1920, 1080], "lens_mm": 60, "fstop": 11, "focus": {"point": [0.0, 0.12, 0.0]},
                   "keys": [{"t": 0.0, "eye": [0.03, -0.12, 0.30], "target": [0.0, 0.11, 0.0]}]},
        "macro": {"resolution": [1080, 1080], "lens_mm": 100, "fstop": 8,
                  "focus": {"point": [0.022, -0.012, 0.003]},
                  "keys": [{"t": 0.0, "eye": [0.06, -0.17, 0.07], "target": [0.022, -0.012, 0.006]}]},
        "strip": {"resolution": [1080, 1080], "lens_mm": 60, "fstop": 11, "focus": {"point": [0.12, 0.06, 0.02]},
                  "keys": [{"t": 0.0, "eye": [0.13, -0.09, 0.08], "target": [0.12, 0.065, 0.018]}]},
        "card": {"resolution": [256, 256], "lens_mm": 100, "fstop": 0,
                 "keys": [{"t": 0.0, "eye": [0.014, -0.06, 0.012], "target": [0.014, 0.034, 0.009]}]},
        "loaf": {"resolution": [1080, 1080], "lens_mm": 70, "fstop": 8, "focus": {"point": [0.049, 0.12, 0.02]},
                 "keys": [{"t": 0.0, "eye": [0.13, 0.0, 0.09], "target": [0.048, 0.135, 0.018]}]},
        "horizontal": {"resolution": [1920, 1080], "lens_mm": 35, "sensor_mm": 36, "fstop": 5.6,
                       "focus": {"distance": 0.8},
                       "keys": [{"t": 0.0, "eye": [0.32, -0.62, 0.42], "target": [0.16, 0.14, 0.0]}]},
    }


def make_look(samples: int = 192, hdri: str = "polyhaven:photo_studio_loft_hall") -> dict:
    """Look with a vertical 85 mm f/2.8 close-up focused on the nut and a wide horizontal camera."""
    look = {
        "look": 1,
        "render": {"resolution": [1080, 1920], "samples": samples, "noise_threshold": 0.01, "denoiser": "OIDN",
                   "view": "AgX", "look": "AgX - Medium High Contrast", "exposure": 0.0,
                   "motion_blur": {"shutter_s": 1 / 60}, "bounces": {"total": 12, "transmission": 16, "volume": 2},
                   "caustics": {"reflective": False, "refractive": True}, "film_transparent": False},
        "frames": {"start": 0, "end": -1, "step": 1},
        "cameras": _cameras(),
        "world": {"hdri": hdri, "strength": 0.7, "rotation_deg": 40, "show_hdri": True,
                  "background_color": [0.03, 0.03, 0.035]},
        "lights": [
            {"type": "area", "eye": [-0.35, -0.3, 0.45], "target": [0, 0.03, 0], "size_m": 0.45, "watts": 7,
             "kelvin": 5200},
            {"type": "area", "eye": [0.3, 0.5, 0.3], "target": [0, 0.05, 0.02], "size_m": 0.3, "watts": 5,
             "kelvin": 6500},
            {"type": "area", "eye": [0.5, -0.3, 0.15], "target": [0, 0, 0.02], "size_m": 0.6, "watts": 1.5,
             "kelvin": 4300},
        ],
        "materials": {"table": {"preset": "wood_oak", "scale_m": 0.5}, "nut": {"preset": "zinc_plated"},
                      "red": {"preset": "plastic_gloss", "base_color": [0.7, 0.06, 0.04]},
                      "rice": {"preset": "rice_grain"}},
        "assign": [{"match": "^table$", "material": "table"}, {"match": "^grey_card$", "material": "grey_card_18"},
                   {"match": "^nut", "material": "nut"},
                   {"match": "^box_fast", "material": "red"}, {"match": "^ball", "material": "steel_polished"},
                   {"match": "^block", "material": "abs_black"}],
        "liquids": {"honey": {"voxel_m": 0.0002, "radius_scale": 1.8, "smooth": 2, "material": "honey"}},
        "points": {"rice_mpm": {"instance": "rice_grain", "scale": 1.0, "material": "rice"}},
        "instances": {"rice": {"material": "rice", "instance": "rice_grain"}},
        "props": [{"model": "polyhaven:potted_plant_01", "pose": [0.2, 0.45, 0.0, 0, 0, 0, 1], "scale": 1.0},
                  {"model": "polyhaven:wooden_bowl_01", "pose": [-0.17, 0.32, 0.0, 0, 0, 0.38, 0.92]},
                  {"model": "file:extra/strip_rest.obj", "pose": [0.0, 0.25, 0.0001, 0, 0, 0, 1],
                   "material": "plaster_paint"}],
        "bevels": [{"match": "^block$", "bevel_m": 0.0008, "segments": 3}, {"match": "^nut$", "bevel_m": 0.0003}],
        "skins": {"strip": {"mesh": "extra/strip_rest.obj", "weights": "extra/strip_weights.json", "rest_frame": 0,
                            "material": "pp_frosted", "thickness_m": 0.001, "subdivide": 1, "hide_bodies": True}},
        "curves": {"cable": {"material": "pvc_cable_black", "resolution": 4}},
        "surfaces": {"towel": {"subdivide": 1, "thickness_m": 0.002, "materials": ["cotton_terry"]},
                     "loaf": {"subdivide": 1, "materials": ["bread_crust", "bread_crumb"]}},
        "hide": ["_collision$"],
    }
    return synth_models.rj45_look(look)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out")
    p.add_argument("--frames", type=int, default=FRAMES)
    p.add_argument("--fps", type=float, default=FPS)
    p.add_argument("--samples", type=int, default=192)
    p.add_argument("--worlds", type=int, default=3, help="worlds on a grid (rigid replicas of world 0)")
    p.add_argument("--scene", choices=("tabletop", "glass"), default="tabletop")
    args = p.parse_args(argv)
    out = Path(args.out)
    if args.scene == "glass":
        synth_models.write_glass(out / "bundle")
        for tag in ("on", "off"):
            look = synth_models.glass_look(args.samples, contact=tag == "on")
            (out / f"look_{tag}.json").write_text(json.dumps(look, indent=1))
        return
    write_synth(out / "bundle", args.frames, args.fps, args.worlds)
    (out / "look.json").write_text(json.dumps(make_look(args.samples), indent=1))


if __name__ == "__main__":
    main()
