# SPDX-License-Identifier: Apache-2.0
"""Write an ARB v1 bundle from captured host frames and a Layout."""
from __future__ import annotations

import datetime
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .bundle import ARB_VERSION
from .chains import apply_pose, centreline
from .plan import Layout

log = logging.getLogger("alloy.render")


def _memmap(path: Path, shape: tuple) -> np.ndarray:
    return open_memmap(path, mode="w+", dtype=np.float32, shape=shape)


def _write_gathered(path: Path, frames: list[np.ndarray], rows: np.ndarray, off: np.ndarray) -> None:
    """Write [T, len(rows), C] float32, adding ``off`` to the position part."""
    out = _memmap(path, (len(frames), len(rows), frames[0].shape[1]))
    for k, f in enumerate(frames):
        out[k] = f[rows]
        out[k, :, :3] += off
    out.flush()


def _write_array(path: Path, arr: np.ndarray) -> None:
    out = _memmap(path, arr.shape)
    out[:] = arr
    out.flush()


def _pose_stack(bq: list[np.ndarray], rows: np.ndarray, off: np.ndarray) -> np.ndarray:
    poses = np.stack([f[rows] for f in bq]).astype(np.float64)
    poses[..., :3] += off
    return poses


def _git_commit(cwd: Path) -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _write_static(root: Path, lay: Layout) -> None:
    for mid, m in lay.store.meshes.items():
        extra = {"uv": m["uv"]} if m["uv"] is not None else {}
        np.savez(root / "meshes" / f"{mid}.npz", vertices=m["vertices"], faces=m["faces"], **extra)
    for s in lay.surfaces:
        np.savez(root / "surfaces" / f"{s.name}.npz", faces=s.faces, src=s.src, face_tag=s.face_tag, twins=s.twins)


def _write_frames(root: Path, lay: Layout, t: list[float], bq: list, pq: list, pqd: list) -> None:
    fr = root / "frames"
    np.save(fr / "t.npy", np.asarray(t, np.float64))
    if lay.bodies:
        _write_gathered(fr / "body_q.npy", bq, lay.body_rows, lay.cap_body_off[lay.body_rows])
    for s in lay.surfaces:
        rows = lay.p_lookup[s.src]
        _write_gathered(fr / f"surf_{s.name}.npy", pq, rows, lay.p_off[rows])
    for p in lay.points:
        rows = lay.p_lookup[p.ids]
        _write_gathered(fr / f"pts_{p.name}.npy", pq, rows, lay.p_off[rows])
        _write_gathered(fr / f"vel_{p.name}.npy", pqd, rows, np.zeros((len(rows), 3), np.float32))
    for c in lay.curves:
        _write_curve(fr, lay, c, bq, pq)
    for i in lay.instancers:
        _write_array(fr / f"inst_{i.name}.npy", _pose_stack(bq, i.rows, lay.cap_body_off[i.rows]))


def _write_curve(fr: Path, lay: Layout, c, bq: list, pq: list) -> None:
    if c.source == "particles":
        rows = lay.p_lookup[c.ids]
        _write_gathered(fr / f"curve_{c.name}.npy", pq, rows, lay.p_off[rows])
        return
    poses = _pose_stack(bq, c.rows, np.zeros((len(c.rows), 3)))
    pts = centreline(c.spec, poses) + lay.cap_body_off[c.rows[0]]
    _write_array(fr / f"curve_{c.name}.npy", pts)


def _scene(lay: Layout, fps: float, n: int, meta: dict, device: str, world_count: int, offsets: list) -> dict:
    created = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    full_meta = dict(meta, alloy_commit=_git_commit(Path(__file__).resolve().parent), created=created,
                     device=device, world_count=world_count)
    shapes = _shape_entries(lay)
    return {
        "arb": ARB_VERSION, "meta": full_meta, "frames": {"count": n, "fps": fps},
        "worlds": lay.worlds.worlds, "world_offsets": offsets, "hero_world": lay.worlds.hero,
        "bodies": lay.bodies, "shapes": shapes, "meshes": lay.store.scene_entries(),
        "surfaces": [{"name": s.name, "kind": s.kind, "world": s.world, "file": f"surfaces/{s.name}.npz",
                      "verts": len(s.src), "faces": len(s.faces), "tags": ["outer", "cut"]} for s in lay.surfaces],
        "points": [{"name": p.name, "world": p.world, "count": len(p.ids), "radius": p.radius} for p in lay.points],
        "curves": [{"name": c.name, "world": c.world, "count": c.count, "radius": c.radius, "from": c.source}
                   for c in lay.curves],
        "instancers": [_instancer_entry(i) for i in lay.instancers],
    }


def _shape_entries(lay: Layout) -> list[dict]:
    """Shape rows for scene.json; shapes of a body-chain curve carry ``curve`` so the renderer hides them."""
    curve_of = {int(r): c.name for c in lay.curves if c.source == "bodies" for r in c.rows}
    out = []
    for s in lay.shapes:
        entry = {k: v for k, v in s.items() if k != "model_index"}
        if entry["body"] in curve_of:
            entry["curve"] = curve_of[entry["body"]]
        out.append(entry)
    return out


def _instancer_entry(i) -> dict:
    g = i.geom
    entry = {"name": i.name, "world": i.world, "count": len(i.rows), "xform": g["xform"], "scale": g["scale"]}
    if g["type"] == "mesh":
        entry["mesh"] = g["mesh"]
    else:
        entry.update(shape_type=g["type"], size=g["size"])
    return entry


def write_bundle(root: Path, lay: Layout, fps: float, meta: dict, device: str, world_count: int,
                 t: list[float], bq: list, pq: list, pqd: list) -> None:
    """Create ``root`` (frames/, meshes/, surfaces/ and scene.json) from captured frames."""
    for sub in ("frames", "meshes", "surfaces"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    _write_static(root, lay)
    _write_frames(root, lay, t, bq, pq, pqd)
    offsets = [[float(x) for x in lay.worlds.offsets[w]] for w in lay.worlds.worlds]
    scene = _scene(lay, fps, len(t), meta, device, world_count, offsets)
    (root / "scene.json").write_text(json.dumps(scene, indent=1))
    log.info("wrote bundle %s: %d frames, %d shapes", root, len(t), len(lay.shapes))
