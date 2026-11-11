# SPDX-License-Identifier: Apache-2.0
"""Skinning math and files for look.skins (pure numpy, no bpy): OBJ and weight readers, dual-quaternion and
linear blend skinning of a rest mesh by body poses T_i(t) T_i(rest)^-1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import motion
import primitives
from io_bundle import Bundle
from look import LookError


def _obj_index(tok: str, count: int) -> int:
    i = int(tok)
    return i - 1 if i > 0 else count + i


class ObjMesh:
    """A Wavefront OBJ: vertices, polygons, per-face `usemtl` group, and per-corner UVs when every face has vt."""

    def __init__(self, path: Path):
        self.path = path
        verts, vts, corners, uv_corners, sizes, groups, face_group = [], [], [], [], [], [], []
        current = None
        for line in path.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "v":
                verts.append([float(x) for x in parts[1:4]])
            elif parts[0] == "vt":
                vts.append([float(x) for x in parts[1:3]])
            elif parts[0] == "usemtl":
                name = " ".join(parts[1:])
                if name not in groups:
                    groups.append(name)
                current = groups.index(name)
            elif parts[0] == "f":
                toks = [t.split("/") for t in parts[1:]]
                corners += [_obj_index(t[0], len(verts)) for t in toks]
                uv_corners += [_obj_index(t[1], len(vts)) if len(t) > 1 and t[1] else -1 for t in toks]
                sizes.append(len(toks))
                face_group.append(-1 if current is None else current)
        if not verts or not sizes:
            raise LookError(f"skins mesh {path}: no vertices or faces")
        self.verts = np.asarray(verts, np.float32)
        self.corners, self.sizes = np.asarray(corners, np.int32), np.asarray(sizes, np.int32)
        self.groups, self.face_group = groups, np.asarray(face_group, np.int32)
        uvc = np.asarray(uv_corners, np.int64)
        self.corner_uv = np.asarray(vts, np.float32)[uvc] if vts and (uvc >= 0).all() else None


def read_obj(path: Path) -> primitives.MeshArrays:
    """Vertices and polygons of a Wavefront OBJ (v / f lines; 'a/b/c' corner tokens; negative indices)."""
    m = ObjMesh(path)
    return m.verts, m.corners, m.sizes


def load_weights(path: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None]:
    """(body labels, idx [N,K], w [N,K], points [N,3] or None) from the JSON or .npz weights file.

    Rows are per mesh vertex, or, when `points` is given, per sample point (a mesh vertex takes the weights of its
    nearest sample, which suits imported .glb meshes whose vertex order the author does not control)."""
    if path.suffix == ".npz":
        with np.load(path) as z:
            labels, idx, w = [str(s) for s in z["labels"]], np.asarray(z["idx"], np.int64), np.asarray(z["w"], float)
            points = np.asarray(z["points"], float) if "points" in z.files else None
        return labels, idx, w, points
    raw = json.loads(path.read_text())
    labels, rows = list(raw["bodies"]), raw["weights"]
    k = max(len(r) for r in rows)
    idx, w = np.zeros((len(rows), k), np.int64), np.zeros((len(rows), k))
    for v, row in enumerate(rows):
        for j, (b, wt) in enumerate(row):
            idx[v, j], w[v, j] = int(b), float(wt)
    return labels, idx, w, np.asarray(raw["points"], float) if "points" in raw else None


def normalise_weights(idx: np.ndarray, w: np.ndarray, where: str) -> np.ndarray:
    total = w.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise LookError(f"{where}: vertex {int(np.argmin(total))} has no positive weight")
    return w / total


def read_weights(path: Path, n_verts: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(body labels, idx [V,K], w [V,K] normalised) from a per-vertex JSON or .npz weights file."""
    labels, idx, w, _ = load_weights(path)
    if idx.shape[0] != n_verts:
        raise LookError(f"skins weights {path}: {idx.shape[0]} rows for {n_verts} mesh vertices")
    return labels, idx, normalise_weights(idx, w, f"skins weights {path}")


def body_ids(bundle: Bundle, labels: list[str], where: str) -> np.ndarray:
    """Body columns for labels, preferring the hero world when a label repeats across worlds."""
    hero = bundle.scene.get("hero_world", (bundle.scene.get("worlds") or [0])[0])
    out = []
    for label in labels:
        hits = [b for b in bundle.scene.get("bodies", []) if b.get("label") == label]
        if not hits:
            raise LookError(f"{where}: body label {label!r} is not in the bundle's bodies")
        out.append(int(next((b for b in hits if b.get("world") == hero), hits[0])["id"]))
    return np.asarray(out, np.int64)


def rigid_matrix(bundle: Bundle, body: int, rest_frame: int, fr: float, local: bool) -> np.ndarray:
    """4x4 transform for a mesh rigidly attached to `body`: T(fr) for a body-local mesh, else T(fr) T(rest)^-1."""
    p, q = motion.interp_poses(bundle.body_q, fr)
    now = motion.pose_matrices(p[body], q[body])[0]
    if local:
        return now
    p0, q0 = motion.interp_poses(bundle.body_q, float(rest_frame))
    return now @ np.linalg.inv(motion.pose_matrices(p0[body], q0[body])[0])


def relative_poses(bundle: Bundle, ids: np.ndarray, rest_frame: int, fr: float) -> tuple[np.ndarray, np.ndarray]:
    """Per body (translation, xyzw rotation) of T(fr) * T(rest)^-1."""
    p_now, q_now = motion.interp_poses(bundle.body_q, fr)
    p_rest, q_rest = motion.interp_poses(bundle.body_q, float(rest_frame))
    q_inv = q_rest[ids] * np.array([-1.0, -1.0, -1.0, 1.0])
    q = motion.quat_mul(q_now[ids], q_inv)
    return p_now[ids] - motion.quat_rotate(q, p_rest[ids]), q


def blend_dual_quat(v0: np.ndarray, idx: np.ndarray, w: np.ndarray, t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Dual-quaternion skinning of rest vertices v0 [V,3] (antipodal-safe, normalised)."""
    real = q
    dual = 0.5 * motion.quat_mul(np.concatenate([t, np.zeros((len(t), 1))], axis=1), q)
    r, d = real[idx], dual[idx]  # [V, K, 4]
    sign = np.where(np.sum(r * r[:, :1], axis=-1) < 0.0, -1.0, 1.0)
    ws = (w * sign)[..., None]
    br, bd = (ws * r).sum(axis=1), (ws * d).sum(axis=1)
    norm = np.linalg.norm(br, axis=1, keepdims=True)
    br, bd = br / norm, bd / norm
    conj = br * np.array([-1.0, -1.0, -1.0, 1.0])
    trans = 2.0 * motion.quat_mul(bd, conj)[:, :3]
    return motion.quat_rotate(br, v0) + trans


def blend_linear(v0: np.ndarray, idx: np.ndarray, w: np.ndarray, t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Linear blend skinning: the weighted mean of each influence's rigidly moved vertex."""
    out = np.zeros_like(v0)
    for k in range(idx.shape[1]):
        out += w[:, k:k + 1] * (motion.quat_rotate(q[idx[:, k]], v0) + t[idx[:, k]])
    return out
