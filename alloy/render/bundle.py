# SPDX-License-Identifier: Apache-2.0
"""ARB v1 bundle reader and validator (pure numpy; no Newton or Warp import)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("alloy.render")

ARB_VERSION = 1
LABEL_KEYS = ("shapes", "bodies", "surfaces", "points", "curves", "instancers")


class BundleError(ValueError):
    """A bundle is missing, malformed, from another ARB version, or cannot be written."""


@dataclass(frozen=True)
class Bundle:
    """An opened bundle: its directory and the parsed ``scene.json``."""

    root: Path
    scene: dict

    @property
    def frame_count(self) -> int:
        return int(self.scene["frames"]["count"])

    @property
    def fps(self) -> float:
        return float(self.scene["frames"]["fps"])

    @property
    def duration_s(self) -> float:
        """Time span first to last frame: ``(count - 1) / fps``."""
        return max(self.frame_count - 1, 0) / self.fps

    def labels(self) -> dict[str, list[str]]:
        """Matchable names per entity kind (shape and body labels, then declared names)."""
        out = {}
        for key in LABEL_KEYS:
            field = "label" if key in ("shapes", "bodies") else "name"
            out[key] = [str(e[field]) for e in self.scene.get(key, [])]
        return out

    def array(self, name: str, mmap: bool = True) -> np.ndarray:
        """Load ``frames/<name>.npy`` (memory-mapped read-only by default)."""
        path = self.root / "frames" / f"{name}.npy"
        if not path.is_file():
            raise BundleError(f"{self.root}: missing frames/{name}.npy")
        return np.load(path, mmap_mode="r" if mmap else None)

    def nbytes(self) -> int:
        """Total size of every file in the bundle."""
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


def _referenced_files(scene: dict) -> list[str]:
    files = ["frames/t.npy"]
    if scene.get("bodies"):
        files.append("frames/body_q.npy")
    files += [m["file"] for m in scene.get("meshes", {}).values()]
    for s in scene.get("surfaces", []):
        files += [s["file"], f"frames/surf_{s['name']}.npy"]
    for p in scene.get("points", []):
        files += [f"frames/pts_{p['name']}.npy", f"frames/vel_{p['name']}.npy"]
    files += [f"frames/curve_{c['name']}.npy" for c in scene.get("curves", [])]
    files += [f"frames/inst_{i['name']}.npy" for i in scene.get("instancers", [])]
    return files


def load_bundle(path) -> Bundle:
    """Open a bundle directory, checking the ARB version and that every referenced file exists."""
    root = Path(path)
    scene_file = root / "scene.json"
    if not scene_file.is_file():
        raise BundleError(f"{root}: no scene.json (not an ARB bundle)")
    try:
        scene = json.loads(scene_file.read_text())
    except json.JSONDecodeError as exc:
        raise BundleError(f"{scene_file}: invalid JSON ({exc})") from exc
    if scene.get("arb") != ARB_VERSION:
        raise BundleError(f"{scene_file}: arb={scene.get('arb')!r}, this reader supports arb={ARB_VERSION}")
    if "frames" not in scene or "count" not in scene["frames"] or "fps" not in scene["frames"]:
        raise BundleError(f"{scene_file}: missing frames.count / frames.fps")
    missing = [f for f in _referenced_files(scene) if not (root / f).is_file()]
    if missing:
        raise BundleError(f"{root}: missing files {missing[:8]}{' ...' if len(missing) > 8 else ''}")
    return Bundle(root=root, scene=scene)


def _check_shape(b: Bundle, name: str, want: tuple, problems: list[str]) -> np.ndarray | None:
    arr = b.array(name)
    if tuple(arr.shape) != want:
        problems.append(f"frames/{name}.npy has shape {tuple(arr.shape)}, expected {want}")
        return None
    return arr


def _check_finite(name: str, arr: np.ndarray, problems: list[str]) -> None:
    for k in range(arr.shape[0]):
        if not np.isfinite(arr[k]).all():
            problems.append(f"frames/{name}.npy has NaN/Inf at frame {k}")
            return


def _validate_frames(b: Bundle, problems: list[str]) -> None:
    sc, n = b.scene, b.frame_count
    t = b.array("t")
    if t.shape != (n,):
        problems.append(f"frames/t.npy has shape {t.shape}, expected ({n},)")
    elif n > 1 and not (np.diff(t) > 0).all():
        problems.append("frames/t.npy is not strictly increasing")
    wants = [("body_q", (n, len(sc.get("bodies", [])), 7))] if sc.get("bodies") else []
    wants += [(f"surf_{s['name']}", (n, s["verts"], 3)) for s in sc.get("surfaces", [])]
    for p in sc.get("points", []):
        wants += [(f"pts_{p['name']}", (n, p["count"], 3)), (f"vel_{p['name']}", (n, p["count"], 3))]
    wants += [(f"curve_{c['name']}", (n, c["count"], 3)) for c in sc.get("curves", [])]
    wants += [(f"inst_{i['name']}", (n, i["count"], 7)) for i in sc.get("instancers", [])]
    for name, want in wants:
        arr = _check_shape(b, name, want, problems)
        if arr is not None:
            _check_finite(name, arr, problems)


def _validate_refs(b: Bundle, problems: list[str]) -> None:
    sc = b.scene
    n_bodies = len(sc.get("bodies", []))
    ids = {s["id"] for s in sc.get("shapes", [])}
    for s in sc.get("shapes", []):
        if not -1 <= s["body"] < n_bodies:
            problems.append(f"shape {s['id']} ({s['label']!r}): body {s['body']} out of range [-1, {n_bodies})")
        if s["type"] == "mesh" and s.get("mesh") not in sc.get("meshes", {}):
            problems.append(f"shape {s['id']} ({s['label']!r}): unknown mesh {s.get('mesh')!r}")
        if s.get("replica_of") is not None and s["replica_of"] not in ids:
            problems.append(f"shape {s['id']}: replica_of {s['replica_of']} is not a shape id")
    for i in sc.get("instancers", []):
        if "mesh" in i and i["mesh"] not in sc.get("meshes", {}):
            problems.append(f"instancer {i['name']!r}: unknown mesh {i['mesh']!r}")


def _validate_npz(b: Bundle, problems: list[str]) -> None:
    for mid, m in b.scene.get("meshes", {}).items():
        with np.load(b.root / m["file"]) as z:
            v, f = z["vertices"], z["faces"]
        if v.shape != (m["verts"], 3) or f.shape != (m["faces"], 3):
            problems.append(f"mesh {mid}: arrays {v.shape}/{f.shape} disagree with scene.json")
        elif f.size and (f.min() < 0 or f.max() >= len(v)):
            problems.append(f"mesh {mid}: face index out of range")
    for s in b.scene.get("surfaces", []):
        with np.load(b.root / s["file"]) as z:
            faces, src, tag, twins = z["faces"], z["src"], z["face_tag"], z["twins"]
        if len(src) != s["verts"] or len(faces) != s["faces"] or len(tag) != len(faces):
            problems.append(f"surface {s['name']!r}: counts disagree with scene.json")
        elif faces.size and (faces.min() < 0 or faces.max() >= len(src)):
            problems.append(f"surface {s['name']!r}: face index out of range")
        elif twins.size and (twins.min() < 0 or twins.max() >= len(src)):
            problems.append(f"surface {s['name']!r}: twin index out of range")


def validate_bundle(b: Bundle) -> list[str]:
    """Return a list of problems (empty when the bundle is consistent)."""
    problems: list[str] = []
    for check in (_validate_refs, _validate_npz, _validate_frames):
        try:
            check(b, problems)
        except (KeyError, BundleError, ValueError, OSError) as exc:
            problems.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
    return problems
