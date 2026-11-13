# SPDX-License-Identifier: Apache-2.0
"""Source mesh loading and the watertight physics mesh (needs the ``import`` extra: trimesh, fast-simplification).

Pipeline for ``physics.obj``:
1. load the source (OBJ/GLB/PLY...) as one mesh, apply ``scale`` and turn ``up_axis`` to +Z;
2. weld texture seams (merge vertices by position only);
3. if not closed: fix the winding, cut pinched edges and fan-fill holes (``manifold_fix``); only if that fails,
   remesh through a voxel grid (``VOXELS`` cells on the longest side, at least ``MIN_PITCH``) and marching cubes;
4. simplify by quadric edge collapse (fast-simplification, MIT) to about ``max_faces`` faces, repairing pinched
   edges (``manifold_fix``); if that fails, keep a clean mesh of up to ``KEEP_FACES`` faces, else start again from
   a voxel remesh.
Every step is recorded in ``RepairInfo`` so a card can say how its geometry was made.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from alloy.importers.common.errors import ImporterError, MissingExtraError, need
from alloy.importers.common.manifold_fix import fix_pinches, is_clean

log = logging.getLogger("alloy.importers")

VOXELS = 128                 # cells on the longest side for the voxel remesh fallback
MIN_PITCH = 5e-4             # m; finer voxels than this cost time without changing contact
MAX_FACES = 5000
KEEP_FACES = 4 * MAX_FACES  # a clean mesh that will not simplify is kept up to this size
MAX_VOLUME_CHANGE = 0.03     # a simplification may change the enclosed volume by at most 3 %
UP_ROTATIONS = {             # source up axis -> rotation taking it to +Z (right-handed)
    "z": np.eye(3),
    "y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    "-y": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
    "x": np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
}


@dataclass(frozen=True)
class RepairInfo:
    """How the physics mesh was made from the source (JSON-ready via ``asdict``)."""

    source_faces: int
    welded_watertight: bool
    method: str                  # "as_is" | "repair" | "voxel"
    voxel_pitch: float | None
    faces: int
    simplify: str                # "none" | "kept" | "qem:<target faces>:a<aggressiveness>"
    volume: float                # m^3 of the final physics mesh
    source_volume: float | None  # m^3 of the welded source when it was watertight

    def to_json(self) -> dict:
        """Plain dict for ``json.dumps``.

        Example::

            json.dumps(result.to_json())
        """
        return asdict(self)


def frame_matrix(scale: float, up_axis: str) -> np.ndarray:
    """3x3 matrix taking source coordinates to Alloy's body frame (metres, +Z up).

    Example::

        frame_matrix(0.001, "y")   # a Y-up millimetre file
    """
    if up_axis not in UP_ROTATIONS:
        raise ImporterError("<mesh>", "frame", f"up_axis {up_axis!r} not in {sorted(UP_ROTATIONS)}")
    if not (scale > 0 and np.isfinite(scale)):
        raise ImporterError("<mesh>", "frame", f"scale must be a positive number, got {scale!r}")
    return UP_ROTATIONS[up_axis] * float(scale)


def frame4(scale: float, up_axis: str) -> np.ndarray:
    """4x4 version of ``frame_matrix`` (source -> body frame).

    Example::

        frame4(1.0, "y") @ [0, 1, 0, 1]   # -> [0, 0, 1, 1]
    """
    m = np.eye(4)
    m[:3, :3] = frame_matrix(scale, up_axis)
    return m


def load_scene(path: Path):
    """The source file as a ``trimesh.Scene`` in source units and axes (for ``visual.export_glb``).

    Example::

        scene = load_scene(Path("chair.glb"))
    """
    trimesh = need("trimesh")
    try:
        return trimesh.load(str(path), force="scene", process=False)
    except (ValueError, OSError) as exc:
        raise ImporterError(str(path), "load", f"trimesh cannot read it: {exc}") from exc


def load_source(path: Path, scale: float = 1.0, up_axis: str = "z"):
    """The source as one textured ``trimesh.Trimesh`` in metres, +Z up (texture and UVs kept, no welding).

    Example::

        mesh = load_source(Path("textured.obj"), scale=1.0, up_axis="z")
    """
    trimesh = need("trimesh")
    try:
        mesh = trimesh.load(str(path), force="mesh", process=False)
    except (ValueError, OSError) as exc:
        raise ImporterError(str(path), "load", f"trimesh cannot read it: {exc}") from exc
    if len(mesh.faces) == 0:
        raise ImporterError(str(path), "load", "mesh has no faces; check the file and its format")
    mesh.apply_transform(frame4(scale, up_axis))
    return mesh


def weld(mesh):
    """Geometry only, vertices merged by position (texture seams closed), degenerate faces dropped.

    Example::

        g = weld(load_source(Path("textured.obj")))
    """
    trimesh = need("trimesh")
    g = trimesh.Trimesh(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), process=True)
    g.merge_vertices(digits_vertex=7)
    g.update_faces(g.nondegenerate_faces())
    g.remove_unreferenced_vertices()
    return g


def voxel_remesh(g, voxels: int = VOXELS, min_pitch: float = MIN_PITCH):
    """Watertight remesh through a filled voxel grid and marching cubes; returns ``(mesh, pitch)``.

    Example::

        closed, pitch = voxel_remesh(open_scan)
    """
    pitch = max(float(np.max(g.extents)) / voxels, min_pitch)
    grid = g.voxelized(pitch).fill()
    out = grid.marching_cubes
    out.apply_transform(grid.transform)
    return out, pitch


def _closed(g) -> bool:
    return bool(is_clean(np.asarray(g.faces)) and g.volume > 0)


def _repaired(g):
    """Winding fixed, then pinched edges cut and holes fan-filled (``manifold_fix``); None if still open."""
    trimesh = need("trimesh")
    h = g.copy()
    trimesh.repair.fix_normals(h)
    v, f, ok = fix_pinches(np.asarray(h.vertices), np.asarray(h.faces), rounds=8)
    out = trimesh.Trimesh(v, f, process=False)
    return out if ok and out.volume > 0 else None


def make_watertight(g) -> tuple[object, str, float | None]:
    """``(mesh, method, voxel_pitch)``: the first closed result of: as is, local repair, voxel remesh.

    Local repair (winding fix, pinch cut, hole fan-fill) keeps the scan's surface; the voxel remesh is the last
    resort because it thickens walls thinner than about two voxels.

    Example::

        closed, method, pitch = make_watertight(weld(source))
    """
    if _closed(g):
        return g, "as_is", None
    h = _repaired(g)
    if h is not None:
        return h, "repair", None
    v, pitch = voxel_remesh(g)
    v = _repaired(v) if not _closed(v) else v
    if v is None:
        raise ImporterError("<mesh>", "repair", "voxel remesh is not watertight; the source may be empty or flat")
    return v, "voxel", pitch


def _qem(g, target: int, agg: int):
    """Quadric simplification to ``target`` faces, welded, pinches repaired; None if not clean afterwards."""
    trimesh, fs = need("trimesh"), need("fast_simplification")
    v, f = fs.simplify(np.asarray(g.vertices, np.float32), np.asarray(g.faces, np.int32), target_count=target, agg=agg)
    w = trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(f, np.int64), process=True)
    v2, f2, ok = fix_pinches(np.asarray(w.vertices), np.asarray(w.faces))
    out = trimesh.Trimesh(v2, f2, process=False)
    return out if ok and out.volume > 0 else None


def simplify(g, max_faces: int = MAX_FACES) -> tuple[object, str]:
    """``(mesh, how)``: a clean (closed, consistently wound) simplification of a watertight mesh.

    Tries quadric collapse at ``max_faces`` (then 1.5x, 2x) with low aggressiveness, repairing pinched edges
    (``manifold_fix``), and accepts the first result within ``MAX_VOLUME_CHANGE`` of the input volume. A mesh
    already within budget is returned as is. Raises ``ImporterError`` if nothing qualifies.

    Example::

        small, how = simplify(closed, max_faces=5000)   # how == "qem:5000:a3"
    """
    if len(g.faces) <= max_faces and is_clean(np.asarray(g.faces)):
        return g, "none"
    targets = [t for t in (max_faces, int(1.5 * max_faces), 2 * max_faces) if t < len(g.faces)]
    for target in targets:
        for agg in (3, 2, 1):
            out = _qem(g, target, agg)
            if out is not None and abs(out.volume / g.volume - 1.0) <= MAX_VOLUME_CHANGE:
                return out, f"qem:{target}:a{agg}"
    raise ImporterError("<mesh>", "simplify", f"no clean simplification within {2 * max_faces} faces")


_WARNED: list[bool] = []


def _warn_kept(faces: int, max_faces: int, exc: MissingExtraError) -> None:
    """Say once that meshes are kept unsimplified because the simplifier is not installed."""
    if not _WARNED:
        _WARNED.append(True)
        log.warning("mesh of %d faces kept unsimplified (target %d): %s; later meshes up to %d faces are kept the "
                    "same way", faces, max_faces, exc.message, KEEP_FACES)


def physics_mesh(source, max_faces: int = MAX_FACES) -> tuple[object, RepairInfo]:
    """Watertight, consistently wound physics mesh of at most ``max_faces`` faces, plus how it was made.

    Example::

        phys, info = physics_mesh(load_source(Path("model.obj")))
        info.volume  # m^3
    """
    g = weld(source)
    src_closed = _closed(g)
    closed, method, pitch = make_watertight(g)
    try:
        small, how = simplify(closed, max_faces)
    except ImporterError as exc:
        if len(closed.faces) <= KEEP_FACES:
            if isinstance(exc, MissingExtraError):          # kept as before, but say the simplifier is missing
                _warn_kept(len(closed.faces), max_faces, exc)
            small, how = closed, "kept"
        else:
            log.info("simplify failed on the %s mesh; retrying from a voxel remesh", method)
            closed, pitch = voxel_remesh(g)
            method = "voxel"
            small, how = simplify(closed, max_faces)
    info = RepairInfo(source_faces=int(len(source.faces)), welded_watertight=src_closed, method=method,
                      voxel_pitch=pitch, faces=int(len(small.faces)), simplify=how,
                      volume=float(small.volume), source_volume=float(g.volume) if src_closed else None)
    return small, info
