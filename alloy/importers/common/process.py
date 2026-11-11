# SPDX-License-Identifier: Apache-2.0
"""One source mesh in, the importer's per-object files and a JSON summary out.

Files written to ``out_dir`` (names fixed so cards and manifests can rely on them):

- ``physics.obj``: watertight, consistently wound, about 5k faces, body frame (metres, +Z up). The card's
  ``shape.mesh.path``; Alloy computes mass as card density x its volume.
- ``physics.hulls.obj``: CoACD convex hulls of ``physics.obj`` (sidecar format in ``objio``), used by
  ``collision: decompose``.
- ``visual.glb``: the textured source mesh, glTF frame (+Y up, metres; see ``visual``).
- ``thumb.png``: 256 px preview.
- ``result.json``: the dict ``process`` returns.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from alloy.importers.common import hulls as hl
from alloy.importers.common import meshops, objio, visual
from alloy.importers.common.errors import ImporterError

log = logging.getLogger("alloy.importers")

PHYSICS = "physics.obj"
HULLS = "physics.hulls.obj"
VISUAL = "visual.glb"
THUMB = "thumb.png"
RESULT = "result.json"
FILES = (PHYSICS, HULLS, VISUAL, THUMB)


@dataclass(frozen=True)
class ProcessOptions:
    """Per-object settings; every field is recorded in ``result.json``."""

    scale: float = 1.0               # source units -> metres
    up_axis: str = "z"               # source up axis ("z", "y", "-y", "x")
    max_faces: int = meshops.MAX_FACES
    hulls: hl.HullParams = hl.HullParams()
    thumb_px: int = 256
    write_visual: bool = True


def _mass_props(mesh) -> dict:
    """Unit-density mass properties of the physics mesh (volume, centre of mass, inertia about it)."""
    return {"volume": float(mesh.volume), "center_mass": [float(x) for x in mesh.center_mass],
            "inertia_unit_density": np.asarray(mesh.moment_inertia, np.float64).round(12).tolist(),
            "bounds": np.asarray(mesh.bounds, np.float64).round(7).tolist(),
            "extents": np.asarray(mesh.extents, np.float64).round(7).tolist()}


def _write_physics(out: Path, name: str, phys, info: meshops.RepairInfo) -> str:
    header = f"{name}: physics mesh, repair={info.method}, simplify={info.simplify}, volume={info.volume:.6e} m^3"
    text = objio.mesh_obj_text(np.asarray(phys.vertices), np.asarray(phys.faces), header=header)
    (out / PHYSICS).write_text(text, encoding="utf-8")
    return objio.sha256_bytes(text.encode("utf-8"))


def hull_words(hulls: dict) -> str:
    """"14 CoACD hulls", or "3 convex hulls (near-convex mesh)" when hullfit skipped CoACD (``HullStats.to_json``).

    Example::

        hull_words(result["hulls"])
    """
    n = hulls["n_hulls"]
    return f"{n} convex hull{'s' * (n != 1)} (near-convex mesh)" if hulls.get("single") else f"{n} CoACD hulls"


def _write_hulls(out: Path, name: str, phys, sha: str, params: hl.HullParams) -> hl.HullStats:
    hulls, stats = hl.decompose(np.asarray(phys.vertices), np.asarray(phys.faces), float(phys.volume), params)
    header = (f"{name}: {hull_words(stats.to_json())}, threshold={params.threshold}, "
              f"max_hull_verts={params.max_hull_verts}, volume_ratio={stats.volume_ratio:.4f}")
    (out / HULLS).write_text(objio.hulls_obj_text(hulls, sha, header=header), encoding="utf-8")
    return stats


def _has_texture(mesh) -> bool:
    return getattr(mesh.visual, "uv", None) is not None and visual.base_image(mesh) is not None


def process(src_mesh_path: Path, out_dir: Path, name: str, opts: ProcessOptions = ProcessOptions()) -> dict:
    """Run the whole per-object pipeline and return its JSON summary (also written to ``result.json``).

    Example::

        r = process(Path("025_mug/google_16k/textured.obj"), Path("out/ycb_025_mug"), "ycb_025_mug")
        r["n_hulls"], r["files"]["physics.obj"]
    """
    t0 = time.monotonic()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        src = meshops.load_source(Path(src_mesh_path), opts.scale, opts.up_axis)
        phys, info = meshops.physics_mesh(src, opts.max_faces)
        sha = _write_physics(out, name, phys, info)
        stats = _write_hulls(out, name, phys, sha, opts.hulls)
        if opts.write_visual:
            visual.export_glb(meshops.load_scene(Path(src_mesh_path)), out / VISUAL,
                              meshops.frame4(opts.scale, opts.up_axis))
        visual.write_png(visual.render_thumbnail(src, opts.thumb_px), out / THUMB)
    except ImporterError as exc:
        raise ImporterError(name, exc.step, exc.message) from exc
    files = {f: objio.sha256_path(out / f) for f in FILES if (out / f).is_file()}
    result = {"name": name, "ok": True, "watertight": True, "repaired": info.method != "as_is",
              "volume": info.volume, "n_hulls": stats.n_hulls, "faces": info.faces,
              "texture_ok": _has_texture(src), "files": files,
              "sizes": {f: (out / f).stat().st_size for f in files}, "repair": info.to_json(),
              "hulls": stats.to_json(), "mass_props": _mass_props(phys),
              "options": {"scale": opts.scale, "up_axis": opts.up_axis, "max_faces": opts.max_faces},
              "seconds": round(time.monotonic() - t0, 2)}
    (out / RESULT).write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    log.info("%s: %d faces, %d hulls (ratio %.2f), %.1f s", name, info.faces, stats.n_hulls, stats.volume_ratio,
             result["seconds"])
    return result
