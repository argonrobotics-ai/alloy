# SPDX-License-Identifier: Apache-2.0
"""Install check for the ``import`` extra: run the importer pipeline on an open mesh, which takes the voxel path.

    python tools/deps/import_extra_check.py        # one JSON line; exit 0 when the mesh came out watertight

Runs the importer's voxel remesh (voxel grid + marching cubes: trimesh -> scikit-image), the fallback for open scans
whose holes local repair cannot close, on an uncapped cylinder, then the whole ``alloy.importers.common.process``
pipeline on it. Without scikit-image the remesh fails with ``No module named 'skimage'``. Needs
``pip install "alloy-engine[import]"``.

Example (Python):
    >>> callable(run)
    True
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path


def run() -> dict[str, object]:
    """Process an uncapped cylinder and report whether the physics mesh is watertight."""
    import trimesh

    from alloy.importers.common.meshops import voxel_remesh
    from alloy.importers.common.process import process

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        tube = trimesh.creation.cylinder(radius=0.04, height=0.1, sections=48)
        tube.update_faces(abs(tube.face_normals[:, 2]) < 0.5)  # drop both caps: an open scan-like shell
        src = Path(tmp) / "tube.obj"
        tube.export(src)
        try:
            closed, pitch = voxel_remesh(tube)
            summary = process(src, Path(tmp) / "out", "open_tube")
        except Exception as exc:  # report what broke, e.g. a missing optional package
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "wall_s": round(time.time() - t0, 1)}
    repair = summary.get("repair", {})
    ok = bool(closed.is_watertight and summary.get("ok") and summary.get("watertight") and not tube.is_watertight)
    return {"ok": ok, "source_watertight": bool(tube.is_watertight), "voxel_watertight": bool(closed.is_watertight),
            "voxel_pitch_m": pitch, "voxel_faces": int(len(closed.faces)), "pipeline_repair": repair.get("method"),
            "n_hulls": summary.get("n_hulls"), "files": sorted(summary.get("files", {})),
            "wall_s": round(time.time() - t0, 1)}


def main() -> int:
    """CLI entry: one JSON line on stdout."""
    doc = run()
    print(json.dumps(doc))
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
