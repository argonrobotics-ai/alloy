# SPDX-License-Identifier: Apache-2.0
"""Command line: ``python -m alloy.importers.gymanything <command> ...``; JSON on stdout, logs on stderr.

Commands:
- ``import --specs DIR --out DIR [--sim S ...]``: convert exported specs to cards in the library root ``--out`` and
  check them (Alloy's card validator, mass via Newton, cited dimensions). Exit 1 if anything failed.
  ``--repo DIR`` (a Gym-Anything checkout) also looks every cited number up at its cited line.
- ``export --python PY --repo DIR --specs DIR [--sim S ...]``: run the exporters in Gym-Anything's own env ``PY``.
  ``--drop`` drop-tests every rigid object; ``--no-hulls`` skips the hull sidecars (no importer extra needed).
- ``fetch --root DIR [--store S]``: download missing or stale meshes (sha256 from ``manifest.json``) from an asset
  store (directory, ``gs://`` or ``https://`` prefix; or ``$ALLOY_ASSET_STORE``).
- ``upload --root DIR --store gs://...``: copy the meshes to ``<store>/gymanything/meshes/``.

Example::

    python -m alloy.importers.gymanything import --specs work/specs --out work/out --sim wine_pouring
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from alloy.importers.gymanything.pipeline import ImportOptions, import_specs

RUNNER = Path(__file__).resolve().parents[3] / "tools" / "gymanything_export" / "run.py"   # runs in their env


def _export(a: argparse.Namespace) -> dict:
    cmd = [a.python, str(RUNNER), "--repo", a.repo, "--out", a.specs]
    for s in a.sim or ():
        cmd += ["--sim", s]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    sys.stderr.write(proc.stderr[-4000:])
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": f"exporter run failed (exit {proc.returncode}); see stderr", "stdout": proc.stdout[-2000:]}


@contextlib.contextmanager
def _stdout_to_stderr():
    """Point fd 1 (this process and the worker processes it starts) at stderr for the duration."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _storage(a: argparse.Namespace) -> int:
    from alloy.importers.common import manifest as mf  # noqa: PLC0415
    from alloy.importers.common.errors import ImporterError  # noqa: PLC0415

    root = Path(a.root)
    try:
        man = mf.load(root / "manifest.json", "gymanything")
        if a.cmd == "fetch":
            res = mf.fetch(root, man, store=a.store)
        else:
            dest = f"{mf.store_root(a.store)}/gymanything/{mf.MESH_DIR}"
            subprocess.run(["gcloud", "storage", "rsync", "-r", "-q", str(root / mf.MESH_DIR), dest], check=True,
                           capture_output=True)
            res = {"uploaded_objects": len(man["objects"])}
    except (ImporterError, subprocess.CalledProcessError, OSError) as exc:
        res = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(res, indent=1, default=str))
    return 0 if "error" not in res else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m alloy.importers.gymanything", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    imp = sub.add_parser("import", help="specs -> checked cards")
    imp.add_argument("--specs", required=True)
    imp.add_argument("--out", required=True)
    imp.add_argument("--sim", action="append")
    imp.add_argument("--repo", help="Gym-Anything checkout, to verify each cited number at its cited line")
    imp.add_argument("--no-hulls", action="store_true", help="skip hull sidecars (convex/sdf collision only)")
    imp.add_argument("--no-visuals", action="store_true", help="skip visual.glb (no trimesh/Pillow needed)")
    imp.add_argument("--hull-cache", help="folder of earlier <card>.hulls.obj sidecars to reuse")
    imp.add_argument("--drop", action="store_true", help="drop-test every rigid object")
    imp.add_argument("--workers", type=int, default=2)
    imp.add_argument("--sources", help="sourcing records root (<dir>/gymanything/<card>.yaml) to fill open values")
    exp = sub.add_parser("export", help="run the exporters in Gym-Anything's env")
    exp.add_argument("--python", required=True, help="Python of an env with gym-anything-robotics installed")
    exp.add_argument("--repo", required=True, help="Gym-Anything for Robotics checkout")
    exp.add_argument("--specs", required=True)
    exp.add_argument("--sim", action="append")
    for name, what in (("fetch", "download and verify the manifest's meshes"), ("upload", "copy meshes to a store")):
        sp = sub.add_parser(name, help=what)
        sp.add_argument("--root", required=True)
        sp.add_argument("--store", help="asset store (default $ALLOY_ASSET_STORE)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    if a.cmd == "import":
        with _stdout_to_stderr():                         # Warp's banner and worker logs; stdout is JSON only
            opt = ImportOptions(repo=Path(a.repo) if a.repo else None, hulls=not a.no_hulls, drop=a.drop,
                                visuals=not a.no_visuals,
                                hull_cache=Path(a.hull_cache) if a.hull_cache else None, workers=a.workers,
                                sources=Path(a.sources) if a.sources else None)
            res = import_specs(Path(a.specs), Path(a.out), a.sim, opt)
        print(json.dumps(res, indent=1, default=str))
        return 0 if not res["failed"] and not res["check_failures"] else 1
    if a.cmd in ("upload", "fetch"):
        return _storage(a)
    res = _export(a)
    print(json.dumps(res, indent=1, default=str))
    return 0 if "error" not in res and not any("crashed" in r for r in res.get("results", [])) else 1


if __name__ == "__main__":
    sys.exit(main())
