# SPDX-License-Identifier: Apache-2.0
"""Path-traced thumbnails for import libraries; JSON on stdout, logs on stderr.

Commands (run with Alloy's python, from the repo root; a GPU makes it fast)::

    python tools/thumbnails/run.py list   --root alloy-assets/imports/ycb_gso
    python tools/thumbnails/run.py check  --root alloy-assets/imports/ycb_gso       # load every visual, no render
    python tools/thumbnails/run.py render --root alloy-assets/imports/ycb_gso --state thumbs-state/ [--redo] [--only N]
    python tools/thumbnails/run.py render --job job.json            # {"libraries": [{"root": ...}], "size": 512, ...}
    python tools/thumbnails/run.py sheet  --images a.png b.png ... --out sheet.png [--cols 4]
    python tools/thumbnails/run.py manifest --root alloy-assets/imports/ycb_gso --state thumbs-state/   # thumbs.json

``render`` is resumable: ``<state>/<lib>.jsonl`` keeps one row per object (ok with the PNG's sha256, or the reason
it failed); a rerun renders only what is not ok (``--redo`` renders all). Thumbnails are written where each library
keeps them (``meshes/<name>/thumb.png``, RoboCasa ``thumbs/<category>/<name>.png``) and nothing else is touched.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))   # the repo root, for `alloy`

from batch import check_jobs, run_jobs  # noqa: E402
from libs import library_jobs  # noqa: E402


@contextlib.contextmanager
def _stdout_to_stderr():
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


def _render(job: dict) -> dict:
    from backend import CameraRenderer  # noqa: PLC0415

    renderer = CameraRenderer(size=int(job.get("size", 512)), spp=int(job.get("spp", 128)), denoise=job.get("denoise"),
                              exposure=float(job.get("exposure", -1.0)))
    out = {}
    for lib in job["libraries"]:
        root = Path(lib["root"])
        jobs = library_jobs(root, lib.get("lib"))
        if job.get("only"):
            jobs = [j for j in jobs if j.name in set(job["only"])]
        state = Path(job.get("state", "thumbs-state")) / f"{jobs[0].lib if jobs else root.name}.jsonl"
        out[root.name] = run_jobs(jobs, renderer, state, bool(job.get("redo")), job.get("limit"))
    return out


def _manifest(root: Path, state: Path, showcase: str | None, update: bool = False) -> dict:
    from batch import read_state  # noqa: PLC0415
    from manifest import SHOWCASE, build, load_showcase, update_library_manifest, write  # noqa: PLC0415

    jobs = library_jobs(root)
    lib = jobs[0].lib if jobs else root.name
    rows = {k: r for k, r in read_state(state / f"{lib}.jsonl").items() if k in {j.name for j in jobs}}
    m = build(root, lib, rows, load_showcase(Path(showcase) if showcase else SHOWCASE),
              {"size": 512, "camera": "alloy.camera.mesh_scene", "view": "3/4, studio light, light ground"})
    path = write(root, m)
    changed = update_library_manifest(root, m) if update else 0
    return {"thumbs": str(path), "lib": lib, "library_manifest_entries_updated": changed, "jobs": len(jobs), "count": m["count"], "showcase_ok": m["showcase_ok"],
            "failed": m["failed"], "missing": len(jobs) - m["count"] - len(m["failed"])}


def _sheet(images: list[str], out: str, cols: int) -> dict:
    from sheet import contact_sheet  # noqa: PLC0415

    return {"sheet": str(contact_sheet([Path(p) for p in images], Path(out), cols))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "check"):
        sub.add_parser(name).add_argument("--root", required=True)
    rd = sub.add_parser("render")
    rd.add_argument("--root", action="append", help="import library root (repeat)")
    rd.add_argument("--job", help="JSON job file instead of the flags")
    rd.add_argument("--state", default="thumbs-state")
    rd.add_argument("--only", action="append")
    rd.add_argument("--redo", action="store_true")
    rd.add_argument("--limit", type=int)
    rd.add_argument("--size", type=int, default=512)
    rd.add_argument("--spp", type=int, default=128)
    rd.add_argument("--exposure", type=float, default=-1.0, help="stops added to the camera's exposure")
    rd.add_argument("--denoise", help="the camera's denoiser: auto, learned or oidn (default none)")
    mf = sub.add_parser("manifest")
    mf.add_argument("--root", required=True)
    mf.add_argument("--state", default="thumbs-state")
    mf.add_argument("--showcase", help="showcase rules JSON (default tools/thumbnails/showcase.json)")
    mf.add_argument("--update-library-manifest", action="store_true",
                    help="also set the thumbnails' sha256/bytes in the library's manifest.json (owner's consent)")
    sh = sub.add_parser("sheet")
    sh.add_argument("--images", nargs="+", required=True)
    sh.add_argument("--out", required=True)
    sh.add_argument("--cols", type=int, default=4)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    if a.cmd == "list":
        res = [j.as_dict() for j in library_jobs(Path(a.root))]
    elif a.cmd == "check":
        with _stdout_to_stderr():
            res = check_jobs(library_jobs(Path(a.root)))
    elif a.cmd == "manifest":
        res = _manifest(Path(a.root), Path(a.state), a.showcase, a.update_library_manifest)
    elif a.cmd == "sheet":
        res = _sheet(a.images, a.out, a.cols)
    else:
        job = json.loads(Path(a.job).read_text()) if a.job else {
            "libraries": [{"root": r} for r in a.root or ()], "state": a.state, "only": a.only, "redo": a.redo,
            "limit": a.limit, "size": a.size, "spp": a.spp, "denoise": a.denoise,
            "exposure": a.exposure}
        with _stdout_to_stderr():
            res = _render(job)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
