# SPDX-License-Identifier: Apache-2.0
"""Batch worker for the Objaverse import: survey (fetch + quality) a list of candidates in parallel.

Runs on batch machines that have only the mesh tooling (``pip install trimesh pillow``), not the engine: started as a
script, it makes ``alloy`` importable without running ``alloy/__init__`` (which needs warp and newton).

    python alloy/importers/objaverse/worker.py survey --candidates cands.json --out survey/ --jobs 32

Writes ``survey/rows.jsonl`` (one JSON row per object: quality report, verdict, raw extents), keeps the GLBs in
``survey/glbs/`` and a review preview of each passing object in ``survey/previews/`` (needs pyrender). Re-running skips uids that already have a row, so an evicted batch resumes.

Example (in-process, with the engine installed):
    >>> from alloy.importers.objaverse.worker import survey_row
    >>> survey_row({"uid": "x", "category": "mug"}, {}, None)["verdict"]
    'fetch: uid not in object-paths.json.gz'
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import time
import types
from pathlib import Path
from typing import Any

if __name__ in ("__main__", "__mp_main__") and "alloy" not in sys.modules:  # engine-free machine: skip alloy/__init__
    _pkg = types.ModuleType("alloy")
    _pkg.__path__ = [str(Path(__file__).resolve().parents[2])]
    sys.modules["alloy"] = _pkg

from alloy.importers.objaverse import fetch as F  # noqa: E402
from alloy.importers.objaverse import quality as Q  # noqa: E402
from alloy.importers.objaverse.categories import load_categories  # noqa: E402
from alloy.importers.objaverse.errors import ImportRejected  # noqa: E402

log = logging.getLogger("alloy.importers.objaverse")
_PATHS: dict[str, str] = {}


def survey_row(cand: dict[str, Any], paths: dict[str, str], glb_dir: Path | None) -> dict[str, Any]:
    """Fetch one candidate's GLB, measure it and apply the quality limits; never raises (the verdict says why)."""
    row: dict[str, Any] = {"uid": cand["uid"], "category": cand["category"]}
    t0 = time.time()
    try:
        glb = F.fetch_glb(cand["uid"], paths, glb_dir or Path("."))
        row["glb_bytes"] = glb.stat().st_size
        geoms = Q.load_zup(str(glb))
        rep = Q.measure(geoms)
        row.update(rep.as_row())
        Q.check_report(cand["uid"], rep, flat_ok=load_categories()[cand["category"]].flat)
        row["verdict"] = "ok"
        if glb_dir is not None:
            _preview(geoms, glb_dir.parent / "previews" / f"{cand['uid']}.png", row)
    except ImportRejected as e:
        row["verdict"] = f"{e.stage}: {e.reason}"
    except Exception as e:  # noqa: BLE001 - a broken upload must not stop the batch
        row["verdict"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    row["sec"] = round(time.time() - t0, 2)
    return row


def _preview(geoms: list[Any], path: Path, row: dict[str, Any]) -> None:
    from alloy.importers.objaverse.review import render_preview  # noqa: PLC0415 - needs pyrender

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        render_preview(geoms).save(path)
    except Exception as e:  # noqa: BLE001 - a preview failure is noted, not fatal
        row["preview_error"] = f"{type(e).__name__}: {str(e)[:120]}"


def _init(paths: dict[str, str]) -> None:
    _PATHS.update(paths)


def _one(args: tuple[dict[str, Any], str]) -> dict[str, Any]:
    cand, glb_dir = args
    return survey_row(cand, _PATHS, Path(glb_dir))


def survey(cands: list[dict[str, Any]], out: Path, jobs: int) -> int:
    """Survey every candidate without a row in ``out/rows.jsonl``; returns the number of new rows."""
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(line)["uid"] for line in rows_path.read_text().splitlines() if line.strip()}
    todo = [(c, str(out / "glbs")) for c in cands if c["uid"] not in done]
    log.info("survey: %d candidates, %d done, %d to do, %d jobs", len(cands), len(done), len(todo), jobs)
    paths = F.object_paths(out / "cache")
    n = 0
    with mp.get_context("spawn").Pool(jobs, initializer=_init, initargs=(paths,), maxtasksperchild=20) as pool, \
            rows_path.open("a") as fh:
        for row in pool.imap_unordered(_one, todo, chunksize=1):
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            n += 1
            if n % 100 == 0:
                log.info("survey: %d / %d", n, len(todo))
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sv = sub.add_parser("survey", help="fetch + quality-check candidates (JSON list from select)")
    sv.add_argument("--candidates", type=Path, required=True)
    sv.add_argument("--out", type=Path, required=True)
    sv.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 1))
    sv.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cands = json.loads(a.candidates.read_text())
    n = survey(cands[: a.limit] if a.limit else cands, a.out, a.jobs)
    print(json.dumps({"new_rows": n, "rows": str(a.out / "rows.jsonl")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
