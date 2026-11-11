# SPDX-License-Identifier: Apache-2.0
"""Run ``process`` over many objects in parallel, one forked process per object with a time limit.

Job file (JSON in)::

    {"jobs": [{"name": "gso_x", "url": "https://.../x.zip", "mesh": "meshes/model.obj",
               "extract": ["meshes", "materials", "metadata.pbtxt"],
               "copy": [["materials/textures/texture.png", "meshes/texture.png"]],
               "keep": ["metadata.pbtxt"], "scale": 1.0, "up_axis": "z", "max_parts": 32}]}

A ``url`` that serves a bare mesh (not a zip or tar, e.g. a ``.glb``) is processed directly; ``mesh`` is then
ignored. ``extract`` lists the archive prefixes to unpack (default: the mesh's folder), ``copy`` places files where the
mesh's material file expects them, ``keep`` copies small members (e.g. metadata) into the output.
Each object lands in ``<out>/<name>/`` (``process`` files plus the ``keep`` members) and
writes ``result.json`` or, on failure, ``error.json`` with the step and the message. ``<out>/summary.jsonl`` gets one
line per object. Objects that already have ``result.json`` are skipped, so a rerun after an eviction only does the
rest. Downloads are deleted after each object, so disk use stays at a few archives.

Example (JSON out on stdout)::

    python -m alloy.importers.common.batch jobs.json out --workers 30 --timeout 900 --mem-gb 6
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path

from alloy.importers.common import hulls as hl
from alloy.importers.common.errors import ImporterError
from alloy.importers.common.process import RESULT, ProcessOptions, process
from alloy.importers.common.sources import download, extract, is_archive

log = logging.getLogger("alloy.importers")

ERROR = "error.json"


def _options(job: dict) -> ProcessOptions:
    params = replace(hl.HullParams(), max_parts=int(job.get("max_parts", hl.MAX_PARTS)))
    return ProcessOptions(scale=float(job.get("scale", 1.0)), up_axis=str(job.get("up_axis", "z")), hulls=params)


def _keep(src_root: Path, out: Path, members: list[str]) -> None:
    for m in members:
        p = src_root / m
        if p.is_file():
            (out / Path(m).name).write_bytes(p.read_bytes())


def _copy(src_root: Path, pairs: list[list[str]]) -> None:
    for a, b in pairs:
        if (src_root / a).is_file():
            (src_root / b).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_root / a, src_root / b)


def run_one(job: dict, out_root: Path) -> dict:
    """Download, unpack (archives only), ``process`` and clean up one job; returns the result dict (raises on failure).

    Example::

        run_one({"name": "gso_x", "url": "https://.../x.zip", "mesh": "meshes/model.obj"}, Path("out"))
    """
    out = Path(out_root) / job["name"]
    with tempfile.TemporaryDirectory(prefix="alloy-imp-") as tmp:
        tmpd = Path(tmp)
        got = download(job["url"], tmpd / Path(job["url"].split("?")[0]).name)
        if not is_archive(got):              # a bare mesh file (e.g. an Objaverse .glb): process it as is
            return process(got, out, job["name"], _options(job))
        prefixes = tuple(job.get("extract") or [str(Path(job["mesh"]).parent)]) + tuple(job.get("keep", []))
        extract(got, tmpd / "src", prefixes)
        got.unlink()
        _copy(tmpd / "src", job.get("copy", []))
        result = process(tmpd / "src" / job["mesh"], out, job["name"], _options(job))
        _keep(tmpd / "src", out, list(job.get("keep", [])))
    return result


def _limit_memory(gb: float) -> None:
    """Cap this process's address space (Linux); a mesh that needs more fails with MemoryError, not the host."""
    if gb > 0:
        import resource  # noqa: PLC0415 - POSIX only

        cap = int(gb * (1 << 30))
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def _child(job: dict, out_root: str, mem_gb: float = 0.0) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _limit_memory(mem_gb)
    out = Path(out_root) / job["name"]
    try:
        run_one(job, Path(out_root))
    except Exception as exc:  # noqa: BLE001 - every failure becomes an error.json the caller can read
        step = exc.step if isinstance(exc, ImporterError) else "unexpected"
        out.mkdir(parents=True, exist_ok=True)
        err = {"name": job["name"], "ok": False, "step": step, "error": f"{type(exc).__name__}: {exc}",
               "trace": traceback.format_exc(limit=6)}
        (out / ERROR).write_text(json.dumps(err, indent=1), encoding="utf-8")


def _record(out_root: Path, name: str, extra: dict | None = None) -> dict:
    d = out_root / name
    for fname in (RESULT, ERROR):
        if (d / fname).is_file():
            row = json.loads((d / fname).read_text(encoding="utf-8"))
            return {"name": name, "ok": row.get("ok", False), **({} if row.get("ok") else
                    {"step": row.get("step"), "error": row.get("error")}), **(extra or {})}
    return {"name": name, "ok": False, "step": "worker", "error": "no result (crashed or killed)", **(extra or {})}


def _reap(running: dict, out_root: Path, timeout: float, rows: list[dict]) -> None:
    now = time.monotonic()
    for proc, (job, start) in list(running.items()):
        if proc.is_alive() and now - start <= timeout:
            continue
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            d = out_root / job["name"]
            d.mkdir(parents=True, exist_ok=True)
            (d / ERROR).write_text(json.dumps({"name": job["name"], "ok": False, "step": "timeout",
                                               "error": f"over {timeout:.0f} s"}), encoding="utf-8")
        proc.join()
        rows.append(_record(out_root, job["name"], {"seconds": round(now - start, 1)}))
        del running[proc]


def run_batch(jobs: list[dict], out_root: Path, workers: int, timeout: float, mem_gb: float = 0.0) -> list[dict]:
    """Process every job not already done, ``workers`` at a time; returns one summary row per job run.

    Example::

        rows = run_batch(json.load(open("jobs.json"))["jobs"], Path("out"), workers=30, timeout=900, mem_gb=6)
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out_root / j["name"] / RESULT).is_file()]
    ctx = mp.get_context("fork")
    running: dict = {}
    rows: list[dict] = []
    while todo or running:
        while todo and len(running) < workers:
            job = todo.pop(0)
            shutil.rmtree(out_root / job["name"], ignore_errors=True)
            proc = ctx.Process(target=_child, args=(job, str(out_root), mem_gb), daemon=False)
            proc.start()
            running[proc] = (job, time.monotonic())
        _reap(running, out_root, timeout, rows)
        time.sleep(0.2)
    with open(out_root / "summary.jsonl", "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    return rows


def main(argv: list[str] | None = None) -> int:
    """CLI: ``batch jobs.json out [--workers N] [--timeout S] [--mem-gb G]``; prints a JSON summary, exit 0 if all
    succeeded.

    Example::

        main(["jobs.json", "out", "--workers", "30", "--mem-gb", "6"])
    """
    ap = argparse.ArgumentParser(prog="alloy.importers.common.batch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--timeout", type=float, default=900.0, help="seconds per object")
    ap.add_argument("--mem-gb", type=float, default=0.0, help="address-space cap per object (0 = none)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    jobs = json.loads(args.jobs.read_text(encoding="utf-8"))["jobs"]
    rows = run_batch(jobs, args.out, args.workers, args.timeout, args.mem_gb)
    failed = [r for r in rows if not r["ok"]]
    json.dump({"ran": len(rows), "ok": len(rows) - len(failed), "failed": failed}, sys.stdout, indent=1)
    sys.stdout.write("\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
