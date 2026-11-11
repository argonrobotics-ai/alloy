#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run teacher-render jobs in parallel: Blender render -> pack -> optional upload, resumable.

    python tools/denoiser/run_jobs.py WORK/jobs.jsonl --work /data/render --out /data/packed --parallel 6 \\
        [--shard 0/2] [--rotate 1/6] [--blender /opt/blender/blender] [--upload "CMD {dir} {id}"] \\
        [--skip-if "CMD {id}"] [--only-split train]

Per job (``jobs.py`` format): ``render_bundle.py`` writes EXRs under ``WORK/<id>``, ``pack.py`` turns them into
arrays under ``OUT/<id>`` and deletes the EXRs, then the ``--upload`` shell command runs with ``{dir}`` (the packed
folder) and ``{id}`` filled in. A job is done when ``OUT/<id>/meta.json`` exists (and, with ``--upload``, the
``OUT/<id>/.uploaded`` marker). Failures are appended to ``OUT/failed.jsonl`` and retried once per run. Each job's
Blender log is ``WORK/<id>.log``. ``--shard k/n`` runs every n-th job starting at k (one shard per machine).
Several machines can also share one list: ``--rotate k/n`` starts at the k-th n-th of the list, and ``--skip-if``
(a shell command with ``{id}``, e.g. a check that the upload exists) skips jobs another machine already finished;
it is checked again just before each job starts.
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
log = logging.getLogger("denoiser.run_jobs")


def blender_cmd(blender: str, job: dict, out: Path, res: str, frames: list[int]) -> list[str]:
    """The Blender command line that renders ``frames`` of ``job`` at ``res``."""
    return [blender, "-b", "--python-exit-code", "1", "-P", str(HERE / "blender" / "render_bundle.py"), "--",
            "--bundle", job["bundle_dir"], "--look", job["look"], "--camera", job["camera"], "--out", str(out),
            "--frames", ",".join(map(str, frames)), "--res", res, "--inputs", job["inputs"],
            *job.get("blender_args", [])]


def run_logged(cmd: list[str] | str, logf: Path, shell: bool = False) -> int:
    """Run a command, appending its output to ``logf``; returns the exit code."""
    with logf.open("a") as f:
        f.write(f"\n$ {cmd if shell else shlex.join(cmd)}\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, shell=shell, check=False).returncode


def done_elsewhere(job: dict, args: argparse.Namespace) -> bool:
    """True if the --skip-if command says another machine finished this job."""
    if not args.skip_if:
        return False
    cmd = args.skip_if.format(id=job["id"])
    return subprocess.run(cmd, shell=True, check=False, capture_output=True).returncode == 0


def render_job(job: dict, args: argparse.Namespace) -> tuple[str, int]:
    """Render, pack and upload one job. Returns (stage, code): ("done", 0), ("skipped", 0) or the failing stage."""
    wdir, odir = Path(args.work) / job["id"], Path(args.out) / job["id"]
    if not (odir / "meta.json").exists() and done_elsewhere(job, args):
        return "skipped", 0
    logf = Path(args.work) / f"{job['id']}.log"
    wdir.mkdir(parents=True, exist_ok=True)
    (wdir / "job.json").write_text(json.dumps(job))
    passes = [(job["res"], job["frames"])] + ([(job["large"]["res"], job["large"]["frames"])] if "large" in job else [])
    if not (odir / "meta.json").exists():
        for res, frames in passes:
            code = run_logged(blender_cmd(args.blender, job, wdir, res, frames), logf)
            if code:
                return "render", code
        pack = [sys.executable, str(HERE / "pack.py"), str(wdir), str(odir), "--delete-exr"]
        code = run_logged(pack + (["--oidn"] if job["split"] == "eval" else []), logf)
        if code:
            return "pack", code
    if args.upload and not (odir / ".uploaded").exists():
        code = run_logged(args.upload.format(dir=shlex.quote(str(odir)), id=job["id"]), logf, shell=True)
        if code:
            return "upload", code
        (odir / ".uploaded").touch()
    return "done", 0


def is_done(job: dict, args: argparse.Namespace) -> bool:
    odir = Path(args.out) / job["id"]
    return (odir / "meta.json").exists() and (not args.upload or (odir / ".uploaded").exists())


def select_jobs(args: argparse.Namespace) -> list[dict]:
    """The jobs of this shard and split that are not done yet."""
    jobs = [json.loads(line) for line in Path(args.jobs).read_text().splitlines() if line.strip()]
    k, n = (int(v) for v in args.shard.split("/"))
    jobs = [j for i, j in enumerate(jobs) if i % n == k]
    if args.only_split:
        jobs = [j for j in jobs if j["split"] == args.only_split]
    r, m = (int(v) for v in args.rotate.split("/"))
    start = len(jobs) * r // m
    jobs = jobs[start:] + jobs[:start]
    return [j for j in jobs if not is_done(j, args)]


def run_all(jobs: list[dict], args: argparse.Namespace) -> list[dict]:
    """Run jobs on ``args.parallel`` workers; returns the failures."""
    failed = []
    with ThreadPoolExecutor(args.parallel) as pool:
        futs = {pool.submit(render_job, j, args): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            job, (stage, code) = futs[fut], fut.result()
            row = {"t": time.strftime("%H:%M:%SZ", time.gmtime()), "id": job["id"], "stage": stage, "code": code,
                   "n": f"{i}/{len(jobs)}"}
            log.info(json.dumps(row))
            if code:
                failed.append(job)
                with (Path(args.out) / "failed.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
    return failed


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--blender", default="blender")
    p.add_argument("--upload", default=None, help="shell command run after packing, with {dir} and {id}")
    p.add_argument("--only-split", dest="only_split", default=None)
    p.add_argument("--rotate", default="0/1", help="k/n: start at the k-th n-th of the job list")
    p.add_argument("--skip-if", dest="skip_if", default=None, help="shell command with {id}; exit 0 = skip the job")
    args = p.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    jobs = select_jobs(args)
    log.info(json.dumps({"todo": len(jobs), "shard": args.shard}))
    failed = run_all(jobs, args)
    if failed:
        log.info(json.dumps({"retry": len(failed)}))
        failed = run_all(failed, args)
    log.info(json.dumps({"summary": True, "jobs": len(jobs), "failed": len(failed)}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
