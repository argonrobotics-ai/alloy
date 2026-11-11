# SPDX-License-Identifier: Apache-2.0
"""Re-run the Newton/Alloy side of every xbench protocol and append the results to results/<protocol>/.

  python bench/xbench/run_all.py --engine-under-test {newton,alloy} [--protocols mujoco_warp,newton,...]
                                 [--tier headline|full] [--only REGEX] [--repeats N] [--dry-run | --list]

Run it on a CUDA VM with the Alloy venv's python (newton @ ac7f7830 + warp 1.18.0, mujoco-warp 3.16.0; torch for
maniskill/isaaclab/newton N3). Each job is one process (scenes/engine_shim.py -> the protocol's scene script), one GPU
job at a time, with the protocol's own settings and clock (xbench_jobs.py). Rows are appended to
results/<protocol>/runall_<engine>.jsonl in the common xbench schema (protocol, benchmark, engine, version, hardware,
envs, metric_name, metric_value, unit, settings, notes) plus `raw` (the script's own record). Logs go to
results/runall/<run_id>/<job>.log. Compare an Alloy row only with a Newton row from the SAME VM and session:
two L4 hosts of one machine type differed by up to 9.5 % (notes/newton.md) and ManiSkill by up to 31 %.
External inputs (paths) are flags or env vars: see xbench_jobs.default_paths() and CROSS_ENGINE.md "Re-running".
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import xbench_jobs as X

RESULTS = X.XB / "results"
SHIM = X.SCENES / "engine_shim.py"
MARKERS = ("XB_RESULT:", "BENCHMARK_RESULT:", "NEWTON_RESULT:")


def gpu_label() -> str:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                      encoding="utf-8", timeout=30)
        return out.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        return "no nvidia-smi (CPU host?)"


def dig(rec: dict, path: str):
    """Value at 'a/b' inside nested dicts, or a flat key that itself contains '/' (ManiSkill's flattened keys)."""
    if path in rec:
        return rec[path]
    cur = rec
    for part in path.split("/"):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def marker_records(text: str, marker: str) -> list[dict]:
    recs = []
    for line in text.splitlines():
        if marker in line:
            try:
                recs.append(json.loads(line.split(marker, 1)[1]))
            except json.JSONDecodeError:
                pass
    return recs


def prepare(job: X.Job, paths: X.Paths, work: Path) -> dict[str, str]:
    """Named preparation steps; returns extra environment for the job."""
    if job.prep == "mjx_panda":
        src = paths.alloy_refs / "mujoco_menagerie/franka_emika_panda"
        dst = work / "menagerie_mjxpanda"
        if not dst.exists():
            shutil.copytree(src, dst)
            shutil.copy(src / "mjx_panda.xml", dst / "panda.xml")
        return {"MENAGERIE_PANDA": str(dst)}
    return {}


def command(job: X.Job, engine: str, python: str, out_file: Path | None) -> list[str]:
    cmd = [python, str(SHIM), "--engine", engine, "--protocol", job.protocol, str(X.SCENES / job.script), *job.args]
    if job.out_flag and out_file is not None:
        cmd += [job.out_flag, str(out_file)]
    return cmd


def rows_from_run(job: X.Job, ctx: dict, text: str, out_file: Path | None, rc: int) -> list[dict]:
    """Normalise the job's output into xbench rows."""
    base = {"protocol": job.protocol, "benchmark": job.name, "engine": ctx["engine"], "version": ctx["version"],
            "hardware": ctx["hardware"], "envs": job.envs,
            "settings": {"script": f"scenes/{job.script}", "args": job.args, "tier": ctx["tier"]},
            "notes": f"run_all.py run_id={ctx['run_id']} rc={rc}" + (f"; {job.note}" if job.note else "")}
    if job.out_flag and out_file is not None and out_file.exists():
        rows = []
        for line in out_file.read_text().splitlines():
            rec = json.loads(line)
            if "metric_name" in rec:  # already an xbench row (newton, isaaclab)
                rows.append({**rec, "engine": ctx["engine"], "notes": f"{base['notes']}; {rec.get('notes', '')}"})
            else:  # one record per run (mujoco_warp)
                rows.append({**base, "metric_name": job.metric, "metric_value": dig(rec, job.metric or ""),
                             "unit": job.unit, "raw": rec})
        return rows
    recs = marker_records(text, job.marker) if job.marker else []
    if not recs:
        return [{**base, "metric_name": job.metric or "run", "metric_value": None, "unit": job.unit,
                 "notes": base["notes"] + "; FAILED or no result record (see log)"}]
    return [{**base, "metric_name": job.metric, "metric_value": dig(r, job.metric or ""), "unit": job.unit, "raw": r}
            for r in recs]


def run_job(job: X.Job, ctx: dict, paths: X.Paths, python: str) -> tuple[int, list[dict]]:
    logdir = RESULTS / "runall" / ctx["run_id"]
    logdir.mkdir(parents=True, exist_ok=True)
    out_file = logdir / f"{job.name}.out" if job.out_flag else None
    env = {**os.environ, **job.env, **prepare(job, paths, logdir), "PYTHONUNBUFFERED": "1"}
    cmd = command(job, ctx["engine"], python, out_file)
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=X.SCENES, env=env, capture_output=True, text=True, timeout=job.timeout_s)
        rc, text = proc.returncode, proc.stdout + "\n--- stderr ---\n" + proc.stderr
    except subprocess.TimeoutExpired as exc:
        rc, text = 124, f"{exc.stdout or ''}\n--- TIMEOUT after {job.timeout_s} s ---"
    (logdir / f"{job.name}.log").write_text(f"$ {' '.join(cmd)}\nwall {time.time() - t0:.1f} s rc {rc}\n{text}")
    engine_line = marker_records(text, "XB_ENGINE:")
    ctx = {**ctx, "version": json.dumps(engine_line[-1]) if engine_line else ctx["version"]}
    return rc, rows_from_run(job, ctx, text, out_file, rc)


def append_rows(protocol: str, engine: str, rows: list[dict]) -> Path:
    path = RESULTS / protocol / f"runall_{engine}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="xbench: re-run every protocol's Newton/Alloy scene")
    p.add_argument("--engine-under-test", choices=("newton", "alloy"), required=True)
    p.add_argument("--protocols", default=",".join(X.BUILDERS), help="comma list; default all")
    p.add_argument("--tier", choices=("headline", "full"), default="headline")
    p.add_argument("--only", default=None, help="regex on job names")
    p.add_argument("--repeats", type=int, default=1, help="repeat the whole job list (interleaved by round)")
    p.add_argument("--python", default=sys.executable, help="interpreter for the scene scripts (the Alloy venv)")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    p.add_argument("--list", action="store_true", help="list the jobs and exit")
    p.add_argument("--allow-cpu", action="store_true", help="run without nvidia-smi (smoke tests only; Warp on CPU)")
    for name in ("newton_src", "mjw_assets", "genesis_assets", "gsb", "maniskill_root", "alloy_refs"):
        p.add_argument("--" + name.replace("_", "-"), default=None, type=Path)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    defaults = X.default_paths()
    paths = X.Paths(**{k: getattr(a, k) or getattr(defaults, k) for k in X.Paths.__dataclass_fields__})
    protocols = [s for s in a.protocols.split(",") if s]
    unknown = set(protocols) - set(X.BUILDERS)
    if unknown:
        sys.exit(f"unknown protocol(s): {sorted(unknown)}; choose from {list(X.BUILDERS)}")
    jobs = [j for j in X.all_jobs(paths, protocols, a.tier) if not a.only or re.search(a.only, j.name)]
    engine = a.engine_under_test
    if a.list or a.dry_run:
        for j in jobs:
            out = Path("<out>") if j.out_flag else None
            print(f"{j.protocol:12s} {j.name:36s} " + (" ".join(command(j, engine, a.python, out)) if a.dry_run else ""))
        print(f"{len(jobs)} jobs", file=sys.stderr)
        return 0
    if not shutil.which("nvidia-smi") and not a.allow_cpu:
        sys.exit("no nvidia-smi: xbench timings need a CUDA GPU (use --allow-cpu for a smoke test only)")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{engine}"
    ctx = {"engine": engine, "version": "", "hardware": gpu_label(), "tier": a.tier, "run_id": run_id}
    failed_groups: set[str] = set()
    n_fail = 0
    for rnd in range(1, a.repeats + 1):
        for j in jobs:
            if j.group and (j.group, rnd) in failed_groups:
                continue
            rc, rows = run_job(j, {**ctx, "run_id": f"{run_id}/r{rnd}"}, paths, a.python)
            dest = append_rows(j.protocol, engine, rows)
            head = rows[0] if rows else {}
            print(f"[r{rnd}] {j.protocol:12s} {j.name:36s} rc={rc} {head.get('metric_name')}={head.get('metric_value')} -> "
                  f"{dest.relative_to(X.XB)}", flush=True)
            if rc == 3 and engine == "alloy":
                sys.exit("alloy.xbench adapter missing (scenes/engine_shim.py): Alloy column stays pending")
            if rc != 0:
                n_fail += 1
                if j.group:
                    failed_groups.add((j.group, rnd))
    print(f"done: {len(jobs) * a.repeats} jobs, {n_fail} failed; logs in results/runall/{run_id}/", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
