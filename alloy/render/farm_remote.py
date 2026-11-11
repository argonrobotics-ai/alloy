# SPDX-License-Identifier: Apache-2.0
"""Remote command construction for the farm: Blender invocations and the background batch script."""
from __future__ import annotations

import shlex
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alloy.render.farm_types import FarmJob

REMOTE_ROOT = "alloy_render"
BLENDER = "/opt/blender/blender"
CYCLES_REMOTE = ".cache/cycles"
# One directory holds both the OptiX shader cache and the CUDA JIT cache, so it can be copied between VMs.
CACHE_ENV = (f"export OPTIX_CACHE_PATH=$HOME/{CYCLES_REMOTE}/optix CUDA_CACHE_PATH=$HOME/{CYCLES_REMOTE}/nv "
             "CUDA_CACHE_MAXSIZE=2147483647; ")
FD_LIMIT = "ulimit -n 65536 2>/dev/null || ulimit -n 4096; "  # 1,024-world bundles have thousands of arrays


class RemoteLayout:
    """Home-relative paths of one run on a VM."""

    def __init__(self, job_name: str, run_id: int) -> None:
        self.job = f"{REMOTE_ROOT}/{job_name}"
        self.bundle = f"{self.job}/bundle"
        self.run_dir = f"{self.job}/{run_id}"
        self.look = f"{self.run_dir}/look.json"
        self.out = f"$HOME/{self.run_dir}/out"


def blender_cmd(job: "FarmJob", lay: RemoteLayout, run: tuple[int, int]) -> str:
    """One Blender invocation for frames run[0]..run[1] inclusive (the Blender side takes B exclusive)."""
    parts = [BLENDER, "-b", "--python-exit-code", "1", "-P", f"$HOME/{REMOTE_ROOT}/blender/build.py", "--",
             "--bundle", f"$HOME/{lay.bundle}", "--look", f"$HOME/{lay.look}", "--camera", shlex.quote(job.camera),
             "--out", lay.out, "--frames", f"{run[0]}:{run[1] + 1}", "--format", job.fmt, "--device", "OPTIX",
             "--assets", f"$HOME/{REMOTE_ROOT}/assets"]
    if job.samples:
        parts += ["--samples", str(job.samples)]
    return " ".join(parts)


def uploader(out: str, prefix: str) -> str:
    """Background loop (GCS mode): push finished frames to `prefix` every 10 s and mark each pushed frame under
    OUT/uploaded; after Blender writes OUT/rc it makes one last pass (retried) and exits. Dot files (Blender's
    in-progress .NNNNN.part.png) are never pushed: a truncated PNG in the stream silently cuts the encoded clip."""
    push = (f"ls {out}/frames > {out}/.snap 2>/dev/null; gcloud storage rsync -q -x '(^|/)[.]' {out}/frames {prefix} > /dev/null 2>&1 "
            f"&& (cd {out}/uploaded && xargs -r touch < {out}/.snap)")
    return (f"while [ ! -e {out}/rc ]; do {push}; sleep 10; done; "
            f"for i in 1 2 3 4 5; do {push} && break; sleep 15; done")


def remote_script(job: "FarmJob", lay: RemoteLayout, runs: list[tuple[int, int]]) -> str:
    """Start the runs in the background (nohup); `done` receives the worst exit code when they finish. With
    job.frames_gcs, the VM uploads its own frames and `done` is written only after the last one is in GCS."""
    cmds = "; ".join(f"{blender_cmd(job, lay, r)} || rc=$?" for r in runs)
    gcs = getattr(job, "frames_gcs", None)
    if gcs:
        up = uploader(lay.out, gcs.rstrip("/"))
        inner = (f"{FD_LIMIT}{CACHE_ENV}rc=0; ({up}) & up=$!; {cmds}; echo $rc > {lay.out}/rc; wait $up; "
                 f"echo $rc > {lay.out}/done")
    else:
        inner = f"{FD_LIMIT}{CACHE_ENV}rc=0; {cmds}; echo $rc > {lay.out}/done"
    return (f"mkdir -p {lay.out}/frames {lay.out}/uploaded && rm -f {lay.out}/done {lay.out}/rc; "
            f"(nohup bash -c {shlex.quote(inner)} > {lay.out}/blender.log 2>&1 < /dev/null &); echo started")


POLL_CMD = ("(cd {out}/frames 2>/dev/null && find . -type f -mmin +0.1 -printf '%f\\n'); "
            "echo ::DONE; cat {out}/done 2>/dev/null")


def poll_cmd(out: str, sub: str = "frames") -> str:
    """List finished frames (sub = frames; files a few seconds old) or frames already in GCS (sub = uploaded:
    marks are written after the upload, so no age filter), then the exit code once the batch is done."""
    if sub == "frames":
        return POLL_CMD.format(out=out)
    return (f"(cd {out}/{sub} 2>/dev/null && find . -type f -printf '%f\\n'); "
            f"echo ::DONE; cat {out}/done 2>/dev/null")


# The last error lines of a failed run, so the farm log says why Blender exited (e.g. DeviceError: no OptiX).
BLENDER_ERR_CMD = "grep -E 'ERROR|Error|Traceback' {out}/blender.log 2>/dev/null | tail -2 | cut -c1-300"


def parse_poll(text: str) -> tuple[list[str], str]:
    """(finished frame file names, exit code or '') from the combined poll command output."""
    head, _, tail = text.partition("::DONE")
    return head.split(), tail.strip()
