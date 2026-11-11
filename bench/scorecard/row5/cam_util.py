# SPDX-License-Identifier: Apache-2.0
"""Timing, GPU-memory sampling, row and PNG helpers shared by the row 5 harnesses (Alloy and MuJoCo Warp)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cam_scenes import OUTPUTS


ROW_FIELDS = (
    "engine", "engine_version", "scene", "outputs", "width", "height", "worlds", "status",
    "images_per_s", "images_per_s_min", "images_per_s_max", "render_only_images_per_s", "physics_only_steps_per_s",
    "gpu_mem_mib", "renderer", "render_settings", "control_dt_s", "physics_substeps", "gpu", "driver",
    "commit_or_version_pins", "command", "utc",
)  # the row format of PROTOCOL.md


def add_cell_arguments(ap: argparse.ArgumentParser, scenes: list[str]) -> None:
    """The command line shared by the Alloy and MuJoCo Warp cell scripts."""
    ap.add_argument("--scene", default="cartpole", choices=scenes)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--outputs", default="rgb", choices=list(OUTPUTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--png-dir", default="png")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--loops", type=int, default=50)
    ap.add_argument("--commit", default=os.environ.get("ALLOY_COMMIT", "unknown"))


def _smi(args: list[str]) -> str:
    try:
        return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_info() -> dict:
    out = _smi(["--query-gpu=name,driver_version", "--format=csv,noheader"]).strip().splitlines()
    name, _, drv = (out[0].partition(", ") if out else ("cpu", "", ""))
    return {"gpu": name, "driver": drv}


class MemSampler:
    """Samples this process's GPU memory (nvidia-smi compute-apps) and the whole GPU at ~2.5 Hz.

    ``peak_timed`` restarts at :meth:`mark`; ``peak_all`` covers the whole run including the build."""

    def __init__(self) -> None:
        self.pid = str(os.getpid())
        self.baseline = self._used_total()
        self.peak_all = self.peak_timed = 0
        self.peak_total_all = self.peak_total_timed = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def _used_total(self) -> int:
        out = _smi(["--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()
        return int(out[0]) if out else 0

    def _sample(self) -> None:
        mine = 0
        for ln in _smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).splitlines():
            pid, _, mem = ln.partition(",")
            if pid.strip() == self.pid and mem.strip().isdigit():
                mine = int(mem)
        total = self._used_total()
        self.peak_all, self.peak_timed = max(self.peak_all, mine), max(self.peak_timed, mine)
        self.peak_total_all, self.peak_total_timed = max(self.peak_total_all, total), max(self.peak_total_timed, total)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(0.4)

    def start(self) -> "MemSampler":
        self._t.start()
        return self

    def mark(self) -> None:
        self.peak_timed = self.peak_total_timed = 0

    def stop(self) -> dict:
        self._sample()
        self._stop.set()
        self._t.join(timeout=5)
        timed = self.peak_timed or max(self.peak_total_timed - self.baseline, 0)
        return {"gpu_mem_mib": timed, "gpu_mem_mib_incl_build": self.peak_all or max(self.peak_total_all - self.baseline, 0),
                "gpu_mem_whole_gpu_peak_mib": self.peak_total_timed, "gpu_idle_baseline_mib": self.baseline}


def time_loops(fn, sync, worlds: int, warmup: int = 20, repeats: int = 3, loops: int = 50, budget_s: float = 150.0,
               on_start=None) -> dict:
    """Median/min/max images per second over ``repeats`` x ``loops`` calls of ``fn`` (sync before each clock stop).

    If the estimated total exceeds ``budget_s``, falls back to 5 warm-up and 10 loops per repeat and says so."""
    for _ in range(2):  # compile, graph capture
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(3):
        fn()
    sync()
    est = (time.perf_counter() - t0) / 3
    reduced = est * (warmup + repeats * loops) > budget_s
    if reduced:
        warmup, loops = 5, 10
        if est * (warmup + repeats * loops) > 1.5 * budget_s:  # very slow cell: 3 repeats of 5 loops
            warmup, loops = 3, 5
    for _ in range(max(warmup - 5, 0)):
        fn()
    sync()
    if on_start:
        on_start()
    rates = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(loops):
            fn()
        sync()
        rates.append(worlds * loops / (time.perf_counter() - t0))
    return {"rate": statistics.median(rates), "min": min(rates), "max": max(rates), "repeats": repeats, "loops": loops,
            "warmup_loops": warmup, "reduced_loops": reduced, "est_loop_s": est}


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_row(path: str, row: dict) -> None:
    row.setdefault("utc", now_utc())
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def save_grid(path: str, imgs: np.ndarray) -> None:
    """imgs: (4, H, W, 3|1) uint8 -> one row PNG."""
    from PIL import Image
    if imgs.shape[-1] == 1:
        imgs = np.repeat(imgs, 3, axis=-1)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(list(imgs), axis=1)).save(path)


def sanity(first: np.ndarray, last: np.ndarray) -> str | None:
    """first/last: (4, H, W, C) uint8 of worlds 0..3 before/after the timed loops. None = fine, else the reason."""
    if last.mean() < 2.0 or last.std() < 1.0:
        return "image is blank or black"
    if np.array_equal(first[0], last[0]):
        return "world 0 image did not change over the timed loops (frozen)"
    if all(np.array_equal(last[0], last[i]) for i in range(1, len(last))):
        return "worlds 0..3 are identical after physics"
    return None


def is_oom(exc: BaseException) -> bool:
    s = str(exc).lower()
    return "out of memory" in s or "cuda error 2" in s or "allocation" in s and "fail" in s
