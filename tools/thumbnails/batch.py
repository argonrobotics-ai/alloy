# SPDX-License-Identifier: Apache-2.0
"""Resumable thumbnail batches: one JSON line per object in a state file; done objects are skipped on a rerun.

Example::

    summary = run_jobs(jobs, renderer, Path("state/ycb_gso.jsonl"))
    summary["ok"], summary["failed"]           # 1110, {"gso_x": "VisualError: ..."}
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from framing import frame
from libs import Job
from visuals import Part, bounds, load_visual

log = logging.getLogger("thumbnails")


class Renderer(Protocol):
    def render(self, parts: list[Part], view: Any) -> np.ndarray: ...


def read_state(path: Path) -> dict[str, dict]:
    """Last row per object name of a state file (empty if missing). Rows a crash left unreadable (the NUL-filled tail a
    VM preemption leaves, a half-written line) are skipped with a warning, so those objects render again."""
    rows: dict[str, dict] = {}
    if not Path(path).is_file():
        return rows
    for n, line in enumerate(Path(path).read_text(errors="replace").splitlines(), 1):
        line = line.replace("\x00", "").strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            log.warning("%s:%d: unreadable state row skipped", path, n)
            continue
        rows[r["name"]] = r
    return rows


def thumb_intact(row: dict[str, Any]) -> bool:
    """True when the row's thumbnail is on disk with the sha256 the row recorded (a crash can leave it truncated)."""
    thumb = Path(row.get("thumb", ""))
    return thumb.is_file() and hashlib.sha256(thumb.read_bytes()).hexdigest() == row.get("sha256")


def _start_on_new_line(state: Path) -> None:
    """Appended rows must not continue a crash-truncated last line."""
    if state.is_file() and state.stat().st_size:
        with open(state, "rb") as fh:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n":
                with open(state, "a", encoding="utf-8") as out:
                    out.write("\n")


def write_png(path: Path, img: np.ndarray) -> str:
    """Save an (H, W, 3) uint8 image as PNG (Pillow); returns its sha256."""
    from PIL import Image  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(np.asarray(img, np.uint8)).save(tmp, optimize=True)
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_one(job: Job, renderer: Renderer) -> dict[str, Any]:
    """Load, frame, render and write one thumbnail; the row records how (or why not)."""
    t0 = time.monotonic()
    row: dict[str, Any] = {"lib": job.lib, "name": job.name, "thumb": str(job.thumb)}
    try:
        if job.source is None or not job.source.is_file():
            raise FileNotFoundError(f"no render source ({job.source}); fetch the library's meshes first")
        parts = load_visual(job.source)
        lo, hi = bounds(parts)
        img = renderer.render(parts, frame(lo, hi))
        row.update(status="ok", sha256=write_png(job.thumb, img), source=job.source.name,
                   textured=any(p.texture is not None for p in parts), parts=len(parts),
                   glass=any(p.glass for p in parts), notes=sorted({p.note for p in parts if p.note}))
    except Exception as exc:  # noqa: BLE001 - every object's failure is recorded with its reason
        row.update(status="failed", reason=f"{type(exc).__name__}: {exc}"[:400])
    row["seconds"] = round(time.monotonic() - t0, 2)
    return row


def check_jobs(jobs: list[Job]) -> dict[str, Any]:
    """Load every job's visual without rendering (CPU): how many load, how many are textured, why the rest fail."""
    ok, textured, glass, failed = 0, 0, [], {}
    for job in jobs:
        try:
            if job.source is None or not job.source.is_file():
                raise FileNotFoundError(f"no render source ({job.source}); fetch the library's meshes first")
            parts = load_visual(job.source)
            ok += 1
            textured += any(p.texture is not None for p in parts)
            if any(p.glass for p in parts):
                glass.append(job.name)
        except Exception as exc:  # noqa: BLE001 - every object's failure is recorded with its reason
            failed[job.name] = f"{type(exc).__name__}: {exc}"[:400]
    return {"jobs": len(jobs), "loads": ok, "textured": textured, "glass": glass, "failed": failed}


def run_jobs(jobs: list[Job], renderer: Renderer, state: Path, redo: bool = False,
             limit: int | None = None) -> dict[str, Any]:
    """Render every job not yet ``ok`` in ``state`` with its thumbnail intact (all with ``redo``); append one row per
    attempt."""
    state.parent.mkdir(parents=True, exist_ok=True)
    rows = read_state(state) if not redo else {}
    done = {k for k, r in rows.items() if r.get("status") == "ok" and thumb_intact(r)}
    todo = [j for j in jobs if j.name not in done][:limit]
    _start_on_new_line(state)
    with open(state, "a", encoding="utf-8") as fh:
        for i, job in enumerate(todo):
            row = render_one(job, renderer)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if i % 25 == 0 or row["status"] != "ok":
                log.info("%d/%d %s %s %s", i + 1, len(todo), job.name, row["status"], row.get("reason", ""))
    rows = read_state(state)
    mine = [rows[j.name] for j in jobs if j.name in rows]
    return {"jobs": len(jobs), "rendered_now": len(todo), "ok": sum(r["status"] == "ok" for r in mine),
            "textured": sum(bool(r.get("textured")) for r in mine if r["status"] == "ok"),
            "glass_opaque": sorted(r["name"] for r in mine if r.get("glass")),
            "simplified": {r["name"]: r["notes"] for r in mine if r.get("notes")},
            "failed": {r["name"]: r.get("reason", "") for r in mine if r["status"] != "ok"},
            "missing": len(jobs) - len(mine)}
