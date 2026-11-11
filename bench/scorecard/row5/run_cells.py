# SPDX-License-Identifier: Apache-2.0
"""Runs every row 5 cell of one engine, one process per cell, appending rows to RAW/<engine>_g4.jsonl.

  python bench/scorecard/row5/run_cells.py --engine alloy --raw raw          # all Alloy cells (about 4 h on one GPU)
  python bench/scorecard/row5/run_cells.py --engine mujoco_warp --raw raw    # all MuJoCo Warp cells (about 25 min)

Resumable: a cell that already has a row is skipped. A cell that dies or times out gets a row with its status.
``--smoke`` runs one tiny cell per kind (64 px, 2 worlds, 3 loops) to check the setup.
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cam_scenes as S

log = logging.getLogger("row5.cells")
HERE = Path(__file__).resolve().parent
SCRIPTS = {"alloy": "alloy_cam.py", "mujoco_warp": "mujoco_warp_cam.py"}
SMOKE_ARGS = ["--warmup", "5", "--repeats", "1", "--loops", "3"]


@dataclass(frozen=True)
class Cell:
    engine: str
    scene: str
    outputs: str
    size: int
    worlds: int
    setting: str | None = None

    @property
    def key(self) -> tuple:
        return (self.scene, self.outputs, self.size, self.worlds, self.setting)


def _grid(smoke: bool) -> list[tuple[str, int, int]]:
    sizes, worlds = ((64,), (2,)) if smoke else (S.SIZES, S.WORLDS)
    return sorted(itertools.product(S.SCENE_NAMES, sizes, worlds), key=lambda c: (c[0], c[1] * c[1] * c[2]))


def alloy_cells(smoke: bool, optix: bool) -> Iterator[Cell]:
    """Settings fast (1 spp), then default (16 spp); the t12 extra rows follow the fast cells; OptiX cells last."""
    grid = _grid(smoke)
    for setting in ("fast", "default"):
        for out in S.OUTPUTS:
            yield from (Cell("alloy", sc, out, sz, w, setting) for sc, sz, w in grid)
        if setting == "fast":
            yield from (Cell("alloy", "t12", "rgb", 128, w, "fast") for w in ((2,) if smoke else S.WORLDS))
    if optix:
        for out in S.OUTPUTS:
            yield from (Cell("alloy", sc, out, sz, w, "fast_optix") for sc, sz, w in grid)


def mujoco_warp_cells(smoke: bool) -> Iterator[Cell]:
    for out in S.OUTPUTS:
        yield from (Cell("mujoco_warp", sc, out, sz, w) for sc, sz, w in _grid(smoke))


def done_keys(path: Path) -> set:
    if not path.exists():
        return set()
    rows = (json.loads(ln) for ln in path.read_text().splitlines() if ln.strip().startswith("{"))
    return {(r.get("scene"), r.get("outputs"), r.get("width"), r.get("worlds"), r.get("setting")) for r in rows}


def cell_command(cell: Cell, path: Path, raw: Path, extra: list[str]) -> list[str]:
    cmd = [sys.executable, str(HERE / SCRIPTS[cell.engine]), "--scene", cell.scene, "--size", str(cell.size),
           "--worlds", str(cell.worlds), "--outputs", cell.outputs, "--out", str(path), "--png-dir", str(raw / "png"), *extra]
    return cmd + (["--setting", cell.setting] if cell.setting else [])


def failure_row(cell: Cell, rc: int, why: str) -> dict:
    status = "oom" if "out of memory" in why.lower() else ("timeout" if rc == -1 else "error")
    return {"engine": cell.engine, "scene": cell.scene, "outputs": cell.outputs, "width": cell.size, "height": cell.size,
            "worlds": cell.worlds, "setting": cell.setting, "status": status, "reason": why,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def run_one(cell: Cell, path: Path, raw: Path, extra: list[str], timeout: int) -> int:
    cmd = cell_command(cell, path, raw, extra)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        rc, why = -1, f"timeout after {timeout} s"
    else:
        (raw / "logs").mkdir(exist_ok=True)
        name = f"{cell.engine}_{cell.scene}_{cell.outputs}_{cell.size}_{cell.worlds}_{cell.setting}.log"
        (raw / "logs" / name).write_text(p.stdout[-4000:] + p.stderr[-6000:])
        rc, why = p.returncode, f"exit {p.returncode}: {p.stderr[-300:]}"
    if cell.key not in done_keys(path):  # the process died before writing its row
        with open(path, "a") as f:
            f.write(json.dumps(failure_row(cell, rc, why)) + "\n")
    return rc


def run_all(cells: Iterator[Cell], raw: Path, smoke: bool, commit: str | None) -> None:
    extra = (SMOKE_ARGS if smoke else []) + (["--commit", commit] if commit else [])
    timeout = 120 if smoke else 600
    for cell in cells:
        path = raw / f"{cell.engine}_g4.jsonl"
        if cell.key in done_keys(path):
            continue
        t0 = time.time()
        rc = run_one(cell, path, raw, extra, timeout)
        log.info("%s %s %s %d px %d worlds %s rc=%d %.0fs", cell.engine, cell.scene, cell.outputs, cell.size,
                 cell.worlds, cell.setting, rc, time.time() - t0)
    log.info("ALL CELLS DONE")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", required=True, choices=sorted(SCRIPTS))
    ap.add_argument("--raw", required=True, type=Path, help="output folder: rows, png/, logs/")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--optix", action="store_true", help="Alloy: add the OptiX cells (after `python -m alloy.camera.optix build`)")
    ap.add_argument("--commit", default=None, help="commit recorded in each row (default: the ALLOY_COMMIT environment variable)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    a.raw.mkdir(parents=True, exist_ok=True)
    cells = alloy_cells(a.smoke, a.optix) if a.engine == "alloy" else mujoco_warp_cells(a.smoke)
    run_all(cells, a.raw, a.smoke, a.commit)


if __name__ == "__main__":
    main()
