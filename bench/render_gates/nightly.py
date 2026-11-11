# SPDX-License-Identifier: Apache-2.0
"""Nightly render-gate run on one GPU: fetch references, render the engine, score, write the matrix.

Steps, all into one work folder:

1. fetch the Cycles reference PNG tars for the chosen scenes and sizes (``refs.fetch``);
2. render the engine images into the same layout by running ``--render`` once per scene and size, a command template
   with ``{scene} {res} {frames} {root} {pack} {gpu}`` placeholders (the camera's own gate renderer; ``{pack}`` is
   the scene pack fetched to ``<work>/packs/<scene>``);
3. score every engine (``score.score_root``) and write ``report_<gpu>.json``;
4. merge with speed rows into ``matrix_<gpu>.json`` and ``matrix_<gpu>.md``.

Exit code: 0 when no cell FAILS (PENDING allowed unless ``strict``), 1 otherwise.

Example:
    >>> cfg = NightlyConfig(work=Path("gate_work"), gpu="l4", render="python -m my_camera.gate --scene {scene} "
    ...                     "--res {res} --frames {frames} --out {root}")
    >>> run_nightly(cfg)  # doctest: +SKIP
    {'counts': {'PASS': 9, 'FAIL': 2, 'PENDING': 4}, ...}
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import matrix, refs, score
from .metrics import Scorer

log = logging.getLogger(__name__)


@dataclass
class NightlyConfig:
    """Inputs of one nightly run (see module docstring)."""

    work: Path
    gpu: str
    render: str | None = None
    scenes: list[str] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)
    uri: str | None = None
    rows: Path | None = None
    strict: bool = False
    skip_fetch: bool = False


def render_all(cfg: NightlyConfig, spec: dict, root: Path) -> list[dict]:
    """Run the render template per scene and size; returns one log row per call (rc, seconds, command)."""
    out = []
    for scene in cfg.scenes or list(spec["scenes"]):
        frames = ",".join(spec["scenes"][scene]["frames"])
        for res in cfg.sizes or spec["sizes"]:
            cmd = cfg.render.format(scene=scene, res=res, frames=frames, root=root, gpu=cfg.gpu,
                                    pack=cfg.work / "packs" / scene)
            log.info("render %s %s: %s", scene, res, cmd)
            r = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
            if r.returncode:
                log.error("render %s %s failed (rc %d): %s", scene, res, r.returncode, r.stderr[-2000:])
            out.append({"scene": scene, "res": res, "rc": r.returncode, "stderr_tail": r.stderr[-500:]})
    return out


def read_rows(path: Path | None) -> list[dict]:
    """Speed rows from a JSONL file (missing file -> no rows).

    Example:
        >>> read_rows(None)
        []
    """
    if not path or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip().startswith("{")]


def run_nightly(cfg: NightlyConfig) -> dict:
    """Run steps 1-4 and return the matrix (also written to the work folder)."""
    spec = matrix.load_spec()
    root = cfg.work / "root"
    root.mkdir(parents=True, exist_ok=True)
    if not cfg.skip_fetch:
        man = refs.load_manifest()
        refs.fetch(refs.select(man, cfg.scenes or None, cfg.sizes or None), root, cfg.uri)
        refs.fetch(refs.select(man, cfg.scenes or None, kind="pack"), cfg.work / "packs", cfg.uri, cache=root / ".refs_cache")
    renders = render_all(cfg, spec, root) if cfg.render else []
    report = score.score_root(root, Scorer(cfg.work / "metric_cache"), gpu=cfg.gpu, meta={"renders": renders})
    (cfg.work / f"report_{cfg.gpu}.json").write_text(json.dumps(report, indent=1))
    m = matrix.build_matrix(spec, {cfg.gpu: report}, read_rows(cfg.rows))
    m["cells"] = [c for c in m["cells"] if c["gpu"] == cfg.gpu]
    m["gpus"] = [cfg.gpu]
    (cfg.work / f"matrix_{cfg.gpu}.json").write_text(json.dumps(m, indent=1))
    (cfg.work / f"matrix_{cfg.gpu}.md").write_text(matrix.to_markdown(m, spec))
    return m


def exit_code(m: dict, strict: bool) -> int:
    """0 when nothing FAILS (and, with strict, nothing is PENDING).

    Example:
        >>> exit_code({"counts": {"PASS": 3, "FAIL": 0, "PENDING": 1}}, strict=False)
        0
    """
    c = m["counts"]
    return 1 if c["FAIL"] or (strict and c["PENDING"]) else 0
