#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fine-tune the denoiser for one scene in minutes: render teacher frames, tune, gate, keep it only if it helps.

    python tools/denoiser/finetune_scene.py --bundle BUNDLE_DIR --look LOOK.json --camera wide --work WORK \\
        [--base default] [--frames 40] [--gate-frames 6] [--steps 2000] [--blender blender] [--res 128x128]

1. Picks ``--gate-frames`` evenly spaced frames to hold out and ``--frames`` training frames at least 60 frames
   away from them.
2. Renders both with Blender Cycles (``blender/render_bundle.py``): training inputs at 1-16 spp (two seeds) with a
   4,096-spp target; gate inputs at 1-128 spp with Cycles' own OIDN output kept as the baseline.
3. Fine-tunes ``--base`` on the training frames in the scene's own display space (``finetune.finetune``).
4. Scores base, tuned and OIDN on the gate frames (per-frame G1: FLIP and LPIPS within 2x of Cycles' 128-spp +
   OIDN), and writes ``WORK/report.json`` plus ``WORK/tuned.pt``. Use the tuned weights only at the spp where the
   report says ``"use": "tuned"``.
Prints the report as JSON. Needs Blender 4.5, torch, OpenEXR, lpips and flip-evaluator.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as E
import jobs as J
import run_jobs as R
from dataset import PackedIndex

from alloy.camera.denoiser import load
from alloy.camera.denoiser.finetune import PairSet, finetune

log = logging.getLogger("denoiser.finetune_scene")
TRAIN_INPUTS = "11:1,2,4,8,16;12:1,4,16"


def pick(count: int, n_gate: int, n_train: int, seed: int = 0,
         gate: list[int] | None = None) -> tuple[list[int], list[int]]:
    """Gate frames (given, or evenly spaced), then seeded training frames 60+ frames away from them."""
    gate = gate or [int(x) for x in np.linspace(0.1 * count, 0.9 * count, n_gate)]
    train = J.pick_frames(count, n_train, random.Random(seed), gate, 60)
    return gate, train


def render(a: argparse.Namespace, frames: list[int], name: str, split: str) -> Path:
    """Render and pack one job; returns the packed folder."""
    job = {"id": name, "bundle": "scene", "bundle_dir": a.bundle, "look": a.look, "camera": a.camera,
           "frames": frames, "res": a.res, "split": split, "weight": 1.0, "blender_args": [],
           "inputs": J.EVAL_INPUTS if split == "eval" else TRAIN_INPUTS}
    ns = argparse.Namespace(work=str(Path(a.work) / "render"), out=str(Path(a.work) / f"packed_{split}"),
                            blender=a.blender, upload=None, skip_if=None)
    stage, code = R.render_job(job, ns)
    if code:
        raise RuntimeError(f"{stage} failed for {name} (exit {code}); see {ns.work}/{name}.log")
    return Path(ns.out)


def pairs_from(packed: Path, res: str) -> tuple[PairSet, torch.Tensor]:
    """The packed training job as a PairSet, plus its display LUT stack."""
    index = PackedIndex.scan([packed])
    job = index.by_res[res][0]
    a = job.arrays
    ps = PairSet.from_numpy(a["noisy"], a["albedo"], a["normal"], a["ref"], [s for _, s in job.keys])
    return ps, index.display(ps.rgb.device)


def gate(a: argparse.Namespace, packed: Path, tuned: Path) -> dict:
    """Per spp: frames passing and worst ratio for base, tuned and OIDN; which one to use."""
    index = PackedIndex.scan([packed])
    disp, scorer = index.display("cuda"), E.Scorer()
    models = {"base": (load(a.base), 1), "tuned": (load(str(tuned)), 1)}
    rows = []
    for job in index.by_res[a.res]:
        for f in range(job.frames):
            rows += E.frame_rows(job, f, a.res, models, disp, scorer, 128)
    out = {}
    for spp in sorted({r["spp"] for r in rows}):
        per = {m: [r["ratio"] for r in rows if r["spp"] == spp and r["method"] == m] for m in models.keys() | {"oidn"}}
        best = min(per, key=lambda m: float(np.mean(per[m])))
        out[spp] = {**{m: {"pass": sum(x <= 1 for x in v), "of": len(v), "worst": round(max(v), 3)}
                       for m, v in per.items()}, "use": best}
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True)
    p.add_argument("--look", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--base", default="default")
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--gate-frames", dest="gate_frames", type=int, default=6)
    p.add_argument("--gate", default=None, help="explicit gate frames, e.g. 100,300,500 (overrides --gate-frames)")
    p.add_argument("--skip-gate", dest="skip_gate", action="store_true", help="tune only; score elsewhere")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--res", default="128x128")
    p.add_argument("--blender", default="blender")
    a = p.parse_args(argv)
    explicit = [int(x) for x in a.gate.split(",")] if a.gate else None
    gate_f, train_f = pick(J.frame_count(a.bundle), a.gate_frames, a.frames, gate=explicit)
    t0 = time.perf_counter()
    packed_train = render(a, train_f, "scene_train", "train")
    t_render = time.perf_counter() - t0
    pairs, display = pairs_from(packed_train, a.res)
    tuned = finetune(a.base, pairs, Path(a.work) / "tuned.pt", steps=a.steps, display=display)
    report = {"base": a.base, "tuned": str(tuned), "gate_frames": gate_f, "train_frames": len(train_f),
              "render_s": round(t_render, 1), "tune_s": round(time.perf_counter() - t0 - t_render, 1)}
    if not a.skip_gate:
        report["per_spp"] = gate(a, render(a, gate_f, "scene_gate", "eval"), tuned)
    (Path(a.work) / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
