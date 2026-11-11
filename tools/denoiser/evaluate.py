#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Per-frame G1 check of denoisers on packed eval jobs, against Cycles' own OIDN, at every input spp.

    python tools/denoiser/evaluate.py --eval /data/packed_eval --weights default,v1.pt [--tta 1,2] \\
        [--bar-spp 128] [--out rows.jsonl] [--table table.md]

For every eval frame: the reference (high spp + OIDN) and each candidate go through the scene's display LUT, are
rounded to 8 bits, and are scored with FLIP (mean, LDR) and LPIPS-alex. The frame's G1 limit is 2x the scores of
Cycles' ``--bar-spp`` input + OIDN (the shipping setting), per metric; a candidate passes when both metrics are
within. The table lists, per (resolution, job, method), frames passing and the worst ratio at each spp, and the
fewest spp at which every frame passes. Needs ``flip-evaluator`` and ``lpips``.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import JobRes, PackedIndex
from display import Display

from alloy.camera.denoiser import load
from alloy.camera.denoiser.training import DisplayLUT

log = logging.getLogger("denoiser.evaluate")


def to_u8(disp: torch.Tensor) -> np.ndarray:
    """Display-encoded [0, 1] (1, H, W, 3) -> (H, W, 3) uint8, round half up, no dither."""
    return (disp[0].clamp(0, 1) * 255 + 0.5).floor().to(torch.uint8).cpu().numpy()


class Scorer:
    """FLIP mean + LPIPS-alex on 8-bit display images. Example: ``Scorer()(ref_u8, test_u8)``."""

    def __init__(self) -> None:
        import lpips
        self.lp = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    @torch.no_grad()
    def __call__(self, ref: np.ndarray, test: np.ndarray) -> tuple[float, float]:
        import flip_evaluator as flip
        _, mean, _ = flip.evaluate(ref.astype(np.float32) / 255, test.astype(np.float32) / 255, "LDR",
                                   inputsRGB=True, applyMagma=False, computeMeanError=True)
        t = lambda a: torch.from_numpy(a).permute(2, 0, 1)[None].float().div(127.5).sub(1).cuda()
        return float(mean), float(self.lp(t(ref), t(test)))


def show(disp, x: torch.Tensor, idx: torch.Tensor) -> np.ndarray:
    """(1, H, W, 3) linear -> uint8 through ``disp``: a DisplayLUT stack (training LUT) or an exact ``Display``."""
    if isinstance(disp, DisplayLUT):
        return to_u8(disp.apply(x, idx))
    return disp(x.float().cpu().numpy())


def frame_rows(job: JobRes, f: int, res: str, models: dict, disp, scorer: Scorer, bar_spp: int) -> list[dict]:
    """Score every method and spp on one frame; the limit comes from the bar_spp OIDN input."""
    a, idx = job.arrays, torch.tensor([job.lut], device="cuda")
    cuda = lambda x: torch.from_numpy(np.asarray(x, np.float32)).cuda()
    disp = Display.from_job(job.path.parent) if disp == "exact" else disp
    ref = show(disp, cuda(a["ref"][f:f + 1]), idx)
    kbar = next((k for k, key in enumerate(job.keys) if key[1] == bar_spp), None)
    if kbar is None or "oidn" not in a:
        raise ValueError(f"{job.path}: no {bar_spp}-spp OIDN input to set the G1 limit; render eval jobs with "
                         f"jobs.py's eval inputs")
    bar = scorer(ref, show(disp, cuda(a["oidn"][f:f + 1, kbar]), idx))
    rows = []
    for k, (_, spp) in enumerate(job.keys):
        ins = [cuda(a[n][f:f + 1, k]) for n in ("noisy", "albedo", "normal")]
        cands = {"oidn": cuda(a["oidn"][f:f + 1, k])}
        cands.update({m: torch.as_tensor(dn.denoise(*ins, spp=spp, tta=tta)).cuda() for m, (dn, tta) in models.items()})
        for m, y in cands.items():
            fl, lp = scorer(ref, show(disp, y, idx))
            rows.append({"res": res, "job": job.path.parent.name, "frame": f, "spp": spp, "method": m, "flip": fl,
                         "lpips": lp, "ratio": max(fl / (2 * bar[0]), lp / (2 * bar[1])), "bar": bar})
    return rows


def summarize(rows: list[dict]) -> list[str]:
    """Markdown: per (res, job, method) the frames passing and worst ratio per spp, and the fewest passing spp."""
    acc = defaultdict(list)
    for r in rows:
        acc[(r["res"], r["job"], r["method"], r["spp"])].append(r["ratio"])
    lines = ["| res | job | method | spp: passing (worst ratio) | fewest spp, all frames pass |", "|---|---|---|---|---|"]
    for key in sorted({k[:3] for k in acc}):
        spps = sorted(k[3] for k in acc if k[:3] == key)
        cells = [f"{s}: {sum(x <= 1 for x in acc[(*key, s)])}/{len(acc[(*key, s)])} ({max(acc[(*key, s)]):.2f})"
                 for s in spps]
        ok = [s for s in spps if max(acc[(*key, s)]) <= 1]
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {'; '.join(cells)} | {min(ok) if ok else '-'} |")
    return lines


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True, help="comma list of packed eval-job folders")
    p.add_argument("--weights", required=True, help="comma list of weights (path, name or default[:res])")
    p.add_argument("--tta", default="1", help="comma list of tta settings to score, e.g. 1,2")
    p.add_argument("--bar-spp", dest="bar_spp", type=int, default=128)
    p.add_argument("--only", default=None, help="only jobs whose id contains this")
    p.add_argument("--exact", action="store_true", help="exact display (OCIO / sRGB, display.py) instead of the LUT")
    p.add_argument("--out", default=None)
    p.add_argument("--table", default=None)
    a = p.parse_args(argv)
    index = PackedIndex.scan(a.eval.split(","), split=None)
    disp, scorer = ("exact" if a.exact else index.display("cuda")), Scorer()
    models = {f"{Path(w).stem}@tta{t}": (load(w), int(t)) for w in a.weights.split(",") for t in a.tta.split(",")}
    rows = []
    for res, jobs in sorted(index.by_res.items()):
        for job in jobs:
            if a.only and a.only not in job.path.parent.name:
                continue
            for f in range(job.frames):
                rows += frame_rows(job, f, res, models, disp, scorer, a.bar_spp)
    if a.out:
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    table = "\n".join(summarize(rows))
    if a.table:
        Path(a.table).write_text(table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
