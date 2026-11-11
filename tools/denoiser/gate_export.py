#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export denoised eval frames into the render-gates layout, so ``python -m bench.render_gates score`` grades them.

    python tools/denoiser/gate_export.py --eval /data/packed_eval --root GATE_ROOT --weights v1.pt \\
        [--tta 1,2] [--map c2gate=c2,c2gate720=c2,c3eval=c3,c4eval=c4,dc5gate=dc5] [--check]

For every packed eval job whose id starts with a ``--map`` prefix, every frame, every input spp and every
``--weights`` x ``--tta``, the denoised frame goes through the scene's exact display transform (``display.py``: view,
look and white balance as Blender applies them), is rounded to 8 bits and written to
``ROOT/<scene>/<frame>/<engine>/<W>x<H>_<spp>spp_on.png`` with engine ``dn_<weights stem>_tta<k>``; Cycles' own
OIDN output of the same input goes to engine ``dn_oidn``. The references and control seeds come from
``bench.render_gates fetch``. ``--check`` compares this display path with the references' own PNGs (the Cycles
control seed 1 at 128 spp) and reports the code-value difference. With ``--passes-jobs scene=JOB_DIR`` the inputs
are instead the gate set's own control renders (``fetch --kind passes``: noisy, albedo and normal of seed ``--seed``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[2]))  # the repo root, for bench.render_gates
from dataset import JobRes, PackedIndex
from display import Display

from alloy.camera.denoiser import load

log = logging.getLogger("denoiser.gate_export")
DEFAULT_MAP = "c2gate=c2,c2gate720=c2,c3eval=c3,c3gate720=c3,c4eval=c4,c4gate720=c4,dc5gate=dc5,dc5gate720=dc5"


def write_png(path: Path, rgb: np.ndarray) -> None:
    """(H, W, 3) uint8 -> PNG (Pillow if present, else the render-gates writer)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(rgb).save(path)
    except ImportError:
        from bench.render_gates.pngio import write_png as w
        w(path, rgb)


def scene_of(job_id: str, mapping: dict[str, str]) -> str | None:
    """The gate scene for an eval job id (longest matching prefix), or None."""
    hits = [p for p in mapping if job_id.startswith(p + "_")]
    return mapping[max(hits, key=len)] if hits else None


def export_job(job: JobRes, res: str, scene: str, models: dict, root: Path) -> int:
    """Write every frame x spp x engine of one job (exact display transform); returns the number of PNGs."""
    meta = json.loads((job.path.parent / "meta.json").read_text())
    frames, a = meta["res"][res]["frames"], job.arrays
    disp, n = Display.from_job(job.path.parent), 0
    cuda = lambda x: torch.from_numpy(np.asarray(x, np.float32)).cuda()
    for i, frame in enumerate(frames):
        for k, (_, spp) in enumerate(job.keys):
            ins = [cuda(a[nm][i:i + 1, k]) for nm in ("noisy", "albedo", "normal")]
            outs = {"dn_oidn": cuda(a["oidn"][i:i + 1, k])} if "oidn" in a else {}
            outs.update({m: torch.as_tensor(dn.denoise(*ins, spp=spp, tta=t)).cuda() for m, (dn, t) in models.items()})
            for eng, y in outs.items():
                write_png(root / scene / str(frame) / eng / f"{res}_{spp}spp_on.png", disp(y.cpu().numpy()))
                n += 1
    return n


def check_display(job: JobRes, res: str, scene: str, root: Path) -> dict | None:
    """Our display transform on the reference set's own seed-1 128-spp OIDN EXR vs its PNG: max / mean code difference."""
    meta = json.loads((job.path.parent / "meta.json").read_text())
    frame = meta["res"][res]["frames"][0]
    exr = root / scene / str(frame) / "cycles_s1" / f"{res}_128spp_on.exr"
    png = exr.with_suffix(".png")
    if not (exr.exists() and png.exists()):
        return None
    import OpenEXR

    from bench.render_gates.pngio import read_png
    with OpenEXR.File(str(exr), separate_channels=True) as f:
        ch = f.parts[0].channels
        lin = np.stack([ch[c].pixels.astype(np.float32) for c in "RGB"], -1)[None]
    ours = Display.from_job(job.path.parent)(lin).astype(int)
    d = np.abs(ours - read_png(png).astype(int))
    return {"scene": scene, "res": res, "frame": frame, "max_code_diff": int(d.max()), "mean": round(float(d.mean()), 3)}


def read_passes(path: Path) -> list[torch.Tensor]:
    """A render-gates ``*_off_passes.exr`` -> [noisy, albedo, normal] as (1, H, W, 3) float32 CUDA tensors."""
    import OpenEXR
    with OpenEXR.File(str(path), separate_channels=True) as f:
        ch = f.parts[0].channels
        get = lambda layer, comps: np.stack([ch[f"ViewLayer.{layer}.{c}"].pixels.astype(np.float32)
                                             for c in comps], -1)
        arrs = [get("Combined", "RGB"), get("Denoising Albedo", "RGB"), get("Denoising Normal", "XYZ")]
    return [torch.from_numpy(x)[None].cuda() for x in arrs]


def export_passes(root: Path, jobs: dict[str, Path], models: dict, seed: int) -> int:
    """Denoise the gate set's own control inputs (seed ``seed``, every spp, every size) of the scenes in ``jobs``
    (scene -> a packed job folder with the same view and white balance); returns the number of PNGs written."""
    n = 0
    for scene, job_dir in jobs.items():
        disp = Display.from_job(job_dir)
        for exr in sorted((root / scene).glob(f"*/cycles_s{seed}/*_off_passes.exr")):
            res, spp = exr.name.split("_")[0], int(exr.name.split("_")[1].replace("spp", ""))
            ins = read_passes(exr)
            for eng, (dn, t) in models.items():
                y = torch.as_tensor(dn.denoise(*ins, spp=spp, tta=t)).cuda()
                write_png(exr.parent.parent / eng / f"{res}_{spp}spp_on.png", disp(y.cpu().numpy()))
                n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--weights", required=True, help="comma list (path, name or default[:res])")
    p.add_argument("--tta", default="1")
    p.add_argument("--map", default=DEFAULT_MAP, help="job-id prefix=gate scene, comma list")
    p.add_argument("--check", action="store_true")
    p.add_argument("--passes-jobs", dest="passes_jobs", default=None,
                   help="scene=packed_job_dir,...: denoise the gate set's own control inputs (display from that job)")
    p.add_argument("--seed", type=int, default=2, help="control seed whose inputs --passes-luts denoises")
    a = p.parse_args(argv)
    if a.passes_jobs:
        models = {f"dn_{Path(w).stem.replace(':', '_')}_tta{t}": (load(w), int(t))
                  for w in a.weights.split(",") for t in a.tta.split(",")}
        jobs = {k: Path(v) for k, v in (kv.split("=") for kv in a.passes_jobs.split(","))}
        print(json.dumps({"pngs": export_passes(Path(a.root), jobs, models, a.seed), "engines": sorted(models)}))
        return 0
    mapping = dict(kv.split("=") for kv in a.map.split(","))
    index = PackedIndex.scan(a.eval.split(","), split=None)
    root = Path(a.root)
    models = {f"dn_{Path(w).stem.replace(':', '_')}_tta{t}": (load(w), int(t))
              for w in a.weights.split(",") for t in a.tta.split(",")}
    total, checks = 0, []
    for res, jobs in sorted(index.by_res.items()):
        for job in jobs:
            scene = scene_of(job.path.parent.name, mapping)
            if scene is None:
                continue
            total += export_job(job, res, scene, models, root)
            if a.check and (c := check_display(job, res, scene, root)):
                checks.append(c)
    print(json.dumps({"pngs": total, "engines": sorted(models) + ["dn_oidn"], "display_check": checks}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
