#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the teacher-render job list from a plan: look variants per bundle, seeded frames, hold-outs.

    python tools/denoiser/jobs.py PLAN.json --out WORK_DIR        # writes WORK_DIR/jobs.jsonl and WORK_DIR/looks/

PLAN.json::

    {"seed": 1, "res": "128x128", "large_res": "640x480",
     "inputs": "11:1,2,4,8,16,32;12:1,4,16",
     "bundles": {
       "c3": {"dir": "/data/in/c3", "look": "/data/in/c3/look.json", "cameras": ["wide", "vertical"],
              "variants": 80, "frames_per_job": 40, "cams_per_variant": 1, "hold": [420, 4400], "margin": 60,
              "large_every": 3, "large_frames": 4, "split": "train", "weight": 1.0}}}

Each bundle gets ``variants`` look variants (``looks.variant``; variant 0 is the production look unchanged), each
rendered from ``cams_per_variant`` cameras (cycled through ``cameras``) on ``frames_per_job`` seeded frames that
stay ``margin`` frames away from every ``hold`` frame and inside 3-97 % of the take (or the fixed ``frames`` list);
``cameras`` defaults to every camera of the look, and ``blender_args`` (e.g. ``["--ref-spp-large", "4096"]``) are
passed to ``render_bundle.py``. Every ``large_every``-th job
also renders ``large_frames`` of its frames at ``large_res``. A bundle with ``"split": "eval"`` keeps the
production look (no variants) and adds its ``eval_inputs`` (default ``1:1,2,4,8,16,32,64,128``).
One job = one Blender process (one scene build). Prints a JSON summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import looks as L

EVAL_INPUTS = "1:1,2,4,8,16,32,64,128"


def frame_count(bundle_dir: str) -> int:
    """Frames in a bundle (``scene.json`` ``frames.count``). Example: ``frame_count("/data/in/c3")``."""
    return int(json.loads((Path(bundle_dir) / "scene.json").read_text())["frames"]["count"])


def pick_frames(count: int, n: int, rng: random.Random, hold: list[int], margin: int) -> list[int]:
    """``n`` distinct frames inside 3-97 % of the take, ``margin`` away from every held-out frame, sorted.

    Example: ``pick_frames(1000, 5, random.Random(0), [500], 60)``."""
    lo, hi = int(0.03 * count), max(int(0.97 * count), int(0.03 * count) + 1)
    pool = [f for f in range(lo, hi) if all(abs(f - h) > margin for h in hold)]
    if not pool:
        raise ValueError(f"no frames left in 0..{count - 1} after hold-outs {hold} +- {margin}; shrink the margin")
    return sorted(rng.sample(pool, min(n, len(pool))))


def drop_missing_liquids(look: dict, bundle_dir: str) -> list[str]:
    """Remove look ``liquids`` entries with no point set of that name in the bundle (a look written for another
    take of the scene); returns the removed names. Example: ``drop_missing_liquids(look, "/data/in/dc5h")``."""
    scene = json.loads((Path(bundle_dir) / "scene.json").read_text())
    names = {p.get("name") for p in scene.get("points", [])}
    liquids = look.get("liquids") or {}
    gone = [k for k in liquids if k not in names]
    for k in gone:
        liquids.pop(k)
    return gone


def write_variants(name: str, spec: dict, seed: int, out: Path) -> list[Path]:
    """Write the bundle's look variants (variant 0 = production look) and return their paths."""
    look = json.loads(Path(spec["look"]).read_text())
    drop_missing_liquids(look, spec["dir"])
    hdris = L.load_hdris()
    vdir = out / "looks" / name
    vdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(int(spec.get("variants", 1))):
        v = look if i == 0 else L.variant(look, seed, i, hdris, spec.get("p_swap", 0.35), spec.get("p_dark", 0.3),
                                          spec.get("p_hdri", 0.6))
        p = vdir / f"v{i:03d}.json"
        p.write_text(json.dumps(v, indent=1))
        paths.append(p)
    return paths


def bundle_jobs(name: str, spec: dict, plan: dict, out: Path) -> list[dict]:
    """All jobs of one bundle."""
    split = spec.get("split", "train")
    if split == "eval":
        spec = {**spec, "variants": 1}
    rng = random.Random(int(hashlib.sha256(f"{plan['seed']}:{name}".encode()).hexdigest()[:16], 16))
    count = frame_count(spec["dir"])
    cams = spec.get("cameras") or sorted(json.loads(Path(spec["look"]).read_text())["cameras"])
    per = int(spec.get("cams_per_variant", 1))
    jobs = []
    for vi, look in enumerate(write_variants(name, spec, plan["seed"], out)):
        for ci in range(per):
            cam = cams[(vi * per + ci) % len(cams)]
            frames = spec.get("frames") or pick_frames(count, int(spec.get("frames_per_job", 40)), rng,
                                                       spec.get("hold", []), int(spec.get("margin", 60)))
            jobs.append(_job(name, spec, plan, look, cam, frames, split, len(jobs)))
    return jobs


def _job(name: str, spec: dict, plan: dict, look: Path, cam: str, frames: list[int], split: str, k: int) -> dict:
    large = spec.get("large_every") and k % int(spec["large_every"]) == 0
    job = {"id": f"{name}_{look.stem}_{cam}", "bundle": name, "bundle_dir": spec["dir"], "look": str(look),
           "camera": cam, "frames": frames, "res": plan.get("res", "128x128"), "split": split,
           "inputs": spec.get("eval_inputs", EVAL_INPUTS) if split == "eval" else plan["inputs"],
           "weight": spec.get("weight", 1.0), "blender_args": spec.get("blender_args", [])}
    if large:
        job["large"] = {"res": plan.get("large_res", "640x480"), "frames": frames[: int(spec.get("large_frames", 4))]}
    return job


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("plan")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    plan = json.loads(Path(a.plan).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [j for name, spec in plan["bundles"].items() for j in bundle_jobs(name, spec, plan, out)]
    with (out / "jobs.jsonl").open("w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    frames = sum(len(j["frames"]) for j in jobs)
    large = sum(len(j.get("large", {}).get("frames", [])) for j in jobs)
    print(json.dumps({"jobs": len(jobs), "frames": frames, "large_frames": large, "out": str(out / "jobs.jsonl")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
