# SPDX-License-Identifier: Apache-2.0
"""Denoiser teacher pairs from an alloy render bundle: one scene build, many frames (run inside Blender 4.5).

    blender -b --python-exit-code 1 -P tools/denoiser/blender/render_bundle.py -- \\
        --bundle BUNDLE_DIR --look LOOK.json --camera NAME --out OUT --frames 10,200 \\
        [--res 128x128,640x480] [--inputs "11:1,2,4,8,16,32;12:1,4,16"] [--ref-spp 4096] [--ref-spp-large 2048] \\
        [--ref-seed 777] [--no-ref] [--device OPTIX] [--assets DIR]

The scene is alloy's own (``alloy/render/blender`` ``build_scene()`` from this repository), then
``teacher_common.configure()``: adaptive sampling off, motion blur off, OIDN on, denoising passes stored.
Per frame and resolution, ``OUT/<W>x<H>/<frame:05d>/``:

- ``in_s<seed>_<spp>.exr``: a noisy input (layers Noisy Image, Combined = OIDN of the same samples, Denoising
  Albedo / Normal / Depth);
- ``ref.exr`` + ``ref.png``: the target, ``ref-spp`` + OIDN with another seed (``ref-spp-large`` above 256x256).

``OUT/view.json`` (colour management), ``OUT/lut.png`` (+ .json, the display LUT), ``OUT/wb.json`` (white-balance
matrix) and ``OUT/timing.jsonl`` sit next to them. Resumable: existing EXRs are skipped, and the scene is not even
built when every file exists. Exit codes: 0 done, 1 bad bundle / look / frames, 2 no render device.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "alloy" / "render" / "blender"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import bpy
import build as alloy_build
import handlers
import stage
import teacher_common as T
from io_bundle import BundleError, load_bundle
from look import LookError, load_look

log = logging.getLogger("denoiser.render_bundle")
LARGE_PX = 256 * 256  # frames above this many pixels use --ref-spp-large


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="render_bundle.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", required=True)
    p.add_argument("--look", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", required=True, help="comma list of bundle frames")
    p.add_argument("--res", default="128x128", help="comma list of WxH")
    p.add_argument("--inputs", default="11:1,2,4,8,16,32;12:1,4,16", help="seed:spp,spp;seed:spp,...")
    p.add_argument("--ref-spp", dest="ref_spp", type=int, default=4096)
    p.add_argument("--ref-spp-large", dest="ref_spp_large", type=int, default=2048)
    p.add_argument("--ref-seed", dest="ref_seed", type=int, default=777)
    p.add_argument("--no-ref", dest="no_ref", action="store_true")
    p.add_argument("--device", choices=["OPTIX", "CUDA", "CPU"], default="OPTIX")
    p.add_argument("--assets", default=None)
    return p.parse_args(argv)


def ints(s: str) -> list[int]:
    """'1,2,4' -> [1, 2, 4]."""
    return [int(x) for x in s.split(",") if x]


def parse_inputs(spec: str) -> list[tuple[int, int]]:
    """'11:1,2;12:4' -> [(11, 1), (11, 2), (12, 4)] as (seed, spp)."""
    out = []
    for part in spec.split(";"):
        if not part.strip():
            continue
        seed, spps = part.split(":")
        out += [(int(seed), n) for n in ints(spps)]
    return out


def resolutions(spec: str) -> list[tuple[int, int]]:
    """'128x128,640x480' -> [(128, 128), (640, 480)]."""
    return [tuple(int(v) for v in r.split("x")) for r in spec.split(",") if r]


def jobs_for(odir: Path, args: argparse.Namespace, w: int, h: int) -> list[tuple[Path, int, int]]:
    """(stem, spp, seed) still to render in one frame/resolution directory."""
    todo = [(odir / f"in_s{seed}_{spp}", spp, seed) for seed, spp in parse_inputs(args.inputs)]
    if not args.no_ref:
        spp = args.ref_spp_large if w * h > LARGE_PX else args.ref_spp
        todo.append((odir / "ref", spp, args.ref_seed))
    return [j for j in todo if not j[0].with_suffix(".exr").exists()]


def any_todo(args: argparse.Namespace, out: Path) -> bool:
    return any(jobs_for(out / f"{w}x{h}" / f"{f:05d}", args, w, h)
               for f in ints(args.frames) for (w, h) in resolutions(args.res))


def render_frame(scene, f: int, args: argparse.Namespace, out: Path, meta: dict,
                 fresh: T.FreshSessions | None = None) -> dict:
    """Render every missing input and the reference of frame ``f`` at every resolution (a fresh Cycles session per
    size and seed: ``fresh``)."""
    stats = {"rendered": 0, "render_s": 0.0}
    fresh = fresh or T.FreshSessions(scene)
    synced = False
    for (w, h) in resolutions(args.res):
        odir = out / f"{w}x{h}" / f"{f:05d}"
        todo = jobs_for(odir, args, w, h)
        if not todo:
            continue
        odir.mkdir(parents=True, exist_ok=True)
        scene.render.resolution_x, scene.render.resolution_y = w, h
        scene.render.resolution_percentage = 100
        if not synced:
            handlers.RUNTIME.center = f
            scene.frame_set(f)
            synced = True
        for stem, spp, seed in todo:
            scene.cycles.samples, scene.cycles.seed = spp, seed
            fresh.before(w, h, seed)
            rs, ws = T.render_exr(scene, stem, png=stem.name == "ref")
            fresh.after()
            if handlers.RUNTIME.error is not None:
                raise RuntimeError(f"frame {f}: handler failed: {handlers.RUNTIME.error!r}")
            stats["rendered"] += 1
            stats["render_s"] += rs
            T.emit(out / "timing.jsonl", {"frame": f, "w": w, "h": h, "spp": spp, "seed": seed, "file": stem.name,
                                          "render_s": round(rs, 3), "write_s": round(ws, 3), **meta})
    return stats


def build(args: argparse.Namespace) -> tuple[list[str], dict]:
    """Build the bundle's scene with alloy's own builder; raises BundleError / LookError on bad inputs."""
    bundle = load_bundle(args.bundle)
    look = load_look(args.look)
    bad = [f for f in ints(args.frames) if not 0 <= f < bundle.count]
    if bad:
        raise LookError(f"frames {bad} are outside 0..{bundle.count - 1}; pick frames inside the take")
    ns = argparse.Namespace(camera=args.camera, device=args.device, samples=None, format="png", assets=args.assets)
    devices, _, wb = alloy_build.build_scene(bundle, look, ns)
    return devices, wb


def write_scene_files(scene, out: Path) -> None:
    (out / "view.json").write_text(json.dumps(T.view_settings(scene), indent=1))
    if not (out / "lut.png").exists():
        T.bake_lut(scene, out / "lut.png")
    if not (out / "wb.json").exists():
        T.bake_wb_matrix(scene, out / "wb.json", out)


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = parse_args(argv)
    t0 = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not any_todo(args, out):
        log.info(json.dumps({"summary": True, "rendered": 0, "note": "all files exist"}))
        return 0
    try:
        devices, wb = build(args)
    except (BundleError, LookError) as exc:
        log.error("ERROR %s: %s", type(exc).__name__, exc)
        return 1
    except stage.DeviceError as exc:
        log.error("ERROR DeviceError: %s (use --device CPU, or check the GPU driver)", exc)
        return 2
    scene = bpy.context.scene
    T.configure(scene)
    write_scene_files(scene, out)
    meta = {"bundle": Path(args.bundle).name, "look": Path(args.look).stem, "camera": args.camera}
    T.emit(out / "timing.jsonl", {"event": "start", "devices": devices, "white_balance": wb,
                                  "build_s": round(time.perf_counter() - t0, 2),
                                  "blender": bpy.app.version_string, **meta})
    total = {"rendered": 0, "render_s": 0.0}
    for f in ints(args.frames):
        st = render_frame(scene, f, args, out, meta)
        total = {k: total[k] + st[k] for k in total}
    T.emit(out / "timing.jsonl", {"summary": True, **total, "total_s": round(time.perf_counter() - t0, 2), **meta})
    return 0


if __name__ == "__main__":
    _argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sys.exit(main(_argv))
