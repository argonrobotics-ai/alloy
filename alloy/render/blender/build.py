# SPDX-License-Identifier: Apache-2.0
"""Render an ARB v1 bundle with a look file in Blender Cycles (docs/RENDER.md).

    blender -b --python-exit-code 1 -P alloy/render/blender/build.py -- --bundle B --look L --camera NAME --out DIR
        [--frames A:B] [--samples S] [--still K] [--format png|png16|exr] [--device OPTIX|CUDA|CPU] [--assets DIR]

Writes DIR/frames/%05d.<ext> (bundle frame index), skipping frames that exist, and appends one JSON line per frame
plus a summary to DIR/timing.jsonl. Exit codes: 1 bad bundle / look / render error, 2 requested device missing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402

import colour  # noqa: E402
import handlers  # noqa: E402
import props  # noqa: E402
import stage  # noqa: E402
from assets import AssetCache  # noqa: E402
from camera import CameraRig  # noqa: E402
from deform import SurfaceSet  # noqa: E402
from grains import InstancerSet, PointSet  # noqa: E402
from io_bundle import Bundle, BundleError, load_bundle  # noqa: E402
from liquids import LiquidSet  # noqa: E402
from look import Look, LookError, frame_list, load_look  # noqa: E402
from materials import MaterialLibrary  # noqa: E402
from rigid import RigidSet  # noqa: E402
from skins import SkinSet  # noqa: E402
from tubes import CurveSet  # noqa: E402

log = logging.getLogger("alloy.render.blender")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="build.py", description=__doc__.splitlines()[0])
    p.add_argument("--bundle", required=True)
    p.add_argument("--look", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frames", default=None, help="A:B bundle frames, A inclusive, B exclusive")
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--still", type=int, default=None)
    p.add_argument("--format", choices=sorted(stage.FORMATS), default="png")
    p.add_argument("--device", choices=["OPTIX", "CUDA", "CPU"], default="OPTIX")
    p.add_argument("--assets", default=None)
    return p.parse_args(argv)


def _window(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    try:
        a, b = spec.split(":")
        return int(a or 0), int(b) if b else 1 << 30
    except ValueError as exc:
        raise LookError(f"--frames {spec!r} is not A:B") from exc


def build_scene(bundle: Bundle, look: Look, args: argparse.Namespace) -> tuple[list[str], int, dict]:
    """Reset Blender, set up device and render settings, and build every entity.

    Returns (devices, samples, white balance applied)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    devices = stage.setup_device(scene, args.device)
    samples = int(args.samples or look.render["samples"])
    stage.setup_render(scene, look, samples, args.format, bundle.fps)
    assets = AssetCache(args.assets, bundle.root)
    coll = bpy.data.collections.new("alloy")
    scene.collection.children.link(coll)
    # Stage first: the white-balance probe renders under lights and environment only.
    env_mean = stage.setup_world(scene, look.world, assets)
    stage.setup_lights(look.lights, coll)
    if args.camera not in look.cameras:
        raise LookError(f"--camera {args.camera!r} not in look.cameras ({sorted(look.cameras)})")
    wb = colour.setup_white_balance(scene, look, env_mean, look.cameras[args.camera]["keys"][0])
    colour.setup_glare(scene, look.render.get("glare"))
    mats = MaterialLibrary(look, assets, bundle.frame_at_time)
    skins = SkinSet(bundle, look, mats, coll)
    rigid = RigidSet(bundle, look, mats, coll, skins.hidden_bodies, skins)
    def walls(rx):
        return rigid.objects_matching(rx) + skins.objects_matching(rx)

    updaters = [rigid, skins, SurfaceSet(bundle, look, mats, coll), CurveSet(bundle, look, mats, coll),
                LiquidSet(bundle, look, mats, coll, walls), InstancerSet(bundle, look, mats, coll),
                PointSet(bundle, look, mats, coll)]
    updaters.append(CameraRig(scene, bundle, look, args.camera, rigid.origin_fn))
    props.build_props(look, bundle, assets, mats, coll)
    props.build_ground(look, bundle, mats, coll)
    stage.setup_horizon_fade(look.world, mats)
    scene.frame_start, scene.frame_end = 0, max(bundle.count - 1, 0)
    handlers.install(updaters)
    return devices, samples, wb


def _emit(timing: Path, row: dict) -> None:
    line = json.dumps(row)
    print(line, flush=True)
    with timing.open("a") as f:
        f.write(line + "\n")


def render_frames(frames: list[int], out: Path, ext: str, overwrite: bool, meta: dict) -> dict:
    """Render each frame to out/frames/%05d.ext (atomic rename), skipping existing files unless overwrite."""
    scene = bpy.context.scene
    fdir = out / "frames"
    fdir.mkdir(parents=True, exist_ok=True)
    timing = out / "timing.jsonl"
    stats = {"rendered": 0, "skipped": 0, "seconds": 0.0}
    for f in frames:
        final = fdir / f"{f:05d}.{ext}"
        if final.exists() and not overwrite:
            stats["skipped"] += 1
            continue
        tmp = fdir / f".{f:05d}.part.{ext}"
        handlers.RUNTIME.center = f
        t0 = time.perf_counter()
        scene.frame_set(f)
        scene.render.filepath = str(tmp)
        bpy.ops.render.render(write_still=True)
        if handlers.RUNTIME.error is not None:
            raise RuntimeError(f"frame {f}: handler failed: {handlers.RUNTIME.error!r}")
        os.replace(tmp, final)
        dt = time.perf_counter() - t0
        stats["rendered"] += 1
        stats["seconds"] += dt
        _emit(timing, {"frame": f, "seconds": round(dt, 3), "samples": meta["samples"], "device": meta["device"]})
    return stats


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    args = parse_args(argv)
    t0 = time.perf_counter()
    try:
        bundle = load_bundle(args.bundle)
        look = load_look(args.look)
        frames = [args.still] if args.still is not None else frame_list(look.frames, bundle.count,
                                                                         _window(args.frames))
        devices, samples, wb = build_scene(bundle, look, args)
    except (BundleError, LookError) as exc:
        log.error("%s: %s", type(exc).__name__, exc)
        return 1
    except stage.DeviceError as exc:
        log.error("DeviceError: %s", exc)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"samples": samples, "device": args.device}
    build_s = time.perf_counter() - t0
    _emit(out / "timing.jsonl", {"event": "start", "devices": devices, "frames": len(frames), "build_s":
                                 round(build_s, 3), "blender": bpy.app.version_string, "camera": args.camera,
                                 "white_balance": wb})
    ext = stage.FORMATS[args.format][2]
    stats = render_frames(frames, out, ext, overwrite=args.still is not None, meta=meta)
    n = max(stats["rendered"], 1)
    _emit(out / "timing.jsonl", {"summary": True, **stats, "seconds": round(stats["seconds"], 3),
                                 "s_per_frame": round(stats["seconds"] / n, 3), "build_s": round(build_s, 3),
                                 "devices": devices, "samples": samples, "resolution": [
                                     bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y]})
    return 0


if __name__ == "__main__":
    _argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sys.exit(main(_argv))
