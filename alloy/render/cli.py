# SPDX-License-Identifier: Apache-2.0
"""python -m alloy.render.cli check | still | render | encode."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from alloy.render import encode as enc
from alloy.render import look as lk
from alloy.render.farm import Farm, FarmError, FarmJob
from alloy.render.vm import ZONES, GcloudBackend, RenderConfig, VmError, sanitise_name, vm_name

log = logging.getLogger("alloy.render.cli")
ASSET_CACHE = Path.home() / ".cache" / "alloy_render" / "assets"
CYCLES_CACHE = Path.home() / ".cache" / "alloy_render" / "cycles"
BLENDER_DIR = Path(__file__).resolve().parent / "blender"


def _load(args: argparse.Namespace):
    """Load and validate bundle + look; returns (bundle, look, errors, warnings)."""
    from alloy.render import bundle as bmod

    try:
        bundle = bmod.load_bundle(args.bundle)
        look = lk.load_look(args.look)
    except (bmod.BundleError, lk.LookError) as exc:
        return None, None, [str(exc)], []
    problems = [f"bundle: {p}" for p in bmod.validate_bundle(bundle)] + lk.validate_look(look, bundle)
    cam = getattr(args, "camera", None)
    if cam and cam not in look.get("cameras", {}):
        problems.append(f"--camera {cam!r} is not in look cameras {sorted(look.get('cameras', {}))}")
    errors = [p for p in problems if not p.startswith(lk.WARN)]
    return bundle, look, errors, [p[len(lk.WARN):] for p in problems if p.startswith(lk.WARN)]


def cmd_check(args: argparse.Namespace) -> int:
    """Validate bundle and look, print a report; exit 1 on errors."""
    bundle, look, errors, warnings = _load(args)
    if bundle is None:
        print("ERROR " + errors[0])
        return 1
    frames = lk.frame_list(look, bundle, args.frames)
    samples = args.samples or look["render"]["samples"]
    cams = look["cameras"] if not args.camera else {args.camera: look["cameras"].get(args.camera, {})}
    print(f"bundle   {bundle.root}  {bundle.frame_count} frames @ {bundle.fps:g} fps = {bundle.duration_s:.2f} s, "
          f"{bundle.nbytes() / 1e9:.2f} GB")
    print("labels   " + ", ".join(f"{k}={len(v)}" for k, v in bundle.labels().items()))
    print(f"render   {len(frames)} frames (step {look.get('frames', {}).get('step', 1)}), {samples} spp")
    for name in cams:
        res = lk.camera_resolution(look, name)
        est = lk.estimate_cost(len(frames), res, samples)
        print(f"camera   {name}: {res[0]}x{res[1]}  ~{est.seconds_per_frame:.0f} s/frame, {est.gpu_hours:.2f} GPU-h, "
              f"~${est.usd:.2f} spot")
    for w in warnings:
        print("WARN  " + w)
    for e in errors:
        print("ERROR " + e)
    print("FAIL" if errors else "OK")
    return 1 if errors else 0


def _job(args: argparse.Namespace, bundle, look, frames: list[int], out: Path, vms: int) -> FarmJob:
    return FarmJob(bundle=Path(args.bundle).resolve(), look=Path(args.look).resolve(), camera=args.camera, out=out,
                   name=vm_name(Path(args.bundle), "").rstrip("-"), frames=frames, samples=args.samples,
                   fmt=getattr(args, "format", "png"), vms=vms, zones=_zones(args), keep_vms=args.keep_vms,
                   region_cap=getattr(args, "region_cap", 8), stage=_stage_arg(args),
                   blender_dir=BLENDER_DIR if BLENDER_DIR.is_dir() else None,
                   assets_dir=ASSET_CACHE if ASSET_CACHE.is_dir() else None, cycles_dir=CYCLES_CACHE,
                   reuse_vms=tuple(getattr(args, "reuse_vm", None) or ()), frames_gcs=_frames_gcs(args),
                   bundle_gcs=getattr(args, "bundle_gcs", None))


def _frames_gcs(args: argparse.Namespace) -> str | None:
    """--frames-gcs: a gs:// prefix, or 'auto' = gs://$ALLOY_RENDER_BUCKET/frames/<out dir name>/frames.

    The render VMs' service account needs write access to that bucket (roles/storage.objectUser).
    """
    value = getattr(args, "frames_gcs", None)
    if not value:
        return None
    if value == "auto":
        try:
            return RenderConfig.load().bucket_uri(f"frames/{Path(args.out).resolve().name}/frames")
        except VmError as exc:
            raise SystemExit(f"--frames-gcs auto: {exc}") from exc
    if not value.startswith("gs://"):
        raise SystemExit(f"--frames-gcs must be a gs:// prefix or 'auto', got {value!r}")
    return value.rstrip("/")


def _stage_arg(args: argparse.Namespace) -> str | None:
    if getattr(args, "no_stage", False):
        return "off"
    return getattr(args, "stage", None)


def _zones(args: argparse.Namespace) -> tuple[str, ...]:
    return (args.zone,) if args.zone else ZONES


def _run_farm(farm: Farm) -> dict:
    """Run a farm; delete created VMs on Ctrl-C, SIGTERM or error."""
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        return farm.run()
    except BaseException:
        log.error("aborting: deleting created VMs")
        farm.delete_all()
        raise


def _alloy_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cmd_render(args: argparse.Namespace) -> int:
    """Render the frame list on spot VMs, encode, write render_log.json."""
    bundle, look, errors, warnings = _load(args)
    if errors:
        print("\n".join("ERROR " + e for e in errors))
        return 1
    frames = lk.frame_list(look, bundle, args.frames)
    if not frames:
        print("ERROR no frames selected")
        return 1
    out = Path(args.out).resolve()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.look, out / "look.json")
    if args.vms < 1 and not args.reuse_vm:
        print("ERROR need --vms >= 1 or at least one --reuse-vm")
        return 1
    farm = Farm(GcloudBackend(), _job(args, bundle, look, frames, out, args.vms))
    try:
        log_data = _run_farm(farm)
    except (FarmError, VmError) as exc:
        print(f"ERROR {exc}")
        return 2
    except KeyboardInterrupt:
        print("interrupted; VMs deleted")
        return 130
    speed = playback_speed(args.fps, int(look["frames"].get("step", 1)), bundle.fps)
    log_data.update(alloy_commit=_alloy_commit(), camera=args.camera,
                    resolution=lk.camera_resolution(look, args.camera),
                    samples=args.samples or look["render"]["samples"], fps=args.fps, playback_x_real_time=speed,
                    frames_gcs=farm.job.frames_gcs)
    (out / "render_log.json").write_text(json.dumps(log_data, indent=2))
    farm.delete_all()
    print(json.dumps(log_data))
    return _encode_after(out, args.fps, args.crf, farm.job.frames_gcs)


def playback_speed(fps: float, step: int, bundle_fps: float) -> float:
    """How fast the encoded clip plays against sim time: 1.0 real time, 0.25 = 4x slow motion."""
    return round(fps * step / bundle_fps, 4)


def _encode_after(out: Path, fps: float, crf: int, frames_gcs: str | None = None) -> int:
    try:
        dest = enc.encode_gcs(frames_gcs, out / "clip.mp4", fps, crf) if frames_gcs else enc.encode(out, fps, crf)
        print(f"encoded {dest}")
    except enc.EncodeError as exc:
        print(f"ERROR {exc}")
        return 3
    return 0


def cmd_still(args: argparse.Namespace) -> int:
    """Render one frame on a VM (reused with --vm, else a new one that is left up)."""
    bundle, look, errors, _ = _load(args)
    if errors:
        print("\n".join("ERROR " + e for e in errors))
        return 1
    if not 0 <= args.frame < bundle.frame_count:
        print(f"ERROR --frame {args.frame} outside 0..{bundle.frame_count - 1}")
        return 1
    out = Path(args.out or f"still_{sanitise_name(Path(args.bundle).resolve().name)}").resolve()
    (out / "frames").mkdir(parents=True, exist_ok=True)
    args.keep_vms = True
    job = _job(args, bundle, look, [args.frame], out, 1)
    job.vm_name = args.vm or vm_name(Path(args.bundle), "still")
    job.poll_s = 10.0
    farm = Farm(GcloudBackend(), job)
    try:
        log_data = _run_farm(farm)
    except (FarmError, VmError) as exc:
        print(f"ERROR {exc}")
        return 2
    png = out / "frames" / f"{args.frame:05d}.png"
    vm = farm.created.get(job.vm_name)
    zone = vm.zone if vm else "ZONE"
    print(f"still {png}  ({log_data['spot_usd']} USD so far); VM {job.vm_name} left up; "
          f"delete: gcloud compute instances delete {job.vm_name} --zone {zone} --quiet")
    return 0 if png.exists() else 2


def cmd_encode(args: argparse.Namespace) -> int:
    """Re-encode DIR/frames, or stream frames from a gs:// prefix into --out (default ./clip.mp4)."""
    try:
        if args.dir.startswith("gs://"):
            dest = enc.encode_gcs(args.dir, Path(args.out or "clip.mp4"), args.fps, args.crf)
        else:
            dest = enc.encode(Path(args.dir), args.fps, args.crf, args.prores)
        print(f"encoded {dest}")
    except enc.EncodeError as exc:
        print(f"ERROR {exc}")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m alloy.render.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("still", cmd_still), ("render", cmd_render)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("bundle")
        p.add_argument("--look", required=True)
        p.add_argument("--camera", required=(name != "check"))
        p.add_argument("--samples", type=int)
        p.add_argument("--frames", help="A:B bundle frame indices, A inclusive, B exclusive")
        p.add_argument("--zone")
        p.add_argument("--keep-vms", action="store_true")
        p.add_argument("--format", choices=("png", "png16", "exr"), default="png")
    sub.choices["still"].add_argument("--frame", type=int, required=True)
    sub.choices["still"].add_argument("--vm")
    sub.choices["still"].add_argument("--out")
    r = sub.choices["render"]
    r.add_argument("--out", required=True)
    r.add_argument("--vms", type=int, default=4, help="new spot VMs to create (0 with --reuse-vm only)")
    r.add_argument("--region-cap", type=int, default=8, help="max VMs of this job per region (quota is 8)")
    r.add_argument("--stage", metavar="GS_URI", help="stage inputs under this gs:// prefix "
                   "(default: automatic through gs://$ALLOY_RENDER_BUCKET/stage when --vms > 8)")
    r.add_argument("--no-stage", action="store_true", help="always rsync inputs from this machine")
    r.add_argument("--reuse-vm", action="append", default=[], metavar="NAME",
                   help="existing VM to use as an extra worker, never deleted (repeatable)")
    r.add_argument("--fps", type=float, default=30.0,
                   help="clip frame rate; real time when fps x frames.step == bundle fps (render_log.json "
                        "records playback_x_real_time)")
    r.add_argument("--crf", type=int, default=16)
    r.add_argument("--frames-gcs", metavar="GS_URI|auto",
                   help="VMs upload frames to this gs:// prefix and none are pulled here; the clip is encoded by "
                        "streaming them ('auto' = gs://$ALLOY_RENDER_BUCKET/frames/<out dir name>/frames)")
    e = sub.add_parser("encode")
    e.set_defaults(fn=cmd_encode)
    e.add_argument("dir")
    e.add_argument("--fps", type=float, default=30.0)
    e.add_argument("--crf", type=int, default=16)
    e.add_argument("--prores", action="store_true")
    e.add_argument("--out", help="output file when DIR is a gs:// prefix (default ./clip.mp4)")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if str(getattr(args, "bundle", "") or "").startswith("gs://"):
        if args.fn not in (cmd_check, cmd_render):
            print("ERROR a gs:// bundle works with check and render only")
            return 1
        from alloy.render.mirror import mirror_gcs_bundle
        args.bundle_gcs, args.bundle = args.bundle.rstrip("/"), str(mirror_gcs_bundle(args.bundle))
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
