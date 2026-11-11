# SPDX-License-Identifier: Apache-2.0
"""Row 5 camera throughput for Alloy (public API only): physics step + one camera per world + render_torch.

One cell per process:
  python alloy_cam.py --scene cartpole|franka_cube --size 128 --worlds 1024 --setting fast|default|fast_optix \
      [--outputs rgb] --out raw/alloy_g4.jsonl --png-dir raw/png
  python alloy_cam.py --scene t12 --size 128 --worlds 1024 --out ...   # extra row: alloy.camera.bench --claim t12
Settings (alloy.Camera, fov 45 deg, no denoiser, path tracer = Cycles port, studio sun+sky light):
  fast       spp=1, backend=warp (megakernel + Warp BVH)   <- like for like with ray casters
  default    spp=16 (Camera's default), backend=warp
  fast_optix spp=1, backend=optix (RT cores; needs `python -m alloy.camera.optix build`)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

import cam_scenes as S
import cam_util as U

log = logging.getLogger("row5.alloy")

SETTINGS = {
    "fast": dict(spp=1, backend="warp", desc="path tracer (Cycles port), 1 spp, Warp BVH megakernel, no denoiser, studio sun+sky"),
    "default": dict(spp=16, backend="warp", desc="path tracer (Cycles port), 16 spp (Camera default), Warp BVH megakernel, no denoiser, studio sun+sky"),
    "fast_optix": dict(spp=1, backend="optix", desc="path tracer (Cycles port), 1 spp, OptiX RT cores wavefront, no denoiser, studio sun+sky"),
}
OUTPUT_NAMES = {"rgb": ("rgb",), "rgb+depth": ("rgb", "depth"), "rgb+depth+seg": ("rgb", "depth", "segmentation")}


@dataclass
class Loop:
    """The pieces of one timed loop. ``extra`` holds the depth / segmentation tensors of the last render."""
    physics: Callable[[], None]
    render: Callable[[], Any]
    sync: Callable[[], None]
    ramp_to_hold: Callable[[], None]
    extra: dict


def build_sim(scene_name: str, worlds: int):
    import alloy
    from alloy.cards.library import default_root
    sc = alloy.Scene(worlds=worlds, dt=S.CONTROL_DT)
    if scene_name == "cartpole":
        sc.add_ground()
        sc.add_robot("cart", mjcf=S.CARTPOLE_MJCF.format(floor=""))
    else:
        sc.add_robot("panda", default_root() / "robots" / "emb_e2_panda" / "scene.xml")
    return alloy.build(sc, seed=0)


def make_loop(sim, cam, scene_name: str, worlds: int, outputs: str = "rgb") -> Loop:
    import torch
    import warp as wp
    joints, hold = S.SCENES[scene_name][:2]
    pool = S.noise_pool(scene_name, worlds)
    names = OUTPUT_NAMES[outputs]
    counter = {"i": 0}
    extra: dict = {}

    def physics() -> None:
        tg = pool[counter["i"] % len(pool)]
        counter["i"] += 1
        sim.set_targets({j: tg[:, c] for c, j in enumerate(joints)})
        sim.step(1)

    def render():
        """rgb: uint8 tensor. Else the dict of torch tensors from cam.render_torch(sim, output=names) (stays on the GPU)."""
        if outputs == "rgb":
            return cam.render_torch(sim)
        extra.update(cam.render_torch(sim, output=names))
        return extra["rgb"]

    def sync() -> None:
        wp.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def ramp_to_hold(steps: int = 150) -> None:
        """Alloy starts an MJCF robot at q=0 (it ignores the MJCF keyframe): ramp the servos to the hold pose, untimed."""
        if not hold.any():
            return
        for i in range(1, steps + 1):
            sim.set_targets({j: float(hold[c]) * i / steps for c, j in enumerate(joints)})
            sim.step(1)

    return Loop(physics, render, sync, ramp_to_hold, extra)


def common_row(a: argparse.Namespace, extra: dict) -> dict:
    import alloy
    import warp as wp
    return {"engine": "alloy", "engine_version": alloy.__version__, "scene": a.scene, "outputs": a.outputs,
            "width": a.size, "height": a.size, "worlds": a.worlds, "setting": a.setting,
            "control_dt_s": S.CONTROL_DT, **U.gpu_info(),
            "commit_or_version_pins": {"warp": wp.config.version, "alloy_commit": a.commit},
            "command": " ".join(sys.argv), **extra}


def _save_extra_images(tag: str, extra: dict, why: str | None) -> str | None:
    """PNG grids of depth and segmentation; returns the first reason one of them is empty (or ``why``)."""
    if "depth" in extra:
        dep = extra["depth"][:4].cpu().numpy()
        fin = np.isfinite(dep)
        if not fin.any() or dep[fin].max() <= 0:
            why = why or "depth image empty"
        scaled = np.clip(np.where(fin, dep, 0) / max(dep[fin].max(), 1e-6), 0, 1)
        U.save_grid(tag + "_depth.png", (scaled * 255).astype(np.uint8)[..., None])
    if "segmentation" in extra:
        sg = extra["segmentation"][:4].cpu().numpy()
        if len(np.unique(sg)) < 2:
            why = why or "segmentation image has one id"
        U.save_grid(tag + "_seg.png", ((sg * 37) % 256).astype(np.uint8)[..., None])
    return why


def _timings(loop: Loop, a: argparse.Namespace, mem: U.MemSampler) -> dict:
    both = lambda: (loop.physics(), loop.render())  # noqa: E731
    return {"both": U.time_loops(both, loop.sync, a.worlds, a.warmup, a.repeats, a.loops, on_start=mem.mark),
            "render": U.time_loops(loop.render, loop.sync, a.worlds, 10, 3, 50, budget_s=30.0),
            "physics": U.time_loops(loop.physics, loop.sync, a.worlds, 10, 3, 50, budget_s=30.0)}


def _measure(sim, loop: Loop, a: argparse.Namespace, mem: U.MemSampler, t_build: float) -> dict:
    """Warm up, time the three loops, check the images. Returns the measured row fields."""
    loop.ramp_to_hold()
    loop.physics(), loop.render()
    loop.sync()
    first = loop.render()[:4].cpu().numpy().copy()
    build_s = time.perf_counter() - t_build
    s0 = sim.physics_steps
    loop.physics()
    loop.sync()
    substeps = sim.physics_steps - s0
    t = _timings(loop, a, mem)
    last = loop.render()[:4].cpu().numpy().copy()
    why = U.sanity(first, last)
    tag = f"{a.png_dir}/alloy_{a.scene}_{a.setting}_{a.outputs}_{a.size}px_{a.worlds}w"
    U.save_grid(tag + ".png", last)
    why = _save_extra_images(tag, loop.extra, why)
    return {"timings": t, "why": why, "substeps": substeps, "build_s": build_s}


def _status(sim, why: str | None) -> tuple[dict, str, str | None]:
    chk = sim.check()
    if not chk["ok"]:
        return chk, "invalid", f"sim.check() not ok: {chk['problems'][:2]}"
    return chk, ("invalid" if why else "ok"), why


def run_cell(a: argparse.Namespace) -> dict:
    import alloy
    cfg = SETTINGS[a.setting]
    mem = U.MemSampler().start()
    t_build = time.perf_counter()
    sim = build_sim(a.scene, a.worlds)
    eye, target = S.SCENES[a.scene][3]
    cam = alloy.Camera("bench", look_at=(eye, target), width=a.size, height=a.size, fov=S.CAM_FOV, spp=cfg["spp"],
                       backend=cfg["backend"])
    loop = make_loop(sim, cam, a.scene, a.worlds, a.outputs)
    m = _measure(sim, loop, a, mem, t_build)
    chk, status, why = _status(sim, m["why"])
    memr = mem.stop()
    res, r_only, p_only = m["timings"]["both"], m["timings"]["render"], m["timings"]["physics"]
    return common_row(a, {
        "status": status, "invalid_reason": why, "images_per_s": round(res["rate"], 1),
        "images_per_s_min": round(res["min"], 1), "images_per_s_max": round(res["max"], 1),
        "render_only_images_per_s": round(r_only["rate"], 1),
        "physics_only_steps_per_s": round(p_only["rate"], 1),  # world-steps/s (control steps x worlds)
        "repeats": res["repeats"], "loops_per_repeat": res["loops"], "warmup_loops": res["warmup_loops"],
        "reduced_loops": res["reduced_loops"], "physics_substeps": m["substeps"], "build_s": round(m["build_s"], 1),
        "renderer": cfg["desc"],
        "render_settings": {"spp": cfg["spp"], "backend": cfg["backend"], "denoise": None, "fov_deg": S.CAM_FOV},
        "sim_check_ok": bool(chk["ok"]), "routes": {k: v["route"] for k, v in sim.routes().items()}, **memr})


def run_t12(a: argparse.Namespace) -> dict:
    """The gated-quality scene through alloy.camera.bench --claim t12 (an extra row, not the comparison)."""
    from alloy.camera import bench
    mem = U.MemSampler().start()
    cfg = bench.with_claim(bench.BenchConfig("t12", a.worlds, a.size, a.size, 1, True, "full", None, a.backend_t12, 10), "t12")
    mem.mark()
    r = bench.run(cfg)
    memr = mem.stop()
    return common_row(a, {
        "status": "ok", "images_per_s": r["images_per_s"], "render_only_images_per_s": r["render_only_images_per_s"],
        "physics_only_steps_per_s": round(a.worlds / (r["ms"]["physics"] / 1e3), 1), "bench_row": r,
        "renderer": f"gate-claimed setting: {cfg.claim}, backend {r['backend']}, quality full",
        "render_settings": {"spp": cfg.spp, "denoise": cfg.denoise, "tta": cfg.tta, "backend": r["backend"]},
        "note": "extra t12 row, 10 timed iterations (alloy.camera.bench), not the comparison; physics 1/120 s per loop",
        "physics_substeps": None, **memr})


def _has_aov() -> bool:
    try:
        import alloy.camera.aov  # noqa: F401
        return True
    except ImportError:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    U.add_cell_arguments(ap, sorted(S.SCENES) + ["t12"])
    ap.add_argument("--setting", default="fast", choices=sorted(SETTINGS))
    ap.add_argument("--backend-t12", default="optix")
    return ap.parse_args(argv)


def run_to_row(a: argparse.Namespace) -> dict:
    """One cell as a row; a failed cell is a row, not a crash."""
    base = {"engine": "alloy", "scene": a.scene, "outputs": a.outputs, "width": a.size, "height": a.size,
            "worlds": a.worlds, "setting": a.setting, "command": " ".join(sys.argv)}
    if a.outputs != "rgb" and not _has_aov():
        return {**base, "status": "pending FEAT-DEPTH"}
    try:
        return run_t12(a) if a.scene == "t12" else run_cell(a)
    except Exception as exc:
        log.error("cell failed:\n%s", traceback.format_exc())
        return {**base, "status": "oom" if U.is_oom(exc) else "error", "reason": f"{type(exc).__name__}: {str(exc)[:500]}"}


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    a = parse_args()
    row = run_to_row(a)
    U.write_row(a.out, row)
    print(row["status"], row.get("images_per_s"), row.get("gpu_mem_mib"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
