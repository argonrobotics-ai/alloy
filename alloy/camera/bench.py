# SPDX-License-Identifier: Apache-2.0
"""Camera speed with physics in the loop: images/s over many worlds, as JSON lines.

Run::

    python -m alloy.camera.bench --scene t12 --worlds 1024 --size 128x128 --spp 16
    python -m alloy.camera.bench --scene t12 --quality full --claim t12 --backend optix    # the gate's T12 setting
    python -m alloy.camera.bench --scene pack:c3_pack@420 --worlds 1024 --spp 5 --denoise learned  # moving poses
    python -m alloy.camera.bench --scene boxes --worlds 64 --size 640x480 --spp 4 --no-physics

Each row: {"scene", "worlds", "width", "height", "spp", "quality", "denoise", "weights", "tta", "claim", "backend",
"images_per_s", "render_only_images_per_s", "ms": {"physics", "render_and_display", "loop"}, "gpu", "physics",
"mujoco"}. Compile and warm-up are not timed. ``--claim <gate scene>`` takes the samples, denoiser, weights and tta
from that gate scene's claimed setting at each size (``gate_claims``: the fastest one that passes its quality gates),
so a row is a speed at a setting whose quality is gated; without it a row is a speed only (``claim`` null).
``mujoco`` (T12 only, else null) is how full MuJoCo Warp's buffers were after the last step (constraint rows of the
fullest world and contacts of the batch, beside their capacities); an overflow raises instead of printing a row.
``quality="full"`` renders the model with Blender materials (shape colours as matte plastic, studio lights); a
``pack:`` scene has no physics, so every loop instead moves each world's meshes (a refit per step).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import warp as wp

from .camera import Camera
from .pose import look_at
from .wavefront.tracers import resolve_backend

log = logging.getLogger(__name__)
T12_SCENE = "robots/task_t12/scene.xml"  # in the shipped library (a source checkout's or an installed package's)
# MuJoCo Warp buffers per world: T12 at rest holds about 410 constraint rows and 96 contacts. The solver's defaults
# (64 rows) overflow, and the blocks then fall through the table.
T12_CAPACITY = {"njmax": 640, "nconmax": 160}


def t12_xml() -> Path:
    """The T12 scene file in the shipped library, in a source checkout or an installed package."""
    from alloy.cards.library import default_root
    return default_root() / T12_SCENE


def build_t12(worlds: int, device: str):
    """The T12 benchmark scene (two arms, eight blocks, a ledge) from MJCF, with MuJoCo Warp physics."""
    from alloy._vendor import newton
    tmpl = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(tmpl)
    tmpl.add_mjcf(str(t12_xml()), ctrl_direct=True, parse_visuals=True)
    scene = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(scene)
    scene.replicate(tmpl, worlds)
    model = scene.finalize(device=device)
    return model, newton.solvers.SolverMuJoCo(model, **T12_CAPACITY)


def mujoco_fill(solver) -> dict:
    """How full MuJoCo Warp's buffers are now: the fullest world's constraint rows and the batch's contacts, each
    beside its capacity. Raises when either overflowed: the step then drops constraints, so a row would time broken
    physics."""
    d = solver.mjw_data
    fill = {"constraint_rows": int(d.nefc.numpy().max()), "njmax": int(d.njmax),
            "contacts": int(d.nacon.numpy().max()), "naconmax": int(d.naconmax)}
    if fill["constraint_rows"] > fill["njmax"] or fill["contacts"] > fill["naconmax"]:
        raise RuntimeError(f"MuJoCo Warp buffers overflowed ({fill}): raise T12_CAPACITY")
    return fill


def build_boxes(worlds: int, device: str):
    """A ground plane, a tilted box and a ball per world, with XPBD physics."""
    from alloy._vendor import newton
    b = newton.ModelBuilder()
    b.add_ground_plane()
    box = b.add_body(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_rpy(0.3, 0.2, 0.1)), label="box")
    b.add_shape_box(box, hx=0.08, hy=0.06, hz=0.04)
    ball = b.add_body(xform=wp.transform((0.15, 0.1, 0.5), wp.quat_identity()), label="ball")
    b.add_shape_sphere(ball, radius=0.05)
    scene = newton.ModelBuilder()
    scene.replicate(b, worlds)
    model = scene.finalize(device=device)
    return model, newton.solvers.SolverXPBD(model)


def t12_view(model, state) -> np.ndarray:
    """The benchmark's camera: three-quarter view of the bodies of world 0 at rest (as the T12 study)."""
    q = state.body_q.numpy()
    world = model.body_world.numpy()
    pts = q[world == 0, :3]
    lo, hi = pts.min(0), pts.max(0)
    target = 0.5 * (lo + hi)
    span = max(float(np.linalg.norm(hi - lo)), 0.3)
    return look_at(target + span * np.array([0.9, -1.1, 0.8]), target)


def _timed(fn, iters: int, device: str) -> float:
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    wp.synchronize_device(device)
    return (time.perf_counter() - t0) / iters


@dataclass
class BenchConfig:
    """One benchmark cell."""

    scene: str = "t12"  # t12 | boxes | pack:<dir>@<frame>
    worlds: int = 1024
    width: int = 128
    height: int = 128
    spp: int = 16
    physics: bool = True
    quality: str = "fast"  # fast (diffuse) | full (Blender materials)
    denoise: str | None = None
    backend: str = "warp"
    iters: int = 10
    device: str = "cuda:0"
    tta: int = 1  # learned denoiser: flipped passes averaged (2 = the 128 px claim)
    weights: str | None = None  # learned denoiser weights (a manifest name); None = the size's default model
    claim: str | None = None  # the gate scene whose claimed setting this cell runs (see with_claim)


def with_claim(cfg: BenchConfig, gate_scene: str) -> BenchConfig:
    """``cfg`` at the claimed setting of ``gate_scene`` for its size: samples, denoiser, tta and weights (the claim's
    own, else the scene's tuned weights for the size, as the gate's speed rows pick them)."""
    from .denoise import weights_for
    from .gate_claims import BY_SIZE, claim_for
    res = f"{cfg.width}x{cfg.height}"
    if res not in BY_SIZE:
        raise ValueError(f"no gate claim at {res}: the gated sizes are {sorted(BY_SIZE)}")
    claim = claim_for(gate_scene, res)
    weights = (claim.weights or weights_for(gate_scene, cfg.width, cfg.height)) if claim.kind == "learned" else None
    return replace(cfg, spp=claim.spp, denoise=claim.kind, tta=claim.tta, weights=weights,
                   claim=f"{gate_scene}: {claim.label()}")


def _model_loop(cfg: BenchConfig):
    """(step, render, camera, fill) for a Newton model scene with physics; ``fill()`` is :func:`mujoco_fill` for
    T12 and None for the boxes."""
    from alloy._vendor import newton
    from .camera import camera_scene
    model, solver = (build_t12 if cfg.scene == "t12" else build_boxes)(cfg.worlds, cfg.device)
    s = {"s0": model.state(), "s1": model.state()}
    ctrl = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s["s0"])
    # MuJoCo Warp finds its own contacts (T12); only the XPBD boxes need Newton's collision pipeline
    pipe = None if cfg.scene == "t12" else newton.CollisionPipeline(model)
    contacts = None if pipe is None else pipe.contacts()
    pose = t12_view(model, s["s0"]) if cfg.scene == "t12" else look_at((0.7, -0.7, 0.5), (0.0, 0.0, 0.1))
    cam = _camera(cfg, pose)
    scene = camera_scene(model, device=cfg.device, quality=cfg.quality)

    def step():
        if pipe is not None:
            pipe.collide(s["s0"], contacts)
        solver.step(s["s0"], s["s1"], ctrl, contacts, 1.0 / 120.0)
        s["s0"], s["s1"] = s["s1"], s["s0"]

    fill = (lambda: mujoco_fill(solver)) if cfg.scene == "t12" else (lambda: None)
    return step, lambda: cam.render(scene, s["s0"]), cam, fill


def _pack_loop(cfg: BenchConfig):
    """(step, render, camera, fill) for a scene pack: each step moves every world's meshes a little (kinematic
    replay)."""
    from .camera import load_pack
    path, frame = cfg.scene[5:].rsplit("@", 1)
    scene = load_pack(path, frame=int(frame), num_worlds=cfg.worlds, device=cfg.device)
    cam = _camera(cfg, None)
    sc = getattr(scene, "sc", scene)  # PbrScene (v2) holds a GpuScene; a v1 pack is one
    base = sc.xf.numpy().copy()
    t = {"k": 0}

    def step():
        t["k"] += 1
        xf = base.copy()
        xf[:, 2] += 1e-4 * np.sin(0.1 * t["k"] + np.arange(len(xf)))  # sub-mm motion: every instance refits
        sc.set_transforms(xf)

    return step, lambda: cam.render(scene), cam, lambda: None


def _camera(cfg: BenchConfig, pose) -> Camera:
    """The bench camera; a pack keeps its own Cycles settings (only the backend changes)."""
    from .denoise import DenoiseSpec
    return Camera("bench", pose=pose, width=cfg.width, height=cfg.height, fov=np.degrees(0.9), spp=cfg.spp,
                  backend=cfg.backend, denoise=DenoiseSpec(cfg.denoise, weights=cfg.weights, tta=cfg.tta))


def run(cfg: BenchConfig) -> dict:
    """One benchmark row (see module docstring)."""
    step, render, cam, fill = (_pack_loop if cfg.scene.startswith("pack:") else _model_loop)(cfg)
    for _ in range(2):  # compile + warm
        step()
        render()
    t_phys = _timed(step, cfg.iters, cfg.device) if cfg.physics else 0.0
    t_rend = _timed(render, cfg.iters, cfg.device)
    t_all = _timed(lambda: (step(), render()), cfg.iters, cfg.device) if cfg.physics else t_rend
    return {**_row(cfg, cam, t_phys, t_rend, t_all), "mujoco": fill()}


def _row(cfg: BenchConfig, cam: Camera, t_phys: float, t_rend: float, t_all: float) -> dict:
    return {"scene": cfg.scene, "worlds": cfg.worlds, "width": cfg.width, "height": cfg.height, "spp": cfg.spp,
            "quality": cfg.quality, "denoise": cfg.denoise, "weights": cfg.weights, "tta": cfg.tta,
            "claim": cfg.claim, "physics": cfg.physics,
            "backend": resolve_backend(cam._bound[1].st.backend, cfg.device),
            "images_per_s": round(cfg.worlds / t_all, 1), "render_only_images_per_s": round(cfg.worlds / t_rend, 1),
            "ms": {"physics": round(1e3 * t_phys, 3), "render_and_display": round(1e3 * t_rend, 3),
                   "loop": round(1e3 * t_all, 3)}, "gpu": wp.get_device(cfg.device).name}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scene", default="t12", help="t12 | boxes | pack:<dir>@<frame>")
    ap.add_argument("--worlds", type=int, nargs="+", default=[1024])
    ap.add_argument("--size", nargs="+", default=["128x128"], help="WxH, e.g. 128x128 640x480 1280x720")
    ap.add_argument("--spp", type=int, nargs="+", default=[16])
    ap.add_argument("--quality", choices=("fast", "full"), default="fast")
    ap.add_argument("--denoise", default=None, choices=("auto", "learned", "oidn"))
    ap.add_argument("--backend", default="warp", help="warp | optix | auto | wavefront-warp")
    ap.add_argument("--tta", type=int, default=1, help="learned denoiser test-time augmentation (2 at 128 px, 5 spp)")
    ap.add_argument("--weights", default=None, help="learned denoiser weights (a manifest name); default: by size")
    ap.add_argument("--claim", default=None, metavar="GATE_SCENE",
                    help="run each size at this gate scene's claimed setting (c3, t12, ...): overrides --spp, "
                         "--denoise, --tta and --weights")
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(argv)
    for n in a.worlds:
        for size in a.size:
            w, h = (int(v) for v in size.lower().split("x"))
            for spp in a.spp[:1] if a.claim else a.spp:
                cfg = BenchConfig(a.scene, n, w, h, spp, not a.no_physics, a.quality, a.denoise, a.backend, a.iters,
                                  a.device, a.tta, a.weights)
                print(json.dumps(run(with_claim(cfg, a.claim) if a.claim else cfg)), flush=True)


if __name__ == "__main__":
    main()
