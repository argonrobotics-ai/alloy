# SPDX-License-Identifier: Apache-2.0
"""Row 5 camera throughput for MuJoCo Warp's own renderer (the copy vendored in Alloy: alloy._vendor.mujoco_warp).

One cell per process:
  python mujoco_warp_cam.py --scene cartpole|franka_cube --size 128 --worlds 1024 --outputs rgb|rgb+depth|rgb+depth+seg \
      --out raw/mujoco_warp_g4.jsonl --png-dir raw/png
Loop = `substeps` x mjw.step (the model's own timestep; dt = 10 ms per loop), hold-position ctrl with noise, then
mjw.kinematics + refit_bvh + render and get_rgb (+ get_depth, get_segmentation) into GPU arrays viewed as torch tensors.
Renderer: ray caster against a BVH, one ray per pixel (samples_per_pixel=1), no shadows, no textures, ambient + one
directional light, Blinn-Phong style shading (MuJoCo Warp's recommended defaults). It draws geom groups 0-3, so the
Panda is drawn as its collision hulls, where Alloy draws the visual meshes.
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

log = logging.getLogger("row5.mujoco_warp")

CAM = "bench"
CAPACITY = {"cartpole": (32, 128), "franka_cube": (128, 640)}  # (nconmax, njmax) per world; overflow is checked
UPSTREAM = "86e2fae005093dcac9680f8a42646da002ebea83 (vendored, no patches)"


@dataclass
class Loop:
    physics: Callable[[], None]
    render: Callable[[], Any]
    sync: Callable[[], None]
    substeps: int
    outs: dict


def _xyaxes(eye, target) -> list[float]:
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return list(x) + list(y)


def load_spec(scene: str, size: int):
    import mujoco
    from alloy.cards.library import default_root
    if scene == "cartpole":
        spec = mujoco.MjSpec.from_string(S.CARTPOLE_MJCF.format(floor=S.CARTPOLE_FLOOR))
    else:
        spec = mujoco.MjSpec.from_file(str(default_root() / "robots" / "emb_e2_panda" / "scene.xml"))
    eye, target = S.SCENES[scene][3]
    spec.worldbody.add_camera(name=CAM, pos=list(eye), mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
                              xyaxes=_xyaxes(eye, target), fovy=S.CAM_FOV, resolution=[size, size])
    spec.worldbody.add_light(name="sun", pos=[0, 0, 4], dir=[-0.3, 0.4, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                             castshadow=False, diffuse=[0.8, 0.8, 0.8], ambient=[0.3, 0.3, 0.3])
    return spec


def build(scene: str, worlds: int, size: int, outputs: str):
    import mujoco
    from alloy._vendor import mujoco_warp as mjw
    mjm = load_spec(scene, size).compile()
    mjd = mujoco.MjData(mjm)
    if mjm.nkey:
        mujoco.mj_resetDataKeyframe(mjm, mjd, 0)
    mujoco.mj_forward(mjm, mjd)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=worlds, nconmax=CAPACITY[scene][0], njmax=CAPACITY[scene][1])
    rc = mjw.create_render_context(
        mjm, nworld=worlds, cam_res=(size, size), render_rgb=True, render_depth="depth" in outputs,
        render_seg="seg" in outputs, use_textures=False, use_shadows=False, enabled_geom_groups=[0, 1, 2, 3],
        cam_active=[CAM], samples_per_pixel=1)
    return mjm, m, d, rc


def make_loop(mjm, m, d, rc, scene: str, worlds: int, size: int, outputs: str) -> Loop:
    import torch
    import warp as wp
    from alloy._vendor import mujoco_warp as mjw
    substeps = max(1, round(S.CONTROL_DT / mjm.opt.timestep))
    pool = torch.from_numpy(S.noise_pool(scene, worlds)).to(wp.device_to_torch(wp.get_device()))
    ctrl = wp.to_torch(d.ctrl)
    counter = {"i": 0}
    outs = {"rgb": wp.zeros((worlds, size, size), dtype=wp.vec3)}
    if "depth" in outputs:
        outs["depth"] = wp.zeros((worlds, size, size), dtype=float)
    if "seg" in outputs:
        outs["seg"] = wp.zeros((worlds, size, size), dtype=wp.vec2i)

    def physics() -> None:
        ctrl[:] = pool[counter["i"] % len(pool)]
        counter["i"] += 1
        for _ in range(substeps):
            mjw.step(m, d)

    def render():
        mjw.kinematics(m, d)
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
        mjw.get_rgb(rc, 0, outs["rgb"])
        if "depth" in outs:
            mjw.get_depth(rc, 0, 10.0, outs["depth"])
        if "seg" in outs:
            mjw.get_segmentation(rc, 0, outs["seg"])
        return wp.to_torch(outs["rgb"])

    def sync() -> None:
        wp.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return Loop(physics, render, sync, substeps, outs)


def to_u8(rgb) -> np.ndarray:
    return (rgb[:4].clamp(0, 1) * 255).to("cpu").numpy().astype(np.uint8)


def _check_extra_outputs(a: argparse.Namespace, outs: dict, why: str | None) -> str | None:
    """Depth and segmentation sanity, plus their PNG grids; returns the first reason found (or ``why``)."""
    tag = f"{a.png_dir}/mujoco_warp_{a.scene}_{a.outputs}_{a.size}px_{a.worlds}w"
    if "depth" in outs:
        dep = outs["depth"].numpy()[:4]
        if not np.isfinite(dep).all() or dep.max() <= 0:
            why = why or "depth image empty"
        U.save_grid(tag + "_depth.png", np.clip(dep * 255, 0, 255).astype(np.uint8)[..., None])
    if "seg" in outs:
        sg = outs["seg"].numpy()[:4, ..., 0]
        if (sg >= 0).sum() == 0:
            why = why or "segmentation image empty"
        U.save_grid(tag + "_seg.png", ((sg + 1) * 37 % 256).astype(np.uint8)[..., None])
    return why


def _measure(loop: Loop, a: argparse.Namespace, mem: U.MemSampler, t_build: float) -> dict:
    both = lambda: (loop.physics(), loop.render())  # noqa: E731
    both()
    loop.sync()
    first = to_u8(loop.render())
    build_s = time.perf_counter() - t_build
    both_t = U.time_loops(both, loop.sync, a.worlds, a.warmup, a.repeats, a.loops, on_start=mem.mark)
    render_t = U.time_loops(loop.render, loop.sync, a.worlds, 10, 3, 50, budget_s=30.0)
    physics_t = U.time_loops(loop.physics, loop.sync, a.worlds, 10, 3, 50, budget_s=30.0)
    last = to_u8(loop.render())
    return {"both": both_t, "render": render_t, "physics": physics_t, "first": first, "last": last, "build_s": build_s}


def _row_pins(a: argparse.Namespace) -> dict:
    import mujoco
    import warp as wp
    return {"mujoco_warp": UPSTREAM, "warp": wp.config.version, "mujoco": mujoco.__version__, "alloy_commit": a.commit}


def run_cell(a: argparse.Namespace) -> dict:
    from alloy._vendor import mujoco_warp as mjw
    mem = U.MemSampler().start()
    t_build = time.perf_counter()
    mjm, m, d, rc = build(a.scene, a.worlds, a.size, a.outputs)
    loop = make_loop(mjm, m, d, rc, a.scene, a.worlds, a.size, a.outputs)
    t = _measure(loop, a, mem, t_build)
    memr = mem.stop()
    ncon, nefc = int(d.nacon.numpy().max()), int(d.nefc.numpy().max())
    why = U.sanity(t["first"], t["last"])
    U.save_grid(f"{a.png_dir}/mujoco_warp_{a.scene}_{a.outputs}_{a.size}px_{a.worlds}w.png", t["last"])
    why = _check_extra_outputs(a, loop.outs, why)
    if nefc > CAPACITY[a.scene][1] or ncon > d.naconmax:
        why = f"MuJoCo Warp buffers overflowed (nefc {nefc} / {CAPACITY[a.scene][1]}, nacon {ncon} / {d.naconmax})"
    res = t["both"]
    return {"engine": "mujoco_warp", "engine_version": f"{mjw.__version__} (vendored in alloy, upstream 86e2fae0)",
            "scene": a.scene, "outputs": a.outputs, "width": a.size, "height": a.size, "worlds": a.worlds,
            "status": "invalid" if why else "ok", "invalid_reason": why, "images_per_s": round(res["rate"], 1),
            "images_per_s_min": round(res["min"], 1), "images_per_s_max": round(res["max"], 1),
            "render_only_images_per_s": round(t["render"]["rate"], 1),
            "physics_only_steps_per_s": round(t["physics"]["rate"], 1),
            "repeats": res["repeats"], "loops_per_repeat": res["loops"], "warmup_loops": res["warmup_loops"],
            "reduced_loops": res["reduced_loops"], "gpu_mem_mib": memr["gpu_mem_mib"], **memr,
            "renderer": "MuJoCo Warp ray caster (BVH, 1 ray/pixel), no shadows, no textures, 1 directional light + ambient",
            "render_settings": {"samples_per_pixel": 1, "shadows": False, "textures": False,
                                "geom_groups": [0, 1, 2, 3], "fov_deg": S.CAM_FOV, "depth_scale": 10.0},
            "control_dt_s": S.CONTROL_DT, "physics_substeps": loop.substeps, "model_timestep_s": float(mjm.opt.timestep),
            "ngeom_rendered": int(rc.bvh_ngeom), "nacon_max": ncon, "nefc_max": nefc,
            "build_s": round(t["build_s"], 1), **U.gpu_info(), "commit_or_version_pins": _row_pins(a),
            "command": " ".join(sys.argv)}


def main() -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    U.add_cell_arguments(ap, sorted(S.SCENES))
    a = ap.parse_args()
    base = {"engine": "mujoco_warp", "scene": a.scene, "outputs": a.outputs, "width": a.size, "height": a.size,
            "worlds": a.worlds, "command": " ".join(sys.argv)}
    try:
        row = run_cell(a)
    except Exception as exc:
        log.error("cell failed:\n%s", traceback.format_exc())
        row = {**base, "status": "oom" if U.is_oom(exc) else "error", "reason": f"{type(exc).__name__}: {str(exc)[:500]}"}
    U.write_row(a.out, row)
    print(row["status"], row.get("images_per_s"), row.get("gpu_mem_mib"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
