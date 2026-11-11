# SPDX-License-Identifier: Apache-2.0
"""Row 5 camera throughput for Madrona-MJX (MuJoCo Playground vision envs). One process = one cell
(scene, backend, size, worlds); Madrona fixes world count and resolution at init. Writes one JSON row per output
(rgb, rgb+depth, rgb+depth+seg -> unsupported) to --out and a PNG grid of 4 worlds.

Loop = one control step of the Playground env (mjx physics n_substeps = ctrl_dt/sim_dt, random small actions) + one
Madrona render of every world, in a jax.lax.scan of 50 loops; images stay on the GPU; token = a few pixels summed
(dependent output), block_until_ready before the clock stops. The env's own vision post-processing (grayscale
history / float32 brightness) is dropped: the step returns the renderer's raw uint8 RGBA (+ float depth).

Run inside the venv made by setup.sh (run_all.sh does it): python bench.py --scene cartpole --backend raytracer \
    --size 128 --worlds 1024 --out madrona_g4.jsonl --png grid.png
"""
from __future__ import annotations

import argparse
import datetime
import functools
import json
import logging
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # JAX default preallocates 75 % and voids memory
os.environ.setdefault("MADRONA_MWGPU_DEVICE_HEAP_SIZE", str(2**31))  # as playground's benchmark.py

import numpy as np  # noqa: E402

log = logging.getLogger("row5.madrona")
ENV_NAMES = {"cartpole": "CartpoleBalance", "franka_cube": "PandaPickCubeCartesian"}
NO_SEG = ("Madrona-MJX exposes only rgb_tensor and depth_tensor (src/bindings.cpp, src/mgr.hpp); "
          "no segmentation / instance-id output")
NOTE = ("depth is produced by the same render call for every output (Madrona returns rgb and depth together): "
        "asking for it adds only the read of the depth tensor in the timed token. Physics = MJX, random "
        "actions uniform +-0.1. The env's grayscale-history / float32-brightness post-processing is removed.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", choices=sorted(ENV_NAMES), required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--worlds", type=int, required=True)
    p.add_argument("--backend", choices=["raytracer", "rasterizer"], default="raytracer")
    p.add_argument("--out", required=True)
    p.add_argument("--png", default=None)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--loops", type=int, default=50)
    return p.parse_args(argv)


# ---- GPU memory (nvidia-smi, 2 Hz) -------------------------------------------------------------------------------
def _smi(query: str) -> str:
    return subprocess.run(["nvidia-smi", f"--query-{query}", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout


def gpu_used() -> int:
    return int(_smi("gpu=memory.used").split()[0])


def proc_used(pid: int) -> int | None:
    for ln in _smi("compute-apps=pid,used_memory").splitlines():
        f = [x.strip() for x in ln.split(",")]
        if len(f) == 2 and f[0] == str(pid):
            return int(f[1])
    return None


class Sampler(threading.Thread):
    """Samples this process's GPU memory and the whole GPU's every 0.5 s."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop_flag = False
        self.proc: list[int] = []
        self.gpu: list[int] = []

    def run(self) -> None:
        while not self.stop_flag:
            mine = proc_used(os.getpid())
            if mine is not None:
                self.proc.append(mine)
            self.gpu.append(gpu_used())
            time.sleep(0.5)


# ---- the env and its timed rollouts ------------------------------------------------------------------------------
@dataclass
class Rig:
    """The Playground env with its renderer patched to return raw outputs, wrapped for ``W`` worlds."""
    envw: Any
    raw_render: Any
    mode: dict
    state0: Any
    act_size: int
    worlds: int
    size: int
    ctrl_dt: float
    sim_dt: float
    name: str


def build_rig(a: argparse.Namespace) -> Rig:
    import jax
    import jax.numpy as jp
    from mujoco_playground import registry
    from mujoco_playground._src import wrapper
    W, S = a.worlds, a.size
    cfg = {"vision": True, "vision_config.render_batch_size": W, "vision_config.render_width": S,
           "vision_config.render_height": S, "vision_config.use_rasterizer": a.backend == "rasterizer",
           "action_repeat": 1}
    if a.scene == "cartpole":
        cfg |= {"ctrl_dt": 0.04, "sim_dt": 0.01, "vision_config.history": 1}  # the training notebook's 0.04 s control step
    env = registry.load(ENV_NAMES[a.scene], config_overrides=cfg)
    mode = {"m": "full"}
    raw_render = env.renderer.render

    def render_shim(token, data):
        """The env's own render call becomes a no-op (its post-processing is dropped); step() renders itself."""
        return token, jp.zeros((1, S, S, 4), jp.uint8), jp.zeros((1, S, S, 1), jp.float32)

    env.renderer.render = render_shim
    orig_step = env.step

    def step(state, action):
        hist, token = state.info.get("obs_history"), state.info["render_token"]
        out = orig_step(state, action)
        if hist is not None:
            out.info["obs_history"] = hist  # drop the env's grayscale history update (post-processing, not physics/render)
        if mode["m"] == "physics":
            return out.replace(obs={"rgb": jp.zeros((S, S, 4), jp.uint8), "depth": jp.zeros((S, S, 1), jp.float32)})
        _, rgb, depth = raw_render(token, out.data)
        return out.replace(obs={"rgb": rgb[0], "depth": depth[0]})

    env.step = step
    envw = wrapper.MadronaWrapper(env, W)
    state0 = jax.jit(envw.reset)(jax.random.split(jax.random.PRNGKey(0), W)).replace(obs=jp.zeros((W,)))
    return Rig(envw, raw_render, mode, state0, env.action_size, W, S, float(env._config.ctrl_dt),
               float(env._config.sim_dt), ENV_NAMES[a.scene])


def make_rollout(rig: Rig):
    import jax
    import jax.numpy as jp
    W = rig.worlds

    @functools.partial(jax.jit, static_argnames=("n", "with_depth", "mode"))
    def rollout(state, key, n, with_depth, mode):
        rig.mode["m"] = mode

        def body(c, _):
            s, k = c
            k, ka = jax.random.split(k)
            act = 0.1 * jax.random.uniform(ka, (W, rig.act_size), minval=-1.0, maxval=1.0)
            if mode == "render":  # render only: no physics, same data every loop
                _, rgb, _depth = jax.vmap(lambda st: rig.raw_render(st.info["render_token"], st.data))(s)
                return (s, k), jp.sum(rgb[:, 0, :2, :2].astype(jp.int32))
            s2 = rig.envw.step(s, act)
            tok = jp.sum(s2.obs["rgb"][:, :2, :2].astype(jp.int32))
            if with_depth:
                tok = tok + jp.sum(s2.obs["depth"][:, :2, :2])
            return (s2.replace(obs=jp.zeros((W,))), k), tok

        (s, _), toks = jax.lax.scan(body, (state, key), None, length=n)
        return s, jp.sum(toks)

    return rollout


def measure(rig: Rig, rollout, a: argparse.Namespace, with_depth: bool, mode: str) -> list[float]:
    import jax

    def timed(n: int, key) -> float:
        t0 = time.perf_counter()
        s, tok = rollout(rig.state0, key, n, with_depth, mode)
        jax.block_until_ready((s.data.qpos, tok))
        return time.perf_counter() - t0

    for i in range(2):  # warm-up: compile + 2 x 25 loops (>= 20 loops after compile)
        timed(25 if i else a.loops, jax.random.PRNGKey(10 + i))
    walls = [timed(a.loops, jax.random.PRNGKey(20 + i)) for i in range(a.repeats)]
    return [rig.worlds * a.loops / w for w in walls]


def run_measurements(rig: Rig, a: argparse.Namespace) -> dict:
    rollout = make_rollout(rig)
    out = {"ips": {"rgb": measure(rig, rollout, a, False, "full"), "rgb+depth": measure(rig, rollout, a, True, "full")}}
    try:
        out["render_only"] = measure(rig, rollout, a, False, "render")
    except Exception as e:
        out["render_only"] = None
        log.warning("render-only failed: %s", repr(e)[:300])
    out["physics"] = measure(rig, rollout, a, False, "physics")
    return out


def sanity_images(rig: Rig, a: argparse.Namespace) -> tuple[str, dict]:
    """8 more loops from the start state, then 4 worlds: status, and the checks behind it."""
    import jax
    import jax.numpy as jp

    @jax.jit
    def imgs(state, key):
        rig.mode["m"] = "full"
        s = state
        for i in range(8):
            act = 0.1 * jax.random.uniform(jax.random.fold_in(key, i), (rig.worlds, rig.act_size), minval=-1.0, maxval=1.0)
            s = rig.envw.step(s, act)
        return s.obs["rgb"][:4], s.obs["depth"][:4], s.data.qpos[:4]

    rgb, depth, qpos_end = [np.asarray(x) for x in imgs(rig.state0, jax.random.PRNGKey(1))]
    qpos_start = np.asarray(rig.state0.data.qpos[:4])
    checks = {"nonblack": float(rgb[..., :3].std()) > 1.0,
              "frozen": bool(np.abs(qpos_end - qpos_start).max() == 0),
              "worlds_differ": bool(np.abs(qpos_end - qpos_end[0]).max() > 0)}
    if a.png:
        save_png(a.png, rgb, depth)
    ok = checks["nonblack"] and not checks["frozen"] and checks["worlds_differ"]
    return ("ok" if ok else "invalid"), checks


def save_png(path: str, rgb_np: np.ndarray, depth_np: np.ndarray) -> None:
    from PIL import Image
    d = depth_np[..., 0]
    fin = d[np.isfinite(d)]
    dn = np.clip((d - fin.min()) / max(fin.max() - fin.min(), 1e-6), 0, 1) if fin.size else d * 0
    top = np.concatenate(list(rgb_np[..., :3]), axis=1)
    bot = np.concatenate([np.repeat((x * 255).astype(np.uint8)[..., None], 3, axis=-1) for x in np.nan_to_num(dn)], axis=1)
    Image.fromarray(np.concatenate([top, bot], axis=0)).save(path)


def stats(v) -> tuple[float, float, float] | None:
    return None if not v else (float(np.median(v)), float(min(v)), float(max(v)))


def common_fields(a: argparse.Namespace, rig: Rig, mem: float | None, mem_src: str | None) -> dict:
    import importlib.metadata as md
    import jax
    import mujoco
    drv = _smi("gpu=driver_version,name").strip().split(",")
    pins = {"jax": jax.__version__, "mujoco": mujoco.__version__, "mujoco-mjx": md.version("mujoco-mjx"),
            "madrona_mjx": "1505699168fd", "madrona": "b46e6ab7", "mujoco_playground": "f2159f3"}
    shadows = "raytracer: yes (light casts shadows)" if a.backend == "raytracer" else "rasterizer: none"
    return {"engine": "madrona_mjx", "engine_version": "madrona_mjx 1505699 + playground f2159f3", "scene": a.scene,
            "width": a.size, "height": a.size, "worlds": a.worlds, "gpu": drv[-1].strip() if len(drv) > 1 else None,
            "driver": drv[0].strip(), "commit_or_version_pins": pins,
            "renderer": f"madrona {a.backend}" + (" (documented default / recommended)" if a.backend == "raytracer" else ""),
            "render_settings": {"backend": a.backend, "use_rasterizer": a.backend == "rasterizer", "cameras": 1,
                                "enabled_geom_groups": [0, 1, 2], "shadows": shadows,
                                "samples": "1 ray/pixel" if a.backend == "raytracer" else "n/a", "denoiser": "none",
                                "rgb": "uint8 RGBA (H,W,4)", "depth": "float32 (H,W,1)", "env": rig.name,
                                "MADRONA_MWGPU_DEVICE_HEAP_SIZE": os.environ["MADRONA_MWGPU_DEVICE_HEAP_SIZE"],
                                "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]},
            "control_dt_s": rig.ctrl_dt, "physics_substeps": int(round(rig.ctrl_dt / rig.sim_dt)), "sim_dt_s": rig.sim_dt,
            "gpu_mem_mib": mem, "gpu_mem_source": mem_src, "command": "python bench.py " + " ".join(sys.argv[1:]),
            "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "loops_per_repeat": a.loops, "repeats": a.repeats, "note": NOTE}


def build_rows(common: dict, status: str, m: dict) -> list[dict]:
    pr, ro = stats(m.get("physics")), stats(m.get("render_only"))
    rows = []
    for outs in ["rgb", "rgb+depth"]:
        st = stats(m.get("ips", {}).get(outs))
        rows.append(dict(common, outputs=outs, status=status, images_per_s=st and st[0], images_per_s_min=st and st[1],
                         images_per_s_max=st and st[2], render_only_images_per_s=ro and ro[0],
                         physics_only_steps_per_s=pr and pr[0]))  # world-steps/s: control steps per second over all worlds
    rows.append(dict(common, outputs="rgb+depth+seg", status="unsupported", images_per_s=None, reason=NO_SEG))
    return rows


def memory_of(smp: Sampler, base_gpu: int) -> tuple[float | None, str]:
    smp.stop_flag = True
    smp.join()
    if smp.proc:
        return max(smp.proc), "nvidia-smi compute-apps (this process)"
    return max(smp.gpu) - base_gpu, "whole-GPU used minus idle baseline"


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    a = parse_args()
    rig = build_rig(a)
    base_gpu = gpu_used()
    smp = Sampler()
    smp.start()
    status, measured, err, mem, mem_src = "error", {}, None, None, None
    try:
        measured = run_measurements(rig, a)
        mem, mem_src = memory_of(smp, base_gpu)
        status, checks = sanity_images(rig, a)
    except Exception as e:
        smp.stop_flag = True
        err = repr(e)[:1500]
        status = "oom" if ("RESOURCE_EXHAUSTED" in err or "out of memory" in err.lower()) else "error"
        measured, checks = {}, {}
    common = common_fields(a, rig, mem, mem_src)
    common["sanity"] = checks
    if err:
        common["error"] = err
    rows = build_rows(common, status, measured)
    with open(a.out, "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps({k: rows[0][k] for k in ("scene", "width", "worlds", "status", "images_per_s", "gpu_mem_mib")}))


if __name__ == "__main__":
    main()
