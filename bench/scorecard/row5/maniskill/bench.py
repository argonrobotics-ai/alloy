# SPDX-License-Identifier: Apache-2.0
"""Row 5 camera throughput, ManiSkill3 (SAPIEN renderer). One process = one cell; appends one JSON row to --out.

Based on mani_skill/examples/benchmarking/gpu_sim.py (same gym.make path, torch actions, GPU tensors, no reset).
Run inside the venv made by setup.sh:
  ~/ms/bin/python bench.py --scene cartpole|franka_cube --outputs rgb|rgb+depth|rgb+depth+seg --size 128 --worlds 1024 \
      --out maniskill_g4.jsonl --pngdir png [--sync torch|host|hostloop]

--sync decides when the clock stops. ``torch`` (default): torch.cuda.synchronize(). ``host``: also read back a value that
depends on every image of the last step. ``hostloop``: read it after every loop. torch.cuda.synchronize alone does not
wait for all of SAPIEN's GPU work in every case, so the host read is the check that the render really finished.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("row5.maniskill")
ENVS = {"cartpole": "CartpoleBalanceBenchmark-v1", "franka_cube": "PickCube-v1"}
OBS_MODES = {"rgb": "rgb", "rgb+depth": "rgb+depth", "rgb+depth+seg": "rgb+depth+segmentation"}
PINS = "ManiSkill@62ff3a5896b4 sapien 3.0.3 torch 2.15.0+cu132 py3.12 driver "


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scene", required=True, choices=sorted(ENVS))
    p.add_argument("--outputs", required=True, choices=sorted(OBS_MODES))
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--worlds", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pngdir", default=None)
    p.add_argument("--loops", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--sync", default="torch", choices=["torch", "host", "hostloop"])
    p.add_argument("--act", type=float, default=0.1)
    return p.parse_args(argv)


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else value


@dataclass
class GpuMem:
    """Peak GPU memory of this process (and of the whole GPU) sampled at 5 Hz through NVML."""
    peak_proc: float = 0.0
    peak_gpu: float = 0.0
    samples: int = 0
    stopped: bool = False
    handle: Any = field(default=None, repr=False)
    baseline: float = 0.0
    name: str = ""
    driver: str = ""

    def start(self) -> "GpuMem":
        import pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.baseline = pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20
        self.name = _text(pynvml.nvmlDeviceGetName(self.handle))
        self.driver = _text(pynvml.nvmlSystemGetDriverVersion())
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _proc_mib(self) -> float | None:
        import pynvml
        for pr in pynvml.nvmlDeviceGetComputeRunningProcesses(self.handle):
            if pr.pid == os.getpid() and pr.usedGpuMemory is not None:
                return pr.usedGpuMemory / 2**20
        return None

    def _loop(self) -> None:
        import pynvml
        while not self.stopped:
            mine = self._proc_mib()
            if mine:
                self.peak_proc = max(self.peak_proc, mine)
            self.peak_gpu = max(self.peak_gpu, pynvml.nvmlDeviceGetMemoryInfo(self.handle).used / 2**20)
            self.samples += 1
            time.sleep(0.2)

    def stop(self) -> None:
        self.stopped = True
        time.sleep(0.3)


def make_env(a: argparse.Namespace):
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import mani_skill.examples.benchmarking.envs  # noqa: F401  (registers *Benchmark-v1)
    if a.scene == "cartpole":
        kw = dict(camera_width=a.size, camera_height=a.size, num_cameras=1)
    else:
        kw = dict(sensor_configs=dict(width=a.size, height=a.size))
    return gym.make(ENVS[a.scene], num_envs=a.worlds, obs_mode=OBS_MODES[a.outputs], render_mode="rgb_array", **kw)


def describe_env(be, a: argparse.Namespace, driver: str) -> dict:
    import mani_skill
    import sapien
    import torch
    cams = list(be._sensors.values())
    packs = ",".join(sorted({c.config.shader_pack for c in cams}))
    return {
        "engine_version": f"mani_skill {mani_skill.__version__} sapien {sapien.__version__} torch {torch.__version__}",
        "commit_or_version_pins": PINS + driver,
        "renderer": "SAPIEN Vulkan rasterizer",
        "render_settings": (f"shader_pack={packs} (ManiSkill default 'minimal': fastest, no shadows); camera fov "
                            f"{cams[0].config.fov:.3f} rad; env lighting/textures per env; 1 camera/world; "
                            "depth/segmentation from the same pass"),
        "control_dt_s": 1.0 / be.control_freq, "physics_substeps": be.sim_freq // be.control_freq,
        "sim_freq": be.sim_freq, "control_freq": be.control_freq,
        "control_mode": getattr(be, "control_mode", None), "env_id": ENVS[a.scene]}


def wanted(outputs: str) -> list[str]:
    return ["rgb"] + (["depth"] if "depth" in outputs else []) + (["segmentation"] if "seg" in outputs else [])


def grab(obs: dict, outputs: str) -> dict:
    cd = next(iter(obs["sensor_data"].values()))
    for k in wanted(outputs):
        assert k in cd, (k, list(cd))
    return {k: cd[k] for k in wanted(outputs)}


def host_read(obs: dict) -> float:
    """Host read of a value depending on every image of the last step (rgb, and depth / segmentation when present)."""
    import torch
    cd = next(iter(obs["sensor_data"].values()))
    return sum(float(cd[k].sum(dtype=torch.float64).item()) for k in ["rgb", "depth", "segmentation"] if k in cd)


def _rate_loop(a: argparse.Namespace, step, per_world: bool = True, check=None) -> tuple[list[float], Any]:
    """``repeats`` timed runs of ``loops`` calls of ``step() -> obs``; the clock stops per --sync.

    ``check(obs)`` runs after each repeat, outside the clock."""
    import torch
    rates, obs = [], None
    for _ in range(a.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.loops):
            obs = step()
            if a.sync == "hostloop" and obs is not None:
                host_read(obs)
        torch.cuda.synchronize()
        if a.sync != "torch" and obs is not None:
            host_read(obs)
        rates.append((a.worlds if per_world else 1) * a.loops / (time.perf_counter() - t0))
        if check:
            check(obs)
    return rates, obs


def run_timings(env, be, a: argparse.Namespace) -> dict:
    import torch
    adim = env.action_space.shape

    def act():
        return a.act * (2 * torch.rand(adim, device=be.device) - 1)

    snaps: list = []

    def env_step():
        return env.step(act())[0]

    def check_rerendered(obs) -> None:
        snaps.append(grab(obs, "rgb")["rgb"][:4].clone())
        if len(snaps) > 1:
            assert not torch.equal(snaps[-1], snaps[-2]), "consecutive frames identical: not re-rendered"

    def physics_step() -> None:
        be._step_action(act())

    with torch.inference_mode():
        obs, _ = env.reset(seed=2022)
        first = grab(obs, a.outputs)["rgb"][:4].clone()
        for _ in range(a.warmup):
            obs, *_ = env.step(act())
        torch.cuda.synchronize()
        reps, obs = _rate_loop(a, env_step, check=check_rerendered if a.sync != "torch" else None)
        last = grab(obs, a.outputs)
        rr, _ = _rate_loop(a, be.get_obs)  # render only: capture all cameras again (no physics)
        # physics only: control step without observations (action + physics substeps + GPU fetch)
        pp, _ = _rate_loop(a, physics_step, per_world=False)
    return {"reps": reps, "render_only": rr, "physics_only": pp, "first": first, "last": last}


def save_png(a: argparse.Namespace, last: dict, img: np.ndarray) -> None:
    from PIL import Image
    os.makedirs(a.pngdir, exist_ok=True)
    tiles = [np.concatenate(list(img), axis=1)]
    if "depth" in last:
        d = last["depth"][:4, ..., 0].float().cpu().numpy()
        d[~np.isfinite(d)] = 0
        d8 = (255 * d / max(d.max(), 1)).astype(np.uint8)
        tiles.append(np.repeat(np.concatenate(list(d8), axis=1)[..., None], 3, 2))
    if "segmentation" in last:
        s = last["segmentation"][:4, ..., 0].cpu().numpy().astype(np.int64)
        s8 = np.stack([(s * 53) % 256, (s * 97) % 256, (s * 29) % 256], -1).astype(np.uint8)
        tiles.append(np.concatenate(list(s8), axis=1))
    name = f"maniskill_{a.scene}_{a.outputs.replace('+', '_')}_{a.size}_n{a.worlds}.png"
    Image.fromarray(np.concatenate(tiles, axis=0)).save(f"{a.pngdir}/{name}")


def sanity(a: argparse.Namespace, t: dict) -> tuple[dict, bool]:
    img, img0 = t["last"]["rgb"][:4].cpu().numpy(), t["first"].cpu().numpy()
    nonblack = bool(img.mean() > 3 and img.std() > 2)
    moved = float(np.abs(img.astype(float) - img0.astype(float)).mean())
    differ = float(np.abs(img[0].astype(float) - img[1].astype(float)).mean()) if len(img) > 1 else 0.0
    if a.pngdir:
        save_png(a, t["last"], img)
    info = {"mean": float(img.mean()), "std": float(img.std()), "changed_since_reset": moved, "world0_vs_world1": differ}
    return info, (not nonblack) or moved < 0.01


def finish(a: argparse.Namespace, row: dict, status: str, **kw) -> None:
    row.update(status=status, utc=datetime.datetime.now(datetime.timezone.utc).strftime("%FT%TZ"),
               command=" ".join(sys.argv), **kw)
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)
    os._exit(0)  # SAPIEN's teardown can hang; the row is written


def measure(a: argparse.Namespace, row: dict, mem: GpuMem) -> None:
    env = make_env(a)
    be = env.unwrapped
    row.update(describe_env(be, a, mem.driver))
    t = run_timings(env, be, a)
    mem.stop()
    info, bad = sanity(a, t)
    row.update(sanity=info, images_per_s_repeats=t["reps"])
    med = lambda v: float(np.median(v))  # noqa: E731
    finish(a, row, "invalid" if bad else "ok", images_per_s=med(t["reps"]), images_per_s_min=min(t["reps"]),
           images_per_s_max=max(t["reps"]), render_only_images_per_s=med(t["render_only"]),
           physics_only_steps_per_s=med(t["physics_only"]), physics_only_env_steps_per_s_all_worlds=med(t["physics_only"]) * a.worlds,
           gpu_mem_mib=mem.peak_proc or None, gpu_mem_whole_gpu_peak_minus_idle_mib=mem.peak_gpu - mem.baseline,
           mem_samples=mem.samples, reason="image blank or frozen" if bad else None)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    a = parse_args()
    mem = GpuMem().start()
    row = dict(sync=a.sync, engine="maniskill", scene=a.scene, outputs=a.outputs, width=a.size, height=a.size,
               worlds=a.worlds, gpu=mem.name, driver=mem.driver)
    try:
        measure(a, row, mem)
    except BaseException as e:
        msg = str(e)
        oom = "out of memory" in msg.lower() or "outofmemory" in type(e).__name__.lower() or "VK_ERROR_OUT_OF" in msg
        row["gpu_mem_mib_at_fail_whole_gpu_peak"] = mem.peak_gpu - mem.baseline
        finish(a, row, "oom" if oom else "failed", reason=msg[:400], traceback=traceback.format_exc()[-1500:])


if __name__ == "__main__":
    main()
